"""Зоны трещиноватости по космоснимкам: непрерывное поле вместо списка линий.

Задача №2 реестра: «выделить зоны трещиноватости по космоснимкам, сопоставить
их с зонами разломов по геологическим картам».

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ЗАДАЧИ №8. :mod:`src.lineaments_sat` уже прогнал
Кэнни -> Хаф прямо по композитам четырёх съёмок и получил отрицательный
вердикт: стоп-правило воспроизводимости (:data:`config.LIN_MIN_SPEARMAN`) не
прошёл ни один источник (Spearman половин выборки +0.23 у Sentinel-2, +0.33 у
Landsat 8/9, -0.20 у Sentinel-1, -0.61 у PALSAR-2). Повторять то же самое
смысла нет. Но там измерялась ДРУГАЯ величина:

* линеамент — ЛИНИЯ, её выделяют порогом; зона трещиноватости — ПЛОЩАДНОЙ
  объект, область повышенной раздробленности и проницаемости;
* воспроизводимость в задаче №8 считалась по растру ПОРОГОВЫХ детекций.
  Отрезок либо прошёл порог Хафа, либо исчез — это самый хрупкий из возможных
  статистиков. Полученные +0.2..+0.3 доказывают, что нет воспроизводимого
  СПИСКА ОТРЕЗКОВ; они не доказывают, что в снимке нет воспроизводимой
  структуры. Вдобавок там 90%+ карты были нулями у обеих половин, и метрику
  приходилось считать по объединению ненулевых пикселей.

Поэтому здесь тот же вопрос задаётся НЕПРЕРЫВНЫМ полем: структурный тензор
градиента даёт в каждом пикселе когерентность (насколько согласованы
направления градиента в окрестности), само направление и энергию — без
порога, без Хафа, без бинаризации. Поле деградирует плавно, метрика считается
по всей площади, и стоп-правило половин выборки применяется к нему честнее.

ПРОТОКОЛ СВЕРКИ С КАРТОЙ РАЗЛОМОВ. Задача №8 зафиксировала, что доля линий
«без соответствия на карте» неотличима от случайного узора сравнимой
плотности. Значит голая доля «поймано N% карты» ничего не значит и подаётся
только вместе с нуль-моделью. Здесь она зашита в протокол с самого начала:

1. метрика — capture-efficiency (lift@area) ячеек разломного коридора под
   картой-кандидатом, то есть та же метрика, что у всей фазы заверки
   (:mod:`src.assessment`), только «точками» выступают ячейки карты разломов;
2. значимость — тороидальный сдвиг карты-кандидата
   (:func:`src.assessment.spatial_null_pvalue`): двигается кандидат, карта
   разломов стоит, пространственная автокорреляция кандидата сохраняется;
3. контроль рельефа — та же метрика на кандидате, из которого линейный вклад
   рельефа вычтен (:func:`rank_residual`). Диагноз задачи №8 был «снимок
   пересказывает рельеф через более шумный канал»; если после вычитания
   рельефа метрика падает до уровня null, диагноз подтверждается уже по
   существу, а не по хрупкому стоп-правилу. Если держится — в снимке есть
   структурная информация, которой нет в рельефе, и это то, что стоит
   показывать.

Карта разломов при этом НЕ эталон истины: геолог рисовал её в том числе по
рельефу и старым снимкам, так что полное совпадение было бы подозрительным, а
расхождение не обязательно ошибка. Сверка здесь — согласование, а защита от
самообмана — пункт 2.

Все метрики протокола РАНГОВЫЕ (Spearman, квантильный порог lift), поэтому
монотонные преобразования полей значения не имеют: логарифмировать энергию
градиента или нет — на результат не влияет, и приводить источники к общей
шкале (дБ у радара против коэффициента отражения у оптики) не требуется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _scale_tag(scale_m: float) -> str:
    """Имя масштаба в стиле :data:`src.terrain_v2.TERRAIN_COLS` (``500`` / ``2km``)."""
    return f"{int(round(scale_m))}" if scale_m < 1000 else f"{scale_m / 1000.0:g}km"


def structure_tensor(image: np.ndarray, scale_m: float, res_m: float | None = None,
                     grad_sigma_px: float | None = None) -> dict[str, np.ndarray]:
    """Структурный тензор градиента: когерентность, энергия и направление оси.

    Тензор ``J = G_scale * (grad I)(grad I)^T`` усредняется гауссианой с сигмой
    ``scale_m``; по его собственным числам считаются

    * ``coh`` — когерентность ``((l1 - l2) / (l1 + l2))^2``, 0 при хаотичных
      направлениях и 1 при идеально согласованных. Это и есть площадная мера
      «здесь есть направленная структура», не требующая решать про каждый
      пиксель «линия или не линия»;
    * ``energy`` — след тензора ``l1 + l2`` (средний квадрат модуля градиента),
      аналог плотности линий у пороговой ветки;
    * ``dir_sin``/``dir_cos`` — направление ОСИ структуры (перпендикуляр к
      градиенту) через удвоенный угол, взвешенное когерентностью.

    Удвоенный угол — потому что у оси нет «начала» и «конца»: 10° и 190° это
    одно направление (та же кодировка, что у ``lin_dir_sin``/``lin_dir_cos`` в
    :func:`src.lineaments.rasters_to_cell_features`). Угол считается в
    РАСТРОВОЙ системе (ось col на восток, ось row на юг) — то есть в той же,
    что у рельефной ветки, чтобы направления двух веток сравнивались напрямую.

    Вес когерентностью нужен для усреднения по ячейке: среднее от
    ``coh * (sin 2a, cos 2a)`` — корректное осевое среднее, у которого длина
    вектора сама падает при разнобое направлений.

    Пропуски композита заполняются медианой (как в
    :func:`src.lineaments_sat.edges_from_image`), но результат маскируется
    обратно: заполнение нужно, чтобы гауссиана не размазала NaN на всю карту,
    а не чтобы придумать данные.
    """
    from scipy.ndimage import gaussian_filter

    res_m = res_m or config.SAT_RES_M
    grad_sigma_px = (config.FZ_GRAD_SIGMA_PX if grad_sigma_px is None
                     else grad_sigma_px)
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    if finite.sum() < config.FZ_MIN_FINITE_PX:
        return {k: np.full(img.shape, np.nan)
                for k in ("coh", "energy", "dir_sin", "dir_cos")}

    filled = np.where(finite, img, np.nanmedian(img[finite]))
    # Сглаживание ДО градиента: у радара спекл, у оптики шум датчика, и без
    # него градиент меряет их, а не форму поверхности.
    sm = gaussian_filter(filled, grad_sigma_px)
    gy, gx = np.gradient(sm, res_m)

    sigma_px = max(config.FZ_MIN_TENSOR_SIGMA_PX, scale_m / res_m)
    jxx = gaussian_filter(gx * gx, sigma_px)
    jxy = gaussian_filter(gx * gy, sigma_px)
    jyy = gaussian_filter(gy * gy, sigma_px)

    trace = jxx + jyy                       # l1 + l2
    delta = np.hypot(jxx - jyy, 2.0 * jxy)  # l1 - l2
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(trace > 0, (delta / trace) ** 2, 0.0)
        # Ось структуры перпендикулярна градиенту: удвоенный угол оси равен
        # удвоенному углу градиента плюс 180°, отсюда общий знак минус.
        sin2 = np.where(delta > 0, -2.0 * jxy / delta, 0.0)
        cos2 = np.where(delta > 0, -(jxx - jyy) / delta, 0.0)

    out = {"coh": coh, "energy": trace,
           "dir_sin": coh * sin2, "dir_cos": coh * cos2}
    # Маска возвращается: заполненные пропуски не должны выдавать себя за замер.
    return {k: np.where(finite, v, np.nan) for k, v in out.items()}


def zone_field(image: np.ndarray, scales_m: tuple[float, ...] | None = None,
               res_m: float | None = None) -> dict[str, np.ndarray]:
    """Многомасштабные растры зон трещиноватости по одному композиту.

    Масштабы (:data:`config.FZ_SCALES_M`) заданы в метрах, а не в пикселях:
    рабочий шаг у источников общий (100 м), но привязка окна к пикселям меняла
    бы физический размер зоны при смене источника.
    """
    scales_m = scales_m or config.FZ_SCALES_M
    out: dict[str, np.ndarray] = {}
    for s in scales_m:
        tag = _scale_tag(s)
        for name, arr in structure_tensor(image, s, res_m=res_m).items():
            out[f"{name}_{tag}"] = arr
    return out


def zone_features(meta, image: np.ndarray, prefix: str,
                  res_m: float | None = None,
                  scales_m: tuple[float, ...] | None = None) -> pd.DataFrame:
    """Растры :func:`zone_field` -> признаки на ячейки сетки (C-порядок).

    Агрегация переиспользует :func:`src.stac_grid.to_cells` — ту же, которой
    собраны все спутниковые признаки датасета, поэтому среднее и разброс
    внутри ячейки считаются здесь ровно так же, как у ``s1_``/``psr_``.
    """
    from . import stac_grid

    res_m = res_m or config.SAT_RES_M
    return stac_grid.to_cells(zone_field(image, scales_m, res_m), meta,
                              res_m=res_m, prefix=prefix)


def reproducibility(field_a: np.ndarray, field_b: np.ndarray,
                    pool: np.ndarray | None = None) -> dict[str, float]:
    """Согласие НЕПРЕРЫВНЫХ полей между двумя независимыми половинами выборки сцен.

    Стоп-правило то же (:data:`config.LIN_MIN_SPEARMAN`) и половины те же
    (чётные/нечётные по дате, :func:`src.lineaments_sat.split_even_odd`), но
    считается иначе, и это главное отличие ветки. Пороговой версии
    (:func:`src.lineaments_sat.reproducibility`) приходилось ограничиваться
    пикселями, где хоть одна половина что-то нашла, иначе метрика мерила долю
    совместных нулей. У непрерывного поля нулей нет, поэтому оно сравнивается
    по всей площади пула — метрика не вырождается и сравнима между источниками.
    """
    from scipy.stats import spearmanr

    a = np.asarray(field_a, dtype=float).ravel()
    b = np.asarray(field_b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if pool is not None:
        keep = np.zeros(a.shape, dtype=bool)
        keep[np.asarray(pool, dtype=int)] = True
        ok &= keep
    if ok.sum() < config.FZ_MIN_FINITE_PX:
        return {"rho": float("nan"), "covered_frac": float(ok.mean())}
    return {"rho": float(spearmanr(a[ok], b[ok]).statistic),
            "covered_frac": float(ok.mean())}


def rank_residual(x: np.ndarray, controls: np.ndarray | None) -> np.ndarray:
    """Ранговый остаток ``x`` после вычитания линейного вклада ``controls``.

    Ранговый, а не обычный МНК-остаток: вся метрология фазы ранговая (Spearman,
    квантильный порог lift), шкалы источников несопоставимы (дБ, коэффициент
    отражения, метры), а связь «яркость — уклон» заведомо нелинейна. Ранги
    переводят задачу в монотонную, где линейная модель осмысленна.

    Ячейки, где не хватает данных (NaN в ``x`` или в любом контроле),
    возвращаются как NaN и дальше по протоколу считаются промахом — так же, как
    в :func:`src.assessment.spatial_null_pvalue`. Ранги считаются ТОЛЬКО по
    полным ячейкам: пул задаётся тем, что подано на вход.
    """
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    if controls is None:
        return np.where(np.isfinite(x), x, np.nan)
    out = np.full(x.shape, np.nan)
    c = np.asarray(controls, dtype=float)
    if c.ndim == 1:
        c = c[:, None]
    ok = np.isfinite(x) & np.isfinite(c).all(axis=1)
    if ok.sum() < c.shape[1] + config.FZ_MIN_RESIDUAL_MARGIN:
        return out

    def ranks(v: np.ndarray) -> np.ndarray:
        return rankdata(v) / (len(v) + 1.0)

    xr = ranks(x[ok])
    design = np.column_stack([np.ones(int(ok.sum())),
                              *(ranks(c[ok, j]) for j in range(c.shape[1]))])
    beta, *_ = np.linalg.lstsq(design, xr, rcond=None)
    out[ok] = xr - design @ beta
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     controls: np.ndarray | None) -> float:
    """Частный ранговый коэффициент: связь ``x`` и ``y`` при фиксированных контролях.

    Описательное число рядом с основной метрикой протокола. Своя значимость у
    него не считается намеренно: 22 946 ячеек пространственно автокоррелированы,
    эффективный объём выборки на порядки меньше номинального, и t-тест по
    формуле дал бы p ~ 0 у сколь угодно слабого эффекта. Значимость в этой ветке
    даёт только сдвиговый null (:func:`fault_agreement`).
    """
    rx = rank_residual(x, controls)
    ry = rank_residual(y, controls)
    ok = np.isfinite(rx) & np.isfinite(ry)
    if ok.sum() < config.FZ_MIN_RESIDUAL_MARGIN:
        return float("nan")
    return float(np.corrcoef(rx[ok], ry[ok])[0, 1])


def fault_corridor_cells(dist_to_fault: np.ndarray, corridor_m: float,
                         pool: np.ndarray | None = None) -> np.ndarray:
    """Индексы ячеек разломного коридора — «точки» метрики заверки.

    Разломы в этих данных уже полигоны-коридоры (см.
    :func:`src.lineaments_sat.geometry_azimuths`), а ``dist_tect*`` — расстояние
    до них, поэтому ``corridor_m = 0`` означает «внутри коридора карты», а
    большие пороги добавляют окрестность: оба источника огрубляют геометрию
    по-своему, и совпадением разумно считать окрестность, а не пиксель в пиксель.
    Пороги те же, что в пороговой ветке (:data:`config.LIN_SAT_MATCH_M`), чтобы
    числа двух задач сравнивались напрямую.
    """
    d = np.asarray(dist_to_fault, dtype=float)
    on = np.isfinite(d) & (d <= corridor_m)
    if pool is not None:
        keep = np.zeros(d.shape, dtype=bool)
        keep[np.asarray(pool, dtype=int)] = True
        on &= keep
    return np.where(on)[0]


def fault_agreement(score: np.ndarray, meta, fault_idx: np.ndarray,
                    pool: np.ndarray | None = None, area: float | None = None,
                    n_shifts: int | None = None,
                    seed: int | None = None) -> dict[str, float]:
    """Согласие карты-кандидата с картой разломов + сдвиговый null.

    Тонкая обёртка над :func:`src.assessment.spatial_null_pvalue`: ячейки
    разломного коридора подставляются вместо точек заверки. Это не натяжка, а
    ровно то же вычисление — «какая доля целевых ячеек попала в верхние ``area``
    карты, делённая на реализованную площадь», — и оно даёт ветке ту же метрику
    и тот же тест значимости, что у всей фазы заверки, вместо самодельной доли
    «поймано/не поймано» без планки.

    Двигается КАНДИДАТ, карта разломов стоит: null отвечает на вопрос «а не даёт
    ли столько же любая карта с такой же зернистостью?».
    """
    from . import assessment

    return assessment.spatial_null_pvalue(
        score, fault_idx, meta, area=area or config.FZ_AREA, pool=pool,
        n_shifts=n_shifts or config.FZ_N_SHIFTS, seed=seed)


def relief_control_matrix(df: pd.DataFrame,
                          cols: tuple[str, ...] | None = None) -> np.ndarray | None:
    """Матрица контролей рельефа из датасета (отсутствующие колонки пропускаются)."""
    cols = cols or config.FZ_RELIEF_CONTROLS
    have = [c for c in cols if c in df.columns]
    if not have:
        return None
    return df[have].to_numpy(dtype=float)


def screen_candidate(score: np.ndarray, df: pd.DataFrame, meta,
                     fault_idx: np.ndarray, pool: np.ndarray,
                     controls: np.ndarray | None = None,
                     area: float | None = None,
                     n_shifts: int | None = None) -> dict[str, float]:
    """Полный протокол для одной карты-кандидата: сырая метрика, остаток, null.

    Возвращает обе строки сравнения сразу — «как есть» и «после вычитания
    рельефа», — потому что порознь они не читаются: высокий lift сырого
    кандидата на этом листе ожидаем у чего угодно, коррелирующего с
    расчленённостью рельефа, и смысл имеет только его судьба после контроля.
    """
    controls = relief_control_matrix(df) if controls is None else controls
    masked = np.asarray(score, dtype=float).copy()
    keep = np.zeros(masked.shape, dtype=bool)
    keep[np.asarray(pool, dtype=int)] = True
    masked[~keep] = np.nan

    raw = fault_agreement(masked, meta, fault_idx, pool=pool, area=area,
                          n_shifts=n_shifts)
    resid = rank_residual(masked, controls)
    res = fault_agreement(resid, meta, fault_idx, pool=pool, area=area,
                          n_shifts=n_shifts)
    return {
        "lift": raw["observed"], "p": raw["p"], "null_q95": raw["null_q95"],
        "lift_resid": res["observed"], "p_resid": res["p"],
        "null_q95_resid": res["null_q95"],
    }
