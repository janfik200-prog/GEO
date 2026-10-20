"""Сроки схода снега по временному ряду Sentinel-1 — задача №2C.

ЧТО ЗДЕСЬ ДРУГОГО. Весь остальной спутниковый конвейер проекта сворачивает
выборку сцен в МЕДИАНУ (:func:`src.stac_grid.median_stack`) и тем самым
выбрасывает временную ось. Для сроков таяния временная ось — это и есть
измеряемая величина, поэтому медианный композит здесь неприменим в принципе, а
не «нежелателен».

ФИЗИКА. Сухой снег для C-диапазона почти прозрачен, и σ0 над ним близок к σ0
мёрзлого грунта. Как только в снеге появляется жидкая вода, поглощение резко
растёт и σ0 падает на 3+ дБ — это и есть подпись мокрого снега (Nagler & Rott
2000). После схода снега σ0 возвращается к летнему уровню. Сезонная кривая
пикселя выглядит так::

    σ0, дБ
      |‾‾‾\\                       ___---‾‾‾   <- лето, снега нет
      |    \\                 ___--
      | сухой \\____       __--
      | снег       \\____--                    <- возврат = сход снега
      |             \\__/                      <- минимум = пик мокрой фазы
      +--------------------------------> день года
        апрель      май-июнь        июль

Отсюда четыре величины на пиксель и год: день минимума (пик таяния), глубина
падения относительно апрельского уровня, день возврата к летнему уровню (сход
снега) и длительность мокрой фазы.

ЗАЧЕМ ЭТО ДЛЯ ТРЕЩИНОВАТОСТИ. Зона повышенной трещиноватости — это зона
повышенной проницаемости: другой дренаж, другое удержание талой воды, в мерзлоте
ещё и другая мощность деятельного слоя. Всё это сдвигает сроки и укорачивает или
удлиняет мокрую фазу. В отличие от яркости и её текстуры (блок 1 задачи №2,
закрыт отрицательно), величина описывает РЕЖИМ, а не уровень.

ОГОВОРКА, КОТОРУЮ НЕЛЬЗЯ ТЕРЯТЬ. От рельефа сроки таяния не свободны: высота,
экспозиция склона и надувы в понижениях управляют ими сильно. Поэтому вердикт
по-прежнему выносит остаток после вычитания рельефа
(:func:`src.fracture_zones.screen_candidate`), а не сырая метрика. Отличие от
блока 1 в том, что здесь механизм (дренаж) физически отличен от артефакта
(геометрия съёмки), то есть у остатка есть содержательный шанс выжить.

Порядок: :func:`bin_by_doy` -> :func:`melt_metrics` (один год) ->
:func:`aggregate_years` (медиана и межгодовой разброс) ->
:func:`melt_features` (признаки на ячейки сетки).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

#: Признаки ветки. Префикс ``sm_`` (snowmelt) с существующими группами не
#: пересекается: ``s1_``/``s2_`` отличаются вторым символом.
#:
#: Межгодового разброса даты (``sm_melt_doy_std``) здесь НЕТ, хотя он и
#: напрашивался. Причина измерена: датировать таяние по одному году удаётся лишь
#: в 11% пикселей (см. :func:`align_and_pool`), то есть разброс по годам был бы
#: определён на девятой части листа, а на остальных ячейках стал бы NaN. По
#: протоколу заверки NaN — промах, и такой признак получил бы механически
#: заниженный lift, несравнимый с остальными. Он вернётся, если появится
#: источник с полным покрытием кадра.
SNOWMELT_COLS: tuple[str, ...] = (
    "sm_melt_doy", "sm_recovery_doy", "sm_drop_db", "sm_wet_len", "sm_n_bins",
)


def bin_by_doy(doys: np.ndarray, cube: np.ndarray,
               bin_days: int | None = None,
               doy_range: tuple[int, int] | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Нерегулярный ряд сцен -> регулярная сезонная кривая по корзинам дня года.

    ``cube`` — ``(n_scenes, H, W)`` в дБ, ``doys`` — день года каждой сцены.
    Внутри корзины берётся МЕДИАНА: у радара спекл, и одиночная сцена с
    всплеском обратного рассеяния сдвинула бы среднее, но не медиану.

    ``doy_range`` задаёт ОБЩИЕ границы корзин. Без него границы берутся по
    краям самой выборки, и у разных лет они разъезжаются — сложить такие годы
    в одну кривую (:func:`align_and_pool`) уже нельзя. По умолчанию
    :data:`config.SM_DOY_RANGE`.

    Возвращает ``(центры корзин, куб (n_bins, H, W))``. Пустые корзины остаются
    NaN и заполнению не подлежат: пропуск в ряде — это отсутствие наблюдения, а
    не ноль.
    """
    import warnings

    bin_days = bin_days or config.SM_DOY_BIN
    doys = np.asarray(doys, dtype=int)
    cube = np.asarray(cube, dtype=float)
    lo, hi = doy_range or config.SM_DOY_RANGE
    edges = np.arange(lo, hi + bin_days, bin_days)
    centers = edges[:-1] + bin_days / 2.0
    out = np.full((len(centers), *cube.shape[1:]), np.nan)
    with warnings.catch_warnings():      # корзина без сцен -> NaN, это норма
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(len(centers)):
            sel = (doys >= edges[i]) & (doys < edges[i + 1])
            if sel.any():
                out[i] = np.nanmedian(cube[sel], axis=0)
    return centers, out


def sheet_melt_doy(centers: np.ndarray, cube: np.ndarray) -> float:
    """День года минимума СРЕДНЕЙ ПО ЛИСТУ кривой — синоптический сдвиг сезона.

    Год к году весна приходит раньше или позже на 1-2 недели, и это общий для
    всего листа сдвиг, а не свойство площадки. Он вычитается перед сведением
    годов (:func:`align_and_pool`), иначе разные годы размажут минимум и
    пространственный рисунок — то единственное, что нас интересует, — утонет в
    межгодовой изменчивости погоды.
    """
    curve = sheet_curve(cube)
    ok = np.isfinite(curve)
    if ok.sum() < config.SM_MIN_BINS:
        return float("nan")
    return float(np.asarray(centers)[int(np.argmin(np.where(ok, curve, np.inf)))])


def sheet_curve(cube: np.ndarray) -> np.ndarray:
    """Средняя по листу сезонная кривая, дБ; NaN там, где корзина пуста всюду."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(cube, dtype=float), axis=(1, 2))


def sheet_wet_amplitude(cube: np.ndarray) -> float:
    """Размах средней по листу кривой, дБ — есть ли у года мокрая фаза вообще.

    Год, у которого размах ниже :data:`config.SM_WET_DROP_DB`, таяния не видел:
    его сцены либо не попали на событие, либо не докачались. Опорную дату по
    такому году брать нельзя — минимум шумной полки не имеет отношения к весне,
    а сдвиг по нему подмешивает шум ровно в окно таяния остальных лет. Измерено
    на 2021: размах 1.8 дБ (отвалились по 403 все майско-июньские сцены S1B),
    «минимум» на 128-й день против 144-164 у полных лет.
    """
    curve = sheet_curve(cube)
    if np.isfinite(curve).sum() < config.SM_MIN_BINS:
        return float("nan")
    return float(np.nanmax(curve) - np.nanmin(curve))


def align_and_pool(binned_years: list[np.ndarray], melt_doys: list[float],
                   bin_days: int | None = None) -> np.ndarray:
    """Годы -> одна климатическая сезонная кривая, выровненная по сдвигу весны.

    ЗАЧЕМ. Измерено пилотом: одна сцена Sentinel-1 RTC покрывает медианно 10.9%
    листа, на пиксель за год приходится 7 заполненных корзин из 25, и датировать
    таяние по одному году нельзя (отсеивается 89% площади). Сведение пяти лет
    даёт ~19 корзин на пиксель — ряд, по которому дата уже определяется.

    КАК. Каждый год сдвигается на целое число корзин так, чтобы его средний по
    листу минимум встал на общую опорную дату (медиану по годам). Приехавшие с
    другого конца края обнуляются в NaN — циклический перенос здесь означал бы,
    что июль одного года выдаёт себя за апрель другого.

    ЧТО ЭТО ЗНАЧИТ ДЛЯ ИНТЕРПРЕТАЦИИ. После выравнивания дата таяния становится
    ОТНОСИТЕЛЬНОЙ: не «когда сошёл снег», а «насколько раньше или позже
    среднего по листу». Для поиска аномалий дренажа нужна именно она —
    абсолютная дата на 70% состоит из погоды конкретного года.
    """
    if not binned_years:
        raise ValueError("нет годов для сведения")
    bin_days = bin_days or config.SM_DOY_BIN
    md = np.asarray(melt_doys, dtype=float)
    if not np.isfinite(md).any():
        raise ValueError("ни у одного года не определён сдвиг сезона")
    ref = float(np.nanmedian(md))

    aligned = []
    for cube, m in zip(binned_years, md):
        a = np.asarray(cube, dtype=np.float32).copy()
        if np.isfinite(m):
            shift = int(round((ref - m) / bin_days))
            if shift:
                a = np.roll(a, shift, axis=0)
                if shift > 0:
                    a[:shift] = np.nan
                else:
                    a[shift:] = np.nan
        aligned.append(a)

    import warnings
    with warnings.catch_warnings():      # корзина пустая во всех годах -> NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(aligned), axis=0)


def _window_level(centers: np.ndarray, cube: np.ndarray,
                  window: tuple[int, int]) -> np.ndarray:
    """Средний уровень σ0 по корзинам, попавшим в окно дней года."""
    import warnings

    sel = (centers >= window[0]) & (centers <= window[1])
    if not sel.any():
        return np.full(cube.shape[1:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(cube[sel], axis=0)


def melt_metrics(centers: np.ndarray, cube: np.ndarray) -> dict[str, np.ndarray]:
    """Метрики таяния за ОДИН год по сезонной кривой ``(n_bins, H, W)`` в дБ.

    * ``melt_doy`` — день минимума σ0: пик мокрой фазы. Берётся как опорная
      точка, потому что не требует ни одного порога — в отличие от даты начала
      таяния, которая зависит от выбора «на сколько дБ упало»;
    * ``drop_db`` — насколько минимум ниже апрельского (сухого) уровня;
    * ``recovery_doy`` — первая корзина ПОСЛЕ минимума, где σ0 прошёл долю
      :data:`config.SM_RECOVERY_FRAC` пути от минимума к июльскому уровню; это
      и есть сход снега;
    * ``wet_len`` — длительность мокрой фазы: от первого пересечения порога
      :data:`config.SM_WET_DROP_DB` вниз до возврата.

    Пиксели, где корзин с данными меньше :data:`config.SM_MIN_BINS` либо не
    определён опорный уровень, возвращаются как NaN целиком: половина кривой не
    даёт даты, а даёт артефакт.
    """
    cube = np.asarray(cube, dtype=float)
    centers = np.asarray(centers, dtype=float)
    shape = cube.shape[1:]
    nan = lambda: np.full(shape, np.nan)  # noqa: E731
    out = {k: nan() for k in ("melt_doy", "drop_db", "recovery_doy", "wet_len")}

    n_obs = np.isfinite(cube).sum(axis=0)
    ref = _window_level(centers, cube, config.SM_REF_DOY)
    summer = _window_level(centers, cube, config.SM_SUMMER_DOY)
    ok = (n_obs >= config.SM_MIN_BINS) & np.isfinite(ref) & np.isfinite(summer)
    if not ok.any():
        return out

    # nanargmin падает на полностью пустом столбце, поэтому пустые заполняются
    # +inf: такой пиксель всё равно отсеян маской ok.
    filled = np.where(np.isfinite(cube), cube, np.inf)
    imin = np.argmin(filled, axis=0)
    vmin = np.take_along_axis(cube, imin[None], axis=0)[0]
    ok &= np.isfinite(vmin)

    melt_doy = centers[imin]
    drop_db = ref - vmin

    # Возврат к летнему уровню: порог у каждого пикселя свой, потому что
    # амплитуда падения разная, а доля пути от неё не зависит.
    level = vmin + config.SM_RECOVERY_FRAC * (summer - vmin)
    after = np.arange(cube.shape[0])[:, None, None] > imin[None]
    reached = after & np.isfinite(cube) & (cube >= level[None])
    has_rec = reached.any(axis=0)
    irec = np.argmax(reached, axis=0)
    recovery_doy = np.where(has_rec, centers[irec], np.nan)

    # Начало мокрой фазы: первое пересечение порога Наглера вниз.
    wet = np.isfinite(cube) & (cube <= (ref - config.SM_WET_DROP_DB)[None])
    has_wet = wet.any(axis=0)
    ionset = np.argmax(wet, axis=0)
    onset_doy = np.where(has_wet, centers[ionset], np.nan)
    wet_len = recovery_doy - onset_doy

    for key, val in (("melt_doy", melt_doy), ("drop_db", drop_db),
                     ("recovery_doy", recovery_doy), ("wet_len", wet_len)):
        out[key] = np.where(ok, val, np.nan)
    return out


def pooled_rasters(centers: np.ndarray, pooled: np.ndarray) -> dict[str, np.ndarray]:
    """Климатическая кривая :func:`align_and_pool` -> именованные растры ветки.

    ``sm_n_bins`` — сколько корзин сезона реально заполнено у пикселя. Это
    служебная величина (качество ряда, а не свойство площади), и в кандидаты
    протокола она не идёт; печатается, чтобы было видно, на каком ряде получены
    остальные признаки.
    """
    m = melt_metrics(centers, pooled)
    out = {f"sm_{k}": v for k, v in m.items()}
    out["sm_n_bins"] = np.isfinite(pooled).sum(axis=0).astype(float)
    return out


def aggregate_years(per_year: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Межгодовая сводка: медиана по годам и разброс даты минимума.

    Медиана, а не среднее: год с ранней или поздней весной — это реальное
    событие, но нас интересует типичный режим площадки, а не конкретный сезон.
    Разброс ``melt_doy_std`` — самостоятельный признак: площадка со стабильным
    сроком таяния ведёт себя иначе, чем та, где срок скачет от года к году.

    Пиксели, где валидных лет меньше :data:`config.SM_MIN_YEARS`, обнуляются в
    NaN: по двум годам разброс — это не разброс.
    """
    import warnings

    if not per_year:
        raise ValueError("нет годовых метрик для сводки")
    keys = ("melt_doy", "drop_db", "recovery_doy", "wet_len")
    stacks = {k: np.stack([y[k] for y in per_year]) for k in keys}
    n_years = np.isfinite(stacks["melt_doy"]).sum(axis=0)
    enough = n_years >= config.SM_MIN_YEARS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {f"sm_{k}": np.where(enough, np.nanmedian(stacks[k], axis=0), np.nan)
               for k in keys}
        out["sm_melt_doy_std"] = np.where(
            enough, np.nanstd(stacks["melt_doy"], axis=0), np.nan)
    out["sm_n_years"] = n_years.astype(float)
    return out


def melt_features(meta, rasters: dict[str, np.ndarray],
                  res_m: float | None = None) -> pd.DataFrame:
    """Растры :func:`aggregate_years` -> признаки на ячейки сетки (C-порядок).

    Агрегация — та же :func:`src.stac_grid.to_cells`, что у всех спутниковых
    групп, поэтому среднее и субъячеечный разброс считаются здесь так же, как у
    ``s1_``/``psr_``. Разброс срока таяния ВНУТРИ ячейки 500 м осмыслен сам по
    себе: он показывает, насколько резко срок меняется на коротком расстоянии —
    а резкая граница срока и есть то, как зона дренажа должна выглядеть.
    """
    from . import stac_grid

    return stac_grid.to_cells(rasters, meta, res_m=res_m or config.SM_RES_M,
                              prefix="")
