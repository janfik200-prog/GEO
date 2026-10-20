"""Накопление вдоль линии: сколько даёт длина, если детектор построен вокруг неё.

Продолжение шага 0 (:mod:`src.detection_limit`) тем же прибором. Шаг 0 измерил
порог обнаружения 1.48-1.81 sigma при контрасте настоящих коридоров карты
0.086-0.194 sigma и заодно показал, ЧТО этот порог двигает: ширина, источник и
профиль давали доли, а длина — почти вдвое (2.86 sigma при 1 км, 2.28 при 3 км,
1.47 при 10 км). При этом оба детектора шага 0 длиной почти не пользуются: Кэнни
решает попиксельно, а когерентность тензора усредняет в ИЗОТРОПНОМ окне, то есть
набирает площадь, а не длину. Здесь добавлены два детектора, для которых длина —
основная величина.

ДВА ДЕТЕКТОРА СТОЯТ ПО КРАЯМ ШКАЛЫ ЧЕСТНОСТИ, и в этом весь смысл пары.

* :func:`matched_response` — согласованный фильтр, которому СООБЩЕНЫ положение,
  азимут, ширина и профиль вставки. Это не метод, а ВЕРХНЯЯ ГРАНИЦА: лучше него
  не может быть ни один линейный детектор, потому что искать уже нечего, вся
  задача сведена к сложению яркостей по известной маске. Граница нужна, чтобы
  отделить «наши детекторы плохи» от «информации такой величины в данных нет»;
* :func:`linked_response` — сращивание фрагментов, детектор полностью СЛЕПОЙ:
  о вставке не знает ничего, как и Хаф шага 0. Порог на пиксель ослаблен вдвое
  (``config.ACC_EDGE_*``), зато от фрагментов требуется выстроиться в длинную
  коллинеарную цепочку. Это то, что в литературе называют segment tracing.

Читать их надо вместе:

* слепой провалился, граница взяла -> дело в ПЕРЕБОРЕ, а не в сигнале, и метод
  искать стоит;
* провалилась и граница -> перебирать нечего, вся семья «накопление вдоль
  линии» закрыта разом, и закрыта не мнением, а измерением.

ПОЧЕМУ ГРАНИЦА МОЖЕТ НЕ ДАТЬ ОЖИДАЕМОГО. Наивно накопление по ``N`` пикселям
линии должно давить шум в ``sqrt(N)`` раз: 3 км при шаге 100 м — это 30 отсчётов
поперёк, то есть выигрыш до 5.5 раз, чего почти хватило бы, чтобы закрыть разрыв
между 1.5 и 0.19 sigma. Но ``sqrt(N)`` верен только для НЕЗАВИСИМЫХ отсчётов, а
композит — не белый шум: у него есть собственная геологическая и растительная
текстура с километровой корреляцией. Сколько независимости на самом деле —
величина измеримая, а не предмет спора: статистика :func:`matched_response`
нормирована так, что на независимом фоне её нуль-модель обязана дать квантиль
стандартного нормального. Насколько реальный квантиль больше — ровно во столько
раз (в квадрате) меньше эффективное число отсчётов, см.
:func:`independence_ratio`. Это и есть главное число, которое здесь добывается:
оно ограничивает не один детектор, а весь класс.

ЭТО ПРИБОР, А НЕ ГИПОТЕЗА — как и шаг 0. Ни одного p-value против карты
разломов, точки заверки не участвуют, знаменатель Бонферрони
(``config.FZ_N_TESTS_PRIOR`` и далее) НЕ растёт.
"""
from __future__ import annotations

import numpy as np

from . import config, detection_limit as dl


# --------------------------------------------------------------------------
# Согласованный фильтр — верхняя граница
# --------------------------------------------------------------------------

def matched_response(image: np.ndarray, center_rc: tuple[float, float],
                     axis_deg: float, *, length_m: float, width_m: float,
                     res_m: float, profile: str | None = None,
                     flank_inner_m: float | None = None,
                     flank_m: float | None = None) -> dict[str, float]:
    """Отклик согласованного фильтра: накопленный контраст в единицах шума.

    Складывает яркость по маске самого объекта и вычитает две полосы сравнения
    по обе стороны от него, после чего делит на СОБСТВЕННЫЙ разброс этой
    разности. Форма вычитания повторяет профиль врезки:

    * ``ridge`` — жёлоб симметричен, поэтому ``среднее на объекте`` минус
      полусумма полос;
    * ``scarp`` — уступ антисимметричен, у него на объекте перепада нет вовсе
      (маска накрывает оба борта), поэтому вычитаются полосы друг из друга.

    Модуль, а не знак: настоящий разлом может быть и светлее, и темнее фона, а
    :func:`src.detection_limit.corridor_contrast`, с которым сравнивается
    результат, тоже берёт модуль. Нуль-модель считается тем же модулем, так что
    проверка получается двусторонней.

    Разброс берётся по ФОНУ (дальше ``config.DL_BACKGROUND_PAD_M`` от отрезка), а
    не по всей вырезке: считай его по всему окну — сильная вставка раздувала бы
    собственный знаменатель и гасила себя же.

    Возвращает:

    * ``acc_matched`` — сама статистика. Нормирована так, что на независимом
      гауссовом фоне это ``|N(0, 1)|``: отклонение нуль-модели от этого и есть
      мера связности фона (:func:`independence_ratio`);
    * ``acc_nominal_n`` — сколько независимых отсчётов статистика СЧИТАЕТ, что
      набрала. Нужен, чтобы перевести долю независимости в эффективное число
      отсчётов.
    """
    profile = config.ACC_PROFILE_PRIMARY if profile is None else profile
    if profile not in config.DL_PROFILES:
        raise ValueError(f"неизвестный профиль {profile!r}, "
                         f"допустимы {config.DL_PROFILES}")
    inner = config.ACC_FLANK_INNER_M if flank_inner_m is None else flank_inner_m
    band = config.ACC_FLANK_M if flank_m is None else flank_m

    img = np.asarray(image, dtype=float)
    s, t = dl.track_coords(img.shape, center_rc, axis_deg, res_m)
    half_l, half_w = length_m / 2.0, width_m / 2.0
    lo_edge, hi_edge = half_w + inner, half_w + inner + band

    fin = np.isfinite(img)
    along = np.abs(s) <= half_l
    on = along & (np.abs(t) <= half_w + res_m) & fin
    lo = along & (t <= -lo_edge) & (t >= -hi_edge) & fin
    hi = along & (t >= lo_edge) & (t <= hi_edge) & fin

    dist = np.hypot(np.clip(np.abs(s) - half_l, 0.0, None), t)
    back = (dist > config.DL_BACKGROUND_PAD_M) & fin

    n_on, n_lo, n_hi = int(on.sum()), int(lo.sum()), int(hi.sum())
    sigma_loc = dl.robust_sigma(img[back]) if back.sum() >= 2 else float("nan")
    if profile == "ridge":
        enough = n_on >= 2 and n_lo >= 2 and n_hi >= 2
        var_factor = (1.0 / max(n_on, 1) + 0.25 / max(n_lo, 1) + 0.25 / max(n_hi, 1))
    else:
        enough = n_lo >= 2 and n_hi >= 2
        var_factor = 1.0 / max(n_lo, 1) + 1.0 / max(n_hi, 1)
    if not enough or not np.isfinite(sigma_loc) or sigma_loc <= 0:
        return {"acc_matched": float("nan"), "acc_nominal_n": float("nan")}

    if profile == "ridge":
        num = img[on].mean() - 0.5 * (img[lo].mean() + img[hi].mean())
    else:
        num = img[hi].mean() - img[lo].mean()
    return {"acc_matched": float(abs(num) / (sigma_loc * np.sqrt(var_factor))),
            "acc_nominal_n": float(1.0 / var_factor)}


def independence_ratio(null_threshold: float, q: float | None = None) -> float:
    """Какая доля номинальной независимости реально есть у фона композита.

    :func:`matched_response` нормирована под ``|N(0, 1)|``, поэтому на белом шуме
    квантиль её нуль-модели равен квантилю модуля стандартного нормального. Если
    измеренный квантиль в ``k`` раз больше, значит шум накапливался в ``k`` раз
    хуже, чем при независимости, то есть эффективных отсчётов в ``k^2`` раз
    меньше номинальных. Умножив на ``acc_nominal_n``, получаем, сколько
    независимых точек реально даёт линия — число, ограничивающее ВЕСЬ класс
    накопительных детекторов, а не один этот.
    """
    from scipy.stats import norm

    q = config.DL_NULL_Q if q is None else q
    thr = float(null_threshold)
    if not np.isfinite(thr) or thr <= 0:
        return float("nan")
    ideal = float(norm.ppf(0.5 * (1.0 + q)))       # квантиль |N(0, 1)|
    return float((ideal / thr) ** 2)


# --------------------------------------------------------------------------
# Сращивание фрагментов — слепой детектор
# --------------------------------------------------------------------------

def fragments(image: np.ndarray):
    """Короткие отрезки по ослабленному вдвое конвейеру Кэнни -> Хаф.

    Отличие от :func:`src.lineaments_sat.lines_from_image` только в порогах
    (``config.ACC_*``): здесь отрезок нужен не как готовый линеамент, а как
    ФРАГМЕНТ под сращивание. Самые длинные ``config.ACC_MAX_FRAGMENTS``
    оставляются, остальные отбрасываются — потолок применяется одинаково к
    сигналу и к нуль-модели.
    """
    from . import lineaments, lineaments_sat

    edges = lineaments_sat.edges_from_image(image,
                                            low_pctl=config.ACC_EDGE_LOW_PCTL,
                                            high_pctl=config.ACC_EDGE_PCTL)
    segs = lineaments.extract_lines(edges, threshold=config.ACC_HOUGH_THRESHOLD,
                                    line_length=config.ACC_MIN_LEN_PX,
                                    line_gap=config.ACC_MAX_GAP_PX)
    if len(segs) <= config.ACC_MAX_FRAGMENTS:
        return segs
    order = sorted(segs, key=lambda s: -np.hypot(s[1][0] - s[0][0],
                                                 s[1][1] - s[0][1]))
    return order[:config.ACC_MAX_FRAGMENTS]


def _segment_arrays(segments):
    """Отрезки -> массивы концов, середин, длин и осей в растровых координатах.

    Координаты Хафа — ``(col, row)``, поэтому здесь они переставляются в
    ``(row, col)``: вся геометрия ветки (:func:`src.detection_limit.track_coords`,
    тензор) работает в ``(row, col)``, и смешивать две системы нельзя.
    """
    p0 = np.array([[y0, x0] for (x0, y0), _ in segments], dtype=float)
    p1 = np.array([[y1, x1] for _, (x1, y1) in segments], dtype=float)
    d = p1 - p0
    return (p0, p1, 0.5 * (p0 + p1), np.hypot(d[:, 0], d[:, 1]),
            np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 180.0)


def _mean_axis_deg(axis_deg: np.ndarray, weights: np.ndarray) -> float:
    """Средняя ОСЬ набора отрезков: удвоенный угол, вес — длина.

    Обычное среднее углов здесь неверно: 179° и 1° — почти одно направление, а
    их среднее дало бы 90°. Удвоение угла снимает эту неоднозначность (та же
    техника, что в :func:`src.fracture_zones.structure_tensor`).
    """
    a = np.radians(np.asarray(axis_deg, dtype=float))
    w = np.asarray(weights, dtype=float)
    ang = 0.5 * np.arctan2(float((w * np.sin(2 * a)).sum()),
                           float((w * np.cos(2 * a)).sum()))
    return dl._canonical_axis(np.degrees(ang))


def link_segments(segments, res_m: float, *, tol_deg: float | None = None,
                  offset_m: float | None = None, gap_m: float | None = None
                  ) -> list[dict]:
    """Сращивание коллинеарных фрагментов в цепочки.

    Два фрагмента попадают в одну цепочку, когда выполнены все три условия
    сразу, и все три взяты из уже опубликованных порогов ветки:

    * оси различаются не больше ``config.ACC_LINK_TOL_DEG`` (лепесток розы);
    * поперечное расстояние между их серединами не больше
      ``config.ACC_LINK_OFFSET_M`` (нижний масштаб тензора);
    * разрыв ВДОЛЬ общей оси между их проекциями не больше
      ``config.ACC_LINK_GAP_M`` (допуск, при котором задача №8 уже называла
      автоматическую линию и разлом карты одним объектом).

    Связность транзитивна: цепочка — компонента связности этого отношения, то
    есть длинная линия собирается из фрагментов, попарно соседних, даже если её
    концы друг с другом напрямую не сцепляются. Это и есть накопление длины.

    Возвращает по цепочке: ось, центр, протяжённость вдоль оси (``span_m``) и
    суммарную длину членов (``cover_m``). ``span_m`` — то, что делает цепочку
    длинной; ``cover_m`` показывает, какая доля этой протяжённости реально
    подтверждена фрагментами, а какая набрана перекрытием разрывов.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if not segments:
        return []
    tol_deg = config.ACC_LINK_TOL_DEG if tol_deg is None else tol_deg
    offset_px = (config.ACC_LINK_OFFSET_M if offset_m is None else offset_m) / res_m
    gap_px = (config.ACC_LINK_GAP_M if gap_m is None else gap_m) / res_m

    p0, p1, mid, length, axis = _segment_arrays(segments)
    n = len(segments)

    d = np.abs(axis[:, None] - axis[None, :]) % 180.0
    ok = np.minimum(d, 180.0 - d) <= tol_deg

    # Общая ось пары — по удвоенному углу, с весом длины: симметрична по (i, j),
    # поэтому проекции ниже можно брать транспонированием.
    a2 = np.radians(axis)
    s2 = length * np.sin(2 * a2)
    c2 = length * np.cos(2 * a2)
    ang = 0.5 * np.arctan2(s2[:, None] + s2[None, :], c2[:, None] + c2[None, :])
    u_r, u_c = np.sin(ang), np.cos(ang)
    n_r, n_c = np.cos(ang), -np.sin(ang)

    dr = mid[:, 0][:, None] - mid[:, 0][None, :]
    dc = mid[:, 1][:, None] - mid[:, 1][None, :]
    ok &= np.abs(dr * n_r + dc * n_c) <= offset_px

    def proj(p):
        return p[:, 0][:, None] * u_r + p[:, 1][:, None] * u_c

    a0, a1 = proj(p0), proj(p1)
    i_lo, i_hi = np.minimum(a0, a1), np.maximum(a0, a1)
    j_lo, j_hi = i_lo.T, i_hi.T
    ok &= np.maximum(0.0, np.maximum(i_lo, j_lo) - np.minimum(i_hi, j_hi)) <= gap_px

    n_comp, labels = connected_components(csr_matrix(ok), directed=False)
    chains = []
    for k in range(n_comp):
        idx = np.where(labels == k)[0]
        ax = _mean_axis_deg(axis[idx], length[idx])
        w = length[idx]
        center = np.average(mid[idx], axis=0, weights=w if w.sum() > 0 else None)
        a = np.radians(ax)
        ends = np.vstack([p0[idx], p1[idx]]) - center
        s = ends[:, 0] * np.sin(a) + ends[:, 1] * np.cos(a)
        chains.append({"axis_deg": ax, "center_rc": (float(center[0]), float(center[1])),
                       "span_m": float((s.max() - s.min()) * res_m),
                       "cover_m": float(w.sum() * res_m),
                       "n_segments": int(idx.size)})
    return chains


def linked_response(image: np.ndarray, center_rc: tuple[float, float],
                    axis_deg: float, *, length_m: float, res_m: float,
                    tol_m: float | None = None, tol_deg: float | None = None
                    ) -> dict[str, float]:
    """Отклик слепого детектора: длиннейшая цепочка, попавшая во вставку.

    Правило попадания — то же, что у :func:`src.detection_limit.hough_response`
    (ось в пределах ``tol_deg``, центр не дальше ``tol_m`` поперёк и
    ``половина длины + tol_m`` вдоль), чтобы два слепых детектора сравнивались
    по одной мерке и разница между ними была разницей ДЕТЕКТОРОВ, а не сверки.

    Статистика — ``acc_link_span`` в метрах, а не счётчик: накопление меряется
    длиной, и порог по длине различает «собралась цепочка на весь объект» и
    «нашёлся случайный обрывок нужного азимута», чего счётчик не различает.
    """
    tol_m = config.DL_MATCH_TOL_M if tol_m is None else tol_m
    tol_deg = config.DL_MATCH_TOL_DEG if tol_deg is None else tol_deg

    segs = fragments(image)
    chains = link_segments(segs, res_m)
    half = length_m / 2.0
    a = np.radians(axis_deg)
    u_r, u_c = np.sin(a), np.cos(a)
    n_r, n_c = np.cos(a), -np.sin(a)

    best, n_match = 0.0, 0
    for ch in chains:
        if dl.axis_difference_deg(ch["axis_deg"], axis_deg) > tol_deg:
            continue
        dr = (ch["center_rc"][0] - center_rc[0]) * res_m
        dc = (ch["center_rc"][1] - center_rc[1]) * res_m
        s = dr * u_r + dc * u_c
        t = dr * n_r + dc * n_c
        if abs(t) <= tol_m and abs(s) <= half + tol_m:
            n_match += 1
            best = max(best, ch["span_m"])
    return {"acc_link_span": float(best), "acc_link_n_match": float(n_match),
            "acc_link_n_chains": float(len(chains)),
            "acc_link_n_frag": float(len(segs))}


# --------------------------------------------------------------------------
# Одна врезка: все четыре детектора на одном и том же окне
# --------------------------------------------------------------------------

def probe_once(image: np.ndarray, center_rc: tuple[int, int], axis_deg: float, *,
               amplitude: float, length_m: float, width_m: float, res_m: float,
               sigma: float, profile: str | None = None,
               scales_m: tuple[float, ...] | None = None) -> dict[str, float]:
    """Врезка -> отклики старых детекторов шага 0 И накопительных, на одном окне.

    Окно со вставкой готовится один раз (:func:`src.detection_limit
    .prepare_probe`) и раздаётся всем четырём детекторам. Считай каждый на своей
    копии — разница порогов между детекторами включала бы разницу площадок, и
    сравнение перестало бы отвечать на вопрос ветки.
    """
    profile = config.ACC_PROFILE_PRIMARY if profile is None else profile
    hit, ctr = dl.prepare_probe(image, center_rc, axis_deg, amplitude=amplitude,
                                length_m=length_m, width_m=width_m, res_m=res_m,
                                sigma=sigma, profile=profile)
    row = dl.tensor_response(hit, ctr, axis_deg, length_m=length_m,
                             width_m=width_m, res_m=res_m, scales_m=scales_m)
    row.update(dl.hough_response(hit, ctr, axis_deg, length_m=length_m,
                                 res_m=res_m))
    row.update(matched_response(hit, ctr, axis_deg, length_m=length_m,
                                width_m=width_m, res_m=res_m, profile=profile))
    row.update(linked_response(hit, ctr, axis_deg, length_m=length_m,
                               res_m=res_m))
    return row
