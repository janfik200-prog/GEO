"""Линеаменты по рельефу: многоазимутная отмывка -> края -> отрезки -> признаки.

Почему по рельефу, а не по оптике: под сплошным тундровым покровом (замер в
:mod:`src.s2_composite`: 98% пикселей — растительность) оптический «линеамент»
чаще всего оказывается границей растительного сообщества или тенью облака.
Рельеф от покрова свободен, а разломы и дайки на щите выражены уступами и
спрямлёнными долинами.

Почему многоазимутная отмывка: односторонняя подсветка усиливает структуры,
перпендикулярные лучу, и гасит параллельные — карта линеаментов по одному
азимуту измеряет направление солнца. Здесь края берутся по восьми азимутам и
объединяются, а расхождение между чётными и нечётными азимутами используется
как ПРОВЕРКА (:func:`azimuth_reproducibility`, стоп-правило ветки): если две
половины дают разные карты, линии — артефакт освещения.

Порядок: :func:`hillshade` -> :func:`edge_map` -> :func:`extract_lines`
(вероятностное преобразование Хафа) -> :func:`line_rasters` -> признаки на
ячейки :func:`lineament_features`.

ИЗМЕРЕННОЕ ОГРАНИЧЕНИЕ. Края отмывки — это прежде всего бровки долин, поэтому
скалярные признаки линеаментов во многом пересказывают расчленённость рельефа:
на листе ``lin_dist`` ~ ``dem_slope_std`` даёт Spearman -0.80, ``lin_dens`` ~
``dem_slope_std`` +0.67. Самостоятельную информацию несут ориентационные
признаки (``dir_sin``/``dir_cos``, связь с рельефом |rho| ≤ 0.08) и, слабее,
плотность узлов (0.39). Дублирование не алгебраическое (в отличие от
``dem_curv`` ~ ``dem_tpi_500``, r = -1.00), поэтому группа не выбраковывается —
решение принимает абляция на этапе 4.

Признаки (префикс ``lin_``): ``dens`` — длина линий на км²; ``node_dens`` —
плотность пересечений разноориентированных линий (узлы структурного каркаса,
классическая позиция рудных узлов); ``dist`` — расстояние до ближайшей линии;
``aniso_r`` — насколько направления в ячейке согласованы (0 — хаос, 1 — все
линии параллельны); ``dir_sin``/``dir_cos`` — само преобладающее направление
(через удвоенный угол: у линии нет «начала» и «конца», 10° и 190° — одно и то же).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, integro_grid, terrain_v2


def hillshade(elev: np.ndarray, azimuth_deg: float,
              altitude_deg: float | None = None,
              res_m: float | None = None) -> np.ndarray:
    """Отмывка рельефа при заданном азимуте и высоте солнца, значения 0..1."""
    res_m = res_m or config.TER_RES_M
    alt = np.radians(config.LIN_SUN_ALT if altitude_deg is None else altitude_deg)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    gy, gx = np.gradient(np.asarray(elev, dtype=float), res_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0.0, 1.0)


def edge_map(elev: np.ndarray, azimuths=None, res_m: float | None = None) -> np.ndarray:
    """Булева карта краёв: объединение краёв отмывок по всем азимутам."""
    from skimage.feature import canny

    azimuths = config.LIN_AZIMUTHS if azimuths is None else azimuths
    acc = np.zeros(np.shape(elev), dtype=bool)
    for az in azimuths:
        hs = hillshade(elev, az, res_m=res_m)
        # Пороги задаются В КВАНТИЛЯХ МОДУЛЯ ГРАДИЕНТА отмывки, а не в её
        # яркости: абсолютная яркость отмывки зависит от азимута, и общий порог
        # по яркости на одних азимутах не находит ничего, на других — всё.
        acc |= canny(hs, sigma=config.LIN_CANNY_SIGMA,
                     low_threshold=config.LIN_EDGE_LOW_PCTL / 100.0,
                     high_threshold=config.LIN_EDGE_PCTL / 100.0,
                     use_quantiles=True)
    return acc


def extract_lines(edges: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Прямолинейные отрезки из карты краёв (вероятностное преобразование Хафа)."""
    from skimage.transform import probabilistic_hough_line

    return probabilistic_hough_line(
        edges, threshold=config.LIN_HOUGH_THRESHOLD,
        line_length=config.LIN_MIN_LEN_PX, line_gap=config.LIN_MAX_GAP_PX,
        rng=config.LIN_SEED)


def _draw(segments, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Растеризация отрезков: длина, сумма sin(2a) и cos(2a) с весом длины."""
    from skimage.draw import line as draw_line

    length = np.zeros(shape, dtype=float)
    s2 = np.zeros(shape, dtype=float)
    c2 = np.zeros(shape, dtype=float)
    for (x0, y0), (x1, y1) in segments:
        rr, cc = draw_line(int(y0), int(x0), int(y1), int(x1))
        ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        rr, cc = rr[ok], cc[ok]
        ang = np.arctan2(y1 - y0, x1 - x0)          # ось, а не вектор -> удвоение угла
        length[rr, cc] += 1.0
        s2[rr, cc] += np.sin(2 * ang)
        c2[rr, cc] += np.cos(2 * ang)
    return length, s2, c2


def _azimuth_class(segments, n_classes: int = 4) -> np.ndarray:
    """Класс направления отрезка (по удвоенному углу), для поиска узлов."""
    ang = np.array([np.arctan2(y1 - y0, x1 - x0) for (x0, y0), (x1, y1) in segments])
    return np.floor(((np.degrees(2 * ang) % 360.0) / 360.0) * n_classes).astype(int)


def density_raster(segments, shape) -> np.ndarray:
    """Плотность линий (сумма длины в пикселях, без нормировки) — для сверки половин выборки."""
    return _draw(segments, shape)[0]


def line_rasters(segments, shape, res_m: float | None = None,
                 exact_nodes: bool = False) -> dict[str, np.ndarray]:
    """Растры линеаментов: длина, узлы, расстояние, ориентация.

    ``exact_nodes`` (по умолчанию выключено, чтобы прежние выходы, в том числе
    ``dataset_v5_rebuilt.parquet``, не менялись): ``uniform_filter`` считает
    скользящую сумму вычитанием и оставляет в строке после каждой линии хвост
    остатка округления ±1e-17. В ``node_dens`` это даёт 52% отрицательных и 3%
    положительных «почти нулей» вместо честного нуля, а любая свёртка по рангу
    превращает их в горизонтальные полосы (найдено 13.11.2026,
    ``experiments/gmo_lineaments_background.py``). При ``True`` сглаженный счёт
    округляется до кратного 1/25 — это точное значение среднего по окну 5x5
    из целых счётов пикселей.
    """
    from scipy.ndimage import distance_transform_edt, uniform_filter

    res_m = res_m or config.TER_RES_M
    length, s2, c2 = _draw(segments, shape)

    # Узел — место, где рядом проходят линии РАЗНЫХ направлений: одиночная
    # длинная линия узлов не создаёт, а пересечение двух систем — создаёт.
    cls = _azimuth_class(segments)
    per_class = []
    for c in np.unique(cls):
        sub = [s for s, k in zip(segments, cls) if k == c]
        sm = uniform_filter(_draw(sub, shape)[0], size=5)
        per_class.append(np.round(sm * 25.0) / 25.0 if exact_nodes else sm)
    nodes = np.zeros(shape)
    for i in range(len(per_class)):
        for j in range(i + 1, len(per_class)):
            nodes += np.minimum(per_class[i], per_class[j])

    dist = distance_transform_edt(length == 0) * res_m
    return {"length": length, "nodes": nodes, "dist": dist, "s2": s2, "c2": c2}


def azimuth_reproducibility(elev: np.ndarray, res_m: float | None = None,
                            smooth_px: int | None = None) -> dict[str, float]:
    """Согласие карт плотности линий по чётным и нечётным азимутам отмывки.

    Стоп-правило ветки: ниже ``LIN_MIN_SPEARMAN`` линии считаются артефактом
    освещения, а не структурой.

    Считается ТОЛЬКО там, где хотя бы одна из половин что-то нашла. На полной
    карте 90%+ пикселей пустые у обеих половин, и Spearman по такой матрице
    измеряет долю совместных нулей, а не согласие линий (первый прогон дал
    ровно 0.500 — характерный признак вырождения на связках).
    """
    from scipy.stats import spearmanr
    from scipy.ndimage import uniform_filter

    smooth_px = smooth_px or max(3, config.LIN_MIN_LEN_PX | 1)
    az = list(config.LIN_AZIMUTHS)
    maps = []
    for half in (az[::2], az[1::2]):
        segs = extract_lines(edge_map(elev, azimuths=half, res_m=res_m))
        maps.append(uniform_filter(_draw(segs, np.shape(elev))[0], size=smooth_px))
    a, b = maps[0].ravel(), maps[1].ravel()
    both = (a > 0) | (b > 0)
    return {"rho": float(spearmanr(a[both], b[both]).statistic),
            "covered_frac": float(both.mean()),
            "overlap_frac": float(((a > 0) & (b > 0)).sum() / max(1, both.sum()))}


def rasters_to_cell_features(meta: integro_grid.GridMeta, r: dict[str, np.ndarray],
                             prefix: str, res_m: float | None = None,
                             pad: int = 0) -> pd.DataFrame:
    """Растры :func:`line_rasters` -> признаки на ячейки сетки (общее для рельефа и снимков).

    ``pad`` — поле в пикселях рабочего растра ДО начала сетки (у рельефа рабочий
    растр шире целевого на :data:`config.TER_PAD_PX` для оконных операций
    отмывки; у спутниковых композитов рабочий растр посажен ровно под сетку,
    ``pad=0``).
    """
    res_m = res_m or config.TER_RES_M
    k = int(round(meta.dx / res_m))

    def blocks(a):
        core = a[pad:pad + meta.prf * k, pad:pad + meta.pic * k]
        return core.reshape(meta.prf, k, meta.pic, k)

    cell_km2 = (meta.dx / 1000.0) * (meta.dy / 1000.0)
    length_km = blocks(r["length"]).sum(axis=(1, 3)) * (res_m / 1000.0)
    s2 = blocks(r["s2"]).sum(axis=(1, 3))
    c2 = blocks(r["c2"]).sum(axis=(1, 3))
    n = blocks(r["length"]).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        aniso = np.hypot(s2, c2) / n
        dir_s, dir_c = s2 / np.hypot(s2, c2), c2 / np.hypot(s2, c2)
    out = {
        f"{prefix}dens": (length_km / cell_km2).ravel(),
        f"{prefix}node_dens": blocks(r["nodes"]).mean(axis=(1, 3)).ravel(),
        f"{prefix}dist": blocks(r["dist"]).mean(axis=(1, 3)).ravel(),
        f"{prefix}aniso_r": np.nan_to_num(aniso, nan=0.0).ravel(),
        f"{prefix}dir_sin": np.nan_to_num(dir_s, nan=0.0).ravel(),
        f"{prefix}dir_cos": np.nan_to_num(dir_c, nan=0.0).ravel(),
    }
    return pd.DataFrame(out)


def coherence_cells(meta: integro_grid.GridMeta, r: dict[str, np.ndarray],
                    res_m: float | None = None, pad: int = 0,
                    win_cells: int = 5, min_px: int = 20) -> np.ndarray:
    """Согласованность направлений линий в окне ``win_cells`` x ``win_cells`` ячеек.

    Замена ``aniso_r`` из :func:`rasters_to_cell_features` (тот остаётся как
    был). У прежнего канала два дефекта: ячейка с одной-единственной линией
    получает 1 («все линии параллельны» по построению — 11% ячеек листа), а
    ячейка без линий — 0, то есть «хаос», хотя направлений там просто нет
    (75% ячеек). Здесь суммы sin 2a, cos 2a и длины берутся по окну (по
    умолчанию 2.5 км — масштаб складки, а не ячейки), согласованность =
    |Σ(sin, cos)| / Σдлины, и там, где в окне меньше ``min_px`` пикселей линий
    (по умолчанию 2 км), канал не определён (NaN), а не ноль.

    Возвращает плоский массив по ячейкам сетки (C-порядок).
    """
    from scipy.ndimage import uniform_filter

    res_m = res_m or config.TER_RES_M
    k = int(round(meta.dx / res_m))

    def cell_sum(a):
        core = a[pad:pad + meta.prf * k, pad:pad + meta.pic * k]
        return core.reshape(meta.prf, k, meta.pic, k).sum(axis=(1, 3))

    w = win_cells * win_cells
    # Счёт целый, sin/cos — нет; окно считается как среднее x площадь окна.
    n = np.round(uniform_filter(cell_sum(r["length"]), size=win_cells, mode="constant") * w)
    s2 = uniform_filter(cell_sum(r["s2"]), size=win_cells, mode="constant") * w
    c2 = uniform_filter(cell_sum(r["c2"]), size=win_cells, mode="constant") * w
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.clip(np.hypot(s2, c2) / n, 0.0, 1.0)
    return np.where(n >= min_px, coh, np.nan).ravel()


def lineament_features(meta: integro_grid.GridMeta,
                       elev: np.ndarray | None = None) -> pd.DataFrame:
    """Признаки линеаментов на ячейки сетки (C-порядок, как датасет)."""
    res_m = config.TER_RES_M
    pad = config.TER_PAD_PX
    if elev is None:
        elev = terrain_v2.projected_dem(meta)
    elev = np.where(np.isfinite(elev), elev, np.nanmedian(elev))

    segments = extract_lines(edge_map(elev, res_m=res_m))
    r = line_rasters(segments, elev.shape, res_m)
    return rasters_to_cell_features(meta, r, "lin_", res_m=res_m, pad=pad)


LINEAMENT_COLS: tuple[str, ...] = ("lin_dens", "lin_node_dens", "lin_dist",
                                   "lin_aniso_r", "lin_dir_sin", "lin_dir_cos")
