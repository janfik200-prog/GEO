"""Линеаменты напрямую по космоснимку — задача №8 реестра.

:mod:`src.lineaments` считает линии по РЕЛЬЕФУ — осознанное отступление от
постановки (задача №8, «выделить разломы (линеаменты) по космоснимку»),
обоснованное измерением: на листе 98.3% пикселей — растительность
(:mod:`src.s2_composite`), и оптический «линеамент» там чаще всего граница
растительного сообщества, а не структура. Но постановка прямо говорит «по
космоснимку», значит нужно посчитать и измерить, а не сослаться на ожидание.

Здесь тот же конвейер (Кэнни -> вероятностный Хаф -> растры -> признаки на
ячейки, всё переиспользуется из :mod:`src.lineaments`) прогоняется НАПРЯМУЮ на
композитах четырёх съёмок:

* ``s2`` — Sentinel-2, серый = среднее видимых+NIR каналов композита (ожидаемо
  отрицательный, но ИЗМЕРЕННЫЙ результат — растительность вместо структуры);
* ``l8`` — Landsat 8/9, красный канал (панхроматического канала в наборе нет,
  разрешение хуже, 30 м вместо 100 м рабочего шага — ограничение честно видно
  в результате);
* ``s1`` — Sentinel-1 RTC, VV в дБ (C-диапазон радара, видит геометрию
  поверхности сквозь покров, не зависит от освещения и облаков; в постановке
  не значится, но съёмка уже скачана для датасета);
* ``psr`` — ALOS PALSAR-2, HH в дБ (L-диапазон, глубже проникает сквозь
  покров, структурный каркас должен читаться чище, чем у C-диапазона).

Отличие от рельефной ветки — только в карте краёв. У рельефа объединяются
отмывки по 8 азимутам, а стоп-правило сравнивает чётные/нечётные азимуты
(односторонняя подсветка усиливает одни структуры и гасит другие). У
композита нет «азимута подсветки» — вместо этого выборка сцен делится на две
НЕЗАВИСИМЫЕ ПОЛОВИНЫ (чётные/нечётные по дате) и строятся два независимых
композита: если линии не переживают смену половины выборки, это шум
конкретных дат (застрявшее облако, ледостав, всплеск обратного рассеяния), а
не структура поверхности. Порог тот же :data:`config.LIN_MIN_SPEARMAN`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, lineaments, s2_composite, sat_sources, stac_grid


def split_even_odd(items, key):
    """Выборка -> две независимые половины (чётные/нечётные по ``key``, обычно дате)."""
    order = sorted(items, key=key)
    return order[0::2], order[1::2]


def s2_grayscale(meta, scenes, res_m: float | None = None):
    """Композит Sentinel-2 -> яркость (среднее b02/b03/b04/b8a)."""
    res_m = res_m or config.S2_RES_M
    comp, n_obs = s2_composite.median_composite(scenes, meta, res_m)
    bands = np.stack([comp["b02"], comp["b03"], comp["b04"], comp["b8a"]])
    with np.errstate(all="ignore"):
        gray = np.nanmean(bands, axis=0)
    return gray, res_m, n_obs


def _stac_composite(meta, items, bands, cache_dir, res_m, mask_fn=None, nodata=None,
                    min_obs=None):
    res_m = res_m or config.SAT_RES_M
    scenes = [stac_grid.scene_on_grid(it, bands, meta, cache_dir, res_m=res_m,
                                      mask_fn=mask_fn, nodata=nodata)
             for it in items]
    return stac_grid.median_stack(scenes, min_obs=min_obs)


def l8_grayscale(meta, items, res_m: float | None = None, min_obs: int | None = None):
    """Композит Landsat 8/9 -> яркость (канал red; панхроматического канала нет).

    ``min_obs`` — переопределить порог наблюдений на пиксель (по умолчанию
    :data:`config.SAT_MIN_OBS`); нужен ниже при делении выборки пополам для
    проверки воспроизводимости — иначе половинная выборка почти вся уйдёт в NaN.
    """
    from cache_paths import L8_ANABAR

    res_m = res_m or config.SAT_RES_M
    comp, n_obs = _stac_composite(meta, items, sat_sources.L8_BANDS, L8_ANABAR,
                                  res_m, mask_fn=sat_sources.l8_mask, min_obs=min_obs)
    return comp["red"], res_m, n_obs


def s1_grayscale(meta, items, res_m: float | None = None, min_obs: int | None = None):
    """Композит Sentinel-1 RTC -> VV в дБ (см. ``min_obs`` в :func:`l8_grayscale`)."""
    from cache_paths import S1_ANABAR

    res_m = res_m or config.SAT_RES_M
    comp, n_obs = _stac_composite(meta, items, sat_sources.S1_BANDS, S1_ANABAR,
                                  res_m, nodata=config.S1_NODATA, min_obs=min_obs)
    return stac_grid.to_db(comp["vv"]), res_m, n_obs


def psr_grayscale(meta, items, res_m: float | None = None, min_obs: int | None = None):
    """Композит ALOS PALSAR-2 -> HH в дБ (см. ``min_obs`` в :func:`l8_grayscale`)."""
    from cache_paths import PSR_ANABAR

    res_m = res_m or config.SAT_RES_M
    comp, n_obs = _stac_composite(meta, items, sat_sources.PSR_BANDS, PSR_ANABAR,
                                  res_m, nodata=0.0, min_obs=min_obs)
    dn = comp["hh"]
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 20.0 * np.log10(np.where(dn > 1, dn, np.nan)) + config.PSR_CAL_OFFSET
    return np.where(np.isfinite(db), db, np.nan), res_m, n_obs


def edges_from_image(image: np.ndarray) -> np.ndarray:
    """Карта краёв Кэнни прямо по композиту (без многоазимутного объединения)."""
    from skimage.feature import canny

    finite = np.isfinite(image)
    if finite.sum() < 100:
        return np.zeros(image.shape, dtype=bool)
    filled = np.where(finite, image, np.nanmedian(image[finite]))
    return canny(filled, sigma=config.LIN_CANNY_SIGMA,
                low_threshold=config.LIN_EDGE_LOW_PCTL / 100.0,
                high_threshold=config.LIN_EDGE_PCTL / 100.0,
                use_quantiles=True, mask=finite)


def lines_from_image(image: np.ndarray):
    """Отрезки (вероятностный Хаф) по одному композиту."""
    return lineaments.extract_lines(edges_from_image(image))


def reproducibility(image_a: np.ndarray, image_b: np.ndarray,
                    smooth_px: int | None = None) -> dict[str, float]:
    """Согласие карт плотности линий между двумя независимыми половинами выборки сцен.

    Стоп-правило ветки (см. :mod:`src.lineaments`.azimuth_reproducibility для
    рельефного аналога): считается только там, где хотя бы одна половина
    что-то нашла, иначе метрика измеряет долю совместных нулей.
    """
    from scipy.ndimage import uniform_filter
    from scipy.stats import spearmanr

    smooth_px = smooth_px or max(3, config.LIN_MIN_LEN_PX | 1)
    da = uniform_filter(lineaments.density_raster(lines_from_image(image_a), image_a.shape),
                        size=smooth_px)
    db = uniform_filter(lineaments.density_raster(lines_from_image(image_b), image_b.shape),
                        size=smooth_px)
    a, b = da.ravel(), db.ravel()
    both = (a > 0) | (b > 0)
    if both.sum() < 10:
        return {"rho": float("nan"), "covered_frac": float(both.mean()), "overlap_frac": 0.0}
    return {"rho": float(spearmanr(a[both], b[both]).statistic),
           "covered_frac": float(both.mean()),
           "overlap_frac": float(((a > 0) & (b > 0)).sum() / max(1, both.sum()))}


def source_features(meta, image: np.ndarray, res_m: float, prefix: str
                    ) -> tuple[pd.DataFrame, list]:
    """Признаки на ячейки сетки + сырые отрезки (для розы азимутов и превью)."""
    segments = lines_from_image(image)
    r = lineaments.line_rasters(segments, image.shape, res_m)
    feats = lineaments.rasters_to_cell_features(meta, r, prefix, res_m=res_m, pad=0)
    return feats, segments


def segment_azimuths(segments) -> tuple[np.ndarray, np.ndarray]:
    """Географический азимут (0..180 град. от севера) и длина (пикс.) отрезков.

    Растровые координаты сегмента — (col, row), строка растёт К ЮГУ (сетка
    считается от ``y_top`` вниз). Смещение растянуто/сжато одинаково по обеим
    осям (рабочий растр квадратный), поэтому масштаб на угол не влияет и
    домножать на шаг сетки не нужно — только знак у строки (юг -> север).
    """
    if not segments:
        return np.array([]), np.array([])
    dx = np.array([x1 - x0 for (x0, y0), (x1, y1) in segments], dtype=float)
    dy_north = np.array([y0 - y1 for (x0, y0), (x1, y1) in segments], dtype=float)
    az = np.degrees(np.arctan2(dx, dy_north)) % 180.0
    length = np.hypot(dx, dy_north)
    return az, length


def _boundary_rings(geom):
    """Линии/кольца геометрии (полигон разлома в этих слоях — буферный коридор)."""
    t = geom.geom_type
    if t == "Polygon":
        yield geom.exterior
        yield from geom.interiors
    elif t == "MultiPolygon":
        for poly in geom.geoms:
            yield poly.exterior
            yield from poly.interiors
    elif t == "MultiLineString":
        yield from geom.geoms
    else:
        yield geom


def geometry_azimuths(gdf) -> tuple[np.ndarray, np.ndarray]:
    """То же самое (азимут, длина), но по вершинам geopandas-слоя в метрах карты.

    Разломные слои в этих данных — ПОЛИГОНЫ (буферные коридоры вдоль разлома), а
    не линии. Азимут берётся по всем рёбрам контура, взвешенным длиной: длинные
    боковые стороны коридора (вдоль разлома) естественно перевешивают короткие
    торцы (поперёк) без отдельного выделения главной оси.
    """
    az_list, len_list = [], []
    for geom in gdf.geometry:
        for ring in _boundary_rings(geom):
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                dx, dy = x1 - x0, y1 - y0
                az_list.append(np.degrees(np.arctan2(dx, dy)) % 180.0)
                len_list.append(np.hypot(dx, dy))
    return np.array(az_list), np.array(len_list)


def azimuth_rose(azimuths: np.ndarray, lengths: np.ndarray,
                 n_bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Гистограмма направлений (0..180 град.), взвешенная длиной; сумма = 1."""
    n_bins = n_bins or config.LIN_SAT_ROSE_BINS
    edges = np.linspace(0.0, 180.0, n_bins + 1)
    if len(azimuths) == 0:
        return edges, np.zeros(n_bins)
    hist, _ = np.histogram(azimuths, bins=edges, weights=lengths)
    total = hist.sum()
    return edges, hist / total if total > 0 else hist


def match_stats(auto_dist: np.ndarray, map_dist: np.ndarray,
                thresholds: tuple[float, ...] | None = None) -> pd.DataFrame:
    """Доля карты, «пойманная» автоматикой, и доля автоматики без карты — по порогам.

    ``auto_dist``/``map_dist`` — расстояния до ближайшей автоматической линии
    и до ближайшего разлома карты на одних и тех же ячейках. Порог задаёт, что
    считать «рядом» (совпадением может быть окрестность, а не пиксель в
    пиксель — оба источника огрубляют геометрию по-своему).
    """
    thresholds = thresholds or config.LIN_SAT_MATCH_M
    rows = []
    for t in thresholds:
        on_map = map_dist <= t
        on_auto = auto_dist <= t
        caught = float(on_auto[on_map].mean()) if on_map.any() else float("nan")
        unmatched = float((~on_map)[on_auto].mean()) if on_auto.any() else float("nan")
        rows.append({"порог_м": t, "доля_карты_поймана": caught,
                    "доля_авто_без_карты": unmatched})
    return pd.DataFrame(rows)
