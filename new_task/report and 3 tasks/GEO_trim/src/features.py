"""Построение признаков: регулярная сетка, расстояния, близость, geo_score.

Все функции дополняют переданный ``grid`` (GeoDataFrame ячеек) новыми столбцами
и возвращают его же. Параметры берутся из :mod:`src.config`.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

from . import config
from .dem_features import add_dem_features
from .utils import robust_normalize_01, smooth_on_regular_grid


def build_grid(mask: gpd.GeoDataFrame, cell_size: int):
    """Построить регулярную сетку ячеек ``cell_size``×``cell_size`` по маске.

    Возвращает ``(grid, mask_union, shape)``: GeoDataFrame ячеек (со столбцами
    ``cell_id``, ``row``, ``col``, ``geometry``), объединённую геометрию маски и
    форму растра ``(n_rows, n_cols)``.
    """
    mask_union = unary_union(mask.geometry)
    prepared_mask = prep(mask_union)
    minx, miny, maxx, maxy = mask.total_bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    rows = []
    cell_id = 0
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            geom = box(x, y, x + cell_size, y + cell_size)
            if prepared_mask.intersects(geom):
                rows.append((cell_id, r, c, geom))
                cell_id += 1
    grid = gpd.GeoDataFrame(
        rows, columns=["cell_id", "row", "col", "geometry"], geometry="geometry", crs=mask.crs
    )
    return grid, mask_union, (len(ys), len(xs))


def add_distance_feature(grid: gpd.GeoDataFrame, source: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Добавить столбец ``name`` — расстояние от каждой ячейки до слоя ``source``."""
    source_union = unary_union(source.geometry)
    d = np.empty(len(grid), dtype=float)
    for i, geom in enumerate(grid.geometry.values):
        d[i] = 0.0 if geom.intersects(source_union) else geom.distance(source_union)
    grid[name] = d
    return grid


def distance_to_proximity(distance, transform: str = "sqrt", q: float = 0.75) -> np.ndarray:
    """Преобразовать расстояние в близость [0, 1] через убывающую экспоненту.

    ``transform`` (``sqrt``/``cbrt``/иное) сжимает шкалу расстояний, ``q`` задаёт
    квантиль, по которому подбирается масштаб затухания.
    """
    d = np.clip(np.asarray(distance, dtype=float), 0, None)
    if transform == "sqrt":
        t = np.sqrt(d)
    elif transform == "cbrt":
        t = np.cbrt(d)
    else:
        t = d
    scale = float(np.nanquantile(t, q))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.nanmean(t)), 1.0)
    return np.clip(np.exp(-t / scale), 0, 1)


def build_features(grid: gpd.GeoDataFrame, layers: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Посчитать признаки близости, их взаимодействия и обогащение.

    Добавляет ``dist_*`` и ``prox_*`` по каждому фактору, парные/тройные
    пересечения, агрегаты тектоники, ``coincidence_score`` (совпадение факторов),
    ``tect_only_penalty`` (штраф за «голую» тектонику без поддержки), через
    :func:`add_enriched_features` — mineral-systems признаки (плотности разломов/
    даек, широкие ореолы), и через :func:`src.dem_features.add_dem_features` —
    производные рельефа Copernicus DEM (уклон, кривизна, TPI, шероховатость) —
    см. :data:`config.FEATURE_COLS`.
    """
    # Расстояния до факторов.
    for role in ("facies", "paleo", "struct", "magm", "tect1", "tect2"):
        grid = add_distance_feature(grid, layers[role], f"dist_{role}")

    # Близость с индивидуальными параметрами преобразования.
    for role, (transform, q) in config.PROXIMITY_PARAMS.items():
        grid[f"prox_{role}"] = distance_to_proximity(grid[f"dist_{role}"], transform, q)

    # Агрегаты и пересечения факторов.
    grid["tect_combo"] = 0.5 * (grid["prox_tect1"] + grid["prox_tect2"])
    grid["tect_intersection"] = grid["prox_tect1"] * grid["prox_tect2"]
    grid["tect_magm_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_magm"])
    grid["tect_struct_intersection"] = np.sqrt(grid["tect_combo"] * grid["prox_struct"])
    grid["paleo_struct_intersection"] = np.sqrt(grid["prox_paleo"] * grid["prox_struct"])

    # Совпадение разнородных факторов в одной точке.
    combo_core = (
        np.clip(grid["tect_combo"], 0, 1)
        * np.clip(0.55 * grid["prox_magm"] + 0.45 * grid["prox_struct"], 0, 1)
        * np.clip(0.60 * grid["prox_paleo"] + 0.40 * grid["prox_facies"], 0, 1)
    )
    grid["coincidence_score"] = robust_normalize_01(np.sqrt(np.clip(combo_core, 0, 1)), 0.02, 0.98)

    # Штраф за тектонику без поддержки магматизмом/структурой/палео.
    tect_support = 0.40 * grid["prox_magm"] + 0.35 * grid["prox_struct"] + 0.25 * grid["prox_paleo"]
    grid["tect_only_penalty"] = robust_normalize_01(
        np.clip(grid["tect_combo"] - tect_support, 0, 1), 0.02, 0.98
    )

    grid = add_enriched_features(grid, layers)
    grid = add_dem_features(grid)
    return grid


def _density_in_radius(centroids, source: gpd.GeoDataFrame, radius: float, measure: str) -> np.ndarray:
    """Плотность слоя вокруг каждой ячейки: длина/площадь ``source`` в радиусе ``radius``.

    Через пространственный индекс берутся только геометрии, пересекающие буфер
    ячейки; ``measure`` = ``length`` (для линий-разломов) или ``area`` (для
    полигонов-даек). Узел у множества разломов/даек получает больший балл, чем у
    одиночного — то, что «расстояние до ближайшего» не отражает.
    """
    sidx = source.sindex
    geoms = source.geometry
    out = np.zeros(len(centroids))
    for i, c in enumerate(centroids):
        buf = c.buffer(radius)
        hit = list(sidx.query(buf, predicate="intersects"))
        if hit:
            inter = geoms.iloc[hit].intersection(buf)
            out[i] = inter.length.sum() if measure == "length" else inter.area.sum()
    return out


def add_enriched_features(grid: gpd.GeoDataFrame, layers: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Mineral-systems признаки из тех же слоёв (плотности + широкие ореолы).

    Набор отобран честной held-out проверкой (см. :data:`config.FEATURE_COLS`):
    ``dens_tect`` — плотность разломов (длина СЗ+СВ в :data:`config.DENSITY_RADIUS`),
    ``dens_magm`` — плотность магматизма (площадь даек в том же радиусе),
    ``prox_<role>_wide`` — широкий ореол (многомасштабная близость) по
    :data:`config.WIDE_PROXIMITY_ROLES`. Требует уже посчитанных ``dist_*``.
    """
    cent = list(grid.geometry.centroid.values)
    faults = gpd.GeoDataFrame(
        geometry=list(layers["tect1"].geometry) + list(layers["tect2"].geometry), crs=grid.crs
    )
    grid["dens_tect"] = robust_normalize_01(
        _density_in_radius(cent, faults, config.DENSITY_RADIUS, "length"), 0.02, 0.98
    )
    grid["dens_magm"] = robust_normalize_01(
        _density_in_radius(cent, layers["magm"].reset_index(drop=True), config.DENSITY_RADIUS, "area"),
        0.02, 0.98,
    )
    for role in config.WIDE_PROXIMITY_ROLES:
        grid[f"prox_{role}_wide"] = distance_to_proximity(
            grid[f"dist_{role}"], config.WIDE_PROXIMITY_TRANSFORM, config.WIDE_PROXIMITY_Q
        )
    return grid


def _criterion_transform(distance, kind: str) -> np.ndarray:
    """Степенная трансформация расстояния для симметризации гистограммы (ГИС Интегро)."""
    d = np.clip(np.asarray(distance, dtype=float), 0, None)
    if kind == "sqrt":
        return np.sqrt(d)
    if kind == "cbrt":
        return np.cbrt(d)
    return d


def compute_geo_score(grid: gpd.GeoDataFrame, grid_shape: tuple[int, int]) -> gpd.GeoDataFrame:
    """Критериальный baseline по методу ГИС Интегро «Таксономия по критериям».

    Воспроизводит нативный расчёт ГИС Интегро (``data/Gis-integro/Расчет``):
    расстояние до фактора → степенная трансформация (симметризация гистограммы) →
    min-max нормировка каждого критерия в [0, 1] → взвешенное манхэттенское (L1)
    расстояние до эталона-минимума (мера развития Плюты). Рецепт восстановлен по
    нативному ``prognoz.prognoz.property`` (Spearman ≈ 0.98 с выходом их программы)
    и подтверждён символами ``igk_prognose.dll`` (distance_manhattan, эталон, веса).
    В ГИС Интегро меньшее значение перспективнее; здесь шкала инвертируется
    (больше = лучше), чтобы baseline сравнивался так же, как ML-прогноз.

    Параметры — :data:`config.TAXONOMY_TRANSFORMS` и :data:`config.TAXONOMY_WEIGHTS`.
    Не входит в признаки модели — используется только для валидации.
    """
    columns, weights = [], []
    for role, kind in config.TAXONOMY_TRANSFORMS.items():
        t = _criterion_transform(grid[f"dist_{role}"].to_numpy(), kind)
        lo, hi = float(t.min()), float(t.max())
        columns.append((t - lo) / (hi - lo) if hi > lo else np.zeros_like(t))  # min-max, эталон=0
        weights.append(config.TAXONOMY_WEIGHTS[role])
    z = np.column_stack(columns)               # критерии в [0, 1] (0 = на эталоне-минимуме)
    w = np.asarray(weights, dtype=float)
    c = (np.abs(z) * w).sum(axis=1)            # взвешенное L1-расстояние до эталона (=0)
    dist = c / (c.mean() + 2.0 * c.std())       # нормировка Плюты: меньше = ближе к эталону
    grid["taxonomy_distance"] = dist            # нативная шкала ГИС Интегро (меньше = лучше)
    grid["geo_score_raw"] = -dist               # инверсия: больше = перспективнее
    grid["geo_score"] = robust_normalize_01(grid["geo_score_raw"], 0.02, 0.98)
    grid["geo_score_sm"] = robust_normalize_01(
        smooth_on_regular_grid(grid, "geo_score", grid_shape, passes=2), 0.02, 0.98
    )
    return grid
