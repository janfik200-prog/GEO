"""Векторные признаки на общей сетке ГИС Интегро (:class:`~src.integro_grid.GridMeta`).

Дистанционные растры (расстояние от центра ячейки до ближайшей геометрии слоя)
и плотностные (суммарная длина/площадь слоя в радиусе вокруг ячейки) — тот же
смысл, что у ``dist_*``/``dens_*`` в :mod:`src.features`, но напрямую на
регулярной сетке ``(Prf, Pic)`` без построения GeoDataFrame-сетки по маске.

Все слои предполагаются в CRS сетки (сверку CRS делает вызывающий код —
см. ``experiments/build_dataset_v1.py``).
"""

import numpy as np
import shapely

from .integro_grid import GridMeta


def _cell_center_points(meta: GridMeta) -> np.ndarray:
    """Плоский массив shapely-точек центров ячеек (порядок C: строка 0 — север)."""
    x, y = meta.cell_centers()
    return shapely.points(x.ravel(), y.ravel())


def distance_raster(meta: GridMeta, geoms) -> np.ndarray:
    """Расстояние (м) от центра каждой ячейки до ближайшей геометрии слоя.

    ``geoms`` — последовательность shapely-геометрий (например,
    ``gdf.geometry.values``). Возвращает float32-массив формы ``meta.shape``;
    ячейка внутри полигона/на линии получает 0.
    """
    geoms = np.asarray(geoms, dtype=object)
    if len(geoms) == 0:
        raise ValueError("distance_raster: слой пуст")
    pts = _cell_center_points(meta)
    tree = shapely.STRtree(geoms)
    nearest = tree.nearest(pts)
    d = shapely.distance(pts, geoms[nearest])
    return d.astype(np.float32).reshape(meta.shape)


def density_raster(meta: GridMeta, geoms, radius: float, measure: str = "length") -> np.ndarray:
    """Плотность слоя вокруг ячейки: длина (``length``) или площадь (``area``)
    пересечения слоя с кругом радиуса ``radius`` (м) вокруг центра ячейки.

    Аналог ``_density_in_radius`` из :mod:`src.features` (узел из многих
    разломов/даек получает больший балл, чем одиночный объект), но на сетке.
    """
    if measure not in ("length", "area"):
        raise ValueError(f"measure должен быть 'length' или 'area', получено {measure!r}")
    geoms = np.asarray(geoms, dtype=object)
    pts = _cell_center_points(meta)
    tree = shapely.STRtree(geoms)
    out = np.zeros(len(pts), dtype=np.float32)
    # Запрос по индексу сразу для всех буферов: пары (индекс точки, индекс геометрии).
    bufs = shapely.buffer(pts, radius, quad_segs=8)
    pt_idx, geom_idx = tree.query(bufs, predicate="intersects")
    for i in np.unique(pt_idx):
        inter = shapely.intersection(geoms[geom_idx[pt_idx == i]], bufs[i])
        out[i] = (
            shapely.length(inter).sum() if measure == "length" else shapely.area(inter).sum()
        )
    return out.reshape(meta.shape)
