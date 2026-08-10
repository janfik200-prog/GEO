"""Геологические производные, не вытащенные при сборке ``dataset_v1``.

Аудит (``docs II presentation/Аудит-признаков-что-не-вытащено.md``) нашёл три
пропуска в уже имеющихся векторных слоях (``data/Gis-integro/shp_dbf``), без
новых скачиваний:

* **раздельные признаки по фациям.** ``dist_facies`` в датасете v1 считан по
  ВСЕМ 15 полигонам ``fasii`` разом и не различает две фации (``CODE_F`` 1 и 2,
  348 и 509 км²) — это прямо блокирует дополнительную задачу заказчика, где
  фации разного генезиса контролируют оруденение по-разному;
* **зона вкрест простирания от контакта чехла с фундаментом.** Полигоны
  ``svita_new`` покрывают 3818 км², их ВНЕШНЯЯ граница (не границы между
  соседними свитами внутри контура) и есть контакт чехла с фундаментом —
  фактор с рис. 1 постановки заказчика. Расстояние до контакта —
  ``dist_contact``; направление контакта в точке (простирание, ось —
  «вкрест» этого направления и есть искомая зона) — ``strike_sin``/
  ``strike_cos``, осевая пара по касательной ближайшего сегмента границы
  (простирание не имеет «головы»/«хвоста», поэтому кодируется как
  ``cos(2*azimuth)``/``sin(2*azimuth)``, тем же приёмом, что чистка
  ``gm_gr_1GFI_25`` в ``src/features_v11.py``);
* **узлы пересечения разломных систем.** ``lin_node_dens`` считает узлы
  только для автоматических линеаментов по рельефу; узлы пересечения ДВУХ
  оцифрованных разломных систем (``glub_raz_nw`` x ``glub_r_nw``) не
  считались вовсе. Оба слоя — полигоны (буферизованные разломные коридоры,
  не линии), поэтому «узел» — это полигон-полигон пересечение: расстояние до
  ближайшего из непустых пересечений.

Все признаки — с префиксом ``geo2_``, отдельная группа от ``geo``/``dist`` в
``config.V2_FEATURE_GROUPS`` (те заняты факторными слоями v1 и гидросетью
через префиксный матчинг ``dist_``/``dens_``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from . import config
from .data_loader import load_layer
from .integro_grid import GridMeta
from .vector_features import distance_raster

GEO2_COLS: tuple[str, ...] = (
    "geo2_facies1", "geo2_facies2", "geo2_contact",
    "geo2_strike_sin", "geo2_strike_cos", "geo2_fault_node",
)


def _shp_dir() -> Path:
    return config.GOLD_TARGET_PGRID.parents[1] / config.SHP_SUBDIR


def _cell_center_points(meta: GridMeta) -> np.ndarray:
    x, y = meta.cell_centers()
    return shapely.points(x.ravel(), y.ravel())


def facies_distance_rasters(meta: GridMeta, facies_gdf) -> dict[str, np.ndarray]:
    """``dist`` до полигонов ``CODE_F`` 1 и 2 раздельно."""
    out = {}
    for code, name in ((1, "facies1"), (2, "facies2")):
        sub = facies_gdf[facies_gdf["CODE_F"] == code]
        if len(sub) == 0:
            raise ValueError(f"fasii: нет полигонов с CODE_F={code}")
        out[f"geo2_{name}"] = distance_raster(meta, sub.geometry.values)
    return out


def contact_boundary_segments(mask_gdf) -> tuple[np.ndarray, np.ndarray]:
    """Сегменты ВНЕШНЕЙ границы объединения ``svita_new`` и их азимуты (рад).

    Внутренние границы между соседними полигонами свит (после ``union_all``)
    исчезают — остаётся только внешний контур, то есть контакт чехла с
    фундаментом, а не межсвитные границы.
    """
    union = shapely.union_all(mask_gdf.geometry.values)
    polys = [union] if union.geom_type == "Polygon" else list(shapely.get_parts(union))
    segs: list = []
    az: list[float] = []
    for poly in polys:
        for ring in [shapely.get_exterior_ring(poly), *[
            shapely.get_interior_ring(poly, i) for i in range(shapely.get_num_interior_rings(poly))
        ]]:
            coords = np.asarray(shapely.get_coordinates(ring))
            for i in range(len(coords) - 1):
                (x0, y0), (x1, y1) = coords[i], coords[i + 1]
                if x0 == x1 and y0 == y1:
                    continue
                segs.append(shapely.LineString([(x0, y0), (x1, y1)]))
                az.append(float(np.arctan2(x1 - x0, y1 - y0)))
    return np.array(segs, dtype=object), np.array(az, dtype=np.float64)


def contact_features(meta: GridMeta, mask_gdf) -> dict[str, np.ndarray]:
    """``dist_contact`` и осевое направление ближайшего сегмента границы."""
    segs, az = contact_boundary_segments(mask_gdf)
    dist = distance_raster(meta, segs)
    pts = _cell_center_points(meta)
    tree = shapely.STRtree(segs)
    nearest = tree.nearest(pts)
    theta = az[nearest]
    sin2 = np.sin(2.0 * theta).astype(np.float32).reshape(meta.shape)
    cos2 = np.cos(2.0 * theta).astype(np.float32).reshape(meta.shape)
    return {"geo2_contact": dist, "geo2_strike_sin": sin2, "geo2_strike_cos": cos2}


def fault_intersection_geoms(tect1_gdf, tect2_gdf) -> list:
    """Непустые полигон-полигон пересечения двух разломных систем."""
    geoms = []
    for g1 in tect1_gdf.geometry.values:
        for g2 in tect2_gdf.geometry.values:
            inter = shapely.intersection(g1, g2)
            if not inter.is_empty:
                geoms.append(inter)
    return geoms


def fault_node_distance(meta: GridMeta, tect1_gdf, tect2_gdf) -> np.ndarray:
    geoms = fault_intersection_geoms(tect1_gdf, tect2_gdf)
    if not geoms:
        raise ValueError("tect1 x tect2: пересечений не найдено")
    return distance_raster(meta, geoms)


def geology_v2_features(meta: GridMeta) -> pd.DataFrame:
    """Таблица ``GEO2_COLS`` на ячейках сетки (C-порядок, как датасет)."""
    shp_dir = _shp_dir()
    facies_gdf = load_layer(shp_dir / f"{config.LAYER_FILES['facies']}.shp")
    mask_gdf = load_layer(shp_dir / f"{config.LAYER_FILES['mask']}.shp")
    tect1_gdf = load_layer(shp_dir / f"{config.LAYER_FILES['tect1']}.shp")
    tect2_gdf = load_layer(shp_dir / f"{config.LAYER_FILES['tect2']}.shp")

    out: dict[str, np.ndarray] = {}
    out.update(facies_distance_rasters(meta, facies_gdf))
    out.update(contact_features(meta, mask_gdf))
    out["geo2_fault_node"] = fault_node_distance(meta, tect1_gdf, tect2_gdf)

    table = {k: v.ravel() for k, v in out.items()}
    return pd.DataFrame(table)[list(GEO2_COLS)]
