"""Признаки минерагенической карты — для задачи № 6 (палеодолины, не золото).

``config.GOLD_FEATURES_STOP`` запрещает прямые признаки минерагенической карты
(геохимические ореолы, опробование, привнос урана) для прогноза ЗОЛОТА — запрет
против циркулярности (минерагеническая карта сама построена по рудопроявлениям).
Здесь целью является фактор «долины и впадины», а не оруденение, поэтому запрет
не действует (реестр задач, п. 6, явная оговорка заказчика).

Слой «геохимическое_опробование» (76 точек) сюда сознательно НЕ включён — это
независимый набор заверки золота (:func:`src.assessment.load_verification_points`);
использование его как признака для другой задачи не было бы утечкой в статистическом
смысле, но обесценило бы его как чистый резерв для будущей заверки золотого прогноза.

Геохимические ореолы (14 линий, ``label`` = Au/U/Co/V/Ag) и зоны привноса урана
(23 линии) — оставшаяся часть минерагенической карты, безопасная и разрешённая.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np

from .integro_grid import GridMeta
from .vector_features import distance_raster


def geochem_distance_features(
    meta: GridMeta, halo: gpd.GeoDataFrame, uran: gpd.GeoDataFrame,
) -> dict[str, np.ndarray]:
    """Расстояния (м) до геохимических ореолов Au/U/прочих металлов и зон привноса урана.

    ``halo`` — слой «геохимические ореолы» с колонкой ``label`` (Au/U/Co/V/Ag);
    Au и U выделены отдельно (по 5 линий каждый, целевые металлы проекта),
    остальные метки (Co/V/Ag, 4 линии) — общей колонкой. ``uran`` — слой
    «привнос урана» целиком (23 линии, одна метка).
    """
    label = halo["label"].astype(str)
    out = {
        "mingeo_dist_au": distance_raster(meta, halo.loc[label == "Au"].geometry.values),
        "mingeo_dist_u": distance_raster(meta, halo.loc[label == "U"].geometry.values),
        "mingeo_dist_uran": distance_raster(meta, uran.geometry.values),
    }
    other = halo.loc[~label.isin(["Au", "U"])]
    if len(other):
        out["mingeo_dist_other"] = distance_raster(meta, other.geometry.values)
    return out
