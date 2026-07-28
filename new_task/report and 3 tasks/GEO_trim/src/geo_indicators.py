"""Бинарные геологические признаки: членство ячейки в объектах-факторах (0/1).

Альтернатива непрерывному расстоянию `dist_*`: для каждой ячейки — 1, если её
центр попадает внутрь полигона фактора (разломы, фации, палеодолины, дайки),
иначе 0. Для линейных слоёв (коры выветривания) — «в буфере»: расстояние до
слоя ≤ порога. Такое evidential/indicator-кодирование — стандарт в прогнозной
металлогении и, в отличие от точного `dist_*`, не воспроизводит формулу
критериального анализа буквально (грубее, без градиента-ореола).

Point-in-polygon считается через ``matplotlib.path.Path`` (без geopandas/shapely).
Соответствие ролей и шейпов — как в исходной сборке (config.LAYER_FILES ГИС Интегро):
разломы = glub_raz_nw + glub_r_nw, фации = fasii, палеодолины = gr_dol_vp_poly,
дайки = dayki_buf, коры выветривания = kory (линии).
"""

from __future__ import annotations

from pathlib import Path as FsPath

import numpy as np

from . import config

# Индикатор -> (список шейпов-полигонов) для point-in-polygon.
POLYGON_INDICATORS: dict[str, tuple[str, ...]] = {
    "in_faults": ("glub_raz_nw", "glub_r_nw"),   # разломы (обе системы)
    "in_facies": ("fasii",),                     # литолого-фациальный
    "in_paleo": ("gr_dol_vp_poly",),             # палеодолины
    "in_dykes": ("dayki_buf",),                  # дайки (магматогенный)
}
# Индикатор -> (шейп-линия, буфер в метрах) для «в буфере».
BUFFER_INDICATORS: dict[str, tuple[str, float]] = {
    "near_kory": ("kory", 1000.0),               # коры выветривания (линии), буфер 1 км
}

_SHP_DIR = config.PROJECT_ROOT / "data" / "Gis-integro" / "shp_dbf"


def _polygon_paths(shp_stem: str):
    """Вернуть список matplotlib.Path по всем кольцам полигонов шейпа."""
    import shapefile
    from matplotlib.path import Path as MplPath

    reader = shapefile.Reader(str(_SHP_DIR / f"{shp_stem}.shp"))
    paths = []
    for shape in reader.shapes():
        pts = np.asarray(shape.points, dtype=float)
        if len(pts) == 0:
            continue
        bounds = list(shape.parts) + [len(pts)]
        for i in range(len(bounds) - 1):
            ring = pts[bounds[i]:bounds[i + 1]]
            if len(ring) >= 3:
                paths.append(MplPath(ring))
    return paths


def _inside_any(shp_stems: tuple[str, ...], XY: np.ndarray) -> np.ndarray:
    """1, если точка внутри ЛЮБОГО полигона перечисленных шейпов."""
    inside = np.zeros(len(XY), dtype=bool)
    for stem in shp_stems:
        for path in _polygon_paths(stem):
            inside |= path.contains_points(XY)
    return inside.astype(np.uint8)


def build_indicators(frame, dist_columns: dict | None = None) -> dict[str, np.ndarray]:
    """Собрать бинарные гео-признаки для ячеек ``frame`` (нужны столбцы x, y).

    ``dist_columns`` — опционально словарь ``{имя_dist: значения}`` для буферных
    индикаторов по уже посчитанным расстояниям (например ``dist_struct`` для kory),
    чтобы не пересчитывать. Если нужного расстояния нет — буферный индикатор
    считается напрямую от линий (медленнее) не здесь; в этом проекте расстояния
    уже есть в датасете.
    """
    XY = np.column_stack([frame["x"].to_numpy(float), frame["y"].to_numpy(float)])
    out: dict[str, np.ndarray] = {}
    for name, stems in POLYGON_INDICATORS.items():
        out[name] = _inside_any(stems, XY)
    dist_columns = dist_columns or {}
    for name, (stem, radius) in BUFFER_INDICATORS.items():
        # kory -> используем dist_struct (расстояние до kory) из датасета
        dcol = "dist_struct" if stem == "kory" else None
        if dcol and dcol in dist_columns:
            out[name] = (np.asarray(dist_columns[dcol]) <= radius).astype(np.uint8)
    return out


def indicator_names() -> list[str]:
    """Имена всех бинарных гео-признаков (полигональные + буферные с известным dist)."""
    return list(POLYGON_INDICATORS) + list(BUFFER_INDICATORS)


# --------------------------------------------------------------------------- #
# Производные гео-признаки (mineral systems): пересечения факторов, coincidence,
# плотности. Пересечения = логическое И бинарных индикаторов (совмещение факторов
# в ячейке — классический рудоконтролирующий признак). coincidence = число
# совпавших факторов. Плотности разломов/даек берутся из датасета (dens_tect/
# dens_magm), т.к. уже посчитаны при сборке.
# --------------------------------------------------------------------------- #

# Пересечение -> пара базовых индикаторов (логическое И).
INTERSECTIONS: dict[str, tuple[str, str]] = {
    "x_faults_dykes": ("in_faults", "in_dykes"),     # разломы × дайки (тект.-магм.)
    "x_faults_paleo": ("in_faults", "in_paleo"),     # разломы × палеодолины
    "x_faults_facies": ("in_faults", "in_facies"),   # разломы × фации
    "x_facies_paleo": ("in_facies", "in_paleo"),     # фации × палеодолины
}
# Плотностные признаки, которые берём из датасета (уже посчитаны при сборке).
DENSITY_FROM_DATASET: tuple[str, ...] = ("dens_tect", "dens_magm")


def build_derived(indicators: dict[str, np.ndarray], frame) -> dict[str, np.ndarray]:
    """Производные признаки из бинарных индикаторов ``indicators`` и датасета ``frame``.

    Возвращает словарь: пересечения (И индикаторов), ``coincidence`` (сумма всех
    базовых индикаторов — число совпавших факторов) и плотности из датасета.
    """
    out: dict[str, np.ndarray] = {}
    for name, (a, b) in INTERSECTIONS.items():
        if a in indicators and b in indicators:
            out[name] = (indicators[a].astype(bool) & indicators[b].astype(bool)).astype(np.uint8)
    if indicators:
        out["coincidence"] = np.sum([v.astype(np.int16) for v in indicators.values()], axis=0).astype(np.int16)
    for c in DENSITY_FROM_DATASET:
        if c in frame.columns:
            out[c] = frame[c].to_numpy(dtype=float)
    return out


def derived_names(has_densities: bool = True) -> list[str]:
    """Имена производных признаков (пересечения + coincidence + плотности)."""
    names = list(INTERSECTIONS) + ["coincidence"]
    if has_densities:
        names += list(DENSITY_FROM_DATASET)
    return names
