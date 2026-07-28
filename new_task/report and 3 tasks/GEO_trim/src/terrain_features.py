"""Производные рельефа из растра ``relief_m`` (уклон, кривизна, TPI, шероховатость).

В ``dataset_v1`` есть единственный рельефный слой ``relief_m``. Боевой пайплайн
GEO (``src/dem_features.py``) использовал 5 производных Copernicus DEM
(dem_elev/slope/curv/tpi/rough) — они значимы для прогноза. Здесь те же
производные считаются напрямую из ``relief_m`` (растр 154×149, шаг 500 м),
без внешнего DEM: ``relief_m`` играет роль высоты (dem_elev).

Формулы совпадают с ``dem_features._derivatives``: уклон = модуль градиента,
кривизна = лапласиан, TPI = высота минус среднее по окну, шероховатость =
локальное СКО высоты. NaN рельефа (~4.7%) заполняются средним перед расчётом.
"""

from __future__ import annotations

import numpy as np

from . import config

TERRAIN_COLS = ["relief_slope", "relief_curv", "relief_tpi", "relief_rough"]


def _derivatives(elev: np.ndarray, window: int) -> dict[str, np.ndarray]:
    from scipy.ndimage import uniform_filter

    e = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    gy, gx = np.gradient(e)
    slope = np.hypot(gx, gy)                                   # уклон
    curv = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)   # кривизна (лапласиан)
    tpi = e - uniform_filter(e, size=window)                   # положение в окрестности
    rough = np.sqrt(np.clip(
        uniform_filter(e ** 2, window) - uniform_filter(e, window) ** 2, 0, None))
    return {"relief_slope": slope, "relief_curv": curv,
            "relief_tpi": tpi, "relief_rough": rough}


def build_terrain(frame, grid_shape: tuple[int, int],
                  window: int | None = None) -> dict[str, np.ndarray]:
    """Производные рельефа для ячеек ``frame`` (нужны столбцы row, col, relief_m)."""
    window = int(window if window is not None else getattr(config, "CRITERIAL_TERRAIN_WINDOW", 5))
    r = frame["row"].to_numpy().astype(int)
    c = frame["col"].to_numpy().astype(int)
    elev = np.full(grid_shape, np.nan, dtype=float)
    elev[r, c] = frame["relief_m"].to_numpy(dtype=float)
    deriv = _derivatives(elev, window)
    return {name: deriv[name][r, c] for name in TERRAIN_COLS}
