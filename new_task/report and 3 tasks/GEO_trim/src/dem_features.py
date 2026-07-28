"""Признаки рельефа из Copernicus DEM GLO-30 (производные топографии).

Отдельный модуль с ПОСТОЯННЫМ кэшем (datacache/anabar_dem) и graceful fallback:
при недоступности сети/GDAL боевой пайплайн не падает — признаки заполняются
нейтрально (0), а модель деградирует к версии без рельефа.

Набор отобран честной held-out проверкой (experiments/dem_prune.py, 8 сидов):
прирост значим — top-5% +1.04x (95% ДИ [+0.54,+1.55]), top-10% +0.56x
([+0.17,+0.95]). Все 5 производных важнее среднего; ``dem_tpi`` (топографическое
положение) — важнейший признак из всех, ловит палеодолины (рудоконтроль Au-U).

SRTM не годится (предел 60° N, Анабар на 71°). Copernicus DEM GLO-30 (30 м, до
полюсов) открыт на AWS (anonymous /vsicurl), тайлы 1°×1°.
"""

import math
import warnings
from pathlib import Path

import numpy as np

from . import config

_AWS = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_COLS = ["dem_elev", "dem_slope", "dem_curv", "dem_tpi", "dem_rough"]


def _tile_url(lat: int, lon: int) -> str:
    la = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    lo = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    name = f"Copernicus_DSM_COG_10_{la}_00_{lo}_00_DEM"
    return f"/vsicurl/{_AWS}/{name}/{name}.tif"


def _bbox_of(grid) -> tuple[float, float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = grid.geometry.centroid.to_crs(4326)
    p = config.DEM_PAD
    return (c.x.min() - p, c.y.min() - p, c.x.max() + p, c.y.max() + p)


def _build_mosaic(bbox, cache_dir: Path):
    """Скачать/обрезать DEM-мозаику по bbox в 4326-растр (с кэшем). None при ошибке."""
    from osgeo import gdal  # локальный импорт: отсутствие GDAL не ломает офлайн-импорт модуля

    lon0, lat0, lon1, lat1 = bbox
    key = f"dem_{lat0:.2f}_{lon0:.2f}_{lat1:.2f}_{lon1:.2f}.tif".replace("-", "m")
    out = cache_dir / key
    if not out.exists():
        gdal.UseExceptions()
        srcs = [_tile_url(lat, lon)
                for lat in range(math.floor(lat0), math.floor(lat1) + 1)
                for lon in range(math.floor(lon0), math.floor(lon1) + 1)]
        gdal.Warp(str(out), srcs, dstSRS="EPSG:4326", outputBounds=(lon0, lat0, lon1, lat1),
                  xRes=config.DEM_RES[0], yRes=config.DEM_RES[1], resampleAlg="bilinear")
    ds = gdal.Open(str(out))
    return ds.GetRasterBand(1).ReadAsArray().astype(float), ds.GetGeoTransform()


def _derivatives(elev: np.ndarray) -> dict[str, np.ndarray]:
    from scipy.ndimage import uniform_filter

    e = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    gy, gx = np.gradient(e)
    slope = np.hypot(gx, gy)
    curv = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)                  # лапласиан
    tpi = e - uniform_filter(e, size=config.DEM_TPI_WINDOW)                   # положение в окрестности
    rough = np.sqrt(np.clip(uniform_filter(e ** 2, config.DEM_ROUGH_WINDOW)
                            - uniform_filter(e, config.DEM_ROUGH_WINDOW) ** 2, 0, None))
    return {"dem_elev": elev, "dem_slope": slope, "dem_curv": curv,
            "dem_tpi": tpi, "dem_rough": rough}


def _sample(arr, gt, lon, lat):
    x0, px, _, y0, _, py = gt
    n_r, n_c = arr.shape
    c = ((np.asarray(lon) - x0) / px).astype(int)
    r = ((np.asarray(lat) - y0) / py).astype(int)
    ok = (c >= 0) & (c < n_c) & (r >= 0) & (r < n_r)
    out = np.full(np.shape(lon), np.nan)
    out[ok] = arr[r[ok], c[ok]]
    return out


def add_dem_features(grid):
    """Добавить ``DEM_COLS`` (производные рельефа) в ``grid``.

    Качает Copernicus DEM по bbox сетки (кэш :data:`cache_paths.ANABAR_DEM`),
    считает уклон/кривизну/TPI/шероховатость, сэмплирует на центроиды ячеек.
    При любой ошибке (нет сети/GDAL/тайлов) колонки заполняются нейтрально (0) с
    предупреждением — боевой прогноз продолжается без рельефа.
    """
    try:
        from cache_paths import ANABAR_DEM

        bbox = _bbox_of(grid)
        arr, gt = _build_mosaic(bbox, ANABAR_DEM)
        deriv = _derivatives(arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cents = grid.geometry.centroid.to_crs(4326)
        glon, glat = cents.x.to_numpy(), cents.y.to_numpy()
        for k in DEM_COLS:
            grid[k] = _sample(deriv[k], gt, glon, glat)
        med = grid[DEM_COLS].median()
        grid[DEM_COLS] = grid[DEM_COLS].fillna(med)
        valid = float(np.isfinite(_sample(deriv["dem_elev"], gt, glon, glat)).mean())
        if valid < 0.9:
            warnings.warn(f"DEM покрывает лишь {valid*100:.0f}% ячеек — рельеф может быть неинформативен")
    except Exception as exc:  # noqa: BLE001 — graceful fallback на офлайн/без GDAL
        warnings.warn(f"DEM недоступен ({exc}); признаки рельефа заполнены нейтрально")
        for k in DEM_COLS:
            grid[k] = 0.0
    return grid
