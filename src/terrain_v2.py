"""Производные рельефа v2: Copernicus DEM в проекции сетки, 100 м -> 500 м.

Отличие от :mod:`src.dem_features` (фаза v1): там производные считались на
географическом растре (градусы) и сэмплировались в центр ячейки — на 71° с.ш.
пиксель по долготе втрое короче, чем по широте, и уклон/кривизна получались
анизотропными артефактами проекции. Здесь DEM сначала перепроецируется в
метрическую систему целевой сетки (Пулково-1942, зона Гаусса-Крюгера) с шагом
``TER_RES_M``, производные считаются на изотропной метрической сетке, и лишь
затем агрегируются на ячейки 500 м — средним И стандартным отклонением
(субъячеечная текстура — самостоятельный признак, в v1 её не было).

Набор признаков (все с префиксом ``dem_``):

* ``elev`` — высота, м; ``elev_std`` — разброс высот внутри ячейки;
* ``slope`` — уклон, м/м; ``slope_std``;
* ``curv`` — лапласиан (кривизна: > 0 ложбина, < 0 гребень);
* ``tpi_500`` / ``tpi_2km`` — топографическое положение в двух масштабах
  (мультимасштабность отличает локальный уступ от регионального склона);
* ``relief_1km`` — размах высот в окне 1 км (расчленённость);
* ``incision`` — глубина вреза: насколько ячейка ниже максимума в окне 2 км
  (0 на водоразделах, максимум в тальвегах) — палеодолинный контроль.

Кэш: перепроецированный DEM 100 м кладётся в ``datacache/anabar_dem`` — сеть
дёргается один раз. Тайлы Copernicus DEM GLO-30 читаются анонимно с AWS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, integro_grid

TERRAIN_COLS: tuple[str, ...] = (
    "dem_elev", "dem_slope", "dem_slope_std",
    "dem_tpi_500", "dem_tpi_2km", "dem_relief_1km", "dem_incision",
)

#: Посчитанные, но НЕ включённые в пул — почти алгебраические дубли соседей
#: (измерено на реальном листе, см. experiments/terrain_v2_run.py):
#: * ``dem_curv`` ~ ``dem_tpi_500``, r = -1.00: TPI в окне 5 пикселей — это и
#:   есть дискретный лапласиан с обратным знаком, признак один и тот же;
#:   оставлен TPI, потому что он в метрах и парен с ``tpi_2km`` (масштаб);
#: * ``dem_elev_std`` ~ ``dem_slope``, r = +0.98: разброс высот внутри ячейки
#:   500 м на почти плоской поверхности равен уклону, умноженному на размер
#:   ячейки; субъячеечную текстуру несёт ``slope_std`` (r = 0.84 с уклоном).
TERRAIN_DROPPED: tuple[str, ...] = ("dem_curv", "dem_elev_std")


def _tile_url(lat: int, lon: int) -> str:
    la = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    lo = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    name = f"Copernicus_DSM_COG_10_{la}_00_{lo}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def projected_dem(meta: integro_grid.GridMeta) -> np.ndarray:
    """DEM в проекции сетки с шагом ``TER_RES_M`` и полем ``TER_PAD_PX`` пикселей.

    Форма: ``(prf * k + 2*pad, pic * k + 2*pad)``, где ``k = 500 / TER_RES_M``.
    Поле нужно оконным операциям (TPI 2 км = 20 пикселей при 100 м), чтобы у
    краёв листа они не считались по обрезанным окнам.
    """
    from cache_paths import ANABAR_DEM

    k = int(round(meta.dx / config.TER_RES_M))
    pad = config.TER_PAD_PX
    h, w = meta.prf * k + 2 * pad, meta.pic * k + 2 * pad
    cache = ANABAR_DEM / f"dem_proj_{config.TER_RES_M:.0f}m_{h}x{w}.npy"
    if cache.exists():
        return np.load(cache)

    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    if proj4 is None:
        raise RuntimeError("proj4 целевой сетки не найден (sidecar .pj4)")

    # Набор тайлов 1x1 градус по охвату сетки (углы -> WGS84)
    from pyproj import CRS, Transformer

    tr = Transformer.from_crs(CRS.from_proj4(proj4), CRS.from_epsg(4326), always_xy=True)
    xs = [meta.x0 - pad * config.TER_RES_M, meta.x0 + (meta.pic * k + pad) * config.TER_RES_M]
    ys = [meta.y_top + pad * config.TER_RES_M, meta.y_top - (meta.prf * k + pad) * config.TER_RES_M]
    corners = [(x, y) for x in xs for y in ys]
    lons, lats = zip(*[tr.transform(x, y) for x, y in corners])
    srcs = [rasterio.open(_tile_url(la, lo))
            for la in range(int(np.floor(min(lats))), int(np.floor(max(lats))) + 1)
            for lo in range(int(np.floor(min(lons))), int(np.floor(max(lons))) + 1)]
    mosaic, mosaic_tf = merge(srcs)
    src_crs = srcs[0].crs
    for s in srcs:
        s.close()

    dst = np.full((h, w), np.nan, dtype="float32")
    dst_tf = Affine(config.TER_RES_M, 0.0, meta.x0 - pad * config.TER_RES_M,
                    0.0, -config.TER_RES_M, meta.y_top + pad * config.TER_RES_M)
    reproject(source=mosaic[0], destination=dst,
              src_transform=mosaic_tf, src_crs=src_crs,
              dst_transform=dst_tf, dst_crs=CRS.from_proj4(proj4),
              resampling=Resampling.bilinear, src_nodata=None, dst_nodata=np.nan)
    ANABAR_DEM.mkdir(parents=True, exist_ok=True)
    np.save(cache, dst)
    return dst


def terrain_derivatives(elev: np.ndarray, res_m: float | None = None) -> dict[str, np.ndarray]:
    """Производные рельефа на изотропном метрическом растре (шаг ``res_m``)."""
    from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

    res_m = res_m or config.TER_RES_M
    e = np.where(np.isfinite(elev), elev, np.nanmedian(elev)).astype(float)
    gy, gx = np.gradient(e, res_m)
    slope = np.hypot(gx, gy)
    curv = np.gradient(gx, res_m, axis=1) + np.gradient(gy, res_m, axis=0)

    def win(metres: float) -> int:
        return max(3, int(round(metres / res_m)) | 1)   # нечётный размер окна

    tpi_500 = e - uniform_filter(e, size=win(500))
    tpi_2km = e - uniform_filter(e, size=win(2000))
    w1 = win(1000)
    relief_1km = maximum_filter(e, size=w1) - minimum_filter(e, size=w1)
    incision = maximum_filter(e, size=win(2000)) - e
    return {"elev": e, "slope": slope, "curv": curv, "tpi_500": tpi_500,
            "tpi_2km": tpi_2km, "relief_1km": relief_1km, "incision": incision}


def _block_stats(arr: np.ndarray, meta: integro_grid.GridMeta,
                 k: int, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и стандартное отклонение по блокам k x k (ячейка 500 м)."""
    core = arr[pad:pad + meta.prf * k, pad:pad + meta.pic * k]
    blocks = core.reshape(meta.prf, k, meta.pic, k)
    return blocks.mean(axis=(1, 3)).ravel(), blocks.std(axis=(1, 3)).ravel()


def terrain_features(meta: integro_grid.GridMeta,
                     keep_dropped: bool = False) -> pd.DataFrame:
    """Таблица признаков рельефа v2 на ячейки сетки (C-порядок, как датасет).

    Среднее по ячейке — для всех производных; дополнительно стандартное
    отклонение уклона внутри ячейки (субъячеечная текстура: на 500 м среднее
    сглаживает уступы, а разброс их сохраняет).

    ``keep_dropped=True`` возвращает и отбракованные ``TERRAIN_DROPPED`` — нужно
    только диагностике, которая эту отбраковку и обосновывает.
    """
    k = int(round(meta.dx / config.TER_RES_M))
    pad = config.TER_PAD_PX
    deriv = terrain_derivatives(projected_dem(meta))
    out = {}
    for name in ("elev", "slope", "curv", "tpi_500", "tpi_2km", "relief_1km", "incision"):
        mean, std = _block_stats(deriv[name], meta, k, pad)
        out[f"dem_{name}"] = mean
        if name in ("elev", "slope"):
            out[f"dem_{name}_std"] = std
    if keep_dropped:
        return pd.DataFrame(out)[list(TERRAIN_COLS) + list(TERRAIN_DROPPED)]
    return pd.DataFrame(out)[list(TERRAIN_COLS)]
