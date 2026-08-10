"""Рельеф: кривизна на другом масштабе, шероховатость (TRI), водосбор в датасете.

Аудит (``docs II presentation/Аудит-признаков-что-не-вытащено.md``) нашёл три
пропуска:

* **кривизна на другом масштабе.** ``dem_curv`` в ``src/terrain_v2.py`` был
  посчитан и ОТБРАКОВАН — на масштабе соседнего пикселя (~100 м) он совпал с
  ``dem_tpi_500`` (окно 500 м) c r=-1.00: разные по названию, но алгебраически
  один и тот же признак. Здесь берётся масштаб 5 км — заведомо крупнее ОБОИХ
  уже занятых масштабов (500 м у ``dem_tpi_500``, 2 км у ``dem_tpi_2km``), то
  есть форма регионального бассейна/свода, а не локальная выпуклость;
* **шероховатость.** Terrain Ruggedness Index (Riley et al., 1999) — RMS
  разницы высоты с 8 соседями на рабочем растре 100 м. Измерение показало:
  среднее TRI по ячейке 500 м совпадает с ``dem_slope`` (Spearman r=0.999) —
  на этом DEM оба по сути линейная функция локального градиента, поэтому
  ``relief2_tri`` (среднее) в пул НЕ включён по тому же основанию, что и
  ``dem_curv`` в terrain_v2. Внутриячеечное СТАНДАРТНОЕ ОТКЛОНЕНИЕ TRI
  (``relief2_tri_std``) — уже другая величина (r=0.79 с уклоном): разброс
  изрезанности внутри ячейки, а не сама изрезанность, и остаётся в пуле;
* **водосбор как признак датасета.** ``src/catchments.py`` до сих пор
  использовался только в заверке (площадь сноса точки). Водосборное дерево
  D8 строится на самой целевой сетке 500 м (тот же вход, что видит заверка,
  не рабочий растр 100 м), и для каждой ячейки листа считается число ячеек
  её собственного водосбора — большой водосбор означает бОльшую площадь
  сноса обломочного материала, что релевантно для шлихового/россыпного
  контроля золота.

Кривизна и TRI используют тот же перепроецированный DEM 100 м, что и
``src/terrain_v2.py`` (кэш ``datacache/anabar_dem`` переиспользуется, повторного
скачивания нет). Водосбор считается на уже посчитанной колонке ``dem_elev``
(500 м) существующего датасета — тоже без новых скачиваний.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catchments, config, integro_grid, terrain_v2

RELIEF2_COLS: tuple[str, ...] = (
    "relief2_curv_5km", "relief2_tri_std", "relief2_catch_log",
)

#: Посчитан, но не включён в пул — измерение см. в докстринге модуля.
RELIEF2_DROPPED: tuple[str, ...] = ("relief2_tri",)


def curvature_at_scale(elev: np.ndarray, res_m: float, window_m: float) -> np.ndarray:
    """Лапласиан DEM, сглаженного гауссианом с sigma ~ ``window_m`` / 2."""
    from scipy.ndimage import gaussian_filter

    e = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    sigma_px = max(1.0, (window_m / res_m) / 2.0)
    smooth = gaussian_filter(e, sigma_px)
    gy, gx = np.gradient(smooth, res_m)
    return np.gradient(gx, res_m, axis=1) + np.gradient(gy, res_m, axis=0)


def terrain_ruggedness(elev: np.ndarray, res_m: float) -> np.ndarray:
    """TRI: RMS разницы высоты с 8 соседями (Riley et al., 1999). Край — edge-pad."""
    e = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    padded = np.pad(e, 1, mode="edge")
    total = np.zeros_like(e)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = padded[1 + dr: 1 + dr + e.shape[0], 1 + dc: 1 + dc + e.shape[1]]
            total += (shifted - e) ** 2
    return np.sqrt(total / 8.0)


def _block_mean_std(arr: np.ndarray, meta: integro_grid.GridMeta,
                     k: int, pad: int) -> tuple[np.ndarray, np.ndarray]:
    core = arr[pad:pad + meta.prf * k, pad:pad + meta.pic * k]
    blocks = core.reshape(meta.prf, k, meta.pic, k)
    return blocks.mean(axis=(1, 3)).ravel(), blocks.std(axis=(1, 3)).ravel()


def catchment_log_area(meta: integro_grid.GridMeta, dem_elev_500: np.ndarray) -> np.ndarray:
    """``log1p`` числа ячеек водосбора на целевой сетке 500 м."""
    dem = dem_elev_500.reshape(meta.shape)
    valid = np.isfinite(dem)
    rec = catchments.flow_dirs_d8(np.where(valid, dem, np.nan), valid)
    acc = catchments.flow_accumulation(rec)
    return np.log1p(acc).astype(np.float32).reshape(meta.shape)


def relief_extra_features(meta: integro_grid.GridMeta, dem_elev_500: np.ndarray,
                           keep_dropped: bool = False) -> pd.DataFrame:
    """Таблица ``RELIEF2_COLS`` на ячейках целевой сетки (C-порядок, как датасет).

    ``dem_elev_500`` — колонка ``dem_elev`` существующего датасета (500 м),
    формы ``meta.shape``. ``keep_dropped=True`` возвращает и ``RELIEF2_DROPPED``
    — нужно только диагностике, которая эту отбраковку и обосновывает.
    """
    res_m = config.TER_RES_M
    pad = config.TER_PAD_PX
    k = int(round(meta.dx / res_m))
    elev_100 = terrain_v2.projected_dem(meta)

    curv = curvature_at_scale(elev_100, res_m, window_m=5000.0)
    tri = terrain_ruggedness(elev_100, res_m)
    curv_mean, _ = _block_mean_std(curv, meta, k, pad)
    tri_mean, tri_std = _block_mean_std(tri, meta, k, pad)

    out = {
        "relief2_curv_5km": curv_mean,
        "relief2_tri": tri_mean,
        "relief2_tri_std": tri_std,
        "relief2_catch_log": catchment_log_area(meta, dem_elev_500).ravel(),
    }
    cols = list(RELIEF2_COLS) + list(RELIEF2_DROPPED) if keep_dropped else list(RELIEF2_COLS)
    return pd.DataFrame(out)[cols]
