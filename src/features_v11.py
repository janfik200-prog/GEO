"""Признаки v1.1 для лестницы нетипичности — single source of truth.

Ревью model-critic (пункт 13): набор v1 содержал алгебраически избыточные
колонки (модуль градиента = f(компонент), угол = atan2 компонент) и сырые DN
Landsat, чувствительные к освещению/теням — детекторы нетипичности ловили
артефакты сцены, а не геологию. Здесь:

* :func:`ladder_features` — ЕДИНСТВЕННЫЙ источник списка признаков лестницы
  (база — :func:`src.cell_mask.feature_columns`, тот остаётся для маски);
* :func:`fill_relief_from_dem` — заливка пропусков ``relief_m`` из Copernicus
  DEM GLO-30 с линейной калибровкой (graceful fallback при недоступности);
* :func:`condition_number` — диагностика мультиколлинеарности набора.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import cell_mask, config

# Алгебраически избыточные градиентные колонки: модуль 1G = f(1GX, 1GY),
# угол 1GFI = atan2(1GY, 1GX) — компоненты 1GX/1GY остаются.
GRADIENT_REDUNDANT: tuple[str, ...] = ("gm_gr_1G_25", "gm_mag_1G_25")
# Полное поле = региональная составляющая (flt_35) + остаток (ost_35): проверено
# численно на dataset_v1, max|отклонения| ~1e-6 при std поля ~10. Точная линейная
# зависимость обнуляла собственное значение корреляционной матрицы (cond ~4e12) и
# делала MinCovDet/PCA численно шаткими. Оставляем разложение, убираем сумму.
FIELD_REDUNDANT: tuple[str, ...] = ("gm_gr_all", "gm_mg_all")
# Поля направлений: азимут горизонтального градиента, завёрнутый на ±180°
# (аудит признаков, п. 2.1: −179.95° и +179.83° численно максимально далеки,
# хотя это почти одно направление). Раскодируются в осевую пару cos(2θ)/sin(2θ)
# ниже в ladder_features — ОСЕВУЮ, а не векторную (sin θ/cos θ как для полярных
# координат): градиент «на северо-восток» и «на юго-запад» задают одну и ту же
# линейную структуру, обычная (не удвоенная) sin/cos-развёртка их бы развела.
# Сырые углы (parquet) и одинарные sin/cos (если появятся) всё равно убираются
# из base — они несут тот же завёрнутый разрыв.
ANGLE_BASES: tuple[str, ...] = ("gm_gr_1GFI_25", "gm_mag_1GFI_25")
# Сырые DN Landsat, заменяемые отношениями (ls_ch6 — тепловой, остаётся).
LS_RAW: tuple[str, ...] = ("ls_ch1", "ls_ch2", "ls_ch3", "ls_ch4", "ls_ch5", "ls_ch7")
# Расстояния до гидросети: только при include_dist=True — первичный набор без
# них, чтобы наивная планка «минус расстояние до рек» была честным контролем.
DIST_COLS: tuple[str, ...] = ("dist_dnl", "dist_dnara")


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Отношение с защитой: знаменатель <= 0 или NaN -> NaN (импутация дальше)."""
    den = np.where(np.isfinite(den) & (den > 0), den, np.nan)
    return num / den


def ladder_features(df: pd.DataFrame, include_dist: bool = False) -> pd.DataFrame:
    """Матрица признаков v1.1 лестницы нетипичности (single source of truth).

    База — независимые признаки :func:`src.cell_mask.feature_columns` (тот
    остаётся источником для маски валидности), далее:

    * убраны избыточные градиентные колонки (``GRADIENT_REDUNDANT``) и суммы
      полей (``FIELD_REDUNDANT``: поле = flt_35 + ost_35);
    * поля направлений 1GFI (``ANGLE_BASES``) перекодированы осево —
      ``<имя>_cos2``/``<имя>_sin2`` = cos(2θ)/sin(2θ) — вместо разрыва на
      ±180° у сырого угла; сырой угол убран (см. аудит признаков, п. 2.1);
    * сырые DN Landsat (кроме теплового ``ls_ch6``) заменены отношениями,
      устойчивыми к освещению/теням: ``ls_r31`` = ch3/ch1 (оксиды железа),
      ``ls_r57`` = ch5/ch7 (глины/гидроксилы), ``ls_r54`` = ch5/ch4
      (ферро-минералы), ``ls_ndvi`` = (ch4-ch3)/(ch4+ch3);
    * пропуски ``relief_m`` заливаются из Copernicus DEM
      (:func:`fill_relief_from_dem`, graceful fallback);
    * ``dist_dnl``/``dist_dnara`` — только при ``include_dist=True``.
    """
    base = cell_mask.feature_columns(df)
    drop = set(GRADIENT_REDUNDANT) | set(FIELD_REDUNDANT) | set(LS_RAW)
    for name in ANGLE_BASES:
        drop |= {name, f"{name}_sin", f"{name}_cos"}
    if not include_dist:
        drop |= set(DIST_COLS)
    out = df[[c for c in base if c not in drop]].copy()

    ch = {i: df[f"ls_ch{i}"].to_numpy(dtype=float) for i in (1, 3, 4, 5, 7)
          if f"ls_ch{i}" in df.columns}
    if len(ch) == 5:
        out["ls_r31"] = _safe_ratio(ch[3], ch[1])
        out["ls_r57"] = _safe_ratio(ch[5], ch[7])
        out["ls_r54"] = _safe_ratio(ch[5], ch[4])
        out["ls_ndvi"] = _safe_ratio(ch[4] - ch[3], ch[4] + ch[3])

    for name in ANGLE_BASES:
        if name in df.columns:
            rad2 = 2.0 * np.deg2rad(df[name].to_numpy(dtype=float))
            out[f"{name}_cos2"] = np.cos(rad2)
            out[f"{name}_sin2"] = np.sin(rad2)

    if "relief_m" in out.columns:
        out["relief_m"] = fill_relief_from_dem(df)
    return out


# --------------------------------------------------------------------------
# Заливка relief_m из Copernicus DEM
# --------------------------------------------------------------------------

def _dem_elevation_at_cells(df: pd.DataFrame) -> np.ndarray:
    """Высота Copernicus DEM GLO-30 в центрах ячеек (x/y сетки -> WGS84).

    Постоянный кэш вектора высот в ``datacache/anabar_dem`` (та же папка, что
    у :mod:`src.dem_features`); тайлы читаются rasterio по HTTPS (COG, AWS,
    anonymous) — GDAL-мозаика старого пайплайна не требуется.
    """
    from cache_paths import ANABAR_DEM

    cache = ANABAR_DEM / f"dem_elev_cells_{len(df)}.npy"
    if cache.exists():
        return np.load(cache)

    import rasterio
    from pyproj import CRS, Transformer

    from . import integro_grid

    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    if proj4 is None:
        raise RuntimeError("proj4 целевой сетки не найден (sidecar .pj4)")
    tr = Transformer.from_crs(CRS.from_proj4(proj4), CRS.from_epsg(4326),
                              always_xy=True)
    lon, lat = tr.transform(df["x"].to_numpy(), df["y"].to_numpy())
    lon, lat = np.asarray(lon), np.asarray(lat)

    elev = np.full(len(df), np.nan)
    tiles = sorted({(int(np.floor(la)), int(np.floor(lo)))
                    for la, lo in zip(lat, lon)})
    for la, lo in tiles:
        sel = (np.floor(lat).astype(int) == la) & (np.floor(lon).astype(int) == lo)
        la_s = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
        lo_s = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
        name = f"Copernicus_DSM_COG_10_{la_s}_00_{lo_s}_00_DEM"
        url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
        with rasterio.open(url) as ds:
            arr = ds.read(1)
            rows, cols = rasterio.transform.rowcol(ds.transform, lon[sel], lat[sel])
            rows = np.clip(np.asarray(rows), 0, arr.shape[0] - 1)
            cols = np.clip(np.asarray(cols), 0, arr.shape[1] - 1)
            elev[sel] = arr[rows, cols]

    ANABAR_DEM.mkdir(parents=True, exist_ok=True)
    np.save(cache, elev)
    return elev


def fill_relief_from_dem(df: pd.DataFrame) -> pd.Series:
    """Заполнить NaN в ``relief_m`` предсказанием из Copernicus DEM.

    Калибровка линейная (OLS ``relief_m ~ a*dem_elev + b`` по ячейкам без
    пропусков, ожидаемый наклон ~1 — обе величины в метрах); заполняются
    ТОЛЬКО пропуски, наблюдённые значения не трогаются. При недоступности
    DEM (нет сети/кэша) — предупреждение и возврат как есть (NaN подхватит
    импутация медианой), пайплайн не падает. Калибровка печатается в консоль.
    """
    relief = df["relief_m"].astype(float).copy()
    na = relief.isna().to_numpy()
    if not na.any():
        return relief
    try:
        dem = _dem_elevation_at_cells(df)
    except Exception as exc:  # noqa: BLE001 — graceful fallback на офлайн
        warnings.warn(f"Copernicus DEM недоступен ({exc}) — relief_m остаётся с NaN")
        return relief

    obs = (~na) & np.isfinite(dem)
    if obs.sum() < 10:
        warnings.warn("Слишком мало ячеек для калибровки relief~DEM — пропуски не заполнены")
        return relief
    y = relief.to_numpy()[obs]
    x = dem[obs]
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    r2 = 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    fill = na & np.isfinite(dem)
    relief.iloc[np.flatnonzero(fill)] = a * dem[fill] + b
    print(f"relief_m ~ DEM: a={a:.4f}, b={b:.1f} м, R^2={r2:.4f}; "
          f"заполнено {int(fill.sum())} из {int(na.sum())} NaN")
    if fill.sum() < na.sum():
        warnings.warn(f"{int(na.sum() - fill.sum())} ячеек relief_m без DEM — остаются NaN")
    return relief


# --------------------------------------------------------------------------
# Диагностика мультиколлинеарности
# --------------------------------------------------------------------------

def condition_number(feat_df: pd.DataFrame, valid: np.ndarray) -> float:
    """Число обусловленности корреляционной матрицы признаков (валидные ячейки).

    Матрица — после робастной стандартизации :func:`src.atypicality.prepare_matrix`
    (медиана/IQR + импутация); колонки с нулевой дисперсией отбрасываются.
    Большое значение (>> 1e3) — почти линейно зависимые признаки: Махаланобис
    и PCA-невязка становятся численно шаткими.
    """
    from .atypicality import prepare_matrix

    X = prepare_matrix(feat_df, valid)
    keep = X.std(axis=0) > 0
    R = np.corrcoef(X[:, keep], rowvar=False)
    ev = np.linalg.eigvalsh(R)
    return float(ev.max() / max(ev.min(), 1e-12))
