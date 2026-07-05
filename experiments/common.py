"""Общий код исследовательских прогонов (experiments/*).

Сюда вынесено всё, что раньше дублировалось по run_*.py:
  • глобальная геофизика GMT (тайлы 60° JP2): sample_global, region_xy, DS, SCALE;
  • чтение GeoTIFF через PIL (без rasterio): sample_raster;
  • метрики/baseline: criterial (линейный аналог ГИС Интегро), lift@N%.

Пути берутся из общего постоянного кэша (см. cache_paths). Запуск скриптов —
из корня репозитория модулем: `python3 -m experiments.<имя>`.
"""
import os
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from cache_paths import WORLD_GEO

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- глобальная геофизика GMT
GEODIR = str(WORLD_GEO)
SERVER = "https://oceania.generic-mapping-tools.org/server/earth"
DS = ["mag4km", "faa", "vgg", "geoid", "relief"]
SCALE = {"mag4km": 0.4, "faa": 0.025, "vgg": 0.03125, "geoid": 0.01, "relief": 0.5}
N_BG, SEED = 3000, 42

_tile_cache: dict = {}


def tile_name(lon, lat):
    """Имя 60°-тайла GMT (N..E..) и координаты его юго-западного угла."""
    lon_sw = int(np.floor(lon / 60.0) * 60)
    lat_sw = int(np.floor((lat + 90) / 60.0) * 60 - 90)
    ln = f"E{lon_sw:03d}" if lon_sw >= 0 else f"W{-lon_sw:03d}"
    lt = f"N{lat_sw:02d}" if lat_sw >= 0 else f"S{-lat_sw:02d}"
    return f"{lt}{ln}", lon_sw, lat_sw


def fetch(ds, tile):
    """Скачать тайл в постоянный кэш (идемпотентно, с докачкой)."""
    p = f"{GEODIR}/{ds}_{tile}.jp2"
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    url = f"{SERVER}/earth_{ds}/earth_{ds}_02m_p/{tile}.earth_{ds}_02m_p.jp2"
    subprocess.run(["curl", "-sS", "-L", "--connect-timeout", "20", "--max-time", "400",
                    "--retry", "8", "--retry-all-errors", "-C", "-", "-o", p, url],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p if (os.path.exists(p) and os.path.getsize(p) > 1000) else None


def sample_global(ds, lon, lat):
    """Сэмпл глобального грида ds в точках (lon, lat); докачивает нужные тайлы."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    out = np.full(len(lon), np.nan)
    names = np.array([tile_name(lo, la)[0] for lo, la in zip(lon, lat)])
    for tl in np.unique(names):
        m = names == tl
        _, lsw, bsw = tile_name(lon[m][0], lat[m][0])
        key = (ds, tl)
        if key not in _tile_cache:
            p = fetch(ds, tl)
            if p is None:
                _tile_cache[key] = None
            else:
                a = np.array(Image.open(p)).astype(float); a[a == 0] = np.nan
                _tile_cache[key] = (a, lsw, bsw)
        if _tile_cache[key] is None:
            continue
        a, lsw, bsw = _tile_cache[key]; n = a.shape[0]; dx = 60.0 / n
        c = ((lon[m] - lsw) / dx).astype(int); r = (((bsw + 60) - lat[m]) / dx).astype(int)
        ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
        vals = np.full(m.sum(), np.nan); vals[ok] = a[r[ok], c[ok]]
        out[np.where(m)[0]] = vals * SCALE[ds]
    return out


def region_xy(occ, admin, lonr, latr, seed, n_bg=N_BG):
    """presence-background матрица признаков (глоб. геофизика) для региона."""
    sub = occ[occ.Admin.isin(admin)].dropna(subset=["Longitude", "Latitude"])
    sub = sub[sub.Longitude.between(*lonr) & sub.Latitude.between(*latr)]
    pl, pa = sub.Longitude.to_numpy(), sub.Latitude.to_numpy()
    rng = np.random.default_rng(seed)
    bl = rng.uniform(*lonr, n_bg * 3); ba = rng.uniform(*latr, n_bg * 3)
    fmat = lambda lo, la: np.column_stack([sample_global(d, lo, la) for d in DS])
    Xp, Xb = fmat(pl, pa), fmat(bl, ba)
    okp = np.isfinite(Xp).all(1); okb = np.isfinite(Xb).all(1)
    Xp, pl, pa = Xp[okp], pl[okp], pa[okp]
    Xb, bl, ba = Xb[okb][:n_bg], bl[okb][:n_bg], ba[okb][:n_bg]
    X = np.vstack([Xp, Xb]); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xb))].astype(int)
    lon = np.r_[pl, bl]; lat = np.r_[pa, ba]
    return X, y, lon, lat


# ---------------------------------------------------------------- чтение GeoTIFF (PIL, без rasterio)
_raster_cache: dict = {}


def _epsg_from_geokeys(t):
    gk = t.get(34735)
    if not gk:
        return None
    n = gk[3]
    for i in range(1, n + 1):
        kid, loc, _cnt, val = gk[i * 4: i * 4 + 4]
        if kid in (3072, 2048) and loc == 0:
            return int(val)
    return None


def _read_geotiff(path):
    img = Image.open(path)
    arr = np.array(img, dtype=float)
    t = img.tag_v2
    scale, tie = t.get(33550), t.get(33922)
    if not (scale and tie):
        raise ValueError(f"{path}: нет ModelPixelScale/Tiepoint")
    sx, sy = float(scale[0]), float(scale[1])
    I, J, _, X, Y, _ = [float(v) for v in tie[:6]]
    x0, y0 = X - I * sx, Y + J * sy          # мир в пикселе (0,0)
    nod = t.get(42113)
    nodata = float(nod) if nod not in (None, "") else None
    return arr, (x0, y0, sx, sy), nodata, _epsg_from_geokeys(t)


def sample_raster(path, lon, lat):
    if path not in _raster_cache:
        _raster_cache[path] = _read_geotiff(path)
    arr, (x0, y0, sx, sy), nodata, epsg = _raster_cache[path]
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    if epsg and epsg != 4326:
        X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    else:
        X, Y = lon, lat
    col = np.floor((np.asarray(X) - x0) / sx).astype(int)
    row = np.floor((y0 - np.asarray(Y)) / sy).astype(int)
    H, W = arr.shape
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    out = np.full(lon.shape, np.nan)
    out[inb] = arr[row[inb], col[inb]]
    if nodata is not None:
        out[out == nodata] = np.nan
    out[np.abs(out) > 1e30] = np.nan
    return out


# ---------------------------------------------------------------- метрики / baseline
def criterial(Xtr, ytr, Xte):
    """Линейный критериальный индекс (аналог ГИС Интегро): z-норм + знак корреляции."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z1, Z2 = (Xtr - mu) / sd, (Xte - mu) / sd
    sg = np.sign([np.corrcoef(Z1[:, j], ytr)[0, 1] for j in range(Xtr.shape[1])])
    return (Z2 * sg).mean(1)


def lift(score, y, a=0.10):
    """Lift@a: во сколько раз доля попавших меток в top-a% выше случайной."""
    thr = np.quantile(score[y == 0], 1 - a)
    return float((score[y == 1] >= thr).mean()) / a
