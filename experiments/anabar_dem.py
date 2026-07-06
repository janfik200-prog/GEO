"""Рельеф Copernicus DEM GLO-30 на Анабаре — производные как признаки, held-out.

SRTM не покрывает 71° N (предел 60°), поэтому берём Copernicus DEM GLO-30 (30 м,
до полюсов, открыто на AWS, anonymous /vsicurl). DEM покрывает сушу ПОЛНОСТЬЮ —
нет missingness-артефакта, погубившего ДЗЗ (там 34% покрытия). Производные под
тип Au-U (палеодолины + коры выветривания):

  * dem_elev   — высота;
  * dem_slope  — уклон (|grad|);
  * dem_curv   — кривизна (лапласиан): долины<0 / хребты>0;
  * dem_tpi    — topographic position (отн. окрестности): палеодолины отрицательны;
  * dem_rough  — шероховатость (локальная дисперсия): структурная раздробленность.

Метрика — lift@10% по Au-U, пространственная CV, 5 сидов. Встроена проверка
валидности (доля валидных у точек vs фон) — урок ДЗЗ.

Запуск: python3 -m experiments.anabar_dem
"""

import tempfile, warnings
from pathlib import Path

import numpy as np
from osgeo import gdal
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, compute_geo_score
from src.model import mark_presence, sample_presence_background
from src.validation import assign_spatial_blocks, _coverage
from experiments.feat_enrich import boot_ci

gdal.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "datacache" / "anabar_dem"
CACHE.mkdir(parents=True, exist_ok=True)

BBOX = (106.0, 70.67, 108.0, 71.34)
TR = (0.004, 0.0016)                        # ~150 м
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def dem_url(lat: str, lon: str) -> str:
    name = f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"
    return f"/vsicurl/{BASE}/{name}/{name}.tif"


def build_dem() -> Path:
    out = CACHE / "dem.tif"
    if out.exists():
        return out
    srcs = [dem_url(lat, lon) for lat in ("N70", "N71") for lon in ("E106", "E107", "E108")]
    print(f"  warp {len(srcs)} DEM-тайлов -> {out.name}")
    gdal.Warp(str(out), srcs, dstSRS="EPSG:4326", outputBounds=BBOX,
              xRes=TR[0], yRes=TR[1], resampleAlg="bilinear")
    return out


def read_raster(path: Path):
    ds = gdal.Open(str(path))
    a = ds.GetRasterBand(1).ReadAsArray().astype(float)
    return a, ds.GetGeoTransform()


def sampler(arr, gt):
    x0, px, _, y0, _, py = gt
    n_r, n_c = arr.shape

    def s(lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        c = ((lon - x0) / px).astype(int)
        r = ((lat - y0) / py).astype(int)
        ok = (c >= 0) & (c < n_c) & (r >= 0) & (r < n_r)
        out = np.full(lon.shape, np.nan)
        out[ok] = arr[r[ok], c[ok]]
        return out
    return s


def derivatives(elev):
    e = np.where(np.isfinite(elev), elev, np.nanmean(elev))
    gy, gx = np.gradient(e)
    slope = np.hypot(gx, gy)
    curv = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)     # лапласиан
    tpi = e - uniform_filter(e, size=15)                         # ~2 км окно
    rough = np.sqrt(np.clip(uniform_filter(e**2, 9) - uniform_filter(e, 9) ** 2, 0, None))
    return {"dem_elev": elev, "dem_slope": slope, "dem_curv": curv,
            "dem_tpi": tpi, "dem_rough": rough}


def main():
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = compute_geo_score(grid, shape)
    grid, pos = mark_presence(grid, points)

    print("Готовлю Copernicus DEM (кэш datacache/anabar_dem):")
    elev, gt = read_raster(build_dem())
    deriv = derivatives(elev)
    samp = {k: sampler(v, gt) for k, v in deriv.items()}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cents = grid.geometry.centroid.to_crs(4326)
    glon, glat = cents.x.to_numpy(), cents.y.to_numpy()
    dem_cols = list(deriv)
    for k in dem_cols:
        grid[k] = samp[k](glon, glat)

    # --- проверка валидности (урок ДЗЗ: missingness не должен кодировать географию) ---
    valid = np.isfinite(grid["dem_elev"].to_numpy())
    presence = grid["presence"].to_numpy().astype(int)
    pv, bv = valid[presence == 1].mean(), valid[presence == 0].mean()
    print(f"  сетка {len(grid)}, точек {len(pos)}; DEM-признаков {len(dem_cols)}")
    print(f"  валидность DEM: у точек {pv*100:.0f}%, у фона {bv*100:.0f}% "
          f"(отношение {pv/bv:.2f}x — близко к 1.0 = нет missingness-артефакта)")

    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    crit = grid["geo_score"].to_numpy()
    AREAS = (0.05, 0.10)
    labels = ["Критериальный (ГИС Интегро)", "ML (текущие)", "ML (+ DEM)", "ML (только DEM)"]
    feat = {"ML (текущие)": config.FEATURE_COLS, "ML (+ DEM)": config.FEATURE_COLS + dem_cols,
            "ML (только DEM)": dem_cols}
    # пары на ОДНОМ разбиении (sd, fold) — для парного bootstrap
    pairs = {a: {lab: [] for lab in labels} for a in AREAS}
    for sd in (1, 7, 13, 21, 42):
        sp, y = sample_presence_background(grid, pos, config.VAL_N_BACKGROUND, sd)
        g = blocks[sp]
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(np.zeros(len(sp)), y, g):
            tp = sp[te][y[te] == 1]
            if tp.size == 0:
                continue
            for a in AREAS:
                pairs[a]["Критериальный (ГИС Интегро)"].append(_coverage(crit, tp, a) / a)
            for lab, cols in feat.items():
                rf = RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=10,
                    class_weight="balanced_subsample", random_state=sd, n_jobs=-1)
                X = grid[cols].fillna(grid[cols].median()).to_numpy()
                rf.fit(X[sp][tr], y[tr])
                sc = rf.predict_proba(X)[:, 1]
                for a in AREAS:
                    pairs[a][lab].append(_coverage(sc, tp, a) / a)

    n = len(pairs[0.10]["ML (текущие)"])
    for a in AREAS:
        print(f"\n=== Анабар, lift@{int(a*100)}% по Au-U (held-out, 5 сидов, {n} пар) ===")
        for lab in labels:
            v = np.array(pairs[a][lab])
            print(f"  {lab:28s} lift {v.mean():.2f} ± {v.std():.2f}")
        d = np.array(pairs[a]["ML (+ DEM)"]) - np.array(pairs[a]["ML (текущие)"])
        ci = boot_ci(d, config.VAL_SEED)
        sig = "ДА" if (ci[0] > 0 or ci[1] < 0) else "НЕТ"
        print(f"  разница (+DEM − текущие): {d.mean():+.2f}x "
              f"(95% ДИ [{ci[0]:+.2f}, {ci[1]:+.2f}]) | значимо: {sig}")


if __name__ == "__main__":
    main()
