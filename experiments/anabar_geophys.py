"""P2 на Анабаре с ПОЛНОЙ глобальной геофизикой (GMT, тайлы N30E060, 2', с сушей):
магнитка ΔT (EMAG2v3 4км), free-air гравика, Буге-прокси, вертикальный градиент
гравики (VGG), геоид (EGM2008), рельеф + градиенты магнитки/гравики.

Сравнение: критериальный (ГИС Интегро) vs ML(текущие) vs ML(+геофизика) vs
ML(только геофизика). Метрика — lift@10% по точкам Au-U, пространственная CV.

Запуск: python3 -m experiments.anabar_geophys
"""

import tempfile, warnings
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, compute_geo_score
from src.model import mark_presence, sample_presence_background
from src.validation import assign_spatial_blocks, _coverage
from cache_paths import ANABAR_GEO

GEO = str(ANABAR_GEO)
EXT = (60.0, 120.0, 30.0, 90.0)   # тайл N30E060: lon0,lon1,lat0,lat1
SCALE = {"mag4km": 0.4, "faa": 0.025, "vgg": 0.03125, "geoid": 0.01, "relief": 0.5}


def load_tile(name):
    a = np.array(Image.open(f"{GEO}/{name}_N30E060.jp2")).astype(float)
    a[a == 0] = np.nan
    return a * SCALE[name]          # в физ. единицы (offset опускаем — для признака неважно)


def sampler(arr):
    n = arr.shape[0]; dx = (EXT[1] - EXT[0]) / n

    def s(lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        c = ((lon - EXT[0]) / dx).astype(int); r = ((EXT[3] - lat) / dx).astype(int)
        ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
        out = np.full(lon.shape, np.nan); out[ok] = arr[r[ok], c[ok]]
        return out
    return s


def gradmag(arr):
    filled = np.where(np.isfinite(arr), arr, np.nanmean(arr))
    gy, gx = np.gradient(filled); g = np.hypot(gx, gy); g[~np.isfinite(arr)] = np.nan
    return g


def main():
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers); grid = compute_geo_score(grid, shape)
    grid, pos = mark_presence(grid, points)

    mag = load_tile("mag4km"); faa = load_tile("faa"); vgg = load_tile("vgg")
    geoid = load_tile("geoid"); relief = load_tile("relief")
    bouguer = faa - 0.1119 * relief          # простая Буге-редукция (ρ=2.67)
    fields = {"mag4km": mag, "mag4km_grad": gradmag(mag), "faa": faa,
              "bouguer": bouguer, "vgg": vgg, "geoid": geoid, "relief": relief}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cents = grid.geometry.centroid.to_crs(4326)
    glon, glat = cents.x.to_numpy(), cents.y.to_numpy()
    geo_cols = []
    for nm, arr in fields.items():
        col = "geo_" + nm; grid[col] = sampler(arr)(glon, glat); geo_cols.append(col)
    print(f"сетка {len(grid)}, точек Au-U {len(pos)}; геофизич. признаков: {len(geo_cols)} {geo_cols}")
    print(f"  магнитка ΔT валидна в {np.isfinite(grid['geo_mag4km']).sum()}/{len(grid)} ячейках")

    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    crit = grid["geo_score"].to_numpy(); a = 0.10
    res = {"Критериальный (ГИС Интегро)": [], "ML (текущие признаки)": [],
           "ML (+ геофизика)": [], "ML (только геофизика)": []}
    for sd in (1, 7, 13, 21, 42):
        sp, y = sample_presence_background(grid, pos, config.VAL_N_BACKGROUND, sd)
        g = blocks[sp]
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(np.zeros(len(sp)), y, g):
            tp = sp[te][y[te] == 1]
            if tp.size == 0:
                continue
            res["Критериальный (ГИС Интегро)"].append(_coverage(crit, tp, a) / a)
            for label, fl in [("ML (текущие признаки)", config.FEATURE_COLS),
                              ("ML (+ геофизика)", config.FEATURE_COLS + geo_cols),
                              ("ML (только геофизика)", geo_cols)]:
                rf = RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=10,
                    class_weight="balanced_subsample", random_state=sd, n_jobs=-1)
                Xs = grid.iloc[sp][fl].fillna(0).to_numpy()
                rf.fit(Xs[tr], y[tr])
                sc = rf.predict_proba(grid[fl].fillna(0).to_numpy())[:, 1]
                res[label].append(_coverage(sc, tp, a) / a)
    print("\n=== Анабар, lift@10% по Au-U (пространственная CV, 5 сидов) ===")
    for k, v in res.items():
        v = np.array(v); print(f"  {k:28s} lift {v.mean():.2f} ± {v.std():.2f}")


if __name__ == "__main__":
    main()
