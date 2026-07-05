"""Перенос U-модели США-Запад -> сетка Анабара R-48 (та же проектная сетка 500 м,
проекция Красовского, что и боевые карты прогноза).

Модель урана обучается на рудопроявлениях MRDS (commod=Uranium, США-Запад) в
пространстве ГЛОБАЛЬНОЙ геофизики (mag4km, faa, vgg, geoid, relief) и применяется
к ячейкам сетки Анабара. ВАЖНО: радиометрия NURE есть только по США — над Сибирью
её нет, поэтому переносится ТОЛЬКО геофизическая часть U-модели (это честно
подписано на карте). Не валидированный прогноз (меток урана по Анабару для
проверки нет), а transfer-эксперимент «подписи фертильности урана по геофизике».

Запуск: python3 -m experiments.anabar_uranium_transfer
"""

import tempfile, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

import glob
from src import config
from src.data_loader import find_base_dir, load_all_layers, load_layer, to_crs_safe
from src.features import build_grid
from src.model import mark_presence
from experiments.common import sample_global, DS, SCALE
from cache_paths import ANABAR_GEO, MRDS_CSV

EXT = (60.0, 120.0, 30.0, 90.0)            # тайл N30E060: lon0,lon1,lat0,lat1
US_WEST = dict(lon=(-124.7, -108.2), lat=(31.3, 49.0))   # как WesternUS у NURE
N_BG, SEED = 5000, 42


def sample_anabar(ds, lon, lat):
    """Сэмпл тайла N30E060 (Анабар) — как в anabar_znpb."""
    a = np.array(Image.open(f"{ANABAR_GEO}/{ds}_N30E060.jp2")).astype(float)
    a[a == 0] = np.nan; a *= SCALE[ds]
    n = a.shape[0]; dx = (EXT[1] - EXT[0]) / n
    c = ((lon - EXT[0]) / dx).astype(int); r = ((EXT[3] - lat) / dx).astype(int)
    ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
    out = np.full(lon.shape, np.nan); out[ok] = a[r[ok], c[ok]]
    return out


def main():
    t0 = time.time()

    # --- обучаем U-модель на США-Запад (MRDS уран + фон) в пространстве DS ---
    mrds = pd.read_csv(str(MRDS_CSV), low_memory=False,
                       usecols=["latitude", "longitude", "commod1", "commod2", "commod3", "country"])
    mrds = mrds[mrds.country == "United States"].dropna(subset=["latitude", "longitude"])
    mrds = mrds[mrds.longitude.between(*US_WEST["lon"]) & mrds.latitude.between(*US_WEST["lat"])]
    comm = (mrds.commod1.fillna("") + ";" + mrds.commod2.fillna("") + ";" + mrds.commod3.fillna(""))
    u = mrds[comm.str.contains("Uranium", case=False, na=False)]
    plon, pla = u.longitude.to_numpy(), u.latitude.to_numpy()

    rng = np.random.default_rng(SEED)
    blon = rng.uniform(*US_WEST["lon"], N_BG * 3); bla = rng.uniform(*US_WEST["lat"], N_BG * 3)
    fmat = lambda lo, la: np.column_stack([sample_global(d, lo, la) for d in DS])
    Xp, Xb = fmat(plon, pla), fmat(blon, bla)
    okp, okb = np.isfinite(Xp).all(1), np.isfinite(Xb).all(1)
    Xp = Xp[okp]; Xb = Xb[okb][:N_BG]
    Xtr = np.vstack([Xp, Xb]); ytr = np.r_[np.ones(len(Xp)), np.zeros(len(Xb))].astype(int)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED).fit(Xtr, ytr)
    print(f"[{time.time()-t0:3.0f}s] U-модель обучена: {len(Xp)} урановых рудопроявл. + {len(Xb)} фон (США-Запад)")

    # --- проектная сетка Анабара R-48 ---
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid, pos = mark_presence(grid, points)
    crs = layers["mask"].crs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cents = grid.geometry.centroid.to_crs(4326)
    glon, glat = cents.x.to_numpy(), cents.y.to_numpy()
    Xg = np.column_stack([sample_anabar(d, glon, glat) for d in DS])

    score = np.full(len(grid), np.nan)
    ok = np.isfinite(Xg).all(1)
    score[ok] = gb.predict_proba(Xg[ok])[:, 1]
    print(f"[{time.time()-t0:3.0f}s] сетка {len(grid)} ячеек, валидных {ok.sum()}; "
          f"U-score min/max {np.nanmin(score):.2f}/{np.nanmax(score):.2f}, среднее {np.nanmean(score):.2f}")

    # --- карта (Красовский, метры), точки Au-U поверх ---
    rows = grid["row"].to_numpy(); cols = grid["col"].to_numpy()
    arr = np.full(shape, np.nan); arr[rows, cols] = score
    minx, miny, maxx, maxy = layers["mask"].total_bounds
    extent = [minx, minx + shape[1] * config.CELL_SIZE, miny, miny + shape[0] * config.CELL_SIZE]

    fig, ax = plt.subplots(figsize=(8, 8.4))
    im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=extent,
                   cmap="YlOrRd", vmin=0, vmax=float(np.nanmax(score)), interpolation="nearest")
    layers["mask"].boundary.plot(ax=ax, color="0.4", linewidth=0.3, alpha=0.5)
    real = grid[grid["cell_id"].isin(pos)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = real.geometry.centroid
    ax.scatter(rc.x, rc.y, s=18, facecolor="none", edgecolor="blue", linewidths=0.9,
               label="известные Au-U точки")

    # независимые геохимические ореолы (НЕ участвовали в обучении) — поверх
    halo_f = glob.glob(str(base / config.SHP_SUBDIR / "*ореол*.shp"))
    if halo_f:
        halos = to_crs_safe(load_layer(Path(halo_f[0])), crs)
        styles = {"U": ("magenta", 2.2, "ореол U (независимый)"),
                  "Au": ("lime", 1.6, "ореол Au (независимый)")}
        for el, (col, lw, lab) in styles.items():
            sub = halos[halos["label"] == el]
            if len(sub):
                sub.plot(ax=ax, color=col, linewidth=lw, label=lab)
        n_u = int((halos["label"] == "U").sum()); n_au = int((halos["label"] == "Au").sum())
        print(f"[{time.time()-t0:3.0f}s] наложено ореолов: U={n_u}, Au={n_au}")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("U-фертильность (перенос США-Запад, ТОЛЬКО геофизика)")
    ax.set_title("Анабар, лист R-48 · уран по переносу модели США-Запад\n"
                 "геофизика mag4km/faa/vgg/geoid/relief (радиометрии по Сибири нет) · сетка 500 м",
                 fontsize=9.5)
    ax.set_xlabel("X, м (Гаусса–Крюгера, Красовский)"); ax.set_ylabel("Y, м")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    fig.savefig("outputs/anabar_uranium_transfer.png", dpi=150, bbox_inches="tight")
    print(f"[{time.time()-t0:3.0f}s] saved outputs/anabar_uranium_transfer.png")


if __name__ == "__main__":
    main()
