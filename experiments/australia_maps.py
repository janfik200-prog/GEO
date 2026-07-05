"""Австралия: карты прогноза двумя методами + метрики.

Слева — критериальный метод (аналог ГИС Интегро: нормировка факторов + взвешенное
комбинирование), справа — наш ML (ансамбль RF+GB). Признаки — геофизические гриды
по Австралии (Moho, LAB, спутниковая гравика). Метки — рудопроявления Zn-Pb (751).
Оценка качества — пространственная CV: ROC-AUC и lift@10%.

Запуск: python3 -m experiments.australia_maps
"""

import glob
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.common import _read_geotiff, sample_raster
from cache_paths import CMMI_AU, CMMI_OCC

DATA_AU = str(CMMI_AU)
OCC_CSV = str(CMMI_OCC)
LON = (112.0, 154.0)
LAT = (-44.0, -9.0)
RES = 0.1
SEED = 42
N_BG = 4000
TOP = 0.10


def criterial(Xtr, ytr, Xq):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zq = (Xtr - mu) / sd, (Xq - mu) / sd
    sign = np.sign([np.corrcoef(Ztr[:, j], ytr)[0, 1] for j in range(Xtr.shape[1])])
    return (Zq * sign).mean(1)


def ml_ensemble(Xtr, ytr, Xq):
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=6,
                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED).fit(Xtr, ytr)
    return 0.5 * (rf.predict_proba(Xq)[:, 1] + gb.predict_proba(Xq)[:, 1])


def main():
    t0 = time.time()
    rasters = sorted(glob.glob(f"{DATA_AU}/*.tif"))
    names = [r.split("/")[-1] for r in rasters]
    print("геофизика Австралии:", names)
    if not rasters:
        print(f"нет .tif в {DATA_AU}"); return

    # регулярная сетка
    lons = np.arange(LON[0], LON[1], RES) + RES / 2
    lats = np.arange(LAT[0], LAT[1], RES) + RES / 2
    GX, GY = np.meshgrid(lons, lats)
    glon, glat = GX.ravel(), GY.ravel()
    Xgrid = np.column_stack([sample_raster(r, glon, glat) for r in rasters])
    valid = np.isfinite(Xgrid).all(1)
    print(f"[{time.time()-t0:3.0f}s] сетка {GX.shape}, валидных ячеек: {valid.sum()}")

    occ = pd.read_csv(OCC_CSV, encoding="latin1", low_memory=False)
    au = occ[(occ.Admin == "Australia")].dropna(subset=["Longitude", "Latitude"])
    plon, plat = au.Longitude.to_numpy(), au.Latitude.to_numpy()
    Xpos = np.column_stack([sample_raster(r, plon, plat) for r in rasters])
    okp = np.isfinite(Xpos).all(1)
    Xpos, plon, plat = Xpos[okp], plon[okp], plat[okp]

    rng = np.random.default_rng(SEED)
    bgidx = rng.choice(np.where(valid)[0], min(N_BG, valid.sum()), replace=False)
    Xbg = Xgrid[bgidx]
    blon, blat = glon[bgidx], glat[bgidx]
    print(f"[{time.time()-t0:3.0f}s] presence={len(Xpos)}, background={len(Xbg)}")

    Xtr = np.vstack([Xpos, Xbg])
    ytr = np.r_[np.ones(len(Xpos)), np.zeros(len(Xbg))].astype(int)

    # карты по всей сетке
    crit_grid = np.full(len(glon), np.nan)
    ml_grid = np.full(len(glon), np.nan)
    crit_grid[valid] = criterial(Xtr, ytr, Xgrid[valid])
    ml_grid[valid] = ml_ensemble(Xtr, ytr, Xgrid[valid])

    def pct(a):
        o = np.full(a.shape, np.nan); m = np.isfinite(a)
        o[m] = np.argsort(np.argsort(a[m])) / (m.sum() - 1); return o

    # метрики: пространственная CV
    lon_all = np.r_[plon, blon]; lat_all = np.r_[plat, blat]
    groups = np.floor(lon_all / 2).astype(int) * 1000 + np.floor(lat_all / 2).astype(int)
    print(f"\n=== Австралия, {len(Xpos)} меток, пространственная CV ===")
    print(f"{'метод':32s} {'ROC-AUC':>10s} {'lift@10%':>10s}")
    for label, fn in [("Критериальный (ГИС Интегро)", criterial), ("Наш ML (ансамбль RF+GB)", ml_ensemble)]:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs, lifts = [], []
        for tr, te in sgkf.split(Xtr, ytr, groups):
            s = fn(Xtr[tr], ytr[tr], Xtr[te])
            aucs.append(roc_auc_score(ytr[te], s))
            thr = np.quantile(s[ytr[te] == 0], 1 - TOP)
            lifts.append(float((s[ytr[te] == 1] >= thr).mean()) / TOP)
        print(f"{label:32s} {np.mean(aucs):.3f}±{np.std(aucs):.3f} {np.mean(lifts):5.2f}±{np.std(lifts):.2f}")

    # рисуем
    extent = [LON[0], LON[1], LAT[0], LAT[1]]
    shp = GX.shape
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)
    for a, grid, title in [(ax[0], crit_grid, "Метод ГИС Интегро (критериальный)\nАвстралия · Zn-Pb"),
                           (ax[1], ml_grid, "Наш метод — ML (ансамбль RF+GB)\nАвстралия · Zn-Pb")]:
        im = a.imshow(np.ma.masked_invalid(pct(grid).reshape(shp)), origin="lower", extent=extent,
                      cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        a.scatter(plon, plat, s=12, facecolor="none", edgecolor="black", linewidths=0.6)
        a.set_title(title, fontsize=11); a.set_xlabel("Долгота"); a.set_aspect("equal")
    ax[0].set_ylabel("Широта")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("перспективность (перцентиль; тёмно-красное = выше)")
    fig.suptitle("Австралия · прогноз Zn-Pb: критериальный (ГИС Интегро) vs наш ML · ○ = известные рудопроявления", fontsize=12)
    fig.savefig("outputs/australia_compare.png", dpi=140, bbox_inches="tight")
    print(f"\n[{time.time()-t0:3.0f}s] saved australia_compare.png")


if __name__ == "__main__":
    main()
