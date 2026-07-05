"""P2: проверка метода на ВТОРОЙ территории (США+Канада) с тысячами меток.

Датасет CMMI/Lawley (USGS): рудопроявления Zn-Pb + геофизические гриды (глубина
LAB, Moho и др.). Гипотеза проекта проверяется там, где меток на 2 порядка
больше, чем на Анабаре: линейное «критериальное» комбинирование факторов против
нелинейного ML, честно — пространственная блочная CV, AUC и lift@10%.

GeoTIFF читается через PIL (без rasterio): гео-теги дают привязку, CRS, nodata.
Данные в постоянном кэше datacache/cmmi/ (cmmi_occ.csv + *.tif), путь — из
cache_paths. Запуск: python3 -m experiments.cmmi_terrain
"""

import glob
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cache_paths import CMMI, CMMI_OCC
from experiments.common import _read_geotiff, sample_raster

DATA = str(CMMI)
OCC_CSV = str(CMMI_OCC)
BBOX = dict(lon=(-170.0, -52.0), lat=(25.0, 75.0))   # США+Канада
N_BG = 6000
SEED = 42
N_SPLITS = 5
TOP_AREA = 0.10
N_BOOT = 2000


def main():
    t0 = time.time()
    rasters = sorted(glob.glob(f"{DATA}/*.tif"))
    names = [r.split("/")[-1].replace(".tif", "") for r in rasters]
    print(f"геофизических слоёв: {len(rasters)} -> {names}")
    if not rasters:
        print(f"НЕТ .tif в {DATA} — сначала скачать гриды."); return
    for r in rasters:
        a, (x0, y0, sx, sy), nod, epsg = _read_geotiff(r)
        print(f"  {r.split('/')[-1]:32s} shape={a.shape} epsg={epsg} nodata={nod}")

    occ = pd.read_csv(OCC_CSV, encoding="latin1", low_memory=False)
    occ = occ[occ.Admin != "Australia"].dropna(subset=["Longitude", "Latitude"])
    occ = occ[occ.Longitude.between(*BBOX["lon"]) & occ.Latitude.between(*BBOX["lat"])]
    pos_lon, pos_lat = occ.Longitude.to_numpy(), occ.Latitude.to_numpy()
    print(f"[{time.time()-t0:4.0f}s] меток присутствия (США+Канада): {len(occ)}")

    rng = np.random.default_rng(SEED)
    bl = rng.uniform(*BBOX["lon"], N_BG * 4)
    ba = rng.uniform(*BBOX["lat"], N_BG * 4)

    def fmat(lon, lat):
        return np.column_stack([sample_raster(r, lon, lat) for r in rasters])

    Xpos, Xbg = fmat(pos_lon, pos_lat), fmat(bl, ba)
    ok = np.isfinite(Xbg).all(1)
    Xbg, bl, ba = Xbg[ok][:N_BG], bl[ok][:N_BG], ba[ok][:N_BG]
    okp = np.isfinite(Xpos).all(1)
    Xpos, pos_lon, pos_lat = Xpos[okp], pos_lon[okp], pos_lat[okp]
    print(f"[{time.time()-t0:4.0f}s] валидных presence={len(Xpos)}, background={len(Xbg)}")

    X = np.vstack([Xpos, Xbg])
    y = np.r_[np.ones(len(Xpos)), np.zeros(len(Xbg))].astype(int)
    lon = np.r_[pos_lon, bl]; lat = np.r_[pos_lat, ba]
    groups = np.floor(lon / 2).astype(int) * 1000 + np.floor(lat / 2).astype(int)

    def criterial(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        sign = np.sign([np.corrcoef(Ztr[:, j], ytr)[0, 1] for j in range(Xtr.shape[1])])
        return (Zte * sign).mean(1)

    models = {
        "Критериальный (линейн. индекс)": "crit",
        "Логистическая (линейная)": Pipeline([("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Random Forest (нелин.)": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=8,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "Gradient Boosting (нелин.)": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED),
    }
    seeds = (1, 7, 13, 21, 42)
    res = {m: {"auc": [], "lift": []} for m in models}   # по (сид,фолд) — идентичные разбиения для всех моделей
    for sd in seeds:
        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=sd)
        for tr, te in sgkf.split(X, y, groups):
            for name, mdl in models.items():
                if mdl == "crit":
                    s = criterial(X[tr], y[tr], X[te])
                else:
                    mdl.fit(X[tr], y[tr]); s = mdl.predict_proba(X[te])[:, 1]
                res[name]["auc"].append(roc_auc_score(y[te], s))
                thr = np.quantile(s[y[te] == 0], 1 - TOP_AREA)
                res[name]["lift"].append(float((s[y[te] == 1] >= thr).mean()) / TOP_AREA)

    n_pairs = len(seeds) * N_SPLITS
    print(f"\n[{time.time()-t0:4.0f}s] === США+Канада, {len(Xpos)} меток, повторная пространственная CV ({n_pairs} фолдов) ===")
    print(f"{'модель':33s} {'ROC-AUC':>14s} {'lift@10%':>14s}")
    for name in models:
        a, l = np.array(res[name]["auc"]), np.array(res[name]["lift"])
        print(f"{name:33s} {a.mean():.3f}±{a.std():.3f}   {l.mean():5.2f}±{l.std():.2f}")
    crit = np.array(res["Критериальный (линейн. индекс)"]["auc"])
    print("\nПарные сравнения ΔAUC (модель − критериальный), bootstrap 95% ДИ:")
    for name in ["Логистическая (линейная)", "Random Forest (нелин.)", "Gradient Boosting (нелин.)"]:
        d = np.array(res[name]["auc"]) - crit
        boot = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(N_BOOT)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        sig = "ЗНАЧИМ" if (lo > 0 or hi < 0) else "—"
        print(f"  {name:30s} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")
    print(f"[{time.time()-t0:4.0f}s] готово.")


if __name__ == "__main__":
    main()
