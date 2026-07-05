"""Под золото-уран: прогноз УРАНА на США (Запад) с КОММОДИТИ-ПРАВИЛЬНЫМ признаком —
радиометрией NURE (eU/K/eTh) + глобальная геофизика. Критериальный vs ML с
абляцией: геофизика / +радиометрия / только радиометрия. Метки — MRDS (уран, золото).

Запуск: python3 -m experiments.nure_uranium
"""

import glob, time
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.common import sample_global, DS   # глобальная геофизика (mag4km,faa,vgg,geoid,relief)
from cache_paths import NURE, MRDS_CSV

N_BG, SEED = 5000, 42


def load_rad(ch):
    f = glob.glob(str(NURE / f"Predictions_{ch}_WesternUS.shp"))[0]
    g = gpd.read_file(f)
    col = "Y_mean_ppm" if "Y_mean_ppm" in g.columns else "Y_mean"
    xy = np.c_[g["long"].to_numpy(), g["lat"].to_numpy()]
    return cKDTree(xy), g[col].to_numpy().astype(float), xy


def main():
    t0 = time.time()
    print("читаю радиометрию NURE (eU/K/eTh, WesternUS) ...")
    RAD = {ch: load_rad(ch) for ch in ("eU", "K", "eTh")}
    XY = RAD["eU"][2]
    lon_min, lon_max = XY[:, 0].min(), XY[:, 0].max()
    lat_min, lat_max = XY[:, 1].min(), XY[:, 1].max()
    print(f"[{time.time()-t0:3.0f}s] радиометрия: {len(XY)} точек, bbox lon[{lon_min:.1f},{lon_max:.1f}] lat[{lat_min:.1f},{lat_max:.1f}]")

    def samp_rad(ch, lon, lat):
        t, V, _ = RAD[ch]; d, i = t.query(np.c_[lon, lat]); v = V[i].copy(); v[d > 0.05] = np.nan; return v

    mrds = pd.read_csv(str(MRDS_CSV), low_memory=False,
                       usecols=["latitude", "longitude", "commod1", "commod2", "commod3", "country"])
    mrds = mrds[(mrds.country == "United States")].dropna(subset=["latitude", "longitude"])
    mrds = mrds[mrds.longitude.between(lon_min, lon_max) & mrds.latitude.between(lat_min, lat_max)]
    comm = (mrds.commod1.fillna("") + ";" + mrds.commod2.fillna("") + ";" + mrds.commod3.fillna(""))

    def feats(lon, lat):
        rad = np.column_stack([samp_rad(c, lon, lat) for c in ("eU", "K", "eTh")])
        geo = np.column_stack([sample_global(d, lon, lat) for d in DS])
        return rad, geo

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(XY), N_BG * 2, replace=False)
    blon, bla = XY[bg_idx, 0], XY[bg_idx, 1]
    rb, gb = feats(blon, bla)
    okb = np.isfinite(rb).all(1) & np.isfinite(gb).all(1)
    rb, gb, blon, bla = rb[okb][:N_BG], gb[okb][:N_BG], blon[okb][:N_BG], bla[okb][:N_BG]

    for target, key in [("УРАН", "Uranium"), ("ЗОЛОТО", "Gold")]:
        sub = mrds[comm.str.contains(key, case=False, na=False)]
        plon, pla = sub.longitude.to_numpy(), sub.latitude.to_numpy()
        rp, gp = feats(plon, pla)
        okp = np.isfinite(rp).all(1) & np.isfinite(gp).all(1)
        rp, gp, plon, pla = rp[okp], gp[okp], plon[okp], pla[okp]
        if len(rp) < 30:
            print(f"\n{target}: мало точек ({len(rp)}) — пропуск"); continue

        Rad = np.vstack([rp, rb]); Geo = np.vstack([gp, gb])
        y = np.r_[np.ones(len(rp)), np.zeros(len(rb))].astype(int)
        lon = np.r_[plon, blon]; lat = np.r_[pla, bla]
        groups = np.floor(lon).astype(int) * 1000 + np.floor(lat).astype(int)

        def crit(X, ytr, Xte):
            mu, sd = X.mean(0), X.std(0) + 1e-9; Z1, Z2 = (X - mu) / sd, (Xte - mu) / sd
            sg = np.sign([np.corrcoef(Z1[:, j], ytr)[0, 1] for j in range(X.shape[1])]); return (Z2 * sg).mean(1)

        def lift(s, yy, a=0.10):
            return float((s[yy == 1] >= np.quantile(s[yy == 0], 1 - a)).mean()) / a

        sets = {"Критериальный (геофизика)": ("crit", Geo),
                "ML геофизика": ("gb", Geo),
                "ML +радиометрия": ("gb", np.column_stack([Geo, Rad])),
                "ML только радиометрия": ("gb", Rad)}
        print(f"\n=== {target}: {len(rp)} рудопроявлений (США, Запад), пространственная CV ===")
        print(f"{'модель':30s} {'AUC':>14s} {'lift@10%':>10s}")
        for name, (kind, X) in sets.items():
            au, lf = [], []
            for sd in (1, 7, 13):
                for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(X, y, groups):
                    if kind == "crit":
                        s = crit(X[tr], y[tr], X[te])
                    else:
                        m = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=sd).fit(X[tr], y[tr])
                        s = m.predict_proba(X[te])[:, 1]
                    au.append(roc_auc_score(y[te], s)); lf.append(lift(s, y[te]))
            print(f"{name:30s} {np.mean(au):.3f}±{np.std(au):.3f} {np.mean(lf):5.2f}")
    print(f"\n[{time.time()-t0:3.0f}s] готово.")


if __name__ == "__main__":
    main()
