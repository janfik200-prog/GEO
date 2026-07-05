"""P1: проверка УСТОЙЧИВОСТИ вывода ML>критериальный при обогащённых признаках.

К базовым геофизическим гридам (гравика, её HGM, глубина LAB, Moho) добавляются
инженерные признаки — пространственные градиенты (края структур) и сглаженная
крупномасштабная компонента. Без новых закачек. Если даже на богатом наборе
нелинейный ML значимо обходит линейное критериальное комбинирование — вывод
устойчив, а не артефакт бедного набора.

Запуск: python3 -m experiments.cmmi_enriched
"""

import glob, time
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.common import _read_geotiff
from cache_paths import CMMI, CMMI_OCC

DATA = str(CMMI)
OCC_CSV = str(CMMI_OCC)
BBOX = dict(lon=(-170.0, -52.0), lat=(25.0, 75.0))
N_BG, SEED, N_SPLITS, TOP, N_BOOT = 6000, 42, 5, 0.10, 2000


def build_layers():
    """Базовые + инженерные слои: (имя, массив, (x0,y0,sx,sy), epsg)."""
    layers = []
    for path in sorted(glob.glob(f"{DATA}/*.tif")):
        nm = path.split("/")[-1].replace(".tif", "").replace("Geophysics", "").replace("_USCanada", "")
        arr, tr, nod, epsg = _read_geotiff(path)
        a = arr.copy()
        if nod is not None:
            a[a == nod] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        layers.append((nm, a, tr, epsg))
        # градиент (заполняем дыры средним, считаем |grad|, маскируем исходные дыры)
        filled = np.where(np.isfinite(a), a, np.nanmean(a))
        gy, gx = np.gradient(filled)
        grad = np.hypot(gx, gy)
        grad[~np.isfinite(a)] = np.nan
        layers.append((nm + "_grad", grad, tr, epsg))
    return layers


def sample(layer, lon, lat):
    _nm, arr, (x0, y0, sx, sy), epsg = layer
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
    return out


def main():
    t0 = time.time()
    layers = build_layers()
    names = [l[0] for l in layers]
    print(f"признаков: {len(names)} -> {names}")

    occ = pd.read_csv(OCC_CSV, encoding="latin1", low_memory=False)
    occ = occ[occ.Admin != "Australia"].dropna(subset=["Longitude", "Latitude"])
    occ = occ[occ.Longitude.between(*BBOX["lon"]) & occ.Latitude.between(*BBOX["lat"])]
    pl, pa = occ.Longitude.to_numpy(), occ.Latitude.to_numpy()
    rng = np.random.default_rng(SEED)
    bl = rng.uniform(*BBOX["lon"], N_BG * 4); ba = rng.uniform(*BBOX["lat"], N_BG * 4)

    fmat = lambda lon, lat: np.column_stack([sample(L, lon, lat) for L in layers])
    Xpos, Xbg = fmat(pl, pa), fmat(bl, ba)
    okb = np.isfinite(Xbg).all(1); Xbg, bl, ba = Xbg[okb][:N_BG], bl[okb][:N_BG], ba[okb][:N_BG]
    okp = np.isfinite(Xpos).all(1); Xpos, pl, pa = Xpos[okp], pl[okp], pa[okp]
    print(f"[{time.time()-t0:3.0f}s] presence={len(Xpos)}, background={len(Xbg)}")

    X = np.vstack([Xpos, Xbg]); y = np.r_[np.ones(len(Xpos)), np.zeros(len(Xbg))].astype(int)
    lon = np.r_[pl, bl]; lat = np.r_[pa, ba]
    groups = np.floor(lon / 2).astype(int) * 1000 + np.floor(lat / 2).astype(int)

    def crit(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z1, Z2 = (Xtr - mu) / sd, (Xte - mu) / sd
        sg = np.sign([np.corrcoef(Z1[:, j], ytr)[0, 1] for j in range(Xtr.shape[1])])
        return (Z2 * sg).mean(1)

    models = {
        "Критериальный (линейн. индекс)": "crit",
        "Логистическая (линейная)": Pipeline([("sc", StandardScaler()),
            ("c", LogisticRegression(max_iter=4000, class_weight="balanced"))]),
        "Random Forest (нелин.)": RandomForestClassifier(n_estimators=500, max_depth=12,
            min_samples_leaf=6, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "Gradient Boosting (нелин.)": GradientBoostingClassifier(n_estimators=300, max_depth=3,
            learning_rate=0.05, subsample=0.8, random_state=SEED),
    }
    res = {m: [] for m in models}
    for sd in (1, 7, 13, 21, 42):
        for tr, te in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=sd).split(X, y, groups):
            for nm, md in models.items():
                s = crit(X[tr], y[tr], X[te]) if md == "crit" else md.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                res[nm].append(roc_auc_score(y[te], s))

    print(f"\n[{time.time()-t0:3.0f}s] === США+Канада, {len(Xpos)} меток, {len(names)} признаков, повторная CV ===")
    for nm in models:
        a = np.array(res[nm]); print(f"  {nm:32s} AUC {a.mean():.3f}±{a.std():.3f}")
    cb = np.array(res["Критериальный (линейн. индекс)"])
    print("\nΔAUC (модель − критериальный), 95% ДИ:")
    for nm in ["Логистическая (линейная)", "Random Forest (нелин.)", "Gradient Boosting (нелин.)"]:
        d = np.array(res[nm]) - cb
        bt = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(N_BOOT)])
        lo, hi = np.quantile(bt, [0.025, 0.975])
        print(f"  {nm:30s} {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {'ЗНАЧИМ' if (lo>0 or hi<0) else '—'}")
    print(f"[{time.time()-t0:3.0f}s] готово.")


if __name__ == "__main__":
    main()
