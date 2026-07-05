"""Анабар при малом N: связка A+C+E + вариант F.

A — расширение позитивов: объединяем прямые индикаторы (опробование Au-U +
    привнос урана) как presence; геохимические ОРЕОЛЫ держим как НЕЗАВИСИМЫЙ тест.
C — знание: критериальный geo_score как признак-приор + монотонные ограничения
    (prox_* и geo_score монотонно ↑, tect_only_penalty ↓) у HistGB; бэггинг по фону.
E — честная оценка: пространственная блочная CV, lift@10% + bootstrap-ДИ,
    плюс независимый тест на ореолах.
F — смена опоры: повтор сравнения на огрублённой сетке (блоки ~5 км).

Запуск: python3 -m experiments.anabar_pu
"""

import tempfile, warnings
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data_loader import find_base_dir, load_all_layers, load_layer, to_crs_safe
from src.features import build_grid, build_features, compute_geo_score
from src.model import mark_presence, sample_presence_background
from src.validation import assign_spatial_blocks, _coverage

SHP = Path("data/Gis-integro/shp_dbf")
SEEDS = (1, 7, 13, 21, 42)
A = 0.10
FEATS = config.FEATURE_COLS + ["geo_score"]
MONO = [(-1 if f == "tect_only_penalty" else (1 if (f.startswith("prox_") or f == "geo_score") else 0))
        for f in FEATS]


def cells_of(grid, gdf):
    j = gpd.sjoin(grid[["cell_id", "geometry"]], gdf[["geometry"]], predicate="intersects", how="inner")
    return set(j["cell_id"].astype(int).unique())


def hgb_bagged(Xtr, ytr, Xq, seed, rounds=6, bg_frac=0.7):
    """PU-бэггинг: усредняем HistGB по подвыборкам фона; монотонные ограничения; баланс весами."""
    rng = np.random.default_rng(seed)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    preds = []
    for r in range(rounds):
        dr = rng.choice(neg, int(bg_frac * len(neg)), replace=False)
        idx = np.concatenate([pos, dr])
        yw = ytr[idx]
        w = np.where(yw == 1, (yw == 0).sum() / max((yw == 1).sum(), 1), 1.0)
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, monotonic_cst=MONO, random_state=seed + r)
        m.fit(Xtr[idx], yw, sample_weight=w)
        preds.append(m.predict_proba(Xq)[:, 1])
    return np.mean(preds, axis=0)


def spatial_cv(grid, pos_cells, feats_rf, block, oreol_pos):
    """Пространственная CV: lift@10% по held-out тренировочным позитивам + независимый тест на ореолах."""
    blocks = assign_spatial_blocks(grid, block)
    crit = grid["geo_score"].to_numpy()
    Xrf_all = grid[feats_rf].fillna(0).to_numpy()
    Xhg_all = grid[FEATS].fillna(0).to_numpy()
    out = {"Критериальный (ГИС Интегро)": [], "RF (текущие)": [], "HistGB монотон.+приор+PU": []}
    ind = {k: [] for k in out}
    for sd in SEEDS:
        sp, y = sample_presence_background(grid, sorted(pos_cells), config.VAL_N_BACKGROUND, sd)
        g = blocks[sp]
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(np.zeros(len(sp)), y, g):
            tp = sp[te][y[te] == 1]
            if tp.size == 0:
                continue
            out["Критериальный (ГИС Интегро)"].append(_coverage(crit, tp, A) / A)
            rf = RandomForestClassifier(n_estimators=400, max_depth=7, min_samples_leaf=10,
                class_weight="balanced_subsample", random_state=sd, n_jobs=-1)
            rf.fit(Xrf_all[sp][tr], y[tr]); s_rf = rf.predict_proba(Xrf_all)[:, 1]
            out["RF (текущие)"].append(_coverage(s_rf, tp, A) / A)
            s_hg = hgb_bagged(Xhg_all[sp][tr], y[tr], Xhg_all, sd)
            out["HistGB монотон.+приор+PU"].append(_coverage(s_hg, tp, A) / A)
            if oreol_pos.size:
                ind["Критериальный (ГИС Интегро)"].append(_coverage(crit, oreol_pos, A) / A)
                ind["RF (текущие)"].append(_coverage(s_rf, oreol_pos, A) / A)
                ind["HistGB монотон.+приор+PU"].append(_coverage(s_hg, oreol_pos, A) / A)
    return out, ind


def main():
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers); grid = compute_geo_score(grid, shape)
    grid, opro = mark_presence(grid, points)

    crs = layers["mask"].crs
    uran = cells_of(grid, to_crs_safe(load_layer(SHP / "привнос урана.shp"), crs))
    oreol = cells_of(grid, to_crs_safe(load_layer(SHP / "геохимические ореолы.shp"), crs))
    train_pos = set(opro)                              # чистая цель: только Au-U опробование
    oreol_test = oreol - train_pos                     # независимый тест (геохим. ореолы)
    cid2pos = {c: i for i, c in enumerate(grid["cell_id"].to_numpy())}
    oreol_pos = np.array([cid2pos[c] for c in oreol_test if c in cid2pos])
    print(f"позитивы: опробование {len(opro)} + привнос урана {len(uran)} = {len(train_pos)} (было {len(opro)})")
    print(f"независимый тест (ореолы, не в train): {len(oreol_pos)} ячеек\n")

    print("=== A+C+E: 500 м, пространственная CV, lift@10% ===")
    out, ind = spatial_cv(grid, train_pos, config.FEATURE_COLS, config.VAL_BLOCK_SIZE, oreol_pos)
    crit = np.array(out["Критериальный (ГИС Интегро)"])
    for k in out:
        v = np.array(out[k]); iv = np.array(ind[k])
        line = f"  {k:28s} CV {v.mean():.2f}±{v.std():.2f}   ореолы {iv.mean():.2f}±{iv.std():.2f}"
        if k != "Критериальный (ГИС Интегро)":
            d = np.array(out[k]) - crit
            bt = np.array([np.random.default_rng(s).choice(d, len(d)).mean() for s in range(2000)])
            lo, hi = np.quantile(bt, [.025, .975])
            line += f"   ΔCV {d.mean():+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if (lo>0 or hi<0) else ''}"
        print(line)

    # F: огрубление опоры до ~5 км
    print("\n=== F: огрублённая опора (блоки ~5 км) ===")
    cb = assign_spatial_blocks(grid, 5000)
    df = grid[config.FEATURE_COLS + ["geo_score"]].copy(); df["b"] = cb
    agg = df.groupby("b").mean()
    pos_b = set(np.unique(cb[grid["cell_id"].isin(train_pos).to_numpy()]))
    y = np.array([1 if b in pos_b else 0 for b in agg.index])
    # координаты блоков для пространственной CV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx = grid.geometry.centroid.x.to_numpy(); cy = grid.geometry.centroid.y.to_numpy()
    bx = pd.Series(cx).groupby(cb).mean(); by = pd.Series(cy).groupby(cb).mean()
    superblk = (np.floor(bx.loc[agg.index] / 20000).astype(int) * 1000
                + np.floor(by.loc[agg.index] / 20000).astype(int)).to_numpy()
    Xc = agg[config.FEATURE_COLS].to_numpy(); critc = agg["geo_score"].to_numpy()
    Xhc = agg[config.FEATURE_COLS + ["geo_score"]].to_numpy()
    resc = {"Критериальный": [], "HistGB монотон.+приор+PU": []}
    for sd in SEEDS:
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=sd).split(Xc, y, superblk):
            tp = np.where(y)[0]; tp = tp[np.isin(tp, te)]
            if tp.size == 0:
                continue
            resc["Критериальный"].append(_coverage(critc, tp, A) / A)
            s = hgb_bagged(Xhc[tr], y[tr], Xhc, sd)
            resc["HistGB монотон.+приор+PU"].append(_coverage(s, tp, A) / A)
    print(f"  блоков {len(agg)}, позитивных {int(y.sum())}")
    for k, v in resc.items():
        v = np.array(v); print(f"  {k:28s} lift {v.mean():.2f}±{v.std():.2f}")


if __name__ == "__main__":
    main()
