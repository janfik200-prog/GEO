"""Прунинг DEM-признаков + 8 сидов — добор мощности к решению об интеграции.

DEM (Copernicus GLO-30) на 5 сидах дал положительный, но НЕзначимый прирост
(top-5% +0.48x ДИ [-0.12,+1.03]). Здесь, как в feat_prune для геометрии:
  1) отбор DEM-признаков по важности на обучении (> среднего, не по held-out);
  2) 8 сидов вместо 5, боевая модель BackgroundEnsemble;
  3) парный bootstrap (pruned−base), (pruned−full) на top-5% и top-10%.

Решение: интегрируем DEM, только если pruned значимо лучше base.

Запуск: python3 -m experiments.dem_prune
"""

import tempfile, warnings
from pathlib import Path

import numpy as np

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, compute_geo_score
from src.model import BackgroundEnsemble, mark_presence
from src.validation import assign_spatial_blocks, _coverage
from experiments.pu_seeds import assign_folds
from experiments.feat_enrich import boot_ci
from experiments.anabar_dem import build_dem, read_raster, derivatives, sampler

SEEDS = (1, 7, 13, 21, 42, 99, 123, 777)
AREAS = (0.05, 0.10)


def fold_lift(X, presence, tr_idx, te_pos, rng, area):
    pos = tr_idx[presence[tr_idx] == 1]
    neg = tr_idx[presence[tr_idx] == 0]
    bg = rng.choice(neg, size=min(config.VAL_N_BACKGROUND, len(neg)), replace=False)
    m = BackgroundEnsemble(random_state=config.VAL_SEED)
    m.fit(np.vstack([X[pos], X[bg]]), np.r_[np.ones(len(pos)), np.zeros(len(bg))].astype(int))
    sc = m.predict_proba(X)[:, 1]
    thr = np.quantile(sc, 1.0 - area)
    return float((sc[te_pos] >= thr).mean()) / area


def main():
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = compute_geo_score(grid, shape)
    grid, pos = mark_presence(grid, points)

    elev, gt = read_raster(build_dem())
    deriv = derivatives(elev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cents = grid.geometry.centroid.to_crs(4326)
    glon, glat = cents.x.to_numpy(), cents.y.to_numpy()
    dem_cols = list(deriv)
    for k in dem_cols:
        grid[k] = sampler(deriv[k], gt)(glon, glat)
    grid[dem_cols] = grid[dem_cols].fillna(grid[dem_cols].median())

    base_cols = list(config.FEATURE_COLS)
    full_cols = base_cols + dem_cols
    presence = grid["presence"].to_numpy().astype(int)
    Xf = grid[full_cols].fillna(0).to_numpy()

    # --- отбор DEM-признаков по важности на обучении ---
    rng0 = np.random.default_rng(config.VAL_SEED)
    p = np.where(presence == 1)[0]
    neg = np.where(presence == 0)[0]
    bg = rng0.choice(neg, size=min(config.TRAIN_N_BACKGROUND, len(neg)), replace=False)
    fm = BackgroundEnsemble(random_state=config.VAL_SEED)
    fm.fit(np.vstack([Xf[p], Xf[bg]]), np.r_[np.ones(len(p)), np.zeros(len(bg))].astype(int))
    imp = fm.feature_importances_
    thr = float(imp.mean())
    kept = [c for c in dem_cols if imp[full_cols.index(c)] > thr]
    pruned_cols = base_cols + kept

    print(f"Точек: {int(presence.sum())} | порог важности = {thr:.4f}")
    print("DEM-признаки по важности:")
    for c in sorted(dem_cols, key=lambda c: imp[full_cols.index(c)], reverse=True):
        print(f"   {imp[full_cols.index(c)]:.4f}  {c:<12} -> {'оставить' if c in kept else 'ОТБРОСИТЬ'}")
    print(f"\nОставлено: {kept}")
    print(f"Наборы: base={len(base_cols)} | pruned={len(pruned_cols)} | full={len(full_cols)}\n")

    Xb = grid[base_cols].fillna(0).to_numpy()
    Xp = grid[pruned_cols].fillna(0).to_numpy()
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    sets = {"base": Xb, "pruned": Xp, "full": Xf}
    pairs = {a: {k: [] for k in sets} for a in AREAS}
    for seed in SEEDS:
        folds = assign_folds(blocks, config.VAL_N_SPLITS, seed)
        for f in range(config.VAL_N_SPLITS):
            te = folds == f
            te_pos = np.where(te & (presence == 1))[0]
            if te_pos.size == 0:
                continue
            tr_idx = np.where(~te)[0]
            for k, X in sets.items():
                rng = np.random.default_rng(seed * 1000 + f)
                for a in AREAS:
                    # один и тот же фон на (seed,fold,area) для парности
                    rng2 = np.random.default_rng(seed * 1000 + f)
                    pairs[a][k].append(fold_lift(X, presence, tr_idx, te_pos, rng2, a))

    n = len(pairs[0.10]["base"])
    print(f"Пар (сид×фолд): {n}")
    for a in AREAS:
        b = np.array(pairs[a]["base"]); pr = np.array(pairs[a]["pruned"]); fu = np.array(pairs[a]["full"])
        print(f"\ntop-{int(a*100)}%:  base {b.mean():.2f}x | pruned {pr.mean():.2f}x | full {fu.mean():.2f}x")
        for name, ref in (("pruned − base", b), ("pruned − full", fu)):
            d = pr - ref
            ci = boot_ci(d, config.VAL_SEED)
            sig = "ДА" if (ci[0] > 0 or ci[1] < 0) else "НЕТ"
            print(f"    {name}: {d.mean():+.2f}x (95% ДИ [{ci[0]:+.2f}, {ci[1]:+.2f}]) | значимо: {sig}")


if __name__ == "__main__":
    main()
