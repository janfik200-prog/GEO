"""Прунинг обогащённых признаков + честная перепроверка (held-out, 6 сидов).

Из 9 новых признаков (feat_enrich) оставляем только полезные. Отбор — по
ВАЖНОСТИ на обучении (не по held-out lift, иначе отбор подгонит итоговую
метрику под тест). Правило: новый признак остаётся, если его важность выше
средней по всем признакам. Затем парное 6-сидовое сравнение:
  base(13)  vs  full-enriched(22)  vs  pruned(13 + отобранные).

Bootstrap-95% ДИ на (pruned − base) и (pruned − full): подтвердить, что прунинг
не хуже полного набора и лучше базового.

Запуск: python3 -m experiments.feat_prune
"""

import sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features
from src.model import BackgroundEnsemble, mark_presence
from src.validation import assign_spatial_blocks
from experiments.pu_seeds import assign_folds
from experiments.feat_enrich import enrich, boot_ci, NEW_COLS

AREAS = (0.05, 0.10)


def fit_full(X, presence, idx_pos, idx_neg, rng):
    bg = rng.choice(idx_neg, size=min(config.TRAIN_N_BACKGROUND, len(idx_neg)), replace=False)
    m = BackgroundEnsemble(random_state=config.VAL_SEED)
    m.fit(np.vstack([X[idx_pos], X[bg]]),
          np.r_[np.ones(len(idx_pos)), np.zeros(len(bg))].astype(int))
    return m


def fold_lifts(X, presence, tr_idx, te_pos, rng) -> dict:
    """Один fit -> lift на всех AREAS (без лишних переобучений)."""
    pos = tr_idx[presence[tr_idx] == 1]
    neg = tr_idx[presence[tr_idx] == 0]
    bg = rng.choice(neg, size=min(config.VAL_N_BACKGROUND, len(neg)), replace=False)
    m = BackgroundEnsemble(random_state=config.VAL_SEED)
    m.fit(np.vstack([X[pos], X[bg]]),
          np.r_[np.ones(len(pos)), np.zeros(len(bg))].astype(int))
    full = m.predict_proba(X)[:, 1]
    return {a: float((full[te_pos] >= np.quantile(full, 1.0 - a)).mean()) / a for a in AREAS}


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, _, _ = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = enrich(grid, layers)
    grid, _ = mark_presence(grid, points)

    base_cols = config.FEATURE_COLS
    enr_cols = base_cols + NEW_COLS
    presence = grid["presence"].to_numpy().astype(int)
    Xe = grid[enr_cols].fillna(0).to_numpy()

    # --- отбор по важности на обучении ---
    rng0 = np.random.default_rng(config.VAL_SEED)
    pos = np.where(presence == 1)[0]; neg = np.where(presence == 0)[0]
    fm = fit_full(Xe, presence, pos, neg, rng0)
    imp = fm.feature_importances_
    thr = float(imp.mean())
    kept_new = [c for c in NEW_COLS if imp[enr_cols.index(c)] > thr]
    dropped = [c for c in NEW_COLS if c not in kept_new]
    pruned_cols = base_cols + kept_new

    print(f"Точек: {int(presence.sum())} | порог важности (среднее) = {thr:.4f}")
    print("Новые признаки по важности:")
    for c in sorted(NEW_COLS, key=lambda c: imp[enr_cols.index(c)], reverse=True):
        keep = "оставить" if c in kept_new else "ОТБРОСИТЬ"
        print(f"   {imp[enr_cols.index(c)]:.4f}  {c:<20} -> {keep}")
    print(f"\nОставлено новых: {kept_new}")
    print(f"Отброшено:       {dropped}")
    print(f"Наборы: base={len(base_cols)} | pruned={len(pruned_cols)} | full={len(enr_cols)}\n")

    Xb = grid[base_cols].fillna(0).to_numpy()
    Xp = grid[pruned_cols].fillna(0).to_numpy()
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)

    sets = {"base": Xb, "pruned": Xp, "full": Xe}
    pairs = {a: {k: [] for k in sets} for a in AREAS}
    print(f"{'сид':>5}{'base@10':>10}{'pruned@10':>12}{'full@10':>10}")
    print("-" * 37)
    for seed in config.VAL_SEEDS:
        folds = assign_folds(blocks, config.VAL_N_SPLITS, seed)
        sm = {a: {k: [] for k in sets} for a in AREAS}
        for f in range(config.VAL_N_SPLITS):
            te = folds == f
            te_pos = np.where(te & (presence == 1))[0]
            if te_pos.size == 0:
                continue
            tr_idx = np.where(~te)[0]
            base_seed = seed * 1000 + f
            for k, X in sets.items():
                L = fold_lifts(X, presence, tr_idx, te_pos, np.random.default_rng(base_seed))
                for a in AREAS:
                    sm[a][k].append(L[a]); pairs[a][k].append(L[a])
        print(f"{seed:>5}{np.mean(sm[0.10]['base']):>9.2f}x"
              f"{np.mean(sm[0.10]['pruned']):>11.2f}x{np.mean(sm[0.10]['full']):>9.2f}x")

    print("-" * 37)
    print(f"\nПар (сид×фолд): {len(pairs[0.10]['base'])}")
    for a in AREAS:
        b = np.array(pairs[a]["base"]); p = np.array(pairs[a]["pruned"]); fu = np.array(pairs[a]["full"])
        print(f"\ntop-{int(a*100)}%:  base {b.mean():.2f}x | pruned {p.mean():.2f}x | full {fu.mean():.2f}x")
        for name, ref in (("pruned − base", b), ("pruned − full", fu)):
            d = p - ref
            ci = boot_ci(d, config.VAL_SEED)
            sig = "ДА" if (ci[0] > 0 or ci[1] < 0) else "НЕТ"
            print(f"    {name}: {d.mean():+.2f}x (95% ДИ [{ci[0]:+.2f}, {ci[1]:+.2f}]) | "
                  f"pruned>{'base' if 'base' in name else 'full'} {(d>0).mean()*100:.0f}% | значимо: {sig}")


if __name__ == "__main__":
    main()
