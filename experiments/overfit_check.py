"""Проверка, что боевая модель НЕ запоминает точки (overfitting / leakage).

Два независимых теста на текущих боевых признаках (config.FEATURE_COLS):

  1. In-sample vs held-out gap. In-sample = обучение на ВСЕХ точках и lift на
     них же (модель их видела). Held-out = пространственная блочная CV (точка и
     её соседи-блок целиком вне обучения). Большой разрыв = запоминание; малый =
     честное обобщение.

  2. Permutation-тест (src.validation.permutation_test): метки переставляются на
     случайные ячейки, модель переобучается. Если на «фантомных» метках lift
     остаётся высоким — протокол ловит артефакт; если падает к ~1.0, а на
     реальных точках высокий и p<0.05 — сигнал настоящий, не утечка.

Запуск: python3 -m experiments.overfit_check
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
from src.validation import assign_spatial_blocks, permutation_test
from experiments.pu_seeds import assign_folds

AREAS = (0.05, 0.10, 0.15)


def lift_at(score, pos_idx, area):
    thr = np.quantile(score, 1.0 - area)
    return float((score[pos_idx] >= thr).mean()) / area


def in_sample_lifts(X, pos, rng):
    """Модель видит все точки, меряем lift на них же — верхняя (нечестная) граница."""
    bg = rng.choice(np.setdiff1d(np.arange(len(X)), pos),
                    size=min(config.TRAIN_N_BACKGROUND, len(X) - len(pos)), replace=False)
    m = BackgroundEnsemble(random_state=config.VAL_SEED)
    m.fit(np.vstack([X[pos], X[bg]]), np.r_[np.ones(len(pos)), np.zeros(len(bg))].astype(int))
    score = m.predict_proba(X)[:, 1]
    return {a: lift_at(score, pos, a) for a in AREAS}


def heldout_fold_lift(X, presence, tr_idx, te_pos, rng):
    pos = tr_idx[presence[tr_idx] == 1]
    neg = tr_idx[presence[tr_idx] == 0]
    bg = rng.choice(neg, size=min(config.VAL_N_BACKGROUND, len(neg)), replace=False)
    m = BackgroundEnsemble(random_state=config.VAL_SEED)
    m.fit(np.vstack([X[pos], X[bg]]), np.r_[np.ones(len(pos)), np.zeros(len(bg))].astype(int))
    score = m.predict_proba(X)[:, 1]
    return {a: lift_at(score, te_pos, a) for a in AREAS}


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, _, _ = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid, _ = mark_presence(grid, points)

    presence = grid["presence"].to_numpy().astype(int)
    X = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    pos = np.where(presence == 1)[0]
    n_pos = int(presence.sum())
    print(f"Точек: {n_pos} | ячеек: {len(grid)} | признаков: {len(config.FEATURE_COLS)}\n")

    # --- Тест 1: in-sample vs held-out ---
    print("=" * 60)
    print("ТЕСТ 1. In-sample (видит точки) vs held-out (блочная CV)")
    print("=" * 60)
    in_s = in_sample_lifts(X, pos, np.random.default_rng(config.VAL_SEED))

    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    held = {a: [] for a in AREAS}
    for seed in config.VAL_SEEDS:
        folds = assign_folds(blocks, config.VAL_N_SPLITS, seed)
        for f in range(config.VAL_N_SPLITS):
            te = folds == f
            te_pos = np.where(te & (presence == 1))[0]
            if te_pos.size == 0:
                continue
            tr_idx = np.where(~te)[0]
            L = heldout_fold_lift(X, presence, tr_idx, te_pos, np.random.default_rng(seed * 1000 + f))
            for a in AREAS:
                held[a].append(L[a])

    print(f"\n{'top-N%':>8}{'in-sample':>12}{'held-out':>11}{'разрыв':>10}")
    print("-" * 41)
    for a in AREAS:
        h = float(np.mean(held[a]))
        gap = in_s[a] - h
        print(f"{int(a*100):>7}%{in_s[a]:>11.2f}x{h:>10.2f}x{gap:>9.2f}x")
    print("\nМалый разрыв = модель обобщает, а не запоминает координаты.")
    print("Запоминание дало бы in-sample 8-15x при held-out ~1x.")

    # --- Тест 2: permutation ---
    print("\n" + "=" * 60)
    print(f"ТЕСТ 2. Permutation-тест (n_perm={config.PERM_N}, top-{int(config.PERM_AREA*100)}%)")
    print("=" * 60)
    positive_cells = grid.loc[presence == 1, "cell_id"].tolist()
    res = permutation_test(grid, positive_cells, area=config.PERM_AREA, n_perm=config.PERM_N)
    print(f"\n  наблюдаемый lift (реальные точки): {res['observed']:.2f}x")
    print(f"  нулевой lift (перемешанные метки): среднее {res['null_mean']:.2f}x, q95 {res['null_q95']:.2f}x")
    print(f"  p-value (доля нулевых >= observed): {res['p_value']:.3f}")
    verdict = "СИГНАЛ ЗНАЧИМ (не утечка/запоминание)" if res["p_value"] < 0.05 else "НЕ значим — насторожиться"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
