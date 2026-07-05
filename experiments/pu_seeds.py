"""PU vs baseline на нескольких сидах — устойчивость прироста (held-out, парно).

Отвечает на вопрос: прирост PU из pu_halo.py (один сид) — реальный или шум?
Для каждого сида блоки заново раскидываются по фолдам (seeded), baseline и PU
оцениваются на ОДНОМ И ТОМ ЖЕ разбиении и с одинаковым стартовым фоном —
значит пары (lift_baseline, lift_PU) корректны. Bootstrap-95% ДИ на разницу:
если ДИ не включает 0 — прирост значим.

Метрика фолда — lift попадания held-out точек (presence==1 из test-блоков) в
top-AREA% площади прогноза фолд-модели по всей сетке.

Запуск: python3 -m experiments.pu_seeds
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
from experiments.pu_halo import RELIABLE_FRAC

AREA = 0.10        # основной порог для парного сравнения
N_BOOT = 5000


def assign_folds(blocks: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Раскидать пространственные блоки по фолдам со случайной перестановкой (seeded)."""
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {int(b): i % n_splits for i, b in enumerate(uniq)}
    return np.array([fold_of[int(b)] for b in blocks])


def _fit(X, pos_idx, neg_idx, rng, reliable: bool):
    """Ансамбль RF+GB на (pos=1, выборка neg=0); reliable=True -> two-step PU."""
    n_bg = min(config.VAL_N_BACKGROUND, len(neg_idx))
    bg = rng.choice(neg_idx, size=n_bg, replace=False)
    if reliable:
        m1 = BackgroundEnsemble(random_state=config.VAL_SEED)
        m1.fit(np.vstack([X[pos_idx], X[bg]]),
               np.r_[np.ones(len(pos_idx)), np.zeros(len(bg))].astype(int))
        s = m1.predict_proba(X[neg_idx])[:, 1]
        rel = neg_idx[s <= np.quantile(s, RELIABLE_FRAC)]
        bg = rng.choice(rel, size=min(config.VAL_N_BACKGROUND, len(rel)), replace=False)
    model = BackgroundEnsemble(random_state=config.VAL_SEED)
    model.fit(np.vstack([X[pos_idx], X[bg]]),
              np.r_[np.ones(len(pos_idx)), np.zeros(len(bg))].astype(int))
    return model


def fold_lift(X, presence, tr_idx, te_pos, rng, reliable: bool) -> float:
    """Lift held-out точек test-блока в top-AREA% прогноза фолд-модели."""
    pos = tr_idx[presence[tr_idx] == 1]
    neg = tr_idx[presence[tr_idx] == 0]
    model = _fit(X, pos, neg, rng, reliable)
    full = model.predict_proba(X)[:, 1]
    thr = np.quantile(full, 1.0 - AREA)
    return float((full[te_pos] >= thr).mean()) / AREA


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, _, _ = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid, _ = mark_presence(grid, points)

    X = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    presence = grid["presence"].to_numpy().astype(int)
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)

    print(f"Точек: {int(presence.sum())} | ячеек: {len(grid)} | "
          f"сиды: {config.VAL_SEEDS} | порог top-{int(AREA*100)}%\n")
    print(f"{'сид':>5}{'baseline':>12}{'PU':>10}{'разница':>12}")
    print("-" * 39)

    pairs_b, pairs_p = [], []
    for seed in config.VAL_SEEDS:
        folds = assign_folds(blocks, config.VAL_N_SPLITS, seed)
        sb, sp = [], []
        for f in range(config.VAL_N_SPLITS):
            te = folds == f
            te_pos = np.where(te & (presence == 1))[0]
            if te_pos.size == 0:
                continue
            tr_idx = np.where(~te)[0]
            base_seed = seed * 1000 + f
            lb = fold_lift(X, presence, tr_idx, te_pos, np.random.default_rng(base_seed), False)
            lp = fold_lift(X, presence, tr_idx, te_pos, np.random.default_rng(base_seed), True)
            sb.append(lb); sp.append(lp)
            pairs_b.append(lb); pairs_p.append(lp)
        mb, mp = float(np.mean(sb)), float(np.mean(sp))
        print(f"{seed:>5}{mb:>11.2f}x{mp:>9.2f}x{mp-mb:>+11.2f}x")

    b = np.array(pairs_b); p = np.array(pairs_p)
    diff = p - b
    rng = np.random.default_rng(config.VAL_SEED)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(N_BOOT)])
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))

    print("-" * 39)
    print(f"\nПар (сид×фолд): {len(diff)}")
    print(f"Средний lift  baseline: {b.mean():.2f}x | PU: {p.mean():.2f}x")
    print(f"Средняя разница PU−baseline: {diff.mean():+.2f}x  (bootstrap-95% ДИ [{ci[0]:+.2f}, {ci[1]:+.2f}])")
    print(f"Доля фолдов, где PU > baseline: {(diff > 0).mean()*100:.0f}%")
    sig = ci[0] > 0 or ci[1] < 0
    print(f"Значимо (ДИ не включает 0): {'ДА' if sig else 'НЕТ'}")


if __name__ == "__main__":
    main()
