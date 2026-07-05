"""Площадные (ореольные) метки vs baseline на 6 сидах — устойчивость (held-out, парно).

Проверяет одно-сидовый намёк из pu_halo.py (хало резко лучше в top-5%, хуже
широко). Протокол как в pu_seeds.py: seeded раскидка блоков по фолдам, baseline и
halo на ОДНОМ разбиении -> корректные пары; bootstrap-95% ДИ на разницу lift.

Хало = положителен ореол радиусом HALO_RADIUS_M вокруг точки, ТОЛЬКО в train-
блоках (без утечки в test). Оценка — попадание ИСХОДНЫХ точек (presence==1) из
test-блоков в top-N%. Считаем на top-5% (заявленная сила хало) и top-10%.

Запуск: python3 -m experiments.halo_seeds
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
from experiments.pu_halo import build_halo
from experiments.pu_seeds import assign_folds

AREAS = (0.05, 0.10)
N_BOOT = 5000


def _fit(X, pos_idx, neg_idx, rng):
    """Ансамбль RF+GB на (pos_idx=1, выборка neg_idx=0)."""
    n_bg = min(config.VAL_N_BACKGROUND, len(neg_idx))
    bg = rng.choice(neg_idx, size=n_bg, replace=False)
    model = BackgroundEnsemble(random_state=config.VAL_SEED)
    model.fit(np.vstack([X[pos_idx], X[bg]]),
              np.r_[np.ones(len(pos_idx)), np.zeros(len(bg))].astype(int))
    return model


def fold_lifts(X, pos_idx, neg_idx, te_pos, rng) -> dict:
    """Lift попадания te_pos в top-AREA% прогноза модели — по всем AREAS."""
    model = _fit(X, pos_idx, neg_idx, rng)
    full = model.predict_proba(X)[:, 1]
    out = {}
    for a in AREAS:
        thr = np.quantile(full, 1.0 - a)
        out[a] = float((full[te_pos] >= thr).mean()) / a
    return out


def boot_ci(diff: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(N_BOOT)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, _, _ = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid, _ = mark_presence(grid, points)

    X = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    presence = grid["presence"].to_numpy().astype(int)
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    halo = build_halo(grid)

    print(f"Точек: {int(presence.sum())} | ячеек: {len(grid)} | сиды: {config.VAL_SEEDS}")
    print(f"хало-радиус {1000} м · пороги {[f'top-{int(a*100)}%' for a in AREAS]}\n")
    print(f"{'сид':>5}{'base@5':>10}{'halo@5':>10}{'Δ@5':>9}{'base@10':>11}{'halo@10':>10}{'Δ@10':>9}")
    print("-" * 64)

    pairs = {a: {"b": [], "h": []} for a in AREAS}
    for seed in config.VAL_SEEDS:
        folds = assign_folds(blocks, config.VAL_N_SPLITS, seed)
        sm = {a: {"b": [], "h": []} for a in AREAS}
        for f in range(config.VAL_N_SPLITS):
            te = folds == f
            te_pos = np.where(te & (presence == 1))[0]
            if te_pos.size == 0:
                continue
            tr_idx = np.where(~te)[0]
            seed_pos = tr_idx[presence[tr_idx] == 1]
            pos_h = np.intersect1d(halo(seed_pos), tr_idx)
            base_seed = seed * 1000 + f
            lb = fold_lifts(X, seed_pos, tr_idx[~np.isin(tr_idx, seed_pos)], te_pos,
                            np.random.default_rng(base_seed))
            lh = fold_lifts(X, pos_h, tr_idx[~np.isin(tr_idx, pos_h)], te_pos,
                            np.random.default_rng(base_seed))
            for a in AREAS:
                sm[a]["b"].append(lb[a]); sm[a]["h"].append(lh[a])
                pairs[a]["b"].append(lb[a]); pairs[a]["h"].append(lh[a])
        b5, h5 = np.mean(sm[0.05]["b"]), np.mean(sm[0.05]["h"])
        b10, h10 = np.mean(sm[0.10]["b"]), np.mean(sm[0.10]["h"])
        print(f"{seed:>5}{b5:>9.2f}x{h5:>9.2f}x{h5-b5:>+8.2f}x{b10:>10.2f}x{h10:>9.2f}x{h10-b10:>+8.2f}x")

    print("-" * 64)
    print(f"\nПар (сид×фолд): {len(pairs[0.05]['b'])}")
    for a in AREAS:
        b = np.array(pairs[a]["b"]); h = np.array(pairs[a]["h"]); d = h - b
        ci = boot_ci(d, config.VAL_SEED)
        sig = "ДА" if (ci[0] > 0 or ci[1] < 0) else "НЕТ"
        print(f"top-{int(a*100)}%: baseline {b.mean():.2f}x | halo {h.mean():.2f}x | "
              f"Δ {d.mean():+.2f}x (95% ДИ [{ci[0]:+.2f}, {ci[1]:+.2f}]) | "
              f"halo>base {(d>0).mean()*100:.0f}% | значимо: {sig}")


if __name__ == "__main__":
    main()
