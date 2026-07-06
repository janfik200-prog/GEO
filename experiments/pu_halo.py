"""PU-learning и площадные (ореольные) метки — честная held-out проверка.

Постановка задачи и данные НЕ меняются: те же признаки, та же территория, те же
точки. Меняем только (1) как трактуем фон и (2) форму положительных меток —
и честно мерим, даёт ли это прирост held-out lift.

Протокол оценки (одинаковый для всех режимов):
  * GroupKFold по пространственным блокам VAL_BLOCK_SIZE (нет утечки соседства);
  * метрика — попадание ИСХОДНЫХ точек-рудопроявлений (presence==1) в top-N%
    площади прогноза модели, не видевшей блок точки;
  * балл ячейки = её перцентиль в прогнозе фолд-модели по всей сетке.

Режимы:
  baseline   — presence-background (текущая боевая схема): фон = случайные ячейки.
  +PU        — two-step PU (spy): надёжные негативы вместо случайного фона.
  +halo      — площадные метки: положителен ореол радиусом HALO_RADIUS_M вокруг
               точки (только train-блоки, без утечки в test).
  +PU+halo   — оба рычага вместе.

Запуск: python3 -m experiments.pu_halo
"""

import sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.model_selection import GroupKFold

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features
from src.model import BackgroundEnsemble, mark_presence
from src.validation import assign_spatial_blocks

HALO_RADIUS_M = 1000      # радиус ореола вокруг точки (2 ячейки по 500 м)
RELIABLE_FRAC = 0.60      # доля нижних по баллу ячеек -> надёжные негативы (PU)
AREAS = (0.05, 0.10, 0.15, 0.20)


def build_halo(grid):
    """Вернуть функцию halo(idx_array) -> индексы ячеек в радиусе HALO_RADIUS_M."""
    row = grid["row"].to_numpy()
    col = grid["col"].to_numpy()
    rc_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(row, col))}
    rad = int(round(HALO_RADIUS_M / config.CELL_SIZE))
    offsets = [(dr, dc) for dr in range(-rad, rad + 1) for dc in range(-rad, rad + 1)
               if dr * dr + dc * dc <= rad * rad]

    def halo(idx_array):
        out = set()
        for i in idx_array:
            r, c = int(row[i]), int(col[i])
            for dr, dc in offsets:
                j = rc_to_idx.get((r + dr, c + dc))
                if j is not None:
                    out.add(j)
        return np.fromiter(out, dtype=int)

    return halo


def _fit(X, pos_idx, neg_idx, rng, reliable=False):
    """Обучить boевой ансамбль на (pos_idx как 1, выборка neg_idx как 0).

    При reliable=True — two-step PU: первый ансамбль ранжирует фон, нижние
    RELIABLE_FRAC берутся надёжными негативами, на них переобучаемся.
    """
    n_bg = min(config.VAL_N_BACKGROUND, len(neg_idx))
    bg = rng.choice(neg_idx, size=n_bg, replace=False)

    if reliable:
        m1 = BackgroundEnsemble(random_state=config.VAL_SEED)
        m1.fit(np.vstack([X[pos_idx], X[bg]]),
               np.r_[np.ones(len(pos_idx)), np.zeros(len(bg))].astype(int))
        s = m1.predict_proba(X[neg_idx])[:, 1]
        rel = neg_idx[s <= np.quantile(s, RELIABLE_FRAC)]   # уверенно непохожие на руду
        n_bg = min(config.VAL_N_BACKGROUND, len(rel))
        bg = rng.choice(rel, size=n_bg, replace=False)

    model = BackgroundEnsemble(random_state=config.VAL_SEED)
    model.fit(np.vstack([X[pos_idx], X[bg]]),
              np.r_[np.ones(len(pos_idx)), np.zeros(len(bg))].astype(int))
    return model


def run_mode(grid, blocks, halo, use_pu: bool, use_halo: bool) -> np.ndarray:
    """OOF-перцентили по всей сетке для заданного режима (PU / halo / оба)."""
    X = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    presence = grid["presence"].to_numpy().astype(int)
    n = len(grid)
    oof = np.full(n, np.nan)
    rng = np.random.default_rng(config.VAL_SEED)

    for tr_idx, te_idx in GroupKFold(n_splits=config.VAL_N_SPLITS).split(X, groups=blocks):
        seed_pos = tr_idx[presence[tr_idx] == 1]            # точки только из train-блоков
        if use_halo:
            pos = np.intersect1d(halo(seed_pos), tr_idx)    # ореол ∩ train (нет утечки в test)
        else:
            pos = seed_pos
        pos_mask = np.isin(tr_idx, pos)
        neg_idx = tr_idx[~pos_mask]

        model = _fit(X, pos, neg_idx, rng, reliable=use_pu)
        full = model.predict_proba(X)[:, 1]
        pct = full.argsort().argsort() / (n - 1)
        oof[te_idx] = pct[te_idx]
    return oof


def lifts(grid, oof) -> dict[float, tuple[float, float]]:
    """Для каждой площади: (доля точек в top-N%, lift)."""
    presence = grid["presence"].to_numpy().astype(int)
    pos_oof = oof[presence == 1]
    pos_oof = pos_oof[np.isfinite(pos_oof)]
    out = {}
    for a in AREAS:
        thr = float(np.nanquantile(oof, 1.0 - a))
        cov = float((pos_oof >= thr).mean())
        out[a] = (cov, cov / a)
    return out


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, _, _ = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid, positive_cells = mark_presence(grid, points)
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    halo = build_halo(grid)

    n_pos = int(grid["presence"].sum())
    print(f"Точек: {n_pos} | ячеек: {len(grid)} | хало-радиус {HALO_RADIUS_M} м | "
          f"надёжных негативов: нижние {int(RELIABLE_FRAC*100)}%\n")

    modes = [
        ("baseline (presence-bg)", False, False),
        ("+PU (надёжные негативы)", True, False),
        ("+halo (площадные метки)", False, True),
        ("+PU + halo",              True,  True),
    ]
    header = f"{'режим':<26}" + "".join(f"top-{int(a*100)}%".rjust(12) for a in AREAS)
    print(header)
    print("-" * len(header))
    for name, use_pu, use_halo in modes:
        oof = run_mode(grid, blocks, halo, use_pu, use_halo)
        L = lifts(grid, oof)
        row = f"{name:<26}" + "".join(f"{L[a][1]:.2f}x".rjust(12) for a in AREAS)
        print(row)
    print("\n(числа — lift held-out: во сколько раз чаще случайного точка попадает в top-N%)")


if __name__ == "__main__":
    main()
