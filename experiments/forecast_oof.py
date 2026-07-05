"""Карта перспективности в HELD-OUT (out-of-fold) режиме.

В отличие от боевой карты (forecast_map.py), где модель обучена на ВСЕХ точках и
поэтому неизбежно «подсвечивает» собственные обучающие точки (in-sample,
циркулярно), здесь каждая ячейка получает балл от модели, которая НЕ видела её
пространственный блок:

  * территория разбивается на блоки VAL_BLOCK_SIZE (как в честной валидации);
  * GroupKFold по блокам: для каждого фолда модель учится на presence-background
    из ОСТАЛЬНЫХ блоков и предсказывает ячейки тестового блока;
  * балл каждой ячейки = её перцентиль в распределении прогноза фолд-модели по
    всей сетке (сопоставимо между фолдами, согласовано с метрикой top-N%).

Так на карте видно ЧЕСТНОЕ попадание точек, а не in-sample иллюзию. Точки
раскрашены: залитые — попали в top-10% held-out прогноза, пустые — нет.

Запуск: python3 -m experiments.forecast_oof
"""

import sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupKFold

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features
from src.model import BackgroundEnsemble, mark_presence
from src.validation import assign_spatial_blocks

AREA_MAIN = 0.10   # порог top-N% для «попадания» точки


def out_of_fold_scores(grid: gpd.GeoDataFrame, blocks: np.ndarray) -> np.ndarray:
    """Перцентильный балл каждой ячейки от модели, не видевшей её блок.

    GroupKFold по пространственным блокам: на каждом фолде боевая модель
    (BackgroundEnsemble RF+GB) учится на presence-background из train-блоков и
    предсказывает по всей сетке; для ячеек test-блока берётся их перцентиль в
    распределении прогноза этой фолд-модели (сопоставимо между фолдами).
    """
    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    presence = grid["presence"].to_numpy().astype(int)
    n = len(grid)
    oof = np.full(n, np.nan)
    rng = np.random.default_rng(config.VAL_SEED)

    gkf = GroupKFold(n_splits=config.VAL_N_SPLITS)
    for tr_idx, te_idx in gkf.split(X_all, groups=blocks):
        pos_tr = tr_idx[presence[tr_idx] == 1]            # позитивы только из train-блоков
        neg_pool = tr_idx[presence[tr_idx] == 0]
        n_bg = min(config.VAL_N_BACKGROUND, len(neg_pool))
        bg = rng.choice(neg_pool, size=n_bg, replace=False)
        train_idx = np.concatenate([pos_tr, bg])
        y = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(bg))]).astype(int)

        model = BackgroundEnsemble(random_state=config.VAL_SEED)
        model.fit(X_all[train_idx], y)
        full = model.predict_proba(X_all)[:, 1]
        pct = full.argsort().argsort() / (n - 1)          # перцентиль каждой ячейки
        oof[te_idx] = pct[te_idx]
    return oof


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    mask = layers["mask"]

    grid, mask_union, grid_shape = build_grid(mask, config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid, positive_cells = mark_presence(grid, points)

    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    grid["oof"] = out_of_fold_scores(grid, blocks)

    # --- метрика честного попадания точек (held-out) ---
    presence = grid["presence"].to_numpy().astype(int)
    pos_oof = grid["oof"].to_numpy()[presence == 1]
    pos_oof = pos_oof[np.isfinite(pos_oof)]
    n_pos = len(pos_oof)
    print(f"Реальных ячеек-точек: {int(presence.sum())} | ячеек сетки: {len(grid)}")
    for area in (0.05, 0.10, 0.15, 0.20):
        thr = float(np.nanquantile(grid["oof"].to_numpy(), 1.0 - area))
        cov = float((pos_oof >= thr).mean())
        print(f"  held-out: точек в top-{int(area*100):>2}% = {cov*100:5.1f}%  (lift {cov/area:.2f}x)")

    # --- карта ---
    fig, ax = plt.subplots(figsize=(10, 10))
    bins = np.linspace(0, 1, config.N_DISPLAY_CLASSES + 1)
    disp = np.digitize(grid["oof"].fillna(0).to_numpy(), bins[1:-1], right=False)
    grid = grid.copy()
    grid["oof_class"] = disp
    norm = BoundaryNorm(np.arange(config.N_DISPLAY_CLASSES + 1), plt.cm.RdBu_r.N)
    grid.plot(column="oof_class", ax=ax, cmap="RdBu_r", norm=norm, linewidth=0)
    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)

    # точки: попадание в top-AREA_MAIN held-out
    thr_main = float(np.nanquantile(grid["oof"].to_numpy(), 1.0 - AREA_MAIN))
    pts = gpd.sjoin(points[["geometry"]], grid[["oof", "geometry"]], how="left", predicate="within")
    hit = pts["oof"] >= thr_main
    pts[hit].plot(ax=ax, color="yellow", markersize=45, edgecolor="black", linewidth=0.6, zorder=5)
    pts[~hit].plot(ax=ax, facecolor="none", markersize=45, edgecolor="black", linewidth=0.6, zorder=5)

    cov_main = float((pos_oof >= thr_main).mean())
    ax.set_title(
        f"Held-out (out-of-fold) перспективность · {n_pos} точек\n"
        f"в top-{int(AREA_MAIN*100)}%: {int(round(cov_main*n_pos))} "
        f"({cov_main*100:.0f}%), lift {cov_main/AREA_MAIN:.2f}x — модель НЕ видела блок точки",
        fontsize=11,
    )
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="yellow",
               markeredgecolor="black", markersize=9, label=f"точка в top-{int(AREA_MAIN*100)}%"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=9, label="точка вне top-N%"),
    ], loc="lower right", frameon=True, fontsize=9)
    ax.set_axis_off()
    plt.tight_layout()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "forecast_oof.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {out_png}")


if __name__ == "__main__":
    main()
