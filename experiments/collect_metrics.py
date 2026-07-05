"""Собрать ОДНУ сводную таблицу метрик по всем методам.

Метрика — lift попадания held-out реальных точек в top-N% площади
(пространственная блочная CV, presence-background), усреднённый по сидам.
Методы: Random Forest, Gradient Boosting, Logistic Regression, ансамбль RF+GB
(боевой), критериальный geo_score, Random baseline.

Выход: metrics/metrics_table.csv (+ _lift_pm.csv с «mean ± std») и
metrics/metrics_table.png (рендер таблицы для презентации).
"""

import sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent   # корень репозитория (на уровень выше experiments/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, compute_geo_score
from src.model import (
    mark_presence, sample_presence_background,
    _make_rf, _make_gb, _make_lr, _make_ensemble,
)
from src.validation import assign_spatial_blocks, _coverage

AREAS = config.VAL_AREAS
SEEDS = config.VAL_SEEDS

MODEL_FACTORIES = {
    "Random Forest": _make_rf,
    "Gradient Boosting": _make_gb,
    "Logistic Regression": _make_lr,
    "Ensemble RF+GB (боевой)": _make_ensemble,
}

ORDER = [
    "Ensemble RF+GB (боевой)", "Random Forest", "Gradient Boosting",
    "Logistic Regression", "geo_score (критериальный)", "Random",
]


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, mask_union, grid_shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = compute_geo_score(grid, grid_shape)
    grid, positive_cells = mark_presence(grid, points)
    print(f"Реальных ячеек-точек: {len(positive_cells)} | ячеек сетки: {len(grid)}")

    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    geo_all = grid["geo_score"].to_numpy()

    records = []  # (model, area, lift)
    for seed in SEEDS:
        blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
        sample_pos, y = sample_presence_background(grid, positive_cells, config.VAL_N_BACKGROUND, seed)
        groups = blocks[sample_pos]
        X_sample = grid.iloc[sample_pos][config.FEATURE_COLS].fillna(0).to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=config.VAL_N_SPLITS, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(X_sample, y, groups):
            test_positions = sample_pos[te][y[te] == 1]
            if test_positions.size == 0:
                continue
            # обучаемые модели: один fit -> оценка всех площадей
            for name, factory in MODEL_FACTORIES.items():
                model = factory()
                model.fit(X_sample[tr], y[tr])
                score = model.predict_proba(X_all)[:, 1]
                for a in AREAS:
                    records.append((name, a, _coverage(score, test_positions, a) / a))
            # критериальный baseline (точки не использует — честно)
            for a in AREAS:
                records.append(("geo_score (критериальный)", a, _coverage(geo_all, test_positions, a) / a))
            # случайный прогноз: покрытие == площадь => lift = 1
            for a in AREAS:
                records.append(("Random", a, 1.0))
        print(f"  сид {seed}: готово")

    raw = pd.DataFrame(records, columns=["model", "area", "lift"])
    agg = raw.groupby(["model", "area"]).agg(
        lift=("lift", "mean"), lift_std=("lift", "std")
    ).reset_index()

    cols = [f"top-{int(a*100)}%" for a in AREAS]
    mean_t = agg.pivot(index="model", columns="area", values="lift").round(2)
    std_t = agg.pivot(index="model", columns="area", values="lift_std").round(2)
    mean_t.columns = cols
    std_t.columns = cols
    order = [m for m in ORDER if m in mean_t.index]
    mean_t = mean_t.reindex(order)
    std_t = std_t.reindex(order)

    # «mean ± std» для презентации
    pm = mean_t.copy().astype(str)
    for c in cols:
        pm[c] = mean_t[c].map(lambda v: f"{v:.2f}") + " ± " + std_t[c].map(lambda v: f"{v:.2f}")

    out_dir = ROOT / "metrics"
    out_dir.mkdir(exist_ok=True)
    mean_t.to_csv(out_dir / "metrics_table.csv")
    pm.to_csv(out_dir / "metrics_table_lift_pm.csv")

    print("\n=== Lift в top-N% площади (среднее по сидам) ===")
    print(mean_t.to_string())
    print("\n=== Lift (mean ± std по сидам) ===")
    print(pm.to_string())

    _render_png(pm, cols, out_dir / "metrics_table.png")
    print(f"\nСохранено: {out_dir/'metrics_table.csv'}")
    print(f"           {out_dir/'metrics_table_lift_pm.csv'}")
    print(f"           {out_dir/'metrics_table.png'}")


def _render_png(pm: pd.DataFrame, cols: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.0 + 1.6 * len(cols), 0.55 * (len(pm) + 1) + 0.6))
    ax.axis("off")
    ax.set_title("Качество методов: lift попадания точек в top-N% площади\n"
                 "(пространственная CV, среднее ± std по сидам; lift>1 = лучше случайного)",
                 fontsize=11, pad=14)
    tbl = ax.table(
        cellText=pm.values,
        rowLabels=pm.index,
        colLabels=cols,
        cellLoc="center", rowLoc="left", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    # подсветка боевой модели (первая строка данных)
    for j in range(len(cols)):
        tbl[(1, j)].set_facecolor("#fff3cd")
    # шапка столбцов (угловой ячейки rowLabel в шапке нет — её не трогаем)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#dfe6ee")
        tbl[(0, j)].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
