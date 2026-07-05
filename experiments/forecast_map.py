"""Построить карту перспективности боевой модели (ансамбль RF+GB).

Боевая модель train_model = BackgroundEnsemble (RF+GB). Скрипт прогоняет
полный боевой пайплайн и сохраняет карту перспективности (с золотыми зонами)
в outputs/forecast_rf_gb.png.
"""

import sys, tempfile, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent   # корень репозитория (на уровень выше experiments/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features
from src.model import (
    mark_presence, train_model, add_uncertainty, compute_prospectivity,
    mark_gold_zones, point_top_coverage,
)
from src.visualization import make_display_classes, plot_final


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    mask = layers["mask"]

    grid, mask_union, grid_shape = build_grid(mask, config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid, positive_cells = mark_presence(grid, points)
    grid, model = train_model(grid, positive_cells)       # боевая модель = RF+GB ансамбль
    grid = compute_prospectivity(grid, grid_shape)
    grid = make_display_classes(grid)
    grid = mark_gold_zones(grid, grid_shape, mask_union)
    grid = add_uncertainty(grid, model)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "forecast_rf_gb.png"
    plot_final(grid, mask, points, out_png)

    from scipy import ndimage
    arr = np.zeros(grid_shape, dtype=np.uint8)
    arr[grid["row"].to_numpy(), grid["col"].to_numpy()] = grid["gold_zone"].to_numpy().astype(np.uint8)
    _, n_zones = ndimage.label(arr, structure=np.ones((3, 3), dtype=np.uint8))

    cov = point_top_coverage(grid, positive_cells)
    print(f"Реальных ячеек-точек: {len(positive_cells)} | ячеек сетки: {len(grid)}")
    print(f"Золотых зон: {n_zones} ({int(grid['gold_zone'].sum())} ячеек)")
    print(f"Доля точек в верхних 15% прогноза (in-sample): {cov:.3f}")
    print(f"Сохранено: {out_png}")


if __name__ == "__main__":
    main()
