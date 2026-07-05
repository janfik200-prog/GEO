"""Headless-прогон валидации: ML vs критериальный baseline (таксономия ГИС Интегро).

Повторяет логику notebook/validation_report.ipynb без Jupyter. Запуск из корня:
    python3 -m experiments.validation [--quick]
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, compute_geo_score
from src.model import mark_presence
from src.validation import spatial_cv_evaluate, repeated_spatial_cv

QUICK = "--quick" in sys.argv


def main() -> None:
    t0 = time.time()
    base = find_base_dir()
    shp_dir = base / config.SHP_SUBDIR
    with tempfile.TemporaryDirectory() as alias_dir:
        layers, points = load_all_layers(shp_dir, Path(alias_dir))
    print(f"[{time.time()-t0:5.1f}s] данные загружены: base={base}")

    grid, _mask_union, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = compute_geo_score(grid, shape)
    grid, pos = mark_presence(grid, points)
    print(f"[{time.time()-t0:5.1f}s] сетка: {len(grid)} ячеек {shape}, "
          f"рудопроявлений (positive cells): {len(pos)}")

    # сводка по новому baseline
    d = grid["taxonomy_distance"].to_numpy()
    print(f"           taxonomy_distance: min={d.min():.3f} max={d.max():.3f} mean={d.mean():.3f} "
          f"(меньше=лучше); geo_score: min={grid['geo_score'].min():.3f} max={grid['geo_score'].max():.3f}")

    if QUICK:
        agg, _ = spatial_cv_evaluate(grid, pos, seed=1)
        print(f"\n[{time.time()-t0:5.1f}s] === ОДИН СИД (1), lift по площадям ===")
        tbl = agg.pivot(index="model", columns="area", values="lift").round(2)
        print(tbl.to_string())
        return

    print(f"[{time.time()-t0:5.1f}s] запуск repeated_spatial_cv по сидам {config.VAL_SEEDS} ...")
    res = repeated_spatial_cv(grid, pos)
    print(f"\n[{time.time()-t0:5.1f}s] === LIFT (среднее по {len(config.VAL_SEEDS)} сидам) ===")
    tbl = res.pivot(index="model", columns="area", values="lift").round(2)
    order = [m for m in ["Random", "geo_score (эвристика)", "Logistic Regression",
                         "Gradient Boosting", "Random Forest"] if m in tbl.index]
    print(tbl.loc[order].to_string())
    print(f"\n[{time.time()-t0:5.1f}s] готово.")


if __name__ == "__main__":
    main()
