"""Считает src/geology_v2.py на целевой сетке и пишет parquet для сборщика.

Запуск: ``python -m experiments.build_geology_v2``.
Выход: ``data/processed/geology_v2_features.parquet`` — 6 столбцов ``geo2_*``
(:data:`src.geology_v2.GEO2_COLS`), одна строка на ячейку, C-порядок.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, geology_v2, integro_grid  # noqa: E402


def main() -> None:
    target_meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    n_cells = target_meta.prf * target_meta.pic
    df = geology_v2.geology_v2_features(target_meta)
    assert len(df) == n_cells, f"{len(df)} строк вместо {n_cells}"
    assert list(df.columns) == list(geology_v2.GEO2_COLS)

    out_path = config.PROCESSED_DIR / "geology_v2_features.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{out_path}: {df.shape[0]} строк x {df.shape[1]} признаков")
    print(df.describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
