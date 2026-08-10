"""Считает src/potential_fields.py на целевой сетке и пишет parquet для сборщика.

Запуск: ``python -m experiments.build_potential_fields``.
Выход: ``data/processed/potfield_features.parquet`` — 16 столбцов ``pf_*``
(:data:`src.potential_fields.PF_COLS`), одна строка на ячейку, C-порядок —
готов к склейке в ``experiments/build_dataset_v2.py`` (группа ``pf``).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, integro_grid, potential_fields  # noqa: E402


def main() -> None:
    target_meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    n_cells = target_meta.prf * target_meta.pic
    df = potential_fields.potential_field_features(target_meta)
    assert len(df) == n_cells, f"{len(df)} строк вместо {n_cells}"
    assert list(df.columns) == list(potential_fields.PF_COLS)

    out_path = config.PROCESSED_DIR / "potfield_features.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{out_path}: {df.shape[0]} строк x {df.shape[1]} признаков")
    print(df.isna().mean().rename("NaN, доля").to_string())


if __name__ == "__main__":
    main()
