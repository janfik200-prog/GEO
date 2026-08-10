"""Считает src/relief_v2_extra.py на целевой сетке и пишет parquet для сборщика.

Запуск: ``python -m experiments.build_relief_extra``.
Выход: ``data/processed/relief_extra_features.parquet`` — 4 столбца ``relief2_*``
(:data:`src.relief_v2_extra.RELIEF2_COLS`), одна строка на ячейку, C-порядок.

``dem_elev`` для водосбора берётся из уже посчитанного ``terrain_v2.parquet``
(та же колонка 500 м, что видит src/catchments.py в заверке) — повторного
скачивания DEM для этого шага не требуется, а перепроецированный растр 100 м
для кривизны/TRI читается из кэша ``datacache/anabar_dem``, заполненного тем
же terrain_v2.projected_dem при первом запуске build_dataset_v2 (группа ``ter``).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config, integro_grid, relief_v2_extra  # noqa: E402


def main() -> None:
    target_meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    n_cells = target_meta.prf * target_meta.pic

    terrain_path = config.PROCESSED_DIR / "terrain_v2.parquet"
    dem_elev = pd.read_parquet(terrain_path)["dem_elev"].to_numpy(float)
    assert dem_elev.size == n_cells, f"{dem_elev.size} строк вместо {n_cells}"

    df_diag = relief_v2_extra.relief_extra_features(target_meta, dem_elev, keep_dropped=True)
    print("relief2_tri (отбракован, дубль dem_slope, см. докстринг модуля):")
    print(df_diag[["relief2_tri"]].describe().T[["mean", "std", "min", "max"]].to_string())

    df = df_diag[list(relief_v2_extra.RELIEF2_COLS)]
    assert len(df) == n_cells, f"{len(df)} строк вместо {n_cells}"
    assert list(df.columns) == list(relief_v2_extra.RELIEF2_COLS)

    out_path = config.PROCESSED_DIR / "relief_extra_features.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{out_path}: {df.shape[0]} строк x {df.shape[1]} признаков")
    print(df.describe().T[["mean", "std", "min", "max"]].to_string())
    print(df.isna().mean().rename("NaN, доля").to_string())


if __name__ == "__main__":
    main()
