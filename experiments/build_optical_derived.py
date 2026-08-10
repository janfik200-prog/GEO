"""Считает src/optical_derived.py на уже собранном датасете, пишет parquet.

Вход — ``dataset_v3.parquet`` (там уже есть ``s2_b*`` и ``ast_b*`` из этапа 9,
новых скачиваний не нужно). Запуск: ``python -m experiments.build_optical_derived``.
Выход: ``data/processed/optical_derived_features.parquet`` — 3 столбца
(:data:`src.optical_derived.OPTICAL_DERIVED_COLS`), тот же порядок строк.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config, optical_derived  # noqa: E402


def main() -> None:
    src_path = config.PROCESSED_DIR / "dataset_v3.parquet"
    df = pd.read_parquet(src_path)
    out = optical_derived.optical_derived_features(df)

    out_path = config.PROCESSED_DIR / "optical_derived_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"{out_path}: {out.shape[0]} строк x {out.shape[1]} признаков")
    print(out.isna().mean().rename("NaN, доля").to_string())

    # Дублирование ast_mgoh с уже посчитанным ast_carb — не постулируется, а измеряется
    corr = out["ast_mgoh"].corr(df["ast_carb"], method="spearman")
    print(f"\nSpearman(ast_mgoh, ast_carb) = {corr:.3f}")


if __name__ == "__main__":
    main()
