"""Превью dataset_v6 в стиле dataset_v3_preview/dataset_v5_preview (build_dataset_v2.py):
шесть представителей разных групп признаков на сетке листа.

Запуск из корня: ``python -m experiments.plot_dataset_v6_preview``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cell_mask, config, criterial_target, integro_grid  # noqa: E402

OUT = config.PROJECT_ROOT / "outputs" / "dataset_v6_preview.png"

SHOW = ["dem_incision", "lin_dens", "s2_clay", "s2_ndvi", "s1_vv_vh", "psr_hh_hv",
        "ast_aloh", "l8_lst", "s2_b11", "dem_tpi_2km"]


def main() -> None:
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    train = criterial_target.training_features(dataset="v6")
    valid = np.asarray(cell_mask.build_valid_mask(raw))
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    shape = (meta.prf, meta.pic)

    show = [c for c in SHOW if c in train.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, c in zip(axes.ravel(), show):
        img = np.where(valid.reshape(shape), raw[c].to_numpy().reshape(shape), np.nan)
        lo, hi = np.nanpercentile(img, [2, 98])
        im = ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(c, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle(f"Датасет v6: {train.shape[1]} признаков для обучения "
                 f"({raw.shape[1]} столбцов в файле), {int(valid.sum())} валидных ячеек", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
