# -*- coding: utf-8 -*-
"""Картинки общей части большой колоды.

crit_wide.png — воссозданный критериальный балл на всей области расчёта: основной
лист (контур) и смежная территория к югу от него. Вход:
`data/processed/criterial_reconstruct_multisheet.parquet` (пишет
`experiments/criterial_reconstruct_multisheet.py`; столбцы воссоздание_вся_область,
в_маске, в_габаритах_листа). Балл показан долей ячеек с таким же или лучшим
значением: меньше — перспективнее, как в ГИС Интегро.
Запуск: python -X utf8 -m experiments.make_report_figures_general
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs"


def crit_wide():
    d = pd.read_parquet(ROOT / "data" / "processed" / "criterial_reconstruct_multisheet.parquet")
    d = d[d["в_маске"]].copy()
    d["r"] = d["воссоздание_вся_область"].rank(pct=True)
    xs = np.sort(d["x"].unique())
    ys = np.sort(d["y"].unique())
    ix = np.searchsorted(xs, d["x"].to_numpy())
    iy = np.searchsorted(ys, d["y"].to_numpy())
    g = np.full((len(ys), len(xs)), np.nan)
    g[iy, ix] = d["r"].to_numpy()
    s = np.zeros_like(g)
    m = d["в_габаритах_листа"].to_numpy()
    s[iy[m], ix[m]] = 1
    ext = [(xs[0] - 250) / 1000, (xs[-1] + 250) / 1000, (ys[0] - 250) / 1000, (ys[-1] + 250) / 1000]
    fig, ax = plt.subplots(figsize=(3.9, 3.45), dpi=200)
    im = ax.imshow(g, origin="lower", extent=ext, cmap="viridis_r", vmin=0, vmax=1)
    ax.contour(np.linspace(ext[0], ext[1], g.shape[1]), np.linspace(ext[2], ext[3], g.shape[0]),
               s, levels=[0.5], colors="#dc2626", linewidths=1.6)
    ax.set_xlabel("X, км", fontsize=8)
    ax.set_ylabel("Y, км", fontsize=8)
    ax.tick_params(labelsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("балл формулы: доля ячеек с таким же или лучшим\n(меньше — перспективнее)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.plot([], [], color="#dc2626", lw=1.6, label="основной лист")
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "crit_wide.png", bbox_inches="tight", pad_inches=0.05)
    print("crit_wide.png", g.shape, int(np.isfinite(g).sum()), "ячеек")


if __name__ == "__main__":
    crit_wide()
