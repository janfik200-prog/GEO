"""Итоговая карта линеаментов Анабара: два представления, общий фильтр.

Линии добываются двумя независимыми представлениями одного и того же снимка —
компонентами PCA и латентом свёрточного автоэнкодера, — и каждая проходит один
и тот же фильтр: её обязаны видеть ОБА прибора (оптика и радар) по отдельности,
вдоль не меньше половины длины и в том же направлении.

Представления пересекаются лишь наполовину, поэтому в карту идёт объединение:
255 веток против 153 у одного PCA при ожидаемых 2.1 случайных линиях на всю
площадь.

Запуск из корня репозитория (после openlin_unsup и openlin_mae):
    python -m experiments.openlin_final_map
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from experiments.openlin_lines_diag import azimuths
from experiments.openlin_lines_map import fat, robust
from experiments.openlin_unet import load_site
from experiments.openlin_unsup import ROSE_BIN, rose

M = Path("outputs/metrics")
C_BOTH, C_PCA, C_MAE, C_FAULT = "#1baf7a", "#2a78d6", "#eb6834", "#e34948"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    d = np.load(M / f"openlin_union_{a.site}.npz")
    pca, mae, uni = d["pca"], d["mae"], d["confirmed"]
    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    gray = robust(z["gray"])
    fault = (z["dist"] <= res_m * 0.7) & valid

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})
    for ax in axes[:2]:
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(gray, cmap="gray", alpha=0.8)
    axes[0].imshow(np.ma.masked_where(~fat(uni), np.zeros_like(gray)),
                   cmap=ListedColormap([C_BOTH]), alpha=1.0)
    axes[0].set_title("линеаменты Анабара: {} линий, {:.0f} км\n"
                      "каждая подтверждена и оптикой, и радаром"
                      .format(255, uni.sum() * res_m / 1000),
                      fontsize=10.5, color=INK)
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(
        valid.shape[1] * res_m / 1000, valid.shape[0] * res_m / 1000),
        fontsize=9, color=INK2)

    axes[1].imshow(gray, cmap="gray", alpha=0.55)
    both = pca & mae
    axes[1].imshow(np.ma.masked_where(~fat(pca & ~mae), np.zeros_like(gray)),
                   cmap=ListedColormap([C_PCA]), alpha=0.9)
    axes[1].imshow(np.ma.masked_where(~fat(mae & ~pca), np.zeros_like(gray)),
                   cmap=ListedColormap([C_MAE]), alpha=0.9)
    axes[1].imshow(np.ma.masked_where(~fat(both), np.zeros_like(gray)),
                   cmap=ListedColormap([C_BOTH]), alpha=1.0)
    axes[1].set_title("вклад представлений\n"
                      "синие видит только PCA, оранжевые только автоэнкодер",
                      fontsize=10.5, color=INK)
    axes[1].legend(handles=[Line2D([], [], color=C_BOTH, lw=3, label="оба"),
                            Line2D([], [], color=C_PCA, lw=3, label="только PCA"),
                            Line2D([], [], color=C_MAE, lw=3,
                                   label="только автоэнкодер")],
                   loc="lower left", fontsize=8.5, framealpha=0.9)

    ax = fig.add_subplot(1, 3, 3, projection="polar")
    axes[2].remove()
    bins = np.radians(np.arange(0, 180, ROSE_BIN) + ROSE_BIN / 2)
    for m_, color, lab in ((uni, C_BOTH, "линеаменты"),
                           (fault, C_FAULT, "контуры зон разломов")):
        h = rose(azimuths(m_, res_m))
        ax.plot(np.concatenate([bins, bins + np.pi, [bins[0]]]),
                np.concatenate([h, h, [h[0]]]), color=color, lw=2.2, label=lab)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title("розы азимутов", fontsize=10.5, color=INK, pad=20)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              frameon=False)

    fig.suptitle("Анабар: итоговая карта линеаментов по космоснимку "
                 "(обучения с учителем нет)", fontsize=13.5, color=INK, y=0.99)
    fig.text(0.5, 0.015, "случайно через фильтр прошло бы 2.1 линии из 255 — "
             "чистота отбора x121", ha="center", fontsize=9, color=INK2)
    out = Path(f"outputs/openlin_final_{a.site}.png")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
