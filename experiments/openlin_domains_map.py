"""Картинка границ доменов: поле скачка признаков и снятые с него линии.

Три панели: сила границы (в сигмах), трассированные с неё линии на фоне карты
разломов и — главное — распределение силы границы НА разломах против фона. Если
два распределения совпадают, значит на разломах признаковый вектор не меняется,
и никакой оператор, работающий с этими признаками, их не найдёт.

Запуск из корня репозитория (после openlin_domains):
    python -m experiments.openlin_domains_map [--site anabar]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from experiments.openlin_anfis import TOL_M
from experiments.openlin_lines import DENSITY_DEFAULT, line_metrics, trace_lines
from experiments.openlin_lines_map import fat, robust
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")
TITLES = {"nevada": "Невада", "anabar": "Анабар, разломы",
          "anabar_dykes": "Анабар, дайки"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    a = ap.parse_args()

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    dist = z["dist"]
    gray = robust(z["gray"])
    fault = (dist <= res_m * 0.7) & valid

    p = M / f"openlin_domains_{a.site}.npz"
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала прогон openlin_domains")
    strength = np.load(p)["strength"]

    lines, br = trace_lines(strength, valid, valid, res_m, a.density)
    r = line_metrics(lines, dist, valid, res_m)
    base = float(((dist <= TOL_M) & valid).sum() / valid.sum())

    on = (dist <= TOL_M) & valid
    off = valid & ~on

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.6))

    im = axes[0].imshow(np.where(valid, strength, np.nan), cmap="magma")
    axes[0].set_title("сила границы доменов\n"
                      "(скачок признакового вектора поперёк линии, сигмы)",
                      fontsize=10.5)
    plt.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)

    axes[1].imshow(gray, cmap="gray", alpha=0.65)
    axes[1].imshow(np.ma.masked_where(~fault, np.zeros_like(gray)),
                   cmap=ListedColormap(["#e8402a"]), alpha=0.85)
    axes[1].imshow(np.ma.masked_where(~fat(lines), np.zeros_like(gray)),
                   cmap=ListedColormap(["#12a150"]), alpha=0.95)
    axes[1].set_title("границы доменов ({} линий, {:.0f} км)\n"
                      "точность {:.3f} против {:.3f} у случайной линии"
                      .format(len(br), r["длина_км"], r["точность"], base),
                      fontsize=10.5)
    axes[1].legend(handles=[Line2D([], [], color="#12a150", lw=3,
                                   label="границы доменов"),
                            Line2D([], [], color="#e8402a", lw=3,
                                   label="разломы по карте")],
                   loc="lower left", fontsize=9, framealpha=0.9)

    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    bins = np.linspace(0, float(np.nanpercentile(strength[valid], 99.5)), 60)
    axes[2].hist(strength[off], bins=bins, density=True, color="#9aa7d8",
                 alpha=0.85, label="вне разломов")
    axes[2].hist(strength[on], bins=bins, density=True, histtype="step",
                 color="#e8402a", lw=2, label="на разломах (допуск 250 м)")
    axes[2].axvline(float(np.mean(strength[off])), color="#4a5aa0", ls="--", lw=1.2)
    axes[2].axvline(float(np.mean(strength[on])), color="#e8402a", ls="--", lw=1.2)
    axes[2].set_xlabel("сила границы, сигмы")
    axes[2].set_ylabel("плотность")
    axes[2].set_title("Меняется ли признаковый вектор на разломах\n"
                      "средние {:.3f} и {:.3f} сигмы, отношение {:.2f}"
                      .format(float(np.mean(strength[on])),
                              float(np.mean(strength[off])),
                              float(np.mean(strength[on]) / np.mean(strength[off]))),
                      fontsize=10.5)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.25)

    fig.suptitle("{}: разлом как граница доменов — прямое измерение"
                 .format(TITLES.get(a.site, a.site)), fontsize=14, y=0.98)
    out = Path(f"outputs/openlin_{a.site}_domains.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
