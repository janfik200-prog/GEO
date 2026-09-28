"""Карта Анабара с линеаментами: одна площадка, крупно, с мерой уверенности.

Линии берутся из ансамблевой трассировки (``openlin_lines_ens``): цвет — доля
подвыборок обучения, в которых линия прошла здесь. Тёмно-синие проходят изредка,
жёлтые — почти всегда. Это и есть уверенность: одиночная трассировка такой меры
не даёт, её список на другой подвыборке будет другим.

Модель обучена на Неваде и Анабар не видела, поэтому вся площадь оценивается
целиком, без деления на полосы.

Запуск из корня репозитория (после openlin_lines_ens):
    python -m experiments.openlin_anabar_lines [--site anabar] [--min-freq 0.25]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D

from experiments.openlin_anfis import TOL_M
from experiments.openlin_lines_ens import CONSENSUS
from experiments.openlin_lines_map import robust
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")

TITLES = {"anabar": "Анабар, разломы", "anabar_dykes": "Анабар, дайки"}
SHORT = {"nevada": "Неваде", "anabar": "Анабаре", "anabar_dykes": "дайках"}


def fat(mask: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, np.ones((k, k), dtype=bool))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--train-on", default="nevada")
    ap.add_argument("--model", default="логрегрессия")
    ap.add_argument("--target", default="smooth")
    ap.add_argument("--tag", default="", help="суффикс прогона, напр. _густо")
    ap.add_argument("--min-freq", type=float, default=0.25,
                    help="ниже этой частоты линия не показывается вовсе")
    a = ap.parse_args()

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    gray = robust(z["gray"])
    fault = (z["dist"] <= res_m * 0.7) & valid

    p = M / (f"openlin_lines_ens_{a.train_on}{a.tag}_{a.model}_{a.target}"
             f"_{a.site}.npz")
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала прогон openlin_lines_ens")
    freq = np.load(p)["freq"]

    tab = pd.read_csv(M / f"openlin_lines_ens_{a.train_on}{a.tag}.csv")
    row = tab[(tab["модель"] == a.model) & (tab["таргет"] == a.target)
              & (tab["площадка"] == a.site)].iloc[0]
    base = float(((z["dist"] <= TOL_M) & valid).sum() / valid.sum())

    from experiments.openlin_lines_ens import thin

    shown = (freq >= a.min_freq) & valid
    axis = thin(shown) & valid                    # осевые линии показанных полос
    fatax = fat(axis)
    # цвет ячейки — её собственная частота: утолщение шире осевой всего на ячейку,
    # а частота накоплена с допуском 250 м и на таком сдвиге не меняется
    col = np.where(fatax, freq, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.2))

    axes[0].imshow(gray, cmap="gray", alpha=0.85)
    im = axes[0].imshow(np.ma.masked_invalid(col), cmap="viridis",
                        norm=Normalize(vmin=a.min_freq, vmax=1.0))
    axes[0].set_title("линеаменты, выделенные программой\n"
                      "цвет — в скольких обучениях из {} линия прошла здесь"
                      .format(int(row["подвыборок"])), fontsize=11)
    cb = plt.colorbar(im, ax=axes[0], fraction=0.040, pad=0.02)
    cb.set_label("частота", fontsize=9)

    axes[1].imshow(gray, cmap="gray", alpha=0.6)
    axes[1].imshow(np.ma.masked_where(~fault, np.zeros_like(gray)),
                   cmap=ListedColormap(["#e8402a"]), alpha=0.85)
    cons = fat(thin((freq >= CONSENSUS) & valid) & valid)
    axes[1].imshow(np.ma.masked_where(~cons, np.zeros_like(gray)),
                   cmap=ListedColormap(["#12a150"]), alpha=1.0)
    weak = fatax & ~cons
    axes[1].imshow(np.ma.masked_where(~weak, np.zeros_like(gray)),
                   cmap=ListedColormap(["#7fd3a6"]), alpha=0.9)
    axes[1].set_title("то же и разломы по карте\n"
                      "у консенсусных линий на карту легло {:.0%} длины "
                      "(у случайной линии {:.0%}), p = {:.3f}"
                      .format(row["точность_50"], base, row["p"]), fontsize=11)
    axes[1].legend(handles=[
        Line2D([], [], color="#12a150", lw=3,
               label=f"консенсус (частота >= {CONSENSUS:.0%})"),
        Line2D([], [], color="#7fd3a6", lw=3, label="реже, но выше порога показа"),
        Line2D([], [], color="#e8402a", lw=3, label="разломы по карте")],
        loc="lower left", fontsize=9, framealpha=0.92)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(
        valid.shape[1] * res_m / 1000, valid.shape[0] * res_m / 1000), fontsize=9.5)

    fig.suptitle("{}: ансамбль трассировок, модель обучена на {} и эту площадку "
                 "не видела".format(TITLES.get(a.site, a.site),
                                    SHORT.get(a.train_on, a.train_on)),
                 fontsize=14, y=0.97)
    fig.text(0.5, 0.03,
             "воспроизводимость списка: {:.2f} (две половины ансамбля); "
             "у одиночной трассировки было 0.18"
             .format(row["совпадение_половин"]),
             ha="center", fontsize=9.5, color="#444")

    out = Path(f"outputs/openlin_{a.site}_lines_ens{a.tag}_from_{a.train_on}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"сохранено: {out}  точность {row['точность_50']:.3f} "
          f"против {base:.3f}  p = {row['p']:.3f}")


if __name__ == "__main__":
    main()
