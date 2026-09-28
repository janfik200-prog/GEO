"""Картинка прогона ANFIS по линеаментному таргету.

Слева — скор ANFIS по всей площадке: модель обучена на обучающих полосах и
видит только спектр ячейки, без признаков окрестности. В середине — тот же скор
на проверочных полосах с подтверждёнными линеаментами поверх: там и меряется
lift. Справа — то, ради чего прогон делался: внутри площадки против переноса,
по трём моделям.

Запуск из корня репозитория (после openlin_anfis_lin):
    python -m experiments.openlin_anfis_lin_map --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from experiments.openlin_anfis import (AREA, N_RULES, N_TRAIN_ROWS, SEED,
                                       TARGET_HALFWIDTH_M, TOL_M,
                                       anfis_predict, block_split, fit_anfis,
                                       rows, score_map)
from experiments.openlin_anfis_lin import lineament_dist
from experiments.openlin_lines_map import fat
from experiments.openlin_unet import CHANNELS, build_stack, lift_at_top, load_site

M = Path("outputs/metrics")

#: Однотонная последовательная шкала — величина меряется светлотой.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab",
        "#104281"]
C_LIN, C_ANFIS, C_LOGREG, C_BOOST = "#1baf7a", "#2a78d6", "#eb6834", "#c98b3a"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SHORT = {"anabar": "Анабар", "nevada": "Невада"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--target", default="hard", choices=("hard", "smooth"))
    a = ap.parse_args()

    z = load_site(a.site)
    x, _y, valid, _use = build_stack(z, CHANNELS)
    res_m = float(z["res_m"])
    dist = lineament_dist(a.site, res_m)
    tr, va = block_split(valid.shape, res_m)
    tr &= valid
    va &= valid

    xtr = rows(x, tr)
    dtr = dist[tr]
    rng = np.random.default_rng(SEED)
    if len(xtr) > N_TRAIN_ROWS:
        sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
        xtr, dtr = xtr[sel], dtr[sel]
    soft = a.target == "smooth"
    ytr = (np.clip(1.0 - dtr / 500.0, 0, 1).astype("float32") if soft
           else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))
    model = fit_anfis(xtr, ytr, N_RULES, soft, lambda *_: None)
    score = score_map(anfis_predict(model, rows(x, valid)), valid)
    got = lift_at_top(np.nan_to_num(score, nan=-1e9), dist, va,
                      fraction=AREA, tol_m=TOL_M)
    lines = np.load(M / "openlin_unsup_{}_предельно.npz".format(a.site))["confirmed"]

    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax in axes[:2]:
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])

    im = axes[0].imshow(np.where(valid, score, np.nan), cmap=cmap)
    axes[0].set_title("скор ANFIS: где ожидается линеамент\n"
                      "вход — только 10 сырых каналов, без окрестности",
                      fontsize=10.5, color=INK)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.040, pad=0.02)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(
        valid.shape[1] * res_m / 1000, valid.shape[0] * res_m / 1000),
        fontsize=9, color=INK2)

    # Вне проверочных полос скор оставлен приглушённым, а не убран: иначе панель
    # теряет географию и по ней нельзя понять, где эти полосы на площадке.
    axes[1].imshow(np.where(valid, score, np.nan), cmap=cmap, alpha=0.28)
    axes[1].imshow(np.where(va, score, np.nan), cmap=cmap)
    axes[1].imshow(np.ma.masked_where(~fat(lines & va), np.zeros_like(score)),
                   cmap=ListedColormap([C_LIN]), alpha=1.0)
    axes[1].set_title("проверочные полосы: скор и подтверждённые линеаменты\n"
                      "именно здесь меряется lift = {:.2f}".format(got["lift"]),
                      fontsize=10.5, color=INK)

    ax = axes[2]
    ax.set_facecolor(SURFACE)
    df = pd.read_csv(M / f"openlin_anfis_lin_{a.site}.csv")
    df = df[df["таргет"] == a.target]
    models = ["ANFIS", "логрегрессия", "бустинг"]
    colors = {"ANFIS": C_ANFIS, "логрегрессия": C_LOGREG, "бустинг": C_BOOST}
    modes = ["проверка", "перенос"]
    pos = np.arange(len(modes))
    for i, m in enumerate(models):
        vals = [float(df[(df["модель"] == m) & (df["режим"] == md)]["lift"].iloc[0])
                for md in modes]
        ax.bar(pos + (i - 1) * 0.27, vals, 0.25, color=colors[m], label=m,
               linewidth=0)
        for xx, v in zip(pos + (i - 1) * 0.27, vals):
            ax.text(xx, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(["своя площадка\n(блочно, буфер 2 км)",
                        "перенос на другую\nплощадку"], fontsize=9, color=INK2)
    ax.set_ylabel("lift@10% по линеаментам", fontsize=9, color=INK2)
    ax.set_ylim(0, 2.05)
    ax.set_title("Подпись есть, но не переносится\n"
                 "пунктир — уровень случайной карты", fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d3")
    ax.tick_params(colors=INK2, labelsize=8)

    fig.suptitle("{}: ANFIS обучен на подтверждённых линеаментах, "
                 "не на разломах".format(SHORT.get(a.site, a.site)),
                 fontsize=13.5, color=INK, y=0.98)
    out = Path(f"outputs/openlin_anfis_lin_{a.site}.png")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"сохранено: {out}  lift на проверке {got['lift']:.3f}")


if __name__ == "__main__":
    main()
