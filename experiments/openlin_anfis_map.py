"""Картинка итога табличного прогона: карты ANFIS и цена утечки разбиения.

Верхние два ряда — по площадке: композит с разметкой, само разбиение
(обучение / буфер / проверка), скор ANFIS и его верхние 10% площади рядом с
разметкой. Нижний ряд — то, ради чего прогон и делался: lift на проверке при
случайном и при блочном разбиении.

Модель переобучается прямо здесь (8 секунд на площадку) — держать её на диске
ради одной картинки незачем.

Запуск из корня репозитория:
    python -m experiments.openlin_anfis_map
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from experiments.openlin_anfis import (AREA, TARGET_HALFWIDTH_M, TOL_M,
                                       anfis_predict, block_split, fit_anfis,
                                       rows, score_map)
from experiments.openlin_unet import CHANNELS, build_stack, lift_at_top, load_site

M = Path("outputs/metrics")
OUT = Path("outputs/openlin_anfis_map.png")

#: Какие площадки показываем и как подписываем.
PANELS = (("nevada", "Невада — обучение и проверка\nна одной площадке"),
          ("anabar", "Анабар — то же самое,\nсвоя разметка"))

#: Короткие подписи площадок для нижних диаграмм.
SHORT = {"nevada": "Невада", "anabar": "Анабар", "anabar_dykes": "Анабар, дайки"}
ORDER = ["Невада", "Анабар", "Анабар, дайки"]


def robust(a: np.ndarray) -> np.ndarray:
    """Растяжка по 2-98 перцентилю: у композита есть остаточные выбросы."""
    v = a[np.isfinite(a)]
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((a - lo) / ((hi - lo) or 1.0), 0, 1)


def draw_site(fig, gs, r: int, site: str, caption: str) -> None:
    z = load_site(site)
    x, _, valid, _ = build_stack(z, CHANNELS)
    dist = z["dist"]
    tr, va = block_split(valid.shape, float(z["res_m"]))
    tr &= valid
    va &= valid

    model = fit_anfis(rows(x, tr), (dist[tr] <= TARGET_HALFWIDTH_M).astype("float32"),
                      16, soft=False, log=lambda *_: None)
    score = score_map(anfis_predict(model, rows(x, valid)), valid)

    got = lift_at_top(np.nan_to_num(score, nan=-1e9), dist, va,
                      fraction=AREA, tol_m=TOL_M)
    thr = float(np.quantile(score[va], 1 - AREA))
    top = (score >= thr) & va
    target = (dist <= TOL_M) & va

    axes = [fig.add_subplot(gs[r, c]) for c in range(4)]

    # 1. композит + разметка
    ax = axes[0]
    ax.imshow(robust(z["gray"]), cmap="gray")
    ax.imshow(np.ma.masked_where(dist > TARGET_HALFWIDTH_M, np.zeros_like(dist)),
              cmap=ListedColormap(["#e8402a"]), alpha=0.9)
    ax.set_title("композит + разметка", fontsize=10)
    ax.set_ylabel(caption, fontsize=11.5, labelpad=10)

    # 2. разбиение
    ax = axes[1]
    lay = np.zeros(valid.shape)
    lay[valid] = 1.0          # буфер
    lay[tr] = 2.0
    lay[va] = 3.0
    ax.imshow(lay, cmap=ListedColormap(["#ffffff", "#d8d8d8", "#8fb8de", "#f0a04b"]),
              vmin=0, vmax=3)
    buf = valid.sum() - tr.sum() - va.sum()
    ax.set_title("обучение {:.0f}% (синее), буфер {:.0f}% (серое),\n"
                 "проверка {:.0f}% (жёлтое)".format(
                     100 * tr.sum() / valid.sum(), 100 * buf / valid.sum(),
                     100 * va.sum() / valid.sum()), fontsize=9.5)

    # 3. скор
    ax = axes[2]
    im = ax.imshow(np.where(valid, score, np.nan), cmap="magma")
    ax.set_title("скор ANFIS\n(обучен только на синей части)", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)

    # 4. верх 10% против разметки, только на проверочных полосах
    ax = axes[3]
    cat = np.zeros(valid.shape)
    cat[valid] = 1.0
    cat[va] = 2.0
    cat[target & ~top] = 3.0
    cat[top & ~target] = 4.0
    cat[top & target] = 5.0
    ax.imshow(cat, vmin=0, vmax=5,
              cmap=ListedColormap(["#ffffff", "#fafafa", "#ececec",
                                   "#e8402a", "#9aa7d8", "#12633f"]))
    hit = float(target[top].mean()) if top.sum() else float("nan")
    ax.set_title("разметка (красное), верх 10% (синее), совпало (зелёное)\n"
                 "lift = {:.2f}, попаданий {:.0f}%".format(got["lift"], 100 * hit),
                 fontsize=9.5)

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])


def main() -> None:
    fig = plt.figure(figsize=(17.0, 12.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.72], hspace=0.34,
                          wspace=0.14)

    for r, (site, caption) in enumerate(PANELS):
        draw_site(fig, gs, r, site, caption)

    # --- нижний ряд: цена утечки ------------------------------------------
    frames = []
    for t in ("nevada", "anabar", "anabar_dykes"):
        p = M / f"openlin_anfis_{t}.csv"
        if p.exists():
            f = pd.read_csv(p)
            f.insert(0, "площадка обучения", SHORT[t])
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    v = df[df["режим"] == "проверка"]
    piv = v.pivot_table(index=["площадка обучения", "модель"], columns="разбиение",
                        values="lift").reindex(ORDER, level=0)

    ax = fig.add_subplot(gs[2, :2])
    pos = np.arange(len(piv))
    ax.bar(pos - 0.19, piv["random"], 0.36, label="случайное 3/4 — 1/4",
           color="#e0a3a3", edgecolor="#b05555")
    ax.bar(pos + 0.19, piv["block"], 0.36, label="блочное, буфер 2 км",
           color="#8fb8de", edgecolor="#3f6fa0")
    ax.axhline(1.0, color="#333", lw=1.2, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(["{}\n{}".format(a.replace(", ", ",\n"), b)
                        for a, b in piv.index], fontsize=8.5)
    ax.set_ylim(0, float(piv.max().max()) * 1.34)
    ax.set_ylabel("lift@10% на проверке")
    ax.set_title("Цена утечки: одно обучение, два разбиения", fontsize=11.5)
    ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.text(len(piv) - 0.45, 1.05, "уровень случайной карты", fontsize=8.5,
            ha="right", color="#333")

    ax = fig.add_subplot(gs[2, 2:])
    b = df[(df["разбиение"] == "block") & (df["режим"] == "проверка")]
    piv2 = b.pivot_table(index="площадка обучения", columns="модель",
                         values="lift").reindex(ORDER)[["ANFIS", "логрегрессия",
                                                        "бустинг"]]
    pos = np.arange(len(piv2))
    for i, c in enumerate(piv2.columns):
        ax.bar(pos + (i - 1) * 0.27, piv2[c], 0.26, label=c,
               color=["#4a3f9f", "#9c9c9c", "#c98b3a"][i])
    ax.axhline(1.0, color="#333", lw=1.2, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels(piv2.index, fontsize=9.5)
    ax.set_ylim(0, float(piv2.max().max()) * 1.34)
    ax.set_ylabel("lift@10%, блочная проверка", labelpad=10)
    ax.set_title("Что даёт именно нечёткий вывод\n(те же строки, то же разбиение)",
                 fontsize=11.5)
    ax.legend(fontsize=9, ncol=3, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Табличная постановка: ANFIS по ячейкам композита", fontsize=15,
                 y=0.975)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"сохранено: {OUT}")


if __name__ == "__main__":
    main()
