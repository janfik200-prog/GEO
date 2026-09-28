"""Непрерывное поле скора по Анабару: без порога, без чистки, без линий.

ЗАЧЕМ ИМЕННО ТАК. Обе пороговые формы вводят в заблуждение, и это измерено:
верхние 10% площади после чистки дают связные пятна, которые глаз читает как
структуру, а на отложенных полосах меряются lift 0.87 (ниже случайной карты);
трассированные линии меряются 1.39, но выглядят рассыпанными. Непрерывное поле
не обещает ни того, ни другого: оно показывает, где модель считает разлом более
вероятным, и ровно настолько, насколько считает.

ЧТО ПОКАЗАНО. Слева поле как есть. В середине оно же с разломами по карте, чтобы
можно было судить глазом. Справа — то, чего по картинке не видно: насколько
сигнал сконцентрирован, то есть какой lift получается, если брать верхние N%
площади. Эта кривая и есть честное содержание поля.

Конфигурация — лучшая по измеренному переносу на Анабар: логрегрессия, жёсткий
таргет, обучение на Неваде и Абитиби с признаками окрестности (lift 1.399,
p = 0.031 при пре-зарегистрированном пороге 0.00026). Модель Анабар не видела.

Запуск из корня репозитория:
    python -m experiments.openlin_anabar_score
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.openlin_anfis import (N_TRAIN_ROWS, SEED, TARGET_HALFWIDTH_M,
                                       TOL_M, baseline_predict, block_split,
                                       fit_baselines, rows, score_map,
                                       soft_target)
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, lift_at_top, load_site

M = Path("outputs/metrics")
OUT = Path("outputs/openlin_anabar_score.png")

#: Однотонная последовательная шкала (синяя ветвь палитры справочника): величина
#: меряется светлотой, а не сменой цвета. Радужные шкалы здесь запрещены — они
#: создают ложные границы там, где поле меняется плавно.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
C_FAULT = "#e34948"
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"

#: Доли площади для кривой концентрации. Нижняя граница 0.5% — примерно столько
#: занимает трассировка, верхняя 30% — дальше отбор перестаёт быть отбором.
FRACS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def build(site: str, support: np.ndarray | None):
    z = load_site(site)
    x, _y, valid, use = build_stack(z, CHANNELS)
    nx, _n = neigh_stack(x, use, valid, float(z["res_m"]),
                         support=valid if support is None else support)
    return z, np.concatenate([x, nx]), valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--train-on", default="nevada,abitibi")
    ap.add_argument("--target", default="hard", choices=("hard", "smooth"))
    ap.add_argument("--model", default="логрегрессия")
    a = ap.parse_args()

    train_sites = [s for s in a.train_on.split(",") if s]
    xtr_parts, dtr_parts = [], []
    for s in train_sites:
        z = load_site(s)
        v = np.isfinite(z["gray"])
        t, _va = block_split(v.shape, float(z["res_m"]))
        t &= v
        _z, x, _v = build(s, t)
        xtr_parts.append(rows(x, t))
        dtr_parts.append(z["dist"][t])
        print(f"  обучение {s}: {int(t.sum()):,} ячеек")
    xtr = np.concatenate(xtr_parts)
    dtr = np.concatenate(dtr_parts)
    rng = np.random.default_rng(SEED)
    if len(xtr) > N_TRAIN_ROWS:
        sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
        xtr, dtr = xtr[sel], dtr[sel]
    soft = a.target == "smooth"
    ytr = (soft_target(dtr) if soft
           else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))
    est = fit_baselines(xtr, ytr, soft)[a.model]

    z, x, valid = build(a.site, None)
    res_m = float(z["res_m"])
    dist = z["dist"]
    score = score_map(baseline_predict(est, rows(x, valid)), valid)
    base = float(((dist <= TOL_M) & valid).sum() / valid.sum())

    lifts = [lift_at_top(np.nan_to_num(score, nan=-1e9), dist, valid,
                         fraction=f, tol_m=TOL_M)["lift"] for f in FRACS]
    print(f"\nбазовая доля площади у карты {base:.3f}")
    for f, l in zip(FRACS, lifts):
        print(f"  верхние {100 * f:>5.1f}% площади: lift {l:.3f}")

    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    for ax in axes:
        ax.set_facecolor(SURFACE)

    im = axes[0].imshow(np.where(valid, score, np.nan), cmap=cmap)
    axes[0].set_title("поле скора: где модель считает разлом вероятнее\n"
                      "порога нет, чистки нет", fontsize=10.5, color=INK)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.040, pad=0.02)
    cb.set_label("скор модели", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)

    axes[1].imshow(np.where(valid, score, np.nan), cmap=cmap)
    fault = (dist <= res_m * 0.7) & valid
    axes[1].imshow(np.ma.masked_where(~fault, np.zeros_like(score)),
                   cmap=ListedColormap([C_FAULT]), alpha=0.9)
    axes[1].set_title("то же и разломы по карте (красным)\n"
                      "судить глазом можно только по этой панели",
                      fontsize=10.5, color=INK)

    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(
        valid.shape[1] * res_m / 1000, valid.shape[0] * res_m / 1000),
        fontsize=9, color=INK2)

    ax = axes[2]
    ax.plot([100 * f for f in FRACS], lifts, "o-", color=BLUE[4], lw=2, ms=7)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.text(30, 1.02, "случайная карта", ha="right", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 30])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "30"])
    ax.set_xlabel("сколько процентов площади отбираем", fontsize=9, color=INK2)
    ax.set_ylabel("lift: во сколько раз чаще, чем наугад", fontsize=9,
                  color=INK2)
    ax.set_ylim(0.9, max(lifts) * 1.12)
    ax.set_title("Насколько сигнал сконцентрирован\n"
                 "это и есть честное содержание поля", fontsize=10.5, color=INK)
    ax.grid(alpha=0.2, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d3")
    ax.tick_params(colors=INK2, labelsize=8)

    fig.suptitle("Анабар: непрерывное поле скора ({}, таргет {}, обучение на {})"
                 .format(a.model, a.target, " + ".join(train_sites)),
                 fontsize=13.5, color=INK, y=0.98)
    fig.text(0.5, 0.02,
             "модель эту площадку не видела; на всей площади lift@10% = "
             f"{lifts[FRACS.index(0.10)]:.2f} при уровне случайной карты 1.00",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    print(f"\nсохранено: {OUT}")


if __name__ == "__main__":
    main()
