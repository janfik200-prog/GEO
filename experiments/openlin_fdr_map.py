"""Карта с оценкой у каждой линии: порог двигает геофизик, цена известна.

Оценка берётся из :mod:`experiments.openlin_fdr` — это мера непохожести линии
на артефакт, откалиброванная по фазовой подделке. Для каждого порога известно,
сколько линий остаётся и какая доля из них мусор, поэтому карта показывается
сразу на нескольких уровнях: видно и что теряется при ужесточении, и чем
платится смягчение.

Запуск из корня репозитория (после openlin_fdr):
    python -m experiments.openlin_fdr_map --site anabar
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

from experiments.openlin_confidence import _fracs
from experiments.openlin_fdr import fdr_curve, line_features, one_pass
from experiments.openlin_null_input import _mask
from experiments.openlin_slides2 import (ACC, CROP, EDGE, GREY, INK, INK2,
                                         SLATE, SURF, crop, fat, robust)
from experiments.openlin_unet import load_site
from experiments.openlin_unsup import OPTIC, RADAR

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

#: Уровни, на которых показывается карта: доля мусора среди оставленных линий.
LEVELS = (0.05, 0.10, 0.20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--operator", default="hessian")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    src = M / f"openlin_fdr_{a.site}_hessian,old.csv"
    if not src.exists():
        src = M / f"openlin_fdr_{a.site}_{a.operator}.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    log(f"таблица линий: {src.name}, строк {len(df)}")

    # заново получаем линии выбранного оператора на настоящем входе: в таблице
    # они идут в том же порядке, в каком возвращает трассировка
    rng = np.random.default_rng(7)
    masks = {name: _mask(a.site, chans, a.density, a.operator, False,
                         np.random.default_rng(7))
             for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                                 ("оптика+радар", OPTIC + RADAR))}
    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))
    coords, _fo, _fr = _fracs(masks["оптика+радар"],
                              {"оптика": masks["оптика"],
                               "радар": masks["радар"]}, res_m)

    sub = df[(df["настоящая"] == 1)
             & (df.get("оператор", a.operator) == a.operator)]
    if len(sub) != len(coords):
        raise SystemExit(f"линий на карте {len(coords)}, в таблице {len(sub)} — "
                         "прогоны разошлись, пересчитайте openlin_fdr")
    score = sub["оценка"].to_numpy()
    tab = fdr_curve(df, df["оценка"].to_numpy())

    thresholds = []
    for lvl in LEVELS:
        ok = tab[tab["доля мусора, %"] <= 100 * lvl]
        if ok.empty:
            continue
        row = ok.iloc[0]
        thresholds.append((float(row["порог"]), int(row["линий"]),
                           float(row["доля мусора, %"])))
    log("пороги: " + "; ".join(f"{n} линий при {p:.0f}% мусора"
                               for _t, n, p in thresholds))

    r0, r1, c0, c1 = CROP
    fig = plt.figure(figsize=(7.0 * len(thresholds), 12.2), facecolor=SURF)
    gs = fig.add_gridspec(2, len(thresholds), height_ratios=[1.0, 0.92],
                          hspace=0.10, wspace=0.05, left=0.01, right=0.99,
                          top=0.93, bottom=0.07)
    for i, (t, n_lines, pct) in enumerate(thresholds):
        keep = np.zeros(valid.shape, dtype=bool)
        kept = 0
        for pts, sc in zip(coords, score):
            if sc >= t:
                keep[pts[:, 0], pts[:, 1]] = True
                kept += 1
        for row, cut in enumerate((False, True)):
            ax = fig.add_subplot(gs[row, i])
            base = crop(gray) if cut else np.ma.masked_where(~valid, gray)
            ax.imshow(base, cmap="gray", alpha=0.9)
            m = crop(keep) if cut else keep
            ax.imshow(np.ma.masked_where(~fat(m, 2 if cut else 2),
                                         np.zeros(np.shape(m))),
                      cmap=ListedColormap([ACC]))
            if not cut:
                ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                           fill=False, edgecolor=SLATE,
                                           lw=1.6, ls="--"))
                ax.set_title(f"мусора не больше {pct:.0f} %",
                             fontsize=18, color=INK, fontweight="bold", pad=10)
                ax.set_xlabel(f"{kept} линий, "
                              f"{keep.sum() * res_m / 1000:.0f} км",
                              fontsize=14, color=INK2)
            else:
                ax.set_xlabel("фрагмент 22 × 18 км", fontsize=14, color=INK2)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(SLATE if cut else EDGE)
    fig.text(0.5, 0.02, "порог выбирается по измеренной доле мусора: линии "
             "с оценкой ниже порога отброшены", ha="center", fontsize=15,
             color=INK2)
    out = OUT / f"карта_по_порогам_{a.site}.png"
    fig.savefig(out, dpi=140, facecolor=SURF)
    plt.close(fig)
    log(f"сохранено: {out}")


if __name__ == "__main__":
    main()
