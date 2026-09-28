"""Итоговая карта: объединение вариантов, порог по измеренной доле мусора.

Линии берутся из нескольких прогонов (оператор x способ сжатия), каждой
присвоена оценка из :mod:`experiments.openlin_fdr`, и на карту попадают те, у
кого оценка выше порога. Порог выбирается не на глаз, а по доле мусора,
измеренной на фазовой подделке.

Маски линий заново пересчитываются (в таблице хранятся только признаки), и
сохраняются, чтобы карту можно было перерисовать без повторного счёта.

Запуск из корня репозитория (после openlin_fdr):
    python -m experiments.openlin_fdr_map2 --site anabar --variants hessian/pca,old/pca
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from experiments.openlin_confidence import _fracs
from experiments.openlin_fdr import fdr_curve
from experiments.openlin_null_input import _mask
from experiments.openlin_slides2 import (ACC, CROP, EDGE, INK, INK2, SLATE,
                                         SURF, crop, fat, robust)
from experiments.openlin_unet import load_site
from experiments.openlin_unsup import OPTIC, RADAR

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

#: Уровни доли мусора, на которых показывается карта.
LEVELS = (0.05, 0.10, 0.20)


def variant_lines(site: str, density: str, op: str, compress: str, log):
    """Координаты веток одного варианта на настоящем входе."""
    cache = M / f"_lines_{site}_{op}_{compress}.npz"
    if cache.exists():
        z = np.load(cache)
        return [z[k] for k in z.files]
    masks = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        masks[name] = _mask(site, chans, density, op, False,
                            np.random.default_rng(7), compress)
    res_m = float(load_site(site)["res_m"])
    coords, _fo, _fr = _fracs(masks["оптика+радар"],
                              {"оптика": masks["оптика"],
                               "радар": masks["радар"]}, res_m)
    np.savez_compressed(cache, **{f"l{i}": p for i, p in enumerate(coords)})
    return coords


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--variants", default="hessian/pca,old/pca")
    ap.add_argument("--table", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    src = Path(a.table) if a.table else M / f"openlin_fdr_{a.site}_hessian,old.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    wanted = [v for v in a.variants.split(",") if v]
    df = df[df["оператор"].isin(wanted) | (df["оператор"].isin(wanted))]
    log(f"таблица {src.name}: {len(df)} строк, варианты {', '.join(wanted)}")

    tab = fdr_curve(df, df["оценка"].to_numpy())
    thresholds = []
    for lvl in LEVELS:
        ok = tab[tab["доля мусора, %"] <= 100 * lvl]
        if not ok.empty:
            r = ok.iloc[0]
            thresholds.append((float(r["порог"]), float(r["доля мусора, %"])))

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["real_res_m"]) if "real_res_m" in z else float(z["res_m"])
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))

    per_variant = {}
    for v in wanted:
        op, compress = v.split("/")
        t0 = time.time()
        coords = variant_lines(a.site, a.density, op, compress, log)
        sub = df[(df["оператор"] == v) & (df["настоящая"] == 1)]
        if len(sub) != len(coords):
            log(f"  {v}: линий {len(coords)}, в таблице {len(sub)} — пропускаю")
            continue
        per_variant[v] = (coords, sub["оценка"].to_numpy())
        log(f"  {v}: {len(coords)} линий ({time.time() - t0:.0f} с)")

    r0, r1, c0, c1 = CROP
    fig = plt.figure(figsize=(7.0 * len(thresholds), 12.2), facecolor=SURF)
    gs = fig.add_gridspec(2, len(thresholds), height_ratios=[1.0, 0.92],
                          hspace=0.10, wspace=0.05, left=0.01, right=0.99,
                          top=0.93, bottom=0.07)
    for i, (t, pct) in enumerate(thresholds):
        keep = np.zeros(valid.shape, dtype=bool)
        kept = 0
        for coords, score in per_variant.values():
            for pts, sc in zip(coords, score):
                if sc >= t:
                    keep[pts[:, 0], pts[:, 1]] = True
                    kept += 1
        for row, cut in enumerate((False, True)):
            ax = fig.add_subplot(gs[row, i])
            ax.imshow(crop(gray) if cut else np.ma.masked_where(~valid, gray),
                      cmap="gray", alpha=0.9)
            m = crop(keep) if cut else keep
            ax.imshow(np.ma.masked_where(~fat(m, 2), np.zeros(np.shape(m))),
                      cmap=ListedColormap([ACC]))
            if not cut:
                ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                           fill=False, edgecolor=SLATE,
                                           lw=1.6, ls="--"))
                ax.set_title(f"мусора не больше {pct:.0f} %", fontsize=18,
                             color=INK, fontweight="bold", pad=10)
                ax.set_xlabel(f"{kept} линий, "
                              f"{keep.sum() * res_m / 1000:.0f} км",
                              fontsize=14, color=INK2)
            else:
                ax.set_xlabel("фрагмент 22 × 18 км", fontsize=14, color=INK2)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(SLATE if cut else EDGE)
    fig.text(0.5, 0.02, "объединение: " + ", ".join(per_variant)
             + "; порог по доле мусора, измеренной на фазовой подделке",
             ha="center", fontsize=14, color=INK2)
    out = Path(a.out) if a.out else OUT / f"итог_{a.site}.png"
    fig.savefig(out, dpi=140, facecolor=SURF)
    plt.close(fig)
    log(f"сохранено: {out}")


if __name__ == "__main__":
    main()
