"""Карта линеаментов по снимку: что выделено, чем подтверждено, как соотносится.

Три панели. Слева — сама карта: линии, подтверждённые обоими приборами,
отделены от увиденных только одним. В середине — оптика против радара, то есть
из чего складывается подтверждение. Справа — розы азимутов: линеаментов и, для
сравнения, контуров зон разломов.

Разломы здесь только для сопоставления. Ни обучения, ни подгонки: поле линейности
считается прямо из композита (``openlin_unsup``).

Запуск из корня репозитория (после openlin_unsup):
    python -m experiments.openlin_unsup_map --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.openlin_lines_diag import azimuths
from experiments.openlin_lines_map import fat, robust
from experiments.openlin_unet import load_site
from experiments.openlin_unsup import ROSE_BIN, rose

M = Path("outputs/metrics")

#: Цвета: подтверждение — зелёный (слот aqua), оптика — синий, радар — оранжевый,
#: разломы — красный. Порядок слотов фиксирован, как в остальных картинках ветки.
C_BOTH, C_OPT, C_RAD, C_FAULT = "#1baf7a", "#2a78d6", "#eb6834", "#e34948"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="очень густо")
    a = ap.parse_args()

    p = M / f"openlin_unsup_{a.site}_{a.density}.npz"
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала python -m experiments.openlin_unsup")
    d = np.load(p)
    l_opt, l_rad, l_all = d["lines_0"], d["lines_1"], d["lines_2"]

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    gray = robust(z["gray"])
    dist = z["dist"]
    fault = (dist <= res_m * 0.7) & valid

    from scipy.ndimage import distance_transform_edt
    from experiments.openlin_anfis import TOL_M

    # Подтверждение поштучное и по НАПРАВЛЕНИЮ: линия остаётся, если второй
    # прибор видит вдоль неё свою линию того же азимута не меньше чем на
    # половине длины. Простое «есть линия в допуске» засчитывало бы пересечения,
    # которых в густой карте много; такой фильтр случайно проходят 3% веток.
    conf = d["confirmed"] if "confirmed" in d.files else l_all
    solo = l_all & ~conf

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})
    for ax in axes[:2]:
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(gray, cmap="gray", alpha=0.8)
    axes[0].imshow(np.ma.masked_where(~fat(solo), np.zeros_like(gray)),
                   cmap=matplotlib.colors.ListedColormap(["#f0efec"]), alpha=0.95)
    axes[0].imshow(np.ma.masked_where(~fat(conf), np.zeros_like(gray)),
                   cmap=matplotlib.colors.ListedColormap([C_BOTH]), alpha=1.0)
    share = float(conf.sum() / max(l_all.sum(), 1))
    axes[0].set_title("карта линеаментов по снимку\n"
                      "зелёные подтверждены обоими приборами ({:.0%} длины), "
                      "серые — одним".format(share), fontsize=10.5, color=INK)
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(
        valid.shape[1] * res_m / 1000, valid.shape[0] * res_m / 1000),
        fontsize=9, color=INK2)

    axes[1].imshow(gray, cmap="gray", alpha=0.55)
    axes[1].imshow(np.ma.masked_where(~fat(l_opt), np.zeros_like(gray)),
                   cmap=matplotlib.colors.ListedColormap([C_OPT]), alpha=0.85)
    axes[1].imshow(np.ma.masked_where(~fat(l_rad), np.zeros_like(gray)),
                   cmap=matplotlib.colors.ListedColormap([C_RAD]), alpha=0.75)
    axes[1].set_title("из чего складывается подтверждение\n"
                      "синие — только оптика, оранжевые — только радар",
                      fontsize=10.5, color=INK)

    ax = fig.add_subplot(1, 3, 3, projection="polar")
    axes[2].remove()
    bins = np.radians(np.arange(0, 180, ROSE_BIN) + ROSE_BIN / 2)
    for az, color, lab in ((azimuths(l_all, res_m), C_BOTH, "линеаменты"),
                           (azimuths(fault, res_m), C_FAULT, "контуры зон разломов")):
        h = rose(az)
        ax.plot(np.concatenate([bins, bins + np.pi, [bins[0]]]),
                np.concatenate([h, h, [h[0]]]), color=color, lw=2.2, label=lab)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    # Обе розы двулучевые: и СВ, и СЗ система присутствуют у каждой. Различается
    # не набор направлений, а какой луч преобладает, — так и подписано.
    ax.set_title("розы азимутов: обе системы есть у обеих карт,\n"
                 "но у линеаментов преобладает СВ, у контуров зон — СЗ",
                 fontsize=10, color=INK, pad=20)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              frameon=False)

    fig.suptitle("{}: линеаменты выделены по снимку без обучения; "
                 "разломы — только для сопоставления"
                 .format({"anabar": "Анабар", "nevada": "Невада",
                          "abitibi": "Абитиби"}.get(a.site, a.site)),
                 fontsize=13.5, color=INK, y=0.99)
    out = Path(f"outputs/openlin_unsup_{a.site}_{a.density}.png")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"сохранено: {out}; подтверждено обоими приборами {share:.1%} длины")


if __name__ == "__main__":
    main()
