"""Что есть после каждого шага: картинки для проверки глазами.

ЗАЧЕМ. Из карты уходит часть линеаментов, и надо понять, на каком шаге они
теряются. Для этого каждый шаг показывается как самостоятельная картинка на
одних и тех же данных: сначала вся площадка, затем тот же фрагмент крупным
планом. Никакой разметки поверх — смотрим на то, что шаг реально произвёл.

Шаги:

* ``--step 0`` — исходные каналы: оптика, NDVI, радар;
* ``--step 1`` — величины окрестности (вытянутость и гребень на двух масштабах);
* ``--step 2`` — сжатое описание: первые компоненты;
* ``--step 3`` — поле силы границы;
* ``--step 4`` — прослеженные линии (до отбора);
* ``--step 5`` — линии после отбора.

Запуск из корня репозитория:
    python -m experiments.openlin_steps_check --step 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from experiments.openlin_lines import _branches
from experiments.openlin_neigh import _one_channel
from experiments.openlin_slides2 import (ACC, CROP, EDGE, INK, INK2, RAMP,
                                         SLATE, SURF, crop, fat, robust)
from experiments.openlin_unet import load_site

OUT = Path("outputs/steps")
M = Path("outputs/metrics")

DENS = "предельно"

#: Крупный фрагмент для второй строки: тот же, что во всех схемах метода.
#: Размер панели подобран так, чтобы ячейка 100 м оставалась различимой.
PANEL_IN = 7.2


def _panels(site: str):
    z = load_site(site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    return z, valid, res_m


def show(name: str, panels: list[tuple[np.ndarray, str, str]], site: str,
         res_m: float, valid: np.ndarray) -> None:
    """Две строки: вся площадка и тот же фрагмент крупным планом."""
    r0, r1, c0, c1 = CROP
    n = len(panels)
    fig = plt.figure(figsize=(PANEL_IN * n, PANEL_IN * 1.72), facecolor=SURF)
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 0.92], hspace=0.10,
                          wspace=0.05, left=0.01, right=0.99, top=0.94,
                          bottom=0.04)
    for i, (img, title, sub) in enumerate(panels):
        full = np.ma.masked_where(~valid, img) if img.dtype != bool else img
        ax = fig.add_subplot(gs[0, i])
        if img.dtype == bool:
            ax.imshow(np.ma.masked_where(~valid, np.zeros(img.shape)),
                      cmap=ListedColormap(["#EFEFEC"]))
            ax.imshow(np.ma.masked_where(~fat(img, 3), np.zeros(img.shape)),
                      cmap=ListedColormap([ACC]))
        else:
            ax.imshow(full, cmap="gray" if "снимок" in title else RAMP)
        ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                   edgecolor=SLATE, lw=1.6, ls="--"))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.set_title(title, fontsize=19, color=INK, fontweight="bold", pad=10)
        if i == 0:
            ax.set_xlabel("вся площадка, 86 × 77 км", fontsize=14, color=INK2)

        ax = fig.add_subplot(gs[1, i])
        if img.dtype == bool:
            ax.imshow(np.zeros(crop(img).shape), cmap=ListedColormap(["#EFEFEC"]))
            ax.imshow(np.ma.masked_where(~fat(crop(img), 2),
                                         np.zeros(crop(img).shape)),
                      cmap=ListedColormap([ACC]))
        else:
            ax.imshow(crop(img), cmap="gray" if "снимок" in title else RAMP)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(SLATE)
            s.set_linewidth(1.6)
        ax.set_xlabel(sub, fontsize=14, color=INK2)
    fig.savefig(OUT / name, dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"{OUT / name}")


def step0(site: str) -> None:
    z, valid, res_m = _panels(site)
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))
    vv = robust(np.nan_to_num(z["vv"], nan=float(np.nanmedian(z["vv"]))))
    show("step0_вход.png",
         [(gray, "снимок: яркость оптики", "фрагмент 22 × 18 км"),
          (vv, "радар VV", "фрагмент 22 × 18 км")], site, res_m, valid)


def step1(site: str) -> None:
    z, valid, res_m = _panels(site)
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    f500 = _one_channel(gray, valid, 500.0 / res_m)
    f2k = _one_channel(gray, valid, 2000.0 / res_m)
    show("step1_признаки.png",
         [(robust(gray), "снимок как есть", "для сравнения"),
          (f500["coh"], "вытянутость, 500 м", "тёмное — есть направление"),
          (robust(f500["rdg"]), "гребень / борозда, 500 м",
           "светлое — гребень, тёмное — борозда")],
         site, res_m, valid)
    show("step1_масштаб2км.png",
         [(robust(gray), "снимок как есть", "для сравнения"),
          (f2k["coh"], "вытянутость, 2 км", "тот же признак на крупном окне"),
          (robust(f2k["rdg"]), "гребень / борозда, 2 км",
           "тот же признак на крупном окне")],
         site, res_m, valid)


def step2(site: str) -> None:
    from experiments.openlin_gate import components
    from experiments.openlin_unsup import OPTIC, RADAR

    _z, valid, res_m, comp = components(site, OPTIC + RADAR)
    gray = np.nan_to_num(_z["gray"], nan=float(np.nanmedian(_z["gray"])))
    show("step2_компоненты.png",
         [(robust(gray), "снимок как есть", "для сравнения"),
          (comp[0], "компонента 1", "смесь 160 признаков"),
          (comp[1], "компонента 2", "смесь 160 признаков"),
          (comp[2], "компонента 3", "смесь 160 признаков")],
         site, res_m, valid)


def step3(site: str) -> None:
    z, valid, res_m = _panels(site)
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    d = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    field = np.nan_to_num(d["field"], nan=0.0)
    show("step3_поле.png",
         [(robust(gray), "снимок как есть", "для сравнения"),
          (robust(field, 2, 99.5), "поле силы границы",
           "тёмное — участки по сторонам различаются")],
         site, res_m, valid)


def step3_overlay(site: str) -> None:
    """Поле силы границы поверх самого снимка: ложится ли оно на местность."""
    z, valid, res_m = _panels(site)
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))
    d = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    field = np.nan_to_num(d["field"], nan=0.0)
    top = field >= np.quantile(field[valid], 0.80)      # верхние 20 % площади
    r0, r1, c0, c1 = CROP

    fig = plt.figure(figsize=(PANEL_IN * 2, PANEL_IN * 1.72), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.10,
                          wspace=0.05, left=0.01, right=0.99, top=0.94,
                          bottom=0.04)
    for row, cut in enumerate((False, True)):
        for col, (title, sub) in enumerate((
                ("поле поверх снимка", "полупрозрачно, вся шкала"),
                ("только сильные значения поля",
                 "верхние 20 % площади"))):
            g = crop(gray) if cut else np.ma.masked_where(~valid, gray)
            f = crop(field) if cut else np.ma.masked_where(~valid, field)
            t = crop(top) if cut else (top & valid)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(g, cmap="gray")
            if col == 0:
                ax.imshow(f, cmap=RAMP, alpha=0.55)
            else:
                ax.imshow(np.ma.masked_where(~t, np.zeros(np.shape(t))),
                          cmap=ListedColormap([ACC]), alpha=0.55)
            if not cut:
                ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                           fill=False, edgecolor=SLATE,
                                           lw=1.6, ls="--"))
                ax.set_title(title, fontsize=19, color=INK, fontweight="bold",
                             pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(SLATE if cut else EDGE)
            ax.set_xlabel(("фрагмент 22 × 18 км: " if cut else
                           "вся площадка: ") + sub, fontsize=14, color=INK2)
    fig.savefig(OUT / "step3_наложение.png", dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"{OUT / 'step3_наложение.png'}")


def step4(site: str) -> None:
    z, valid, res_m = _panels(site)
    d = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    la = d["lines_2"]
    n = len([p for p in _branches(la, res_m) if len(p) >= 3])
    show("step4_линии.png",
         [(la, f"прослежено линий: {n}",
           f"{la.sum() * res_m / 1000:.0f} км, до отбора")],
         site, res_m, valid)


def step5(site: str) -> None:
    z, valid, res_m = _panels(site)
    d = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    keep = d["confirmed"]
    n = len([p for p in _branches(keep, res_m) if len(p) >= 3])
    show("step5_отбор.png",
         [(keep, f"после отбора: {n} линий",
           f"{keep.sum() * res_m / 1000:.0f} км")],
         site, res_m, valid)




def compare(site: str) -> None:
    """Старое поле и новое рядом, на одном и том же фрагменте."""
    z, valid, res_m = _panels(site)
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))
    old = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    import os
    tag = os.environ.get("CMP_TAG", "v3")
    new = np.load(M / f"openlin_unsup_{site}_{DENS}_{tag}.npz")
    show("compare_поле.png",
         [(robust(gray), "снимок как есть", "для сравнения"),
          (robust(np.nan_to_num(old["field"], nan=0.0), 2, 99.5),
           "поле: как было", "оператор ступени, без нормировки"),
          (robust(np.nan_to_num(new["field"], nan=0.0), 2, 99.5),
           "поле: исправленное", "линия + ступень, с нормировкой")],
         site, res_m, valid)
    show("compare_линии.png",
         [(old["lines_2"], "линии: как было",
           f"{old['lines_2'].sum() * res_m / 1000:.0f} км"),
          (new["lines_2"], "линии: исправленные",
           f"{new['lines_2'].sum() * res_m / 1000:.0f} км")],
         site, res_m, valid)

def lines_on_map(site: str) -> None:
    """Линии поверх снимка: прежние и новые, вся площадка и фрагмент."""
    import os

    z, valid, res_m = _panels(site)
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))
    tag = os.environ.get("CMP_TAG", "v3")
    old = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    new = np.load(M / f"openlin_unsup_{site}_{DENS}_{tag}.npz")
    r0, r1, c0, c1 = CROP

    panels = [(old["lines_2"], "линии: как было",
               f"{old['lines_2'].sum() * res_m / 1000:.0f} км"),
              (new["lines_2"], "линии: исправленный оператор",
               f"{new['lines_2'].sum() * res_m / 1000:.0f} км")]
    fig = plt.figure(figsize=(PANEL_IN * 2, PANEL_IN * 1.72), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.10,
                          wspace=0.05, left=0.01, right=0.99, top=0.94,
                          bottom=0.04)
    for i, (mask, title, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.ma.masked_where(~valid, gray), cmap="gray", alpha=0.9)
        ax.imshow(np.ma.masked_where(~fat(mask, 3), np.zeros(mask.shape)),
                  cmap=ListedColormap([ACC]))
        ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                                   edgecolor=SLATE, lw=1.6, ls="--"))
        ax.set_title(title, fontsize=19, color=INK, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.set_xlabel(f"вся площадка, 86 × 77 км — {sub}", fontsize=14,
                      color=INK2)

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(crop(gray), cmap="gray", alpha=0.9)
        m = crop(mask)
        ax.imshow(np.ma.masked_where(~fat(m, 2), np.zeros(m.shape)),
                  cmap=ListedColormap([ACC]))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(SLATE)
            s.set_linewidth(1.6)
        ax.set_xlabel("фрагмент 22 × 18 км", fontsize=14, color=INK2)
    fig.savefig(OUT / "compare_линии_на_карте.png", dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"{OUT / 'compare_линии_на_карте.png'}")


STEPS = {0: step0, 1: step1, 2: step2, 3: step3, 4: step4, 5: step5,
         31: step3_overlay, 9: compare, 10: lines_on_map}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--step", type=int, required=True,
                choices=sorted(STEPS),
                help="31 — поле третьего шага поверх снимка")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    STEPS[a.step](a.site)


if __name__ == "__main__":
    main()
