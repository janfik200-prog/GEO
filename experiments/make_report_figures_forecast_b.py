# -*- coding: utf-8 -*-
"""Рисунки части B раздела «Прогнозирование» большой колоды-отчёта.

ЗАЧЕМ. Готовые PNG из `outputs/` сделаны под холст 13.3": подписи внутри
панелей на холсте 10" нечитаемы. Здесь собираются рисунки под размер слайда
(физический размер рисунка = размер на слайде, кегль 8-11 пт):
  * столбики из файлов результатов (ветки 1-5, потолок проверки);
  * карты — обрезка нужных панелей готовых PNG с собственными крупными
    заголовками (сами карты не пересчитываются).

ИСТОЧНИКИ ЧИСЕЛ в столбиках:
  fb_branch1  — outputs/metrics/branch1_detrended_agreement.csv
                (карта boostGCN_58ms: none / whole2 / whole10);
  fb_branch23 — outputs/metrics/branch4_aster_maps.csv, строки «58 сырые»
                (круглое 2 км; вдоль северо-западное 5 км, 4:1);
  fb_branch4  — outputs/metrics/branch4_aster_maps.csv (ядро 5 км, 4:1);
  fb_branch5  — outputs/metrics/branch5_align.log (среднее по 8 полосам);
  fb_ceiling  — outputs/metrics/full_sheet_map.csv и ceiling_decomposition.csv.
ВЫХОД: outputs/report_figs/fb_*.png
Запуск из корня: python -X utf8 -m experiments.make_report_figures_forecast_b
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "report_figs"
MET = OUT / "metrics"

BLUE = "#1E3A8A"
LBLUE = "#93C5FD"
GREY = "#475569"
GREEN = "#0D9488"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=220, facecolor="white")
    plt.close(fig)
    print("OK", name)


def _csv(name):
    return pd.read_csv(MET / name, encoding="utf-8-sig")


def _bars(ax, labels, series, colors, names, ymax, fmt="{:.0f}", rand=10.0):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for i, (vals, col, nm) in enumerate(zip(series, colors, names)):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=col,
                   label=nm)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + ymax * 0.015,
                    fmt.format(v), ha="center", va="bottom", fontsize=9,
                    color="#0F172A", fontweight="bold")
    if rand is not None:
        ax.axhline(rand, color="#DC2626", lw=1.0, ls="--",
                   label="случайный уровень 10%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, ymax)


def fig_branch1():
    d = _csv("branch1_detrended_agreement.csv")
    d = d[d["карта"] == "boostGCN_58ms"].set_index("как")
    keys = ["none", "whole2", "whole10"]
    raw = [d.loc[k, "agr"] * 100 for k in keys]
    adj = [d.loc[k, "agr_detrend"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(4.4, 2.35))
    _bars(ax, ["без\nсглаживания", "сглаживание\n2 км", "сглаживание\n10 км"],
          [raw, adj], [LBLUE, BLUE],
          ["совпадение", "после поправки на наклон"], 80)
    ax.set_ylabel("совпадение верхних 10%, %", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    _save(fig, "fb_branch1.png")


def fig_branch23():
    d = _csv("branch4_aster_maps.csv")
    d = d[d["вход"] == "58 сырые"].set_index("постобработка")
    keys = ["круглое 2 км", "вдоль северо-западное 5 км, 4:1"]
    raw = [d.loc[k, "совпадение_верх_10"] * 100 for k in keys]
    adj = [d.loc[k, "совпадение_без_тренда"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(4.2, 2.35))
    _bars(ax, ["ветка 2:\nкруглое 2 км", "ветка 3:\nядро 5 × 1.25 км"],
          [raw, adj], [LBLUE, BLUE],
          ["совпадение", "после поправки на наклон"], 56)
    ax.set_ylabel("совпадение верхних 10%, %", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    _save(fig, "fb_branch23.png")


def fig_branch4():
    d = _csv("branch4_aster_maps.csv")
    d = d[d["постобработка"] == "вдоль северо-западное 5 км, 4:1"]
    d = d.set_index("вход")
    keys = ["58 сырые", "37 перенос", "58 + контур ASTER (контроль)",
            "58 + индексы ASTER (6)", "58 + ASTER (24)"]
    names = ["58 сырых признаков", "набор переноса 37", "58 + признак «контур»",
             "58 + 6 индексов ASTER", "58 + 24 канала ASTER"]
    vals = [d.loc[k, "совпадение_без_тренда"] * 100 for k in keys]
    cols = [BLUE, BLUE, GREY, LBLUE, LBLUE]
    fig, ax = plt.subplots(figsize=(4.3, 2.35))
    y = np.arange(len(keys))[::-1]
    ax.barh(y, vals, color=cols, height=0.65)
    for yy, v in zip(y, vals):
        ax.text(v + 0.6, yy, "%.1f" % v, va="center", fontsize=9,
                fontweight="bold", color="#0F172A")
    ax.axvline(10, color="#DC2626", lw=1.0, ls="--")
    ax.text(10.4, y[0] + 0.55, "случайный 10%", color="#DC2626", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 38)
    ax.set_xlabel("после поправки на наклон, %", fontsize=9)
    fig.tight_layout()
    _save(fig, "fb_branch4.png")


def fig_branch5():
    vals = [0.7488, 0.7633, 0.7623]
    names = ["сборка\nлиста", "широкая\nсборка", "широкая,\nсогласована\nпо квантилям"]
    fig, ax = plt.subplots(figsize=(3.9, 2.35))
    b = ax.bar(range(3), vals, 0.6, color=[BLUE, LBLUE, LBLUE])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.01, "%.3f" % v,
                ha="center", fontsize=9, fontweight="bold", color="#0F172A")
    ax.axhline(0.5, color="#DC2626", lw=1.0, ls="--", label="случайно 0.5")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_ylim(0.4, 0.95)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("ROC-AUC по 8 полосам\n(шкала от 0.4)", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fb_branch5.png")


def fig_ceiling():
    f = _csv("full_sheet_map.csv")
    row = f[(f["цель"] == "порог верхних 10%") & (f["ядро_км"] == "5 x 1.25")]
    full = row[row["сборка"] == "обучение на всём листе"]["совпадение_без_тренда"].iloc[0]
    blind = row[row["сборка"].str.contains("полосам")]["совпадение_без_тренда"].iloc[0]
    c = _csv("ceiling_decomposition.csv")
    ceil = c[c["вариант"].str.contains("потолок")]["совпадение_без_тренда"].iloc[0]
    vals = [full, blind, ceil]
    names = ["итоговая карта\n(модель видела\nвсе метки)", "слепая проверка\nпо полосам\nс разрывом 15 км",
             "потолок\nпроверки"]
    fig, ax = plt.subplots(figsize=(3.9, 2.35))
    b = ax.bar(range(3), vals, 0.62, color=[LBLUE, BLUE, GREEN])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.012, "%.3f" % v,
                ha="center", fontsize=10, fontweight="bold", color="#0F172A")
    ax.axhline(0.10, color="#DC2626", lw=1.0, ls="--", label="случайный уровень 0.10")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("совпадение с поправкой на наклон", fontsize=8)
    fig.tight_layout()
    _save(fig, "fb_ceiling.png")


# --- карты: обрезка панелей готовых PNG --------------------------------------
def crop(rel, factor, box):
    """box — координаты в масштабе показа (как при просмотре), factor — пересчёт."""
    im = Image.open(OUT / rel).convert("RGB")
    x0, y0, x1, y1 = [int(round(v * factor)) for v in box]
    return np.asarray(im.crop((x0, y0, x1, y1)))


def compose(name, panels, figsize, fs=9.5):
    ratios = [a.shape[1] / a.shape[0] for a, _ in panels]
    fig, axs = plt.subplots(1, len(panels), figsize=figsize,
                            gridspec_kw={"width_ratios": ratios})
    axs = np.atleast_1d(axs)
    for ax, (arr, title) in zip(axs, panels):
        ax.imshow(arr)
        ax.axis("off")
        ax.set_title(title, fontsize=fs, fontweight="bold", color="#0F172A")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.86, wspace=0.04)
    _save(fig, name)


def maps():
    r = "07_перенос_на_смежные/analog_search_v2.png"
    compose("fb_analog.png", [
        (crop(r, 1.61, (95, 70, 535, 930)), "все признаки\n(обнажённые ячейки)"),
        (crop(r, 1.61, (722, 70, 1170, 930)), "рельеф и\nлинеаменты")],
        (3.3, 2.4), 9)
    r = "02_критериальный_метод/cores_anatomy.png"
    compose("fb_cores.png", [
        (crop(r, 1.30, (160, 60, 830, 1385)), "критериальная"),
        (crop(r, 1.30, (1165, 60, 1835, 1385)), "наш прогноз")],
        (3.0, 2.4), 9.5)
    r = "07_перенос_на_смежные/transfer_map_ядро5км.png"
    compose("fb_transfer.png", [
        (crop(r, 2.25, (5, 176, 440, 566)), "классы перспективности"),
        (crop(r, 2.25, (520, 157, 975, 566)), "перенос и формула")],
        (5.6, 2.6), 10)
    r2 = "06_прогнозные_карты/full_sheet_map_ядро5км.png"
    compose("fb_worknow.png", [
        (crop(r2, 2.15, (555, 157, 945, 585)), "лист"),
        (crop(r, 2.25, (5, 176, 440, 566)), "перенос на смежную площадь")],
        (4.6, 2.5), 10)
    r = "07_перенос_на_смежные/forecast_vs_criterial_wide.png"
    compose("fb_wide.png", [
        (crop(r, 1.22, (70, 105, 825, 1370)), "наш прогноз"),
        (crop(r, 1.22, (1090, 105, 1830, 1370)), "формула")],
        (3.5, 2.5), 10)
    r = "02_критериальный_метод/criterial_reconstruct_multisheet.png"
    compose("fb_multisheet.png", [
        (crop(r, 1.67, (80, 55, 560, 975)), "наше"),
        (crop(r, 1.67, (765, 55, 1230, 950)), "организации"),
        (crop(r, 1.67, (1435, 55, 1900, 950)), "верхние 10%")],
        (4.5, 2.4), 9)


if __name__ == "__main__":
    fig_branch1()
    fig_branch23()
    fig_branch4()
    fig_branch5()
    fig_ceiling()
    maps()
