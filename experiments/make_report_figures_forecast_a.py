# -*- coding: utf-8 -*-
"""Рисунки для части A раздела «Прогнозирование» большой колоды-отчёта.

Строит столбиковые диаграммы под размер слайда 10 x 5.625" из файлов
`outputs/metrics/`; результат — `outputs/report_figs/pa_*.png`. Размер рисунка
(дюймы) равен месту на слайде, поэтому кегль в точках на картинке равен кеглю
на экране; везде не меньше 8 пт. Источник чисел — в докстринге функции.

Запуск: python -X utf8 -m experiments.make_report_figures_forecast_a
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
OUT = ROOT / "outputs" / "report_figs"

NAVY, GREY, LGREY = "#2563EB", "#64748B", "#CBD5E1"
GREEN, RED = "#0D9488", "#A3122B"
H = 2.0
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans",
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 8})


def _csv(name):
    return pd.read_csv(MET / name, encoding="utf-8-sig")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)


def fig_probes(w=3.9, h=H):
    """П-02. Доля проб в верхних 10% площади — metrics/cores_anatomy.log, раздел 5."""
    labels = ["Формула,\nвсе пробы", "Формула,\nнесмещ.",
              "Прогноз,\nвсе пробы", "Прогноз,\nнесмещ."]
    vals = [21, 24, 26, 29]
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    bars = ax.bar(labels, vals, color=[GREY, GREY, NAVY, NAVY], width=0.62)
    ax.axhline(10, color=RED, ls="--", lw=1.2, label="случайная карта: 10%")
    ax.legend(loc="upper left", frameon=False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v}%", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, 38)
    ax.set_ylabel("доля проб, %")
    _save(fig, "pa_probes.png")


def fig_neigh(w=4.1, h=H):
    """П-07. Совпадение верхних 10%: сырые 37 -> с окрестностью 111 —
    metrics/supervised_sweep.csv, столбец совпадение_верх_10 (до поправки на наклон)."""
    s = _csv("supervised_sweep.csv")
    a = s[s["вход"] == "сырые 37"].set_index("метод")["совпадение_верх_10"] * 100
    b = s[s["вход"] == "с окрестностью 111"].set_index("метод")["совпадение_верх_10"] * 100
    a = a.drop("ТОЛЬКО география (планка)")
    names = {"логистическая (аналог MaxEnt)": "логистическая",
             "ближайшие соседи (k=25)": "ближ. соседи",
             "опорные векторы, ядро RBF*": "метод опор. векторов",
             "случайный лес": "случайный лес",
             "экстремально рандомиз. деревья": "рандом. деревья",
             "гистограммный бустинг": "бустинг",
             "перцептрон (ANN литературы)": "перцептрон",
             "положительные-неразмеченные": "положит./неразмеч.",
             "голосование трёх сильнейших": "голосование трёх"}
    order = b.loc[a.index].sort_values().index
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    y = range(len(order))
    ax.barh([i + 0.2 for i in y], [a[k] for k in order], height=0.38,
            color=LGREY, label="сырые (37)")
    ax.barh([i - 0.2 for i in y], [b[k] for k in order], height=0.38,
            color=NAVY, label="с окрестностью (111)")
    ax.set_yticks(list(y))
    ax.set_yticklabels([names[k] for k in order])
    ax.axvline(20.4, color=RED, ls="--", lw=1, label="география 20.4%")
    ax.set_xlabel("совпадение верхних 10%, %")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    ax.set_xlim(0, 54)
    _save(fig, "pa_neigh.png")


def fig_elev(w=4.3, h=H):
    """П-10. Совпадение с поправкой на наклон — metrics/elevation_detrend.csv."""
    d = _csv("elevation_detrend.csv")
    d = d[~d["вариант"].str.startswith("обучение на всём листе")]
    short = {"как сейчас": "как сейчас (рабочая точка)",
             "без высоты": "без высоты",
             "высота -> превышение над окрестностью 10 км": "высота → превышение 10 км",
             "высота -> превышения 5/10/20 км": "высота → превышения 5/10/20 км",
             "высота -> минус региональная поверхность": "высота − регион. поверхность",
             "все признаки минус региональная поверхность": "все признаки − регион. пов.",
             "все признаки -> превышения над окрестностью 10 км": "все признаки → превышения",
             "сырые признаки + превышения 10 км": "сырые + превышения 10 км"}
    d = d.iloc[::-1]
    cols = [NAVY if v == "как сейчас" else GREY for v in d["вариант"]]
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    ax.barh([short[v] for v in d["вариант"]], d["совпадение_без_тренда"], color=cols)
    for i, v in enumerate(d["совпадение_без_тренда"]):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0.10, color=RED, ls="--", lw=1, label="случайно 0.10")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 0.58)
    ax.set_xlabel("совпадение с поправкой на наклон")
    _save(fig, "pa_elev.png")


def fig_exposure(w=3.3, h=H):
    """П-11. Совпадение с поправкой на наклон — metrics/spectral_exposure_control.csv
    (столбец без_тренда_сглаж; случайная маска — среднее трёх запусков)."""
    c = _csv("spectral_exposure_control.csv")
    rnd = c[c["вариант"].str.startswith("случайная маска")]["без_тренда_сглаж"].mean()
    base = c.loc[c["вариант"].str.startswith("база"), "без_тренда_сглаж"].iloc[0]
    nos = c.loc[c["вариант"].str.startswith("без спектров"), "без_тренда_сглаж"].iloc[0]
    mask = c.loc[c["вариант"].str.startswith("маска"), "без_тренда_сглаж"].iloc[0]
    labels = ["все\nячейки", "без\nспектров", "случайная\nмаска", "маска\nобнажён."]
    vals = [base, nos, rnd, mask]
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    bars = ax.bar(labels, vals, color=[LGREY, GREY, GREY, NAVY], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("совпадение с поправкой")
    _save(fig, "pa_exposure.png")


def fig_factor(w=4.1, h=H):
    """П-13. ROC восстановимости факторов — metrics/factor_from_remote.csv,
    столбец roc_благоприятен."""
    d = _csv("factor_from_remote.csv")
    nm = {"dist_struct": "структурное несогласие", "dist_facies": "границы фаций",
          "dist_paleo": "палеодолины", "dist_magm": "магматические (расст.)",
          "dens_magm": "магматические (плотн.)", "dist_tect2": "разломы 2-й системы",
          "dist_tect1": "разломы 1-й системы", "dens_tect": "плотность разломов"}
    d = d.sort_values("roc_благоприятен")
    cols = [NAVY if v >= 0.8 else (GREY if v >= 0.6 else LGREY) for v in d["roc_благоприятен"]]
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    ax.barh([nm[f] for f in d["фактор"]], d["roc_благоприятен"], color=cols)
    for i, v in enumerate(d["roc_благоприятен"]):
        ax.text(max(v, 0.5) + 0.015, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0.5, color=RED, ls="--", lw=1, label="случайно 0.5")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("ROC-AUC восстановления фактора")
    _save(fig, "pa_factor.png")


def fig_cascade_lift(w=4.2, h=H):
    """П-16. Кратность по 19 несмещённым точкам, лучший вариант каждого
    детектора — metrics/atypicality_v6_verdict.csv; каскад —
    atypicality_cascade_verdict.csv; критериальная карта — lift_criterial."""
    v = _csv("atypicality_v6_verdict.csv")
    v["fam"] = v["method"].str.split("_").str[0]
    best = v.groupby("fam")["lift"].max()
    nm = {"isoforest": "изолирующий лес", "gmm": "смесь гауссиан", "mahalanobis": "Махаланобис",
          "pca": "метод глав. компонент", "ocsvm": "одноклассовый SVM",
          "knn": "расстояние до соседей", "shallow": "автокодировщик",
          "lof": "локальный выброс", "rank": "ансамбль рангов"}
    nm["ocsvm"] = "одноклассовые опор. векторы"
    casc = _csv("atypicality_cascade_verdict.csv")["lift"].max()
    crit = v["lift_criterial"].iloc[0]
    items = [(nm[k], x) for k, x in best.items()] + [("каскад по группам", casc)]
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    cols = [NAVY if x >= crit - 1e-9 else GREY for _, x in items]
    ax.barh([n for n, _ in items], [x for _, x in items], color=cols)
    for i, (_, x) in enumerate(items):
        ax.text(x + 0.04, i, f"{x:.2f}", va="center", fontsize=8)
    ax.axvline(crit, color=GREEN, ls="--", lw=1.2, label=f"критер. карта {crit:.2f}")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("кратность к случайной карте")
    _save(fig, "pa_cascade_lift.png")


def fig_clusters(w=4.7, h=H):
    """П-15. Число признаков в каждом из 20 кластеров — metrics/feature_clusters_v6.csv."""
    d = _csv("feature_clusters_v6.csv")
    n = d.groupby("cluster").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    bars = ax.bar([str(i) for i in n.index], n.values, color=NAVY, width=0.7)
    for b, v in zip(bars, n.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=8)
    ax.set_xlabel("номер кластера (по убыванию размера)")
    ax.set_ylabel("признаков")
    ax.set_ylim(0, n.max() + 3)
    _save(fig, "pa_clusters.png")


if __name__ == "__main__":
    for f in (fig_probes, fig_neigh, fig_elev, fig_exposure, fig_factor,
              fig_cascade_lift, fig_clusters):
        f()
    print("готово:", OUT)
