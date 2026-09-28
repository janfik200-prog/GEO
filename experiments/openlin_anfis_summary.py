"""Итог по ANFIS одной картинкой: что померено за всю ветку табличной постановки.

Четыре панели, каждая отвечает на свой вопрос:

1. Даёт ли что-нибудь сам нечёткий вывод — ANFIS против логрегрессии и бустинга
   НА ТЕХ ЖЕ строках и том же разбиении, отдельно на 10 поячеечных признаках и на
   55 с окрестностью.
2. Сколько стоит разбиение по исходному рецепту — случайное 3/4-1/4 против
   блочного с буфером 2 км, парой на каждый прогон.
3. Что получается при переносе на другую площадку.
4. Что ANFIS выдаёт такого, чего не даёт логрегрессия, — центры правил.

Запуск из корня репозитория:
    python -m experiments.openlin_anfis_summary
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = Path("outputs/metrics")
OUT = Path("outputs/openlin_anfis_summary.png")

#: Палитра — первые три слота категориальной палитры справочника dataviz,
#: взятые дословно: порядок фиксирован (модель всегда одного цвета на всех
#: панелях), тройка проверена на различимость при дальтонизме во всех парах.
C_ANFIS, C_LOGREG, C_BOOST = "#2a78d6", "#eb6834", "#1baf7a"
#: Расходящаяся пара для тепловой карты правил: синий-красный с СЕРЫМ нулём.
C_NEG, C_MID, C_POS = "#2a78d6", "#f0efec", "#e34948"
SURFACE = "#fcfcfb"
INK, INK2 = "#0b0b0b", "#52514e"

MODELS = ("ANFIS", "логрегрессия", "бустинг")
COLORS = {"ANFIS": C_ANFIS, "логрегрессия": C_LOGREG, "бустинг": C_BOOST}
SHORT = {"nevada": "Невада", "abitibi": "Абитиби", "anabar": "Анабар",
         "anabar_dykes": "дайки"}
RUN_LABEL = {"nevada": "Невада, 10 призн.",
             "nevada_neigh": "Невада, 55 призн.",
             "anabar": "Анабар, 10 призн.",
             "anabar_neigh": "Анабар, 55 призн.",
             "anabar_dykes": "дайки, 10 призн.",
             "anabar_dykes_neigh": "дайки, 55 призн.",
             "nevada+abitibi_neigh": "Невада+Абитиби, 55"}


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(M.glob("openlin_anfis_*.csv")):
        if f.name.endswith("_rules.csv") or f.stem.endswith("_all"):
            continue
        d = pd.read_csv(f)
        d["прогон"] = f.stem.replace("openlin_anfis_", "")
        rows.append(d)
    if not rows:
        raise SystemExit("нет таблиц openlin_anfis_*.csv")
    d = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["прогон", "разбиение", "таргет", "модель", "площадка", "режим"])
    d["площадка"] = d["площадка"].map(SHORT).fillna(d["площадка"])
    return d


def panel_models(ax, df: pd.DataFrame) -> None:
    """Нечёткий вывод против планок на тех же строках."""
    groups = [("nevada", "hard", "10 признаков\nжёсткий"),
              ("nevada", "smooth", "10 признаков\nплавный"),
              ("nevada_neigh", "hard", "55 признаков\nжёсткий"),
              ("nevada_neigh", "smooth", "55 признаков\nплавный")]
    pos = np.arange(len(groups))
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = []
        for run, tgt, _ in groups:
            g = df[(df["прогон"] == run) & (df["разбиение"] == "block")
                   & (df["режим"] == "проверка") & (df["таргет"] == tgt)
                   & (df["модель"] == m)]
            vals.append(float(g["lift"].iloc[0]) if len(g) else np.nan)
        # 2 px просвета между соседними столбцами: ширина чуть меньше шага
        ax.bar(pos + (i - 1) * w, vals, w * 0.92, color=COLORS[m], label=m,
               linewidth=0)
        for x, v in zip(pos + (i - 1) * w, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK2)
    # Подпись пунктира вынесена в заголовок: на уровне 1.0 везде стоят столбцы,
    # и любая надпись там легла бы поверх данных.
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels([g[2] for g in groups], fontsize=8.5, color=INK2)
    ax.set_ylabel("lift@10% на блочной проверке", fontsize=9, color=INK2)
    ax.set_ylim(0, 2.75)
    ax.set_title("1. Что даёт именно нечёткий вывод\n"
                 "одни и те же строки и разбиение (Невада); "
                 "пунктир — случайная карта", fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, ncol=3, loc="upper left", frameon=False)


def panel_leak(ax, df: pd.DataFrame) -> None:
    """Цена утечки: случайное 3/4-1/4 против блочного, только ANFIS."""
    a = df[(df["модель"] == "ANFIS") & (df["режим"] == "проверка")]
    piv = a.pivot_table(index=["прогон", "таргет"], columns="разбиение",
                        values="lift").dropna()
    piv["ratio"] = piv["random"] / piv["block"]
    piv = piv.sort_values("ratio")
    y = np.arange(len(piv))
    for yi, (blk, rnd) in enumerate(zip(piv["block"], piv["random"])):
        ax.plot([blk, rnd], [yi, yi], color="#d8d7d3", lw=2, zorder=1)
    ax.scatter(piv["block"], y, s=46, color=C_ANFIS, zorder=3,
               label="блочно, буфер 2 км (честно)")
    ax.scatter(piv["random"], y, s=46, color=C_POS, zorder=3,
               label="случайно 3/4 — 1/4 (рецепт)")
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{RUN_LABEL.get(r, r)} · {t}" for r, t in piv.index],
                       fontsize=8, color=INK2)
    for yi, r in enumerate(piv["ratio"]):
        ax.text(3.55, yi, f"×{r:.1f}", fontsize=8, va="center", color=INK2)
    ax.set_xlim(0.2, 3.9)
    ax.set_xlabel("lift@10% на проверке", fontsize=9, color=INK2)
    ax.set_title("2. Цена утечки: одно обучение, два разбиения\n"
                 "справа — во сколько раз случайное разбиение завышает",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, loc="lower left", frameon=False)


def panel_transfer(ax, df: pd.DataFrame) -> None:
    """Перенос ANFIS на площадку, которую модель не видела."""
    t = df[(df["модель"] == "ANFIS") & (df["разбиение"] == "block")
           & (df["режим"] == "перенос")].copy()
    t["подпись"] = [f"{RUN_LABEL.get(r, r)} → {s}"
                    for r, s in zip(t["прогон"], t["площадка"])]
    t = t.sort_values("lift")
    y = np.arange(len(t))
    ax.barh(y, t["lift"], 0.72, color=C_ANFIS, linewidth=0)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    for yi, (v, p) in enumerate(zip(t["lift"], t["p"])):
        ax.text(v + 0.02, yi, f"{v:.2f}   p={p:.3f}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(t["подпись"], fontsize=7.5, color=INK2)
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("lift@10% на чужой площадке", fontsize=9, color=INK2)
    ax.set_title("3. Перенос: площадка, которую модель не видела\n"
                 "пунктир — уровень случайной карты", fontsize=10.5, color=INK)


def panel_rules(ax, fig) -> None:
    """Центры правил — то единственное, чего логрегрессия не даёт."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    p = M / "openlin_anfis_nevada_rules.csv"
    if not p.exists():
        ax.axis("off")
        return
    d = pd.read_csv(p)
    chans = [c for c in d.columns if c not in ("правило", "смещение")]
    v = d[chans].to_numpy()
    lim = float(np.abs(v).max())
    cmap = LinearSegmentedColormap.from_list("bwr", [C_NEG, C_MID, C_POS])
    im = ax.imshow(v, cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim),
                   aspect="auto")
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([c.replace("coh_summer_vv_", "лето ")
                        .replace("coh_winter_vv_", "зима ")
                        for c in chans], rotation=45, ha="right", fontsize=7.5,
                       color=INK2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"правило {i}" for i in d["правило"]], fontsize=7,
                       color=INK2)
    ax.set_title("4. Центры 16 правил (Невада, 10 признаков)\n"
                 "красное — «высокое», синее — «низкое» значение канала",
                 fontsize=10.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("центр правила, нормированные единицы", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)


def main() -> None:
    df = load()
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=INK2, labelsize=8)

    panel_models(axes[0, 0], df)
    axes[0, 0].grid(axis="y", alpha=0.18, color=INK2)
    panel_leak(axes[0, 1], df)
    axes[0, 1].grid(axis="x", alpha=0.18, color=INK2)
    panel_transfer(axes[1, 0], df)
    axes[1, 0].grid(axis="x", alpha=0.18, color=INK2)
    panel_rules(axes[1, 1], fig)

    fig.suptitle("ANFIS на табличной постановке: что померено",
                 fontsize=15, color=INK, y=0.975)
    fig.text(0.5, 0.015,
             "везде lift@10% с допуском 250 м; блочное разбиение — полосы с "
             "буфером 2 км; значимость — тороидальный сдвиговый нуль",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    print(f"сохранено: {OUT}")


if __name__ == "__main__":
    main()
