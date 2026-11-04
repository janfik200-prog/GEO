"""Рисунки для колоды «Предметная модель» в крупном шрифте — читаются с экрана.

Тот же смысл, что у рабочих рисунков, но без служебных имён слоёв и с
кеглем, рассчитанным на слайд:

* `gmo_svity_labels_deck.png` — пять свит из списка ВНИГНИ на подложке
  ГГК-200 (данные — `data/processed/svita_labeled.gpkg`,
  `experiments/gmo_svita_labels.py`);
* `gmo_h7_deck.png` — высота рельефа по подразделениям (данные и расчёт — `experiments/gmo_h2_h7_checks.py`).

Чисел заверки на рисунках нет.

Запуск::

    python -X utf8 -m experiments.gmo_deck_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gmo_h2_h7_checks as h27  # noqa: E402
from experiments.gmo_svita_labels import COLORS, MAIN  # noqa: E402
from src import anabar_inset, ggk200_base, mapstyle, outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def svity() -> None:
    g = gpd.read_file(ROOT / "data/processed/svita_labeled.gpkg").to_crs(4326)
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.subplots_adjust(left=0.08, right=0.60, top=0.97, bottom=0.10)
    ggk200_base.draw(ax, bounds=MAIN, decimate=2, alpha=0.5, sheets=("R-48-XI,XII",))
    rock = g[g.in_vnigni_list.astype(bool)]
    for sv, c in COLORS.items():
        rock[rock.svita == sv].plot(ax=ax, facecolor=c, edgecolor="k", linewidth=0.3, alpha=0.6)
    g[~g.in_vnigni_list.astype(bool)].plot(ax=ax, facecolor="0.85", edgecolor="0.4",
                                           linewidth=0.3, alpha=0.8)
    ax.set_xlim(MAIN[0], MAIN[2]); ax.set_ylim(MAIN[1], MAIN[3])
    handles = [Patch(facecolor=c, alpha=0.6, edgecolor="k", label=f"{sv}")
               for sv, c in COLORS.items()]
    handles.append(Patch(facecolor="0.85", edgecolor="0.4",
                         label="вне списка: четвертичные,\nдолериты, кембрий, архей"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=13,
              framealpha=0.92, title="Свиты из списка ВНИГНИ", title_fontsize=13)
    ax.set_xlabel("долгота, ° в.д.", fontsize=12)
    ax.set_ylabel("широта, ° с.ш.", fontsize=12)
    ax.tick_params(labelsize=11)
    anabar_inset.draw(ax.inset_axes([1.04, 0.0, 0.5, 0.42]), fontsize=10, title=False,
                      legend=False)
    out = outputs.png("gmo_svity_labels_deck.png")
    mapstyle.save_fig(fig, out, dpi=160)
    print(f"Рисунок: {out}")


def h7() -> None:
    cols = ["row", "col", "x", "y", "dem_elev", "dem_tpi_2km"]
    df = pd.read_parquet(h27.DATA, columns=cols)
    df["unit"] = h27.main_unit(df)
    d = df[df.unit.notna() & df.dem_elev.notna() & df.dem_tpi_2km.notna()].copy()
    d["level"] = d.unit.map(h27.LEVEL).astype(float)
    short = {"архей и нижний протерозой": "архей", "кембрий (немакит-далдынская, медвежинская)": "кембрий"}
    # Одна панель: правая (уровень по положению на склоне) немонотонна и после
    # вычета наклона моноклинали незначима — на слайде она только путает.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    order = [u for u in h27.LEVEL if u in set(d.unit)]
    ax.boxplot([d.dem_elev[d.unit == u] for u in order], showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), [short.get(u, u) for u in order],
                  rotation=35, ha="right", fontsize=12)
    ax.set_ylabel("высота рельефа, м", fontsize=13)
    ax.set_title("Высота выходов по подразделениям, от древних к молодым", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = outputs.png("gmo_h7_deck.png")
    mapstyle.save_fig(fig, out, dpi=160)
    print(f"Рисунок: {out}")


if __name__ == "__main__":
    svity()
    h7()
