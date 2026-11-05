"""Рисунки для колоды «Предметная модель» в крупном шрифте — читаются с экрана.

Тот же смысл, что у рабочих рисунков, но без служебных имён слоёв и с
кеглем, рассчитанным на слайд:

* `gmo_svity_labels_deck.png` — пять свит из списка ВНИГНИ на подложке
  ГГК-200 (данные — `data/processed/svita_labeled.gpkg`,
  `experiments/gmo_svita_labels.py`);
* `gmo_map_v0_2_deck.png` — итог и уверенность пробной карты v0.2, отмечена
  ячейка-пример слайда «одна ячейка — цепочка назад»;
* `gmo_map_v0_2_b_deck.png` — то же для варианта Б (слои формулы ⚠), та же
  ячейка-пример;
* `gmo_lineaments_deck.png` — отрезки вне долин и вдоль долин поверх отмывки
  (`experiments/gmo_lineaments_clean.py`);
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


def map_v02(src: str = "knowledge_map_v0_2", out_name: str = "gmo_map_v0_2_deck.png") -> None:
    m = pd.read_parquet(ROOT / f"data/processed/{src}.parquet")
    nrow, ncol = int(m.row.max()) + 1, int(m.col.max()) + 1
    x0 = (m.x - (m.col + 0.5) * 500).iloc[0] / 1e3
    ytop = (m.y + (m.row + 0.5) * 500).iloc[0] / 1e3
    ext = [x0, x0 + ncol * 0.5, ytop - nrow * 0.5, ytop]
    # Ячейка-пример всегда из варианта А (115, 28) — на варианте Б та же ячейка
    a = pd.read_parquet(ROOT / "data/processed/knowledge_map_v0_2.parquet").nlargest(1, "итог").iloc[0]
    ex = m[(m.row == a.row) & (m.col == a.col)].iloc[0]
    lon, lat = gpd.GeoSeries(gpd.points_from_xy([ex.x], [ex.y]), crs=anabar_inset.SHEET_CRS).to_crs(4326).iloc[0].coords[0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, col, cm, ttl in ((axs[0], "итог", "viridis", "Итог ячейки"),
                             (axs[1], "уверенность", "Greys", "Уверенность: доля наблюдённых звеньев")):
        a = np.full((nrow, ncol), np.nan)
        a[m.row, m.col] = m[col]
        im = ax.imshow(a, extent=ext, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.72)
        cb.set_label("доля 0–1", fontsize=12); cb.ax.tick_params(labelsize=11)
        ax.plot(ex.x / 1e3, ex.y / 1e3, "o", ms=13, mfc="none", mec="#e74c3c", mew=2.5)
        ax.set_title(ttl, fontsize=14)
        ax.set_xlabel("X, км", fontsize=12); ax.set_ylabel("Y, км", fontsize=12)
        ax.tick_params(labelsize=10)
    axs[0].annotate(f"ячейка-пример\n{lon:.2f}° в.д., {lat:.2f}° с.ш.", (ex.x / 1e3, ex.y / 1e3),
                    xytext=(ex.x / 1e3 + 9, ex.y / 1e3 - 14), fontsize=12, color="white",
                    weight="bold", arrowprops=dict(arrowstyle="-", color="#e74c3c", lw=1.5))
    fig.text(0.5, 0.02, "Белое — ячейка не покрыта геологической картой свит", ha="center",
             fontsize=12, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = outputs.png(out_name)
    mapstyle.save_fig(fig, out, dpi=160)
    print(f"Рисунок: {out}; пример {lon:.3f}, {lat:.3f}")


def lineaments_panel() -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from experiments import gmo_lineaments_clean as lc
    from src import config, integro_grid, lineaments, terrain_v2
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    raw = terrain_v2.projected_dem(meta)
    valid = np.isfinite(raw)
    elev = np.where(valid, raw, np.nanmedian(raw))
    acc, ang = lc.thalwegs(raw, valid)
    v = lc.Valleys(acc, ang, lc.A0_KM2)
    segs = lineaments.extract_lines(lineaments.edge_map(elev, res_m=lc.RES))
    valley = lc.classify(segs, v, lc.W_M)
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.imshow(lineaments.hillshade(elev, 315.0), cmap="gray", alpha=0.75)
    from matplotlib.colors import ListedColormap
    ax.imshow(np.ma.masked_where(~v.mask, v.mask), cmap=ListedColormap(["#4fc3f7"]), alpha=0.8)
    for flag, col in ((False, "#c0392b"), (True, "#1a237e")):
        ax.add_collection(LineCollection([[a, b] for (a, b), q in zip(segs, valley) if q == flag],
                                         colors=col, linewidths=1.1))
    pad = config.TER_PAD_PX
    ax.set_xlim(pad, pad + meta.pic * 5); ax.set_ylim(pad + meta.prf * 5, pad)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Line2D([0], [0], color="#c0392b", lw=2.5, label=f"вне долин — {int((~valley).sum())} отрезков"),
                       Line2D([0], [0], color="#1a237e", lw=2.5, label=f"вдоль долин — {int(valley.sum())} отрезков"),
                       Line2D([0], [0], color="#4fc3f7", lw=4, label="тальвеги по рельефу")],
              loc="lower left", fontsize=13, framealpha=0.92)
    fig.tight_layout()
    out = outputs.png("gmo_lineaments_deck.png")
    mapstyle.save_fig(fig, out, dpi=160)
    print(f"Рисунок: {out}")


if __name__ == "__main__":
    svity()
    h7()
    map_v02()
    map_v02("knowledge_map_v0_2_b", "gmo_map_v0_2_b_deck.png")   # вариант Б
    lineaments_panel()
