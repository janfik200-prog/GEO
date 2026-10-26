"""Генетическая модель ВНИГНИ v2 на карте листа: что из неё видно нашими данными.

Цель: показать каждый элемент ГМО v2 («Генетическая схема наследования и
перераспределения золота», `Генетические-модели/`) на территории листа
R-48-XI,XII (Анабарский щит) тем слоем, которым он у нас наблюдается, — и
отдельно отметить, где наблюдение идёт только через слои критериальной формулы.

Входные данные:
* `data/processed/dataset_v6.parquet` — каналы на сетке 500 м;
* `data/processed/dataset_v5_rebuilt.parquet` — доля обнажённости `s2_bare_frac`;
* `data/Gis-integro/shp_dbf/*.shp` — слои формулы ВНИГНИ и ореолы.

Слои формулы (разломные зоны, дайки, коры, фации, палеодолины, контакт чехла)
решением 21.10.2026 допускаются **с осторожностью**: на карте они штрихуются и
помечены «⚠ слой формулы». Использование их в признаках ещё не решено —
совпадение с картой ВНИГНИ по ним не будет независимой заверкой.

Уран не показывается (решение 21.10.2026): ореолы — только Au.

Спектры ASTER показаны только на обнажённых ячейках (`src/spectral_mask`):
под растительностью индекс меряет ландшафт, а не породу.

Запуск::

    python -m experiments.gmo_v2_map
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, mapstyle, outputs, spectral_mask  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SHP = ROOT / "data" / "Gis-integro" / "shp_dbf"
DATASET = ROOT / "data" / "processed" / "dataset_v6.parquet"
# доля обнажённости есть только в v5_rebuilt; сетка та же (row, col, x, y)
BARE_SRC = ROOT / "data" / "processed" / "dataset_v5_rebuilt.parquet"
CELL = 500.0

CAUTION = "#c0392b"          # цвет слоёв формулы
CAUTION_TAG = "⚠ слой формулы — с осторожностью"


def grid_image(df: pd.DataFrame, col: str, keep: np.ndarray | None = None):
    """Столбец датасета -> растр (row, col) и охват в метрах."""
    nr, nc = int(df.row.max()) + 1, int(df.col.max()) + 1
    img = np.full((nr, nc), np.nan)
    v = df[col].to_numpy(dtype=float).copy()
    if keep is not None:
        v[~keep] = np.nan
    img[df.row.to_numpy(), df.col.to_numpy()] = v
    # ряд 0 — север или юг: ориентируем по координате y
    if df.loc[df.row == 0, "y"].mean() < df.loc[df.row == nr - 1, "y"].mean():
        img = img[::-1]
    ext = (df.x.min() - CELL / 2, df.x.max() + CELL / 2,
           df.y.min() - CELL / 2, df.y.max() + CELL / 2)
    return img, ext


def clipped(img: np.ndarray, lo=2, hi=98):
    a, b = np.nanpercentile(img, [lo, hi])
    return dict(vmin=a, vmax=b)


def formula_poly(ax, gdf, label):
    gdf.plot(ax=ax, facecolor="none", edgecolor=CAUTION, hatch="///",
             linewidth=0.9, alpha=0.9)
    return Patch(facecolor="none", edgecolor=CAUTION, hatch="///",
                 label=f"{label} ⚠")


def formula_line(ax, gdf, label, ls="-"):
    gdf.plot(ax=ax, color=CAUTION, linewidth=1.2, linestyle=ls)
    return Line2D([0], [0], color=CAUTION, lw=1.2, ls=ls, label=f"{label} ⚠")


def contact_gdf(svita: gpd.GeoDataFrame, df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Контакт как его берёт формула: внешний контур объединения свит
    (`src/geology_v2.contact_boundary_segments`), без участков по рамке листа —
    там контур обрезан границей листа, а не геологией."""
    union = shapely.union_all(svita.geometry.values)
    frame = shapely.box(df.x.min() - CELL / 2, df.y.min() - CELL / 2,
                        df.x.max() + CELL / 2, df.y.max() + CELL / 2)
    line = union.boundary.difference(frame.exterior.buffer(1500))
    return gpd.GeoDataFrame(geometry=[line])


def contact_line(ax, contact: gpd.GeoDataFrame):
    contact.plot(ax=ax, color=CAUTION, linewidth=1.4, linestyle="--")
    return Line2D([0], [0], color=CAUTION, lw=1.4, ls="--",
                  label="контакт: внешний контур свит ⚠")


def main():
    df = pd.read_parquet(DATASET)
    bare = pd.read_parquet(BARE_SRC, columns=["row", "col", spectral_mask.BARE_COL])
    df = df.merge(bare, on=["row", "col"], how="left", validate="one_to_one")
    expo = spectral_mask.exposure_mask(df)

    layers = {n: gpd.read_file(SHP / f"{n}.shp") for n in (
        "svita_new", "glub_raz_nw", "glub_r_nw", "dayki_buf", "kory",
        "fasii", "gr_dol_vp_poly", "геохимические ореолы")}
    halos_au = layers["геохимические ореолы"].query("label == 'Au'")
    contact = contact_gdf(layers["svita_new"], df)

    # (заголовок, элемент ГМО, канал, подпись шкалы, cmap, маска обнажённости,
    #  функция наложения слоёв формулы)
    panels = [
        ("I.2 Интрузивные комплексы",
         "gm_mag_ost_35", "остаточное магнитное поле, отн. ед.", "RdBu_r", False,
         lambda ax: [formula_poly(ax, layers["dayki_buf"], "дайки (буфер)")]),
        ("I.3 Глубинные разломные зоны",
         "lin_dens", "плотность линеаментов, км/км²", "magma", False,
         lambda ax: [formula_poly(ax, layers["glub_raz_nw"], "разломные зоны СЗ"),
                     formula_line(ax, layers["glub_r_nw"].boundary.to_frame("geometry"),
                                  "разломные зоны СВ", ":")]),
        ("I.4 Зоны диафтореза (хлорит, эпидот)",
         "ast_mgoh", "ASTER Mg-OH 2.33 мкм, отношение", "viridis", True,
         lambda ax: [formula_line(ax, layers["glub_raz_nw"].boundary.to_frame("geometry"),
                                  "контур разломных зон СЗ")]),
        ("II.3–4 Реактивация, выщелачивание",
         "ast_alter", "ASTER индекс изменений, отношение", "viridis", True,
         lambda ax: []),
        ("II.5 Пликативные дислокации",
         "lin_aniso_r", "анизотропия линеаментов, 0–1 (где они есть)", "cividis", False,
         lambda ax: []),
        ("II.6 Коры выветривания",
         "ast_kaolin", "ASTER каолинит, отношение", "viridis", True,
         lambda ax: [formula_line(ax, layers["kory"], "коры выветривания")]),
        ("III.7–8 Алевропелиты и грубообломочные",
         "dem_tpi_2km", "положение в рельефе (TPI 2 км), м", "BrBG", False,
         lambda ax: [formula_poly(ax, layers["fasii"], "фации"),
                     formula_line(ax, layers["gr_dol_vp_poly"].boundary.to_frame("geometry"),
                                  "палеодолины", "-.")]),
        ("Сохранность и несогласие (нет в v2)",
         "dem_elev", "высота, м", "terrain", False,
         lambda ax: [contact_line(ax, contact)]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(19, 20), constrained_layout=True)
    axes = axes.ravel()

    # Обзорная панель: геология и заверка
    ax = axes[0]
    layers["svita_new"].plot(ax=ax, facecolor="0.85", edgecolor="0.55",
                             linewidth=0.3)
    img, ext = grid_image(df, spectral_mask.BARE_COL)
    ax.contourf(np.where(np.isfinite(img), img >= spectral_mask.BARE_MIN, 0),
                levels=[0.5, 1.5], colors=["#2e86c1"], alpha=0.35,
                extent=ext, origin="upper")
    halos_au.plot(ax=ax, color="#d4a017", linewidth=2.0)
    h = [Patch(facecolor="0.85", edgecolor="0.55", label="полигоны свит (svita_new)"),
         Patch(facecolor="#2e86c1", alpha=0.35,
               label=f"обнажённые ячейки ({np.mean(expo):.0%} листа)"),
         Line2D([0], [0], color="#d4a017", lw=2, label="ореолы Au — только заверка")]
    h.append(contact_line(ax, contact))
    ax.legend(handles=h, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Лист R-48-XI,XII, Анабарский щит\nобзор: геология и заверка",
                 fontsize=11)
    anabar_inset.draw(ax.inset_axes([0.0, 0.6, 0.44, 0.4]), fontsize=6.5, title=False)

    for ax, (title, col, unit, cmap, spectral, over) in zip(axes[1:], panels):
        keep = expo if spectral else None
        if col == "lin_aniso_r":
            keep = df["lin_dens"].to_numpy() > 0
        img, ext = grid_image(df, col, keep)
        im = ax.imshow(img, extent=ext, cmap=cmap, interpolation="nearest",
                       **clipped(img))
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label(unit, fontsize=8)
        handles = over(ax)
        sub = f"канал: {col}"
        if spectral:
            sub += f" (только обнажённые ячейки, {np.mean(keep):.0%} листа)"
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=8,
                      framealpha=0.9, title=CAUTION_TAG, title_fontsize=7)

    for ax in axes:
        ax.set_xlim(df.x.min() - CELL, df.x.max() + CELL)
        ax.set_ylim(df.y.min() - CELL, df.y.max() + CELL)
        ax.set_aspect("equal")
        ax.set_xlabel("восток, м (Гаусс–Крюгер, осевой 105°)", fontsize=8)
        ax.set_ylabel("север, м", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(style="plain", useOffset=False)

    fig.suptitle(
        "Генетическая модель ВНИГНИ v2 (золото) на территории: чем наблюдается каждый элемент\n"
        "Не наблюдаются нашими данными: I.1 золотоносные кварцевые жилы, III.9 кварц-карбонатные жилы, "
        "механизм ловушки.\n"
        "Красная штриховка и линии — слои критериальной формулы: пока допускаются, с осторожностью.",
        fontsize=12)
    out = outputs.png("gmo_v2_map.png")
    mapstyle.save_fig(fig, out, dpi=130)
    print(f"Карта: {out}")


if __name__ == "__main__":
    main()
