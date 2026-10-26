"""Врезка «положение листа на Анабарском щите» для карт проекта.

Контур щита и фоновые единицы — «Геологическая карта Арктики» (Harrison и др.,
2011, 1:5 000 000), подробности и происхождение в
`data/external/anabar_shield/README.md`. Масштаб обзорный: только для врезки,
не для признаков.

Использование::

    from src import anabar_inset
    anabar_inset.draw(ax.inset_axes([0.0, 0.6, 0.42, 0.4]))
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "external" / "anabar_shield"

# Система координат листа (из .pj4 слоёв ГИС Интегро)
SHEET_CRS = ("+proj=tmerc +lon_0=105 +lat_0=0 +x_0=500000 +y_0=0 +k_0=1 "
             "+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +ellps=krass "
             "+units=m +no_defs")

# Рамка основного листа R-48-XI,XII в метрах листа
SHEET_BOX = (536000.0, 7843600.0, 611000.0, 7920600.0)

LON = (104.0, 116.0)
LAT = (68.7, 73.4)

SHIELD = "#c0392b"
RIPHEAN = "#e8d9b5"
POPIGAI_ID = 634119


def sheet_lonlat() -> shapely.Polygon:
    return gpd.GeoSeries([shapely.box(*SHEET_BOX)], crs=SHEET_CRS).to_crs(4326).iloc[0]


def draw(ax, legend: bool = True, fontsize: float = 7, title: bool = True) -> None:
    """Врезка в долготе-широте с поправкой масштаба на широту листа."""
    units = gpd.read_file(SRC / "arctic_map_units.gpkg")
    shield = gpd.read_file(SRC / "anabar_shield.gpkg")
    units["t_age"] = units.t_age.astype(float)
    units["b_age"] = units.b_age.astype(float)

    ax.set_facecolor("#f4f4f4")
    # рифей: между палеопротерозоем и венд-кембрием (1600–1000 млн лет снизу)
    riph = units[(units.b_age <= 1800) & (units.b_age >= 1000) & (units.t_age < 1600)]
    riph.plot(ax=ax, color=RIPHEAN, edgecolor="none")
    units[units.map_id == POPIGAI_ID].plot(ax=ax, facecolor="none",
                                           edgecolor="0.3", linewidth=0.6,
                                           linestyle=":")
    shield.plot(ax=ax, facecolor="#f5c6bf", edgecolor=SHIELD, linewidth=1.0)
    sheet = sheet_lonlat()
    gpd.GeoSeries([sheet]).plot(ax=ax, facecolor="none", edgecolor="black",
                                linewidth=1.6)
    ax.annotate("R-48-XI,XII", (sheet.bounds[0], sheet.bounds[3]),
                xytext=(sheet.bounds[0] - 0.3, sheet.bounds[3] + 0.5),
                fontsize=fontsize, fontweight="bold")
    ax.text(111.2, 71.95, "Попигай", fontsize=fontsize - 1, color="0.3",
            ha="center")

    ax.set_xlim(*LON)
    ax.set_ylim(*LAT)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(LAT))))
    ax.set_xticks(range(int(LON[0]), int(LON[1]) + 1, 4))
    ax.set_yticks(range(int(np.ceil(LAT[0])), int(LAT[1]) + 1, 2))
    ax.set_xticklabels([f"{v}°" for v in ax.get_xticks().astype(int)], fontsize=fontsize - 1)
    ax.set_yticklabels([f"{v}°" for v in ax.get_yticks().astype(int)], fontsize=fontsize - 1)
    ax.grid(color="white", linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title("Положение листа на Анабарском щите", fontsize=fontsize + 1)
    if legend:
        ax.legend(handles=[
            Patch(facecolor="#f5c6bf", edgecolor=SHIELD, label="щит: архей–палеопротерозой"),
            Patch(facecolor=RIPHEAN, label="рифейский чехол"),
            Line2D([0], [0], color="black", lw=1.6, label="наш лист"),
        ], loc="lower right", fontsize=fontsize - 1, framealpha=0.9)
    ax.text(0.01, 0.01, "по Harrison и др., 2011 (1:5 млн)",
            transform=ax.transAxes, fontsize=fontsize - 2, color="0.35")


def main() -> None:
    """Отдельная карта положения листа (крупно)."""
    import matplotlib.pyplot as plt
    from src import mapstyle, outputs
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    draw(ax, fontsize=10, title=False)
    ax.set_title("Анабарский щит и положение листа R-48-XI,XII\n"
                 "контур щита — выходы архея и палеопротерозоя (Harrison и др., 2011)",
                 fontsize=11)
    ax.set_xlabel("долгота, ° в.д.")
    ax.set_ylabel("широта, ° с.ш.")
    out = outputs.png("gmo_anabar_location.png")
    mapstyle.save_fig(fig, out, dpi=150)
    print(f"Карта: {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
