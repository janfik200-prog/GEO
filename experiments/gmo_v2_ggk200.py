"""Генетическая модель ВНИГНИ v2 на подложке геологических карт ГГК-200.

Цель: показать элементы ГМО v2 не на голых растрах признаков
(`experiments/gmo_v2_map.py`), а поверх геологической карты, где геолог
сразу видит свиты, разломы и дайки листа.

Входные данные:
* `datacache/ggk200/*/*.tif` — 8 листов ГГК-200 (`src/ggk200_base`);
* `data/external/anabar_shield/anabar_shield.gpkg` — контур щита (1:5 млн);
* `data/Gis-integro/shp_dbf/*.shp` — слои формулы и ореолы;
* `data/processed/dataset_v6.parquet` + `dataset_v5_rebuilt.parquet` — канал
  ASTER Mg-OH и доля обнажённости.

Левая панель — регион: 8 листов, граница щита, наш лист, листы с золотом.
Правая — основной лист: слои формулы (⚠ с осторожностью, решение
21.10.2026), ореолы Au (только заверка), аномалии хлорит-эпидотового сигнала
I.4 (верхние 10% по обнажённым ячейкам) — кандидат в новый фактор сверх
формулы (гипотеза Г1 разбора ГМО).

Запуск::

    python -m experiments.gmo_v2_ggk200
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, ggk200_base, mapstyle, outputs, spectral_mask  # noqa: E402
from experiments.gmo_v2_map import BARE_SRC, CAUTION, DATASET, SHP  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLD_SHEETS = ("R-49-I,II", "R-49-VII,VIII", "R-49-XIII,XIV")
REGION = (103.9, 69.95, 110.1, 72.05)
MAIN = (105.98, 70.65, 108.02, 71.35)
TOP = 0.90        # порог аномалии Mg-OH по обнажённым ячейкам


def layer(name):
    return ggk200_base.to_lonlat(gpd.read_file(SHP / f"{name}.shp"))


def mgoh_anomaly() -> gpd.GeoDataFrame:
    """Ячейки 500 м с Mg-OH в верхних 10% среди обнажённых — квадратами."""
    df = pd.read_parquet(DATASET, columns=["row", "col", "x", "y", "ast_mgoh"])
    bare = pd.read_parquet(BARE_SRC, columns=["row", "col", spectral_mask.BARE_COL])
    df = df.merge(bare, on=["row", "col"], validate="one_to_one")
    keep = spectral_mask.exposure_mask(df) & np.isfinite(df.ast_mgoh.to_numpy())
    sub = df[keep]
    sub = sub[sub.ast_mgoh >= sub.ast_mgoh.quantile(TOP)]
    sq = gpd.GeoSeries.from_xy(sub.x, sub.y).buffer(250, cap_style=3)
    g = gpd.GeoDataFrame(geometry=[sq.union_all()], crs=anabar_inset.SHEET_CRS)
    return g.to_crs(4326)


def main():
    fig, (axr, axm) = plt.subplots(1, 2, figsize=(22, 11.5),
                                   gridspec_kw=dict(width_ratios=[1, 1.05]),
                                   constrained_layout=True)

    # --- регион
    ggk200_base.draw(axr, bounds=REGION, decimate=8)
    shield = gpd.read_file(anabar_inset.SRC / "anabar_shield.gpkg")
    shield.boundary.plot(ax=axr, color="black", linewidth=1.6, linestyle="--")
    sheet = anabar_inset.sheet_lonlat()
    gpd.GeoSeries([sheet]).boundary.plot(ax=axr, color="red", linewidth=2.6)
    for s in GOLD_SHEETS:
        with __import__("rasterio").open(ggk200_base.sheet_tif(s)) as src:
            b = src.bounds
        axr.add_patch(plt.Rectangle((b.left, b.bottom), b.right - b.left,
                                    b.top - b.bottom, fill=False,
                                    edgecolor="gold", linewidth=2.4))
    axr.legend(handles=[
        Line2D([0], [0], color="black", lw=1.6, ls="--",
               label="граница щита (карта Арктики, 1:5 млн)"),
        Line2D([0], [0], color="red", lw=2.6, label="основной лист R-48-XI,XII"),
        Line2D([0], [0], color="gold", lw=2.4, label="листы R-49 с золотом"),
    ], loc="lower left", fontsize=9, framealpha=0.92)
    axr.set_title("Регион: 8 листов ГГК-200 и граница Анабарского щита", fontsize=12)

    # --- основной лист
    ggk200_base.draw(axm, bounds=MAIN, decimate=2, alpha=0.75)
    faults = layer("glub_raz_nw")
    faults.plot(ax=axm, facecolor="none", edgecolor=CAUTION, hatch="//",
                linewidth=1.2, alpha=0.8)
    layer("glub_r_nw").boundary.plot(ax=axm, color=CAUTION, linewidth=1.2,
                                     linestyle=":")
    layer("kory").plot(ax=axm, color="#7d3c98", linewidth=2.0)
    layer("gr_dol_vp_poly").boundary.plot(ax=axm, color="#1f618d",
                                          linewidth=1.4, linestyle="-.")
    mgoh_anomaly().plot(ax=axm, facecolor="#1e8449", edgecolor="none",
                        alpha=0.75)
    halos = layer("геохимические ореолы")
    halos[halos.label == "Au"].plot(ax=axm, color="gold", linewidth=3.2)
    axm.set_xlim(MAIN[0], MAIN[2])
    axm.set_ylim(MAIN[1], MAIN[3])
    axm.legend(handles=[
        Patch(facecolor="#1e8449", alpha=0.75,
              label="I.4 диафторез: Mg-OH 2.33 мкм, верхние 10% обнажённых ячеек"),
        Line2D([0], [0], color="gold", lw=3.2, label="ореолы Au — только заверка"),
        Patch(facecolor="none", edgecolor=CAUTION, hatch="//",
              label="I.3 разломные зоны СЗ ⚠"),
        Line2D([0], [0], color=CAUTION, lw=1.2, ls=":", label="I.3 разломные зоны СВ ⚠"),
        Line2D([0], [0], color="#7d3c98", lw=2, label="II.6 коры выветривания ⚠"),
        Line2D([0], [0], color="#1f618d", lw=1.4, ls="-.", label="III.8 палеодолины ⚠"),
    ], loc="lower left", fontsize=9, framealpha=0.92,
        title="⚠ слой формулы — с осторожностью", title_fontsize=8)
    axm.set_title("Основной лист R-48-XI,XII: элементы ГМО v2 на геологической карте",
                  fontsize=12)
    anabar_inset.draw(axm.inset_axes([0.73, 0.70, 0.27, 0.30]), fontsize=6,
                      title=False, legend=False)

    for ax in (axr, axm):
        ax.set_xlabel("долгота, ° в.д.")
        ax.set_ylabel("широта, ° с.ш.")

    fig.suptitle("Генетическая модель ВНИГНИ v2 (золото) на подложке ГГК-200", fontsize=14)
    out = outputs.png("gmo_v2_ggk200.png")
    mapstyle.save_fig(fig, out, dpi=140)
    print(f"Карта: {out}")


if __name__ == "__main__":
    main()
