"""Обзорная карта Анабарского щита: 8 листов ГГК-200 + контуры наших сеток.

Зачем. Перед решением, стоит ли гнаться за золоторудной зоной на R-49-I,II
(см. :mod:`experiments.digitize_au_zone`) или расширять `dataset_wide.parquet`
(см. :mod:`experiments.fetch_wide_area`), нужна одна картинка, на которой видно
всё сразу: реальные геологические карты 8 смежных листов (мозаика в EPSG:4326,
как их отдаёт geokniga), поверх — контур основной расчётной сетки (прогноз,
74.5x77 км), контур `WIDE_META` (сетка-расширение на смежную территорию,
построенная в этой сессии) и оцифрованная золоторудная зона на R-49-I,II.

Растры читаются decimated (read с out_shape) — полное разрешение (~5000x1700 px
на лист) не нужно для обзорной картинки.

Запуск из корня: ``python -m experiments.plot_ggk200_overview``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from pyproj import CRS, Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache_paths import GGK200  # noqa: E402
from experiments.fetch_ggk200 import SHEETS  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402
from src import config  # noqa: E402
from src.integro_grid import read_grid_proj4, read_pgrid  # noqa: E402

GOLD_SHEETS = {"R-49-I,II", "R-49-VII,VIII", "R-49-XIII,XIV"}
DECIMATE = 6
AU_ZONE = config.PROJECT_ROOT / "data" / "processed" / "au_zone_r49_i_ii.geojson"
OUT = config.PROJECT_ROOT / "outputs" / "ggk200_overview_map.png"


def sheet_tif(sheet: str) -> Path | None:
    folder = GGK200 / sheet.replace(",", "_")
    tifs = list(folder.glob("*geologicheskayakarta.tif")) or list(folder.glob("*kartageologicheskaya.tif"))
    return tifs[0] if tifs else None


def grid_bounds_lonlat(meta) -> np.ndarray:
    """4 угла сетки в WGS84 (lon, lat), по часовой стрелке от СЗ."""
    proj4 = read_grid_proj4(config.GOLD_TARGET_PGRID)
    to_wgs = Transformer.from_crs(CRS.from_proj4(proj4), CRS.from_epsg(4326), always_xy=True)
    x0, x1 = meta.x0, meta.x0 + meta.pic * meta.dx
    y0, y1 = meta.y0, meta.y_top
    xs = [x0, x1, x1, x0, x0]
    ys = [y1, y1, y0, y0, y1]
    lon, lat = to_wgs.transform(xs, ys)
    return np.column_stack([lon, lat])


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 15))

    print("Листы ГГК-200:")
    for sheet, (node, side) in SHEETS.items():
        tif = sheet_tif(sheet)
        if tif is None:
            print(f"  {sheet:<17} ({side}) — GeoTIFF не найден, пропущен")
            continue
        with rasterio.open(tif) as src:
            out_shape = (src.count, src.height // DECIMATE, src.width // DECIMATE)
            img = src.read(out_shape=out_shape).transpose(1, 2, 0)
            b = src.bounds
        ax.imshow(img, extent=[b.left, b.right, b.bottom, b.top], origin="upper", zorder=1)
        is_gold = sheet in GOLD_SHEETS
        edge = "gold" if is_gold else "0.5"
        lw = 2.5 if is_gold else 1.0
        ax.add_patch(plt.Rectangle((b.left, b.bottom), b.right - b.left, b.top - b.bottom,
                                    fill=False, edgecolor=edge, linewidth=lw, zorder=3))
        cx, cy = (b.left + b.right) / 2, (b.top - 0.05)
        label = f"{sheet}\n({side})" + (" — Au" if is_gold else "")
        ax.annotate(label, (cx, cy), ha="center", va="top", fontsize=9,
                    color="darkgoldenrod" if is_gold else "black", weight="bold" if is_gold else "normal",
                    zorder=4)
        print(f"  {sheet:<17} ({side}), {img.shape[1]}x{img.shape[0]} px decim, "
              f"{'ЗОЛОТО' if is_gold else '—'}")

    # Контур основной расчётной сетки (prognoz)
    main_meta = read_pgrid(config.GOLD_TARGET_PGRID)
    main_ring = grid_bounds_lonlat(main_meta)
    ax.plot(main_ring[:, 0], main_ring[:, 1], color="red", linewidth=2.5, zorder=5,
            label="основная сетка (prognoz, 74.5x77 км)")

    # Контур WIDE_META (расширение)
    wide_ring = grid_bounds_lonlat(WIDE_META)
    ax.plot(wide_ring[:, 0], wide_ring[:, 1], color="cyan", linewidth=2.5, zorder=5,
            label="WIDE_META (сетка-расширение)")

    # Контур grav-mag съёмки (покрытие gm_*/pf_* признаков)
    try:
        gm_meta = read_pgrid(config.GRAVMAG_PGRID)
        gm_ring = grid_bounds_lonlat(gm_meta)
        ax.plot(gm_ring[:, 0], gm_ring[:, 1], color="magenta", linewidth=1.5, linestyle="--", zorder=5,
                label="грав-маг съёмка (gm_*/pf_* доступны)")
    except Exception as e:
        print(f"  грав-маг сетка не прочитана: {e}")

    # Оцифрованная золоторудная зона
    if AU_ZONE.exists():
        geo = json.loads(AU_ZONE.read_text(encoding="utf-8"))
        coords = np.array(geo["features"][0]["geometry"]["coordinates"][0])
        poly = MplPolygon(coords, closed=True, facecolor="gold", edgecolor="darkgoldenrod",
                           alpha=0.7, linewidth=1.5, zorder=6)
        ax.add_patch(poly)
        ax.annotate("зона Au\n(оцифровано)", coords.mean(axis=0), ha="center", fontsize=8,
                    color="darkgoldenrod", weight="bold", zorder=7)

    handles = [Line2D([0], [0], color="red", lw=2.5, label="основная сетка (prognoz)"),
               Line2D([0], [0], color="cyan", lw=2.5, label="WIDE_META (расширение)"),
               Line2D([0], [0], color="magenta", lw=1.5, ls="--", label="грав-маг съёмка (gm_*/pf_*)"),
               Line2D([0], [0], color="gold", lw=2.5, label="листы с золотом (R-49)")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xlabel("долгота, °в.д.")
    ax.set_ylabel("широта, °с.ш.")
    ax.set_title("Анабарский щит: основной лист R-48-XI,XII + 7 смежных листов ГГК-200")
    ax.set_aspect(1 / np.cos(np.radians(71)))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\nOK: {OUT}")


if __name__ == "__main__":
    main()
