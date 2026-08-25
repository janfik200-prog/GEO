"""Карта: реально присланные заказчиком листы vs заявленное покрытие грав-магнитки.

Зачем. Заказчик утверждает, что грав-магнитная съёмка есть на всю
присланную им территорию. Проверка по факту:

1. Точная территория "сколько листов прислал заказчик" — не по названиям
   листов ГГК-200, а по реальной геометрии: `data/SBORKA_NESKOLKO_LISTOV/
   .../Сцены 2D/Геология/рамка_листов.shp` — единственный полигон в архиве
   "СБОРКА_НА_НЕСКОЛЬКО_ЛИСТОВ" (присланном отдельно от основного архива
   `Gis-integro/`, который содержит только один лист R-48-XI,XII).
   Проверка по атрибуту `CLASSNAME` слоя `r48_geol_gk.shp` (749 объектов,
   код листа зашит в префиксе, например `R48_1718_g_314`) подтверждает
   независимо: объекты относятся к трём листам — q48_1112 (=R-48-XI,XII,
   61 объект), R48_1718 (=R-48-XVII,XVIII, 470 объектов, основной массив),
   R48_1516 (=R-48-XV,XVI, всего 18 объектов — гораздо реже, лист оцифрован
   не полностью).
2. Покрытие грав-магнитки — фактический экстент `грав_маг.pgrid`
   (`config.GRAVMAG_PGRID`, 305x455 ячеек, шаг 500 м).

Пересечение полигона `рамка_листов` с прямоугольником листов ГГК-200 (по
геотиффам geokniga, наша отдельная докачка для контекста) даёт: R-48-XI,XII
и R-48-XVII,XVIII — покрыты практически полностью (>99.9% площади листа),
R-48-XV,XVI — только восточная половина (~50%), R-48-IX,X и все листы
R-49 — не задеты (0%). Итого содержательно это НЕ 3 целых листа, а 2 целых
+ половина третьего, плюс небольшой довесок южнее XV,XVI за пределами
формальной сетки известных 8 листов.

Пересечение `рамка_листов` с прямоугольником `грав_маг.pgrid` (в той же
проекции, без перехода в WGS84) даёт покрытие 99.93% площади присланной
заказчиком территории (не покрыта только полоска ~11 км² по краю,
объяснимая округлением рамки при дискретизации сетки шагом 500 м) —
заявление заказчика подтверждается количественно.

Запуск из корня: ``python -m experiments.plot_customer_sheets_gravmag``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
from matplotlib.lines import Line2D
from shapely.geometry import box
from pyproj import CRS, Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cache_paths import GGK200  # noqa: E402
from experiments.fetch_ggk200 import SHEETS  # noqa: E402
from src import config  # noqa: E402
from src.integro_grid import read_grid_proj4, read_pgrid  # noqa: E402

DECIMATE = 6
CUSTOMER_FRAME_SHP = (
    config.PROJECT_ROOT / "data" / "SBORKA_NESKOLKO_LISTOV"
    / "СБОРКА_НА_НЕСКОЛЬКО_ЛИСТОВ" / "Сцены 2D" / "Геология" / "рамка_листов.shp"
)
OUT = config.PROJECT_ROOT / "outputs" / "customer_sheets_gravmag_map.png"

# Число оцифрованных геологических объектов на лист по CLASSNAME-префиксу
# r48_geol_gk.shp (749 объектов всего) — см. docstring выше, п.1.
GEOL_OBJECTS_BY_SHEET = {
    "R-48-XI,XII": 61,
    "R-48-XVII,XVIII": 470,
    "R-48-XV,XVI": 18,
}


def sheet_tif(sheet: str) -> Path | None:
    folder = GGK200 / sheet.replace(",", "_")
    tifs = list(folder.glob("*geologicheskayakarta.tif")) or list(folder.glob("*kartageologicheskaya.tif"))
    return tifs[0] if tifs else None


def grid_ring_lonlat(meta, proj4: str) -> np.ndarray:
    """4 угла сетки (в её собственной проекции) в WGS84, замкнутое кольцо."""
    to_wgs = Transformer.from_crs(CRS.from_proj4(proj4), CRS.from_epsg(4326), always_xy=True)
    x0, x1 = meta.x0, meta.x0 + meta.pic * meta.dx
    y0, y1 = meta.y0, meta.y_top
    xs = [x0, x1, x1, x0, x0]
    ys = [y1, y1, y0, y0, y1]
    lon, lat = to_wgs.transform(xs, ys)
    return np.column_stack([lon, lat])


def main() -> None:
    main_proj4 = read_grid_proj4(config.GOLD_TARGET_PGRID)

    # Полигон "рамка_листов" — реальная территория, присланная заказчиком.
    frame_native = gpd.read_file(CUSTOMER_FRAME_SHP).set_crs(main_proj4, allow_override=True)
    frame_wgs = frame_native.to_crs(4326)
    frame_geom = frame_native.geometry.iloc[0]

    # Проверка покрытия грав-магнитки (в проекции, без искажений WGS84).
    gm_meta = read_pgrid(config.GRAVMAG_PGRID)
    gm_proj4 = read_grid_proj4(config.GRAVMAG_PGRID)
    gm_box_native = box(gm_meta.x0, gm_meta.y0, gm_meta.x0 + gm_meta.pic * gm_meta.dx, gm_meta.y_top)
    covered_frac = frame_geom.intersection(gm_box_native).area / frame_geom.area
    print(f"Грав-магнитка покрывает {covered_frac:.2%} площади присланной заказчиком территории "
          f"(рамка_листов, {frame_geom.area / 1e6:.0f} км²)")

    fig, ax = plt.subplots(figsize=(18, 15))

    print("\nЛисты ГГК-200 (контекст, наша докачка с geokniga) и доля их площади, "
          "покрытая присланной заказчиком территорией:")
    for sheet, (node, side) in SHEETS.items():
        tif = sheet_tif(sheet)
        if tif is None:
            print(f"  {sheet:<17} ({side}) — GeoTIFF не найден, пропущен")
            continue
        with rasterio.open(tif) as src:
            out_shape = (src.count, src.height // DECIMATE, src.width // DECIMATE)
            img = src.read(out_shape=out_shape).transpose(1, 2, 0)
            b = src.bounds
        ax.imshow(img, extent=[b.left, b.right, b.bottom, b.top], origin="upper", zorder=1, alpha=0.85)

        sheet_box = box(b.left, b.bottom, b.right, b.top)
        overlap = frame_wgs.geometry.iloc[0].intersection(sheet_box).area / sheet_box.area
        is_sent = overlap > 0.01
        edge = "steelblue" if is_sent else "0.5"
        lw = 2.0 if is_sent else 1.0
        ax.add_patch(plt.Rectangle((b.left, b.bottom), b.right - b.left, b.top - b.bottom,
                                    fill=False, edgecolor=edge, linewidth=lw, zorder=3))
        cx, cy = (b.left + b.right) / 2, (b.top - 0.05)
        geol_n = GEOL_OBJECTS_BY_SHEET.get(sheet)
        geol_note = f"\nоцифровано геологии: {geol_n} объектов" if geol_n is not None else ""
        label = f"{sheet}\n({side})" + (f"\nприслано {overlap:.0%}{geol_note}" if is_sent else "")
        ax.annotate(label, (cx, cy), ha="center", va="top", fontsize=9,
                    color="steelblue" if is_sent else "0.35",
                    weight="bold" if is_sent else "normal", zorder=4)
        print(f"  {sheet:<17} ({side:<12}) присланная территория покрывает {overlap:.1%} листа")

    # Реальная территория, присланная заказчиком (рамка_листов).
    coords = np.asarray(frame_wgs.geometry.iloc[0].exterior.coords)
    ax.add_patch(plt.Polygon(coords, closed=True, fill=True, facecolor="steelblue", alpha=0.15,
                              edgecolor="steelblue", linewidth=3.0, zorder=5, hatch="//"))

    # Покрытие грав-магнитки.
    gm_ring = grid_ring_lonlat(gm_meta, gm_proj4)
    ax.plot(gm_ring[:, 0], gm_ring[:, 1], color="magenta", linewidth=2.5, linestyle="--", zorder=6)

    # Основной расчётный лист (prognoz) — для ориентира.
    main_meta = read_pgrid(config.GOLD_TARGET_PGRID)
    main_ring = grid_ring_lonlat(main_meta, main_proj4)
    ax.plot(main_ring[:, 0], main_ring[:, 1], color="red", linewidth=2.0, zorder=6)

    handles = [
        Line2D([0], [0], color="steelblue", lw=3, label=f"территория, присланная заказчиком (рамка_листов, {frame_geom.area/1e6:.0f} км²)"),
        Line2D([0], [0], color="magenta", lw=2.5, ls="--", label=f"покрытие грав-магнитки (грав_маг.pgrid, {gm_box_native.area/1e6:.0f} км², {covered_frac:.1%} присланной территории)"),
        Line2D([0], [0], color="red", lw=2, label="основной расчётный лист (prognoz, R-48-XI,XII)"),
        Line2D([0], [0], color="steelblue", lw=1.5, label="листы ГГК-200, частично/полностью присланные\n(подпись «оцифровано геологии: N» — густота векторной геологии по листу)"),
        Line2D([0], [0], color="0.5", lw=1, label="листы ГГК-200 без данных заказчика (контекст)"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, framealpha=0.95,
                        title="Какие данные какую территорию покрывают:")
    legend.get_title().set_fontweight("bold")
    coverage_note = (
        "Космоснимки (Sentinel-1/2, Landsat-8, PALSAR-2, ASTER) и глобальная\n"
        "геофизика (EMAG2v3, гравика/VGG/EGM2008) — открытые источники, покрывают\n"
        "весь показанный на карте участок независимо от заказчика.\n"
        "Локальная грав-магнитка (пунктир) и векторная геология (плотность подписана\n"
        "у каждого листа) — данные заказчика, доступны ТОЛЬКО в присланной территории."
    )
    ax.annotate(coverage_note, xy=(1.01, 0.0), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=8.5, style="italic", color="0.25",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F2E3", edgecolor="0.6"))
    ax.set_xlabel("долгота, °в.д.")
    ax.set_ylabel("широта, °с.ш.")
    ax.set_title("Территория, присланная заказчиком (SBORKA_NESKOLKO_LISTOV), "
                  "vs покрытие грав-магнитной съёмки")
    ax.set_aspect(1 / np.cos(np.radians(71)))
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\nOK: {OUT}")


if __name__ == "__main__":
    main()
