"""Генетическая модель ВНИГНИ v2 на широкой территории: признак — отдельная карта.

Цель: посмотреть, как каналы, которыми наблюдаются элементы ГМО v2, ложатся
на большую территорию (5 листов R-48 вокруг основного), а не только на
основной лист. Каждый признак — отдельный PNG поверх геологической подложки
ГГК-200, с одинаковыми рамками и легендой.

Входные данные:
* `data/processed/ast_wide_v3.parquet` — ASTER на той же сетке, сборка v3
  (заменяет ASTER из `dataset_wide`, где основной лист почти не покрыт);
* `data/processed/dataset_wide.parquet` — каналы на широкой сетке 500 м
  (`WIDE_META`, 299 × 303 ячейки, листы R-48-V,VI / IX,X / XI,XII / XV,XVI /
  XVII,XVIII);
* многолистовая сетка ГИС Интегро (`criterial_reconstruct_multisheet`) — слои
  критериальной формулы за пределами основного листа (полоса 105–108° в.д.;
  западнее 105° слоёв нет);
* `datacache/ggk200` — подложка; `data/external/anabar_shield` — граница щита;
* `data/Gis-integro/shp_dbf/геохимические ореолы.shp` — ореолы Au (основной
  лист), `data/processed/sw_shlikh_points.csv` — шлихи с золотом на юго-западе
  (только заверка).

Ожидаемый результат: `outputs/13_генетическая_модель/gmo_wide_*.png` и сводка
покрытия каждого канала по территории в выводе скрипта.

Слои формулы — «⚠ с осторожностью» (решение 21.10.2026). Спектры (ASTER,
Sentinel-2) — только на обнажённых ячейках. Уран не показывается.

Запуск::

    python -m experiments.gmo_v2_wide
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
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import shapely  # noqa: E402

from src import (anabar_inset, ggk200_base, mapstyle, optical_derived,  # noqa: E402
                 outputs, spectral_mask)
from experiments.criterial_reconstruct_multisheet import load_layers  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WIDE = ROOT / "data" / "processed" / "dataset_wide.parquet"
# ASTER v3 для широкой сетки: покрытие 70% против 39% в dataset_wide, тот же
# порядок ячеек (проверено: старый ast_wide совпадает с колонками dataset_wide)
AST_V3 = ROOT / "data" / "processed" / "ast_wide_v3.parquet"
SHP = ROOT / "data" / "Gis-integro" / "shp_dbf"
SHLIKH = ROOT / "data" / "processed" / "sw_shlikh_points.csv"
BOUNDS = (103.85, 69.97, 108.25, 71.40)
CAUTION = "#c0392b"
CELL = 500.0

TO_LL = Transformer.from_crs(anabar_inset.SHEET_CRS, "EPSG:4326", always_xy=True)

# (файл, элемент ГМО, канал, подпись шкалы, cmap, только обнажённые)
CHANNELS = [
    ("01_I2_magnit_ost", "I.2 Интрузивные комплексы", "gm_mag_ost_35",
     "остаточное магнитное поле, отн. ед.", "RdBu_r", False),
    ("02_I2_magnit_asig", "I.2 Интрузивные комплексы", "pf_mag_asig",
     "аналитический сигнал магнитного поля, отн. ед.", "magma", False),
    ("03_I3_lineamenty", "I.3 Глубинные разломные зоны", "lin_dens",
     "плотность линеаментов, км/км²", "magma", False),
    ("04_I3_gravika_grad", "I.3 Глубинные разломные зоны", "gm_gr_1G_25",
     "горизонтальный градиент гравитационного поля, отн. ед.", "viridis", False),
    ("05_I4_diaftorez_mgoh", "I.4 Зоны диафтореза (хлорит, эпидот)", "ast_mgoh",
     "ASTER Mg-OH 2.33 мкм, отношение", "viridis", True),
    ("06_II34_izmeneniya_aster", "II.3–4 Реактивация, выщелачивание", "ast_alter",
     "ASTER индекс изменений, отношение", "viridis", True),
    ("07_II34_okisly_fe_s2", "II.3–4 Реактивация, выщелачивание", "s2_iron_ox",
     "Sentinel-2 окислы железа, отношение", "YlOrBr", True),
    ("08_II5_anizotropiya", "II.5 Пликативные дислокации", "lin_aniso_r",
     "анизотропия линеаментов, 0–1", "cividis", False),
    ("09_II6_kaolinit", "II.6 Коры выветривания", "ast_kaolin",
     "ASTER каолинит, отношение", "viridis", True),
    ("10_III78_rel_polozhenie", "III.7–8 Алевропелиты и грубообломочные", "dem_tpi_2km",
     "положение в рельефе (TPI 2 км), м", "BrBG", False),
    ("11_III78_rassechennost", "III.7–8 Алевропелиты и грубообломочные", "relief2_tri_std",
     "изрезанность рельефа, м", "cividis", False),
    ("12_III78_radar", "III.7–8 Алевропелиты и грубообломочные", "psr_hh",
     "радар ALOS PALSAR HH, дБ", "Greys_r", False),
    ("13_sohrannost_vysota", "Сохранность и несогласие (нет в v2)", "dem_elev",
     "высота, м", "terrain", False),
]

# (файл, элемент ГМО, слои формулы {роль: подпись}, цвет)
FORMULA = [
    ("14_I2_formula_dayki", "I.2 Интрузивные комплексы",
     {"magm": "дайки"}),
    ("15_I3_formula_razlomy", "I.3 Глубинные разломные зоны",
     {"tect1": "разломные зоны СЗ", "tect2": "разломные зоны СВ"}),
    ("16_II6_formula_kory", "II.6 Коры выветривания",
     {"struct": "коры выветривания"}),
    ("17_III78_formula_facii_paleo", "III.7–8 Алевропелиты и грубообломочные",
     {"facies": "благоприятные фации", "paleo": "палеодолины"}),
]
FORMULA_STYLE = {"tect1": ("#c0392b", "//"), "tect2": ("#e67e22", "\\\\"),
                 "magm": ("#8e44ad", "xx"), "struct": ("#6c3483", ".."),
                 "facies": ("#c0392b", "//"), "paleo": ("#1f618d", "\\\\")}


def regular_mesh(x: np.ndarray, y: np.ndarray, v: np.ndarray):
    """Значения на правильной сетке -> (lon, lat) углов и 2D массив."""
    xs, ys = np.unique(x), np.unique(y)
    img = np.full((ys.size, xs.size), np.nan)
    img[np.searchsorted(ys, y), np.searchsorted(xs, x)] = v
    xe = np.r_[xs - CELL / 2, xs[-1] + CELL / 2]
    ye = np.r_[ys - CELL / 2, ys[-1] + CELL / 2]
    gx, gy = np.meshgrid(xe, ye)
    lon, lat = TO_LL.transform(gx, gy)
    return lon, lat, img


def outline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Рамка сетки в lon/lat (по крайним ячейкам)."""
    x0, x1 = x.min() - CELL / 2, x.max() + CELL / 2
    y0, y1 = y.min() - CELL / 2, y.max() + CELL / 2
    t = np.linspace(0, 1, 50)
    xs = np.r_[x0 + (x1 - x0) * t, np.full(50, x1), x1 - (x1 - x0) * t, np.full(50, x0)]
    ys = np.r_[np.full(50, y1), y1 - (y1 - y0) * t, np.full(50, y0), y0 + (y1 - y0) * t]
    return np.column_stack(TO_LL.transform(xs, ys))


def zone_polygon(x: np.ndarray, y: np.ndarray) -> gpd.GeoSeries:
    """Ячейки зоны -> один (мульти)полигон в lon/lat."""
    boxes = shapely.box(x - CELL / 2, y - CELL / 2, x + CELL / 2, y + CELL / 2)
    geom = shapely.union_all(boxes).simplify(50)
    return gpd.GeoSeries([geom], crs=anabar_inset.SHEET_CRS).to_crs(4326)


class Context:
    def __init__(self):
        self.wide = pd.read_parquet(WIDE)
        ast = pd.read_parquet(AST_V3)
        for c in ("ast_kaolin", "ast_alter"):
            self.wide[c] = ast[c].to_numpy()
        self.wide["ast_mgoh"] = optical_derived.aster_mgoh_index(ast)
        self.bare = spectral_mask.exposure_mask(self.wide)
        self.wide_ring = outline(self.wide.x.to_numpy(), self.wide.y.to_numpy())
        meta, self.formula, msk, _, fx, fy = load_layers()
        self.fx, self.fy = fx, fy
        self.formula_ring = outline(fx, fy)
        self.shield = gpd.read_file(anabar_inset.SRC / "anabar_shield.gpkg")
        halos = ggk200_base.to_lonlat(gpd.read_file(SHP / "геохимические ореолы.shp"))
        self.halos = halos[halos.label == "Au"]
        sh = pd.read_csv(SHLIKH, sep=";", encoding="utf-8-sig")
        self.shlikh = sh[sh["элемент"].str.contains("Au")]
        self.main = anabar_inset.sheet_lonlat()


def base(ctx: Context, title: str, sub: str):
    fig, ax = plt.subplots(figsize=(13, 9.6), constrained_layout=True)
    ggk200_base.draw(ax, bounds=BOUNDS, decimate=6, alpha=0.55)
    return fig, ax


def decorate(fig, ax, ctx: Context, title: str, sub: str, extra=(),
             caution: bool = False):
    ctx.shield.boundary.plot(ax=ax, color="black", linewidth=1.3, linestyle="--",
                             zorder=6)
    ax.plot(*ctx.wide_ring.T, color="#17a589", linewidth=1.6, zorder=6)
    gpd.GeoSeries([ctx.main]).boundary.plot(ax=ax, color="red", linewidth=1.8,
                                            zorder=6)
    ctx.halos.plot(ax=ax, color="gold", linewidth=2.6, zorder=7)
    ax.scatter(ctx.shlikh["долгота"], ctx.shlikh["широта"], marker="*", s=110,
               color="gold", edgecolor="black", linewidth=0.6, zorder=8)
    handles = list(extra) + [
        Line2D([0], [0], color="black", lw=1.3, ls="--", label="граница щита (1:5 млн)"),
        Line2D([0], [0], color="#17a589", lw=1.6, label="широкая сетка (5 листов)"),
        Line2D([0], [0], color="red", lw=1.8, label="основной лист R-48-XI,XII"),
        Line2D([0], [0], color="gold", lw=2.6, label="ореолы Au — только заверка"),
        Line2D([0], [0], marker="*", ls="", color="gold", markeredgecolor="black",
               ms=11, label="шлихи с Au (юго-запад) — только заверка"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.92,
                    title="⚠ слой формулы — с осторожностью" if caution else None,
                    title_fontsize=8)
    leg.set_zorder(20)
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    ax.set_xlabel("долгота, ° в.д.")
    ax.set_ylabel("широта, ° с.ш.")
    ax.set_title(f"{title}\n{sub}", fontsize=12)
    anabar_inset.draw(ax.inset_axes([-0.34, 0.66, 0.3, 0.34]), fontsize=6,
                      title=False, legend=False)


def channel_map(ctx: Context, fname, title, col, unit, cmap, spectral):
    w = ctx.wide
    v = w[col].to_numpy(dtype=float).copy()
    measured = np.isfinite(v)
    if spectral:
        v[~ctx.bare] = np.nan
    if col == "lin_aniso_r":
        v[w["lin_dens"].to_numpy() <= 0] = np.nan
    cover = np.isfinite(v).mean()
    lon, lat, img = regular_mesh(w.x.to_numpy(), w.y.to_numpy(), v)
    fig, ax = base(ctx, title, col)
    lo, hi = np.nanpercentile(img, [2, 98])
    m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(img), cmap=cmap,
                      vmin=lo, vmax=hi, alpha=0.8, shading="flat", zorder=2,
                      rasterized=True)
    cb = fig.colorbar(m, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label(unit)
    sub = f"канал {col}: значения на {cover:.0%} широкой сетки"
    if spectral:
        sub += (f"\nсъёмка есть на {measured.mean():.0%}; показаны только обнажённые ячейки, "
                "спектр под растительностью снят")
    decorate(fig, ax, ctx, title, sub)
    mapstyle.save_fig(fig, outputs.png(f"gmo_wide_{fname}.png"), dpi=130)
    plt.close(fig)
    return cover


def formula_map(ctx: Context, fname, title, roles: dict):
    fig, ax = base(ctx, title, "")
    extra = []
    for role, label in roles.items():
        zone = ctx.formula[role] == 0
        color, hatch = FORMULA_STYLE[role]
        zone_polygon(ctx.fx[zone], ctx.fy[zone]).plot(
            ax=ax, facecolor=color, edgecolor=color, hatch=hatch, alpha=0.35,
            linewidth=0.8, zorder=3)
        extra.append(Patch(facecolor=color, alpha=0.35, edgecolor=color, hatch=hatch,
                           label=f"{label} ⚠"))
    ax.plot(*ctx.formula_ring.T, color=CAUTION, linewidth=1.3, linestyle=":", zorder=6)
    extra.append(Line2D([0], [0], color=CAUTION, lw=1.3, ls=":",
                        label="где слои формулы есть (многолистовая сборка)"))
    decorate(fig, ax, ctx, title,
             "слой критериальной формулы: есть только в многолистовой сборке "
             "(пунктир, 105–108° в.д.); западнее 105° слоёв формулы нет", extra, caution=True)
    mapstyle.save_fig(fig, outputs.png(f"gmo_wide_{fname}.png"), dpi=130)
    plt.close(fig)


def main():
    ctx = Context()
    print(f"обнажено (s2_bare_frac ≥ {spectral_mask.BARE_MIN}): {ctx.bare.mean():.0%} широкой сетки")
    for row in CHANNELS:
        cov = channel_map(ctx, *row)
        print(f"{row[0]:<30} {row[2]:<18} покрытие {cov:.0%}")
    for row in FORMULA:
        formula_map(ctx, *row)
        print(f"{row[0]:<30} слой формулы")
    print(f"карты: {outputs.png('gmo_wide_x.png').parent}")


if __name__ == "__main__":
    main()
