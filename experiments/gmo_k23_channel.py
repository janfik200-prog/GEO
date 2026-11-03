"""Канал К23 предметной модели: доля ячейки 500 м в свитах из списка ВНИГНИ.

Цель: числовой канал для критерия К23 «ячейка в свите из списка ВНИГНИ»
(объект О11, звено «место накопления» этапа III; ответ ВНИГНИ 4 от
02.11.2026). Пять свит: ильинская, бурдурская, лабастахская, усть-ильинская,
котуйканская.

Входные данные:
* `data/processed/svita_labeled.gpkg` — свиты листа R-48-XI,XII
  (`experiments/gmo_svita_labels.py`: `svita_new` по легенде ГГК-200 и
  котуйканская со скана);
* сетка основного листа — `data/processed/dataset_v6.parquet` (row, col, x, y),
  500 м, проекция листа.

Как считается: полигоны переносятся на сетку в 10 раз мельче (50 м), доля —
среднее по блоку 10 x 10. Доля отнесена к той части ячейки, что покрыта
геологической картой (`svk_cover`): ячейки у рамки листа покрыты частично,
за рамкой — нет вовсе (там канал не определён, а не ноль).

Ожидаемый результат:
* `data/processed/k23_svity_grid.parquet` — row, col, x, y, `svk_cover`,
  доли пяти свит `svk_kt`, `svk_ul`, `svk_lb`, `svk_br`, `svk_il`, их сумма
  `svk_list` и преобладающая свита `svk_main`;
* `outputs/13_генетическая_модель/gmo_k23_channel.png` — карта канала.

Оговорка: канал описывает выход свиты на поверхность по геологической карте,
а не наличие свиты на глубине. Под четвертичными отложениями и долеритами
доля свит из списка меньше единицы, хотя свита под ними, скорее всего, есть.

Запуск::

    python -X utf8 -m experiments.gmo_k23_channel
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, mapstyle, outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SVITY = ROOT / "data/processed/svita_labeled.gpkg"
GRID = ROOT / "data/processed/dataset_v6.parquet"
OUT = ROOT / "data/processed/k23_svity_grid.parquet"
CELL, SUB = 500.0, 10

CODES = {"котуйканская": "svk_kt", "усть-ильинская": "svk_ul",
         "лабастахская": "svk_lb", "бурдурская": "svk_br", "ильинская": "svk_il"}


def fractions(svity: gpd.GeoDataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    from rasterio import features
    from rasterio.transform import from_origin

    nrow, ncol = int(grid.row.max()) + 1, int(grid.col.max()) + 1
    x0 = grid.x.min() - CELL / 2
    ytop = grid.y.max() + CELL / 2
    T = from_origin(x0, ytop, CELL / SUB, CELL / SUB)
    shape = (nrow * SUB, ncol * SUB)

    def frac(geoms) -> np.ndarray:
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros((nrow, ncol))
        r = features.rasterize(((g, 1) for g in geoms), out_shape=shape,
                               transform=T, fill=0, dtype="uint8")
        return r.reshape(nrow, SUB, ncol, SUB).mean(axis=(1, 3))

    cover = frac(svity.geometry)
    out = grid[["row", "col", "x", "y"]].copy()
    rc = (out.row.to_numpy(), out.col.to_numpy())
    out["svk_cover"] = cover[rc]
    with np.errstate(invalid="ignore", divide="ignore"):
        for sv, code in CODES.items():
            f = frac(svity.geometry[svity.svita == sv])
            out[code] = np.where(cover > 0, f / cover, np.nan)[rc]
    out["svk_list"] = out[list(CODES.values())].sum(axis=1, min_count=1)
    vals = out[list(CODES.values())].to_numpy()
    main = np.array(list(CODES))[np.nan_to_num(vals, nan=-1).argmax(1)]
    out["svk_main"] = np.where((out.svk_cover > 0) & (np.nan_to_num(vals).max(1) > 0),
                               main, "вне списка")
    out.loc[out.svk_cover == 0, "svk_main"] = "нет карты"
    return out


def draw(df: pd.DataFrame) -> Path:
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    ext = [df.x.min() - CELL / 2, df.x.max() + CELL / 2,
           df.y.min() - CELL / 2, df.y.max() + CELL / 2]
    a = np.full((nrow, ncol), np.nan)
    a[df.row, df.col] = df.svk_list
    b = np.full((nrow, ncol), np.nan)
    order = list(CODES) + ["вне списка"]
    b[df.row, df.col] = [order.index(s) if s in order else np.nan for s in df.svk_main]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6.6))
    im = axs[0].imshow(a, extent=ext, cmap="YlOrBr", vmin=0, vmax=1)
    fig.colorbar(im, ax=axs[0], shrink=0.8, label="доля открытой части ячейки, 0–1")
    axs[0].set_title("К23: доля ячейки в свитах из списка ВНИГНИ", fontsize=11)

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ["#7b241c", "#c0392b", "#e67e22", "#f1c40f", "#8e44ad", "#bdbdbd"]
    axs[1].imshow(b, extent=ext, cmap=ListedColormap(colors), vmin=-0.5, vmax=5.5,
                  interpolation="nearest")
    axs[1].legend(handles=[Patch(color=c, label=s) for c, s in zip(colors, order)],
                  loc="lower left", fontsize=8, framealpha=0.9)
    axs[1].set_title("Преобладающая свита в ячейке", fontsize=11)
    for ax in axs:
        ax.set_xlabel("X, м (Гаусс–Крюгер, осевой 105°)")
        ax.set_ylabel("Y, м")
        ax.ticklabel_format(style="plain")
        ax.tick_params(labelsize=7)
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м: канал свит рифея (О11, звено «место накопления»)",
                 fontsize=12)
    out = outputs.png("gmo_k23_channel.png")
    mapstyle.save_fig(fig, out, dpi=140)
    return out


def main() -> None:
    svity = gpd.read_file(SVITY).to_crs(anabar_inset.SHEET_CRS)
    grid = pd.read_parquet(GRID, columns=["row", "col", "x", "y"])
    df = fractions(svity, grid)
    df.to_parquet(OUT, index=False)

    covered = df[df.svk_cover > 0]
    print(f"Ячеек сетки {len(df)}, покрыто картой {len(covered)} "
          f"(полностью {int((df.svk_cover > 0.999).sum())})")
    print("Средняя доля по покрытым ячейкам:")
    print(covered[list(CODES.values()) + ["svk_list"]].mean().round(3).to_string())
    print("\nПреобладающая свита, ячеек:")
    print(df.svk_main.value_counts().to_string())
    print(f"\nКарта: {draw(df)}")
    print(f"Канал: {OUT}")


if __name__ == "__main__":
    main()
