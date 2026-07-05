"""Zn-Pb по переносу — на ТОЙ ЖЕ проектной сетке Анабара (лист R-48, 500 м, маска
свит, проекция Красовского), как наши карты прогноза.

Модель Zn-Pb обучается на США+Канаде в общем пространстве глобальной геофизики
(mag4km, faa, vgg, geoid, relief) и применяется к ячейкам проектной сетки Анабара.
"""

import tempfile, warnings, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier

from src import config
from src.data_loader import find_base_dir, load_all_layers, load_layer, to_crs_safe
from src.features import build_grid
from src.model import mark_presence
from experiments.common import region_xy, DS, SCALE
from cache_paths import ANABAR_GEO, CMMI_OCC

GEO = str(ANABAR_GEO)
EXT = (60.0, 120.0, 30.0, 90.0)   # тайл N30E060


def sample_grid(ds, lon, lat):
    a = np.array(Image.open(f"{GEO}/{ds}_N30E060.jp2")).astype(float)
    a[a == 0] = np.nan; a *= SCALE[ds]
    n = a.shape[0]; dx = (EXT[1] - EXT[0]) / n
    c = ((lon - EXT[0]) / dx).astype(int); r = ((EXT[3] - lat) / dx).astype(int)
    ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
    out = np.full(lon.shape, np.nan); out[ok] = a[r[ok], c[ok]]
    return out


def main():
    t0 = time.time()
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid, pos = mark_presence(grid, points)
    crs = layers["mask"].crs

    # признаки на сетке Анабара (репроекция центров в lon/lat) — тот же порядок DS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cents = grid.geometry.centroid.to_crs(4326)
    glon, glat = cents.x.to_numpy(), cents.y.to_numpy()
    Xg = np.column_stack([sample_grid(d, glon, glat) for d in DS])

    # обучаем Zn-Pb на США+Канаде в том же пространстве
    occ = pd.read_csv(str(CMMI_OCC), encoding="latin1", low_memory=False)
    Xu, yu, _, _ = region_xy(occ, ["United States of America", "Canada"], (-170, -52), (25, 75), 42)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=42).fit(Xu, yu)
    score = np.full(len(grid), np.nan)
    ok = np.isfinite(Xg).all(1)
    score[ok] = gb.predict_proba(Xg[ok])[:, 1]
    print(f"[{time.time()-t0:3.0f}s] обучено на {int(yu.sum())} Zn-Pb; сетка {len(grid)} ячеек; "
          f"Zn-Pb score min/max {np.nanmin(score):.2f}/{np.nanmax(score):.2f}, среднее {np.nanmean(score):.2f}")

    # растеризация на проектную сетку (Красовский, метры)
    rows = grid["row"].to_numpy(); cols = grid["col"].to_numpy()
    arr = np.full(shape, np.nan); arr[rows, cols] = score
    minx, miny, maxx, maxy = layers["mask"].total_bounds
    extent = [minx, minx + shape[1] * config.CELL_SIZE, miny, miny + shape[0] * config.CELL_SIZE]
    svita = layers["mask"]

    fig, ax = plt.subplots(figsize=(8, 8.4))
    im = ax.imshow(np.ma.masked_invalid(arr), origin="lower", extent=extent,
                   cmap="YlOrRd", vmin=0, vmax=float(np.nanmax(score)), interpolation="nearest")
    svita.boundary.plot(ax=ax, color="0.4", linewidth=0.3, alpha=0.5)
    real = grid[grid["cell_id"].isin(pos)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = real.geometry.centroid
    ax.scatter(rc.x, rc.y, s=16, facecolor="none", edgecolor="blue", linewidths=0.8,
               label="известные Au-U точки")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Zn-Pb фертильность (перенос США+Канада)")
    ax.set_title("Анабар, лист R-48 · Zn-Pb по переносу модели США+Канада\n"
                 "(проектная сетка 500 м, та же, что и карты прогноза)", fontsize=10)
    ax.set_xlabel("X, м (Гаусса–Крюгера, Красовский)"); ax.set_ylabel("Y, м")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    fig.savefig("outputs/anabar_znpb_sheet.png", dpi=150, bbox_inches="tight")
    print(f"[{time.time()-t0:3.0f}s] saved anabar_znpb_sheet.png")


if __name__ == "__main__":
    main()
