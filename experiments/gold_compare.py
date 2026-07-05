"""Сравнение «золота» наш ML vs ГИС Интегро при ОДИНАКОВОМ пороге.

ГИС Интегро не строит морфологический фильтр золотых зон — высший класс
перспективности это просто верхние N% площади по критериальному баллу. Чтобы
сравнить методы честно (яблоко к яблоку), берём ОДИН простой фильтр для обоих:

  «золото» = верхние q% площади по баллу (без локальных пиков, связности и
  геологических И-условий боевого mark_gold_zones).

Скоринг:
  * ГИС Интегро — критериальный geo_score (src.features.compute_geo_score),
    детерминирован, точек не видит;
  * наш ML — HELD-OUT (out-of-fold) перцентиль: модель оценивает ячейку, не
    видя её блок (иначе сравнение нечестно — ML «подсветил» бы свои точки).

При равной площади золота метод лучше, если в неё попало больше реальных точек.

Запуск: python3 -m experiments.gold_compare
"""

import sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, compute_geo_score
from src.model import mark_presence
from src.validation import assign_spatial_blocks
from experiments.forecast_oof import out_of_fold_scores

GOLD_LEVELS = (0.03, 0.05, 0.10)   # доли площади под «золото» (top-q%)
MAP_LEVEL = 0.03                    # порог для карты (как боевой GOLD_Q_BEST=0.97)


def gold_mask(score: np.ndarray, q: float) -> np.ndarray:
    """Верхние q% площади по баллу (NaN -> не золото)."""
    thr = float(np.nanquantile(score, 1.0 - q))
    return np.nan_to_num(score, nan=-np.inf) >= thr


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    mask = layers["mask"]
    grid, _, grid_shape = build_grid(mask, config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = compute_geo_score(grid, grid_shape)
    grid, _ = mark_presence(grid, points)

    presence = grid["presence"].to_numpy().astype(int)
    n_pos = int(presence.sum())
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)

    geo = grid["geo_score"].to_numpy()                 # ГИС Интегро (точек не видит)
    oof = out_of_fold_scores(grid, blocks)             # наш ML, held-out перцентиль
    grid["oof"] = oof

    print(f"Точек: {n_pos} | ячеек: {len(grid)}\n")
    print(f"{'порог':>7}{'площадь':>9}{'  ГИС Интегро (точек / lift)':>30}{'  наш ML held-out (точек / lift)':>32}")
    print("-" * 78)
    scores = {"ГИС Интегро": geo, "наш ML": oof}
    for q in GOLD_LEVELS:
        cells = int(round(q * len(grid)))
        row = f"{int(q*100):>6}%{cells:>9}"
        for name in ("ГИС Интегро", "наш ML"):
            gm = gold_mask(scores[name], q)
            hit = int((gm & (presence == 1)).sum())
            cov = hit / n_pos
            lift = cov / q
            row += f"{hit:>13} / {lift:>4.2f}x"
        print(row)
    print("-" * 78)
    print("Площадь золота одинакова у обоих -> кто поймал больше точек, тот лучше.")

    # --- карта: две панели при MAP_LEVEL ---
    gm_geo = gold_mask(geo, MAP_LEVEL)
    gm_ml = gold_mask(oof, MAP_LEVEL)
    pts = gpd.sjoin(points[["geometry"]].reset_index(drop=True), grid[["geometry"]],
                    how="inner", predicate="within")
    cell_of_pt = pts["index_right"].to_numpy()
    geom_of_pt = pts.geometry.values

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, (name, gm) in zip(axes, (("ГИС Интегро (критериальный)", gm_geo),
                                      ("Наш ML (held-out)", gm_ml))):
        grid.plot(ax=ax, color="#eeeeee", linewidth=0)
        grid[gm].plot(ax=ax, color="#f2d200", linewidth=0)
        mask.boundary.plot(ax=ax, color="black", linewidth=0.5)
        in_gold = np.isin(cell_of_pt, np.where(gm)[0])   # раскраска точек
        gpd.GeoSeries(geom_of_pt[in_gold], crs=points.crs).plot(
            ax=ax, color="red", markersize=40, edgecolor="black", linewidth=0.5, zorder=5)
        gpd.GeoSeries(geom_of_pt[~in_gold], crs=points.crs).plot(
            ax=ax, facecolor="none", markersize=40, edgecolor="black", linewidth=0.5, zorder=5)
        hit = int((gm & (presence == 1)).sum())          # метрика по ячейкам-точкам (= таблица)
        ax.set_title(f"{name}\nзолото = top-{int(MAP_LEVEL*100)}% площади · ячеек-точек в золоте: "
                     f"{hit}/{n_pos} (lift {hit/n_pos/MAP_LEVEL:.2f}x)", fontsize=12)
        ax.set_axis_off()
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="red",
               markeredgecolor="black", markersize=9, label="точка в золоте"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=9, label="точка вне золота"),
    ], loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "gold_compare.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nСохранено: {out_png}")


if __name__ == "__main__":
    main()
