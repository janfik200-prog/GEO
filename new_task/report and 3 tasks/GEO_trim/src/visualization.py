"""Визуализация: классы отображения и итоговая карта прогноза (PNG)."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch

from . import config
from .utils import robust_normalize_01


def make_display_classes(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Разбить ``prospectivity`` на :data:`config.N_DISPLAY_CLASSES` классов для карты."""
    disp = robust_normalize_01(grid["prospectivity"].to_numpy(), 0.02, 0.98)
    grid["display_score"] = disp
    bins = np.linspace(0, 1, config.N_DISPLAY_CLASSES + 1)
    grid["display_class"] = np.digitize(disp, bins[1:-1], right=False)
    return grid


def _set_mask_extent(ax, mask: gpd.GeoDataFrame) -> None:
    """Выставить пределы осей по границам маски с небольшим отступом."""
    minx, miny, maxx, maxy = mask.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)


def plot_final(
    grid: gpd.GeoDataFrame,
    mask: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame | None,
    out_png,
) -> None:
    """Отрисовать карту прогноза с золотыми зонами и сохранить PNG в ``out_png``."""
    fig, ax = plt.subplots(figsize=(10, 10))
    bins = np.arange(config.N_DISPLAY_CLASSES + 1)
    norm = BoundaryNorm(bins, plt.cm.bwr_r.N)
    grid.plot(column="display_class", ax=ax, cmap="bwr_r", norm=norm, linewidth=0, legend=False)

    gold = grid[grid["gold_zone"] == 1]
    if len(gold) > 0:
        gold.plot(ax=ax, color="#f2d200", linewidth=0)

    mask.boundary.plot(ax=ax, color="black", linewidth=0.5)

    if config.SHOW_POINTS and points is not None and len(points) > 0:
        points.plot(ax=ax, color="yellow", markersize=8, edgecolor="black", linewidth=0.25)

    ax.legend(
        handles=[Patch(facecolor="#f2d200", edgecolor="black", label="Gold zone")],
        loc="lower right", frameon=True,
    )
    _set_mask_extent(ax, mask)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uncertainty(grid: gpd.GeoDataFrame, mask: gpd.GeoDataFrame, out_png) -> None:
    """Карта неопределённости прогноза (разброс деревьев RF)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    grid.plot(column="uncertainty", ax=ax, cmap="viridis", linewidth=0, legend=True,
              legend_kwds={"label": "Неопределённость (разброс деревьев)", "shrink": 0.6})
    mask.boundary.plot(ax=ax, color="white", linewidth=0.5)
    _set_mask_extent(ax, mask)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_success_rate(curve_df, out_png) -> None:
    """Построить success-rate кривые (покрытие точек vs доля площади) по моделям.

    ``curve_df`` — результат :func:`src.validation.success_rate_curve`. Диагональ —
    случайный baseline; чем выше кривая, тем лучше модель.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, sub in curve_df.groupby("model"):
        sub = sub.sort_values("area")
        style = "--" if model == "Random" else "-"
        ax.plot(sub["area"], sub["coverage"], style, marker="o", markersize=3, label=model)
    ax.plot([0, 0.5], [0, 0.5], ":", color="gray", linewidth=1, label="Случайный (диагональ)")
    ax.set_xlabel("Доля обследованной площади")
    ax.set_ylabel("Доля найденных рудопроявлений")
    ax.set_title("Success-rate (пространственная CV)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
