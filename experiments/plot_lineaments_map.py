"""Карта линеаментов листа: наши линии по рельефу против разломов организации.

Зачем. В проекте под словом «линеаменты» живут ТРИ разные вещи, и их постоянно
путают при чтении признаков:

1. `lin_*` (6 признаков) — линии, которые считаем МЫ по рельефу
   (Copernicus DEM GLO-30 -> отмывка по 8 азимутам -> Кэнни -> вероятностное
   преобразование Хафа, `src/lineaments.py`). Это автоматика, не карта геолога.
2. `lin_sat_*` — тот же конвейер напрямую по космоснимкам (Sentinel-2,
   Landsat 8/9, Sentinel-1, ALOS PALSAR-2, `src/lineaments_sat.py`). Ветка
   закрыта стоп-правилом воспроизводимости, в датасеты не вошла.
3. `glub_raz_nw` / `glub_r_nw` — ГЛУБИННЫЕ РАЗЛОМЫ СЗ и СВ из шейп-файлов
   организации (ГИС Интегро). Это интерпретация геолога, а не линеаменты;
   именно из них считаются `dist_tect1`, `dist_tect2`, `dens_tect` и
   `geo2_fault_node` — прямые входы критериальной формулы.

Карта показывает 1 и 3 в одних координатах, чтобы было видно: автоматика
рисует густую сеть бровок долин, а карта геолога — редкие протяжённые линии
двух направлений. Розы азимутов рядом отвечают на вопрос, совпадают ли
направления, а не только положение.

Выход: `outputs/lineaments_vs_faults_map.png`.
Запуск из корня: ``python -m experiments.plot_lineaments_map``.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from src import (assessment, cell_mask, config, criterial_target,  # noqa: E402
                 data_loader, integro_grid, lineaments, terrain_v2)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def world_to_px(x, y, meta) -> tuple[np.ndarray, np.ndarray]:
    """Метровые координаты слоя -> пиксели сетки (тот же кадр, что у imshow)."""
    return ((np.asarray(x) - meta.x0) / meta.dx - 0.5,
            (meta.y_top - np.asarray(y)) / meta.dy - 0.5)


def line_segments(gdf, meta) -> list:
    """Геометрии слоя -> ломаные в пикселях сетки.

    Слои разломов организации оказались ПОЛИГОНАМИ (зоны влияния, а не осевые
    линии), поэтому у площадных геометрий берётся контур.
    """
    segs = []
    for geom in gdf.geometry.values:
        g = geom.boundary if geom.geom_type in ("Polygon", "MultiPolygon") else geom
        parts = [g] if g.geom_type == "LineString" else list(g.geoms)
        for p in parts:
            xs, ys = np.asarray(p.coords).T
            px, py = world_to_px(xs, ys, meta)
            segs.append(np.column_stack([px, py]))
    return segs


def azimuths_deg(segs_xy) -> np.ndarray:
    """Азимуты отрезков в 0..180 град. (у линии нет направления), вес — длина."""
    a, w = [], []
    for s in segs_xy:
        d = np.diff(np.asarray(s), axis=0)
        for dx, dy in d:
            a.append(np.degrees(np.arctan2(dx, -dy)) % 180.0)   # 0 = север
            w.append(float(np.hypot(dx, dy)))
    return np.asarray(a), np.asarray(w)


def rose(ax, ang, w, color, title) -> None:
    """Роза-диаграмма азимутов, симметричная (линия = ось, а не вектор)."""
    bins = np.arange(0, 181, 10)
    h, _ = np.histogram(ang, bins=bins, weights=w)
    h = h / max(h.sum(), 1e-9)
    theta = np.radians(bins[:-1] + 5)
    ax.bar(np.concatenate([theta, theta + np.pi]), np.concatenate([h, h]),
           width=np.radians(10), color=color, edgecolor="none", alpha=0.85)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10)


def main() -> None:
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    shape = (meta.prf, meta.pic)
    elev = terrain_v2.projected_dem(meta)
    elev = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    log(f"DEM в проекции сетки {elev.shape}, шаг {config.TER_RES_M:.0f} м; "
        f"сетка {shape}, шаг {meta.dx:.0f} м")

    segments = lineaments.extract_lines(lineaments.edge_map(elev))
    pad, k = config.TER_PAD_PX, config.TER_RES_M / meta.dx
    own = [np.array([[(x0 - pad) * k, (y0 - pad) * k],
                     [(x1 - pad) * k, (y1 - pad) * k]])
           for (x0, y0), (x1, y1) in segments]
    lens_km = np.array([np.hypot(s[1, 0] - s[0, 0], s[1, 1] - s[0, 1])
                        for s in own]) * meta.dx / 1000.0
    log(f"наших отрезков по рельефу: {len(own)}, суммарно {lens_km.sum():.0f} км, "
        f"медиана {np.median(lens_km):.1f} км")

    shp = config.GOLD_TARGET_PGRID.parents[1] / config.SHP_SUBDIR
    faults = {}
    for role, color, name in (("tect1", "#ff3b30", "разломы СЗ (glub_raz_nw)"),
                              ("tect2", "#0a84ff", "разломы СВ (glub_r_nw)")):
        gdf = data_loader.load_layer(shp / f"{config.LAYER_FILES[role]}.shp")
        faults[role] = (line_segments(gdf, meta), color, name, gdf)
        log(f"{name}: объектов {len(gdf)}, тип {sorted(set(gdf.geom_type))}, "
            f"площадь {gdf.area.sum() / 1e6:.0f} км2")

    valid = cell_mask.build_valid_mask(criterial_target.training_features("v5_rebuilt"))
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    pr, pc = np.divmod(unb, meta.pic)

    fig = plt.figure(figsize=(17.5, 7.4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(lineaments.hillshade(elev, 315.0)[pad:-pad or None, pad:-pad or None],
               cmap="gray", extent=(-0.5, meta.pic - 0.5, meta.prf - 0.5, -0.5))
    ax0.add_collection(LineCollection(own, colors="#ffcc00", linewidths=0.5))
    ax0.set_xlim(-0.5, meta.pic - 0.5)
    ax0.set_ylim(meta.prf - 0.5, -0.5)
    ax0.set_title(f"Наши линеаменты по рельефу: {len(own)} отрезков\n"
                  "Copernicus DEM GLO-30, отмывка по 8 азимутам")

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(np.where(valid, 1.0, np.nan).reshape(shape), cmap="Greys", vmin=0, vmax=8,
               origin="upper")
    ax1.add_collection(LineCollection(own, colors="#5a5a5a", linewidths=0.45))
    for segs, color, name, gdf in faults.values():
        ax1.add_collection(LineCollection(segs, colors=color, linewidths=1.8,
                                          label=name))
        for poly in gdf.geometry.values:
            xs, ys = np.asarray(poly.exterior.coords).T
            ax1.fill(*world_to_px(xs, ys, meta), color=color, alpha=0.12, lw=0)
    ax1.scatter(pc, pr, s=22, c="white", edgecolors="black", linewidths=0.6,
                zorder=5, label=f"точки заверки, непредвзятые ({unb.size})")
    ax1.set_xlim(-0.5, meta.pic - 0.5)
    ax1.set_ylim(meta.prf - 0.5, -0.5)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_title("Наши линеаменты (серым) против зон разломов организации\n"
                  "зоны — вход критериальной формулы, линеаменты — нет")

    ax2 = fig.add_subplot(2, 3, 3, projection="polar")
    a, w = azimuths_deg(own)
    rose(ax2, a, w, "#ffa000", f"Азимуты наших линеаментов\n(взвешено длиной, "
                               f"{lens_km.sum():.0f} км)")
    ax3 = fig.add_subplot(2, 3, 6, projection="polar")
    fa, fw = azimuths_deg(faults["tect1"][0] + faults["tect2"][0])
    rose(ax3, fa, fw, "#d32f2f", "Азимуты зон разломов организации\n"
                                 "(по контурам полигонов СЗ + СВ)")

    for ax in (ax0, ax1):
        ax.set_xlabel("столбец сетки (шаг 500 м)")
    ax0.set_ylabel("строка сетки")
    fig.suptitle("Линеаменты листа: автоматика по рельефу и разломы с карты организации",
                 fontsize=13)
    fig.tight_layout()
    out = ROOT / "outputs" / "lineaments_vs_faults_map.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    log(f"OK: {out}")


if __name__ == "__main__":
    main()
