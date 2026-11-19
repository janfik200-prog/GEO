"""Рисунки Пакета 01 ТЗ К1: реальные окна, источники данных, проверка геометрий.

Окна выбираются по правилу ``audit/SELECTION_RULE.md`` (записано до
построения): линии при выборе не участвуют. Линии на рисунках берутся из
векторного экспорта (``outputs/k1/vectors/*.gpkg``,
:mod:`openlin_export_vectors`), а не из маски: рисунок заодно проверяет
экспорт. Подложка — зимний B04, растяжка 2-98 % по всему листу.

Выходы (``outputs/k1/figures/``):
* fig_anabar_optical_dem.png / .pdf — три окна Анабара, три панели;
* fig_lapland_auto_gtk.png / .pdf — два окна Лапландии, три панели;
* fig_windows_overview.png — где лежат окна на листах;
* fig_data_sources.png / .pdf — покрытие источников;
* geometry_check_<site>.png — 20 случайных линий: ячейки и вектор;
* selection_manifest.csv, а также audit/geometry_check_<site>.csv.

Запуск из корня:
    python -m experiments.openlin_k1_figures
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COMP = Path("datacache/s2_lowsun/composite")
VEC = Path("outputs/k1/vectors")
FIG = Path("outputs/k1/figures")
AUDIT = Path("audit")

# ---- правило выбора (audit/SELECTION_RULE.md) -------------------------------
SEED = 20261026
WIN = 250                   # ячеек 20 м = 5 км
STEP = 125                  # шаг центров 2.5 км
Q_LOW, Q_HIGH = 0.25, 0.75  # квартили медианного уклона окон
N_GEOM_CHECK = 20
# ----------------------------------------------------------------------------

RED, BLUE, YEL = "#e0141e", "#1a66f2", "#ffd21a"


def load_site(site):
    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    res, x0, ytop, epsg = float(w["res_m"]), float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    b04 = w["B04"].astype("float32")
    valid = np.isfinite(b04)
    extra = {}
    if site == "anabar":
        from experiments.openlin_dem_lines import load_alos
        from experiments.openlin_relief_probe import load_dem
        glo, alos = load_dem(site, res), load_alos(site, res)
        extra = {"glo": glo, "alos": alos}
        valid &= np.isfinite(glo) & np.isfinite(alos)
    v = b04[valid]
    lo, hi = np.percentile(v, [2, 98])
    img = np.clip((np.where(valid, b04, hi) - lo) / (hi - lo), 0, 1)
    return dict(w=w, res=res, x0=x0, ytop=ytop, epsg=epsg, img=img, valid=valid, **extra)


def eligible_windows(valid):
    H, W = valid.shape
    from scipy.ndimage import uniform_filter
    full = uniform_filter(valid.astype("float32"), WIN, mode="constant") > 0.9999
    out = []
    for r in range(WIN // 2, H - WIN // 2 + 1, STEP):
        for c in range(WIN // 2, W - WIN // 2 + 1, STEP):
            if full[r, c]:
                out.append((r, c))
    return out


def select(site, S, log):
    rng = np.random.default_rng(SEED)
    cand = eligible_windows(S["valid"])
    log(f"{site}: допустимых окон {len(cand)}")
    chosen = []

    def overlaps(rc):
        return any(abs(rc[0] - q[0]) < WIN and abs(rc[1] - q[1]) < WIN for q, *_ in chosen)

    def pick(pool, name, stratum, note=""):
        pool = list(pool)
        order = rng.permutation(len(pool))
        for k in order:
            rc = pool[k]
            if not overlaps(rc):
                chosen.append((rc, name, stratum, note))
                return
        raise SystemExit(f"{site}: для «{name}» нет неперекрывающегося окна")

    if site == "anabar":
        g = S["glo"]
        gy, gx = np.gradient(np.where(np.isfinite(g), g, np.nanmedian(g)), S["res"])
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        med = np.array([np.median(slope[r - WIN // 2:r + WIN // 2, c - WIN // 2:c + WIN // 2])
                        for r, c in cand])
        lo, hi = np.quantile(med, [Q_LOW, Q_HIGH])
        log(f"  медиана уклона окон: квартили {lo:.2f}° и {hi:.2f}°")
        pick(cand, "случайное", "все допустимые")
        pick([rc for rc, m in zip(cand, med) if m < lo], "пологий рельеф",
             f"медиана уклона < {lo:.2f}° (Q25)")
        pick([rc for rc, m in zip(cand, med) if m > hi], "расчленённый рельеф",
             f"медиана уклона > {hi:.2f}° (Q75)")
        mmap = dict(zip(cand, med))
        chosen[:] = [(rc, n, s, f"медиана уклона {mmap[rc]:.2f}°") for rc, n, s, _ in chosen]
    else:
        pick(cand, "случайное 1", "все допустимые")
        pick(cand, "случайное 2", "все допустимые")
    return chosen


def win_extent(S, rc):
    r, c = rc
    res, x0, ytop = S["res"], S["x0"], S["ytop"]
    r0, c0 = r - WIN // 2, c - WIN // 2
    xmin, xmax = x0 + c0 * res, x0 + (c0 + WIN) * res
    ymax, ymin = ytop - r0 * res, ytop - (r0 + WIN) * res
    return (slice(r0, r0 + WIN), slice(c0, c0 + WIN)), (xmin, xmax, ymin, ymax)


def scalebar(ax, ext, km=1.0):
    xmin, xmax, ymin, ymax = ext
    x, y = xmin + 0.06 * (xmax - xmin), ymin + 0.06 * (ymax - ymin)
    ax.plot([x, x + km * 1000], [y, y], color="white", lw=4, solid_capstyle="butt")
    ax.plot([x, x + km * 1000], [y, y], color="black", lw=2, solid_capstyle="butt")
    ax.text(x + km * 500, y + 0.025 * (ymax - ymin), f"{km:.0f} км", ha="center",
            fontsize=9, color="black",
            bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))


def draw_lines(ax, g, ext, color, lw=1.1):
    from shapely.geometry import box
    xmin, xmax, ymin, ymax = ext
    sub = g.clip(box(xmin, ymin, xmax, ymax))
    sub = sub[~sub.geometry.is_empty]
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=lw)
    return len(sub)


def window_figure(site, S, chosen, layers, title_sfx, out, log):
    import matplotlib.pyplot as plt

    n = len(chosen)
    fig, axs = plt.subplots(n, 3, figsize=(16.5, 5.6 * n), facecolor="white")
    axs = np.atleast_2d(axs)
    rows = []
    for i, (rc, name, stratum, note) in enumerate(chosen):
        sl, ext = win_extent(S, rc)
        sub = S["img"][sl]
        for j, (ttl, lay) in enumerate(layers):
            ax = axs[i, j]
            ax.imshow(sub, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
            counts = []
            for g, col in lay:
                counts.append(draw_lines(ax, g, ext, col))
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(f"{ttl}" + (f"\nобъектов в окне: {', '.join(map(str, counts))}"
                                     if counts else "\n"), fontsize=10, loc="left")
            ax.set_xlabel("x, м")
            if j:
                ax.set_ylabel("y, м")
            ax.tick_params(labelsize=7)
            ax.ticklabel_format(style="plain", useOffset=False)
            scalebar(ax, ext)
        axs[i, 0].set_ylabel(f"окно {i + 1}: {name}\n{note}\nцентр x={0.5 * (ext[0] + ext[1]):.0f}, "
                             f"y={0.5 * (ext[2] + ext[3]):.0f}", fontsize=10)
        rows.append({"site": site, "window": i + 1, "rule": name, "stratum": stratum,
                     "note": note, "seed": SEED, "row": rc[0], "col": rc[1],
                     "xmin": ext[0], "ymin": ext[2], "xmax": ext[1], "ymax": ext[3],
                     "epsg": S["epsg"], "chosen_before_viewing": True})
    fig.suptitle(f"{title_sfx}; EPSG:{S['epsg']}, оси в метрах, север сверху, окна 5 × 5 км "
                 f"по правилу audit/SELECTION_RULE.md", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{out}.png", dpi=300)
    fig.savefig(FIG / f"{out}.pdf")
    plt.close(fig)
    log(f"сохранено: {FIG / out}.png и .pdf")
    return rows


def overview(sites, log):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axs = plt.subplots(1, len(sites), figsize=(9 * len(sites), 8.5), facecolor="white")
    for ax, (site, S, chosen) in zip(np.atleast_1d(axs), sites):
        H, W = S["img"].shape
        ext = (S["x0"], S["x0"] + W * S["res"], S["ytop"] - H * S["res"], S["ytop"])
        ax.imshow(S["img"][::5, ::5], cmap="gray", vmin=0, vmax=1, extent=ext)
        for i, (rc, name, *_r) in enumerate(chosen):
            _sl, e = win_extent(S, rc)
            ax.add_patch(Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fill=False,
                                   ec=YEL, lw=2))
            ax.text(e[0], e[3], f" {i + 1}", color=YEL, fontsize=13, va="bottom",
                    fontweight="bold")
        ax.set_title(f"{site}: окна рисунков (EPSG:{S['epsg']})", loc="left")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_windows_overview.png", dpi=150)
    plt.close(fig)
    log(f"сохранено: {FIG / 'fig_windows_overview.png'}")


def data_sources(sites, log):
    import matplotlib.pyplot as plt

    panels = []
    for site, S, _c in sites:
        w = S["w"]
        hb = np.load(COMP / f"winter_{site}_20m_лист_Б.npz")["count"]
        panels.append((site, "зимних сцен в ячейке (полный композит, 12)", w["count"], "viridis"))
        panels.append((site, "сцен в ячейке, половина «Б» (подтверждение)", hb, "viridis"))
        if site == "anabar":
            panels.append((site, "GLO-30: есть / нет (TanDEM-X)", np.isfinite(S["glo"]), "gray"))
            panels.append((site, "ALOS AW3D30: есть / нет (PRISM)", np.isfinite(S["alos"]), "gray"))
        else:
            z = np.load(Path("outputs/metrics") / f"openlin_{site}.npz")
            panels.append((site, "летний S2 100 м: >= 3 наблюдений (вход «Коры»)",
                           np.isfinite(z["gray"]), "gray"))
            panels.append((site, "S1 VV 100 м: есть / нет (вход «Коры»)",
                           np.isfinite(z["vv"]), "gray"))
    fig, axs = plt.subplots(len(sites), 4, figsize=(22, 5.4 * len(sites)), facecolor="white")
    for ax, (site, ttl, arr, cmap) in zip(axs.ravel(), panels):
        a = np.asarray(arr)
        step = max(1, a.shape[0] // 800)
        im = ax.imshow(a[::step, ::step].astype("float32"), cmap=cmap, interpolation="nearest",
                       **({} if cmap == "viridis" else {"vmin": 0, "vmax": 1}))
        if cmap == "viridis":
            fig.colorbar(im, ax=ax, fraction=0.04)
            ttl += f"; мин {int(a.min())}, макс {int(a.max())}"
        else:
            ttl += f"; покрыто {100 * a.mean():.1f} %"
        ax.set_title(f"{site}: {ttl}", fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "fig_data_sources.png", dpi=300)
    fig.savefig(FIG / "fig_data_sources.pdf")
    plt.close(fig)
    log(f"сохранено: {FIG / 'fig_data_sources.png'} и .pdf")


def geometry_check(site, S, g, mask, log):
    """20 случайных линий: ячейки маски (красные) и вектор (жёлтый) поверх B04."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED + 1)
    pick = rng.choice(len(g), N_GEOM_CHECK, replace=False)
    fig, axs = plt.subplots(4, 5, figsize=(20, 16.5), facecolor="white")
    res, x0, ytop = S["res"], S["x0"], S["ytop"]
    rows = []
    for ax, k in zip(axs.ravel(), pick):
        row = g.iloc[int(k)]
        xmin, ymin, xmax, ymax = row.geometry.bounds
        pad = 300.0
        # рамка обрезается краем листа, иначе срез короче рамки и картинку тянет
        H, W = mask.shape
        c0 = int(max((xmin - pad - x0) // res, 0)); c1 = min(int((xmax + pad - x0) // res) + 1, W)
        r0 = int(max((ytop - ymax - pad) // res, 0)); r1 = min(int((ytop - ymin + pad) // res) + 1, H)
        ext = (x0 + c0 * res, x0 + c1 * res, ytop - r1 * res, ytop - r0 * res)
        ax.imshow(S["img"][r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, extent=ext,
                  interpolation="nearest")
        parts = row.geometry.geoms if row.geometry.geom_type == "MultiLineString" else [row.geometry]
        # ячейки самой линии — по вершинам её вектора (они стоят в центрах ячеек)
        own = np.zeros(mask.shape, bool)
        for p in parts:
            xy = np.asarray(p.coords)
            own[np.floor((ytop - xy[:, 1]) / res).astype(int),
                np.floor((xy[:, 0] - x0) / res).astype(int)] = True
        m, o = mask[r0:r1, c0:c1], own[r0:r1, c0:c1]
        ov = np.zeros(m.shape + (4,), "float32")
        ov[m & ~o] = (0.2, 0.75, 0.95, 0.55)       # соседние линии набора
        ov[o] = (0.9, 0.05, 0.05, 0.7)             # сама линия
        if not mask[own].all():
            raise SystemExit(f"{site}: вершины линии {row.line_id} вне маски")
        ax.imshow(ov, extent=ext, interpolation="nearest")
        for p in parts:
            xy = np.asarray(p.coords)
            ax.plot(xy[:, 0], xy[:, 1], color=YEL, lw=1.4)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
        ratio = row.len_vector_km / max(row.len_trace_km, 1e-9)
        ax.set_title(f"id {row.line_id}: трасса {row.len_trace_km:.2f} км, вектор "
                     f"{row.len_vector_km:.2f} км, частей {row.n_parts}", fontsize=8.5)
        rows.append({"site": site, "set": row.set, "line_id": int(row.line_id),
                     "n_cells": int(row.n_cells), "n_parts": int(row.n_parts),
                     "len_trace_km": round(row.len_trace_km, 3),
                     "len_vector_km": round(row.len_vector_km, 3),
                     "ratio_vector_to_trace": round(ratio, 3),
                     "x_center": round(0.5 * (xmin + xmax)), "y_center": round(0.5 * (ymin + ymax))})
    fig.suptitle(f"{site}: {N_GEOM_CHECK} случайных линий набора «{g.set.iloc[0]}» (зерно {SEED + 1}); "
                 "красное — ячейки этой линии в маске тракта, голубое — соседние линии набора, "
                 "жёлтое — экспортированный вектор",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / f"geometry_check_{site}.png", dpi=110)
    plt.close(fig)
    with open(AUDIT / f"geometry_check_{site}.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    r = np.array([x["ratio_vector_to_trace"] for x in rows])
    log(f"{site}: проверка {N_GEOM_CHECK} линий — вектор/трасса медиана {np.median(r):.2f}, "
        f"диапазон {r.min():.2f}-{r.max():.2f}; {FIG / f'geometry_check_{site}.png'}")


def main() -> None:
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")

    def log(m=""):
        print(m, flush=True)

    FIG.mkdir(parents=True, exist_ok=True)
    manifest, sites = [], []

    S = load_site("anabar")
    ch = select("anabar", S, log)
    opt = gpd.read_file(VEC / "anabar_lines.gpkg", layer="optical_full")
    add = gpd.read_file(VEC / "anabar_lines.gpkg", layer="dem1_added")
    manifest += window_figure("anabar", S, ch, [
        ("зимний S2 B04, без линий", []),
        ("B04 + линии по снимку (весь лист)", [(opt, RED)]),
        ("B04 + линии рельефа, добавленные к снимку (DEM1)", [(add, BLUE)]),
    ], "Анабар, лист R-48-XI,XII", "fig_anabar_optical_dem", log)
    geometry_check("anabar", S, opt,
                   np.load("outputs/metrics/openlin_zone_lines_anabar_v3_бл.npz")["C_all"], log)
    sites.append(("anabar", S, ch))

    S = load_site("lapland")
    ch = select("lapland", S, log)
    auto = gpd.read_file(VEC / "lapland_lines.gpkg", layer="auto_full")
    gtk = gpd.read_file("datacache/gtk_lineaments/lin500k_292.gpkg").to_crs(epsg=S["epsg"])
    manifest += window_figure("lapland", S, ch, [
        ("зимний S2 B04, без линий", []),
        ("B04 + наши линии (весь лист)", [(auto, RED)]),
        ("B04 + наши линии + GTK 292 (только для оценки)", [(auto, RED), (gtk, BLUE)]),
    ], "Лапландия (перенос XFER)", "fig_lapland_auto_gtk", log)
    geometry_check("lapland", S, auto,
                   np.load("outputs/metrics/openlin_zone_lines_lapland_v3_перенос.npz")["C_all"], log)
    sites.append(("lapland", S, ch))

    overview(sites, log)
    data_sources(sites, log)
    with open(FIG / "selection_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, list(manifest[0].keys()))
        wr.writeheader(); wr.writerows(manifest)
    log(f"сохранено: {FIG / 'selection_manifest.csv'}")


if __name__ == "__main__":
    main()
