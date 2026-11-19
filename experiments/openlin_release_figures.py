"""WP21–WP23 (ТЗ v3.0): настоящие карты трёх площадок и сцены демонстрации.

Окна выбираются по правилу ``audit/RELEASE_FIGURE_RULE.md`` (записано до
построения). Подложка — зимний B04 текущего композита, линии — из
GeoPackage (``outputs/k1/vectors``), маски простых методов — из
``openlin_release_baselines``. Ничего не подрисовывается.

Выходы (``outputs/k1/figures/release/``):
* real_anabar.png, real_lapland_gtk.png, real_finland65.png — обзор листа и
  окна «без линий / с линиями»;
* side_by_side_examples.png — одно окно, наша карта / LSD / Хаф при одной длине;
* problem_cases.png — берега, дорога, шов, долина вдоль света, Finland65;
* transfer_3sites.png — три листа рядом с итогом проверок;
* line_trace_<id>.png и line_traces.json — две линии от пикселя до GeoPackage;
* fig_scheme.png — схема тракта (F01);
* figure_manifest.csv — окно, правило, зерно, координаты каждого рисунка.

Запуск из корня:
    python -m experiments.openlin_release_figures
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.openlin_k1_figures import WIN, eligible_windows, load_site

VEC = Path("outputs/k1/vectors")
RES = Path("outputs/k1/results")
AA = Path("outputs/k1/artifact_audit")
FIG = Path("outputs/k1/figures/release")
SEED_P01, SEED_REL = 20261026, 20261028
SMALL = 150                                   # окно сцен E: 150 ячеек = 3 км
SITE_NAME = {"anabar": "Анабар", "lapland": "Лапландия", "finland65": "Finland65"}
LAYER = {"anabar": "optical_full", "lapland": "auto_full", "finland65": "auto_full"}
GTK = {"lapland": "datacache/gtk_lineaments/lin500k_292.gpkg",
       "finland65": "datacache/gtk_lineaments/finland65_lin500k_292.gpkg"}
RED, ORANGE, CYAN, YEL, BLUE, GREEN = "#e0141e", "#ff8c00", "#00b7d4", "#ffd21a", "#1a66f2", "#16a34a"
MANIFEST: list[dict] = []


def scalebar(ax, ext, km=1.0):
    """Масштабная линейка; подпись с дробными километрами (0.5 км)."""
    xmin, xmax, ymin, ymax = ext
    x, y = xmin + 0.06 * (xmax - xmin), ymin + 0.06 * (ymax - ymin)
    ax.plot([x, x + km * 1000], [y, y], color="white", lw=4, solid_capstyle="butt")
    ax.plot([x, x + km * 1000], [y, y], color="black", lw=2, solid_capstyle="butt")
    ax.text(x + km * 500, y + 0.025 * (ymax - ymin), f"{km:g} км", ha="center", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))


def extent_of(S, r0, c0, n):
    res, x0, ytop = S["res"], S["x0"], S["ytop"]
    return (x0 + c0 * res, x0 + (c0 + n) * res, ytop - (r0 + n) * res, ytop - r0 * res)


def base(ax, S, sl, ext, title):
    ax.imshow(S["img"][sl], cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(title, fontsize=9, loc="left")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=6)


def lines_in(g, ext):
    from shapely.geometry import box
    sub = g.clip(box(ext[0], ext[2], ext[1], ext[3]))
    return sub[~sub.geometry.is_empty]


def plot_lines(ax, g, ext, color, lw=1.0, label=None):
    sub = lines_in(g, ext)
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=lw)
    if label:
        ax.plot([], [], color=color, lw=2, label=f"{label} ({len(sub)})")
    return len(sub)


def plot_mask(ax, m, sl, ext, color, label):
    from matplotlib.colors import ListedColormap
    from scipy.ndimage import binary_dilation
    sub = binary_dilation(m[sl], iterations=1)
    ax.imshow(np.ma.masked_where(~sub, sub), cmap=ListedColormap([color]), extent=ext,
              interpolation="nearest", alpha=0.95)
    ax.plot([], [], color=color, lw=3, label=f"{label} ({m[sl].sum() * 0.02:.0f} км ячеек)")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"сохранено: {FIG / name}.png / .pdf", flush=True)


def p01_windows(site, S):
    """Окна Пакета 01 (Анабар, Лапландия) или то же правило для Finland65."""
    if site != "finland65":
        rows = [r for r in csv.DictReader(open("outputs/k1/figures/selection_manifest.csv", encoding="utf-8-sig"))
                if r["site"] == site]
        return [((int(r["row"]), int(r["col"])), r["rule"], r["note"] or r["stratum"]) for r in rows]
    rng = np.random.default_rng(SEED_P01)
    cand = eligible_windows(S["valid"])
    chosen = []
    for name in ("случайное 1", "случайное 2"):
        for k in rng.permutation(len(cand)):
            rc = cand[k]
            if not any(abs(rc[0] - q[0]) < WIN and abs(rc[1] - q[1]) < WIN for q, *_ in chosen):
                chosen.append((rc, name, f"все допустимые ({len(cand)})"))
                break
    return chosen


def overview(ax, S, g, color, title, lw=0.35):
    H, W = S["img"].shape
    ext = extent_of(S, 0, 0, max(H, W))
    ext = (S["x0"], S["x0"] + W * S["res"], S["ytop"] - H * S["res"], S["ytop"])
    ax.imshow(S["img"][::4, ::4], cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="bilinear")
    if g is not None:
        g.plot(ax=ax, color=color, linewidth=lw)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(title, fontsize=10, loc="left")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=6)
    scalebar(ax, ext, km=10)
    return ext


def real_site(site, S, gp, log):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    lines = gpd.read_file(gp, layer=LAYER[site])
    dem1 = gpd.read_file(gp, layer="dem1_added") if site == "anabar" else None
    gtk = gpd.read_file(GTK[site]).to_crs(epsg=S["epsg"]) if site in GTK else None
    wins = p01_windows(site, S)
    ncol = 3
    fig = plt.figure(figsize=(16, 5.6 * (1 + len(wins))), facecolor="white")
    gs = fig.add_gridspec(1 + len(wins), ncol)
    ax = fig.add_subplot(gs[0, :2] if gtk is None else gs[0, 0])
    overview(ax, S, lines, RED, f"{SITE_NAME[site]}: C_all, {len(lines)} линий / {lines.len_trace_km.sum():.0f} км "
             f"(хорды)")
    for i, (rc, *_r) in enumerate(wins):
        sl, ext = (slice(rc[0] - WIN // 2, rc[0] + WIN // 2), slice(rc[1] - WIN // 2, rc[1] + WIN // 2)), None
        ext = extent_of(S, rc[0] - WIN // 2, rc[1] - WIN // 2, WIN)
        ax.add_patch(plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, ec=YEL, lw=1.5))
        ax.text(ext[0], ext[3], f" {i + 1}", color=YEL, fontsize=10, va="bottom", weight="bold")
    if dem1 is not None:
        ax2 = fig.add_subplot(gs[0, 2])
        overview(ax2, S, dem1, ORANGE, f"DEM1 добавленные к снимку: {len(dem1)} линий")
    if gtk is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        overview(ax2, S, gtk, CYAN, f"GTK 292, 1:500 000 — ОЦЕНКА, НЕ ВХОД ({len(gtk)} объектов)", lw=0.8)
        ax3 = fig.add_subplot(gs[0, 2])
        overview(ax3, S, lines, RED, "C_all (красный) и GTK 292 (голубой)")
        gtk.plot(ax=ax3, color=CYAN, linewidth=0.8)
    for i, (rc, name, note) in enumerate(wins):
        r0, c0 = rc[0] - WIN // 2, rc[1] - WIN // 2
        sl, ext = (slice(r0, r0 + WIN), slice(c0, c0 + WIN)), extent_of(S, r0, c0, WIN)
        a0, a1, a2 = (fig.add_subplot(gs[1 + i, j]) for j in range(3))
        base(a0, S, sl, ext, f"окно {i + 1} ({name}; {note}): зимний B04 без линий\n"
             f"центр x = {0.5 * (ext[0] + ext[1]):.0f}, y = {0.5 * (ext[2] + ext[3]):.0f}")
        base(a1, S, sl, ext, "те же пиксели + линии C_all")
        plot_lines(a1, lines, ext, RED, 1.3, "C_all")
        if site == "anabar":
            base(a2, S, sl, ext, "+ линии, добавленные по рельефу (DEM1)")
            plot_lines(a2, lines, ext, RED, 1.1, "снимок C_all")
            plot_lines(a2, dem1, ext, ORANGE, 1.3, "DEM1 добавленные")
        else:
            base(a2, S, sl, ext, "+ GTK 292 (только для оценки)")
            plot_lines(a2, lines, ext, RED, 1.1, "C_all")
            plot_lines(a2, gtk, ext, CYAN, 1.6, "GTK 292")
        for a in (a1, a2):
            a.legend(fontsize=7, loc="upper right")
        for a in (a0, a1, a2):
            scalebar(a, ext)
        MANIFEST.append({"figure": f"real_{site}", "panel": f"окно {i + 1}", "site": site, "rule": name,
                         "seed": SEED_P01, "xmin": ext[0], "ymin": ext[2], "xmax": ext[1], "ymax": ext[3],
                         "epsg": S["epsg"], "raster": f"winter_{site}_20m_лист.npz:B04", "lines": f"{gp}:{LAYER[site]}",
                         "reference": GTK.get(site, "—"), "selection_doc": "audit/SELECTION_RULE.md"
                         if site != "finland65" else "audit/RELEASE_FIGURE_RULE.md §2"})
    fig.suptitle(f"{SITE_NAME[site]}: EPSG:{S['epsg']}, оси в метрах UTM, север сверху; подложка — зимний "
                 f"Sentinel-2 B04 (12 сцен, низкое солнце, азимут {float(np.median(S['w']['azimuth'])):.0f}°), "
                 f"растяжка 2–98 %; линии — GeoPackage {gp.name}", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, {"anabar": "real_anabar", "lapland": "real_lapland_gtk", "finland65": "real_finland65"}[site])


def side_by_side(sites, log):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from experiments.openlin_auto_reference import BLOCK_CELLS
    fig, axs = plt.subplots(4, 4, figsize=(17, 17.5), facecolor="white")
    r_ = 0
    for site, S in sites.items():
        if site == "finland65":
            continue
        mk = np.load(RES / f"masks_at_our_budget_{site}.npz")
        shape = tuple(mk["shape"])
        unpack = lambda k: np.unpackbits(mk[k])[:shape[0] * shape[1]].reshape(shape).astype(bool)
        ours, lsd, hough = unpack("наша_карта"), unpack("LSD"), unpack("Canny_plus_Хаф")
        d0 = pd.read_csv(RES / f"D09_F1_AUDIT_{site}.csv")
        d = d0[(d0.budget == "полная длина") & (d0.reference == "v2")]
        pv = d.pivot_table(index="tile_id", columns="method", values="F1")
        # на панели Хаф обрезан до нашей длины — в подписи его F1 при том же бюджете
        hb = d0[(d0.budget == "бюджет наш") & (d0.reference == "v2") & (d0.method == "Хаф")].set_index("tile_id")["F1"]
        pv["Хаф"] = hb
        delta = (pv["наша"] - pv["LSD"]).sort_index()
        picks = [("удачный: наибольшая разность «наша − LSD»", delta.idxmax()),
                 ("сложный: наименьшая разность «наша − LSD»", delta.idxmin())]
        for label, tile in picks:
            i, j = int(tile[1:tile.index("c")]), int(tile[tile.index("c") + 1:])
            rc = (i * BLOCK_CELLS + BLOCK_CELLS // 2, j * BLOCK_CELLS + BLOCK_CELLS // 2)
            r0, c0 = rc[0] - WIN // 2, rc[1] - WIN // 2
            sl, ext = (slice(r0, r0 + WIN), slice(c0, c0 + WIN)), extent_of(S, r0, c0, WIN)
            row = pv.loc[tile]
            base(axs[r_, 0], S, sl, ext, f"{SITE_NAME[site]}, участок {tile} — {label}\nB04 без линий; "
                 f"F1 участка 20 км (v2): наша {row['наша']:.2f}, LSD {row['LSD']:.2f}, Хаф при нашей длине {row['Хаф']:.2f}")
            for a, (m, col, nm) in zip(axs[r_, 1:], ((ours, RED, "наша карта"), (lsd, BLUE, "LSD (своя длина)"),
                                                     (hough, GREEN, "Canny + Хаф, обрезан до нашей длины"))):
                base(a, S, sl, ext, nm)
                plot_mask(a, m, sl, ext, col, nm)
                a.legend(fontsize=7, loc="upper right")
            for a in axs[r_]:
                scalebar(a, ext)
            MANIFEST.append({"figure": "side_by_side_examples", "panel": label, "site": site, "rule":
                             "RELEASE_FIGURE_RULE §3", "seed": "—", "xmin": ext[0], "ymin": ext[2], "xmax": ext[1],
                             "ymax": ext[3], "epsg": S["epsg"], "raster": "B04", "lines":
                             f"masks_at_our_budget_{site}.npz", "reference": "v2 (для выбора)",
                             "selection_doc": "audit/RELEASE_FIGURE_RULE.md §3"})
            r_ += 1
    fig.suptitle("Одно окно 5 × 5 км, одна длина карты: наша карта (полная), LSD (полная, короче нашей), Хаф "
                 "(обрезан до длины нашей карты, от длинных отрезков). Окна — центры участков 20 км по таблице D09.",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "side_by_side_examples")


def problem_cases(sites, log):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    rng = np.random.default_rng(SEED_REL)
    lap = pd.read_csv(AA / "lapland_auto_full_lines.csv")
    ana = pd.read_csv(AA / "anabar_optical_full_lines.csv")
    dem = pd.read_csv(AA / "anabar_dem1_added_lines.csv")
    oth = lap[lap["класс"] == "other"]["прямолинейность"]
    cases = [
        ("изогнутая береговая дуга", "lapland", "auto_full",
         lap[(lap["класс"] == "shore") & (lap["прямолинейность"] < oth.quantile(0.10))]),
        ("прямая линия у цепочки озёр (берег, но прямая)", "lapland", "auto_full",
         lap[(lap["класс"] == "shore") & (lap["прямолинейность"] >= oth.median())]),
        ("вдоль дороги", "lapland", "auto_full", lap[lap["класс"] == "road"]),
        ("вдоль шва съёмки", "anabar", "optical_full", ana[ana["класс"] == "seam"]),
        ("долина вдоль освещения, добавлена рельефом", "anabar", "dem1_added", dem[dem["угол_к_солнцу"] < 15]),
        ("Finland65: случайная линия набора", "finland65", "auto_full", None),
    ]
    water = gpd.read_file(AA / "lapland_water_mask.gpkg")
    fig, axs = plt.subplots(len(cases), 2, figsize=(11, 5.4 * len(cases)), facecolor="white")
    for k, (name, site, layer, pool) in enumerate(cases):
        S = sites[site]
        g = gpd.read_file(VEC / f"{site}_lines.gpkg", layer=layer)
        ids = g.line_id.values if pool is None else pool.line_id.values
        lid = int(ids[rng.integers(len(ids))])
        geom = g.set_index("line_id").geometry.loc[lid]
        c = geom.interpolate(0.5, normalized=True) if geom.geom_type == "LineString" else geom.centroid
        cc = int((c.x - S["x0"]) / S["res"]); rr = int((S["ytop"] - c.y) / S["res"])
        H, W = S["img"].shape
        r0 = int(np.clip(rr - SMALL // 2, 0, H - SMALL)); c0 = int(np.clip(cc - SMALL // 2, 0, W - SMALL))
        sl, ext = (slice(r0, r0 + SMALL), slice(c0, c0 + SMALL)), extent_of(S, r0, c0, SMALL)
        row = g.set_index("line_id").loc[lid]
        base(axs[k, 0], S, sl, ext, f"{name}\n{SITE_NAME[site]}, line_id {lid} ({layer}), {row.len_trace_km:.2f} км; "
             f"B04 без линий")
        base(axs[k, 1], S, sl, ext, "эта линия (жёлтая), прочие линии набора (красные)"
             + ("; контуры воды OSM (синие)" if site == "lapland" else ""))
        plot_lines(axs[k, 1], g, ext, RED, 0.9)
        axs[k, 1].plot(*np.asarray(geom.geoms[0].coords if geom.geom_type == "MultiLineString" else geom.coords).T,
                       color=YEL, lw=2.4)
        if site == "lapland":
            from shapely.geometry import box
            wsub = water.to_crs(epsg=S["epsg"]).clip(box(ext[0], ext[2], ext[1], ext[3]))
            if len(wsub):
                wsub.boundary.plot(ax=axs[k, 1], color=BLUE, linewidth=0.9)
        for a in axs[k]:
            scalebar(a, ext, km=0.5)
        MANIFEST.append({"figure": "problem_cases", "panel": name, "site": site, "rule": "RELEASE_FIGURE_RULE §4",
                         "seed": SEED_REL, "xmin": ext[0], "ymin": ext[2], "xmax": ext[1], "ymax": ext[3],
                         "epsg": S["epsg"], "raster": "B04", "lines": f"{site}_lines.gpkg:{layer}:{lid}",
                         "reference": "OSM (Лапландия)" if site == "lapland" else "—",
                         "selection_doc": "audit/RELEASE_FIGURE_RULE.md §4", "pool": len(ids)})
    fig.suptitle("Проблемные структуры: окна 3 × 3 км. Класс — признак по OSM и форме (Пакет 06), "
                 "НЕ экспертный вердикт", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save(fig, "problem_cases")


def transfer_3sites(sites, log):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    txt = {"anabar": "все 4 условия пройдены\nавто-эталон v2: P 50 % при случайных 7.6 %\nGTK нет (Россия)",
           "lapland": "все 4 условия пройдены\nGTK 292: подъём 1.51×, p = 0.0002 (5000)\nавто-эталон v2: P 56 % / 26 %",
           "finland65": "длина НЕ пройдена (18.5 % < 25 %)\nGTK 292: подъём 0.65×, p = 0.99\n"
                        "«Кора»: простирание неустойчиво (68°)\nПЕРЕНОС НЕ ПОДТВЕРЖДЁН"}
    fig, axs = plt.subplots(1, 3, figsize=(19, 7.6), facecolor="white")
    for ax, (site, S) in zip(axs, sites.items()):
        g = gpd.read_file(VEC / f"{site}_lines.gpkg", layer=LAYER[site])
        overview(ax, S, g, RED, f"{SITE_NAME[site]} (EPSG:{S['epsg']}): {len(g)} линий / "
                 f"{g.len_trace_km.sum():.0f} км")
        ax.text(0.01, 0.01, txt[site], transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(facecolor="white", alpha=0.85, lw=0))
    fig.suptitle("Один тракт, одни пороги на трёх листах 86 × 77 км. Пороги не менялись после Анабара.",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "transfer_3sites")


def line_traces(S, log):
    """Две линии Анабара: от ячеек композита до записи в GeoPackage."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.ndimage import distance_transform_edt as edt
    from experiments.openlin_auto_reference import grid, rast, REF
    rng = np.random.default_rng(SEED_REL)
    d = np.load("outputs/metrics/openlin_zone_lines_anabar_v3_бл_линии.npz")
    pts, ids = d["C_all_pts"], d["C_all_id"]
    masks = np.load("outputs/metrics/openlin_zone_lines_anabar_v3_бл.npz")
    zoneC, A_all, zone = masks["C"], masks["A_all"], masks["zone"]
    g = gpd.read_file(VEC / "anabar_lines.gpkg", layer="optical_full").set_index("line_id")
    aud = pd.read_csv(AA / "anabar_optical_full_lines.csv").set_index("line_id")
    gr = grid("anabar")
    v2 = rast(gpd.read_file(REF / "anabar_arcticdem_mlsd.gpkg").geometry, gr)
    dv2 = edt(~v2, sampling=20.0)
    inC = np.array([zoneC[pts[ids == i][:, 0], pts[ids == i][:, 1]].mean() for i in range(len(g))])
    out = []
    for label, pool in (("в зонах согласия (есть в наборе C)", np.flatnonzero(inC > 0.5)),
                        ("вне зон (только C_all)", np.flatnonzero(inC == 0))):
        lid = int(pool[rng.integers(len(pool))])
        cells = pts[ids == lid]
        row, a = g.loc[lid], aud.loc[lid]
        rec = {"line_id": lid, "пример": label, "site": "anabar", "run_id": "TWIN (перегон 26.10, D08)",
               "ячеек": int(len(cells)), "строки": [int(cells[:, 0].min()), int(cells[:, 0].max())],
               "столбцы": [int(cells[:, 1].min()), int(cells[:, 1].max())],
               "длина_хорда_км": round(float(row.len_trace_km), 3), "длина_вектор_км": round(float(row.len_vector_km), 3),
               "азимут_град": round(float(row.azimuth_deg), 1), "угол_к_солнцу_град": round(float(a["угол_к_солнцу"]), 1),
               "стадии": {"A_all (до подтверждения половинами)": round(float(A_all[cells[:, 0], cells[:, 1]].mean()), 3),
                          "C_all (подтверждена обеими половинами)": 1.0,
                          "в зоне согласия «Коры»": round(float(zone[cells[:, 0], cells[:, 1]].mean()), 3),
                          "C (в зонах, итог)": round(float(inC[lid]), 3)},
               "проверки": {"класс по OSM/гидросети (Пакет 06, признак)": a["класс"],
                            "прямолинейность": round(float(a["прямолинейность"]), 3),
                            "доля ячеек в 60 м от эталона v2": round(float((dv2[cells[:, 0], cells[:, 1]] <= 60).mean()), 3)},
               "gpkg": "outputs/k1/vectors/anabar_lines.gpkg:optical_full",
               "концы_UTM47N": [list(map(lambda v: round(v, 1), row.geometry.boundary.geoms[0].coords[0]))
                                if not row.geometry.boundary.is_empty else None,
                                list(map(lambda v: round(v, 1), row.geometry.boundary.geoms[-1].coords[0]))
                                if not row.geometry.boundary.is_empty else None]}
        out.append(rec)
        rr, cc = int(cells[:, 0].mean()), int(cells[:, 1].mean())
        H, W = S["img"].shape
        r0 = int(np.clip(rr - SMALL // 2, 0, H - SMALL)); c0 = int(np.clip(cc - SMALL // 2, 0, W - SMALL))
        sl, ext = (slice(r0, r0 + SMALL), slice(c0, c0 + SMALL)), extent_of(S, r0, c0, SMALL)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5.6), facecolor="white")
        base(axs[0], S, sl, ext, f"1. зимний B04, окно 3 км")
        base(axs[1], S, sl, ext, "2. маска A_all (до подтверждения) — голубая")
        plot_mask(axs[1], A_all, sl, ext, CYAN, "A_all")
        base(axs[2], S, sl, ext, "3. итог C_all (красный), эталон v2 (зелёный)")
        plot_mask(axs[2], v2, sl, ext, GREEN, "эталон v2")
        plot_mask(axs[2], masks["C_all"], sl, ext, RED, "C_all")
        base(axs[3], S, sl, ext, f"4. GeoPackage: line_id {lid} (жёлтый)")
        geom = row.geometry
        for part in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
            axs[3].plot(*np.asarray(part.coords).T, color=YEL, lw=2.4)
        for ax_ in axs:
            scalebar(ax_, ext, km=0.5)
        for ax_ in axs[1:3]:
            ax_.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Анабар, line_id {lid} — {label}: хорда {row.len_trace_km:.2f} км, азимут {row.azimuth_deg:.0f}°, "
                     f"угол к солнцу {a['угол_к_солнцу']:.0f}°, у эталона v2 {rec['проверки']['доля ячеек в 60 м от эталона v2']:.0%}",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"line_trace_{lid}")
    (FIG / "line_traces.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> None:
    log = lambda m="": print(m, flush=True)
    sites = {s: load_site(s) for s in ("anabar", "lapland", "finland65")}
    for site, S in sites.items():
        real_site(site, S, VEC / f"{site}_lines.gpkg", log)
    side_by_side(sites, log)
    problem_cases(sites, log)
    transfer_3sites(sites, log)
    line_traces(sites["anabar"], log)
    scheme()
    with open(FIG / "figure_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = sorted({k for r in MANIFEST for k in r}, key=lambda k: list(MANIFEST[0]).index(k)
                      if k in MANIFEST[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(MANIFEST)
    log(f"сохранено: {FIG / 'figure_manifest.csv'}")


if __name__ == "__main__":
    main()


def scheme() -> None:
    """F01: схема тракта по рабочей конфигурации (OPENLIN_RULES.md, «Текущий метод»)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    rows = [
        ("Вход (разрешён)", ["S2 B04, зима,\n12 сцен, солнце 6–10°", "GLO-30 + ALOS\n(Анабар, DEM1)", "S1 VH, лето S2,\nкогерентность"], "#dbe8f7"),
        ("Выделение", ["гребень тени, нормировка 2 км,\nNMS, гистерезис 97/90 %", "продолжение концов\n200 м, до 900 м", "сшивка «цепочки»,\nсдвиг ≤ 127 м; отсев швов"], "#fde7c8"),
        ("Безэталонная проверка", ["подтверждение двумя\nполовинами (8 / 4 сцены)", "4 условия: мусор на IAAFT ≤ 20 %,\nсдвиг p ≤ 0.05/N, устойчивость,\nдлина > P95 подделки ≥ 25 %", "«Кора»: зоны согласия\nприборов — признак,\nне фильтр"], "#d9f0d9"),
        ("Внешняя сверка (не вход)", ["эталон v2:\nArcticDEM + LSD", "GTK 292,\n1:500 000", "LSD, Canny + Хаф\nпри равной длине"], "#f3d9d9"),
    ]
    fig, ax = plt.subplots(figsize=(13, 7.2), facecolor="white")
    ax.set_xlim(0, 14); ax.set_ylim(0, len(rows) * 1.8); ax.axis("off")
    for i, (name, boxes, col) in enumerate(rows):
        y = (len(rows) - 1 - i) * 1.8 + 0.25
        ax.text(0.1, y + 0.65, name, fontsize=11, weight="bold", va="center")
        for j, t in enumerate(boxes):
            x = 3.9 + j * 3.4
            ax.add_patch(FancyBboxPatch((x, y), 3.1, 1.3, boxstyle="round,pad=0.05", fc=col, ec="#555"))
            ax.text(x + 1.55, y + 0.65, t, ha="center", va="center", fontsize=9)
        if i < len(rows) - 1:
            ax.annotate("", xy=(8.6, y - 0.22), xytext=(8.6, y + 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))
    ax.text(0.1, 0.02, "Пороги заморожены с 09.10.2026 (прогон TWIN) и не менялись при переносе на Лапландию и Finland65. "
            "Геология, гравика, магнитка, GTK во вход не идут.", fontsize=9, style="italic")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_scheme.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_scheme.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {FIG / 'fig_scheme.png'}")
