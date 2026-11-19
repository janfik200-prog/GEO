"""Пакет 06 ТЗ К1: берега, водоёмы и ложные структуры — признаки, не вердикты.

Объявление — ``audit/PACKET06_DECLARATION.md`` (26.10.2026, до расчёта).
Экспертов нет: доля ошибок в процентах здесь НЕ считается (ТЗ требует для неё
экспертную классификацию). Считаются близость к объектам, прямолинейность,
классы по правилам и готовится заранее отобранная выборка для экспертов.

Запуск из корня:
    python -m experiments.openlin_artifact_audit
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import grid, rast

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
VEC = Path("outputs/k1/vectors")
OUT = Path("outputs/k1/artifact_audit")
FIG = Path("outputs/k1/figures")

# ---- объявление (audit/PACKET06_DECLARATION.md) ----------------------------
TOLS_M = (30.0, 60.0, 120.0)
TOL_M, FRAC = 60.0, 0.30                       # CONF_TOL_M, CONF_FRAC тракта
ARC_Q = 0.10                                   # договорённость; рядом 0.05 и 0.25
ARC_Q_SENS = (0.05, 0.10, 0.25)
SEED = 20261027
PER_STRATUM = 5
SUN_AZ = {"anabar": 184.0}                     # Лапландия — медиана по сценам
SETS = (("anabar", "optical_full", "openlin_zone_lines_anabar_v3_бл", "C_all"),
        ("anabar", "dem1_added", "openlin_dem_lines_anabar_одна", "add"),
        ("lapland", "auto_full", "openlin_zone_lines_lapland_v3_перенос", "C_all"))
# ----------------------------------------------------------------------------


def adiff(a, b):
    return abs(((a - b) + 90.0) % 180.0 - 90.0)


def object_dists(site, g):
    """Расстояние до каждого класса объектов на сетке 20 м."""
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt as edt

    osm = gpd.read_file(f"outputs/k1/reference/{site}_osm.gpkg")
    pick = {"road": ["road", "power"], "shore": ["shore", "wetland_edge"], "waterway": ["waterway"]}
    d = {k: edt(~rast(osm[osm.cls.isin(v)].geometry, g), sampling=20.0) for k, v in pick.items()}
    if site == "anabar":
        from experiments.openlin_hydro_check import hydro_mask
        hm = hydro_mask((g["H"], g["W"]), 20.0, g["x0"], g["ytop"], g["epsg"], lambda *_: None)
        d["waterway"] = np.minimum(d["waterway"], edt(~hm, sampling=20.0))
    cnt = np.load(COMP / f"winter_{site}_20m_лист.npz")["count"]
    seam = np.zeros(cnt.shape, bool)
    seam[:, :-1] |= cnt[:, :-1] != cnt[:, 1:]
    seam[:-1, :] |= cnt[:-1, :] != cnt[1:, :]
    d["seam"] = edt(~seam, sampling=20.0)
    return d, osm


def water_polygons(osm, site, epsg):
    """Маска водоёмов: полигоны из замкнутых контуров берегов и болот OSM."""
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union
    sub = osm[osm.cls.isin(["shore", "wetland_edge"])]
    rows = []
    for cls in ("shore", "wetland_edge"):
        lines = sub[sub.cls == cls].geometry.values
        if not len(lines):
            continue
        for p in polygonize(unary_union(list(lines))):
            rows.append({"cls": "water" if cls == "shore" else "wetland", "geometry": p})
    g = gpd.GeoDataFrame(rows, crs=f"EPSG:{epsg}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{site}_water_mask.gpkg"
    if len(g):
        g.to_file(p, driver="GPKG")
    return g, p


def line_geometry(cells):
    """Размах вдоль главной оси, разброс поперёк (в долях размаха)."""
    c = (cells - cells.mean(0)).astype("float64")
    if len(c) < 3:
        return 0.0, 0.0
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    along, across = c @ vt[0], c @ vt[1]
    span = float(along.max() - along.min()) * 20.0
    return span, float(across.std() * 20.0 / max(span, 1e-9))


def main() -> None:
    import geopandas as gpd

    def log(m=""):
        print(m, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    summary, manifest, label_rows = {}, [], []
    cache = {}
    for site, layer, stem, key in SETS:
        g = grid(site)
        if site not in cache:
            cache[site] = object_dists(site, g)
            if site == "lapland":
                wp, wpath = water_polygons(cache[site][1], site, g["epsg"])
                log(f"{site}: маска водоёмов из OSM — {int((wp.cls == 'water').sum())} водоёмов, "
                    f"{int((wp.cls == 'wetland').sum())} болот -> {wpath}")
        dists, _osm = cache[site]
        sun = SUN_AZ.get(site) or float(np.median(np.load(COMP / f"winter_{site}_20m_лист.npz")["azimuth"]))
        d = np.load(M / f"{stem}_линии.npz")
        pts, ids, az = d[f"{key}_pts"], d[f"{key}_id"], d[f"{key}_az"]
        vec = gpd.read_file(VEC / f"{site}_lines.gpkg", layer=layer).set_index("line_id")
        n = len(az)
        rows = []
        for i in range(n):
            c = pts[ids == i]
            near = {k: float((v[c[:, 0], c[:, 1]] <= TOL_M).mean()) for k, v in dists.items()}
            cls = next((k for k in ("road", "seam", "shore", "waterway") if near[k] >= FRAC), "other")
            span, spread = line_geometry(c)
            path = float(vec.loc[i, "len_vector_km"]) * 1000
            rows.append({"line_id": i, "класс": cls, "прямолинейность": min(span / max(path, 1e-9), 1.0),
                         "разброс_поперёк": spread, "длина_км": span / 1000,
                         "угол_к_солнцу": adiff(float(az[i]), sun),
                         **{f"у_{k}": v for k, v in near.items()}})
        st = {"линий": n}
        # близость (косвенно): доля длины (ячеек) набора у объектов
        allc = pts
        st["доля_длины_у"] = {k: {f"{t:.0f}м": float((v[allc[:, 0], allc[:, 1]] <= t).mean())
                                  for t in TOLS_M} for k, v in dists.items()}
        cls_arr = np.array([r["класс"] for r in rows])
        st["классы"] = {k: float((cls_arr == k).mean()) for k in ("road", "seam", "shore", "waterway", "other")}
        sl = np.array([r["прямолинейность"] for r in rows])
        sp = np.array([r["разброс_поперёк"] for r in rows])
        st["прямолинейность_по_классам"] = {
            k: {"медиана": float(np.median(sl[cls_arr == k])), "Q10": float(np.quantile(sl[cls_arr == k], 0.1)),
                "разброс_медиана": float(np.median(sp[cls_arr == k])), "n": int((cls_arr == k).sum())}
            for k in ("road", "seam", "shore", "waterway", "other") if (cls_arr == k).sum() >= 5}
        other = sl[cls_arr == "other"]
        st["береговые_дуги"] = {f"Q{int(q * 100)}": float(((cls_arr == "shore") & (sl < np.quantile(other, q))).mean())
                                for q in ARC_Q_SENS}
        summary[f"{site}/{layer}"] = st
        log(f"\n{site}/{layer}: линий {n}")
        log("  доля длины у объектов (30/60/120 м): " + "; ".join(
            f"{k} " + "/".join(f"{100 * v:.1f}" for v in st['доля_длины_у'][k].values()) + " %"
            for k in st["доля_длины_у"]))
        log("  классы по правилам (признак): " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in st["классы"].items()))
        log("  прямолинейность (медиана; Q10; разброс поперёк): " + "; ".join(
            f"{k} {v['медиана']:.2f}; {v['Q10']:.2f}; {100 * v['разброс_медиана']:.1f} % (n={v['n']})"
            for k, v in st["прямолинейность_по_классам"].items()))
        log("  береговые дуги (shore и кривее Q5/Q10/Q25 прочих): " + " / ".join(
            f"{100 * v:.1f} %" for v in st["береговые_дуги"].values()) + " линий")

        # выборка для экспертов — только для наборов по снимку
        if layer in ("optical_full", "auto_full"):
            rng = np.random.default_rng(SEED)
            med = np.median([r["длина_км"] for r in rows])
            strata = {}
            for r in rows:
                key_s = (r["класс"] in ("shore", "waterway"), r["угол_к_солнцу"] < 30, r["длина_км"] > med)
                strata.setdefault(key_s, []).append(r)
            for key_s in sorted(strata):
                grp = strata[key_s]
                pick = rng.choice(len(grp), min(PER_STRATUM, len(grp)), replace=False)
                for k in sorted(pick):
                    r = grp[k]
                    geom = vec.loc[r["line_id"], "geometry"]
                    cx, cy = geom.centroid.x, geom.centroid.y
                    s_name = (f"{'вода' if key_s[0] else 'не_вода'}|"
                              f"{'<30°' if key_s[1] else '>=30°'}|{'длинная' if key_s[2] else 'короткая'}")
                    manifest.append({"site": site, "set": layer, "line_id": r["line_id"], "стратум": s_name,
                                     "seed": SEED, "x": round(cx), "y": round(cy), "epsg": g["epsg"],
                                     "длина_км": round(r["длина_км"], 3)})
                    label_rows.append({"site": site, "line_id": r["line_id"], "стратум": s_name,
                                       "expert_A_class": "", "expert_B_class": "", "confidence": "",
                                       "rule_class_NOT_EXPERT": r["класс"],
                                       "прямолинейность": round(r["прямолинейность"], 3)})
        with open(OUT / f"{site}_{layer}_lines.csv", "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
    for name, rows_ in (("sample_manifest.csv", manifest), ("expert_labels.csv", label_rows)):
        with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, list(rows_[0].keys()))
            wr.writeheader(); wr.writerows(rows_)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\nвыборка для экспертов: {len(manifest)} линий -> {OUT / 'sample_manifest.csv'}")
    expert_sheets(manifest, log)
    shore_cases(log)


def _stretch_site(site):
    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    b = w["B04"].astype("float32")
    v = np.isfinite(b)
    lo, hi = np.percentile(b[v], [2, 98])
    return w, np.clip((np.where(v, b, hi) - lo) / (hi - lo), 0, 1)


def expert_sheets(manifest, log):
    """Листы для экспертов: окно 3 x 3 км вокруг линии, B04, линия без класса."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    for site in ("anabar", "lapland"):
        rows = [r for r in manifest if r["site"] == site]
        w, img = _stretch_site(site)
        x0, yt = float(w["x0"]), float(w["y_top"])
        layer = rows[0]["set"]
        vec = gpd.read_file(VEC / f"{site}_lines.gpkg", layer=layer).set_index("line_id")
        for page in range(0, len(rows), 20):
            chunk = rows[page:page + 20]
            fig, axs = plt.subplots(4, 5, figsize=(20, 17), facecolor="white")
            for ax, r in zip(axs.ravel(), chunk):
                half = 1500.0
                xmin, xmax, ymin, ymax = r["x"] - half, r["x"] + half, r["y"] - half, r["y"] + half
                c0, c1 = int((xmin - x0) / 20), int((xmax - x0) / 20)
                r0, r1 = int((yt - ymax) / 20), int((yt - ymin) / 20)
                c0, r0 = max(c0, 0), max(r0, 0)
                ext = (x0 + c0 * 20, x0 + c1 * 20, yt - r1 * 20, yt - r0 * 20)
                ax.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, extent=ext,
                          interpolation="nearest")
                gpd.GeoSeries([vec.loc[r["line_id"], "geometry"]]).plot(ax=ax, color="#ffd21a", lw=1.6)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{site} id {r['line_id']}", fontsize=9)
            for ax in axs.ravel()[len(chunk):]:
                ax.axis("off")
            fig.suptitle(f"{site}: линии для экспертной классификации (выборка по стратам, зерно {SEED}); "
                         "окно 3 x 3 км, зимний B04, жёлтое — линия; класс не показан", x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            p = OUT / f"expert_sheet_{site}_{page // 20 + 1}.png"
            fig.savefig(p, dpi=100)
            plt.close(fig)
            log(f"сохранено: {p}")


def shore_cases(log):
    """Три пары окон на Лапландии: shore, waterway, other (случайно, то же зерно)."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    site = "lapland"
    rows = list(csv.DictReader(open(OUT / f"{site}_auto_full_lines.csv", encoding="utf-8-sig")))
    rng = np.random.default_rng(SEED + 6)
    w, img = _stretch_site(site)
    x0, yt = float(w["x0"]), float(w["y_top"])
    vec = gpd.read_file(VEC / f"{site}_lines.gpkg", layer="auto_full").set_index("line_id")
    osm = gpd.read_file(f"outputs/k1/reference/{site}_osm.gpkg")
    fig, axs = plt.subplots(3, 3, figsize=(16, 16), facecolor="white")
    for i, cls in enumerate(("shore", "waterway", "other")):
        grp = [r for r in rows if r["класс"] == cls]
        r = grp[int(rng.integers(len(grp)))]
        geom = vec.loc[int(r["line_id"]), "geometry"]
        cx, cy, half = geom.centroid.x, geom.centroid.y, 1500.0
        ext_m = (cx - half, cx + half, cy - half, cy + half)
        c0, c1 = int((ext_m[0] - x0) / 20), int((ext_m[1] - x0) / 20)
        r0, r1 = int((yt - ext_m[3]) / 20), int((yt - ext_m[2]) / 20)
        ext = (x0 + c0 * 20, x0 + c1 * 20, yt - r1 * 20, yt - r0 * 20)
        from shapely.geometry import box
        sub = osm.clip(box(ext[0], ext[2], ext[1], ext[3]))
        for j in range(3):
            ax = axs[i, j]
            ax.imshow(img[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
            if j >= 1 and len(sub):
                for k, col in (("shore", "#1f8fff"), ("wetland_edge", "#7fd0ff"), ("waterway", "#00c0c0"),
                               ("road", "#ff7f00"), ("power", "#ff00ff")):
                    s_ = sub[sub.cls == k]
                    if len(s_):
                        s_.plot(ax=ax, color=col, lw=1.0)
            if j == 2:
                gpd.GeoSeries([geom]).plot(ax=ax, color="#e0141e", lw=2.0)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])
        axs[i, 0].set_title(f"класс по правилам: {cls} (НЕ эксперт); id {r['line_id']}", loc="left", fontsize=11)
        axs[i, 1].set_title("+ OSM: берег (синий), болото (голубой), водоток, дорога (оранж.)", loc="left", fontsize=9)
        axs[i, 2].set_title(f"+ линия; прямолинейность {float(r['прямолинейность']):.2f}, у берега "
                            f"{100 * float(r['у_shore']):.0f} % длины", loc="left", fontsize=10)
    fig.suptitle("Лапландия: по одной случайной линии классов shore / waterway / other; окна 3 x 3 км",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = FIG / "shore_cases.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
