r"""Автоматический эталон линеаментов по независимому прибору вместо экспертов.

Объявление — ``audit/AUTO_REFERENCE_DECLARATION.md`` (26.10.2026, до прогона);
здесь те же числа. Экспертов нет (решение пользователя 26.10.2026): G3 ТЗ К1
это не выполняет, результат — «согласие с другим алгоритмом на другом, более
точном приборе», а не точность в глазах специалиста.

Шаги (каждый кэшируется, повторный запуск не качает заново):
1. ``dem``    — ArcticDEM mosaic v4.1 2 м -> среднее на сетку 10 м листа;
2. ``ref``    — отмывка 0/45/90/135°, высота 45°; LSD по умолчанию; отрезки
               от 500 м -> outputs/k1/reference/<site>_arcticdem_lsd.gpkg;
3. ``osm``    — вода, водотоки, болота, дороги, ЛЭП из OpenStreetMap;
4. ``eval``   — P, R, F1 наших наборов против эталона (τ = 60 м, нуль 159
               сдвигов), по листу и по участкам 20 x 20 км; классы ложных
               линий; на Лапландии — эталон против GTK 292 (300 м).

Запуск из корня:
    python -m experiments.openlin_auto_reference --site anabar
    python -m experiments.openlin_auto_reference --site lapland
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
CACHE = Path("datacache/arcticdem")
REF = Path("outputs/k1/reference")
ADEM = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/2m/"
OVERPASS = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"

# ---- объявление (audit/AUTO_REFERENCE_DECLARATION.md) ----------------------
REF_RES_M = 10.0                  # масштаб глаза эксперта по протоколу (20 м, 2x)
AZIMUTHS = (0.0, 45.0, 90.0, 135.0)
ALTITUDE = 45.0
MIN_LEN_M = 500.0                 # определение линеамента в протоколе и тракте
TOL_M = 60.0                      # CONF_TOL_M тракта
TOL_SWEEP_M = (30.0, 60.0, 120.0)
N_SHIFTS = 159
SEED = 2510
CLASS_FRAC = 0.30                 # CONF_FRAC тракта: «лежит вдоль»
BLOCK_CELLS = 1000                # участок 20 км на сетке 20 м
GTK_TOL_M = 300.0
GTK_SITES = ("lapland", "finland65")   # площадки с эталоном GTK (Финляндия)
#: версия 2 (audit/AUTO_REFERENCE_V2_DECLARATION.md, после просмотра версии 1):
#: сшивка через пирамиду масштабов; 80 м — отрезок 500 м ещё 6 пикселей
SCALES_V2_M = (10.0, 20.0, 40.0, 80.0)
V1_F1_VS_GTK = 0.070              # правило выбора версии: v2 основной, если выше
# ----------------------------------------------------------------------------


def grid(site):
    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    H, W = w["B04"].shape
    return dict(H=H, W=W, x0=float(w["x0"]), ytop=float(w["y_top"]), epsg=int(w["epsg"]),
                valid=np.isfinite(w["B04"]))


def wgs_bbox(g, pad=0.0):
    from pyproj import Transformer
    tr = Transformer.from_crs(g["epsg"], 4326, always_xy=True)
    xs = np.linspace(g["x0"] - pad, g["x0"] + g["W"] * 20 + pad, 9)
    ys = np.linspace(g["ytop"] - g["H"] * 20 - pad, g["ytop"] + pad, 9)
    ll = np.array([tr.transform(x, y) for x in xs for y in ys])
    return ll[:, 0].min(), ll[:, 1].min(), ll[:, 0].max(), ll[:, 1].max()


def adem_items(g, log):
    from pyproj import Transformer
    tr = Transformer.from_crs(g["epsg"], 3413, always_xy=True)
    xs = np.linspace(g["x0"], g["x0"] + g["W"] * 20, 9)
    ys = np.linspace(g["ytop"] - g["H"] * 20, g["ytop"], 9)
    sup = sorted({f"{int((y + 4e6) // 1e5) + 1:02d}_{int((x + 4e6) // 1e5) + 1:02d}"
                  for x, y in (tr.transform(a, b) for a in xs for b in ys)})
    lo0, la0, lo1, la1 = wgs_bbox(g)
    out = []
    for s in sup:
        cat = json.load(urllib.request.urlopen(ADEM + s + ".json", timeout=120))
        for l in cat["links"]:
            if l["rel"] != "child":
                continue
            it = json.load(urllib.request.urlopen(l["href"], timeout=120))
            b = it["bbox"]
            if b[0] < lo1 and b[2] > lo0 and b[1] < la1 and b[3] > la0:
                out.append(it)
    log(f"  плиток ArcticDEM 2 м над листом: {len(out)} ({', '.join(i['id'] for i in out)})")
    return out


def step_dem(site, log):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{site}_10m.npy"
    if p.exists():
        return np.load(p)
    g = grid(site)
    k = int(round(20 / REF_RES_M))
    H, W = g["H"] * k, g["W"] * k
    tr = Affine(REF_RES_M, 0, g["x0"], 0, -REF_RES_M, g["ytop"])
    dem = np.full((H, W), np.nan, "float32")
    meta = []
    for it in adem_items(g, log):
        href = it["assets"]["dem"]["href"]
        t0 = time.time()
        with rasterio.open(href) as src, WarpedVRT(
                src, crs=CRS.from_epsg(g["epsg"]), transform=tr, height=H, width=W,
                resampling=Resampling.average, src_nodata=src.nodata, nodata=np.nan,
                dtype="float32") as vrt:
            a = vrt.read(1)
        fill = np.isnan(dem) & np.isfinite(a)
        dem[fill] = a[fill]
        meta.append({"id": it["id"], "start": it["properties"].get("start_datetime"),
                     "end": it["properties"].get("end_datetime")})
        log(f"    {it['id']}: +{100 * fill.mean():.1f} % листа ({time.time() - t0:.0f} с)")
    np.save(p, dem)
    (CACHE / f"{site}_10m_tiles.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    log(f"  ArcticDEM на сетке {REF_RES_M:.0f} м: покрыто {100 * np.isfinite(dem).mean():.2f} % -> {p}")
    return dem


def hillshade(dem, az, alt, res):
    """Освещённость: скалярное произведение нормали к поверхности и света.

    Строки растра идут на юг, столбцы на восток; свет — азимут от севера по
    часовой и высота над горизонтом."""
    gy, gx = np.gradient(dem, res)
    dzdx, dzdy = gx, -gy                       # на восток и на север
    n = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    a, h = np.radians(az), np.radians(alt)
    lx, ly, lz = np.sin(a) * np.cos(h), np.cos(a) * np.cos(h), np.sin(h)
    return (-dzdx * lx - dzdy * ly + lz) / n


def step_ref2(site, dem, log):
    """Версия 2: тот же LSD на пирамиде масштабов (сшивка через масштаб, MLSD).

    Объявление — audit/AUTO_REFERENCE_V2_DECLARATION.md. Координата отрезка
    LSD — в центрах пикселей, поэтому к ней прибавляется полпикселя: на 80 м
    это 40 м, сравнимо с допуском, и без поправки эталон съехал бы."""
    import cv2
    import geopandas as gpd
    from shapely.geometry import LineString

    REF.mkdir(parents=True, exist_ok=True)
    p = REF / f"{site}_arcticdem_mlsd.gpkg"
    if p.exists():
        return gpd.read_file(p)
    g = grid(site)
    base = np.where(np.isfinite(dem), dem, np.nanmean(dem)).astype("float64")
    lsd = cv2.createLineSegmentDetector()      # параметры по умолчанию
    rows = []
    for s in SCALES_V2_M:
        f = int(round(s / REF_RES_M))
        h, w = base.shape[0] // f * f, base.shape[1] // f * f
        d = base[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
        for az in AZIMUTHS:
            img = np.clip(hillshade(d, az, ALTITUDE, s) * 255, 0, 255).astype("uint8")
            segs = lsd.detect(img)[0]
            segs = np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4)
            L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) * s
            keep = segs[L >= MIN_LEN_M]
            for x1, y1, x2, y2 in keep:
                xy = [(g["x0"] + (x1 + 0.5) * s, g["ytop"] - (y1 + 0.5) * s),
                      (g["x0"] + (x2 + 0.5) * s, g["ytop"] - (y2 + 0.5) * s)]
                rows.append({"scale_m": s, "light_az": az, "geometry": LineString(xy)})
            log(f"  масштаб {s:.0f} м, отмывка {az:.0f}°: отрезков LSD {len(segs)}, "
                f"от {MIN_LEN_M:.0f} м {len(keep)}")
    ref = gpd.GeoDataFrame(rows, crs=f"EPSG:{g['epsg']}")
    ref["length_m"] = ref.length
    ref.to_file(p, driver="GPKG")
    log(f"  эталон Р2: {len(ref)} отрезков, {ref.length_m.sum() / 1000:.0f} км (с дублями) -> {p}")
    return ref


def step_ref(site, dem, log):
    import cv2
    import geopandas as gpd
    from shapely.geometry import LineString

    REF.mkdir(parents=True, exist_ok=True)
    p = REF / f"{site}_arcticdem_lsd.gpkg"
    if p.exists():
        return gpd.read_file(p)
    g = grid(site)
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    lsd = cv2.createLineSegmentDetector()      # параметры по умолчанию
    rows = []
    for az in AZIMUTHS:
        t0 = time.time()
        hs = hillshade(filled, az, ALTITUDE, REF_RES_M)
        img = np.clip(hs * 255, 0, 255).astype("uint8")
        segs = lsd.detect(img)[0]
        segs = np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4)
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) * REF_RES_M
        keep = segs[L >= MIN_LEN_M]
        for x1, y1, x2, y2 in keep:
            xy = [(g["x0"] + x1 * REF_RES_M, g["ytop"] - y1 * REF_RES_M),
                  (g["x0"] + x2 * REF_RES_M, g["ytop"] - y2 * REF_RES_M)]
            rows.append({"light_az": az, "geometry": LineString(xy)})
        log(f"  отмывка {az:.0f}°: отрезков LSD {len(segs)}, от {MIN_LEN_M:.0f} м {len(keep)} "
            f"({time.time() - t0:.0f} с)")
    ref = gpd.GeoDataFrame(rows, crs=f"EPSG:{g['epsg']}")
    ref["length_m"] = ref.length
    ref.to_file(p, driver="GPKG")
    log(f"  эталон Р: {len(ref)} отрезков, {ref.length_m.sum() / 1000:.0f} км (с дублями "
        f"между отмывками) -> {p}")
    return ref


def overpass(bbox, log):
    s, w, n, e = bbox[1], bbox[0], bbox[3], bbox[2]
    q = f"""[out:json][timeout:300];
(way["natural"="water"]({s},{w},{n},{e}); relation["natural"="water"]({s},{w},{n},{e});
 way["waterway"="riverbank"]({s},{w},{n},{e});
 way["waterway"~"^(river|stream|canal)$"]({s},{w},{n},{e});
 way["natural"="wetland"]({s},{w},{n},{e}); relation["natural"="wetland"]({s},{w},{n},{e});
 way["highway"]({s},{w},{n},{e}); way["power"="line"]({s},{w},{n},{e}););
out geom;"""
    data = urllib.parse.urlencode({"data": q}).encode()
    with urllib.request.urlopen(OVERPASS, data=data, timeout=600) as r:
        return json.load(r)


def osm_class(tags):
    if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
        return "shore"
    if tags.get("natural") == "wetland":
        return "wetland_edge"
    if tags.get("waterway") in ("river", "stream", "canal"):
        return "waterway"
    if "highway" in tags:
        return "road"
    if tags.get("power") == "line":
        return "power"
    return None


def step_osm(site, log):
    import geopandas as gpd
    from shapely.geometry import LineString

    REF.mkdir(parents=True, exist_ok=True)
    p = REF / f"{site}_osm.gpkg"
    if p.exists():
        return gpd.read_file(p)
    g = grid(site)
    js = overpass(wgs_bbox(g, pad=1000), log)
    rows = []
    for el in js.get("elements", []):
        cls = osm_class(el.get("tags", {}))
        if cls is None:
            continue
        geoms = ([el["geometry"]] if el["type"] == "way" else
                 [m["geometry"] for m in el.get("members", []) if m.get("type") == "way"
                  and "geometry" in m])
        for gm in geoms:
            if len(gm) >= 2:
                rows.append({"cls": cls, "osm_id": el["id"],
                             "geometry": LineString([(q["lon"], q["lat"]) for q in gm])})
    o = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(epsg=g["epsg"])
    o.to_file(p, driver="GPKG")
    log("  OSM: " + ", ".join(f"{c} {int((o.cls == c).sum())} ({o[o.cls == c].length.sum() / 1000:.0f} км)"
                              for c in sorted(o.cls.unique())) + f" -> {p}")
    return o


def rast(geoms, g):
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    tr = Affine(20.0, 0, g["x0"], 0, -20.0, g["ytop"])
    geoms = [x for x in geoms if x is not None and not x.is_empty]
    if not geoms:
        return np.zeros((g["H"], g["W"]), bool)
    return rasterize(((x, 1) for x in geoms), out_shape=(g["H"], g["W"]), transform=tr,
                     fill=0, dtype="uint8").astype(bool)


def prf(ours, ref, d_ours, d_ref, tol):
    p = float((d_ref[ours] <= tol).mean()) if ours.any() else np.nan
    r = float((d_ours[ref] <= tol).mean()) if ref.any() else np.nan
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def step_eval(site, dem, ref, osm, log, version: int = 1):
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt as edt

    g = grid(site)
    H, W = g["H"], g["W"]
    # ячейки 20 м, где есть ArcticDEM во всех четырёх 10 м ячейках
    k = int(round(20 / REF_RES_M))
    adem_ok = np.isfinite(dem).reshape(H, k, W, k).all(axis=(1, 3))
    ok = g["valid"] & adem_ok
    ok &= edt(ok, sampling=20.0) > TOL_M            # края данных — объявление
    log(f"  область оценки: {100 * ok.mean():.2f} % листа")
    refm = rast(ref.geometry, g) & ok
    d_ref = edt(~refm, sampling=20.0)
    log(f"  эталон Р на сетке 20 м: ячеек {int(refm.sum())} ({refm.sum() * 0.02:.0f} км)")

    if site == "anabar":
        z = np.load(M / "openlin_zone_lines_anabar_v3_бл.npz")
        d1 = np.load(M / "openlin_dem_lines_anabar_одна.npz")
        sets = {"снимок, весь лист": z["C_all"], "снимок, зоны": z["C"],
                "DEM1, добавленные": d1["add"], "снимок + DEM1": z["C_all"] | d1["add"]}
        cells = {"снимок, весь лист": ("openlin_zone_lines_anabar_v3_бл", "C_all"),
                 "DEM1, добавленные": ("openlin_dem_lines_anabar_одна", "add")}
    else:
        z = np.load(M / f"openlin_zone_lines_{site}_v3_перенос.npz")
        sets = {"весь лист": z["C_all"], "зоны": z["C"]}
        cells = {"весь лист": (f"openlin_zone_lines_{site}_v3_перенос", "C_all")}

    rng = np.random.default_rng(SEED)
    m = min(200, H // 8), min(200, W // 8)
    shifts = [(int(rng.integers(m[0], H - m[0] + 1)), int(rng.integers(m[1], W - m[1] + 1)))
              for _ in range(N_SHIFTS)]
    res = {"site": site, "область_доля": float(ok.mean()), "эталон_ячеек": int(refm.sum()),
           "наборы": {}}
    for name, mask in sets.items():
        ours = mask[:H, :W] & ok
        d_o = edt(~ours, sampling=20.0)
        row = {"ячеек": int(ours.sum())}
        for tol in TOL_SWEEP_M:
            row[f"tau{tol:.0f}"] = dict(zip(("P", "R", "F1"), prf(ours, refm, d_o, d_ref, tol)))
        f_obs = row[f"tau{TOL_M:.0f}"]["F1"]
        null = []
        for dy, dx in shifts:
            sh = np.roll(np.roll(mask[:H, :W], dy, 0), dx, 1) & ok
            null.append(prf(sh, refm, edt(~sh, sampling=20.0), d_ref, TOL_M)[2])
        null = np.array(null)
        row["нуль_F1_ср"], row["нуль_F1_макс"] = float(null.mean()), float(null.max())
        row["p"] = (1 + int((null >= f_obs).sum())) / (N_SHIFTS + 1)
        # по участкам 20 x 20 км
        blocks = []
        for i in range(H // BLOCK_CELLS):
            for j in range(W // BLOCK_CELLS):
                sl = (slice(i * BLOCK_CELLS, (i + 1) * BLOCK_CELLS),
                      slice(j * BLOCK_CELLS, (j + 1) * BLOCK_CELLS))
                if ok[sl].mean() < 0.95:
                    continue
                o_, r_ = ours[sl], refm[sl]
                blocks.append({"участок": f"{i}_{j}",
                               **dict(zip(("P", "R", "F1"),
                                          prf(o_, r_, d_o[sl], d_ref[sl], TOL_M)))})
        row["участки"] = blocks
        fb = np.array([b["F1"] for b in blocks])
        t = row[f"tau{TOL_M:.0f}"]
        log(f"  {name}: P {100 * t['P']:.1f} %, R {100 * t['R']:.1f} %, F1 {t['F1']:.3f}; "
            f"нуль F1 {row['нуль_F1_ср']:.3f} / {row['нуль_F1_макс']:.3f}, p = {row['p']:.4f}; "
            f"по {len(blocks)} участкам F1 медиана {np.median(fb):.3f}, {fb.min():.3f}-{fb.max():.3f}")
        log("    чувствительность к τ 30/60/120 м: F1 " + " / ".join(
            f"{row[f'tau{x:.0f}']['F1']:.3f}" for x in TOL_SWEEP_M))
        res["наборы"][name] = row

    # классы ложных линий (OSM; на Анабаре ещё гидросеть заказчика)
    dist_cls = {c: edt(~rast(osm[osm.cls == c].geometry, g), sampling=20.0)
                for c in ("shore", "wetland_edge", "waterway", "road", "power")}
    if site == "anabar":
        from experiments.openlin_hydro_check import hydro_mask
        dist_cls["гидросеть заказчика"] = edt(~hydro_mask((H, W), 20.0, g["x0"], g["ytop"],
                                                        g["epsg"], lambda *_: None), sampling=20.0)
    res["классы"] = {}
    for name, (stem, key) in cells.items():
        d = np.load(M / f"{stem}_линии.npz")
        pts, ids = d[f"{key}_pts"], d[f"{key}_id"]
        n = int(ids.max()) + 1 if len(ids) else 0
        share = {}
        for c, dc in dist_cls.items():
            near = dc[pts[:, 0], pts[:, 1]] <= TOL_M
            frac = np.bincount(ids, weights=near, minlength=n) / np.bincount(ids, minlength=n)
            share[c] = float((frac >= CLASS_FRAC).mean())
        res["классы"][name] = share
        log(f"  классы ({name}, доля линий, лежащих вдоль): " + ", ".join(
            f"{c} {100 * v:.1f} %" for c, v in share.items()))
    if site in GTK_SITES:
        from experiments.openlin_transfer_check_k1 import gtk_path
        gtk = gpd.read_file(gtk_path(site, 292)).to_crs(epsg=g["epsg"])
        gm = rast(gtk.geometry, g) & ok
        p_, r_, f_ = prf(refm, gm, d_ref, edt(~gm, sampling=20.0), GTK_TOL_M)
        res["эталон_против_GTK"] = {"P": p_, "R": r_, "F1": f_}
        ow = z["C_all"][:H, :W] & ok
        p2, r2, f2 = prf(ow, gm, edt(~ow, sampling=20.0), edt(~gm, sampling=20.0), GTK_TOL_M)
        res["наши_против_GTK"] = {"P": p2, "R": r2, "F1": f2}
        log(f"  проверка эталона: эталон Р против GTK 292 при {GTK_TOL_M:.0f} м — "
            f"P {100 * p_:.1f} %, R {100 * r_:.1f} %, F1 {f_:.3f}; для сравнения наши "
            f"(весь лист) против GTK: P {100 * p2:.1f} %, R {100 * r2:.1f} %, F1 {f2:.3f}")
    if site == "lapland" and version == 2:
        better = res["эталон_против_GTK"]["F1"] > V1_F1_VS_GTK
        res["основной_эталон"] = 2 if better else 1
        log(f"  правило выбора версии: F1 против GTK {res['эталон_против_GTK']['F1']:.3f} "
            f"против {V1_F1_VS_GTK:.3f} у версии 1 -> основной эталон — версия {res['основной_эталон']}")
    out = REF / (f"{site}_eval.json" if version == 1 else f"{site}_eval_v{version}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  сохранено: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--версия", dest="version", type=int, default=1, choices=(1, 2),
                    help="эталон: 1 — LSD на 10 м; 2 — LSD на пирамиде 10-80 м (сшивка)")
    a = ap.parse_args()
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    log(f"площадка {a.site}: ArcticDEM")
    dem = step_dem(a.site, log)
    log("эталон Р (LSD по отмывкам):")
    ref = step_ref(a.site, dem, log) if a.version == 1 else step_ref2(a.site, dem, log)
    log("OSM:")
    osm = step_osm(a.site, log)
    log("сравнение:")
    step_eval(a.site, dem, ref, osm, log, a.version)


if __name__ == "__main__":
    main()
