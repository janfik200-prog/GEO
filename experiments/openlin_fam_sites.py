"""FAM2: выбор листов для обучения модели по эталону GTK 292 (север Финляндии).

ЗАЧЕМ. FAM1 (Утсйоки, обучение запад → проверка восток) показал: на одном
листе для обучения около десятка линий эталона, модель учит частности и
проигрывает одиночному признаку. Эталон GTK 292 есть по всей Финляндии —
берутся ещё листы.

ПРАВИЛО ОТБОРА (объявлено 10.11.2026 до расчёта, в audit/FAM2_DECLARATION.md):
  1) квадрат ``TILE_KM`` (как лист Анабара), шаг перебора центров 20 км, в
     TM35FIN — те же числа, что в openlin_site_scan;
  2) центр не южнее Полярного круга (66.56° с. ш.) — ближе к Анабару по ландшафту;
  3) в Финляндии не меньше 95 % (иначе часть листа без эталона);
  4) не ближе 20 км к контрольной Лапландии и к Утсйоки (настройка/проверка);
  5) GTK 292 внутри не меньше 947 км (половина лапландских, как в site_scan);
  6) порядок — по широте центра с севера; жадно, без перекрытия с уже
     выбранными; не больше ``N_PICK`` = 8.

Запуск: python -m experiments.openlin_fam_sites   (сеть через прокси)

FAM3 (audit/FAM3_DECLARATION.md): ``--relief`` — те же условия 1–4, порядок по
убыванию расчленённости R (медиана размаха высот GLO-30 в окне 1 км на сетке
200 м), 3 листа, список в audit/fam3_sites.json.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from experiments.openlin_probe import TILE_KM

GTK_DIR = Path("datacache/gtk_lineaments")
NORTH = GTK_DIR / "north_lin500k_292.gpkg"
NE_DIR = Path("datacache/natural_earth")
NE_ZIP = NE_DIR / "ne_10m_admin_0_countries.zip"
NE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
GTK_URL = ("https://gtkdata.gtk.fi/arcgis/rest/services/Rajapinnat/"
           "kokoavaWMS/MapServer/292/query")
OUT = Path("audit/fam2_sites.json")
EPSG_FI = 3067
STEP_M = 20_000
LAT_MIN = 66.56
FIN_MIN, GAP_M, L292_MIN_KM = 0.95, 20_000, 0.0   # поправка 1: условие 5 снято
N_PICK = 8


def fetch_north():
    import geopandas as gpd
    if NORTH.exists():
        return gpd.read_file(NORTH)
    feats, off = [], 0
    while True:
        q = {"where": "1=1", "outFields": "*", "f": "geojson",
             "geometry": "19.0,66.0,32.0,70.2", "geometryType": "esriGeometryEnvelope",
             "inSR": 4326, "outSR": EPSG_FI, "spatialRel": "esriSpatialRelIntersects",
             "resultOffset": off, "resultRecordCount": 1000}
        with urllib.request.urlopen(GTK_URL + "?" + urllib.parse.urlencode(q), timeout=180) as r:
            got = json.load(r).get("features", [])
        feats += got
        print(f"  GTK 292: {len(feats)} объектов", flush=True)
        if len(got) < 1000:
            break
        off += len(got)
    g = gpd.GeoDataFrame.from_features(feats, crs=f"EPSG:{EPSG_FI}")
    GTK_DIR.mkdir(parents=True, exist_ok=True)
    g.to_file(NORTH, driver="GPKG")
    return g


def finland():
    import geopandas as gpd
    if not NE_ZIP.exists():
        NE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NE_URL, NE_ZIP)
    f = gpd.read_file(NE_ZIP)
    return f[f["ADMIN"] == "Finland"].to_crs(EPSG_FI).union_all()


def site_box(name):
    import geopandas as gpd
    from shapely.geometry import box
    from experiments.openlin_probe import SITES, build_grid
    g = build_grid(SITES[name], 1000.0)
    b = box(g.x0, g.y_top - g.height * g.res_m, g.x0 + g.width * g.res_m, g.y_top)
    return gpd.GeoSeries([b], crs=g.epsg).to_crs(EPSG_FI).iloc[0]


# ---- FAM3: расчленённость рельефа -----------------------------------------
OUT3 = Path("audit/fam3_sites.json")
R_RES_M, R_WIN_M, N_PICK3 = 200.0, 1000.0, 3
MOSAIC = Path("datacache/anabar_dem/north_fi_glo30_200m.npy")


def relief_R(dem):
    """Медиана размаха высот в окне R_WIN_M (дыры — NaN, окна с дырами не идут)."""
    from scipy import ndimage as ndi
    w = int(round(R_WIN_M / R_RES_M))
    ok = np.isfinite(dem)
    d = np.where(ok, dem, 0.0)
    rng = ndi.maximum_filter(d, w) - ndi.minimum_filter(d, w)
    full = ndi.minimum_filter(ok.astype(np.uint8), w).astype(bool)
    return float(np.median(rng[full])) if full.any() else float("nan")


def north_mosaic(fin):
    """GLO-30 по всему северу Финляндии на 200 м (TM35FIN), кэш."""
    from rasterio.enums import Resampling
    from experiments.openlin_probe import Grid, _read_on_grid
    from src.dem_features import _tile_url
    x0, y0, x1, y1 = fin.bounds
    import pyproj
    ymin = pyproj.Transformer.from_crs(4326, EPSG_FI, always_xy=True).transform(25.0, LAT_MIN - 0.5)[1]
    y0 = max(y0, ymin)
    g = Grid(x0=x0, y_top=y1, res_m=R_RES_M, height=int((y1 - y0) // R_RES_M) + 1,
             width=int((x1 - x0) // R_RES_M) + 1, epsg=EPSG_FI)
    if MOSAIC.exists():
        return np.load(MOSAIC), g
    lon0, lat0, lon1, lat1 = g.bbox_wgs84(0)
    dem = np.zeros((g.height, g.width), "float32")
    tdir = MOSAIC.parent / "north_fi_tiles"
    tdir.mkdir(parents=True, exist_ok=True)
    failed = []
    for lat in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1):
        for lon in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1):
            tp = tdir / f"{lat}_{lon}.npy"
            if not tp.exists():
                for k in range(5):                  # обрывы сети — повтор
                    try:
                        np.save(tp, _read_on_grid(_tile_url(lat, lon), g, Resampling.bilinear))
                        break
                    except Exception as exc:
                        msg = repr(exc)
                        if "No such file" in msg or "not recognized" in msg or "404" in msg:
                            np.save(tp, np.zeros((g.height, g.width), "float32"))  # плитки нет (море)
                            print(f"  плитка {lat},{lon}: нет в каталоге", flush=True)
                            break
                        print(f"  плитка {lat},{lon}: обрыв, попытка {k + 1} ({msg:.60})", flush=True)
                else:
                    failed.append((lat, lon))
                    continue
            part = np.load(tp)
            dem = np.where(part != 0, part, dem)
            print(f"  плитка {lat},{lon}: {100 * (part != 0).mean():.0f} % сетки", flush=True)
    if failed:
        raise SystemExit(f"не скачаны плитки {failed} — мозаика не сохранена, запустить ещё раз")
    dem = np.where(dem != 0, dem, np.nan).astype("float32")
    np.save(MOSAIC, dem)
    return dem, g


def main_relief():
    import geopandas as gpd
    from shapely.geometry import box
    from experiments.openlin_relief_probe import load_dem
    fin = finland()
    dem, g = north_mosaic(fin)
    keep_out = [site_box("lapland").buffer(GAP_M), site_box("utsjoki").buffer(GAP_M)]
    w, h = TILE_KM[0] * 1000, TILE_KM[1] * 1000
    x0, y0, x1, y1 = fin.bounds
    rows = []
    for cy in np.arange(y0 + h / 2, y1 - h / 2 + 1, STEP_M):
        for cx in np.arange(x0 + w / 2, x1 - w / 2 + 1, STEP_M):
            t = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            c = gpd.GeoSeries([t.centroid], crs=EPSG_FI).to_crs(4326).iloc[0]
            if c.y < LAT_MIN:
                continue
            ff = t.intersection(fin).area / t.area
            if ff < FIN_MIN or any(t.intersects(k) for k in keep_out):
                continue
            r0, c0 = int((g.y_top - (cy + h / 2)) / R_RES_M), int((cx - w / 2 - g.x0) / R_RES_M)
            sub = dem[max(r0, 0):r0 + int(h / R_RES_M), max(c0, 0):c0 + int(w / R_RES_M)]
            rows.append(dict(lat=round(c.y, 3), lon=round(c.x, 3), fin=round(ff, 3),
                             R=round(relief_R(sub), 1), box=t))
    print(f"квадратов, прошедших 1–4: {len(rows)}; R от {min(r['R'] for r in rows)} "
          f"до {max(r['R'] for r in rows)} м, медиана {np.median([r['R'] for r in rows]):.1f}")
    rows.sort(key=lambda r: -r["R"])
    pick = []
    for r in rows:
        if all(not r["box"].intersects(p["box"]) for p in pick):
            pick.append(r)
        if len(pick) == N_PICK3:
            break
    out = [{"name": f"fam_r{i + 1}", **{k: v for k, v in r.items() if k != "box"}}
           for i, r in enumerate(pick)]
    for r in out:
        print(r)
    print("R прочих листов (описательно, GLO-30 на сетке листа 200 м):")
    for s in ("anabar", "utsjoki", "lapland", "finland65", "fam_n1", "fam_n2", "fam_n3"):
        print(f"  {s}: R = {relief_R(load_dem(s, R_RES_M)):.1f} м", flush=True)
    OUT3.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("сохранено:", OUT3)


def main():
    import geopandas as gpd
    from shapely.geometry import box
    gtk = fetch_north().to_crs(EPSG_FI)
    fin = finland()
    keep_out = [site_box("lapland").buffer(GAP_M), site_box("utsjoki").buffer(GAP_M)]
    w, h = TILE_KM[0] * 1000, TILE_KM[1] * 1000
    x0, y0, x1, y1 = fin.bounds
    rows = []
    for cy in np.arange(y0 + h / 2, y1 - h / 2 + 1, STEP_M):
        for cx in np.arange(x0 + w / 2, x1 - w / 2 + 1, STEP_M):
            t = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            c = gpd.GeoSeries([t.centroid], crs=EPSG_FI).to_crs(4326).iloc[0]
            if c.y < LAT_MIN:
                continue
            ff = t.intersection(fin).area / t.area
            if ff < FIN_MIN or any(t.intersects(k) for k in keep_out):
                continue
            L = float(gtk.clip(t).length.sum() / 1000)
            if L < L292_MIN_KM:
                continue
            rows.append(dict(lat=round(c.y, 3), lon=round(c.x, 3), fin=round(ff, 3),
                             L292=round(L, 1), box=t))
    print(f"квадратов, прошедших 1–5: {len(rows)}")
    rows.sort(key=lambda r: -r["lat"])
    pick = []
    for r in rows:
        if all(not r["box"].intersects(p["box"]) for p in pick):
            pick.append(r)
        if len(pick) == N_PICK:
            break
    out = [{"name": f"fam_n{i + 1}", **{k: v for k, v in r.items() if k != "box"}}
           for i, r in enumerate(pick)]
    for r in out:
        print(r)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("сохранено:", OUT)


if __name__ == "__main__":
    import sys
    main_relief() if "--relief" in sys.argv else main()
