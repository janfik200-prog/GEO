"""Проверочный (не тракт): выгрузка OSM (Overpass) и WorldCover 10 м -> сетка 20 м для любой площадки.

Обобщение openlin_artifact_lapland_osm / _wc (те не тронуты). Кэш:
datacache/osm/<site>_<слой>.json, datacache/worldcover/wc_<site>_20m.npz.
Сеть — только через прокси (HTTPS_PROXY=http://127.0.0.1:10809).
Запуск:  python -m experiments.openlin_artifact_site_data --site utsjoki
"""
from __future__ import annotations
import argparse, json, os, urllib.parse, urllib.request
from pathlib import Path
import numpy as np

from experiments.openlin_artifact_lapland_osm import ENDPOINTS, PROXY, QUERIES

OSM = Path("datacache/osm")
WCD = Path("datacache/worldcover")


def comp_path(site):
    return Path(f"datacache/s2_lowsun/composite/winter_{site}_20m_лист.npz")


def bbox_wgs(site):
    from pyproj import Transformer
    w = np.load(comp_path(site))
    tr = Transformer.from_crs(f"EPSG:{int(w['epsg'])}", "EPSG:4326", always_xy=True)
    H, W = w["count"].shape
    x0, yt, r = float(w["x0"]), float(w["y_top"]), float(w["res_m"])
    pts = [tr.transform(x, y) for x in (x0, x0 + W * r) for y in (yt, yt - H * r)]
    lon = [p[0] for p in pts]; lat = [p[1] for p in pts]
    return min(lat), min(lon), max(lat), max(lon)


def fetch(q, bb):
    s, w, n, e = bb
    body = f'[out:json][timeout:180];{q}({s:.4f},{w:.4f},{n:.4f},{e:.4f});out geom;'
    op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY}))
    last = None
    for ep in ENDPOINTS:
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": body}).encode(),
                                         headers={"User-Agent": "geo-openlin-artifact-check/1.0"})
            return json.loads(op.open(req, timeout=240).read())
        except Exception as ex:  # noqa
            last = f"{ep}: {ex}"
    raise RuntimeError(last)


def osm(site):
    OSM.mkdir(parents=True, exist_ok=True)
    bb = bbox_wgs(site)
    print("bbox s,w,n,e", bb, flush=True)
    for k, q in QUERIES.items():
        f = OSM / f"{site}_{k}.json"
        if f.exists():
            continue
        try:
            d = fetch(q, bb)
        except Exception as ex:
            print(k, "ОШИБКА", ex, flush=True); continue
        f.write_text(json.dumps(d), encoding="utf-8")
        print(k, len(d.get("elements", [])), "ways", flush=True)


def wc(site):
    import planetary_computer as pc, rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    from pystac_client import Client
    out_f = WCD / f"wc_{site}_20m.npz"
    s, w, n, e = bbox_wgs(site)
    z = np.load(comp_path(site))
    H, W = z["count"].shape
    x0, yt, r = float(z["x0"]), float(z["y_top"]), float(z["res_m"])
    pad = 0.01
    lo, la0, hi, la1 = w - pad, s - pad, e + pad, n + pad
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    its = list(cat.search(collections=["esa-worldcover"], bbox=[lo, la0, hi, la1],
                          query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    res = 8.333333333333333e-05 * 2
    NW = int(np.ceil((hi - lo) / res)); NH = int(np.ceil((la1 - la0) / res))
    mos = np.zeros((NH, NW), np.uint8)
    for it in its:
        with rasterio.open(it.assets["map"].href) as src:
            b = src.bounds
            L, B, R, T = max(lo, b.left), max(la0, b.bottom), min(hi, b.right), min(la1, b.top)
            if L >= R or B >= T:
                continue
            win = from_bounds(L, B, R, T, src.transform)
            c0 = int(round((L - lo) / res)); r0 = int(round((la1 - T) / res))
            h = int(round((T - B) / res)); ww = int(round((R - L) / res))
            a = src.read(1, window=win, out_shape=(h, ww), resampling=rasterio.enums.Resampling.nearest)
            mos[r0:r0 + h, c0:c0 + ww] = a[:NH - r0, :NW - c0]
        print(it.id, flush=True)
    tr = Transformer.from_crs(f"EPSG:{int(z['epsg'])}", "EPSG:4326", always_xy=True)
    out = np.zeros((H, W), np.uint8)
    xx = x0 + (np.arange(W) + 0.5) * r
    for i in range(0, H, 200):
        rr = np.arange(i, min(i + 200, H)); yy = yt - (rr + 0.5) * r
        X, Y = np.meshgrid(xx, yy)
        lon, lat = tr.transform(X, Y)
        ci = np.clip(((lon - lo) / res).astype(int), 0, NW - 1)
        ri = np.clip(((la1 - lat) / res).astype(int), 0, NH - 1)
        out[i:i + len(rr)] = mos[ri, ci]
    WCD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_f, wc=out)
    v, c = np.unique(out, return_counts=True)
    print(dict(zip(v.tolist(), (100 * c / c.sum()).round(2).tolist())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--site", required=True)
    ap.add_argument("--что", default="osm,wc")
    a = ap.parse_args()
    for part in a.что.split(","):
        {"osm": osm, "wc": wc}[part](a.site)
