"""Проверочный: выгрузка дорог, ЛЭП, просек и воды OSM для листа Лапландии (Overpass через прокси).

Только кэш в datacache/osm/; в тракт ничего не идёт.
"""
from __future__ import annotations
import json, sys, urllib.request, urllib.parse
from pathlib import Path
import numpy as np

OUT = Path("datacache/osm")
PROXY = "http://127.0.0.1:10809"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
QUERIES = {
    "roads": 'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|track|path)$"]',
    "power": 'way["power"~"^(line|minor_line)$"]',
    "cutline": 'way["man_made"="cutline"]',
    "railway": 'way["railway"~"^(rail|abandoned|disused)$"]',
    "water": 'way["natural"="water"]',
    "waterway": 'way["waterway"~"^(river|stream|canal|ditch)$"]',
}


def bbox_wgs(npz="datacache/s2_lowsun/composite/winter_lapland_20m_лист.npz"):
    from pyproj import Transformer
    w = np.load(npz)
    tr = Transformer.from_crs(f"EPSG:{int(w['epsg'])}", "EPSG:4326", always_xy=True)
    H, W = w["count"].shape
    x0, yt, r = float(w["x0"]), float(w["y_top"]), float(w["res_m"])
    xs = [x0, x0 + W * r]; ys = [yt, yt - H * r]
    pts = [tr.transform(x, y) for x in xs for y in ys]
    lon = [p[0] for p in pts]; lat = [p[1] for p in pts]
    return min(lat), min(lon), max(lat), max(lon)


def fetch(name, q, bb):
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bb = bbox_wgs()
    print("bbox s,w,n,e", bb)
    for k, q in QUERIES.items():
        f = OUT / f"lapland_{k}.json"
        if f.exists():
            continue
        try:
            d = fetch(k, q, bb)
        except Exception as ex:
            print(k, "ОШИБКА", ex); continue
        f.write_text(json.dumps(d), encoding="utf-8")
        print(k, len(d.get("elements", [])), "ways")


if __name__ == "__main__":
    main()
