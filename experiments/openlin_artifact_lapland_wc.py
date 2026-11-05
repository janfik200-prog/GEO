"""Проверочный: WorldCover 10 м -> сетка листа Лапландии 20 м (кэш datacache/worldcover/wc_lapland_20m.npz).

Только для проверки артефактов (вода класс 80, болото 90, вырубки не различаются
отдельно: класс 10 лес, 20 кустарник, 30 луг, 40 пашня). В тракт не идёт.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

OUT = Path("datacache/worldcover/wc_lapland_20m.npz")
COMP = "datacache/s2_lowsun/composite/winter_lapland_20m_лист.npz"


def main():
    import planetary_computer as pc, rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    from pystac_client import Client
    from experiments.openlin_artifact_lapland_osm import bbox_wgs
    s, w, n, e = bbox_wgs()
    z = np.load(COMP)
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
    cc = np.arange(W); xx = x0 + (cc + 0.5) * r
    for i in range(0, H, 200):
        rr = np.arange(i, min(i + 200, H)); yy = yt - (rr + 0.5) * r
        X, Y = np.meshgrid(xx, yy)
        lon, lat = tr.transform(X, Y)
        ci = np.clip(((lon - lo) / res).astype(int), 0, NW - 1)
        ri = np.clip(((la1 - lat) / res).astype(int), 0, NH - 1)
        out[i:i + len(rr)] = mos[ri, ci]
    np.savez_compressed(OUT, wc=out)
    v, c = np.unique(out, return_counts=True)
    print(dict(zip(v.tolist(), (100 * c / c.sum()).round(2).tolist())))


if __name__ == "__main__":
    main()
