"""REAL1: свидетели на сетке листа Утсйоки 20 м — летний S2 (B04, B08) и летний Landsat 8/9.

audit/REAL1_DECLARATION.md. Лидар не используется. Слои кладутся в
datacache/typesites/_utsjoki/ (s2w_summer.npz, landsat_summer.npz); готовые не качаются повторно.
    python -m experiments.openlin_real1_fetch
"""
from __future__ import annotations

import numpy as np

from experiments.openlin_typesites_fetch import (DEST, PC_STAC, S2_BAD_SCL, _distinct_dates,
                                                 _s2_stack, date_groups, months_for, read_on_grid,
                                                 save_layer, search)

N_DATES = 8          # как --s2-max участков


def layer_s2w(site, grid, log):
    items = [it for it in search(PC_STAC, ["sentinel-2-l2a"], grid, query={"eo:cloud_cover": {"lt": 10}})
             if it.datetime.month in months_for(site["lat"])]
    groups = date_groups(items, grid, N_DATES)
    log(f"    S2 лето: дат {len(groups)}")
    out, used = _s2_stack(groups, grid, ("B04", "B08"), log, "S2 лето", S2_BAD_SCL + (11,))
    return out, used


def layer_landsat(site, grid, log):
    from rasterio.enums import Resampling
    ms = months_for(site["lat"])
    items = [it for it in search(PC_STAC, ["landsat-c2-l2"], grid,
                                 query={"eo:cloud_cover": {"lt": 10},
                                        "platform": {"in": ["landsat-8", "landsat-9"]}})
             if it.datetime.month in ms]
    items = _distinct_dates(sorted(items, key=lambda it: it.properties["eo:cloud_cover"]), N_DATES)
    reds, nirs, used = [], [], []
    for it in items:
        try:
            qa = read_on_grid(it.assets["qa_pixel"].href, grid, Resampling.nearest)
            # биты QA_PIXEL C2: 1 dilated cloud, 3 cloud, 4 cloud shadow, 5 snow; 0 — нет данных
            q = np.nan_to_num(qa, nan=1).astype(np.int64)
            bad = (q == 1) | ((q >> 1) & 1).astype(bool) | ((q >> 3) & 1).astype(bool) \
                | ((q >> 4) & 1).astype(bool) | ((q >> 5) & 1).astype(bool)
            r = read_on_grid(it.assets["red"].href, grid, Resampling.bilinear)
            n = read_on_grid(it.assets["nir08"].href, grid, Resampling.bilinear)
            reds.append(np.where(bad, np.nan, r)); nirs.append(np.where(bad, np.nan, n))
            used.append(it.id)
            log(f"    Landsat: {len(used)}/{len(items)} {it.id}")
        except Exception as exc:
            log(f"    Landsat: {it.id} ошибка {exc!r:.100}")
    if not used:
        raise RuntimeError("Landsat: ни одной сцены")
    return {"red": np.nanmedian(np.stack(reds), 0).astype("float32"),
            "nir": np.nanmedian(np.stack(nirs), 0).astype("float32")}, used


def main():
    import json

    import sys

    from experiments.openlin_probe import SITES
    from experiments.openlin_winter import site_grid
    name = sys.argv[sys.argv.index("--site") + 1] if "--site" in sys.argv else "utsjoki"   # LEN1: + lapland
    grid = site_grid(name, 20.0)
    d = DEST / f"_{name}"
    d.mkdir(parents=True, exist_ok=True)
    site = {"lat": 69.551, "lon": 27.142} if name == "utsjoki" else {"lat": SITES[name].lat, "lon": SITES[name].lon}
    meta = {"layers": {}}
    log = lambda m: print(m, flush=True)
    save_layer(d, "s2w_summer", layer_s2w, site, grid, log, meta)
    save_layer(d, "landsat_summer", layer_landsat, site, grid, log, meta)
    (d / "meta_real1.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
