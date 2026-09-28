"""Разведка: какие сцены с низким солнцем реально накрывают фрагмент."""
from __future__ import annotations

import numpy as np
from shapely.geometry import shape, box
from pyproj import CRS, Transformer

from experiments.openlin_lowsun import scene_grid, FRAG_KM, RES_M, MAX_CLOUD


def frag_poly(grid):
    tr = Transformer.from_crs(CRS.from_epsg(grid.epsg), CRS.from_epsg(4326),
                              always_xy=True)
    x1 = grid.x0 + grid.width * grid.res_m
    y1 = grid.y_top - grid.height * grid.res_m
    pts = [tr.transform(x, y) for x, y in
           ((grid.x0, grid.y_top), (x1, grid.y_top), (x1, y1), (grid.x0, y1))]
    from shapely.geometry import Polygon
    return Polygon(pts)


def main():
    import planetary_computer as pc
    from pystac_client import Client
    from experiments.openlin_probe import STAC_URL

    grid = scene_grid("anabar", FRAG_KM, RES_M)
    fp = frag_poly(grid)
    cat = Client.open(STAC_URL, modifier=pc.sign_inplace)
    res = cat.search(collections=["sentinel-2-l2a"], bbox=grid.bbox_wgs84(),
                     datetime="2019-01-01/2025-12-31",
                     query={"eo:cloud_cover": {"lt": 25}})
    rows = []
    for it in res.items():
        sz = it.properties.get("s2:mean_solar_zenith")
        if sz is None:
            continue
        sun = 90.0 - float(sz)
        g = shape(it.geometry)
        cov = g.intersection(fp).area / fp.area
        rows.append((sun, cov, float(it.properties.get("eo:cloud_cover", 99)),
                     it.datetime, it.id,
                     it.properties.get("s2:mean_solar_azimuth"),
                     it.properties.get("s2:nodata_pixel_percentage")))
    rows.sort(key=lambda r: r[0])
    print(f"всего сцен: {len(rows)}")
    full = [r for r in rows if r[1] > 0.99]
    print(f"полностью накрывают фрагмент: {len(full)}")
    print("\nсамые низкие солнца среди полностью накрывающих:")
    print(f"{'солнце':>7} {'покр':>6} {'обл':>5} {'пусто%':>7} {'азимут':>7}  дата")
    for sun, cov, cl, dt, sid, az, nod in full[:25]:
        nd = -1 if nod is None else float(nod)
        print(f"{sun:7.1f} {cov:6.2f} {cl:5.1f} {nd:7.1f} "
              f"{(az or 0):7.1f}  {dt:%Y-%m-%d} {sid}")
    print("\nдля справки — низкие солнца независимо от покрытия:")
    for sun, cov, cl, dt, sid, az, nod in rows[:12]:
        print(f"{sun:7.1f} {cov:6.2f} {cl:5.1f}  {dt:%Y-%m-%d}")


if __name__ == "__main__":
    main()
