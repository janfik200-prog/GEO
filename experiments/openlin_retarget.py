"""Смена цели сверки на уже собранной площадке (снимок не перекачивается).

ЗАЧЕМ. Нуль-тест на Анабаре показал не слабость метода, а слабость ЦЕЛИ: шесть
сплошных полигонов зон разломов дают крупные связные пятна, и случайно сдвинутая
карта попадает в них с lift'ом до 2.9 (q95). При такой ширине нуля наблюдённые
1.57 неотличимы от случайности при любом методе.

Дайки для этой роли годятся лучше по трём причинам сразу: их 151 против 6, они
раскиданы по всему листу, и на щите дайка — это заполненная трещина, то есть
законный прокси разломной сети (в задаче №3 проекта они использовались именно
так). Более дробная и разбросанная цель сужает нулевое распределение, то есть
возвращает тесту разрешающую способность.

Композит, NDVI и радар берутся из готового файла площадки — меняется только
поле расстояний ``dist``, поэтому пересчёт идёт секунды.

Запуск из корня репозитория:
    python -m experiments.openlin_retarget --site anabar --target dykes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_probe import Grid, SITES, _read_pj4, build_grid

#: Доступные цели: имя -> (файлы слоя, брать ли границу вместо самой геометрии).
#: У зон разломов цель — ГРАНИЦА полигона (сам полигон это коридор шириной
#: 8-21 км, его «внутренность» разломом не является). У даек наоборот: полигон и
#: есть тело дайки, у него ширина в первые сотни метров.
TARGETS: dict[str, tuple[tuple[str, ...], bool]] = {
    "zones": (("data/Gis-integro/shp_dbf/glub_raz_nw.shp",
               "data/Gis-integro/shp_dbf/glub_r_nw.shp"), True),
    "dykes": (("data/Gis-integro/shp_dbf/dayki_buf.shp",), False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--target", choices=sorted(TARGETS), default="dykes")
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt

    src = Path("outputs/metrics") / f"openlin_{args.site}.npz"
    if not src.exists():
        raise SystemExit(f"нет {src} — сначала прогоните openlin_probe --site {args.site}")
    z = dict(np.load(src))

    site = SITES[args.site]
    grid = build_grid(site, float(z["res_m"]))
    paths, use_boundary = TARGETS[args.target]

    parts = []
    for p in paths:
        shp = Path(p)
        g = gpd.read_file(shp)
        if g.crs is None:
            g = g.set_crs(_read_pj4(shp))
        g = g.to_crs(f"EPSG:{site.epsg}")
        parts.append(g.boundary if use_boundary else g.geometry)
        print(f"  {shp.name}: {len(g)} объектов")
    f = gpd.GeoDataFrame(geometry=pd.concat(parts, ignore_index=True),
                         crs=f"EPSG:{site.epsg}")

    minx, maxx = grid.x0, grid.x0 + grid.width * grid.res_m
    miny = grid.y_top - grid.height * grid.res_m
    f = f.cx[minx:maxx, miny:grid.y_top]
    print(f"на площадке: {len(f)} объектов, {f.length.sum() / 1000:.0f} км")
    if len(f) == 0:
        raise SystemExit("на площадке нет объектов цели")

    mask = rasterize(((geom, 1) for geom in f.geometry), out_shape=grid.shape,
                     transform=grid.transform, fill=0, dtype="uint8")
    z["dist"] = distance_transform_edt(mask == 0,
                                       sampling=grid.res_m).astype("float32")
    print(f"ячеек цели: {int((z['dist'] <= 250).sum())} при допуске 250 м "
          f"({100 * np.mean(z['dist'] <= 250):.2f}% площадки)")

    out = Path("outputs/metrics") / f"openlin_{args.site}_{args.target}.npz"
    np.savez_compressed(out, **z)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
