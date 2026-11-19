"""Пакет 09 ТЗ К1: выбор третьей площадки по правилу audit/PACKET09_DECLARATION.md.

Кандидаты — центры 61.0-68.5° с. ш. (шаг 0.5°) x 21-31° в. д. (шаг 1°);
допустим, если лист 86 x 77 км с полями 5 км целиком в Финляндии (Natural
Earth 1:50 млн), не перекрывает Лапландию и набирает 12 зимних сцен с
солнцем <= 12° так, что все пробные точки накрыты >= 3 сценами (cover_pick).
Порядок — default_rng(20261028), берётся первый допустимый. Эталон GTK и
результаты тракта в выборе не участвуют.

Запуск из корня:
    python -m experiments.openlin_pick_site
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

SEED = 20261028
NE = Path("datacache/naturalearth/ne_50m_admin_0_countries.zip")
OUT = Path("audit/third_site_selection.json")


def main() -> None:
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box

    from experiments.openlin_probe import SITES, TILE_KM, Site, build_grid
    from experiments import openlin_lowsun
    from experiments.openlin_winter import MIN_SCENES, cover_pick

    def log(m=""):
        print(m, flush=True)

    with zipfile.ZipFile(NE) as z:
        z.extractall(NE.parent / "ne50")
    ne = gpd.read_file(NE.parent / "ne50" / "ne_50m_admin_0_countries.shp")
    fin = ne[ne["ADMIN"] == "Finland"].geometry.union_all()
    lap = build_grid(SITES["lapland"], 100.0)

    cands = [(lat, lon) for lat in np.arange(61.0, 68.51, 0.5) for lon in np.arange(21.0, 31.01, 1.0)]
    order = np.random.default_rng(SEED).permutation(len(cands))
    log(f"кандидатов {len(cands)}, порядок — зерно {SEED}")
    tried = []
    for k in order:
        lat, lon = map(float, cands[k])
        epsg = 32600 + int((lon + 180) // 6) + 1
        s = Site(name="cand", lat=lat, lon=lon, faults=None, fault_query=None, epsg=epsg, comment="")
        g = build_grid(s, 100.0)
        b = g.bbox_wgs84(pad_m=5000.0)
        rec = {"lat": lat, "lon": lon, "epsg": epsg}
        if not fin.contains(box(*b)):
            rec["отказ"] = "рамка не целиком в Финляндии"
            tried.append(rec); continue
        # перекрытие с Лапландией — в проекции Лапландии
        tr = Transformer.from_crs(4326, lap.epsg, always_xy=True)
        cx, cy = tr.transform(lon, lat)
        lx = lap.x0 + lap.width * 50.0
        ly = lap.y_top - lap.height * 50.0
        if abs(cx - lx) < TILE_KM[0] * 1000 and abs(cy - ly) < TILE_KM[1] * 1000:
            rec["отказ"] = "перекрывает Лапландию"
            tried.append(rec); continue
        g20 = build_grid(s, 20.0)
        items = openlin_lowsun.find_scenes(g20, lambda *_: None, min_cover=0.02)
        picked = cover_pick(items, g20, 12, lambda *_: None) if items else []
        # доля пробных точек с >= MIN_SCENES сценами — как в cover_pick
        from shapely.geometry import Point, shape
        from shapely.prepared import prep
        from experiments.openlin_lowsun import frag_polygon
        fp = frag_polygon(g20)
        x0, y0, x1, y1 = fp.bounds
        n = int(np.sqrt(120))
        probes = [Point(x, y) for x in np.linspace(x0, x1, n) for y in np.linspace(y0, y1, n)]
        need = np.zeros(len(probes), int)
        for _s, _a, it in picked:
            gp = prep(shape(it.geometry))
            need += np.array([gp.contains(p) for p in probes])
        full = float((need >= MIN_SCENES).mean()) if picked else 0.0
        rec |= {"сцен_найдено": len(items), "отобрано": len(picked), "покрытие_3+": full}
        if len(picked) < 12 or full < 1.0:
            rec["отказ"] = f"зимних сцен мало: {len(picked)}, покрытие {100 * full:.0f} %"
            tried.append(rec); log(f"  {lat:.1f}, {lon:.0f}: {rec['отказ']}"); continue
        rec["выбран"] = True
        tried.append(rec)
        log(f"ВЫБРАН: центр {lat:.1f}° с. ш., {lon:.0f}° в. д., EPSG:{epsg}; сцен {len(items)}, отобрано "
            f"{len(picked)}, покрытие 3+ {100 * full:.0f} %")
        break
    OUT.write_text(json.dumps({"seed": SEED, "проверено": tried}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"сохранено: {OUT} (проверено кандидатов {len(tried)})")


if __name__ == "__main__":
    main()
