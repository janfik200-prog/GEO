"""ANFU, поправка 1 п. 4–5: ранжирование листов по R (правило FAM3), резерв и листы шага 5.

Пишет audit/ANFU_RANKING.txt (весь список кандидатов, условия 1–4, по убыванию R)
и audit/anfu_sheets.json (резерв — 2 листа; шаг 5 — остаток). Эталон не читается.
Запуск: python -m experiments.openlin_anfu_ranking
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.openlin_fam_sites import (EPSG_FI, FIN_MIN, GAP_M, LAT_MIN, R_RES_M, STEP_M,
                                           finland, north_mosaic, relief_R, site_box)
from experiments.openlin_probe import TILE_KM

N_RESERVE = 2
N_EXTRA_MAX = 5


def main():
    import geopandas as gpd
    from shapely.geometry import box
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
    rows.sort(key=lambda r: -r["R"])
    used_names = ["fam_r1", "fam_r2", "fam_r3", "utsjoki", "lapland", "finland65"]
    used = [site_box(s) for s in used_names]
    lines = [f"# ANFU: кандидаты по правилу FAM3 (условия 1–4), по убыванию R; {len(rows)} шт."]
    for i, r in enumerate(rows):
        lines.append(f"{i:3d} lat {r['lat']:.3f} lon {r['lon']:.3f} fin {r['fin']:.3f} R {r['R']:.1f}")
    txt = "\n".join(lines) + "\n"
    Path("audit/ANFU_RANKING.txt").write_text(txt, encoding="utf-8")
    sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    reserve, extra = [], []
    for r in rows:
        if any(r["box"].intersects(u) for u in used):
            continue
        if any(r["box"].intersects(p["box"]) for p in reserve + extra):
            continue
        (reserve if len(reserve) < N_RESERVE else extra).append(r)
    n_all = len(extra)
    extra = extra[:N_EXTRA_MAX]
    out = {"ranking_sha256": sha, "n_candidates": len(rows),
           "reserve": [{"name": f"anfu_test{i + 1}", **{k: v for k, v in r.items() if k != "box"}}
                       for i, r in enumerate(reserve)],
           "extra_all_count": n_all,
           "extra": [{"name": f"fam_r{i + 4}", **{k: v for k, v in r.items() if k != "box"}}
                     for i, r in enumerate(extra)]}
    Path("audit/anfu_sheets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))



# ---- поправка 2: резерв с зазором 20 км ко всем использованным, полоса до 64,0° ----
LAT_RESERVE = 64.0
USED = ("utsjoki", "lapland", "finland65", "fam_n1", "fam_n2", "fam_n3", "fam_r1", "fam_r2", "fam_r3")
MOSAIC_S = Path("datacache/anabar_dem/south_fi_glo30_200m.npy")


def band_mosaic(fin):
    """GLO-30 200 м для полосы LAT_RESERVE…LAT_MIN+0,5 (как north_mosaic, свой кэш)."""
    import pyproj
    from rasterio.enums import Resampling
    from experiments.openlin_probe import Grid, _read_on_grid
    from src.dem_features import _tile_url
    tr = pyproj.Transformer.from_crs(4326, EPSG_FI, always_xy=True)
    x0, _, x1, _ = fin.bounds
    y0 = tr.transform(25.0, LAT_RESERVE - 0.6)[1]
    y1 = tr.transform(25.0, LAT_MIN + 0.6)[1]
    g = Grid(x0=x0, y_top=y1, res_m=R_RES_M, height=int((y1 - y0) // R_RES_M) + 1,
             width=int((x1 - x0) // R_RES_M) + 1, epsg=EPSG_FI)
    if MOSAIC_S.exists():
        return np.load(MOSAIC_S), g
    tdir = MOSAIC_S.parent / "south_fi_tiles"
    tdir.mkdir(parents=True, exist_ok=True)
    lon0, lat0, lon1, lat1 = g.bbox_wgs84(0)
    dem = np.zeros((g.height, g.width), "float32")
    failed = []
    for lat in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1):
        for lon in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1):
            tp = tdir / f"{lat}_{lon}.npy"
            if not tp.exists():
                for k in range(5):
                    try:
                        np.save(tp, _read_on_grid(_tile_url(lat, lon), g, Resampling.bilinear))
                        break
                    except Exception as exc:
                        msg = repr(exc)
                        if "No such file" in msg or "not recognized" in msg or "404" in msg:
                            np.save(tp, np.zeros((g.height, g.width), "float32"))
                            break
                        print(f"  плитка {lat},{lon}: обрыв {k + 1}", flush=True)
                else:
                    failed.append((lat, lon))
                    continue
            dem = np.where((part := np.load(tp)) != 0, part, dem)
            print(f"  плитка {lat},{lon}: {100 * (part != 0).mean():.0f} %", flush=True)
    if failed:
        raise SystemExit(f"не скачаны плитки {failed}")
    dem = np.where(dem != 0, dem, np.nan).astype("float32")
    np.save(MOSAIC_S, dem)
    return dem, g


def reserve2():
    import geopandas as gpd
    from shapely.geometry import box
    fin = finland()
    dem, g = band_mosaic(fin)
    used = [site_box(s).buffer(GAP_M) for s in USED]
    w, h = TILE_KM[0] * 1000, TILE_KM[1] * 1000
    x0, _, x1, _ = fin.bounds
    rows = []
    for cy in np.arange(g.y_top - g.height * R_RES_M + h / 2, g.y_top - h / 2 + 1, STEP_M):
        for cx in np.arange(x0 + w / 2, x1 - w / 2 + 1, STEP_M):
            t = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            c = gpd.GeoSeries([t.centroid], crs=EPSG_FI).to_crs(4326).iloc[0]
            if c.y < LAT_RESERVE:
                continue
            if t.intersection(fin).area / t.area < FIN_MIN or any(t.intersects(u) for u in used):
                continue
            r0, c0 = int((g.y_top - (cy + h / 2)) / R_RES_M), int((cx - w / 2 - g.x0) / R_RES_M)
            sub = dem[max(r0, 0):r0 + int(h / R_RES_M), max(c0, 0):c0 + int(w / R_RES_M)]
            rows.append(dict(lat=round(c.y, 3), lon=round(c.x, 3), R=round(relief_R(sub), 1), box=t))
    rows.sort(key=lambda r: -r["R"])
    pick = []
    for r in rows:
        if all(not r["box"].buffer(GAP_M).intersects(p["box"]) for p in pick):
            pick.append(r)
        if len(pick) == N_RESERVE:
            break
    out = json.loads(Path("audit/anfu_sheets.json").read_text(encoding="utf-8"))
    out["reserve_v1_rejected"] = out.pop("reserve")
    out["reserve"] = [{"name": f"anfu_test{i + 1}", **{k: v for k, v in r.items() if k != "box"}}
                      for i, r in enumerate(pick)]
    out["reserve_rule"] = "поправка 2: ≥20 км ко всем использованным, полоса ≥ 64,0°, по R"
    out["reserve_candidates"] = len(rows)
    Path("audit/anfu_sheets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps(out["reserve"], ensure_ascii=False, indent=1), "кандидатов", len(rows))


if __name__ == "__main__":
    import sys
    reserve2() if "--reserve2" in sys.argv else main()
