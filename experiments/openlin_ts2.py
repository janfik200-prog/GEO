"""TS2: доля опубликованных разломов (GEM, GIF), найденная наборами TS1 — audit/TS2_DECLARATION.md.

Описательно, ``TS2_N_TESTS`` = 0. Наборы — маски TS1 (``ts1_masks.npz``), не пересчитываются.
Запуск: python -m experiments.openlin_ts2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REF = Path("datacache/ref_lines")
TS2_N_TESTS = 0
D_M = (100.0, 300.0, 600.0, 1000.0)     # 100 — ячейка Анфисы, 300 — допуск S (в вывод); 600/1000 — для сведения (поправка 1 п. 2)
SEEDS = tuple(range(1, 20))             # 19 сдвигов: ранг из 20
MIN_SHIFT_M = 1000.0                    # больше наибольшего d
# участок -> (база, отбор строк); Крафла (Bird 2003) исключена объявлением
SITES = {"ts_scarp_lostriver": "gem", "ts_strikeslip_carrizo": "gem", "ts_strikeslip_arava": "gem",
         "ts_thermal_dixie": "gem", "ts_shore_bonneville": "gem", "ts_folds_zagros": "gem",
         "ts_scarp_stuoragurra": "gif", "ts_scarp_parvie": "gif"}
SETS = {"снимок": "снимок", "рельеф": "рельеф", "Анфиса LINB2": "анфиса2", "Анфиса LINB1 все": "анфиса1"}


def ref_mask(name, site, kind):
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box

    from experiments.openlin_typesites_fetch import make_grid
    gr = make_grid(site, 20.0)
    b = box(*gr.bbox_wgs84(pad_m=0))
    g = gpd.read_file(REF / "gem_af.geojson") if kind == "gem" else \
        gpd.read_file(REF / "gif" / "GIF.shp").to_crs(4326)
    sub = g[g.intersects(b)]
    if kind == "gem":
        sub = sub[sub["catalog_name"].astype(str) != "Bird 2003"]
    geoms = [x for x in sub.to_crs(gr.epsg).geometry if x is not None]
    m = rasterize([(x, 1) for x in geoms], out_shape=gr.shape, transform=gr.transform,
                  fill=0, all_touched=True, dtype="uint8").astype(bool)
    return m, len(sub), "; ".join(sorted(set(sub.get("catalog_name", sub.get("Reference")).astype(str))))[:120]


def recall(ref, dist, d):
    return float((dist[ref] <= d).mean()) if ref.any() else np.nan


def main():
    import pandas as pd

    from experiments.openlin_fam import log_to
    from experiments.openlin_typesites_fetch import DEST, OUT, load_sites
    log = log_to(OUT / "_ts2.log")
    log("TS2; объявление audit/TS2_DECLARATION.md")
    S = load_sites()
    rows = []
    for name, kind in SITES.items():
        site = S[name]
        mk = np.load(DEST / name / "ts1_masks.npz")
        ref, nseg, cat = ref_mask(name, site, kind)
        H, W = ref.shape
        km = ref.sum() * 0.02
        if not ref.any():                                # аудит: Загрос — эталон вне окна
            log(f"=== {name}: эталон в окне пуст — исключён")
            continue
        log(f"=== {name}: эталон {kind} {cat}, отрезков {nseg}, {km:.1f} км (ячейками 20 м)")
        # поправка 1 п. 5: сдвиг по нормали к главной оси эталона (вдоль простирания прямой
        # разлом лёг бы на себя), от 1000 м до половины окна; нуль — только в годных ячейках
        z = np.load(DEST / name / "s2_img.npz")
        valid = np.isfinite(z["shade"]) & np.isfinite(np.load(DEST / name / "glo30.npz")["dem"])
        rr_, cc_ = np.nonzero(ref)
        q = np.stack([rr_, cc_], 1).astype(float)
        q -= q.mean(0)
        _w, v = np.linalg.eigh(q.T @ q)
        nrm = v[:, 0]                                    # нормаль = наименьшая ось
        dref = distance_transform_edt(~ref, sampling=(20.0, 20.0))
        rng_shifts, selfov, dropped = [], [], []
        for s in SEEDS:
            r = np.random.default_rng(s)
            t = r.uniform(MIN_SHIFT_M, 10.0 * min(H, W) - 1) * r.choice([-1, 1])   # 20 м·N/2
            sh = (int(round(t * nrm[0] / 20)), int(round(t * nrm[1] / 20)))
            rs = np.roll(ref, sh, (0, 1))
            selfov.append(float((dref[rs] <= 1000).mean()) if rs.any() else np.nan)
            dropped.append(float((rs & ~valid).sum() / max(rs.sum(), 1)))
            rng_shifts.append(sh)
        log(f"  сдвиги по нормали: самоналожение (1000 м) медиана {100 * np.nanmedian(selfov):.0f} %, "
            f"вне годных ячеек отброшено медиана {100 * np.median(dropped):.0f} %")
        sets = {s: mk[k] for s, k in SETS.items()}
        from experiments.openlin_union_recall import add_new
        sets["снимок + рельеф"] = mk["снимок"] | add_new(mk["рельеф"], mk["снимок"], 20.0)
        for s, m in sets.items():
            dist = distance_transform_edt(~m, sampling=(20.0, 20.0)) if m.any() else np.full(m.shape, np.inf)
            row = dict(участок=name, вид=site["type"], эталон=kind, эталон_км=round(km, 1), набор=s,
                       набор_км=round(m.sum() * 0.02, 1))
            parts = []
            for d in D_M:
                obs = recall(ref, dist, d)
                nul = np.array([recall(np.roll(ref, sh, (0, 1)) & valid, dist, d) for sh in rng_shifts])
                above = int((nul < obs).sum())        # аудит: ничьи не в пользу набора
                row.update({f"R{int(d)}": round(obs, 3), f"нуль{int(d)}_мед": round(float(np.median(nul)), 3),
                            f"нуль{int(d)}_90": round(float(np.percentile(nul, 90)), 3), f"выше_из19_{int(d)}": above})
                parts.append(f"{int(d)} м: {100 * obs:.0f} % (нуль {100 * np.median(nul):.0f}/"
                             f"{100 * np.percentile(nul, 90):.0f}, выше {above} из 19)")
            log(f"  {s:18s} {m.sum() * 0.02:6.0f} км | " + " | ".join(parts))
            rows.append(row)
        picture(name, site, ref, sets)
    pd.DataFrame(rows).to_csv(OUT / "ts2.csv", index=False, encoding="utf-8-sig")


def picture(name, site, ref, sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    from experiments.openlin_typesites_fetch import DEST, OUT, hillshade
    dem = np.load(DEST / name / "glo30.npz")["dem"]
    hs = np.mean([hillshade(dem, 20.0, az) for az in (0, 45, 90, 135)], 0)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.4))
    rr = ndi.binary_dilation(ref, iterations=2)
    for a_, (s, col) in zip(ax, (("снимок", "orange"), ("рельеф", "deepskyblue"), ("Анфиса LINB2", "gold"))):
        a_.imshow(hs, cmap="gray")
        a_.imshow(np.ma.masked_where(~rr, rr), cmap=matplotlib.colors.ListedColormap(["red"]), alpha=0.5,
                  interpolation="nearest")
        m = ndi.binary_dilation(sets[s], iterations=1)
        a_.imshow(np.ma.masked_where(~m, m), cmap=matplotlib.colors.ListedColormap([col]), alpha=0.9,
                  interpolation="nearest")
        a_.set_title(f"{s} (цвет) и эталон (красный)")
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{name}: {site['type']}")
    fig.savefig(OUT / f"ts2_{name}.png", dpi=75, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
