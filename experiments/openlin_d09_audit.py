"""D09 (дельта-ТЗ v2 от 27.10.2026): таблица F1 по участкам для нашей карты, LSD и Хафа.

Вопрос ТЗ: как получено +0.288 при F1 = 0.251. Ничего нового не настраивается:
те же входы, детекторы, эталоны, участки 20 x 20 км и зерно бутстрэпа, что в
``openlin_baselines`` (Пакет 04). Скрипт только выписывает по каждому участку
P, R, F1 и длины для трёх бюджетов:

- «полная длина» — каждая карта целиком (объявленное сравнение Пакета 04);
- «бюджет наш» — карта метода обрезана до длины нашей (от длинных линий);
- «бюджет LSD» — наша карта обрезана до длины LSD (добавлено после Пакета 04).

и считает по каждому сравнению micro-F1 (вся область), macro-F1 (среднее по
участкам) и парную среднюю разность с 95 % ДИ.

Запуск из корня:
    python -m experiments.openlin_d09_audit --site anabar
    python -m experiments.openlin_d09_audit --site lapland
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import BLOCK_CELLS, GTK_TOL_M, REF, TOL_M, grid, prf, rast
from experiments.openlin_baselines import (COMP, M, N_BOOT, RES_M, SEED, eval_area, mask_of,
                                           run_method)

OUT = Path("outputs/k1/results")


def budget_mask(cells, L, ok, km, shape):
    """Маска линий от длинных к коротким, пока длина в области не достигнет km."""
    cum, tot = np.zeros(shape, bool), 0
    for k in np.argsort(-np.asarray(L)):
        c = cells[k]
        new = c[~cum[c[:, 0], c[:, 1]]]
        cum[new[:, 0], new[:, 1]] = True
        tot += int(ok[new[:, 0], new[:, 1]].sum())
        if tot * RES_M / 1000 >= km:
            return cum & ok
    return cum & ok


def main() -> None:
    import geopandas as gpd
    import pandas as pd
    from scipy.ndimage import distance_transform_edt as edt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    site = ap.parse_args().site
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    g = grid(site)
    H, W = g["H"], g["W"]
    shape = (H, W)
    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    halves = [np.load(COMP / f"winter_{site}_20m_лист_{n}.npz")["shade"] for n in ("А", "Б")]
    valid = np.isfinite(w["shade"]) & np.isfinite(halves[0]) & np.isfinite(halves[1])
    ok = eval_area(site, g)

    stem = ("openlin_zone_lines_anabar_v3_бл" if site == "anabar"
            else "openlin_zone_lines_lapland_v3_перенос")
    run_id = {"anabar": "TWIN (перегон 26.10, D08)", "lapland": "XFER (восстановлен 26.10)"}[site]
    d = np.load(M / f"{stem}_линии.npz")
    pts, ids, lens = d["C_all_pts"], d["C_all_id"], d["C_all_len_km"]
    ours = ([pts[ids == i] for i in range(len(lens))], lens * 1000)
    lsd = run_method(site, "lsd", w, halves, valid, log)[0]["raw"]
    hough = run_method(site, "hough", w, halves, valid, log)[0]["raw"]

    refs = {"v2": (rast(gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg").geometry, g) & ok, TOL_M)}
    if site == "lapland":
        gtk = gpd.read_file("datacache/gtk_lineaments/lin500k_292.gpkg").to_crs(epsg=g["epsg"])
        refs["GTK292"] = (rast(gtk.geometry, g) & ok, GTK_TOL_M)
    dref = {k: edt(~m, sampling=RES_M) for k, (m, _t) in refs.items()}

    km = lambda m: float(m.sum() * RES_M / 1000)
    ours_full = mask_of(ours[0], shape) & ok
    lsd_full = mask_of(lsd[0], shape) & ok
    maps = {
        ("полная длина", "наша"): ours_full,
        ("полная длина", "LSD"): lsd_full,
        ("полная длина", "Хаф"): mask_of(hough[0], shape) & ok,
        ("бюджет наш", "наша"): ours_full,
        ("бюджет наш", "Хаф"): budget_mask(*hough, ok, km(ours_full), shape),
        ("бюджет LSD", "наша"): budget_mask(*ours, ok, km(lsd_full), shape),
        ("бюджет LSD", "LSD"): lsd_full,
    }
    blocks = [(slice(i * BLOCK_CELLS, (i + 1) * BLOCK_CELLS), slice(j * BLOCK_CELLS, (j + 1) * BLOCK_CELLS))
              for i in range(H // BLOCK_CELLS) for j in range(W // BLOCK_CELLS)]
    tiles = [(i, j) for i in range(H // BLOCK_CELLS) for j in range(W // BLOCK_CELLS)]
    keep = [k for k, b in enumerate(blocks) if ok[b].mean() >= 0.95]
    log(f"участков 20 x 20 км с областью оценки ≥ 95 %: {len(keep)}")

    rows, glob = [], []
    per = {}
    for (budget, meth), mm in maps.items():
        dm = edt(~mm, sampling=RES_M)
        for rn, (rm, tol) in refs.items():
            p, r, f = prf(mm, rm, dm, dref[rn], tol)
            glob.append({"site": site, "budget": budget, "method": meth, "reference": rn,
                         "tolerance_m": tol, "auto_length_km": round(km(mm), 1),
                         "reference_length_km": round(km(rm), 1), "P": round(p, 4), "R": round(r, 4),
                         "micro_F1": round(f, 4), "run_id": run_id})
            fs = []
            for k in keep:
                b = blocks[k]
                p_, r_, f_ = prf(mm[b], rm[b], dm[b], dref[rn][b], tol)
                fs.append(f_)
                rows.append({"site": site, "tile_id": f"r{tiles[k][0]}c{tiles[k][1]}", "budget": budget,
                             "method": meth, "reference": rn, "tolerance_m": tol,
                             "valid_AOI_km2": round(ok[b].sum() * RES_M ** 2 / 1e6, 1),
                             "auto_length_km": round(km(mm[b]), 1),
                             "reference_length_km": round(km(rm[b]), 1),
                             "P": round(p_, 4), "R": round(r_, 4), "F1": round(f_, 4), "run_id": run_id})
            per[(budget, meth, rn)] = np.array(fs)
            glob[-1]["macro_F1"] = round(float(np.mean(fs)), 4)

    # парные разности тем же способом, что в Пакете 04 (зерно 2510, 10 000 повторов)
    pairs = []
    for budget, other in (("полная длина", "LSD"), ("полная длина", "Хаф"),
                          ("бюджет наш", "Хаф"), ("бюджет LSD", "LSD")):
        for rn in refs:
            dlt = per[(budget, "наша", rn)] - per[(budget, other, rn)]
            rng = np.random.default_rng(SEED)
            bs = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            gm = {(x["budget"], x["method"]): x for x in glob if x["reference"] == rn}
            pairs.append({"site": site, "budget": budget, "comparison": f"наша − {other}", "reference": rn,
                          "n_tiles": len(dlt), "paired_mean_delta": round(float(dlt.mean()), 4),
                          "CI95_lo": round(float(lo), 4), "CI95_hi": round(float(hi), 4),
                          "micro_F1_ours": gm[(budget, "наша")]["micro_F1"],
                          "micro_F1_other": gm[(budget, other)]["micro_F1"],
                          "micro_delta": round(gm[(budget, "наша")]["micro_F1"] - gm[(budget, other)]["micro_F1"], 4),
                          "macro_delta": round(float(dlt.mean()), 4)})
            log(f"  {budget}, {rn}, наша − {other}: micro {pairs[-1]['micro_F1_ours']:.3f} − "
                f"{pairs[-1]['micro_F1_other']:.3f} = {pairs[-1]['micro_delta']:+.3f}; "
                f"средняя по участкам {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"D09_F1_AUDIT_{site}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(glob).to_csv(OUT / f"baselines_global_{site}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(OUT / f"baselines_paired_ci_{site}.csv", index=False, encoding="utf-8-sig")
    log(f"сохранено: {OUT}")


if __name__ == "__main__":
    main()
