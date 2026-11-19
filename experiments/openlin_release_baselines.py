"""WP19 (ТЗ v3.0): кривые «длина карты -> P, R, F1» для нашей карты, LSD и Хафа.

Ничего не настраивается: входы, детекторы, эталоны и область оценки — те же,
что в ``openlin_baselines`` (Пакет 04) и ``openlin_d09_audit``. Добавлено
только то, чего там не было:

* F1 (а не только P и R) на **общих** точках бюджета для всех методов —
  геометрическая сетка от 20 км до полной длины самой длинной карты;
* та же кривая против GTK 292 (Лапландия, 300 м);
* маски трёх методов при бюджете нашей карты — для рисунка «одно окно,
  одна длина» (``openlin_release_figures``).

Линии каждой карты набираются от длинных к коротким, как в Пакете 04.

Запуск из корня:
    python -m experiments.openlin_release_baselines --site anabar
    python -m experiments.openlin_release_baselines --site lapland
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import GTK_TOL_M, REF, TOL_M, grid, prf, rast
from experiments.openlin_baselines import COMP, M, RES_M, eval_area, mask_of, run_method

OUT = Path("outputs/k1/results")
FIG = Path("outputs/k1/figures")
N_LEVELS = 30          # точек на кривой (только рисунок и таблица кривой)
KM_MIN = 20.0


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
    d = np.load(M / f"{stem}_линии.npz")
    pts, ids, lens = d["C_all_pts"], d["C_all_id"], d["C_all_len_km"]
    methods = {"наша карта": ([pts[ids == i] for i in range(len(lens))], lens * 1000)}
    if site == "anabar":
        d1 = np.load(M / "openlin_dem_lines_anabar_одна_линии.npz")
        p1, i1, l1 = d1["add_pts"], d1["add_id"], d1["add_len_km"]
        methods["наша карта + DEM1"] = (methods["наша карта"][0] + [p1[i1 == i] for i in range(len(l1))],
                                        np.r_[lens, l1] * 1000)
    methods["LSD"] = run_method(site, "lsd", w, halves, valid, log)[0]["raw"]
    methods["Canny + Хаф"] = run_method(site, "hough", w, halves, valid, log)[0]["raw"]

    refs = {"v2": (rast(gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg").geometry, g) & ok, TOL_M)}
    if site == "lapland":
        gtk = gpd.read_file("datacache/gtk_lineaments/lin500k_292.gpkg").to_crs(epsg=g["epsg"])
        refs["GTK292"] = (rast(gtk.geometry, g) & ok, GTK_TOL_M)
    dref = {k: edt(~m, sampling=RES_M) for k, (m, _t) in refs.items()}

    # длины полных карт в области оценки -> общая сетка бюджетов
    full_km = {n: float((mask_of(c, shape) & ok).sum() * RES_M / 1000) for n, (c, _L) in methods.items()}
    ours_km = full_km["наша карта"]
    levels = np.unique(np.r_[np.geomspace(KM_MIN, max(full_km.values()), N_LEVELS), ours_km])
    log("полные длины, км ячеек: " + ", ".join(f"{n} {v:.0f}" for n, v in full_km.items()))

    rows, at_budget = [], {}
    for name, (cells, L) in methods.items():
        order = np.argsort(-np.asarray(L))
        cum, tot, li = np.zeros(shape, bool), 0, 0
        todo = [lv for lv in levels if lv <= full_km[name] + 1e-6] + [None]   # None = полная длина
        for k in order:
            c = cells[k]
            new = c[~cum[c[:, 0], c[:, 1]]]
            cum[new[:, 0], new[:, 1]] = True
            tot += int(ok[new[:, 0], new[:, 1]].sum())
            while todo[li] is not None and tot * RES_M / 1000 >= todo[li]:
                mm = cum & ok
                dm = edt(~mm, sampling=RES_M)
                for rn, (rm, tol) in refs.items():
                    p, r, f = prf(mm, rm, dm, dref[rn], tol)
                    rows.append({"site": site, "method": name, "reference": rn, "tolerance_m": tol,
                                 "budget_target_km": round(float(todo[li]), 1),
                                 "length_km": round(float(mm.sum() * RES_M / 1000), 1),
                                 "P": round(p, 4), "R": round(r, 4), "F1": round(f, 4), "mode": "кривая"})
                if abs(todo[li] - ours_km) < 1e-6:
                    at_budget[name] = mm.copy()
                li += 1
        mm = cum & ok
        dm = edt(~mm, sampling=RES_M)
        for rn, (rm, tol) in refs.items():
            p, r, f = prf(mm, rm, dm, dref[rn], tol)
            rows.append({"site": site, "method": name, "reference": rn, "tolerance_m": tol,
                         "budget_target_km": None, "length_km": round(full_km[name], 1),
                         "P": round(p, 4), "R": round(r, 4), "F1": round(f, 4), "mode": "полная длина"})
        at_budget.setdefault(name, mm)
        log(f"  {name}: точек {li}, полная длина {full_km[name]:.0f} км")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"length_budget_curves_{site}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(OUT / f"masks_at_our_budget_{site}.npz", ours_km=ours_km,
                        **{k.replace(" ", "_").replace("+", "plus"): np.packbits(v) for k, v in at_budget.items()},
                        shape=np.array(shape))
    plot(site, df, ours_km, refs)
    log(f"сохранено: {OUT / f'length_budget_curves_{site}.csv'}")


def plot(site, df, ours_km, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"наша карта": dict(color="#c0392b", lw=2.4), "наша карта + DEM1": dict(color="#c0392b", lw=1.6, ls="--"),
             "LSD": dict(color="#1f5fd6", lw=1.7), "Canny + Хаф": dict(color="#2a9d3a", lw=1.7)}
    fig, axs = plt.subplots(len(refs), 3, figsize=(15, 4.6 * len(refs)), facecolor="white", squeeze=False)
    for ri, rn in enumerate(refs):
        for ci, met in enumerate(("F1", "P", "R")):
            ax = axs[ri, ci]
            for name, st in style.items():
                s = df[(df.method == name) & (df.reference == rn)]
                if s.empty:
                    continue
                cur = s[s["mode"] == "кривая"]
                ax.plot(cur.length_km, cur[met], label=name, marker=".", ms=4, **st)
                fl = s[s["mode"] == "полная длина"]
                ax.plot(fl.length_km, fl[met], marker="o", ms=7, mfc="white", color=st["color"], ls="none")
            ax.axvline(ours_km, color="gray", lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            tol = refs[rn][1]
            ax.set_title(f"{met} против {'эталона v2 (ArcticDEM + LSD)' if rn == 'v2' else 'GTK 292 (1:500 000)'},"
                         f" τ = {tol:.0f} м", loc="left", fontsize=10)
            ax.set_xlabel("длина карты в области оценки, км ячеек 20 м (лог.)")
        axs[ri, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{site}: качество при одинаковой длине карты, micro по всей области. Линии набираются от длинных"
                 f" к коротким; кружок — полная длина метода; пунктир — длина нашей карты ({ours_km:.0f} км)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"fig_budget_vs_f1_{site}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
