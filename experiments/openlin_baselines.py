"""Пакет 04 ТЗ К1: простые методы (LSD, Canny + вероятностный Хаф) против нашей карты.

Объявление — ``audit/PACKET04_DECLARATION.md`` (26.10.2026, до прогона): вход
тот же, что у нашего детектора (``shade`` зимнего композита, 20 м), параметры
по умолчанию, наименьшая длина 500 м; два режима — raw и с тем же
подтверждением половинами; сравнение с эталонами автоматической разметки
(v2 основной, v1) и на Лапландии с GTK 292; при полной и при равной длине
карты; парный бутстрэп разности F1 по участкам 20 x 20 км.

Запуск из корня:
    python -m experiments.openlin_baselines --site anabar
    python -m experiments.openlin_baselines --site lapland
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import (BLOCK_CELLS, GTK_TOL_M, REF, TOL_M,
                                                grid, prf, rast)

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
OUT = Path("outputs/k1/baselines")
FIG = Path("outputs/k1/figures")

# ---- объявление (audit/PACKET04_DECLARATION.md) ----------------------------
RES_M = 20.0
MIN_LEN_M = 500.0                 # наименьшая длина линии тракта и протокола
CONF_TOL_M, CONF_FRAC = 60.0, 0.30
HOUGH = dict(rho=1, theta=np.pi / 180, threshold=10,        # умолчания skimage
             minLineLength=int(MIN_LEN_M / RES_M), maxLineGap=10)
N_BOOT, SEED = 10_000, 2510
N_BUDGET = 25                     # точек на кривой бюджета (только рисунок)
# ----------------------------------------------------------------------------


def to8(a, valid):
    lo, hi = np.percentile(a[valid], [2, 98])
    x = np.clip((np.where(valid, a, np.median(a[valid])) - lo) / (hi - lo), 0, 1)
    return (x * 255).astype("uint8")


def detect(img8, method):
    """Отрезки (r0, c0, r1, c1) в ячейках сетки от 500 м."""
    import cv2
    if method == "lsd":
        s = cv2.createLineSegmentDetector().detect(img8)[0]
        s = np.zeros((0, 4)) if s is None else s.reshape(-1, 4)     # x1 y1 x2 y2
    else:
        from skimage.feature import canny
        e = canny(img8.astype("float64") / 255.0).astype("uint8") * 255
        s = cv2.HoughLinesP(e, **HOUGH)
        s = np.zeros((0, 4)) if s is None else s.reshape(-1, 4).astype("float64")
    seg = np.c_[s[:, 1], s[:, 0], s[:, 3], s[:, 2]]                  # r0 c0 r1 c1
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) * RES_M
    return seg[L >= MIN_LEN_M], L[L >= MIN_LEN_M]


def seg_cells(seg, shape):
    from skimage.draw import line
    out = []
    H, W = shape
    for r0, c0, r1, c1 in np.rint(seg).astype(int):
        rr, cc = line(r0, c0, r1, c1)
        k = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        out.append(np.c_[rr[k], cc[k]])
    return out


def mask_of(cells, shape, idx=None):
    m = np.zeros(shape, bool)
    for i in (range(len(cells)) if idx is None else idx):
        c = cells[i]
        m[c[:, 0], c[:, 1]] = True
    return m


def run_method(site, method, w, halves, valid, log):
    from scipy.ndimage import distance_transform_edt as edt
    t0 = time.time()
    shape = valid.shape
    seg, L = detect(to8(w["shade"], valid), method)
    cells = seg_cells(seg, shape)
    dts = []
    for h in halves:
        s_, _l = detect(to8(h, valid), method)
        dts.append(edt(~mask_of(seg_cells(s_, shape), shape), sampling=RES_M))
    conf = [i for i, c in enumerate(cells)
            if all((d[c[:, 0], c[:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC for d in dts)]
    log(f"  {method}: отрезков от 500 м {len(seg)} ({L.sum() / 1000:.0f} км), подтверждено "
        f"обеими половинами {len(conf)} ({L[conf].sum() / 1000:.0f} км) ({time.time() - t0:.0f} с)")
    return {"raw": (cells, L), "conf": ([cells[i] for i in conf], L[conf])}, seg, conf


def save_gpkg(site, name, seg, L, g):
    import geopandas as gpd
    from shapely.geometry import LineString
    x0, yt = g["x0"], g["ytop"]
    geoms = [LineString([(x0 + (c0 + 0.5) * RES_M, yt - (r0 + 0.5) * RES_M),
                         (x0 + (c1 + 0.5) * RES_M, yt - (r1 + 0.5) * RES_M)])
             for r0, c0, r1, c1 in seg]
    OUT.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"length_m": L}, geometry=geoms, crs=f"EPSG:{g['epsg']}").to_file(
        OUT / f"{site}_baselines.gpkg", layer=name, driver="GPKG")


def eval_area(site, g):
    from scipy.ndimage import distance_transform_edt as edt
    H, W = g["H"], g["W"]
    dem = np.load(f"datacache/arcticdem/{site}_10m.npy")
    ok = g["valid"] & np.isfinite(dem).reshape(H, 2, W, 2).all(axis=(1, 3))
    return ok & (edt(ok, sampling=RES_M) > TOL_M)


def main() -> None:
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt as edt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    site = a.site
    g = grid(site)
    H, W = g["H"], g["W"]
    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    halves = [np.load(COMP / f"winter_{site}_20m_лист_{n}.npz")["shade"] for n in ("А", "Б")]
    valid = np.isfinite(w["shade"]) & np.isfinite(halves[0]) & np.isfinite(halves[1])
    ok = eval_area(site, g)
    log(f"площадка {site}: область оценки {100 * ok.mean():.2f} %")

    # наша карта поштучно (для кривой бюджета) и маской
    stem = ("openlin_zone_lines_anabar_v3_бл" if site == "anabar"
            else "openlin_zone_lines_lapland_v3_перенос")
    d = np.load(M / f"{stem}_линии.npz")
    pts, ids, lens = d["C_all_pts"], d["C_all_id"], d["C_all_len_km"]
    ours_cells = [pts[ids == i] for i in range(len(lens))]
    methods = {"наша карта (снимок)": (ours_cells, lens * 1000)}
    if site == "anabar":
        d1 = np.load(M / "openlin_dem_lines_anabar_одна_линии.npz")
        p1, i1, l1 = d1["add_pts"], d1["add_id"], d1["add_len_km"]
        methods["наша карта + DEM1 (видит рельеф)"] = (
            ours_cells + [p1[i1 == i] for i in range(len(l1))], np.r_[lens, l1] * 1000)
    for mth in ("lsd", "hough"):
        res, seg, conf = run_method(site, mth, w, halves, valid, log)
        save_gpkg(site, f"{mth}_raw", seg, res["raw"][1], g)
        save_gpkg(site, f"{mth}_conf", seg[conf], res["conf"][1], g)
        label = {"lsd": "LSD", "hough": "Canny + Хаф"}[mth]
        methods[f"{label}, raw"] = res["raw"]
        methods[f"{label}, с подтверждением"] = res["conf"]

    # эталоны
    refs = {}
    for v, fn in ((2, "mlsd"), (1, "lsd")):
        r = gpd.read_file(REF / f"{site}_arcticdem_{fn}.gpkg")
        refs[f"эталон v{v}"] = (rast(r.geometry, g) & ok, TOL_M)
    if site == "lapland":
        gtk = gpd.read_file("datacache/gtk_lineaments/lin500k_292.gpkg").to_crs(epsg=g["epsg"])
        refs["GTK 292"] = (rast(gtk.geometry, g) & ok, GTK_TOL_M)
    dref = {k: edt(~m, sampling=RES_M) for k, (m, _t) in refs.items()}

    ours_km = float(mask_of(ours_cells, (H, W)).__and__(ok).sum() * RES_M / 1000)
    log(f"бюджет длины = наша карта по снимку: {ours_km:.0f} км ячеек")
    blocks = [(slice(i * BLOCK_CELLS, (i + 1) * BLOCK_CELLS), slice(j * BLOCK_CELLS, (j + 1) * BLOCK_CELLS))
              for i in range(H // BLOCK_CELLS) for j in range(W // BLOCK_CELLS)]
    blocks = [b for b in blocks if ok[b].mean() >= 0.95]
    rows, curves, perblock = [], {}, {}
    for name, (cells, L) in methods.items():
        order = np.argsort(-np.asarray(L))
        full = mask_of(cells, (H, W)) & ok
        # маска при бюджете = длине нашей карты
        cum, bud, tot = np.zeros((H, W), bool), None, 0
        for k in order:
            c = cells[k]
            new = c[~cum[c[:, 0], c[:, 1]]]
            cum[new[:, 0], new[:, 1]] = True
            tot += int(ok[new[:, 0], new[:, 1]].sum())
            if tot * RES_M / 1000 >= ours_km:
                bud = cum & ok
                break
        short = bud is None
        bud = full if short else bud
        for mode, mm in (("полная длина", full), ("бюджет", bud)):
            dm = edt(~mm, sampling=RES_M)
            row = {"метод": name, "режим": mode, "км": round(mm.sum() * RES_M / 1000, 1),
                   "короче_бюджета": short}
            for rn, (rm, tol) in refs.items():
                p, r, f = prf(mm, rm, dm, dref[rn], tol)
                row |= {f"{rn} P": round(p, 4), f"{rn} R": round(r, 4), f"{rn} F1": round(f, 4)}
            rows.append(row)
            if mode == "полная длина":
                rm, tol = refs["эталон v2"]
                perblock[name] = np.array([prf(mm[b], rm[b], dm[b], dref["эталон v2"][b], tol)[2]
                                           for b in blocks])
        # кривая бюджета по эталону v2
        rm, tol = refs["эталон v2"]
        # точки — равномерно по логарифму набранной длины, а не по числу линий:
        # иначе у метода с сотней тысяч отрезков первая точка уже за 10 000 км
        seen, km_cum = np.zeros((H, W), bool), np.zeros(len(order))
        tot = 0
        for n_, k in enumerate(order):
            c = cells[k]
            new = c[~seen[c[:, 0], c[:, 1]]]
            seen[new[:, 0], new[:, 1]] = True
            tot += int(ok[new[:, 0], new[:, 1]].sum())
            km_cum[n_] = tot * RES_M / 1000
        levels = np.geomspace(max(km_cum[0], 1.0), km_cum[-1], N_BUDGET)
        steps = np.unique(np.minimum(np.searchsorted(km_cum, levels) + 1, len(order)))
        cur, xs, ps, rs = np.zeros((H, W), bool), [], [], []
        done = 0
        for st in steps:
            for k in order[done:st]:
                c = cells[k]
                cur[c[:, 0], c[:, 1]] = True
            done = st
            mm = cur & ok
            p, r, _f = prf(mm, rm, edt(~mm, sampling=RES_M), dref["эталон v2"], tol)
            xs.append(mm.sum() * RES_M / 1000); ps.append(p); rs.append(r)
        curves[name] = (xs, ps, rs)
        log(f"  {name}: {rows[-2]['км']:.0f} км; F1 v2 полная {rows[-2]['эталон v2 F1']:.3f}, "
            f"при бюджете {rows[-1]['эталон v2 F1']:.3f}" + (" (короче бюджета)" if short else ""))

    # парный бутстрэп по участкам: наша карта минус каждый простой метод (эталон v2)
    rng = np.random.default_rng(SEED)
    base = perblock["наша карта (снимок)"]
    boot = {}
    for name, fb in perblock.items():
        if name.startswith("наша"):
            continue
        dlt = base - fb
        bs = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "лучше" if lo > 0 else ("хуже" if hi < 0 else "данных недостаточно")
        boot[name] = {"разность_F1_ср": float(dlt.mean()), "ДИ95": [float(lo), float(hi)],
                      "участков": len(dlt), "вывод": verdict,
                      "по_участкам_наша": base.round(3).tolist(), "по_участкам_метод": fb.round(3).tolist()}
        log(f"  бутстрэп, наша минус «{name}»: разность F1 {dlt.mean():+.3f}, 95 % ДИ "
            f"[{lo:+.3f}, {hi:+.3f}] по {len(dlt)} участкам -> {verdict}")

    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{site}_baseline_comparison.csv", index=False,
                              encoding="utf-8-sig")
    (OUT / f"{site}_bootstrap.json").write_text(json.dumps(boot, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    (OUT / f"{site}_configs.json").write_text(json.dumps(
        {"вход": "shade, 8 бит, растяжка 2-98 % по листу", "LSD": "cv2.createLineSegmentDetector() по умолчанию",
         "Canny": "skimage.feature.canny по умолчанию (sigma 1, пороги 0.1/0.2 шкалы 0-1)",
         "HoughLinesP": {k: (float(v) if k == "theta" else v) for k, v in HOUGH.items()},
         "min_len_m": MIN_LEN_M, "подтверждение": {"tol_m": CONF_TOL_M, "frac": CONF_FRAC},
         "opencv": __import__("cv2").__version__, "skimage": __import__("skimage").__version__},
        ensure_ascii=False, indent=1), encoding="utf-8")
    plot(site, curves, ours_km, log)
    log(f"сохранено: {OUT}")


def plot(site, curves, ours_km, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), facecolor="white")
    style = {"наша карта (снимок)": dict(lw=2.6, color="#d0101a"),
             "наша карта + DEM1 (видит рельеф)": dict(lw=2.0, color="#d0101a", ls="--"),
             "LSD, raw": dict(lw=1.6, color="#1f5fd6"),
             "LSD, с подтверждением": dict(lw=1.6, color="#1f5fd6", ls=":"),
             "Canny + Хаф, raw": dict(lw=1.6, color="#2a9d3a"),
             "Canny + Хаф, с подтверждением": dict(lw=1.6, color="#2a9d3a", ls=":")}
    for name, (xs, ps, rs) in curves.items():
        st = style.get(name, dict(lw=1.3))
        ax[0].plot(xs, ps, label=name, marker=".", ms=4, **st)
        ax[1].plot(xs, rs, label=name, marker=".", ms=4, **st)
    for a_, t in zip(ax, ("точность P (доля наших у эталона v2)", "полнота R (доля эталона v2 у наших)")):
        a_.axvline(ours_km, color="gray", lw=0.8, ls=":")
        a_.set_xscale("log")
        a_.set_xlabel("длина карты, км ячеек 20 м, логарифм (линии от длинных к коротким)")
        a_.set_title(t, loc="left", fontsize=11)
        a_.grid(alpha=0.3)
    ax[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{site}: качество при равной длине карты; пунктир — длина нашей карты по снимку",
                 x=0.01, ha="left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"length_budget_curves_{site}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
