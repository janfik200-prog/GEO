"""Пакет 07 ТЗ К1: абляции рабочего тракта линеаментов по шагам.

Объявление — ``audit/PACKET07_DECLARATION.md`` (26.10.2026, до расчёта).
Варианты — части рабочего тракта (TWIN: «цепочки», сдвиг 127 м) с теми же
числами; соседние отличаются одним шагом:

    V0 детектор -> V1 + продолжение концов -> V2 + сшивка -> V3 + подтверждение
    (рабочий набор C_all) -> V4 + зона «Кора» как фильтр (набор C);
    V5 = V3 + DEM1 (Анабар); отдельно V0+п и V1+п — подтверждение без сшивки.

Проверка восстановления: V2 и V3 обязаны совпасть с A_all и C_all рабочего
прогона бит в бит, иначе — остановка.

Запуск из корня:
    python -m experiments.openlin_ablation --site anabar
    python -m experiments.openlin_ablation --site lapland
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import GTK_TOL_M, REF, TOL_M, grid, prf, rast
from experiments.openlin_baselines import eval_area

M = Path("outputs/metrics")
COMP = Path("datacache/s2_lowsun/composite")
OUT = Path("outputs/k1/ablations")
FIG = Path("outputs/k1/figures")
N_BOOT, SEED = 10_000, 2510
BANDS_DEG = ((0, 15), (15, 30), (30, 60), (60, 90.01))
STEM = {"anabar": "openlin_zone_lines_anabar_v3_бл", "lapland": "openlin_zone_lines_lapland_v3_перенос"}


def build_variants(site, log):
    """Части тракта — те же вызовы и числа, что в openlin_zone_lines_v3.main."""
    from experiments import openlin_zone_lines_v3 as Z
    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize
    from scipy.ndimage import distance_transform_edt as edt

    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = Z.build_halves(site, log, "B04")
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])
    k = int(round(100 / res_m))
    zone, strike = Z.zone_and_strike(site, k)
    zone = zone[:valid.shape[0], :valid.shape[1]] & valid
    strike = strike[:valid.shape[0], :valid.shape[1]]
    up = lambda x: np.repeat(np.repeat(x, k, 0), k, 1)[:valid.shape[0], :valid.shape[1]]
    zone = up(Z.close_along_strike(zone[::k, ::k], strike[::k, ::k], 100.0, log)) & valid
    cnt = w["count"][:valid.shape[0], :valid.shape[1]]
    seam = np.zeros(valid.shape, bool)
    seam[:, :-1] |= cnt[:, :-1] != cnt[:, 1:]
    seam[:-1, :] |= cnt[:-1, :] != cnt[1:, :]
    seam |= ~valid
    xx = np.where(valid, standardize(opt, valid), 0.0).astype("float32")
    zz = local_norm(np.where(valid, ridge_response(xx), 0.0), valid, res_m)

    t0 = time.time()
    raw = Z.trace(opt, valid, res_m, local=True, min_len_m=0)
    log(f"  трассировка: обрывков {len(raw)} ({time.time() - t0:.0f} с)")
    keep = lambda L: [x for x in L if x["длина_км"] * 1000 >= Z.MIN_LEN_M]

    v0 = Z.drop_seam_lines(keep(copy.deepcopy(raw)), seam, res_m)
    ext = Z.extend_ends(copy.deepcopy(raw), zz, valid, res_m, None)
    v1 = Z.drop_seam_lines(keep(copy.deepcopy(ext)), seam, res_m)
    L = Z.link_same_side(copy.deepcopy(ext), zone, strike, res_m, None, z=zz,
                         support="mean", offset_m=Z.TWIN_OFFSET_M)
    v2 = Z.drop_seam_lines(keep(L), seam, res_m)
    W_ = valid.shape[1]
    for x in v2:                                    # «цепочки» — как в main
        kul = x.get("кулисы")
        if kul is not None and len(kul):
            x["pts"] = x["pts"][~np.isin(x["pts"][:, 0] * W_ + x["pts"][:, 1],
                                         kul[:, 0] * W_ + kul[:, 1])]
    dts = [edt(~Z.mask_of(Z.trace(h, valid, res_m, local=True), valid.shape), sampling=(res_m, res_m))
           for h in (halves["А"], halves["Б"])]
    conf = lambda L_: Z.select2(L_, zone, strike, dts, gate=False)[1]
    v3 = conf(v2)
    v4 = Z.select2(v2, zone, strike, dts, gate=True)[1]
    out = {"V0 детектор": v0, "V0+п": conf(v0), "V1 + продолжение": v1, "V1+п": conf(v1),
           "V2 + сшивка": v2, "V3 + подтверждение (рабочий)": v3, "V4 + зона как фильтр": v4}
    return out, valid, Z.mask_of


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
    V, valid, mask_of = build_variants(site, log)
    saved = np.load(M / f"{STEM[site]}.npz")
    chk = {"V2 + сшивка": "A_all", "V3 + подтверждение (рабочий)": "C_all", "V4 + зона как фильтр": "C"}
    for name, key in chk.items():
        same = bool(np.array_equal(mask_of(V[name], valid.shape), saved[key][:H, :W]))
        log(f"  проверка восстановления: {name} = {key} рабочего прогона -> "
            f"{'СОВПАДАЕТ' if same else 'НЕ совпадает'}")
        if not same:
            raise SystemExit("восстановление вариантов не сошлось — абляция остановлена")

    sets = {n: [x["pts"] for x in L] for n, L in V.items()}
    lens = {n: np.array([x["длина_км"] for x in L]) for n, L in V.items()}
    if site == "anabar":
        d1 = np.load(M / "openlin_dem_lines_anabar_одна_линии.npz")
        p1, i1, l1 = d1["add_pts"], d1["add_id"], d1["add_len_km"]
        dem = [p1[i1 == i] for i in range(len(l1))]
        sets["DEM1 добавленные"], lens["DEM1 добавленные"] = dem, l1
        sets["V5 + рельеф"] = sets["V3 + подтверждение (рабочий)"] + dem
        lens["V5 + рельеф"] = np.r_[lens["V3 + подтверждение (рабочий)"], l1]

    ok = eval_area(site, g)
    refs = {"v2": (rast(gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg").geometry, g) & ok, TOL_M),
            "v1": (rast(gpd.read_file(REF / f"{site}_arcticdem_lsd.gpkg").geometry, g) & ok, TOL_M)}
    if site == "lapland":
        gtk = gpd.read_file("datacache/gtk_lineaments/lin500k_292.gpkg").to_crs(epsg=g["epsg"])
        refs["GTK"] = (rast(gtk.geometry, g) & ok, GTK_TOL_M)
    dref = {k: edt(~m, sampling=20.0) for k, (m, _t) in refs.items()}

    def mask_budget(name, km=None):
        cells, L = sets[name], lens[name]
        m = np.zeros((H, W), bool)
        tot = 0
        for k in np.argsort(-L):
            c = cells[k]
            new = c[~m[c[:, 0], c[:, 1]]]
            m[new[:, 0], new[:, 1]] = True
            tot += int(ok[new[:, 0], new[:, 1]].sum())
            if km is not None and tot * 0.02 >= km:
                break
        return m & ok

    blocks = [(slice(i * 1000, (i + 1) * 1000), slice(j * 1000, (j + 1) * 1000))
              for i in range(H // 1000) for j in range(W // 1000)]
    blocks = [b for b in blocks if ok[b].mean() >= 0.95]

    def metrics(m):
        dm = edt(~m, sampling=20.0)
        r = {k: prf(m, rm, dm, dref[k], t) for k, (rm, t) in refs.items()}
        fb = np.array([prf(m[b], refs["v2"][0][b], dm[b], dref["v2"][b], TOL_M)[2] for b in blocks])
        # ДОБАВЛЕНО ПОСЛЕ ПРОГОНА (26.10.2026): то же по GTK — единственной
        # человеческой карте; в объявлении бутстрэп был только по эталону v2
        if "GTK" in refs:
            fg = np.array([prf(m[b], refs["GTK"][0][b], dm[b], dref["GTK"][b], GTK_TOL_M)[2] for b in blocks])
            return r, (fb, fg)
        return r, (fb, None)

    rows, full = [], {}
    log("полная длина каждого варианта:")
    for name in sets:
        m = mask_budget(name)
        r, _fb = metrics(m)
        full[name] = m
        km = m.sum() * 0.02
        rows.append({"вариант": name, "линий": len(sets[name]), "км": round(km, 1),
                     **{f"{k} {q}": round(v[i], 4) for k, v in r.items() for i, q in enumerate("PRF")}})
        log(f"  {name}: линий {len(sets[name])}, {km:.0f} км; F1 v2 {r['v2'][2]:.3f} (P {r['v2'][0]:.3f}, "
            f"R {r['v2'][1]:.3f})" + (f"; GTK F1 {r['GTK'][2]:.3f}" if "GTK" in r else ""))

    pairs = [("V0 детектор", "V1 + продолжение"), ("V1 + продолжение", "V2 + сшивка"),
             ("V2 + сшивка", "V3 + подтверждение (рабочий)"),
             ("V3 + подтверждение (рабочий)", "V4 + зона как фильтр"),
             ("V0 детектор", "V0+п"), ("V1 + продолжение", "V1+п"),
             ("V1+п", "V3 + подтверждение (рабочий)")]
    if site == "anabar":
        pairs.append(("V3 + подтверждение (рабочий)", "V5 + рельеф"))
    rng = np.random.default_rng(SEED)
    boots = []
    log("соседние варианты при равной длине (короткий из пары), бутстрэп по участкам:")
    for a_, b_ in pairs:
        km = min(full[a_].sum(), full[b_].sum()) * 0.02
        ma, mb = mask_budget(a_, km), mask_budget(b_, km)
        (ra, (fa, ga)), (rb, (fb, gb)) = metrics(ma), metrics(mb)
        dl = fb - fa
        gtk_txt = ""
        if ga is not None:
            dg = gb - ga
            bg = np.array([dg[rng.integers(0, len(dg), len(dg))].mean() for _ in range(N_BOOT)])
            glo, ghi = np.percentile(bg, [2.5, 97.5])
            gtk_txt = (f"; ПОСЛЕ ПРОГОНА, по GTK по участкам {dg.mean():+.3f} [{glo:+.3f}, {ghi:+.3f}] -> "
                       + ("помогает" if glo > 0 else ("вредит" if ghi < 0 else "не различимо")))
        bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "помогает" if lo > 0 else ("вредит" if hi < 0 else "не различимо")
        rec = {"от": a_, "к": b_, "км": round(km, 1),
               **{f"{k} F1 от": round(ra[k][2], 4) for k in ra}, **{f"{k} F1 к": round(rb[k][2], 4) for k in rb},
               "разность_F1_v2": float(dl.mean()), "ДИ95": [float(lo), float(hi)], "вывод": verdict}
        boots.append(rec)
        log(f"  {a_} -> {b_} при {km:.0f} км: F1 v2 {ra['v2'][2]:.3f} -> {rb['v2'][2]:.3f}; по участкам "
            f"{dl.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}"
            + (f"; GTK {ra['GTK'][2]:.3f} -> {rb['GTK'][2]:.3f}" if "GTK" in ra else "") + gtk_txt)
        rec["GTK_по_участкам_после_прогона"] = gtk_txt

    band = {}
    if site == "anabar":
        ref = gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg")
        xy = np.array([list(gm.coords) for gm in ref.geometry])            # отрезки: 2 точки
        az = np.degrees(np.arctan2(xy[:, 1, 0] - xy[:, 0, 0], xy[:, 1, 1] - xy[:, 0, 1])) % 180
        sun = 184.0 % 180
        ang = np.abs(((az - sun) + 90) % 180 - 90)
        log("полнота R по углу отрезков эталона v2 к солнцу (184°), полная длина:")
        for lo_, hi_ in BANDS_DEG:
            rm = rast(ref.geometry[(ang >= lo_) & (ang < hi_)], g) & ok
            row = {}
            for name in ("V3 + подтверждение (рабочий)", "DEM1 добавленные", "V5 + рельеф"):
                dm = edt(~full[name], sampling=20.0)
                row[name] = float((dm[rm] <= TOL_M).mean()) if rm.any() else np.nan
            row["ячеек эталона"] = int(rm.sum())
            band[f"{lo_}-{int(hi_)}"] = row
            log(f"  {lo_:2d}-{int(hi_):2d}°: ячеек {row['ячеек эталона']}; R снимок "
                f"{100 * row['V3 + подтверждение (рабочий)']:.1f} %, DEM1 {100 * row['DEM1 добавленные']:.1f} %, "
                f"снимок + DEM1 {100 * row['V5 + рельеф']:.1f} % (прибавка "
                f"{100 * (row['V5 + рельеф'] - row['V3 + подтверждение (рабочий)']):+.1f} п.п.)")

    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{site}_stage_counts.csv", index=False, encoding="utf-8-sig")
    (OUT / f"{site}_ablation.json").write_text(json.dumps({"варианты": rows, "пары": boots, "слепая_полоса": band},
                                                          ensure_ascii=False, indent=1), encoding="utf-8")
    plot(site, rows, log)
    log(f"сохранено: {OUT}")


def plot(site, rows, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor="white")
    for r in rows:
        ax.scatter(r["v2 P"], r["v2 R"], s=30 + r["км"] / 30, color="#d0101a" if "рабочий" in r["вариант"] else "#555")
        ax.annotate(f"{r['вариант']}\n{r['км']:.0f} км, F1 {r['v2 F'] :.3f}", (r["v2 P"], r["v2 R"]),
                    fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("точность P (эталон v2)"); ax.set_ylabel("полнота R (эталон v2)")
    ax.set_title(f"{site}: варианты тракта; размер точки — длина карты", loc="left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"ablation_tradeoff_{site}.png", dpi=150)
    plt.close(fig)
    log(f"сохранено: {FIG / f'ablation_tradeoff_{site}.png'}")


if __name__ == "__main__":
    main()
