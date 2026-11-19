"""ANFU шаг 6: сдвиг точек линий LINB2 к оси вогнутости GLO-30 (audit/ANFU_DECLARATION.md,
поправка 1 п. 7 и поправка 3).

Линии отбираются правилом LINB2 (среднее P, бюджет = длина RELU1) по несдвинутым
линиям; сдвигаются только отобранные. Каждая точка ломаной (центр ячейки 100 м)
сдвигается по нормали к максимуму вогнутости (σ 60 м, сетка 20 м) среди 15
положений −140…+140 м; сдвиги сглаживаются скользящей медианой по 5 точкам.
Плацебо: те же |сдвиги|, знак случайный один на линию, зёрна 1…19.
Решение (поправка 4): ΔS > 0 и > ΔS каждого из 19 плацебо (0,029 отменено).

Запуск: python -m experiments.openlin_anfu_snap [--pmap outputs/fam/linb1_utsjoki_карта.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from experiments.openlin_anfisa_lines import (M, TAG, add_lines, length_m, line_mean_p,
                                              rasterize)
from experiments.openlin_fam import CELL_M, DEM, OUT, RES_IN, log_to, multi_ridge

SNAP_SIG_M = (60,)               # меньший из масштабов вогнутости FAM (DEM_SIG_M)
SNAP_HALF_PX = 7                 # 7 × 20 м = 140 м: 15 положений, не дальше ±150 м
MED_PTS = 5                      # 5 точек × 100 м = 500 м
PLACEBO_SEEDS = tuple(range(1, 20))  # 19 = 1/0,05 − 1 (поправка 4 п. 2): ранг 1 из 20
ANFU6_N_TESTS = 0
#: ИТОГ (11.11.2026): S LINB2 0,1328 → после сдвига 0,1193 (ΔS −0,0135), плацебо −0,0244…−0,0091,
#: ранг 6 из 20 — ОТКАТ; длина +7,2 %.
# Порядок проверки положений: от центра наружу — при равенстве (ровное место, вогнутость 0)
# точка остаётся на месте, а не уезжает к краю окна.
_OFFS = np.array(sorted(range(-SNAP_HALF_PX, SNAP_HALF_PX + 1), key=lambda o: (abs(o), o)), float)


def normals(q):
    """Единичные нормали к хорде соседей (i−1, i+1); у концов — к крайней хорде."""
    n = len(q)
    a = q[np.maximum(np.arange(n) - 1, 0)]
    b = q[np.minimum(np.arange(n) + 1, n - 1)]
    t = b - a
    nt = np.hypot(t[:, 0], t[:, 1])
    nv = np.stack([-t[:, 1], t[:, 0]], 1) / np.maximum(nt, 1e-9)[:, None]
    nv[nt == 0] = 0
    return nv


def snap_shifts(pl, conc, valid, k):
    """Сдвиги (в пикселях 20 м) по нормали для точек ломаной pl (координаты сетки 100 м)."""
    q = pl * k + (k - 1) / 2
    n = len(q)
    if n < 2:
        return q, np.zeros((n, 2)), np.zeros(n)
    nv = normals(q)
    pts = q[:, None, :] + _OFFS[None, :, None] * nv[:, None, :]       # (n, 15, 2)
    ij = np.round(pts).astype(int)
    H, W = conc.shape
    inside = ((ij[..., 0] >= 0) & (ij[..., 0] < H) & (ij[..., 1] >= 0) & (ij[..., 1] < W)).all(1)
    sh = np.zeros(n)
    ok = inside & (np.hypot(nv[:, 0], nv[:, 1]) > 0)
    if ok.any():
        r, c = ij[ok, :, 0], ij[ok, :, 1]
        good = valid[r, c].all(1)
        v = conc[r, c]
        best = _OFFS[np.argmax(v, 1)]                   # первый максимум — ближе к центру
        sh[np.flatnonzero(ok)[good]] = best[good]
    sh = ndi.median_filter(sh, size=MED_PTS, mode="nearest")
    return q, nv, sh


def apply(q, nv, sh, k):
    return (q + sh[:, None] * nv - (k - 1) / 2) / k


def main():
    from experiments import openlin_ref_match as rm
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    from experiments.openlin_union_recall import add_new
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--pmap", default=None, help="карта P принятой модели (поправка 3 п. 6)")
    ap.add_argument("--lines", default=None, help="линии LINB1 принятой модели (_линии.npz)")
    a = ap.parse_args()
    site = a.site
    if site != "utsjoki":
        raise SystemExit("ANFU — только площадка настройки")
    log = log_to(OUT / f"_anfu_step6_{site}.log")
    log("ANFU шаг 6; объявление audit/ANFU_DECLARATION.md (поправка 1 п. 7, поправка 3)")
    lp = Path(a.lines) if a.lines else M / f"openlin_zone_lines_{site}_v3{TAG}_линии.npz"
    pp = Path(a.pmap) if a.pmap else OUT / f"linb1_{site}_карта.npz"
    log(f"линии {lp}; карта P {pp}")
    V = np.load(lp)
    flat, off = V["rc100"], V["offsets"]
    lines = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    P = np.load(pp)["P"]
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res = float(w["res_m"])
    o = np.load(M / f"openlin_zone_lines_{site}_v3_настройка1.npz")
    r = np.load(M / f"openlin_zone_lines_{site}_v3_рельеф1.npz")
    shape20 = o["C_all"].shape
    valid20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    om, rmk = o["C_all"], r["C_all"][:shape20[0], :shape20[1]]
    k = int(round(CELL_M / RES_IN))

    # ---- отбор LINB2 (как main_linb2, основной вариант с мостиками)
    ids_all = np.where(valid20, rasterize(lines, shape20, k), 0)
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    mp = np.array([line_mean_p(pl, P) for pl in lines])
    lenm = np.array([length_m(pl) for pl in lines])
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    budget = rmk.sum() * res / 1000
    cum = np.cumsum(cells[order]) * res / 1000
    n = int(np.searchsorted(cum, budget, side="right"))
    pick = order[:n]
    keep = np.zeros(len(lines) + 1, bool)
    keep[pick + 1] = True
    idsB = np.where(keep[ids_all], ids_all, 0)
    log(f"отбор LINB2: бюджет {budget:.0f} км, линий {n}, {(idsB > 0).sum() * res / 1000:.0f} км")
    if a.pmap is None and a.lines is None:
        saved = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса2.npz")["C_all"]
        same = bool(np.array_equal(saved, idsB > 0))
        log(f"  совпадение с сохранённым LINB2: {same}")
        if not same:
            raise SystemExit("отбор не воспроизводит LINB2 — остановка")

    # ---- сдвиг
    dem = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy").astype("float64")[:shape20[0], :shape20[1]]
    vdem = valid20 & np.isfinite(dem)
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    conc, _ = multi_ridge(dem, SNAP_SIG_M)
    sel = [lines[i] for i in pick]
    geo = [snap_shifts(pl, conc, vdem, k) for pl in sel]
    allsh = np.concatenate([g[2] for g in geo]) * RES_IN
    log(f"сдвиги, м: доля нулевых {100 * (allsh == 0).mean():.1f} %, |сдвиг| медиана "
        f"{np.median(np.abs(allsh)):.0f}, 90-й проц. {np.percentile(np.abs(allsh), 90):.0f}; "
        f"средний знак {np.mean(np.sign(allsh)):+.3f}")
    snapped = [apply(q, nv, sh, k) for q, nv, sh in geo]

    def ids_of(pls):
        return np.where(valid20, rasterize(pls, shape20, k), 0)
    sets = {}
    base = om | add_new(rmk, om, res)
    sets.update({"снимок": om, "рельеф": rmk, "снимок + рельеф": base,
                 "LINB2": idsB > 0, "LINB2 сдвиг": ids_of(snapped) > 0})
    for s in PLACEBO_SEEDS:
        rng = np.random.default_rng(s)
        sg = rng.choice([-1.0, 1.0], len(geo))
        pls = [apply(q, nv, np.abs(sh) * g, k) for (q, nv, sh), g in zip(geo, sg)]
        sets[f"плацебо {s}"] = ids_of(pls) > 0

    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(valid20, ang, np.nan)
    pos = rm.positions(ang, fid, valid20, rm.shifts(shape20))
    ev = {s: rm.evaluate(m_, pos, res) for s, m_ in sets.items()}
    K = int(np.floor(min(e[0].sum() for e in ev.values()) / rm.LEN_STEP_KM))
    log(f"семейство: K = {K} (L до {K * rm.LEN_STEP_KM:.0f} км)")
    S = {}
    rows = []
    for s, (lens, mins) in ev.items():
        cs = rm.curve_score(lens, mins, K)
        S[s] = cs["S"]
        log(f"  {s:18s} {lens.sum():6.0f} км | S = {cs['S']:.4f}")
        rows.append(dict(набор=s, км=lens.sum(), K=K, S=cs["S"]))
    dS = S["LINB2 сдвиг"] - S["LINB2"]
    dP = np.array([S[f"плацебо {s}"] - S["LINB2"] for s in PLACEBO_SEEDS])
    log(f"ΔS сдвига {dS:+.4f}; ΔS плацебо: среднее {dP.mean():+.4f}, SD {dP.std():.4f}, "
        f"от {dP.min():+.4f} до {dP.max():+.4f}; ранг сдвига {1 + int((dP >= dS).sum())} из "
        f"{1 + len(dP)}")
    lB, lS = ev["LINB2"][0].sum(), ev["LINB2 сдвиг"][0].sum()
    log(f"длина: LINB2 {lB:.0f} км, после сдвига {lS:.0f} км ({100 * (lS / lB - 1):+.1f} %)")
    log("ШАГ 6: " + ("ОСТАВИТЬ" if dS > 0 and dS > dP.max() else "ОТКАТИТЬ"))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / f"anfu_step6_{site}.csv", index=False, encoding="utf-8-sig")
    fl = np.concatenate(snapped)
    of = np.cumsum([0] + [len(p) for p in snapped])
    np.savez_compressed(M / f"openlin_zone_lines_{site}_v3_анфиса2_сдвиг.npz",
                        C_all=sets["LINB2 сдвиг"], C=sets["LINB2 сдвиг"])
    np.savez_compressed(M / f"openlin_zone_lines_{site}_v3_анфиса2_сдвиг_линии.npz",
                        rc100=fl, offsets=of, x0=V["x0"], y_top=V["y_top"], cell_m=CELL_M,
                        epsg=V["epsg"])


if __name__ == "__main__":
    main()
