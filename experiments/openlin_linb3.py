"""LINB3: отбор линий Анфисы по среднему (P − фон 2 км) — audit/LINB3_DECLARATION.md.

Всё как LINB2 (линии LINB1, мостики, бюджет = длина «рельефа»), меняется только величина,
по которой упорядочены линии: P минус openlin_winter.detrend(P) с DETREND_M на сетке 100 м.
Описательно, LINB3_N_TESTS = 0.
    python -m experiments.openlin_linb3
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

LINB3_N_TESTS = 0


def p_local(P):
    """P минус местный фон (тот же вычет фона, что у снимка тракта, масштаб DETREND_M)."""
    from experiments.openlin_fam import CELL_M
    from experiments.openlin_winter import detrend
    return detrend(P, np.isfinite(P), CELL_M)


def utsjoki(log):
    from pathlib import Path

    from experiments import openlin_ref_match as rm
    from experiments.openlin_fam import OUT
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    from experiments.openlin_ts1 import linb2_pick
    M = Path("outputs/metrics")
    site = "utsjoki"
    V = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса1_линии.npz")
    lines = [V["rc100"][V["offsets"][i]:V["offsets"][i + 1]] for i in range(len(V["offsets"]) - 1)]
    P = np.load(OUT / f"linb1_{site}_карта.npz")["P"]
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res = float(w["res_m"])
    o = np.load(M / f"openlin_zone_lines_{site}_v3_настройка1.npz")["C_all"]
    shape20 = o.shape
    r = np.load(M / f"openlin_zone_lines_{site}_v3_рельеф1.npz")["C_all"][:shape20[0], :shape20[1]]
    v20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    budget = r.sum() * res / 1000
    l2 = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса2.npz")["C_all"]
    l2c = linb2_pick(lines, P, shape20, v20, budget, res) > 0
    Pl = p_local(P)
    ok = np.isfinite(P) & np.isfinite(Pl)
    log(f"Утсйоки: доля дисперсии P, ушедшая в фон 2 км: {100 * np.var(P[ok] - Pl[ok]) / np.var(P[ok]):.0f} %")
    l3 = linb2_pick(lines, Pl, shape20, v20, budget, res) > 0
    log(f"Утсйоки: LINB2 воспроизведён: {bool(np.array_equal(l2, l2c))}; общих ячеек LINB3 и LINB2 "
        f"{100 * (l3 & l2).sum() / max(l2.sum(), 1):.0f} %")
    np.savez_compressed(M / f"openlin_zone_lines_{site}_v3_анфиса3.npz", C_all=l3, C=l3)
    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(v20, ang, np.nan)
    pos = rm.positions(ang, fid, v20, rm.shifts(shape20))
    sets = {"снимок": o, "рельеф": r, "Анфиса LINB2": l2, "Анфиса LINB3": l3}
    ev = {s: rm.evaluate(m, pos, res) for s, m in sets.items()}
    K = int(np.floor(min(e[0].sum() for e in ev.values()) / rm.LEN_STEP_KM))
    log(f"  семейство «весь лист»: K = {K}")
    Sv = {}
    for s, (lens, mins) in ev.items():
        Sv[s] = rm.curve_score(lens, mins, K)["S"]
        log(f"  {s:14s} {lens.sum():6.0f} км | S = {Sv[s]:.4f}")
    # поправка 1 п. 3: шум ΔS пары LINB3 − LINB2 по половинам эталона, как П3 в MEAS1
    lines_ref = np.unique(fid[fid > 0])
    dd = []
    for k in range(20):
        half = np.random.default_rng(5000 + k).permutation(lines_ref)[:len(lines_ref) // 2]
        d2 = []
        for pick in (np.isin(lines_ref, half), ~np.isin(lines_ref, half)):
            keep = lines_ref[pick]
            ph = [(rc[np.isin(f, keep)], a_[np.isin(f, keep)], f[np.isin(f, keep)]) for rc, a_, f in pos]
            e3, e2 = rm.evaluate(l3, ph, res), rm.evaluate(l2, ph, res)
            d2.append(rm.curve_score(*e3, K)["S"] - rm.curve_score(*e2, K)["S"])
        dd.append((d2[0] - d2[1]) / 2)
    sd3 = float(np.std(dd, ddof=1))
    thr = 1.96 * np.sqrt(2) * sd3
    dS = Sv["Анфиса LINB3"] - Sv["Анфиса LINB2"]
    log(f"  ΔS LINB3 − LINB2 = {dS:+.4f}; SD_П3 пары {sd3:.4f}, 1,96·√2·SD = {thr:.4f} -> "
        + ("больше шума выборки эталона" if abs(dS) > thr else "в пределах шума выборки эталона"))


def sites(log):
    import pandas as pd

    from experiments.openlin_fam import all_windows, quantile_norm, win_mask  # noqa: F401
    from experiments.openlin_ts1 import (anfisa_features, anfisa_line_az, anfisa_lines,
                                         az_hist, linb2_pick)
    from experiments.openlin_ts2 import SITES as TS2_SITES
    from experiments.openlin_ts2 import ref_mask
    from experiments.openlin_ts2_diag import above, shifts
    from experiments.openlin_typesites_fetch import DEST, OUT, load_sites
    S = load_sites()
    rows = []
    for name, site in S.items():
        d = DEST / name
        if site["lat"] is None or not (d / "ts1_masks.npz").exists():
            continue
        z = np.load(d / "s2_img.npz")
        g = np.load(d / "glo30.npz")["dem"]
        valid20 = (z["count"] > 0) & np.isfinite(z["shade"]) & np.isfinite(g)
        F, valid = anfisa_features(z["shade"], g, valid20)
        lines, P = anfisa_lines(F, valid, lambda *_: None)
        mk = np.load(d / "ts1_masks.npz")
        budget = mk["рельеф"].sum() * 0.02 if mk["рельеф"].any() else None
        v20 = np.isfinite(z["shade"])
        ids3 = linb2_pick(lines, p_local(P), g.shape, v20, budget, 20.0)
        l3 = ids3 > 0
        keep3 = set(np.unique(ids3[l3]).tolist())
        h = az_hist(anfisa_line_az(lines, keep3))
        row = dict(участок=name, вид=site["type"], LINB2_км=round(mk["анфиса2"].sum() * 0.02, 1),
                   LINB3_км=round(l3.sum() * 0.02, 1),
                   общих_с_LINB2=round((l3 & mk["анфиса2"]).sum() / max(mk["анфиса2"].sum(), 1), 3),
                   LINB3_азимут=int(15 * np.argmax(h) + 7.5) if l3.any() else None)
        if name in TS2_SITES:
            ref, _n, _c = ref_mask(name, site, TS2_SITES[name])
            if ref.any():
                sh = shifts(ref, *ref.shape)
                for lab, m in (("LINB2", mk["анфиса2"]), ("LINB3", l3)):
                    dist = distance_transform_edt(~m, sampling=(20.0, 20.0))
                    for dd in (100, 300):
                        ob = float((dist[ref] <= dd).mean())
                        nul = [float((dist[np.roll(ref, s_, (0, 1)) & valid20] <= dd).mean()) for s_ in sh]
                        row[f"{lab}_выше_{dd}"] = above(ob, nul)
                # поправка 1 п. 4: разброс случайного отбора той же длины (100 раз) на этом участке
                from experiments.openlin_anfisa_lines import rasterize
                ids_all = np.where(v20, rasterize(lines, g.shape, 5), 0)
                cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
                rk = []
                for s_ in range(100):
                    order = np.random.default_rng(10_000 + s_).permutation(len(lines))
                    nsel = int(np.searchsorted(np.cumsum(cells[order]) * 0.02, budget, side="right"))
                    kp = np.zeros(len(lines) + 1, bool)
                    kp[order[:nsel] + 1] = True
                    dist = distance_transform_edt(~kp[ids_all], sampling=(20.0, 20.0))
                    ob = float((dist[ref] <= 300).mean())
                    rk.append(above(ob, [float((dist[np.roll(ref, s2, (0, 1)) & valid20] <= 300).mean()) for s2 in sh]))
                q1, q3 = np.percentile(rk, [25, 75])
                row["случ_IQR_300"] = f"{q1:.0f}-{q3:.0f}"
                log(f"  {name}: выше k из 19 при 100/300 м — LINB2 {row['LINB2_выше_100']}/{row['LINB2_выше_300']}, "
                    f"LINB3 {row['LINB3_выше_100']}/{row['LINB3_выше_300']}"
                    + f"; случайный отбор при 300 м: квартили {q1:.0f}–{q3:.0f}; разница LINB3−LINB2 "
                    + ("читается" if abs(row['LINB3_выше_300'] - row['LINB2_выше_300']) > q3 - q1 else "не читается")
                    + ("  (участок, по которому придумано правило)" if name == "ts_strikeslip_carrizo" else ""))
        np.savez_compressed(d / "linb3_mask.npz", LINB3=l3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "linb3_sites.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    log(f"  участки: общих ячеек LINB3 с LINB2 — медиана {100 * df['общих_с_LINB2'].median():.0f} %")


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_typesites_fetch import OUT
    log = log_to(OUT / "_linb3.log")
    log("LINB3; объявление audit/LINB3_DECLARATION.md")
    utsjoki(log)
    sites(log)


if __name__ == "__main__":
    main()
