"""LINB4: отбор линий Анфисы по геометрии — audit/LINB4_DECLARATION.md (+ поправка 1).

LINB4 — порядок по хорде (расстояние между концами, м), ничьи по длине пути, затем по номеру;
LINB4L (контроль поправки 1) — порядок по длине пути. Бюджет = длина «рельефа», как LINB2.
Описательно, LINB4_N_TESTS = 0.   python -m experiments.openlin_linb4
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

LINB4_N_TESTS = 0


def geom(lines):
    from experiments.openlin_anfisa_lines import length_m
    from experiments.openlin_fam import CELL_M
    lenm = np.array([length_m(pl) for pl in lines])
    chord = np.array([float(np.hypot(*(pl[-1] - pl[0]))) * CELL_M for pl in lines])
    return lenm, chord


def pick(lines, ids_all, key, lenm, budget_km, res=20.0, order=None):
    """Отбор до бюджета в заданном порядке (как linb2_pick, другой ключ)."""
    if order is None:
        order = np.lexsort((np.arange(len(lines)), -lenm, -key))
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    n = int(np.searchsorted(np.cumsum(cells[order]) * res / 1000, budget_km, side="right"))
    keep = np.zeros(len(lines) + 1, bool)
    keep[order[:n] + 1] = True
    return np.where(keep[ids_all], ids_all, 0)


def utsjoki(log):
    from pathlib import Path

    from experiments import openlin_ref_match as rm
    from experiments.openlin_anfisa_lines import rasterize
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    M = Path("outputs/metrics")
    site = "utsjoki"
    V = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса1_линии.npz")
    lines = [V["rc100"][V["offsets"][i]:V["offsets"][i + 1]] for i in range(len(V["offsets"]) - 1)]
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res = float(w["res_m"])
    o = np.load(M / f"openlin_zone_lines_{site}_v3_настройка1.npz")["C_all"]
    shape20 = o.shape
    r = np.load(M / f"openlin_zone_lines_{site}_v3_рельеф1.npz")["C_all"][:shape20[0], :shape20[1]]
    v20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    ids_all = np.where(v20, rasterize(lines, shape20, 5), 0)
    lenm, chord = geom(lines)
    budget = r.sum() * res / 1000
    l2 = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса2.npz")["C_all"]
    l4 = pick(lines, ids_all, chord, lenm, budget) > 0
    l4L = pick(lines, ids_all, lenm, lenm, budget, order=np.lexsort((np.arange(len(lines)), -lenm))) > 0
    log(f"Утсйоки: замкнутых линий (хорда 0) {int((chord == 0).sum())} из {len(lines)}")
    np.savez_compressed(M / f"openlin_zone_lines_{site}_v3_анфиса4.npz", C_all=l4, C=l4)
    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(v20, ang, np.nan)
    pos = rm.positions(ang, fid, v20, rm.shifts(shape20))
    sets = {"снимок": o, "рельеф": r, "LINB2": l2, "LINB4": l4, "LINB4L": l4L}
    ev = {s: rm.evaluate(m, pos, res) for s, m in sets.items()}
    K = int(np.floor(min(e[0].sum() for e in ev.values()) / rm.LEN_STEP_KM))
    Sv = {s: rm.curve_score(*e, K)["S"] for s, e in ev.items()}
    log(f"  K = {K}; " + "; ".join(f"{s} S = {v:.4f}" for s, v in Sv.items()))
    lines_ref = np.unique(fid[fid > 0])
    for a_, b_ in (("LINB4", "LINB2"), ("LINB4", "LINB4L")):
        dd = []
        for k in range(20):
            half = np.random.default_rng(5000 + k).permutation(lines_ref)[:len(lines_ref) // 2]
            d2 = []
            for sel in (np.isin(lines_ref, half), ~np.isin(lines_ref, half)):
                keep = lines_ref[sel]
                ph = [(rc[np.isin(f, keep)], an[np.isin(f, keep)], f[np.isin(f, keep)]) for rc, an, f in pos]
                d2.append(rm.curve_score(*rm.evaluate(sets[a_], ph, res), K)["S"]
                          - rm.curve_score(*rm.evaluate(sets[b_], ph, res), K)["S"])
            dd.append((d2[0] - d2[1]) / 2)
        thr = 1.96 * np.sqrt(2) * float(np.std(dd, ddof=1))
        dS = Sv[a_] - Sv[b_]
        log(f"  ΔS {a_} − {b_} = {dS:+.4f}; порог шума половин {thr:.4f} -> "
            + ("больше шума" if abs(dS) > thr else "в пределах шума"))


def sites(log):
    import pandas as pd

    from experiments.openlin_anfisa_lines import rasterize
    from experiments.openlin_ts1 import anfisa_features, anfisa_lines
    from experiments.openlin_ts2 import SITES as TS2_SITES
    from experiments.openlin_ts2 import ref_mask
    from experiments.openlin_ts2_diag import above, shifts
    from experiments.openlin_typesites_fetch import DEST, OUT, load_sites
    S = load_sites()
    rows = []
    for name, kind in TS2_SITES.items():
        site = S[name]
        ref, _n, _c = ref_mask(name, site, kind)
        if not ref.any():
            continue
        d = DEST / name
        z = np.load(d / "s2_img.npz")
        g = np.load(d / "glo30.npz")["dem"]
        valid20 = (z["count"] > 0) & np.isfinite(z["shade"]) & np.isfinite(g)
        F, valid = anfisa_features(z["shade"], g, valid20)
        lines, _P = anfisa_lines(F, valid, lambda *_: None)
        mk = np.load(d / "ts1_masks.npz")
        budget = mk["рельеф"].sum() * 0.02
        ids_all = np.where(np.isfinite(z["shade"]), rasterize(lines, g.shape, 5), 0)
        lenm, chord = geom(lines)
        sets = {"LINB2": mk["анфиса2"],
                "LINB4": pick(lines, ids_all, chord, lenm, budget) > 0,
                "LINB4L": pick(lines, ids_all, lenm, lenm, budget,
                               order=np.lexsort((np.arange(len(lines)), -lenm))) > 0}
        sh = shifts(ref, *ref.shape)

        def k19(m, dd=300):
            dist = distance_transform_edt(~m, sampling=(20.0, 20.0))
            ob = float((dist[ref] <= dd).mean())
            return above(ob, [float((dist[np.roll(ref, s_, (0, 1)) & valid20] <= dd).mean()) for s_ in sh])

        def rand_iqr(pool):
            rk = []
            for s_ in range(100):
                order = np.random.default_rng(10_000 + s_).permutation(pool)
                rk.append(k19(pick(lines, ids_all, lenm, lenm, budget, order=order) > 0))
            return np.percentile(rk, [25, 75])
        q_all = rand_iqr(np.arange(len(lines)))
        med2 = np.median(lenm)                       # медиана длины линий набора (поправка 1 п. 2)
        q_long = rand_iqr(np.flatnonzero(lenm >= med2))
        row = dict(участок=name, случ_IQR=f"{q_all[0]:.0f}-{q_all[1]:.0f}",
                   случ_длинные_IQR=f"{q_long[0]:.0f}-{q_long[1]:.0f}",
                   замкнутых=int((chord == 0).sum()))
        for s_, m in sets.items():
            row[f"{s_}_100"], row[f"{s_}_300"] = k19(m, 100), k19(m, 300)
        w_ = q_all[1] - q_all[0]
        verdict = lambda a, b: "читается" if abs(row[f"{a}_300"] - row[f"{b}_300"]) > w_ else "не читается"
        log(f"  {name}: выше k из 19 при 100/300 м — " + ", ".join(
            f"{s_} {row[f'{s_}_100']}/{row[f'{s_}_300']}" for s_ in sets)
            + f"; случайный {q_all[0]:.0f}–{q_all[1]:.0f}, случайный среди длинных {q_long[0]:.0f}–{q_long[1]:.0f}; "
            f"LINB4−LINB2 {verdict('LINB4', 'LINB2')}, LINB4−LINB4L {verdict('LINB4', 'LINB4L')}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "linb4_sites.csv", index=False, encoding="utf-8-sig")


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_typesites_fetch import OUT
    log = log_to(OUT / "_linb4.log")
    log("LINB4; объявление audit/LINB4_DECLARATION.md (+ поправка 1)")
    utsjoki(log)
    sites(log)


if __name__ == "__main__":
    main()
