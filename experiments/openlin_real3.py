"""REAL3: REAL2 с исправленным боковым окном — жёсткий целый перенос (audit/REAL3_DECLARATION.md).

Утсйоки, описательно, REAL3_N_TESTS = 0. Лидар не используется. Клетка бокового положения k
линии в положении o: P + round(o·n) + round(k·n) — у положений 0 и ±d один и тот же набор клеток
окна, сдвинутый на целый вектор. Допуск: R = 15 сразу, по каждому набору отдельно, 13 проверок,
порог t₀,₉₉₈₀₈(14)·SE (Бонферрони на семью), повтора нет.
    python -m experiments.openlin_real3
"""
from __future__ import annotations

import numpy as np

from experiments.openlin_real1 import LAT, RES, SAME_M, geometry, load_sets, witnesses
from experiments.openlin_real2 import (BLOCKS_PX, DEPENDENT, ENV_PX, MAIN_PX, PAIRS_PX, USE, boot,
                                       envelope, phase_surrogate, pi_hat, prep_witness, summarize,
                                       wins)

REAL3_N_TESTS = 0
R_SYN = 15                               # выбор дизайна по времени (объявление п. 3)
N_CHECKS = 13                            # 3 (шум × набор) + 4 + 3 + 3 (свидетель × набор по USE)
T_BONF = 3.4576                          # t_{1−0,025/13}(14), scipy (поправка 1 п. 1)
NEG_PX = int(round(3400 / RES))
assert N_CHECKS == 3 + sum(len(v) for v in USE.values())
assert ENV_PX == int(2000 / RES)


def int_vec(N, idx, o):
    """Целый вектор round(o·n) для каждой ячейки (n — одна на линию)."""
    return np.round(o * N).astype(int)[idx]


def window(N, idx):
    """Целые смещения бокового окна round(k·n), k = −6…6: массив (13, ncells, 2)."""
    return np.stack([np.round(k * N).astype(int)[idx] for k in range(-LAT, LAT + 1)])


def t_stats(P, idx, T, N, nl, hs, valid, shape, pairs=PAIRS_PX, swap=False):
    """T линии в положениях 0, ±d: (nl, 1 + 2·len(pairs)). Все клетки — целые, без обрезки края."""
    H, W = shape
    Pi = P.astype(int)
    offs = [0] + [s * d for d in pairs for s in (1, -1)]
    win = window(N, idx)
    n_pt, t_pt = N[idx], T[idx]
    a_pt, b_pt = (t_pt, n_pt) if swap else (n_pt, t_pt)
    tot = np.bincount(idx, minlength=nl)
    best = np.full((nl, len(offs)), -np.inf)
    cells = []
    for o in offs:
        ov = int_vec(N, idx, o)
        cc = [Pi + ov + kv for kv in win]
        # проверка-утверждение (поправка 1 п. 2г): клетки минус R(o) = клетки положения 0, все линии
        assert all(np.array_equal(c - ov, Pi + kv) for c, kv in zip(cc, win))
        ins = [(c[:, 0] >= 0) & (c[:, 0] < H) & (c[:, 1] >= 0) & (c[:, 1] < W) for c in cc]
        rc = [(np.where(i_, c[:, 0], 0), np.where(i_, c[:, 1], 0)) for c, i_ in zip(cc, ins)]
        cells.append((rc, ins))
    for (hrr, hrc, hcc, s, scale) in hs:
        q2 = lambda r, c, v: hrr[r, c] * v[:, 0] ** 2 + 2 * hrc[r, c] * v[:, 0] * v[:, 1] + hcc[r, c] * v[:, 1] ** 2
        for j, (rc, ins) in enumerate(cells):
            mx = np.full(len(P), -np.inf)
            allv = np.ones(len(P), bool)
            for (r, c), i_ in zip(rc, ins):
                L = s * s * np.maximum(np.abs(q2(r, c, a_pt)) - np.abs(q2(r, c, b_pt)), 0) / scale
                allv &= valid[r, c] & i_
                mx = np.fmax(mx, L)
            mx[~allv | ~np.isfinite(mx)] = np.nan
            sums = np.bincount(idx, np.nan_to_num(mx), nl)
            cnt = np.bincount(idx, np.isfinite(mx), nl)
            Tj = np.where(cnt >= tot, sums / np.maximum(cnt, 1), np.nan)
            best[:, j] = np.fmax(best[:, j], Tj)
    return best


def admissible(P, idx, N, nl, shape):
    """Все клетки окон (o ± 6) всех положений ±d из PAIRS_PX внутри листа (поправка 1 п. 2в)."""
    H, W = shape
    Pi = P.astype(int)
    win = window(N, idx)
    ok = np.ones(nl, bool)
    for d in PAIRS_PX:
        for s in (1, -1):
            ov = int_vec(N, idx, s * d)
            for kv in (win[0], win[-1]):
                c = Pi + ov + kv
                inside = (c[:, 0] >= 0) & (c[:, 0] < H) & (c[:, 1] >= 0) & (c[:, 1] < W)
                ok &= np.bincount(idx, ~inside, nl) == 0
    return ok


def overlap_report(lines, log):
    """|round(13n)| мин/макс и доля линий, у которых окна положений 0 и +13 делят клетку."""
    _P, _idx, _T, N = geometry(lines)
    v = np.round(MAIN_PX * N).astype(int)
    ln = np.hypot(v[:, 0], v[:, 1])
    ov = 0
    for i in range(len(N)):
        w0 = {tuple(np.round(k * N[i]).astype(int)) for k in range(-LAT, LAT + 1)}
        w1 = {(a + v[i, 0], b + v[i, 1]) for a, b in w0}
        ov += bool(w0 & w1)
    log(f"  перенос: |round({MAIN_PX}·n)| от {ln.min():.2f} до {ln.max():.2f} яч.; окна 0 и +{MAIN_PX} "
        f"пересекаются у {100 * ov / len(N):.1f} % линий")


def admission(sets, W, shape, log):
    """Допуск по каждому набору: белый шум и суррогаты свидетелей; поле и зерно общие для наборов."""
    all_lines = [x for v in sets.values() for x in v]
    lab = np.concatenate([np.full(len(v), i) for i, v in enumerate(sets.values())])
    names = list(sets)
    P, idx, T, N = geometry(all_lines)
    nl = len(all_lines)
    ok = admissible(P, idx, N, nl, shape)
    log(f"  линий в счёт (край листа): " + ", ".join(f"{n} {int((ok & (lab == i)).sum())}" for i, n in enumerate(names)))
    H, Wd = shape
    wit = sorted(set(sum(USE.values(), ())))                       # Landsat 0, S1 1, S2 2, рельеф 3
    vals = {}                                                      # (набор, проверка) -> [π̂ по r]
    floors = {}
    for wi, wn in enumerate([None] + wit):
        for r in range(R_SYN):
            if wn is None:
                rng = np.random.default_rng(9000 + r)
                syn = {"шум": rng.standard_normal(shape).astype("float32")}
            else:
                rng = np.random.default_rng(9100 + 100 * (wi - 1) + r)
                ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (H, Wd // 2 + 1)))
                ph[0, 0] = 1
                syn = {k: np.where(np.isfinite(f), phase_surrogate(f, ph), np.nan) for k, f in W[wn].items()}
            hs, valid = prep_witness(syn, shape)
            st = t_stats(P, idx, T, N, nl, hs, valid, shape, pairs=(MAIN_PX,))
            fin = ok & np.all(np.isfinite(st), 1)
            if wn is not None:
                env = {k: (syn[k] - np.nanmean(syn[k])) * envelope(f) for k, f in W[wn].items()}
                hs2, valid2 = prep_witness(env, shape)
                st2 = t_stats(P, idx, T, N, nl, hs2, valid2, shape, pairs=(MAIN_PX,))
                f2 = ok & np.all(np.isfinite(st2), 1)
            for i, sn in enumerate(names):
                if wn is not None and wn not in USE[sn]:
                    continue
                key = (sn, wn or "белый шум")
                m = fin & (lab == i)
                vals.setdefault(key, []).append(pi_hat(wins(st[m], MAIN_PX)))
                if wn is not None:
                    m2 = f2 & (lab == i)
                    floors.setdefault(key, []).append(pi_hat(wins(st2[m2], MAIN_PX)))
        log(f"  … {wn or 'белый шум'}: {R_SYN} реализаций готово")
    assert len(vals) == N_CHECKS, len(vals)
    passed, thr = True, {}
    for (sn, wn), v in vals.items():
        m, se = float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))
        okt = abs(m) <= T_BONF * se
        passed &= okt
        thr[(sn, wn)] = T_BONF * se
        fl = f"; пол фактуры {100 * np.mean(floors[(sn, wn)]):+.1f} %" if (sn, wn) in floors else ""
        log(f"  допуск {sn:14s} × {wn:13s}: среднее {100 * m:+.2f} п. п., SE {100 * se:.2f}, "
            f"порог {T_BONF}·SE = {100 * T_BONF * se:.2f} -> {'ПРОЙДЕН' if okt else 'НЕ ПРОЙДЕН'}{fl}")
    return passed, {k: float(np.mean(v)) for k, v in floors.items()}, thr


def evaluate(name, lines, Wp, use, shape, log, dependent=(), swap=False):
    P, idx, T, N = geometry(lines)
    nl = len(lines)
    ok = admissible(P, idx, N, nl, shape)
    cen = np.array([x.mean(0) for x in lines])
    lens = np.array([len(x) for x in lines]) * RES / 1000
    log(f"=== {name}: линий {nl}, в счёт (край листа) {int(ok.sum())}")
    lows, allst = {}, None
    for wn in use:
        hs, valid = Wp[wn]
        st = t_stats(P, idx, T, N, nl, hs, valid, shape, swap=swap)
        _ph, lo = summarize(wn + (" (зависим)" if wn in dependent else ""), st, ok, cen, lens, log)
        if wn not in dependent:
            lows[wn] = lo
            allst = st if allst is None else np.fmax(allst, st)
    out = {"ok": ok, "allst": allst}
    if lows:
        best = max(lows, key=lambda k: -9 if np.isnan(lows[k]) else lows[k])
        log(f"  ХОТЯ БЫ ОДНИМ независимым: не меньше {100 * lows[best]:+.0f} % ({best}; отбор по максимуму)")
        out["any"] = summarize("все независимые вместе (справка)", allst, ok, cen, lens, log)
        out["lows"] = lows
    return out


def main():
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_fam import OUT, log_to
    from experiments.openlin_real1 import M
    log = log_to(OUT / "_real3.log")
    log("REAL3; объявление audit/REAL3_DECLARATION.md (с поправкой 1); лидар не используется")
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    sets = load_sets(shape)
    W = witnesses(shape)
    W["рельеф"] = {"GLO-30 высота": W["GLO-30"]["высота"], "GLO-30 уклон": W["GLO-30"]["уклон"],
                   "ALOS высота": W["ALOS"]["высота"], "ALOS уклон": W["ALOS"]["уклон"]}
    overlap_report([x for v in sets.values() for x in v], log)
    log(f"ДОПУСК: R = {R_SYN}, проверок {N_CHECKS}, порог {T_BONF}·SE, повтора нет")
    passed, floors, thr = admission(sets, W, shape, log)
    if not passed:
        log("ДОПУСК НЕ ПРОЙДЕН — мера не откалибрована, наборы не считаются")
        return
    log("ДОПУСК ПРОЙДЕН")
    Wp = {wn: prep_witness(W[wn], shape) for wn in set(sum(USE.values(), ()))}
    res = {}
    for name, lines in sets.items():
        res[name] = evaluate(name, lines, Wp, USE[name], shape, log, DEPENDENT[name])
        log(f"  неопределённость допуска ({name}): " + ", ".join(
            f"{wn} ±{100 * thr[(name, wn)]:.1f} п. п." for wn in USE[name]) +
            f"; белый шум ±{100 * thr[(name, 'белый шум')]:.1f}")
        log(f"  пол фактуры ({name}): " + ", ".join(f"{wn} {100 * floors[(name, wn)]:+.1f} %" for wn in USE[name]))
    log("справка: пары 19/25/32 допуском не проверены, выводов по ним нет")
    # ---- тор-контроль по диагоналям
    log("=== КОНТРОЛЬ отрицательный: тор по диагоналям ±3,4 км, перескочившие линии выброшены; "
        "интервал сводной π̂ должен содержать 0")
    H, Wd = shape
    neg_ok = True
    for name, lines in sets.items():
        Pm = np.concatenate(lines)
        own = np.zeros(shape, bool)
        own[Pm[:, 0], Pm[:, 1]] = True
        d_own = distance_transform_edt(~own, sampling=(RES, RES))
        moved, near_cells, dirs = [], [], []
        for dr in (NEG_PX, -NEG_PX):
            for dc in (NEG_PX, -NEG_PX):
                for x in lines:
                    r, c = x[:, 0] + dr, x[:, 1] + dc
                    if r.min() < 0 or r.max() >= H or c.min() < 0 or c.max() >= Wd:
                        continue
                    y = np.stack([r, c], 1)
                    moved.append(y)
                    dirs.append((dr > 0) * 2 + (dc > 0))
                    near_cells.append(d_own[y[:, 0], y[:, 1]] <= SAME_M)
        share = float(np.concatenate(near_cells).mean())
        log(f"  {name}: сдвинутых линий {len(moved)}; ячеек в 127 м от исходных {100 * share:.0f} %")
        r = evaluate(name + " (тор)", moved, Wp, USE[name], shape, log, DEPENDENT[name])
        set_any = res[name].get("any", (np.nan, np.nan))[0]
        log(f"  потолок плотности = {100 * share:.0f} % × π̂ набора {100 * set_any:+.0f} % = {100 * share * set_any:+.1f} %")
        allst, ok_ = r["allst"], r["ok"]
        fin = ok_ & np.all(np.isfinite(allst[:, :3]), 1)
        w = wins(allst[fin], MAIN_PX)
        cen = np.array([x.mean(0) for x in moved])[fin]
        ci = boot(w, cen, BLOCKS_PX[0], 0)
        inc = ci[0] <= 0 <= ci[1]
        dd = np.array(dirs)[fin]
        nb = len(np.unique((cen[:, 0] // BLOCKS_PX[0]).astype(int) * 100000 + (cen[:, 1] // BLOCKS_PX[0]).astype(int)))
        log("  по направлениям: " + ", ".join(f"{['--', '-+', '+-', '++'][k]} {100 * pi_hat(w[dd == k]):+.1f} %"
                                          for k in range(4)) + f"; блоков 12 км {nb}")
        neg_ok &= inc
        log(f"  СВОДНО {name}: π̂ {100 * pi_hat(w):+.1f} %, интервал {100 * ci[0]:+.1f}…{100 * ci[1]:+.1f} -> "
            + ("содержит 0 — пройдено" if inc else "НЕ содержит 0 — не пройдено"))
    log("ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ: " + ("ПРОЙДЕН" if neg_ok else "НЕ ПРОЙДЕН — итог по наборам не выдаётся"))
    # ---- положительный и ориентационный
    log("=== КОНТРОЛЬ положительный: trace по S2 лето (B04) -> Landsat лето; нижний край должен быть >= 0,5")
    from experiments.openlin_zone_lines import trace
    f = W["S2 лето"]["красный"]
    tl = [x["pts"] for x in trace(f, np.isfinite(f), RES, local=True) if len(x["pts"]) >= 2]
    log(f"  линий trace: {len(tl)}")
    pr = evaluate("trace S2 -> Landsat", tl, Wp, ("Landsat лето",), shape, log)
    lo = pr.get("lows", {}).get("Landsat лето", np.nan)
    log("  ПОЛОЖИТЕЛЬНЫЙ: " + ("пройден" if lo >= 0.5 else "НЕ пройден — чувствительность меры не подтверждена"))
    evaluate("trace S2 -> S2 (проверка кода)", tl, Wp, ("S2 лето",), shape, log)
    log("=== КОНТРОЛЬ ориентационный: n и t переставлены в отклике; ожидается нижний край <= 0")
    for name, lines in sets.items():
        evaluate(name + " (n↔t)", lines, Wp, USE[name], shape, log, DEPENDENT[name], swap=True)


if __name__ == "__main__":
    main()
