"""REAL2: доля настоящих линий — доля побед в симметричных парах сдвигов (audit/REAL2_DECLARATION.md).

Утсйоки, описательно, REAL2_N_TESTS = 0. Лидар не используется. Порядок: допуск меры на
синтетике (белый шум, фазовые суррогаты свидетелей) — при провале выход; затем наборы,
тор-контроль по диагоналям, положительный и ориентационный контроли.
    python -m experiments.openlin_real2
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from experiments.openlin_real1 import (LAT, RES, SAME_M, admissible, geometry, hessians,
                                       load_sets, witnesses)

REAL2_N_TESTS = 0
PAIRS_PX = (13, 19, 25, 32)            # 260 м = 2·6+1 (основная); 380/500/640 — для сведения
MAIN_PX = 13
ERODE_PX = 8                           # 4σ при σ = 2 (обрезка гауссова ядра)
BLOCKS_PX = (int(12000 / RES), int(6000 / RES))
N_BOOT = 1000
R_SYN = 5
R_SYN_MAX = 15                         # поправка 1: один раз для всех проверок разом
T975_DF = {4: 2.776, 14: 2.145}        # t₀,₉₇₅(R − 1)
NEG_PX = int(round(3400 / RES))
ENV_PX = int(2000 / RES)               # LOCAL_WIN_M — окно огибающей

USE = {"снимок UTS1": ("S1", "S2 лето", "Landsat лето", "рельеф"),
       "рельеф RELU1": ("S1", "S2 лето", "Landsat лето"),
       "Анфиса LINB2": ("S1", "S2 лето", "Landsat лето")}
DEPENDENT = {"рельеф RELU1": {"S1"}, "Анфиса LINB2": {"S1"}, "снимок UTS1": set()}
OTHER_PHYSICS = ("S1", "S2 лето", "Landsat лето")


def t_stats(P, idx, T, N, nl, hs, valid, shape, swap=False):
    """T линии в положениях 0, ±d (d из PAIRS_PX): массив (nl, 1 + 2·len(PAIRS_PX))."""
    H, W = shape
    offs = [0.0] + [float(s * d) for d in PAIRS_PX for s in (1, -1)]
    lat = np.arange(-LAT, LAT + 1, dtype=float)
    n_pt, t_pt = N[idx], T[idx]
    a_pt, b_pt = (t_pt, n_pt) if swap else (n_pt, t_pt)       # ориентационный контроль
    tot = np.bincount(idx, minlength=nl)
    best = np.full((nl, len(offs)), -np.inf)
    for (hrr, hrc, hcc, s, scale) in hs:
        q2 = lambda r, c, v: hrr[r, c] * v[:, 0] ** 2 + 2 * hrc[r, c] * v[:, 0] * v[:, 1] + hcc[r, c] * v[:, 1] ** 2
        for j, o in enumerate(offs):
            mx = np.full(len(P), -np.inf)
            allv = np.ones(len(P), bool)                 # поправка 1 п. 3: все боковые положения в маске
            for k in lat:
                q = P + (o + k) * n_pt
                r = np.clip(np.round(q[:, 0]).astype(int), 0, H - 1)
                c = np.clip(np.round(q[:, 1]).astype(int), 0, W - 1)
                L = s * s * np.maximum(np.abs(q2(r, c, a_pt)) - np.abs(q2(r, c, b_pt)), 0) / scale
                inside = (q[:, 0] >= 0) & (q[:, 0] <= H - 1) & (q[:, 1] >= 0) & (q[:, 1] <= W - 1)
                allv &= valid[r, c] & inside
                mx = np.fmax(mx, L)
            mx[~allv | ~np.isfinite(mx)] = np.nan
            sums = np.bincount(idx, np.nan_to_num(mx), nl)
            cnt = np.bincount(idx, np.isfinite(mx), nl)
            Tj = np.where(cnt >= tot, sums / np.maximum(cnt, 1), np.nan)   # поправка 1: всё положение в маске
            best[:, j] = np.fmax(best[:, j], Tj)
    return best


def wins(st, d_px):
    """w линии для пары ±d: (c(T0,T+d) + c(T0,T−d)) / 2, c ∈ {1, 0,5, 0}."""
    j = 1 + 2 * PAIRS_PX.index(d_px)
    c = lambda a, b: np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))
    return (c(st[:, 0], st[:, j]) + c(st[:, 0], st[:, j + 1])) / 2


def pi_hat(w):
    return float(2 * np.mean(w) - 1) if len(w) else np.nan


def boot(w, cen, block_px, seed):
    rng = np.random.default_rng(seed)
    b = (cen[:, 0] // block_px).astype(int) * 100000 + (cen[:, 1] // block_px).astype(int)
    ub = np.unique(b)
    groups = {k: w[b == k] for k in ub}
    vals = [pi_hat(np.concatenate([groups[k] for k in rng.choice(ub, len(ub), replace=True)]))
            for _ in range(N_BOOT)]
    return np.percentile(vals, [2.5, 97.5])


def summarize(label, st, ok, cen, lens, log):
    fin = ok & np.all(np.isfinite(st[:, :3]), 1)
    w = wins(st[fin], MAIN_PX)
    if not len(w):
        log(f"  {label}: линий нет")
        return np.nan, np.nan
    ph = pi_hat(w)
    c12 = [boot(w, cen[fin], BLOCKS_PX[0], sd) for sd in (0, 1)]
    c6 = boot(w, cen[fin], BLOCKS_PX[1], 0)
    lo, hi = min(c[0] for c in c12), max(c[1] for c in c12)
    far = []
    for d in PAIRS_PX[1:]:
        f2 = ok & np.all(np.isfinite(st[:, [0, 1 + 2 * PAIRS_PX.index(d), 2 + 2 * PAIRS_PX.index(d)]]), 1)
        far.append(f"{d * RES:.0f} м {100 * pi_hat(wins(st[f2], d)):.0f}")
    terc = np.quantile(lens[fin], [1 / 3, 2 / 3])
    parts = []
    for a_, b_ in ((0, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)):
        m = (lens[fin] > a_) & (lens[fin] <= b_)
        parts.append(f"{100 * pi_hat(w[m]):.0f} (n={int(m.sum())})")
    log(f"  {label:36s}: π̂ {100 * ph:+.0f} %, НЕ МЕНЬШЕ {100 * lo:+.0f} % [12 км {100 * lo:+.0f}…{100 * hi:+.0f}; "
        f"6 км {100 * c6[0]:+.0f}…{100 * c6[1]:+.0f}] (линий {int(fin.sum())}); по дальности: {', '.join(far)}; "
        f"терцили длины: {' / '.join(parts)}")
    return ph, lo


def prep_witness(fields, shape):
    hs = [h for f in fields.values() for h in hessians(f)]
    valid = np.all([np.isfinite(f) for f in fields.values()], 0)
    valid = ndi.binary_erosion(valid, iterations=ERODE_PX, border_value=0)
    return hs, valid


def phase_surrogate(f, ph):
    """Суррогат поля с заданными фазами ph (общими для всех полей одного свидетеля — поправка 1)."""
    x = np.where(np.isfinite(f), f, np.nanmedian(f)).astype("float64")
    F = np.fft.rfft2(x - x.mean())
    return (np.fft.irfft2(np.abs(F) * ph, s=x.shape) + x.mean()).astype("float32")


def envelope(f):
    x = np.where(np.isfinite(f), f, np.nanmedian(f)).astype("float64")
    m = ndi.uniform_filter(x, ENV_PX)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(x * x, ENV_PX) - m * m, 0))
    return sd / max(float(np.median(sd)), 1e-12)


def admission(all_lines, W, shape, log, R):
    """Допуск: белый шум и фазовые суррогаты свидетелей дают π̂ ≈ 0; огибающая — пол фактуры."""
    P, idx, T, N = geometry(all_lines)
    nl = len(all_lines)
    ok = admissible(P, idx, N, nl, shape, np.zeros(shape, bool))
    passed, floors = True, {}
    tests = {"белый шум": None}
    tests.update({w: W[w] for w in sorted(set(sum(USE.values(), ())))})
    H, Wd = shape
    for name, fields in tests.items():
        vals, env_vals = [], []
        for r in range(R):
            rng = np.random.default_rng(7000 + r)
            if fields is None:
                syn = {"шум": rng.standard_normal(shape).astype("float32")}
            else:
                ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (H, Wd // 2 + 1)))
                ph[0, 0] = 1
                masks = {k: np.isfinite(f) for k, f in fields.items()}
                syn = {k: np.where(masks[k], phase_surrogate(f, ph), np.nan) for k, f in fields.items()}
            hs, valid = prep_witness(syn, shape)
            st = t_stats(P, idx, T, N, nl, hs, valid, shape)
            fin = ok & np.all(np.isfinite(st[:, :3]), 1)
            vals.append(pi_hat(wins(st[fin], MAIN_PX)))
            if fields is not None and r < 2:                 # пол фактуры: описание, 2 реализации
                env = {k: (syn[k] - np.nanmean(syn[k])) * envelope(f) for k, f in fields.items()}
                hs2, valid2 = prep_witness(env, shape)
                st2 = t_stats(P, idx, T, N, nl, hs2, valid2, shape)
                f2 = ok & np.all(np.isfinite(st2[:, :3]), 1)
                env_vals.append(pi_hat(wins(st2[f2], MAIN_PX)))
        m, se = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(R))
        tq = T975_DF[R - 1]
        ok_t = abs(m) <= tq * se
        passed &= ok_t
        extra = ""
        if env_vals:
            floors[name] = float(np.mean(env_vals))
            extra = f"; пол фактуры (суррогат × огибающая) {100 * floors[name]:+.1f} %"
        log(f"  допуск {name:13s}: π̂ по {R} реализациям {', '.join(f'{100 * v:+.1f}' for v in vals)} %; "
            f"среднее {100 * m:+.2f}, порог {tq}·SE = {100 * tq * se:.2f} -> {'ПРОЙДЕН' if ok_t else 'НЕ ПРОЙДЕН'}{extra}")
    return passed, floors


def evaluate(name, lines, Wp, use, shape, log, dependent=(), swap=False):
    P, idx, T, N = geometry(lines)
    nl = len(lines)
    ok = admissible(P, idx, N, nl, shape, np.zeros(shape, bool))
    cen = np.array([x.mean(0) for x in lines])
    lens = np.array([len(x) for x in lines]) * RES / 1000
    log(f"=== {name}: линий {nl}, в счёт (край листа) {int(ok.sum())}")
    lows, allst = {}, None
    for wn in use:
        hs, valid = Wp[wn]
        st = t_stats(P, idx, T, N, nl, hs, valid, shape, swap)
        _ph, lo = summarize(wn + (" (зависим)" if wn in dependent else ""), st, ok, cen, lens, log)
        if wn not in dependent:
            lows[wn] = lo
            allst = st if allst is None else np.fmax(allst, st)
    out = {}
    if lows:
        best = max(lows, key=lambda k: -9 if np.isnan(lows[k]) else lows[k])
        log(f"  ХОТЯ БЫ ОДНИМ независимым: не меньше {100 * lows[best]:+.0f} % ({best})")
        out["any"] = summarize("все независимые вместе (справка)", allst, ok, cen, lens, log)
        out["lows"] = lows
    return out


def main():
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_fam import OUT, log_to
    from experiments.openlin_real1 import M
    log = log_to(OUT / "_real2.log")
    log("REAL2; объявление audit/REAL2_DECLARATION.md; лидар не используется")
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    sets = load_sets(shape)
    W = witnesses(shape)
    W["рельеф"] = {"GLO-30 высота": W["GLO-30"]["высота"], "GLO-30 уклон": W["GLO-30"]["уклон"],
                   "ALOS высота": W["ALOS"]["высота"], "ALOS уклон": W["ALOS"]["уклон"]}
    all_lines = [x for v in sets.values() for x in v]
    log(f"пары сдвигов, ячеек: {PAIRS_PX} (основная {MAIN_PX} = {MAIN_PX * RES:.0f} м); маска свидетеля "
        f"сужена на {ERODE_PX} яч.")
    passed, floors = admission(all_lines, W, shape, log, R_SYN)
    if not passed:
        log(f"допуск при R = {R_SYN} не пройден — один повтор с R = {R_SYN_MAX} для всех проверок (поправка 1)")
        passed, floors = admission(all_lines, W, shape, log, R_SYN_MAX)
    if not passed:
        log("ДОПУСК НЕ ПРОЙДЕН — мера не откалибрована, наборы не считаются")
        return
    Wp = {wn: prep_witness(W[wn], shape) for wn in set(sum(USE.values(), ()))}
    res = {name: evaluate(name, lines, Wp, USE[name], shape, log, DEPENDENT[name]) for name, lines in sets.items()}
    log("пол фактуры по свидетелям (справка к итогу): " + ", ".join(f"{k} {100 * v:+.1f} %" for k, v in floors.items()))
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
                        continue                                   # перескочила бы край — выброс
                    y = np.stack([r, c], 1)
                    moved.append(y)
                    dirs.append((dr > 0) * 2 + (dc > 0))
                    near_cells.append(d_own[y[:, 0], y[:, 1]] <= SAME_M)
        share = float(np.concatenate(near_cells).mean())
        log(f"  {name}: сдвинутых линий {len(moved)}; ячеек в 127 м от исходных {100 * share:.0f} %")
        r = evaluate(name + " (тор)", moved, Wp, USE[name], shape, log, DEPENDENT[name])
        set_any = res[name].get("any", (np.nan, np.nan))[0]
        log(f"  потолок плотности = {100 * share:.0f} % × π̂ набора {100 * set_any:+.0f} % = {100 * share * set_any:+.1f} %")
        # критерий — по сводной (все независимые вместе): интервал содержит 0
        P_, idx_, T_, N_ = geometry(moved)
        ok_ = admissible(P_, idx_, N_, len(moved), shape, np.zeros(shape, bool))
        allst = None
        for wn in USE[name]:
            if wn in DEPENDENT[name]:
                continue
            st = t_stats(P_, idx_, T_, N_, len(moved), Wp[wn][0], Wp[wn][1], shape)
            allst = st if allst is None else np.fmax(allst, st)
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
    evaluate("trace S2 -> Landsat", tl, Wp, ("Landsat лето",), shape, log)
    evaluate("trace S2 -> S2 (проверка кода)", tl, Wp, ("S2 лето",), shape, log)
    log("=== КОНТРОЛЬ ориентационный: n и t переставлены в отклике; ожидается π̂ <= 0")
    for name, lines in sets.items():
        evaluate(name + " (n↔t)", lines, Wp, USE[name], shape, log, DEPENDENT[name], swap=True)


if __name__ == "__main__":
    main()
