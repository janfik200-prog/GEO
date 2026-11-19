"""REAL1: доля настоящих линий по независимым глобальным свидетелям, без лидара.

audit/REAL1_DECLARATION.md (+ поправки). Утсйоки, описательно, REAL1_N_TESTS = 0.
Для каждой линии набора: ориентированная линейность поля свидетеля по нормали линии,
максимум по боковому смещению ±127 м, среднее вдоль линии (T); ранг T среди 8 параллельных
сдвигов (±254…±635 м); доля настоящих снизу — по Стори (ранги 6–9, вероятность 4/9).
    python -m experiments.openlin_real1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REAL1_N_TESTS = 0
RES = 20.0
SAME_M = 127.0                                    # «та же линия», OPENLIN_RULES
SHIFTS_M = (254.0, 381.0, 508.0, 635.0)           # 2..5 × 127, обе стороны -> K = 8 (выбор дизайна)
SHIFTS_PX = tuple(int(round(d / 20.0)) for d in SHIFTS_M)   # поправка 1 п. 1: целые ячейки 13/19/25/32
LAT = int(SAME_M // RES)                          # 6 ячеек
BLOCKS_PX = (int(6000 / RES), int(12000 / RES))   # поправка 1 п. 1: 6 и 12 км, итог по большему
N_BOOT = 1000
NEG_SHIFT_PX = int(round(3400 / RES))             # отрицательный контроль, 3,4 км
M = Path("outputs/metrics")


# ------------------------------------------------------------------ наборы
def lines_from_pts(pts, lid):
    order = np.argsort(lid, kind="stable")
    pts, lid = pts[order], lid[order]
    cut = np.flatnonzero(np.diff(lid)) + 1
    return np.split(pts, cut)


def load_sets(shape):
    out = {}
    for name, tag in (("снимок UTS1", "_настройка1"), ("рельеф RELU1", "_рельеф1")):
        L = np.load(M / f"openlin_zone_lines_utsjoki_v3{tag}_линии.npz")
        pts, lid = L["C_all_pts"], L["C_all_id"]
        ok = (pts[:, 0] < shape[0]) & (pts[:, 1] < shape[1])
        out[name] = lines_from_pts(pts[ok], lid[ok])
    ids = np.load(M / "openlin_zone_lines_utsjoki_v3_анфиса2.npz")["ids"][:shape[0], :shape[1]]
    rr, cc = np.nonzero(ids)
    out["Анфиса LINB2"] = lines_from_pts(np.stack([rr, cc], 1), ids[rr, cc])
    return {k: [x for x in v if len(x) >= 2] for k, v in out.items()}


# ------------------------------------------------------------------ свидетели
def witnesses(shape):
    """Поля свидетелей на сетке листа: {свидетель: {поле: массив}}."""
    from cache_paths import CACHE
    from experiments.openlin_relief_lines import ALOS_BAD_M
    from experiments.openlin_typesites_fetch import DEST
    H, W = shape
    cut = lambda a: a[:H, :W].astype("float32")
    s1 = np.load(CACHE / "s1_look" / "s1_vv_utsjoki_лист_20m_половины.npz")
    keys = [k for k in s1.files if not k.startswith("orbits")]
    w = {"S1": {"VV": cut(np.nanmean(np.stack([s1[k] for k in keys]), 0))}}
    if (DEST / "_utsjoki" / "s2w_summer.npz").exists():   # поправка 3: может ещё качаться
        s2 = np.load(DEST / "_utsjoki" / "s2w_summer.npz")
        r, n = cut(s2["B04"]), cut(s2["B08"])
        w["S2 лето"] = {"красный": r, "NDVI": (n - r) / np.maximum(n + r, 1e-6)}
    ls = np.load(DEST / "_utsjoki" / "landsat_summer.npz")
    r, n = cut(ls["red"]), cut(ls["nir"])
    w["Landsat лето"] = {"красный": r, "NDVI": (n - r) / np.maximum(n + r, 1e-6)}
    from cache_paths import ANABAR_DEM
    g = cut(np.load(ANABAR_DEM / "dem_glo30_utsjoki_лист_20m.npy"))
    a = cut(np.load(ANABAR_DEM / "dem_alos_utsjoki_лист_20m.npy"))
    a = np.where(np.abs(a - g) > ALOS_BAD_M, np.nan, a)
    slope = lambda z: np.hypot(*np.gradient(np.where(np.isfinite(z), z, np.nanmedian(z)), RES))
    w["GLO-30"] = {"высота": g, "уклон": np.where(np.isfinite(g), slope(g), np.nan)}
    w["ALOS"] = {"высота": a, "уклон": np.where(np.isfinite(a), slope(a), np.nan)}
    return w


USE = {"снимок UTS1": ("S1", "S2 лето", "Landsat лето", "GLO-30", "ALOS"),
       "рельеф RELU1": ("S1", "S2 лето", "Landsat лето"),
       "Анфиса LINB2": ("S1", "S2 лето", "Landsat лето")}
#: поправка 1 п. 2а и поправка 2 п. 2: S1 RTC выправлен по GLO-30 — для наборов, построенных по
#: GLO-30, печатается с пометкой и в «хотя бы одним» не входит
DEPENDENT = {"рельеф RELU1": {"S1"}, "Анфиса LINB2": {"S1"}, "снимок UTS1": set()}


def hessians(field):
    """[(Hrr, Hrc, Hcc, масштаб нормировки)] по SIGMAS тракта; NaN -> медиана, маска отдельно."""
    from skimage.feature import hessian_matrix

    from experiments.openlin_iso import SIGMAS
    f = np.where(np.isfinite(field), field, np.nanmedian(field)).astype("float32")
    out = []
    for s in SIGMAS:
        hrr, hrc, hcc = hessian_matrix(f, sigma=s, order="rc", use_gaussian_derivatives=True)
        lam = 0.5 * np.abs(hrr + hcc) + np.sqrt(0.25 * (hrr - hcc) ** 2 + hrc ** 2)
        scale = float(np.nanmedian(lam[np.isfinite(field)]))
        out.append((hrr, hrc, hcc, s, max(scale, 1e-12)))
    return out


# ------------------------------------------------------------------ мера
def geometry(lines):
    """Ячейки всех линий, номер линии у ячейки, касательная и нормаль линии (PCA)."""
    P = np.concatenate(lines).astype(float)
    idx = np.concatenate([np.full(len(x), i) for i, x in enumerate(lines)])
    T = np.zeros((len(lines), 2))
    for i, x in enumerate(lines):
        q = x - x.mean(0)
        _w, v = np.linalg.eigh(q.T @ q)
        T[i] = v[:, -1]
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    return P, idx, T, N


def line_stats(P, idx, T, N, nlines, hs, valid, shape):
    """T линии для 9 положений (0 — сама, 1..8 — сдвиги), максимум по кандидатам (поле × σ)."""
    H, W = shape
    offs = [0.0] + [float(s * d) for d in SHIFTS_PX for s in (1, -1)]
    lat = np.arange(-LAT, LAT + 1, dtype=float)
    n_pt, t_pt = N[idx], T[idx]
    best = np.full((nlines, len(offs)), -np.inf)
    for (hrr, hrc, hcc, s, scale) in hs:
        nhn_c = lambda r, c, nv: hrr[r, c] * nv[:, 0] ** 2 + 2 * hrc[r, c] * nv[:, 0] * nv[:, 1] + hcc[r, c] * nv[:, 1] ** 2
        for j, o in enumerate(offs):
            mx = np.full(len(P), -np.inf)
            for k in lat:
                q = P + (o + k) * n_pt
                r = np.clip(np.round(q[:, 0]).astype(int), 0, H - 1)
                c = np.clip(np.round(q[:, 1]).astype(int), 0, W - 1)
                L = s * s * np.maximum(np.abs(nhn_c(r, c, n_pt)) - np.abs(nhn_c(r, c, t_pt)), 0) / scale
                L = np.where(valid[r, c], L, np.nan)
                mx = np.fmax(mx, L)
            mx[~np.isfinite(mx)] = np.nan
            sums = np.bincount(idx, np.nan_to_num(mx), nlines)
            cnt = np.bincount(idx, np.isfinite(mx), nlines)
            tot = np.bincount(idx, minlength=nlines)
            # объявление: > 50 % ячеек свидетеля пусто — положение не считается (линия выбывает в ranks_of)
            Tj = np.where(cnt >= 0.5 * tot, sums / np.maximum(cnt, 1), np.nan)
            best[:, j] = np.fmax(best[:, j], Tj)
    return best


def admissible(P, idx, N, nlines, shape, near):
    """Все 8 сдвигов внутри листа и не более половины ячеек сдвига ближе 127 м к чужим линиям."""
    H, W = shape
    ok = np.ones(nlines, bool)
    for d in SHIFTS_PX:
        for s in (1, -1):
            q = P + s * d * N[idx]
            r, c = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
            inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            ok &= np.bincount(idx, ~inside, nlines) == 0
            rr, cc = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
            frac = np.bincount(idx, near[rr, cc], nlines) / np.maximum(np.bincount(idx, minlength=nlines), 1)
            ok &= frac <= 0.5
    return ok


def storey(ranks):
    """Доля настоящих снизу: 1 − π₀, π₀ = доля рангов 6–9 / (4/9)."""
    if len(ranks) == 0:
        return np.nan
    return float(1 - min(1.0, np.mean(ranks >= 6) / (4 / 9)))


def boot(ranks, blocks_px, cen, seed):
    rng = np.random.default_rng(seed)
    blocks = (cen[:, 0] // blocks_px).astype(int) * 100000 + (cen[:, 1] // blocks_px).astype(int)
    ub = np.unique(blocks)
    groups = {b: ranks[blocks == b] for b in ub}
    vals = [storey(np.concatenate([groups[b] for b in rng.choice(ub, len(ub), replace=True)]))
            for _ in range(N_BOOT)]
    return np.nanpercentile(vals, [2.5, 97.5])


def ranks_of(st, ok):
    """Ранг линии среди 9: пессимистично при ничьих (поправка 1 п. 3 iii)."""
    fin = ok & np.all(np.isfinite(st), 1)
    return fin, 1 + (st[fin, 1:] >= st[fin, :1]).sum(1)


def summarize(label, rank, cen, lens, log):
    if len(rank) == 0:
        log(f"  {label}: линий нет")
        return np.nan, np.nan
    sh = storey(rank)
    ci = {b: [boot(rank, b, cen, sd) for sd in (0, 1)] for b in BLOCKS_PX}
    lo = min(c[0] for c in ci[BLOCKS_PX[1]])          # нижний край по большему блоку, худшее из зёрен
    hi = max(c[1] for c in ci[BLOCKS_PX[1]])
    hist = np.bincount(rank, minlength=10)[1:]
    terc = np.quantile(lens, [1 / 3, 2 / 3])
    parts = []
    for a_, b_ in ((0, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)):
        m = (lens > a_) & (lens <= b_)
        parts.append(f"{100 * storey(rank[m]):.0f} (n={int(m.sum())})")
    c6 = ci[BLOCKS_PX[0]][0]
    log(f"  {label:34s}: точка {100 * sh:.0f} %, НЕ МЕНЬШЕ {100 * lo:.0f} % [12 км: {100 * lo:.0f}–{100 * hi:.0f}; "
        f"6 км: {100 * c6[0]:.0f}–{100 * c6[1]:.0f}] (линий {len(rank)}); ранги 1..9: {' '.join(map(str, hist))}; "
        f"терцили длины: {' / '.join(parts)}")
    return sh, lo


def evaluate(name, lines, hs_by_w, use, shape, near, log, valid_by_w, dependent=()):
    P, idx, T, N = geometry(lines)
    nl = len(lines)
    ok_edge = admissible(P, idx, N, nl, shape, np.zeros(shape, bool))   # только край листа
    ok_near = admissible(P, idx, N, nl, shape, near)                       # + фильтр соседства
    cen = np.array([x.mean(0) for x in lines])
    lens = np.array([len(x) for x in lines]) * RES / 1000
    log(f"=== {name}: линий {nl}; в счёт (край листа) {int(ok_edge.sum())} ({100 * ok_edge.mean():.0f} %); "
        f"с фильтром соседства {int(ok_near.sum())} ({100 * ok_near.mean():.0f} %)")
    lows, allst = {}, None
    for wname in use:
        st = line_stats(P, idx, T, N, nl, hs_by_w[wname], valid_by_w[wname], shape)
        fin, rank = ranks_of(st, ok_edge)
        mark = " (зависим)" if wname in dependent else ""
        _sh, lo = summarize(wname + mark, rank, cen[fin], lens[fin], log)
        _f2, rank2 = ranks_of(st, ok_near)
        log(f"    чувствительность, с фильтром соседства: точка {100 * storey(rank2):.0f} % (линий {len(rank2)})")
        if wname == "Landsat лето":
            for h in hs_by_w[wname][:2]:                    # поле «красный», σ отдельно (поправка 1 п. 2г)
                stk = line_stats(P, idx, T, N, nl, [h], valid_by_w[wname], shape)
                _f3, r3 = ranks_of(stk, ok_edge)
                log(f"    Landsat, красный, σ = {h[3]:.0f} яч.: точка {100 * storey(r3):.0f} %")
        if wname not in dependent:
            lows[wname] = lo
            allst = st if allst is None else np.fmax(allst, st)
    if lows:
        best = max(lows, key=lambda k: -1 if np.isnan(lows[k]) else lows[k])
        log(f"  ХОТЯ БЫ ОДНИМ независимым свидетелем: не меньше {100 * lows[best]:.0f} % ({best}; максимум нижних краёв)")
        fin, rank = ranks_of(allst, ok_edge)
        summarize("все независимые вместе (T = максимум)", rank, cen[fin], lens[fin], log)


def main():
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_fam import OUT, log_to
    from experiments.openlin_zone_lines import COMP
    log = log_to(OUT / "_real1.log")
    log("REAL1; объявление audit/REAL1_DECLARATION.md; лидар не используется")
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    sets = load_sets(shape)
    union = np.zeros(shape, bool)
    for v in sets.values():
        P = np.concatenate(v)
        union[P[:, 0], P[:, 1]] = True
    near = distance_transform_edt(~union, sampling=(RES, RES)) <= SAME_M
    W = witnesses(shape)
    if "S2 лето" not in W:
        log("ЧАСТИЧНЫЙ ПРОГОН (поправка 3): слоя S2 лето нет — свидетели без S2, положительный контроль позже")
        for k in USE:
            USE[k] = tuple(x for x in USE[k] if x != "S2 лето")
    need = sorted({w for u in USE.values() for w in u})
    hs_by_w, valid_by_w = {}, {}
    for wn in need:
        hs_by_w[wn] = [h for f in W[wn].values() for h in hessians(f)]
        valid_by_w[wn] = np.all([np.isfinite(f) for f in W[wn].values()], 0)
        log(f"  свидетель {wn}: полей {len(W[wn])}, годных ячеек {100 * valid_by_w[wn].mean():.1f} %")
    # «near» для каждого набора — чужие линии; своя линия, сдвинутая на ≥ 254 м, себя не задевает,
    # но соседние линии своего же набора — тоже «чужие» (поля параллельных форм)
    log(f"  сдвиги, ячеек: {SHIFTS_PX} = {[d * RES for d in SHIFTS_PX]} м; боковой допуск {LAT} яч. = {LAT * RES:.0f} м")
    for name, lines in sets.items():
        evaluate(name, lines, hs_by_w, USE[name], shape, near, log, valid_by_w, DEPENDENT[name])
    # ---- контроли
    log("=== КОНТРОЛЬ отрицательный: сдвиг по тору на 3,4 км по ±x и ±y; нижний край должен быть 0")
    H, Wd = shape
    for name, lines in sets.items():
        Pm = np.concatenate(lines)
        own = np.zeros(shape, bool)
        own[Pm[:, 0], Pm[:, 1]] = True
        d_own = distance_transform_edt(~own, sampling=(RES, RES))
        for dr, dc in ((NEG_SHIFT_PX, 0), (-NEG_SHIFT_PX, 0), (0, NEG_SHIFT_PX), (0, -NEG_SHIFT_PX)):
            moved = [np.stack([(x[:, 0] + dr) % H, (x[:, 1] + dc) % Wd], 1) for x in lines]
            Q = np.concatenate(moved)
            log(f"  {name} сдвиг ({dr * RES / 1000:+.1f}, {dc * RES / 1000:+.1f}) км: ячеек сдвинутых линий "
                f"в 127 м от исходных {100 * (d_own[Q[:, 0], Q[:, 1]] <= SAME_M).mean():.0f} %")
            evaluate(f"{name} (контроль {dr},{dc})", moved, hs_by_w, USE[name], shape, near, log, valid_by_w,
                     DEPENDENT[name])
    log("=== КОНТРОЛЬ положительный (поправка 2 п. 1): trace по S2 лето (красный) -> свидетель Landsat; "
        "нижний край должен быть >= 0,5; S2 -> S2 — только проверка кода")
    if "S2 лето" not in W:
        log("  положительный контроль отложен: нет S2 лето")
        return
    from experiments.openlin_zone_lines import trace
    f = W["S2 лето"]["красный"]
    tl = [x["pts"] for x in trace(f, np.isfinite(f), RES, local=True)]
    tl = [x for x in tl if len(x) >= 2]
    log(f"  линий trace по S2 лето: {len(tl)}")
    evaluate("trace S2 лето -> Landsat", tl, hs_by_w, ("Landsat лето",), shape, near, log, valid_by_w)
    evaluate("trace S2 лето -> S2 лето (проверка кода)", tl, hs_by_w, ("S2 лето",), shape, near, log, valid_by_w)


if __name__ == "__main__":
    main()
