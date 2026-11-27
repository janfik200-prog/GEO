"""ACON1: линии «от противного» (a contrario) — линия засчитана, если NFA = N_тестов · B(n, k, p) <= 1
(audit/ACON1_DECLARATION.md, поправки 1–2). Без учителя: GTK и разметка в выбор не входят.

    python -m experiments.openlin_acon1 --site utsjoki          # настройка: шаги по подделке А, (а) на подделке Б
    python -m experiments.openlin_acon1 --site anabar           # только если (а) пройдено на Утсйоки
    python -m experiments.openlin_acon1 --site utsjoki --windows   # окна (после охотника)
ACON1_N_TESTS = 0. Только процессор. Соглашение углов: θ из openlin_cand1.response — направление черты,
вектор (строка, столбец) = (−cos θ, sin θ) (поправка 2).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy.stats import binom

from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments.openlin_fam import OUT, RES_IN

ACON1_N_TESTS = 0
TAU = np.pi / 8                                   # 22,5° — LSD, Sidiropoulou-Velidou 2015
LAGS_M = (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000)
SEED_A, SEED_B = 2311, 2411                       # подделки А (выбор шага) и Б (проверка); + номер канала
FAKE_B_MAX = 3                                    # наименьшее k: P(Пуассон(1) > k) <= 0,05 (хвост 0,019)
AMPS = (1.0, 2.0, 4.0, 8.0)                       # врезки, σ — для сведения (б)
TAG = "acon1"


def angdiff(a, b):
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


# ---------------------------------------------------------------- линии и оси
def traced(img, dem, valid, log, label):
    """Линии CAND2 по каналам (тот же тракт) + поля «черта есть» и направление черты."""
    out = {}
    for ch in range(5):
        r, th = C1.response(ch, img, dem)
        z = C2.local_norm_nz(r, valid)
        lines, km = C2.trace_z(z, th, valid, lambda *a: None, label)
        out[ch] = {"lines": lines, "pos": (r > 0) & valid, "dir": (th % np.pi).astype("float32"),
                   "p": float(2 * TAU / np.pi * np.mean(r[valid] > 0))}
        log(f"    {label} к{ch + 1}: линий {len(lines)}, {km:.0f} км; p = {out[ch]['p']:.4f}")
    return out


def axes(lines):
    """Главная ось каждой линии: центр, единичный вектор u (строка, столбец), угол a (соглашение ядер), s0, s1."""
    res = []
    for pl in lines:
        c = pl.mean(0)
        q = pl - c
        _w, v = np.linalg.eigh(q.T @ q)
        u = v[:, -1]
        s = q @ u
        a = math.atan2(u[1], -u[0]) % math.pi              # (−cos a, sin a) = ±u
        res.append((c, u, a, float(s.min()), float(s.max())))
    return res


def counts(ax, F, valid, d_px):
    """n и k по оси с шагом d_px ячеек."""
    H, W = valid.shape
    n = np.zeros(len(ax), int)
    k = np.zeros(len(ax), int)
    for i, (c, u, a, s0, s1) in enumerate(ax):
        t = np.arange(s0, s1 + 1e-9, d_px)
        rr = np.round(c[0] + t * u[0]).astype(int)
        cc = np.round(c[1] + t * u[1]).astype(int)
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[ok], cc[ok]
        ok = valid[rr, cc]
        rr, cc = rr[ok], cc[ok]
        n[i] = len(rr)
        k[i] = int(np.sum(F["pos"][rr, cc] & (angdiff(F["dir"][rr, cc], a) <= TAU)))
    return n, k


def nfa(n, k, p, n_tests):
    return n_tests * binom.sf(k - 1, n, p)                 # P(X >= k)


def n_tests_of(T):
    return sum(len(T[ch]["lines"]) for ch in T)


def choose_steps(TA, valid, log):
    """Поправка 1 п. 2: d_ch — наименьший лаг, начиная с которого на подделке А засчитано 0 на ВСЕХ бо́льших лагах."""
    N = n_tests_of(TA)
    steps = {}
    for ch in range(5):
        ax = axes(TA[ch]["lines"])
        cnt = []
        for lag in LAGS_M:
            n, k = counts(ax, TA[ch], valid, lag / RES_IN)
            cnt.append(int(np.sum(nfa(n, k, TA[ch]["p"], N) <= 1)))
        log(f"    подделка А к{ch + 1}: засчитано по лагам " + ", ".join(f"{l} м:{c}" for l, c in zip(LAGS_M, cnt)))
        d = None
        for j in range(len(LAGS_M)):
            if all(c == 0 for c in cnt[j:]):
                d = LAGS_M[j]
                break
        steps[ch] = d
        log(f"    к{ch + 1} {C1.CHANNELS[ch]}: шаг d = {d} м" + ("" if d else " — канал исключается (нет лага с 0)"))
    return steps


def reach(T, steps, N, log):
    """Поправка 1 п. 3: n_min (k = n) и n при k = n−1; длина; доля линий не короче."""
    keep = {}
    for ch, d in steps.items():
        if d is None:
            keep[ch] = False
            continue
        p = T[ch]["p"]
        n_min = math.ceil(math.log(N) / math.log(1 / p))
        n1 = next(n for n in range(2, 10000) if nfa(n, n - 1, p, N) <= 1)
        ax = axes(T[ch]["lines"])
        L = np.array([(s1 - s0) * RES_IN for _c, _u, _a, s0, s1 in ax])
        share = float(np.mean(L >= (n_min - 1) * d))
        keep[ch] = share > 0
        log(f"    к{ch + 1}: n_min {n_min} (k = n), {n1} (k = n−1); наименьшая длина {(n_min - 1) * d / 1000:.2f} км; "
            f"линий CAND2 не короче {100 * share:.1f} %" + ("" if keep[ch] else " — канал исключается"))
    return keep


def detect(T, steps, keep, valid, N):
    res = {}
    for ch in range(5):
        if not keep.get(ch):
            res[ch] = None
            continue
        ax = axes(T[ch]["lines"])
        n, k = counts(ax, T[ch], valid, steps[ch] / RES_IN)
        v = nfa(n, k, T[ch]["p"], N)
        res[ch] = {"nfa": v, "sel": v <= 1, "n": n, "k": k, "ax": ax}
    return res


# ---------------------------------------------------------------- описание
def raster(lines, sel, shape):
    from experiments.openlin_anfisa_lines import rasterize
    return rasterize([pl for pl, s in zip(lines, sel) if s], shape, 1) > 0


def sensitivity(site, img, dem, valid, steps, keep, log):
    """(б) для сведения: врезки прямых полос 4 км на 12 азимутах в снимок, амплитуды AMPS σ; каналы снимка 1–3."""
    import experiments.openlin_azimuth_response as ar
    from scipy.spatial import cKDTree

    from src.detection_limit import robust_sigma
    ar.INS_LEN_M = 4000.0
    sigma = robust_sigma(img[valid])
    rng = np.random.default_rng(ar.SEED)
    grid = ar.sites(img.shape, RES_IN, valid, rng)
    need = ar.AZ_BINS * min(ar.N_PER_BIN, len(grid) // ar.AZ_BINS)
    azs = np.array([(b + 0.5) * 180.0 / ar.AZ_BINS for b in range(ar.AZ_BINS)])
    plan = [(azs[i % ar.AZ_BINS], grid[i]) for i in range(need)]
    for amp in AMPS:
        x = img.astype("float64").copy()
        tracks = []
        for a_deg, ctr in plan:
            ar.insert_window(x, ctr, a_deg, amp, sigma, RES_IN)
            tracks.append(ar.track(img.shape, ctr, a_deg, RES_IN))
        T = traced(x, dem, valid, lambda *a: None, f"врезка {amp}σ")
        N = n_tests_of(T)
        det = detect(T, steps, {ch: keep.get(ch, False) and ch < 3 for ch in range(5)}, valid, N)
        hit = 0
        pts, own = [], []
        for ch in range(3):
            if det[ch] is None:
                continue
            for j in np.flatnonzero(det[ch]["sel"]):
                pl = T[ch]["lines"][j]
                pts.append(pl)
                own.append(np.full(len(pl), len(own)))
        if pts:
            tree = cKDTree(np.concatenate(pts))
            for tr in tracks:
                q = tree.query_ball_point(tr.astype(float), r=ar.MATCH_TOL_M / RES_IN)
                hit += np.mean([len(h) > 0 for h in q]) >= ar.MATCH_FRAC
        log(f"    (б) врезка {amp:g} σ: обнаружено {hit} из {len(tracks)} ({100 * hit / max(len(tracks), 1):.0f} %)")


def describe(site, T, det, TA_det, TB_det, valid, meta, log):
    shape = valid.shape
    area = valid.sum() * RES_IN ** 2 / 1e6
    pool = np.zeros(shape, bool)
    log("=== засчитанные: снимок / подделка А / подделка Б (по каналам)")
    for ch in range(5):
        if det[ch] is None:
            log(f"  к{ch + 1} {C1.CHANNELS[ch]}: исключён")
            continue
        sel = det[ch]["sel"]
        m = raster(T[ch]["lines"], sel, shape)
        pool |= m
        a_ = int(TA_det[ch]["sel"].sum()) if TA_det[ch] is not None else 0
        b_ = int(TB_det[ch]["sel"].sum()) if TB_det[ch] is not None else 0
        q = np.log10(np.maximum(det[ch]["nfa"][sel], 1e-300)) if sel.any() else np.array([np.nan])
        log(f"  к{ch + 1} {C1.CHANNELS[ch]}: {int(sel.sum())} из {len(sel)} линий, {m.sum() * RES_IN / 1000:.0f} км; "
            f"подделка А {a_}, Б {b_}; log10 NFA засчитанных: медиана {np.nanmedian(q):.1f}, мин {np.nanmin(q):.1f}")
    km = pool.sum() * RES_IN / 1000
    log(f"  всего засчитано {km:.0f} км (объединение), {km / area:.3f} км/км²")
    from scipy.ndimage import distance_transform_edt as edt
    c3 = OUT / f"cand3_{site}.npz"
    if c3.exists():
        z = np.load(c3)
        cp = np.zeros(shape, bool)
        for ch in range(1, 6):
            cp |= z[f"ids_{ch}"][:shape[0], :shape[1]] > 0
        d = edt(~cp) * RES_IN
        log(f"  засчитанные в 127 м от пула CAND3: {100 * np.mean(d[pool] <= 127):.1f} %; "
            f"пул CAND3 в 127 м от засчитанных: {100 * np.mean((edt(~pool) * RES_IN)[cp] <= 127):.1f} %")
    l6p = OUT / f"linb6_{site}.npz"
    if l6p.exists():
        l6 = np.load(l6p)["ids"][:shape[0], :shape[1]] > 0
        log(f"  LINB6 в 127 м от засчитанных: {100 * np.mean((edt(~pool) * RES_IN)[l6] <= 127):.1f} %")
    if site == "utsjoki":
        gtk = C1.gtk_mask(site, meta, shape, valid)
        log(f"  для сведения: GTK 292 в 300 м от засчитанных {100 * C1.cover(gtk, pool, 300):.1f} % при {km:.0f} км "
            f"(без вывода «лучше/хуже»)")
    return pool


def save(site, T, det, steps, meta):
    d = {"res_m": RES_IN, **meta, "steps": json.dumps(steps)}
    for ch in range(5):
        if det[ch] is None:
            continue
        sel = det[ch]["sel"]
        d[f"nfa_{ch + 1}"] = det[ch]["nfa"]
        d[f"sel_{ch + 1}"] = sel
        pts = [pl for pl, s in zip(T[ch]["lines"], sel) if s]
        d[f"pts_{ch + 1}"] = np.concatenate(pts) if pts else np.zeros((0, 2))
        d[f"id_{ch + 1}"] = np.concatenate([np.full(len(p), i) for i, p in enumerate(pts)]) if pts else np.zeros(0, int)
    np.savez_compressed(OUT / f"{TAG}_{site}.npz", **d)


def run(site, log):
    from experiments.openlin_null_input import phase_randomize
    img, dem, valid, meta = C1.load_inputs(site)
    log(f"=== {site}: {img.shape}, годных {int(valid.sum())}")

    def fakes(seed0):
        fi = {ch: phase_randomize(img, valid, np.random.default_rng(seed0 + ch)).astype("float64") for ch in range(3)}
        fd = {ch: phase_randomize(dem, valid, np.random.default_rng(seed0 + ch)).astype("float64") for ch in (3, 4)}
        T = {}
        # каждый канал — на своей подделке (зерно seed0 + канал, как CAND1/2)
        for ch in range(5):
            x = fi[ch] if ch < 3 else img
            y = fd[ch] if ch >= 3 else dem
            r, th = C1.response(ch, x, y)
            z = C2.local_norm_nz(r, valid)
            lines, km = C2.trace_z(z, th, valid, lambda *a: None, "подделка")
            T[ch] = {"lines": lines, "pos": (r > 0) & valid, "dir": (th % np.pi).astype("float32"),
                     "p": float(2 * TAU / np.pi * np.mean(r[valid] > 0))}
            log(f"    подделка {seed0} к{ch + 1}: линий {len(lines)}, {km:.0f} км; p = {T[ch]['p']:.4f}")
        return T

    log("  подделка А (выбор шага):")
    TA = fakes(SEED_A)
    steps = choose_steps(TA, valid, log)
    log("  снимок:")
    T = traced(img, dem, valid, log, "снимок")
    N = n_tests_of(T)
    log(f"  N тестов (линий снимка, 5 каналов) = {N}")
    log("  достижимость (до засчёта):")
    keep = reach(T, steps, N, log)
    log("  подделка Б (проверка):")
    TB = fakes(SEED_B)
    NB = n_tests_of(TB)
    TB_det = detect(TB, steps, keep, valid, NB)
    nb = sum(int(TB_det[ch]["sel"].sum()) for ch in range(5) if TB_det[ch] is not None)
    a_ok = nb <= FAKE_B_MAX
    log(f"(а) подделка Б: засчитано {nb} линий по 5 каналам <= {FAKE_B_MAX} -> {'да' if a_ok else 'НЕТ — модель фона неверна'}"
        f" (условие слабое: при истинном среднем 3 проходит с вероятностью 0,65)")
    TA_det = detect(TA, steps, keep, valid, n_tests_of(TA))
    det = detect(T, steps, keep, valid, N)
    describe(site, T, det, TA_det, TB_det, valid, meta, log)
    save(site, T, det, steps, meta)
    if site == "utsjoki":
        log("  (б) чувствительность:")
        sensitivity(site, img, dem, valid, steps, keep, log)
    log(f"ИТОГ ACON1 {site}: (а) {'да' if a_ok else 'НЕТ'}")
    return a_ok


def windows(site, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img, dem, valid, _m = C1.load_inputs(site)
    z = np.load(OUT / f"{TAG}_{site}.npz")
    gy, gx = np.gradient(dem, RES_IN)
    az = np.radians(315)
    shade = np.clip(0.6 + 0.4 * (-(gx * np.sin(az) - gy * np.cos(az)) / np.maximum(np.hypot(gx, gy), 1e-3)
                                 * np.tanh(np.hypot(gx, gy) * 10)), 0, 1)
    sel_pts = [z[f"pts_{ch}"] for ch in range(1, 6) if f"pts_{ch}" in z.files]
    P = np.concatenate(sel_pts) if sel_pts else np.zeros((0, 2))
    dst = OUT / f"{TAG}_{site}"
    dst.mkdir(exist_ok=True)
    win_km = 6 if site == "utsjoki" else 8
    H, Wd = img.shape
    wp = int(win_km * 1000 / RES_IN)
    k = 0
    for i in range(3):
        for j in range(3):
            k += 1
            r0, c0 = int((i + 0.5) * H / 3 - wp / 2), int((j + 0.5) * Wd / 3 - wp / 2)
            fig, axs = plt.subplots(1, 3, figsize=(27, 9.4))
            for ax, (t, a, show) in zip(axs, (("зимний S2, без линий", img, False), ("S2 + засчитанные", img, True),
                                              ("рельеф GLO-30 + засчитанные", shade, True))):
                b = a[r0:r0 + wp, c0:c0 + wp]
                lo, hi = np.nanpercentile(b, [2, 98])
                ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=(0, win_km, win_km, 0))
                ax.set_title(t, fontsize=14)
                if show and len(P):
                    y = P - [r0, c0]
                    m = (y[:, 0] >= 0) & (y[:, 0] < wp) & (y[:, 1] >= 0) & (y[:, 1] < wp)
                    ax.plot(y[m, 1] * RES_IN / 1000, y[m, 0] * RES_IN / 1000, ".", ms=1.8, color="red")
                ax.set_xlim(0, win_km)
                ax.set_ylim(win_km, 0)
            fig.suptitle(f"ACON1 «от противного», {site}, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}. "
                         f"Красные — NFA <= 1", fontsize=14)
            fig.tight_layout()
            p = dst / f"окно_{k}.png"
            fig.savefig(p, dpi=70)
            plt.close(fig)
            log(f"  {p}")


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_len1 import sha
    site = sys.argv[sys.argv.index("--site") + 1]
    head = f"sha256 кода {sha(__file__)}, CAND2 {sha(C2.__file__)}, CAND1 {sha(C1.__file__)}"
    if "--windows" in sys.argv:
        log = log_to(OUT / f"_{TAG}_{site}_windows.log")
        log(f"ACON1 окна {site}; {head}")
        windows(site, log)
        return
    log = log_to(OUT / f"_{TAG}_{site}.log")
    log(f"ACON1 {site}; audit/ACON1_DECLARATION.md (поправки 1–2); {head}; τ = {np.degrees(TAU):.1f}°")
    if site == "anabar":
        prev = (OUT / f"_{TAG}_utsjoki.log").read_text(encoding="utf-8")
        assert "ИТОГ ACON1 utsjoki: (а) да" in prev, "Анабар — только если (а) пройдено на Утсйоки"
    run(site, log)


if __name__ == "__main__":
    main()
