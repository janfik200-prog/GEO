"""ANZ2b: чёткие линии по карте ANZ2 — оси уверенных клеток, длинные пути по графу скелета, сглаживание 60 м,
прямые мостики LIRA.

Объявление audit/ANZ2B_DECLARATION.md. Сеть ANZ2 не переобучается; порог карты — из ANZ2.
    python -m experiments.openlin_anz2b --run
    python -m experiments.openlin_anz2b --windows
ANZ2B_N_TESTS = 0.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, label

from experiments import openlin_anz1 as A1
from experiments import openlin_anz2 as A2
from experiments import openlin_syn1 as S1
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_iso import Q_LOW
from experiments.openlin_zone_lines import DIR_TOL_DEG
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, LINK_MIN_LEN_M, TWIN_OFFSET_M, V3_GAP_M

ANZ2B_N_TESTS = 0
TEACHER = "Анфиса-карта ANZ2, линии ANZ2b (длинные пути, сглаживание 60 м, прямые мостики LIRA); учитель — искусственные линии"
SMOOTH_M = CONF_TOL_M            # σ сглаживания вдоль линии — допуск ветки «линия на объекте» (объявление п. 3)
OUT_DIR = OUT / "anz2b"
_EIGHT = np.ones((3, 3), bool)
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- граф скелета → длинные пути
def _graph(cells):
    from scipy.sparse import csr_matrix
    idx = {tuple(c): i for i, c in enumerate(cells)}
    r, c, w = [], [], []
    for i, (y, x) in enumerate(cells):
        for dy, dx in _NB:
            j = idx.get((y + dy, x + dx))
            if j is not None and j > i:
                r.append(i); c.append(j); w.append(math.hypot(dy, dx))
    n = len(cells)
    return csr_matrix((w, (r, c)), shape=(n, n))


def _longest_path(cells):
    """Приближённый геодезический диаметр (два прохода Дейкстры). → (длина в клетках, индексы пути)."""
    from scipy.sparse.csgraph import dijkstra
    g = _graph(cells)
    d0 = dijkstra(g, directed=False, indices=0)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    return float(da[b]), path[::-1]


def long_paths(region):
    """Скелет области → в каждой связной части снимается самый длинный путь, пока остаются куски ≥ 200 м (п. 2)."""
    from skimage.morphology import skeletonize
    sk = skeletonize(region)
    lab, n = label(sk, structure=_EIGHT)
    out = []
    objs = __import__("scipy.ndimage", fromlist=["find_objects"]).find_objects(lab)
    min_px = LINK_MIN_LEN_M / RES_IN
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        off = np.array([sl[0].start, sl[1].start])
        stack = [sub]
        while stack:
            m = stack.pop()
            cells = np.argwhere(m)
            if len(cells) < min_px:
                continue
            ln, p = _longest_path(cells)
            if ln * RES_IN < LINK_MIN_LEN_M:
                continue
            out.append((cells[p] + off).astype("float64"))
            m = m.copy()
            m[cells[p, 0], cells[p, 1]] = False
            l2, n2 = label(m, structure=_EIGHT)
            for j in range(1, n2 + 1):
                stack.append(l2 == j)
    return out


# ---------------------------------------------------------------- сглаживание
def _resample(pl, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pl, axis=0).T))]
    if d[-1] < step:
        return pl
    s = np.arange(0, d[-1] + 1e-9, step)
    return np.c_[np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])]


def smooth(pl):
    """Равный шаг 1 клетка → гаусс вдоль линии σ = 60 м (концы — mode nearest). → (линия, смещения в м)."""
    q = _resample(pl)
    s = SMOOTH_M / RES_IN
    if len(q) < 3:
        return q, np.zeros(len(q))
    sm = np.c_[gaussian_filter1d(q[:, 0], s, mode="nearest"), gaussian_filter1d(q[:, 1], s, mode="nearest")]
    return sm, np.hypot(*(sm - q).T) * RES_IN


def plen_m(pl):
    return float(np.hypot(*np.diff(pl, axis=0).T).sum() * RES_IN) if len(pl) > 1 else 0.0


# ---------------------------------------------------------------- мостики LIRA (прямой отрезок)
def _end_dir(pl, head):
    k = min(len(pl) - 1, max(2, int(round(SMOOTH_M / RES_IN))))
    p = pl[0] if head else pl[-1]
    q = pl[k] if head else pl[-1 - k]
    d = p - q
    n = np.hypot(*d)
    return p, d / n if n > 0 else d


def bridges(lines, score, low):
    from scipy.spatial import cKDTree
    ends = []
    for i, pl in enumerate(lines):
        for head in (True, False):
            p, d = _end_dir(pl, head)
            ends.append((i, head, p, d))
    gap, off = V3_GAP_M / RES_IN, TWIN_OFFSET_M / RES_IN
    cos_tol = math.cos(math.radians(DIR_TOL_DEG))
    H, W = score.shape
    m = int(off) + 2
    found = []
    if ends:
        tree = cKDTree(np.array([e[2] for e in ends]))
        for a, b in tree.query_pairs(gap):
            ia, _ha, pa, da = ends[a]
            ib, _hb, pb, db = ends[b]
            if ia == ib or np.dot(da, -db) < cos_tol:
                continue
            v = pb - pa
            if np.hypot(*v) > 0:
                if np.dot(v, da) <= 0 or np.dot(-v, db) <= 0:
                    continue
                if max(abs(da[0] * v[1] - da[1] * v[0]), abs(db[0] * v[1] - db[1] * v[0])) > off:
                    continue
            ga, gb = tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int))
            box = (max(min(ga[0], gb[0]) - m, 0), min(max(ga[0], gb[0]) + m + 1, H),
                   max(min(ga[1], gb[1]) - m, 0), min(max(ga[1], gb[1]) + m + 1, W))
            s, path = A1.widest_path(score, ga, gb, box)
            if path and s >= low:
                found.append((s, a, b))
    found.sort(key=lambda t: -t[0])
    used, acc = set(), []
    parent = list(range(len(lines)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, a, b in found:
        if a in used or b in used:
            continue
        ra, rb = root(ends[a][0]), root(ends[b][0])
        if ra == rb:                      # не замыкать петли
            continue
        used |= {a, b}
        parent[ra] = rb
        acc.append((a, b))
    return ends, acc


def bridges_out(ends, acc, score, low):
    """Сколько принятых мостиков (прямой отрезок) лежат не целиком в клетках ≥ нижнего уровня (поправка 1, п. 5)."""
    n_out = 0
    for a, b in acc:
        pa, pb = ends[a][2], ends[b][2]
        k = max(int(np.hypot(*(pb - pa))) + 2, 2)
        t = np.linspace(0, 1, k)[:, None]
        q = np.clip(np.round(pa + t * (pb - pa)).astype(int), 0, np.array(score.shape) - 1)
        if (score[q[:, 0], q[:, 1]] < low).any():
            n_out += 1
    return n_out


def merge(lines, ends, acc):
    """Цепочки линий по принятым мостикам → одна полилиния на цепочку (мостик — прямой отрезок)."""
    link = {}
    for a, b in acc:
        link[a] = b; link[b] = a
    eid = lambda i, head: 2 * i + (0 if head else 1)
    seen, out, kmb = set(), [], 0.0
    deg = {i: sum(eid(i, h) in link for h in (True, False)) for i in range(len(lines))}
    for start in [i for i in range(len(lines)) if deg[i] < 2] + list(range(len(lines))):
        if start in seen:
            continue
        # начать со свободного конца
        head_free = eid(start, True) not in link
        cur, entered_head = start, head_free
        chain = []
        while cur is not None and cur not in seen:
            seen.add(cur)
            pl = lines[cur] if entered_head else lines[cur][::-1]
            if chain:
                kmb += float(np.hypot(*(chain[-1][-1] - pl[0]))) * RES_IN / 1000
            chain.append(pl)
            exit_end = eid(cur, not entered_head)
            nxt = link.get(exit_end)
            if nxt is None:
                break
            cur, entered_head = nxt // 2, (nxt % 2 == 0)
        out.append((np.vstack(chain), len(chain)))
    return out, kmb


def rasterize(lines, shape):
    from skimage.draw import line as dline
    m = np.zeros(shape, bool)
    for pl in lines:
        q = np.clip(np.round(pl).astype(int), 0, np.array(shape) - 1)
        for (r0, c0), (r1, c1) in zip(q[:-1], q[1:]):
            rr, cc = dline(r0, c0, r1, c1)
            m[rr, cc] = True
    return m


def lines_clean(score, valid, thr, low, log, lbl):
    """→ (маска, полилинии, сводка)."""
    score = np.where(valid, score, 0.0).astype("float32")
    # поправка 2: область — гистерезис (семена ≥ thr, рост по ≥ low, как ANZ1b), оси — длинные пути
    lab, _n = label((score >= low) & valid, structure=_EIGHT)
    seeds = np.unique(lab[(score >= thr) & valid])
    region = np.isin(lab, seeds[seeds > 0])
    raw = long_paths(region)
    sm0 = [smooth(p) for p in raw]
    base = [p for p, _d in sm0 if plen_m(p) >= LINK_MIN_LEN_M]
    ends, acc = bridges(base, score, low)
    merged, kmb = merge(base, ends, acc)
    n_out = bridges_out(ends, acc, score, low)
    sm = [smooth(p) if n_parts > 1 else (p, np.zeros(len(p))) for p, n_parts in merged]   # заново — только слитые
    final = [p for p, _d in sm if plen_m(p) >= LINK_MIN_LEN_M]
    mask = rasterize(final, score.shape)
    disp = np.concatenate([d for _p, d in sm0 + sm]) if sm else np.zeros(1)
    tort = np.array([plen_m(p) / max(np.hypot(*(p[-1] - p[0])) * RES_IN, 1e-9) for p in final]) if final else np.ones(1)
    st = {"область км²": float(region.sum()) * RES_IN ** 2 / 1e6, "км путей": sum(map(plen_m, raw)) / 1000,
          "км после сглаживания": sum(plen_m(p) for p, _d in sm0) / 1000,
          "путей": len(raw), "линий до сшивки": len(base), "км до сшивки": sum(map(plen_m, base)) / 1000,
          "мостиков": len(acc), "км мостиков": kmb, "мостиков не целиком в клетках ≥ нижнего уровня": n_out,
          "замкнутых ячеек": int(label(~mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))[1]) - 1,
          "линий": len(final), "км": sum(map(plen_m, final)) / 1000,
          "смещение сглаживания м (99 %, макс)": [float(np.quantile(disp, 0.99)), float(disp.max())],
          "извилистость (медиана, 90 %)": [float(np.median(tort)), float(np.quantile(tort, 0.9))]}
    log(f"    {lbl}: область {st['область км²']:.0f} км²; путей {st['путей']} ({st['км путей']:.0f} км, после сглаживания "
        f"{st['км после сглаживания']:.0f} км), до сшивки {st['линий до сшивки']} ({st['км до сшивки']:.0f} км), мостиков "
        f"{st['мостиков']} ({kmb:.1f} км, {100 * kmb / max(st['км'], 1e-9):.1f} % линий; вне коридора {n_out}), "
        f"замкнутых ячеек {st['замкнутых ячеек']}, итог {st['линий']} линий ({st['км']:.0f} км); смещение сглаживания 99 % "
        f"{st['смещение сглаживания м (99 %, макс)'][0]:.0f} м, макс {st['смещение сглаживания м (99 %, макс)'][1]:.0f} м; "
        f"извилистость медиана {st['извилистость (медиана, 90 %)'][0]:.2f}, 90 % {st['извилистость (медиана, 90 %)'][1]:.2f}")
    return mask, final, st


# ---------------------------------------------------------------- прогон
def run(log):
    import torch
    from src.detection_limit import robust_sigma
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a2 = json.loads((A2.OUT_DIR / "anz2_итог.json").read_text(encoding="utf-8"))
    thr = float(a2["порог"]["значение"])
    res = {"объявление": "audit/ANZ2B_DECLARATION.md", "учитель": TEACHER, "ANZ2B_N_TESTS": ANZ2B_N_TESTS,
           "порог карты ANZ2": thr, "σ сглаживания м": SMOOTH_M}
    dev = A1.device()
    net = A1.make_net(12).to(dev)
    net.load_state_dict(torch.load(A2.OUT_DIR / "anz2_net.pt", map_location=dev))
    log(f"  сеть ANZ2 загружена ({dev}); порог карты {thr:.4f}; σ сглаживания {SMOOTH_M:g} м; нижний уровень прохода "
        f"мостика = {Q_LOW:.0%} карты подделки Б листа")

    def chain(site, sc, sB, v):
        low = float(np.quantile(sB[v], Q_LOW))
        L, polys, st = lines_clean(sc, v, thr, low, log, f"{site} снимок")
        LB, _pB, stB = lines_clean(sB, v, thr, low, log, f"{site} подделка Б")
        r = {"нижний уровень": low, "снимок": st, "подделка Б": stB,
             "доля шума до сшивки": stB["км до сшивки"] / max(st["км до сшивки"], 1e-9),
             "доля шума после сшивки": stB["км"] / max(st["км"], 1e-9), "лист обучения": site != "utsjoki"}
        log(f"  {site}: нижний уровень {low:.4f}; линий {st['км']:.1f} км (Б {stB['км']:.1f}); доля шума до сшивки "
            f"{r['доля шума до сшивки']:.3f}, после {r['доля шума после сшивки']:.3f}")
        return L, polys, r
    train_sites = S1.SITES[:4]
    out = {}
    for i, site in enumerate(train_sites):
        img, dem, raw, valid, _m = A1._sheet(site)
        sc = A1.predict(net, A2.feature_stack2(img, dem, raw, valid))
        sB = A1.predict(net, A2.feature_stack2(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i), valid))
        L, polys, res[site] = chain(site, sc, sB, valid)
        if site == "anabar":
            out["anabar"] = (sc, L, polys)
    iu = S1.SITES.index("utsjoki")
    img, dem, raw, valid, _m = A1._sheet("utsjoki")
    su = A1.predict(net, A2.feature_stack2(img, dem, raw, valid))
    sBu = A1.predict(net, A2.feature_stack2(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + iu), valid))
    Lu, polys_u, res["utsjoki"] = chain("utsjoki", su, sBu, valid)
    old_sig = S.IMG_SIG_M
    S.IMG_SIG_M = A2.IMG_SIG_ANZ2             # план (а) с ширинами ANZ2, только на время построения
    try:
        plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(S1.SEED_PLAN_UTS), S1.N_PROPOSALS)
        i2, d2, a2_, truth = S.apply_plan(img, dem, raw, plan, robust_sigma(img[valid].astype("float64")),
                                          S.relief_sigma(dem, valid), RES_IN, valid=valid)
    finally:
        S.IMG_SIG_M = old_sig
    a2_[~np.isfinite(raw)] = np.nan
    si = A1.predict(net, A2.feature_stack2(i2, d2, a2_, valid))
    Li, _p, _s = lines_clean(si, valid, thr, res["utsjoki"]["нижний уровень"], log, "utsjoki + врезки")
    cells, dirs = A1.line_dirs(Li)
    cov = S1.recall_of_insertions(cells, dirs, truth, plan, CONF_TOL_M, TAU)
    amps = np.array([p["amp"] for p in plan])
    thin = np.array([A2._is_thin(p) for p in plan])
    rng_s = np.random.default_rng(A1.SEED_RANDOM)
    shifts = [(int(rng_s.integers(valid.shape[0])), int(rng_s.integers(valid.shape[1]))) for _ in range(A1.N_SHIFTS)]
    covs_rnd = [S1.recall_of_insertions((cells + np.array(sh)) % np.array(valid.shape), dirs, truth, plan, CONF_TOL_M, TAU)
                for sh in shifts]
    rows = []
    for grp, gm in (("узкие 20 м", thin), ("прежние ширины", ~thin)):
        for lo, hi in zip(S1.AMP_BINS[:-1], S1.AMP_BINS[1:]):
            sel = gm & (amps >= lo) & (amps < hi) & np.isfinite(cov)
            rate = float(np.mean(cov[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
            rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in covs_rnd])) if sel.any() else float("nan")
            rows.append({"группа": grp, "корзина": f"{lo:g}–{hi:g}σ", "врезок": int(sel.sum()), "найдено": rate,
                         "случайный сдвиг": rnd})
            log(f"  (а) {grp} {lo:g}–{hi:g}σ: врезок {int(sel.sum())}, найдено {rate:.3f}; сдвиги {rnd:.3f}")
    res["(а)"] = rows
    fb_ = sum(res[k]["подделка Б"]["км"] for k in train_sites)
    fa_ = sum(res[k]["снимок"]["км"] for k in train_sites)
    log(f"=== сводно по листам обучения (оптимистично, не контроль): линии {fa_:.0f} км, Б {fb_:.0f} км, доля шума "
        f"{fb_ / max(fa_, 1e-9):.3f}; описательно, оценка снизу, не точность")
    for name, path, key in (("ANZ1", A1.OUT_DIR / "anz1_итог.json", None), ("ANZ2", A2.OUT_DIR / "anz2_итог.json", None)):
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
            for k in ("utsjoki", "anabar"):
                log(f"  рядом {name} {k}: линий {j[k]['км линий']:.1f} км, доля шума {j[k]['доля шума после сшивки']:.3f}")
        except (OSError, KeyError) as e:
            log(f"  {name} рядом не прочитан: {e}")
    A1.OUT_DIR, A1.TEACHER, A1.NAME = OUT_DIR, TEACHER, "ANZ2b"
    A1._save("utsjoki", su, Lu, polys_u, res["utsjoki"]["доля шума после сшивки"], S1.load_site("utsjoki")[4], log)
    sa, La, pa = out["anabar"]
    A1._save("anabar", sa, La, pa, res["anabar"]["доля шума после сшивки"], S1.load_site("anabar")[4], log)
    res["код"] = {"anz2b": __import__("experiments.openlin_len1", fromlist=["sha"]).sha(__file__)}
    (OUT_DIR / "anz2b_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz2b_итог.json'}; время {time.time() - t0:.0f} с")


def main():
    from experiments.openlin_fam import log_to
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--run" in sys.argv:
        log = log_to(OUT_DIR / "_anz2b.log")
        log("ANZ2b; audit/ANZ2B_DECLARATION.md")
        run(log)
    elif "--windows" in sys.argv:
        A1.OUT_DIR, A1.NAME = OUT_DIR, "ANZ2b"
        A1.windows(log_to(OUT_DIR / "_anz2b_windows.log"))


if __name__ == "__main__":
    main()
