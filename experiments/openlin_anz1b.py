"""ANZ1b: линии по карте ANZ1 заново — гистерезис, сшивка до отсева, отсев по длине связной линии.

Объявление audit/ANZ1B_DECLARATION.md. Сеть ANZ1 (outputs/fam/anz1/anz1_net.pt) не переобучается; карты пересчитываются
той же функцией с теми же зёрнами; порог карты 0,9854 — из ANZ1.
    python -m experiments.openlin_anz1b --run
    python -m experiments.openlin_anz1b --windows
ANZ1B_N_TESTS = 0.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from experiments import openlin_anz1 as A1
from experiments import openlin_syn1 as S1
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_iso import Q_LOW
from experiments.openlin_zone_lines import DIR_TOL_DEG
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, LINK_MIN_LEN_M, TWIN_OFFSET_M, V3_GAP_M

ANZ1B_N_TESTS = 0
TEACHER = "Анфиса-карта (ANZA, сеть ANZ1), линии LIRA с гистерезисом (ANZ1b); учитель — искусственные линии"
OUT_DIR = OUT / "anz1b"
COVER_LEVELS = (0.5, 0.8)          # печать покрытия (описательно, объявление «Мера»)
_EIGHT = np.ones((3, 3), bool)


def _bridges(pieces, score, low, tips=None):
    """Мостики LIRA между встречными концами ветвей — геометрия и правило ANZ1 (разрыв ≤ 900 м, сдвиг ≤ 127 м,
    угол ≤ 20°, слабое звено ≥ low, каждый конец — один мостик, жадно по убыванию слабого звена)."""
    from scipy.spatial import cKDTree

    from experiments.openlin_linb5 import _end
    ends = []
    for i, pl in enumerate(pieces):
        if len(pl) < 2:
            continue
        for head in (True, False):
            p, d = _end(pl, head)
            q = np.round(pl[0] if head else pl[-1]).astype(int)
            if tips is not None and not tips[q[0], q[1]]:
                continue          # конец у развилки — не конец линии
            ends.append((i, p, d))
    gap, off = V3_GAP_M / RES_IN, TWIN_OFFSET_M / RES_IN
    cos_tol = math.cos(math.radians(DIR_TOL_DEG))
    cand = []
    if ends:
        tree = cKDTree(np.array([e[1] for e in ends]))
        for a, b in tree.query_pairs(gap):
            ia, pa, da = ends[a]
            ib, pb, db = ends[b]
            if ia == ib or np.dot(da, -db) < cos_tol:
                continue
            v = pb - pa
            if np.hypot(*v) > 0:
                if np.dot(v, da) <= 0 or np.dot(-v, db) <= 0:
                    continue
                if max(abs(da[0] * v[1] - da[1] * v[0]), abs(db[0] * v[1] - db[1] * v[0])) > off:
                    continue
            cand.append((a, b))
    H, W = score.shape
    m = int(off) + 2
    found = []
    for a, b in cand:
        pa = tuple(np.round(ends[a][1]).astype(int)); pb = tuple(np.round(ends[b][1]).astype(int))
        box = (max(min(pa[0], pb[0]) - m, 0), min(max(pa[0], pb[0]) + m + 1, H),
               max(min(pa[1], pb[1]) - m, 0), min(max(pa[1], pb[1]) + m + 1, W))
        s, path = A1.widest_path(score, pa, pb, box)
        if path and s >= low:
            found.append((s, a, b, path))
    found.sort(key=lambda t: -t[0])
    used, paths = set(), []
    for s, a, b, path in found:
        if a in used or b in used:
            continue
        used |= {a, b}
        paths.append(np.array(path))
    return paths, len(cand)


def _keep_long(mask):
    """Ветви скелета маски, связная часть которых (8 соседей) имеет длину ≥ 200 м. → (маска, полилинии)."""
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches
    from experiments.openlin_linb5 import length_px_m
    sk = skeletonize(mask)
    lab, _n = label(sk, structure=_EIGHT)
    polys = [pl for pl in branches(sk) if len(pl) >= 1]
    comp_len = {}
    comp_of = []
    for pl in polys:
        q = np.round(pl).astype(int)
        c = int(lab[q[0, 0], q[0, 1]])
        comp_of.append(c)
        comp_len[c] = comp_len.get(c, 0.0) + length_px_m(pl)
    keep = [pl for pl, c in zip(polys, comp_of) if comp_len[c] >= LINK_MIN_LEN_M]
    good = [c for c, ln in comp_len.items() if ln >= LINK_MIN_LEN_M]
    out = sk & np.isin(lab, good)        # скелет целиком, с клетками развилок (в ветви они не входят)
    return out, [pl for pl in keep if len(pl) >= 2]


def lines_hyst(score, valid, thr, low, log, lbl):
    """Гистерезис (верх thr, низ low) → скелет области → мостики между концами всех ветвей → отсев по связной линии.
    Возвращает (маска линий, маска до сшивки, число мостиков, км мостиков, полилинии)."""
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches
    score = np.where(valid, score, 0.0).astype("float32")
    lab, _n = label((score >= low) & valid, structure=_EIGHT)
    seeds = np.unique(lab[(score >= thr) & valid])
    region = np.isin(lab, seeds[seeds > 0])
    sk = _prune(skeletonize(region), region)
    pieces = [pl for pl in branches(sk) if len(pl) >= 2]
    paths, n_cand = _bridges(pieces, score, low, tips=_degree(sk) == 1)   # только тупиковые концы (поправка 2)
    mask = sk.copy()
    br_cells = 0
    for pp in paths:
        mask[pp[:, 0], pp[:, 1]] = True
        br_cells += len(pp)
    base, _pb = _keep_long(sk)
    final, polys = _keep_long(mask)
    # поправка 1 п. 3–4: перколяция, отростки, форма пятна
    a_reg = region.sum() * RES_IN ** 2 / 1e6
    a_top = ((score >= thr) & valid).sum() * RES_IN ** 2 / 1e6
    n_comp = len(seeds[seeds > 0])
    deg = _degree(final)
    bridged = np.zeros(final.shape, bool)
    for pp in paths:
        bridged[pp[:, 0], pp[:, 1]] = True
    spurs = [pl for pl in polys if _is_spur(pl, deg, bridged)]
    from experiments.openlin_linb5 import length_px_m
    st = {"площадь области км²": a_reg, "доля листа": float(region.sum() / max(valid.sum(), 1)),
          "связных частей": n_comp, "область / клетки ≥ порога": a_reg / max(a_top, 1e-9),
          "отростков": len(spurs), "км отростков": sum(length_px_m(p) for p in spurs) / 1000,
          "скелет / площадь области (км/км²)": (sk.sum() * RES_IN / 1000) / max(a_reg, 1e-9)}
    log(f"    {lbl}: область гистерезиса {a_reg:.1f} км² ({st['доля листа']:.3f} листа; ×{st['область / клетки ≥ порога']:.2f} "
        f"к клеткам ≥ порога), связных частей {n_comp}, ветвей {len(pieces)}, пар-кандидатов {n_cand}, мостиков "
        f"{len(paths)} ({br_cells * RES_IN / 1000:.1f} км); до сшивки {base.sum() * RES_IN / 1000:.0f} км, итог {len(polys)} "
        f"ветвей ({final.sum() * RES_IN / 1000:.0f} км); отростков {len(spurs)} ({st['км отростков']:.1f} км); скелет/площадь "
        f"{st['скелет / площадь области (км/км²)']:.2f} км/км²")
    lines_hyst.stats = st
    return final, base, len(paths), br_cells * RES_IN / 1000, polys


def _prune(sk, region):
    """Чистка срединной оси (поправка 2): отросток (тупик — у развилки) убирается, если его длина ≤ толщины области у
    развилки (2 × расстояние до края области): он целиком лежит в ширине ленты. Один проход, новых чисел нет."""
    from experiments.openlin_anfisa_lines import branches
    from experiments.openlin_linb5 import length_px_m
    dt = distance_transform_edt(region) * RES_IN
    deg = _degree(sk)
    out = sk.copy()
    for pl in branches(sk):
        q = np.round(pl).astype(int)
        d0, d1 = deg[q[0, 0], q[0, 1]], deg[q[-1, 0], q[-1, 1]]
        if not (min(d0, d1) == 1 and max(d0, d1) >= 2):
            continue
        j = q[-1] if d0 == 1 else q[0]
        if length_px_m(pl) <= 2 * dt[j[0], j[1]]:
            out[q[:, 0], q[:, 1]] = False
    return out


def _degree(mask):
    from scipy.ndimage import convolve
    return np.where(mask, convolve(mask.astype("int16"), np.ones((3, 3), "int16"), mode="constant") - 1, 0)


def _is_spur(pl, deg, bridged):
    """Отросток: ветвь, у которой один конец — тупик (1 сосед), другой — у развилки (≥ 2: клетки развилки в ветвь не
    входят), и в ней нет мостика."""
    q = np.round(pl).astype(int)
    d0, d1 = deg[q[0, 0], q[0, 1]], deg[q[-1, 0], q[-1, 1]]
    return min(d0, d1) == 1 and max(d0, d1) >= 2 and not bridged[q[:, 0], q[:, 1]].any()


def coverage(score, valid, L, thr):
    d = distance_transform_edt(~L) * RES_IN
    out = {}
    for t in COVER_LEVELS + (thr,):
        m = (score >= t) & valid
        out[f"≥{t:.4g}"] = float((d[m] <= CONF_TOL_M).mean()) if m.any() else float("nan")
    return out


def run(log):
    import time

    import torch
    t0 = time.time()
    from src.detection_limit import robust_sigma
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a1 = json.loads((A1.OUT_DIR / "anz1_итог.json").read_text(encoding="utf-8"))
    thr = float(a1["порог"]["значение"])
    res = {"объявление": "audit/ANZ1B_DECLARATION.md", "учитель": TEACHER, "ANZ1B_N_TESTS": ANZ1B_N_TESTS,
           "порог карты ANZ1": thr}
    dev = A1.device()
    net = A1.make_net(10).to(dev)
    net.load_state_dict(torch.load(A1.OUT_DIR / "anz1_net.pt", map_location=dev))
    log(f"  сеть ANZ1 загружена ({dev}); порог карты из ANZ1 {thr:.4f}; нижний уровень = {Q_LOW:.0%} карты подделки Б листа")
    train_sites = S1.SITES[:4]
    maps = {}
    for i, site in enumerate(train_sites):
        img, dem, raw, valid, _m = A1._sheet(site)
        sc = A1.predict(net, A1.feature_stack(img, dem, raw, valid))
        sB = A1.predict(net, A1.feature_stack(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i), valid))
        maps[site] = (sc, sB, valid)
    allr = np.concatenate([maps[k][0][maps[k][2]] for k in maps])
    thr_re = float(np.quantile(allr, A1.Q_TOP))
    log(f"  сверка: порог по пересчитанным картам {thr_re:.4f} (ANZ1 {thr:.4f}); используется ANZ1")

    def chain(site, sc, sB, v):
        low = float(np.quantile(sB[v], Q_LOW))
        L, base, nb, kmb, polys = lines_hyst(sc, v, thr, low, log, f"{site} снимок")
        st_r = lines_hyst.stats
        LB, baseB, nbB, kmbB, _p = lines_hyst(sB, v, thr, low, log, f"{site} подделка Б")
        st_b = lines_hyst.stats
        km = lambda m: m.sum() * RES_IN / 1000
        # старая цепочка ANZ1 рядом (поправка 1 п. 5)
        Lo, *_o = A1.lines_from_map(sc, v, thr, low, log, f"{site} снимок, цепочка ANZ1")
        LoB, *_ob = A1.lines_from_map(sB, v, thr, low, log, f"{site} подделка Б, цепочка ANZ1")
        old = {"км линий": km(Lo), "км линий подделки Б": km(LoB), "доля шума": km(LoB) / max(km(Lo), 1e-9),
               "покрытие снимок": coverage(sc, v, Lo, thr), "покрытие подделка Б": coverage(sB, v, LoB, thr)}
        log(f"  {site}, цепочка ANZ1 рядом: {old['км линий']:.1f} км, Б {old['км линий подделки Б']:.1f} км, доля шума "
            f"{old['доля шума']:.3f}; покрытие " + ", ".join(f"{k} {x:.2f}" for k, x in old["покрытие снимок"].items()))
        r = {"нижний уровень": low, "км линий": km(L), "км линий подделки Б": km(LB), "км до сшивки": km(base),
             "км до сшивки подделки Б": km(baseB), "мостиков": nb, "км мостиков": kmb, "мостиков на подделке Б": nbB,
             "доля шума до сшивки": km(baseB) / max(km(base), 1e-9), "доля шума после сшивки": km(LB) / max(km(L), 1e-9),
             "покрытие снимок": coverage(sc, v, L, thr), "покрытие подделка Б": coverage(sB, v, LB, thr),
             "лист обучения": site != "utsjoki", "область снимок": st_r, "область подделка Б": st_b, "цепочка ANZ1": old}
        log(f"  {site}: нижний уровень {low:.4f}; линий {km(L):.1f} км (подделка Б {km(LB):.1f}); доля шума до сшивки "
            f"{r['доля шума до сшивки']:.3f}, после {r['доля шума после сшивки']:.3f}; покрытие карты линиями (60 м) снимок "
            + ", ".join(f"{k} {x:.2f}" for k, x in r["покрытие снимок"].items()) + "; Б "
            + ", ".join(f"{k} {x:.2f}" for k, x in r["покрытие подделка Б"].items()))
        return L, polys, r
    out = {}
    for k in train_sites:
        L, polys, res[k] = chain(k + (" (лист обучения)" if k != "anabar" else ""), *maps[k])
        out[k] = (maps[k][0], L, polys)
    iu = S1.SITES.index("utsjoki")
    img, dem, raw, valid, _m = A1._sheet("utsjoki")
    su = A1.predict(net, A1.feature_stack(img, dem, raw, valid))
    sBu = A1.predict(net, A1.feature_stack(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + iu), valid))
    Lu, polys_u, res["utsjoki"] = chain("utsjoki", su, sBu, valid)
    plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(S1.SEED_PLAN_UTS), S1.N_PROPOSALS)
    i2, d2, a2, truth = S.apply_plan(img, dem, raw, plan, robust_sigma(img[valid].astype("float64")),
                                     S.relief_sigma(dem, valid), RES_IN, valid=valid)
    a2[~np.isfinite(raw)] = np.nan
    si = A1.predict(net, A1.feature_stack(i2, d2, a2, valid))
    Li, *_r = lines_hyst(si, valid, thr, res["utsjoki"]["нижний уровень"], log, "utsjoki + врезки")
    cells, dirs = A1.line_dirs(Li)
    cov = S1.recall_of_insertions(cells, dirs, truth, plan, CONF_TOL_M, TAU)
    amps = np.array([p["amp"] for p in plan])
    rng_s = np.random.default_rng(A1.SEED_RANDOM)
    shifts = [(int(rng_s.integers(valid.shape[0])), int(rng_s.integers(valid.shape[1]))) for _ in range(A1.N_SHIFTS)]
    covs_rnd = [S1.recall_of_insertions((cells + np.array(sh)) % np.array(valid.shape), dirs, truth, plan, CONF_TOL_M, TAU)
                for sh in shifts]
    rows = []
    for lo, hi in zip(S1.AMP_BINS[:-1], S1.AMP_BINS[1:]):
        sel = (amps >= lo) & (amps < hi) & np.isfinite(cov)
        rate = float(np.mean(cov[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
        rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in covs_rnd])) if sel.any() else float("nan")
        rows.append({"корзина": f"{lo:g}–{hi:g}σ", "врезок": int(sel.sum()), "найдено": rate, "случайный сдвиг": rnd})
        log(f"  (а) {lo:g}–{hi:g}σ: врезок {int(sel.sum())}, найдено {rate:.3f}; сдвиги ({A1.N_SHIFTS}) {rnd:.3f}")
    res["(а)"] = rows
    tk = train_sites
    fb_, fa_ = sum(res[k]["км линий подделки Б"] for k in tk), sum(res[k]["км линий"] for k in tk)
    sb_, sa_ = sum(res[k]["км до сшивки подделки Б"] for k in tk), sum(res[k]["км до сшивки"] for k in tk)
    log(f"=== сводно по листам обучения (оптимистично, не контроль): линии {fa_:.0f} км, подделка Б {fb_:.0f} км; доля "
        f"шума до сшивки {sb_ / max(sa_, 1e-9):.3f}, после {fb_ / max(fa_, 1e-9):.3f}; описательно, оценка снизу, не точность")
    A1.OUT_DIR, A1.TEACHER, A1.NAME = OUT_DIR, TEACHER, "ANZ1b"
    A1._save("utsjoki", su, Lu, polys_u, res["utsjoki"]["доля шума после сшивки"], S1.load_site("utsjoki")[4], log)
    sa, La, pa = out["anabar"]
    A1._save("anabar", sa, La, pa, res["anabar"]["доля шума после сшивки"], S1.load_site("anabar")[4], log)
    res["код"] = {"anz1b": __import__("experiments.openlin_len1", fromlist=["sha"]).sha(__file__)}
    (OUT_DIR / "anz1b_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz1b_итог.json'}; время {time.time() - t0:.0f} с")


def main():
    from experiments.openlin_fam import log_to
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--run" in sys.argv:
        log = log_to(OUT_DIR / "_anz1b.log")
        log("ANZ1b; audit/ANZ1B_DECLARATION.md")
        run(log)
    elif "--windows" in sys.argv:
        A1.OUT_DIR, A1.NAME = OUT_DIR, "ANZ1b"
        A1.windows(log_to(OUT_DIR / "_anz1b_windows.log"))


if __name__ == "__main__":
    main()
