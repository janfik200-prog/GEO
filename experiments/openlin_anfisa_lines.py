"""LINB1: линии из карты Анфисы (нечёткая аддитивная модель FAM3), вариант Б.

Объявление: audit/LINB1_DECLARATION.md (11.11.2026, до кода). Площадка — Утсйоки
(настройка, описательно, ``LINB1_N_TESTS`` = 0).

Шаги (сетка 100 м): карта P (среднее логитов трёх зёрен замороженной FAM3) →
сила D и направление θ (среднее P вдоль отрезка 1,1 км по 12 направлениям) →
кандидаты D > 0 → подавление немаксимумов поперёк θ → скелет, ветви → сшивка
(разрыв ≤ 900 м, сдвиг ≤ 127 м, ≤ 20°) → отброс < 200 м → растр 20 м.
Сравнение с трактом — главная мера S ``openlin_ref_match`` (семейство «весь лист»).

Запуск: python -m experiments.openlin_anfisa_lines --site utsjoki
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from experiments.openlin_fam import (CELL_M, FAM_NAME, LINE_LEN_M, N_DIR, OUT, RES_IN,
                                     all_windows, features, log_to, quantile_norm, win_mask)
from experiments.openlin_zone_lines import DIR_TOL_DEG
from experiments.openlin_zone_lines_v3 import LINK_MIN_LEN_M, TWIN_OFFSET_M, V3_GAP_M

M = Path("outputs/metrics")
TAG = "_анфиса1"
LINB1_N_TESTS = 0


def score_map(site, log):
    import cloudpickle
    F, valid, meta = features(site, log)
    wins, n = all_windows(valid)                      # поправка 1.3: как site_table
    Z = quantile_norm(F, win_mask(valid.shape, wins, n) & valid)
    preds = cloudpickle.load(open(OUT / "frozen_fam3.pkl", "rb"))["models"][FAM_NAME]
    P = np.full(valid.shape, np.nan)
    P[valid] = np.mean([p(Z[valid]) for p in preds], 0)
    return P, valid, meta


def directional(P, valid):
    """Среднее P вдоль отрезка LINE_LEN_M по N_DIR направлениям (ядра — как у FAM)."""
    L = int(round(LINE_LEN_M / CELL_M))
    half = L // 2
    Pf = np.where(valid, P, np.nanpercentile(P, 1))   # вне листа — низкий фон
    resp = []
    for i in range(N_DIR):
        th = np.pi * i / N_DIR
        k = np.zeros((L, L))
        for t in np.linspace(-half, half, 4 * L):
            k[int(round(half - t * np.cos(th))), int(round(half + t * np.sin(th)))] = 1
        resp.append(ndi.convolve(Pf, k / k.sum(), mode="nearest"))
    resp = np.stack(resp)
    idx = resp.argmax(0)
    return resp.max(0), np.pi * idx / N_DIR


def nms(D, th):
    """Ячейка остаётся, если D не меньше D двух соседей по нормали к θ."""
    H, W = D.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    # направление ядра в (строка, столбец): (-cos θ, sin θ); нормаль: (sin θ, cos θ)
    nr, nc = np.sin(th), np.cos(th)
    a = ndi.map_coordinates(D, [rr + nr, cc + nc], order=1, mode="nearest")
    b = ndi.map_coordinates(D, [rr - nr, cc - nc], order=1, mode="nearest")
    return (D >= a) & (D >= b)


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def branches(skel):
    """Ветви скелета: узлы (> 2 соседей) убираются, остаток — ломаные по ячейкам."""
    n = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    n = np.where(skel, n, 0)
    body = skel & (n <= 2)
    lab, k = ndi.label(body, np.ones((3, 3)))
    out = []
    objs = ndi.find_objects(lab)
    for j, sl in enumerate(objs, 1):
        if sl is None:
            continue
        pts = set(zip(*np.nonzero(lab[sl] == j)))
        pts = {(r + sl[0].start, c + sl[1].start) for r, c in pts}
        deg = {p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in _NB) for p in pts}
        start = next((p for p, d in deg.items() if d <= 1), next(iter(pts)))
        path, seen, cur = [start], {start}, start
        while True:
            nxt = [(cur[0] + dr, cur[1] + dc) for dr, dc in _NB
                   if (cur[0] + dr, cur[1] + dc) in pts and (cur[0] + dr, cur[1] + dc) not in seen]
            if not nxt:
                break
            # сначала 4-соседи, чтобы не перескакивать
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nxt[0]
            seen.add(cur)
            path.append(cur)
        out.append(np.array(path, float))
    return out


def length_m(pl):
    return float(np.hypot(*np.diff(pl, axis=0).T).sum() * CELL_M) if len(pl) > 1 else 0.0


def _end(pl, head):
    """Точка конца и наружное единичное направление (по 3 ячейкам от конца)."""
    k = min(3, len(pl) - 1)
    if head:
        p, q = pl[0], pl[k]
    else:
        p, q = pl[-1], pl[-1 - k]
    v = p - q
    nv = np.hypot(*v)
    return p, (v / nv if nv else v)


def stitch(lines, log):
    """Жадная сшивка концов: разрыв ≤ V3_GAP_M, сдвиг ≤ TWIN_OFFSET_M, угол ≤ DIR_TOL_DEG."""
    gap, off = V3_GAP_M / CELL_M, TWIN_OFFSET_M / CELL_M
    cos_tol = np.cos(np.radians(DIR_TOL_DEG))
    ends = []
    for i, pl in enumerate(lines):
        if len(pl) < 2:
            continue
        for head in (True, False):
            p, d = _end(pl, head)
            ends.append((i, head, p, d))
    pts = np.array([e[2] for e in ends])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    cand = []
    for a, b in tree.query_pairs(gap):
        ia, ha, pa, da = ends[a]
        ib, hb, pb, db = ends[b]
        if ia == ib:
            continue
        v = pb - pa
        g = np.hypot(*v)
        if np.dot(da, -db) < cos_tol:                  # концы смотрят навстречу
            continue
        if g > 0:
            if np.dot(v, da) <= 0 or np.dot(-v, db) <= 0:
                continue
            lat = max(abs(da[0] * v[1] - da[1] * v[0]), abs(db[0] * v[1] - db[1] * v[0]))
            if lat > off:
                continue
        cand.append((g, a, b))
    cand.sort()
    used, link = set(), {}
    for g, a, b in cand:
        if a in used or b in used:
            continue
        used |= {a, b}
        link[a], link[b] = b, a
    # обход цепочек
    other = {}
    for e_idx, (i, head, _p, _d) in enumerate(ends):
        other.setdefault(i, {})[head] = e_idx
    done, out = set(), []
    for i in range(len(lines)):
        if i in done or len(lines[i]) < 2:
            continue
        # найти начало цепочки: идём от головы i назад, пока есть связь
        cur, head = i, True
        guard = 0
        while other[cur][head] in link and guard < len(lines):
            j = ends[link[other[cur][head]]]
            cur, head = j[0], not j[1]
            guard += 1
            if cur == i:
                break
        # теперь cur, и свободный конец — head; собираем вперёд
        chain, start = [], cur
        h = head
        while True:
            done.add(cur)
            pl = lines[cur] if h else lines[cur][::-1]
            chain.append(pl)
            e = other[cur][not h]
            if e not in link:
                break
            j = ends[link[e]]
            if j[0] in done:
                break
            cur, h = j[0], j[1]
        out.append(np.concatenate(chain))
    log(f"  сшивка: пар-кандидатов {len(cand)}, сшито {len(link) // 2}; линий {len(lines)} -> {len(out)}")
    return out


def rasterize(lines, shape20, k):
    """Метки линий на сетке 20 м (0 — пусто; на пересечении — последняя линия)."""
    from skimage.draw import line as dline
    m = np.zeros(shape20, np.int32)
    for j, pl in enumerate(lines, 1):
        q = np.round(pl * k + (k - 1) / 2).astype(int)
        for (r0, c0), (r1, c1) in zip(q[:-1], q[1:]):
            rr, cc = dline(r0, c0, r1, c1)
            ok = (rr >= 0) & (rr < shape20[0]) & (cc >= 0) & (cc < shape20[1])
            m[rr[ok], cc[ok]] = j
    return m


def add_lines(ids, base, res):
    """Правило add_new, но по линиям (метки), а не по компонентам (поправка 1.5)."""
    from experiments.openlin_union_recall import ADD_FRAC, ADD_TOL_M
    d = ndi.distance_transform_edt(~base, sampling=(res, res))
    n = int(ids.max())
    near = np.bincount(ids.ravel(), weights=(d <= ADD_TOL_M).ravel(), minlength=n + 1)
    size = np.bincount(ids.ravel(), minlength=n + 1)
    keep = near / np.maximum(size, 1) < ADD_FRAC
    keep[0] = False
    return keep[ids]


def build(site, log):
    P, valid, meta = score_map(site, log)
    log(f"карта P: ячеек {int(valid.sum())}, P > 0 у {100 * np.nanmean(P > 0):.1f} %")
    D, th = directional(P, valid)
    cand = (D > 0) & valid
    keep = cand & nms(D, th)
    log(f"  D > 0: {100 * cand[valid].mean():.1f} % ячеек; после NMS {int(keep.sum())}")
    from skimage.morphology import skeletonize
    sk = skeletonize(keep)
    br = branches(sk)
    log(f"  скелет {int(sk.sum())} ячеек, ветвей {len(br)}")
    lines = stitch(br, log)
    lines = [pl for pl in lines if length_m(pl) >= LINK_MIN_LEN_M]
    L = np.array([length_m(pl) for pl in lines]) / 1000
    log(f"  линий >= {LINK_MIN_LEN_M:.0f} м: {len(lines)}, {L.sum():.0f} км; длина медиана "
        f"{np.median(L):.2f} км, 90-й проц. {np.percentile(L, 90):.2f}, макс {L.max():.1f}")
    # азимуты отрезков (контроль «зубцов» по 15°)
    az = np.concatenate([np.degrees(np.arctan2(np.diff(pl[:, 1]), -np.diff(pl[:, 0]))) % 180
                         for pl in lines if len(pl) > 1])
    h, _ = np.histogram(az, bins=12, range=(0, 180))
    log("  азимуты шагов ломаных (по 15°): " + " ".join(str(x) for x in h))
    return lines, P, D, th, valid, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="utsjoki")
    a = ap.parse_args()
    if a.site != "utsjoki":
        raise SystemExit("LINB1 — только площадка настройки")
    log = log_to(OUT / f"_linb1_{a.site}.log")
    log("LINB1; объявление audit/LINB1_DECLARATION.md")
    lines, P, D, th, valid, meta = build(a.site, log)
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    o = np.load(M / f"openlin_zone_lines_{a.site}_v3_настройка1.npz")
    shape20 = o["C_all"].shape
    k = int(round(CELL_M / RES_IN))
    valid20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    ids = np.where(valid20, rasterize(lines, shape20, k), 0)     # поправка 1.6
    anf = ids > 0
    if not anf.any():
        log("ПУСТО: линий Анфисы нет — S не считается, вариант А по решению Артура")
        return
    lab, nlab = ndi.label(anf, np.ones((3, 3)))
    big = np.bincount(lab.ravel())[1:].max() * float(w["res_m"]) / 1000
    log(f"растр: доля листа {100 * anf[valid20].mean():.2f} %, компонент {nlab}, "
        f"наибольшая {big:.0f} км")
    flat = np.concatenate(lines)
    off = np.cumsum([0] + [len(pl) for pl in lines])
    np.savez_compressed(M / f"openlin_zone_lines_{a.site}_v3{TAG}.npz", C_all=anf, C=anf)
    np.savez_compressed(M / f"openlin_zone_lines_{a.site}_v3{TAG}_линии.npz",
                        rc100=flat, offsets=off, x0=meta["x0"], y_top=meta["y_top"],
                        cell_m=CELL_M, epsg=meta["epsg"])
    np.savez_compressed(OUT / f"linb1_{a.site}_карта.npz", P=P, D=D, th=th, valid=valid)
    log(f"растр 20 м: {int(anf.sum())} ячеек")

    # ---- мера S (openlin_ref_match без изменений), семейство «весь лист»
    from experiments import openlin_ref_match as rm
    from experiments.openlin_union_recall import add_new
    r = np.load(M / f"openlin_zone_lines_{a.site}_v3_рельеф1.npz")
    res = float(w["res_m"])
    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(valid20, ang, np.nan)
    pos = rm.positions(ang, fid, valid20, rm.shifts(shape20))
    om, rmk = o["C_all"], r["C_all"][:shape20[0], :shape20[1]]
    base = om | add_new(rmk, om, res)
    sets = {"снимок": om, "рельеф": rmk, "снимок + рельеф": base, "Анфиса": anf,
            "Анфиса + снимок + рельеф": base | add_lines(ids, base, res)}
    ev = {s: rm.evaluate(m, pos, res) for s, m in sets.items()}
    K = int(np.floor(min(l_.sum() for l_, _m in ev.values()) / rm.LEN_STEP_KM))
    log(f"семейство «весь лист»: K = {K} (L до {K * rm.LEN_STEP_KM:.0f} км)")
    if big > K * rm.LEN_STEP_KM:
        log(f"  ВНИМАНИЕ: наибольшая компонента Анфисы {big:.0f} км > K·10 км — в кривую не входит "
            "(свойство меры, поправка 1.4)")
    rows = []
    for s, (lens, mins) in ev.items():
        cs = rm.curve_score(lens, mins, K)
        rd, rn = rm.recall_dir(mins)
        ll = rm.long_lines(mins, pos, res)
        log(f"  {s:26s} {lens.sum():6.0f} км | S = {cs['S']:.3f} | при L = "
            f"{K * rm.LEN_STEP_KM:.0f} км {100 * cs['R'][-1]:.1f} % (нуль {100 * cs['N'][-1]:.1f}) "
            f"| весь набор {100 * rd:.1f} % (нуль {100 * rn:.1f}) | длинных эталона найдено "
            f"{100 * ll['obs']:.0f} % из {ll['n_long']} (нуль {100 * ll['null']:.0f} %)")
        rows.append(dict(набор=s, км=lens.sum(), K=K, S=cs["S"], R_K=cs["R"][-1],
                         N_K=cs["N"][-1], полнота_напр=rd, нуль_напр=rn,
                         найдено_длинных=ll["obs"], нуль_длинных=ll["null"]))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / f"linb1_{a.site}.csv", index=False, encoding="utf-8-sig")
    best = max(r_["S"] for r_ in rows if r_["набор"] in ("снимок", "рельеф", "снимок + рельеф"))
    sa = next(r_["S"] for r_ in rows if r_["набор"] == "Анфиса")
    log(f"ИТОГ (описательно): S Анфисы {sa:.3f} против лучшего набора тракта {best:.3f} -> "
        + ("больше" if sa > best else "не больше — по решению Артура вариант А"))


# ---- LINB2 (audit/LINB2_DECLARATION.md): отбор линий LINB1 по среднему P до бюджета длины
def line_mean_p(pl, P, bridges=True):
    """Среднее P по ячейкам ломаной на 100 м (bridges=False — только ячейки ветвей)."""
    from skimage.draw import line as dline
    q = np.round(pl).astype(int)
    if not bridges:
        v = P[q[:, 0], q[:, 1]]
        return float(np.nanmean(v)) if np.isfinite(v).any() else -np.inf
    rr, cc = [q[0, 0]], [q[0, 1]]
    for (r0, c0), (r1, c1) in zip(q[:-1], q[1:]):
        r_, c_ = dline(r0, c0, r1, c1)
        rr += list(r_[1:]); cc += list(c_[1:])
    v = P[np.array(rr), np.array(cc)]
    return float(np.nanmean(v)) if np.isfinite(v).any() else -np.inf


def main_linb2(site="utsjoki"):
    from experiments import openlin_ref_match as rm
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    from experiments.openlin_union_recall import add_new
    log = log_to(OUT / f"_linb2_{site}.log")
    log("LINB2; объявление audit/LINB2_DECLARATION.md (+ поправки, если есть)")
    V = np.load(M / f"openlin_zone_lines_{site}_v3{TAG}_линии.npz")
    flat, off = V["rc100"], V["offsets"]
    lines = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    P = np.load(OUT / f"linb1_{site}_карта.npz")["P"]
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res = float(w["res_m"])
    o = np.load(M / f"openlin_zone_lines_{site}_v3_настройка1.npz")
    r = np.load(M / f"openlin_zone_lines_{site}_v3_рельеф1.npz")
    shape20 = o["C_all"].shape
    valid20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    om, rmk = o["C_all"], r["C_all"][:shape20[0], :shape20[1]]
    k = int(round(CELL_M / RES_IN))
    ids_all = np.where(valid20, rasterize(lines, shape20, k), 0)
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]   # ячейки 20 м на линию
    mp = np.array([line_mean_p(pl, P) for pl in lines])
    mp_nb = np.array([line_mean_p(pl, P, bridges=False) for pl in lines])
    lenm = np.array([length_m(pl) for pl in lines])
    # поправка 1.4: по убыванию P, затем по убыванию длины, затем по номеру
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    order_nb = np.lexsort((np.arange(len(lines)), -lenm, -mp_nb))
    log(f"линий LINB1 {len(lines)}, {cells.sum() * res / 1000:.0f} км (растр); среднее P: "
        f"медиана {np.median(mp):.3f}")

    def pick(budget_km, order=order, mp=mp):
        cum = np.cumsum(cells[order]) * res / 1000
        n = int(np.searchsorted(cum, budget_km, side="right"))
        keep = np.zeros(len(lines) + 1, bool)
        keep[order[:n] + 1] = True
        log(f"  бюджет {budget_km:.0f} км: линий {n}, порог среднего P {mp[order[n - 1]]:.3f}")
        return np.where(keep[ids_all], ids_all, 0)
    budgets = {"рельеф": rmk.sum() * res / 1000, "снимок": om.sum() * res / 1000}
    idsR, idsU = pick(budgets["рельеф"]), pick(budgets["снимок"])
    log("  контроль без мостиков:")
    idsN = pick(budgets["рельеф"], order_nb, mp_nb)
    base = om | add_new(rmk, om, res)
    sets = {"снимок": om, "рельеф": rmk, "снимок + рельеф": base,
            "Анфиса, бюджет RELU1": idsR > 0, "Анфиса, бюджет UTS1": idsU > 0,
            "Анфиса LINB1 (все)": ids_all > 0,
            "Анфиса, бюджет RELU1, без мостиков": idsN > 0,
            "Анфиса (бюджет RELU1) + снимок + рельеф": base | add_lines(idsR, base, res)}
    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(valid20, ang, np.nan)
    pos = rm.positions(ang, fid, valid20, rm.shifts(shape20))
    ev = {s: rm.evaluate(m, pos, res) for s, m in sets.items()}
    rows = []
    for fam, names in (("K по самому короткому", list(sets)),
                       ("K по рельефу (для сведения)", ["рельеф", "снимок + рельеф",
                                                        "Анфиса, бюджет RELU1",
                                                        "Анфиса (бюджет RELU1) + снимок + рельеф"])):
        K = int(np.floor(min(ev[s][0].sum() for s in names) / rm.LEN_STEP_KM))
        log(f"=== {fam}: K = {K} (L до {K * rm.LEN_STEP_KM:.0f} км)")
        for s in names:
            lens, mins = ev[s]
            cs = rm.curve_score(lens, mins, K)
            rd, rn = rm.recall_dir(mins)
            ll = rm.long_lines(mins, pos, res)
            log(f"  {s:42s} {lens.sum():6.0f} км | S = {cs['S']:.3f} | при L = "
                f"{K * rm.LEN_STEP_KM:.0f} км {100 * cs['R'][-1]:.1f} % (нуль {100 * cs['N'][-1]:.1f}) "
                f"| весь набор {100 * rd:.1f} % (нуль {100 * rn:.1f}) | длинных найдено "
                f"{100 * ll['obs']:.0f} % (нуль {100 * ll['null']:.0f} %)")
            rows.append(dict(семейство=fam, набор=s, км=lens.sum(), K=K, S=cs["S"],
                             полнота_напр=rd, нуль_напр=rn))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / f"linb2_{site}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(M / f"openlin_zone_lines_{site}_v3_анфиса2.npz", C_all=idsR > 0, C=idsR > 0,
                        ids=idsR)
    sR = next(r_["S"] for r_ in rows if r_["семейство"].startswith("K по самому") and r_["набор"] == "рельеф")
    sA = next(r_["S"] for r_ in rows if r_["семейство"].startswith("K по самому")
              and r_["набор"] == "Анфиса, бюджет RELU1")
    log(f"ИТОГ (описательно): S «Анфиса, бюджет RELU1» {sA:.3f} против рельефа {sR:.3f} -> "
        + ("больше — согласуется с разбором (вес: правило не сломано; нужен нетронутый лист)"
           if sA > sR else "не больше — вариант А"))


if __name__ == "__main__":
    import sys
    main_linb2() if "--linb2" in sys.argv else main()
