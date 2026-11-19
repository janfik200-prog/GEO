"""LEN1: порог длины для линий LINB2 по мере REAL3 (audit/LEN1_DECLARATION.md, поправки 1–2).

Утсйоки — выбор порога по правилу; Лапландия — разовая проверка числом. Свидетель — Landsat лето.
LEN1_N_TESTS = 0. Лидар не используется.
    python -m experiments.openlin_len1 --utsjoki
    python -m experiments.openlin_len1 --lapland
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from experiments.openlin_real1 import RES, geometry, lines_from_pts
from experiments.openlin_real2 import BLOCKS_PX, MAIN_PX, N_BOOT, boot, phase_surrogate, pi_hat, prep_witness, wins
from experiments.openlin_real3 import NEG_PX, admissible, t_stats

LEN1_N_TESTS = 0
N_GROUPS = 5                     # квинтили (выбор дизайна)
GOOD = 0.5                       # кажущаяся доля ≥ половины
E_U = 0.78                       # положительный контроль REAL3, Утсйоки (trace S2 -> Landsat, точка)
BUDGET_SHARE = 2858 / 6082       # 0,470: бюджет RELU1 / LINB1 на Утсйоки (_linb2_utsjoki.log)
N_MIN = 91                       # поправка 2 п. 4
R_SYN = 15
T_LAP = 2.510                    # t_{1−0,025/2}(14)
WIT = "Landsat лето"


def line_t(lines, Wp, wn, shape, swap=False):
    P, idx, T, N = geometry(lines)
    nl = len(lines)
    ok = admissible(P, idx, N, nl, shape)
    st = t_stats(P, idx, T, N, nl, Wp[wn][0], Wp[wn][1], shape, swap=swap)
    fin = ok & np.all(np.isfinite(st[:, :3]), 1)
    w = np.full(nl, np.nan)
    w[fin] = wins(st[fin], MAIN_PX)
    cen = np.array([x.mean(0) for x in lines])
    lens = np.array([len(x) for x in lines]) * RES          # м, как в REAL3
    return w, fin, cen, lens


def lo_hi(w, cen):
    c = [boot(w, cen, BLOCKS_PX[0], sd) for sd in (0, 1)]
    return min(x[0] for x in c), max(x[1] for x in c)


def diff_ci(w, cen, keep):
    """Нижний/верхний край π̂(оставленные) − π̂(отсечённые): общий блочный бутстреп 12 км, 2 зерна."""
    b = (cen[:, 0] // BLOCKS_PX[0]).astype(int) * 100000 + (cen[:, 1] // BLOCKS_PX[0]).astype(int)
    ub = np.unique(b)
    out = []
    for sd in (0, 1):
        rng = np.random.default_rng(sd)
        vals = []
        for _ in range(N_BOOT):
            m = np.concatenate([np.flatnonzero(b == k) for k in rng.choice(ub, len(ub), replace=True)])
            vals.append(pi_hat(w[m][keep[m]]) - pi_hat(w[m][~keep[m]]))
        out.append(np.percentile(vals, [2.5, 97.5]))
    return min(o[0] for o in out), max(o[1] for o in out)


def utsjoki(log):
    from experiments.openlin_real1 import M, load_sets, witnesses
    from experiments.openlin_fam import OUT
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    lines = load_sets(shape)["Анфиса LINB2"]
    W = witnesses(shape)
    Wp = {wn: prep_witness(W[wn], shape) for wn in (WIT, "S2 лето")}
    w, fin, cen, lens = line_t(lines, Wp, WIT, shape)
    wS, finS, _, _ = line_t(lines, Wp, "S2 лето", shape)
    L = lens[fin]
    edges = np.quantile(L, np.linspace(0, 1, N_GROUPS + 1))
    grp = np.clip(np.searchsorted(edges, lens, side="right") - 1, 0, N_GROUPS - 1)
    log(f"LINB2 Утсйоки: линий {len(lines)}, в счёт {int(fin.sum())}; e_U = {E_U}; группы по длине (м): "
        + ", ".join(f"{edges[g]:.0f}–{edges[g + 1]:.0f}" for g in range(N_GROUPS)))
    good = []
    for g in range(N_GROUPS):
        m = fin & (grp == g)
        ph = pi_hat(w[m])
        lo, hi = lo_hi(w[m], cen[m])
        mS = finS & (grp == g)
        good.append(ph / E_U >= GOOD)
        log(f"  группа {g + 1} ({edges[g]:.0f}–{edges[g + 1]:.0f} м, линий {int(m.sum())}): π̂ {100 * ph:+.0f} % "
            f"[{100 * lo:+.0f}…{100 * hi:+.0f}], кажущаяся доля π̂/e {ph / E_U:.2f} -> "
            f"{'годна' if good[-1] else 'не годна'}; S2 лето (справка) π̂ {100 * pi_hat(wS[mS]):+.0f} %")
    res = {"edges_m": edges.tolist(), "good": good}
    if good[0]:
        log("ИТОГ: годна самая короткая группа — ПОРОГА НЕТ, Лапландия не запускается")
        res["L_star_m"] = None
    elif not good[-1]:
        log("ИТОГ: не годна самая длинная группа — ПОРОГ НЕ НАЙДЕН, Лапландия не запускается")
        res["L_star_m"] = None
    else:
        j = min(g for g in range(N_GROUPS) if all(good[g:]))
        Ls = float(edges[j])
        keep = lens >= Ls
        res["L_star_m"] = Ls
        nl_all = len(lines)
        km_all = lens.sum() / 1000
        log(f"ИТОГ: ПОРОГ L* = {Ls:.0f} м (начало группы {j + 1}); отсекается линий "
            f"{int((~keep).sum())} из {nl_all} ({100 * (~keep).mean():.0f} %), км {lens[~keep].sum() / 1000:.0f} "
            f"из {km_all:.0f} ({100 * lens[~keep].sum() / 1000 / km_all:.0f} %)")
        for lab, mm in (("оставленные", fin & keep), ("отсечённые", fin & ~keep)):
            lo, hi = lo_hi(w[mm], cen[mm])
            log(f"  {lab}: линий {int(mm.sum())}, π̂ {100 * pi_hat(w[mm]):+.0f} % [{100 * lo:+.0f}…{100 * hi:+.0f}], "
                f"кажущаяся доля {pi_hat(w[mm]) / E_U:.2f}")
        dlo, dhi = diff_ci(w[fin], cen[fin], keep[fin])
        log(f"  разность оставленные − отсечённые: [{100 * dlo:+.0f}…{100 * dhi:+.0f}] п. п. (справка)")
    (OUT / "len1_utsjoki.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def lapland_lines(shape, log):
    """LINB2 на Лапландии: замороженный build + отбор по рангу P до 47,0 % км LINB1 (поправка 2 п. 5)."""
    from experiments import openlin_anfisa_lines as A
    from experiments.openlin_fam import OUT
    from experiments.openlin_transfer_check import COMP
    log(f"  sha256 до: openlin_anfisa_lines {sha(A.__file__)}, openlin_fam {sha('experiments/openlin_fam.py')}")
    lines, P, _D, _th, _valid, _meta = A.build("lapland", log)
    w = dict(np.load(COMP / "winter_lapland_20m_лист.npz"))
    res = float(w["res_m"])
    valid20 = np.isfinite(w["shade"][:shape[0], :shape[1]])
    k = int(round(A.CELL_M / A.RES_IN))
    ids_all = np.where(valid20, A.rasterize(lines, shape, k), 0)
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    mp = np.array([A.line_mean_p(pl, P) for pl in lines])
    lenm = np.array([A.length_m(pl) for pl in lines])
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    total = cells.sum() * res / 1000
    budget = BUDGET_SHARE * total
    cum = np.cumsum(cells[order]) * res / 1000
    n = int(np.searchsorted(cum, budget, side="right"))
    keep = np.zeros(len(lines) + 1, bool)
    keep[order[:n] + 1] = True
    ids = np.where(keep[ids_all], ids_all, 0)
    log(f"  LINB1 Лапландии: линий {len(lines)}, {total:.0f} км (растр); бюджет {100 * BUDGET_SHARE:.1f} % = "
        f"{budget:.0f} км: линий {n}, порог среднего P {mp[order[n - 1]]:.3f}")
    np.savez_compressed(OUT / "len1_lapland_linb2.npz", ids=ids)
    log(f"  sha256 после: openlin_anfisa_lines {sha(A.__file__)}, openlin_fam {sha('experiments/openlin_fam.py')}")
    rr, cc = np.nonzero(ids)
    return [x for x in lines_from_pts(np.stack([rr, cc], 1), ids[rr, cc]) if len(x) >= 2]


def lapland(log, lines_only=False):
    from experiments.openlin_fam import OUT
    from experiments.openlin_typesites_fetch import DEST
    res_u = json.loads((OUT / "len1_utsjoki.json").read_text(encoding="utf-8"))
    Ls = res_u["L_star_m"]
    if Ls is None:
        log("на Утсйоки порога нет / не найден — Лапландия не запускается (поправка 2 п. 2)")
        return
    shape = np.load("outputs/metrics/openlin_zone_lines_lapland_v3_перенос.npz")["C_all"].shape
    p = OUT / "len1_lapland_linb2.npz"
    if p.exists():
        ids = np.load(p)["ids"]
        rr, cc = np.nonzero(ids)
        lines = [x for x in lines_from_pts(np.stack([rr, cc], 1), ids[rr, cc]) if len(x) >= 2]
        log(f"  LINB2 Лапландии из {p.name}")
    else:
        lines = lapland_lines(shape, log)
    log(f"  линий LINB2 Лапландии {len(lines)}")
    if lines_only:
        return
    H, Wd = shape
    cut = lambda a: a[:H, :Wd].astype("float32")
    ls = np.load(DEST / "_lapland" / "landsat_summer.npz")
    s2 = np.load(DEST / "_lapland" / "s2w_summer.npz")
    nd = lambda r, n: (n - r) / np.maximum(n + r, 1e-6)
    W = {WIT: {"красный": cut(ls["red"]), "NDVI": nd(cut(ls["red"]), cut(ls["nir"]))},
         "S2 лето": {"красный": cut(s2["B04"]), "NDVI": nd(cut(s2["B04"]), cut(s2["B08"]))}}
    log(f"ПОРОГ с Утсйоки L* = {Ls:.0f} м (не пересчитывается)")
    # ---- (а) калибровка: допуск
    P, idx, T, N = geometry(lines)
    nl = len(lines)
    ok = admissible(P, idx, N, nl, shape)
    adm = True
    for name, seed in (("белый шум", 9500), (WIT, 9600)):
        vals = []
        for r in range(R_SYN):
            rng = np.random.default_rng(seed + r)
            if name == "белый шум":
                syn = {"шум": rng.standard_normal(shape).astype("float32")}
            else:
                ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (H, Wd // 2 + 1)))
                ph[0, 0] = 1
                syn = {k: np.where(np.isfinite(f), phase_surrogate(f, ph), np.nan) for k, f in W[WIT].items()}
            hs, valid = prep_witness(syn, shape)
            st = t_stats(P, idx, T, N, nl, hs, valid, shape, pairs=(MAIN_PX,))
            fin = ok & np.all(np.isfinite(st), 1)
            vals.append(pi_hat(wins(st[fin], MAIN_PX)))
        m, se = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(R_SYN))
        okt = abs(m) <= T_LAP * se
        adm &= okt
        log(f"  допуск {name}: среднее {100 * m:+.2f} п. п., SE {100 * se:.2f}, порог {T_LAP}·SE = "
            f"{100 * T_LAP * se:.2f} -> {'ПРОЙДЕН' if okt else 'НЕ ПРОЙДЕН'}")
    Wp = {wn: prep_witness(W[wn], shape) for wn in W}
    # тор
    moved = []
    for dr in (NEG_PX, -NEG_PX):
        for dc in (NEG_PX, -NEG_PX):
            for x in lines:
                r_, c_ = x[:, 0] + dr, x[:, 1] + dc
                if r_.min() < 0 or r_.max() >= H or c_.min() < 0 or c_.max() >= Wd:
                    continue
                moved.append(np.stack([r_, c_], 1))
    wt, ft, ct, _ = line_t(moved, Wp, WIT, shape)
    tlo, thi = boot(wt[ft], ct[ft], BLOCKS_PX[0], 0)
    tor = tlo <= 0 <= thi
    log(f"  тор: линий {len(moved)}, π̂ {100 * pi_hat(wt[ft]):+.1f} % [{100 * tlo:+.1f}…{100 * thi:+.1f}] -> "
        + ("содержит 0" if tor else "НЕ содержит 0"))
    # положительный
    from experiments.openlin_zone_lines import trace
    f = W["S2 лето"]["красный"]
    tl = [x["pts"] for x in trace(f, np.isfinite(f), RES, local=True) if len(x["pts"]) >= 2]
    wp, fp, cp, _ = line_t(tl, Wp, WIT, shape)
    e_L = pi_hat(wp[fp])
    plo, phi = lo_hi(wp[fp], cp[fp])
    pos = plo >= 0.5
    log(f"  положительный trace S2 -> Landsat: линий {int(fp.sum())}, π̂ {100 * e_L:+.0f} % "
        f"[{100 * plo:+.0f}…{100 * phi:+.0f}] -> {'пройден' if pos else 'НЕ пройден'}; e_L = {e_L:.2f}")
    # ---- оставленные / отсечённые
    w, fin, cen, lens = line_t(lines, Wp, WIT, shape)
    keep = lens >= Ls
    wo, fo, _, _ = line_t(lines, Wp, WIT, shape, swap=True)
    olo, ohi = lo_hi(wo[fo & keep], cen[fo & keep])
    ori = olo <= 0
    log(f"  ориентационный (оставленные, n↔t): π̂ {100 * pi_hat(wo[fo & keep]):+.0f} % [{100 * olo:+.0f}…"
        f"{100 * ohi:+.0f}] -> {'пройден' if ori else 'НЕ пройден'}")
    calib = adm and tor and pos and ori
    log("(а) КАЛИБРОВКА НА ЛАПЛАНДИИ: " + ("ПРОЙДЕНА" if calib else "НЕ ПРОЙДЕНА — мера не откалибрована, о пороге не говорим"))
    nk, nr = int((fin & keep).sum()), int((fin & ~keep).sum())
    log(f"  в счёт: оставленных {nk}, отсечённых {nr} (из {nl}); км отсечено {lens[~keep].sum() / 1000:.0f} "
        f"из {lens.sum() / 1000:.0f}")
    for lab, mm in (("оставленные", fin & keep), ("отсечённые", fin & ~keep)):
        if mm.sum():
            lo, hi = lo_hi(w[mm], cen[mm])
            log(f"  {lab}: π̂ {100 * pi_hat(w[mm]):+.0f} % [{100 * lo:+.0f}…{100 * hi:+.0f}], "
                f"кажущаяся доля {pi_hat(w[mm]) / e_L:.2f} [{lo / e_L:.2f}…{hi / e_L:.2f}]")
    if not calib:
        return
    if min(nk, nr) < N_MIN:
        log(f"ИТОГ: НЕДОСТАТОЧНО ДАННЫХ (в группе меньше {N_MIN} линий)")
        return
    lo_k, _ = lo_hi(w[fin & keep], cen[fin & keep])
    b_ok = lo_k / e_L >= GOOD
    dlo, dhi = diff_ci(w[fin], cen[fin], keep[fin])
    c_ok = dlo > 0
    log(f"(б) нижний край кажущейся доли оставленных {lo_k / e_L:.2f} >= {GOOD} -> {'да' if b_ok else 'НЕТ'}")
    log(f"(в) разность оставленные − отсечённые [{100 * dlo:+.0f}…{100 * dhi:+.0f}] п. п., нижний край > 0 -> "
        f"{'да' if c_ok else 'НЕТ'}")
    log("ИТОГ ЛАПЛАНДИИ: " + ("ПОРОГ ПОДТВЕРЖДЁН" if b_ok and c_ok else "порог НЕ подтверждён — не переносится"))


def main():
    from experiments.openlin_fam import OUT, log_to
    if "--utsjoki" in sys.argv:
        log = log_to(OUT / "_len1_utsjoki.log")
        log("LEN1 Утсйоки; объявление audit/LEN1_DECLARATION.md (поправки 1–2); свидетель Landsat лето")
        utsjoki(log)
    elif "--lapland-lines" in sys.argv:
        log = log_to(OUT / "_len1_lapland_lines.log")
        log("LEN1 Лапландия: только построение LINB2 (для openlin-artifact-hunter)")
        lapland(log, lines_only=True)
    elif "--lapland" in sys.argv:
        log = log_to(OUT / "_len1_lapland.log")
        log("LEN1 Лапландия (контроль, разово); объявление audit/LEN1_DECLARATION.md")
        lapland(log)


if __name__ == "__main__":
    main()
