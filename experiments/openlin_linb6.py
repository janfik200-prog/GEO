"""LINB6: LINB5 без двойников — отбор с запретом линий, у которых > 50 % ячеек в 127 м от уже взятых
(audit/LINB6_DECLARATION.md, поправка 1). Трассировка — как LINB5 (openlin_linb5).

    python -m experiments.openlin_linb6 --utsjoki     # настройка: (а), (б), (в), (г1), (г2)
    python -m experiments.openlin_linb6 --anabar      # цель: картинки и описание
LINB6_N_TESTS = 0. Лидар не используется. Окна LINB5, которые смотрели до правила: Утсйоки 1 и 5.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments.openlin_fam import OUT
from experiments.openlin_linb5 import (BUDGET_U_KM, ISO_AXES, PHASE_MAX, RES_IN, as_lines, build20,
                                       grid_stats, length_px_m)

LINB6_N_TESTS = 0
TWIN_M = 127.0                   # TWIN_OFFSET_M (правило сшивки, подобрано на настройке)
TWIN_FRAC = 0.5                  # «не более половины» — по аналогии с openlin_real1.admissible
TWIN_G2 = 0.102                  # доля двойников RELU1 (twins.py) — выбор границы (г2)
R_PX = TWIN_M / RES_IN
DISK = [(dr, dc) for dr in range(-7, 8) for dc in range(-7, 8) if dr * dr + dc * dc <= R_PX * R_PX]


def stamp(near, cells):
    H, W = near.shape
    for dr, dc in DISK:
        r, c = cells[:, 0] + dr, cells[:, 1] + dc
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        near[r[ok], c[ok]] = True


def select_twins(lines, P, valid20, budget_km, log, share=None):
    """Отбор LINB2/LINB5 + пропуск двойников. Бюджет — сумма ячеек растра по линиям (у пересечения
    ячейка принадлежит одной линии — последней при растеризации), как у LINB5."""
    from experiments.openlin_anfisa_lines import line_mean_p, rasterize
    ids_all = np.where(valid20, rasterize(lines, valid20.shape, 1), 0)
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    total = cells.sum() * RES_IN / 1000
    if share is not None:
        budget_km = share * total
    mp = np.array([line_mean_p(pl, P) for pl in lines])
    lenm = np.array([length_px_m(pl) for pl in lines])
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    # ячейки каждой линии (по растру)
    rr, cc = np.nonzero(ids_all)
    lab = ids_all[rr, cc]
    srt = np.argsort(lab, kind="stable")
    rr, cc, lab = rr[srt], cc[srt], lab[srt]
    start = np.searchsorted(lab, np.arange(1, len(lines) + 2))
    near = np.zeros(valid20.shape, bool)
    taken, skipped, used = [], [], 0.0
    for i in order:
        a, b = start[i], start[i + 1]
        if b == a:
            continue
        cl = np.stack([rr[a:b], cc[a:b]], 1)
        if used + (b - a) * RES_IN / 1000 > budget_km:
            break
        if near[cl[:, 0], cl[:, 1]].mean() > TWIN_FRAC:
            skipped.append(i)
            continue
        taken.append(i)
        used += (b - a) * RES_IN / 1000
        stamp(near, cl)
    keep = np.zeros(len(lines) + 1, bool)
    keep[np.array(taken) + 1] = True
    log(f"  все линии {len(lines)}, {total:.0f} км (растр); бюджет {budget_km:.0f} км: взято {len(taken)}, "
        f"пропущено двойников {len(skipped)}; порог среднего P {mp[taken[-1]]:.3f}; доля бюджета {budget_km / total:.4f}")
    return np.where(keep[ids_all], ids_all, 0), total, taken, skipped, (rr, cc, start)


def check_g1(taken, cellinfo, shape):
    """(г1) пересчёт в порядке отбора независимой функцией (distance_transform_edt): нарушений 0."""
    rr, cc, start = cellinfo
    viol = 0
    sel = np.zeros(shape, bool)
    for i in taken:
        a, b = start[i], start[i + 1]
        r, c = rr[a:b], cc[a:b]
        if sel.any():
            r0, c0 = max(r.min() - 8, 0), max(c.min() - 8, 0)
            r1, c1 = min(r.max() + 9, shape[0]), min(c.max() + 9, shape[1])
            sub = sel[r0:r1, c0:c1]
            if sub.any():
                d = distance_transform_edt(~sub) * RES_IN
                viol += np.mean(d[r - r0, c - c0] <= TWIN_M) > TWIN_FRAC
        sel[r, c] = True
    return int(viol)


def synthetic_test():
    """Две параллельные линии в 60 м — вторая пропускается; ветвь Y под 45° длиной 600 м — берётся."""
    shape = (300, 300)
    a = np.array([[100, c] for c in range(50, 250)], float)
    b = np.array([[103, c] for c in range(50, 250)], float)              # 60 м
    y = np.array([[100 + k, 150 + k] for k in range(0, 30)], float)      # 30·√2·20 ≈ 850 м
    P = np.zeros(shape)
    P[100, :] = 1.0
    P[103, :] = 0.9
    for k in range(30):
        P[100 + k, 150 + k] = 0.8
    valid = np.ones(shape, bool)
    ids, _t, taken, skipped, _ = select_twins([a, b, y], P, valid, 1e9, lambda *_: None)
    assert taken == [0, 2] and skipped == [1], (taken, skipped)


def twins_share(lines, shape):
    lab = np.zeros(shape, np.int32)
    for i, x in enumerate(lines, 1):
        lab[x[:, 0], x[:, 1]] = i
    fr = []
    for i, x in enumerate(lines, 1):
        r0, c0 = np.maximum(x.min(0) - 8, 0)
        r1, c1 = np.minimum(x.max(0) + 9, shape)
        sub = lab[r0:r1, c0:c1]
        other = (sub > 0) & (sub != i)
        if not other.any():
            fr.append(0.0)
            continue
        d = distance_transform_edt(~other) * RES_IN
        fr.append(float(np.mean(d[x[:, 0] - r0, x[:, 1] - c0] <= TWIN_M)))
    return float(np.mean(np.array(fr) > TWIN_FRAC))


def side_report(lines, taken, skipped, L5ids, L6ids, log):
    """Для сведения: потери — пропущенные по длине и углу к ближайшей взятой; покрытие LINB5 линиями LINB6."""
    lens = np.array([length_px_m(lines[i]) for i in skipped])
    log(f"  пропущено: {len(skipped)} линий, {lens.sum() / 1000:.0f} км (ломаная); < 500 м: {int((lens < 500).sum())}, "
        f">= 500 м: {int((lens >= 500).sum())}")
    d = distance_transform_edt(L6ids == 0) * RES_IN
    rr, cc = np.nonzero(L5ids)
    log(f"  покрытие: ячеек LINB5 в 127 м от LINB6 {100 * np.mean(d[rr, cc] <= TWIN_M):.1f} %")
    # угол к «победившей»: ближайшая взятая линия по центру
    def ax(x):
        q = x - x.mean(0)
        _w, v = np.linalg.eigh(q.T @ q)
        return v[:, -1]
    from scipy.spatial import cKDTree
    cen_t = np.array([lines[i].mean(0) for i in taken])
    tr = cKDTree(cen_t)
    small = 0
    for i in skipped:
        _, j = tr.query(lines[i].mean(0))
        cosang = abs(float(np.dot(ax(lines[i]), ax(lines[taken[j]]))))
        small += cosang >= np.cos(np.radians(20))
    log(f"  угол пропущенной к ближайшей взятой (по центрам): < 20° {small}, >= 20° {len(skipped) - small}")


def utsjoki(log):
    from experiments.openlin_len1 import line_t, lo_hi
    from experiments.openlin_linb5 import select, windows_pics
    from experiments.openlin_real1 import M, load_sets, witnesses
    from experiments.openlin_real2 import BLOCKS_PX, N_BOOT, pi_hat, prep_witness
    synthetic_test()
    log("  синтетика (г1): параллельная в 60 м пропущена, ветвь под 45° взята — ок")
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    lines, P, valid20 = build20("utsjoki", log)
    P, valid20 = P[:shape[0], :shape[1]], valid20[:shape[0], :shape[1]]
    log("  LINB5 (те же линии, отбор без пропуска):")
    ids5, _t = select(lines, P, valid20, BUDGET_U_KM, log)
    log("  LINB6:")
    ids6, total, taken, skipped, cellinfo = select_twins(lines, P, valid20, BUDGET_U_KM, log)
    np.savez_compressed(OUT / "linb6_utsjoki.npz", ids=ids6)
    (OUT / "linb6_utsjoki.json").write_text(json.dumps({"share": BUDGET_U_KM / total, "total_km": total}),
                                            encoding="utf-8")
    side_report(lines, taken, skipped, ids5, ids6, log)
    L5, L6 = as_lines(ids5), as_lines(ids6)
    sets = load_sets(shape)
    L2 = sets["Анфиса LINB2"]
    log("=== (а), (б)")
    for k, v in sets.items():
        grid_stats(v, log, k.split()[-1])
    grid_stats(L5, log, "LINB5")
    s6 = grid_stats(L6, log, "LINB6")
    a_ok = s6["фаза"] <= PHASE_MAX
    n, ax, _ = s6["<620"]
    thr = ISO_AXES + 2 * np.sqrt(ISO_AXES * (1 - ISO_AXES) / max(n, 1))
    b_ok = None if n < 100 else ax <= thr
    log(f"(а) фаза {s6['фаза']:.3f} <= {PHASE_MAX} -> {'да' if a_ok else 'НЕТ'}")
    log(f"(б) короткие у осей {100 * ax:.1f} % <= {100 * thr:.1f} % (n={n}) -> "
        + ("не оценивается" if b_ok is None else "да" if b_ok else "НЕТ"))
    log("=== (г)")
    v = check_g1(taken, cellinfo, shape)
    g1 = v == 0
    log(f"(г1) нарушений правила при пересчёте: {v} -> {'да' if g1 else 'НЕТ — ошибка кода'}")
    tw = {k: twins_share(x, shape) for k, x in (("LINB2", L2), ("LINB5", L5), ("LINB6", L6))}
    g2 = tw["LINB6"] <= TWIN_G2
    log(f"  двойники: LINB2 {100 * tw['LINB2']:.1f} %, LINB5 {100 * tw['LINB5']:.1f} %, LINB6 {100 * tw['LINB6']:.1f} %")
    log(f"(г2) LINB6 {100 * tw['LINB6']:.1f} % <= {100 * TWIN_G2:.1f} % -> {'да' if g2 else 'НЕТ — двойники назад остались'}")
    log("=== (в): мера REAL3")
    W = witnesses(shape)
    Wp = {wn: prep_witness(W[wn], shape) for wn in ("Landsat лето", "S2 лето")}
    c_ok = None
    for wn in ("Landsat лето", "S2 лето"):
        res = {}
        for lab, L in (("LINB2", L2), ("LINB5", L5), ("LINB6", L6)):
            w, f, c, _ = line_t(L, Wp, wn, shape)
            lo, hi = lo_hi(w[f], c[f])
            res[lab] = (w[f], c[f])
            log(f"  {wn} {lab}: линий {int(f.sum())}, π̂ {100 * pi_hat(w[f]):+.1f} % [{100 * lo:+.1f}…{100 * hi:+.1f}]")
        for a_, b_ in (("LINB6", "LINB2"), ("LINB6", "LINB5")):
            wa, ca = res[a_]
            wb, cb = res[b_]
            ba = (ca[:, 0] // BLOCKS_PX[0]).astype(int) * 100000 + (ca[:, 1] // BLOCKS_PX[0]).astype(int)
            bb = (cb[:, 0] // BLOCKS_PX[0]).astype(int) * 100000 + (cb[:, 1] // BLOCKS_PX[0]).astype(int)
            ub = np.union1d(ba, bb)
            ga = {k: wa[ba == k] for k in ub}
            gb = {k: wb[bb == k] for k in ub}
            los, his = [], []
            for sd in (0, 1):
                rng = np.random.default_rng(sd)
                vals = []
                for _ in range(N_BOOT):
                    s = rng.choice(ub, len(ub), replace=True)
                    vals.append(pi_hat(np.concatenate([ga[k] for k in s])) - pi_hat(np.concatenate([gb[k] for k in s])))
                lo, hi = np.percentile(vals, [2.5, 97.5])
                los.append(lo)
                his.append(hi)
            log(f"  {wn}: {a_} − {b_} {100 * (pi_hat(wa) - pi_hat(wb)):+.1f} п. п. [{100 * min(los):+.1f}…{100 * max(his):+.1f}]")
            if wn == "Landsat лето" and b_ == "LINB2":
                c_ok = min(los) >= -0.03
                log(f"(в) нижний край {100 * min(los):+.1f} >= −3 п. п. -> {'да' if c_ok else 'НЕТ'}")
        # общие линии LINB5 ∩ LINB6 против только-LINB6 (справка)
        com = [x for x in L6 if ids5[x[0, 0], x[0, 1]] > 0 and ids5[x[0, 0], x[0, 1]] == ids6[x[0, 0], x[0, 1]]]
        only = [x for x in L6 if not (ids5[x[0, 0], x[0, 1]] > 0 and ids5[x[0, 0], x[0, 1]] == ids6[x[0, 0], x[0, 1]])]
        for lab, L in (("общие с LINB5", com), ("только LINB6", only)):
            if L:
                w, f, c, _ = line_t(L, Wp, wn, shape)
                log(f"  {wn} {lab}: линий {int(f.sum())}, π̂ {100 * pi_hat(w[f]):+.1f} %")
    verdict = ("ошибка кода (г1)" if not g1 else "сетка не убрана" if (not a_ok or b_ok is False) else
               "хуже по мере" if not c_ok else
               "ПРИНЯТО" + ("" if g2 else ", но двойники назад остались (г2)") +
               (" (б не оценивается)" if b_ok is None else ""))
    log(f"ИТОГ LINB6 Утсйоки: {verdict}")
    windows_pics3("utsjoki", L2, L5, L6, shape, log, win_km=6)


def windows_pics3(site, L2, L5, L6, shape, log, n_win=(3, 3), win_km=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_transfer_check import COMP
    out = OUT / f"linb6_{site}"
    out.mkdir(exist_ok=True)
    img = np.load(COMP / f"winter_{site}_20m_лист.npz")["shade"][:shape[0], :shape[1]].astype("float32")
    H, Wd = shape
    wp = int(win_km * 1000 / RES_IN)
    k = 0
    for i in range(n_win[0]):
        for j in range(n_win[1]):
            k += 1
            r0 = int((i + 0.5) * H / n_win[0] - wp / 2)
            c0 = int((j + 0.5) * Wd / n_win[1] - wp / 2)
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            b = img[r0:r0 + wp, c0:c0 + wp]
            lo, hi = np.nanpercentile(b, [2, 98])
            for ax, (title, L) in zip(axs.ravel(), (("зимний S2, без линий", None), ("LINB2 (сетка 100 м)", L2),
                                                    ("LINB5 (20 м, с двойниками)", L5), ("LINB6 (20 м, без двойников)", L6))):
                ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=(0, win_km, win_km, 0))
                ax.set_title(title, fontsize=13)
                if L is None:
                    continue
                for x in L:
                    y = x - [r0, c0]
                    m = (y[:, 0] >= 0) & (y[:, 0] < wp) & (y[:, 1] >= 0) & (y[:, 1] < wp)
                    if m.any():
                        y = y[m] * RES_IN / 1000
                        ax.plot(y[:, 1], y[:, 0], ".", ms=1.6, color="red")
            fig.suptitle(f"{site}, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}", fontsize=14)
            fig.tight_layout()
            p = out / f"окно_{k}.png"
            fig.savefig(p, dpi=65)
            plt.close(fig)
            log(f"  {p}")


def anabar(log):
    from experiments.openlin_linb5 import select
    share = json.loads((OUT / "linb6_utsjoki.json").read_text(encoding="utf-8"))["share"]
    lines, P, valid20 = build20("anabar", log)
    shape = valid20.shape
    ids5, _ = select(lines, P, valid20, None, log, share=share)
    ids6, total, taken, skipped, cellinfo = select_twins(lines, P, valid20, None, log, share=share)
    np.savez_compressed(OUT / "linb6_anabar.npz", ids=ids6)
    side_report(lines, taken, skipped, ids5, ids6, log)
    L2 = as_lines(np.load(OUT / "anabar_linb2" / "linb2_anabar.npz")["ids"])
    L5, L6 = as_lines(ids5), as_lines(ids6)
    log("=== описание, без критериев")
    for lab, L in (("LINB2", L2), ("LINB5", L5), ("LINB6", L6)):
        grid_stats(L, log, lab)
        log(f"  {lab}: линий {len(L)}, {sum(len(x) for x in L) * RES_IN / 1000:.0f} км (ячейки), "
            f"двойников {100 * twins_share(L, shape):.1f} %")
    log(f"  (г1) нарушений {check_g1(taken, cellinfo, shape)}")
    windows_pics3("anabar", L2, L5, L6, shape, log, win_km=8)


def main():
    from experiments.openlin_fam import log_to
    if "--utsjoki" in sys.argv:
        log = log_to(OUT / "_linb6_utsjoki.log")
        log("LINB6 Утсйоки; объявление audit/LINB6_DECLARATION.md (поправка 1)")
        utsjoki(log)
    elif "--anabar" in sys.argv:
        log = log_to(OUT / "_linb6_anabar.log")
        log("LINB6 Анабар (описательно, картинки)")
        anabar(log)


if __name__ == "__main__":
    main()
