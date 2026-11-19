"""CAND3: кандидаты CAND2 без дублей каналов — одна черта, одна линия (audit/CAND3_DECLARATION.md).
Правило LINB6: по убыванию силы (среднее z своего канала), линия берётся, если <= 50 % её ячеек в 127 м
от уже взятых линий любого канала.

    python -m experiments.openlin_cand3 --utsjoki     # настройка: (б), (в1), (в2), (г1)
    python -m experiments.openlin_cand3 --anabar      # цель: только если Утсйоки ПРИНЯТО
CAND3_N_TESTS = 0. Лидар не используется. Только процессор.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_linb6 import DISK, TWIN_FRAC, TWIN_M, stamp

CAND3_N_TESTS = 0
#: лог CAND2 Утсйоки (линий >= 200 м, км ломаной) — поправка 1 registrar: расхождение = прогон не засчитан
CAND2_UTS = {0: (9385, 6561), 1: (9971, 6923), 2: (11599, 8153), 3: (1902, 2588), 4: (2032, 2261)}


def build(site, log):
    """Линии каналов как CAND2 (без подделки) + карты z для силы линии."""
    from experiments.openlin_anfisa_lines import rasterize
    img, dem, valid, meta = C1.load_inputs(site)
    log(f"  лист {site}: {img.shape}, годных ячеек {int(valid.sum())}")
    chans = {}
    for ch in range(len(C1.CHANNELS)):
        log(f"  канал {ch + 1} «{C1.CHANNELS[ch]}»")
        r, th = C1.response(ch, img, dem)
        z = C2.local_norm_nz(r, valid)
        lines, km = C2.trace_z(z, th, valid, log, "снимок")
        ids = rasterize(lines, valid.shape, 1)
        n = int(ids.max())
        cells = np.bincount(ids.ravel(), minlength=n + 1)
        zsum = np.bincount(ids.ravel(), weights=z.ravel().astype("float64"), minlength=n + 1)
        mz = zsum / np.maximum(cells, 1)
        chans[ch] = {"ids": ids, "cells": cells, "mz": mz, "n": n, "km": km}
        ok = cells[1:] > 0
        q = np.percentile(mz[1:][ok], [10, 50, 90])
        rho = np.corrcoef(mz[1:][ok], np.log(cells[1:][ok]))[0, 1]
        log(f"    линий {len(lines)}, {km:.0f} км (ломаная), {cells[1:].sum() * RES_IN / 1000:.0f} км (растр); "
            f"сила (среднее z) 10/50/90 % {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}; corr(сила, log длины) {rho:+.2f}")
        if site == "utsjoki":
            want = CAND2_UTS[ch]
            same = len(lines) == want[0] and round(km) == want[1]
            log(f"    сверка с CAND2: {len(lines)}/{km:.0f} против {want[0]}/{want[1]} -> {'совпало' if same else 'НЕ СОВПАЛО'}")
            assert same, "трассировка не совпала с CAND2 — прогон не засчитывается (поправка 1)"
    return chans, img, dem, valid, meta


def cell_lists(ids, n):
    rr, cc = np.nonzero(ids)
    lab = ids[rr, cc]
    srt = np.argsort(lab, kind="stable")
    rr, cc, lab = rr[srt], cc[srt], lab[srt]
    start = np.searchsorted(lab, np.arange(1, n + 2))
    return rr, cc, start


def dedupe(chans, shape, log):
    items = [(ch, j) for ch in chans for j in range(1, chans[ch]["n"] + 1) if chans[ch]["cells"][j] > 0]
    mz = np.array([chans[ch]["mz"][j] for ch, j in items])
    ln = np.array([chans[ch]["cells"][j] for ch, j in items])
    chn = np.array([ch for ch, _ in items])
    jn = np.array([j for _, j in items])
    order = np.lexsort((jn, chn, -ln, -mz))
    cl = {ch: cell_lists(chans[ch]["ids"], chans[ch]["n"]) for ch in chans}
    near = np.zeros(shape, bool)
    taken, skipped = [], []
    for i in order:
        ch, j = items[i]
        rr, cc, start = cl[ch]
        a, b = start[j - 1], start[j]
        c = np.stack([rr[a:b], cc[a:b]], 1)
        if near[c[:, 0], c[:, 1]].mean() > TWIN_FRAC:
            skipped.append((ch, j))
            continue
        taken.append((ch, j))
        stamp(near, c)
    keep = {ch: np.zeros(chans[ch]["n"] + 1, bool) for ch in chans}
    for ch, j in taken:
        keep[ch][j] = True
    log(f"  отбор: кандидатов {len(items)}, взято {len(taken)}, пропущено двойников {len(skipped)}")
    for ch in chans:
        nt = int(keep[ch][1:].sum())
        km0 = chans[ch]["cells"][1:].sum() * RES_IN / 1000
        km1 = chans[ch]["cells"][keep[ch]].sum() * RES_IN / 1000
        log(f"    {C1.CHANNELS[ch]}: линий {chans[ch]['n']} -> {nt}, км (растр) {km0:.0f} -> {km1:.0f} "
            f"(пропущено {100 * (1 - km1 / max(km0, 1e-9)):.1f} % км)")
    return taken, keep, cl


def check_g1(taken, cl, shape):
    """(г1) пересчёт в порядке отбора независимой функцией: нарушений 0."""
    sel = np.zeros(shape, bool)
    viol = 0
    for ch, j in taken:
        rr, cc, start = cl[ch]
        r, c = rr[start[j - 1]:start[j]], cc[start[j - 1]:start[j]]
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
    """Две параллельные линии разных каналов в 60 м — вторая (слабее) пропускается; ветвь под 45° — берётся."""
    shape = (300, 300)
    a = np.zeros(shape, np.int32)
    b = np.zeros(shape, np.int32)
    a[100, 50:250] = 1
    b[103, 50:250] = 1
    for k in range(30):
        b[100 + k + 1, 150 + k + 1] = 2
    chans = {0: {"ids": a, "n": 1, "cells": np.bincount(a.ravel(), minlength=2), "mz": np.array([0, 3.0])},
             1: {"ids": b, "n": 2, "cells": np.bincount(b.ravel(), minlength=3), "mz": np.array([0, 2.0, 1.0])}}
    taken, _k, _c = dedupe(chans, shape, lambda *a: None)
    assert taken == [(0, 1), (1, 2)], taken


def union_thinned_cover(M, chs, gtk, km_target, n_rep, seed):
    """(в2): случайные линии, пока км ОБЪЕДИНЕНИЯ ячеек <= km_target (наибольшее такое число линий)."""
    items = [(ch, j) for ch in chs for j in range(1, int(M[ch][0].max()) + 1)]

    def mask(perm, n):
        sel = np.zeros(gtk.shape, bool)
        for ch in chs:
            keep = np.zeros(int(M[ch][0].max()) + 1, bool)
            js = [items[i][1] for i in perm[:n] if items[i][0] == ch]
            keep[js] = True
            keep[0] = False
            sel |= keep[M[ch][0]]
        return sel
    out, kms = [], []
    for rep in range(n_rep):
        perm = np.random.default_rng(seed + rep).permutation(len(items))   # зерно 1900 + номер повтора
        lo, hi = 0, len(items)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if mask(perm, mid).sum() * RES_IN / 1000 <= km_target:
                lo = mid
            else:
                hi = mid - 1
        sel = mask(perm, lo)
        out.append(C1.cover(gtk, sel, C1.TOL_GTK_M))
        kms.append(sel.sum() * RES_IN / 1000)
    return np.array(out), float(np.mean(kms))


def pool_masks(chans, keep):
    from experiments.openlin_linb5 import as_lines
    M = {}
    for ch in chans:
        ids = np.where(keep[ch][chans[ch]["ids"]], chans[ch]["ids"], 0)
        M[ch] = (ids, as_lines(ids))
    return M


def save(site, M, meta):
    d = {"res_m": RES_IN, **meta}
    for ch in M:
        d[f"ids_{ch + 1}"] = M[ch][0]
    np.savez_compressed(OUT / f"cand3_{site}.npz", **d)


def run(site, log):
    synthetic_test()
    log("  синтетика (г1): параллельная в 60 м другого канала пропущена, ветвь под 45° взята — ок")
    chans, img, dem, valid, meta = build(site, log)
    shape = valid.shape
    taken, keep, cl = dedupe(chans, shape, log)
    v = check_g1(taken, cl, shape)
    log(f"(г1) нарушений правила при пересчёте: {v} -> {'да' if v == 0 else 'НЕТ — ошибка кода'}")
    M = pool_masks(chans, keep)
    save(site, M, meta)
    out = {ch: (None, chans[ch]["km"], np.nan, True) for ch in chans}
    l6 = np.zeros(shape, bool)
    a = np.load(OUT / f"linb6_{site}.npz")["ids"] > 0
    h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    l6[:h, :w] = a[:h, :w]
    pool, pool_ch, b_res = C1.describe(site, out, M, valid, meta, log, l6=l6)
    z2 = np.load(OUT / f"cand2_{site}.npz") if (OUT / f"cand2_{site}.npz").exists() else None
    if z2 is not None:
        c2 = np.zeros(shape, bool)
        for ch in range(1, 6):
            c2 |= z2[f"ids_{ch}"][:shape[0], :shape[1]] > 0
        log(f"  CAND2 в 127 м от CAND3 (потеря территории): {100 * C1.cover(c2, pool, 127):.1f} %; "
            f"км CAND2 {c2.sum() * RES_IN / 1000:.0f} -> CAND3 {pool.sum() * RES_IN / 1000:.0f} (объединение)")
    return M, pool, pool_ch, b_res, v, img, dem, valid, meta


def utsjoki(log):
    M, pool, pool_ch, b_res, v, img, dem, valid, meta = run("utsjoki", log)
    shape = valid.shape
    gtk = C1.gtk_mask("utsjoki", meta, shape, valid)
    log("=== (в) GTK 292 в 300 м")
    for ch in M:
        log(f"  {C1.CHANNELS[ch]}: {100 * C1.cover(gtk, M[ch][0] > 0, C1.TOL_GTK_M):.1f} %")
    c = C1.cover(gtk, pool, C1.TOL_GTK_M)
    c1 = c >= C1.GTK_OLD_POOL
    log(f"(в1) пул CAND3 {100 * c:.1f} % >= {100 * C1.GTK_OLD_POOL:.1f} % -> {'да' if c1 else 'НЕТ'}")
    # проверка процедуры (поправка 1): та же функция на прежнем пуле должна дать ~6826 км и ~68,0 %
    from experiments.openlin_real1 import M as MET
    Mold = {}
    for k_, tag in enumerate(("_настройка1", "_рельеф1")):
        z_ = np.load(MET / f"openlin_zone_lines_utsjoki_v3{tag}_линии.npz")
        p, lid = z_["A_all_pts"].astype(int), z_["A_all_id"].astype(int)
        okp = (p[:, 0] >= 0) & (p[:, 0] < shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < shape[1])
        _u, lab = np.unique(lid[okp], return_inverse=True)
        ids_ = np.zeros(shape, np.int32)
        ids_[p[okp, 0], p[okp, 1]] = lab + 1
        Mold[k_] = (ids_, None)
    cvo, kmo = union_thinned_cover(Mold, [0, 1], gtk, C1.OLD_POOL_KM, 1, C1.SEED0)
    log(f"  проверка процедуры на прежнем пуле: взято {kmo:.0f} км (объединение), полнота {100 * cvo[0]:.1f} %")
    km_pool = pool.sum() * RES_IN / 1000
    if km_pool > C1.OLD_POOL_KM:
        cv, km_mean = union_thinned_cover(M, pool_ch, gtk, C1.OLD_POOL_KM, C1.N_THIN, C1.SEED0)
        c2 = float(np.mean(cv)) >= C1.GTK_OLD_POOL
        log(f"(в2) пул {km_pool:.0f} км (объединение) > {C1.OLD_POOL_KM:.0f}: прореживание до {C1.OLD_POOL_KM:.0f} км "
            f"по объединению, взято в среднем {km_mean:.0f} км, {C1.N_THIN} повторов: полнота {100 * np.mean(cv):.1f} % "
            f"[мин {100 * np.min(cv):.1f}, макс {100 * np.max(cv):.1f}] >= {100 * C1.GTK_OLD_POOL:.1f} % -> "
            f"{'да' if c2 else 'НЕТ'}")
    else:
        c2 = c1
        log(f"(в2) пул {km_pool:.0f} км <= {C1.OLD_POOL_KM:.0f}: прореживания нет, сырая полнота {100 * c:.1f} % -> "
            f"{'да' if c2 else 'НЕТ'}")
    b_txt = ", ".join(f"{C1.CHANNELS[ch] if ch != 'пул' else 'пул'} "
                      f"{'не оценено' if x is None else 'да' if x else 'НЕТ'}" for ch, x in b_res.items())
    log(f"ИТОГ CAND3 Утсйоки: (г1) {'да' if v == 0 else 'НЕТ'}; (б) {b_txt}; (в1) {'да' if c1 else 'НЕТ'}; "
        f"(в2) {'да' if c2 else 'НЕТ'} -> {'ПРИНЯТО' if c1 and c2 and v == 0 else 'НЕ ПРИНЯТО'}")
    windows("utsjoki", img, dem, M, log, 6)


def windows(site, img, dem, M, log, win_km):
    """Раскладка окон CAND2, своя папка cand3_<лист> (окна CAND2 не трогаются)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = OUT / f"cand3_{site}"
    out.mkdir(exist_ok=True)
    gy, gx = np.gradient(dem, RES_IN)
    az = np.radians(315)
    shade = np.clip(0.6 + 0.4 * (-(gx * np.sin(az) - gy * np.cos(az)) / np.maximum(np.hypot(gx, gy), 1e-3)
                                 * np.tanh(np.hypot(gx, gy) * 10)), 0, 1)
    col = {0: "red", 1: "cyan", 2: "yellow", 3: "blue", 4: "orange"}
    H, Wd = img.shape
    wp = int(win_km * 1000 / RES_IN)
    k = 0
    for i in range(3):
        for j in range(3):
            k += 1
            r0 = int((i + 0.5) * H / 3 - wp / 2)
            c0 = int((j + 0.5) * Wd / 3 - wp / 2)
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            panels = (("зимний S2, без линий", img, ()),
                      ("S2 + все кандидаты (цвет — канал)", img, (0, 1, 2, 3, 4)),
                      ("рельеф GLO-30, без линий", shade, ()), ("рельеф + все кандидаты", shade, (0, 1, 2, 3, 4)))
            for ax, (title, a, chs) in zip(axs.ravel(), panels):
                b = a[r0:r0 + wp, c0:c0 + wp]
                lo, hi = np.nanpercentile(b, [2, 98])
                ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=(0, win_km, win_km, 0))
                ax.set_title(title, fontsize=12)
                for ch in chs:
                    for x in M[ch][1]:
                        y = x - [r0, c0]
                        m = (y[:, 0] >= 0) & (y[:, 0] < wp) & (y[:, 1] >= 0) & (y[:, 1] < wp)
                        if m.any():
                            y = y[m] * RES_IN / 1000
                            ax.plot(y[:, 1], y[:, 0], ".", ms=1.4, color=col[ch])
            fig.suptitle(f"CAND3 {site}, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}; "
                         f"красн. тёмная, голуб. светлая, жёлт. перепад, син. ложбина, оранж. гребень", fontsize=13)
            fig.tight_layout()
            p = out / f"окно_{k}.png"
            fig.savefig(p, dpi=65)
            plt.close(fig)
            log(f"  {p}")


def anabar(log):
    M, _pool, _ch, _b, _v, img, dem, _valid, _meta = run("anabar", log)
    windows("anabar", img, dem, M, log, 8)


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_len1 import sha
    site = "utsjoki" if "--utsjoki" in sys.argv else "anabar" if "--anabar" in sys.argv else None
    if site is None:
        return
    log = log_to(OUT / f"_cand3_{site}.log")
    log(f"CAND3 {site}; audit/CAND3_DECLARATION.md; sha256 кода {sha(__file__)}, CAND2 {sha(C2.__file__)}, "
        f"CAND1 {sha(C1.__file__)}")
    utsjoki(log) if site == "utsjoki" else anabar(log)


if __name__ == "__main__":
    main()
