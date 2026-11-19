"""CAND2: кандидаты для Анфисы — каналы CAND1, порог рабочего тракта: отклик в сигмах окрестности 2 км
(openlin_iso.local_norm), гистерезис по квантилям 97/90 % самого листа (audit/CAND2_DECLARATION.md).

    python -m experiments.openlin_cand2 --utsjoki     # настройка: (б), (в1), (в2)
    python -m experiments.openlin_cand2 --anabar      # цель: только если Утсйоки ПРИНЯТО
CAND2_N_TESTS = 0. Лидар не используется. Только процессор.
"""
from __future__ import annotations

import sys

import numpy as np

from experiments import openlin_cand1 as C1
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_iso import LOCAL_WIN_M, Q_HIGH, Q_LOW

CAND2_N_TESTS = 0
WIN5 = (1775, 2000, 300)            # окно 5 CAND1 (строка, столбец, размер в ячейках) — тонкие штрихи


def local_norm_nz(r, valid, res_m=RES_IN, win_m=LOCAL_WIN_M):
    """openlin_iso.local_norm, но среднее и разброс логарифма — только по НЕНУЛЕВЫМ ячейкам окна, eps —
    квантиль 1 % ненулевых; нули получают значение ниже любого порога (поправка 2 CAND2: у каналов 1, 2, 4
    нулей 4,9–12,7 %, и z при общей нормировке мерил долю нулей в окне)."""
    from scipy.ndimage import uniform_filter
    k = max(int(round(win_m / res_m)), 3)
    nz = valid & (r > 0)
    eps = max(float(np.quantile(r[nz], 0.01)), 1e-9)
    lr = np.log(np.where(nz, r, 0.0) + eps).astype("float64")
    w = nz.astype("float64")
    den = np.maximum(uniform_filter(w, k, mode="nearest"), 1e-6)
    m = uniform_filter(np.where(nz, lr, 0.0), k, mode="nearest") / den
    sq = uniform_filter(np.where(nz, lr * lr, 0.0), k, mode="nearest") / den
    sd = np.sqrt(np.maximum(sq - m * m, 1e-12))
    z = (lr - m) / sd
    floor = float(z[nz].min()) - 1.0 if nz.any() else 0.0
    return np.where(nz, z, floor).astype("float32")


def trace_z(z, th, valid, log, label, win=None):
    """Как C1.trace, но по нормированному отклику; печать доли ячеек, ушедших в куски < 200 м."""
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches, nms, rasterize
    from experiments.openlin_iso import hysteresis8
    from experiments.openlin_linb5 import length_px_m, stitch20
    v = z[valid]
    hi, lo = float(np.quantile(v, Q_HIGH)), float(np.quantile(v, Q_LOW))
    floor = float(v.min())
    thin = np.where(nms(z, th) & valid, z, floor)
    keep = hysteresis8(thin, lo, hi) & valid
    sk = skeletonize(keep)
    br = branches(sk)
    lines = stitch20(br, lambda *a: None)
    lines = [pl for pl in lines if length_px_m(pl) >= C1.LINK_MIN_LEN_M]
    km = sum(length_px_m(pl) for pl in lines) / 1000
    kept = rasterize(lines, z.shape, 1) > 0
    lost = 1 - float((kept & sk).sum()) / max(int(sk.sum()), 1)
    txt = ""
    if win is not None:
        r0, c0, n = win
        s = sk[r0:r0 + n, c0:c0 + n]
        txt = f"; окно 5: скелет {int(s.sum())} ячеек, ушло в куски < 200 м " \
              f"{100 * (1 - float((kept[r0:r0 + n, c0:c0 + n] & s).sum()) / max(int(s.sum()), 1)):.1f} %"
    log(f"    {label}: z-пороги {Q_HIGH:.0%} {hi:.3f}, {Q_LOW:.0%} {lo:.3f}; после nms+гистерезиса {int(keep.sum())} "
        f"ячеек, ветвей {len(br)}, линий >= 200 м {len(lines)}, {km:.0f} км; ячеек скелета в куски < 200 м "
        f"{100 * lost:.1f} %{txt}")
    return lines, km


def channel(ch, img, dem, valid, fake_img, fake_dem, log, win=None):
    log(f"  канал {ch + 1} «{C1.CHANNELS[ch]}»")
    r, th = C1.response(ch, img, dem)
    lines, km = trace_z(local_norm_nz(r, valid), th, valid, log, "снимок", win)
    rf, thf = C1.response(ch, fake_img, fake_dem)
    _f, kmf = trace_z(local_norm_nz(rf, valid), thf, valid, log, "подделка")
    junk = kmf / km if km else np.inf
    log(f"    справка: км подделки / км канала {100 * junk:.1f} % (не критерий, объявление CAND2)")
    return lines, km, junk, True


def build(site, log, win=None):
    from experiments.openlin_null_input import phase_randomize
    img, dem, valid, meta = C1.load_inputs(site)
    log(f"  лист {site}: {img.shape}, годных ячеек {int(valid.sum())}")
    out = {}
    for ch in range(len(C1.CHANNELS)):
        rng = np.random.default_rng(C1.SEED0 + ch)
        fi = phase_randomize(img, valid, rng).astype("float64") if ch < 3 else img
        fd = phase_randomize(dem, valid, rng).astype("float64") if ch >= 3 else dem
        out[ch] = channel(ch, img, dem, valid, fi, fd, log, win)
    return out, img, dem, valid, meta


def save(site, out, M, meta):
    d = {"res_m": RES_IN, **meta}
    for ch in out:
        d[f"ids_{ch + 1}"] = M[ch][0]
        d[f"junk_{ch + 1}"] = out[ch][2]
    np.savez_compressed(OUT / f"cand2_{site}.npz", **d)


def windows(site, img, dem, M, valid, log, win_km):
    """Окна как CAND1 (та же раскладка), в папку cand2_<лист>."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = OUT / f"cand2_{site}"
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
                      ("S2 + тёмные (красн.), светлые (голуб.), перепады (жёлт.)", img, (0, 1, 2)),
                      ("рельеф GLO-30, без линий", shade, ()), ("рельеф + ложбины (син.), гребни (оранж.)", shade, (3, 4)))
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
            fig.suptitle(f"CAND2 {site}, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}", fontsize=14)
            fig.tight_layout()
            p = out / f"окно_{k}.png"
            fig.savefig(p, dpi=65)
            plt.close(fig)
            log(f"  {p}")


def utsjoki(log):
    from scipy import ndimage as ndi

    from experiments.openlin_real1 import M as MET
    out, img, dem, valid, meta = build("utsjoki", log, win=WIN5)
    M = C1.masks(out, valid.shape)
    save("utsjoki", out, M, meta)
    shape = valid.shape
    l6 = np.zeros(shape, bool)
    a = np.load(OUT / "linb6_utsjoki.npz")["ids"] > 0
    h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    l6[:h, :w] = a[:h, :w]
    old = np.zeros(shape, bool)
    for tag in ("_настройка1", "_рельеф1"):
        p = np.load(MET / f"openlin_zone_lines_utsjoki_v3{tag}_линии.npz")["A_all_pts"].astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < shape[1])
        old[p[ok, 0], p[ok, 1]] = True
    pool, pool_ch, b_res = C1.describe("utsjoki", out, M, valid, meta, log, l6=l6, old_pool=old)
    z1 = np.load(OUT / "cand1_utsjoki.npz")
    c1pool = np.zeros(shape, bool)
    for ch in range(1, 6):
        c1pool |= z1[f"ids_{ch}"][:shape[0], :shape[1]] > 0
    log(f"  CAND1 в 127 м от CAND2: {100 * C1.cover(c1pool, pool, 127):.1f} %; "
        f"CAND2 в 127 м от CAND1: {100 * C1.cover(pool, c1pool, 127):.1f} %")
    gtk = C1.gtk_mask("utsjoki", meta, shape, valid)
    log("=== (в) GTK 292 в 300 м")
    for ch in out:
        log(f"  {C1.CHANNELS[ch]}: {100 * C1.cover(gtk, M[ch][0] > 0, C1.TOL_GTK_M):.1f} %")
    c = C1.cover(gtk, pool, C1.TOL_GTK_M)
    c1 = c >= C1.GTK_OLD_POOL
    log(f"(в1) пул CAND2 {100 * c:.1f} % >= {100 * C1.GTK_OLD_POOL:.1f} % -> {'да' if c1 else 'НЕТ'}")
    cv, km_mean = C1.thinned_cover(M, pool_ch, gtk, C1.OLD_POOL_KM, C1.N_THIN, C1.SEED0)
    c2 = float(np.mean(cv)) >= C1.GTK_OLD_POOL
    own_km = sum(int(np.count_nonzero(M[ch][0])) for ch in pool_ch) * C1.RES_IN / 1000
    log(f"  длина пула CAND2 по сумме ячеек линий {own_km:.0f} км; "
        + ("< 6826 км: прореживания нет, взят весь пул — сырая полнота при своей длине (поправка 1 п. 3)"
           if own_km < C1.OLD_POOL_KM else ">= 6826 км: случайное прореживание до 6826 км"))
    log(f"(в2) при равной длине ({C1.OLD_POOL_KM:.0f} км по сумме ячеек линий; взято в среднем {km_mean:.0f} км, "
        f"{C1.N_THIN} повторов): полнота {100 * np.mean(cv):.1f} % [мин {100 * np.min(cv):.1f}, "
        f"макс {100 * np.max(cv):.1f}] >= {100 * C1.GTK_OLD_POOL:.1f} % -> {'да' if c2 else 'НЕТ'}")
    b_txt = ", ".join(f"{C1.CHANNELS[ch] if ch != 'пул' else 'пул'} "
                      f"{'не оценено' if v is None else 'да' if v else 'НЕТ'}" for ch, v in b_res.items())
    log(f"ИТОГ CAND2 Утсйоки: (б) {b_txt}; (в1) {'да' if c1 else 'НЕТ'}; (в2) {'да' if c2 else 'НЕТ'}"
        f" -> {'ПРИНЯТО' if c1 and c2 else 'НЕ ПРИНЯТО'}")
    _ = ndi
    windows("utsjoki", img, dem, M, valid, log, win_km=6)


def anabar(log):
    out, img, dem, valid, meta = build("anabar", log)
    M = C1.masks(out, valid.shape)
    save("anabar", out, M, meta)
    shape = valid.shape
    l6 = np.zeros(shape, bool)
    a = np.load(OUT / "linb6_anabar.npz")["ids"] > 0
    h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    l6[:h, :w] = a[:h, :w]
    C1.describe("anabar", out, M, valid, meta, log, l6=l6)
    windows("anabar", img, dem, M, valid, log, win_km=8)


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_len1 import sha
    if "--utsjoki" in sys.argv:
        log = log_to(OUT / "_cand2_utsjoki.log")
        log(f"CAND2 Утсйоки; audit/CAND2_DECLARATION.md; sha256 кода {sha(__file__)}, CAND1 {sha(C1.__file__)}")
        utsjoki(log)
    elif "--anabar" in sys.argv:
        log = log_to(OUT / "_cand2_anabar.log")
        log(f"CAND2 Анабар (описательно); sha256 кода {sha(__file__)}, CAND1 {sha(C1.__file__)}")
        anabar(log)


if __name__ == "__main__":
    main()
