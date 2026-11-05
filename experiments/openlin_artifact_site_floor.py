"""Проверочный (не тракт): пол кривизны трассировки и азимутальный отклик на изотропном фоне.

1. ПОЛ УЛЬТРАПРЯМЫХ. В реальный композит врезаются заведомо прямые уступы 4 км
   (больше порога LONG_KM = 3 км батареи) на 12 азимутах; тот же тракт
   (trace + extend_ends, как в openlin_azimuth_response) их находит, и
   меряется кривизна найденной линии по формуле батареи. «Ни одной ультрапрямой»
   что-то значит, только если доля пойманных врезок с кривизной < 0.5 % высока.
2. ИЗОТРОПНЫЙ ФОН. Поле со случайными фазами и РАДИАЛЬНО усреднённым спектром
   композита (анизотропия снята): азимутальная гистограмма ложных линий при
   амплитуде 0 и обнаружение врезок по азимутам при амплитуде 8 sigma. Если
   провал по азимуту остаётся и здесь — это код или растр, а не освещение.
Запуск:  python -m experiments.openlin_artifact_site_floor --site utsjoki
"""
from __future__ import annotations
import time
import numpy as np

import experiments.openlin_azimuth_response as ar
from experiments.openlin_zone_lines import COMP, MIN_LEN_M, trace
from experiments.openlin_zone_lines_v3 import extend_ends

import sys
SITE = sys.argv[sys.argv.index("--site") + 1] if "--site" in sys.argv else "utsjoki"
RELIEF = "--relief" in sys.argv   # вход — уклон GLO-30, отклик scarp (как в ключе --рельеф)
AMP = 8.0
LEN_M = 4000.0


def curvature(pts, res=20.0):
    q = pts.astype(float) - pts.mean(0)
    vt = np.linalg.svd(q, full_matrices=False)[2]
    t = q @ vt[0]
    ln = (t.max() - t.min()) * res
    return float(np.abs(q @ vt[1]).mean()) * res / ln, ln / 1000


def iso_field(img, valid, rng):
    from scipy.fft import irfft2, rfft2
    x = np.where(valid, img, np.nanmedian(img)).astype("float32")
    x = x - x.mean()
    F = rfft2(x)
    P = np.abs(F) ** 2
    H, W = x.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    k = np.hypot(ky, kx)
    bins = np.minimum((k / k.max() * 400).astype(int), 399)
    rad = np.bincount(bins.ravel(), P.ravel(), 400) / np.maximum(np.bincount(bins.ravel(), minlength=400), 1)
    amp = np.sqrt(rad[bins]).astype("float32")
    noise = rfft2(rng.standard_normal((H, W)).astype("float32"))
    ph = noise / np.maximum(np.abs(noise), 1e-12)
    y = irfft2(amp * ph, s=(H, W)).astype("float32")
    return y * (x.std() / y.std()) + float(np.nanmedian(img))


def main():
    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize
    from scipy.spatial import cKDTree
    from src.detection_limit import robust_sigma
    ar.INS_LEN_M = LEN_M
    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{SITE}_20m_лист.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    rfn = None
    if RELIEF:
        from experiments.openlin_relief_lines import load_slopes, scarp_response
        img = load_slopes(SITE, res_m)[0].astype("float32")
        rfn = scarp_response
        print("релиф: вход — уклон GLO-30, отклик scarp", flush=True)
    valid = np.isfinite(img)
    sigma = robust_sigma(img[valid])
    rng = np.random.default_rng(ar.SEED)
    grid = ar.sites(img.shape, res_m, valid, rng)
    need = ar.AZ_BINS * min(ar.N_PER_BIN, len(grid) // ar.AZ_BINS)
    az = np.array([(b + 0.5) * 180.0 / ar.AZ_BINS for b in range(ar.AZ_BINS)])
    plan = [(az[i % ar.AZ_BINS], grid[i]) for i in range(need)]
    print(f"sigma {sigma:.4f}, врезок {need}, длина {LEN_M / 1000:.0f} км", flush=True)

    def run(bg, amp):
        x = bg.astype("float64").copy()
        tracks = []
        for a_deg, ctr in plan:
            if amp > 0:
                ar.insert_window(x, ctr, a_deg, amp, sigma, res_m)
            tracks.append((ar.track(img.shape, ctr, a_deg, res_m), a_deg))
        xx = np.where(valid, x, np.nan).astype("float32")
        lines = trace(xx, valid, res_m, local=True, min_len_m=0, response_fn=rfn)
        zz = local_norm(np.where(valid, (rfn or ridge_response)(np.where(
            valid, standardize(xx, valid), 0.0).astype("float32")), 0.0), valid, res_m)
        lines = extend_ends(lines, zz, valid, res_m)
        lines = [r for r in lines if r["длина_км"] * 1000 >= MIN_LEN_M]
        return lines, tracks

    def match(lines, tracks):
        pts = np.concatenate([r["pts"] for r in lines])
        own = np.concatenate([np.full(len(r["pts"]), i) for i, r in enumerate(lines)])
        tree = cKDTree(pts.astype(float))
        out = []
        for tr_, a in tracks:
            hits = tree.query_ball_point(tr_.astype(float), r=ar.MATCH_TOL_M / res_m)
            cnt = {}
            for h in hits:
                for j in np.unique(own[np.asarray(h, int)]) if len(h) else []:
                    cnt[int(j)] = cnt.get(int(j), 0) + 1
            if not cnt:
                out.append(None); continue
            b = max(cnt, key=cnt.get)
            ok = cnt[b] / len(tr_) >= ar.MATCH_FRAC and \
                abs(((lines[b]["азимут"] - a) + 90.0) % 180.0 - 90.0) <= ar.MATCH_TOL_DEG
            out.append(b if ok else None)
        return out

    # 1. пол кривизны на реальном фоне
    lines, tracks = run(img, AMP)
    m = match(lines, tracks)
    ks = []
    for b in m:
        if b is None:
            ks.append((np.nan, np.nan))
        else:
            ks.append(curvature(lines[b]["pts"], res_m))
    ks = np.array(ks)
    got = np.isfinite(ks[:, 0])
    print(f"\n1. реальный фон, {AMP} sigma: поймано {got.sum()} из {need}; "
          f"кривизна найденных 10/50/90 % = "
          + "/".join(f"{100 * np.percentile(ks[got, 0], q):.2f}" for q in (10, 50, 90))
          + " %; длина найденной 10/50/90 % = "
          + "/".join(f"{np.percentile(ks[got, 1], q):.1f}" for q in (10, 50, 90)) + " км", flush=True)
    ult = got & (ks[:, 0] < 0.005) & (ks[:, 1] >= 3.0)
    print(f"   из {need} врезок с кривизной < 0.5 % и длиной >= 3 км: {ult.sum()} "
          f"({100 * ult.sum() / need:.0f} %) — столько ультрапрямых тракт ДАЁТ на заведомо прямой врезке", flush=True)
    for b in range(ar.AZ_BINS):
        sel = np.arange(need) % ar.AZ_BINS == b
        print(f"   азимут {az[b]:5.1f}: ультрапрямых {ult[sel].sum()}/{sel.sum()}, "
              f"медиана кривизны {100 * np.nanmedian(ks[sel, 0]):.2f} %", flush=True)

    # 2. изотропный фон
    iso = iso_field(img, valid, np.random.default_rng(5))
    bins = np.arange(need) % ar.AZ_BINS
    for amp in (0.0, AMP):
        lines, tracks = run(iso, amp)
        azs = np.array([r["азимут"] for r in lines]); ln = np.array([r["длина_км"] for r in lines])
        h = np.histogram(azs, bins=12, range=(0, 180), weights=ln)[0]
        print(f"\n2. изотропный фон, амплитуда {amp}: линий {len(lines)} ({ln.sum():.0f} км); "
              f"доли длины по секторам 15°: " + " ".join(f"{100 * v / h.sum():.0f}" for v in h), flush=True)
        if amp > 0:
            m = match(lines, tracks)
            hit = np.array([b is not None for b in m])
            print("   обнаружение по азимутам, %: " + " ".join(
                f"{100 * hit[bins == b].mean():.0f}" for b in range(ar.AZ_BINS)), flush=True)
    print(f"готово ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
