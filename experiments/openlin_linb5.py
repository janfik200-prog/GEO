"""LINB5: LINB2 без сетки 100 м — та же модель, P на 20 м по 25 сдвинутым сеткам 100 м,
трассировка (directional, NMS, скелет, ветви, сшивка) на 20 м (audit/LINB5_DECLARATION.md, поправки 1–2).

    python -m experiments.openlin_linb5 --utsjoki     # настройка: критерии (а), (б), (в)
    python -m experiments.openlin_linb5 --anabar      # цель: только картинки и описание
LINB5_N_TESTS = 0. Лидар не используется.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import ndimage as ndi

from experiments.openlin_fam import (CELL_M, DEM, DEM_SIG_M, FAM_NAME, GRAD_SIG_M, IMG_SIG_M,
                                     LINE_LEN_M, N_DIR, OUT, RES_IN, STD_WIN_M, all_windows, block,
                                     features, multi_ridge, straightness, win_mask)
from experiments.openlin_zone_lines import DIR_TOL_DEG
from experiments.openlin_zone_lines_v3 import LINK_MIN_LEN_M, TWIN_OFFSET_M, V3_GAP_M

LINB5_N_TESTS = 0
K = int(CELL_M / RES_IN)                # 5 ячеек 20 м в ячейке 100 м
END_PX = int(300 / RES_IN)              # 300 м = 3 ячейки 100 м прежде (поправка 1 п. 1)
PHASE_MAX = 1.041                       # поправка 2 п. 2 (а)
ISO_AXES = 2 * 6 / 180                  # 6,7 %
BUDGET_U_KM = 2858.0                    # RELU1 Утсйоки, растр (LINB2)
SHORT_M = 620                           # только разбиение для (б), не фильтр


# ---------------------------------------------------------------- P на 20 м
def maps20(site, log):
    """Те же карты 20 м, что в openlin_fam._features (до блоков)."""
    from experiments.openlin_transfer_check import COMP
    z = np.load(COMP / f"winter_{site}_20m_лист.npz")
    img = z["shade"].astype("float64")
    valid20 = (z["count"] > 0) & np.isfinite(img)
    dem = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy").astype("float64")
    valid20 &= np.isfinite(dem)
    fill = lambda a: np.where(np.isfinite(a), a, np.nanmedian(a))
    img, dem = fill(img), fill(dem)
    dark, bright = multi_ridge(img, IMG_SIG_M)
    grad = ndi.gaussian_gradient_magnitude(img, GRAD_SIG_M / RES_IN)
    w = int(STD_WIN_M / RES_IN)
    m1 = ndi.uniform_filter(img, w)
    std = np.sqrt(np.maximum(ndi.uniform_filter(img ** 2, w) - m1 ** 2, 0))
    gy, gx = np.gradient(dem, RES_IN)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    conc, conv = multi_ridge(dem, DEM_SIG_M)
    log("  карты 20 м посчитаны")
    return [(dark, "max"), (bright, "max"), (grad, "mean"), (std, "mean"), (slope, "mean"),
            (conc, "max"), (conv, "max")], valid20


def feats_shift(mp, valid20, dr, dc):
    """Признаки 1–9 на сетке 100 м, сдвинутой на (dr, dc) ячеек 20 м — как _features."""
    F = [block(x[dr:, dc:], K, how) for x, how in mp]
    F.append(straightness(F[0]))
    F.append(straightness(F[5]))
    valid = block(valid20[dr:, dc:].astype(float), K, "mean") >= 1.0
    return np.stack(F, -1).astype("float32"), valid


def score20(site, log):
    """P на 20 м: модель на 25 сдвигах; квантили — по несдвинутой сетке (как score_map)."""
    import cloudpickle
    from sklearn.preprocessing import QuantileTransformer
    F0, valid0, meta = features(site, log)
    wins, n = all_windows(valid0)
    fit = win_mask(valid0.shape, wins, n) & valid0
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=0)
    qt.fit(F0[fit])
    preds = cloudpickle.load(open(OUT / "frozen_fam3.pkl", "rb"))["models"][FAM_NAME]
    mp, valid20 = maps20(site, log)
    H, W = valid20.shape
    P20 = np.full((H, W), np.nan, "float32")
    for dr in range(K):
        for dc in range(K):
            F, v = feats_shift(mp, valid20, dr, dc)
            if dr == 0 and dc == 0:                      # проверка: тот же вход, что LINB2
                assert F.shape == F0.shape and np.array_equal(v, valid0)
                assert np.allclose(F[v], F0[v], rtol=1e-5, atol=1e-6), "сдвиг (0,0) не совпал с features()"
            flat = F.reshape(-1, 9)
            ok = v.ravel() & np.all(np.isfinite(flat), 1)
            Z = qt.transform(flat[ok])
            p = np.full(flat.shape[0], np.nan, "float32")
            p[ok] = np.mean([f(Z) for f in preds], 0)
            p = p.reshape(F.shape[:2])
            P20[dr + 2:dr + 2 + K * p.shape[0]:K, dc + 2:dc + 2 + K * p.shape[1]:K] = p
        log(f"  P 20 м: сдвиги строки {dr} готовы")
    P0 = P20[2::K, 2::K][:valid0.shape[0], :valid0.shape[1]]
    from experiments.openlin_anfisa_lines import score_map
    Pref, vref, _ = score_map(site, log)
    d = np.nanmax(np.abs(P0[vref] - Pref[vref]))
    log(f"  проверка: P сдвига (0,0) против score_map LINB2 — макс. расхождение {d:.2e}")
    assert d < 1e-4
    valid = np.isfinite(P20)
    log(f"  P 20 м: ячеек {int(valid.sum())}, P > 0 у {100 * np.nanmean(P20 > 0):.1f} %")
    return P20, valid, valid20


# ---------------------------------------------------------------- трассировка на 20 м
def line_offsets(L, th):
    half = L // 2
    pts = set()
    for t in np.linspace(-half, half, 4 * L):
        # те же точки, что ядро directional/straightness: round(half - t cos) - half (ничьи округляются
        # так же), свёртка ndi.convolve сэмплирует f[x - o], поэтому смещение = -o
        o = (int(round(half - t * np.cos(th))) - half, int(round(half + t * np.sin(th))) - half)
        pts.add((-o[0], -o[1]))
    return sorted(pts)


def directional20(P, valid):
    """Как openlin_anfisa_lines.directional, ядро LINE_LEN_M = 55 ячеек 20 м (сдвиги вместо свёртки)."""
    L = int(round(LINE_LEN_M / RES_IN))
    half = L // 2
    Pf = np.where(valid, P, np.nanpercentile(P, 1)).astype("float32")
    pad = np.pad(Pf, half, mode="edge")
    H, W = Pf.shape
    best = np.full((H, W), -np.inf, "float32")
    arg = np.zeros((H, W), np.int8)
    for i in range(N_DIR):
        offs = line_offsets(L, np.pi * i / N_DIR)
        acc = np.zeros((H, W), "float32")
        for dr, dc in offs:
            acc += pad[half + dr:half + dr + H, half + dc:half + dc + W]
        acc /= len(offs)
        m = acc > best
        best[m], arg[m] = acc[m], i
    return best, np.pi * arg / N_DIR


def length_px_m(pl):
    return float(np.hypot(*np.diff(pl, axis=0).T).sum() * RES_IN) if len(pl) > 1 else 0.0


def _end(pl, head):
    k = min(END_PX, len(pl) - 1)
    p, q = (pl[0], pl[k]) if head else (pl[-1], pl[-1 - k])
    v = p - q
    nv = np.hypot(*v)
    return p, (v / nv if nv else v)


def stitch20(lines, log):
    """openlin_anfisa_lines.stitch с порогами в ячейках 20 м и концом 300 м."""
    from experiments import openlin_anfisa_lines as A
    old = (A.CELL_M, A._end)
    A.CELL_M, A._end = RES_IN, _end
    try:
        return A.stitch(lines, log)
    finally:
        A.CELL_M, A._end = old


def build20(site, log):
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches, nms
    P, valid, valid20 = score20(site, log)
    D, th = directional20(P, valid)
    keep = (D > 0) & valid & nms(D, th)
    log(f"  D > 0: {100 * ((D > 0) & valid).sum() / valid.sum():.1f} % ячеек; после NMS {int(keep.sum())}")
    sk = skeletonize(keep)
    br = branches(sk)
    log(f"  скелет {int(sk.sum())} ячеек, ветвей {len(br)}")
    lines = stitch20(br, log)
    lines = [pl for pl in lines if length_px_m(pl) >= LINK_MIN_LEN_M]
    L = np.array([length_px_m(pl) for pl in lines]) / 1000
    log(f"  линий >= {LINK_MIN_LEN_M:.0f} м: {len(lines)}, {L.sum():.0f} км (ломаная); медиана "
        f"{np.median(L):.2f} км, 90-й проц. {np.percentile(L, 90):.2f}, макс {L.max():.1f}")
    return lines, P, valid20


def select(lines, P, valid20, budget_km, log, share=None):
    """Отбор как LINB2: по убыванию среднего P, длины, номера; бюджет — км растра."""
    from experiments.openlin_anfisa_lines import line_mean_p, rasterize
    ids_all = np.where(valid20, rasterize(lines, valid20.shape, 1), 0)
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    total = cells.sum() * RES_IN / 1000
    if share is not None:
        budget_km = share * total
    mp = np.array([line_mean_p(pl, P) for pl in lines])
    lenm = np.array([length_px_m(pl) for pl in lines])
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    cum = np.cumsum(cells[order]) * RES_IN / 1000
    n = int(np.searchsorted(cum, budget_km, side="right"))
    keep = np.zeros(len(lines) + 1, bool)
    keep[order[:n] + 1] = True
    log(f"  все линии {len(lines)}, {total:.0f} км (растр); бюджет {budget_km:.0f} км: линий {n}, "
        f"порог среднего P {mp[order[n - 1]]:.3f}; доля бюджета {budget_km / total:.4f}")
    return np.where(keep[ids_all], ids_all, 0), total


def as_lines(ids):
    from experiments.openlin_real1 import lines_from_pts
    rr, cc = np.nonzero(ids)
    return [x for x in lines_from_pts(np.stack([rr, cc], 1), ids[rr, cc]) if len(x) >= 2]


# ---------------------------------------------------------------- проверки (а), (б)
def grid_stats(lines, log, label):
    P = np.concatenate(lines)
    ph = [np.bincount(P[:, a] % 5, minlength=5) / len(P) / 0.2 for a in (0, 1)]
    mx = float(max(x.max() for x in ph))
    out = {"фаза": mx}
    for lab, sel in (("<620", lambda n: n * RES_IN < SHORT_M), (">=620", lambda n: n * RES_IN >= SHORT_M)):
        az = []
        for x in lines:
            if sel(len(x)):
                q = x - x.mean(0)
                _w, v = np.linalg.eigh(q.T @ q)
                t = v[:, -1]
                az.append(np.degrees(np.arctan2(t[1], -t[0])) % 180)
        az = np.array(az)
        ax = float(np.mean((np.minimum(az, 180 - az) <= 3) | (np.abs(az - 90) <= 3))) if len(az) else np.nan
        dg = float(np.mean((np.abs(az - 45) <= 3) | (np.abs(az - 135) <= 3))) if len(az) else np.nan
        out[lab] = (len(az), ax, dg)
    log(f"  {label:12s}: ячеек {len(P)}, фаза max/0,2 {mx:.3f} (строки {np.round(ph[0], 2).tolist()}, "
        f"столбцы {np.round(ph[1], 2).tolist()}); короткие n={out['<620'][0]} у осей {100 * out['<620'][1]:.1f} % "
        f"диаг {100 * out['<620'][2]:.1f} %; длинные n={out['>=620'][0]} у осей {100 * out['>=620'][1]:.1f} % "
        f"диаг {100 * out['>=620'][2]:.1f} %")
    return out


def utsjoki(log):
    from experiments.openlin_real1 import M, load_sets, witnesses
    from experiments.openlin_real2 import BLOCKS_PX, N_BOOT, pi_hat, prep_witness
    from experiments.openlin_len1 import line_t, lo_hi
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    lines, P, valid20 = build20("utsjoki", log)
    valid20 = valid20[:shape[0], :shape[1]]
    P = P[:shape[0], :shape[1]]
    ids, total = select(lines, P, valid20, BUDGET_U_KM, log)
    np.savez_compressed(OUT / "linb5_utsjoki.npz", ids=ids)
    share = BUDGET_U_KM / total
    (OUT / "linb5_utsjoki.json").write_text(json.dumps({"share": share, "total_km": total}), encoding="utf-8")
    L5 = as_lines(ids)
    sets = load_sets(shape)
    log("=== (а), (б): сетка и азимуты")
    st = {k: grid_stats(v, log, k.split()[-1]) for k, v in sets.items()}
    s5 = grid_stats(L5, log, "LINB5")
    a_ok = s5["фаза"] <= PHASE_MAX
    n, ax, _ = s5["<620"]
    thr_b = ISO_AXES + 2 * np.sqrt(ISO_AXES * (1 - ISO_AXES) / max(n, 1))
    log(f"(а) фаза LINB5 {s5['фаза']:.3f} <= {PHASE_MAX} -> {'да' if a_ok else 'НЕТ'}")
    if n < 100:
        b_ok = None
        log(f"(б) коротких хорд {n} < 100 — не оценивается")
    else:
        b_ok = ax <= thr_b
        log(f"(б) короткие у осей {100 * ax:.1f} % <= {100 * thr_b:.1f} % (6,7 % + 2·SE, n={n}) -> {'да' if b_ok else 'НЕТ'}")
    log("=== (в): мера REAL3, Landsat лето (критерий) и S2 лето (справка)")
    W = witnesses(shape)
    Wp = {wn: prep_witness(W[wn], shape) for wn in ("Landsat лето", "S2 лето")}
    L2 = sets["Анфиса LINB2"]
    c_ok = None
    for wn in ("Landsat лето", "S2 лето"):
        w2, f2, c2, _ = line_t(L2, Wp, wn, shape)
        w5, f5, c5, _ = line_t(L5, Wp, wn, shape)
        for lab, w, f, c in (("LINB2", w2, f2, c2), ("LINB5", w5, f5, c5)):
            lo, hi = lo_hi(w[f], c[f])
            log(f"  {wn} {lab}: линий {int(f.sum())}, π̂ {100 * pi_hat(w[f]):+.1f} % [{100 * lo:+.1f}…{100 * hi:+.1f}]")
        b2 = (c2[f2, 0] // BLOCKS_PX[0]).astype(int) * 100000 + (c2[f2, 1] // BLOCKS_PX[0]).astype(int)
        b5 = (c5[f5, 0] // BLOCKS_PX[0]).astype(int) * 100000 + (c5[f5, 1] // BLOCKS_PX[0]).astype(int)
        ub = np.union1d(b2, b5)
        g2 = {k: w2[f2][b2 == k] for k in ub}
        g5 = {k: w5[f5][b5 == k] for k in ub}
        los, his = [], []
        for sd in (0, 1):
            rng = np.random.default_rng(sd)
            vals = []
            for _ in range(N_BOOT):
                s = rng.choice(ub, len(ub), replace=True)
                vals.append(pi_hat(np.concatenate([g5[k] for k in s])) - pi_hat(np.concatenate([g2[k] for k in s])))
            lo, hi = np.percentile(vals, [2.5, 97.5])
            los.append(lo)
            his.append(hi)
        d = pi_hat(w5[f5]) - pi_hat(w2[f2])
        log(f"  {wn}: разность LINB5 − LINB2 {100 * d:+.1f} п. п. [{100 * min(los):+.1f}…{100 * max(his):+.1f}]")
        if wn == "Landsat лето":
            c_ok = min(los) >= -0.03
            log(f"(в) нижний край {100 * min(los):+.1f} >= −3 п. п. -> {'да' if c_ok else 'НЕТ'}")
    verdict = ("сетка не убрана" if not a_ok or b_ok is False else
               "без сетки хуже по мере" if not c_ok else "ПРИНЯТО: не хуже наборов на 20 м, мера не хуже")
    if b_ok is None and verdict.startswith("ПРИНЯТО"):
        verdict = "(а) и (в) пройдены, (б) не оценивается (n < 100)"
    log(f"ИТОГ LINB5 Утсйоки: {verdict}")
    windows_pics("utsjoki", L2, L5, shape, log)


def windows_pics(site, L2, L5, shape, log, n_win=(3, 3), win_km=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_transfer_check import COMP
    out = OUT / f"linb5_{site}"
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
            fig, axs = plt.subplots(1, 3, figsize=(24, 8.4))
            b = img[r0:r0 + wp, c0:c0 + wp]
            lo, hi = np.nanpercentile(b, [2, 98])
            for ax, (title, L) in zip(axs, (("зимний S2, без линий", None), ("LINB2 (сетка 100 м)", L2),
                                            ("LINB5 (трассировка 20 м)", L5))):
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
            fig.savefig(p, dpi=70)
            plt.close(fig)
            log(f"  {p}")


def anabar(log):
    from experiments.openlin_transfer_check import COMP
    share = json.loads((OUT / "linb5_utsjoki.json").read_text(encoding="utf-8"))["share"]
    lines, P, valid20 = build20("anabar", log)
    ids, total = select(lines, P, valid20, None, log, share=share)
    np.savez_compressed(OUT / "linb5_anabar.npz", ids=ids)
    L5 = as_lines(ids)
    ids2 = np.load(OUT / "anabar_linb2" / "linb2_anabar.npz")["ids"]
    L2 = as_lines(ids2)
    shape = valid20.shape
    log(f"  LINB5 Анабар: линий {len(L5)}, {sum(len(x) for x in L5) * RES_IN / 1000:.0f} км (ячейки); "
        f"LINB2: линий {len(L2)}")
    log("=== описание (а), (б), без критериев")
    grid_stats(L2, log, "LINB2")
    grid_stats(L5, log, "LINB5")
    windows_pics("anabar", L2, L5, shape, log, win_km=8)
    _ = COMP


def main():
    from experiments.openlin_fam import log_to
    if "--utsjoki" in sys.argv:
        log = log_to(OUT / "_linb5_utsjoki.log")
        log("LINB5 Утсйоки; объявление audit/LINB5_DECLARATION.md (поправки 1–2)")
        utsjoki(log)
    elif "--anabar" in sys.argv:
        log = log_to(OUT / "_linb5_anabar.log")
        log("LINB5 Анабар (описательно, картинки)")
        anabar(log)


if __name__ == "__main__":
    main()
