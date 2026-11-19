"""CAND1: кандидаты в линии для Анфисы — гребни и перепады снимка и рельефа на 20 м
(audit/CAND1_DECLARATION.md). Анфиса потом выбирает из них (ANFL1, на разметке геологов).

    python -m experiments.openlin_cand1 --utsjoki     # настройка: (а), (б), (в)
    python -m experiments.openlin_cand1 --anabar      # цель: картинки и описание
CAND1_N_TESTS = 0. Лидар не используется. Только процессор.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage as ndi

from experiments.openlin_iso import Q_HIGH, Q_LOW          # 97/90 % рабочего тракта; квантили берутся по подделке
from experiments.openlin_fam import DEM, DEM_SIG_M, GRAD_SIG_M, IMG_SIG_M, OUT, RES_IN
from experiments.openlin_zone_lines_v3 import LINK_MIN_LEN_M, TWIN_OFFSET_M

CAND1_N_TESTS = 0
JUNK_MAX = 0.20                     # мусор ветки (OPENLIN_RULES п. 3)
GTK_OLD_POOL = 0.680                # прежний пул снимок+рельеф A_all, замер pool_probe.py
OLD_POOL_KM = 6826.0                # его длина (объединение ячеек), тот же замер — (в2), поправка 1 п. 3
N_THIN = 20                         # повторов прореживания (в2), поправка 1 п. 3
TOL_GTK_M = 300.0                   # openlin_transfer_check.TOL_M
SEED0 = 1900
SHORT_M = 620                       # разбиение для (б), как LINB5
CHANNELS = ("тёмная полоса", "светлая полоса", "перепад снимка", "ложбина рельефа", "гребень рельефа")


# ---------------------------------------------------------------- входы
def load_inputs(site):
    from experiments.openlin_transfer_check import COMP
    z = np.load(COMP / f"winter_{site}_20m_лист.npz")
    img = z["shade"].astype("float64")
    valid = (z["count"] > 0) & np.isfinite(img)
    dem = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy").astype("float64")
    H, W = min(img.shape[0], dem.shape[0]), min(img.shape[1], dem.shape[1])
    img, dem, valid = img[:H, :W], dem[:H, :W], valid[:H, :W] & np.isfinite(dem[:H, :W])
    fill = lambda a: np.where(np.isfinite(a), a, np.nanmedian(a))
    meta = {k: float(z[k]) for k in ("res_m", "x0", "y_top", "epsg")}
    return fill(img), fill(dem), valid, meta


# ---------------------------------------------------------------- отклик и нормаль
def ridge_dir(x, sigs_m, sign):
    """Многомасштабный гессиан как openlin_fam.multi_ridge; sign=+1 — тёмная полоса/ложбина (λmax > 0),
    -1 — светлая/гребень (λmin < 0). Возвращает отклик и угол θ для nms (нормаль = (sin θ, cos θ))."""
    best = np.zeros(x.shape)
    th = np.zeros(x.shape)
    for s_m in sigs_m:
        s = s_m / RES_IN
        hrr = ndi.gaussian_filter(x, s, order=(2, 0))
        hcc = ndi.gaussian_filter(x, s, order=(0, 2))
        hrc = ndi.gaussian_filter(x, s, order=(1, 1))
        tr, det = hrr + hcc, hrr * hcc - hrc ** 2
        disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        lam = tr / 2 + disc if sign > 0 else tr / 2 - disc
        r = np.maximum(sign * lam, 0) * s ** 2
        # собственный вектор λ: (hrc, λ − hrr) или (λ − hcc, hrc) — берём более длинный
        a_r, a_c = hrc, lam - hrr
        b_r, b_c = lam - hcc, hrc
        use_a = np.hypot(a_r, a_c) >= np.hypot(b_r, b_c)
        nr, nc = np.where(use_a, a_r, b_r), np.where(use_a, a_c, b_c)
        m = r > best
        best[m] = r[m]
        th[m] = np.arctan2(nr, nc)[m]
    return best, th


def edge_dir(x):
    g = ndi.gaussian_gradient_magnitude(x, GRAD_SIG_M / RES_IN)
    gr = ndi.gaussian_filter(x, GRAD_SIG_M / RES_IN, order=(1, 0))
    gc = ndi.gaussian_filter(x, GRAD_SIG_M / RES_IN, order=(0, 1))
    return g, np.arctan2(gr, gc)


def response(ch, img, dem):
    if ch == 0:
        return ridge_dir(img, IMG_SIG_M, +1)
    if ch == 1:
        return ridge_dir(img, IMG_SIG_M, -1)
    if ch == 2:
        return edge_dir(img)
    if ch == 3:
        return ridge_dir(dem, DEM_SIG_M, +1)
    return ridge_dir(dem, DEM_SIG_M, -1)


# ---------------------------------------------------------------- трассировка
def trace(r, th, valid, lo, hi, log, label):
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches, nms
    from experiments.openlin_iso import hysteresis8
    from experiments.openlin_linb5 import length_px_m, stitch20
    thin = np.where(nms(r, th) & valid, r, 0.0)
    keep = hysteresis8(thin, lo, hi) & valid
    sk = skeletonize(keep)
    br = branches(sk)
    lines = stitch20(br, lambda *a: None)
    lines = [pl for pl in lines if length_px_m(pl) >= LINK_MIN_LEN_M]
    km = sum(length_px_m(pl) for pl in lines) / 1000
    log(f"    {label}: после nms+гистерезиса {int(keep.sum())} ячеек, ветвей {len(br)}, "
        f"линий >= {LINK_MIN_LEN_M:.0f} м {len(lines)}, {km:.0f} км (ломаная)")
    return lines, km


def channel(ch, img, dem, valid, fake_img, fake_dem, log):
    name = CHANNELS[ch]
    log(f"  канал {ch + 1} «{name}»")
    rf, thf = response(ch, fake_img, fake_dem)
    v = rf[valid]
    hi, lo = float(np.quantile(v, Q_HIGH)), float(np.quantile(v, Q_LOW))
    r, th = response(ch, img, dem)
    log(f"    пороги по подделке: {Q_HIGH:.0%} {hi:.4g}, {Q_LOW:.0%} {lo:.4g}; "
        f"доля ячеек снимка выше верхнего {100 * np.mean(r[valid] >= hi):.1f} %")
    lines, km = trace(r, th, valid, lo, hi, log, "снимок")
    _fl, kmf = trace(rf, thf, valid, lo, hi, log, "подделка")
    junk = kmf / km if km else np.inf
    ok = True                                        # поправка 2: (а) — справка, все каналы в пуле
    log(f"    (а) справка: км подделки / км снимка {100 * junk:.1f} % (условие ветки {100 * JUNK_MAX:.0f} % "
        f"недостижимо по замеру до прогона, поправка 2)")
    return lines, km, junk, ok


def build(site, log):
    from experiments.openlin_null_input import phase_randomize
    img, dem, valid, meta = load_inputs(site)
    log(f"  лист {site}: {img.shape}, годных ячеек {int(valid.sum())}")
    out = {}
    for ch in range(len(CHANNELS)):
        rng = np.random.default_rng(SEED0 + ch)
        fi = phase_randomize(img, valid, rng).astype("float64") if ch < 3 else img
        fd = phase_randomize(dem, valid, rng).astype("float64") if ch >= 3 else dem
        out[ch] = channel(ch, img, dem, valid, fi, fd, log)
    return out, img, dem, valid, meta


# ---------------------------------------------------------------- описание
def masks(out, shape):
    from experiments.openlin_anfisa_lines import rasterize
    from experiments.openlin_linb5 import as_lines
    res = {}
    for ch, (lines, _km, _j, _ok) in out.items():
        ids = rasterize(lines, shape, 1)
        res[ch] = (ids, as_lines(ids))
    return res


def gtk_mask(site, meta, shape, valid):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    from experiments.openlin_fam import gtk_file
    tr = Affine(RES_IN, 0, meta["x0"], 0, -RES_IN, meta["y_top"])
    g = gpd.read_file(gtk_file(site)).to_crs(epsg=int(meta["epsg"]))
    m = rasterize([(x, 1) for x in g.geometry if x is not None], out_shape=shape, transform=tr, fill=0,
                  all_touched=True, dtype="uint8").astype(bool)
    return m & valid


def thinned_cover(M, pool_ch, gtk, km_target, n_rep, seed):
    """(в2): случайные линии пула (все каналы вместе), пока сумма их ячеек не дойдёт до km_target;
    полнота GTK в 300 м по объединению взятых. n_rep повторов, одно зерно."""
    items = [(ch, j) for ch in pool_ch for j in range(1, int(M[ch][0].max()) + 1)]
    size = {ch: np.bincount(M[ch][0].ravel(), minlength=int(M[ch][0].max()) + 1) for ch in pool_ch}
    cells = np.array([size[ch][j] for ch, j in items])
    rng = np.random.default_rng(seed)
    out, kms = [], []
    for _ in range(n_rep):
        perm = rng.permutation(len(items))
        cum = np.cumsum(cells[perm]) * RES_IN / 1000
        n = int(np.searchsorted(cum, km_target, side="right"))
        sel = np.zeros(gtk.shape, bool)
        for ch in pool_ch:
            keep = np.zeros(int(M[ch][0].max()) + 1, bool)
            js = [items[i][1] for i in perm[:n] if items[i][0] == ch]
            keep[js] = True
            keep[0] = False
            sel |= keep[M[ch][0]]
        out.append(cover(gtk, sel, TOL_GTK_M))
        kms.append(cum[n - 1] if n else 0.0)
    return np.array(out), float(np.mean(kms))


def cover(src, dst_mask, tol):
    d = ndi.distance_transform_edt(~dst_mask) * RES_IN
    return float(np.mean(d[src] <= tol)) if src.any() else np.nan


def multi_channel_share(M, log):
    """Доля кандидатов, у которых >= 50 % ячеек в 127 м от линий ДРУГОГО канала."""
    chs = list(M)
    dist = {ch: ndi.distance_transform_edt(M[ch][0] == 0) * RES_IN for ch in chs}
    for ch in chs:
        ids = M[ch][0]
        n = int(ids.max())
        if n == 0:
            continue
        near_any = np.zeros(ids.shape, bool)
        for o in chs:
            if o != ch:
                near_any |= dist[o] <= TWIN_OFFSET_M
        frac = np.bincount(ids.ravel(), weights=near_any.ravel(), minlength=n + 1)[1:] / \
            np.maximum(np.bincount(ids.ravel(), minlength=n + 1)[1:], 1)
        log(f"    {CHANNELS[ch]}: видны в другом канале {100 * np.mean(frac >= 0.5):.1f} % линий")


def describe(site, out, M, valid, meta, log, l6=None, old_pool=None):
    from experiments.openlin_linb5 import ISO_AXES, grid_stats
    area = valid.sum() * RES_IN ** 2 / 1e6
    pool_ch = [ch for ch in out if out[ch][3]]
    pool = np.zeros(valid.shape, bool)
    for ch in pool_ch:
        pool |= M[ch][0] > 0
    log(f"=== пул: каналы {[CHANNELS[c] for c in pool_ch]}, {pool.sum() * RES_IN / 1000:.0f} км (растр), "
        f"{pool.sum() * RES_IN / 1000 / area:.2f} км/км²")
    b_res = {}
    for ch in out:
        L = M[ch][1]
        km = sum(len(x) for x in L) * RES_IN / 1000
        log(f"  {CHANNELS[ch]}: линий {len(L)}, {km:.0f} км (растр), {km / area:.2f} км/км²")
        if L:
            s = grid_stats(L, log, f"к{ch + 1}")
            n, ax, _ = s["<620"]
            thr = ISO_AXES + 2 * np.sqrt(ISO_AXES * (1 - ISO_AXES) / max(n, 1))
            b_res[ch] = None if n < 100 else ax <= thr
            log(f"    (б) короткие у осей {100 * ax:.1f} % <= {100 * thr:.1f} % (n={n}) -> "
                + ("не оценивается" if b_res[ch] is None else "да" if b_res[ch] else "НЕТ — пиксельная ступенька"))
    Lpool = [x for ch in pool_ch for x in M[ch][1]]
    if Lpool:
        s = grid_stats(Lpool, log, "пул")
        n, ax, _ = s["<620"]
        thr = ISO_AXES + 2 * np.sqrt(ISO_AXES * (1 - ISO_AXES) / max(n, 1))
        b_res["пул"] = None if n < 100 else ax <= thr
        log(f"    (б) пул: короткие у осей {100 * ax:.1f} % <= {100 * thr:.1f} % (n={n}) -> "
            + ("не оценивается" if b_res["пул"] is None else "да" if b_res["пул"] else "НЕТ — пиксельная ступенька"))
    log("  видимость в нескольких каналах (для сведения):")
    multi_channel_share(M, log)
    if l6 is not None:
        log(f"  покрытие LINB6 пулом: в 60 м {100 * cover(l6, pool, 60):.1f} %, в 127 м {100 * cover(l6, pool, 127):.1f} %")
    if old_pool is not None:
        log(f"  прежний пул (A_all снимок+рельеф) в 127 м от CAND1: {100 * cover(old_pool, pool, 127):.1f} %; "
            f"CAND1 в 127 м от прежнего: {100 * cover(pool, old_pool, 127):.1f} %")
    return pool, pool_ch, b_res


def windows(site, img, dem, M, valid, log, win_km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = OUT / f"cand1_{site}"
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
            panels = (("зимний S2, без линий", img, ()), ("S2 + тёмные (красн.), светлые (голуб.), перепады (жёлт.)", img, (0, 1, 2)),
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
            fig.suptitle(f"CAND1 {site}, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}", fontsize=14)
            fig.tight_layout()
            p = out / f"окно_{k}.png"
            fig.savefig(p, dpi=65)
            plt.close(fig)
            log(f"  {p}")


def save(site, out, M, meta):
    d = {"res_m": RES_IN, **meta}
    for ch in out:
        ids = M[ch][0]
        d[f"ids_{ch + 1}"] = ids
        d[f"junk_{ch + 1}"] = out[ch][2]
        d[f"ok_{ch + 1}"] = out[ch][3]
    np.savez_compressed(OUT / f"cand1_{site}.npz", **d)


def utsjoki(log):
    from experiments.openlin_real1 import M as MET
    out, img, dem, valid, meta = build("utsjoki", log)
    M = masks(out, valid.shape)
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
    pool, pool_ch, b_res = describe("utsjoki", out, M, valid, meta, log, l6=l6, old_pool=old)
    gtk = gtk_mask("utsjoki", meta, shape, valid)
    log("=== (в) GTK 292 в 300 м")
    for ch in out:
        log(f"  {CHANNELS[ch]}: {100 * cover(gtk, M[ch][0] > 0, TOL_GTK_M):.1f} %")
    log(f"  прежний пул (замер до кода, проверка): {100 * cover(gtk, old, TOL_GTK_M):.1f} %; LINB6 {100 * cover(gtk, l6, TOL_GTK_M):.1f} %")
    c = cover(gtk, pool, TOL_GTK_M)
    c1 = c >= GTK_OLD_POOL
    log(f"(в1) пул CAND1 {100 * c:.1f} % >= {100 * GTK_OLD_POOL:.1f} % -> {'да' if c1 else 'НЕТ — пул беднее прежнего'}")
    cv, km_mean = thinned_cover(M, pool_ch, gtk, OLD_POOL_KM, N_THIN, SEED0)
    c2 = float(np.mean(cv)) >= GTK_OLD_POOL
    log(f"(в2) при равной длине ({OLD_POOL_KM:.0f} км по сумме ячеек линий; взято в среднем {km_mean:.0f} км, "
        f"{N_THIN} повторов): полнота {100 * np.mean(cv):.1f} % [мин {100 * np.min(cv):.1f}, макс {100 * np.max(cv):.1f}] "
        f">= {100 * GTK_OLD_POOL:.1f} % -> {'да' if c2 else 'НЕТ — на километр беднее прежнего'}")
    b_txt = ", ".join(f"{CHANNELS[ch] if ch != 'пул' else 'пул'} "
                      f"{'не оценено' if v is None else 'да' if v else 'НЕТ'}" for ch, v in b_res.items())
    log(f"ИТОГ CAND1 Утсйоки: (а) справка; (б) {b_txt}; (в1) {'да' if c1 else 'НЕТ'}; (в2) {'да' if c2 else 'НЕТ'}"
        f" -> {'ПРИНЯТО' if c1 and c2 else 'НЕ ПРИНЯТО'}")
    windows("utsjoki", img, dem, M, valid, log, win_km=6)


def anabar(log):
    out, img, dem, valid, meta = build("anabar", log)
    M = masks(out, valid.shape)
    save("anabar", out, M, meta)
    shape = valid.shape
    l6 = np.zeros(shape, bool)
    a = np.load(OUT / "linb6_anabar.npz")["ids"] > 0
    h, w = min(shape[0], a.shape[0]), min(shape[1], a.shape[1])
    l6[:h, :w] = a[:h, :w]
    describe("anabar", out, M, valid, meta, log, l6=l6)
    windows("anabar", img, dem, M, valid, log, win_km=8)


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_len1 import sha
    if "--utsjoki" in sys.argv:
        log = log_to(OUT / "_cand1_utsjoki.log")
        log(f"CAND1 Утсйоки; audit/CAND1_DECLARATION.md; sha256 кода {sha(__file__)}")
        utsjoki(log)
    elif "--anabar" in sys.argv:
        log = log_to(OUT / "_cand1_anabar.log")
        log(f"CAND1 Анабар (описательно); sha256 кода {sha(__file__)}")
        anabar(log)


if __name__ == "__main__":
    main()
