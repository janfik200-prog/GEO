"""META1: сравнение способов оценки карт линеаментов без эталона.

Объявление — ``audit/META1_DECLARATION.md`` (07.11.2026, до расчёта, с
уточнениями до кода). Сравниваются СПОСОБЫ ОЦЕНКИ, а не детекторы:

* Н  — набор ветки: суррогат IAAFT через весь тракт карты, сдвиговый нуль,
  устойчивость, длина против нуля;
* К1 — Kitchen, Rosenfeld (1981), связность границ (формула не сверена с
  оригиналом, см. объявление, уточнение 2);
* К2 — a contrario, NFA по модели LSD;
* К3 — консенсус карт, Yitzhaky, Peli (2003);
* К4 — только устойчивость; К5 — только половины; К6 — суррогат только на
  детекторе (части Н).

Эта часть — **Р** (настоящие листы, описательно). Синтетика (часть С) —
отдельный запуск того же модуля, ключ ``--часть С`` (будет добавлен).

Запуск из корня, только процессор:
    CUDA_VISIBLE_DEVICES="" python -m experiments.openlin_meta_eval --часть Р
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import GTK_TOL_M, REF, TOL_M, grid, prf, rast
from experiments.openlin_baselines import detect, eval_area, mask_of, seg_cells, to8

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
OUT = Path("outputs/meta1")

# ---- объявление META1 (audit/META1_DECLARATION.md) ---------------------------
RES_M = 20.0
CONF_TOL_M, CONF_FRAC = 60.0, 0.30       # подтверждение половинами (ветка)
JUNK_MAX = 0.20                          # мусор (ветка)
STAB_MIN, STAB_OVER_NULL, STAB_TOL_M = 0.5, 2.0, 40.0
STAB_NULL_N, STAB_NULL_SEED = 20, 7      # как в openlin_zone_lines
LEN_Q, LEN_SHARE_MIN = 0.95, 0.25        # длина против нуля (ветка)
N_FAKES, FAKE_SEED0 = 3, 900             # суррогаты IAAFT, зёрна тракта
ALPHA_NEW = 0.05 / 160                   # длина цепочки на 07.11.2026
N_SHIFTS_NEW, SHIFT_SEED = 7000, 11      # запас 2.19x; зерно сдвигов тракта
KR_GAMMA = 0.8                           # К1: НЕ сверено с оригиналом
LSD_TAU, LSD_LOG_EPS = np.pi / 8, 0.0  # К2: tau = 22.5 градуса, eps = 1 (LSD)
#: К2: у LSD угол направленный и p = tau/pi = 1/8. Линия тени — два борта с
#: противоположным градиентом, поэтому сравнение без знака (по модулю pi), и
#: вероятность совпадения в случайной модели та же по смыслу: 2*tau/pi = 1/4.
LSD_P = 2 * LSD_TAU / np.pi
LSD_SIGMA = 0.6 / 0.8                    # К2: сглаживание входа, как в LSD
YP_BLOCK = int(CONF_TOL_M / RES_M)       # К3: сетка 60 м (3 x 3 ячейки)
GTK_SHIFTS, GTK_SEED = 159, 2510         # как openlin_transfer_check_k1
# -----------------------------------------------------------------------------
assert 1.0 / (N_SHIFTS_NEW + 1) * 2 <= ALPHA_NEW, "порог недостижим с запасом 2x"

# Значения Н наших карт — из логов их прогонов (уточнение 4): мусор этапа
# подтверждения по ВСЕМУ листу («подтверждение: снимок …, подделка …,
# отношение …») и вердикт листа по правилу того прогона.
OURS = {
    # имя: (лист, файл маски, лог, строка, отношение, вердикт листа, что решило)
    "CHAIN":   ("anabar", "openlin_zone_lines_anabar_v3_цеп.npz", "_zone_lines_chain.log", 31,
                0.041, False, "медиана 1.27 км < 1.5 (старое правило)"),
    "WIDE":    ("anabar", "openlin_zone_lines_anabar_v3_путь.npz", "_zone_lines_anabar_v3_путь.log", 31,
                0.044, False, "медиана 1.21 км < 1.5 (старое правило)"),
    "TWIN":    ("anabar", "openlin_zone_lines_anabar_v3_бл.npz", "_zone_lines_anabar_v3_бл.log", 31,
                0.048, True, "все условия листа"),
    "ALL":     ("anabar", "openlin_zone_lines_anabar_v3_все.npz", "_zone_lines_all.log", 28,
                0.029, False, "медиана 1.23 км < 1.5 (старое правило)"),
    "NDVI":    ("anabar", "openlin_zone_lines_anabar_v3_ndvi.npz", "_zone_lines_ndvi.log", 32,
                0.157, False, "медиана 1.25 км < 1.5 (старое правило)"),
    "CLEAN":   ("anabar", "openlin_zone_lines_anabar_v3_чист.npz", "_zone_lines_anabar_v3_чист.log", 31,
                0.317, False, "мусор 0.317 > 0.20"),
    "CLEAN1K": ("anabar", "openlin_zone_lines_anabar_v3_чист1.npz", "_zone_lines_anabar_v3_чист1.log", 31,
                0.356, False, "мусор 0.356 > 0.20"),
    "DEM1":    ("anabar", "openlin_dem_lines_anabar_одна_линии.npz", "_dem_lines_anabar_одна.log", 7,
                0.015, True, "все условия"),
    "наш, Лапландия": ("lapland", "openlin_zone_lines_lapland_v3_перенос.npz",
                       "_zone_lines_lapland_v3_перенос.log", 33, 0.006, True, "все условия листа"),
    "наш, С. Остроботния": ("finland65", "openlin_zone_lines_finland65_v3_перенос.npz",
                            "_zone_lines_finland65_v3_перенос.log", 33, 0.012, False,
                            "длина 18.5 % < 25 %"),
}
# Независимые вердикты (не из Н): глаз и внешняя сверка
INDEPENDENT = {"TWIN": "принят глазом 09.10", "CHAIN": "рабочий 05–09.10 после просмотра",
               "ALL": "отклонён глазом («стало хуже»)",
               "наш, Лапландия": "GTK выше нуля, p = 0.0063",
               "наш, С. Остроботния": "GTK ниже нуля (0.65)"}


# ---- общие куски ---------------------------------------------------------------
def cells_to_ids(cells):
    """Список массивов ячеек -> (точки, номер линии) одним куском."""
    if not cells:
        return np.zeros((0, 2), int), np.zeros(0, int)
    pts = np.concatenate(cells)
    ids = np.repeat(np.arange(len(cells)), [len(c) for c in cells])
    return pts, ids


def conf_frac(pts, ids, n, dts, dr=0, dc=0, shape=None):
    """Доля линий, у которых >= CONF_FRAC длины в CONF_TOL_M от КАЖДОЙ половины."""
    if n == 0:
        return 0.0, np.zeros(0, bool)
    r, c = pts[:, 0], pts[:, 1]
    if dr or dc:
        r, c = (r + dr) % shape[0], (c + dc) % shape[1]
    ok = np.ones(n, bool)
    cnt = np.bincount(ids, minlength=n)
    for d in dts:
        hit = np.bincount(ids, weights=(d[r, c] <= CONF_TOL_M), minlength=n)
        ok &= hit >= CONF_FRAC * cnt
    return float(ok.mean()), ok


def shift_p(pts, ids, n, dts, shape, n_sh=N_SHIFTS_NEW, threads=8):
    """Сдвиговый нуль подтверждения: как shift_null_conf тракта, на точках.

    Те же сдвиги в том же порядке (зерно SHIFT_SEED), то же правило. Ускорено
    08.11.2026 без изменения результата: маски «в допуске» считаются один раз,
    суммы по линиям — reduceat по точкам, упорядоченным по линии, сдвиги
    считаются в потоках. Равенство со старой записью проверено (см. объявление).
    """
    from concurrent.futures import ThreadPoolExecutor
    H, W = shape
    rng = np.random.default_rng(SHIFT_SEED)
    obs = conf_frac(pts, ids, n, dts)[0]
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    sh = [(int(rng.integers(lo_r, H - lo_r)), int(rng.integers(lo_c, W - lo_c))) for _ in range(n_sh)]
    if n == 0:
        return obs, 0.0, 1.0
    order = np.argsort(ids, kind="stable")
    r = pts[order, 0].astype(np.int64); c = pts[order, 1].astype(np.int64)
    sid = ids[order]
    starts = np.r_[0, np.nonzero(np.diff(sid))[0] + 1]
    cnt = np.diff(np.r_[starts, len(sid)])
    need = CONF_FRAC * cnt
    near = [(d <= CONF_TOL_M).ravel() for d in dts]
    nl = len(starts)

    def one(s):
        dr, dc = s
        idx = ((r + dr) % H) * W + (c + dc) % W
        ok = np.ones(nl, bool)
        for b in near:
            ok &= np.add.reduceat(b[idx].astype(np.int32), starts) >= need
        return ok.sum() / n          # линии без точек не бывают: доля от всех n

    with ThreadPoolExecutor(threads) as ex:
        out = np.fromiter(ex.map(one, sh), float, count=n_sh)
    return obs, float(out.mean()), float((1 + (out >= obs).sum()) / (1 + n_sh))


def shift_p_old(pts, ids, n, dts, shape, n_sh=N_SHIFTS_NEW):
    """Прежняя запись (для проверки равенства)."""
    H, W = shape
    rng = np.random.default_rng(SHIFT_SEED)
    obs = conf_frac(pts, ids, n, dts)[0]
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    out = np.empty(n_sh)
    for i in range(n_sh):
        dr = int(rng.integers(lo_r, H - lo_r)); dc = int(rng.integers(lo_c, W - lo_c))
        out[i] = conf_frac(pts, ids, n, dts, dr, dc, shape)[0]
    return obs, float(out.mean()), float((1 + (out >= obs).sum()) / (1 + n_sh))


def stability(m, mp):
    """Устойчивость S и её нуль — правило openlin_zone_lines (20 сдвигов, зерно 7)."""
    from experiments.openlin_zone_lines import cover
    s = cover(m, mp, RES_M, STAB_TOL_M)
    rng = np.random.default_rng(STAB_NULL_SEED)
    null = np.mean([cover(m, np.roll(np.roll(mp, int(rng.integers(200, 3000)), 0),
                                     int(rng.integers(200, 3000)), 1), RES_M, STAB_TOL_M)
                    for _ in range(STAB_NULL_N)])
    return float(s), float(null)


def tensor_angle(img, valid, rho=1.0):
    """Направление градиента по структурному тензору (окрестность rho ячеек).

    Для К1: на оси гребня (линия тени) свой градиент почти нулевой, и
    направление в пикселе случайно; тензор берёт его с бортов. Для границы
    (Canny) совпадает с градиентом.
    """
    from scipy.ndimage import gaussian_filter
    x = np.where(valid, img, np.nanmedian(img[valid])).astype("float64")
    gy, gx = np.gradient(gaussian_filter(x, 1.0))
    jxx, jyy, jxy = (gaussian_filter(v, rho) for v in (gx * gx, gy * gy, gx * gy))
    return 0.5 * np.arctan2(2 * jxy, jxx - jyy), valid & ((jxx + jyy) > 0)


def grad_angle(img, valid, sigma=None):
    """Направление градиента (рад) и маска, где оно определено."""
    from scipy.ndimage import gaussian_filter
    x = np.where(valid, img, np.nanmedian(img[valid])).astype("float64")
    if sigma:
        x = gaussian_filter(x, sigma)
    gy, gx = np.gradient(x)
    return np.arctan2(gy, gx), valid & (np.hypot(gx, gy) > 0)


# ---- К1: Kitchen–Rosenfeld --------------------------------------------------------
_OFF = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def components(m):
    """Связные компоненты маски (8-связность) как список массивов ячеек."""
    from scipy.ndimage import find_objects
    from skimage.measure import label
    lab = label(m, connectivity=2)
    out = []
    for i, sl in enumerate(find_objects(lab), 1):
        if sl is None:
            continue
        out.append(np.argwhere(lab[sl] == i) + [sl[0].start, sl[1].start])
    return out


def k1_kitchen_rosenfeld(mask, ang, defined, chunk=1_000_000):
    """Среднее E по пикселям карты; считается кусками, чтобы не держать всё в памяти."""
    r, c = np.nonzero(mask)
    if len(r) == 0:
        return np.nan
    tot = 0.0
    for i in range(0, len(r), chunk):
        tot += _k1_part(mask, ang, defined, r[i:i + chunk], c[i:i + chunk]).sum()
    return float(tot / len(r))


def _k1_part(mask, ang, defined, r, c):
    """Средняя связность E = g*C + (1-g)*T по пикселям карты (окно 3 x 3).

    Направление линии в пикселе — градиент входа + 90 градусов. Спереди и
    сзади — по три соседа вокруг направления линии; согласие соседа =
    |cos(разность направлений)| * |cos(угол на соседа - направление)|; C —
    среднее лучших согласий спереди и сзади; T — доля прочих соседей вне линии.
    """
    H, W = mask.shape
    phi = (ang[r, c] + np.pi / 2) % np.pi
    phi_all = (ang + np.pi / 2) % np.pi
    # угол на соседа k в системе «x вправо, y вниз» как у np.gradient по (row, col)
    th = np.array([np.arctan2(dr, dc) for dr, dc in _OFF])
    nb_on = np.zeros((len(r), 8), bool)
    agree = np.zeros((len(r), 8))
    for k, (dr, dc) in enumerate(_OFF):
        rr, cc = r + dr, c + dc
        inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
        on = inside & mask[rr, cc]
        nb_on[:, k] = on
        a = np.abs(np.cos(phi - phi_all[rr, cc])) * np.abs(np.cos(th[k] - phi))
        agree[:, k] = np.where(on & defined[rr, cc], a, 0.0)
    # сектор «спереди»: три соседа, ближайшие по углу к phi; «сзади» — к phi + pi
    d_f = np.abs(np.angle(np.exp(1j * (th[None, :] - phi[:, None]))))
    d_b = np.abs(np.angle(np.exp(1j * (th[None, :] - phi[:, None] - np.pi))))
    front = np.argsort(d_f, axis=1)[:, :3]
    back = np.argsort(d_b, axis=1)[:, :3]
    rows = np.arange(len(r))[:, None]
    af, ab = agree[rows, front], agree[rows, back]
    bf, bb = front[rows[:, 0], af.argmax(1)], back[rows[:, 0], ab.argmax(1)]
    cont = 0.5 * (af.max(1) + ab.max(1))
    used = np.zeros_like(nb_on)
    used[rows[:, 0], bf] = af.max(1) > 0
    used[rows[:, 0], bb] |= ab.max(1) > 0
    rest = (~used).sum(1)
    thin = ((~nb_on) & (~used)).sum(1) / np.maximum(rest, 1)
    return KR_GAMMA * cont + (1 - KR_GAMMA) * thin


# ---- К2: a contrario (NFA по модели LSD) ---------------------------------------------
def k2_nfa(cells, ang, defined, shape):
    """Доля длины карты в линиях с NFA <= 1 (log10 NFA <= 0)."""
    from scipy.stats import binom
    H, W = shape
    log_tests = 2.5 * np.log10(H * W)
    tot, good = 0, 0
    for cl in cells:
        n = len(cl)
        if n < 2:
            continue
        xy = cl.astype("float64") - cl.mean(0)
        _u, _s, vt = np.linalg.svd(xy, full_matrices=False)
        psi = np.arctan2(vt[0, 0], vt[0, 1])           # направление линии (x = col, y = row)
        # прямоугольник LSD: ось линии и по ячейке с каждого борта (ширина 60 м
        # = допуск подтверждения): у линии тени градиент на оси почти нулевой,
        # выровненные пиксели — на бортах
        pr, pc = int(round(np.cos(psi))), int(round(-np.sin(psi)))
        cl = np.unique(np.concatenate([cl, cl + [pr, pc], cl - [pr, pc]]), axis=0)
        cl = cl[(cl[:, 0] >= 0) & (cl[:, 0] < H) & (cl[:, 1] >= 0) & (cl[:, 1] < W)]
        n = len(cl)
        g = ang[cl[:, 0], cl[:, 1]]
        # линия уровня перпендикулярна градиенту; выровнена с линией до tau
        lvl = g + np.pi / 2
        diff = np.abs(np.angle(np.exp(2j * (lvl - psi)))) / 2
        k = int(((diff <= LSD_TAU) & defined[cl[:, 0], cl[:, 1]]).sum())
        lnfa = log_tests + binom.logsf(k - 1, n, LSD_P) / np.log(10)
        tot += n
        good += n if lnfa <= LSD_LOG_EPS else 0
    return good / tot if tot else np.nan


# ---- К3: консенсус Yitzhaky–Peli -------------------------------------------------------
def _blocks(m, ok):
    b = YP_BLOCK
    H, W = (m.shape[0] // b) * b, (m.shape[1] // b) * b
    mm = m[:H, :W].reshape(H // b, b, W // b, b).any(axis=(1, 3))
    oo = ok[:H, :W].reshape(H // b, b, W // b, b).all(axis=(1, 3))
    return mm, oo


def _chi2(tp, fp, tn, fn):
    p, q = tp + fn, tp + fp
    if not (0 < q < 1) or not (0 < p < 1):
        return np.nan
    sens, spec = tp / p, tn / (1 - p)
    return (sens - q) * (spec - (1 - q)) / (q * (1 - q))


def k3_consensus(masks: dict, ok, log):
    """χ² каждой карты против оценённой «правды» согласия (EGT)."""
    names = list(masks)
    D, oo = [], None
    for n in names:
        mm, oo = _blocks(masks[n], ok)
        D.append(mm[oo])
    D = np.stack(D).astype(bool)                       # (N, ячейки)
    corr = D.sum(0)
    best, best_i = -np.inf, None
    for i in range(1, len(names) + 1):
        pgt = corr >= i
        tp = np.mean([(pgt & d).mean() for d in D]); fp = np.mean([(pgt & ~d).mean() for d in D])
        tn = np.mean([(~pgt & ~d).mean() for d in D]); fn = np.mean([(~pgt & d).mean() for d in D])
        x = _chi2(tp, fp, tn, fn)
        log(f"    К3: уровень согласия {i}/{len(names)}: χ² {x:.4f}")
        if np.isfinite(x) and x > best:
            best, best_i = x, i
    egt = corr >= best_i
    log(f"    К3: EGT — согласие не меньше {best_i} из {len(names)} карт, "
        f"{100 * egt.mean():.2f} % ячеек 60 м")
    out = {}
    for n, d in zip(names, D):
        out[n] = _chi2((d & egt).mean(), (d & ~egt).mean(), (~d & ~egt).mean(), (~d & egt).mean())
    return out, best_i


# ---- внешняя мера ---------------------------------------------------------------------
def external(site, g, ok, log):
    """Функция: маска карты -> {ΔP v2, отношение GTK к нулю}."""
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt as edt
    v2 = rast(gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg").geometry, g) & ok
    dv2 = edt(~v2, sampling=RES_M)
    p0 = float((dv2[ok] <= TOL_M).mean())
    log(f"  ориентир v2: {int(v2.sum())} ячеек, случайный уровень точности P0 = {p0:.4f}")
    gtk = None
    if site in ("lapland", "finland65"):
        from experiments.openlin_transfer_check_k1 import gtk_path
        gg = gpd.read_file(gtk_path(site, 292)).to_crs(epsg=g["epsg"])
        gtk = rast(gg.geometry, g) & g["valid"]
        H, W = g["H"], g["W"]
        mm = min(200, H // 8), min(200, W // 8)
        rng = np.random.default_rng(GTK_SEED)
        shifts = [(int(rng.integers(mm[0], H - mm[0] + 1)), int(rng.integers(mm[1], W - mm[1] + 1)))
                  for _ in range(GTK_SHIFTS)]
        gr, gc = np.nonzero(gtk)

    def f(mask):
        mm_ = mask & ok
        dm = edt(~mm_, sampling=RES_M)
        p, _r, _f = prf(mm_, v2, dm, dv2, TOL_M)
        row = {"P_v2": p, "P0_v2": p0, "dP_v2": p - p0}
        if gtk is not None:
            dfull = edt(~mask, sampling=RES_M)
            valid = g["valid"]
            obs = float((dfull[gr, gc] <= GTK_TOL_M).mean())
            null = []
            for dy, dx in shifts:
                rr, cc = (gr + dy) % H, (gc + dx) % W
                keep = valid[rr, cc]
                null.append(float((dfull[rr[keep], cc[keep]] <= GTK_TOL_M).mean()))
            row |= {"GTK_покрытие": obs, "GTK_нуль": float(np.mean(null)),
                    "GTK_отношение": obs / max(np.mean(null), 1e-12)}
        return row
    return f


# ---- Н, К4–К6 для простых детекторов ---------------------------------------------------
def baseline_eval(method, shade, hA, hB, valid, log):
    """Полный Н и К4–К6 для LSD или Canny + Хаф, режимы raw и с половинами."""
    from scipy.ndimage import distance_transform_edt as edt

    from experiments.openlin_iso import perturb
    from experiments.openlin_kora import surrogate
    shape = valid.shape

    def run(img, ha, hb):
        seg, L = detect(to8(img, valid), method)
        cells = seg_cells(seg, shape)
        dts = [edt(~mask_of(seg_cells(detect(to8(h, valid), method)[0], shape), shape),
                   sampling=RES_M) for h in (ha, hb)]
        pts, ids = cells_to_ids(cells)
        share, okc = conf_frac(pts, ids, len(cells), dts)
        return dict(cells=cells, L=L, dts=dts, pts=pts, ids=ids, share=share, okc=okc)

    t0 = time.time()
    R = run(shade, hA, hB)
    log(f"  {method}: отрезков {len(R['cells'])} ({R['L'].sum() / 1000:.0f} км), "
        f"подтверждено {int(R['okc'].sum())} ({100 * R['share']:.1f} %) ({time.time() - t0:.0f} с)")
    stack = np.stack([np.where(valid, x, np.nan) for x in (shade, hA, hB)])
    F = []
    for k in range(N_FAKES):
        sg = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(FAKE_SEED0 + k))
        F.append(run(sg[0], sg[1], sg[2]))
        log(f"    суррогат {k + 1}: отрезков {len(F[-1]['cells'])}, подтверждено "
            f"{int(F[-1]['okc'].sum())} ({time.time() - t0:.0f} с)")
    P = run(perturb(shade, valid), perturb(hA, valid), perturb(hB, valid))
    log(f"    пересэмплирование: отрезков {len(P['cells'])} ({time.time() - t0:.0f} с)")

    n_real = len(R["cells"])
    junk_trace = np.mean([len(f["cells"]) for f in F]) / max(n_real, 1)
    share_fake = sum(int(f["okc"].sum()) for f in F) / max(sum(len(f["cells"]) for f in F), 1)
    junk_conf = share_fake / max(R["share"], 1e-12)
    fake_len_q = float(np.quantile(np.concatenate([f["L"] for f in F]), LEN_Q)) \
        if any(len(f["L"]) for f in F) else 0.0
    out = {}
    for mode in ("raw", "conf"):
        sel = np.arange(n_real) if mode == "raw" else np.nonzero(R["okc"])[0]
        cells = [R["cells"][i] for i in sel]
        L = R["L"][sel]
        pts, ids = cells_to_ids(cells)
        obs, nul, p = shift_p(pts, ids, len(cells), R["dts"], shape)
        psel = (np.arange(len(P["cells"])) if mode == "raw" else np.nonzero(P["okc"])[0])
        m = mask_of(R["cells"], shape, sel)
        mp = mask_of(P["cells"], shape, psel)
        s, s0 = stability(m, mp)
        long_share = float((L > fake_len_q).mean()) if len(L) else 0.0
        junk = junk_trace if mode == "raw" else junk_conf
        verdict = dict(мусор=junk <= JUNK_MAX, сдвиги=p <= ALPHA_NEW,
                       устойчивость=(s >= STAB_MIN) and (s >= STAB_OVER_NULL * s0),
                       длина=long_share >= LEN_SHARE_MIN)
        out[mode] = dict(mask=m, cells=cells, Н=1 - junk, мусор=junk, p=p,
                         сдвиг_доля=obs, сдвиг_нуль=nul, S=s, S_нуль=s0,
                         длинных=long_share, q95_нуля_км=fake_len_q / 1000,
                         Н_вердикт=all(verdict.values()), Н_условия=verdict,
                         К4=s, К4_вердикт=verdict["устойчивость"], К5=R["share"],
                         К6=1 - junk_trace, К6_вердикт=junk_trace <= JUNK_MAX)
        log(f"    {mode}: мусор {junk:.3f}, p = {p:.5f} (порог {ALPHA_NEW:.6f}), S {s:.3f} "
            f"(нуль {s0:.3f}), длиннее нуля ({fake_len_q / 1000:.2f} км) {100 * long_share:.1f} % "
            f"-> Н {'ПРИНЯТ' if all(verdict.values()) else 'отклонён'} {verdict}")
    return out


# ---- часть Р -----------------------------------------------------------------------------
def part_r(log):
    from experiments.openlin_relief_lines import slope as slope_of
    rows = []
    for site in ("anabar", "lapland", "finland65"):
        log(f"\n=== {site}")
        g = grid(site)
        ok = eval_area(site, g)
        ext = external(site, g, ok, log)
        w = np.load(COMP / f"winter_{site}_20m_лист.npz")
        shade = w["shade"].astype("float32")
        hA, hB = [np.load(COMP / f"winter_{site}_20m_лист_{n}.npz")["shade"] for n in ("А", "Б")]
        valid = np.isfinite(shade) & np.isfinite(hA) & np.isfinite(hB)
        ang_kr, def_kr = tensor_angle(shade, valid)
        ang_lsd, def_lsd = grad_angle(shade, valid, sigma=LSD_SIGMA)
        if site == "anabar":
            from cache_paths import ANABAR_DEM
            # вход DEM1 — уклон GLO-30 (ALOS только подтверждал, градиент — по входу)
            slope = slope_of(np.load(ANABAR_DEM / "dem_glo30_anabar_лист_20m.npy"),
                             RES_M)[:valid.shape[0], :valid.shape[1]]
            sv = np.isfinite(slope)
            dem_kr, dem_lsd = tensor_angle(slope, sv), grad_angle(slope, sv, LSD_SIGMA)
        maps = {}
        for name, (st, fn, lg, ln, ratio, verd, why) in OURS.items():
            if st != site:
                continue
            z = np.load(M / fn)
            if "C_all" in z.files:
                m = z["C_all"][:valid.shape[0], :valid.shape[1]]
            else:                                           # DEM1: только полилинии
                m = np.zeros(valid.shape, bool)
                m[z["C_pts"][:, 0], z["C_pts"][:, 1]] = True
            cells = components(m)
            maps[name] = dict(mask=m, cells=cells, Н=1 - ratio, мусор=ratio, Н_вердикт=verd,
                              Н_почему=why, источник=f"{lg}:{ln}",
                              вход="dem" if name == "DEM1" else "opt")
        for meth in ("lsd", "hough"):
            res = baseline_eval(meth, shade, hA, hB, valid, log)
            for mode, d in res.items():
                maps[f"{'LSD' if meth == 'lsd' else 'Хаф'} {mode}, {site}"] = d | {"вход": "opt"}
        log("  К3 — консенсус по всем картам листа:")
        k3, lvl = k3_consensus({n: d["mask"] for n, d in maps.items()}, ok & valid, log)
        for name, d in maps.items():
            a_kr, d_kr = (dem_kr if d["вход"] == "dem" else (ang_kr, def_kr))
            a_l, d_l = (dem_lsd if d["вход"] == "dem" else (ang_lsd, def_lsd))
            row = {"лист": site, "карта": name, "км": float(d["mask"].sum() * RES_M / 1000),
                   "Н": d["Н"], "мусор": d["мусор"], "Н_вердикт": d["Н_вердикт"],
                   "К1": k1_kitchen_rosenfeld(d["mask"], a_kr, d_kr),
                   "К2": k2_nfa(d["cells"], a_l, d_l, valid.shape),
                   "К3": k3[name], "К3_уровень": lvl,
                   "независимый_вердикт": INDEPENDENT.get(name, "")}
            for key in ("К4", "К4_вердикт", "К5", "К6", "К6_вердикт", "p", "S", "S_нуль",
                        "длинных", "Н_почему", "источник"):
                if key in d:
                    row[key] = d[key]
            row |= ext(d["mask"])
            rows.append(row)
            log(f"  {name}: {row['км']:.0f} км; Н {row['Н']:.3f} ({'прин.' if row['Н_вердикт'] else 'откл.'}); "
                f"К1 {row['К1']:.3f}; К2 {row['К2']:.3f}; К3 {row['К3']:.4f}; "
                f"ΔP v2 {row['dP_v2']:+.3f}"
                + (f"; GTK {row['GTK_отношение']:.2f}x" if "GTK_отношение" in row else ""))
    return rows


# ---- часть С: одна синтетическая сцена ---------------------------------------------------
def parse_tract_log(path):
    """Числа Н всего листа из лога тракта (раздел «набор без фильтра зоны»)."""
    import re
    txt = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    i0 = next(i for i, s in enumerate(txt) if "набор без фильтра зоны" in s)
    ratio = p = None
    for s in txt[i0:i0 + 8]:
        m = re.search(r"подтверждение: снимок ([\d.]+) %, подделка ([\d.]+) %, отношение ([\d.]+)", s)
        if m and ratio is None:
            ratio = float(m.group(3))
        m = re.search(r"сдвиговый нуль: p = ([\d.]+)", s)
        if m and p is None:
            p = float(m.group(1))
    long_share = None
    for s in txt:
        m = re.match(r"\s+весь лист: ([\d.]+) % ->", s)
        if m:
            long_share = float(m.group(1)) / 100
    return dict(мусор=ratio, p=p, длинных=long_share)


def tract_k6(shade, hA, hB, valid):
    """К6 для нашего тракта: трассировка (отрезки от 500 м) на снимке и на 3 суррогатах."""
    from experiments.openlin_kora import surrogate
    from experiments.openlin_zone_lines import trace
    n_real = len(trace(shade, valid, RES_M, local=True, min_len_m=500.0))
    stack = np.stack([np.where(valid, x, np.nan) for x in (shade, hA, hB)])
    nf = []
    for k in range(N_FAKES):
        sg = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(FAKE_SEED0 + k))
        nf.append(len(trace(sg[0], valid, RES_M, local=True, min_len_m=500.0)))
    return np.mean(nf) / max(n_real, 1), n_real, nf


def part_c_site(name, log):
    """Все способы оценки на одной синтетической сцене; правда — вписанные линии."""
    from scipy.ndimage import distance_transform_edt as edt

    from experiments.openlin_zone_lines import save_line_sets  # noqa: F401  (формат файлов)
    tr = np.load(Path("outputs/meta1/synth") / f"{name}_truth.npz")
    truth = tr["truth"]
    dtruth = edt(~truth, sampling=RES_M)
    w = np.load(COMP / f"winter_{name}_20m_лист.npz")
    shade = w["shade"].astype("float32")
    hA, hB = [np.load(COMP / f"winter_{name}_20m_лист_{n}.npz")["shade"] for n in ("А", "Б")]
    valid = np.isfinite(shade) & np.isfinite(hA) & np.isfinite(hB)
    ang_kr, def_kr = tensor_angle(shade, valid)
    ang_l, def_l = grad_angle(shade, valid, sigma=LSD_SIGMA)
    fc = tr["false_c"]
    from experiments.openlin_azimuth_response import INS_LEN_M
    near_false = np.zeros(shade.shape, bool)
    if str(tr["false"]) != "none" and len(fc):
        fm = np.zeros(shade.shape, bool); fm[fc[:, 0], fc[:, 1]] = True
        near_false = edt(~fm, sampling=RES_M) <= INS_LEN_M / 2 + CONF_TOL_M

    maps = {}
    # наш тракт: карта, Н из лога, устойчивость листа по второму прогону
    z = np.load(M / f"openlin_zone_lines_{name}_v3_бл.npz")
    zp = np.load(M / f"openlin_zone_lines_{name}_p_v3_бл.npz")
    m = z["C_all"][:valid.shape[0], :valid.shape[1]]
    d = np.load(M / f"openlin_zone_lines_{name}_v3_бл_линии.npz")
    cells = [d["C_all_pts"][d["C_all_id"] == i] for i in range(len(d["C_all_len_km"]))]
    lg = parse_tract_log(Path("outputs/meta1") / f"_zl_{name}.log")
    s, s0 = stability(m, zp["C_all"][:valid.shape[0], :valid.shape[1]])
    k6, n6, nf6 = tract_k6(shade, hA, hB, valid)
    verdict = dict(мусор=lg["мусор"] <= JUNK_MAX, сдвиги=lg["p"] <= ALPHA_NEW,
                   устойчивость=(s >= STAB_MIN) and (s >= STAB_OVER_NULL * s0),
                   длина=lg["длинных"] >= LEN_SHARE_MIN)
    maps["наш тракт"] = dict(mask=m, cells=cells, Н=1 - lg["мусор"], мусор=lg["мусор"], p=lg["p"],
                            S=s, S_нуль=s0, длинных=lg["длинных"], Н_вердикт=all(verdict.values()),
                            Н_условия=verdict, К4=s, К4_вердикт=verdict["устойчивость"],
                            К6=1 - k6, К6_вердикт=k6 <= JUNK_MAX)
    log(f"  наш тракт: мусор {lg['мусор']}, p {lg['p']}, S {s:.3f} (нуль {s0:.3f}), длинных "
        f"{lg['длинных']}, К6 {k6:.3f} ({n6} -> {nf6}) -> Н {verdict}")
    for meth in ("lsd", "hough"):
        res = baseline_eval(meth, shade, hA, hB, valid, log)
        for mode, dd in res.items():
            maps[f"{'LSD' if meth == 'lsd' else 'Хаф'} {mode}"] = dd
    k3, lvl = k3_consensus({n: dd["mask"] for n, dd in maps.items()}, valid, log)
    rows = []
    for n, dd in maps.items():
        mm = dd["mask"] & valid
        prec = float((dtruth[mm] <= CONF_TOL_M).mean()) if mm.any() else np.nan
        row = {"сцена": name, "фон": name.split("_")[1], "амп": float(tr["amp"]),
               "ложные": str(tr["false"]), "карта": n, "км": float(mm.sum() * RES_M / 1000),
               "точность": prec, "плохая": (1 - prec) > JUNK_MAX if np.isfinite(prec) else None,
               "у_ложных_мест": float(near_false[mm].mean()) if mm.any() else np.nan,
               "Н": dd["Н"], "мусор": dd["мусор"], "Н_вердикт": dd["Н_вердикт"],
               "К1": k1_kitchen_rosenfeld(mm, ang_kr, def_kr),
               "К2": k2_nfa(dd["cells"], ang_l, def_l, valid.shape),
               "К3": k3[n], "К3_уровень": lvl}
        for key in ("К4", "К4_вердикт", "К5", "К6", "К6_вердикт", "p", "S", "S_нуль", "длинных"):
            if key in dd:
                row[key] = dd[key]
        if "Н_условия" in dd:
            row |= {f"Н_{k}": v for k, v in dd["Н_условия"].items()}
        rows.append(row)
        log(f"  {n}: {row['км']:.0f} км, точность {prec:.3f} ({'ПЛОХАЯ' if row['плохая'] else 'хорошая'}); "
            f"Н {row['Н']:.3f} ({'прин.' if row['Н_вердикт'] else 'откл.'}); К1 {row['К1']:.3f}; "
            f"К2 {row['К2']:.3f}; К3 {row['К3']:.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--часть", dest="part", choices=("Р", "С"), default="Р")
    ap.add_argument("--сцена", dest="site", default=None, help="часть С: имя синтетической сцены")
    a = ap.parse_args()
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"META1, часть {a.part}; объявление audit/META1_DECLARATION.md")
    if a.part == "С":
        import pandas as pd
        rows = part_c_site(a.site, log)
        (OUT / "synth").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / "synth" / f"{a.site}_eval.csv", index=False,
                                  encoding="utf-8-sig")
        log(f"сохранено: {OUT / 'synth' / (a.site + '_eval.csv')}")
        return
    rows = part_r(log)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part_R.csv", index=False, encoding="utf-8-sig")
    log(f"\nсохранено: {OUT / 'part_R.csv'}")
    # описательная связь с внешней мерой (без вердиктов, объявление)
    from scipy.stats import kendalltau
    sub = df[~df["карта"].str.startswith("LSD")]          # LSD против v2 не сравнивается
    summ = {}
    for k in ("Н", "К1", "К2", "К3", "К4", "К5", "К6"):
        if k not in sub:
            continue
        s = sub[[k, "dP_v2"]].dropna()
        if len(s) >= 3:
            tau, _p = kendalltau(s[k], s["dP_v2"])
            rng = np.random.default_rng(2510)
            bs = []
            for _ in range(10_000):
                i = rng.integers(0, len(s), len(s))
                t_, _ = kendalltau(s[k].values[i], s["dP_v2"].values[i])
                bs.append(t_)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            summ[k] = {"tau": float(tau), "ДИ95": [float(lo), float(hi)], "карт": len(s)}
            log(f"  Кендалл τ({k}, ΔP v2) = {tau:+.2f} [{lo:+.2f}; {hi:+.2f}] по {len(s)} картам")
    (OUT / "part_R_tau.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
