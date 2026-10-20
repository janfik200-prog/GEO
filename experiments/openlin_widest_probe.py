"""Проба: «самый широкий путь» вместо прямого мостика при сшивке обрывков.

ОТКУДА ПОСТАНОВКА. Сшивка — слабое место тракта по глазной проверке: «иногда
сшивает два близлежащих линеамента» и «линеамент из маленьких идущих подряд
линий не собран». Сейчас мостик между концами — ПРЯМОЙ отрезок, а его опора —
СРЕДНИЙ отклик вдоль него (:func:`openlin_zone_lines_v3.link_same_side`):
сильные края могут вытянуть пустую середину, а изогнутый разрыв прямая
промахивает. В репозитории ANZA-LIRA (github.com/fims9000/anza_lira,
``path_completion/widest_path.py``) та же задача решается иначе: между концами
ищется путь, у которого САМОЕ СЛАБОЕ ЗВЕНО максимально, в коридоре вокруг пары.
Нейросетевые части той работы нам недоступны (весов нет, обучение с учителем),
а это ядро — чистый алгоритм на сетке, и подать на него можно наш отклик.

ЧТО МЕРИТСЯ. Пары концов строятся теми же геометрическими условиями, что в
рабочей сшивке. Для каждой пары считаются две опоры:
  «прямая» — средний отклик вдоль отрезка (нынешнее правило);
  «путь»   — слабое звено самого широкого пути в коридоре.
Обе сравниваются со СВОИМ нулём (те же отрезки и коридоры, перенесённые по
тору), порог — тот же ``V7_BRIDGE_Q``-й процентиль, что в рабочей сшивке.

КАК РЕШАЕТСЯ, ЧТО ЛУЧШЕ. Доля прошедших пар ничего не говорит: правило, которое
пропускает больше, не обязательно право. Судит НЕЗАВИСИМАЯ ПОЛОВИНА СЪЁМКИ:
пары и мостики строятся по отклику одной половины, а проверяется, есть ли
вдоль предложенного мостика отклик в другой (средний по дорожке против нуля из
той же дорожки, перенесённой по тору). Путь выбирался по первой половине,
поэтому во второй он ничем не подогнан. Лучше то правило, у чьих принятых пар
доля подтверждённых второй половиной выше. Считается в обе стороны (А->Б, Б->А).

Число ширины коридора не подбирается: берётся ``CONF_TOL_M`` (60 м) — допуск,
в пределах которого ветка уже считает две линии одной. Концы пути, как и в
рабочей сшивке, исключаются на две ячейки: они лежат на гребнях.

Это замер, а не прогон: рабочий набор не меняется, в счётчик не идёт.

РЕЗУЛЬТАТ (08.10.2026, лог ``outputs/metrics/_widest_probe.log``). ЗАМЕНОЙ
нынешнему правилу путь НЕ является; как ДОБАВКА — кандидат. Пар-кандидатов
26.7 тыс. (86 % — через кулису). Прямая принимает 33-35 %, путь 32-33 %.
Подтверждение второй половиной (А->Б / Б->А), у нуля 5 %:
  приняты обоими        5918 / 6424 пар: по прямой 82 / 80 %, по пути 91 / 89 %;
  только прямой         2918 / 2964:     по прямой 64 / 68 %;
  только путём          2499 / 2524:     по прямой 20 / 21 %, по пути 75 / 76 %;
  отвергнуты обоими    15342 / 14922:    по прямой  5 /  6 %, по пути 21 / 26 %.
ЧТО ИЗ ЭТОГО СЛЕДУЕТ.
  1. Путь находит ~2500 соединений, которых прямая не видит (изогнутые: вдоль
     прямой во второй половине пусто, вдоль пути отклик есть). Извилистость
     принятых путей 1.04 / 1.4 / 2.1 (50 / 90 / 99 %).
  2. Путь отвергает ~2900 мостиков, две трети которых вторая половина
     подтверждает: слабое звено — минимум, одна шумовая дыра рвёт настоящий
     мостик. Замена правила означала бы эту потерю.
  3. Ожидание «путь не станет соединять соседние параллельные линии» НЕ
     подтвердилось: пары через кулису прямая принимает в 30-31 %, путь в
     28-30 % — разницы нет.
ПОПРАВКА К ПУНКТУ 2 (08.10.2026, после просмотра окон; лог
``_widest_probe2.log``). Глазом красные мостики (нынешнее правило принимает,
путь отвергает) «выглядят очень плохо», и после жадного отбора тоже: это
прямые, перечёркивающие чужие уступы и соединяющие разные объекты. После
отбора таких стыков 1002 из 2607 (38 %). Их «подтверждение второй половиной»
(57-64 % по прямой) доверия не заслуживает: средний отклик вдоль прямой высок
и тогда, когда прямая ПЕРЕСЕКАЕТ настоящие гребни, а это свойство мостика, а
не разрыва, и во второй половине оно повторяется. То есть сверка половиной
наследует изъян проверяемой статистики и «две трети настоящих» из пункта 2 —
не установлено. Стыков, которые добавил бы путь, после отбора 720.
ОГОВОРКА К СТОЛБЦУ «ПО ПУТИ»: он завышен. Путь ищет гребни и может идти по
чужим линиям в коридоре, поэтому даже у пар, отвергнутых обоими правилами,
«по пути» 21-26 % при 5 % у нуля. Сравнивать 75 % надо с этими 21-26 %, а не
с пятью. Окна (``outputs/линеаменты_широкий_путь/``) показывают ВСЕ пары до
жадного отбора, поэтому мостики лежат пучками — в рабочей сшивке из пучка
остаётся один.

Запуск из корня репозитория:
    python -m experiments.openlin_widest_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import COMP, MIN_LEN_M, OUT, _adiff, mask_of, trace

OUT_DIR = OUT.parent / "линеаменты_широкий_путь"
END_SKIP = 2                 # ячеек у каждого конца вне оценки — как в рабочей сшивке
WIN_KM = 5.0
N_WINDOWS = 12


def candidate_pairs(lines, res_m: float):
    """Пары концов по геометрии рабочей сшивки (без опоры на отклик)."""
    from scipy.spatial import cKDTree

    from experiments.openlin_lines import _axis

    cand = [r for r in lines if r["длина_км"] * 1000 >= v3.LINK_MIN_LEN_M]
    ends, own = [], []
    for i, r in enumerate(cand):
        _ax, a, b = _axis(r["pts"])
        ends += [a, b]
        own += [i, i]
    E = np.asarray(ends, dtype="float64")
    rad = np.hypot(v3.V5_GAP_M, v3.V4_OFFSET_M) / res_m
    out, meta = [], []
    for p_, q_ in cKDTree(E).query_pairs(rad):
        i, j = own[p_], own[q_]
        if i == j:
            continue
        v = E[q_] - E[p_]
        if np.hypot(*v) < 1:
            continue
        if _adiff(cand[i]["азимут"], cand[j]["азимут"]) > v3.DIR_TOL_DEG:
            continue
        t = np.radians(2 * np.array([cand[i]["азимут"], cand[j]["азимут"]]))
        st = (np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())) / 2) % 180.0
        a = np.radians(st)
        u = np.array([-np.cos(a), np.sin(a)])
        n = np.array([np.sin(a), np.cos(a)])
        along, across = abs(v @ u) * res_m, (v @ n) * res_m
        if along > v3.V5_GAP_M or abs(across) > v3.V4_OFFSET_M:
            continue
        ti, tj = cand[i]["pts"] @ u, cand[j]["pts"] @ u
        if (min(ti.max(), tj.max()) - max(ti.min(), tj.min())) * res_m \
                > v3.V6_OVERLAP_MAX_M:
            continue
        p = (int(E[p_][0]), int(E[p_][1]))
        q = (int(E[q_][0]), int(E[q_][1]))
        if max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 2 * END_SKIP + 1:
            continue                      # внутренних ячеек нет — мерить нечего
        out.append((p, q, abs(across) >= v3.V4_SIDE_MIN_M))
        meta.append((float(np.hypot(*v)), p_, q_, own[p_], own[q_],
                     0 if abs(across) < v3.V4_SIDE_MIN_M else int(np.sign(across))))
    return cand, out, meta


def greedy(meta, ok, n_cand: int) -> np.ndarray:
    """Жадный отбор стыков — тот же, что в рабочей сшивке.

    От коротких к длинным; конец занят одним стыком; обрывки одной цепочки
    второй раз не соединяются; сдвиги внутри цепочки — в одну сторону.
    """
    parent = list(range(n_cand))
    side = [0] * n_cand

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used, take = set(), np.zeros(len(meta), bool)
    for k in sorted(np.flatnonzero(ok), key=lambda k: meta[k][0]):
        _d, p_, q_, i, j, sg = meta[k]
        if p_ in used or q_ in used:
            continue
        ra, rb = find(i), find(j)
        if ra == rb:
            continue
        if sg and ((side[ra] and side[ra] != sg) or (side[rb] and side[rb] != sg)):
            continue
        used.update((p_, q_))
        parent[rb] = ra
        side[ra] = side[ra] or side[rb] or sg
        take[k] = True
    return take


def _box(p, q, m: int):
    r0, r1 = min(p[0], q[0]) - m, max(p[0], q[0]) + m + 1
    c0, c1 = min(p[1], q[1]) - m, max(p[1], q[1]) + m + 1
    return r0, r1, c0, c1


def bottleneck(zb: np.ndarray, p, q) -> float:
    """Слабое звено самого широкого пути p->q в окне (8-связность).

    Наибольший порог t, при котором p и q лежат в одной связной области
    {z >= t}; ячейки у концов в оценку не входят.
    """
    from scipy.ndimage import label

    zz = zb.copy()
    for e in (p, q):
        zz[max(e[0] - END_SKIP, 0):e[0] + END_SKIP + 1,
           max(e[1] - END_SKIP, 0):e[1] + END_SKIP + 1] = np.inf
    vals = np.unique(zz[np.isfinite(zz)])
    lo, hi = 0, len(vals) - 1
    s8 = np.ones((3, 3), bool)
    lab, _ = label(zz >= vals[0], structure=s8)
    if lab[p] != lab[q]:
        return float("-inf")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lab, _ = label(zz >= vals[mid], structure=s8)
        if lab[p] and lab[p] == lab[q]:
            lo = mid
        else:
            hi = mid - 1
    return float(vals[lo])


def path_of(zb: np.ndarray, p, q, b: float) -> np.ndarray:
    """Кратчайший путь внутри области z >= слабое звено (для рисунка и сверки)."""
    from skimage.graph import route_through_array

    zz = zb.copy()
    for e in (p, q):
        zz[max(e[0] - END_SKIP, 0):e[0] + END_SKIP + 1,
           max(e[1] - END_SKIP, 0):e[1] + END_SKIP + 1] = np.inf
    cost = np.where(zz >= b, 1.0, 1e6)
    idx, _ = route_through_array(cost, p, q, fully_connected=True, geometric=True)
    return np.asarray(idx, dtype=int)


def supports(z: np.ndarray, valid: np.ndarray, pairs, res_m: float, rng):
    """Обе опоры для каждой пары, их нули и дорожки (прямая и путь)."""
    from skimage.draw import line as draw_line

    H, W = z.shape
    m = max(int(round(v3.CONF_TOL_M / res_m)), 1)
    zf = np.where(valid, z, -np.inf).astype("float32")
    s_mean, s_wide, n_mean, n_wide, straight, paths = [], [], [], [], [], []
    for p, q, _k in pairs:
        rr, cc = draw_line(p[0], p[1], q[0], q[1])
        rr, cc = rr[END_SKIP:-END_SKIP], cc[END_SKIP:-END_SKIP]
        straight.append(np.stack([rr, cc], 1))
        s_mean.append(float(np.nanmean(z[rr, cc])))
        r0, r1, c0, c1 = _box(p, q, m)
        ri, ci = np.arange(r0, r1) % H, np.arange(c0, c1) % W
        zb = zf[np.ix_(ri, ci)]
        pl, ql = (p[0] - r0, p[1] - c0), (q[0] - r0, q[1] - c0)
        b = bottleneck(zb, pl, ql)
        s_wide.append(b)
        if np.isfinite(b):
            pt = path_of(zb, pl, ql, b) + np.array([r0, c0])
            pt = pt[END_SKIP:-END_SKIP]
            pt[:, 0] %= H; pt[:, 1] %= W
        else:
            pt = straight[-1]
        paths.append(pt)
        for _ in range(v3.V7_BRIDGE_SHIFTS):
            dr = int(rng.integers(200, H - 200)); dc = int(rng.integers(200, W - 200))
            n_mean.append(float(np.nanmean(z[(rr + dr) % H, (cc + dc) % W])))
            zs = zf[np.ix_((ri + dr) % H, (ci + dc) % W)]
            if np.isfinite(zs).all():       # коридор целиком в данных, иначе не нуль
                n_wide.append(bottleneck(zs, pl, ql))
    return (np.array(s_mean), np.array(s_wide), np.array(n_mean),
            np.array(n_wide), straight, paths)


def track_support(z: np.ndarray, tracks, rng):
    """Средний отклик вдоль готовых дорожек и порог из тех же дорожек по тору."""
    H, W = z.shape
    val = np.array([np.nanmean(z[t[:, 0], t[:, 1]]) if len(t) else np.nan
                    for t in tracks])
    null = []
    for _ in range(v3.V7_BRIDGE_SHIFTS):
        dr = int(rng.integers(200, H - 200)); dc = int(rng.integers(200, W - 200))
        null += [np.nanmean(z[(t[:, 0] + dr) % H, (t[:, 1] + dc) % W])
                 for t in tracks if len(t)]
    return val, float(np.nanquantile(np.array(null), v3.V7_BRIDGE_Q))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = v3.build_halves(a.site, log)
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])

    def response(o):
        xx = np.where(valid, standardize(o, valid), 0.0).astype("float32")
        return local_norm(np.where(valid, ridge_response(xx), 0.0), valid, res_m)

    def prepare(o):
        z = response(o)
        raw = trace(o, valid, res_m, local=True, min_len_m=0)
        raw = v3.extend_ends(raw, z, valid, res_m)
        return z, raw

    q = v3.V7_BRIDGE_Q
    log(f"ширина коридора {v3.CONF_TOL_M:.0f} м, порог — {100 * q:.0f}-й процентиль "
        f"своего нуля, переносов на пару {v3.V7_BRIDGE_SHIFTS}")

    # ---- независимая сверка: строим по одной половине, проверяем по другой
    Z = {}
    for name in ("А", "Б"):
        Z[name] = prepare(halves[name])
        log(f"половина «{name}»: обрывков {len(Z[name][1])} ({time.time() - t0:.0f} с)")
    for src, dst in (("А", "Б"), ("Б", "А")):
        z, raw = Z[src]
        zd = Z[dst][0]
        _cand, pairs, meta = candidate_pairs(raw, res_m)
        rng = np.random.default_rng(31)
        sm, sw, nm, nw, straight, paths = supports(z, valid, pairs, res_m, rng)
        tm, tw = float(np.nanquantile(nm, q)), float(np.quantile(nw, q))
        ok_m, ok_w = sm > tm, sw > tw
        kul = np.array([k for _p, _q, k in pairs])
        log(f"\n=== строим по «{src}», проверяем по «{dst}»: пар {len(pairs)}, "
            f"из них через кулису {int(kul.sum())} ({time.time() - t0:.0f} с)")
        log(f"  прямая: порог {tm:.3f}, проходит {100 * ok_m.mean():.1f} %  |  "
            f"путь: порог {tw:.3f}, проходит {100 * ok_w.mean():.1f} %")
        log(f"  квартили опоры «путь»: пары "
            f"{np.round(np.quantile(sw[np.isfinite(sw)], [.25, .5, .75]), 2)}, "
            f"нуль {np.round(np.quantile(nw[np.isfinite(nw)], [.25, .5, .75]), 2)}")
        rng = np.random.default_rng(32)
        vs, ts = track_support(zd, straight, rng)
        vp, tp = track_support(zd, paths, rng)
        cs, cp = vs > ts, vp > tp
        log(f"  подтверждение второй половиной (доля пар, чья дорожка выше "
            f"{100 * q:.0f} % нуля; у нуля по построению {100 * (1 - q):.0f} %):")

        def row(title, sel):
            n = int(sel.sum())
            if not n:
                log(f"    {title:38s}: пар 0"); return
            log(f"    {title:38s}: пар {n:5d} | по прямой {100 * cs[sel].mean():5.1f} % "
                f"| по пути {100 * cp[sel].mean():5.1f} %")

        row("приняты прямой (нынешнее правило)", ok_m)
        row("приняты путём", ok_w)
        row("приняты обоими", ok_m & ok_w)
        row("только прямой", ok_m & ~ok_w)
        row("только путём", ~ok_m & ok_w)
        row("отвергнуты обоими", ~ok_m & ~ok_w)
        gm, gw = greedy(meta, ok_m, len(_cand)), greedy(meta, ok_w, len(_cand))
        gu = greedy(meta, ok_m | ok_w, len(_cand))
        log("  после жадного отбора (то, что реально попадает в линии):")
        row("стыки нынешнего правила", gm)
        row("  из них отвергнуты путём", gm & ~ok_w)
        row("  из них приняты и путём", gm & ok_w)
        row("стыки правила «путь»", gw)
        row("стыки «прямая ИЛИ путь»", gu)
        row("  из них добавлены путём", gu & ~ok_m)
        for nm_, sel in (("в створе", ~kul), ("через кулису", kul)):
            log(f"  {nm_}: пар {int(sel.sum())}; прямая принимает "
                f"{100 * ok_m[sel].mean():.1f} %, путь {100 * ok_w[sel].mean():.1f} %")
            row(f"  {nm_}, приняты прямой", sel & ok_m)
            row(f"  {nm_}, приняты путём", sel & ok_w)
        det = np.array([len(pp) for pp in paths]) / np.maximum(
            np.array([len(s) for s in straight]), 1)
        log(f"  извилистость пути у принятых путём (длина пути / прямая): "
            f"{np.round(np.quantile(det[ok_w], [.5, .9, .99]), 2)} (50/90/99 %)")

    # ---- полный композит: где правила расходятся, окна для глаза
    z, raw = prepare(opt)
    cand, pairs, meta = candidate_pairs(raw, res_m)
    rng = np.random.default_rng(31)
    sm, sw, nm, nw, straight, paths = supports(z, valid, pairs, res_m, rng)
    ok_m = sm > float(np.nanquantile(nm, q))
    ok_w = sw > float(np.quantile(nw, q))
    log(f"\n=== полный композит: пар {len(pairs)}; прямая принимает {int(ok_m.sum())}, "
        f"путь {int(ok_w.sum())}, оба {int((ok_m & ok_w).sum())}, только прямая "
        f"{int((ok_m & ~ok_w).sum())}, только путь {int((~ok_m & ok_w).sum())} "
        f"({time.time() - t0:.0f} с)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_review_tiles import pick_windows
    from experiments.openlin_shade import _stretch

    shape = valid.shape

    def paint(tracks, sel):
        mk = np.zeros(shape, bool)
        for t, s in zip(tracks, sel):
            if s and len(t):
                mk[t[:, 0], t[:, 1]] = True
        return mk

    # в окнах — только стыки ПОСЛЕ жадного отбора: до него мостики лежат пучками
    # (все пары-кандидаты), а в линии из пучка попадает один
    gm = greedy(meta, ok_m, len(cand))
    gu = greedy(meta, ok_m | ok_w, len(cand))
    log(f"  после жадного отбора: стыков нынешнего правила {int(gm.sum())}, из них "
        f"отвергнуты путём {int((gm & ~ok_w).sum())}; при «прямая ИЛИ путь» стыков "
        f"{int(gu.sum())}, из них добавлены путём {int((gu & ~ok_m).sum())}")
    both = paint(straight, gm & ok_w)
    only_m = paint(straight, gm & ~ok_w)
    only_w = paint(paths, gu & ~ok_m)
    frag = mask_of([r for r in cand if r["длина_км"] * 1000 >= MIN_LEN_M], shape)
    small = mask_of([r for r in cand if r["длина_км"] * 1000 < MIN_LEN_M], shape)
    base = _stretch(np.where(valid, opt, np.nan))
    n_cell = int(WIN_KM * 1000 / res_m)
    half = n_cell // 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    wins = pick_windows(only_m | only_w, n_cell, N_WINDOWS, 1.0)
    px = 1000.0 / res_m
    for i, (r, c, _n) in enumerate(wins, 1):
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        fig, ax = plt.subplots(1, 2, figsize=(19, 10), facecolor="#fcfcfb")
        for k in (0, 1):
            ax[k].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[k].plot([10, 10 + px], [n_cell - 10] * 2, color="#111", lw=3)
            ax[k].text(10, n_cell - 16, "1 км", color="#111", fontsize=12)
            ax[k].set_xticks([]); ax[k].set_yticks([])
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[small[sl]] = (0.55, 0.80, 0.95, 0.9)
        ov[frag[sl]] = (0.10, 0.45, 0.85, 0.95)
        ov[both[sl]] = (0.10, 0.65, 0.20, 1.0)
        ov[only_m[sl]] = (0.90, 0.10, 0.10, 1.0)
        ov[only_w[sl]] = (0.95, 0.60, 0.0, 1.0)
        ax[1].imshow(ov)
        ax[0].set_title("без разметки — смотрите сами", loc="left", fontsize=14)
        ax[1].set_title("синие — обрывки (светлые короче 500 м); стыки после отбора: "
                        "зелёный — нынешнее правило, путь согласен,\nкрасный — нынешнее "
                        "правило, путь против; оранжевый — добавил бы путь",
                        loc="left", fontsize=13)
        fig.suptitle(f"Окно {i} из {len(wins)} · {WIN_KM:.0f} × {WIN_KM:.0f} км · "
                     f"центр ({r}, {c}) · север сверху", x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT_DIR / f"{i:02d}_окно_{r}_{c}.png", dpi=105,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"окна: {OUT_DIR} ({len(wins)} шт., {time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
