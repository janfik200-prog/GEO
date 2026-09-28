"""Линии внутри зон согласия: выделение, отбор по направлению, радар, проверки.

ОТКУДА ПОСТАНОВКА. Устойчивая «Кора» (:mod:`openlin_kora_stable`) дала карту
зон, где оптика и радар согласны о направлении рисунка местности (AUC 0.85 и
0.83 на двух половинах листа, карты совпадают на 0.84, к гидросети не
привязано). Пользователь посмотрел крупные окна и подтвердил: в зонах видны
параллельные вытянутые формы, которые он называет линеаментами. Здесь из зон
делаются сами линии.

КАК ВЫДЕЛЯЕТСЯ:

1. вход — зимний снимок при низком солнце (20 м, канал ``shade``), весь лист;
   второй прибор — радар Sentinel-1 (20 м, нисходящие витки, ``shade``);
2. линии — генератор, прошедший проверки в :mod:`openlin_iso`: осевая
   трассировка без встроенной прямизны и сшивка разрывов только там, где под
   мостиком есть отклик (``осевая+сшивка по порогу``); не короче ``MIN_LEN_M``;
3. отбор — линия лежит в зоне согласия (``ZONE_SHARE`` листа с наибольшей
   оценкой) большей частью длины и идёт вдоль направления своей зоны с
   отклонением не больше ``DIR_TOL_DEG``;
4. подтверждение — не меньше ``RADAR_FRAC`` длины линии лежит в пределах
   ``RADAR_TOL_M`` от радарной линии. Набор А — только отбор, набор Б — отбор
   плюс радар; условия проверяются на наборе Б.

ГЛАВНАЯ ЛОВУШКА И КАК ОНА ЗАКРЫТА. Если искать линии «вдоль заданного
направления», вдоль него они найдутся и в шуме. Поэтому подделка (многоканальная,
снимок и радар вместе, :func:`openlin_kora.surrogate`) обрабатывается ровно так
же — с теми же зонами и тем же полем направлений, взятыми с настоящего снимка.
Мусор меряется именно тем, что этот приор делает из текстуры.

Запуск из корня репозитория:
    python -m experiments.openlin_zone_lines --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
OUT = Path("outputs/steps")
COMP = Path("datacache/s2_lowsun/composite")
RADAR = Path("datacache/s1_look/composite")

# ---- пре-регистрация (22.09.2026, до прогона) ------------------------------
MIN_LEN_M = 500.0
ZONE_SHARE = 0.20
ZONE_FRAC = 0.5              # доля длины линии, лежащая в зоне
DIR_TOL_DEG = 20.0
RADAR_TOL_M = 60.0
RADAR_FRAC = 0.3
N_FAKES = 3
JUNK_MAX = 0.20              # набор Б
STAB_MIN = 0.5               # взаимное покрытие при пересэмплировании
STAB_OVER_NULL = 2.0
STAB_TOL_M = 40.0
STRIKE_SCALE = 1000
# ---------------------------------------------------------------------------
#: РЕЗУЛЬТАТ первой версии (22.09.2026): набор Б — 31 линия, 27 км; мусор 0 %,
#: устойчивость 0.83 при нуле ~0 — оба условия пройдены. Но линии — обрывки
#: (медиана 0.67 км у всех линий оптики), прямая структура в 4 км режется на
#: 3-4 куска, пологие участки почти пусты: порог генератора глобальный по
#: кадру. Независимо: линии в зонах идут вдоль направления зоны в 3-4 раза
#: чаще случайного (66 % в пределах 20° при ожидаемых 22 %).
# ---- вторая версия (22.09.2026, ДО прогона) ------------------------------
#: Локальный порог — восьмая конфигурация :mod:`openlin_iso` без изменений.
#: Сшивка вдоль зоны: оба обрывка в зоне и вдоль её направления, мостик тоже
#: вдоль (все три угла <= DIR_TOL_DEG), разрыв до V2_GAP_M, ближние первыми,
#: каждый конец один раз. Сшивка — ДО фильтра длины.
V2_GAP_M = 300.0
V2_MEDIAN_KM_MIN = 1.5       # новое условие: медиана длины набора Б
# ---------------------------------------------------------------------------


def cover(a: np.ndarray, b: np.ndarray, res_m: float, tol: float) -> float:
    from scipy.ndimage import distance_transform_edt

    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    da = distance_transform_edt(~a, sampling=(res_m, res_m))
    db = distance_transform_edt(~b, sampling=(res_m, res_m))
    return 0.5 * (float((db[a] <= tol).mean()) + float((da[b] <= tol).mean()))


def trace(img: np.ndarray, valid: np.ndarray, res_m: float,
          local: bool = False, min_len_m: float = MIN_LEN_M) -> list[dict]:
    """Линии замороженным генератором; к каждой — азимут.

    ``local`` — порог гистерезиса по окрестности (восьмая конфигурация
    :mod:`openlin_iso`), иначе по всему кадру.
    """
    from experiments.openlin_iso import (axial_local_with_support,
                                         axial_with_strict_support,
                                         to_lines_strict)
    from experiments.openlin_lines import _axis
    from experiments.openlin_shadow import standardize

    x = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    raw = (axial_local_with_support(x, valid, res_m) if local
           else axial_with_strict_support(x, valid))
    _m, rows = to_lines_strict(raw, res_m)
    out = []
    for r in rows:
        if r["длина_км"] * 1000 < min_len_m:
            continue
        ax, _a, _b = _axis(r["pts"])
        r["азимут"] = float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180.0)
        out.append(r)
    return out


def _line_strike(p, strike):
    s2 = np.radians(2 * strike[p[:, 0], p[:, 1]])
    return (np.degrees(np.arctan2(np.sin(s2).mean(), np.cos(s2).mean())) / 2
            ) % 180.0


def _adiff(a, b):
    return abs(((a - b) + 90.0) % 180.0 - 90.0)


def link_along_zone(lines: list[dict], zone: np.ndarray, strike: np.ndarray,
                    res_m: float) -> list[dict]:
    """Сшивка обрывков вдоль направления зоны (вторая версия).

    Сшиваются только обрывки, которые оба в зоне и оба вдоль её направления,
    и только мостиком, который сам идёт вдоль направления. Остальные линии
    возвращаются как были — их судьбу решит обычный отбор.
    """
    from scipy.spatial import cKDTree
    from skimage.draw import line as draw_line

    from experiments.openlin_iso import _describe
    from experiments.openlin_lines import _axis

    cand, rest = [], []
    for r in lines:
        p = r["pts"]
        st = _line_strike(p, strike)
        if zone[p[:, 0], p[:, 1]].mean() >= ZONE_FRAC and \
                _adiff(r["азимут"], st) <= DIR_TOL_DEG:
            r["_st"] = st
            cand.append(r)
        else:
            rest.append(r)
    if len(cand) < 2:
        return lines
    ends, own = [], []
    for i, r in enumerate(cand):
        _ax, a, b = _axis(r["pts"])
        ends += [a, b]; own += [i, i]
    E = np.asarray(ends, dtype="float64")
    pairs = []
    for p, q in cKDTree(E).query_pairs(V2_GAP_M / res_m):
        i, j = own[p], own[q]
        if i == j:
            continue
        v = E[q] - E[p]
        if np.hypot(*v) < 1:
            continue
        az = float(np.degrees(np.arctan2(v[1], -v[0])) % 180.0)
        # среднее направление — по удвоенным углам, иначе 175° и 5° дали бы 90°
        t = np.radians(2 * np.array([cand[i]["_st"], cand[j]["_st"]]))
        st = (np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())) / 2) % 180.0
        if _adiff(cand[i]["_st"], cand[j]["_st"]) > DIR_TOL_DEG or \
                _adiff(az, st) > DIR_TOL_DEG:
            continue
        pairs.append((float(np.hypot(*v)), p, q))
    pairs.sort()
    parent = list(range(len(cand)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = np.zeros(len(E), dtype=bool)
    bridges = {}
    for _d, p, q in pairs:
        if used[p] or used[q]:
            continue
        a, b = find(own[p]), find(own[q])
        if a == b:
            continue
        used[p] = used[q] = True
        parent[b] = a
        rr, cc = draw_line(int(E[p][0]), int(E[p][1]), int(E[q][0]), int(E[q][1]))
        bridges.setdefault(own[p], []).append(np.stack([rr, cc], 1))
    groups = {}
    for i, r in enumerate(cand):
        groups.setdefault(find(i), []).append(r["pts"])
        groups[find(i)].extend(bridges.get(i, []))
    out = list(rest)
    for g in groups.values():
        pts = np.unique(np.concatenate(g), axis=0)
        d = _describe(pts, res_m)
        if d is None:
            continue
        ax, _a, _b = _axis(pts)
        d["азимут"] = float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180.0)
        out.append(d)
    return out


def select(lines: list[dict], zone: np.ndarray, strike: np.ndarray,
           radar_dt: np.ndarray | None) -> tuple[list, list]:
    """Набор А (зона + направление) и набор Б (А + радар)."""
    A, B = [], []
    for r in lines:
        p = r["pts"]
        if zone[p[:, 0], p[:, 1]].mean() < ZONE_FRAC:
            continue
        s2 = np.radians(2 * strike[p[:, 0], p[:, 1]])
        st = (np.degrees(np.arctan2(np.sin(s2).mean(), np.cos(s2).mean())) / 2
              ) % 180.0
        d = abs(((r["азимут"] - st) + 90.0) % 180.0 - 90.0)
        if d > DIR_TOL_DEG:
            continue
        r["простирание_зоны"] = float(st)
        A.append(r)
        if radar_dt is not None and \
                (radar_dt[p[:, 0], p[:, 1]] <= RADAR_TOL_M).mean() >= RADAR_FRAC:
            B.append(r)
    return A, B


def mask_of(lines: list, shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for r in lines:
        m[r["pts"][:, 0], r["pts"][:, 1]] = True
    return m


def zone_and_strike(site: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Зона согласия и поле простирания со 100 м на сетку 20 м."""
    from experiments.openlin_forest_kora import neigh_all
    from experiments.openlin_invariant import INSTRUMENTS
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR as RAD

    z = load_site(site)
    use = tuple(c for c in OPTIC + RAD if c in z)
    x, _y, valid, use = build_stack(z, use)
    d = np.load(M / f"openlin_kora_stable_{site}.npz")
    s = 0.5 * (d["score_1"] + d["score_2"])
    ok = valid & np.isfinite(s)
    zone = ok & (s >= np.quantile(s[ok], 1 - ZONE_SHARE))
    stack, names = neigh_all(x, use, valid, float(z["res_m"]), valid)
    idx = {n: i for i, n in enumerate(names)}
    c2 = sum(stack[idx[f"{ch}_cos{STRIKE_SCALE}"]] for ch in INSTRUMENTS)
    s2 = sum(stack[idx[f"{ch}_sin{STRIKE_SCALE}"]] for ch in INSTRUMENTS)
    grad = (90.0 + np.degrees(np.arctan2(s2, c2) / 2.0)) % 180.0
    strike = (grad + 90.0) % 180.0
    up = lambda a: np.repeat(np.repeat(a, k, 0), k, 1)
    return up(zone), up(strike.astype("float32"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--v2", action="store_true",
                    help="локальный порог и сшивка вдоль направления зоны")
    a = ap.parse_args()
    tag = "_v2" if a.v2 else ""

    def log(m=""):
        print(m, flush=True)

    import pandas as pd
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_hydro_check import hydro_mask
    from experiments.openlin_iso import perturb
    from experiments.openlin_kora import surrogate

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    r = dict(np.load(RADAR / f"radar_{a.site}_20m_descending.npz"))
    res_m = float(w["res_m"])
    opt, rad = w["shade"].astype("float32"), r["shade"].astype("float32")
    valid = np.isfinite(opt) & np.isfinite(rad)
    zone, strike = zone_and_strike(a.site, int(round(100 / res_m)))
    zone, strike = zone[:valid.shape[0], :valid.shape[1]], \
        strike[:valid.shape[0], :valid.shape[1]]
    zone &= valid
    log(f"площадка {a.site}: 20 м, {valid.shape}, зона согласия "
        f"{100 * zone.sum() / valid.sum():.1f} % листа ({time.time() - t0:.0f} с)")

    def optics(o):
        if not a.v2:
            return trace(o, valid, res_m)
        L = link_along_zone(trace(o, valid, res_m, local=True, min_len_m=0),
                            zone, strike, res_m)
        return [x for x in L if x["длина_км"] * 1000 >= MIN_LEN_M]

    def run(o, rd):
        lo = optics(o)
        lr = trace(rd, valid, res_m, local=a.v2)
        rdt = distance_transform_edt(~mask_of(lr, valid.shape),
                                     sampling=(res_m, res_m))
        A, B = select(lo, zone, strike, rdt)
        return lo, lr, A, B, rdt

    lo, lr, A, B, rdt = run(opt, rad)
    km = lambda L: sum(x["длина_км"] for x in L)
    log(f"снимок: линий оптики {len(lo)}, радара {len(lr)}; набор А {len(A)} "
        f"({km(A):.0f} км), набор Б {len(B)} ({km(B):.0f} км) "
        f"({time.time() - t0:.0f} с)")

    fA, fB = [], []
    stack = np.stack([np.where(valid, opt, np.nan), np.where(valid, rad, np.nan)])
    for k in range(N_FAKES):
        sg = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(700 + k))
        _lo, _lr, a_, b_, _ = run(sg[0], sg[1])
        fA.append(len(a_)); fB.append(len(b_))
        log(f"  подделка {k + 1}: набор А {len(a_)}, набор Б {len(b_)} "
            f"({time.time() - t0:.0f} с)")
    junkA = np.mean(fA) / max(len(A), 1)
    junkB = np.mean(fB) / max(len(B), 1)

    lo_p = optics(perturb(opt, valid))
    Ap, Bp = select(lo_p, zone, strike, rdt)
    mB, mBp = mask_of(B, valid.shape), mask_of(Bp, valid.shape)
    stab = cover(mB, mBp, res_m, STAB_TOL_M)
    rng = np.random.default_rng(7)
    null = np.mean([cover(mB, np.roll(np.roll(mBp, int(rng.integers(200, 3000)), 0),
                                      int(rng.integers(200, 3000)), 1),
                          res_m, STAB_TOL_M) for _ in range(20)])
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    log(f"\nмусор (подделка с теми же зонами и направлениями, {N_FAKES} шт.):")
    log(f"  набор А: {100 * junkA:.1f} %;  набор Б: {100 * junkB:.1f} % "
        f"(порог {100 * JUNK_MAX:.0f} %) -> {mark[junkB <= JUNK_MAX]}")
    okS = stab >= STAB_MIN and stab >= STAB_OVER_NULL * null
    med = float(np.median([x["длина_км"] for x in B])) if B else 0.0
    log(f"устойчивость набора Б к пересэмплированию: {stab:.3f}, нуль "
        f"{null:.3f} ({stab / max(null, 1e-9):.1f}x) -> {mark[okS]}")
    if a.v2:
        log(f"медиана длины набора Б: {med:.2f} км (порог {V2_MEDIAN_KM_MIN} км) -> "
            f"{mark[med >= V2_MEDIAN_KM_MIN]}")

    water = hydro_mask(valid.shape, res_m, float(w["x0"]), float(w["y_top"]),
                       int(w["epsg"]), lambda *_: None)
    dw = distance_transform_edt(~water, sampling=(res_m, res_m))
    base = (dw[valid] <= 100).mean()
    for name, L in (("А", A), ("Б", B)):
        m = mask_of(L, valid.shape)
        obs = (dw[m] <= 100).mean() if m.any() else 0.0
        log(f"для сведения, набор {name}: у воды (100 м) {100 * obs:.1f} % "
            f"длины при {100 * base:.1f} % площади ({obs / max(base, 1e-9):.2f}x)")

    pd.DataFrame([{"набор": n, "длина_км": x["длина_км"],
                   "прямолинейность": x["прямолинейность"],
                   "азимут": x["азимут"], "простирание_зоны": x["простирание_зоны"],
                   "строка": int(x["pts"][:, 0].mean()),
                   "столбец": int(x["pts"][:, 1].mean())}
                  for n, L in (("А", A), ("Б", B)) for x in L]).to_csv(
        M / f"openlin_zone_lines_{a.site}{tag}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(M / f"openlin_zone_lines_{a.site}{tag}.npz", A=mask_of(A, valid.shape),
                        B=mB, zone=zone)
    plot(opt, valid, zone, A, B, a.site + tag, junkB, stab, log)


def plot(img, valid, zone, A, B, site, junkB, stab, log,
         conf: str = "подтверждены радаром") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from experiments.openlin_shade import _stretch

    base = _stretch(np.where(valid, img, np.nan))
    fig, ax = plt.subplots(1, 3, figsize=(27, 9.4), facecolor="#fcfcfb")

    def segs(L):
        out = []
        for r in L:
            p = r["pts"]; c = p.mean(0)
            a = np.radians(r["азимут"]); h = r["длина_км"] * 1000 / 20.0 / 2
            dx, dy = np.sin(a) * h, -np.cos(a) * h
            out.append([(c[1] - dx, c[0] - dy), (c[1] + dx, c[0] + dy)])
        return out

    for a_ in ax:
        a_.imshow(base, cmap="gray", vmin=0, vmax=1)
        ov = np.zeros(zone.shape + (4,), dtype="float32")
        ov[zone] = (0.89, 0.29, 0.28, 0.18)
        a_.imshow(ov)
        a_.set_xticks([]); a_.set_yticks([])
    from scipy.ndimage import binary_dilation

    ax[0].add_collection(LineCollection(segs(A), colors="#eda100", linewidths=1.2))
    ax[0].add_collection(LineCollection(segs(B), colors="#b3261e", linewidths=2.0))
    mA = binary_dilation(mask_of(A, zone.shape), np.ones((3, 3), bool))
    mB = binary_dilation(mask_of(B, zone.shape), np.ones((3, 3), bool))
    # центры двух красных окон из зоны_согласия_крупно: (342, 589) и (337, 229)
    # в ячейках 100 м, то есть x5 на сетке 20 м; окно 6 км = 300 ячеек
    for a_, (rc, cc) in ((ax[1], (1710, 2945)), (ax[2], (1685, 1145))):
        ov = np.zeros(zone.shape + (4,), dtype="float32")
        ov[mA] = (0.93, 0.63, 0.0, 0.9)
        ov[mB] = (0.70, 0.15, 0.12, 1.0)
        a_.imshow(ov)
        a_.set_ylim(rc + 150, rc - 150); a_.set_xlim(cc - 150, cc + 150)
    ax[0].set_title(f"линии в зонах согласия: красные — {conf} "
                    f"({len(B)}), жёлтые — только оптикой ({len(A) - len(B)})\n"
                    f"мусор {100 * junkB:.0f} %, устойчивость {stab:.2f}",
                    fontsize=13, loc="left")
    ax[1].set_title("крупно: красное окно 1 (6 × 6 км)", fontsize=13, loc="left")
    ax[2].set_title("крупно: красное окно 2 (6 × 6 км)", fontsize=13, loc="left")
    fig.tight_layout()
    p = OUT / f"линии_в_зонах_{site}.png"
    fig.savefig(p, dpi=85, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
