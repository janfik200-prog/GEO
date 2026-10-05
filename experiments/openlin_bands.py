"""Вторая ветка детектора: осевые линии ПОЛОС отклика — для прерывистых линеаментов.

ОТКУДА ПОСТАНОВКА. Глазная проверка 28.09.2026: детектор «видит всё» — карта
отклика покрывает все линеаменты. Но утончение (шаг 3) рассчитано на узкую тень
20-40 м; у широкой или прерывистой полосы («бусины») одного хребта нет, и вместо
осевой линии выходит цепочка коротких сдвинутых обрывков, из которой сшивка
собирает зигзаг. Решение пользователя (05.10.2026): шаг 3 не трогать — с
обычными линиями он справляется, — а для прерывистых добавить отдельную ветку.

ЧТО ДЕЛАЕТ ВЕТКА. Работает по готовой карте отклика, а не по снимку:
  1. отклик сглаживается в масштабе полосы — бусины сливаются в ленту;
  2. у ленты берётся хребет тем же утончением и теми же двумя порогами
     (квантили 0.97 / 0.90 сглаженного отклика) — это осевая линия полосы;
  3. осевая линия оставляется, только если шаг 3 на этом месте НЕ дал
     нормальной линии (иначе вторая ветка дублировала бы первую).

МАСШТАБ — ИЗ ЗАМЕРА (05.10.2026, 10025 обрывков от 200 м по листу): у
обрывков в цепочке (2 и больше соседей того же направления в 600 м) вокруг
оси есть «плечо» отклика, которого нет у одиночных; избыток максимален на
140 м от оси и спадает вдвое к 280 м. Сглаживание берётся 140 м.

ИЗВЕСТНЫЙ РИСК. Широкие ядра прямо в детекторе ветка уже пробовала
(``openlin_iso.SIGMAS_WIDE``) — шум на подделке склеивался в длинные гребни,
173 линии против 10. Здесь сглаживается не снимок, а отклик; спасает ли это,
решает двойник.

Запуск из корня репозитория (проба на вырезке 24 x 24 км, один двойник):
    python -m experiments.openlin_bands --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, M, MIN_LEN_M, OUT, mask_of, trace
from experiments.openlin_zone_lines_v3 import CONF_FRAC, CONF_TOL_M

BAND_SIGMA_M = 140.0         # максимум «плеча» отклика у цепочек, см. шапку
#: Осевая линия полосы считается уже найденной шагом 3, если не меньше
#: половины её длины лежит в пределах масштаба полосы от линии шага 3
#: длиной от MIN_LEN_M (обрывки короче — это и есть бусины, они не в счёт).
COVERED_FRAC = 0.5
CROP = (slice(400, 1600), slice(300, 1500))     # та же вырезка, что у пробы голосования

# ---- РЕЗУЛЬТАТ ПРОБЫ (05.10.2026, вырезка 24 x 24 км; в счётчик не идёт) ----
#: Геометрия получается: линеамент из «бусин» из контрольного окна рисуется
#: одной осевой линией и подтверждается обеими половинами. Но шум ветка
#: пропускает: новых осевых линий на двойнике 100 против 23 на снимке,
#: подтверждено 56 против 19 — мусор этапа 68 % при пороге 20 %. Условие
#: «внутри полосы лежит цепочка обрывков шага 3» не спасает (мусор 43-75 %):
#: обрывков 200-500 м на снимке и на двойнике поровну, 674 и 677, то есть
#: короткие обрывки — уровень шума. Различает ДЛИНА: у трёх двойников новые
#: линии 0.70 / 1.20 / 2.66 км (50 % / 95 % / макс), у снимка 0.87 / 3.74 /
#: 5.37 км. Длинных и подтверждённых: при пороге 1.20 км — 5 против 4.7 у
#: двойника, при 1.42 км — 4 против 1.0, при 2.66 км — 2 против 0.
# ---- ПОЛНЫЙ ПРОГОН (объявлено 05.10.2026, до прогона) -----------------------
#: Вторая ветка берёт только ДЛИННЫЕ полосы. Порог длины — самая длинная
#: новая осевая линия у калибровочных двойников (зёрна CAL_SEEDS) на полном
#: листе. Мусор меряется на ДРУГИХ двойниках (TEST_SEEDS): мерить его теми
#: же, по которым выбран порог, значило бы получить ноль по построению.
#: Условия приёмки (один набор — длинные подтверждённые осевые линии):
#:   1. мусор: подтверждённых длинных у проверочных двойников в среднем не
#:      больше 20 % от числа у снимка (STAGE_JUNK_MAX);
#:   2. сдвиговый нуль подтверждения: p <= BAND_ALPHA;
#:   3. устойчивость: взаимное покрытие при возмущении входа >= STAB_MIN и
#:      >= STAB_OVER_NULL нулей.
#: Медиана длины здесь не условие: она выше порога длины по построению.
#: Подтверждение — прежнее правило (30 % длины в 60 м от линий каждой
#: половины); эталон половины — её линии шага 3 и её осевые линии полос.
#: Счётчик: после NDVI 57. Голосование по сценам (openlin_vote) остановлено
#: на пробе, до прогона не дошло и слагаемого не добавляет. Этот прогон: +3.
#: РЕЗУЛЬТАТ ПОЛНОГО ПРОГОНА (05.10.2026): ДВА УСЛОВИЯ ИЗ ТРЁХ, ветка не
#: принята. Порог длины вышел 2.46 км. На снимке длинных и подтверждённых 7
#: линий (20 км), из них 5 новых и 2 идут вдоль уже подтверждённых.
#:   мусор: у проверочных двойников 0 / 2 / 1, в среднем 1.0 против 7 =
#:     14.3 % (порог 20 %) — пройдено, но на счёте в единицы линий;
#:   сдвиговый нуль: 100 % на месте против 1.2 %, p = 0.00033 — пройдено;
#:   устойчивость: 0.316 при пороге 0.5 — НЕ пройдено.
#: Главное число — не эти три, а соотношение до отбора: новых осевых линий
#: на снимке 271, на каждом двойнике около 1230. Сглаженный отклик на шуме
#: даёт в 4.5 раза больше полос, чем на снимке, и ветка держится только на
#: отсечке по длине. Выход — 5 новых линий на лист — усилий не оправдывает.
CAL_SEEDS = (910, 911, 912)
TEST_SEEDS = (900, 901, 902)
BAND_N_TESTS_PRIOR = 57
BAND_N_TESTS = 3
BAND_ALPHA = 0.05 / (BAND_N_TESTS_PRIOR + BAND_N_TESTS)     # 0.000833
#: Сдвигов: n >= 2/alpha - 1 = 2399; берётся 3000, пол 0.00033, запас 2.5x.
BAND_SHIFT_N = 3000


def band_lines(z: np.ndarray, valid: np.ndarray, res_m: float) -> list:
    """Осевые линии полос отклика: хребет сглаженного отклика, те же пороги."""
    from experiments.openlin_iso import Q_HIGH, Q_LOW, hysteresis8, to_lines_strict
    from experiments.openlin_lines import _axis, _ridge_nms

    s, ridge = _ridge_nms(np.clip(z, 0.0, None), valid, BAND_SIGMA_M / res_m)
    v = s[valid]
    hi, lo = np.quantile(v, Q_HIGH), np.quantile(v, Q_LOW)
    mask = hysteresis8(np.where(ridge, s, float(v.min())), lo, hi)
    _m, rows = to_lines_strict((mask, s >= lo), res_m)
    out = []
    for r in rows:
        if r["длина_км"] * 1000 < MIN_LEN_M:
            continue
        ax, _a, _b = _axis(r["pts"])
        r["азимут"] = float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180.0)
        out.append(r)
    return out


def near_share(lines, dist, tol_m: float) -> np.ndarray:
    return np.array([(dist[r["pts"][:, 0], r["pts"][:, 1]] <= tol_m).mean()
                     for r in lines]) if lines else np.zeros(0)


def branch(img, valid, res_m):
    """Линии шага 3 (от MIN_LEN_M) и НОВЫЕ осевые линии полос на одном снимке."""
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_vote import response

    good = trace(img, valid, res_m, local=True)
    bands = band_lines(response(img, valid, res_m), valid, res_m)
    d = distance_transform_edt(~mask_of(good, valid.shape), sampling=(res_m, res_m))
    new = [b for b, sh in zip(bands, near_share(bands, d, BAND_SIGMA_M))
           if sh < COVERED_FRAC]
    return good, bands, new


def full(site: str) -> None:
    """Полный прогон по листу — см. блок «ПОЛНЫЙ ПРОГОН» в шапке констант."""
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_iso import perturb
    from experiments.openlin_kora import surrogate
    from experiments.openlin_zone_lines import (STAB_MIN, STAB_OVER_NULL,
                                                STAB_TOL_M, cover)
    from experiments.openlin_zone_lines_v3 import (STAGE_JUNK_MAX, build_halves,
                                                   shift_null_conf)

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    halves = build_halves(site, lambda *_: None)
    stack = np.stack([w["shade"], halves["А"], halves["Б"]]).astype("float32")
    valid = np.all(np.isfinite(stack), axis=0)
    filled = np.nan_to_num(stack)
    log(f"порог значимости 0.05/{BAND_N_TESTS_PRIOR + BAND_N_TESTS} = "
        f"{BAND_ALPHA:.6f}, сдвигов {BAND_SHIFT_N}")

    def prep(x):
        return np.where(valid, x, np.nan).astype("float32")

    def confirm(new, st):
        dts = []
        for h in st[1:]:
            g_h, b_h, _n = branch(prep(h), valid, res_m)
            dts.append(distance_transform_edt(~mask_of(g_h + b_h, valid.shape),
                                              sampling=(res_m, res_m)))
        sh = [near_share(new, d, CONF_TOL_M) for d in dts]
        return (sh[0] >= CONF_FRAC) & (sh[1] >= CONF_FRAC), dts

    # 1. снимок
    _g, bands, new = branch(prep(stack[0]), valid, res_m)
    conf, dts = confirm(new, stack)
    Lr = np.array([b["длина_км"] for b in new])
    log(f"снимок: осевых линий полос {len(bands)}, новых {len(new)} "
        f"({Lr.sum():.0f} км), подтверждено {int(conf.sum())} ({time.time() - t0:.0f} с)")

    # 2. порог длины — по калибровочным двойникам (только основной канал)
    cal = []
    for s in CAL_SEEDS:
        tw = surrogate(filled, valid, np.random.default_rng(s))
        _g, _b, n_t = branch(prep(tw[0]), valid, res_m)
        cal += [b["длина_км"] for b in n_t]
        log(f"  калибровочный двойник {s}: новых {len(n_t)}, самая длинная "
            f"{max((b['длина_км'] for b in n_t), default=0):.2f} км ({time.time() - t0:.0f} с)")
        del tw
    cal = np.array(cal)
    thr = float(cal.max())
    log(f"порог длины = {thr:.2f} км (у калибровочных двойников 50/95/99 %: "
        f"{' / '.join(f'{v:.2f}' for v in np.percentile(cal, [50, 95, 99]))})")
    long_r = Lr > thr
    keep = long_r & conf
    lines = [b for b, k in zip(new, keep) if k]
    log(f"снимок: длинных {int(long_r.sum())}, длинных и подтверждённых "
        f"{len(lines)} ({Lr[keep].sum():.0f} км)")

    # 3. мусор — на других двойниках, через всю ветку с подтверждением
    fake = []
    for s in TEST_SEEDS:
        tw = surrogate(filled, valid, np.random.default_rng(s))
        _g, _b, n_t = branch(prep(tw[0]), valid, res_m)
        c_t, _d = confirm(n_t, tw)
        L_t = np.array([b["длина_км"] for b in n_t])
        fake.append(int(((L_t > thr) & c_t).sum()))
        log(f"  проверочный двойник {s}: новых {len(n_t)}, длинных "
            f"{int((L_t > thr).sum())}, длинных и подтверждённых {fake[-1]} "
            f"({time.time() - t0:.0f} с)")
        del tw
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    junk = float(np.mean(fake)) / max(len(lines), 1)
    log(f"\nмусор: у двойников в среднем {np.mean(fake):.1f} против {len(lines)} у "
        f"снимка = {100 * junk:.1f} % (порог {100 * STAGE_JUNK_MAX:.0f} %) -> "
        f"{mark[bool(lines) and junk <= STAGE_JUNK_MAX]}")

    # 4. сдвиговый нуль подтверждения длинных линий
    longs = [b for b, k in zip(new, long_r) if k]
    if longs:
        _sh, p = shift_null_conf(longs, dts, valid.shape, log, BAND_SHIFT_N)
        log(f"нуль подтверждения: p = {p:.5f} (порог {BAND_ALPHA:.6f}) -> "
            f"{mark[p <= BAND_ALPHA]}")

    # 5. устойчивость: возмущённый снимок, те же половины
    _g, _b, new_p = branch(perturb(prep(stack[0]), valid), valid, res_m)
    sh = [near_share(new_p, d, CONF_TOL_M) for d in dts]
    keep_p = [b for b, a, c in zip(new_p, sh[0], sh[1])
              if b["длина_км"] > thr and a >= CONF_FRAC and c >= CONF_FRAC]
    mC, mP = mask_of(lines, valid.shape), mask_of(keep_p, valid.shape)
    stab = cover(mC, mP, res_m, STAB_TOL_M) if lines and keep_p else 0.0
    rr = np.random.default_rng(7)
    null = np.mean([cover(mC, np.roll(np.roll(mP, int(rr.integers(200, 3000)), 0),
                                      int(rr.integers(200, 3000)), 1), res_m, STAB_TOL_M)
                    for _ in range(20)]) if lines and keep_p else 0.0
    log(f"устойчивость: {stab:.3f}, нуль {null:.3f} -> "
        f"{mark[stab >= STAB_MIN and stab >= STAB_OVER_NULL * max(null, 1e-9)]}")

    # для сведения: сколько из них уже есть в действующем наборе
    old = np.load(M / f"openlin_zone_lines_{site}_v3.npz")
    dold = distance_transform_edt(~old["C_all"], sampling=(res_m, res_m))
    dup = int((near_share(lines, dold, BAND_SIGMA_M) >= COVERED_FRAC).sum()) if lines else 0
    az = np.array([b["азимут"] for b in lines])
    log(f"для сведения: из {len(lines)} линий {dup} идут вдоль уже подтверждённых "
        f"линий действующего набора, {len(lines) - dup} — новые; азимуты: "
        f"{np.histogram(az, bins=6, range=(0, 180))[0].tolist()} по секторам 30°")
    np.savez_compressed(M / f"openlin_bands_{site}.npz", C=mC,
                        all_new=mask_of(new, valid.shape), thr_km=thr,
                        centers=np.array([b["pts"].mean(0) for b in lines]))
    log(f"готово ({time.time() - t0:.0f} с)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--полный", dest="full", action="store_true",
                    help="полный прогон по листу с объявленными условиями")
    a = ap.parse_args()
    if a.full:
        full(a.site)
        return

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_kora import surrogate
    from experiments.openlin_zone_lines_v3 import build_halves

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    halves = build_halves(a.site, lambda *_: None)
    stack = np.stack([w["shade"][CROP], halves["А"][CROP], halves["Б"][CROP]]
                     ).astype("float32")
    valid = np.all(np.isfinite(stack), axis=0)
    print("ПРОБА на вырезке 24 x 24 км, один двойник — проверка идеи, в счётчик "
          "не идёт", flush=True)

    def run(st, name):
        img = [np.where(valid, x, np.nan).astype("float32") for x in st]
        good, bands, new = branch(img[0], valid, res_m)
        dts = []
        for h in img[1:]:
            g_h, b_h, _n = branch(h, valid, res_m)
            dts.append(distance_transform_edt(~mask_of(g_h + b_h, valid.shape),
                                              sampling=(res_m, res_m)))
        conf = {}
        for tol in (CONF_TOL_M, BAND_SIGMA_M):
            sh = [near_share(new, d, tol) for d in dts]
            conf[tol] = (sh[0] >= CONF_FRAC) & (sh[1] >= CONF_FRAC) if new else np.zeros(0, bool)
        km = sum(b["длина_км"] for b in new)
        print(f"{name}: линий шага 3 — {len(good)}; осевых линий полос {len(bands)}, "
              f"из них новых {len(new)} ({km:.0f} км); подтверждено половинами: "
              f"{int(conf[CONF_TOL_M].sum())} при допуске {CONF_TOL_M:.0f} м, "
              f"{int(conf[BAND_SIGMA_M].sum())} при {BAND_SIGMA_M:.0f} м "
              f"({time.time() - t0:.0f} с)", flush=True)
        return good, new, conf

    good, new, conf = run(stack, "снимок ")
    tw = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900))
    _g, new_t, conf_t = run(tw, "двойник")

    print()
    for tol in (CONF_TOL_M, BAND_SIGMA_M):
        r = conf[tol].mean() if len(new) else 0.0
        f = conf_t[tol].mean() if len(new_t) else 0.0
        print(f"допуск {tol:.0f} м: подтверждение у снимка {100 * r:.0f} %, у двойника "
              f"{100 * f:.0f} %, мусор этапа {100 * f / max(r, 1e-9):.0f} % (порог 20 %); "
              f"подтверждённых: снимок {int(conf[tol].sum())}, двойник "
              f"{int(conf_t[tol].sum())}")
    print(f"число новых осевых линий: снимок {len(new)}, двойник {len(new_t)} "
          f"(отношение {len(new_t) / max(len(new), 1):.2f})")

    # картинка: окно 1 из просмотра (линеамент из «бусин»), оно внутри вырезки
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    r0, c0 = 1176 - CROP[0].start, 733 - CROP[1].start
    sl = (slice(r0 - 125, r0 + 125), slice(c0 - 125, c0 + 125))
    base = _stretch(np.where(valid, stack[0], np.nan))[sl]
    old = np.load(M / f"openlin_zone_lines_{a.site}_v3.npz")
    k3 = np.ones((3, 3), bool)
    fig, ax = plt.subplots(1, 3, figsize=(27, 9.6), facecolor="#fcfcfb")
    for axx in ax:
        axx.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axx.set_xticks([]); axx.set_yticks([])
    ov = np.zeros(base.shape + (4,), "float32")
    ov[old["A_all"][CROP][sl]] = (0.95, 0.60, 0.0, 0.95)
    ov[old["C_all"][CROP][sl]] = (0.85, 0.05, 0.05, 0.95)
    ax[1].imshow(ov)
    ov = np.zeros(base.shape + (4,), "float32")
    c140 = conf[BAND_SIGMA_M]
    ov[binary_dilation(mask_of([b for b, c in zip(new, c140) if not c], valid.shape)[sl], k3)] = (0.15, 0.75, 0.95, 1.0)
    ov[binary_dilation(mask_of([b for b, c in zip(new, c140) if c], valid.shape)[sl], k3)] = (0.1, 0.7, 0.2, 1.0)
    ax[2].imshow(ov)
    ax[0].set_title("снимок", loc="left", fontsize=15)
    ax[1].set_title("сейчас: красные подтверждены, жёлтые — нет", loc="left", fontsize=15)
    ax[2].set_title("вторая ветка: зелёные подтверждены (140 м), голубые — нет",
                    loc="left", fontsize=15)
    fig.tight_layout()
    p = OUT / "вторая_ветка_проба.png"
    fig.savefig(p, dpi=80, facecolor=fig.get_facecolor())
    print(f"сохранено: {p}")


if __name__ == "__main__":
    main()
