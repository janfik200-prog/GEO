"""Отдельный слой: меридиональные линии по рельефу — для слепой полосы оптики.

ОТКУДА ПОСТАНОВКА. Оптика при солнце 184° не видит уступы в полосе ±15° у
меридиана. Проба (:mod:`openlin_relief_probe`, 06.10.2026): отмывки рельефа
наискосок, с 45° и 135°, дают в этой полосе 402 и 582 линии при 11 и 41 у
двойника; в секторах, которые оптика видит, 45-59 % линий рельефа совпадают с
рабочим набором оптики. Отмывка строго с востока негодна: 1003 линии при 701
у двойника — боковой свет вытягивает вдоль меридиана и шум.

ЭТО НЕ ОСНОВНОЙ РЕЗУЛЬТАТ. Заказчик разрешил рельеф с оговоркой «лучше без
него». Основная карта строится по снимку; этот слой подаётся отдельно и
помечается как построенный по рельефу.

ПРАВИЛО (объявлено 06.10.2026, до прогона). Линия берётся с косой отмывки,
если она лежит в слепом секторе оптики и длиннее 500 м. Подтверждение — другой
косой отмывкой: не меньше 30 % длины в 60 м от её линий (те же допуски, что у
подтверждения двумя съёмками). Модель рельефа одна, поэтому это не две
независимые съёмки, а два независимых освещения одной модели; насколько это
отсекает шум, решает двойник. Слой — объединение по двум отмывкам без повторов.
Продолжения концов и сшивки нет: только то, что нашёл детектор.

УСЛОВИЯ ПРИЁМКИ:
  1. мусор: подтверждённых линий у двойников в среднем не больше 20 % от
     числа у слоя;
  2, 3. сдвиговый нуль подтверждения — отдельно для линий с 45° и со 135°;
  4. устойчивость при возмущении отмывок: >= 0.5 и >= 2 нулей.
Счётчик: после «цепочек» 72, этот прогон добавляет 4.

РЕЗУЛЬТАТ (06.10.2026): ВСЕ ЧЕТЫРЕ УСЛОВИЯ ПРОЙДЕНЫ. В слое 578 линий, 529 км,
медиана длины 0.76 км. Мусор: у трёх двойников по одной линии против 578 —
0.2 %. Сдвиговый нуль: 84 % и 76 % на месте против 6.6 % и 6.3 %, p = 0.00025
для обеих отмывок. Устойчивость 0.803 при нуле 0.010. Из 578 линий 252 идут
вдоль линий оптики (включая неподтверждённые), 326 оптика не видит совсем.
Окна для глаза: outputs/линеаменты_рельеф/. Глазом слой ещё не оценён.

Запуск из корня репозитория:
    python -m experiments.openlin_relief_layer --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_relief_probe import (BLIND_DEG, SUN_AZ_OPT, hillshade,
                                              load_dem)
from experiments.openlin_zone_lines import (COMP, M, OUT, STAB_MIN,
                                            STAB_OVER_NULL, STAB_TOL_M, _adiff,
                                            cover, mask_of, trace)
from experiments.openlin_zone_lines_v3 import (CONF_FRAC, CONF_TOL_M,
                                               STAGE_JUNK_MAX, shift_null_conf,
                                               work_npz)

OBLIQUE = (45.0, 135.0)
N_TWINS = 3
RELIEF_N_TESTS_PRIOR = 72
RELIEF_N_TESTS = 4
RELIEF_ALPHA = 0.05 / (RELIEF_N_TESTS_PRIOR + RELIEF_N_TESTS)    # 0.000658
#: Сдвигов: n >= 2/alpha - 1 = 3039; берётся 4000, пол 0.00025, запас 2.6x.
RELIEF_SHIFT_N = 4000
OUT_DIR = OUT.parent / "линеаменты_рельеф"
#: Вариант с продолжением концов и сшивкой (06.10.2026, объявлено до прогона).
#: ПОВОД: слой без них короткий — медиана 0.76 км, а пользователь после
#: просмотра попросил добавить оба шага. ЧТО ДЕЛАЕТСЯ: на каждой косой отмывке
#: линии проходят тот же тракт, что в рабочем методе по снимку — продолжение
#: концов по отклику, сшивка с опорой мостика, «цепочки» (мостики через кулисы
#: не рисуются) — с теми же числами; под рельеф ничего не подбирается. В слепой
#: сектор линия отбирается ПОСЛЕ сшивки, по азимуту цепочки. ЭТАЛОН
#: подтверждения прежний — линии другой отмывки БЕЗ сшивки: сшивать и эталон в
#: ветке пробовали дважды, оба раза это впускало шум (мусор 52.5 % и 20.7 %).
#: УСЛОВИЯ — те же четыре. Счётчик: после слоя без сшивки 76, этот прогон +4.
#: РЕЗУЛЬТАТ (06.10.2026): ВСЕ ЧЕТЫРЕ УСЛОВИЯ ПРОЙДЕНЫ. В слое 563 линии, 749 км,
#: медиана длины 1.06 км (без сшивки было 578 линий, 529 км, 0.76 км). Мусор:
#: у двойников 5 / 3 / 3 линии против 563 — 0.7 %. Сдвиговый нуль: 78 % и 66 %
#: на месте против 5.6 % и 5.2 %, p = 0.00025. Устойчивость 0.728 при нуле
#: 0.013. Кандидатов у двойника стало на порядок больше (101-260 против 11-41):
#: сшивка собирает линии и из шума, отсекает их подтверждение второй отмывкой.
#: 308 линий из 563 оптика не видит совсем. Глазом слой ещё не оценён.
LINK_N_TESTS_PRIOR = RELIEF_N_TESTS_PRIOR + RELIEF_N_TESTS     # 76
LINK_N_TESTS = 4
LINK_ALPHA = 0.05 / (LINK_N_TESTS_PRIOR + LINK_N_TESTS)        # 0.000625
#: Сдвигов: n >= 2/alpha - 1 = 3199; берётся 4000, пол 0.00025, запас 2.5x.
LINK_SHIFT_N = 4000


def blind(lines) -> list:
    return [x for x in lines if _adiff(x["азимут"], SUN_AZ_OPT) <= BLIND_DEG]


def tract(im, valid, res_m):
    """Тракт рабочего метода на одной отмывке: продолжение, сшивка, «цепочки»."""
    from experiments.openlin_vote import response
    from experiments.openlin_zone_lines import MIN_LEN_M
    from experiments.openlin_zone_lines_v3 import extend_ends, link_same_side

    x = np.where(valid, im, np.nan).astype("float32")
    z = response(x, valid, res_m)
    raw = extend_ends(trace(x, valid, res_m, local=True, min_len_m=0), z, valid, res_m)
    L = [r for r in link_same_side(raw, None, None, res_m, None, z=z)
         if r["длина_км"] * 1000 >= MIN_LEN_M]
    W_ = valid.shape[1]
    for r in L:
        kul = r.get("кулисы")
        if kul is not None and len(kul):
            r["pts"] = r["pts"][~np.isin(r["pts"][:, 0] * W_ + r["pts"][:, 1],
                                         kul[:, 0] * W_ + kul[:, 1])]
    return L


def layer(imgs, valid, res_m, link: bool = False):
    """Слой по паре косых отмывок: (линии, откуда взята, карты расстояний)."""
    from scipy.ndimage import distance_transform_edt

    L = [trace(np.where(valid, im, np.nan), valid, res_m, local=True) for im in imgs]
    dts = [distance_transform_edt(~mask_of(x, valid.shape), sampling=(res_m, res_m))
           for x in L]
    if link:                       # эталон (dts) остаётся без сшивки, см. шапку
        L = [tract(im, valid, res_m) for im in imgs]
    cand = [blind(L[0]), blind(L[1])]

    def ok(x, d):
        return (d[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC

    first = [x for x in cand[0] if ok(x, dts[1])]
    d_first = distance_transform_edt(~mask_of(first, valid.shape),
                                     sampling=(res_m, res_m))
    second = [x for x in cand[1] if ok(x, dts[0]) and not ok(x, d_first)]
    return first + second, cand, dts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--без-сшивки", dest="plain", action="store_true",
                    help="первый вариант слоя: только то, что нашёл детектор")
    a = ap.parse_args()
    link = not a.plain
    alpha, n_sh = (LINK_ALPHA, LINK_SHIFT_N) if link else (RELIEF_ALPHA, RELIEF_SHIFT_N)
    n_tot = (LINK_N_TESTS_PRIOR + LINK_N_TESTS if link
             else RELIEF_N_TESTS_PRIOR + RELIEF_N_TESTS)
    out_dir = OUT_DIR.with_name(OUT_DIR.name + "_сшивка") if link else OUT_DIR
    tag = "_сш" if link else ""

    def log(m=""):
        print(m, flush=True)

    from scipy.ndimage import distance_transform_edt, uniform_filter

    from experiments.openlin_iso import perturb
    from experiments.openlin_kora import surrogate
    from experiments.openlin_winter import detrend

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    dem = load_dem(a.site, res_m)
    valid = np.isfinite(dem) & np.isfinite(w["shade"])
    filled = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    hs = [detrend(hillshade(filled, res_m, az), valid, res_m) for az in OBLIQUE]
    log(f"{'со сшивкой и продолжением' if link else 'без сшивки'}; порог значимости "
        f"0.05/{n_tot} = {alpha:.6f}, сдвигов {n_sh}")

    lines, cand, dts = layer(hs, valid, res_m, link)
    km = sum(x["длина_км"] for x in lines)
    log(f"рельеф: кандидатов в слепом секторе {len(cand[0])} и {len(cand[1])}; "
        f"в слое {len(lines)} линий ({km:.0f} км), медиана длины "
        f"{np.median([x['длина_км'] for x in lines]):.2f} км ({time.time() - t0:.0f} с)")

    stack = np.stack(hs).astype("float32")
    fake = []
    for j in range(N_TWINS):
        tw = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900 + j))
        Lt, ct, _d = layer([tw[0], tw[1]], valid, res_m, link)
        fake.append(len(Lt))
        log(f"  двойник {j + 1}: кандидатов {len(ct[0])} и {len(ct[1])}, в слое "
            f"{len(Lt)} ({time.time() - t0:.0f} с)")
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    junk = float(np.mean(fake)) / max(len(lines), 1)
    log(f"\nмусор: у двойников в среднем {np.mean(fake):.1f} против {len(lines)} = "
        f"{100 * junk:.1f} % (порог {100 * STAGE_JUNK_MAX:.0f} %) -> "
        f"{mark[junk <= STAGE_JUNK_MAX]}")
    for k, other in ((0, 1), (1, 0)):
        _s, p = shift_null_conf(cand[k], [dts[other]], valid.shape, log, n_sh)
        log(f"нуль подтверждения, линии с {OBLIQUE[k]:.0f}°: p = {p:.5f} (порог "
            f"{alpha:.6f}) -> {mark[p <= alpha]}")

    Lp, _c, _d = layer([perturb(h, valid) for h in hs], valid, res_m, link)
    mC, mP = mask_of(lines, valid.shape), mask_of(Lp, valid.shape)
    stab = cover(mC, mP, res_m, STAB_TOL_M)
    rr = np.random.default_rng(7)
    null = np.mean([cover(mC, np.roll(np.roll(mP, int(rr.integers(200, 3000)), 0),
                                      int(rr.integers(200, 3000)), 1), res_m, STAB_TOL_M)
                    for _ in range(20)])
    log(f"устойчивость: {stab:.3f}, нуль {null:.3f} -> "
        f"{mark[stab >= STAB_MIN and stab >= STAB_OVER_NULL * max(null, 1e-9)]}")

    work = np.load(work_npz(a.site))
    d_any = distance_transform_edt(~work["A_all"], sampling=(res_m, res_m))
    seen = sum(1 for x in lines
               if (d_any[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC)
    log(f"для сведения: из {len(lines)} линий слоя {seen} идут вдоль линий оптики "
        f"(включая неподтверждённые), {len(lines) - seen} оптика не видит совсем")
    np.savez_compressed(M / f"openlin_relief_layer_{a.site}{tag}.npz", C=mC)

    # ---- парные окна: снимок | отмывка | снимок с линиями ---------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    base = _stretch(np.where(valid, w["shade"], np.nan))
    shade = _stretch(np.where(valid, hs[0], np.nan))
    dens = uniform_filter(mC.astype("float32"), 250)
    dens[:125] = 0; dens[-125:] = 0; dens[:, :125] = 0; dens[:, -125:] = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    k3 = np.ones((3, 3), bool)
    for i in range(1, 21):
        r, c = np.unravel_index(int(np.argmax(dens)), dens.shape)
        if dens[r, c] <= 0:
            break
        dens[max(r - 250, 0):r + 250, max(c - 250, 0):c + 250] = 0
        sl = (slice(r - 125, r + 125), slice(c - 125, c + 125))
        fig, ax = plt.subplots(1, 3, figsize=(27, 9.6), facecolor="#fcfcfb")
        ax[0].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[0].set_title("зимний снимок — смотрите сами", loc="left", fontsize=15)
        ax[1].imshow(shade[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[1].set_title("отмывка рельефа, свет с северо-востока (45°)", loc="left",
                        fontsize=15)
        ax[2].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ov = np.zeros((250, 250, 4), "float32")
        ov[work["C_all"][sl]] = (0.85, 0.05, 0.05, 0.9)
        ov[binary_dilation(mC[sl], k3)] = (0.1, 0.4, 0.95, 1.0)
        ax[2].imshow(ov)
        ax[2].set_title("красные — по снимку (рабочий набор), синие — по рельефу",
                        loc="left", fontsize=15)
        for axx in ax:
            axx.set_xticks([]); axx.set_yticks([])
        fig.suptitle(f"Окно {i} · 5 × 5 км · центр ({r}, {c}) · север сверху",
                     x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"{i:02d}_окно_{r}_{c}.png", dpi=80,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"готово: {out_dir} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
