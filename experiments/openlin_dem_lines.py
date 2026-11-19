"""Линеаменты по рельефу: две независимые модели рельефа вместо снимка.

ОТКУДА ПОСТАНОВКА. 21.10.2026 заказчик снял оговорку «рельеф — лучше без
него». Карта по снимку остаётся независимым вторым прибором для сверки, но
основной вариант теперь может строиться по рельефу.

ДВЕ МОДЕЛИ. Подтверждение в рабочем методе — две независимые съёмки (половины
сцен). У рельефа аналог — две модели, снятые разными приборами:
  * Copernicus GLO-30 — радарная интерферометрия TanDEM-X, 2011-2015;
  * ALOS AW3D30 — оптическая стереосъёмка PRISM, 2006-2011.
Обе 30 м, обе покрывают лист (8 плиток каждая). Вторая отмывка той же модели
(как в :mod:`openlin_relief_layer`) независимым измерением не является.
Замер связи (21.10.2026): после вычитания фона 100 м корреляция моделей 0.867,
СКО разности 1.05 м при СКО мелкого рельефа 1.8-2.1 м.

ВХОД — УКЛОН (замер :mod:`openlin_dem_probe`, 21.10.2026, детектор без
продления и сшивки): подтверждено второй моделью 1657 линий / 1809 км при
10 у двойника — мусор 0.6 %. «Рельеф минус фон» даёт больше (1890 / 2599 км),
но мусор 13.2 %; отмывки 45° и 135° — 1455-1480 линий, мусор 9.3 и 5.9 %.
Уклон взят по наименьшему мусору; направление света у него не выбирается,
слепых секторов нет.

ПРОГОН DEM (объявлено 21.10.2026, до прогона).
ЧТО ДЕЛАЕТСЯ: уклон GLO-30 (градусы, минус фон ``detrend``) -> тракт рабочего
метода без изменений: детектор, продление концов, сшивка с пределом бокового
сдвига ``TWIN_OFFSET_M`` = 127 м, «цепочки» (мостики через кулисы не
рисуются), линии от ``MIN_LEN_M`` = 500 м. Отсева швов съёмки нет: у модели
рельефа нет сцен. Подтверждение — линии детектора по уклону ALOS БЕЗ сшивки
(эталон без сшивки — правило из :mod:`openlin_relief_layer`: сшитый эталон
дважды впускал шум): не меньше ``CONF_FRAC`` = 30 % длины в ``CONF_TOL_M`` =
60 м. Под рельеф ни одно число не подбирается — все из рабочего метода.
УСЛОВИЯ ПРИЁМКИ (те же, что у рабочего метода):
  1. мусор: подтверждённых линий у двойников (``N_TWINS`` общих подделок
     обоих уклонов) в среднем не больше ``STAGE_JUNK_MAX`` = 20 % от снимка;
  2. сдвиговый нуль подтверждения: p <= ``DEM_ALPHA``;
  3. устойчивость при возмущении обоих входов: >= 0.5 и >= 2 нулей;
  4. длина: доля подтверждённых линий длиннее 95-го процентиля длины линий
     двойников после сшивки — не меньше 25 % (``LEN_*`` рабочего метода).
СЧЁТЧИК: после ``CLEAN1K`` 104, этот прогон добавляет 4.
РЕЗУЛЬТАТ (21.10.2026): ВСЕ ЧЕТЫРЕ УСЛОВИЯ ПРОЙДЕНЫ. После сшивки 3212 линий /
3738 км, подтверждено второй моделью 1690 линий / 2236 км, медиана 1.13 км.
Мусор: у двойников 16 / 24 / 46 против 1690 — 1.7 %. Сдвиговый нуль: 53 % на
месте против 4.2 %, p = 0.0002 (пол). Устойчивость 0.821 при нуле 0.033.
Длина: 95-й процентиль двойников 1.16 км, длиннее — 47.6 %. Прогон 11 минут.
Для сравнения рабочий набор по снимку: 1861 / 2381 км, мусор 4.8 %,
устойчивость 0.705, длинных 37.4 %. Наборы НЕ одно и то же: 1072 линии
рельефа идут вдоль линий оптики, 618 оптика не видит, а в 60 м от линий
рельефа лежит лишь 39 % ячеек рабочего набора оптики. Глазом не оценён.

ПРОГОН DEM1 — одна линия на склон (объявлено 21.10.2026, до прогона).
ПОВОД — просмотр окон прогона DEM: борт долины выходит двумя-тремя
параллельными линиями. Причина в отклике: ``ridge_response`` берёт модуль
кривизны, поэтому на поле уклона ловит и гребень (максимум крутизны —
середина уступа), и ложбины по бокам (перегиб у бровки и у подножия).
ЧТО МЕНЯЕТСЯ: только знак — отклик ``scarp_response`` оставляет ячейки, где
наибольшая по модулю кривизна уклона отрицательна (гребень поля уклона), и
обнуляет остальные. Применяется везде, где у прогона DEM был отклик: детектор
GLO, продление и опора мостика (z), эталон ALOS. Ни одно число не меняется.
УСЛОВИЯ — те же четыре. СЧЁТЧИК: после ``DEM`` 108, этот прогон добавляет 4.
РЕЗУЛЬТАТ (21.10.2026): ВСЕ ЧЕТЫРЕ УСЛОВИЯ ПРОЙДЕНЫ. Подтверждено 1568 линий /
2129 км (было 1690 / 2236), медиана 1.16 км. Мусор 1.5 % (29 / 18 / 23 у
двойников), сдвиговый нуль 56 % против 6.7 %, p = 0.0002, устойчивость 0.868
при нуле 0.033, длиннее 95-го процентиля двойников (0.95 км) — 64.8 %.
Глазом: двойные и тройные линии по бортам ушли, остаётся одна по самому
крутому месту. Но добавленные линии рельефа часто извилистые — повторяют
бровку склона вокруг холмов, местами замыкаются в петли.
Сложение: к карте по снимку добавлено 932 линии / 1140 км, 636 уже на ней.

СЛОЖЕНИЕ С КАРТОЙ ПО СНИМКУ (ключ ``--сложить``, после прогона, не проверка):
линия рельефа добавляется к рабочему набору оптики, если меньше ``CONF_FRAC``
её длины лежит в ``CONF_TOL_M`` от линий набора оптики — то же правило «это
тот же объект», что у подтверждения. В счётчик не идёт.

Запуск из корня репозитория:
    python -m experiments.openlin_dem_lines --site anabar
    python -m experiments.openlin_dem_lines --site anabar --одна-линия --сложить
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from cache_paths import ANABAR_DEM

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
N_TWINS = 3                                             # как N_FAKES рабочего метода
DEM_N_TESTS_PRIOR = 104                                 # после CLEAN1K (98 + 6)
DEM_N_TESTS = 4
DEM_ALPHA = 0.05 / (DEM_N_TESTS_PRIOR + DEM_N_TESTS)    # 0.000463
#: Сдвигов: n >= 2/alpha - 1 = 4319; берётся 5000 (как CLEAN_SHIFT_N), пол
#: 0.0002, запас 2.3x.
DEM_SHIFT_N = 5000
DEM1_N_TESTS_PRIOR = DEM_N_TESTS_PRIOR + DEM_N_TESTS    # 108
DEM1_N_TESTS = 4
DEM1_ALPHA = 0.05 / (DEM1_N_TESTS_PRIOR + DEM1_N_TESTS)  # 0.000446
#: Сдвигов: n >= 2/alpha - 1 = 4479; 5000, пол 0.0002, запас 2.2x.
DEM1_SHIFT_N = 5000


def scarp_response(img: np.ndarray) -> np.ndarray:
    """``ridge_response``, но только гребни поля (кривизна поперёк отрицательна)."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    from experiments.openlin_iso import SIGMAS

    best = np.zeros(img.shape, dtype="float32")
    for s in SIGMAS:
        l1, l2 = hessian_matrix_eigvals(hessian_matrix(img, sigma=s, order="rc",
                                                       use_gaussian_derivatives=True))
        big = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
        r = s ** 2 * (np.maximum(np.abs(l1), np.abs(l2)) - np.minimum(np.abs(l1), np.abs(l2)))
        best = np.maximum(best, np.where(big < 0, r, 0.0).astype("float32"))
    return best


def load_alos(site: str, res_m: float) -> np.ndarray:
    """ALOS AW3D30 на сетке листа (кэш); плитки с Planetary Computer, окном."""
    import planetary_computer
    from pystac_client import Client
    from rasterio.enums import Resampling

    from experiments.openlin_probe import _read_on_grid
    from experiments.openlin_winter import site_grid

    path = ANABAR_DEM / f"dem_alos30_{site}_лист_{int(res_m)}m.npy"
    if path.exists():
        return np.load(path)
    grid = site_grid(site, res_m)
    items = list(Client.open(PC_STAC).search(collections=["alos-dem"],
                                             bbox=grid.bbox_wgs84()).items())
    dem = np.zeros((grid.height, grid.width), "float32")
    for it in items:
        href = planetary_computer.sign(it.assets["data"].href)
        part = _read_on_grid(href, grid, Resampling.bilinear)
        dem = np.where(part != 0, part, dem)
        print(f"  {it.id}: покрыто {100 * (part != 0).mean():.0f} % листа", flush=True)
    dem = np.where(dem != 0, dem, np.nan).astype("float32")
    ANABAR_DEM.mkdir(parents=True, exist_ok=True)
    np.save(path, dem)
    return dem


def slope(dem: np.ndarray, valid: np.ndarray, res_m: float) -> np.ndarray:
    """Уклон в градусах минус плавный фон — вход детектора."""
    from experiments.openlin_winter import detrend

    f = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    gy, gx = np.gradient(f, res_m)
    return detrend(np.degrees(np.arctan(np.hypot(gx, gy))), valid, res_m)


def response(img, valid, res_m, fn=None):
    """Отклик в сигмах окрестности — как :func:`openlin_vote.response`, с подменой."""
    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize

    x = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    z = local_norm(np.where(valid, (fn or ridge_response)(x), 0.0), valid, res_m)
    return np.where(valid, z, 0.0).astype("float32")


def tract(im, valid, res_m, fn=None):
    """Тракт рабочего метода: продление, сшивка со сдвигом до 127 м, «цепочки»."""
    from experiments.openlin_zone_lines import MIN_LEN_M, trace
    from experiments.openlin_zone_lines_v3 import (TWIN_OFFSET_M, extend_ends,
                                                   link_same_side)

    x = np.where(valid, im, np.nan).astype("float32")
    z = response(x, valid, res_m, fn)
    raw = extend_ends(trace(x, valid, res_m, local=True, min_len_m=0, response_fn=fn),
                      z, valid, res_m)
    L = [r for r in link_same_side(raw, None, None, res_m, None, z=z,
                                   offset_m=TWIN_OFFSET_M)
         if r["длина_км"] * 1000 >= MIN_LEN_M]
    W_ = valid.shape[1]
    for r in L:
        kul = r.get("кулисы")
        if kul is not None and len(kul):
            r["pts"] = r["pts"][~np.isin(r["pts"][:, 0] * W_ + r["pts"][:, 1],
                                         kul[:, 0] * W_ + kul[:, 1])]
    return L


def build(xg, xa, valid, res_m, fn=None):
    """(все линии GLO, подтверждённые ALOS, карта расстояний до эталона ALOS)."""
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_zone_lines import mask_of, trace
    from experiments.openlin_zone_lines_v3 import CONF_FRAC, CONF_TOL_M

    ref = trace(np.where(valid, xa, np.nan), valid, res_m, local=True, response_fn=fn)
    d = distance_transform_edt(~mask_of(ref, valid.shape), sampling=(res_m, res_m))
    L = tract(xg, valid, res_m, fn)
    C = [x for x in L
         if (d[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC]
    return L, C, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--одна-линия", dest="one", action="store_true",
                    help="прогон DEM1: только гребни поля уклона (scarp_response)")
    ap.add_argument("--сложить", dest="union", action="store_true",
                    help="сложить подтверждённые линии с рабочим набором оптики")
    a = ap.parse_args()
    fn = scarp_response if a.one else None
    name = "DEM1, одна линия на склон" if a.one else "DEM"
    n_tot = (DEM1_N_TESTS_PRIOR + DEM1_N_TESTS) if a.one else (DEM_N_TESTS_PRIOR + DEM_N_TESTS)
    alpha, n_sh = (DEM1_ALPHA, DEM1_SHIFT_N) if a.one else (DEM_ALPHA, DEM_SHIFT_N)
    tag = "_одна" if a.one else ""

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_iso import perturb
    from experiments.openlin_kora import surrogate
    from experiments.openlin_relief_probe import load_dem
    from experiments.openlin_zone_lines import (COMP, M, STAB_MIN, STAB_OVER_NULL,
                                                STAB_TOL_M, cover, mask_of)
    from experiments.openlin_zone_lines_v3 import (CONF_FRAC, CONF_TOL_M,
                                                   LEN_NULL_Q, LEN_SHARE_MIN,
                                                   STAGE_JUNK_MAX, shift_null_conf,
                                                   work_npz)

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    g, al = load_dem(a.site, res_m), load_alos(a.site, res_m)
    valid = np.isfinite(g) & np.isfinite(al) & np.isfinite(w["shade"])
    xg, xa = slope(g, valid, res_m), slope(al, valid, res_m)
    log(f"прогон {name}: уклон GLO-30, подтверждение уклоном ALOS; порог значимости "
        f"0.05/{n_tot} = {alpha:.6f}, сдвигов {n_sh}")

    L, C, d_ref = build(xg, xa, valid, res_m, fn)
    km = lambda S: sum(x["длина_км"] for x in S)
    log(f"линий после сшивки {len(L)} ({km(L):.0f} км); подтверждено {len(C)} "
        f"({km(C):.0f} км), медиана длины {np.median([x['длина_км'] for x in C]):.2f} км "
        f"({time.time() - t0:.0f} с)")

    stack = np.stack([xg, xa]).astype("float32")
    fake, fLen = [], []
    for j in range(N_TWINS):
        tw = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900 + j))
        Lt, Ct, _d = build(tw[0], tw[1], valid, res_m, fn)
        fake.append(len(Ct))
        fLen += [x["длина_км"] for x in Lt]
        log(f"  двойник {j + 1}: линий {len(Lt)}, подтверждено {len(Ct)} "
            f"({time.time() - t0:.0f} с)")

    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    junk = float(np.mean(fake)) / max(len(C), 1)
    log(f"\n1. мусор: у двойников в среднем {np.mean(fake):.1f} против {len(C)} = "
        f"{100 * junk:.1f} % (порог {100 * STAGE_JUNK_MAX:.0f} %) -> "
        f"{mark[junk <= STAGE_JUNK_MAX]}")
    _s, p = shift_null_conf(L, [d_ref], valid.shape, log, n_sh)
    log(f"2. сдвиговый нуль подтверждения: p = {p:.5f} (порог {alpha:.6f}) -> "
        f"{mark[p <= alpha]}")

    _L, Cp, _d = build(perturb(xg, valid), perturb(xa, valid), valid, res_m, fn)
    mC, mP = mask_of(C, valid.shape), mask_of(Cp, valid.shape)
    stab = cover(mC, mP, res_m, STAB_TOL_M)
    rr = np.random.default_rng(7)
    null = np.mean([cover(mC, np.roll(np.roll(mP, int(rr.integers(200, 3000)), 0),
                                      int(rr.integers(200, 3000)), 1), res_m, STAB_TOL_M)
                    for _ in range(20)])
    log(f"3. устойчивость: {stab:.3f}, нуль {null:.3f} -> "
        f"{mark[stab >= STAB_MIN and stab >= STAB_OVER_NULL * max(null, 1e-9)]}")

    len_thr = float(np.quantile(np.array(fLen), LEN_NULL_Q)) if fLen else 0.0
    sh = float(np.mean([x["длина_км"] > len_thr for x in C])) if C else 0.0
    log(f"4. длина: {100 * LEN_NULL_Q:.0f}-й процентиль длины линий двойников "
        f"{len_thr:.2f} км ({len(fLen)} линий); длиннее — {100 * sh:.1f} % "
        f"(порог {100 * LEN_SHARE_MIN:.0f} %) -> {mark[sh >= LEN_SHARE_MIN]}")
    log(f"   время {time.time() - t0:.0f} с")

    work = np.load(work_npz(a.site))
    d_opt = distance_transform_edt(~work["A_all"], sampling=(res_m, res_m))
    seen = sum(1 for x in C
               if (d_opt[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC)
    d_dem = distance_transform_edt(~mC, sampling=(res_m, res_m))
    cw = work["C_all"]
    opt_cov = float((d_dem[cw] <= CONF_TOL_M).mean()) if cw.any() else 0.0
    log(f"\nдля сведения: из {len(C)} линий рельефа {seen} идут вдоль линий оптики "
        f"(весь набор A_all), {len(C) - seen} оптика не видит; "
        f"{100 * opt_cov:.0f} % ячеек рабочего набора оптики лежит в 60 м от линий рельефа")
    out = M / f"openlin_dem_lines_{a.site}{tag}.npz"
    extra = {}
    if a.union:
        d_map = distance_transform_edt(~cw, sampling=(res_m, res_m))
        add = [x for x in C
               if (d_map[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() < CONF_FRAC]
        n_opt = int(work["n_C_all"]) if "n_C_all" in work else None
        extra = {"add": mask_of(add, valid.shape), "union": cw | mask_of(add, valid.shape)}
        log(f"\nсложение: к карте по снимку добавлено {len(add)} линий рельефа "
            f"({km(add):.0f} км); остальные {len(C) - len(add)} уже есть на карте"
            + (f" (линий оптики {n_opt})" if n_opt else ""))
    np.savez_compressed(out, C=mC, A=mask_of(L, valid.shape), res_m=res_m, **extra)
    # поштучно — для векторного экспорта (openlin_export_vectors)
    from experiments.openlin_zone_lines import save_line_sets
    save_line_sets(out.with_name(out.stem + "_линии.npz"), A=L, C=C,
                   **({"add": add} if a.union else {}))
    log(f"сохранено: {out}")


if __name__ == "__main__":
    main()
