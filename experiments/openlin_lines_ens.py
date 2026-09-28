"""Ансамбль трассировок: вместо одного списка линий — частота линии в ячейке.

ЗАЧЕМ. Одиночная трассировка не воспроизводится: два списка линий, полученные на
двух подвыборках ОДНОГО И ТОГО ЖЕ обучения, совпадают по медиане на 0.18
(``openlin_lines --stability``). Ломается шаг бинаризации, а не модель — тот же
вывод ветка уже получала на Canny->Hough. Лечение то же, что и там: вернуться к
непрерывной величине, но уже в линейной геометрии. Обучение повторяется на
``N_ENSEMBLE`` подвыборках, каждая даёт свой список линий, и в ответ идёт доля
подвыборок, в которых линия прошла через ячейку. Это снова поле, а не список,
и оно наследует устойчивость от усреднения; заодно частота — готовая мера
уверенности для каждого отрезка.

ЧТО ПРОВЕРЯЕТСЯ.

1. Качество консенсусных линий (частота >= ``CONSENSUS``) той же мерой, что и у
   одиночной трассировки: доля их длины в допуске от карты, против сдвигового
   нуля по тем же линиям.
2. Устойчивость самого ансамбля: подвыборки делятся пополам, две половины дают
   два независимых поля частоты, и они сравниваются между собой — и как поля
   (Спирмен), и как консенсусные списки линий (взаимное покрытие). Если
   усреднение работает, эти числа должны быть заметно выше 0.18 у одиночной
   трассировки.

Запуск из корня репозитория:
    python -m experiments.openlin_lines_ens --site nevada --apply-to anabar,anabar_dykes
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import (N_TRAIN_ROWS, SEED, TARGET_HALFWIDTH_M,
                                       TOL_M, anfis_predict, baseline_predict,
                                       block_split, fit_anfis, fit_baselines,
                                       rows, score_map, soft_target)
from experiments.openlin_lines import (DENSITY, DENSITY_DEFAULT, agreement,
                                       build, line_metrics, shift_null,
                                       trace_lines)

M = Path("outputs/metrics")

# --------------------------------------------------------------------------
# Параметры ансамбля
# --------------------------------------------------------------------------

#: Сколько подвыборок в ансамбле. 12 — компромисс: разрешение частоты 1/12
#: достаточно, чтобы отличить «линия была всегда» от «линия была раз-два», и
#: ансамбль ещё делится пополам на две независимые половины для проверки
#: устойчивости. Больше упирается во время: одна подвыборка — три обучения.
N_ENSEMBLE = 12

#: Порог консенсуса: линия попадает в ответ, если прошла через ячейку не меньше
#: чем в половине подвыборок. Половина — не подобранное число, а определение
#: большинства; пороги 0.25 и 0.75 считаются и печатаются рядом, но тест по ним
#: не проводится, иначе на каждую клетку приходилось бы по три попытки.
CONSENSUS = 0.5
REPORT_LEVELS = (0.25, 0.5, 0.75)

N_SHIFTS = 8000

#: Пре-регистрация. Свой addend в своей секции. Список кандидатов тот же самый —
#: 7 пар (площадка обучения, площадка оценки) x 3 модели x 2 таргета = 42.
#: ПЕРВИЧНАЯ МЕТРИКА: точность консенсусных линий при частоте >= CONSENSUS.
#: Устойчивость (Спирмен полей и взаимное покрытие списков) — диагностика, а не
#: тест: у неё нет нулевой гипотезы против карты.
LNE_N_TESTS_PRIOR = 126        # 42 поячеечных + 42 с окрестностью + 42 линий
LNE_N_TESTS = 42
LNE_ALPHA = 0.05 / (LNE_N_TESTS_PRIOR + LNE_N_TESTS)

#: Достижимость: 1/(1+8000) = 0.000125 против alpha = 0.00030, запас 2.4x.
#: При прежних 4000 сдвигах запас был бы 1.2x — тест почти непроходим, поэтому
#: сдвигов здесь вдвое больше, чем в прогоне одиночной трассировки.
assert 1.0 / (1 + N_SHIFTS) < LNE_ALPHA, "порог недостижим при таком числе сдвигов"


def _disk(radius_px: float) -> np.ndarray:
    r = int(round(radius_px))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def widen(lines: np.ndarray, res_m: float) -> np.ndarray:
    """Линия -> полоса шириной в допуск сверки.

    Частоту нельзя копить по точному совпадению ячеек: линия толщиной в одну
    ячейку почти никогда не попадает в ту же ячейку дважды, и две находки одного
    и того же разлома со смещением в 200 м считались бы разными. Весь проект
    сверяет положение с допуском ``TOL_M``, и накопление идёт с ним же.
    """
    from scipy.ndimage import binary_dilation

    return binary_dilation(lines, _disk(TOL_M / res_m))


def thin(band: np.ndarray) -> np.ndarray:
    """Полоса консенсуса -> осевая линия, чтобы мерить длину, а не площадь."""
    from skimage.morphology import skeletonize

    return skeletonize(band)


def spearman(f1: np.ndarray, f2: np.ndarray, area: np.ndarray,
             res_m: float) -> float:
    """Спирмен между двумя полями частоты по ВСЕЙ области оценки.

    Считать его по объединению ненулевых ячеек нельзя: поля разреженные, и на
    таком объединении «единица там, где у другого ноль» даёт корреляцию около
    -1 просто по построению, независимо от того, похожи поля или нет. Поэтому
    поля сначала размываются на километр (тогда близкая, но не совпавшая линия
    даёт вклад, а не обнуляется), и корреляция берётся по всей области.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import spearmanr

    s = 1000.0 / res_m
    a = gaussian_filter(f1.astype("float32"), s)[area]
    b = gaussian_filter(f2.astype("float32"), s)[area]
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="nevada",
                    help="площадка(и) обучения, через запятую")
    ap.add_argument("--apply-to", default="")
    ap.add_argument("--members", type=int, default=N_ENSEMBLE)
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--rules", type=int, default=16)
    ap.add_argument("--density", default=DENSITY_DEFAULT, choices=tuple(DENSITY))
    ap.add_argument("--models", default="ANFIS,логрегрессия,бустинг",
                    help="какие модели считать, через запятую")
    ap.add_argument("--targets", default="hard,smooth")
    ap.add_argument("--tag", default="", help="суффикс к именам файлов результата")
    args = ap.parse_args()
    want_models = [m for m in args.models.split(",") if m]
    want_targets = [t for t in args.targets.split(",") if t]

    def log(m=""):
        print(m, flush=True)

    train_sites = [s for s in args.site.split(",") if s]
    apply_sites = [s for s in args.apply_to.split(",") if s
                   and s not in train_sites]
    log(f"обучение на {', '.join(train_sites)}, "
        f"перенос на {', '.join(apply_sites) or '-'}")
    log(f"ансамбль: {args.members} подвыборок, консенсус {CONSENSUS}, "
        f"плотность '{args.density}'")
    log(f"пре-регистрация: первичная метрика — точность консенсусных линий; "
        f"тестов в семействе {LNE_N_TESTS_PRIOR + LNE_N_TESTS}, "
        f"alpha = {LNE_ALPHA:.5f}, "
        f"минимально достижимое p = {1 / (1 + N_SHIFTS):.6f}")

    # --- данные: признаки строятся один раз, они не зависят от подвыборки ---
    jobs = []          # (площадка, маска области оценки, режим, z, x)
    t0 = time.time()
    from experiments.openlin_unet import load_site

    # Обучающие строки собираются со ВСЕХ площадок обучения; окрестность у
    # каждой берётся только из её собственной обучающей части, как и в прогоне
    # с одной площадкой.
    xtr_parts, dtr_parts = [], []
    for s in train_sites:
        zz = load_site(s)
        vv = np.isfinite(zz["gray"])
        t, v = block_split(vv.shape, float(zz["res_m"]))
        t &= vv
        v &= vv
        _z, xt, _v = build(s, t, True)
        _zv, xv, _vv = build(s, v, True)
        jobs.append((s, v, "проверка", zz, xv))
        xtr_parts.append(rows(xt, t))
        dtr_parts.append(zz["dist"][t])
        log(f"  {s}: обучение {int(t.sum()):,} яч., проверка {int(v.sum()):,} яч.")
    for s in apply_sites:
        z, x, valid = build(s, None, True)
        jobs.append((s, valid, "перенос", z, x))
    xtr_full = np.concatenate(xtr_parts)
    dtr_full = np.concatenate(dtr_parts)
    log(f"признаки построены за {time.time() - t0:.0f} с; "
        f"обучающих ячеек {len(xtr_full):,}, признаков {xtr_full.shape[1]}\n")

    # --- накопление частоты -------------------------------------------------
    tag = "+".join(train_sites)
    acc: dict[tuple, list[np.ndarray]] = {}
    for tname, soft in [t for t in (("hard", False), ("smooth", True))
                        if t[0] in want_targets]:
        for sub in range(args.members):
            rng = np.random.default_rng(SEED + 1000 * sub)
            xtr, dtr = xtr_full, dtr_full
            if len(xtr) > N_TRAIN_ROWS:
                sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
                xtr, dtr = xtr[sel], dtr[sel]
            ytr = (soft_target(dtr) if soft
                   else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))

            t1 = time.time()
            models = ({"ANFIS": fit_anfis(xtr, ytr, args.rules, soft,
                                          lambda *_: None)}
                      if "ANFIS" in want_models else {})
            models.update({k: v for k, v in fit_baselines(xtr, ytr, soft).items()
                           if k in want_models})
            for mname, est in models.items():
                for site, area, mode, z, x in jobs:
                    pred_rows = (anfis_predict(est, rows(x, area)) if mname == "ANFIS"
                                 else baseline_predict(est, rows(x, area)))
                    sc = score_map(pred_rows, area)
                    lines, _br = trace_lines(sc, np.isfinite(z["gray"]), area,
                                             float(z["res_m"]), args.density)
                    acc.setdefault((tname, mname, site), []).append(
                        widen(lines, float(z["res_m"])))
            log(f"  {tname}, подвыборка {sub + 1}/{args.members}: "
                f"{time.time() - t1:.0f} с")

    # --- оценка -------------------------------------------------------------
    log()
    out = []
    for (tname, mname, site), masks in acc.items():
        area, mode, z = next((a, m, zz) for s, a, m, zz, _x in jobs if s == site)
        res_m = float(z["res_m"])
        freq = np.mean(np.stack(masks).astype("float32"), axis=0)

        row = {"таргет": tname, "модель": mname, "площадка": site, "режим": mode,
               "подвыборок": len(masks),
               "ячеек_хоть_раз": int((freq > 0).sum())}
        for lv in REPORT_LEVELS:
            cons = thin((freq >= lv) & area) & area
            r = line_metrics(cons, z["dist"], area, res_m)
            lvl = f"{int(lv * 100)}"        # не 'tag': им зовётся имя прогона
            row[f"точность_{lvl}"] = r["точность"]
            row[f"полнота_{lvl}"] = r["полнота"]
            row[f"длина_км_{lvl}"] = r["длина_км"]

        cons = thin((freq >= CONSENSUS) & area) & area
        if not args.no_null and cons.any():
            row.update(shift_null(cons, z["dist"], area,
                                  row[f"точность_{int(CONSENSUS * 100)}"],
                                  n_shifts=N_SHIFTS))
            row["значим"] = row["p"] < LNE_ALPHA

        # устойчивость: две половины ансамбля друг против друга
        h = len(masks) // 2
        f1 = np.mean(np.stack(masks[:h]).astype("float32"), axis=0)
        f2 = np.mean(np.stack(masks[h:]).astype("float32"), axis=0)
        row["спирмен_половин"] = spearman(f1, f2, area, res_m)
        row["совпадение_половин"] = agreement(
            thin((f1 >= CONSENSUS) & area) & area,
            thin((f2 >= CONSENSUS) & area) & area, res_m)
        out.append(row)

        np.savez_compressed(
            M / f"openlin_lines_ens_{tag}{args.tag}_{mname}_{tname}"
                f"_{site}.npz",
            freq=freq, half1=f1, half2=f2)
        pp = f"  p = {row['p']:.5f}" if "p" in row else ""
        log(f"  {mname:<14} {site:<14} {mode:<9} "
            f"точность {row['точность_50']:.3f}  полнота {row['полнота_50']:.3f}  "
            f"спирмен половин {row['спирмен_половин']:.2f}  "
            f"совпадение половин {row['совпадение_половин']:.2f}{pp}")

    df = pd.DataFrame(out)
    path = M / f"openlin_lines_ens_{tag}{args.tag}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"\nсохранено: {path}")
    log(f"медиана спирмена половин: {df['спирмен_половин'].median():.3f}  "
        f"(у одиночной трассировки совпадение списков было 0.18)")


if __name__ == "__main__":
    main()
