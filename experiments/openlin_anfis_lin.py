"""ANFIS с новым таргетом: не разломы, а подтверждённые линеаменты.

ОТКУДА ВОПРОС. Всё обучение в этой ветке шло по расстоянию до РАЗЛОМА, то есть
подгонялось под карту, которую со снимка воспроизвести нельзя (на Анабаре целью
вообще оказался контур зоны шириной 5-9 км). Теперь цель есть своя: 255 линий,
выделенных по снимку без разметки и подтверждённых независимо оптикой и радаром
(``openlin_unsup``). Вопрос: есть ли у линеамента ПОЯЧЕЕЧНАЯ спектральная
подпись — такая, чтобы модель, обученная на одной площадке, находила линеаменты
на другой.

ДВА РЕШЕНИЯ КОНСТРУКЦИИ, БЕЗ КОТОРЫХ ВОПРОС БЫЛ БЫ ПУСТЫМ.

1. ВХОД — ТОЛЬКО 10 СЫРЫХ КАНАЛОВ. Признаки окрестности сюда не подаются: сам
   оператор, которым добыты линеаменты, построен на них, и модель с ними на
   входе просто повторила бы его. Спрашивается именно про спектр точки.
2. ГЛАВНАЯ ПРОВЕРКА — ПЕРЕНОС. Внутри своей площадки высокий отклик означал бы
   лишь, что поячеечная модель воспроизводит соседский оператор на тех же
   данных (дистилляция). Содержателен только перенос на площадку, линеаменты
   которой добыты независимо тем же способом.

Блочное разбиение с буфером 2 км и сдвиговый нуль — как во всей ветке.

Пре-регистрация: свой addend в своей секции, знаменатель накопительный.

Запуск из корня репозитория:
    python -m experiments.openlin_anfis_lin --site anabar --apply-to nevada
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.openlin_anfis import (AREA, N_RULES, N_TRAIN_ROWS, SEED,
                                       TARGET_HALFWIDTH_M, TOL_M,
                                       TARGET_SOFT_M, anfis_predict,
                                       baseline_predict, block_split,
                                       fit_anfis, fit_baselines, rows,
                                       score_map)
from experiments.openlin_unet import CHANNELS, build_stack, lift_at_top, load_site
from src import assessment

M = Path("outputs/metrics")

#: С какой плотности берём подтверждённые линеаменты. Та же, на которой построена
#: итоговая карта.
LIN_DENSITY = "предельно"

N_SHIFTS = 4000

#: Пре-регистрация. Своё слагаемое в своей секции; предыдущие константы ветки не
#: трогаем, на них ссылаются опубликованные пороги.
#: Кандидаты объявлены до просмотра результатов: 2 площадки обучения
#: (anabar, nevada) x 2 режима (своя проверка, перенос) x 3 модели x 2 таргета
#: (жёсткий, плавный) = 24.
#: ПЕРВИЧНАЯ МЕТРИКА — lift@10% на ПЕРЕНОСЕ: внутри своей площадки высокий
#: отклик не является ответом на поставленный вопрос (см. шапку).
LIN_N_TESTS_PRIOR = 195        # 171 структурной ветки + 24 многоплощадочных
LIN_N_TESTS = 24
LIN_ALPHA = 0.05 / (LIN_N_TESTS_PRIOR + LIN_N_TESTS)

#: Достижимость: 1/(1+4000) = 0.00025 против alpha = 0.00023 — НЕ проходит.
#: Поэтому сдвигов вдвое больше: 1/(1+8000) = 0.000125, запас 1.8x.
N_SHIFTS = 8000
assert 1.0 / (1 + N_SHIFTS) < LIN_ALPHA, "порог недостижим при таком числе сдвигов"


def lineament_dist(site: str, res_m: float) -> np.ndarray:
    """Расстояние до ближайшего подтверждённого линеамента, м."""
    from scipy.ndimage import distance_transform_edt

    p = M / f"openlin_unsup_{site}_{LIN_DENSITY}.npz"
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала python -m experiments.openlin_unsup "
                         f"--site {site} --density '{LIN_DENSITY}'")
    lines = np.load(p)["confirmed"]
    return distance_transform_edt(~lines, sampling=(res_m, res_m)).astype("float32")


def soft(dist: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - dist / TARGET_SOFT_M, 0.0, 1.0).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--apply-to", default="nevada")
    ap.add_argument("--rules", type=int, default=N_RULES)
    ap.add_argument("--no-null", action="store_true")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    apply_sites = [s for s in a.apply_to.split(",") if s]
    log(f"обучение {a.site}, перенос {', '.join(apply_sites)}; "
        f"таргет — подтверждённые линеаменты ('{LIN_DENSITY}')")
    log(f"вход: только {len(CHANNELS)} сырых каналов, без признаков окрестности")
    log(f"пре-регистрация: первичная метрика — lift на ПЕРЕНОСЕ; тестов "
        f"{LIN_N_TESTS_PRIOR + LIN_N_TESTS}, alpha = {LIN_ALPHA:.6f}, "
        f"минимальное p = {1 / (1 + N_SHIFTS):.6f}\n")

    data = {}
    for s in [a.site] + apply_sites:
        z = load_site(s)
        x, _y, valid, use = build_stack(z, CHANNELS)
        res_m = float(z["res_m"])
        d = lineament_dist(s, res_m)
        data[s] = {"x": x, "valid": valid, "res": res_m, "dist": d}
        log(f"  {s}: каналов {len(use)}, доля ячеек у линеамента "
            f"{100 * float(((d <= TOL_M) & valid).mean()):.1f}%")

    d0 = data[a.site]
    tr, va = block_split(d0["valid"].shape, d0["res"])
    tr &= d0["valid"]
    va &= d0["valid"]
    xtr = rows(d0["x"], tr)
    dtr = d0["dist"][tr]
    rng = np.random.default_rng(SEED)
    if len(xtr) > N_TRAIN_ROWS:
        sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
        xtr, dtr = xtr[sel], dtr[sel]
    log(f"\nобучающих строк {len(xtr):,}")

    out = []
    for tname, is_soft in (("hard", False), ("smooth", True)):
        ytr = (soft(dtr) if is_soft
               else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))
        t0 = time.time()
        models = {"ANFIS": fit_anfis(xtr, ytr, a.rules, is_soft, lambda *_: None)}
        models.update(fit_baselines(xtr, ytr, is_soft))
        log(f"\n-- таргет {tname} (положительных "
            f"{100 * float((ytr > 0.5).mean()):.1f}%), обучено за "
            f"{time.time() - t0:.0f} с")

        for mname, est in models.items():
            def predict(xr, _e=est, _m=mname):
                return (anfis_predict(_e, xr) if _m == "ANFIS"
                        else baseline_predict(_e, xr))

            jobs = [(a.site, va, "проверка")] + [(s, data[s]["valid"], "перенос")
                                                 for s in apply_sites]
            for site, area, mode in jobs:
                d = data[site]
                sc = score_map(predict(rows(d["x"], area)), area)
                got = lift_at_top(np.nan_to_num(sc, nan=-1e9), d["dist"], area,
                                  fraction=AREA, tol_m=TOL_M)
                row = {"обучение": a.site, "площадка": site, "режим": mode,
                       "модель": mname, "таргет": tname, "lift": got["lift"],
                       "точность": got["точность"]}
                if not a.no_null:
                    meta = SimpleNamespace(prf=sc.shape[0], pic=sc.shape[1])
                    null = assessment.spatial_null_pvalue(
                        np.where(area, sc.astype(float), np.nan),
                        np.flatnonzero(((d["dist"] <= TOL_M) & area).ravel()),
                        meta, area=AREA, pool=np.flatnonzero(area.ravel()),
                        n_shifts=N_SHIFTS, seed=SEED)
                    row["p"] = null["p"]
                    row["значим"] = null["p"] < LIN_ALPHA
                out.append(row)
                pp = f"  p = {row['p']:.5f}" if "p" in row else ""
                log(f"   {mname:<14} {site:<10} {mode:<9} "
                    f"lift {got['lift']:.3f}{pp}")

    df = pd.DataFrame(out)
    path = M / f"openlin_anfis_lin_{a.site}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"\nсохранено: {path}")


if __name__ == "__main__":
    main()
