"""Сдвиговый нуль-тест для карт-кандидатов на Анабаре.

ЗАЧЕМ. Гибрид CNN+SVM даёт на Анабаре lift 1.57 против 1.27 у Кэнни и 0.92 у
случайной маски. Разрыв небольшой, а карта с сильной пространственной
автокорреляцией набирает lift выше единицы без всякой связи с целью — просто
потому, что её пятна совпадают с пятнами цели по случайности. Наивная планка
(случайная маска) этого не ловит: у неё автокорреляции нет вовсе.

Тест — тот же, что во всей структурной ветке проекта
(:func:`src.assessment.spatial_null_pvalue`): карта-кандидат циклически
сдвигается по обеим осям с отражениями, цель и пул ячеек стоят на месте. Нуль
отвечает на вопрос «а не даст ли столько же ЛЮБАЯ карта с такой же
зернистостью?».

ПРЕ-РЕГИСТРАЦИЯ ЗНАМЕНАТЕЛЯ. Против анабарской разметки в этой ветке уже
меряно шесть карт: перенос U-Net (оптика; оптика+радар), гибрид CNN+SVM
(оптика; оптика+радар; то же с признаками двух площадок) и Кэнни -> Хаф. Каждая
считалась на трёх допусках, то есть ``N_TESTS = 6 * 3 = 18``, и порог берётся
по всем ним сразу, а не по одному текущему прогону — иначе получается «перебирай,
пока что-нибудь не выиграет». Знаменатель структурной ветки
(``config.FZ_N_TESTS_PRIOR`` и далее) при этом НЕ трогается: там другие
кандидаты и другой протокол, и подмешивать сюда её счёт нельзя.

ДОСТИЖИМОСТЬ ПОРОГА проверяется явно: при ``n_shifts`` сдвигах минимальное
возможное p равно ``1/(1 + n_shifts)``, и если оно выше alpha, тест не сможет
пройти ни при каком результате.

Запуск из корня репозитория:
    python -m experiments.openlin_null
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import assessment

#: Число тороидальных сдвигов. То же, что в структурной ветке (FZ_N_SHIFTS).
N_SHIFTS = 5000
#: Пре-регистрированный знаменатель Бонферрони (см. докстринг).
N_TESTS = 18
ALPHA = 0.05 / N_TESTS
TOLERANCES_M = (250.0, 500.0, 1000.0)
SEED = 42

M = Path("outputs/metrics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--area", type=float, default=None,
                    help="доля площади для lift@area. По умолчанию — площадь "
                         "маски Кэнни (сравнение при равной площади). Но она "
                         "мала (~0.5%%, сотни ячеек), и нуль от этого широкий; "
                         "0.10 — стандартная площадь протокола проекта.")
    args = ap.parse_args()

    z = dict(np.load(M / f"openlin_{args.site}.npz"))
    dist = z["dist"]
    shape = dist.shape

    # Для целевой площадки берётся файл переноса, для площадки обучения — её
    # собственный прогон (файла «_to_себя» не бывает).
    up = M / f"openlin_unet_nevada_to_{args.site}_pred.npz"
    if not up.exists():
        up = M / f"openlin_unet_{args.site}_pred.npz"
    unet = dict(np.load(up))
    canny = unet["canny"].astype(float)
    area_mask = unet["area"].astype(bool)

    label = "U-Net перенос" if "_to_" in up.name else "U-Net"
    cands: dict[str, np.ndarray] = {"Кэнни→Хаф": canny,
                                    label: unet["prob"].astype(float)}
    for feat in ("nevada", "nevada+abitibi"):
        p = M / f"openlin_cnnsvm_{feat}_to_{args.site}_pred.npz"
        if p.exists():
            d = dict(np.load(p))
            cands[f"CNN+SVM ({feat})"] = d["score"].astype(float)
            # У гибрида сверка идёт только по своим полосам — берём их пул.
            area_mask = d["va_area"].astype(bool)

    # Площадь сравнения — та же, что у маски Кэнни: все кандидаты меряются на
    # одной доле площади, иначе сравнение включало бы выбор порога.
    area_frac = args.area or float(
        (canny.astype(bool) & area_mask).sum() / area_mask.sum())
    meta = SimpleNamespace(prf=shape[0], pic=shape[1])
    pool = np.flatnonzero(area_mask.ravel())

    p_min = 1.0 / (1 + N_SHIFTS)
    print(f"площадка «{args.site}»: сдвигов {N_SHIFTS}, alpha = 0.05/{N_TESTS} = "
          f"{ALPHA:.5f}; минимально достижимое p = {p_min:.5f} "
          f"({'порог достижим' if p_min < ALPHA else 'ПОРОГ НЕДОСТИЖИМ'})")
    print(f"площадь сравнения {100 * area_frac:.2f}%, ячеек пула {len(pool)}\n")

    rows = []
    for tol in TOLERANCES_M:
        target = np.flatnonzero(((dist <= tol) & area_mask).ravel())
        for name, score in cands.items():
            s = np.where(area_mask, score, np.nan)
            got = assessment.spatial_null_pvalue(
                s, target, meta, area=area_frac, pool=pool,
                n_shifts=N_SHIFTS, seed=SEED)
            rows.append({"кандидат": name, "допуск_м": tol, **got,
                         "значим": got["p"] < ALPHA})
            print(f"  {name:<26} допуск {tol:6.0f} м: lift {got['observed']:.3f}, "
                  f"нуль {got['null_mean']:.3f} (q95 {got['null_q95']:.3f}), "
                  f"p = {got['p']:.4f}"
                  f"{'  ЗНАЧИМ' if got['p'] < ALPHA else ''}")

    df = pd.DataFrame(rows)
    tag = f"{args.site}_area{int(100 * area_frac)}" if args.area else args.site
    df.to_csv(M / f"openlin_null_{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\nсохранено: {M / f'openlin_null_{tag}.csv'}")


if __name__ == "__main__":
    main()
