"""Что несёт каждый канал сам по себе: lift сырого канала против разметки.

ЗАЧЕМ. Модель может не показать прироста от нового канала по двум разным
причинам: канал пуст, либо канал полезен, но тонет среди девяти других. Разделить
их можно только измерив каналы поодиночке — как самостоятельные карты-кандидаты,
без всякого обучения.

Каждый канал берётся и как есть, и со знаком минус: у когерентности направление
связи заранее неизвестно (зона дробления может как удерживать фазу, так и терять
её), а lift чувствителен к знаку. Оба варианта идут в общий счёт тестов, чтобы
выбор знака постфактум не удваивал число попыток скрытно.

Значимость — тот же сдвиговый нуль, что в :mod:`experiments.openlin_null`.

Запуск из корня репозитория:
    python -m experiments.openlin_channels --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import assessment

M = Path("outputs/metrics")
N_SHIFTS = 2000
AREA = 0.10
TOL_M = 250.0
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="anabar")
    args = ap.parse_args()

    z = dict(np.load(M / f"openlin_{args.site}.npz"))
    dist = z["dist"]
    shape = dist.shape
    valid = np.isfinite(z["gray"])
    meta = SimpleNamespace(prf=shape[0], pic=shape[1])
    pool = np.flatnonzero(valid.ravel())
    target = np.flatnonzero(((dist <= TOL_M) & valid).ravel())

    skip = {"dist", "sigma", "res_m", "x0", "y_top", "epsg"}
    names = [k for k in z if k not in skip and z[k].shape == shape]
    n_tests = 2 * len(names)
    alpha = 0.05 / n_tests
    print(f"площадка «{args.site}»: каналов {len(names)}, тестов {n_tests} "
          f"(канал x знак), alpha = {alpha:.5f}; "
          f"цель {100 * len(target) / len(pool):.1f}% площади, "
          f"площадь сравнения {100 * AREA:.0f}%")
    print(f"минимально достижимое p = {1 / (1 + N_SHIFTS):.5f}\n")

    rows = []
    for name in names:
        for sign, label in ((1.0, "+"), (-1.0, "-")):
            a = sign * np.where(valid, z[name].astype(float), np.nan)
            got = assessment.spatial_null_pvalue(
                a, target, meta, area=AREA, pool=pool,
                n_shifts=N_SHIFTS, seed=SEED)
            rows.append({"канал": f"{label}{name}", **got,
                         "значим": got["p"] < alpha})
    df = pd.DataFrame(rows).sort_values("observed", ascending=False)
    for _, r in df.head(12).iterrows():
        print(f"  {r['канал']:<28} lift {r['observed']:.3f}  "
              f"нуль {r['null_mean']:.3f} (q95 {r['null_q95']:.3f})  "
              f"p = {r['p']:.4f}{'  ЗНАЧИМ' if r['значим'] else ''}")

    df.to_csv(M / f"openlin_channels_{args.site}.csv", index=False,
              encoding="utf-8-sig")
    print(f"\nсохранено: {M / f'openlin_channels_{args.site}.csv'}")


if __name__ == "__main__":
    main()
