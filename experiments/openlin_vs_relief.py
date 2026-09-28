"""Наши линеаменты против рельефных: единственная доступная внешняя сверка.

ЗАЧЕМ. Всё, чем проверялась наша карта, — внутренняя согласованность: линию
обязаны видеть оба прибора. Это доказывает, что линия не артефакт датчика, но не
говорит, насколько она совпадает с тем, что считает линеаментом независимый
метод. Ручной разметки геофизика в поставке нет, но в проекте есть СВОЙ
линеаментный слой, построенный совсем иначе — по рельефу
(:mod:`src.lineaments`: восьмиазимутная отмывка -> Кэнни -> Хаф). Он и служит
здесь внешней картой.

ЧТО СЧИТАЕТСЯ, В ОБЕ СТОРОНЫ:

* какая доля НАШИХ линий проходит рядом с рельефной линией — это согласие;
* какая доля РЕЛЬЕФНЫХ линий покрыта нашими — это полнота относительно них.

Обе величины сравниваются со случайным уровнем (тороидальный сдвиг нашей карты),
иначе на густой сети любая линия «совпадёт» с чем-нибудь.

ЧТО ЭТО НЕ ЗНАЧИТ. Высокое согласие не подтверждает структурную природу линий:
рельефный слой сам по себе — это в основном бровки долин (в его шапке измерено:
``lin_dist`` против ``dem_slope_std`` даёт Спирмен -0.80). Совпадение означает
«мы находим то же, что находят по рельефу», а не «мы находим разломы». Низкое
согласие тоже не приговор: наш метод видит радар и когерентность, которых у
рельефного нет.

СЕТКА. Рельефный слой лежит на сетке прогноза 500 м, наш растр 100 м в другой
зоне UTM; перенос — ``src.openlin_features`` (перепроецирование и усреднение).
Сравнение идёт на общей маске: валидные ячейки, покрытые нашим растром.

Запуск из корня репозитория:
    python -m experiments.openlin_vs_relief
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target, openlin_features  # noqa: E402

M = ROOT / "outputs" / "metrics"

#: Допуск совпадения, м. Ячейка сетки 500 м, поэтому меньше брать бессмысленно:
#: положение линии внутри ячейки всё равно неизвестно.
TOL_M = 500.0

#: Сколько сдвигов для случайного уровня.
N_SHIFTS = 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", default="openlin_union_anabar.npz",
                    help="файл с маской подтверждённых линий")
    a = ap.parse_args()

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    meta, _prognoz = criterial_target.load_prognoz_grid()
    lin_dist = df["lin_dist"].to_numpy("float64")

    z = dict(np.load(M / "openlin_anabar.npz"))
    ours = np.load(M / a.lines)["confirmed"]
    res_m = float(z["res_m"])

    # Наши линии на сетке прогноза: доля ячейки, занятая линией.
    dens = openlin_features.to_grid(ours.astype("float32"), z, meta, how="frac")
    cover = np.isfinite(dens) & np.isfinite(lin_dist)
    pool = valid & cover
    print(f"общая маска: {int(pool.sum()):,} ячеек "
          f"({100 * pool.sum() / valid.sum():.0f}% валидных)")

    ours_cells = pool & (dens > 0)
    relief_cells = pool & (lin_dist <= res_m)      # ячейка с рельефной линией
    print(f"наших линий касается {int(ours_cells.sum()):,} ячеек, "
          f"рельефных — {int(relief_cells.sum()):,}")

    # --- 1. согласие: доля наших линий рядом с рельефной --------------------
    base = float((lin_dist[pool] <= TOL_M).mean())
    obs = float(np.average((lin_dist[ours_cells] <= TOL_M),
                           weights=dens[ours_cells]))

    # Случайный уровень: сдвигаем НАШУ карту по растру и переносим заново.
    rng = np.random.default_rng(0)
    h, w = ours.shape
    null = []
    for _ in range(25):
        sh = (int(rng.integers(h)), int(rng.integers(w)))
        d2 = openlin_features.to_grid(
            np.roll(ours, sh, (0, 1)).astype("float32"), z, meta, how="frac")
        m2 = pool & np.isfinite(d2) & (d2 > 0)
        if m2.sum() < 100:
            continue
        null.append(float(np.average((lin_dist[m2] <= TOL_M),
                                     weights=d2[m2])))
    null = np.asarray(null)

    print(f"\n=== 1. согласие: наши линии рядом с рельефными " + "=" * 20)
    print(f"  доля длины наших линий в {TOL_M:.0f} м от рельефной: {obs:.3f}")
    print(f"  фон (доля площади в том же допуске):                {base:.3f}")
    print(f"  случайный сдвиг наших линий:                        "
          f"{null.mean():.3f} +- {null.std():.3f}")
    print(f"  во сколько раз выше случайного: x{obs / null.mean():.2f}")

    # --- 2. полнота относительно рельефного слоя ----------------------------
    from scipy.ndimage import distance_transform_edt

    grid = np.zeros(meta.shape, dtype=bool)
    grid.ravel()[np.flatnonzero(ours_cells)] = True
    dt = distance_transform_edt(~grid, sampling=(meta.dy, meta.dx))
    cov = float((dt.ravel()[np.flatnonzero(relief_cells)] <= TOL_M).mean())

    grid_all = np.zeros(meta.shape, dtype=bool)
    grid_all.ravel()[np.flatnonzero(pool)] = True
    share = float(ours_cells.sum() / pool.sum())
    print(f"\n=== 2. полнота относительно рельефной карты " + "=" * 23)
    print(f"  доля рельефных линий, у которых наша линия ближе "
          f"{TOL_M:.0f} м: {cov:.3f}")
    print(f"  при том что наши линии занимают {100 * share:.1f}% ячеек пула")

    pd.DataFrame([{"согласие": obs, "фон": base, "случайно": float(null.mean()),
                   "во_сколько_раз": obs / null.mean(),
                   "полнота_по_рельефу": cov,
                   "доля_ячеек_наших": share}]).to_csv(
        M / "openlin_vs_relief.csv", index=False, encoding="utf-8-sig")
    print(f"\nсохранено: {M / 'openlin_vs_relief.csv'}")


if __name__ == "__main__":
    main()
