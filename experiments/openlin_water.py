"""Сводится ли текстурный сигнал коридоров к воде?

ЗАЧЕМ. Шаг 0 на Abitibi (:mod:`experiments.openlin_probe`) дал контраст РАЗБРОСА
-0.47...-0.52 sigma при контрасте медиан всего +0.05...+0.10: внутри коридора
разломов яркость ОДНОРОДНЕЕ, чем в кольце вокруг. На щите у такой однородности
есть очевидный кандидат — линейные озёра в разломных котловинах: вода тёмная и
гладкая, её IQR близок к нулю.

Различить надо две вещи, потому что они ведут к разным решениям:

* сигнал сводится к воде -> обучать сеть на нём бессмысленно, она выучит
  «озеро = разлом». Это тривиально (озёра и так есть в топооснове) и на Анабар
  не переносится;
* после исключения воды заметная часть сигнала остаётся -> это текстурная
  подпись самой зоны дробления, и вот на ней обучение осмысленно.

КАК. Тот же :func:`src.detection_limit.corridor_contrast`, но пул ячеек
сокращается: вода выбрасывается по порогу NDVI. Порог 0.0 — не подобран, а взят
по физике: у воды NIR поглощается почти нацело, поэтому NDVI воды отрицателен, у
любой растительности и открытого грунта — положителен.

ЭТО ОПИСАНИЕ, А НЕ ПРОВЕРКА ГИПОТЕЗЫ: ни p-value, ни alpha, знаменатель
Бонферрони структурной ветки не растёт.

Запуск из корня репозитория:
    python -m experiments.openlin_water --site abitibi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import detection_limit as dl

#: Порог маски воды по NDVI (см. докстринг: физика, а не подбор).
WATER_NDVI = 0.0
#: Полуширины коридора — те же, что в openlin_probe, ради сравнимости строк.
CORRIDORS_M = (0.0, 250.0, 500.0, 1000.0)


def load(site: str) -> dict:
    path = Path("outputs/metrics") / f"openlin_{site}.npz"
    if not path.exists():
        raise SystemExit(f"нет {path} — сначала прогоните openlin_probe --site {site}")
    return dict(np.load(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="abitibi")
    args = ap.parse_args()

    z = load(args.site)
    gray, ndvi, dist = z["gray"], z["ndvi"], z["dist"]
    sigma = float(z["sigma"])
    water = np.isfinite(ndvi) & (ndvi < WATER_NDVI)

    print(f"площадка «{args.site}», sigma = {sigma:.4g}")
    print(f"вода (NDVI < {WATER_NDVI}): {100 * water.mean():.2f}% площадки\n")

    # Насколько вода тяготеет к коридорам — это и есть проверка озёрной гипотезы
    # в самой прямой форме: если разломы трассируются озёрами, доля воды внутри
    # коридора обязана быть выше, чем в кольце.
    ring = (dist > 2000.0) & (dist <= 6000.0)
    print("доля воды:")
    for c in CORRIDORS_M:
        inside = dist <= c
        print(f"  коридор {c:6.0f} м: внутри {100 * water[inside].mean():5.2f}%  "
              f"против {100 * water[ring].mean():5.2f}% в кольце")

    rows = []
    for c in CORRIDORS_M:
        full = dl.corridor_contrast(gray.ravel(), dist.ravel(), sigma, corridor_m=c)
        dry_gray = np.where(water, np.nan, gray)
        dry = dl.corridor_contrast(dry_gray.ravel(), dist.ravel(), sigma, corridor_m=c)
        rows.append({
            "площадка": args.site, "полуширина_м": c,
            "медиан_всё": full["contrast_median"], "медиан_без_воды": dry["contrast_median"],
            "разброс_всё": full["contrast_spread"], "разброс_без_воды": dry["contrast_spread"],
            "ячеек_внутри_без_воды": dry["n_inside"], "ячеек_кольцо_без_воды": dry["n_ring"],
        })

    df = pd.DataFrame(rows)
    print("\nконтраст, sigma (всё / без воды):")
    print("  полуширина |    медиан    |   разброс")
    for r in rows:
        print(f"  {r['полуширина_м']:9.0f} | {r['медиан_всё']:+.3f} / "
              f"{r['медиан_без_воды']:+.3f} | {r['разброс_всё']:+.3f} / "
              f"{r['разброс_без_воды']:+.3f}")

    keep = (rows[1]["разброс_без_воды"] / rows[1]["разброс_всё"]
            if rows[1]["разброс_всё"] else float("nan"))
    print(f"\nдоля текстурного сигнала, пережившая исключение воды "
          f"(коридор 250 м): {100 * keep:.0f}%")

    out = Path("outputs/metrics") / f"openlin_water_{args.site}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
