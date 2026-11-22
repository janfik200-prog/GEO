"""Г12 на юго-западе: тяготеют ли шлихи с золотом к ильинской свите.

Гипотеза Г12 (записана 04.11.2026 после просмотра данных основного листа,
поэтому подтверждать её надо на независимой площади): ильинская свита
богаче золотом, чем остальные свиты списка ВНИГНИ. Независимая площадь —
юго-запад, листы R-48-XV,XVI и R-48-XXI,XXII: 12 шлиховых проб с золотом
(`data/processed/sw_shlikh_points.csv`, ±0.5 км) и геология сборки смежных
листов (`r48_geol_gk.shp`). Шлих — не рудопроявление, а признак коренного
источника выше по течению (Ог6: россыпи модель не объясняет; здесь шлихи —
только проверка).

Записано до расчёта. Половина шлихов лежит на аллювии, поэтому
сравнивается не порода в точке, а доля ильинской свиты в круге `R_M`
вокруг пробы (замена водосбора: рельефа на юго-западе в проекте нет).
Контроль — случайные точки на голоценовом аллювии той же площади (шлихи
отбирают в долинах). Г12 подтверждается, если средняя доля ильинской
вокруг шлихов выше, чем у контроля, с p < 0.05 (перестановочный тест, 10 000
выборок того же размера). Иначе — «не подтверждена этой выборкой»; при 12
пробах обнаружимо только сильное различие, поэтому дополнительно печатается,
во сколько раз должна отличаться доля, чтобы её поймать.

Запуск::

    python -X utf8 -m experiments.gmo_h12_sw_shlikh
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GEOL = ROOT / "data/SBORKA_NESKOLKO_LISTOV/СБОРКА_НА_НЕСКОЛЬКО_ЛИСТОВ/Сцены 2D/Геология/r48_geol_gk.shp"
SHLIKH = ROOT / "data/processed/sw_shlikh_points.csv"
FRAME = (104.95, 69.35, 106.0, 70.7)          # юго-западная рамка, градусы
R_M, N_CTRL, N_PERM = 5000.0, 2000, 10000
rng = np.random.default_rng(12)
SVITY = {"ильинская": "<il>", "бурдурская": "<br>", "лабастахская": "<lb>",
         "усть-ильинская": "<ul>", "котуйканская": "<kt>"}


def shares(pts: gpd.GeoSeries, geol: gpd.GeoDataFrame) -> pd.DataFrame:
    sidx = geol.sindex
    rows = []
    for p in pts:
        c = p.buffer(R_M)
        hit = geol.iloc[list(sidx.query(c, predicate="intersects"))]
        inter = hit.geometry.intersection(c).area
        rows.append({n: float(inter[hit.INDEX.str.contains(k, regex=False)].sum() / c.area)
                     for n, k in SVITY.items()})
    return pd.DataFrame(rows)


def main() -> None:
    crs = anabar_inset.SHEET_CRS
    geol = gpd.read_file(GEOL).set_crs(crs, allow_override=True)
    frame = gpd.GeoSeries.from_xy([FRAME[0], FRAME[2]], [FRAME[1], FRAME[3]], crs=4326).to_crs(crs)
    (x0, x1), (y0, y1) = sorted(frame.x), sorted(frame.y)
    geol = geol.cx[x0 - R_M:x1 + R_M, y0 - R_M:y1 + R_M]
    p = pd.read_csv(SHLIKH, sep=";")
    pts = gpd.GeoSeries.from_xy(p.долгота, p.широта, crs=4326).to_crs(crs)
    inside = geol.union_all().contains(pts)
    pts = pts[inside.to_numpy()]
    alluv = geol[geol.INDEX.str.contains("Qh", regex=False)
                 & ~geol.INDEX.str.contains("lb", regex=False)].union_all()
    cand = gpd.GeoSeries.from_xy(rng.uniform(x0, x1, 40 * N_CTRL), rng.uniform(y0, y1, 40 * N_CTRL),
                                 crs=crs)
    ctrl = cand[cand.within(alluv).to_numpy()][:N_CTRL]
    s_pts, s_ctrl = shares(pts, geol), shares(ctrl, geol)
    print(f"Шлихов в контуре геологии: {len(pts)} из {len(p)}; контрольных точек на аллювии: "
          f"{len(ctrl)}; круг {R_M / 1000:.0f} км")
    print("Средняя доля свиты в круге, %:")
    t = pd.DataFrame({"у шлихов": (100 * s_pts.mean()).round(1),
                      "у контроля": (100 * s_ctrl.mean()).round(1)})
    print(t.to_string())
    obs = s_pts.ильинская.mean()
    perm = np.array([s_ctrl.ильинская.sample(len(pts), random_state=int(i)).mean()
                     for i in range(N_PERM)])
    pval = float((perm >= obs).mean())
    q95 = float(np.quantile(perm, 0.95))
    base = s_ctrl.ильинская.mean()
    print(f"\nГ12: доля ильинской у шлихов {100 * obs:.1f}% против {100 * base:.1f}% у контроля; "
          f"p = {pval:.3f}. Поймать можно только превышение в {q95 / base:.1f} раза и больше "
          f"(95-й процентиль контроля {100 * q95:.1f}%).")
    print("Вердикт:", "подтверждается" if pval < 0.05 else "не подтверждена этой выборкой")


if __name__ == "__main__":
    main()
