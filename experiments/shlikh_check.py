# -*- coding: utf-8 -*-
"""Проверка карт по шлиховым пробам золота юго-западного квадранта.

Единственная внешняя заверка на площади за рамкой основного листа: своих
точек опробования там нет. 12 проб сняты с карт полезных ископаемых ГГК-200
Анабарской серии (`outputs/12_внешние_данные/шлиховые_пробы.csv`, точность
привязки 0.3-0.5 км). Шлиховой ореол — поисковый признак, указывающий на
коренной источник выше по потоку, а НЕ рудопроявление, поэтому пробы годятся
только как слабая проверка и никогда как цель обучения.

Мера — «процент сверху»: положение пробы в ранге карты по всей области
расчёта. Чем меньше, тем ближе проба к перспективным участкам; у случайной
карты медиана 50%.

Сравниваются: наш прогноз (`forecast_arch_ядро5км.parquet`, рабочая точка на
архивной сетке) и воссоздание критериальной формулы на той же области
(`criterial_reconstruct_multisheet.parquet`). Расчёт организации `by_crt`
считается отдельно: его сетка накрывает лишь часть проб. Поэтому в логе две
сводки — по всем пробам, попавшим в сетку каждой карты, и по общему
подмножеству проб, попавших в сетку всех трёх карт сразу (там карты
сравниваются на одних и тех же точках).

Выход: `outputs/metrics/shlikh_check.log` и `.csv`.
Запуск из корня: ``python -X utf8 -m experiments.shlikh_check``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXT = ROOT / "outputs" / "12_внешние_данные"
MET = ROOT / "outputs" / "metrics"

# Проекция проекта: Гаусса-Крюгера, зона 18, система 1942 г.
CRS = ("+proj=tmerc +lat_0=0 +lon_0=105 +x_0=500000 +y_0=0 +k_0=1 "
       "+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +ellps=krass "
       "+units=m +no_defs")
MAX_DIST = 1500.0          # ячейка 500 м: дальше 1.5 км привязки уже нет


def ranks_from_top(frame, column, xs, ys):
    """«Процент сверху» каждой пробы; NaN, если пробы нет в сетке этой карты."""
    grid = frame.dropna(subset=[column])
    tree = cKDTree(np.c_[grid["x"], grid["y"]])
    dist, idx = tree.query(np.c_[xs, ys])
    hit = dist < MAX_DIST
    rank = pd.Series(grid[column].to_numpy()).rank(pct=True).to_numpy()
    out = np.full(len(xs), np.nan)
    out[hit] = (1.0 - rank[idx[hit]]) * 100.0
    return out


def main():
    shlikh = pd.read_csv(EXT / "шлиховые_пробы.csv", sep=";",
                         encoding="utf-8-sig")
    to_grid = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    xs, ys = to_grid.transform(shlikh["долгота"].to_numpy(),
                               shlikh["широта"].to_numpy())

    ours = pd.read_parquet(PROC / "forecast_arch_ядро5км.parquet")
    crit = pd.read_parquet(PROC / "criterial_reconstruct_multisheet.parquet")

    maps = [("наш прогноз", ranks_from_top(
                ours, "число: сам балл (опора)", xs, ys)),
            ("критериальное воссоздание", ranks_from_top(
                crit, "воссоздание_вся_область", xs, ys)),
            ("критериальный расчёт организации (by_crt)", ranks_from_top(
                crit, "by_crt", xs, ys))]
    # сетка расчёта организации накрывает лишь часть проб; чтобы карты
    # сравнивались на одних и тех же пробах, считается ещё и общее
    # подмножество — пробы, попавшие во все три сетки сразу
    common = np.all([~np.isnan(v) for _, v in maps], axis=0)
    rows = [(name, float(np.nanmedian(v)), int(np.sum(~np.isnan(v))),
             float(np.median(v[common]))) for name, v in maps]
    table = pd.DataFrame(rows, columns=["карта", "медиана_процент_сверху",
                                        "проб_в_сетке",
                                        "медиана_на_общих_пробах"])
    table["всего_проб"] = len(shlikh)
    table["общих_проб"] = int(common.sum())

    MET.mkdir(parents=True, exist_ok=True)
    table.to_csv(MET / "shlikh_check.csv", index=False, encoding="utf-8")
    lines = [f"шлиховых проб с золотом: {len(shlikh)}",
             "мера: медианный «процент сверху», случайный уровень 50%",
             "(меньше — проба ближе к перспективным участкам карты)", ""]
    lines += [f"{r.карта:45s} {r.медиана_процент_сверху:5.0f}%   "
              f"проб в сетке {r.проб_в_сетке}" for r in table.itertuples()]
    lines += ["", f"на общих {int(common.sum())} пробах (попали в сетку всех "
                  f"трёх карт):"]
    lines += [f"{r.карта:45s} {r.медиана_на_общих_пробах:5.0f}%"
              for r in table.itertuples()]
    text = "\n".join(lines)
    (MET / "shlikh_check.log").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
