"""Коры выветривания листа по положению: какие могут быть дорифейскими (О10).

Цель. Возраста контуров кор (`kory`) в открытой литературе нет, поэтому
древние коры (О10, этап II, «источник материала» и место накопления) от
молодых (О16д, этап IV) по возрасту не отделить. Черновик этапа IV (раздел
6.2) предложил отделить их по положению. Здесь это предложение проверено.

Правило, записанное до расчёта. Кора развита по породе, на которой лежит, и
моложе её:
* кора на рифее или кембрии моложе этих пород — **послерифейская (О16д)**;
* кора на фундаменте не дальше `NEAR_M` от контакта фундамента с ильинской
  свитой может быть дорифейской: её недавно освободил размыв рифея, а
  дорифейская кора лежит как раз под подошвой ильинской — **возможно
  дорифейская (О10)**;
* кора на фундаменте дальше от контакта — **не различить**: это может быть
  и дорифейский остаток, и молодая кора.
Порода под корой — та, что занимает большую часть длины линии кор (полигоны
`svita_labeled.gpkg`); четвертичные отложения, долериты и диабазы не в
счёт. Линии `kory` — открытые знаки длиной 0.25–7.7 км, а не контуры.

Входные данные: `data/Gis-integro/shp_dbf/kory.shp` (19 линий, слой формулы
⚠ — здесь он только классифицируется, в признаки не входит),
`data/processed/svita_labeled.gpkg`.

Результат: таблица в консоли и `data/processed/kory_position.csv`, рисунок
`outputs/13_генетическая_модель/gmo_kory_position.png`.

Запуск::

    python -X utf8 -m experiments.gmo_kory_position
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, ggk200_base, mapstyle, outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KORY = ROOT / "data/Gis-integro/shp_dbf/kory.shp"
SVITY = ROOT / "data/processed/svita_labeled.gpkg"
OUT = ROOT / "data/processed/kory_position.csv"
NEAR_M = 2000.0
BASE = "архей и нижний протерозой"
RIPHEAN = {"ильинская", "бурдурская", "лабастахская", "усть-ильинская", "котуйканская",
           "юсмастахская", "кембрий (немакит-далдынская, медвежинская)"}
CLASS_COLOR = {"послерифейская (О16д)": "#c0392b", "возможно дорифейская (О10)": "#1f77b4",
               "не различить": "#7f7f7f"}


def main() -> None:
    crs = anabar_inset.SHEET_CRS
    k = gpd.read_file(KORY).set_crs(crs, allow_override=True)
    sv = gpd.read_file(SVITY).to_crs(crs)
    sv = sv[sv.svita.isin(RIPHEAN | {BASE})]
    base = sv[sv.svita == BASE].union_all()
    il = sv[sv.svita == "ильинская"].union_all()
    contact = base.boundary.intersection(il.buffer(50))       # контакт фундамента с ильинской
    rows = []
    for i, line in enumerate(k.geometry):
        share = {}
        for name, geom in sv.dissolve("svita").geometry.items():
            ln = line.intersection(geom).length
            if ln > 0:
                share[name] = ln / line.length
        under = max(share, key=share.get) if share else "чехол или интрузии"
        d_contact = line.distance(contact)
        if under in RIPHEAN:
            cls = "послерифейская (О16д)"
        elif under == BASE and d_contact <= NEAR_M:
            cls = "возможно дорифейская (О10)"
        else:
            cls = "не различить"
        rows.append({"линия": i, "длина, м": round(line.length), "порода под корой": under,
                     "доля длины, %": round(100 * share.get(under, 0)),
                     "до контакта фундамента с ильинской, м": round(d_contact),
                     "класс": cls})
    t = pd.DataFrame(rows)
    t.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(t.to_string(index=False))
    s = t.groupby("класс")["длина, м"].agg(["count", "sum"])
    s["доля длины, %"] = (100 * s["sum"] / s["sum"].sum()).round(0)
    print("\nИтог по классам:\n" + s.to_string())

    fig, ax = plt.subplots(figsize=(11, 10))
    ggk200_base.draw(ax, decimate=2, alpha=0.45, sheets=("R-48-XI,XII",))
    to_ll = lambda g: gpd.GeoSeries(g, crs=crs).to_crs(4326)  # noqa: E731
    to_ll(contact).plot(ax=ax, color="k", linewidth=1.2, zorder=4)
    for cls, color in CLASS_COLOR.items():
        sel = k.geometry[t["класс"] == cls]
        if len(sel):
            gpd.GeoSeries(sel, crs=crs).to_crs(4326).plot(ax=ax, color=color, linewidth=3.5,
                                                          zorder=5)
            gpd.GeoSeries(sel.buffer(1500), crs=crs).to_crs(4326).boundary.plot(
                ax=ax, color=color, linewidth=1, zorder=5)
    ax.set_xlabel("долгота, ° в.д.")
    ax.set_ylabel("широта, ° с.ш.")
    handles = [Line2D([], [], color=c, linewidth=3.5, label=f"кора: {n}")
               for n, c in CLASS_COLOR.items()]
    handles += [Line2D([], [], color="k", linewidth=1.2,
                       label="контакт фундамента с ильинской свитой"),
                Patch(facecolor="none", edgecolor="0.4", label="кольцо 1.5 км — чтобы видеть короткие линии")]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_title("Лист R-48-XI,XII: коры выветривания по положению — какие могут быть "
                 "дорифейскими\n(кора моложе породы, на которой лежит; возраст по "
                 "литературе не известен)", fontsize=11)
    anabar_inset.draw(ax.inset_axes([0.72, 0.72, 0.27, 0.27]), fontsize=7, title=False,
                      legend=False)
    out = outputs.png("gmo_kory_position.png")
    mapstyle.save_fig(fig, out, dpi=130)
    print(f"\nРисунок: {out}")


if __name__ == "__main__":
    main()
