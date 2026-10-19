# -*- coding: utf-8 -*-
"""Пара карт для слайда «Где мы сейчас» колоды «ИИ во ВНИГНИ».

Слева — критериальный прогноз ВНИГНИ, справа — итоговая карта листа
(обучение на всём листе, 37 переносимых дистанционных признаков, порог верхних
10%, ядро 5 x 1.25 км). Раскраска и классы те же, что в
`experiments/full_sheet_map.py` (`src/mapstyle.py`: пять классов по процентилю
50/75/90/97, граница верхних 10% линией, точки опробования поверх), но картинка
рисуется под слайд: размер фигуры равен месту на слайде (7.7 x 5.0 дюйма),
поэтому кегль в пунктах совпадает с кеглем на экране (деления и легенда 11 пт).

Входы:
* `data/processed/full_sheet_map_ядро5км.parquet` — балл итоговой карты,
  маска валидных ячеек (пишет `experiments/full_sheet_map.py`);
* сетка критериального прогноза — `criterial_target.load_prognoz_grid`;
* точки опробования — `assessment.load_verification_points`.

Выход: `outputs/11_презентации/ai_vnigni_map_pair.png`.
Запуск из корня: ``python -X utf8 -m experiments.plot_ai_vnigni_maps``.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, config, crit_reference, criterial_target, mapstyle  # noqa: E402

OUT = ROOT / "outputs" / "11_презентации" / "ai_vnigni_map_pair.png"
FIGSIZE = (7.7, 5.0)   # дюймы = место на слайде
FS = 11                # деления, подписи осей, легенда
FS_TITLE = 12.5


def main() -> Path:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    crit = crit_reference.criterial_score(prognoz)

    m = pd.read_parquet(config.PROCESSED_DIR / "full_sheet_map_ядро5км.parquet")
    m = m.sort_values(["row", "col"])
    valid = m["валидная"].to_numpy(bool)
    score = m["score"].to_numpy(float)
    top_f = m["верхние_10"].to_numpy(bool)
    pool = np.flatnonzero(valid)
    top_c = crit_reference.reference_mask(crit, pool, config.CREF_AREA) & valid

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    ext = [0, (cx.max() - cx.min()) / 1000.0, 0, (cy.max() - cy.min()) / 1000.0]
    px, py = (cx - cx.min()) / 1000.0, (cy - cy.min()) / 1000.0

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    plt.rcParams.update({"font.family": "Calibri", "font.size": FS})
    fig = plt.figure(figsize=FIGSIZE)
    # две карты сверху, общая шкала классов и легенда точек снизу
    ax_l = fig.add_axes([0.085, 0.36, 0.40, 0.57])
    ax_r = fig.add_axes([0.575, 0.36, 0.40, 0.57])
    cax = fig.add_axes([0.085, 0.175, 0.89, 0.035])

    for ax, v, top, title in ((ax_l, crit, top_c, "Критериальный прогноз ВНИГНИ"),
                              (ax_r, score, top_f, "Итоговая карта листа")):
        h = ax.imshow(mapstyle.percentile(v, valid).reshape(shape), extent=ext,
                      cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                      interpolation="nearest")
        mapstyle.outline(ax, top, shape, ext, flip=True)
        ax.scatter(px[allp], py[allp], s=18, facecolors="none",
                   edgecolors="#101010", linewidths=0.8, zorder=3)
        ax.scatter(px[unb], py[unb], s=38, marker="^", facecolors="#22d3a7",
                   edgecolors="#0b3b32", linewidths=0.7, zorder=4)
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=5)
        ax.set_xlabel("восток, км", fontsize=FS)
        ax.set_ylabel("север, км", fontsize=FS, labelpad=2)
        ax.tick_params(labelsize=FS)

    cb = fig.colorbar(h, cax=cax, orientation="horizontal",
                      ticks=mapstyle.CLASS_EDGES)
    cb.ax.set_xticklabels(["0", "50", "75", "90", "97", "100"])
    cb.ax.tick_params(labelsize=FS)
    cb.set_label("класс перспективности, процентиль по листу", fontsize=FS,
                 labelpad=2)

    fig.legend(handles=[
        Line2D([], [], color="#101010", lw=1.2, label="граница верхних 10%"),
        Line2D([], [], marker="o", color="none", markeredgecolor="#101010",
               markersize=6, label="точки опробования, %d" % allp.size),
        Line2D([], [], marker="^", color="none", markerfacecolor="#22d3a7",
               markeredgecolor="#0b3b32", markersize=7.5,
               label="из них независимые от формулы, %d" % unb.size)],
        loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3, fontsize=FS,
        frameon=False, handletextpad=0.3, columnspacing=1.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mapstyle.save_fig(fig, OUT, dpi=250)
    plt.close(fig)
    print("OK: %s; точек %d, независимых %d" % (OUT, allp.size, unb.size))
    return OUT


if __name__ == "__main__":
    main()
