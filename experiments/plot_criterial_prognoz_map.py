"""Карта прогноза критериального анализа (ГИС Интегро, ``prognoz.pgrid``)
для слайда «Воссоздание критериального метода».

Показывает нативный результат формулы (меньше = перспективнее) с контурами
трёх выделенных рудных объектов (:func:`src.criterial_target.label_ore_objects`)
для ориентира. Не сама реконструкция (та живёт в отдельном репозитории), а её
исходные данные — единственное, что физически доступно в этом проекте.

Запуск из корня: ``python -m experiments.plot_criterial_prognoz_map``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, criterial_target  # noqa: E402

OUT = config.PROJECT_ROOT / "outputs" / "criterial_prognoz_map.png"


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    img = prognoz.reshape(shape)
    objects = criterial_target.label_ore_objects(prognoz).reshape(shape)

    fig, ax = plt.subplots(figsize=(9, 8.2))
    im = ax.imshow(img, cmap="viridis_r")
    ax.contour(objects > 0, levels=[0.5], colors="#C8391B", linewidths=1.6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("prognoz (меньше = перспективнее)")
    ax.set_xlabel("столбец сетки (500 м)")
    ax.set_ylabel("строка сетки (500 м)")
    ax.set_title("Критериальный прогноз ГИС Интегро (prognoz.pgrid)\nконтур — 3 выделенных рудных объекта")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("сохранено:", OUT)


if __name__ == "__main__":
    main()
