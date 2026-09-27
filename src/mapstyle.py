"""Оформление карт перспективности: резкие классы вместо плавной заливки.

ЗАЧЕМ. Карта строится по сглаженному баллу (ядро рабочей точки — 5 x 1.25 км
вдоль северо-запада), поэтому непрерывная заливка процентилей выглядит размытой:
соседние ячейки отличаются на доли процентиля и глаз не видит границ зон. Здесь
собрано оформление, которое ничего не меняет в самих числах, но делает картинку
читаемой:

* `class_norm` — деление процентиля на 5 классов перспективности с резкими
  границами (50 / 75 / 90 / 97), как принято на прогнозных картах;
* рисовать растр с `interpolation="nearest"` — matplotlib по умолчанию
  сглаживает изображение при масштабировании, добавляя размытость поверх той,
  что уже есть в данных;
* `outline` — граница верхних 10% отдельной линией, чтобы сдаваемый контур был
  виден точно, а не угадывался по цвету.

`save_fig` повторяет запись файла: на Windows PNG иногда занят просмотрщиком или
синхронизацией, и первая попытка падает с OSError 22.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Границы классов в процентилях по территории. Верхний класс — сдаваемые
# верхние 3%, следующий добирает до верхних 10%.
CLASS_EDGES = (0.0, 50.0, 75.0, 90.0, 97.0, 100.0)

CLASS_COLORS = ("#f2f0f7", "#c7c5e0", "#f2b544", "#e2662c", "#a3122b")

CLASS_LABELS = ("фон, нижние 50%", "50-75%", "75-90%",
                "90-97%", "верхние 3%")


def class_cmap() -> ListedColormap:
    return ListedColormap(CLASS_COLORS)


def class_norm() -> BoundaryNorm:
    return BoundaryNorm(CLASS_EDGES, len(CLASS_COLORS))


def percentile(v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Балл в процентилях по территории; вне маски — пропуск."""
    from scipy import stats as sps

    out = np.full(v.size, np.nan)
    m = np.ones(v.size, dtype=bool) if mask is None else mask
    out[m] = sps.rankdata(v[m]) / max(int(m.sum()), 1) * 100.0
    return out


def outline(ax, mask: np.ndarray, shape, ext, flip: bool = False,
            color: str = "#101010", lw: float = 0.9) -> None:
    """Граница области `mask` (например верхних 10%) одной линией."""
    img = mask.reshape(shape).astype(float)
    if flip:
        img = np.flipud(img)
    ax.contour(np.linspace(ext[0], ext[1], shape[1]),
               np.linspace(ext[2], ext[3], shape[0]), img,
               levels=[0.5], colors=color, linewidths=lw)


def save_fig(fig, path: Path, dpi: int = 200, tries: int = 4) -> None:
    """Сохранение с повтором: файл может быть временно занят."""
    for k in range(tries):
        try:
            fig.savefig(path, dpi=dpi)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(1.5)
