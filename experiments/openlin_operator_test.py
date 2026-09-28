"""Что оператор границы делает с заведомо известной картинкой.

ПОВОД. На реальном поле силы видны широкие «перья» длиной в километры,
идущие наискось через формы местности. Похоже, что оператор откликается не на
границу между участками, а на само присутствие контрастной текстуры в окне.
Здесь это проверяется на синтетике, где правильный ответ известен заранее.

ЧЕТЫРЕ СЛУЧАЯ:

* ``линия``   — ровный фон и одна прямая линия. Правильный ответ: узкая полоса
  отклика вдоль линии шириной порядка окна поперёк (200 м), и больше ничего.
* ``ступень`` — два ровных блока с разной яркостью. Правильный ответ: отклик
  вдоль их границы.
* ``шум``     — только коррелированная текстура, линий нет вовсе. Правильный
  ответ: ровное слабое поле. Если появятся «перья» — оператор их придумывает.
* ``шум+линия`` — то же плюс линия: видно ли её на фоне собственных артефактов.

Считается ровно тот же оператор, что в методе (``edges_from_components``), без
единой правки.

Запуск из корня репозитория:
    python -m experiments.openlin_operator_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.openlin_domains import (OFFSET_M, WIN_LONG_M, WIN_SHORT_M,
                                         edges_from_components)
from experiments.openlin_slides2 import EDGE, INK, INK2, RAMP, SLATE, SURF

OUT = Path("outputs/steps")

N = 300          # ячеек в стороне
RES_M = 100.0    # тот же шаг, что у площадки
SEED = 42

#: Масштаб корреляции синтетической текстуры, м. 500 м — как самый мелкий
#: масштаб окрестности в методе: текстура должна быть похожа на настоящую.
NOISE_M = 500.0


def _line(shape: tuple[int, int], angle_deg: float, width: int = 1
          ) -> np.ndarray:
    """Прямая линия через центр растра под заданным углом."""
    yy, xx = np.indices(shape).astype("float64")
    cy, cx = shape[0] / 2, shape[1] / 2
    th = np.radians(angle_deg)
    # расстояние до прямой, проходящей через центр под углом th
    d = np.abs((xx - cx) * np.cos(th) - (yy - cy) * np.sin(th))
    return (d <= width / 2).astype("float32")


def _noise(shape: tuple[int, int], rng) -> np.ndarray:
    """Коррелированная текстура: белый шум, сглаженный до масштаба NOISE_M."""
    from scipy.ndimage import gaussian_filter

    a = gaussian_filter(rng.standard_normal(shape), NOISE_M / RES_M)
    return (a / a.std()).astype("float32")


def _arc(shape: tuple[int, int], radius_px: float = 260.0,
         width: int = 1) -> np.ndarray:
    """Дуга: линия с плавной кривизной, какой бывает настоящий линеамент.

    Радиус 260 ячеек = 26 км: на протяжении окна 2 км такая дуга отклоняется
    от хорды примерно на 100 м, то есть ровно на ширину центральной полосы.
    Прямое окно длиной 2 км такую линию уже теряет.
    """
    yy, xx = np.indices(shape).astype("float64")
    cy, cx = shape[0] * 1.35, shape[1] * 0.1      # центр дуги вне кадра
    d = np.abs(np.hypot(yy - cy, xx - cx) - radius_px)
    return (d <= width / 2).astype("float32")


def cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    shape = (N, N)
    line = _line(shape, 30.0, width=1)
    arc = _arc(shape)
    step = np.zeros(shape, dtype="float32")
    step[:, N // 2:] = 1.0
    noise = _noise(shape, rng)
    return {
        "линия": line,
        "ступень": step,
        "шум": noise,
        "шум + линия": noise + 2.0 * line,
        "шум + дуга": noise + 2.0 * arc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="operator_test.png")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = cases()
    valid = np.ones((N, N), dtype=bool)
    fig = plt.figure(figsize=(4.6 * len(data), 9.4), facecolor=SURF)
    gs = fig.add_gridspec(2, len(data), hspace=0.14, wspace=0.06,
                          left=0.015, right=0.985, top=0.92, bottom=0.06)

    print(f"окно {WIN_LONG_M:.0f} x {WIN_SHORT_M:.0f} м, сдвиг "
          f"{OFFSET_M:.0f} м, растр {N}x{N} по {RES_M:.0f} м\n")
    for i, (name, img) in enumerate(data.items()):
        comp = img[None].astype("float32")
        strength, _ang = edges_from_components(comp, valid, RES_M)
        s = np.nan_to_num(strength, nan=0.0)

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=17, color=INK, fontweight="bold", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(EDGE)
        if i == 0:
            ax.set_xlabel("вход", fontsize=13, color=INK2)

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(s, cmap=RAMP)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(SLATE)
        ax.set_xlabel("поле силы границы", fontsize=13, color=INK2)

        # --- числа: насколько отклик привязан к самой линии ---------------
        mid = s[N // 2 - 2:N // 2 + 3].mean()
        far = s[:40].mean()
        prof = s[N // 2]
        wide = int((prof >= 0.5 * prof.max()).sum())
        print(f"{name:12s} max {s.max():6.2f}  на линии {mid:5.2f}  "
              f"вдали {far:5.2f}  отношение {mid / max(far, 1e-9):5.2f}  "
              f"ширина отклика по центру {wide * RES_M:5.0f} м")

    fig.savefig(OUT / a.out, dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"\n{OUT / a.out}")


if __name__ == "__main__":
    main()
