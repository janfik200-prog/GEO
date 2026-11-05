"""Линии по рельефу для тракта снимка: уклон и отклик «одна линия на склон».

ОТКУДА. Повтор прогона DEM1 (21–23.10.2026, другой компьютер; код сюда не
вернулся — пишется по описанию из отчёта Артура от 25.10.2026, раздел 5.3):

* вход детектора — уклон по GLO-30, выбран пробой по наименьшему мусору
  (0.6 % у двойника против 13.2 % у сырого рельефа);
* отклик scarp: гребни поля уклона, где наибольшая по модулю кривизна
  ОТРИЦАТЕЛЬНА — одна линия на склон по самому крутому месту. Обычная
  гессианова линейность (:func:`openlin_iso.ridge_response`) берёт модуль и
  поэтому даёт на склоне три линии: бровку, подошву и середину; здесь
  остаётся одна;
* подтверждение — уклон ALOS AW3D30 (независимое измерение: оптическая
  стереосъёмка против радара TanDEM-X у GLO-30).

Тракт (трассировка, продолжение, сшивка, длина) — тот же, что у снимка; здесь
только вход и отклик. Ключ ``--рельеф`` в :mod:`openlin_zone_lines_v3`.
"""
from __future__ import annotations

import numpy as np

from cache_paths import ANABAR_DEM
from experiments.openlin_iso import SIGMAS


def slope(dem: np.ndarray, res_m: float) -> np.ndarray:
    """Уклон, м/м; пустые ячейки модели остаются пустыми."""
    z = np.where(np.isfinite(dem), dem, np.nanmedian(dem)).astype("float64")
    gy, gx = np.gradient(z, res_m)
    s = np.hypot(gx, gy).astype("float32")
    return np.where(np.isfinite(dem), s, np.nan)


def load_slopes(site: str, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Уклон GLO-30 (строит) и ALOS (подтверждает) на сетке листа.

    Точечные провалы ALOS (расхождение с GLO-30 больше ``ALOS_BAD_M``)
    выкидываются: сверка 05.11.2026 — медиана расхождения около 1 м, больше
    30 м у 0.03 % ячеек, и это одиночные ямы, а не рельеф.
    """
    g = np.load(ANABAR_DEM / f"dem_glo30_{site}_лист_{int(res_m)}m.npy")
    a = np.load(ANABAR_DEM / f"dem_alos_{site}_лист_{int(res_m)}m.npy")
    a = np.where(np.abs(a - g) > ALOS_BAD_M, np.nan, a)
    return slope(g, res_m), slope(a, res_m)


ALOS_BAD_M = 30.0


def load_dems(site: str, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Высота GLO-30 (строит) и ALOS (подтверждает) на сетке листа, без ям ALOS."""
    g = np.load(ANABAR_DEM / f"dem_glo30_{site}_лист_{int(res_m)}m.npy")
    a = np.load(ANABAR_DEM / f"dem_alos_{site}_лист_{int(res_m)}m.npy")
    return g, np.where(np.abs(a - g) > ALOS_BAD_M, np.nan, a)


def valley_response(img: np.ndarray, sigmas: tuple = SIGMAS) -> np.ndarray:
    """Оси ложбин: линейность по гессиану только там, где наибольшая кривизна положительна.

    Поверхность поперёк линии вогнута — дно ложбины (RELU2, 05.11.2026: под
    найденной частью эталона GTK рельеф — ложбина в 79 % мест). Это гребни
    перевёрнутого рельефа; масштаб ядра — те же SIGMAS, под эталон не
    подбирается.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    best = np.zeros(img.shape, dtype="float32")
    for s in sigmas:
        h = hessian_matrix(img, sigma=s, order="rc", use_gaussian_derivatives=True)
        l1, l2 = hessian_matrix_eigvals(h)
        dom = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
        big = np.maximum(np.abs(l1), np.abs(l2))
        small = np.minimum(np.abs(l1), np.abs(l2))
        r = np.where(dom > 0, s ** 2 * (big - small), 0.0)
        best = np.maximum(best, r.astype("float32"))
    return best


def scarp_response(img: np.ndarray, sigmas: tuple = SIGMAS) -> np.ndarray:
    """Гребни поля уклона: та же линейность, что ridge_response, но только знак «гребень».

    Наибольшая по модулю кривизна отрицательна — поле уклона поперёк линии
    выпукло вверх, то есть это самое крутое место склона. Ложбины поля уклона
    (подошва и бровка — места, где уклон меньше соседнего) гасятся.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    best = np.zeros(img.shape, dtype="float32")
    for s in sigmas:
        h = hessian_matrix(img, sigma=s, order="rc", use_gaussian_derivatives=True)
        l1, l2 = hessian_matrix_eigvals(h)          # l1 >= l2
        dom = np.where(np.abs(l1) >= np.abs(l2), l1, l2)
        big = np.maximum(np.abs(l1), np.abs(l2))
        small = np.minimum(np.abs(l1), np.abs(l2))
        r = np.where(dom < 0, s ** 2 * (big - small), 0.0)
        best = np.maximum(best, r.astype("float32"))
    return best
