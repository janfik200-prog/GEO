"""Формы окрестности: вытянутые ядра и статистики соседства, кроме среднего.

Зачем. Всё сглаживание в проекте до сих пор было круглым и усредняющим: гауссово
ядро с одной сигмой, одинаковое во все стороны, и единственная статистика —
среднее по окрестности. Обе оговорки спорны для этой территории. Рудные зоны
здесь вытянуты на северо-запад вслед за тектоно-флюидитными зонами, поэтому
окрестность, вытянутая вдоль местного простирания, должна усреднять «вдоль
рудной зоны», а не размазывать её поперёк. И среднее — не единственная сводка
окрестности: разброс, квантили и местный контраст (значение минус сглаженное)
описывают то, чего среднее не видит, — насколько ячейка выделяется на фоне
соседей.

Ориентация. Направление структур берётся из признаков линеаментов
``lin_dir_sin``/``lin_dir_cos``: это sin и cos удвоенного угла (данные осевые,
у линии нет начала и конца), угол считается в системе растра — ось X вправо по
столбцам, ось Y вниз по строкам (:func:`src.lineaments._draw`). Поэтому все углы
здесь тоже в системе растра; для отчёта их переводит в географический азимут
:func:`axis_azimuth_deg`.

Пропуски и маски. Как и в :func:`experiments.branch1_multiscale_ensemble
.smooth_field`, свёртка нормируется на маску: пропуски и ячейки вне маски не
дают вклада, а результат за пределами маски — пропуск. Свёртка идёт через БПФ:
ядра здесь крупные (сигма вдоль до 20 км = 40 ячеек, ядро 241x241), прямая
свёртка была бы на порядки дороже.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import fftconvolve

__all__ = ["oriented_kernel", "masked_convolve", "aniso_smooth", "steered_smooth",
           "local_axis", "axis_azimuth_deg", "orientation_field", "local_contrast",
           "local_spread", "local_quantile"]


def oriented_kernel(sigma_along: float, sigma_across: float, theta: float,
                    trunc: float = 3.0) -> np.ndarray:
    """Гауссово ядро, вытянутое вдоль направления `theta` (радианы, система растра).

    Сигмы — в ячейках. При ``sigma_along == sigma_across`` ядро круглое.
    """
    half = int(max(1, np.ceil(trunc * max(sigma_along, sigma_across))))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    u = xx * np.cos(theta) + yy * np.sin(theta)          # вдоль
    v = -xx * np.sin(theta) + yy * np.cos(theta)         # поперёк
    k = np.exp(-0.5 * ((u / sigma_along) ** 2 + (v / sigma_across) ** 2))
    return k / k.sum()


def masked_convolve(v: np.ndarray, shape, kernel: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Свёртка с нормировкой на маску (аналог `smooth_field` для любого ядра)."""
    ok = np.isfinite(v)
    if mask is not None:
        ok = ok & np.asarray(mask, dtype=bool)
    g = np.where(ok, v, 0.0).reshape(shape)
    w = ok.astype(float).reshape(shape)
    num = fftconvolve(g, kernel, mode="same")
    den = fftconvolve(w, kernel, mode="same")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).ravel()
    return np.where(ok, out, np.nan) if mask is not None else out


def aniso_smooth(v: np.ndarray, shape, sigma_along: float, sigma_across: float,
                 theta: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Сглаживание ядром, вытянутым вдоль одного общего для листа направления."""
    return masked_convolve(v, shape, oriented_kernel(sigma_along, sigma_across, theta),
                           mask)


def local_axis(dir_sin: np.ndarray, dir_cos: np.ndarray) -> np.ndarray:
    """Угол оси структур в каждой ячейке (радианы, система растра, по модулю pi)."""
    return 0.5 * np.arctan2(np.asarray(dir_sin, float), np.asarray(dir_cos, float))


def axis_azimuth_deg(theta: np.ndarray | float) -> np.ndarray | float:
    """Угол оси в системе растра -> географический азимут в градусах (0-180).

    В системе растра строки растут на юг, поэтому направление (cos t, sin t) в
    (столбец, строка) имеет восточную составляющую cos t и северную -sin t.
    """
    return np.degrees(np.arctan2(np.cos(theta), -np.sin(theta))) % 180.0


def steered_smooth(v: np.ndarray, shape, sigma_along: float, sigma_across: float,
                   theta_field: np.ndarray, mask: np.ndarray | None = None,
                   nbins: int = 8, iso_weight: np.ndarray | None = None) -> np.ndarray:
    """Сглаживание вытянутым ядром, чья ориентация меняется от ячейки к ячейке.

    Карта фильтруется `nbins` ядрами с равномерно расставленными ориентациями, а
    в каждой ячейке берётся взвешенная смесь двух ближайших по углу результатов
    (треугольные веса по угловому расстоянию между осями). Это обычный приём для
    полей ориентации: честного «ядра, поворачивающегося непрерывно», в свёртке
    не существует, а восьми ориентаций хватает — соседние отличаются на 22.5
    градуса, что меньше разброса самих направлений линеаментов внутри ячейки.

    ``iso_weight`` (0..1) — доля круглого ядра, подмешиваемого в каждой ячейке:
    там, где структуры не выражены, вытягивать окрестность не по чему.
    """
    th = np.mod(np.asarray(theta_field, float), np.pi)
    bins = np.arange(nbins) * np.pi / nbins
    step = np.pi / nbins
    acc = np.zeros(th.size)
    wsum = np.zeros(th.size)
    finite = None
    for tb in bins:
        s = aniso_smooth(v, shape, sigma_along, sigma_across, tb, mask)
        d = np.abs((th - tb + np.pi / 2) % np.pi - np.pi / 2)   # угол между осями
        w = np.clip(1.0 - d / step, 0.0, 1.0)
        ok = np.isfinite(s)
        finite = ok if finite is None else (finite | ok)
        acc += np.where(ok, s, 0.0) * w
        wsum += np.where(ok, w, 0.0)
    out = np.where(wsum > 1e-9, acc / np.maximum(wsum, 1e-9), np.nan)
    out = np.where(finite, out, np.nan)
    if iso_weight is not None:
        sig = float(np.sqrt(sigma_along * sigma_across))
        iso = aniso_smooth(v, shape, sig, sig, 0.0, mask)
        a = np.clip(np.asarray(iso_weight, float), 0.0, 1.0)
        out = np.where(np.isfinite(out) & np.isfinite(iso), (1 - a) * out + a * iso, out)
    return out


def local_contrast(v: np.ndarray, shape, sigma_cells: float,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Значение минус сглаженное по окрестности: насколько ячейка выделяется."""
    return v - aniso_smooth(v, shape, sigma_cells, sigma_cells, 0.0, mask)


def local_spread(v: np.ndarray, shape, sigma_cells: float,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """Разброс (стандартное отклонение) по гауссовой окрестности."""
    m1 = aniso_smooth(v, shape, sigma_cells, sigma_cells, 0.0, mask)
    m2 = aniso_smooth(np.asarray(v, float) ** 2, shape, sigma_cells, sigma_cells, 0.0,
                      mask)
    return np.sqrt(np.clip(m2 - m1 ** 2, 0.0, None))


def local_quantile(v: np.ndarray, shape, size_cells: int, q: float) -> np.ndarray:
    """Квантиль по квадратной окрестности (`size_cells` — сторона окна в ячейках).

    Пропуски заполняются медианой: скользящий квантиль в scipy их не понимает, а
    в признаках после сборки датасета пропусков почти нет.
    """
    g = np.asarray(v, float).reshape(shape)
    g = np.where(np.isfinite(g), g, np.nanmedian(g))
    size = int(max(3, size_cells) | 1)
    return percentile_filter(g, percentile=100.0 * q, size=size, mode="nearest").ravel()


def orientation_field(dir_sin: np.ndarray, dir_cos: np.ndarray, weight: np.ndarray,
                      shape, sigma_cells: float) -> tuple[np.ndarray, np.ndarray]:
    """Поле направлений структур по ячейкам: угол оси и согласованность.

    В сыром виде ``lin_dir_*`` заданы только там, где через ячейку прошёл хотя бы
    один линеамент (в этом листе таких ячеек около четверти), в остальных стоят
    нули, то есть ложное «направление на восток». Поэтому удвоенные углы
    складываются как векторы с весом ``weight`` (длина линеаментов на анизотропию)
    и сглаживаются: получается непрерывное поле, покрывающее лист, и мера
    согласованности — длина усреднённого вектора, деленная на сумму весов
    (1 — все линеаменты окрестности параллельны, 0 — направления гасят друг друга).
    """
    ws, wc, ww = (aniso_smooth(a, shape, sigma_cells, sigma_cells, 0.0)
                  for a in (weight * dir_sin, weight * dir_cos, weight))
    theta = 0.5 * np.arctan2(ws, wc)
    coh = np.hypot(ws, wc) / np.maximum(ww, 1e-9)
    return theta, np.clip(np.nan_to_num(coh), 0.0, 1.0)


def theta_from_azimuth(az_deg: float) -> float:
    """Географический азимут (градусы) -> угол оси в системе растра (радианы)."""
    a = np.radians(az_deg)
    return float(np.arctan2(-np.cos(a), np.sin(a)))
