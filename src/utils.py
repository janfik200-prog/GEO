"""Вспомогательные функции: нормализация, сглаживание, морфология сетки.

Все функции чистые — работают только со своими аргументами, без глобального
состояния. Функции сглаживания и морфологии ожидают в ``grid`` целочисленные
столбцы ``row`` и ``col`` (растровые координаты ячейки).
"""

import numpy as np
import pandas as pd


def normalize_01(values) -> np.ndarray:
    """Линейная нормализация массива в диапазон [0, 1] по min/max.

    NaN сохраняются; константный массив отображается в 0.5.
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    mn = np.nanmin(arr[finite])
    mx = np.nanmax(arr[finite])
    if np.isclose(mx, mn):
        return np.full_like(arr, 0.5, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    out[finite] = (arr[finite] - mn) / (mx - mn)
    return out


def robust_normalize_01(values, q_low: float = 0.03, q_high: float = 0.97) -> np.ndarray:
    """Нормализация в [0, 1] по квантилям — устойчива к выбросам.

    При вырожденных квантилях откатывается на :func:`normalize_01`.
    """
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr, dtype=float)
    lo = np.nanquantile(arr[finite], q_low)
    hi = np.nanquantile(arr[finite], q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return normalize_01(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def smooth_on_regular_grid(
    grid: pd.DataFrame, value_col: str, shape: tuple[int, int], passes: int = 1
) -> np.ndarray:
    """Сгладить значения столбца ядром 3×3 с учётом пропусков.

    Раскладывает значения ячеек в растр ``shape``, применяет ``passes`` проходов
    взвешенной свёртки (NaN не размывают соседей) и возвращает значения обратно
    в порядке строк ``grid``.
    """
    try:
        from scipy.signal import convolve2d
    except Exception:
        return grid[value_col].to_numpy()
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    arr = np.full(shape, np.nan, dtype=float)
    arr[rows, cols] = grid[value_col].to_numpy()
    kernel = np.array([[1.0, 1.2, 1.0], [1.2, 3.0, 1.2], [1.0, 1.2, 1.0]], dtype=float)
    smoothed = arr.copy()
    for _ in range(max(1, passes)):
        valid = np.isfinite(smoothed).astype(float)
        filled = np.nan_to_num(smoothed, nan=0.0)
        num = convolve2d(filled, kernel, mode="same", boundary="fill", fillvalue=0)
        den = convolve2d(valid, kernel, mode="same", boundary="fill", fillvalue=0)
        smoothed = np.where(den > 0, num / den, np.nan)
    return smoothed[rows, cols]


def local_max_mask(grid: pd.DataFrame, value_col: str, shape: tuple[int, int]) -> np.ndarray:
    """Булева маска ячеек — локальных максимумов в окне 3×3.

    Без scipy откатывается на порог по квантилю 0.98.
    """
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    vals = grid[value_col].to_numpy()
    try:
        from scipy.ndimage import maximum_filter
    except Exception:
        thr = np.nanquantile(vals, 0.98)
        return vals >= thr
    arr = np.full(shape, np.nan, dtype=float)
    arr[rows, cols] = vals
    filled = np.nan_to_num(arr, nan=-9999.0)
    locmax = maximum_filter(filled, size=3, mode="nearest")
    return (np.isfinite(arr) & (filled >= locmax))[rows, cols]


def keep_large_components(
    grid: pd.DataFrame, bool_col: str, shape: tuple[int, int], min_cells: int = 4
) -> np.ndarray:
    """Оставить только связные компоненты булевого столбца размером ≥ ``min_cells``.

    Связность — по 8 соседям. Без scipy возвращает исходную маску без фильтрации.
    """
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    try:
        from scipy import ndimage
    except Exception:
        return grid[bool_col].to_numpy().astype(bool)
    arr = np.zeros(shape, dtype=np.uint8)
    arr[rows, cols] = grid[bool_col].to_numpy().astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, _ = ndimage.label(arr, structure=structure)
    sizes = np.bincount(labeled.ravel())
    keep_ids = np.where(sizes >= min_cells)[0]
    keep = np.isin(labeled, keep_ids) & (labeled > 0)
    return keep[rows, cols]


def top_n_components(
    grid: pd.DataFrame, bool_col: str, strength_col: str, shape: tuple[int, int],
    min_cells: int = 4, n: int = 8,
) -> np.ndarray:
    """Маска ячеек ``n`` сильнейших связных компонент булева ``bool_col``.

    Связность по 8 соседям; компоненты размером < ``min_cells`` отбрасываются;
    остальные ранжируются по сумме ``strength_col`` внутри (размер × интенсивность)
    и берутся верхние ``n``. В отличие от абсолютных порогов, всегда возвращает
    зоны (если связные пятна есть) — устойчиво к смене распределения/признаков.
    Без scipy возвращает исходную маску без обработки.
    """
    rows = grid["row"].to_numpy()
    cols = grid["col"].to_numpy()
    try:
        from scipy import ndimage
    except Exception:
        return grid[bool_col].to_numpy().astype(bool)
    arr = np.zeros(shape, dtype=np.uint8)
    arr[rows, cols] = grid[bool_col].to_numpy().astype(np.uint8)
    labeled, n_lab = ndimage.label(arr, structure=np.ones((3, 3), dtype=np.uint8))
    cell_lab = labeled[rows, cols]
    strength = grid[strength_col].to_numpy()
    scored = []
    for lid in range(1, n_lab + 1):
        m = cell_lab == lid
        if m.sum() >= min_cells:
            scored.append((float(strength[m].sum()), lid))
    top_ids = [lid for _, lid in sorted(scored, reverse=True)[:n]]
    return np.isin(cell_lab, top_ids)
