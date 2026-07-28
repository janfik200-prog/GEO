"""Загрузка датасета признаков (`dataset_v1`) и построение КРИТЕРИАЛЬНОГО таргета.

В отличие от боевого пайплайна (`src.model`, обучение presence-background на
реальных рудопроявлениях), здесь целевая переменная — результат критериального
анализа ГИС Интегро «Таксономия по критериям» (`prognoz.prognoz.property`),
а НЕ геохимия. Геохимические слои в `dataset_v1` намеренно отсутствуют
(стоп-лист, см. `data/dataset_v1/dataset_v1_sources.md`).

Схема таргета (см. решение пользователя): бинарная классификация —
положительный класс = верхние ``CRITERIAL_TOP_FRAC`` доли ячеек по
перспективности критериального прогноза. В нативной шкале ГИС Интегро
МЕНЬШЕЕ значение = перспективнее (мера Плюты: ближе к эталону-минимуму),
что подтверждено знаком корреляции с расстояниями до факторов
(corr(prognoz, dist_facies) = +0.80). Поэтому положительный класс —
ячейки с НАИМЕНЬШИМ значением ``prognoz.prognoz.property``.

Признаки: НЕЗАВИСИМЫЙ от критериального набор (гравика/магнитка, Landsat-7,
рельеф, dist_dnl/dnara, маска свиты). Факторные признаки, из которых сам
критериальный и построен, исключаются через config.CRITERIAL_EXCLUDE_FEATURES —
иначе модель тривиально восстанавливает формулу (утечка). Сетка датасета совпадает с целевой сеткой критериального
расчёта (154×149, шаг 500 м), поэтому таргет цепляется по (row, col) без
пересчёта координат.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Служебные столбцы датасета (не признаки).
_META_COLS = ("row", "col", "x", "y")


@dataclass
class CriterialDataset:
    """Готовый к обучению набор: признаки, бинарный таргет, координаты, CV-группы."""

    frame: pd.DataFrame            # полный датафрейм (признаки + row/col/x/y + служебные столбцы)
    feature_cols: list[str]       # имена столбцов-признаков (все 36)
    X: np.ndarray                 # матрица признаков (n, 36), NaN импутированы медианой
    y: np.ndarray                 # бинарный таргет (1 — перспективно по критериальному)
    crit: np.ndarray              # непрерывное значение критериального прогноза [0.05..0.91]
    groups: np.ndarray            # id пространственного блока для GroupKFold
    grid_shape: tuple[int, int]   # (rows=Prf, cols=Pic) = (154, 149)


def read_criterial_property(path: Path, grid_shape: tuple[int, int]) -> np.ndarray:
    """Прочитать нативный `.property` ГИС Интегро (плоский float32 LE) в сетку.

    Формат подтверждён размером файла (ObjCount×4) и совпадает с
    :mod:`src.integro_grid`. Возвращает массив формы ``grid_shape``
    (строка 0 — северный край), значения в нативной шкале (меньше = лучше).
    """
    arr = np.fromfile(path, dtype="<f4")
    if arr.size != grid_shape[0] * grid_shape[1]:
        raise ValueError(
            f"{path.name}: {arr.size} значений, ожидалось {grid_shape[0] * grid_shape[1]}"
        )
    return arr.reshape(grid_shape)


def _spatial_blocks(rows: np.ndarray, cols: np.ndarray, block: int) -> np.ndarray:
    """Сгруппировать ячейки в квадратные пространственные блоки `block×block` ячеек.

    Блочная (а не построчная) группировка нужна, чтобы соседние коррелированные
    ячейки не попадали одновременно в train и test — иначе метрики оптимистично
    смещены (утечка через пространственную автокорреляцию).
    """
    br = (rows // block).astype(int)
    bc = (cols // block).astype(int)
    n_bc = bc.max() + 1
    return br * n_bc + bc


def load_criterial_dataset(
    parquet_path: Path | None = None,
    criterial_path: Path | None = None,
    top_frac: float | None = None,
    block_cells: int | None = None,
) -> CriterialDataset:
    """Собрать :class:`CriterialDataset` из `dataset_v1.parquet` и критериального `.property`.

    ``top_frac`` — доля наиболее перспективных ячеек в положительном классе
    (по умолчанию :data:`config.CRITERIAL_TOP_FRAC`). ``block_cells`` — сторона
    пространственного блока для CV (по умолчанию :data:`config.CRITERIAL_BLOCK_CELLS`).
    """
    parquet_path = Path(parquet_path or config.CRITERIAL_DATASET_PARQUET)
    criterial_path = Path(criterial_path or config.GOLD_TARGET_CRITERIAL)
    top_frac = float(top_frac if top_frac is not None else config.CRITERIAL_TOP_FRAC)
    block_cells = int(block_cells if block_cells is not None else config.CRITERIAL_BLOCK_CELLS)

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    rows = df["row"].to_numpy().astype(int)
    cols = df["col"].to_numpy().astype(int)
    grid_shape = (int(rows.max()) + 1, int(cols.max()) + 1)

    # Критериальный таргет: цепляем по (row, col), сетки совпадают.
    crit_grid = read_criterial_property(criterial_path, grid_shape)
    crit = crit_grid[rows, cols].astype(float)
    df = df.assign(criterial=crit)

    # Бинаризация: перспективно = НАИМЕНЬШИЕ значения (нативная шкала: меньше = лучше).
    thresh = np.quantile(crit, top_frac)
    y = (crit <= thresh).astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in _META_COLS
        and c != "criterial"
        and c not in config.CRITERIAL_EXCLUDE_FEATURES
    ]
    # Бинарные геологические признаки (разломы/фации/палеодолины/дайки/коры) —
    # осмысленная замена исключённым непрерывным dist_* факторам.
    if getattr(config, "CRITERIAL_GEO_INDICATORS", False):
        from . import geo_indicators
        dist_cols = {c: df[c].to_numpy() for c in df.columns if c.startswith("dist_")}
        ind = geo_indicators.build_indicators(df, dist_columns=dist_cols)
        for name, vals in ind.items():
            df[name] = vals
            feature_cols.append(name)

        # Производные mineral-systems признаки: пересечения факторов, coincidence,
        # плотности (из индикаторов + датасета).
        if getattr(config, "CRITERIAL_GEO_DERIVED", False):
            for name, vals in geo_indicators.build_derived(ind, df).items():
                df[name] = vals
                feature_cols.append(name)

    # Производные рельефа из relief_m (уклон/кривизна/TPI/шероховатость).
    if getattr(config, "CRITERIAL_TERRAIN_DERIVED", False) and "relief_m" in df.columns:
        from . import terrain_features
        for name, vals in terrain_features.build_terrain(df, grid_shape).items():
            df[name] = vals
            feature_cols.append(name)

    X = df[feature_cols].to_numpy(dtype=float)
    # Импутация NaN медианой столбца (relief_m ~4.7% NaN, гравика/магнитка ~0.1%).
    col_median = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_median, inds[1])

    groups = _spatial_blocks(rows, cols, block_cells)

    return CriterialDataset(
        frame=df,
        feature_cols=feature_cols,
        X=X,
        y=y,
        crit=crit,
        groups=groups,
        grid_shape=grid_shape,
    )
