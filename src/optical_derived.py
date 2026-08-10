"""Оптические индексы поверх УЖЕ материализованных каналов датасета.

Аудит (``docs II presentation/Аудит-признаков-что-не-вытащено.md``) отметил
отсутствие индексов красного края и канонического Mg-OH индекса ASTER. Оба
считаются здесь напрямую из колонок ``s2_b05``/``s2_b06``/``s2_b07``/``s2_b8a``
и ``ast_b08``/``ast_b09``, уже присутствующих в ``dataset_v3.parquet`` (этап 9,
``src/s2_composite.py`` и ``src/sat_sources.py``) — новых скачиваний и
пересборки композитов не требуется.

Ограничение: колонки датасета — это среднее (и std) по ячейке, посчитанное
на этапе сборки композита. Индексы здесь считаются как отношение СРЕДНИХ, а
не среднее отношений по пикселям (иначе потребовался бы доступ к исходным
поканальным растрам) — смещение от неравенства Йенсена при типичном для
композита разбросе внутри ячейки пренебрежимо мало. Из-за этого std новых
индексов не считается: для него нужна поканальная ковариация внутри ячейки,
которой в материализованном датасете нет.

* ``s2_ndre`` — Normalised Difference Red Edge (Barnes et al., 2000):
  ``(b8a - b05) / (b8a + b05)``. В композите нет широкого NIR (``b08``),
  поэтому используется узкий ``b8a`` — стандартная замена в спутниковых
  индексах Sentinel-2.
* ``s2_ireci`` — Inverted Red-Edge Chlorophyll Index (Frampton et al., 2013):
  ``(b07 - b04) / (b05 / b06)``.
* ``ast_mgoh`` — простое двухканальное отношение ``b08 / b09``, выделяющее
  полосу поглощения Mg-OH на 2.33 мкм (``b09``) относительно плеча 2.29 мкм
  (``b08``). Отдельно от уже посчитанного ``ast_carb`` = ``(b06+b09)/(b07+b08)``
  (Rowan & Mars, 2003, карбонат/хлорит/эпидот) — при сборке проверяется
  корреляция между ними как тест на скрытое дублирование.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OPTICAL_DERIVED_COLS: tuple[str, ...] = ("s2_ndre", "s2_ireci", "ast_mgoh")


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return np.where(np.isfinite(r), r, np.nan)


def red_edge_indices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """NDRE и IRECI из колонок ``s2_b04``, ``s2_b05``, ``s2_b06``, ``s2_b07``, ``s2_b8a``."""
    b04, b05, b06, b07, b8a = (
        df[f"s2_{c}"].to_numpy(dtype=float) for c in ("b04", "b05", "b06", "b07", "b8a")
    )
    ndre = _safe_div(b8a - b05, b8a + b05)
    ireci = _safe_div(b07 - b04, _safe_div(b05, b06))
    return {"s2_ndre": ndre.astype(np.float32), "s2_ireci": ireci.astype(np.float32)}


def aster_mgoh_index(df: pd.DataFrame) -> np.ndarray:
    """Mg-OH индекс ``ast_b08 / ast_b09``."""
    b08 = df["ast_b08"].to_numpy(dtype=float)
    b09 = df["ast_b09"].to_numpy(dtype=float)
    return _safe_div(b08, b09).astype(np.float32)


def optical_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица ``OPTICAL_DERIVED_COLS``, тот же порядок строк, что и ``df``."""
    out = red_edge_indices(df)
    out["ast_mgoh"] = aster_mgoh_index(df)
    return pd.DataFrame(out)[list(OPTICAL_DERIVED_COLS)]
