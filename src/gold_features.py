"""Признаковая матрица золотой задачи из датасета v1 + автопроверка стоп-листа.

Реестр категорий — в :mod:`src.config` (``GOLD_FEATURES_*``): обучение,
поля направлений (только sin/cos), стоп-лист (прямые признаки минерагенической
карты — только заверка), служебные слои. Здесь — код, который делает реестр
обязательным: сборка матрицы падает, если в датасете появился слой стоп-листа
или слой вне реестра.
"""

import numpy as np
import pandas as pd

from . import config


def validate_features(layer_names: list[str]) -> None:
    """Проверить слои датасета против реестра; нарушение — ``ValueError``.

    Ловит два сценария: (1) в датасет просочился слой стоп-листа — прямой
    признак не должен попасть в обучение даже случайно; (2) появился слой,
    не описанный в реестре, — его категорию нужно решить явно, а не молча.
    """
    names = set(layer_names)
    stop_hit = names & set(config.GOLD_FEATURES_STOP)
    if stop_hit:
        raise ValueError(f"Слои стоп-листа в датасете: {sorted(stop_hit)} — "
                         "прямые признаки минерагенической карты не идут в обучение")
    known = (set(config.GOLD_FEATURES_TRAIN) | set(config.GOLD_FEATURES_ANGLE)
             | set(config.GOLD_FEATURES_AUX))
    unknown = names - known
    if unknown:
        raise ValueError(f"Слои вне реестра признаков: {sorted(unknown)} — "
                         "добавь их в GOLD_FEATURES_* в src/config.py осознанно")


def build_feature_matrix(layers: dict[str, np.ndarray]) -> pd.DataFrame:
    """Собрать матрицу «пиксель x признак» из слоёв датасета по реестру.

    Порядок строк — C-порядок сетки (строка 0 — север). Поля направлений
    раскладываются в ``<имя>_sin``/``<имя>_cos``; служебные и стоп-слои
    в матрицу не попадают. Вызывает :func:`validate_features`.
    """
    validate_features(list(layers))
    cols: dict[str, np.ndarray] = {}
    for name in config.GOLD_FEATURES_TRAIN:
        if name in layers:
            cols[name] = layers[name].ravel().astype(np.float32)
    for name in config.GOLD_FEATURES_ANGLE:
        if name in layers:
            rad = np.deg2rad(layers[name].ravel().astype(np.float32))
            cols[f"{name}_sin"] = np.sin(rad)
            cols[f"{name}_cos"] = np.cos(rad)
    return pd.DataFrame(cols)


def load_feature_matrix(npz_path=None) -> pd.DataFrame:
    """Загрузить ``dataset_v1.npz`` и собрать матрицу признаков по реестру."""
    path = npz_path or (config.PROCESSED_DIR / "dataset_v1.npz")
    with np.load(path) as data:
        layers = {k: data[k] for k in data.files if not k.startswith("_")}
    return build_feature_matrix(layers)
