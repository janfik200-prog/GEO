"""Маска валидных ячеек и литологические домены для оценки нетипичности.

Нетипичность != рудоносность: вода, края покрытия и ячейки с большой долей
пропусков дают ложные максимумы у любых детекторов аномалий (Махаланобис,
PCA-невязка, автоэнкодер). Такие ячейки исключаются из обучения детекторов
и из пула оценки ДО расчёта скора, а не постфактум по карте.

Вход — таблица датасета (`data/processed/dataset_v1.parquet` и будущий v2):
22 946 строк в C-порядке сетки prognoz (строка 0 — север), колонки-слои
по реестру ``GOLD_FEATURES_*`` в :mod:`src.config`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Колонки-координаты: не признаки, в учёте пропусков не участвуют.
_COORD_COLS = ("row", "col", "x", "y")

# Служебные колонки источников v2: число наблюдений и доля валидных пикселей
# описывают качество съёмки, а не свойства площади.
SERVICE_COLS: tuple[str, ...] = ("s2_n_obs", "s2_valid_frac", "mask_valid",
                                 "domain_id")

# Группы, добавленные в датасете v2. Маска фазы ФИКСИРОВАНА на признаках v1:
# иначе площадь оценки менялась бы при каждом новом источнике и протокол
# заверки этапа 1 стал бы несравним с этапами 4-7. Дыры в Sentinel-2 (облака,
# край полосы) не должны выкидывать ячейку из площади — их лечит импутация
# внутри детектора. Пул признаков лестницы v2 собирает src.features_v2.
V2_SOURCE_PREFIXES: tuple[str, ...] = ("dem_", "lin_", "s2_", "ast_", "gamma_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Независимые признаки датасета: без координат, служебных, стоп-листа
    и факторных слоёв критериального анализа (те исключены из всей фазы
    «собственный алгоритм» по той же причине циркулярности, что и в LOO).

    Используется МАСКОЙ (учёт доли пропусков). Список признаков лестницы
    нетипичности — :func:`src.features_v11.ladder_features` (single source
    of truth фазы): он берёт этот набор за базу, но чистит избыточные
    градиентные колонки и заменяет сырые каналы Landsat отношениями."""
    drop = (set(_COORD_COLS) | set(config.GOLD_FEATURES_AUX) | set(SERVICE_COLS)
            | set(config.GOLD_FEATURES_STOP) | set(config.CRIT_EXCLUDE_FACTOR_FEATURES))
    return [c for c in df.columns
            if c not in drop and not c.startswith(V2_SOURCE_PREFIXES)]


def build_valid_mask(df: pd.DataFrame) -> np.ndarray:
    """Булева маска валидных ячеек (длина = числу строк датасета).

    Невалидны: ячейки воды (внутри ареалов гидросети, ``dist_dnara == 0``)
    и ячейки, где доля NaN среди независимых признаков превышает
    ``config.MASK_MAX_NAN_FRAC``. Остальные пропуски (например, ``relief_m``
    в 4.7% ячеек) остаются валидными — их обрабатывает импутация в детекторе,
    а не маска.
    """
    nan_frac = df[feature_columns(df)].isna().mean(axis=1).to_numpy()
    valid = nan_frac <= config.MASK_MAX_NAN_FRAC
    if "dist_dnara" in df.columns:
        valid &= df["dist_dnara"].to_numpy() > 0
    return valid


def build_domains(df: pd.DataFrame) -> np.ndarray:
    """Целочисленный код литодомена на ячейку.

    Пока домены = ``mask_svita`` (0 — вне полигона свит, 1 — внутри; свиты
    покрывают 66.6% листа, все критериальные объекты внутри). Нормировка
    детекторов по доменам снимает эффект «нетипично просто потому, что другая
    геология»: внутридоменный выброс ищется относительно своего фона.
    """
    if "mask_svita" not in df.columns:
        return np.zeros(len(df), dtype=int)
    dom = df["mask_svita"].to_numpy()
    return np.nan_to_num(dom, nan=0.0).astype(int)
