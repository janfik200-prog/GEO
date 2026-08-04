"""Пул признаков v2: единственный источник правды для лестницы на датасете v2.

Роль та же, что у :mod:`src.features_v11` в фазе v1, — собрать матрицу признаков
по ЯВНЫМ правилам, а не по «всё, что есть в parquet». Правила ровно две штуки:

1. алгебраические дубли не входят (иначе корреляционная матрица вырождается и
   расстояние Махаланобиса теряет смысл): модуль градиента = f(компонент),
   полное поле = регионалка + остаток, ``dem_curv`` = -``dem_tpi_500``,
   ``dem_elev_std`` = f(``dem_slope``);
2. один и тот же физический признак не входит дважды из разных источников:
   ``relief_m`` (v1, 4.7% пропусков) и ``dem_elev`` (v2, пропусков нет) —
   Spearman ровно +1.00, остаётся вариант без пропусков.

Сырые каналы Landsat заменены отношениями (см. :mod:`src.features_v11`):
абсолютная яркость на 71° с.ш. — это освещённость склона, а не свойство породы.

Группы (``config.V2_FEATURE_GROUPS``) нужны абляциям этапа 4: выключается
источник целиком, потому что вопрос ставится как «что дал Sentinel-2», а не
«что дал канал b11».
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cell_mask, config, features_v11, lineaments, terrain_v2

#: Признаки, исключённые из пула v2, и причина (проверено измерением).
V2_DROPPED: dict[str, str] = {
    "gm_gr_1G_25": "модуль градиента = f(gm_gr_1GX_25, gm_gr_1GY_25)",
    "gm_mag_1G_25": "модуль градиента = f(gm_mag_1GX_25, gm_mag_1GY_25)",
    "gm_gr_all": "полное поле = gm_gr_flt_35 + gm_gr_ost_35 (max|d| 1e-6)",
    "gm_mg_all": "полное поле = gm_mag_flt_35 + gm_mag_ost_35 (max|d| 1e-6)",
    "dem_curv": "= -dem_tpi_500 (r = -1.00): TPI в окне 5 px и есть лапласиан",
    "dem_elev_std": "= f(dem_slope) (r = +0.98) на ячейке 500 м",
    "relief_m": "= dem_elev (Spearman +1.00), но с 4.7% пропусков",
    # Сырые средние каналы Sentinel-2: то же правило, что для Landsat в v1.1 —
    # на 71° с.ш. абсолютная яркость это освещённость склона. Отношения
    # предсказываются каналами линейно на R^2 = 0.97-0.99, то есть в пуле
    # присутствовали и функция, и её аргументы: обусловленность падает
    # 1.16e5 -> 2.55e4. Текстуры (``*_std``) остаются: это разброс внутри
    # ячейки, а не её яркость.
    "s2_b02": "среднее сырого канала = f(отношений и освещённости склона)",
    "s2_b03": "среднее сырого канала = f(отношений и освещённости склона)",
    "s2_b04": "среднее сырого канала = f(отношений и освещённости склона)",
    "s2_b8a": "среднее сырого канала = f(отношений и освещённости склона)",
    "s2_b11": "среднее сырого канала = f(отношений и освещённости склона)",
    "s2_b12": "среднее сырого канала = f(отношений и освещённости склона)",
    # ferric = b04/b8a, ndvi = (b8a-b04)/(b8a+b04) = (1-ferric)/(1+ferric):
    # дробно-линейное преобразование ОДНОЙ величины, r = -0.996 (у std +0.987).
    # Содержательный вывод: «минеральный» ферриковый индекс на этом листе не
    # несёт ничего сверх вегетационного, что согласуется с замером покрова.
    "s2_ferric": "= монотонная функция s2_ndvi (r = -0.996), та же пара каналов",
    "s2_ferric_std": "= монотонная функция s2_ndvi_std (r = +0.987)",
}

#: Отброс, у которого есть УСЛОВИЕ: колонка -> чем она заменена.
#: Дубль убирается, только если замена реально пришла в датасет. Иначе сборка
#: на неполном наборе источников (нет terrain_v2.parquet) молча теряла бы
#: рельеф целиком — а он в v1 был сильнейшим признаком лестницы.
V2_DROPPED_IF: dict[str, str] = {"relief_m": "dem_elev"}


def dropped_columns(df: pd.DataFrame) -> dict[str, str]:
    """Что именно исключается из пула для ДАННОГО датасета, с причиной."""
    out = {c: why for c, why in V2_DROPPED.items()
           if c not in V2_DROPPED_IF or V2_DROPPED_IF[c] in df.columns}
    return out


def pool_features(df: pd.DataFrame, include_dist: bool = False,
                  groups: tuple[str, ...] | None = None,
                  include_factors: bool = False) -> pd.DataFrame:
    """Матрица признаков v2 из собранного ``dataset_v2.parquet``.

    ``groups`` — какие группы оставить (для абляций); ``None`` = все доступные.
    Отсутствующие группы (закрытая ветка, недосчитанный источник) молча
    пропускаются: сборка обязана работать при неполном наборе источников.

    ``include_factors`` — пул v3: вернуть в признаки 8 геологических факторных
    слоёв (``config.V3_GEO_FACTORS``). По умолчанию выключено, чтобы прогоны
    этапа 4 остались воспроизводимыми; включать осмысленно ТОЛЬКО при обучении
    без учителя — при обучении на псевдометках критериального анализа это
    циркулярность (см. комментарий у ``config.V3_GEO_FACTORS``).

    База (``gm``, ``ls``, ``relief_v1``, опционально ``dist``) берётся у
    :func:`src.features_v11.ladder_features`, группы v2 добавляются здесь
    ЯВНЫМИ списками — ``cell_mask.feature_columns`` их не отдаёт намеренно
    (маска фазы зафиксирована на v1, см. ``cell_mask.V2_SOURCE_PREFIXES``).
    """
    base = features_v11.ladder_features(df, include_dist=include_dist)

    extra: list[str] = [c for c in terrain_v2.TERRAIN_COLS if c in df.columns]
    extra += [c for c in lineaments.LINEAMENT_COLS if c in df.columns]
    extra += [c for c in df.columns
              if c.startswith("s2_") and c not in cell_mask.SERVICE_COLS]
    if include_factors:
        extra += [c for c in config.V3_GEO_FACTORS if c in df.columns]
    extra = [c for c in dict.fromkeys(extra) if c not in base.columns]

    out = pd.concat([base, df[extra]], axis=1) if extra else base.copy()
    out = out.drop(columns=[c for c in dropped_columns(df) if c in out.columns])
    if groups is not None:
        out = out[[c for c in out.columns if feature_group(c) in groups]]
    return out


def feature_group(col: str) -> str | None:
    """Группа признака по реестру ``config.V2_FEATURE_GROUPS``."""
    for name, prefixes in config.V2_FEATURE_GROUPS.items():
        if any(col.startswith(p) for p in prefixes):
            return name
    return None


def group_sizes(feat: pd.DataFrame) -> dict[str, int]:
    """Сколько признаков в каждой группе пула (для отчётов и абляций)."""
    sizes: dict[str, int] = {}
    for c in feat.columns:
        g = feature_group(c) or "прочее"
        sizes[g] = sizes.get(g, 0) + 1
    return dict(sorted(sizes.items()))


def condition_number(feat: pd.DataFrame, valid: np.ndarray) -> float:
    """Обусловленность корреляционной матрицы пула (диагностика вырождения)."""
    return features_v11.condition_number(feat, valid)
