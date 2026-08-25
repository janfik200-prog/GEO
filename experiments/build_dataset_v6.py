"""Пересборка dataset_v5_rebuilt -> dataset_v6: снятие шума/помех/дублей.

Повторная проверка датасета на шумы, корреляцию и помехи (17.08.2026),
по итогам которой из dataset_v5_rebuilt.parquet (153 столбца = 5 id/маска +
10 служебных колонок покрытия + 138 признаков) убираются НАВСЕГДА:

1. 10 служебных колонок покрытия (`*_valid_frac`, `*_n_obs`) — это метаданные
   о геометрии съёмки (доля покрытия ячейки, число сцен), не геологический
   сигнал. `s1_n_obs` уже был спорным топ-признаком (r=-0.637 к target) из-за
   общего плавного ареального тренда съёмки, а не геологии (см. docstring
   experiments/forecast_dense.py). Раньше их приходилось убирать вручную
   фильтром `cell_mask.is_service` в каждом скрипте отдельно (легко забыть) —
   теперь они убраны из файла один раз.
2. 33 признака с честной permutation importance <=0 на 8 leave-one-strip-out
   фолдах (буфер 15 км, experiments/feature_relevance_check.py ->
   outputs/metrics/feature_relevance_v5_rebuilt.csv) — список `NOISE_FEATURES`
   из experiments/forecast_dense.py, полностью совпадает с
   `mean_importance<=0` в CSV.
3. `l8_green` — единственная оставшаяся после (2) дублирующая пара по
   target-free корреляции (|r|>0.95 с `l8_red`, r=0.9506); `l8_red` сохраняет
   более высокую честную importance (0.000122 против 0.000078 у l8_green) и
   frac_folds_nonpositive ниже (0.125 против 0.375) — оставлен как
   представитель пары.

Группы `ast`/`ls` НЕ трогаются: их исключение в forecast_dense.py/
feature_relevance_check.py — только ради совместимости с dataset_wide.parquet
(в широкой сетке нет ast/ls покрытия), не потому что они шумные.

Круговые факторные столбцы критериальной формулы (группы `geo`/`geo2`,
13 колонок: 8 CRIT_EXCLUDE_FACTOR_FEATURES + 5 geo2_* дублей фаций/узлов
разломов) остаются в файле НЕТРОНУТЫМИ, как и в v5_rebuilt — они нужны
отдельным скриптам воспроизведения факторов (reproduce_paleo_factor.py,
facies_significance.py и т.д.), исключаются только на уровне
criterial_target.training_features(dataset="v6") тем же паттерном, что и
v5_rebuilt (V5_EXCLUDE_GROUPS + V5_EXCLUDE_COLS).

Итог: 153 -> 109 столбцов (5 id/маска + 104 признака-кандидата, из которых
13 круговых факторных остаются только для факторных скриптов и всё равно
исключаются training_features(dataset="v6") -> 91 признак для обучения).
"""

from pathlib import Path

import pandas as pd

SRC = Path("data/processed/dataset_v5_rebuilt.parquet")
DST = Path("data/processed/dataset_v6.parquet")
NOTES = Path("data/processed/dataset_v6_notes.md")

SERVICE_COLS = [
    "s2_valid_frac", "s2_n_obs", "s1_valid_frac", "s1_n_obs",
    "psr_valid_frac", "psr_n_obs", "ast_valid_frac", "ast_n_obs",
    "l8_valid_frac", "l8_n_obs",
]

# честная permutation importance <=0 на 8 leave-one-strip-out фолдах
# (experiments/feature_relevance_check.py, buffer=15km) — совпадает с
# experiments.forecast_dense.NOISE_FEATURES
NOISE_FEATURES = [
    "pf_gr_up1", "gm_mag_flt_35", "psr_linci", "pf_mag_up10", "gm_gr_1GY_25",
    "l8_lst", "s2_b8a", "l8_ferrous", "pf_mag_up5", "pf_mag_up2", "pf_gr_asig",
    "gm_mag_1GFI_25", "pf_gr_vz", "gm_mg_all", "pf_mag_up1", "s2_ferrous",
    "pf_gr_tdr", "s2_bare_frac", "s2_ireci", "pf_mag_tdr", "pf_mag_vz",
    "l8_lst_std", "pf_gm_corr", "l8_iron_ox_std", "psr_hv_std", "s2_b05_std",
    "psr_hh_hv_std", "psr_linci_std", "s2_b03", "l8_swir16_std", "s2_b04_std",
    "s2_b12_std", "gm_mag_2G_25",
]

RESIDUAL_DUP_COLS = ["l8_green"]  # r=0.9506 с l8_red, l8_red сохраняется

DROP_COLS = SERVICE_COLS + NOISE_FEATURES + RESIDUAL_DUP_COLS


def main() -> None:
    df = pd.read_parquet(SRC)
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"колонки для удаления отсутствуют в источнике: {missing}")

    out = df.drop(columns=DROP_COLS)
    out.to_parquet(DST)

    n_id = 5  # row, col, x, y, mask_svita
    n_total_before = df.shape[1]
    n_total_after = out.shape[1]
    n_features_before = n_total_before - n_id - len(SERVICE_COLS)
    n_features_after = n_total_after - n_id

    notes = f"""# dataset_v6 — пересборка после повторной проверки на шум/корреляцию/помехи (17.08.2026)

Источник: `dataset_v5_rebuilt.parquet` ({n_total_before} столбцов = {n_id} id/маска
+ {len(SERVICE_COLS)} служебных колонок покрытия + {n_features_before} признаков,
см. `dataset_v5_rebuilt_notes.md`).

## Что убрано ({len(DROP_COLS)} колонок)

**Служебные колонки покрытия ({len(SERVICE_COLS)}):** {', '.join(SERVICE_COLS)}.
Метаданные съёмки (доля покрытия ячейки, число сцен), не геологический сигнал;
`s1_n_obs` ранее давал ложный топ-признак (r=-0.637 к target) из-за общего
ареального тренда съёмки. Раньше исключались вручную фильтром
`cell_mask.is_service` в каждом скрипте (риск забыть) — теперь убраны из
файла один раз и навсегда.

**Шумовые признаки ({len(NOISE_FEATURES)}):** честная permutation importance
<=0 на 8 leave-one-strip-out фолдах (буфер 15 км от края тестовой полосы,
`experiments/feature_relevance_check.py` ->
`outputs/metrics/feature_relevance_v5_rebuilt.csv`), список идентичен
`NOISE_FEATURES` из `experiments/forecast_dense.py`:
{', '.join(NOISE_FEATURES)}.

**Остаточный дубль (1):** `l8_green` — после снятия (1)-(2) остаётся
единственная пара с target-free корреляцией |r|>0.95: `l8_green`~`l8_red`
(r=0.9506). Оставлен `l8_red` (честная importance 0.000122 против 0.000078,
доля фолдов с importance<=0 — 12.5% против 37.5%).

## Что НЕ тронуто

- Группы `ast`/`ls` — их исключение в forecast_dense.py/
  feature_relevance_check.py было только ради совместимости с
  `dataset_wide.parquet` (нет ast/ls покрытия на широкой сетке), не потому
  что они шумные. В `dataset_v6` они остаются как обычные признаки.
- Круговые факторные столбцы критериальной формулы (группы `geo`/`geo2`,
  13 колонок: `dist_tect1, dist_tect2, dens_tect, dist_magm, dens_magm,
  dist_struct, dist_facies, dist_paleo, geo2_contact, geo2_facies2,
  geo2_strike_cos, geo2_strike_sin, geo2_fault_node`) — остаются в файле,
  как и в v5_rebuilt, поскольку нужны отдельным скриптам воспроизведения
  факторов (`reproduce_paleo_factor.py`, `facies_significance.py` и т.д.).
  При обучении независимой модели по-прежнему исключаются на уровне
  `criterial_target.training_features(dataset="v6")` (та же логика
  `V5_EXCLUDE_GROUPS`/`V5_EXCLUDE_COLS`, что и для v5/v5_rebuilt).

## Итог

{n_total_before} -> {n_total_after} столбцов ({n_id} id/маска +
{n_features_after} признаков-кандидатов, из которых 13 круговых факторных
исключаются на уровне training_features(dataset="v6") -> 91 признак для
независимого обучения).

Построено `experiments/build_dataset_v6.py`.
"""
    NOTES.write_text(notes, encoding="utf-8")

    print(f"{SRC} -> {DST}")
    print(f"columns: {n_total_before} -> {n_total_after}")
    print(f"dropped: {len(DROP_COLS)} ({len(SERVICE_COLS)} service + {len(NOISE_FEATURES)} noise + {len(RESIDUAL_DUP_COLS)} dup)")
    print(f"training features (after geo/geo2 exclusion): {n_features_after - 13}")


if __name__ == "__main__":
    main()
