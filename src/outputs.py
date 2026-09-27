"""Куда складывать картинки: папка на задачу, а не общая свалка.

К сентябрю 2026 в `outputs/` лежало 119 PNG одним списком, и понять, какая
карта к какой задаче относится, можно было только по имени файла. Здесь задан
разбор по задачам проекта: имя файла -> папка. Правила проверяются по порядку,
первое совпадение выигрывает, поэтому частные правила стоят выше общих.

Как пользоваться:

* новый скрипт пишет картинку через :func:`png` — файл сразу ложится в свою
  папку;
* старые скрипты пишут в корень `outputs/` как раньше; `python -m
  experiments.sort_outputs` раскладывает накопившееся и показывает, что не
  попало ни под одно правило.

Папки нумерованы по ходу работы, чтобы в проводнике они шли в осмысленном
порядке, а не по алфавиту.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Порядок важен: правила проверяются сверху вниз.
RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("01_исходные_данные", (
        "ast_preview", "astir_preview", "l8_preview", "s1_preview",
        "s2_composite_preview", "terrain_v2_preview", "psr_preview",
        "quartz_index", "lineaments_preview", "lineaments_sat_overlay",
        "lineaments_sat_roses", "lineaments_vs_faults_map", "catchment_check",
        "ggk200_overview_map", "customer_sheets_gravmag_map",
        "dataset_v2_preview", "dataset_v3_preview", "dataset_v3_full",
        "dataset_v5_preview", "dataset_v6_preview")),
    ("02_критериальный_метод", (
        "criterial_prognoz_map", "criterial_target_v5", "crit_agreement",
        "criterial_reconstruct",
        "criterial_side_by_side",
        "critmulti_formula_diff",
        "compare_methods", "training_points_vs_criterial",
        "verification_points_vs_criterial", "main_sheet_train_test_map",
        "facies_significance", "no_paleo_map", "paleo_factor_map")),
    ("03_признаковое_пространство", (
        "feature_space_", "correlation_clustering", "feature_batch_elimination",
        "feature_elimination_precise", "feature_group_redundancy",
        "nested_feature_elimination", "elastic_net_batch_elimination",
        "rf_vs_elastic_net_elimination", "dataset_v6_noise_breakdown")),
    ("04_нетипичность", ("atypicality_",)),
    ("05_подбор_рабочей_точки", (
        "branch1_", "branch2_", "branch3_", "branch4_", "branch6_",
        "operating_point", "seed_stability", "vote_ensemble",
        "supervised_sweep", "map_sharpness")),
    ("06_прогнозные_карты", (
        "full_sheet_map", "honest_forecast", "forecast_compare",
        "forecast_dense_map", "forecast_dense_points",
        "forecast_dense_blend_compare", "forecast_dense_compare",
        "forecast_dense_pruned")),
    ("07_перенос_на_смежные", (
        "transfer_map", "transfer_cnn", "transfer_distil", "branch5_",
        "forecast_wide_map", "forecast_dense_wide_map", "supervised_wide",
        "honest_wide_coverage", "nn_geo_mpm_wide")),
    ("08_нейросети", ("nn_geo_mpm_map", "mae_")),
    ("09_минерагения", ("honest_mineragenic", "mineragenic_", "org_")),
    ("10_методы_из_литературы", ("lit6b_", "gm_gr_")),
    ("11_презентации", ("project_story", "deck_", "roadmap_")),
)

OTHER = "прочее"


def task_of(name: str) -> str:
    """Папка задачи по имени файла; `прочее`, если правила не сработали."""
    stem = Path(name).name
    for folder, prefixes in RULES:
        if any(stem.startswith(p) for p in prefixes):
            return folder
    return OTHER


def png(name: str, mkdir: bool = True) -> Path:
    """Путь к картинке в папке своей задачи."""
    path = OUT / task_of(name) / name
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
