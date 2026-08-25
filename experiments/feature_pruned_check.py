"""Проверка: помогает ли честной генерализации удаление вредных/шумовых признаков.

По итогам `experiments/feature_relevance_check.py` (permutation importance на
8 честных leave-one-strip-out фолдах) у 33 из 95 признаков средняя важность
<=0 — среди них не просто "шум", а признаки с явно ОТРИЦАТЕЛЬНЫМ вкладом
(`pf_gr_up1`, `gm_mag_flt_35`, `psr_linci` и т.д.), причём почти все они
совпадают с парами, найденными в корреляционной проверке (|r|>0.9) —
гипотеза: почти-дубли увеличивают дисперсию RF/GB и портят перенос на
незнакомую полосу.

Этот скрипт гоняет ТОТ ЖЕ протокол, что и `forecast_dense.leave_one_strip_out`
(дерево-ансамбль SimpleRegEnsemble, честный буфер 15 км), дважды: на полном
пуле (95 признаков) и на урезанном (95 минус 33 = 62). Сравнивается сшитый
Spearman по всем честным полосам (X и Y) — если урезанный пул даёт заметно
лучший (не худший) результат, это подтверждает гипотезу и даёт готовый список
признаков на удаление из канонического пула.

Запуск из корня репозитория: ``python -m experiments.feature_pruned_check``.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target, features_v2  # noqa: E402
from experiments.forecast_dense import DROP_GROUPS, leave_one_strip_out  # noqa: E402

NOISE_FEATURES = [
    "pf_gr_up1", "gm_mag_flt_35", "psr_linci", "pf_mag_up10", "gm_gr_1GY_25",
    "l8_lst", "s2_b8a", "l8_ferrous", "pf_mag_up5", "pf_mag_up2", "pf_gr_asig",
    "gm_mag_1GFI_25", "pf_gr_vz", "gm_mg_all", "pf_mag_up1", "s2_ferrous",
    "pf_gr_tdr", "s2_bare_frac", "s2_ireci", "pf_mag_tdr", "pf_mag_vz",
    "l8_lst_std", "pf_gm_corr", "l8_iron_ox_std", "psr_hv_std", "s2_b05_std",
    "psr_hh_hv_std", "psr_linci_std", "s2_b03", "l8_swir16_std", "s2_b04_std",
    "s2_b12_std", "gm_mag_2G_25",
]


def run_pool(label: str, feat_names: list[str], meta, xy, cx, cy, y, perspective, seed: int) -> None:
    feat_df = criterial_target.training_features(dataset="v5_rebuilt")
    X = feat_df[feat_names].fillna(0).to_numpy()
    print(f"\n########## Пул '{label}': {len(feat_names)} признаков ##########")
    stitched_x, _, _ = leave_one_strip_out("X", cx, xy, X, y, perspective, feat_names, seed)
    stitched_y, _, _ = leave_one_strip_out("Y", cy, xy, X, y, perspective, feat_names, seed)
    for axis_name, stitched in [("X", stitched_x), ("Y", stitched_y)]:
        valid = ~np.isnan(stitched)
        rho, _ = spearmanr(stitched[valid], y[valid])
        r2 = r2_score(y[valid], stitched[valid])
        print(f"[{label}] сшитый честный Spearman по оси {axis_name}: {rho:.3f}, R2={r2:.3f}")


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    prognoz_flat = prognoz.ravel().astype(float)
    y = -prognoz_flat
    perspective = prognoz_flat <= config.CRIT_TARGET_THRESHOLD

    feat_df = criterial_target.training_features(dataset="v5_rebuilt")
    full_names = [c for c in feat_df.columns
                  if features_v2.feature_group(c) not in DROP_GROUPS and not cell_mask.is_service(c)]
    pruned_names = [c for c in full_names if c not in NOISE_FEATURES]
    print(f"Полный пул: {len(full_names)}, урезанный: {len(pruned_names)} "
          f"(удалено {len(full_names) - len(pruned_names)})")

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    xy = np.column_stack([cx, cy])

    run_pool("полный (95)", full_names, meta, xy, cx, cy, y, perspective, config.CRIT_SEED)
    run_pool("урезанный (62)", pruned_names, meta, xy, cx, cy, y, perspective, config.CRIT_SEED)


if __name__ == "__main__":
    main()
