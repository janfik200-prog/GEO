"""Честная проверка полезности и избыточности признаков `dataset_v5_rebuilt` (138 шт).

Запускается параллельно с `experiments.forecast_dense`. Использует ту же честную
схему leave-one-strip-out (4 полосы по X и отдельно по Y, буфер 15 км между
train/test — см. `forecast_dense.py`), но с ОДНИМ `RandomForestRegressor` на фолд
(не полный 16-моделный `SimpleRegEnsemble`), чтобы:
1. не конкурировать по CPU с основным прогоном (который уже занят полным
   ансамблем на тех же данных);
2. получить permutation importance по ВСЕМ 138 признакам на ВСЕХ 8 честных
   фолдах (4 по X + 4 по Y), а не только топ-10 на одной полосе, как сейчас
   печатает `forecast_dense.leave_one_strip_out`.

Отдельно — корреляционная проверка (|r|>0.95, target-free, без завязки на
prognoz): ищем остаточные почти-дубли внутри уже пересобранного набора (138).
Если чистка (`experiments/criterial_target_v5_rebuilt.py`) была верной, таких
пар быть не должно — если найдутся, это повод разобраться, не ожидаемый
результат.

Итог: `outputs/metrics/feature_relevance_v5_rebuilt.csv` — на признак:
mean permutation importance (усреднено по 8 честным фолдам), доля фолдов с
importance<=0 ("шум по большинству голосов"), ближайший коррелированный сосед
(|r| максимальный среди остальных 137) и его значение.

Запуск из корня репозитория: ``python -m experiments.feature_relevance_check``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target, features_v2  # noqa: E402

DROP_GROUPS = ("ast", "ls")
N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0
CORR_THRESHOLD = 0.95


def strip_id_and_edges(coord: np.ndarray, n_strips: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(coord.min(), coord.max(), n_strips + 1)
    strip_id = np.clip(np.digitize(coord, edges[1:-1]), 0, n_strips - 1)
    return strip_id, edges


def honest_importance(axis_name: str, coord: np.ndarray, X: np.ndarray, y: np.ndarray,
                       feat_names: list[str], seed: int) -> pd.DataFrame:
    strip_id, edges = strip_id_and_edges(coord, N_STRIPS)
    records = []
    for s in range(N_STRIPS):
        lo, hi = edges[s], edges[s + 1]
        test_mask = strip_id == s
        train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
        if test_mask.sum() == 0 or train_mask.sum() < 50:
            continue
        model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            random_state=seed, n_jobs=-1,
        ).fit(X[train_mask], y[train_mask])
        perm = permutation_importance(model, X[test_mask], y[test_mask], n_repeats=5,
                                       random_state=seed, scoring="r2")
        print(f"  [{axis_name}] полоса {s}: готово (test={test_mask.sum()}, train={train_mask.sum()})", flush=True)
        for i, name in enumerate(feat_names):
            records.append((f"{axis_name}{s}", name, perm.importances_mean[i]))
    return pd.DataFrame(records, columns=["fold", "feature", "importance"])


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = -prognoz.ravel().astype(float)

    feat_df = criterial_target.training_features(dataset="v5_rebuilt")
    feat_names = [c for c in feat_df.columns
                  if features_v2.feature_group(c) not in DROP_GROUPS and not cell_mask.is_service(c)]
    print(f"Признаков: {len(feat_names)}")
    X_df = feat_df[feat_names].fillna(0)
    X = X_df.to_numpy()

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    # --- 1. Корреляционная проверка остаточных дублей ---
    print("\n=== Корреляция между признаками (target-free), поиск |r|>0.95 ===")
    corr = X_df.corr().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    pairs = []
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):
            r = corr[i, j]
            if abs(r) > CORR_THRESHOLD:
                pairs.append((feat_names[i], feat_names[j], r))
    if pairs:
        pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "r"]).sort_values(
            "r", key=lambda s: s.abs(), ascending=False)
        print(f"Найдено {len(pairs_df)} пар с |r|>{CORR_THRESHOLD} (не должно было остаться после чистки):")
        print(pairs_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print(f"Пар с |r|>{CORR_THRESHOLD} не найдено — остаточных дублей нет, чистка v5_rebuilt подтверждена.")

    nearest_r = np.abs(corr).max(axis=1)
    nearest_partner = [feat_names[j] for j in np.abs(corr).argmax(axis=1)]

    # --- 2. Честная permutation importance по всем 8 фолдам ---
    print("\n=== Honest permutation importance (RF, leave-one-strip-out, X+Y, 8 фолдов) ===")
    imp_x = honest_importance("X", cx, X, y, feat_names, config.CRIT_SEED)
    imp_y = honest_importance("Y", cy, X, y, feat_names, config.CRIT_SEED)
    imp_all = pd.concat([imp_x, imp_y], ignore_index=True)

    summary = imp_all.groupby("feature")["importance"].agg(
        mean_importance="mean", frac_folds_nonpositive=lambda s: float((s <= 0).mean())
    ).reset_index()
    summary["nearest_corr_partner"] = summary["feature"].map(dict(zip(feat_names, nearest_partner)))
    summary["nearest_corr_r"] = summary["feature"].map(dict(zip(feat_names, nearest_r)))
    summary = summary.sort_values("mean_importance", ascending=False)

    out_path = ROOT / "outputs" / "metrics" / "feature_relevance_v5_rebuilt.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path} ({len(summary)} признаков)")

    print("\nТоп-15 самых полезных (по честному permutation importance, усреднено по 8 фолдам):")
    print(summary.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    noise_candidates = summary[summary["frac_folds_nonpositive"] >= 0.75].sort_values("mean_importance")
    print(f"\nКандидаты в 'шум' (importance<=0 минимум в 6 из 8 фолдов): {len(noise_candidates)} признаков")
    if len(noise_candidates) > 0:
        print(noise_candidates.head(30).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
