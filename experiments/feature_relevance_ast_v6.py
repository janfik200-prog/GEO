"""Честная permutation importance для группы `ast` (ASTER) на dataset_v6.

Отдельный прогон от `feature_relevance_check.py`: тот целиком выбрасывает
группу `ast` через `DROP_GROUPS=("ast","ls")` — причина не слабый сигнал, а
совместимость `forecast_dense.py` с `dataset_wide.parquet` (см. докстринг
`forecast_dense.py`, п. «Признаки»). На основном листе (без нужды в широкой
совместимости) это ограничение снимать не нужно — здесь его и снимаем, чтобы
впервые получить честную importance для `ast_aloh` (серицит-мусковит),
`ast_kaolin`, `ast_alter`, `ast_carb`, `ast_ferric`, `ast_ferrous` и их `_std`.
`ls` по-прежнему исключён (не про ASTER, отдельная причина — легаси Landsat 7).

Протокол — тот же leave-one-strip-out (4 полосы по X + 4 по Y, буфер 15 км),
что и в `feature_relevance_check.py`; используется `dataset_v6.parquet`
(актуальная чистка) вместо `dataset_v5_rebuilt`.

Запуск из корня репозитория: ``python -m experiments.feature_relevance_ast_v6``.
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

DROP_GROUPS = ("ls",)  # ast НЕ исключается — это и есть цель прогона
N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0


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

    feat_df = criterial_target.training_features(dataset="v6")
    feat_names = [c for c in feat_df.columns
                  if features_v2.feature_group(c) not in DROP_GROUPS and not cell_mask.is_service(c)]
    ast_names = [c for c in feat_names if c.startswith("ast")]
    print(f"Признаков всего: {len(feat_names)}, из них ast_*: {len(ast_names)} -> {ast_names}")
    X_df = feat_df[feat_names].fillna(0)
    X = X_df.to_numpy()

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    print("\n=== Honest permutation importance (RF, leave-one-strip-out, X+Y, 8 фолдов) ===")
    imp_x = honest_importance("X", cx, X, y, feat_names, config.CRIT_SEED)
    imp_y = honest_importance("Y", cy, X, y, feat_names, config.CRIT_SEED)
    imp_all = pd.concat([imp_x, imp_y], ignore_index=True)

    summary = imp_all.groupby("feature")["importance"].agg(
        mean_importance="mean", frac_folds_nonpositive=lambda s: float((s <= 0).mean())
    ).reset_index().sort_values("mean_importance", ascending=False)

    out_path = ROOT / "outputs" / "metrics" / "feature_relevance_ast_v6.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path} ({len(summary)} признаков)")

    print("\nТоп-15 самых полезных (по честному permutation importance, усреднено по 8 фолдам):")
    print(summary.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\n--- Только ast_* ({len(ast_names)}) ---")
    ast_summary = summary[summary["feature"].isin(ast_names)].sort_values("mean_importance", ascending=False)
    print(ast_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nсумма importance по группе ast: {ast_summary['mean_importance'].sum():.4f}")


if __name__ == "__main__":
    main()
