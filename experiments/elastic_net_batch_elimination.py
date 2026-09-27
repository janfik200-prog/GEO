"""Пакетное обратное исключение признаков той же схемой, что в
`feature_batch_elimination.py` (RF + честная permutation importance,
leave-one-strip-out), но моделью-кандидатом на роль ранжировщика служит
ElasticNetCV, а не Random Forest — линейная модель другого семейства.

Зачем: RF-прогон нашёл резкий обрыв честной метрики между 13 и 12 признаками
(`feature_elimination_precise_21_to_11.py`). Вопрос — это свойство данных или
артефакт конкретной модели (RF нелинеен, ловит другие взаимодействия, чем
линейная регуляризация)? Если ElasticNet на тех же полосах находит похожий
топ-13 и похожий обрыв — граница подтверждена независимо от модели; если нет —
граница модель-зависима, и об этом важно знать раньше, чем закладывать её в
финальный пайплайн.

Признаки перед ElasticNet стандартизируются (z-score по train-фолду) — в
отличие от RF, линейная регуляризация чувствительна к масштабу.

Запуск из корня репозитория: ``python -m experiments.elastic_net_batch_elimination``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.feature_batch_elimination import strip_id_and_edges, N_STRIPS, STRIP_BUFFER_M  # noqa: E402

BATCH = 10
STOP_AT = 11


def honest_round_enet(axis_name: str, coord: np.ndarray, X: np.ndarray, y: np.ndarray,
                       feat_names: list[str], seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    strip_id, edges = strip_id_and_edges(coord, N_STRIPS)
    records = []
    pred = np.full(len(y), np.nan)
    covered = np.zeros(len(y), dtype=bool)
    for s in range(N_STRIPS):
        lo, hi = edges[s], edges[s + 1]
        test_mask = strip_id == s
        train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
        if test_mask.sum() == 0 or train_mask.sum() < 50:
            continue
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_mask])
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            cv=5, random_state=seed, n_jobs=-1, max_iter=10000,
        ).fit(Xtr, y[train_mask])
        pred[test_mask] = model.predict(Xte)
        covered[test_mask] = True
        perm = permutation_importance(model, Xte, y[test_mask], n_repeats=5,
                                       random_state=seed, scoring="r2")
        for i, name in enumerate(feat_names):
            records.append((f"{axis_name}{s}", name, perm.importances_mean[i]))
        print(f"    [{axis_name}] полоса {s}: готово (test={test_mask.sum()}, train={train_mask.sum()}, "
              f"alpha={model.alpha_:.4g}, l1_ratio={model.l1_ratio_:.2f})", flush=True)
    return pd.DataFrame(records, columns=["fold", "feature", "importance"]), pred, covered


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = -prognoz.ravel().astype(float)

    feat_df = criterial_target.training_features(dataset="v6")
    pool = [c for c in feat_df.columns if not cell_mask.is_service(c)]
    print(f"Стартовый пул признаков dataset_v6: {len(pool)}")

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    round_rows = []
    dropped_rows = []
    round_no = 0

    while True:
        round_no += 1
        n_feat = len(pool)
        print(f"\n=== Раунд {round_no}: {n_feat} признаков ===", flush=True)
        X = feat_df[pool].fillna(0).to_numpy()

        imp_x, pred_x, cov_x = honest_round_enet("X", cx, X, y, pool, config.CRIT_SEED)
        imp_y, pred_y, cov_y = honest_round_enet("Y", cy, X, y, pool, config.CRIT_SEED)
        imp_all = pd.concat([imp_x, imp_y], ignore_index=True)

        sp_x = spearmanr(y[cov_x], pred_x[cov_x]).statistic
        sp_y = spearmanr(y[cov_y], pred_y[cov_y]).statistic
        sp_mean = float(np.mean([sp_x, sp_y]))
        print(f"Раунд {round_no}: n_features={n_feat}  spearman_X={sp_x:.4f}  "
              f"spearman_Y={sp_y:.4f}  mean={sp_mean:.4f}", flush=True)
        round_rows.append({
            "round": round_no, "n_features": n_feat,
            "spearman_x": sp_x, "spearman_y": sp_y, "spearman_mean": sp_mean,
        })

        summary = imp_all.groupby("feature")["importance"].mean().reset_index(
            name="mean_importance").sort_values("mean_importance")

        if n_feat <= STOP_AT:
            print(f"Пул <= {STOP_AT} признаков — останов.", flush=True)
            break

        drop_n = min(BATCH, n_feat - STOP_AT)
        to_drop = summary.head(drop_n)
        print(f"Убираем {drop_n} худших: {list(to_drop['feature'])}", flush=True)
        for _, row in to_drop.iterrows():
            dropped_rows.append({"round": round_no, "feature": row["feature"],
                                  "mean_importance": row["mean_importance"]})
        pool = [c for c in pool if c not in set(to_drop["feature"])]

    out_dir = ROOT / "outputs" / "metrics"
    rounds_df = pd.DataFrame(round_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    rounds_path = out_dir / "elastic_net_batch_elimination_rounds.csv"
    dropped_path = out_dir / "elastic_net_batch_elimination_dropped.csv"
    rounds_df.to_csv(rounds_path, index=False)
    dropped_df.to_csv(dropped_path, index=False)
    print(f"\nСохранено: {rounds_path}")
    print(f"Сохранено: {dropped_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds_df["n_features"], rounds_df["spearman_x"], marker="o", label="Spearman, полосы по X")
    ax.plot(rounds_df["n_features"], rounds_df["spearman_y"], marker="o", label="Spearman, полосы по Y")
    ax.plot(rounds_df["n_features"], rounds_df["spearman_mean"], marker="o", linewidth=2.5,
            color="black", label="среднее X/Y")
    ax.invert_xaxis()
    ax.set_xlabel("число признаков в пуле")
    ax.set_ylabel("честная корреляция Спирмена (leave-one-strip-out)")
    ax.set_title("Пакетное обратное исключение признаков, ElasticNetCV, dataset_v6")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = ROOT / "outputs" / "elastic_net_batch_elimination.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Сохранено: {plot_path}")

    print("\nИтоговая таблица по раундам:")
    print(rounds_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
