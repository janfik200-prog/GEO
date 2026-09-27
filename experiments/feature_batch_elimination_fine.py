"""Продолжение пакетного обратного исключения признаков ниже 11 (см. `feature_batch_elimination.py`).

Первый прогон (пачки по 10, `dataset_v6`) остановился на 11 признаках искусственно
(`STOP_AT=11`) — честная метрика (Spearman, leave-one-strip-out) там ещё не начала
падать, а росла. Здесь продолжаем с того же 11-признакового пула более мелкими
пачками (по 2), чтобы найти точку, где сокращение реально начинает вредить.

Раунды нумеруются продолжением предыдущего прогона (раунд 9 = 11 признаков — общая
точка стыковки), результат сохраняется отдельно и объединяется с первым прогоном
только на графике.

Запуск из корня репозитория: ``python -m experiments.feature_batch_elimination_fine``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.feature_batch_elimination import honest_round  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import numpy as np  # noqa: E402

BATCH = 2
STOP_AT = 1
START_ROUND = 9  # раунд 9 = 11 признаков в первом прогоне (стыковка)

# 11 признаков, выживших в первом прогоне (outputs/metrics/feature_batch_elimination_dropped.csv)
START_POOL = [
    "ast_alter", "ast_b04", "dist_dnl", "gm_gr_1GFI_25", "gm_gr_1G_25",
    "l8_iron_ox", "lin_dist", "ls_ch6", "relief2_curv_5km", "s2_b11", "s2_iron_ox",
]


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = -prognoz.ravel().astype(float)
    feat_df = criterial_target.training_features(dataset="v6")

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    pool = list(START_POOL)
    round_rows = []
    dropped_rows = []
    round_no = START_ROUND

    while True:
        round_no += 1
        n_feat = len(pool)
        print(f"\n=== Раунд {round_no}: {n_feat} признаков ===", flush=True)
        X = feat_df[pool].fillna(0).to_numpy()

        imp_x, pred_x, cov_x = honest_round("X", cx, X, y, pool, config.CRIT_SEED)
        imp_y, pred_y, cov_y = honest_round("Y", cy, X, y, pool, config.CRIT_SEED)
        imp_all = pd.concat([imp_x, imp_y], ignore_index=True)

        sp_x = spearmanr(y[cov_x], pred_x[cov_x]).statistic
        sp_y = spearmanr(y[cov_y], pred_y[cov_y]).statistic
        sp_mean = float(np.mean([sp_x, sp_y]))
        print(f"Раунд {round_no}: n_features={n_feat}  spearman_X={sp_x:.4f}  "
              f"spearman_Y={sp_y:.4f}  mean={sp_mean:.4f}", flush=True)
        round_rows.append({
            "round": round_no, "n_features": n_feat,
            "spearman_x": sp_x, "spearman_y": sp_y, "spearman_mean": sp_mean,
            "features": ",".join(pool),
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
    rounds_path = out_dir / "feature_batch_elimination_fine_rounds.csv"
    dropped_path = out_dir / "feature_batch_elimination_fine_dropped.csv"
    rounds_df.to_csv(rounds_path, index=False)
    dropped_df.to_csv(dropped_path, index=False)
    print(f"\nСохранено: {rounds_path}")
    print(f"Сохранено: {dropped_path}")

    # объединённый график: первый прогон (91->11) + продолжение (11->1)
    coarse_path = out_dir / "feature_batch_elimination_rounds.csv"
    coarse_df = pd.read_csv(coarse_path) if coarse_path.exists() else pd.DataFrame()
    combined = pd.concat([coarse_df, rounds_df], ignore_index=True).drop_duplicates(
        subset="n_features", keep="last").sort_values("n_features", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(combined["n_features"], combined["spearman_x"], marker="o", label="Spearman, полосы по X")
    ax.plot(combined["n_features"], combined["spearman_y"], marker="o", label="Spearman, полосы по Y")
    ax.plot(combined["n_features"], combined["spearman_mean"], marker="o", linewidth=2.5,
            color="black", label="среднее X/Y")
    ax.axvline(11, color="gray", linestyle="--", linewidth=1, label="стык прогонов (11 признаков)")
    ax.invert_xaxis()
    ax.set_xlabel("число признаков в пуле")
    ax.set_ylabel("честная корреляция Спирмена (leave-one-strip-out)")
    ax.set_title("Пакетное обратное исключение признаков, dataset_v6 (полная кривая)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = ROOT / "outputs" / "feature_batch_elimination_full.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Сохранено: {plot_path}")

    print("\nОбъединённая таблица по раундам:")
    print(combined.drop(columns=["features"], errors="ignore").to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
