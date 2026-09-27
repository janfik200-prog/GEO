"""Точное однопризнаковое обратное исключение в диапазоне 21->11 признаков.

Пакетный прогон (`feature_batch_elimination.py`, шаг 10) показал плато честной
метрики (Spearman, leave-one-strip-out) между 21 и 51 признаками и заметный скачок
на 11 (см. `feature_batch_elimination_full.png`). Шаг в 10 признаков слишком грубый,
чтобы понять, где именно внутри 21->11 проходит реальная граница пользы — здесь
убираем строго по одному худшему признаку за раунд, стартуя с того же 21-признакового
пула, что выжил в раунде 7 пакетного прогона (`outputs/metrics/feature_batch_elimination_dropped.csv`).

Запуск из корня репозитория: ``python -m experiments.feature_elimination_precise_21_to_11``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target  # noqa: E402
from experiments.feature_batch_elimination import honest_round  # noqa: E402

STOP_AT = 10  # на 1 признак ниже уже честно посчитано в feature_batch_elimination_fine

START_POOL = [
    "ast_alter", "ast_b02_std", "ast_b04", "dist_dnara", "dist_dnl",
    "gm_gr_1GFI_25", "gm_gr_1G_25", "l8_iron_ox", "l8_swir16", "lin_dist",
    "ls_ch1", "ls_ch4", "ls_ch6", "psr_hh_hv", "relief2_curv_5km", "s1_vv",
    "s2_b04", "s2_b11", "s2_b12", "s2_iron_ox", "s2_iron_ox_std",
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
    round_no = 0

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
            "n_features": n_feat, "spearman_x": sp_x, "spearman_y": sp_y,
            "spearman_mean": sp_mean, "features": ",".join(pool),
        })

        summary = imp_all.groupby("feature")["importance"].mean().reset_index(
            name="mean_importance").sort_values("mean_importance")

        if n_feat <= STOP_AT:
            print(f"Пул <= {STOP_AT} признаков — останов.", flush=True)
            break

        worst = summary.iloc[0]
        print(f"Убираем: {worst['feature']} (importance={worst['mean_importance']:.4f})", flush=True)
        dropped_rows.append({"n_features_before": n_feat, "feature": worst["feature"],
                              "mean_importance": worst["mean_importance"]})
        pool = [c for c in pool if c != worst["feature"]]

    out_dir = ROOT / "outputs" / "metrics"
    rounds_df = pd.DataFrame(round_rows)
    dropped_df = pd.DataFrame(dropped_rows)
    rounds_path = out_dir / "feature_elimination_precise_21_to_11_rounds.csv"
    dropped_path = out_dir / "feature_elimination_precise_21_to_11_dropped.csv"
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
    ax.set_title("Точное однопризнаковое исключение, 21 -> 10 признаков")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = ROOT / "outputs" / "feature_elimination_precise_21_to_11.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Сохранено: {plot_path}")

    print("\nТаблица по раундам:")
    print(rounds_df.drop(columns=["features"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
