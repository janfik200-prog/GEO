"""Корреляционная кластеризация ровно того 91-признакового пула, которым
пользуются эксперименты backward elimination (`criterial_target.training_features`
минус служебные, см. `feature_batch_elimination.py`) — в отличие от
`feature_space_clustering.py`, который кластеризует более широкий "сырой" пул
104 признаков-кандидатов и не привязан к elimination-результатам напрямую.

Метод (независимый от целевой переменной): иерархическая кластеризация по
матрице Spearman-корреляций, complete linkage, дистанция = 1 - |corr|. Отвечает
на другой вопрос, чем backward elimination: не "что важно для прогноза", а
"какие признаки взаимозаменяемы" — RF/ElasticNet могли выбрасывать признак не
потому что он бесполезен, а потому что его роль дублирует оставшийся сосед по
кластеру.

Отдельно отмечаем на дендрограмме и в CSV: (a) 13 признаков, выживших в точном
однопризнаковом прогоне (`feature_elimination_precise_21_to_11_dropped.csv`,
пул после последнего дропа перед остановом STOP_AT=10 — фактически то, что
осталось на границе 13->12, где произошёл обрыв метрики), (b) признак,
удаление которого вызвало обрыв (`gm_gr_1G_25`).

Запуск из корня репозитория: ``python -m experiments.correlation_clustering_91``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, criterial_target  # noqa: E402

N_CLUSTERS = 20

# 13 признаков, выживших в outputs/metrics/feature_elimination_precise_21_to_11_rounds.csv
# на строке n_features=13 (последний раунд до обрыва метрики 0.354 -> 0.283)
SURVIVORS_13 = {
    "ast_alter", "ast_b04", "dist_dnl", "gm_gr_1GFI_25", "gm_gr_1G_25",
    "l8_iron_ox", "lin_dist", "ls_ch6", "s2_b04", "s2_b11", "s2_b12",
    "s2_iron_ox", "s1_vv",
}
# признак, чьё удаление (13 -> 12) вызвало обрыв честной метрики
CLIFF_FEATURE = "gm_gr_1G_25"

DARK = "#2B2B2B"
GREY = "#999999"
SURVIVOR_COLOR = "#c23c6b"
CLIFF_COLOR = "#1c8fd6"


def load_pool() -> pd.DataFrame:
    feat_df = criterial_target.training_features(dataset="v6")
    pool = [c for c in feat_df.columns if not cell_mask.is_service(c)]
    return feat_df[pool].fillna(0)


def cluster(X: pd.DataFrame):
    corr = X.corr(method="spearman")
    dist = (1 - corr.abs()).to_numpy().copy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, N_CLUSTERS, criterion="maxclust")
    return corr, dist, Z, labels


def plot_dendrogram(Z, feat_cols) -> None:
    fig, ax = plt.subplots(figsize=(11, 16))
    dendrogram(Z, labels=feat_cols, orientation="left", ax=ax,
               color_threshold=0, above_threshold_color=GREY)
    for tick in ax.get_yticklabels():
        name = tick.get_text()
        if name == CLIFF_FEATURE:
            tick.set_color(CLIFF_COLOR)
            tick.set_fontweight("bold")
            tick.set_fontsize(9)
        elif name in SURVIVORS_13:
            tick.set_color(SURVIVOR_COLOR)
            tick.set_fontweight("bold")
            tick.set_fontsize(9)
        else:
            tick.set_fontsize(7)
    ax.set_xlabel("расстояние = 1 - |корреляция Спирмена| (complete linkage)")
    ax.set_title(
        "Корреляционная кластеризация 91-признакового пула (dataset_v6, elimination-пул)\n"
        "розовый жирный = выжил в точном прогоне (13 признаков), синий жирный = обрыв метрики (gm_gr_1G_25)",
        fontsize=11,
    )
    fig.tight_layout()
    out = ROOT / "outputs" / "correlation_clustering_91_dendrogram.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("сохранено:", out)


def main() -> None:
    X = load_pool()
    feat_cols = list(X.columns)
    print("признаков в пуле:", len(feat_cols))

    corr, dist, Z, labels = cluster(X)

    out_dir = ROOT / "outputs" / "metrics"
    clu = pd.DataFrame({"feature": feat_cols, "cluster": labels})
    clu["is_survivor_13"] = clu["feature"].isin(SURVIVORS_13)
    clu["is_cliff_feature"] = clu["feature"] == CLIFF_FEATURE
    clu = clu.sort_values(["cluster", "feature"])
    out_csv = out_dir / "correlation_clusters_91.csv"
    clu.to_csv(out_csv, index=False)
    print("сохранено:", out_csv)

    plot_dendrogram(Z, feat_cols)

    # какие кластеры представлены выжившими 13 признаками
    survivor_clusters = clu.loc[clu["is_survivor_13"], "cluster"].unique()
    print(f"\n13 выживших признаков покрывают {len(survivor_clusters)} из {N_CLUSTERS} кластеров:")
    for c in sorted(survivor_clusters):
        members = clu.loc[clu["cluster"] == c, "feature"].tolist()
        survivors_in_c = [f for f in members if f in SURVIVORS_13]
        print(f"  кластер {c} ({len(members)} признаков всего): выжившие -> {survivors_in_c}; "
              f"остальные в кластере -> {[f for f in members if f not in SURVIVORS_13]}")

    # признаки, вообще не представленные ни одним выжившим соседом по кластеру
    covered_clusters = set(survivor_clusters)
    all_clusters = set(clu["cluster"].unique())
    uncovered = sorted(all_clusters - covered_clusters)
    print(f"\nКластеры БЕЗ единого выжившего признака ({len(uncovered)} из {N_CLUSTERS}):")
    for c in uncovered:
        members = clu.loc[clu["cluster"] == c, "feature"].tolist()
        print(f"  кластер {c}: {members}")


if __name__ == "__main__":
    main()
