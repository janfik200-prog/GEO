"""Признаковое пространство dataset_v6: корреляционная кластеризация 104
признаков-кандидатов (Spearman + иерархическая кластеризация, complete
linkage) — математическая основа презентации "признаковое пространство"
(``docs II presentation/``).

Почему Spearman: признаки разной физической природы (отражение, дБ, м,
безразмерные индексы) — ранговая корреляция не требует линейности и
одинаковых единиц.

Почему complete linkage, а не average: на average linkage признаки слипаются
в одну "мега-кучу" (62 из 104) — слабые фоновые корреляции между ВСЕМИ
съёмками (общий ареальный тренд освещённости/влажности) цепляют кластеры друг
за друга транзитивно. Complete linkage требует, чтобы ВСЕ пары между
кластерами были похожи, а не только ближайшая — даёт компактные, читаемые
кластеры (проверено: k=18..26 даёт стабильную структуру, крупные кластеры не
дробятся дальше).

Запуск из корня: ``python -X utf8 -m experiments.feature_space_clustering``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset_v6.parquet"
OUT_DIR = ROOT / "outputs"
METRICS_DIR = ROOT / "outputs" / "metrics"

ID_COLS = {"row", "col", "x", "y", "mask_svita"}
N_CLUSTERS = 20

GOLD = "#C89B2C"
DARK = "#2B2B2B"
GREY = "#999999"


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(DATASET)
    mask = df["mask_svita"].astype(bool)
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    return df.loc[mask, feat_cols]


def cluster(X: pd.DataFrame):
    corr = X.corr(method="spearman")
    dist = (1 - corr.abs()).to_numpy().copy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # numerical symmetry
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, N_CLUSTERS, criterion="maxclust")
    return corr, dist, Z, labels


def plot_dendrogram(Z, feat_cols, labels) -> None:
    order_colors = {}
    palette = plt.cm.tab20(np.linspace(0, 1, N_CLUSTERS))
    for i, lab in enumerate(sorted(set(labels))):
        order_colors[lab] = "#%02x%02x%02x" % tuple(int(c * 255) for c in palette[i][:3])

    fig, ax = plt.subplots(figsize=(15, 22))
    dendrogram(
        Z, labels=feat_cols, orientation="left", ax=ax,
        color_threshold=0, above_threshold_color=GREY,
    )
    # раскрасить подписи листьев по кластеру
    label_to_cluster = dict(zip(feat_cols, labels))
    for tick in ax.get_yticklabels():
        c = label_to_cluster.get(tick.get_text())
        if c is not None:
            tick.set_color(order_colors[c])
            tick.set_fontsize(8)
    ax.set_xlabel("расстояние = 1 − |корреляция Спирмена| (complete linkage)")
    ax.set_title(
        f"Признаковое пространство dataset_v6: иерархическая кластеризация "
        f"104 признаков ({N_CLUSTERS} кластеров, цвет подписи = кластер)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_space_dendrogram.png", dpi=130)
    plt.close(fig)


def plot_heatmap(corr, feat_cols, labels) -> None:
    order = np.argsort(labels)
    cols_sorted = [feat_cols[i] for i in order]
    corr_sorted = corr.loc[cols_sorted, cols_sorted]

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(corr_sorted.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols_sorted)))
    ax.set_yticks(range(len(cols_sorted)))
    ax.set_xticklabels(cols_sorted, rotation=90, fontsize=5)
    ax.set_yticklabels(cols_sorted, fontsize=5)

    # разделительные линии между кластерами
    sorted_labels = labels[order]
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color=DARK, lw=0.6)
        ax.axvline(b, color=DARK, lw=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("корреляция Спирмена")
    ax.set_title(
        "Корреляционная матрица 104 признаков dataset_v6\n"
        "(строки/столбцы переупорядочены по кластеру)", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_space_corr_heatmap.png", dpi=140)
    plt.close(fig)


def plot_heatmap_slide(corr, feat_cols, labels) -> None:
    """Та же матрица, но подписи по номеру кластера (20), а не по каждому
    из 104 признаков — 5pt подписи на полной версии физически нечитаемы на
    слайде презентации, что бы мы с размером картинки ни делали."""
    order = np.argsort(labels)
    cols_sorted = [feat_cols[i] for i in order]
    sorted_labels = labels[order]
    corr_sorted = corr.loc[cols_sorted, cols_sorted]

    ticks, ticklabels = [], []
    i = 0
    n = len(sorted_labels)
    while i < n:
        j = i
        while j < n and sorted_labels[j] == sorted_labels[i]:
            j += 1
        ticks.append((i + j - 1) / 2)
        ticklabels.append(str(sorted_labels[i]))
        i = j

    fig, ax = plt.subplots(figsize=(10, 8.6))
    im = ax.imshow(corr_sorted.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=11)
    ax.set_yticklabels(ticklabels, fontsize=11)
    ax.set_xlabel("№ кластера")
    ax.set_ylabel("№ кластера")

    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color=DARK, lw=0.6)
        ax.axvline(b, color=DARK, lw=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("корреляция Спирмена")
    ax.set_title(
        "Корреляционная матрица 104 признаков, сгруппированных\n"
        "в 20 кластеров (подписи блоков = № кластера)", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_space_corr_heatmap_slide.png", dpi=150)
    plt.close(fig)


CATEGORY_RULES = [
    (lambda f: f in {
        "geo2_contact", "geo2_facies2", "geo2_strike_cos", "geo2_strike_sin", "geo2_fault_node",
        "dist_facies", "dist_struct", "dist_tect1", "dist_tect2", "dist_magm", "dens_magm",
        "dist_paleo", "dens_tect",
    }, "циркулярные факторы формулы"),
    (lambda f: f.startswith("ast_") or f in {"s2_clay", "s2_ferrous", "s2_ferrous_std"},
     "гидротермальные изменения (ASTER/S2)"),
    (lambda f: f in {"s2_iron_ox", "s2_ferric", "s2_ferric_std", "l8_iron_ox", "l8_iron_ox_std"},
     "оксиды железа"),
    (lambda f: f.startswith("lin_") or f in {"dist_dnl", "dist_dnara"},
     "структурный контроль (линеаменты/разломы)"),
    (lambda f: f.startswith("dem_") or f.startswith("relief2_"),
     "рельеф и водосбор (россыпной контекст)"),
    (lambda f: f.startswith("pf_") or f.startswith("gm_"),
     "потенциальные поля (грав-магнитка)"),
    (lambda f: f.startswith("s1_") or f.startswith("psr_"),
     "радар (текстура/влажность)"),
    (lambda f: True, "сырые оптические каналы"),
]


def categorize(feat: str) -> str:
    for rule, cat in CATEGORY_RULES:
        if rule(feat):
            return cat
    return "прочее"


def plot_feature_map(dist, feat_cols, labels) -> None:
    """2D карта признакового пространства (многомерное шкалирование по матрице
    расстояний 1 - |корреляция|) — визуально "где" находится каждый признак
    относительно других, а не просто список/дендрограмма. Цвет = категория по
    литературной привязке к золоту (структурный контроль, гидротермальные
    изменения, циркулярные факторы формулы и т.д.), звёздочка = признак,
    прошедший перестановочную проверку значимости на dataset_v6
    (experiments/feature_relevance_ast_v6.py) — подтверждён статистически,
    а не только по литературе."""
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
              normalized_stress="auto", n_init=4)
    coords = mds.fit_transform(dist)

    imp_path = METRICS_DIR / "feature_relevance_ast_v6.csv"
    confirmed = set()
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        confirmed = set(imp.loc[imp["mean_importance"] > 0, "feature"])

    cats = [categorize(f) for f in feat_cols]
    uniq_cats = sorted(set(cats))
    palette = plt.cm.tab10(np.linspace(0, 1, len(uniq_cats)))
    cat_color = dict(zip(uniq_cats, palette))

    fig, ax = plt.subplots(figsize=(15, 6.5))
    for cat in uniq_cats:
        idx = [i for i, c in enumerate(cats) if c == cat]
        xs = coords[idx, 0]
        ys = coords[idx, 1]
        star = [i for i in idx if feat_cols[i] in confirmed]
        plain = [i for i in idx if feat_cols[i] not in confirmed]
        ax.scatter(coords[plain, 0], coords[plain, 1], color=cat_color[cat], s=55,
                   alpha=0.85, label=cat, edgecolors="none")
        if star:
            ax.scatter(coords[star, 0], coords[star, 1], color=cat_color[cat], s=140,
                       alpha=0.95, marker="*", edgecolors=DARK, linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("координата 1 (условная — только взаимные расстояния имеют смысл)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, title="категория (литературная привязка к золоту)")
    ax.set_title(
        "Признаковое пространство: карта 104 признаков по расстоянию 1 − |корреляция Спирмена|\n"
        "(метод многомерного шкалирования); ★ — признак прошёл перестановочную проверку значимости\n"
        "на dataset_v6, подтверждён статистически, не только по литературе",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_space_map.png", dpi=150)
    plt.close(fig)


def plot_cluster_meta_dendrogram(X: pd.DataFrame, labels) -> None:
    """Дендрограмма НЕ по 104 признакам (там подписи нечитаемы на слайде
    в принципе — некуда девать 104 строки), а по 20 кластерам-агрегатам:
    для каждого кластера берём среднее по z-score его признаков ("кластерный
    счёт" на ячейку), коррелируем эти 20 счётов между собой Спирменом и
    строим ту же complete-linkage дендрограмму, но уже читаемого размера."""
    ranks = X.rank()
    z = (ranks - ranks.mean()) / ranks.std()
    sizes = pd.Series(labels).value_counts()
    cluster_ids = sorted(set(labels))
    scores = pd.DataFrame({
        c: z[[f for f, lab in zip(X.columns, labels) if lab == c]].mean(axis=1)
        for c in cluster_ids
    })
    corr_meta = scores.corr(method="spearman")
    dist = (1 - corr_meta.abs()).to_numpy().copy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    Zc = linkage(squareform(dist, checks=False), method="complete")
    leaf_labels = [f"кластер {c} ({sizes[c]} призн.)" for c in corr_meta.columns]

    fig, ax = plt.subplots(figsize=(8.5, 8))
    dendrogram(Zc, labels=leaf_labels, orientation="left", ax=ax,
               color_threshold=0, above_threshold_color=DARK)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(11)
    ax.set_xlabel("расстояние = 1 − |корреляция Спирмена| между кластерными счётами")
    ax.set_title("Как 20 кластеров признаков соотносятся друг с другом", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_space_dendrogram_slide.png", dpi=150)
    plt.close(fig)


def main() -> None:
    X = load_features()
    print("признаков:", X.shape[1], "| строк (в маске листа):", X.shape[0])

    corr, dist, Z, labels = cluster(X)
    feat_cols = list(X.columns)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / "feature_clusters_v6.csv"
    pd.DataFrame({"feature": feat_cols, "cluster": labels}).sort_values(
        ["cluster", "feature"]
    ).to_csv(out_csv, index=False)
    print("сохранено:", out_csv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dendrogram(Z, feat_cols, labels)
    plot_heatmap(corr, feat_cols, labels)
    plot_heatmap_slide(corr, feat_cols, labels)
    plot_cluster_meta_dendrogram(X, labels)
    plot_feature_map(dist, feat_cols, labels)
    print("сохранено:", OUT_DIR / "feature_space_dendrogram.png")
    print("сохранено:", OUT_DIR / "feature_space_corr_heatmap.png")
    print("сохранено:", OUT_DIR / "feature_space_corr_heatmap_slide.png")
    print("сохранено:", OUT_DIR / "feature_space_dendrogram_slide.png")
    print("сохранено:", OUT_DIR / "feature_space_map.png")

    clu = pd.DataFrame({"feature": feat_cols, "cluster": labels})
    for c in sorted(clu["cluster"].unique()):
        members = clu[clu["cluster"] == c]["feature"].tolist()
        print(f"\n--- кластер {c} ({len(members)}) ---")
        print(members)


if __name__ == "__main__":
    main()
