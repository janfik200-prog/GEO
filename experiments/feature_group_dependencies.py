"""Зависимости между группами признаков — для порядка подачи в каскад.

Зачем: пользователь предложил каскадную архитектуру (последовательная подача
групп признаков — геология-прокси -> геофизика -> радар -> оптика — с
attention/memory между стадиями, см. обсуждение 20-27.08.2026). Раньше порядок
стадий был предложен только по физическому смыслу (глубина проникновения
метода: структура -> потенциальные поля -> радар -> спектр). Здесь —
эмпирическая проверка на данных: (а) какие группы дублируют друг друга
(порядок между ними неважен, дублирование сигнала), (б) какие группы несут
уникальный сигнал, не восстановимый из остальных (их стоит подавать раньше —
греедное правило: сначала независимое, потом то, что подхватит остаток).

Метод не завязан на `prognoz` — чисто структура данных (features_v2.feature_group):

1. Матрица избыточности между 16 группами (`ast/dist/geo/geo2/gm/l8/lin/ls/
   opt/pf/psr/relief2/s1/s2/s2raw/ter`) — среднее |Spearman r| по всем
   межгрупповым парам признаков (Spearman инвариантен к z-нормировке, поэтому
   считается прямо на исходных значениях).
2. "Уникальность" группы — честная (leave-one-strip-out, 4 полосы x 2 оси,
   буфер 15 км, как в `feature_batch_elimination.py`) Spearman-корреляция
   между 1-й главной компонентой группы (цель, определена только по своим
   признакам — утечки нет) и её честным OOF-предсказанием по RidgeCV на ВСЕХ
   ОСТАЛЬНЫХ группах. Низкая честная корреляция = группу нельзя
   реконструировать из соседей = уникальный сигнал = кандидат на раннюю
   стадию каскада.

geo/geo2 (прямые факторные слои формулы, 13 признаков) считаются отдельно,
для контекста — из рекомендованного порядка каскада исключены (это входы
самой критериальной формулы, ставить их стадией каскада, который потом
предполагается сравнивать с prognoz, было бы той же циркулярностью, от
которой избавлена только чистая лестница нетипичности, не сравнение с целью).

Запуск из корня: ``python -m experiments.feature_group_dependencies``.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import atypicality, cell_mask, config, criterial_target, features_v2  # noqa: E402
from experiments.feature_batch_elimination import strip_id_and_edges  # noqa: E402

N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0
SERVICE_COLS = ("row", "col", "x", "y", "mask_svita")

MACRO = {
    "структурный прокси": ["ter", "relief2", "lin", "dist"],
    "геофизика": ["gm", "pf"],
    "радар": ["s1", "psr"],
    "оптика/спектр": ["ast", "opt", "l8", "ls", "s2", "s2raw"],
    "формула (не для каскада)": ["geo", "geo2"],
}
GROUP_MACRO = {g: m for m, gs in MACRO.items() for g in gs}


def build_groups(cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in cols:
        g = features_v2.feature_group(c)
        groups.setdefault(g, []).append(c)
    return groups


def redundancy_matrix(corr: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    names = sorted(groups)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for g1 in names:
        for g2 in names:
            block = corr.loc[groups[g1], groups[g2]].abs().to_numpy()
            if g1 == g2:
                n = block.shape[0]
                if n > 1:
                    mask = ~np.eye(n, dtype=bool)
                    M.loc[g1, g2] = block[mask].mean()
                else:
                    M.loc[g1, g2] = np.nan
            else:
                M.loc[g1, g2] = block.mean()
    return M


def honest_uniqueness(X: np.ndarray, cols: list[str], groups: dict[str, list[str]],
                       cx: np.ndarray, cy: np.ndarray) -> pd.DataFrame:
    """Честная (leave-one-strip-out) корреляция PC1 группы с её OOF-предсказанием
    по RidgeCV на всех остальных группах. Ниже -> уникальнее (хуже реконструируется)."""
    col_idx = {c: i for i, c in enumerate(cols)}
    rows = []
    for g, gcols in sorted(groups.items()):
        g_idx = [col_idx[c] for c in gcols]
        rest_idx = [i for i in range(len(cols)) if i not in g_idx]
        Xg = X[:, g_idx]
        Xrest = X[:, rest_idx]
        target = PCA(n_components=1, random_state=config.ANOM_SEED).fit_transform(Xg).ravel()

        fold_rhos = []
        for axis_name, coord in [("X", cx), ("Y", cy)]:
            strip_id, edges = strip_id_and_edges(coord, N_STRIPS)
            oof_pred = np.full(len(target), np.nan)
            for s in range(N_STRIPS):
                lo, hi = edges[s], edges[s + 1]
                test_mask = strip_id == s
                train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
                if test_mask.sum() == 0 or train_mask.sum() < 200:
                    continue
                model = RidgeCV(alphas=np.logspace(-2, 3, 12))
                model.fit(Xrest[train_mask], target[train_mask])
                oof_pred[test_mask] = model.predict(Xrest[test_mask])
            ok = ~np.isnan(oof_pred)
            if ok.sum() > 10:
                rho = spearmanr(target[ok], oof_pred[ok]).statistic
                fold_rhos.append(abs(rho))
        rows.append({
            "group": g, "n_features": len(gcols),
            "macro": GROUP_MACRO.get(g, "?"),
            "explained_by_rest": float(np.mean(fold_rhos)) if fold_rhos else np.nan,
        })
        print(f"  {g} ({len(gcols)} признаков, {GROUP_MACRO.get(g, '?')}): "
              f"explained_by_rest={rows[-1]['explained_by_rest']:.3f}", flush=True)
    return pd.DataFrame(rows).sort_values("explained_by_rest")


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    cols = [c for c in df.columns if c not in SERVICE_COLS]
    groups = build_groups(cols)
    print(f"датасет: {df.shape[0]} ячеек, {len(cols)} признаков в {len(groups)} группах, "
          f"валидных {int(valid.sum())}", flush=True)

    X = atypicality.prepare_matrix(df[cols], valid)
    corr = pd.DataFrame(X, columns=cols).corr(method="spearman")

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # --- 1. матрица избыточности между группами ---
    M = redundancy_matrix(corr, groups)
    M.to_csv(out_dir / "metrics" / "feature_group_redundancy.csv")
    print("сохранено:", out_dir / "metrics" / "feature_group_redundancy.csv")

    dist = (1 - M.fillna(0)).to_numpy().copy()
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    Z = linkage(squareform(dist, checks=False), method="average")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    dendrogram(Z, labels=list(M.index), orientation="left", ax=axes[0])
    axes[0].set_xlabel("расстояние = 1 - средняя |корреляция| между группами")
    axes[0].set_title("Иерархия избыточности групп признаков")
    im = axes[1].imshow(M.to_numpy(dtype=float), cmap="magma", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(M.columns))); axes[1].set_xticklabels(M.columns, rotation=90)
    axes[1].set_yticks(range(len(M.index))); axes[1].set_yticklabels(M.index)
    axes[1].set_title("Средняя |Spearman r| между группами")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_group_redundancy.png", dpi=130)
    print("сохранено:", out_dir / "feature_group_redundancy.png")

    # --- 2. честная уникальность группы ---
    cx, cy = criterial_target.load_prognoz_grid()[0].cell_centers()
    cx, cy = cx.ravel()[valid], cy.ravel()[valid]
    print("\nЧестная уникальность групп (leave-one-strip-out, RidgeCV на остальных группах):")
    uniq = honest_uniqueness(X, cols, groups, cx, cy)
    uniq.to_csv(out_dir / "metrics" / "feature_group_uniqueness.csv", index=False)
    print("\nсохранено:", out_dir / "metrics" / "feature_group_uniqueness.csv")

    # --- рекомендованный порядок каскада ---
    print("\nРекомендованный порядок стадий каскада "
          "(макро-порядок физический, внутри стадии — по возрастанию explained_by_rest, "
          "т.е. сначала самое уникальное):")
    cascade_uniq = uniq[uniq["macro"] != "формула (не для каскада)"]
    for macro in ["структурный прокси", "геофизика", "радар", "оптика/спектр"]:
        sub = cascade_uniq[cascade_uniq["macro"] == macro].sort_values("explained_by_rest")
        order = ", ".join(f"{r.group}({r.explained_by_rest:.2f})" for r in sub.itertuples())
        print(f"  {macro}: {order}")

    print("\nСправочно (не для каскада — прямые входы формулы):")
    formula = uniq[uniq["macro"] == "формула (не для каскада)"].sort_values("explained_by_rest")
    for r in formula.itertuples():
        print(f"  {r.group}: explained_by_rest={r.explained_by_rest:.3f}")


if __name__ == "__main__":
    main()
