"""Проверка гипотезы: RF ценит `gm_gr_1G_25` не через собственную маргинальную
нелинейность (её partial dependence почти плоская, см.
`gm_gr_1G_25_nonlinearity.py`), а через ВЗАИМОДЕЙСТВИЕ с другими признаками —
двугорбая форма в сырых данных гасится при усреднении по остальным 90
признакам, но может проявляться на 2D partial dependence с конкретным
соседом.

Метод: берём топ-N признаков по impurity importance RF (кроме самого
gm_gr_1G_25), для каждого считаем 2D partial dependence (gm_gr_1G_25, сосед) и
сравниваем с "аддитивным" предсказанием (сумма двух 1D PDP минус среднее) —
простая приближённая версия H-статистики Фридмана: доля дисперсии 2D PDP, не
объяснённая суммой одномерных эффектов. Высокое значение = сильное
взаимодействие, низкое = эффекты независимы (аддитивны).

Запуск из корня репозитория: ``python -m experiments.gm_gr_1G_25_interaction``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402

FEATURE = "gm_gr_1G_25"
N_CANDIDATES = 8
GRID_RES_1D = 40
GRID_RES_2D = 25


def h_statistic(pd1_j: np.ndarray, pd1_k: np.ndarray, pd2: np.ndarray) -> float:
    """Приближённая H-статистика Фридмана по сетке partial dependence.

    pd2[a, b] — 2D PDP в точке (grid_j[a], grid_k[b]).
    pd1_j, pd1_k — 1D PDP на тех же сетках grid_j, grid_k.
    Аддитивная модель: pd1_j[a] + pd1_k[b] - mean(pd2). Взаимодействие —
    остаток после вычитания аддитивной части.
    """
    additive = pd1_j[:, None] + pd1_k[None, :] - pd2.mean()
    resid = pd2 - additive
    denom = np.sum((pd2 - pd2.mean()) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum(resid ** 2) / denom)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = -prognoz.ravel().astype(float)

    feat_df = criterial_target.training_features(dataset="v6")
    pool = [c for c in feat_df.columns if not cell_mask.is_service(c)]
    X = feat_df[pool].fillna(0).to_numpy()
    gm_idx = pool.index(FEATURE)

    print(f"Обучаю RF на всём листе (91 признак) для 2D partial dependence по {FEATURE}...", flush=True)
    model = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.CRIT_SEED, n_jobs=-1,
    ).fit(X, y)

    importances = pd.Series(model.feature_importances_, index=pool).sort_values(ascending=False)
    candidates = [f for f in importances.index if f != FEATURE][:N_CANDIDATES]
    print(f"Топ-{N_CANDIDATES} кандидатов-соседей по impurity importance RF: {candidates}", flush=True)

    pdp_j = partial_dependence(model, X, [gm_idx], kind="average", grid_resolution=GRID_RES_1D)
    grid_j = pdp_j["grid_values"][0]
    pd1_j = pdp_j["average"][0]

    results = []
    pdp_cache = {}
    for feat_k in candidates:
        k_idx = pool.index(feat_k)
        pdp_k = partial_dependence(model, X, [k_idx], kind="average", grid_resolution=GRID_RES_1D)
        grid_k = pdp_k["grid_values"][0]
        pd1_k = pdp_k["average"][0]

        pdp_2d = partial_dependence(model, X, [(gm_idx, k_idx)], kind="average",
                                     grid_resolution=GRID_RES_2D)
        grid_j2, grid_k2 = pdp_2d["grid_values"]
        pd2 = pdp_2d["average"][0]  # shape (len(grid_j2), len(grid_k2))

        pd1_j_on2 = np.interp(grid_j2, grid_j, pd1_j)
        pd1_k_on2 = np.interp(grid_k2, grid_k, pd1_k)
        h2 = h_statistic(pd1_j_on2, pd1_k_on2, pd2)

        results.append({"feature": feat_k, "h_stat_sq": h2})
        pdp_cache[feat_k] = (grid_j2, grid_k2, pd2)
        print(f"  {FEATURE} x {feat_k}: H^2 = {h2:.4f}", flush=True)

    res_df = pd.DataFrame(results).sort_values("h_stat_sq", ascending=False)
    out_csv = ROOT / "outputs" / "metrics" / "gm_gr_1G_25_interaction_h_stat.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nСохранено: {out_csv}")
    print(res_df.to_string(index=False))

    top4 = res_df.head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, feat_k in zip(axes.ravel(), top4):
        grid_j2, grid_k2, pd2 = pdp_cache[feat_k]
        im = ax.contourf(grid_k2, grid_j2, pd2, levels=20, cmap="RdBu_r")
        h2 = res_df.loc[res_df["feature"] == feat_k, "h_stat_sq"].iloc[0]
        ax.set_xlabel(feat_k)
        ax.set_ylabel(FEATURE)
        ax.set_title(f"2D PDP: {FEATURE} x {feat_k}  (H^2={h2:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="предсказание RF (-prognoz)")
    fig.suptitle(f"Взаимодействие {FEATURE} с топ-4 соседями по важности RF "
                 "(чем ярче искривлены линии уровня — тем сильнее взаимодействие)", fontsize=11)
    fig.tight_layout()
    out_png = ROOT / "outputs" / "gm_gr_1G_25_interaction.png"
    fig.savefig(out_png, dpi=150)
    print(f"Сохранено: {out_png}")


if __name__ == "__main__":
    main()
