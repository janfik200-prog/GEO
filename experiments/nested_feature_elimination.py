"""Честная (nested) проверка backward elimination — ревью `model-critic` от 27.08.2026.

Разбор `feature_batch_elimination.py`/`feature_elimination_precise_21_to_11.py`/
`elastic_net_batch_elimination.py` нашёл двухуровневый double dipping: признак
для удаления на каждом раунде выбирается по permutation importance, посчитанной
НА ТЕХ ЖЕ leave-one-strip-out полосах, на которых потом репортится честная
метрика раунда — чистого held-out не остаётся вообще. Плюс отдельные проблемы:
маска валидных ячеек нигде не применялась (пик на 17 признаках получен
удалением `dist_dnara` — индикатора воды), скорер отбора (`r2`) не совпадает с
репортируемой метрикой (Spearman), баз для сравнения (xy-only, 1-NN) не было,
сид всегда один и тот же (`config.CRIT_SEED=42`).

Здесь для каждой из 8 внешних полос (leave-one-strip-out, буфер 15 км, как и
раньше) считаются ДВА варианта ранжирования признаков при ОДИНАКОВОЙ финальной
оценке (переобучение на всём внешнем train, предсказание на внешнем test —
том самом held-out, который никогда не участвует в ранжировании):

* ``leaky`` — как в исходных экспериментах: модель обучена на внешнем train,
  permutation importance посчитана прямо на внешнем test (том же, на котором
  потом репортится метрика). Двойное использование test-полосы воспроизведено
  намеренно, чтобы измерить именно масштаб оптимистичного смещения, а не
  просто продекларировать его.
* ``nested`` — внешний train делится ещё раз (2 внутренние полосы по той же
  оси, буфер вдвое меньше — 7.5 км) на inner-fit/inner-holdout; ранжирование
  считается на inner-holdout, который внешний test никогда не видит. Финальная
  модель для оценки всё равно обучается на ПОЛНОМ внешнем train (не только на
  inner-fit) — иначе сравнение leaky/nested было бы нечестным по объёму данных,
  а не по чистоте отбора.

Раз обе ветки заканчиваются идентичным шагом (переобучение на внешнем train,
метрика на внешнем test), расхождение leaky-nested при одном и том же K —
прямая оценка величины double-dipping-смещения, а не косвенная.

Дополнительно к ревью:
* маска валидных ячеек `cell_mask.build_valid_mask` применяется один раз до
  всех разбиений (убирает 41 ячейку воды/края — включая источник подозрительного
  пика на 17 признаках через `dist_dnara`);
* скорер permutation importance — Spearman (`spearman_scorer`), не `r2`,
  согласован с репортируемой метрикой;
* базы для сравнения на каждой внешней полосе: модель только на координатах
  (xy-only) и 1-NN по координатам (`nn_baseline_predict`, как в
  `forecast_dense.py`) — если nested-кривая не отрывается от них, «сигнал в
  признаках» не доказан;
* два сида (`config.CRIT_SEED=42` и запасной 7) — проверка, что выводы не
  особенность одного прогона RF.

Упрощение ради стоимости (честно, не спрятано): ранжирование каждого K берётся
из ОДНОЙ importance-оценки на все 91 признак (top-K по значению), а не из
итеративного пересчёта важности после каждого удаления, как в оригинальных
скриптах. Это дешевле (не 91 отдельный переобучающийся раунд), но может
недооценивать пользу признаков, чья важность растёт только после удаления
конкурентов из того же кластера — см. `correlation_clustering_91.py`.

Запуск из корня репозитория: ``python -m experiments.nested_feature_elimination``.
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.feature_batch_elimination import strip_id_and_edges  # noqa: E402

N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0
INNER_BUFFER_M = STRIP_BUFFER_M / 2
N_REPEATS = 3
SEEDS = [config.CRIT_SEED, 7]
CANDIDATE_SIZES = [91, 51, 21, 17, 13, 12, 11, 5, 1]


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(spearmanr(y_true, y_pred).statistic)


spearman_scorer = make_scorer(_spearman)


def nn_baseline_predict(coord_train: np.ndarray, y_train: np.ndarray, coord_test: np.ndarray) -> np.ndarray:
    tree = cKDTree(coord_train)
    _, idx = tree.query(coord_test, k=1)
    return y_train[idx]


def fit_rf(X: np.ndarray, y: np.ndarray, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=seed, n_jobs=-1,
    ).fit(X, y)


def rank_features(model: RandomForestRegressor, X: np.ndarray, y: np.ndarray,
                   feat_names: list[str], seed: int) -> list[str]:
    """Ранжирование признаков по убыванию permutation importance (Spearman-скорер)."""
    perm = permutation_importance(model, X, y, n_repeats=N_REPEATS, random_state=seed,
                                   scoring=spearman_scorer)
    order = np.argsort(-perm.importances_mean)
    return [feat_names[i] for i in order]


def inner_split(train_mask: np.ndarray, coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Делит внешний train на inner-fit/inner-holdout (2 полосы той же оси, буфер вдвое меньше)."""
    idx_train = np.where(train_mask)[0]
    coord_train = coord[idx_train]
    inner_id, inner_edges = strip_id_and_edges(coord_train, 2)
    lo, hi = inner_edges[0], inner_edges[1]
    local_holdout = inner_id == 0
    local_fit = coord_train > hi + INNER_BUFFER_M
    holdout_mask = np.zeros_like(train_mask)
    fit_mask = np.zeros_like(train_mask)
    holdout_mask[idx_train[local_holdout]] = True
    fit_mask[idx_train[local_fit]] = True
    return fit_mask, holdout_mask


def run_outer_fold(axis_name: str, s: int, coord: np.ndarray, xy: np.ndarray,
                    X: np.ndarray, y: np.ndarray, feat_names: list[str], seed: int) -> list[dict]:
    strip_id, edges = strip_id_and_edges(coord, N_STRIPS)
    lo, hi = edges[s], edges[s + 1]
    test_mask = strip_id == s
    train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
    if test_mask.sum() == 0 or train_mask.sum() < 200:
        return []

    fit_mask, holdout_mask = inner_split(train_mask, coord)
    if fit_mask.sum() < 100 or holdout_mask.sum() < 100:
        return []

    # ---- базы: xy-only и 1-NN (по координатам, без единого признака) ----
    model_xy = fit_rf(xy[train_mask], y[train_mask], seed)
    rho_xy = _spearman(y[test_mask], model_xy.predict(xy[test_mask]))
    pred_nn = nn_baseline_predict(xy[train_mask], y[train_mask], xy[test_mask])
    rho_nn = _spearman(y[test_mask], pred_nn)

    # ---- leaky: ранжирование прямо на внешнем test (как в исходных скриптах) ----
    model_full_train = fit_rf(X[train_mask], y[train_mask], seed)
    rank_leaky = rank_features(model_full_train, X[test_mask], y[test_mask], feat_names, seed)

    # ---- nested: ранжирование на inner-holdout, внешний test не участвует ----
    model_inner = fit_rf(X[fit_mask], y[fit_mask], seed)
    rank_nested = rank_features(model_inner, X[holdout_mask], y[holdout_mask], feat_names, seed)

    rows = []
    for K in CANDIDATE_SIZES:
        for variant, ranking in [("leaky", rank_leaky), ("nested", rank_nested)]:
            feats = ranking[:K]
            cols = [feat_names.index(f) for f in feats]
            model_k = fit_rf(X[train_mask][:, cols], y[train_mask], seed)
            pred = model_k.predict(X[test_mask][:, cols])
            rho = _spearman(y[test_mask], pred)
            rows.append({
                "axis": axis_name, "strip": s, "seed": seed, "n_features": K,
                "variant": variant, "spearman": rho,
                "n_test": int(test_mask.sum()), "n_train": int(train_mask.sum()),
                "spearman_xy_only": rho_xy, "spearman_1nn": rho_nn,
            })
    print(f"    [{axis_name}{s} seed={seed}] готово (train={train_mask.sum()}, "
          f"test={test_mask.sum()}, inner_fit={fit_mask.sum()}, inner_holdout={holdout_mask.sum()})",
          flush=True)
    return rows


def main() -> None:
    t0 = time.time()
    meta, prognoz = criterial_target.load_prognoz_grid()
    y_full = -prognoz.ravel().astype(float)

    feat_df = criterial_target.training_features(dataset="v6")
    pool = [c for c in feat_df.columns if not cell_mask.is_service(c)]
    valid = cell_mask.build_valid_mask(feat_df)
    print(f"Пул признаков: {len(pool)}. Валидных ячеек: {valid.sum()}/{len(valid)} "
          f"(исключено {(~valid).sum()} — вода/край покрытия/много NaN)", flush=True)

    X = feat_df[pool].fillna(0).to_numpy()[valid]
    y = y_full[valid]
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel()[valid], cy.ravel()[valid]
    xy = np.column_stack([cx, cy])

    # быстрая проверка гипотезы критика: скачок на 1 признаке (l8_iron_ox) —
    # это просто маргинальная ранговая корреляция по всему листу, не честная
    # пространственная метрика.
    if "l8_iron_ox" in pool:
        col = feat_df["l8_iron_ox"].to_numpy()[valid]
        finite = np.isfinite(col)
        rho_marg = _spearman(y[finite], col[finite])
        print(f"Маргинальная (не пространственная) корреляция l8_iron_ox с -prognoz "
              f"по {finite.sum()}/{valid.sum()} валидным ячейкам (без NaN самого признака): "
              f"{rho_marg:.4f} — для сравнения со скачком на 1 признаке в "
              f"fine-elimination (0.278)", flush=True)

    all_rows = []
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===", flush=True)
        for axis_name, coord in [("X", cx), ("Y", cy)]:
            for s in range(N_STRIPS):
                all_rows.extend(run_outer_fold(axis_name, s, coord, xy, X, y, pool, seed))

    per_fold = pd.DataFrame(all_rows)
    out_dir = ROOT / "outputs" / "metrics"
    per_fold_path = out_dir / "nested_feature_elimination_per_fold.csv"
    per_fold.to_csv(per_fold_path, index=False)
    print(f"\nСохранено: {per_fold_path} ({len(per_fold)} строк)")

    summary = (per_fold.groupby(["n_features", "variant"])["spearman"]
               .agg(["mean", "std", "count"]).reset_index())
    baseline_xy = per_fold.drop_duplicates(["axis", "strip", "seed"])["spearman_xy_only"].mean()
    baseline_nn = per_fold.drop_duplicates(["axis", "strip", "seed"])["spearman_1nn"].mean()
    summary_path = out_dir / "nested_feature_elimination_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Сохранено: {summary_path}")
    print(f"\nБаза xy-only (среднее по {per_fold['strip'].nunique()*len(SEEDS)*2} внешним фолдам "
          f"x сидам): {baseline_xy:.4f}")
    print(f"База 1-NN: {baseline_nn:.4f}")

    print("\nСводная таблица (среднее honest Spearman по 8 внешним полосам x "
          f"{len(SEEDS)} сидам = {8*len(SEEDS)} прогонов на точку):")
    piv = summary.pivot(index="n_features", columns="variant", values="mean").sort_index(ascending=False)
    piv["leaky_minus_nested"] = piv["leaky"] - piv["nested"]
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    # разбор конкретно обрыва 13->12 по осям
    print("\nОбрыв 13->12 признаков, разбивка по осям (nested, среднее по сидам):")
    sub = per_fold[(per_fold["n_features"].isin([13, 12])) & (per_fold["variant"] == "nested")]
    print(sub.groupby(["axis", "n_features"])["spearman"].mean().to_string(float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(9, 6))
    for variant, color in [("leaky", "tab:red"), ("nested", "tab:blue")]:
        sub = summary[summary["variant"] == variant].sort_values("n_features")
        ax.plot(sub["n_features"], sub["mean"], marker="o", color=color, label=variant)
        ax.fill_between(sub["n_features"], sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                         color=color, alpha=0.15)
    ax.axhline(baseline_xy, color="gray", linestyle="--", linewidth=1, label="база: только координаты")
    ax.axhline(baseline_nn, color="gray", linestyle=":", linewidth=1, label="база: 1-NN по координатам")
    ax.invert_xaxis()
    ax.set_xlabel("число признаков K (top-K по ранжированию)")
    ax.set_ylabel("честная корреляция Спирмена на внешнем held-out")
    ax.set_title("Nested vs leaky backward elimination, dataset_v6\n"
                  "(расхождение leaky-nested = масштаб double-dipping-смещения)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = ROOT / "outputs" / "nested_feature_elimination.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nСохранено: {plot_path}")
    print(f"\nВсего времени: {(time.time()-t0)/60:.1f} мин")


if __name__ == "__main__":
    main()
