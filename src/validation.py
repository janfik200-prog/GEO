"""Честная валидация прогноза: presence-background + пространственная блочная CV.

В отличие от обучения на псевдометках (где метки выведены из тех же признаков и
ROC-AUC завышен циркулярностью), здесь:

* positive — ТОЛЬКО реальные рудопроявления;
* background — случайная фоновая выборка ячеек;
* разбиение train/test — пространственными блоками (убирает утечку из-за
  автокорреляции соседних ячеек);
* метрика — доля held-out реальных точек, попавших в top-N% площади прогноза
  (prediction-rate), и её отношение к случайному baseline (lift).

Это стандартная схема оценки прогнозных карт (mineral prospectivity) и она
честно отвечает на вопрос: даёт ли ML прирост над ручной эвристикой geo_score.
"""

import warnings
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .model import (
    BackgroundEnsemble, _make_ensemble, _make_gb, _make_lr, _make_rf, sample_presence_background,
)

# Реестр доступных моделей: имя -> фабрика (создаёт estimator с fit/predict_proba).
MODEL_FACTORIES = {
    "Random Forest": _make_rf,
    "Gradient Boosting": _make_gb,
    "Logistic Regression": _make_lr,
    "Ensemble RF+GB": _make_ensemble,
}


def assign_spatial_blocks(grid: gpd.GeoDataFrame, block_size: int) -> np.ndarray:
    """Назначить каждой ячейке id пространственного блока ``block_size``×``block_size``.

    Блоки служат группами для CV: ячейки одного блока никогда не попадают
    одновременно в train и test, что исключает пространственную утечку.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx = grid.geometry.centroid.x.to_numpy()
        cy = grid.geometry.centroid.y.to_numpy()
    bx = np.floor((cx - cx.min()) / block_size).astype(int)
    by = np.floor((cy - cy.min()) / block_size).astype(int)
    return bx * (by.max() + 1) + by


def _coverage(score_all: np.ndarray, test_positions: np.ndarray, area: float) -> float:
    """Доля held-out точек, попавших в top-``area`` ячеек по ``score_all``."""
    thr = np.quantile(score_all, 1.0 - area)
    return float((score_all[test_positions] >= thr).mean())


def _fold_lifts(grid, positive_cells, make_estimator, area, seed) -> list[float]:
    """Lift по каждому фолду пространственной CV для произвольной модели.

    Общий низкоуровневый помощник: presence-background выборка, блочная CV,
    обучение ``make_estimator()`` на train-фолде, прогноз по всей сетке, lift
    попадания held-out точек в top-``area``.
    """
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
    sample_pos, y = sample_presence_background(grid, positive_cells, config.VAL_N_BACKGROUND, seed)
    groups = blocks[sample_pos]
    X_sample = grid.iloc[sample_pos][config.FEATURE_COLS].fillna(0).to_numpy()
    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=config.VAL_N_SPLITS, shuffle=True, random_state=seed)
    lifts = []
    for tr, te in sgkf.split(X_sample, y, groups):
        test_positions = sample_pos[te][y[te] == 1]
        if test_positions.size == 0:
            continue
        model = make_estimator()
        model.fit(X_sample[tr], y[tr])
        score = model.predict_proba(X_all)[:, 1]
        lifts.append(_coverage(score, test_positions, area) / area)
    return lifts


def spatial_cv_evaluate(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    n_splits: int | None = None,
    block_size: int | None = None,
    n_background: int | None = None,
    areas: tuple[float, ...] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оценить модели пространственной CV на попадание реальных точек.

    Для каждого фолда обучаемые модели (RF, LogReg) учатся на presence-background
    из train-блоков и предсказывают по всей сетке; затем считается покрытие
    held-out точек из test-блока. Эвристика ``geo_score`` и случайный baseline
    оцениваются на тех же held-out точках для честного сравнения.

    Возвращает ``(agg, raw)``: агрегированную таблицу (lift mean/std по моделям и
    площадям) и сырые значения по фолдам.
    """
    n_splits = n_splits or config.VAL_N_SPLITS
    block_size = block_size or config.VAL_BLOCK_SIZE
    n_background = n_background or config.VAL_N_BACKGROUND
    areas = areas or config.VAL_AREAS
    seed = config.VAL_SEED if seed is None else seed

    blocks = assign_spatial_blocks(grid, block_size)
    sample_pos, y = sample_presence_background(grid, positive_cells, n_background, seed)
    groups = blocks[sample_pos]

    X_sample = grid.iloc[sample_pos][config.FEATURE_COLS].fillna(0).to_numpy()
    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    geo_all = grid["geo_score"].to_numpy()

    model_factories = {
        name: MODEL_FACTORIES[name]
        for name in ("Random Forest", "Gradient Boosting", "Logistic Regression")
    }
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    records: list[tuple] = []
    for fold, (tr, te) in enumerate(sgkf.split(X_sample, y, groups)):
        test_positions = sample_pos[te][y[te] == 1]
        if test_positions.size == 0:
            continue
        for mname, factory in model_factories.items():
            model = factory()
            model.fit(X_sample[tr], y[tr])
            score = model.predict_proba(X_all)[:, 1]
            for a in areas:
                records.append((mname, fold, a, _coverage(score, test_positions, a)))
        for a in areas:  # эвристика geo_score (точки не использует — честный baseline)
            records.append(("geo_score (эвристика)", fold, a, _coverage(geo_all, test_positions, a)))
        for a in areas:  # случайный прогноз: покрытие == площадь
            records.append(("Random", fold, a, a))

    raw = pd.DataFrame(records, columns=["model", "fold", "area", "coverage"])
    raw["lift"] = raw["coverage"] / raw["area"]
    agg = (
        raw.groupby(["model", "area"])
        .agg(coverage=("coverage", "mean"), lift=("lift", "mean"), lift_std=("lift", "std"))
        .reset_index()
    )
    return agg, raw


def repeated_spatial_cv(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    seeds: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Повторить пространственную CV на нескольких сидах и агрегировать lift.

    Разные сиды дают разные блоки/фон/фолды — устойчивость вывода к разбиению.
    Возвращает таблицу ``model × area`` со средним lift и std между сидами.
    """
    seeds = seeds or config.VAL_SEEDS
    frames = []
    for s in seeds:
        agg, _ = spatial_cv_evaluate(grid, positive_cells, seed=s)
        agg["seed"] = s
        frames.append(agg)
    allruns = pd.concat(frames, ignore_index=True)
    return (
        allruns.groupby(["model", "area"])
        .agg(lift=("lift", "mean"), lift_std=("lift", "std"),
             lift_min=("lift", "min"), lift_max=("lift", "max"))
        .reset_index()
    )


def permutation_test(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    area: float | None = None,
    n_perm: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Permutation-тест значимости RF-прогноза.

    Сравнивает наблюдаемый lift RF (пространственная CV на реальных точках) с
    нулевым распределением, где метки присвоены случайным ячейкам того же числа.
    Возвращает ``{observed, null_mean, null_q95, p_value}``. ``p_value`` — доля
    нулевых прогонов с lift ≥ наблюдаемого (значимо при p < 0.05).
    """
    area = area or config.PERM_AREA
    n_perm = n_perm or config.PERM_N
    seed = config.VAL_SEED if seed is None else seed
    n_pos = len(positive_cells)
    all_cells = grid["cell_id"].to_numpy()

    def _rf_cv_lift(pos_cells: list[int], s: int) -> float:
        blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)
        sample_pos, y = sample_presence_background(grid, pos_cells, config.VAL_N_BACKGROUND, s)
        groups = blocks[sample_pos]
        X_sample = grid.iloc[sample_pos][config.FEATURE_COLS].fillna(0).to_numpy()
        X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=config.VAL_N_SPLITS, shuffle=True, random_state=s)
        covs = []
        for tr, te in sgkf.split(X_sample, y, groups):
            test_positions = sample_pos[te][y[te] == 1]
            if test_positions.size == 0:
                continue
            model = _make_rf(n_estimators=config.PERM_RF_TREES)
            model.fit(X_sample[tr], y[tr])
            score = model.predict_proba(X_all)[:, 1]
            covs.append(_coverage(score, test_positions, area))
        return float(np.mean(covs)) / area if covs else np.nan

    observed = _rf_cv_lift(positive_cells, seed)
    rng = np.random.default_rng(seed)
    null = np.array([
        _rf_cv_lift(rng.choice(all_cells, size=n_pos, replace=False).tolist(), seed)
        for _ in range(n_perm)
    ])
    null = null[np.isfinite(null)]
    p_value = float((null >= observed).mean()) if null.size else np.nan
    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": p_value,
    }


def success_rate_curve(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    seeds: tuple[int, ...] | None = None,
    n_areas: int = 20,
) -> pd.DataFrame:
    """Success-rate кривые: покрытие точек vs доля площади для каждой модели.

    Возвращает таблицу ``model × area → coverage`` (усреднённую по сидам и фолдам),
    пригодную для построения кривых «найденные точки vs обследованная площадь».
    """
    areas = tuple(np.round(np.linspace(0.02, 0.5, n_areas), 4))
    seeds = seeds or config.VAL_SEEDS
    frames = []
    for s in seeds:
        agg, _ = spatial_cv_evaluate(grid, positive_cells, areas=areas, seed=s)
        frames.append(agg)
    allruns = pd.concat(frames, ignore_index=True)
    return (
        allruns.groupby(["model", "area"])
        .agg(coverage=("coverage", "mean"))
        .reset_index()
    )


# Сетка гиперпараметров GB по умолчанию для tune_gb.
GB_PARAM_GRID: dict[str, list] = {
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
    "subsample": [0.8],
}


def tune_gb(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    param_grid: dict[str, list] | None = None,
    area: float | None = None,
    seeds: tuple[int, ...] = (1, 7),
) -> pd.DataFrame:
    """Подобрать гиперпараметры Gradient Boosting по пространственной CV.

    Перебирает ``param_grid``, для каждой комбинации усредняет lift (top-``area``)
    по фолдам и ``seeds``. ``seeds`` для тюнинга должны отличаться от сидов
    итоговой оценки, иначе выбор параметров даст оптимистичное смещение.
    Возвращает таблицу комбинаций, отсортированную по убыванию lift.
    """
    param_grid = param_grid or GB_PARAM_GRID
    area = area or config.PERM_AREA
    keys = list(param_grid)
    rows = []
    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        lifts = [
            lift
            for s in seeds
            for lift in _fold_lifts(grid, positive_cells, lambda p=params: _make_gb(**p), area, s)
        ]
        rows.append({**params, "lift_mean": float(np.mean(lifts)), "lift_std": float(np.std(lifts))})
    return pd.DataFrame(rows).sort_values("lift_mean", ascending=False).reset_index(drop=True)


# Сетка гиперпараметров BackgroundEnsemble по умолчанию для tune_ensemble.
ENS_PARAM_GRID: dict[str, list] = {
    "n_rounds": [4, 8],
    "bg_frac": [0.5, 0.7, 0.9],
}


def tune_ensemble(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    param_grid: dict[str, list] | None = None,
    area: float | None = None,
    seeds: tuple[int, ...] = (1, 7),
) -> pd.DataFrame:
    """Подобрать гиперпараметры BackgroundEnsemble по пространственной CV.

    Перебирает ``param_grid`` (``n_rounds`` × ``bg_frac``), для каждой комбинации
    усредняет lift (top-``area``) по фолдам и ``seeds``. ``seeds`` для тюнинга
    должны отличаться от сидов итоговой оценки (иначе выбор даст оптимистичное
    смещение). Возвращает таблицу комбинаций, отсортированную по убыванию lift.
    """
    param_grid = param_grid or ENS_PARAM_GRID
    area = area or config.PERM_AREA
    keys = list(param_grid)
    rows = []
    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        lifts = [
            lift
            for s in seeds
            for lift in _fold_lifts(grid, positive_cells, lambda p=params: BackgroundEnsemble(**p), area, s)
        ]
        rows.append({**params, "lift_mean": float(np.mean(lifts)), "lift_std": float(np.std(lifts))})
    return pd.DataFrame(rows).sort_values("lift_mean", ascending=False).reset_index(drop=True)


def paired_model_comparison(
    grid: gpd.GeoDataFrame,
    positive_cells: list[int],
    area: float | None = None,
    seeds: tuple[int, ...] | None = None,
    n_boot: int = 5000,
    model_a: str = "Random Forest",
    model_b: str = "Gradient Boosting",
) -> tuple[dict, pd.DataFrame]:
    """Парное сравнение двух моделей на ИДЕНТИЧНЫХ фолдах пространственной CV.

    Для каждого (сид, фолд) обе модели обучаются и оцениваются на одном и том же
    разбиении, что даёт корректные пары lift. Возвращает сводку (средний lift
    каждой модели, средняя разница b−a с bootstrap-95% ДИ, доля фолдов где b>a) и
    таблицу пар. Разница значима, если 95%% ДИ не включает 0.
    """
    area = area or config.PERM_AREA
    seeds = seeds or config.VAL_SEEDS
    factories = {model_a: MODEL_FACTORIES[model_a], model_b: MODEL_FACTORIES[model_b]}

    pairs = []
    for s in seeds:
        # один и тот же split для обеих моделей -> корректные пары
        a_lifts = _fold_lifts(grid, positive_cells, factories[model_a], area, s)
        b_lifts = _fold_lifts(grid, positive_cells, factories[model_b], area, s)
        for fold, (la, lb) in enumerate(zip(a_lifts, b_lifts)):
            pairs.append({"seed": s, "fold": fold, "lift_a": la, "lift_b": lb})
    pair_df = pd.DataFrame(pairs)
    diff = (pair_df["lift_b"] - pair_df["lift_a"]).to_numpy()

    rng = np.random.default_rng(config.VAL_SEED)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    summary = {
        "n_pairs": int(len(diff)),
        f"mean_lift_{model_a}": float(pair_df["lift_a"].mean()),
        f"mean_lift_{model_b}": float(pair_df["lift_b"].mean()),
        "mean_diff_b_minus_a": float(diff.mean()),
        "ci95_diff": ci,
        "win_rate_b": float((diff > 0).mean()),
        "significant": bool(ci[0] > 0 or ci[1] < 0),
    }
    return summary, pair_df
