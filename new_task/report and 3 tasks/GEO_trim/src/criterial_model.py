"""Классификатор критериального таргета + честная пространственная валидация.

Задача: научить ML предсказывать перспективные (по критериальному анализу ГИС
Интегро) ячейки из геофизики / Landsat / структурных признаков (`dataset_v1`).
Таргет и признаки готовит :mod:`src.criterial_data`.

Модель — ансамбль RandomForest + GradientBoosting (усреднение вероятностей),
как и в боевом :mod:`src.model`, но здесь метки есть у ВСЕХ ячеек (полная
разметка от критериального прогноза), поэтому обучение — обычная классификация
с балансировкой классов, а не presence-background.

Валидация — out-of-fold по пространственным блокам (GroupKFold), метрики:
ROC-AUC, PR-AUC (average precision) и lift@N% (во сколько раз доля
перспективных в топ-N% прогноза выше базовой). lift@N% — ключевая метрика
для геологоразведки: показывает выигрыш при заверке ограниченной площади.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from . import config
from .criterial_data import CriterialDataset


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.CRITERIAL_RF_TREES,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        class_weight="balanced_subsample",
        random_state=config.CRITERIAL_RANDOM_STATE,
        n_jobs=-1,
    )


def _make_gb() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=config.CRITERIAL_GB_TREES,
        max_depth=config.GB_MAX_DEPTH,
        learning_rate=config.GB_LEARNING_RATE,
        subsample=config.GB_SUBSAMPLE,
        random_state=config.CRITERIAL_RANDOM_STATE,
    )


class RfGbEnsemble:
    """Ансамбль RF+GB: среднее вероятностей двух декоррелированных моделей."""

    def __init__(self) -> None:
        self.rf = _make_rf()
        self.gb = _make_gb()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RfGbEnsemble":
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        return self

    def predict_proba1(self, X: np.ndarray) -> np.ndarray:
        return 0.5 * (self.rf.predict_proba(X)[:, 1] + self.gb.predict_proba(X)[:, 1])

    @property
    def feature_importances_(self) -> np.ndarray:
        return 0.5 * (self.rf.feature_importances_ + self.gb.feature_importances_)


def lift_at(y_true: np.ndarray, score: np.ndarray, area: float) -> float:
    """Lift@area: (доля позитивов в верхних `area` по score) / (базовая доля позитивов)."""
    n = len(score)
    k = max(1, int(round(area * n)))
    top = np.argsort(-score)[:k]
    base = y_true.mean()
    if base <= 0:
        return float("nan")
    return float(y_true[top].mean() / base)


@dataclass
class CriterialCVResult:
    """Итог пространственной кросс-валидации."""

    oof_score: np.ndarray                       # OOF-вероятность класса для каждой ячейки
    roc_auc: float
    pr_auc: float
    lift: dict[float, float]                    # lift@area по площадям
    per_fold: list[dict] = field(default_factory=list)


def spatial_cv(data: CriterialDataset, n_splits: int | None = None) -> CriterialCVResult:
    """Out-of-fold прогноз по пространственным блокам + агрегированные метрики."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_splits = int(n_splits or config.CRITERIAL_N_SPLITS)
    X, y, groups = data.X, data.y, data.groups
    oof = np.full(len(y), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    per_fold: list[dict] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model = RfGbEnsemble().fit(X[tr], y[tr])
        p = model.predict_proba1(X[te])
        oof[te] = p
        fold_metrics = {
            "fold": fold,
            "n_test": int(len(te)),
            "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te])) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y[te], p)),
            "lift@10%": lift_at(y[te], p, 0.10),
        }
        per_fold.append(fold_metrics)

    lift = {a: lift_at(y, oof, a) for a in config.CRITERIAL_LIFT_AREAS}
    return CriterialCVResult(
        oof_score=oof,
        roc_auc=float(roc_auc_score(y, oof)),
        pr_auc=float(average_precision_score(y, oof)),
        lift=lift,
        per_fold=per_fold,
    )


def fit_full(data: CriterialDataset) -> RfGbEnsemble:
    """Обучить финальную модель на всех ячейках (для карты прогноза по всей сетке)."""
    return RfGbEnsemble().fit(data.X, data.y)


def predict_grid(model: RfGbEnsemble, data: CriterialDataset) -> np.ndarray:
    """Вернуть карту вероятности перспективности формы ``data.grid_shape``."""
    p = model.predict_proba1(data.X)
    grid = np.full(data.grid_shape, np.nan, dtype=float)
    rows = data.frame["row"].to_numpy().astype(int)
    cols = data.frame["col"].to_numpy().astype(int)
    grid[rows, cols] = p
    return grid


def feature_importance(model: RfGbEnsemble, data: CriterialDataset) -> list[tuple[str, float]]:
    """Важности признаков ансамбля, отсортированные по убыванию."""
    imp = model.feature_importances_
    pairs = sorted(zip(data.feature_cols, imp), key=lambda t: -t[1])
    return [(name, float(v)) for name, v in pairs]
