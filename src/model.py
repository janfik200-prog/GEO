"""ML-ядро прогноза: presence-background обучение, прогноз, золотые зоны.

Логика построена на честной схеме (см. SECURITY-замечания в истории проекта):
positive — ТОЛЬКО реальные рудопроявления, фон — случайная выборка ячеек.
Псевдометки по geo_score и геологический стабилизатор намеренно убраны — они
вносили циркулярность и гасили реальный сигнал. ``geo_score`` остаётся только
как критериальный baseline в :mod:`src.validation`.

Порядок вызова: ``mark_presence`` → ``train_model`` → ``compute_prospectivity``
→ ``mark_gold_zones``. Все функции дополняют ``grid`` столбцами и возвращают его.
"""

import geopandas as gpd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import robust_normalize_01, smooth_on_regular_grid, top_n_components


def _make_rf(n_estimators: int | None = None) -> RandomForestClassifier:
    """Создать Random Forest с параметрами из конфига (число деревьев переопределяемо)."""
    return RandomForestClassifier(
        n_estimators=n_estimators or config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        class_weight="balanced_subsample",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def _make_gb(**overrides) -> GradientBoostingClassifier:
    """Создать Gradient Boosting с параметрами из конфига (переопределяемы для тюнинга)."""
    params = dict(
        n_estimators=config.GB_N_ESTIMATORS,
        max_depth=config.GB_MAX_DEPTH,
        learning_rate=config.GB_LEARNING_RATE,
        subsample=config.GB_SUBSAMPLE,
        random_state=config.RANDOM_STATE,
    )
    params.update(overrides)
    return GradientBoostingClassifier(**params)


class BackgroundEnsemble:
    """Ансамбль RF+GB, усреднённый по нескольким подвыборкам фона (negatives).

    На ``fit`` за ``n_rounds`` раундов берётся случайная подвыборка отрицательного
    класса (доля ``bg_frac``); на (все позитивы + подвыборка) обучаются RF и GB.
    ``predict_proba`` возвращает среднее вероятностей всех ``2*n_rounds`` моделей.

    Цель — снизить дисперсию прогноза: (1) усреднение по разным фонам убирает шум
    от произвольного выбора фоновой выборки; (2) RF и GB декоррелированы, их
    усреднение стабилизирует оценку. Совместим с интерфейсом sklearn
    (``fit``/``predict_proba``), поэтому подключается в CV как обычная модель.
    """

    def __init__(self, n_rounds: int | None = None, bg_frac: float | None = None,
                 random_state: int | None = None):
        self.n_rounds = n_rounds or config.ENS_N_ROUNDS
        self.bg_frac = bg_frac or config.ENS_BG_FRAC
        self.random_state = config.RANDOM_STATE if random_state is None else random_state
        self.models_: list = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_draw = max(1, int(round(self.bg_frac * len(neg_idx))))
        self.models_ = []
        for r in range(self.n_rounds):
            draw = rng.choice(neg_idx, size=min(n_draw, len(neg_idx)), replace=False)
            idx = np.concatenate([pos_idx, draw])
            Xr, yr = X[idx], y[idx]
            for make in (_make_rf, _make_gb):
                model = make()
                model.fit(Xr, yr)
                self.models_.append(model)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1.0 - probs, probs])

    def member_probas(self, X) -> np.ndarray:
        """Матрица P(class=1) по каждому члену ансамбля: форма (n_members, n)."""
        X = np.asarray(X)
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models_])

    @property
    def feature_importances_(self) -> np.ndarray:
        """Усреднённая важность признаков по всем членам ансамбля (RF и GB)."""
        return np.mean([m.feature_importances_ for m in self.models_], axis=0)


def _make_ensemble() -> BackgroundEnsemble:
    """Создать ансамбль RF+GB с усреднением по фону (параметры из конфига)."""
    return BackgroundEnsemble()


def _make_lr() -> Pipeline:
    """Создать Logistic Regression (со стандартизацией) — для сравнения в валидации."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="lbfgs",
            random_state=config.RANDOM_STATE,
        )),
    ])


def safe_binary_metrics(y_true, y_prob) -> dict[str, float]:
    """ROC-AUC, AP и средние вероятности по классам, без падения на одном классе."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {
        "ROC_AUC": np.nan,
        "Average_Precision": np.nan,
        "Positive_mean_prob": np.nan,
        "Negative_mean_prob": np.nan,
    }
    if len(np.unique(y_true)) >= 2:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
        out["Average_Precision"] = float(average_precision_score(y_true, y_prob))
    if np.any(y_true == 1):
        out["Positive_mean_prob"] = float(np.mean(y_prob[y_true == 1]))
    if np.any(y_true == 0):
        out["Negative_mean_prob"] = float(np.mean(y_prob[y_true == 0]))
    return out


def mark_presence(grid: gpd.GeoDataFrame, points: gpd.GeoDataFrame | None) -> tuple[gpd.GeoDataFrame, list[int]]:
    """Отметить ячейки с реальными рудопроявлениями (``presence`` = 1).

    Возвращает ``(grid, positive_cells)`` — список ``cell_id`` с точками.
    """
    grid["presence"] = 0
    if points is None or len(points) == 0:
        return grid, []
    try:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(points[["geometry"]], grid[["cell_id", "geometry"]], how="left", op="within")
    positive_cells = joined["cell_id"].dropna().astype(int).unique().tolist()
    grid.loc[grid["cell_id"].isin(positive_cells), "presence"] = 1
    return grid, positive_cells


def sample_presence_background(
    grid: gpd.GeoDataFrame, positive_cells: list[int], n_background: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Сформировать выборку presence-background.

    Возвращает ``(sample_pos, y)``: позиции выбранных ячеек в ``grid`` (реальные
    точки + случайный фон) и метки (1 — точка, 0 — фон). Общий помощник для
    боевого обучения и для CV в :mod:`src.validation`.
    """
    rng = np.random.default_rng(seed)
    is_pos = grid["cell_id"].isin(positive_cells).to_numpy()
    pos_pos = np.where(is_pos)[0]
    bg_pool = np.where(~is_pos)[0]
    n_bg = min(n_background, len(bg_pool))
    bg_pos = rng.choice(bg_pool, size=n_bg, replace=False)
    sample_pos = np.concatenate([pos_pos, bg_pos])
    y = np.concatenate([np.ones(len(pos_pos)), np.zeros(len(bg_pos))]).astype(int)
    return sample_pos, y


def train_model(grid: gpd.GeoDataFrame, positive_cells: list[int]) -> tuple[gpd.GeoDataFrame, BackgroundEnsemble | None]:
    """Обучить ансамбль RF+GB на presence-background и записать ``ml_score`` по сетке.

    Боевая модель — :class:`BackgroundEnsemble` (усреднение RF и GB по нескольким
    подвыборкам фона): честным сплит-тюнингом и проверкой на 50 независимых фолдах
    показала значимый прирост среднего lift над чистым RF (+0.36, 95% ДИ
    [0.08, 0.65]). Обучается на всех реальных точках + случайном фоне
    (``TRAIN_N_BACKGROUND``), предсказывает вероятность присутствия по всей сетке.
    При нехватке точек (< ``MIN_POS_CELLS``) ``ml_score`` остаётся неинформативным
    (0.5). Возвращает ``(grid, model)``.
    """
    if len(positive_cells) < config.MIN_POS_CELLS:
        grid["ml_score"] = 0.5
        grid["ml_score_sm"] = 0.5
        return grid, None

    sample_pos, y = sample_presence_background(
        grid, positive_cells, config.TRAIN_N_BACKGROUND, config.TRAIN_SEED
    )
    X = grid.iloc[sample_pos][config.FEATURE_COLS].fillna(0).to_numpy()
    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()

    model = _make_ensemble()
    model.fit(X, y)
    grid["ml_score"] = robust_normalize_01(model.predict_proba(X_all)[:, 1], 0.02, 0.98)
    grid["ml_score_sm"] = robust_normalize_01(
        smooth_on_regular_grid(grid, "ml_score", _grid_shape(grid), passes=config.SMOOTH_PASSES),
        0.02, 0.98,
    )
    return grid, model


def add_uncertainty(grid: gpd.GeoDataFrame, model) -> gpd.GeoDataFrame:
    """Добавить столбец ``uncertainty`` — разброс предсказаний членов модели по сетке.

    Высокая неопределённость = члены модели расходятся в оценке ячейки. Важно для
    геологоразведки: точечная оценка перспективности без меры разброса вводит в
    заблуждение. Источник разброса: для :class:`BackgroundEnsemble` — расхождение
    членов ансамбля (RF/GB по разным фонам), для Random Forest — разброс деревьев.
    Без обученной модели (или без доступа к членам) заполняется нулями.
    """
    if model is None:
        grid["uncertainty"] = 0.0
        return grid
    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    if hasattr(model, "member_probas"):          # BackgroundEnsemble
        member_probs = model.member_probas(X_all)
    elif hasattr(model, "estimators_"):          # Random Forest
        member_probs = np.stack([t.predict_proba(X_all)[:, 1] for t in model.estimators_])
    else:
        grid["uncertainty"] = 0.0
        return grid
    grid["uncertainty"] = robust_normalize_01(member_probs.std(axis=0), 0.02, 0.98)
    return grid


def compute_prospectivity(grid: gpd.GeoDataFrame, grid_shape: tuple[int, int]) -> gpd.GeoDataFrame:
    """Итоговая перспективность = сглаженный ``ml_score`` (без geo-стабилизатора).

    Также пишет ``local_bonus`` (для выделения золотых зон) и инвертированный
    ``prognoz``.
    """
    grid["local_bonus"] = robust_normalize_01(
        sum(w * grid[col] for col, w in config.LOCAL_BONUS_WEIGHTS.items()), 0.02, 0.98
    )
    grid["prospectivity_raw_sm"] = smooth_on_regular_grid(grid, "ml_score", grid_shape, passes=2)
    grid["prospectivity"] = robust_normalize_01(grid["prospectivity_raw_sm"], 0.02, 0.98)
    grid["prognoz"] = 1.0 - grid["prospectivity"]
    return grid


def mark_gold_zones(grid: gpd.GeoDataFrame, grid_shape: tuple[int, int], mask_union) -> gpd.GeoDataFrame:
    """Выделить золотые зоны: все компактные связные ядра (число — из данных).

    Относительный отбор (вместо абсолютных порогов): пул кандидатов — верхние
    ``1 − GOLD_SEED_Q`` доли площади по ``prospectivity``; золотыми становятся все
    связные пятна ≥ ``GOLD_ZONE_MIN_CELLS`` ячеек. Число зон НЕ навязывается —
    сколько реально сильных компактных ядер, столько и показывается; ``GOLD_MAX_ZONES``
    лишь ограничивает сверху (предохранитель от шумовой россыпи), отсекая слабейшие
    по суммарной силе (перспективность × мягкий бонус поддержки магматизмом и
    совпадением факторов). Пул относительный, поэтому слой не «пустеет» и не
    раздувается при смене набора признаков (например при добавлении DEM).
    """
    p = grid["prospectivity"].to_numpy()
    grid["gold_seed"] = (p >= float(np.quantile(p, config.GOLD_SEED_Q))).astype(int)

    # сила ячейки: перспективность с мягким бонусом за геологическую поддержку
    support = 0.5 * grid["prox_magm"].to_numpy() + 0.5 * grid["coincidence_score"].to_numpy()
    grid["_gold_strength"] = p * (0.5 + support)
    grid["gold_zone"] = top_n_components(
        grid, "gold_seed", "_gold_strength", grid_shape,
        min_cells=config.GOLD_ZONE_MIN_CELLS, n=config.GOLD_MAX_ZONES,
    ).astype(int)
    grid.drop(columns="_gold_strength", inplace=True)
    return grid


def point_top_coverage(grid: gpd.GeoDataFrame, positive_cells: list[int]) -> float:
    """Доля реальных точек, попавших в верхние (1 - POINT_COVERAGE_Q) прогноза."""
    if not positive_cells:
        return float("nan")
    thr = float(grid["prospectivity"].quantile(config.POINT_COVERAGE_Q))
    real = grid[grid["cell_id"].isin(positive_cells)]
    if len(real) == 0:
        return float("nan")
    return float((real["prospectivity"] >= thr).mean())


def _grid_shape(grid: gpd.GeoDataFrame) -> tuple[int, int]:
    """Восстановить форму растра по максимальным row/col сетки."""
    return int(grid["row"].max()) + 1, int(grid["col"].max()) + 1
