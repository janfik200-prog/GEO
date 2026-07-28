"""Модели-кандидаты для прогноза критериальной перспективности (эксперименты).

Единый интерфейс, чтобы честно сравнивать разные подходы на одной валидации:

    model = MODELS["som"]()          # фабрика по имени
    model.fit(X_train, y_train)
    score = model.predict_score(X)   # непрерывная оценка перспективности [0..1]

Общая пространственная валидация — :func:`spatial_cv` (GroupKFold по блокам,
как в :mod:`src.criterial_model`). Метрики: ROC-AUC, PR-AUC, lift@N%.

Первая модель — карта Кохонена (Self-Organizing Map). SOM сам по себе
безнадзорный (кластеризация + снижение размерности на 2D-решётку нейронов);
для прогноза перспективности поверх обучается калибровка: каждый нейрон
получает долю положительного класса среди обучающих ячеек, попавших в него,
и эта доля становится оценкой перспективности для всех ячеек нейрона.
Добавить свою модель — реализовать интерфейс и вписать в :data:`MODELS`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src import config


class ProspectivityModel(Protocol):
    """Единый интерфейс экспериментальной модели."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProspectivityModel": ...
    def predict_score(self, X: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Карта Кохонена (Self-Organizing Map), пакетный (batch) алгоритм обучения.
# --------------------------------------------------------------------------- #
class KohonenSOM:
    """Самоорганизующаяся карта Кохонена на регулярной 2D-решётке нейронов.

    Пакетное обучение (batch SOM): на каждой эпохе все ячейки относятся к своему
    нейрону-победителю (BMU), затем веса нейронов пересчитываются как взвешенное
    (гауссовым соседством по решётке) среднее приписанных наблюдений. Радиус
    соседства ``sigma`` линейно убывает. Признаки стандартизуются внутри модели.

    Параметры
    ---------
    grid : (rows, cols) — размер решётки нейронов.
    n_epochs : число эпох пакетного обучения.
    sigma0 : начальный радиус соседства (в узлах решётки); к концу -> ~0.5.
    random_state : сид инициализации весов.
    """

    def __init__(self, grid: tuple[int, int] = (10, 10), n_epochs: int = 30,
                 sigma0: float | None = None, random_state: int = 42) -> None:
        self.rows, self.cols = grid
        self.M = self.rows * self.cols
        self.n_epochs = n_epochs
        self.sigma0 = sigma0 if sigma0 is not None else max(self.rows, self.cols) / 2.0
        self.random_state = random_state
        # координаты нейронов на решётке + попарные расстояния по решётке
        gr, gc = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        self._pos = np.column_stack([gr.ravel(), gc.ravel()]).astype(float)  # (M, 2)
        self._grid_d2 = (
            (self._pos[:, None, :] - self._pos[None, :, :]) ** 2
        ).sum(axis=2)  # (M, M) квадраты расстояний по решётке
        self.mu_ = None
        self.sd_ = None
        self.W_ = None

    def _standardize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mu_ = X.mean(axis=0)
            self.sd_ = X.std(axis=0)
            self.sd_[self.sd_ == 0] = 1.0
        return (X - self.mu_) / self.sd_

    def _bmu(self, Z: np.ndarray) -> np.ndarray:
        # индекс ближайшего нейрона для каждой строки Z: argmin ||Z - W||^2
        d2 = (
            (Z ** 2).sum(axis=1)[:, None]
            - 2.0 * Z @ self.W_.T
            + (self.W_ ** 2).sum(axis=1)[None, :]
        )
        return np.argmin(d2, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KohonenSOM":
        rng = np.random.default_rng(self.random_state)
        Z = self._standardize(X, fit=True)
        # инициализация весов случайной выборкой наблюдений
        self.W_ = Z[rng.choice(len(Z), size=self.M, replace=True)].copy()
        for ep in range(self.n_epochs):
            sigma = self.sigma0 * (1.0 - ep / max(1, self.n_epochs)) + 0.5
            bmu = self._bmu(Z)                              # (n,)
            h = np.exp(-self._grid_d2[bmu] / (2.0 * sigma ** 2))  # (n, M) соседство
            num = h.T @ Z                                  # (M, d)
            den = h.sum(axis=0)[:, None]                   # (M, 1)
            den[den == 0] = 1e-12
            self.W_ = num / den
        return self

    def bmu(self, X: np.ndarray) -> np.ndarray:
        """Индексы нейронов-победителей для наблюдений."""
        return self._bmu(self._standardize(X, fit=False))

    def u_matrix(self) -> np.ndarray:
        """U-matrix: среднее расстояние весов нейрона до соседей по решётке (rows×cols).

        Светлые (большие) значения — «границы» между кластерами.
        """
        W = self.W_
        u = np.zeros(self.M)
        for i in range(self.M):
            neigh = np.where((self._grid_d2[i] > 0) & (self._grid_d2[i] <= 2.0))[0]
            if len(neigh):
                u[i] = np.linalg.norm(W[i] - W[neigh], axis=1).mean()
        return u.reshape(self.rows, self.cols)


class SomProspectivity:
    """SOM + калибровка нейронов долей положительного класса -> оценка перспективности.

    Реализует :class:`ProspectivityModel`: ``fit`` обучает SOM и запоминает по
    каждому нейрону среднее ``y`` обучающих ячеек; ``predict_score`` возвращает
    эту долю для нейрона-победителя каждой ячейки. Пустые нейроны получают
    глобальную базовую долю.
    """

    def __init__(self, grid: tuple[int, int] = (12, 12), n_epochs: int = 30,
                 random_state: int | None = None) -> None:
        self.som = KohonenSOM(
            grid=grid, n_epochs=n_epochs,
            random_state=random_state if random_state is not None else config.CRITERIAL_RANDOM_STATE,
        )
        self.neuron_score_ = None
        self.base_rate_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SomProspectivity":
        self.som.fit(X)
        bmu = self.som.bmu(X)
        self.base_rate_ = float(np.mean(y))
        score = np.full(self.som.M, self.base_rate_, dtype=float)
        for m in range(self.som.M):
            mask = bmu == m
            if mask.any():
                score[m] = float(y[mask].mean())
        self.neuron_score_ = score
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.neuron_score_[self.som.bmu(X)]

# --------------------------------------------------------------------------- #
# Супервизорные табличные модели (единый интерфейс fit/predict_score).
# --------------------------------------------------------------------------- #
class _SklearnProba:
    """База: обёртка над sklearn-классификатором. predict_score = P(класс 1)."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_score(self, X):
        return self.estimator.predict_proba(X)[:, 1]


def _logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    est = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            class_weight="balanced", max_iter=2000,
            random_state=config.CRITERIAL_RANDOM_STATE)),
    ])
    return _SklearnProba(est)


def _extratrees():
    from sklearn.ensemble import ExtraTreesClassifier
    return _SklearnProba(ExtraTreesClassifier(
        n_estimators=config.CRITERIAL_RF_TREES, max_depth=None,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, class_weight="balanced_subsample",
        random_state=config.CRITERIAL_RANDOM_STATE, n_jobs=-1))


def _knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return _SklearnProba(Pipeline([
        ("sc", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=25, weights="distance")),
    ]))


def _mlp():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return _SklearnProba(Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                              early_stopping=True, random_state=config.CRITERIAL_RANDOM_STATE)),
    ]))


class LightGBMModel:
    """LightGBM (градиентный бустинг). Нативно работает с NaN, обычно топ на табличке."""

    def __init__(self, **overrides):
        import lightgbm as lgb
        params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
                      random_state=config.CRITERIAL_RANDOM_STATE, n_jobs=-1, verbose=-1)
        params.update(overrides)
        self.clf = lgb.LGBMClassifier(**params)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_score(self, X):
        return self.clf.predict_proba(X)[:, 1]


class XGBoostModel:
    """XGBoost (градиентный бустинг, hist). scale_pos_weight под дисбаланс классов."""

    def __init__(self, **overrides):
        import xgboost as xgb
        self._xgb = xgb
        self._overrides = overrides

    def fit(self, X, y):
        pos = float((y == 1).sum()); neg = float((y == 0).sum())
        params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      eval_metric="logloss", scale_pos_weight=(neg / max(pos, 1.0)),
                      random_state=config.CRITERIAL_RANDOM_STATE, n_jobs=-1)
        params.update(self._overrides)
        self.clf = self._xgb.XGBClassifier(**params)
        self.clf.fit(X, y)
        return self

    def predict_score(self, X):
        return self.clf.predict_proba(X)[:, 1]


class RfGbBaseline:
    """Ансамбль RF+GB из боевого src.criterial_model — для сравнения в одном ряду."""

    def __init__(self):
        from src.criterial_model import RfGbEnsemble
        self.model = RfGbEnsemble()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_score(self, X):
        return self.model.predict_proba1(X)


# --------------------------------------------------------------------------- #
# Безнадзорные (кластеризация) + калибровка долей класса — семейство SOM.
# --------------------------------------------------------------------------- #
class _ClusterCalibrated:
    """База: кластеризация + оценка перспективности = доля класса 1 в кластере."""

    def __init__(self, clusterer, soft=False):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.clusterer = clusterer
        self.soft = soft            # мягкое отнесение (GMM) vs жёсткое (k-means)
        self.cluster_score_ = None
        self.base_rate_ = 0.0

    def fit(self, X, y):
        Z = self.scaler.fit_transform(X)
        self.clusterer.fit(Z)
        self.base_rate_ = float(np.mean(y))
        if self.soft:
            R = self.clusterer.predict_proba(Z)              # (n, k) ответственности
            w = R.sum(axis=0)
            self.cluster_score_ = (R * y[:, None]).sum(axis=0) / np.where(w > 0, w, 1)
        else:
            lab = self.clusterer.predict(Z)
            k = self.clusterer.n_clusters
            sc = np.full(k, self.base_rate_)
            for c in range(k):
                mask = lab == c
                if mask.any():
                    sc[c] = float(y[mask].mean())
            self.cluster_score_ = sc
        return self

    def predict_score(self, X):
        Z = self.scaler.transform(X)
        if self.soft:
            return self.clusterer.predict_proba(Z) @ self.cluster_score_
        return self.cluster_score_[self.clusterer.predict(Z)]


def _kmeans(n_clusters=25):
    from sklearn.cluster import KMeans
    return _ClusterCalibrated(
        KMeans(n_clusters=n_clusters, n_init=10, random_state=config.CRITERIAL_RANDOM_STATE),
        soft=False)


def _gmm(n_components=20):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_components, covariance_type="diag",
                         random_state=config.CRITERIAL_RANDOM_STATE)
    gm.n_clusters = n_components  # для единообразия
    return _ClusterCalibrated(gm, soft=True)


# Реестр моделей: сюда добавляются новые кандидаты (единый интерфейс fit/predict_score).
MODELS: dict = {
    "logreg":     _logreg,            # логистическая регрессия (elastic-net) — интерпретируемый пол
    "lgbm":       LightGBMModel,      # LightGBM — быстрый максимум качества
    "xgb":        XGBoostModel,       # XGBoost
    "rfgb":       RfGbBaseline,       # боевой RF+GB (src) — сильный baseline
    "extratrees": _extratrees,        # Extremely Randomized Trees
    "knn":        _knn,               # k ближайших соседей
    "mlp":        _mlp,               # многослойный перцептрон (табличный)
    "kmeans":     _kmeans,            # k-means + калибровка (семейство SOM)
    "gmm":        _gmm,               # гауссова смесь + мягкая калибровка
    "som":        SomProspectivity,   # карта Кохонена
}


def available_models() -> list[str]:
    """Имена моделей, чьи зависимости реально установлены (lgbm/xgb могут отсутствовать)."""
    ok = []
    for name, factory in MODELS.items():
        try:
            factory()  # пробная инициализация (ленивый импорт внутри)
            ok.append(name)
        except Exception:
            pass
    return ok


# --------------------------------------------------------------------------- #
# Общая пространственная валидация (как в src.criterial_model, но для любой модели).
# --------------------------------------------------------------------------- #
def lift_at(y_true: np.ndarray, score: np.ndarray, area: float) -> float:
    n = len(score)
    k = max(1, int(round(area * n)))
    top = np.argsort(-score)[:k]
    base = y_true.mean()
    return float(y_true[top].mean() / base) if base > 0 else float("nan")


def spatial_cv(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_splits: int | None = None, **model_kwargs) -> dict:
    """Out-of-fold прогноз выбранной модели по пространственным блокам + метрики."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_splits = int(n_splits or config.CRITERIAL_N_SPLITS)
    factory = MODELS[model_name]
    oof = np.full(len(y), np.nan)
    per_fold = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = factory(**model_kwargs).fit(X[tr], y[tr])
        p = m.predict_score(X[te])
        oof[te] = p
        per_fold.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[te], p)) if len(np.unique(y[te])) > 1 else float("nan"),
        })
    return {
        "model": model_name,
        "oof_score": oof,
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "lift": {a: lift_at(y, oof, a) for a in config.CRITERIAL_LIFT_AREAS},
        "per_fold": per_fold,
    }


# --------------------------------------------------------------------------- #
# Tier 2: U-Net по растру (свёрточная сеть, пространственный контекст).
#
# Единственная модель, использующая ПРОСТРАНСТВЕННУЮ структуру: вход — весь лист
# как многоканальное изображение (C=число признаков, H×W = сетка), выход —
# карта вероятности перспективности. Ловит текстуру геофизических полей, которую
# пиксельные модели (RF/SOM/GBM) не видят.
#
# Не вписана в MODELS/spatial_cv (там интерфейс по строкам-ячейкам, а сети нужен
# растр). Своя блочная кросс-валидация — :func:`spatial_cv_unet`.
#
# ТРЕБУЕТ PyTorch (`pip install torch`). В текущей сборочной среде torch скачать
# не удалось (заблокирован индекс), поэтому код предоставлен, но здесь не
# прогонялся — запускать на машине с установленным torch.
# --------------------------------------------------------------------------- #
def _build_unet(n_ch: int, base: int = 32):
    import torch
    import torch.nn as nn

    def conv(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = conv(n_ch, base)
            self.e2 = conv(base, base * 2)
            self.pool = nn.MaxPool2d(2)
            self.b = conv(base * 2, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = conv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = conv(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            import torch.nn.functional as F
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            b = self.b(self.pool(e2))
            d2 = self.up2(b)
            d2 = F.pad(d2, (0, e2.shape[-1] - d2.shape[-1], 0, e2.shape[-2] - d2.shape[-2]))
            d2 = self.d2(torch.cat([d2, e2], 1))
            d1 = self.up1(d2)
            d1 = F.pad(d1, (0, e1.shape[-1] - d1.shape[-1], 0, e1.shape[-2] - d1.shape[-2]))
            d1 = self.d1(torch.cat([d1, e1], 1))
            return self.out(d1)

    return UNet()


def _rasterize(data, values, fill=0.0):
    """Разложить вектор по (row, col) в растр grid_shape."""
    g = np.full(data.grid_shape, fill, dtype=np.float32)
    r = data.frame["row"].to_numpy().astype(int)
    c = data.frame["col"].to_numpy().astype(int)
    g[r, c] = values
    return g


def spatial_cv_unet(data, n_splits: int | None = None, epochs: int = 200,
                    lr: float = 1e-3, base: int = 32, seed: int = 42) -> dict:
    """Блочная пространственная CV для U-Net по растру. Требует torch.

    На каждом фолде тестовые ячейки маскируются из функции потерь (обучение
    только по train-ячейкам, пиксельный BCE), затем сеть предсказывает весь лист,
    а OOF-оценки собираются по test-ячейкам. Метрики — как в :func:`spatial_cv`.
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.manual_seed(seed)
    n_splits = int(n_splits or config.CRITERIAL_N_SPLITS)
    H, W = data.grid_shape
    C = data.X.shape[1]

    # каналы признаков -> тензор (1, C, H, W), стандартизация по каналу
    chans = []
    for j in range(C):
        g = _rasterize(data, data.X[:, j], fill=np.nan)
        m, sd = np.nanmean(g), np.nanstd(g) or 1.0
        g = np.nan_to_num((g - m) / sd, nan=0.0)
        chans.append(g)
    Xr = torch.tensor(np.stack(chans)[None], dtype=torch.float32)  # (1,C,H,W)

    r = data.frame["row"].to_numpy().astype(int)
    c = data.frame["col"].to_numpy().astype(int)
    y_grid = torch.tensor(_rasterize(data, data.y.astype(np.float32)), dtype=torch.float32)
    valid = torch.tensor(_rasterize(data, np.ones(len(data.y), np.float32)), dtype=torch.float32)

    oof = np.full(len(data.y), np.nan)
    per_fold = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(data.X, data.y, data.groups)):
        train_mask = np.zeros((H, W), np.float32)
        train_mask[r[tr], c[tr]] = 1.0
        tm = torch.tensor(train_mask) * valid
        net = _build_unet(C, base=base)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        pos = data.y[tr].mean()
        pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = net(Xr)[0, 0]
            loss = (lossf(logits, y_grid) * tm).sum() / tm.sum()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            prob = torch.sigmoid(net(Xr)[0, 0]).numpy()
        oof[te] = prob[r[te], c[te]]
        per_fold.append({"fold": fold, "roc_auc": float(roc_auc_score(data.y[te], oof[te]))})

    return {
        "model": "unet",
        "oof_score": oof,
        "roc_auc": float(roc_auc_score(data.y, oof)),
        "pr_auc": float(average_precision_score(data.y, oof)),
        "lift": {a: lift_at(data.y, oof, a) for a in config.CRITERIAL_LIFT_AREAS},
        "per_fold": per_fold,
    }


# --------------------------------------------------------------------------- #
# Анализ признаков: корреляции и важность (permutation / нативная).
# --------------------------------------------------------------------------- #
def feature_correlations(data, method: str = "spearman") -> list[tuple]:
    """Корреляция каждого признака с таргетом.

    Возвращает отсортированный по |corr| список
    ``(признак, corr_с_критериальным, corr_с_классом_y)``. ``corr_с_критериальным``
    — связь с непрерывным критериальным значением (по умолчанию Spearman, так как
    зависимости нелинейные), ``corr_с_классом_y`` — с бинарным таргетом.
    Знак у критериального инвертирован: критериальный «меньше = лучше», поэтому
    возвращается корреляция с перспективностью (−crit), чтобы «плюс = помогает».
    """
    import pandas as pd
    df = pd.DataFrame(data.X, columns=data.feature_cols)
    prosp = -data.crit                       # больше = перспективнее
    y = data.y.astype(float)
    out = []
    for c in data.feature_cols:
        rc = df[c].corr(pd.Series(prosp), method=method)
        ry = df[c].corr(pd.Series(y), method=method)
        out.append((c, float(rc), float(ry)))
    out.sort(key=lambda t: -abs(t[1]))
    return out


def feature_corr_matrix(data, method: str = "spearman"):
    """Матрица корреляций признак-признак (pandas DataFrame) — для теплокарты."""
    import pandas as pd
    return pd.DataFrame(data.X, columns=data.feature_cols).corr(method=method)


def permutation_importance(model, X, y, feature_names, n_repeats: int = 5,
                           seed: int = 42) -> list[tuple]:
    """Модель-агностичная важность: падение ROC-AUC при перемешивании признака.

    Работает для ЛЮБОЙ модели с ``predict_score`` (в т.ч. SOM, kNN, кластеры).
    ``model`` должна быть уже обучена. Возвращает отсортированный список
    ``(признак, среднее_падение_AUC, std)``. Больше падение — важнее признак.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_score(X))
    res = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, model.predict_score(Xp)))
        res.append((name, float(np.mean(drops)), float(np.std(drops))))
    res.sort(key=lambda t: -t[1])
    return res


def native_importance(model, feature_names) -> list[tuple] | None:
    """Встроенная важность признаков, если модель её даёт (деревья/бустинг/линейная).

    Ищет ``feature_importances_`` (леса/бустинг) или ``coef_`` (линейная) на самой
    модели и типичных вложенных атрибутах (``.clf``, ``.estimator``, пайплайн).
    Возвращает отсортированный список ``(признак, важность)`` или ``None``.
    """
    candidates = [model, getattr(model, "clf", None), getattr(model, "estimator", None),
                  getattr(model, "model", None)]
    est = None
    for o in candidates:
        if o is None:
            continue
        if hasattr(o, "feature_importances_") or hasattr(o, "coef_"):
            est = o
            break
        steps = getattr(o, "named_steps", None)  # sklearn Pipeline
        if steps:
            for step in steps.values():
                if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                    est = step
                    break
        if est is not None:
            break
    if est is None:
        return None
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    else:
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    pairs = sorted(zip(feature_names, imp), key=lambda t: -t[1])
    return [(n, float(v)) for n, v in pairs]
