"""Смоук-тесты экспериментальной площадки (карта Кохонена)."""
import numpy as np

from experiment import model as M


def _toy():
    rng = np.random.default_rng(0)
    # два разделимых облака: класс 1 смещён по признакам
    X0 = rng.normal(0, 1, size=(400, 6))
    X1 = rng.normal(2.5, 1, size=(120, 6))
    X = np.vstack([X0, X1]); y = np.r_[np.zeros(400), np.ones(120)].astype(int)
    groups = np.arange(len(y)) % 5  # 5 псевдо-блоков
    return X, y, groups


def test_registry_has_expected_models():
    for name in ("logreg", "lgbm", "xgb", "rfgb", "extratrees", "knn", "mlp",
                 "kmeans", "gmm", "som"):
        assert name in M.MODELS


def test_available_models_runs():
    avail = M.available_models()
    assert "som" in avail and "logreg" in avail  # эти без внешних зависимостей


def test_tabular_models_separate_classes():
    X, y, groups = _toy()
    for name in ("logreg", "extratrees", "kmeans", "gmm"):
        r = M.spatial_cv(name, X, y, groups, n_splits=3)
        assert r["roc_auc"] > 0.75, name


def test_som_fit_predict_shape():
    X, y, _ = _toy()
    m = M.MODELS["som"](grid=(6, 6), n_epochs=10).fit(X, y)
    s = m.predict_score(X)
    assert s.shape == (len(y),)
    assert np.all((s >= 0) & (s <= 1))


def test_som_separates_classes():
    X, y, groups = _toy()
    res = M.spatial_cv("som", X, y, groups, n_splits=5, grid=(6, 6), n_epochs=15)
    assert res["roc_auc"] > 0.8  # разделимые облака -> уверенно выше случайного


def test_u_matrix_shape():
    X, y, _ = _toy()
    m = M.MODELS["som"](grid=(6, 6), n_epochs=10).fit(X, y)
    assert m.som.u_matrix().shape == (6, 6)


def test_feature_correlations_and_importance():
    X, y, _ = _toy()
    # смок для корреляции/важности на игрушечном датасете через SOM-обёртку данных
    class D:  # мини-датасет с нужными полями
        pass
    d = D(); d.X = X; d.y = y
    d.feature_cols = [f"f{i}" for i in range(X.shape[1])]
    d.crit = -y.astype(float)  # чтобы -crit ~ y
    cor = M.feature_correlations(d)
    assert len(cor) == X.shape[1] and len(cor[0]) == 3
    m = M.MODELS["logreg"]().fit(X, y)
    perm = M.permutation_importance(m, X, y, d.feature_cols, n_repeats=2)
    assert len(perm) == X.shape[1]
    assert M.native_importance(m, d.feature_cols) is not None  # у логрегрессии есть coef_
