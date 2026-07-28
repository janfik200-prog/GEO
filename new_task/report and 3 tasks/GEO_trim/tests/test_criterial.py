"""Смоук-тесты критериального пайплайна (таргет = критериальный анализ, не геохимия)."""
import numpy as np
import pytest

from src import config
from src.criterial_data import load_criterial_dataset
from src import criterial_model as cm


@pytest.fixture(scope="module")
def data():
    return load_criterial_dataset()


def test_shapes_and_target(data):
    assert data.grid_shape == (154, 149)
    n_feat = len(data.feature_cols)
    assert data.X.shape == (22946, n_feat)
    # 28 базовых + 5 бинарных индикаторов + 7 производных + 4 терренных = 44
    assert n_feat == 44
    assert not np.isnan(data.X).any()


def test_target_polarity_and_balance(data):
    # положительный класс = НАИМЕНЬШИЕ значения критериального (меньше = перспективнее)
    assert abs(data.y.mean() - config.CRITERIAL_TOP_FRAC) < 0.02
    assert data.crit[data.y == 1].max() <= data.crit[data.y == 0].min() + 1e-9


def test_no_geochem_features(data):
    joined = " ".join(data.feature_cols).lower()
    for banned in ("геохим", "geochem", "opro", "ореол", "halo"):
        assert banned not in joined


def test_factor_features_excluded(data):
    # непрерывные dist_* факторы критерия не должны попасть (прямые входы формулы).
    # Плотности dens_* возвращаются намеренно как производные гео-признаки.
    for f in config.CRITERIAL_EXCLUDE_FEATURES:
        if f.startswith("dist_"):
            assert f not in data.feature_cols


def test_geo_indicators_present(data):
    # бинарные гео-признаки: разломы/фации/палеодолины и т.д.
    for f in ("in_faults", "in_facies", "in_paleo"):
        assert f in data.feature_cols
        vals = data.frame[f].to_numpy()
        assert set(np.unique(vals)).issubset({0, 1})


def test_geo_derived_present(data):
    # производные: пересечения, coincidence, плотности
    for f in ("x_faults_paleo", "coincidence", "dens_tect", "dens_magm"):
        assert f in data.feature_cols
    # пересечение = И индикаторов
    inter = data.frame["x_faults_paleo"].to_numpy()
    both = (data.frame["in_faults"].to_numpy() & data.frame["in_paleo"].to_numpy())
    assert np.array_equal(inter, both)


def test_lift_helper():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert cm.lift_at(y, score, 0.2) == pytest.approx(5.0)


def test_model_learns_signal(data):
    import src.config as C
    C.CRITERIAL_RF_TREES, C.CRITERIAL_GB_TREES = 60, 30
    model = cm.fit_full(data)
    p = model.predict_proba1(data.X)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(data.y, p) > 0.9


def test_terrain_features_present(data):
    # производные рельефа из relief_m
    for f in ("relief_slope", "relief_curv", "relief_tpi", "relief_rough"):
        assert f in data.feature_cols
