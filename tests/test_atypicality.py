"""Тесты src/atypicality.py: каждая ступень находит вброшенные выбросы (синтетика)."""
import numpy as np
import pandas as pd
import pytest

from src import atypicality


N, D, N_OUT = 2000, 8, 20


@pytest.fixture(scope="module")
def toy():
    """Низкоранговый фон (3 латентных фактора + шум) + 20 выбросов вне
    многообразия в последних строках. Низкий ранг важен: PCA-невязка и AE
    детектируют именно уход с многообразия фона, а не сдвиг вдоль него."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(N, 3))
    W = rng.normal(size=(3, D))
    X = latent @ W + rng.normal(scale=0.1, size=(N, D))
    X[-N_OUT:] += rng.choice([-6.0, 6.0], size=(N_OUT, D))
    return X


def _top_share(score, k=N_OUT):
    """Доля вброшенных выбросов в топ-k скора."""
    top = np.argsort(score)[-k:]
    return np.isin(top, np.arange(N - N_OUT, N)).mean()


@pytest.mark.parametrize("fn", [
    atypicality.robust_mahalanobis,
    atypicality.pca_residual,
    atypicality.isoforest_score,
    atypicality.ocsvm_score,
    atypicality.lof_score,
    atypicality.knn_distance_score,
    atypicality.gmm_nll_score,
])
def test_each_rung_ranks_outliers_top(toy, fn):
    assert _top_share(fn(toy)) >= 0.9


def test_shallow_ae_ranks_outliers_top(toy):
    # AE стохастичнее линейных ступеней — планка мягче
    assert _top_share(atypicality.shallow_ae_score(toy)) >= 0.7


def test_rank_ensemble_not_worse_than_members(toy):
    scores = {
        "maha": atypicality.robust_mahalanobis(toy),
        "pca": atypicality.pca_residual(toy),
    }
    ens = atypicality.rank_ensemble(scores)
    assert ens.min() >= 0 and ens.max() <= 1
    assert _top_share(ens) >= min(_top_share(s) for s in scores.values())


def test_per_domain_finds_within_domain_outlier():
    # два домена с разными средними; выброс внутри домена A типичен для B
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=(500, 4))
    b = rng.normal(8.0, 1.0, size=(500, 4))
    a[0] = 4.0  # аномален для A (4 сигмы), неотличим от «между доменами»
    X = np.vstack([a, b])
    domains = np.repeat([0, 1], 500)
    dom_score = atypicality.per_domain(atypicality.robust_mahalanobis, X, domains)
    assert dom_score[0] > np.quantile(dom_score[1:500], 0.99)


def test_prepare_matrix_imputes_median_and_scales():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan, 100.0],
                       "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    valid = np.array([True, True, True, True, False])
    X = atypicality.prepare_matrix(df, valid)
    assert X.shape == (4, 2)
    assert not np.isnan(X).any()
    # импутированное значение = медиана колонки -> после центрирования ~0
    assert abs(X[3, 0]) < 1e-9


def test_prepare_matrix_iqr_before_imputation():
    # IQR считается ДО импутации: медианные вставки не сжимают масштаб
    df = pd.DataFrame({"a": [0.0, 10.0, np.nan, np.nan],
                       "b": [1.0, 2.0, 3.0, 4.0]})
    valid = np.ones(4, dtype=bool)
    X = atypicality.prepare_matrix(df, valid)
    # nanIQR колонки a = 5 (по [0, 10]); после импутации был бы 2.5 -> X[1,0]=2
    assert abs(X[1, 0] - 1.0) < 1e-9
    assert abs(X[2, 0]) < 1e-9              # импутированная медиана -> 0


def test_prepare_matrix_does_not_explode_on_zero_inflated_column():
    # Плотность узлов линеаментов: ноль в 90% ячеек -> IQR вырожден.
    # До правки деление на IQR давало z ~ 1e14, и колонка одна определяла
    # расстояние Махаланобиса.
    rng = np.random.default_rng(11)
    n = 500
    sparse = np.zeros(n)
    sparse[:50] = rng.uniform(1.0, 3.0, size=50)
    df = pd.DataFrame({"lin_node_dens": sparse, "gm": rng.normal(size=n)})
    valid = np.ones(n, dtype=bool)
    with pytest.warns(UserWarning, match="IQR"):
        X = atypicality.prepare_matrix(df, valid)
    assert np.isfinite(X).all()
    assert np.abs(X[:, 0]).max() < 20.0
    # масштабы колонок сопоставимы: ни одна не задавит остальные в Махаланобисе
    assert 0.05 < X[:, 0].std() / X[:, 1].std() < 20.0


def test_feature_contributions_finds_planted_feature():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    names = [f"f{i}" for i in range(5)]
    score = X[:, 2] + rng.normal(scale=0.05, size=400)   # аномальность гонит f2
    contrib = atypicality.feature_contributions(X, score, names, top_frac=0.10)
    assert contrib.loc[0, "feature"] == "f2"
    assert (contrib["abs_dz"].to_numpy()[:-1] >= contrib["abs_dz"].to_numpy()[1:]).all()


def test_shallow_ae_return_info_n_iter(toy):
    score, info = atypicality.shallow_ae_score(toy, return_info=True)
    assert score.shape == (N,)
    assert info["n_iter"] >= 1


def test_expand_to_grid_keeps_invalid_out_of_top():
    valid = np.array([True, False, True])
    full = atypicality.expand_to_grid(np.array([0.5, 0.9]), valid)
    assert np.isneginf(full[1])
    assert full[0] == 0.5 and full[2] == 0.9
