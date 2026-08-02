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


def test_expand_to_grid_keeps_invalid_out_of_top():
    valid = np.array([True, False, True])
    full = atypicality.expand_to_grid(np.array([0.5, 0.9]), valid)
    assert np.isneginf(full[1])
    assert full[0] == 0.5 and full[2] == 0.9
