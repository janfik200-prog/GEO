"""Юниты глубоких моделей без учителя (этап 7b) — синтетика, без данных.

Проверяется ровно то, ради чего эти модели берутся: во вживлённом в фон
компактном «теле» с иной сигнатурой скор обязан быть выше, чем в фоне. Точность
не проверяется — на синтетике она ничего не сказала бы о листе.
"""
import numpy as np
import pytest

from src import deep_unsup as du

N_BG, N_BODY, D = 1500, 40, 8
EPOCHS = 40


def _data(seed=0, contrast=4.0):
    """Коррелированный гауссов фон + компактное тело со сдвигом сигнатуры."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((D, D))
    cov = a @ a.T / D + np.eye(D) * 0.5
    X = rng.multivariate_normal(np.zeros(D), cov, size=N_BG + N_BODY)
    body = np.arange(N_BG, N_BG + N_BODY)
    X[body, :3] += contrast
    X = (X - X.mean(0)) / X.std(0)
    return X, body


def _top_share(score, body, area=0.10):
    """Доля тела, попавшая в верхние ``area`` скора."""
    thr = np.quantile(score, 1.0 - area)
    return float((score[body] >= thr).mean())


@pytest.mark.parametrize("name", ["vae", "dagmm", "svdd"])
def test_detector_finds_implanted_body(name):
    """Каждый детектор обязан вытащить вживлённое тело в верхние 10%."""
    X, body = _data()
    score = du.METHODS[name](X, seed=0, epochs=EPOCHS)
    assert np.isfinite(score).all(), f"{name}: нефинитные значения скора"
    share = _top_share(score, body)
    assert share > 0.5, f"{name}: в top-10% попало лишь {share:.0%} тела"


def test_dec_returns_two_distinct_scores():
    """DEC отдаёт расстояние и редкость — это разные величины, не копии."""
    X, body = _data()
    out = du.dec_scores(X, seed=0, epochs=EPOCHS)
    assert set(out) == {"dec_dist", "dec_rarity"}
    for k, v in out.items():
        assert np.isfinite(v).all(), f"{k}: нефинитные значения"
    rho = np.corrcoef(out["dec_dist"], out["dec_rarity"])[0, 1]
    assert abs(rho) < 0.99, "расстояние и редкость выродились в одно и то же"


def test_dec_rarity_marks_rare_cluster():
    """Редкость обязана быть выше у малочисленной группы, чем у фона."""
    X, body = _data()
    out = du.dec_scores(X, seed=0, epochs=EPOCHS)
    assert out["dec_rarity"][body].mean() > out["dec_rarity"].mean()


def test_svdd_does_not_collapse():
    """Вырождение Deep SVDD: константный выход = нулевой разброс скора."""
    X, _ = _data()
    score = du.deep_svdd_score(X, seed=0, epochs=EPOCHS)
    assert score.std() > 1e-6, "гиперсфера схлопнулась в точку"


def test_dagmm_energy_is_not_degenerate():
    """Энергия DAGMM обязана различать ячейки, а не быть константой."""
    X, _ = _data()
    score = du.dagmm_score(X, seed=0, epochs=EPOCHS)
    assert np.unique(np.round(score, 6)).size > 100


def test_dagmm_survives_degenerate_pool():
    """Вырожденный пул (признаки — линейные комбинации трёх) не роняет Холецкого.

    Ровно на этом упал прогон на пуле эмбеддинга MAE: ковариация компоненты
    смеси теряла положительную определённость, и разложение падало с ошибкой.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal((N_BG, 3))
    X = base @ rng.standard_normal((3, D))          # ранг 3 при D столбцах
    X += rng.standard_normal(X.shape) * 1e-6        # шум ниже уровня сигнала
    score = du.dagmm_score(X.astype(float), seed=0, epochs=EPOCHS)
    assert np.isfinite(score).all()


def test_ensemble_averages_ranks():
    """Ансамбль по семенам: ранговая шкала [0, 1] и стабильнее одиночного семени."""
    X, body = _data()
    ens, per = du.ensemble_score("vae", X, seeds=(0, 1, 2), epochs=EPOCHS)
    assert set(ens) == {"vae"}
    v = ens["vae"]
    assert v.min() >= 0.0 and v.max() <= 1.0
    assert len(per["vae"]) == 3
    singles = [_top_share(s, body) for s in per["vae"]]
    assert _top_share(v, body) >= min(singles) - 1e-9


def test_seeds_are_reproducible():
    """Одно семя — один результат: иначе сравнение конфигураций бессмысленно."""
    X, _ = _data()
    a = du.vae_score(X, seed=3, epochs=20)
    b = du.vae_score(X, seed=3, epochs=20)
    assert np.allclose(a, b)
