"""Тесты src/transfer_nn.py: доменная адаптация и способность сети выучить
направление (синтетика, без сети и без реальных данных)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src import transfer_nn  # noqa: E402


def test_quantile_match_moves_to_reference_marginal_and_keeps_order():
    rng = np.random.default_rng(0)
    x = rng.normal(100, 3, 500)          # «наши» единицы
    ref = rng.normal(0, 1, 4000)         # единицы обучающей выборки
    out = transfer_nn.quantile_match(x, ref)
    # маргинал переехал в распределение ref
    assert abs(np.median(out)) < 0.2 and abs(out.std() - 1.0) < 0.25
    # порядок ячеек сохранён — от него и зависит lift@10%
    assert np.array_equal(np.argsort(x), np.argsort(out))


def test_quantile_match_keeps_nan_and_survives_empty_reference():
    x = np.array([1.0, np.nan, 3.0, 2.0])
    out = transfer_nn.quantile_match(x, np.array([10.0, 20.0, 30.0]))
    assert np.isnan(out[1]) and np.isfinite(out[[0, 2, 3]]).all()
    assert np.isnan(transfer_nn.quantile_match(x, np.array([np.nan]))).all()


def test_spatial_groups_are_blocks_not_points():
    lon = np.array([-120.4, -120.1, -119.0, -110.0])
    lat = np.array([40.1, 40.9, 40.2, 50.0])
    g = transfer_nn.spatial_groups(lon, lat, deg=2.0)
    assert g[0] == g[1]              # обе в блоке lon[-122,-120) lat[40,42)
    assert g[0] != g[2] != g[3]


def test_robust_stats_ignore_outliers_and_clip():
    X = np.repeat(np.arange(100.0)[:, None], 2, axis=1)
    X[0, 0] = 1e9                                   # выброс грида
    med, sc = transfer_nn.robust_stats(X)
    assert abs(med[0] - med[1]) < 2 and sc.min() > 0
    Z = transfer_nn.apply_stats(X, med, sc, clip=5.0)
    assert np.abs(Z).max() <= 5.0 and Z.dtype == np.float32


def test_harmonize_local_uses_local_analogues(monkeypatch):
    """Канал с локальным аналогом обязан браться из нашего слоя, а не из грида."""
    n = 200
    df = pd.DataFrame({src: np.arange(n, dtype=float)
                       for src in transfer_nn.LOCAL_ANALOGUE.values() if src})
    Xtr = np.tile(np.linspace(-3, 3, 1000)[:, None], (1, 5))
    called = {"global": 0}

    def fake_sample(ds, lon, lat):
        called["global"] += 1
        return np.zeros(len(lon))

    monkeypatch.setattr(transfer_nn, "sample_global", fake_sample)
    M = transfer_nn.harmonize_local(df, Xtr, np.zeros(n), np.zeros(n))
    assert M.shape == (n, 5)
    # ровно один канал (geoid) без локального аналога -> один поход в глобальный грид
    assert called["global"] == 1
    j = list(transfer_nn.LOCAL_ANALOGUE).index("mag4km")
    assert np.array_equal(np.argsort(M[:, j]), np.argsort(df["gm_mg_all"].to_numpy()))


def test_net_is_tiny_and_learns_direction():
    """Ключевая проверка: сеть обязана выучить, КАКОЙ конец шкалы благоприятен.

    Именно этого не может дать обучение без учителя (этап 5): детектор
    нетипичности одинаково подсветил бы оба хвоста, а классификатор на метках —
    только правильный.
    """
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(0, 1, (n, 5))
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    tr, te = np.arange(n) < n * 3 // 4, np.arange(n) >= n * 3 // 4
    net = transfer_nn.FertilityNet(5, seed=0)
    assert net.n_params() < 5000
    hist = net.fit(X[tr], y[tr], X[te], y[te], epochs=40, seed=0)
    assert hist["val_loss"].iloc[-5:].min() < hist["val_loss"].iloc[0]

    p = net.predict(X[te])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[te], p) > 0.85
    # направление, а не «необычность»: высокий X0 -> высокий скор, низкий -> низкий
    hi = X[te][:, 0] > 1.5
    lo = X[te][:, 0] < -1.5
    assert p[hi].mean() > p[lo].mean() + 0.3


def test_predict_returns_nan_for_incomplete_rows():
    net = transfer_nn.FertilityNet(5, seed=0)
    X = np.zeros((4, 5))
    X[2, 1] = np.nan
    p = net.predict(X)
    assert np.isnan(p[2]) and np.isfinite(p[[0, 1, 3]]).all()
