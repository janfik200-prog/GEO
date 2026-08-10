"""Тесты src/optical_derived.py: формулы индексов на синтетике (без сети)."""
import numpy as np
import pandas as pd

from src import optical_derived as od


def test_ndre_known_values():
    df = pd.DataFrame({
        "s2_b04": [0.1, 0.1], "s2_b05": [0.2, 0.0],
        "s2_b06": [0.3, 1.0], "s2_b07": [0.35, 1.0], "s2_b8a": [0.4, 0.0],
    })
    out = od.red_edge_indices(df)
    # (0.4-0.2)/(0.4+0.2) = 0.3333
    assert np.isclose(out["s2_ndre"][0], 1.0 / 3.0, atol=1e-4)
    # обе полосы нулевые -> 0/0 -> NaN, не деление на ноль
    assert np.isnan(out["s2_ndre"][1])


def test_ireci_known_value():
    df = pd.DataFrame({
        "s2_b04": [0.10], "s2_b05": [0.20],
        "s2_b06": [0.40], "s2_b07": [0.50], "s2_b8a": [0.45],
    })
    out = od.red_edge_indices(df)
    # (0.50-0.10)/(0.20/0.40) = 0.40/0.50 = 0.8
    assert np.isclose(out["s2_ireci"][0], 0.8, atol=1e-4)


def test_aster_mgoh_index_known_value_and_zero_guard():
    df = pd.DataFrame({"ast_b08": [0.6, 0.5], "ast_b09": [0.3, 0.0]})
    out = od.aster_mgoh_index(df)
    assert np.isclose(out[0], 2.0, atol=1e-4)
    assert np.isnan(out[1])


def test_optical_derived_features_preserves_row_order_and_columns():
    n = 5
    df = pd.DataFrame({
        "s2_b04": np.linspace(0.05, 0.15, n), "s2_b05": np.linspace(0.1, 0.2, n),
        "s2_b06": np.linspace(0.2, 0.3, n), "s2_b07": np.linspace(0.25, 0.35, n),
        "s2_b8a": np.linspace(0.3, 0.4, n),
        "ast_b08": np.linspace(0.4, 0.5, n), "ast_b09": np.linspace(0.3, 0.4, n),
    })
    out = od.optical_derived_features(df)
    assert list(out.columns) == list(od.OPTICAL_DERIVED_COLS)
    assert len(out) == n
