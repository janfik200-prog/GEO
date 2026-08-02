"""Тесты src/cell_mask.py: маска валидных ячеек и домены (синтетика)."""
import numpy as np
import pandas as pd

from src import cell_mask, config


def _toy_df(n=8):
    """Мини-датасет с реальными именами колонок из реестра признаков."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
        "gm_gr_all": rng.normal(size=n),
        "gm_mg_all": rng.normal(size=n),
        "ls_ch1": rng.normal(size=n),
        "relief_m": rng.normal(size=n),
        "dist_dnl": np.full(n, 100.0),
        "dist_dnara": np.full(n, 500.0),
        "dist_tect1": rng.normal(size=n),   # факторный — не признак фазы
        "mask_svita": np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=float),
    })
    return df


def test_feature_columns_exclude_coords_aux_factors():
    cols = cell_mask.feature_columns(_toy_df())
    assert "row" not in cols and "x" not in cols
    assert "mask_svita" not in cols            # служебный (GOLD_FEATURES_AUX)
    assert "dist_tect1" not in cols            # факторный (CRIT_EXCLUDE_FACTOR_FEATURES)
    assert {"gm_gr_all", "ls_ch1", "relief_m", "dist_dnl", "dist_dnara"} <= set(cols)


def test_valid_mask_drops_water_and_nan_heavy_cells():
    df = _toy_df()
    df.loc[2, "dist_dnara"] = 0.0                        # вода
    feat = cell_mask.feature_columns(df)
    df.loc[5, feat] = np.nan                             # 100% NaN > MASK_MAX_NAN_FRAC
    df.loc[6, "relief_m"] = np.nan                       # одиночный NaN — валидна
    valid = cell_mask.build_valid_mask(df)
    assert not valid[2] and not valid[5]
    assert valid[6]
    assert valid.sum() == len(df) - 2


def test_nan_fraction_threshold_respects_config():
    df = _toy_df()
    feat = cell_mask.feature_columns(df)
    n_nan = int(np.ceil(len(feat) * config.MASK_MAX_NAN_FRAC)) + 1
    df.loc[1, feat[:n_nan]] = np.nan
    assert not cell_mask.build_valid_mask(df)[1]


def test_domains_follow_mask_svita_and_default_zero():
    df = _toy_df()
    dom = cell_mask.build_domains(df)
    assert dom.dtype.kind == "i"
    np.testing.assert_array_equal(dom, df["mask_svita"].to_numpy().astype(int))
    dom0 = cell_mask.build_domains(df.drop(columns="mask_svita"))
    assert (dom0 == 0).all()
