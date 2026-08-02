"""Тесты src/features_v11.py: набор признаков лестницы v1.1 (синтетика)."""
import numpy as np
import pandas as pd

from src import features_v11


def _toy_df(n=8):
    """Мини-датасет с реальными именами колонок parquet (включая избыточные)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "row": np.arange(n), "col": np.arange(n),
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
        "gm_gr_all": rng.normal(size=n),
        "gm_gr_1G_25": rng.normal(size=n),        # избыточный модуль градиента
        "gm_gr_1GX_25": rng.normal(size=n),
        "gm_gr_1GY_25": rng.normal(size=n),
        "gm_gr_1GFI_25": rng.uniform(0, 360, n),  # избыточный угол
        "gm_mag_1G_25": rng.normal(size=n),
        "gm_mag_1GFI_25": rng.uniform(0, 360, n),
        "ls_ch1": rng.uniform(50, 150, n),
        "ls_ch2": rng.uniform(50, 150, n),
        "ls_ch3": rng.uniform(40, 120, n),
        "ls_ch4": rng.uniform(30, 100, n),
        "ls_ch5": rng.uniform(10, 140, n),
        "ls_ch6": rng.uniform(100, 200, n),
        "ls_ch7": rng.uniform(10, 110, n),
        "relief_m": rng.uniform(100, 400, n),     # без NaN — DEM не дёргается
        "dist_dnl": np.full(n, 100.0),
        "dist_dnara": np.full(n, 500.0),
        "dist_tect1": rng.normal(size=n),         # факторный — вне фазы
        "mask_svita": np.ones(n),
    })
    return df


def test_ladder_features_drops_redundant_and_raw_landsat():
    feat = features_v11.ladder_features(_toy_df())
    cols = set(feat.columns)
    assert not cols & {"gm_gr_1G_25", "gm_mag_1G_25"}
    assert not any("1GFI" in c for c in cols)                # угол в любом виде
    assert not cols & set(features_v11.LS_RAW)               # сырые DN заменены
    assert {"ls_r31", "ls_r57", "ls_r54", "ls_ndvi", "ls_ch6"} <= cols
    assert {"gm_gr_1GX_25", "gm_gr_1GY_25", "relief_m"} <= cols
    assert not cols & set(features_v11.FIELD_REDUNDANT)   # поле = flt + ost
    assert "dist_tect1" not in cols and "mask_svita" not in cols


def test_ladder_features_include_dist_toggle():
    df = _toy_df()
    no_dist = features_v11.ladder_features(df, include_dist=False)
    with_dist = features_v11.ladder_features(df, include_dist=True)
    assert not {"dist_dnl", "dist_dnara"} & set(no_dist.columns)
    assert {"dist_dnl", "dist_dnara"} <= set(with_dist.columns)
    assert set(with_dist.columns) - set(no_dist.columns) == {"dist_dnl", "dist_dnara"}


def test_ladder_features_ratio_nan_on_bad_denominator():
    df = _toy_df()
    df.loc[0, "ls_ch1"] = 0.0            # знаменатель 0 -> NaN
    df.loc[1, "ls_ch7"] = -5.0           # знаменатель < 0 -> NaN
    df.loc[2, "ls_ch4"] = np.nan         # NaN -> NaN
    feat = features_v11.ladder_features(df)
    assert np.isnan(feat.loc[0, "ls_r31"])
    assert np.isnan(feat.loc[1, "ls_r57"])
    assert np.isnan(feat.loc[2, "ls_r54"]) and np.isnan(feat.loc[2, "ls_ndvi"])
    assert np.isfinite(feat.loc[3, "ls_r31"])


def test_ladder_features_ratio_values():
    df = _toy_df()
    feat = features_v11.ladder_features(df)
    i = 4
    assert np.isclose(feat.loc[i, "ls_r31"], df.loc[i, "ls_ch3"] / df.loc[i, "ls_ch1"])
    ndvi = ((df.loc[i, "ls_ch4"] - df.loc[i, "ls_ch3"])
            / (df.loc[i, "ls_ch4"] + df.loc[i, "ls_ch3"]))
    assert np.isclose(feat.loc[i, "ls_ndvi"], ndvi)


def test_fill_relief_no_nan_returns_as_is():
    df = _toy_df()
    out = features_v11.fill_relief_from_dem(df)
    np.testing.assert_array_equal(out.to_numpy(), df["relief_m"].to_numpy())


def test_condition_number_orthogonal_vs_collinear():
    rng = np.random.default_rng(1)
    n = 500
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    valid = np.ones(n, dtype=bool)
    indep = pd.DataFrame({"a": a, "b": b})
    coll = pd.DataFrame({"a": a, "b": b, "c": a + 1e-6 * rng.normal(size=n)})
    cn_i = features_v11.condition_number(indep, valid)
    cn_c = features_v11.condition_number(coll, valid)
    assert cn_i < 2.0
    assert cn_c > 1e3
