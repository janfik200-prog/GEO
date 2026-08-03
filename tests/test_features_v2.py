"""Тесты src/features_v2.py: правила пула v2 на синтетическом датасете."""
import numpy as np
import pandas as pd
import pytest

from src import features_v2, lineaments, terrain_v2


@pytest.fixture
def fake_df():
    n = 60
    rng = np.random.default_rng(0)
    cols = {c: rng.normal(size=n) for c in (
        "gm_gr_flt_35", "gm_gr_ost_35", "gm_mag_flt_35", "gm_mag_ost_35",
        "gm_gr_1GX_25", "gm_gr_1GY_25", "gm_gr_2V_25", "gm_gr_2G_25",
        "gm_mag_1GX_25", "gm_mag_1GY_25", "gm_mag_2G_25",
        "ls_ch1", "ls_ch2", "ls_ch3", "ls_ch4", "ls_ch5", "ls_ch6", "ls_ch7",
        "relief_m", "dist_dnl", "dist_dnara")}
    cols["gm_gr_all"] = cols["gm_gr_flt_35"] + cols["gm_gr_ost_35"]
    cols["gm_mg_all"] = cols["gm_mag_flt_35"] + cols["gm_mag_ost_35"]
    cols["gm_gr_1G_25"] = np.hypot(cols["gm_gr_1GX_25"], cols["gm_gr_1GY_25"])
    cols["gm_mag_1G_25"] = np.hypot(cols["gm_mag_1GX_25"], cols["gm_mag_1GY_25"])
    for c in terrain_v2.TERRAIN_COLS + terrain_v2.TERRAIN_DROPPED:
        cols[c] = rng.normal(size=n)
    for c in lineaments.LINEAMENT_COLS:
        cols[c] = rng.normal(size=n)
    for c in ("s2_b04", "s2_clay", "s2_ndvi", "s2_n_obs", "s2_valid_frac"):
        cols[c] = rng.normal(size=n)
    return pd.DataFrame(cols)


def test_pool_drops_every_registered_duplicate(fake_df):
    feat = features_v2.pool_features(fake_df)
    for col in features_v2.V2_DROPPED:
        assert col not in feat.columns, f"{col} должен быть исключён"


def test_pool_takes_dem_elev_instead_of_relief_m(fake_df):
    feat = features_v2.pool_features(fake_df)
    assert "dem_elev" in feat.columns
    assert "relief_m" not in feat.columns


def test_pool_includes_new_groups_and_skips_service_columns(fake_df):
    feat = features_v2.pool_features(fake_df)
    assert {"dem_incision", "lin_dens", "s2_clay"} <= set(feat.columns)
    assert "s2_n_obs" not in feat.columns        # служебные, не признаки
    assert "s2_valid_frac" not in feat.columns
    assert "dist_dnl" not in feat.columns        # dist только по явному запросу


def test_ablation_keeps_only_requested_groups(fake_df):
    feat = features_v2.pool_features(fake_df, groups=("gm",))
    assert feat.columns.size > 0
    assert all(features_v2.feature_group(c) == "gm" for c in feat.columns)


def test_pool_survives_missing_optional_sources(fake_df):
    lean = fake_df.drop(columns=[c for c in fake_df.columns
                                 if c.startswith(("s2_", "lin_", "dem_"))])
    feat = features_v2.pool_features(lean)
    assert feat.columns.size > 0
    assert "relief_m" in feat.columns   # без dem_elev старый рельеф остаётся


def test_group_sizes_cover_all_columns(fake_df):
    feat = features_v2.pool_features(fake_df)
    assert sum(features_v2.group_sizes(feat).values()) == feat.columns.size
