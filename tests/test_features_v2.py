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
    # Съёмки этапа 8: у радара отношение поляризаций — это РАЗНОСТЬ каналов в
    # дБ, и синтетика обязана воспроизводить именно тождество, иначе тест на
    # алгебраический дубль ничего не проверяет.
    for c in ("s1_vv", "s1_vh", "psr_hh", "psr_hv", "l8_lst", "l8_ndvi",
              "ast_aloh", "ast_b05", "s2_b02", "s2_b05"):
        cols[c] = rng.normal(size=n)
    cols["s1_vv_vh"] = cols["s1_vv"] - cols["s1_vh"]
    cols["psr_hh_hv"] = cols["psr_hh"] - cols["psr_hv"]
    for c in ("s1_n_obs", "psr_valid_frac", "ast_n_obs", "l8_valid_frac"):
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


# ------------------------------------------------------ съёмки этапа 8 (v3)
def test_pool_drops_radar_ratio_as_exact_duplicate(fake_df):
    """Отношение поляризаций в дБ = разность каналов; в пуле ему места нет.

    Не косметика: с ним обусловленность пула была 2.5e13 вместо 9.5e4, то есть
    корреляционная матрица вырождалась и расстояние Махаланобиса теряло смысл.
    """
    feat = features_v2.pool_features(fake_df)
    assert "s1_vv_vh" not in feat.columns and "psr_hh_hv" not in feat.columns
    assert {"s1_vv", "s1_vh", "psr_hh", "psr_hv"} <= set(feat.columns)


def test_pool_keeps_only_thermal_from_landsat8(fake_df):
    """Оптика Landsat 8/9 повторяет Sentinel-2 (|r| 0.83..1.00) — остаётся тепло."""
    feat = features_v2.pool_features(fake_df)
    assert "l8_lst" in feat.columns
    assert "l8_ndvi" not in feat.columns


def test_pool_drops_raw_aster_bands_but_keeps_indices(fake_df):
    """Сырые каналы ASTER — уровень одной сцены; индексы от него свободны."""
    feat = features_v2.pool_features(fake_df)
    assert "ast_b05" not in feat.columns
    assert "ast_aloh" in feat.columns


def test_service_columns_of_every_sensor_are_excluded(fake_df):
    """n_obs/valid_frac — качество съёмки, а не свойство площади (все сенсоры)."""
    feat = features_v2.pool_features(fake_df)
    assert not [c for c in feat.columns if c.endswith(("_n_obs", "_valid_frac"))]


def test_restore_returns_named_columns_without_touching_old_pools(fake_df):
    """``restore`` возвращает сырые каналы S2, не трогая остальные отбросы."""
    raw = tuple(features_v2.config.V2_FEATURE_GROUPS["s2raw"])
    base = features_v2.pool_features(fake_df)
    with_raw = features_v2.pool_features(fake_df, restore=raw)
    assert "s2_b02" not in base.columns and "s2_b02" in with_raw.columns
    assert "l8_ndvi" not in with_raw.columns      # чужие отбросы не воскресают
    assert set(base.columns) < set(with_raw.columns)


def test_raw_s2_channels_form_their_own_ablation_group(fake_df):
    """Группа s2raw задана полными именами: текстура остаётся в группе s2.

    По префиксу ``s2_b02`` в s2raw утянуло бы и ``s2_b02_std``, и абляция
    отвечала бы уже не на тот вопрос, ради которого группа заведена.
    """
    assert features_v2.feature_group("s2_b02") == "s2raw"
    assert features_v2.feature_group("s2_b02_std") == "s2"
    assert features_v2.feature_group("s2_clay") == "s2"


def test_new_sensor_groups_do_not_collide(fake_df):
    """``l8_`` не должен попасть в группу ``ls_`` и наоборот."""
    assert features_v2.feature_group("l8_lst") == "l8"
    assert features_v2.feature_group("ls_ch6") == "ls"
    assert features_v2.feature_group("s1_vv") == "s1"
    assert features_v2.feature_group("psr_hh") == "psr"
    assert features_v2.feature_group("ast_aloh") == "ast"
