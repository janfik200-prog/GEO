"""Тесты src/s2_composite.py: маски, медиана, отношения, агрегация — без сети."""
import numpy as np
import pandas as pd

from src import config, s2_composite


def test_scl_mask_keeps_only_configured_classes():
    scl = np.arange(12).reshape(3, 4).astype("float32")
    keep = s2_composite.scl_mask(scl)
    assert set(scl[keep].astype(int)) == set(config.S2_SCL_KEEP)
    assert not keep[np.isin(scl, [6, 8, 9, 10, 11])].any()   # вода, облака, снег


def test_veg_mask_drops_tundra_and_keeps_bare_rock():
    # задернованный пиксель: NIR >> red -> NDVI ~ 0.8; порода: NIR ~ red
    nir = np.array([[9000.0, 1200.0]], dtype="float32")
    red = np.array([[1000.0, 1000.0]], dtype="float32")
    keep = s2_composite.veg_mask(nir, red)
    assert not keep[0, 0]
    assert keep[0, 1]


def test_median_ignores_outlier_scene_and_min_obs_cuts_thin_pixels(monkeypatch):
    # три «сцены»: два согласованных наблюдения и одно облако-выброс
    base = np.full((2, 2), 1000.0, dtype="float32")
    fakes = [dict.fromkeys(config.S2_BANDS.values(), base.copy()),
             dict.fromkeys(config.S2_BANDS.values(), base.copy() + 10),
             dict.fromkeys(config.S2_BANDS.values(), base.copy() + 8000)]
    # у последнего пикселя данных нет ни в одной сцене, кроме первой
    for i, f in enumerate(fakes):
        f = {k: v.copy() for k, v in f.items()}
        if i > 0:
            for v in f.values():
                v[1, 1] = np.nan
        fakes[i] = f

    it = iter(fakes)
    monkeypatch.setattr(s2_composite, "scene_on_grid", lambda *a, **k: next(it))

    class Meta:
        prf, pic, dx, dy, x0, y_top = 2, 2, 500.0, 500.0, 0.0, 0.0

    comp, n_obs = s2_composite.median_composite([{}, {}, {}], Meta, res_m=500.0)
    assert np.isclose(comp["b04"][0, 0], 1010.0)      # медиана, а не среднее 3336
    assert n_obs[1, 1] == 1
    assert np.isnan(comp["b04"][1, 1])                # < S2_MIN_OBS наблюдений


def test_band_ratios_are_finite_and_named():
    comp = {b: np.full((2, 2), 1000.0, dtype="float32") for b in config.S2_BANDS.values()}
    comp["b04"][0, 0] = 2000.0
    comp["b02"][1, 1] = 0.0                            # деление на ноль -> NaN, не inf
    r = s2_composite.band_ratios(comp)
    assert set(s2_composite.RATIOS) | {"s2_ndvi"} == set(r)
    assert np.isclose(r["s2_iron_ox"][0, 0], 2.0)
    assert np.isnan(r["s2_iron_ox"][1, 1])
    assert np.isfinite(r["s2_iron_ox"][0, 1])


def test_grid_features_give_mean_std_and_valid_fraction():
    class Meta:
        prf, pic, dx, dy, x0, y_top = 1, 2, 500.0, 500.0, 0.0, 0.0

    k = 5
    comp = {b: np.zeros((k, 2 * k), dtype="float32") for b in config.S2_BANDS.values()}
    comp["b04"][:, :k] = 100.0                         # ячейка 0: константа
    comp["b04"][:, k:] = np.arange(k * k).reshape(k, k)  # ячейка 1: разброс
    comp["b04"][0, k] = np.nan                         # один субпиксель без данных
    ratios = s2_composite.band_ratios(comp)
    n_obs = np.full((k, 2 * k), 5.0)
    out = s2_composite.to_grid_features(comp, ratios, n_obs, Meta, res_m=100.0)
    assert isinstance(out, pd.DataFrame) and len(out) == 2
    assert np.isclose(out["s2_b04"][0], 100.0) and out["s2_b04_std"][0] == 0.0
    assert out["s2_b04_std"][1] > 0
    assert out["s2_valid_frac"][0] == 1.0
    assert np.isclose(out["s2_valid_frac"][1], (k * k - 1) / (k * k))
    assert 0.0 <= out["s2_bare_frac"].min() and out["s2_bare_frac"].max() <= 1.0


def test_veg_mask_is_off_by_default_under_continuous_tundra():
    # На листе NDVI 0.56-0.66 (измерено): жёсткая маска обнулила бы карту,
    # поэтому по умолчанию она выключена, а спектр трактуется геоботанически.
    assert config.S2_APPLY_VEG_MASK is False
