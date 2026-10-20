"""Тесты src/snowmelt.py на синтетическом сезонном ряде (без сети и STAC)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src import config, snowmelt as sm


class _Meta:
    def __init__(self, prf, pic, dx, dy):
        self.prf, self.pic, self.dx, self.dy = prf, pic, dx, dy


def _season(melt_doy=160, recovery_doy=175, dry_db=-8.0, wet_db=-16.0,
            summer_db=-9.0, doys=None):
    """Сезонная кривая одного пикселя: сухой снег -> мокрый провал -> лето.

    Возвращает ``(doys, values_db)``. Форма кусочно-линейная — от детектора
    требуется найти минимум и возврат, а не угадать физическую форму кривой.
    """
    doys = np.arange(95, 210, 3) if doys is None else np.asarray(doys)
    v = np.empty(len(doys), dtype=float)
    for i, d in enumerate(doys):
        if d < melt_doy - 20:
            v[i] = dry_db                                  # сухой снег
        elif d <= melt_doy:                                # спуск в мокрую фазу
            v[i] = dry_db + (wet_db - dry_db) * (d - (melt_doy - 20)) / 20.0
        elif d <= recovery_doy:                            # возврат к лету
            v[i] = wet_db + (summer_db - wet_db) * (d - melt_doy) / (recovery_doy - melt_doy)
        else:
            v[i] = summer_db                               # снега нет
    return doys, v


def _pix(v):
    """Кривая одного пикселя -> куб (n_scenes, 1, 1)."""
    return np.asarray(v, dtype=float)[:, None, None]


def _pixels(*curves):
    """Несколько кривых -> куб (n_scenes, 1, n_pixels)."""
    return np.stack(curves, axis=-1)[:, None, :]


# --- Свёртка нерегулярного ряда в корзины ---

def test_bin_by_doy_shapes_and_centers():
    doys, v = _season()
    centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    assert cube.shape[0] == len(centers)
    assert np.all(np.diff(centers) == 5)


def test_bin_by_doy_uses_median_against_speckle():
    """Одиночный всплеск обратного рассеяния не должен двигать корзину."""
    doys = np.array([101, 102, 103])          # все три — в одну корзину
    vals = np.array([-8.0, -8.0, 40.0])[:, None, None]   # третья сцена — выброс
    _, cube = sm.bin_by_doy(doys, vals, bin_days=5)
    filled = cube[np.isfinite(cube[:, 0, 0]), 0, 0]
    assert filled.size == 1 and filled[0] == pytest.approx(-8.0)


def test_bin_by_doy_leaves_empty_bins_nan():
    doys = np.array([100, 130])          # между ними корзины пустые
    vals = np.array([-8.0, -12.0])[:, None, None]
    _, cube = sm.bin_by_doy(doys, vals, bin_days=5)
    assert np.isnan(cube[:, 0, 0]).any()


# --- Метрики одного года ---

def test_melt_metrics_recovers_known_dates():
    doys, v = _season(melt_doy=160, recovery_doy=175)
    centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    m = sm.melt_metrics(centers, cube)
    assert abs(float(m["melt_doy"][0, 0]) - 160) <= config.SM_DOY_BIN
    assert abs(float(m["recovery_doy"][0, 0]) - 175) <= 2 * config.SM_DOY_BIN
    assert float(m["drop_db"][0, 0]) > config.SM_WET_DROP_DB


def test_melt_metrics_orders_early_and_late_pixels():
    """Главное требование: детектор должен РАЗЛИЧАТЬ ранний и поздний сход."""
    doys, early = _season(melt_doy=145, recovery_doy=158)
    _, late = _season(melt_doy=170, recovery_doy=185)
    cube_scenes = _pixels(early, late)
    centers, cube = sm.bin_by_doy(doys, cube_scenes, bin_days=5)
    m = sm.melt_metrics(centers, cube)
    assert m["melt_doy"][0, 0] < m["melt_doy"][0, 1]
    assert m["recovery_doy"][0, 0] < m["recovery_doy"][0, 1]


def test_melt_metrics_all_nan_pixel_is_nan_not_crash():
    doys, v = _season()
    cube_scenes = _pixels(v, np.full(len(v), np.nan))
    centers, cube = sm.bin_by_doy(doys, cube_scenes, bin_days=5)
    m = sm.melt_metrics(centers, cube)
    assert np.isfinite(m["melt_doy"][0, 0])
    assert np.isnan(m["melt_doy"][0, 1])


def test_melt_metrics_rejects_too_short_series():
    """Половина кривой не даёт дату, а даёт артефакт — такой пиксель отсеивается."""
    doys = np.array([100, 105, 110])          # корзин меньше SM_MIN_BINS
    vals = np.array([-8.0, -9.0, -10.0])[:, None, None]
    centers, cube = sm.bin_by_doy(doys, vals, bin_days=5)
    assert np.isnan(sm.melt_metrics(centers, cube)["melt_doy"]).all()


def test_melt_metrics_flat_series_has_no_wet_phase():
    """Без провала мокрой фазы нет: wet_len не определён."""
    doys = np.arange(95, 210, 3)
    centers, cube = sm.bin_by_doy(doys, _pix(np.full(len(doys), -9.0)), bin_days=5)
    m = sm.melt_metrics(centers, cube)
    assert np.isnan(m["wet_len"]).all() or float(np.nanmax(m["drop_db"])) < config.SM_WET_DROP_DB


# --- Межгодовая сводка ---

def test_aggregate_years_median_and_std():
    def year(md):
        doys, v = _season(melt_doy=md, recovery_doy=md + 15)
        centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
        return sm.melt_metrics(centers, cube)

    agg = sm.aggregate_years([year(150), year(160), year(170)])
    assert abs(float(agg["sm_melt_doy"][0, 0]) - 160) <= config.SM_DOY_BIN
    assert float(agg["sm_melt_doy_std"][0, 0]) > 5.0
    assert float(agg["sm_n_years"][0, 0]) == 3


def test_aggregate_years_requires_min_years():
    doys, v = _season()
    centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    good = sm.melt_metrics(centers, cube)
    empty = {k: np.full_like(v_, np.nan) for k, v_ in good.items()}
    agg = sm.aggregate_years([good, empty, empty])
    assert np.isnan(agg["sm_melt_doy"]).all()      # валидный год один, порог 3


def test_aggregate_years_stable_pixel_has_low_std():
    def year(md):
        doys, v = _season(melt_doy=md, recovery_doy=md + 15)
        centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
        return sm.melt_metrics(centers, cube)

    stable = sm.aggregate_years([year(160), year(160), year(160)])
    jumpy = sm.aggregate_years([year(140), year(160), year(180)])
    assert float(stable["sm_melt_doy_std"][0, 0]) < float(jumpy["sm_melt_doy_std"][0, 0])


# --- Признаки на ячейки ---

def test_melt_features_shape_and_columns():
    meta = _Meta(prf=4, pic=4, dx=500.0, dy=500.0)
    rasters = {"sm_melt_doy": np.full((20, 20), 160.0),
               "sm_melt_doy_std": np.full((20, 20), 4.0)}
    feats = sm.melt_features(meta, rasters, res_m=100.0)
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == meta.prf * meta.pic
    assert "sm_melt_doy" in feats.columns and "sm_melt_doy_std" in feats.columns


def test_snowmelt_cols_match_pooled_output():
    """Реестр колонок не должен разъезжаться с тем, что реально считается."""
    doys, v = _season()
    centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    pooled = sm.align_and_pool([cube], [sm.sheet_melt_doy(centers, cube)], bin_days=5)
    assert set(sm.pooled_rasters(centers, pooled)) == set(sm.SNOWMELT_COLS)


# --- Сведение годов ---

def test_sheet_melt_doy_finds_seasonal_minimum():
    doys, v = _season(melt_doy=160, recovery_doy=175)
    centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    assert abs(sm.sheet_melt_doy(centers, cube) - 160) <= config.SM_DOY_BIN


def test_sheet_wet_amplitude_separates_melt_year_from_flat_year():
    """Год с провалом и год-полка обязаны различаться порогом SM_WET_DROP_DB.

    Это правило отбраковки года перед сведением: 2021 отдал по 403 все
    майско-июньские сцены, дал размах 1.8 дБ и опорную дату из шума.
    """
    doys, v = _season(melt_doy=160, recovery_doy=175)
    _, melt = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    rng = np.random.default_rng(3)
    flat = rng.normal(-9.0, 0.3, size=len(doys))
    _, polka = sm.bin_by_doy(doys, _pix(flat), bin_days=5)
    assert sm.sheet_wet_amplitude(melt) > config.SM_WET_DROP_DB
    assert sm.sheet_wet_amplitude(polka) < config.SM_WET_DROP_DB


def test_sheet_wet_amplitude_nan_on_too_short_series():
    doys = np.array([100, 105, 110])          # корзин меньше SM_MIN_BINS
    _, cube = sm.bin_by_doy(doys, np.array([-8.0, -9.0, -16.0])[:, None, None],
                            bin_days=5)
    assert np.isnan(sm.sheet_wet_amplitude(cube))


def test_align_and_pool_removes_synoptic_shift():
    """Годы с разной датой весны после выравнивания дают ОДИН резкий минимум.

    Без выравнивания медиана трёх сдвинутых кривых размазала бы провал: это и
    есть причина, по которой годы нельзя складывать наивно.
    """
    years, mds = [], []
    for md in (145, 160, 175):
        doys, v = _season(melt_doy=md, recovery_doy=md + 15)
        centers, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
        years.append(cube)
        mds.append(sm.sheet_melt_doy(centers, cube))
    pooled = sm.align_and_pool(years, mds, bin_days=5)
    with warnings.catch_warnings():      # у наивной медианы корзина пуста во всех годах
        warnings.simplefilter("ignore", RuntimeWarning)
        naive = np.nanmedian(np.stack(years), axis=0)
    depth = lambda c: float(np.nanmax(c) - np.nanmin(c))  # noqa: E731
    assert depth(pooled) > depth(naive)      # выровненный провал глубже размазанного


def test_align_and_pool_does_not_wrap_july_into_april():
    """Края, приехавшие с другого конца, обязаны стать NaN, а не подменить сезон.

    Проверяется на паре «год + пустышка»: при нескольких реальных годах край
    законно заполняется соседним годом, и инвариант переноса там не наблюдаем.
    """
    doys, v = _season()
    _, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    empty = np.full_like(cube, np.nan)
    # опорная дата = медиана(130, 190) = 160, значит год сдвигается на +6 корзин
    pooled = sm.align_and_pool([cube, empty], [130.0, 190.0], bin_days=5)
    assert np.isnan(pooled[:6]).all()


def test_align_and_pool_raises_without_any_shift():
    doys, v = _season()
    _, cube = sm.bin_by_doy(doys, _pix(v), bin_days=5)
    with pytest.raises(ValueError):
        sm.align_and_pool([cube], [float("nan")], bin_days=5)


def test_pooling_lifts_bin_coverage():
    """Смысл всей конструкции: свести годы -> у пикселя заполнено больше корзин."""
    rng = np.random.default_rng(7)
    doys_all, years, mds = np.arange(95, 210, 3), [], []
    for _ in range(5):
        _, v = _season()
        seen = rng.random(len(doys_all)) < 0.25        # сцена видит пиксель в 25% случаев
        vv = np.where(seen, v, np.nan)
        centers, cube = sm.bin_by_doy(doys_all, _pix(vv), bin_days=5)
        years.append(cube)
        mds.append(160.0)
    per_year = np.isfinite(years[0]).sum()
    pooled = sm.align_and_pool(years, mds, bin_days=5)
    assert np.isfinite(pooled).sum() > per_year


def test_bin_by_doy_uses_common_edges_across_years():
    """Общие границы корзин — предусловие сведения: у годов разные края выборки."""
    _, v = _season()
    c1, _ = sm.bin_by_doy(np.arange(100, 200, 3), _pix(v[:34]), bin_days=5)
    c2, _ = sm.bin_by_doy(np.arange(95, 210, 3), _pix(v), bin_days=5)
    np.testing.assert_array_equal(c1, c2)
