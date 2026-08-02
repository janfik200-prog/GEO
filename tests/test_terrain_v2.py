"""Тесты src/terrain_v2.py: производные рельефа на синтетике (без сети)."""
import numpy as np

from src import terrain_v2


RES = 100.0


def test_tilted_plane_has_constant_slope_and_zero_curvature():
    # плоскость с уклоном 0.05 (5 м на 100 м пикселя)
    rows, cols = np.mgrid[0:60, 0:60]
    e = 0.05 * cols * RES
    d = terrain_v2.terrain_derivatives(e, res_m=RES)
    core = (slice(10, 50), slice(10, 50))          # без краевых эффектов
    assert np.allclose(d["slope"][core], 0.05, atol=1e-6)
    assert np.allclose(d["curv"][core], 0.0, atol=1e-6)


def test_valley_is_incised_and_ridge_is_not():
    # V-образная долина вдоль столбца 30: глубина 100 м, борт 20 м на пиксель
    rows, cols = np.mgrid[0:80, 0:80]
    e = 500.0 + np.minimum(np.abs(cols - 30) * 20.0, 100.0)
    d = terrain_v2.terrain_derivatives(e, res_m=RES)
    valley = d["incision"][40, 30]
    ridge = d["incision"][40, 5]
    assert valley > 90.0                            # тальвег врезан почти на 100 м
    assert ridge < 1.0                              # водораздел на уровне максимума
    assert d["relief_1km"][40, 30] > 50.0
    assert d["tpi_500"][40, 30] < 0                 # ниже окрестности


def test_dropped_duplicates_are_not_in_the_pool():
    # отбракованные алгебраические дубли не должны утекать в пул признаков
    assert not set(terrain_v2.TERRAIN_COLS) & set(terrain_v2.TERRAIN_DROPPED)
    assert "dem_curv" not in terrain_v2.TERRAIN_COLS       # ~ -1.00 с dem_tpi_500
    assert "dem_elev_std" not in terrain_v2.TERRAIN_COLS   # ~ +0.98 с dem_slope


def test_block_stats_aggregates_5x5_blocks_in_c_order():
    class Meta:
        prf, pic, dx, dy = 2, 3, 500.0, 500.0

    k, pad = 5, 2
    arr = np.zeros((Meta.prf * k + 2 * pad, Meta.pic * k + 2 * pad))
    arr[pad:pad + k, pad:pad + k] = 7.0             # ячейка (0, 0)
    arr[pad + k:pad + 2 * k, pad + 2 * k:pad + 3 * k] = np.arange(k * k).reshape(k, k)
    mean, std = terrain_v2._block_stats(arr, Meta, k, pad)
    assert mean.shape == (Meta.prf * Meta.pic,)
    assert mean[0] == 7.0 and std[0] == 0.0         # ячейка (0,0): константа
    assert np.isclose(mean[1 * Meta.pic + 2], np.arange(k * k).mean())
    assert std[1 * Meta.pic + 2] > 0
