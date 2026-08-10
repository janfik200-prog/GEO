"""Синтетические тесты src/relief_v2_extra.py и catchments.flow_accumulation.

Сетевая часть (projected_dem -> relief_extra_features) не тестируется юнитами,
как и в tests/test_terrain_v2.py: сетевой доступ там не мокается.
"""
import numpy as np

from src import catchments
from src.relief_v2_extra import curvature_at_scale, terrain_ruggedness


def test_flow_accumulation_linear_chain():
    # 0 <- 1 <- 2 <- 3 (3 стекает в 2, 2 в 1, 1 в 0, 0 - выход)
    rec = np.array([-1, 0, 1, 2])
    acc = catchments.flow_accumulation(rec)
    assert list(acc) == [4, 3, 2, 1]


def test_flow_accumulation_branching_tree():
    # 1 и 2 стекают в 0; 3 стекает в 1
    rec = np.array([-1, 0, 0, 1])
    acc = catchments.flow_accumulation(rec)
    assert acc[0] == 4
    assert acc[1] == 2
    assert acc[2] == 1
    assert acc[3] == 1


def test_curvature_at_scale_bowl_is_positive():
    n = 61
    yy, xx = np.mgrid[0:n, 0:n]
    elev = 0.001 * ((xx - n // 2) ** 2 + (yy - n // 2) ** 2)
    curv = curvature_at_scale(elev, res_m=100.0, window_m=2000.0)
    # чаша (выпуклая вверх по краям, вниз в центре) -> положительный лапласиан
    assert curv[n // 2, n // 2] > 0


def test_curvature_at_scale_flat_is_near_zero():
    elev = np.full((41, 41), 500.0)
    curv = curvature_at_scale(elev, res_m=100.0, window_m=2000.0)
    assert np.allclose(curv, 0.0, atol=1e-9)


def test_terrain_ruggedness_flat_is_zero():
    elev = np.full((20, 20), 300.0)
    tri = terrain_ruggedness(elev, res_m=100.0)
    assert np.allclose(tri, 0.0)


def test_terrain_ruggedness_checkerboard_is_positive():
    elev = np.indices((10, 10)).sum(axis=0) % 2 * 50.0
    tri = terrain_ruggedness(elev, res_m=100.0)
    assert np.all(tri[1:-1, 1:-1] > 0)
