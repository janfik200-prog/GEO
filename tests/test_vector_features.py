"""Тесты векторных признаков на сетке (синтетические геометрии, без данных на диске).

Запуск из корня репозитория: ``python -m pytest -q``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, Polygon  # noqa: E402

from src.integro_grid import GridMeta  # noqa: E402
from src.vector_features import density_raster, distance_raster  # noqa: E402


def _meta(pic=10, prf=8, dx=1.0):
    return GridMeta(obj_count=pic * prf, prop_count=0, pic=pic, prf=prf,
                    dx=dx, dy=dx, x0=0.0, y0=0.0)


def test_distance_raster_vertical_line():
    # вертикальная линия x=0.5 проходит через центры столбца 0
    meta = _meta()
    d = distance_raster(meta, [LineString([(0.5, -10), (0.5, 20)])])
    assert d.shape == meta.shape
    assert np.allclose(d[:, 0], 0.0)
    # расстояние растёт на dx с каждым столбцом
    assert np.allclose(d[:, 3], 3.0)


def test_distance_raster_inside_polygon_is_zero():
    meta = _meta()
    poly = Polygon([(0, 0), (5, 0), (5, 8), (0, 8)])  # левая половина сетки
    d = distance_raster(meta, [poly])
    assert np.allclose(d[:, :5], 0.0)
    assert np.all(d[:, 5:] > 0)


def test_distance_raster_orientation_north_up():
    # линия по северному краю (y = y_top): строка 0 ближе всех
    meta = _meta()
    d = distance_raster(meta, [LineString([(-10, 8.0), (20, 8.0)])])
    assert d[0, 0] < d[-1, 0]
    assert np.allclose(d[0, :], 0.5)


def test_density_raster_line_length():
    # одна горизонтальная линия через всю сетку: в радиусе r вокруг центра на линии
    # лежит отрезок длиной 2r
    meta = _meta()
    r = 2.0
    dens = density_raster(meta, [LineString([(-100, 3.5), (100, 3.5)])], radius=r)
    row = meta.prf - 1 - 3  # y=3.5 -> строка с юга 3-я, а строка 0 — север
    assert np.isclose(dens[row, 5], 2 * r, rtol=1e-3)
    # вдали от линии плотность нулевая
    far_row = 0
    assert dens[far_row, 5] < 2 * r


def test_density_raster_polygon_area():
    # большой полигон покрывает буфер целиком: площадь = pi*r^2
    meta = _meta()
    r = 1.5
    poly = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
    dens = density_raster(meta, [poly], radius=r, measure="area")
    assert np.allclose(dens, np.pi * r**2, rtol=1e-2)


def test_density_raster_bad_measure():
    with pytest.raises(ValueError):
        density_raster(_meta(), [], radius=1.0, measure="volume")
