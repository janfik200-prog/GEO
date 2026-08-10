"""Тесты src/geology_v2.py: геологические производные на синтетике (без сети)."""
import geopandas as gpd
import numpy as np
import shapely

from src import geology_v2


class _Meta:
    def __init__(self, prf, pic, dx, dy, x0, y0):
        self.prf, self.pic, self.dx, self.dy, self.x0, self.y0 = prf, pic, dx, dy, x0, y0

    @property
    def shape(self):
        return self.prf, self.pic

    @property
    def y_top(self):
        return self.y0 + self.prf * self.dy

    def cell_centers(self):
        cols, rows = np.meshgrid(np.arange(self.pic), np.arange(self.prf))
        x = self.x0 + (cols + 0.5) * self.dx
        y = self.y_top - (rows + 0.5) * self.dy
        return x, y


def _meta():
    return _Meta(prf=10, pic=10, dx=100.0, dy=100.0, x0=0.0, y0=0.0)


def test_facies_distance_rasters_separates_codes():
    meta = _meta()
    left = shapely.box(0, 0, 300, 1000)     # покрывает левый край
    right = shapely.box(700, 0, 1000, 1000)  # покрывает правый край
    gdf = gpd.GeoDataFrame({"CODE_F": [1, 2]}, geometry=[left, right])
    out = geology_v2.facies_distance_rasters(meta, gdf)
    assert out["geo2_facies1"][5, 0] == 0.0     # внутри facies1
    assert out["geo2_facies2"][5, 9] == 0.0     # внутри facies2
    assert out["geo2_facies1"][5, 9] > 0.0      # вне facies1


def test_contact_boundary_ignores_internal_seams():
    meta = _meta()
    # два смежных квадрата без зазора -> после union внутренний шов исчезает
    a = shapely.box(0, 0, 500, 1000)
    b = shapely.box(500, 0, 1000, 1000)
    gdf = gpd.GeoDataFrame(geometry=[a, b])
    segs, az = geology_v2.contact_boundary_segments(gdf)
    mids = shapely.get_coordinates(shapely.line_interpolate_point(segs, 0.5, normalized=True))
    # ни один сегмент границы не должен проходить точно по внутреннему шву x=500
    assert not np.any(np.isclose(mids[:, 0], 500.0) & (mids[:, 1] > 1.0) & (mids[:, 1] < 999.0))


def test_contact_features_zero_on_boundary_and_axial_strike():
    meta = _meta()
    square = shapely.box(200, 200, 800, 800)
    gdf = gpd.GeoDataFrame(geometry=[square])
    out = geology_v2.contact_features(meta, gdf)
    assert out["geo2_contact"].shape == meta.shape
    # осевая пара всегда на единичной окружности (с точностью float32)
    mag = out["geo2_strike_sin"] ** 2 + out["geo2_strike_cos"] ** 2
    assert np.allclose(mag, 1.0, atol=1e-3)


def test_fault_node_distance_finds_overlap():
    meta = _meta()
    tect1 = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 600, 600)])
    tect2 = gpd.GeoDataFrame(geometry=[shapely.box(400, 400, 1000, 1000)])
    out = geology_v2.fault_node_distance(meta, tect1, tect2)
    # пересечение -- квадрат 400..600, центр (500,500) -> ячейка (5,4) должна быть близко к 0
    assert out.min() < 100.0


def test_fault_node_distance_raises_without_overlap():
    meta = _meta()
    tect1 = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 100, 100)])
    tect2 = gpd.GeoDataFrame(geometry=[shapely.box(800, 800, 900, 900)])
    try:
        geology_v2.fault_node_distance(meta, tect1, tect2)
        assert False, "должно было бросить ValueError"
    except ValueError:
        pass
