"""Тесты src/mineragenic_features.py на синтетике (без сети/шейпов)."""
import geopandas as gpd
import numpy as np
import shapely

from src import mineragenic_features


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


def test_geochem_distance_features_separates_au_u_other():
    meta = _meta()
    au_line = shapely.LineString([(50, 0), (50, 1000)])       # левая часть
    u_line = shapely.LineString([(950, 0), (950, 1000)])       # правая часть
    co_line = shapely.LineString([(500, 500), (600, 500)])     # середина
    halo = gpd.GeoDataFrame({"label": ["Au", "U", "Co"]}, geometry=[au_line, u_line, co_line])
    uran = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 950), (1000, 950)])])  # верхний край

    out = mineragenic_features.geochem_distance_features(meta, halo, uran)
    assert set(out) == {"mingeo_dist_au", "mingeo_dist_u", "mingeo_dist_uran", "mingeo_dist_other"}
    for arr in out.values():
        assert arr.shape == meta.shape

    # ближе к au_line (x=50) слева, чем к u_line (x=950) справа
    assert out["mingeo_dist_au"][5, 0] < out["mingeo_dist_au"][5, 9]
    assert out["mingeo_dist_u"][5, 9] < out["mingeo_dist_u"][5, 0]
    # верхний ряд (y ближе к 950) ближе к uran, чем нижний
    assert out["mingeo_dist_uran"][0, 5] < out["mingeo_dist_uran"][9, 5]


def test_geochem_distance_features_no_other_label_omits_key():
    meta = _meta()
    au_line = shapely.LineString([(50, 0), (50, 1000)])
    u_line = shapely.LineString([(950, 0), (950, 1000)])
    halo = gpd.GeoDataFrame({"label": ["Au", "U"]}, geometry=[au_line, u_line])
    uran = gpd.GeoDataFrame(geometry=[shapely.LineString([(0, 950), (1000, 950)])])

    out = mineragenic_features.geochem_distance_features(meta, halo, uran)
    assert "mingeo_dist_other" not in out
