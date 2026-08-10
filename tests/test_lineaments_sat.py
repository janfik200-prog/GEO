"""Тесты src/lineaments_sat.py на синтетике (без сети/STAC)."""
import numpy as np
import pandas as pd

from src import lineaments_sat


class _Meta:
    def __init__(self, prf, pic, dx, dy, x0, y0):
        self.prf, self.pic, self.dx, self.dy, self.x0, self.y0 = prf, pic, dx, dy, x0, y0


def _meta():
    return _Meta(prf=10, pic=10, dx=100.0, dy=100.0, x0=0.0, y0=0.0)


def _stripe_image(shape=(100, 100), angle_deg=0.0):
    """Синтетический композит с параллельными полосами (структура) под заданным углом."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    theta = np.radians(angle_deg)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    return (np.sin(proj / 4.0) > 0).astype(float)


def test_edges_from_image_empty_returns_false():
    img = np.full((50, 50), np.nan)
    edges = lineaments_sat.edges_from_image(img)
    assert edges.shape == img.shape
    assert not edges.any()


def test_edges_from_image_finds_stripe_boundaries():
    img = _stripe_image()
    edges = lineaments_sat.edges_from_image(img)
    assert edges.any()


def test_reproducibility_identical_images_high_rho():
    img = _stripe_image()
    rep = lineaments_sat.reproducibility(img, img)
    assert rep["rho"] > 0.9


def test_reproducibility_noise_vs_noise_low_rho():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(80, 80))
    b = rng.normal(size=(80, 80))
    rep = lineaments_sat.reproducibility(a, b)
    assert rep["covered_frac"] >= 0.0
    assert not np.isnan(rep["rho"]) or rep["covered_frac"] < (10.0 / a.size)


def test_source_features_shapes(monkeypatch):
    meta = _meta()
    img = _stripe_image(shape=(100, 100), angle_deg=0.0)
    feats, segments = lineaments_sat.source_features(meta, img, res_m=10.0, prefix="linopt_")
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == meta.prf * meta.pic
    for c in ("linopt_dens", "linopt_node_dens", "linopt_dist",
              "linopt_aniso_r", "linopt_dir_sin", "linopt_dir_cos"):
        assert c in feats.columns
    assert isinstance(segments, list)


def test_segment_azimuths_vertical_line_is_zero():
    # (x0,y0)->(x1,y1) в пиксельных координатах: строка растёт к югу.
    segments = [((5, 10), (5, 0))]   # dx=0, идёт строго на север
    az, length = lineaments_sat.segment_azimuths(segments)
    assert np.isclose(az[0], 0.0)
    assert np.isclose(length[0], 10.0)


def test_segment_azimuths_horizontal_line_is_90():
    segments = [((0, 5), (10, 5))]   # dx=10, dy_north=0 -> восток
    az, length = lineaments_sat.segment_azimuths(segments)
    assert np.isclose(az[0], 90.0)


def test_azimuth_rose_sums_to_one():
    az = np.array([10.0, 10.0, 100.0])
    length = np.array([1.0, 1.0, 2.0])
    edges, hist = lineaments_sat.azimuth_rose(az, length, n_bins=18)
    assert len(edges) == 19
    assert np.isclose(hist.sum(), 1.0)


def test_azimuth_rose_empty_returns_zeros():
    edges, hist = lineaments_sat.azimuth_rose(np.array([]), np.array([]), n_bins=18)
    assert np.all(hist == 0.0)


def test_match_stats_perfect_overlap():
    map_dist = np.array([0.0, 0.0, 5000.0, 5000.0])
    auto_dist = np.array([0.0, 0.0, 5000.0, 5000.0])
    df = lineaments_sat.match_stats(auto_dist, map_dist, thresholds=(100.0,))
    assert np.isclose(df.loc[0, "доля_карты_поймана"], 1.0)
    assert np.isclose(df.loc[0, "доля_авто_без_карты"], 0.0)


def test_match_stats_no_overlap():
    map_dist = np.array([0.0, 0.0, 5000.0, 5000.0])
    auto_dist = np.array([5000.0, 5000.0, 0.0, 0.0])
    df = lineaments_sat.match_stats(auto_dist, map_dist, thresholds=(100.0,))
    assert np.isclose(df.loc[0, "доля_карты_поймана"], 0.0)
    assert np.isclose(df.loc[0, "доля_авто_без_карты"], 1.0)


def test_split_even_odd_alternates():
    items = [3, 1, 2, 5, 4]
    even, odd = lineaments_sat.split_even_odd(items, key=lambda x: x)
    assert even == [1, 3, 5]
    assert odd == [2, 4]
