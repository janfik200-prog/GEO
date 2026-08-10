"""Тесты src/potential_fields.py: трансформанты БПФ на синтетике (без сети)."""
import numpy as np

from src import potential_fields as pf

DX = DY = 500.0


def _sine(shape, period_px):
    ny, nx = shape
    _, cols = np.mgrid[0:ny, 0:nx]
    k = 2 * np.pi / (period_px * DX)
    return np.sin(k * cols * DX)


def test_vertical_derivative_amplitude_scales_with_wavenumber():
    shape = (64, 64)
    lo = _sine(shape, period_px=32)   # длинный период -> малое k
    hi = _sine(shape, period_px=8)    # короткий период -> большое k (в 4 раза)
    core = (slice(16, 48), slice(16, 48))
    vz_lo = pf.vertical_derivative(lo, DX, DY, taper_frac=0.0)
    vz_hi = pf.vertical_derivative(hi, DX, DY, taper_frac=0.0)
    ratio = np.std(vz_hi[core]) / np.std(vz_lo[core])
    assert 3.0 < ratio < 5.0    # ожидание x4 (k пропорционально 1/период)


def test_tilt_derivative_is_bounded():
    shape = (48, 48)
    field = _sine(shape, period_px=12)
    tdr = pf.tilt_derivative(field, DX, DY, taper_frac=0.1)
    assert np.nanmax(tdr) <= np.pi / 2 + 1e-6
    assert np.nanmin(tdr) >= -np.pi / 2 - 1e-6


def test_analytic_signal_is_nonnegative():
    shape = (48, 48)
    field = _sine(shape, period_px=10) + 0.5 * _sine(shape, period_px=25)
    asig = pf.analytic_signal(field, DX, DY, taper_frac=0.1)
    assert np.all(asig[np.isfinite(asig)] >= 0.0)


def test_upward_continuation_attenuates_and_preserves_mean():
    shape = (64, 64)
    field = 10.0 + 5.0 * _sine(shape, period_px=8)   # регион + короткая аномалия
    core = (slice(16, 48), slice(16, 48))
    up = pf.upward_continuation(field, DX, DY, height_m=2000.0, taper_frac=0.1)
    assert np.std(up[core]) < np.std(field[core])          # высокочастотное затухает
    assert abs(np.nanmean(up) - np.nanmean(field)) < 1.0    # регионал (k=0) не меняется


def test_reduce_to_pole_is_identity_at_the_pole():
    # I=90 -> Theta_f = K*sin(90) = K, RTP = K^2/K^2 = 1 для всех k>0: фильтр тождественный
    shape = (48, 48)
    field = _sine(shape, period_px=12) + 0.3 * _sine(shape, period_px=6)
    rtp = pf.reduce_to_pole(field, DX, DY, inclination_deg=90.0, declination_deg=0.0,
                             taper_frac=0.0)
    core = (slice(12, 36), slice(12, 36))
    assert np.allclose(rtp[core], field[core], atol=1e-3)


def test_moving_correlation_perfect_and_inverse():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(40, 40))
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(a, 3)   # гладкое поле, иначе корреляция в окне вырождена
    same = pf.moving_correlation(a, a, win_px=8)
    opp = pf.moving_correlation(a, -a, win_px=8)
    core = (slice(10, 30), slice(10, 30))
    assert np.allclose(same[core], 1.0, atol=1e-3)
    assert np.allclose(opp[core], -1.0, atol=1e-3)


def test_fill_nan_replaces_holes_without_shifting_valid_cells():
    arr = np.ones((10, 10))
    arr[3:6, 3:6] = np.nan
    filled, valid = pf._fill_nan(arr)
    assert np.isfinite(filled).all()
    assert not valid[4, 4]
    assert valid[0, 0]
    assert filled[0, 0] == 1.0
