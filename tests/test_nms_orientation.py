"""Регрессия: утончение гребня (_ridge_nms) одинаково видит линии любого азимута.

До 23.09.2026 компоненты собственного вектора гессиана стояли переставленными,
и утончение подавляло немаксимумы ВДОЛЬ линий, идущих по осям растра: такие
линии рвались (замер 23.09: покрытие следа 39-48 % у осевых азимутов против
87-96 % после правки). Тест ловит именно эту ошибку: правильная версия держит
покрытие >= 80 % на всех азимутах, а версия с перестановкой — нет.
Пороги — ТЗ К1, Пакет 08 (audit/PACKET08_DECLARATION.md).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("scipy")

from experiments.openlin_lines import _ridge_nms  # noqa: E402

COVER_MIN = 0.80
SPREAD_MAX = 0.15
AZIMUTHS = (0.0, 45.0, 90.0, 135.0)


def _scene(az_deg, n=241, sigma_line=1.5, noise=0.35, seed=7):
    """Гребень гауссова профиля через центр под углом az (от оси x) на шуме."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((n, n)).astype("float64")
    c = (n - 1) / 2
    a = np.radians(az_deg)
    d = np.abs(-(xx - c) * np.sin(a) + (yy - c) * np.cos(a))        # расстояние до линии
    along = (xx - c) * np.cos(a) + (yy - c) * np.sin(a)
    score = np.exp(-0.5 * (d / sigma_line) ** 2) * (np.abs(along) <= 80)
    score += noise * rng.standard_normal((n, n))
    t = np.linspace(-70, 70, 141)                                     # след без концов
    trace = np.c_[np.rint(c + t * np.sin(a)), np.rint(c + t * np.cos(a))].astype(int)
    return score.astype("float32"), np.ones((n, n), bool), trace


def _cover(ridge, trace):
    """Доля ячеек следа, попавших в маску гребня, БЕЗ допуска в соседнюю ячейку.

    С допуском 3 x 3 ошибка не видна: подавление вдоль гребня рвёт его через
    ячейку, и разрывы закрываются соседями (проба 26.10.2026: с допуском
    версия с ошибкой давала 85-94 % у осевых азимутов, без допуска 23-30 %;
    верная версия без допуска — 87-99 %, как в замере 23.09)."""
    return float(ridge[trace[:, 0], trace[:, 1]].mean())


def _ridge_nms_swapped(score, valid, sigma_px):
    """Копия _ridge_nms с ошибкой до 23.09.2026 (компоненты переставлены)."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    num = gaussian_filter(np.where(valid, score, 0.0).astype("float32"), sigma_px)
    den = gaussian_filter(valid.astype("float32"), sigma_px)
    s = num / np.maximum(den, 1e-3)
    hxx = gaussian_filter(s, sigma_px, order=(0, 2))
    hyy = gaussian_filter(s, sigma_px, order=(2, 0))
    hxy = gaussian_filter(s, sigma_px, order=(1, 1))
    lam_min = 0.5 * (hxx + hyy) - np.sqrt((0.5 * (hxx - hyy)) ** 2 + hxy ** 2)
    vx, vy = lam_min - hxx, hxy                                       # ОШИБКА
    nrm = np.hypot(vx, vy)
    flat = nrm < 1e-12
    vx = np.where(flat, 1.0, vx / np.maximum(nrm, 1e-12))
    vy = np.where(flat, 0.0, vy / np.maximum(nrm, 1e-12))
    yy, xx = np.indices(s.shape).astype("float32")
    fwd = map_coordinates(s, [yy + vy, xx + vx], order=1, mode="nearest")
    bwd = map_coordinates(s, [yy - vy, xx - vx], order=1, mode="nearest")
    return s, valid & (lam_min < 0) & (s >= fwd) & (s >= bwd)


def _covers(fn):
    out = {}
    for az in AZIMUTHS:
        score, valid, trace = _scene(az)
        _s, ridge = fn(score, valid, 1.0)
        out[az] = _cover(ridge, trace)
    return out


def test_ridge_nms_sees_all_azimuths():
    cov = _covers(_ridge_nms)
    assert min(cov.values()) >= COVER_MIN, cov
    assert max(cov.values()) - min(cov.values()) <= SPREAD_MAX, cov


def test_swapped_eigenvector_fails_on_axes():
    cov = _covers(_ridge_nms_swapped)
    assert min(cov[0.0], cov[90.0]) < COVER_MIN, cov
