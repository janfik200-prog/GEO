"""Тесты пакета src: юниты чистых функций + smoke-тест пайплайна на синтетике.

Запуск из корня репозитория: ``python -m pytest -q``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

geopandas = pytest.importorskip("geopandas")
from shapely.geometry import Point, box  # noqa: E402

from src import config  # noqa: E402
from src.utils import normalize_01, robust_normalize_01  # noqa: E402
from src.features import build_grid, build_features, distance_to_proximity  # noqa: E402
from src.model import (  # noqa: E402
    mark_presence, train_model, compute_prospectivity, sample_presence_background,
)
from src.validation import _coverage, assign_spatial_blocks  # noqa: E402


# --- Юниты: нормализация ---

def test_normalize_01_range():
    out = normalize_01([0.0, 5.0, 10.0])
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize_01_constant_is_half():
    out = normalize_01([3.0, 3.0, 3.0])
    assert np.allclose(out, 0.5)


def test_normalize_01_preserves_nan():
    out = normalize_01([1.0, np.nan, 2.0])
    assert np.isnan(out[1])


def test_robust_normalize_clips_outlier():
    out = robust_normalize_01([0, 1, 2, 3, 1000], 0.1, 0.9)
    assert out.max() <= 1.0 and out.min() >= 0.0


# --- Юниты: близость ---

def test_distance_to_proximity_monotonic():
    prox = distance_to_proximity([0.0, 100.0, 1000.0], "sqrt", 0.75)
    assert prox[0] > prox[1] > prox[2]  # ближе -> выше близость
    assert prox[0] <= 1.0 and prox[-1] >= 0.0


# --- Юниты: presence-background и coverage ---

def _toy_grid(n=200):
    cells = [(i, i // 20, i % 20, box(i, 0, i + 1, 1)) for i in range(n)]
    return geopandas.GeoDataFrame(
        cells, columns=["cell_id", "row", "col", "geometry"], geometry="geometry"
    )


def test_sample_presence_background_shapes_and_determinism():
    grid = _toy_grid()
    pos = [1, 2, 3]
    s1, y1 = sample_presence_background(grid, pos, n_background=50, seed=42)
    s2, y2 = sample_presence_background(grid, pos, n_background=50, seed=42)
    assert y1.sum() == len(pos)
    assert len(s1) == len(pos) + 50
    assert np.array_equal(s1, s2)  # детерминизм по сиду


def test_coverage_perfect_ranking():
    score = np.arange(100, dtype=float)
    top_positions = np.array([99, 98, 97])  # самые высокие
    assert _coverage(score, top_positions, area=0.10) == 1.0


def test_assign_spatial_blocks_groups_neighbors():
    grid = _toy_grid()
    blocks = assign_spatial_blocks(grid, block_size=10)
    assert len(np.unique(blocks)) < len(grid)  # ячейки группируются в блоки


# --- Smoke: мини-пайплайн на синтетических слоях ---

def _toy_layers():
    mask = geopandas.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)])
    line = geopandas.GeoDataFrame(geometry=[box(400, 0, 600, 1000)])
    # точки в центрах ячеек столбца x=[400,500) при сетке 100 м
    pts = geopandas.GeoDataFrame(geometry=[Point(450, 50 + 100 * r) for r in range(10)])
    layers = {r: line for r in ("facies", "paleo", "struct", "magm", "tect1", "tect2")}
    layers["mask"] = mask
    return mask, layers, pts


def test_pipeline_smoke_runs_and_scores():
    mask, layers, pts = _toy_layers()
    grid, mask_union, shape = build_grid(mask, cell_size=100)
    grid = build_features(grid, layers)
    grid, pos = mark_presence(grid, pts)
    assert len(pos) > 0
    # с малым числом точек модель не обучается (fallback 0.5), снижаем порог
    orig = config.MIN_POS_CELLS
    config.MIN_POS_CELLS = 1
    try:
        grid, model = train_model(grid, pos)
        grid = compute_prospectivity(grid, shape)
    finally:
        config.MIN_POS_CELLS = orig
    assert "prospectivity" in grid
    assert grid["prospectivity"].between(0, 1).all()
    assert model is not None
