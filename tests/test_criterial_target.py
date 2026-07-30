"""Тесты пайплайна «обучение на критериальном анализе» (пункт 3 постановки)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.criterial_target import (  # noqa: E402
    _random_object_placement,
    build_dataset,
    label_ore_objects,
    leave_one_object_out,
    permutation_significance,
    training_features,
)

PGRID = config.GOLD_TARGET_PGRID
DATASET = config.PROCESSED_DIR / "dataset_v1.npz"


def test_label_ore_objects_ranks_by_size():
    grid = np.full((10, 10), 0.5, dtype=np.float32)
    grid[0:1, 0:3] = 0.05      # 3 ячейки — второй по размеру
    grid[5:9, 5:9] = 0.05      # 16 ячеек — самый крупный
    grid[9:10, 0:1] = 0.05     # 1 ячейка — ниже min_cells, шум (не соседствует с другими)

    labels = label_ore_objects(grid, threshold=0.15, min_cells=2, n_objects=3)
    assert set(np.unique(labels)) == {0, 1, 2}      # третий кандидат отсечён min_cells
    assert (labels == 1).sum() == 16                # ранг 1 — крупнейшая компонента
    assert (labels == 2).sum() == 3
    assert labels[9, 0] == 0                         # шумовая ячейка не попала в объекты


def test_label_ore_objects_threshold_is_absolute_not_quantile():
    grid = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
    labels = label_ore_objects(grid, threshold=0.15, min_cells=1, n_objects=5)
    assert (grid[labels > 0] <= 0.15).all()
    assert (labels > 0).sum() < grid.size * 0.5      # не половина сетки, как было бы при квантиле 0.5


def test_exclude_factor_features_kept_minimal():
    excl = set(config.CRIT_EXCLUDE_FACTOR_FEATURES)
    # гидросеть и служебная маска — НЕ факторные признаки критериального, должны остаться
    assert "dist_dnl" not in excl
    assert "dist_dnara" not in excl
    assert "mask_svita" not in excl
    # а прямые факторные слои — обязаны быть исключены
    for f in ("dist_tect1", "dist_tect2", "dens_tect", "dist_magm", "dens_magm",
              "dist_struct", "dist_facies", "dist_paleo"):
        assert f in excl


def test_coverage_pool_excludes_train_budget():
    from src.validation import _coverage

    score = np.array([1.0, 0.9, 0.8, 0.2, 0.15, 0.1, 0.05, 0.0, 0.3, 0.25])
    held = np.array([8, 9])
    # порог по всей сетке: top-30% съедают «обучающие» ячейки 0-2, held мимо
    assert _coverage(score, held, 0.3) == 0.0
    # порог по eval-пулу (без ячеек 0-2): held — лучшие в пуле, полный захват
    pool = np.arange(3, 10)
    assert _coverage(score, held, 0.3, pool=pool) == 1.0


def test_permutation_significance_shape():
    rng = np.random.default_rng(0)
    n = 200
    score = rng.random(n)
    held_idx = np.arange(0, 10)
    train_idx = np.arange(10, 60)
    result = permutation_significance(score, held_idx, train_idx, area=0.1, n_perm=20, seed=1)
    assert set(result) == {"observed_lift", "null_mean", "null_q95", "p_value"}
    assert 0.0 < result["p_value"] <= 1.0   # сглаживание Phipson-Smyth: p=0 невозможен


def test_random_object_placement_preserves_shape():
    """Размещение — тот же контур (с точностью до сдвига/поворота/отражения)."""
    shape = (30, 40)
    # Г-образный объект из 4 ячеек
    rel_r = np.array([0, 1, 2, 2])
    rel_c = np.array([0, 0, 0, 1])
    valid = np.ones(shape[0] * shape[1], dtype=bool)
    rng = np.random.default_rng(7)
    for _ in range(50):
        idx = _random_object_placement(rel_r, rel_c, shape, valid, rng)
        assert idx is not None
        assert idx.size == 4 and np.unique(idx).size == 4
        assert (idx >= 0).all() and (idx < shape[0] * shape[1]).all()
        r, c = np.divmod(idx, shape[1])
        # Г-фигура при любом элементе диэдра остаётся в bbox 3x2 или 2x3
        hh, ww = r.max() - r.min() + 1, c.max() - c.min() + 1
        assert {hh, ww} == {2, 3}


def test_random_object_placement_respects_mask():
    shape = (10, 10)
    rel_r = np.array([0, 0])
    rel_c = np.array([0, 1])
    valid = np.zeros(100, dtype=bool)
    valid[55:57] = True                      # единственное допустимое место: ячейки 55-56
    rng = np.random.default_rng(0)
    placements = [
        idx for _ in range(3000)
        if (idx := _random_object_placement(rel_r, rel_c, shape, valid, rng)) is not None
    ]
    assert placements                         # место находится
    for idx in placements:
        assert set(idx.tolist()) == {55, 56}


def test_permutation_shape_preserving_null():
    """Компактный контур с высоким скором значим против форм-сохраняющего null."""
    shape = (20, 20)
    rng = np.random.default_rng(3)
    score = rng.random(400)
    held_idx = np.array([0, 1, 20, 21])       # блок 2x2 в углу
    score[held_idx] = 2.0                     # заведомый максимум
    result = permutation_significance(
        score, held_idx, train_idx=np.array([], dtype=int),
        area=0.1, n_perm=50, seed=1, shape=shape,
    )
    assert result["observed_lift"] == pytest.approx(10.0)
    assert result["p_value"] < 0.15           # 1/(1+50) + случайные попадания null на угол


@pytest.mark.skipif(not DATASET.exists(), reason="dataset_v1.npz не собран")
def test_training_features_excludes_factor_layers():
    df = training_features()
    for f in config.CRIT_EXCLUDE_FACTOR_FEATURES:
        assert f not in df.columns
    assert "dist_dnl" in df.columns
    assert "dist_dnara" in df.columns


@pytest.mark.skipif(not (PGRID.exists() and DATASET.exists()), reason="критериальная сетка/датасет не собраны")
def test_build_dataset_three_objects():
    X, labels_flat, coords, feature_names, meta = build_dataset()
    assert X.shape[0] == labels_flat.size == coords.shape[0]
    assert set(np.unique(labels_flat)) <= {-1, 0, 1, 2, 3}   # -1 — перспективные вне топ-3
    assert (labels_flat == -1).sum() > 0   # шумовые компоненты <50 ячеек существуют (см. config)
    assert (labels_flat > 0).sum() > 0
    assert not any(f in feature_names for f in config.CRIT_EXCLUDE_FACTOR_FEATURES)


@pytest.mark.skipif(not (PGRID.exists() and DATASET.exists()), reason="критериальная сетка/датасет не собраны")
def test_loo_train_sets_are_clean(monkeypatch):
    """Фон обучения не содержит перспективных ячеек, обучающие объекты — вне буфера контроля."""
    from scipy.spatial import cKDTree

    from src.criterial_target import load_prognoz_grid

    monkeypatch.setattr(config, "CRIT_N_BACKGROUND", 200)
    monkeypatch.setattr(config, "ENS_N_ROUNDS", 2)
    X, labels_flat, coords, _feature_names, _meta = build_dataset()
    _, prognoz = load_prognoz_grid()
    prognoz_flat = prognoz.ravel()

    buffer_removed_any = False
    for r in leave_one_object_out(X, labels_flat, coords, seed=0):
        train_idx = r["train_idx"]
        # перспективные вне топ-3 (метка -1) не участвуют в обучении вообще
        assert (labels_flat[train_idx] >= 0).all()
        bg = train_idx[labels_flat[train_idx] == 0]
        pos = train_idx[labels_flat[train_idx] > 0]
        # фон чист: ни одной критериально-перспективной ячейки
        assert (prognoz_flat[bg] > config.CRIT_TARGET_THRESHOLD).all()
        # и фон, и обучающие положительные — дальше буфера от контрольного объекта
        dist, _ = cKDTree(coords[r["held_idx"]]).query(coords[train_idx])
        assert (dist > config.CRIT_HOLDOUT_BUFFER_M).all()
        assert pos.size > 0
        if pos.size < r["summary"]["n_train_pos_total"]:
            buffer_removed_any = True
        # eval-пул: только контрольный объект + чистый фон, без обучающих ячеек
        ep = r["eval_pool"]
        assert np.intersect1d(ep, train_idx).size == 0
        assert set(np.unique(labels_flat[ep])) <= {0, r["summary"]["object"]}
        assert np.isin(r["held_idx"], ep).all()
    # буфер нетривиален: хотя бы в одном фолде он реально вырезал обучающие ячейки
    # (на датасете v1 объекты 1 и 3 лежат ближе буфера друг к другу)
    assert buffer_removed_any


@pytest.mark.skipif(not (PGRID.exists() and DATASET.exists()), reason="критериальная сетка/датасет не собраны")
def test_leave_one_object_out_smoke(monkeypatch):
    monkeypatch.setattr(config, "CRIT_N_BACKGROUND", 200)
    monkeypatch.setattr(config, "ENS_N_ROUNDS", 2)
    X, labels_flat, coords, _feature_names, _meta = build_dataset()
    results = leave_one_object_out(X, labels_flat, coords, seed=0)
    assert len(results) == config.CRIT_N_OBJECTS
    for r in results:
        assert r["score_all"].shape[0] == X.shape[0]
        assert 0 <= r["summary"]["lift@10%"]
