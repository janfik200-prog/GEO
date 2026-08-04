"""Юниты заверки по критериальному эталону (этап 7a) — синтетика, без данных."""
import numpy as np
import pytest

from src import crit_reference as cr
from src.integro_grid import GridMeta


def _meta(prf=40, pic=40):
    return GridMeta(obj_count=1, prop_count=1, pic=pic, prf=prf,
                    dx=500.0, dy=500.0, x0=0.0, y0=0.0)


def _smooth_field(prf, pic, seed=0, sigma=3.0):
    """Автокоррелированное поле: сглаженный шум (имитирует геофизическую карту)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((prf, pic)), sigma).ravel()


def test_reference_mask_takes_top_area():
    """Эталон = верхние area долей пула, не больше и не меньше."""
    crit = np.arange(1000, dtype=float)
    pool = np.arange(1000)
    ref = cr.reference_mask(crit, pool, area=0.10)
    assert 95 <= ref.sum() <= 105
    assert ref[-50:].all(), "верхние ячейки обязаны попасть в эталон"


def test_identical_map_is_perfect():
    """Карта, совпадающая с эталоном, даёт предельные значения всех метрик."""
    crit = _smooth_field(40, 40)
    pool = np.arange(crit.size)
    a = cr.agreement(crit, crit, pool, area=0.10)
    assert a["auc"] == pytest.approx(1.0, abs=1e-6)
    assert a["capture"] == pytest.approx(10.0, rel=0.05)
    assert a["kappa"] == pytest.approx(1.0, abs=1e-6)
    assert a["orient"] == 1.0


def test_inverted_map_is_recognized_by_orientation():
    """Перевёрнутый скор — та же карта: ориентация обязана это поймать."""
    crit = _smooth_field(40, 40)
    pool = np.arange(crit.size)
    assert cr.orientation(-crit, crit, pool) == -1
    a = cr.agreement(-crit, crit, pool, area=0.10)
    assert a["auc"] == pytest.approx(1.0, abs=1e-6)
    assert a["orient"] == -1.0


def test_random_map_is_chance_level():
    """У независимой случайной карты AUC ~ 0.5, capture ~ 1."""
    crit = _smooth_field(60, 60, seed=1)
    rng = np.random.default_rng(7)
    noise = rng.random(crit.size)
    pool = np.arange(crit.size)
    a = cr.agreement(noise, crit, pool, area=0.10)
    assert 0.42 < a["auc"] < 0.58
    assert 0.6 < a["capture"] < 1.5
    assert abs(a["kappa"]) < 0.15


def test_block_labels_cover_grid():
    """Блоки нарезают сетку без дыр и пересечений."""
    meta = _meta(40, 40)
    lab = cr.block_labels(meta, block=20)
    assert lab.size == 1600
    assert np.unique(lab).size == 4
    assert np.bincount(lab).max() == 400


def test_shift_null_separates_signal_from_noise():
    """Сдвиговый null: копия эталона значима, независимая карта — нет."""
    meta = _meta(40, 40)
    crit = _smooth_field(40, 40, seed=2)
    pool = np.arange(crit.size)
    good = cr.shift_null(crit, crit, meta, pool, area=0.10, n_shifts=99)
    bad = cr.shift_null(_smooth_field(40, 40, seed=99), crit, meta, pool,
                        area=0.10, n_shifts=99)
    assert good["p"] <= 0.02, "точная копия эталона обязана быть значимой"
    assert bad["p"] > 0.05, f"независимое поле дало p={bad['p']:.3f}"


def test_shift_null_uses_fixed_orientation():
    """Знак фиксируется по наблюдению и не переигрывается на каждом сдвиге."""
    meta = _meta(40, 40)
    crit = _smooth_field(40, 40, seed=3)
    pool = np.arange(crit.size)
    plus = cr.shift_null(crit, crit, meta, pool, n_shifts=49, seed=1)
    minus = cr.shift_null(-crit, crit, meta, pool, n_shifts=49, seed=1)
    assert plus["observed"] == pytest.approx(minus["observed"], abs=1e-9)


def test_block_bootstrap_interval_contains_observed():
    """Наблюдённый AUC обязан лежать внутри блочного интервала."""
    meta = _meta(40, 40)
    crit = _smooth_field(40, 40, seed=4)
    rng = np.random.default_rng(5)
    noisy = crit + 0.5 * rng.standard_normal(crit.size)
    pool = np.arange(crit.size)
    b = cr.block_bootstrap(noisy, crit, meta, pool, block=10, n_boot=100)
    assert b["auc_ci_lo"] <= b["auc_a"] <= b["auc_ci_hi"]
    assert b["n_blocks"] == 16


def test_block_bootstrap_delta_of_identical_maps_covers_zero():
    """Две одинаковые карты не должны различаться значимо."""
    meta = _meta(40, 40)
    crit = _smooth_field(40, 40, seed=6)
    pool = np.arange(crit.size)
    b = cr.block_bootstrap(crit, crit, meta, pool, score_b=crit.copy(),
                           block=10, n_boot=100)
    assert b["delta"] == pytest.approx(0.0, abs=1e-9)
    assert b["delta_ci_lo"] <= 0 <= b["delta_ci_hi"]


def test_block_bootstrap_is_wider_than_cellwise():
    """Блочный интервал обязан быть шире поячеечного — иначе он бесполезен."""
    meta = _meta(60, 60)
    crit = _smooth_field(60, 60, seed=8)
    rng = np.random.default_rng(9)
    noisy = crit + 1.0 * rng.standard_normal(crit.size)
    pool = np.arange(crit.size)
    wide = cr.block_bootstrap(noisy, crit, meta, pool, block=20, n_boot=200)
    narrow = cr.block_bootstrap(noisy, crit, meta, pool, block=1, n_boot=200)
    assert (wide["auc_ci_hi"] - wide["auc_ci_lo"]) > \
           (narrow["auc_ci_hi"] - narrow["auc_ci_lo"])


def test_kappa_edges():
    """Каппа: полное совпадение = 1, независимость ~ 0."""
    rng = np.random.default_rng(0)
    a = rng.random(5000) < 0.1
    b = rng.random(5000) < 0.1
    assert cr._kappa(a, a) == pytest.approx(1.0)
    assert abs(cr._kappa(a, b)) < 0.1


def test_protocol_verdict_has_no_superiority_column():
    """Вердикт по эталону не может утверждать превосходство над эталоном."""
    meta = _meta(30, 30)
    crit = _smooth_field(30, 30, seed=11)
    valid = np.ones(crit.size, dtype=bool)
    rng = np.random.default_rng(3)
    res = cr.run_protocol({"copy": crit.copy(), "noise": rng.random(crit.size)},
                          crit, meta, valid, alpha=0.05)
    assert "superior" not in res["verdict"].columns
    assert "reproduces" in res["verdict"].columns
    assert res["verdict"].iloc[0]["method"] == "copy"
