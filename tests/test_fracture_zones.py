"""Тесты src/fracture_zones.py на синтетике (без сети, STAC и датасета)."""
import numpy as np
import pandas as pd
import pytest

from src import config, fracture_zones as fz


class _Meta:
    """Заглушка GridMeta: to_cells нужны только шаг и размер сетки."""

    def __init__(self, prf, pic, dx, dy):
        self.prf, self.pic, self.dx, self.dy = prf, pic, dx, dy


def _stripes(shape=(200, 200), grad_deg=0.0, period_px=6.0):
    """Параллельные полосы: градиент под ``grad_deg``, ось полос — перпендикуляр."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    th = np.radians(grad_deg)
    return np.sin((xx * np.cos(th) + yy * np.sin(th)) * (2 * np.pi / period_px))


def _mixed(shape=(200, 200), seed=0):
    """Полосы слева, шум справа — поле, у которого когерентность МЕНЯЕТСЯ по площади.

    Ранговые метрики на равномерном поле не определены, а реальная карта
    однородной не бывает: структура есть на части листа, а не везде.
    """
    img = _stripes(shape)
    half = shape[1] // 2
    img[:, half:] = np.random.default_rng(seed).normal(size=(shape[0], shape[1] - half))
    return img


# --- Структурный тензор ---

def test_coherence_high_for_stripes_low_for_noise():
    """Когерентность разделяет направленную структуру и изотропный шум."""
    coh_stripes = np.nanmean(fz.structure_tensor(_stripes(), 1000.0, res_m=100.0)["coh"])
    noise = np.random.default_rng(0).normal(size=(200, 200))
    coh_noise = np.nanmean(fz.structure_tensor(noise, 1000.0, res_m=100.0)["coh"])
    assert coh_stripes > 0.9
    assert coh_noise < 0.1


@pytest.mark.parametrize("grad_deg", [0.0, 30.0, 60.0, 135.0])
def test_direction_recovers_stripe_axis(grad_deg):
    """Ось структуры — перпендикуляр к градиенту, в растровой системе arctan2(drow, dcol)."""
    st = fz.structure_tensor(_stripes(grad_deg=grad_deg), 1000.0, res_m=100.0)
    coh = np.nanmean(st["coh"])
    s, c = np.nanmean(st["dir_sin"]) / coh, np.nanmean(st["dir_cos"]) / coh
    got = np.degrees(np.arctan2(s, c)) / 2.0 % 180.0
    th = np.radians(grad_deg)
    want = np.degrees(np.arctan2(np.cos(th), -np.sin(th))) % 180.0
    # 0 и 180 — одна ось, поэтому сравнение по кратчайшей дуге.
    # Допуск — половина лепестка розы азимутов (config.LIN_SAT_ROSE_BINS = 18,
    # то есть 10°): на косых направлениях конечные разности на пиксельной сетке
    # анизотропны и стоят пары градусов, но до различимости в розе это далеко.
    diff = abs(got - want) % 180.0
    tol = (180.0 / config.LIN_SAT_ROSE_BINS) / 2.0
    assert min(diff, 180.0 - diff) < tol


def test_all_nan_image_returns_nan_not_crash():
    st = fz.structure_tensor(np.full((50, 50), np.nan), 1000.0, res_m=100.0)
    assert set(st) == {"coh", "energy", "dir_sin", "dir_cos"}
    assert all(np.isnan(v).all() for v in st.values())


def test_nan_mask_is_preserved():
    """Заполнение пропусков нужно гауссиане, но не должно выдавать себя за замер."""
    img = _stripes()
    img[20:40, 20:40] = np.nan
    st = fz.structure_tensor(img, 1000.0, res_m=100.0)
    assert np.isnan(st["coh"][20:40, 20:40]).all()
    assert np.isfinite(st["coh"][100:120, 100:120]).all()


def test_zone_field_has_every_scale():
    keys = set(fz.zone_field(_stripes(), scales_m=(300.0, 1000.0), res_m=100.0))
    assert keys == {f"{n}_{t}" for t in ("300", "1km")
                    for n in ("coh", "energy", "dir_sin", "dir_cos")}


def test_candidate_column_names_match_zone_field():
    """Имена, которые блок 2 спрашивает у поля, обязаны в поле существовать.

    Блок 2 берёт кандидатов как ``coh_<масштаб>`` из :func:`fz.zone_field` и
    ``fz<источник>_coh_<масштаб>`` из :func:`fz.zone_features`. Разъедься эти
    два имени — прогон упал бы уже после часов закачки композитов.
    """
    meta = _Meta(prf=10, pic=10, dx=500.0, dy=500.0)
    img = _mixed(shape=(50, 50))
    field = fz.zone_field(img, res_m=100.0)
    feats = fz.zone_features(meta, img, "fzs1_", res_m=100.0)
    for s in config.FZ_SCALES_M:
        tag = fz._scale_tag(s)
        assert f"coh_{tag}" in field
        assert f"fzs1_coh_{tag}" in feats.columns


def test_zone_features_shape_and_prefix():
    meta = _Meta(prf=10, pic=10, dx=500.0, dy=500.0)
    feats = fz.zone_features(meta, _stripes(shape=(50, 50)), "fzs1_",
                             res_m=100.0, scales_m=(300.0,))
    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == meta.prf * meta.pic
    assert "fzs1_coh_300" in feats.columns and "fzs1_coh_300_std" in feats.columns


# --- Стоп-правило воспроизводимости ---

def test_reproducibility_identical_fields_high_rho():
    f = fz.structure_tensor(_mixed(), 1000.0, res_m=100.0)["coh"]
    assert fz.reproducibility(f, f)["rho"] > 0.99


def test_reproducibility_independent_noise_low_rho():
    rng = np.random.default_rng(1)
    a = fz.structure_tensor(rng.normal(size=(150, 150)), 1000.0, res_m=100.0)["coh"]
    b = fz.structure_tensor(rng.normal(size=(150, 150)), 1000.0, res_m=100.0)["coh"]
    assert abs(fz.reproducibility(a, b)["rho"]) < config.LIN_MIN_SPEARMAN


def test_reproducibility_covers_whole_area_unlike_thresholded_branch():
    """Главное отличие от пороговой ветки: метрика считается по всей площади.

    У src.lineaments_sat.reproducibility 90%+ карты были нулями у обеих половин
    и сравнение приходилось ограничивать объединением ненулевых пикселей.
    """
    f = fz.structure_tensor(_mixed(), 1000.0, res_m=100.0)["coh"]
    assert fz.reproducibility(f, f)["covered_frac"] == pytest.approx(1.0)


# --- Ранговый остаток и частная связь ---

def test_rank_residual_removes_dependence_on_control():
    rng = np.random.default_rng(2)
    ctrl = rng.normal(size=2000)
    x = 3.0 * ctrl + 0.1 * rng.normal(size=2000)
    resid = fz.rank_residual(x, ctrl)
    from scipy.stats import spearmanr
    assert abs(spearmanr(resid, ctrl).statistic) < 0.05


def test_rank_residual_invariant_to_monotone_rescaling_of_control():
    """Ранговый остаток не зависит от шкалы контроля — то, ради чего он ранговый."""
    rng = np.random.default_rng(3)
    ctrl = rng.uniform(1.0, 5.0, size=1000)
    x = rng.normal(size=1000) + ctrl
    a = fz.rank_residual(x, ctrl)
    b = fz.rank_residual(x, np.exp(ctrl))       # монотонная замена шкалы
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_rank_residual_propagates_nan_as_missing():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0] * 5)
    ctrl = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0] * 5)
    resid = fz.rank_residual(x, ctrl)
    assert np.isnan(resid[1]) and np.isnan(resid[2])
    assert np.isfinite(resid[0])


def test_rank_residual_without_controls_is_passthrough():
    x = np.array([3.0, 1.0, np.nan, 2.0])
    out = fz.rank_residual(x, None)
    assert np.isnan(out[2])
    np.testing.assert_allclose(out[[0, 1, 3]], x[[0, 1, 3]])


def test_partial_spearman_kills_common_confounder():
    """Связь, целиком наведённая общим контролем, после его фиксации исчезает."""
    rng = np.random.default_rng(4)
    ctrl = rng.normal(size=3000)
    x = ctrl + 0.3 * rng.normal(size=3000)
    y = ctrl + 0.3 * rng.normal(size=3000)
    from scipy.stats import spearmanr
    assert spearmanr(x, y).statistic > 0.8          # сырая связь сильная
    assert abs(fz.partial_spearman(x, y, ctrl)) < 0.15   # частная — нет


def test_partial_spearman_keeps_genuine_link():
    """Обратный контроль: настоящую связь фиксация постороннего контроля не съедает."""
    rng = np.random.default_rng(5)
    ctrl = rng.normal(size=3000)
    x = rng.normal(size=3000)
    y = x + 0.3 * rng.normal(size=3000)
    assert fz.partial_spearman(x, y, ctrl) > 0.8


# --- Ячейки разломного коридора ---

def test_fault_corridor_cells_threshold_and_pool():
    dist = np.array([0.0, 400.0, 900.0, 1500.0, np.nan])
    np.testing.assert_array_equal(fz.fault_corridor_cells(dist, 1000.0), [0, 1, 2])
    np.testing.assert_array_equal(
        fz.fault_corridor_cells(dist, 1000.0, pool=np.array([1, 2, 3])), [1, 2])
