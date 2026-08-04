"""Юниты водосборной заверки (этап 6) — на синтетике, без сети и без данных."""
import numpy as np
import pandas as pd
import pytest

from src import catchments


def _plane(prf=20, pic=20, slope=1.0):
    """Наклонная плоскость: высота растёт с номером строки."""
    r = np.arange(prf)[:, None] * slope
    return np.repeat(r, pic, axis=1).astype(float)


def test_flow_goes_downhill_on_plane():
    """На наклонной плоскости приёмник обязан лежать ниже по склону."""
    dem = _plane()
    prf, pic = dem.shape
    rec = catchments.flow_dirs_d8(dem)
    inner = [(r, c) for r in range(1, prf - 1) for c in range(1, pic - 1)]
    for r, c in inner:
        p = rec[r * pic + c]
        assert p >= 0, "внутренняя ячейка осталась без приёмника"
        assert dem.ravel()[p] <= dem[r, c], "сток пошёл вверх по склону"


def test_catchment_is_upslope_wedge():
    """Водосбор точки на плоскости — клин ВЫШЕ неё, не ниже."""
    dem = _plane()
    prf, pic = dem.shape
    rec = catchments.flow_dirs_d8(dem)
    seed = 10 * pic + 10
    cells = catchments.upstream_catchment(rec, seed, max_cells=500)
    rows = cells // pic
    assert seed in cells, "сама точка обязана входить в свой водосбор"
    assert (dem.ravel()[cells] >= dem.ravel()[seed]).all(), \
        "в водосбор попали ячейки ниже точки по течению"
    assert rows.min() >= 10


def test_catchment_respects_size_cap():
    """На плоскости водосбор — узкая цепочка, в долине — широкий клин.

    Ограничение размера обязано срабатывать именно во втором случае: это тот
    самый выположенный рельеф, на котором водосбор без ограничения расползается
    на пол-листа.
    """
    prf, pic = 20, 20
    dem = _plane(prf, pic)
    rec = catchments.flow_dirs_d8(dem)
    chain = catchments.upstream_catchment(rec, 15 * pic + 10, max_cells=7)
    assert chain.size <= 7

    axis = np.abs(np.arange(pic) - pic // 2)[None, :] * 0.5
    valley = dem + axis                      # сходящаяся к оси долина
    rec_v = catchments.flow_dirs_d8(valley)
    wedge = catchments.upstream_catchment(rec_v, 15 * pic + pic // 2,
                                          max_cells=500)
    assert wedge.size > 7, "в долине водосбор обязан быть шире цепочки"
    capped = catchments.upstream_catchment(rec_v, 15 * pic + pic // 2,
                                           max_cells=7)
    assert capped.size == 7


def test_pit_does_not_break_tree():
    """Замкнутое понижение не должно оставлять бессточных ячеек."""
    dem = _plane()
    dem[8:12, 8:12] = -50.0          # яма
    rec = catchments.flow_dirs_d8(dem)
    prf, pic = dem.shape
    inner = np.ones((prf, pic), bool)
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    assert (rec.reshape(prf, pic)[inner] >= 0).all()


def test_capture_is_one_for_random_map():
    """Нормировка: у случайной карты водосборная метрика ~1 при любом размере."""
    rng = np.random.default_rng(0)
    n = 4000
    score = rng.random(n)
    pool = np.arange(n)
    for size in (1, 10, 200):
        cs = [rng.choice(n, size=size, replace=False) for _ in range(300)]
        got = catchments.catchment_capture(score, cs, area=0.10, pool=pool)
        assert 0.8 < got < 1.2, f"размер {size}: метрика {got:.3f}, ожидалась ~1"


def test_capture_rewards_hitting_catchment():
    """Карта, подсветившая водосборы, обязана обойти карту, подсветившую фон."""
    n = 4000
    pool = np.arange(n)
    cs = [np.arange(i, i + 20) for i in range(0, 400, 20)]
    inside = np.concatenate(cs)
    good = np.zeros(n)
    good[inside] = 1.0
    bad = np.ones(n)
    bad[inside] = 0.0
    assert (catchments.catchment_capture(good, cs, 0.10, pool)
            > catchments.catchment_capture(bad, cs, 0.10, pool))


def test_build_catchments_uses_valid_mask():
    """Невалидные ячейки не попадают в водосбор."""
    prf, pic = 12, 12
    dem = _plane(prf, pic)
    df = pd.DataFrame({"dem_elev": dem.ravel()})
    meta = type("M", (), {"prf": prf, "pic": pic})()
    valid = np.ones(prf * pic, bool)
    valid[dem.ravel() > 8] = False        # верх склона объявляем невалидным
    seed = np.array([5 * pic + 5])
    cs = catchments.build_catchments(df, meta, seed, valid, max_cells=500)
    assert valid[cs[0]].all()


@pytest.mark.parametrize("size", [1, 5, 50])
def test_capture_zero_area_is_safe(size):
    """Вырожденный порог не должен ронять метрику в исключение."""
    n = 500
    score = np.zeros(n)
    cs = [np.arange(size)]
    assert catchments.catchment_capture(score, cs, 0.10, np.arange(n)) >= 0.0
