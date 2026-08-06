"""Юниты общего слоя спутниковых каталогов (этап 8) — синтетика, без сети.

Сеть здесь не трогается принципиально: проверяется арифметика, которая
превращает сцены в признаки, а не доступность чужого сервера. Всё, что зависит
от каталога (поиск, подпись ссылок, чтение COG), проверяется прогоном
``experiments/fetch_sat.py`` и его контролем качества.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, integro_grid, sat_sources, stac_grid


class _Item:
    """Минимальная замена pystac.Item: только то, что читает отбор сцен."""

    def __init__(self, ident, dt, cloud=None):
        self.id = ident
        self.datetime = dt
        self.properties = {} if cloud is None else {"eo:cloud_cover": cloud}


def _meta(prf=4, pic=3, dx=500.0):
    return integro_grid.GridMeta(obj_count=1, prop_count=1, pic=pic, prf=prf,
                                 dx=dx, dy=dx, x0=0.0, y0=0.0)


# ------------------------------------------------------------------ отбор сцен
def test_pick_prefers_clean_scenes():
    """Там, где облачность известна, отбираются самые чистые сцены."""
    items = [_Item(f"s{i}", pd.Timestamp(f"2020-07-0{i + 1}"), cloud=c)
             for i, c in enumerate([50.0, 5.0, 90.0, 20.0])]
    got = stac_grid._pick(items, 2)
    assert [i.id for i in got] == ["s1", "s3"]


def test_pick_spreads_over_time_without_cloud():
    """У радара облачности нет: выборка обязана растянуться по времени."""
    items = [_Item(f"s{i}", pd.Timestamp("2018-06-01") + pd.Timedelta(days=90 * i))
             for i in range(10)]
    got = stac_grid._pick(items, 3)
    assert len(got) == 3
    assert got[0].datetime == items[0].datetime and got[-1].datetime == items[-1].datetime


def test_pick_returns_all_when_group_is_small():
    items = [_Item("a", pd.Timestamp("2020-07-01"))]
    assert stac_grid._pick(items, 5) == items


# ------------------------------------------------------------------ композит
def test_median_stack_ignores_outlier_scene():
    """Одиночное облако смещает среднее, но не медиану — ради этого медиана и берётся."""
    base = np.full((4, 4), 2.0, dtype="float32")
    scenes = [{"b": base.copy()} for _ in range(4)]
    scenes[0]["b"][:] = 100.0                     # «облако» на всю сцену
    comp, n_obs = stac_grid.median_stack(scenes, min_obs=3)
    assert np.allclose(comp["b"], 2.0)
    assert (n_obs == 4).all()


def test_median_stack_blanks_thin_coverage():
    """Где наблюдений меньше min_obs — там NaN, а не «медиана по двум точкам»."""
    a = np.array([[1.0, np.nan], [1.0, np.nan]], dtype="float32")
    b = np.array([[3.0, 5.0], [np.nan, np.nan]], dtype="float32")
    comp, n_obs = stac_grid.median_stack([{"b": a}, {"b": b}], min_obs=2)
    assert np.isfinite(comp["b"][0, 0])
    assert np.isnan(comp["b"][0, 1]) and np.isnan(comp["b"][1, 0])
    assert n_obs[0, 0] == 2 and n_obs[1, 1] == 0


def test_median_stack_requires_scenes():
    with pytest.raises(ValueError):
        stac_grid.median_stack([])


# ------------------------------------------------------------------ агрегация
def test_to_cells_mean_std_and_coverage():
    """Ячейка = блок субпикселей: среднее, разброс внутри неё и доля с данными."""
    meta = _meta(prf=2, pic=2, dx=500.0)
    res = 250.0                                   # 2x2 субпикселя на ячейку
    arr = np.arange(16, dtype="float32").reshape(4, 4)
    arr[0, 0] = np.nan                            # один субпиксель без данных
    out = stac_grid.to_cells({"x": arr}, meta, res_m=res, prefix="t_")
    assert list(out.columns) == ["t_x", "t_x_std", "t_valid_frac"]
    assert len(out) == 4
    assert out["t_x"][0] == pytest.approx(np.nanmean(arr[:2, :2]))
    assert out["t_x_std"][0] == pytest.approx(np.nanstd(arr[:2, :2]))
    assert out["t_valid_frac"][0] == pytest.approx(0.75)
    assert out["t_valid_frac"][1] == pytest.approx(1.0)


def test_to_cells_keeps_std_as_separate_signal():
    """Ровная и пятнистая ячейки с одним средним обязаны различаться по std."""
    meta = _meta(prf=1, pic=2, dx=500.0)
    arr = np.array([[5.0, 5.0, 0.0, 10.0],
                    [5.0, 5.0, 10.0, 0.0]], dtype="float32")
    out = stac_grid.to_cells({"x": arr}, meta, res_m=250.0)
    assert out["x"][0] == pytest.approx(out["x"][1])
    assert out["x_std"][0] == pytest.approx(0.0)
    assert out["x_std"][1] > 4.0


# ------------------------------------------------------------------ радар
def test_to_db_converts_and_floors():
    """Мощность -> дБ, нули и отрицательные значения не уходят в -inf."""
    p = np.array([1.0, 0.1, 0.0, -1.0, np.nan], dtype="float32")
    db = stac_grid.to_db(p, floor_db=-30.0)
    assert db[0] == pytest.approx(0.0)
    assert db[1] == pytest.approx(-10.0)
    assert np.isnan(db[2]) and np.isnan(db[3]) and np.isnan(db[4])
    assert np.nanmin(stac_grid.to_db(np.array([1e-9]), floor_db=-30.0)) == -30.0


# ------------------------------------------------------------------ индексы
def test_ratio_handles_zero_denominator():
    """Деление на ноль даёт NaN, а не бесконечность, иначе NaN расползётся дальше."""
    comp = {"a": np.array([2.0, 4.0]), "b": np.array([1.0, 0.0])}
    r = sat_sources._ratio(comp, ("a",), ("b",))
    assert r[0] == pytest.approx(2.0) and np.isnan(r[1])


def test_ratio_sums_bands():
    """Индексы вида (b5+b7)/b6 считаются по суммам, а не по первому каналу."""
    comp = {"b05": np.array([1.0]), "b06": np.array([2.0]), "b07": np.array([3.0])}
    assert sat_sources._ratio(comp, ("b05", "b07"), ("b06",))[0] == pytest.approx(2.0)


@pytest.mark.parametrize("registry,bands", [("AST_INDICES", "AST_BANDS"),
                                            ("L8_RATIOS", "L8_BANDS")])
def test_indices_reference_existing_bands(registry, bands):
    """Индекс обязан ссылаться на реальный канал, а числитель — быть КОРТЕЖЕМ.

    Строка вместо кортежа не ловится глазами и не роняет импорт: ``_ratio``
    переберёт её по буквам и упадёт уже после скачивания всех сцен (так и
    случилось на прогоне Landsat — KeyError 'r' спустя 19 минут закачки).
    """
    known = getattr(sat_sources, bands)
    for name, (num, den) in getattr(sat_sources, registry).items():
        assert isinstance(num, tuple) and isinstance(den, tuple), \
            f"{name}: каналы должны быть кортежами, а не строкой"
        for b in num + den:
            assert b in known, f"{name}: нет канала {b}"


def test_sensor_registry_is_callable():
    """Реестр сенсоров не должен разъезжаться с функциями."""
    assert set(sat_sources.SENSORS) == {"s1", "l8", "psr", "ast", "astir"}
    for fn, title in sat_sources.SENSORS.values():
        assert callable(fn) and isinstance(title, str)


def test_s2_bands_include_red_edge():
    """Красный край добавлен в пул каналов S2 (этап 8) и не потерян правкой конфига."""
    assert {"b05", "b06", "b07"} <= set(config.S2_BANDS.values())
