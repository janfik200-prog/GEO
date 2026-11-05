"""Инварианты линеаментной ветки (experiments/openlin_*): первые тесты ветки.

Закрепляют то, что уже опубликовано или чинилось: цепочку счётчика проверок
(по образцу tests/test_potential_fields_edges.py), достижимость порогов,
ограничения ключей прогонов и азимутальную нейтральность трассировки.
Данные не нужны: все проверки на константах или на синтетике.

Запуск из корня репозитория: ``python -m pytest tests/test_openlin_invariants.py -q``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("scipy")
from experiments import openlin_transfer_check as tc  # noqa: E402
from experiments import openlin_zone_lines_v3 as v3  # noqa: E402

# (имя, prior, свои проверки, alpha, число сдвигов, опубликованный prior)
# Порядок и состав — по коду openlin_zone_lines_v3.py.
_RUNS = [
    ("ZL", v3.ZL_N_TESTS_PRIOR, v3.ZL_N_TESTS, v3.ZL_ALPHA, v3.SHIFT_N, 18),
    ("NDVI", v3.NDVI_N_TESTS_PRIOR, v3.NDVI_N_TESTS, v3.NDVI_ALPHA,
     v3.NDVI_SHIFT_N, 51),
    ("ALL", v3.ALL_N_TESTS_PRIOR, v3.ALL_N_TESTS, v3.ALL_ALPHA,
     v3.ALL_SHIFT_N, 60),
    ("CHAIN", v3.CHAIN_N_TESTS_PRIOR, v3.CHAIN_N_TESTS, v3.CHAIN_ALPHA,
     v3.CHAIN_SHIFT_N, 66),
    ("WIDE", v3.WIDE_N_TESTS_PRIOR, v3.WIDE_N_TESTS, v3.WIDE_ALPHA,
     v3.WIDE_SHIFT_N, 80),
    ("TWIN", v3.TWIN_N_TESTS_PRIOR, v3.TWIN_N_TESTS, v3.TWIN_ALPHA,
     v3.TWIN_SHIFT_N, 86),
    ("CLEAN", v3.CLEAN_N_TESTS_PRIOR, v3.CLEAN_N_TESTS, v3.CLEAN_ALPHA,
     v3.CLEAN_SHIFT_N, 92),
    ("CLEAN1K", v3.CLEAN1K_N_TESTS_PRIOR, v3.CLEAN1K_N_TESTS, v3.CLEAN1K_ALPHA,
     v3.CLEAN1K_SHIFT_N, 98),
    ("XFER", v3.XFER_N_TESTS_PRIOR, v3.XFER_N_TESTS, v3.XFER_ALPHA,
     v3.XFER_SHIFT_N, 104),
    ("UTS0", v3.UTS0_N_TESTS_PRIOR, v3.UTS0_N_TESTS, v3.UTS0_ALPHA,
     v3.UTS0_SHIFT_N, 118),
]


# ---------------- цепочка счётчика ----------------

@pytest.mark.parametrize("name,prior,_n,_a,_s,published", _RUNS,
                         ids=[r[0] for r in _RUNS])
def test_prior_pinned_to_published_value(name, prior, _n, _a, _s, published):
    """Опубликованный prior не двигается: правка слагаемого сдвигает пороги
    уже опубликованных прогонов (ср. цепочку 92/108 у потенциальных полей)."""
    assert prior == published, name


def test_chain_chain_is_continuous_where_declared():
    """Prior следующего = prior + свои предыдущего для звеньев, связанных в
    коде выражением: TWIN -> CLEAN -> CLEAN1K -> XFER, ALL -> CHAIN."""
    assert v3.CHAIN_N_TESTS_PRIOR == v3.ALL_N_TESTS_PRIOR + v3.ALL_N_TESTS
    assert v3.CLEAN_N_TESTS_PRIOR == v3.TWIN_N_TESTS_PRIOR + v3.TWIN_N_TESTS
    assert v3.CLEAN1K_N_TESTS_PRIOR == v3.CLEAN_N_TESTS_PRIOR + v3.CLEAN_N_TESTS
    assert v3.XFER_N_TESTS_PRIOR == v3.CLEAN1K_N_TESTS_PRIOR + v3.CLEAN1K_N_TESTS
    assert v3.TWIN_N_TESTS_PRIOR == v3.WIDE_N_TESTS_PRIOR + v3.WIDE_N_TESTS
    assert v3.NMS_N_TESTS_PRIOR == v3.EXT_N_TESTS_PRIOR + v3.EXT_N_TESTS
    assert v3.NDVI_N_TESTS_PRIOR == v3.RED_N_TESTS_TOTAL == 51


def test_uts0_gap_is_the_declared_eight():
    """UTS0 = XFER + свои 6 + 4 (DEM) + 4 (DEM1, с другого компьютера) = 118."""
    assert v3.XFER_N_TESTS == 6
    assert v3.UTS0_N_TESTS_PRIOR - (v3.XFER_N_TESTS_PRIOR + v3.XFER_N_TESTS) == 8
    assert v3.XFER_N_TESTS_PRIOR + v3.XFER_N_TESTS == 110


def test_other_literal_jumps_are_the_known_ones():
    """Два прежних разрыва записаны в коде литералами: ALL 60 (после NDVI 57 —
    вторая ветка openlin_bands, +3) и WIDE 80 (после CHAIN 72 — слой по рельефу,
    +8). Тест фиксирует их размер: молча изменить нельзя."""
    assert v3.ALL_N_TESTS_PRIOR - (v3.NDVI_N_TESTS_PRIOR + v3.NDVI_N_TESTS) == 3
    assert v3.WIDE_N_TESTS_PRIOR - (v3.CHAIN_N_TESTS_PRIOR + v3.CHAIN_N_TESTS) == 8


@pytest.mark.parametrize("name,prior,n,alpha,_s,_p", _RUNS,
                         ids=[r[0] for r in _RUNS])
def test_alpha_is_bonferroni_over_prior_plus_own(name, prior, n, alpha, _s, _p):
    assert alpha == pytest.approx(0.05 / (prior + n)), name


# ---------------- достижимость ----------------

@pytest.mark.parametrize("name,prior,n,alpha,n_shifts,_p", _RUNS,
                         ids=[r[0] for r in _RUNS])
def test_threshold_is_reachable(name, prior, n, alpha, n_shifts, _p):
    """Пол p = 1/(n_shifts+1) не выше alpha/2 (запас не меньше 2)."""
    floor = 1.0 / (n_shifts + 1)
    assert floor <= alpha / 2, f"{name}: пол {floor:.5f} > alpha/2 {alpha / 2:.5f}"


def test_threshold_reachable_for_red_channel():
    assert 1.0 / (v3.SHIFT_N + 1) <= v3.SHIFT_P_MAX / 2


def test_gtk_counter_and_reachability():
    assert tc.GTK_N_TESTS == 2
    assert tc.GTK_ALPHA == pytest.approx(0.025)
    assert tc.GTK_SHIFT_N == 159
    assert 1.0 / (tc.GTK_SHIFT_N + 1) <= tc.GTK_ALPHA / 2


# ---------------- ограничения ключей ----------------

def _run_v3(monkeypatch, argv):
    """main() v3 без данных: np.load подменён, build_halves — маркер «дошли»."""
    fake = {"res_m": np.float64(20.0), "shade": np.zeros((4, 4), "float32"),
            "ndvi": np.zeros((4, 4), "float32")}
    monkeypatch.setattr(v3.np, "load", lambda *a, **k: dict(fake))

    def reached(*a, **k):
        raise RuntimeError("дошли до сборки половин")

    monkeypatch.setattr(v3, "build_halves", reached)
    monkeypatch.setattr(sys, "argv", ["openlin_zone_lines_v3"] + argv)
    v3.main()


def test_xfer_key_only_for_lapland(monkeypatch):
    with pytest.raises(SystemExit, match="перенос объявлен"):
        _run_v3(monkeypatch, ["--site", "utsjoki", "--перенос"])
    with pytest.raises(SystemExit, match="перенос объявлен"):
        _run_v3(monkeypatch, ["--site", "anabar", "--перенос"])
    with pytest.raises(RuntimeError, match="дошли"):      # lapland проходит проверку
        _run_v3(monkeypatch, ["--site", "lapland", "--перенос"])


def test_tune0_key_only_for_utsjoki(monkeypatch):
    with pytest.raises(SystemExit, match="настройки"):
        _run_v3(monkeypatch, ["--site", "lapland", "--настройка0"])
    with pytest.raises(SystemExit, match="настройки"):
        _run_v3(monkeypatch, ["--site", "anabar", "--настройка0"])
    with pytest.raises(RuntimeError, match="дошли"):
        _run_v3(monkeypatch, ["--site", "utsjoki", "--настройка0"])


def test_xfer_and_tune0_are_mutually_exclusive(monkeypatch):
    with pytest.raises(SystemExit):
        _run_v3(monkeypatch, ["--site", "utsjoki", "--перенос", "--настройка0"])


def test_loss_by_stage_refuses_non_tuning_site(monkeypatch):
    from experiments import openlin_loss_by_stage as lbs
    assert lbs.TUNE_SITES == ("utsjoki",)
    for site in ("lapland", "anabar"):
        monkeypatch.setattr(sys, "argv", ["openlin_loss_by_stage", "--site", site])
        with pytest.raises(SystemExit, match="площадке настройки"):
            lbs.main()


# ---------------- азимутальная нейтральность трассировки ----------------

def _stripe(theta_deg, n=400, sigma=1.5, noise=0.1, seed=0):
    """Одна прямая яркая полоса через центр под азимутом theta (от севера по
    часовой; север — вверх, то есть против оси строк) на шумном фоне."""
    t = np.radians(theta_deg)
    d_row, d_col = -np.cos(t), np.sin(t)
    yy, xx = np.mgrid[:n, :n].astype(float)
    dist = np.abs((yy - n / 2) * d_col - (xx - n / 2) * d_row)
    rng = np.random.default_rng(seed)
    img = np.exp(-dist ** 2 / (2 * sigma ** 2)) + noise * rng.standard_normal((n, n))
    return img.astype("float32")


@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0, 135.0])
def test_trace_is_azimuth_neutral(theta):
    """Старая ошибка утончения (до 23.09.2026): слепота к меридиональным и
    широтным линиям. Самая длинная линия обязана лежать под азимутом полосы.

    Допуск 2 градуса: на линии в 5 км (250 ячеек по 20 м) шаг сетки даёт
    ошибку направления порядка atan(1/250) = 0.23 градуса; запас ~9x. Длина —
    не меньше 70 % ожидаемой (полоса идёт через весь кадр 400 ячеек = 8 км;
    диагональ длиннее на sqrt(2)); полное отсутствие линии — провал.
    """
    from experiments.openlin_zone_lines import trace
    n, res = 400, 20.0
    img = _stripe(theta, n=n)
    lines = trace(img, np.ones((n, n), bool), res)
    assert lines, f"азимут {theta}: ни одной линии (слепота к направлению)"
    best = max(lines, key=lambda r: r["длина_км"])
    diff = abs(best["азимут"] - theta) % 180.0
    diff = min(diff, 180.0 - diff)                  # азимут осевой: 0 == 180
    assert diff < 2.0, f"азимут {theta}: получен {best['азимут']:.1f}"
    expected_km = n * res / 1000 * (1.0 if theta % 90 == 0 else 2 ** 0.5)
    assert best["длина_км"] >= 0.7 * expected_km


# ---------------- кэш сцен по площадкам (аудит 05.11.2026) ----------------

def test_scene_cache_separate_per_site():
    """Утсйоки брал из общего кэша 5 сцен, вырезанных по сетке Лапландии."""
    from experiments import openlin_lowsun as ls
    a, l, u = (ls.scene_cache(s) for s in ("anabar", "lapland", "utsjoki"))
    assert len({a, l, u}) == 3
    assert a == ls.CACHE
    assert "lapland" in l.parts and "utsjoki" in u.parts
    assert l != ls.CACHE and u != ls.CACHE


def _fake_item_and_grid():
    from datetime import datetime
    from types import SimpleNamespace as NS
    item = NS(id="S2X_TEST", datetime=datetime(2026, 3, 1, 12),
              assets={"B04": NS(href="b04"), "B08": NS(href="b08")})
    return [(10.0, 180.0, item)], NS(res_m=20.0)


def test_fetch_writes_to_cache_dir_and_not_shared(monkeypatch, tmp_path):
    pytest.importorskip("rasterio")
    from experiments import openlin_lowsun as ls
    from experiments import openlin_probe
    monkeypatch.setattr(openlin_probe, "_read_on_grid",
                        lambda href, grid, rs: np.ones((3, 3), "float32"))
    items, grid = _fake_item_and_grid()
    msgs = []
    d1, d2 = tmp_path / "a", tmp_path / "b"
    shared_before = set(ls.CACHE.glob("S2X_TEST*")) if ls.CACHE.exists() else set()
    ls.fetch(items, grid, msgs.append, cache_dir=d1)
    assert len(list(d1.glob("S2X_TEST_*_20m.npz"))) == 1
    shared_after = set(ls.CACHE.glob("S2X_TEST*")) if ls.CACHE.exists() else set()
    assert shared_after == shared_before            # в общий CACHE не писали
    assert not any("из кэша" in m for m in msgs)
    msgs.clear()
    ls.fetch(items, grid, msgs.append, cache_dir=d2)  # другая папка: не «из кэша»
    assert not any("из кэша" in m for m in msgs)
    assert len(list(d2.glob("S2X_TEST_*_20m.npz"))) == 1
    msgs.clear()
    ls.fetch(items, grid, msgs.append, cache_dir=d1)  # та же папка: из кэша
    assert any("из кэша" in m for m in msgs)


def test_callers_use_scene_cache():
    import inspect
    from experiments import openlin_winter as w
    assert "scene_cache" in inspect.getsource(w.main)
    assert "cache_dir=" in inspect.getsource(w.main)
    assert "scene_cache(site)" in inspect.getsource(v3.build_halves)


# ---------------- UTS1 ----------------

def test_uts1_chain_and_alpha_and_reachability():
    assert v3.UTS1_N_TESTS_PRIOR == v3.UTS0_N_TESTS_PRIOR + v3.UTS0_N_TESTS == 124
    assert v3.UTS1_N_TESTS == 6
    assert v3.UTS1_ALPHA == pytest.approx(0.05 / 130)
    assert v3.UTS1_SHIFT_N == 6000
    assert 1.0 / (v3.UTS1_SHIFT_N + 1) <= v3.UTS1_ALPHA / 2


def test_tune1_key_only_for_utsjoki(monkeypatch):
    for site in ("lapland", "anabar"):
        with pytest.raises(SystemExit, match="повтор базового"):
            _run_v3(monkeypatch, ["--site", site, "--настройка1"])
    with pytest.raises(RuntimeError, match="дошли"):
        _run_v3(monkeypatch, ["--site", "utsjoki", "--настройка1"])


def test_tune1_incompatible_with_tune0_and_xfer(monkeypatch):
    for other in ("--настройка0", "--перенос"):
        with pytest.raises(SystemExit):
            _run_v3(monkeypatch, ["--site", "utsjoki", "--настройка1", other])
