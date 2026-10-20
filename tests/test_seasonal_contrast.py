"""Тесты сезонного контраста коридоров (src/seasonal_contrast.py).

Главный контракт — СДВИГОВАЯ ШКАЛА. Она нужна, чтобы число «0.3 sigma» можно
было прочитать, и потому обязана двигать снимок вместе с его невалидными
ячейками, сохраняя автокорреляцию. Сдвинь она ячейки пула отдельно от значений —
шкала считалась бы на другом наборе ячеек, чем сам замер, и сравнение потеряло
бы смысл, не подав никакого признака поломки.

Второй контракт — ослабленная маска Landsat. Штатная бракует снег, и если сюда
случайно вернётся она, весеннее окно вымаскируется целиком, а прогон покажет не
«контраста нет», а «данных нет».
"""
from pathlib import Path

import numpy as np
import pytest

from src import config, detection_limit as dl, sat_sources, seasonal_contrast as sea


class _Meta:
    """Минимальная сетка: seasonal_contrast пользуется только prf/pic."""

    def __init__(self, prf, pic):
        self.prf, self.pic = prf, pic


# --------------------------------------------------------------------------
# Окна и реестр
# --------------------------------------------------------------------------

def test_window_months_rejects_unknown_name():
    with pytest.raises(ValueError, match="неизвестное окно"):
        sea.window_months("бабье лето")


def test_every_preregistered_candidate_names_a_real_window_and_field():
    """Опечатка в пре-регистрированном списке обязана ловиться тестом, а не прогоном."""
    for kind, field, window in config.SEA_CANDIDATES:
        assert window in config.SEA_WINDOWS, (kind, field, window)
        assert kind in sea.SENSOR_FIELDS, kind
        assert field in sea.SENSOR_FIELDS[kind], (kind, field)


def test_every_sensor_field_has_its_own_summer_baseline():
    """У каждой пары «сенсор+поле» обязана быть своя летняя строка.

    Иначе сезонное число сравнивалось бы с летним из другого прогона, снятым
    другой маской и другой квотой, и часть разницы, созданная обработкой,
    записалась бы на счёт сезона.
    """
    summer = {(k, f) for k, f, w in config.SEA_CANDIDATES if w == "лето"}
    for kind, field, window in config.SEA_CANDIDATES:
        assert (kind, field) in summer, f"{kind}/{field} без летней планки"


def test_summer_baseline_is_first_in_the_list():
    """Планка обязана считаться тем же кодом и стоять первой, а не браться из шага 0.

    Летний контраст шага 0 измерен на окне S1_MONTHS = (6,7,8,9); здесь окно
    (7,8). Сравнивать сезонные числа с чужим окном значило бы приписывать сезону
    разницу, часть которой создана составом месяцев.
    """
    assert config.SEA_CANDIDATES[0][2] == "лето"


def test_windows_do_not_overlap_by_month():
    """Пересекись окна по месяцу — «зима» и «наледи» мерили бы частично одно и то же."""
    seen: set[int] = set()
    for months in config.SEA_WINDOWS.values():
        assert not (seen & set(months)), months
        seen |= set(months)


def test_primary_window_is_the_one_the_literature_names():
    assert config.SEA_PRIMARY == "наледи"
    assert config.SEA_WINDOWS[config.SEA_PRIMARY] == (4, 5)


def test_unsupported_sensor_raises_instead_of_silently_returning_nothing():
    with pytest.raises(ValueError, match="не поддержан"):
        sea.window_items("s2", _Meta(4, 4), "наледи")


def test_unknown_field_raises_for_each_sensor():
    with pytest.raises(ValueError, match="vv_db"):
        sea.window_field("s1", "ndwi", _Meta(4, 4), [])
    with pytest.raises(ValueError, match="red"):
        sea.window_field("l8", "vv_db", _Meta(4, 4), [])


# --------------------------------------------------------------------------
# Терпимость к битым сценам
# --------------------------------------------------------------------------

def test_one_broken_scene_does_not_kill_the_window():
    """Регрессия на прогон 26.08.2026: одна битая сцена уронила окно из 76.

    Сцена S1B от 08.03.2021 отдавала RasterioIOError после всех повторов чтения
    (битой была она сама, а не соединение) и обнулила 45 минут скачивания.
    Композит — медиана, ей одна сцена из 76 не нужна.
    """
    def reader(it):
        if it == 3:
            raise OSError("not recognized as being in a supported file format")
        return {"vv": it}

    scenes, failed = sea._read_scenes(list(range(10)), reader)
    assert failed == 1
    assert len(scenes) == 9
    assert {"vv": 3} not in scenes


def test_all_scenes_broken_is_reported_as_zero_not_as_success():
    scenes, failed = sea._read_scenes([1, 2], lambda it: (_ for _ in ()).throw(OSError()))
    assert scenes == [] and failed == 2


def test_parallel_reading_keeps_order_and_still_drops_broken_scenes():
    """Потоки ускоряют ожидание сети, но не должны менять состав и порядок композита."""
    def reader(it):
        if it % 7 == 3:
            raise OSError("битая сцена")
        return {"vv": it}

    items = list(range(40))
    fast, failed_fast = sea._read_scenes(items, reader, workers=8)
    slow, failed_slow = sea._read_scenes(items, reader, workers=1)
    assert fast == slow                                # тот же порядок, не «кто успел»
    assert failed_fast == failed_slow == len([i for i in items if i % 7 == 3])


def test_scene_reader_returning_none_is_not_counted_as_broken():
    """``None`` — законное значение читателя; битую сцену метит отдельный объект."""
    scenes, failed = sea._read_scenes([1, 2, 3], lambda it: None, workers=4)
    assert failed == 0 and scenes == [None, None, None]


def test_window_is_refused_when_too_few_scenes_survive():
    """Терпимость к битой сцене не должна становиться терпимостью к пустому окну."""
    with pytest.raises(RuntimeError, match="окно не считается"):
        sea._require_enough([], 20, list(range(20)))
    survivors = list(range(config.SEA_MIN_ITEMS))
    sea._require_enough(survivors, 1, list(range(config.SEA_MIN_ITEMS + 1)))


# --------------------------------------------------------------------------
# Ослабленная маска Landsat
# --------------------------------------------------------------------------

def test_seasonal_landsat_mask_keeps_snow_and_still_drops_clouds():
    """Ветка ищет снег и лёд: бит 5 браковать нельзя, биты облаков — обязательно."""
    assert 5 in config.L8_QA_REJECT_BITS               # штатная маска снег бракует
    assert 5 not in config.SEA_L8_QA_REJECT_BITS       # сезонная — нет
    for bit in (1, 3, 4):                              # облако, облако, тень
        assert bit in config.SEA_L8_QA_REJECT_BITS


def test_l8_mask_honours_the_reject_bits_argument():
    """Параметр обязан реально менять результат, а не приниматься и игнорироваться."""
    qa = np.array([[0, 1 << 5, 1 << 3]], dtype=float)

    class _Item:
        pass

    def fake_read(item, band, meta, res_m, nearest=False):
        return qa

    orig = sat_sources.stac_grid.read_on_grid
    sat_sources.stac_grid.read_on_grid = fake_read
    try:
        strict = sat_sources.l8_mask(_Item(), None, 100.0)
        relaxed = sat_sources.l8_mask(_Item(), None, 100.0,
                                      reject_bits=config.SEA_L8_QA_REJECT_BITS)
    finally:
        sat_sources.stac_grid.read_on_grid = orig
    # Пиксель со снегом (бит 5): штатная маска бракует, сезонная оставляет.
    assert not strict[0, 1] and relaxed[0, 1]
    # Пиксель с облаком (бит 3) бракуется обеими.
    assert not strict[0, 2] and not relaxed[0, 2]
    assert strict[0, 0] and relaxed[0, 0]


# --------------------------------------------------------------------------
# Ячейки и пул
# --------------------------------------------------------------------------

def test_cells_outside_the_pool_become_nan():
    """Пул зануляется ДО сдвига, иначе шкала поедет на другом наборе ячеек."""
    meta = _Meta(2, 3)
    img = np.arange(6, dtype=float).reshape(2, 3)

    class _FakeGrid:
        @staticmethod
        def to_cells(layers, meta, res_m, prefix):
            import pandas as pd
            return pd.DataFrame({prefix + "gray": layers["gray"].ravel()})

    orig = sea.stac_grid
    sea.stac_grid = _FakeGrid
    try:
        got = sea.cells_from_image(img, meta, 100.0, pool=np.array([0, 2, 4]))
    finally:
        sea.stac_grid = orig
    np.testing.assert_array_equal(np.isnan(got), [False, True, False, True, False, True])


# --------------------------------------------------------------------------
# Сдвиговая шкала
# --------------------------------------------------------------------------

def _corridor_field(prf, pic, band=2):
    """Расстояние до вертикальной полосы посередине сетки."""
    cols = np.tile(np.arange(pic, dtype=float), (prf, 1))
    return np.clip(np.abs(cols - pic // 2) - band, 0.0, None).ravel() * 100.0


def test_shift_reference_returns_one_value_per_shift():
    meta = _Meta(20, 20)
    rng = np.random.default_rng(0)
    values = rng.normal(size=meta.prf * meta.pic)
    dist = _corridor_field(meta.prf, meta.pic)
    got = sea.shift_reference(values, dist, 1.0, meta, n_shifts=25,
                              ring_m=(300.0, 900.0))
    assert got.shape == (25,)
    assert np.isfinite(got).sum() >= 20


def test_shift_reference_is_reproducible_for_a_fixed_seed():
    meta = _Meta(16, 16)
    values = np.random.default_rng(1).normal(size=256)
    dist = _corridor_field(16, 16)
    kw = dict(n_shifts=12, seed=7, ring_m=(300.0, 900.0))
    a = sea.shift_reference(values, dist, 1.0, meta, **kw)
    b = sea.shift_reference(values, dist, 1.0, meta, **kw)
    np.testing.assert_allclose(a, b)


def test_shift_reference_brackets_a_planted_corridor_signal():
    """Настоящий контраст должен выходить за шкалу, случайный — лежать внутри.

    Это проверка самой шкалы: если подсадить в коридор реальный перепад, замер
    обязан оказаться выше почти всех сдвигов, а на чистом шуме — внутри них.
    Иначе шкала не мерила бы ничего.
    """
    meta = _Meta(40, 40)
    dist = _corridor_field(40, 40, band=3)
    rng = np.random.default_rng(3)
    noise = rng.normal(size=1600)

    plain = dl.corridor_contrast(noise, dist, 1.0, ring_m=(500.0, 1500.0))
    ref_plain = sea.shift_reference(noise, dist, 1.0, meta, n_shifts=60,
                                    ring_m=(500.0, 1500.0))
    assert plain["contrast_median"] <= np.nanquantile(ref_plain, 0.95)

    planted = noise + np.where(dist <= 0.0, 3.0, 0.0)
    signal = dl.corridor_contrast(planted, dist, 1.0, ring_m=(500.0, 1500.0))
    ref_signal = sea.shift_reference(planted, dist, 1.0, meta, n_shifts=60,
                                     ring_m=(500.0, 1500.0))
    assert signal["contrast_median"] > np.nanquantile(ref_signal, 0.95)


def test_shift_reference_moves_nan_together_with_the_image():
    """Невалидные ячейки едут вместе с картой — иначе сдвиг менял бы набор ячеек."""
    meta = _Meta(10, 10)
    values = np.full(100, 1.0)
    values[:10] = np.nan                      # первая строка невалидна
    dist = _corridor_field(10, 10, band=1)
    got = sea.shift_reference(values, dist, 1.0, meta, n_shifts=15,
                              ring_m=(300.0, 900.0))
    # Значения постоянные всюду, кроме NaN, поэтому контраст обязан быть 0 или NaN,
    # но никогда не бесконечным: NaN не должен подменяться нулём при сдвиге.
    assert not np.isinf(got).any()


# --------------------------------------------------------------------------
# Дисциплина ветки
# --------------------------------------------------------------------------

def test_branch_declares_no_significance_machinery():
    """Ветка описательная: ни alpha, ни собственного слагаемого ledger у неё нет."""
    assert not hasattr(config, "SEA_N_TESTS")
    assert not hasattr(config, "SEA_ALPHA")
    assert not hasattr(config, "SEA_N_TESTS_PRIOR")


def test_scene_quota_is_raised_only_for_the_sensor_that_needs_coverage():
    """Квота поднята у радара и НЕ поднята у Landsat — по разнице покрытия кадра.

    Одна сцена S1 RTC накрывает 10.9% листа (измерено в блоке SM_*), поэтому там
    нужен запас сцен. Кадр Landsat WRS-2 накрывает лист целиком, и поднятая квота
    там означала бы только часы лишнего скачивания: при 45 весеннее окно набирало
    272 сцены по 7 маршрутам.
    """
    assert config.SEA_MAX_PER_ORBIT > config.S1_MAX_SCENES_PER_ORBIT
    assert config.SEA_MIN_ITEMS > 4      # MIN_ITEMS задачи №8
    src = Path(sea.__file__).read_text(encoding="utf-8")
    l8_call = src.split('if kind == "l8":')[1].split("raise ValueError")[0]
    assert "config.L8_MAX_SCENES_PER_PATH" in l8_call
    assert "SEA_MAX_PER_ORBIT" not in l8_call


def test_shift_scale_is_large_enough_to_place_a_95th_percentile():
    assert config.SEA_N_SHIFTS >= 100


def test_verdict_is_refused_when_the_primary_window_is_missing(capsys):
    """Регрессия на прогон 26.08.2026: сезонные окна упали, вердикт всё равно вышел.

    Три окна пропали на таймаутах каталога, в таблице осталась одна летняя
    планка — и скрипт напечатал «смена сезона разломы видимыми не делает».
    Отрицательный вывод из отсутствия данных хуже отсутствия вывода.
    """
    import pandas as pd
    from experiments.seasonal_contrast_run import _report

    only_summer = pd.DataFrame([{
        "сенсор": "s1", "поле": "vv_db", "окно": "лето", "система": "СЗ (tect1)",
        "сцен": 63, "покрытие": 1.0, "sigma": 1.38,
        "контраст_медиан": 0.243, "контраст_разброса": 0.0,
        "шкала_медиана": 0.115, "шкала_q95": 0.313, "шкала_max": 0.4,
        "ячеек_внутри": 8432, "ячеек_в_кольце": 5742,
    }])
    _report(only_summer)
    out = capsys.readouterr().out
    assert "ВЫВОД НЕ ФОРМУЛИРУЕТСЯ" in out
    assert config.SEA_PRIMARY in out
    assert "видимыми не делает" not in out


def _row(sensor, field, window, contrast):
    return {"сенсор": sensor, "поле": field, "окно": window, "система": "СЗ (tect1)",
            "сцен": 90, "покрытие": 1.0, "sigma": 1.0,
            "контраст_медиан": contrast, "контраст_разброса": 0.0,
            "шкала_медиана": 0.07, "шкала_q95": 0.20, "шкала_max": 0.27,
            "ячеек_внутри": 8432, "ячеек_в_кольце": 5742}


def test_verdict_is_refused_when_the_primary_window_has_no_primary_sensor(capsys):
    """Регрессия на прогон 28.08.2026: `--only l8` посчитал «наледи» и вынес вердикт.

    Проверка на окно её пропустила — окно-то посчиталось. Но посчиталось оптикой,
    которая бракует снег по построению и зиму с ледоставом не видит вообще.
    """
    import pandas as pd
    from experiments.seasonal_contrast_run import _report

    optics_only = pd.DataFrame([_row("l8", "red", "лето", 0.083),
                                _row("l8", "red", "наледи", 0.141)])
    _report(optics_only)
    out = capsys.readouterr().out
    assert "ВЫВОД НЕ ФОРМУЛИРУЕТСЯ" in out
    assert config.SEA_PRIMARY_SENSOR in out
    assert "видимыми не делает" not in out


def test_verdict_is_given_once_the_primary_sensor_covers_the_primary_window(capsys):
    """Обратная сторона: с радаром в основном окне вердикт обязан выноситься."""
    import pandas as pd
    from experiments.seasonal_contrast_run import _report

    full = pd.DataFrame([_row("s1", "vv_db", w, 0.2) for w in config.SEA_WINDOWS])
    _report(full)
    out = capsys.readouterr().out
    assert "ВЫВОД НЕ ФОРМУЛИРУЕТСЯ" not in out
    assert "ВЫВОД:" in out


def test_primary_sensor_is_the_one_that_sees_every_window():
    """Основной сенсор обязан быть тем, у которого есть строки во всех окнах."""
    windows = {w for _, _, w in config.SEA_CANDIDATES
               if any(s == config.SEA_PRIMARY_SENSOR
                      for s, _, ww in config.SEA_CANDIDATES if ww == w)}
    assert windows == set(config.SEA_WINDOWS)
    assert config.SEA_PRIMARY_SENSOR in sea.SENSOR_FIELDS
