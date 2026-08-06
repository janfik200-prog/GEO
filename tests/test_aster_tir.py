"""Юниты ASTER TIR — синтетика, без сети.

Проверяется арифметика, которая превращает DN в индекс окварцевания. Порядок
проверок повторяет порядок ошибок, которые здесь реально возможны: перепутанный
коэффициент перевода, потерянная нормировка на температуру, переставленные
каналы в формуле индекса. Каждая из них не роняет прогон, а тихо выдаёт карту,
на которой видно «что-то» — и разбираться пришлось бы уже по результату.

Доступность LP DAAC и авторизация Earthdata здесь не проверяются: это
ответственность прогона ``experiments/fetch_sat.py astir``.
"""
import numpy as np
import pytest

from src import aster_tir, config, sat_sources


def _blackbody(t_k: float) -> dict[str, np.ndarray]:
    """Радианс абсолютно чёрного тела при температуре ``t_k`` во всех каналах."""
    out = {}
    for b, lam in config.ASTIR_LAMBDA_UM.items():
        v = config.ASTIR_C1 / (np.pi * lam ** 5
                               * (np.exp(config.ASTIR_C2 / (lam * t_k)) - 1.0))
        out[b] = np.full((2, 2), v, dtype="float64")
    return out


# ------------------------------------------------------------------ радианс
def test_radiance_uses_band_coefficients():
    """L = (DN - 1) * UCC, и коэффициент у каждого канала свой."""
    dn = {b: np.array([[101.0]]) for b in aster_tir.BANDS}
    rad = aster_tir.radiance(dn)
    for b in aster_tir.BANDS:
        assert rad[b][0, 0] == pytest.approx(100.0 * config.ASTIR_UCC[b])
    assert rad["b10"][0, 0] != pytest.approx(rad["b14"][0, 0])


def test_radiance_blanks_background_zeros():
    """Ноль в L1T — фон повёрнутой сцены, а не холодная поверхность."""
    dn = {b: np.array([[0.0, 1.0, 50.0]]) for b in aster_tir.BANDS}
    rad = aster_tir.radiance(dn)
    assert np.isnan(rad["b10"][0, 0]) and np.isnan(rad["b10"][0, 1])
    assert np.isfinite(rad["b10"][0, 2])


# ------------------------------------------------------------------ Планк
@pytest.mark.parametrize("t_k", [260.0, 300.0, 330.0])
def test_brightness_temp_inverts_planck(t_k):
    """Яркостная температура обязана вернуть ту, из которой считался радианс."""
    got = aster_tir.brightness_temp(_blackbody(t_k)["b13"])
    assert got[0, 0] == pytest.approx(t_k, rel=1e-3)


def test_normalize_is_identity_at_reference_temperature():
    """Поверхность уже при 300 К — нормировка ничего не меняет."""
    rad = _blackbody(config.ASTIR_NORM_T_K)
    nl = aster_tir.normalize(rad)
    for b in aster_tir.BANDS:
        assert nl[b][0, 0] == pytest.approx(rad[b][0, 0], rel=1e-3)


def test_normalize_removes_temperature_difference():
    """Главное свойство: два одинаковых материала разной температуры сходятся.

    Без нормировки карта кварцевого индекса — это карта прогрева склонов.
    Тёплый и холодный участки одного и того же вещества обязаны дать один
    нормированный радианс.
    """
    warm = aster_tir.normalize(_blackbody(320.0))
    cold = aster_tir.normalize(_blackbody(275.0))
    for b in aster_tir.BANDS:
        assert warm[b][0, 0] == pytest.approx(cold[b][0, 0], rel=1e-2)


def test_indices_survive_temperature_change():
    """Тот же вывод в терминах индексов: разброс по температуре не создаёт кварц."""
    i_warm = aster_tir.indices(aster_tir.normalize(_blackbody(320.0)))
    i_cold = aster_tir.indices(aster_tir.normalize(_blackbody(275.0)))
    for k in ("qi", "ci", "mi"):
        assert i_warm[k][0, 0] == pytest.approx(i_cold[k][0, 0], rel=1e-2)


# ------------------------------------------------------------------ индексы
def test_indices_follow_published_formulas():
    """Формулы Ниномии как опубликованы: qi = 11^2/(10*12), ci = 13/14,
    mi = 12*14^3/13^4. Степени не сокращаются — каналы разные."""
    nl = {b: np.array([[float(i + 2)]]) for i, b in enumerate(aster_tir.BANDS)}
    idx = aster_tir.indices(nl)                    # b10=2, b11=3, b12=4, b13=5, b14=6
    assert idx["qi"][0, 0] == pytest.approx(9.0 / 8.0)
    assert idx["ci"][0, 0] == pytest.approx(5.0 / 6.0)
    assert idx["mi"][0, 0] == pytest.approx(4.0 * 216.0 / 625.0)


def test_quartz_index_rises_on_reststrahlen_shape():
    """Спектральная суть: у кварца эмиссия в канале 11 выше, чем в 10 и 12.

    Проверяется знак реакции, а не абсолютное значение: индекс обязан отличать
    такую форму спектра от ровной.
    """
    flat = {b: np.array([[1.0]]) for b in aster_tir.BANDS}
    quartz = dict(flat)
    quartz["b10"] = np.array([[0.9]])
    quartz["b12"] = np.array([[0.9]])
    assert aster_tir.indices(quartz)["qi"][0, 0] > aster_tir.indices(flat)["qi"][0, 0]


def test_indices_keep_nan_instead_of_infinity():
    """Деление на ноль даёт NaN: бесконечность расползлась бы по медиане."""
    nl = {b: np.array([[1.0]]) for b in aster_tir.BANDS}
    nl["b12"] = np.array([[0.0]])
    assert np.isnan(aster_tir.indices(nl)["qi"][0, 0])


# ------------------------------------------------------------------ сборка
def test_scene_layers_names_are_stable():
    """Состав слоёв сцены зафиксирован: по ним потом называются признаки."""
    dn = {b: np.full((2, 2), 100.0) for b in aster_tir.BANDS}
    lay = aster_tir.scene_layers(dn)
    assert set(lay) == {"nb10", "nb11", "nb12", "nb13", "nb14", "t13",
                        "qi", "ci", "mi"}
    assert all(v.shape == (2, 2) for v in lay.values())


def test_band_urls_picks_only_tir_geotiffs():
    """Из десятков ссылок гранулы берутся ровно пять тепловых каналов."""
    base = "https://data.lpdaac.earthdatacloud.nasa.gov/x/G"
    entry = {"links": [{"href": f"{base}_TIR_B{n}.tif"} for n in range(10, 15)]
             + [{"href": f"{base}_VNIR_B01.tif"}, {"href": f"{base}_TIR.tif"},
                {"href": f"s3://bucket/G_TIR_B10.tif"}]}
    urls = aster_tir._band_urls(entry)
    assert set(urls) == set(aster_tir.BANDS)
    assert urls["b10"].startswith("https://") and urls["b10"].endswith("_TIR_B10.tif")


def test_polygon_center_from_cmr_entry():
    entry = {"polygons": [["71.0 106.0 70.0 106.0 70.0 108.0 71.0 108.0 71.0 106.0"]]}
    lat, lon = aster_tir._polygon_center(entry)
    assert lat == pytest.approx(70.6, abs=0.3) and lon == pytest.approx(106.8, abs=0.3)


def test_sensor_registry_contains_tir():
    """Реестр сенсоров не должен разъезжаться с модулями."""
    assert "astir" in sat_sources.SENSORS
    fn, title = sat_sources.SENSORS["astir"]
    assert callable(fn) and "TIR" in title
