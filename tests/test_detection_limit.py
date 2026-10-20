"""Тесты src/detection_limit.py на синтетике (без сети, STAC и датасета)."""
import numpy as np
import pytest

from src import config, detection_limit as dl, fracture_zones as fz

RES = 100.0


def _noise(shape=(240, 240), seed=0, scale=1.0):
    """Изотропный шум — фон, в котором вставки заведомо нет."""
    return np.random.default_rng(seed).normal(scale=scale, size=shape)


# --- Единица амплитуды ---

def test_robust_sigma_recovers_normal_scale():
    """MAD*1.4826 на нормальном шуме возвращает его сигму."""
    assert dl.robust_sigma(_noise(scale=3.0, seed=1)) == pytest.approx(3.0, rel=0.05)


def test_robust_sigma_ignores_outliers_unlike_std():
    """Устойчивость — то, ради чего сигма именно MAD, а не СКО."""
    img = _noise(seed=2)
    img[:5, :5] = 1e4                      # остаточное облако / блик радара
    assert dl.robust_sigma(img) == pytest.approx(1.0, rel=0.1)
    assert np.std(img) > 10.0              # СКО тем же выбросом снесено


def test_robust_sigma_all_nan_is_nan_not_crash():
    assert np.isnan(dl.robust_sigma(np.full((10, 10), np.nan)))


# --- Система углов ---

@pytest.mark.parametrize("geo, axis", [(0.0, 90.0), (90.0, 0.0), (45.0, 135.0),
                                       (135.0, 45.0)])
def test_axis_from_geographic_known_directions(geo, axis):
    """Север даёт вертикаль в растре (ось row смотрит на юг), восток — горизонталь."""
    assert dl.axis_from_geographic(geo) == pytest.approx(axis, abs=1e-9)


@pytest.mark.parametrize("geo", [0.0, 17.0, 45.0, 90.0, 145.0, 179.0])
def test_geographic_axis_roundtrip(geo):
    assert dl.geographic_from_axis(dl.axis_from_geographic(geo)) == pytest.approx(geo)


@pytest.mark.parametrize("geo", [0.0, 17.0, 45.0, 90.0, 145.0, 179.0, 180.0])
def test_axis_always_inside_half_open_interval(geo):
    """Ровно 180.0 — не значение оси: розы строят бины на [0, 180).

    Регрессия на численный заворот: cos(90°) в double равен 6.1e-17, arctan2
    возвращает угол чуть меньше нуля, и modulo даёт ровно 180.0.
    """
    a = dl.axis_from_geographic(geo)
    assert 0.0 <= a < 180.0
    assert 0.0 <= dl.geographic_from_axis(a) < 180.0


@pytest.mark.parametrize("a, b, want", [(10.0, 20.0, 10.0), (10.0, 190.0, 0.0),
                                        (5.0, 175.0, 10.0), (0.0, 90.0, 90.0)])
def test_axis_difference_is_modulo_180_and_folded(a, b, want):
    """У оси нет начала и конца: 10° и 190° — одно направление."""
    assert dl.axis_difference_deg(a, b) == pytest.approx(want)


# --- Геометрия вставки ---

def test_track_coords_zero_on_the_axis():
    """Поперечная координата на самой оси равна нулю, продольная растёт вдоль неё."""
    s, t = dl.track_coords((101, 101), (50, 50), 0.0, RES)   # ось вдоль col
    assert t[50, :].max() == pytest.approx(0.0, abs=1e-9)
    assert s[50, 60] == pytest.approx(10 * RES)


def test_fault_masks_line_inside_background_outside():
    line, back = dl.fault_masks((240, 240), (120, 120), 0.0, length_m=3000.0,
                                width_m=200.0, res_m=RES)
    assert line[120, 120] and not back[120, 120]
    # За полудлиной вдоль оси следа уже нет: 3000/2 = 1500 м = 15 пикселей.
    assert not line[120, 120 + 20]
    # Фон начинается не ближе DL_BACKGROUND_PAD_M поперёк оси.
    pad_px = int(config.DL_BACKGROUND_PAD_M / RES)
    assert not back[120 + pad_px - 1, 120]
    assert back[120 + pad_px + 2, 120]


def test_fault_masks_background_excludes_beyond_segment_ends():
    """Отступ фона считается от ОТРЕЗКА, а не от его центра.

    Иначе точка прямо за концом вставки, куда гауссиана окна уносит её отклик,
    попала бы в фон и занизила бы контраст против самого себя.
    """
    _, back = dl.fault_masks((300, 300), (150, 150), 0.0, length_m=6000.0,
                             width_m=200.0, res_m=RES)
    end_px = int(6000.0 / 2 / RES)                       # конец отрезка
    assert not back[150, 150 + end_px + 5]               # сразу за концом — не фон
    assert back[150, 150 + end_px + int(config.DL_BACKGROUND_PAD_M / RES) + 5]


def test_insert_zero_amplitude_is_identity():
    """Нулевая амплитуда обязана быть точным no-op: на ней стоит вся нуль-модель."""
    img = _noise()
    out = dl.insert_fault(img, (120, 120), 45.0, length_m=3000.0, width_m=200.0,
                          amplitude=0.0, sigma=1.0, res_m=RES)
    np.testing.assert_array_equal(out, img)


def test_insert_ridge_peaks_on_the_axis():
    flat = np.zeros((240, 240))
    out = dl.insert_fault(flat, (120, 120), 0.0, length_m=3000.0, width_m=400.0,
                          amplitude=2.0, sigma=1.0, res_m=RES, profile="ridge")
    assert out[120, 120] == pytest.approx(2.0, rel=1e-3)      # пик = amplitude*sigma
    assert out[122, 120] < out[120, 120]                      # спад поперёк оси
    assert out[120, 220] == pytest.approx(0.0, abs=1e-6)      # далеко вдоль — ноль


def test_insert_scarp_is_a_step_with_requested_total_jump():
    flat = np.zeros((240, 240))
    out = dl.insert_fault(flat, (120, 120), 0.0, length_m=6000.0, width_m=200.0,
                          amplitude=3.0, sigma=2.0, res_m=RES, profile="scarp")
    top, bottom = out[110, 120], out[130, 120]
    assert top < 0 < bottom                                   # борта разного знака
    assert bottom - top == pytest.approx(3.0 * 2.0, rel=0.02)  # полный перепад
    assert out[120, 120] == pytest.approx(0.0, abs=1e-9)      # на самой оси — ноль


def test_insert_unknown_profile_raises():
    with pytest.raises(ValueError, match="профиль"):
        dl.insert_fault(np.zeros((50, 50)), (25, 25), 0.0, length_m=1000.0,
                        width_m=200.0, amplitude=1.0, sigma=1.0, res_m=RES,
                        profile="ступенька")


# --- Вырезка и выбор площадок ---

def test_cut_window_keeps_center_and_is_smaller_than_sheet():
    img = _noise(shape=(600, 600))
    win, ctr = dl.cut_window(img, (300, 300), 3000.0, RES)
    half_px = int(np.ceil((1500.0 + config.DL_WINDOW_PAD_M) / RES))
    assert win.shape == (2 * half_px + 1, 2 * half_px + 1)
    assert win.size < img.size
    np.testing.assert_allclose(win[int(ctr[0]), int(ctr[1])], img[300, 300])


def test_cut_window_clipped_at_border_still_reports_right_center():
    img = _noise(shape=(600, 600))
    win, ctr = dl.cut_window(img, (5, 5), 1000.0, RES)
    np.testing.assert_allclose(win[int(ctr[0]), int(ctr[1])], img[5, 5])


def test_sample_centers_all_fit_and_are_finite():
    img = _noise(shape=(500, 500))
    rng = np.random.default_rng(0)
    centers = dl.sample_centers(img, 3000.0, RES, 10, rng)
    assert len(centers) == 10
    half_px = int(np.ceil((1500.0 + config.DL_WINDOW_PAD_M) / RES))
    for r, c in centers:
        assert half_px <= r < 500 - half_px and half_px <= c < 500 - half_px


def test_sample_centers_rejects_mostly_empty_windows():
    """Врезка в угол с NaN мерила бы покрытие съёмкой, а не обнаружимость."""
    img = _noise(shape=(500, 500))
    img[:, 250:] = np.nan
    rng = np.random.default_rng(0)
    centers = dl.sample_centers(img, 3000.0, RES, 15, rng, min_finite=0.9)
    assert centers, "не выбрано ни одной площадки"
    assert all(c < 250 for _, c in centers)


def test_sample_centers_raises_when_window_larger_than_raster():
    with pytest.raises(ValueError, match="меньше окна"):
        dl.sample_centers(_noise(shape=(40, 40)), 3000.0, RES, 1,
                          np.random.default_rng(0))


# --- Детектор непрерывной ветки ---

def test_tensor_response_positive_for_inserted_ridge_and_zero_without_it():
    """Прибор обязан отвечать на вставку и молчать без неё — на одном и том же фоне."""
    img = _noise(shape=(300, 300), seed=3)
    kw = dict(length_m=6000.0, width_m=400.0, res_m=RES, scales_m=(1000.0,))
    hit = dl.insert_fault(img, (150, 150), 30.0, amplitude=4.0, sigma=1.0,
                          length_m=6000.0, width_m=400.0, res_m=RES,
                          profile="ridge")
    assert dl.tensor_response(hit, (150, 150), 30.0, **kw)["coh_1km"] > 0.1
    assert dl.tensor_response(img, (150, 150), 30.0, **kw)["coh_1km"] < 0.05


def test_tensor_response_grows_with_amplitude():
    img = _noise(shape=(300, 300), seed=4)
    kw = dict(length_m=6000.0, width_m=400.0, res_m=RES)
    got = []
    for amp in (0.5, 1.0, 4.0):
        hit = dl.insert_fault(img, (150, 150), 0.0, amplitude=amp, sigma=1.0,
                              profile="ridge", **kw)
        got.append(dl.tensor_response(hit, (150, 150), 0.0, scales_m=(1000.0,),
                                      **kw)["coh_1km"])
    assert got[0] < got[1] < got[2]


def test_tensor_response_has_every_scale():
    img = _noise(shape=(300, 300), seed=5)
    keys = set(dl.tensor_response(img, (150, 150), 0.0, length_m=3000.0,
                                  width_m=200.0, res_m=RES))
    assert keys == {f"coh_{fz._scale_tag(s)}" for s in config.FZ_SCALES_M}


@pytest.mark.parametrize("axis_deg", [0.0, 45.0, 90.0, 135.0])
def test_inserted_axis_is_what_the_tensor_reports(axis_deg):
    """Договор о системе углов между этим модулем и fracture_zones.

    Самый ценный тест файла: разъедься две конвенции — врезка под 45° мерилась бы
    как структура под 135°, кривая предела обнаружения молча поехала бы, и
    заметить это по итоговым числам было бы уже нельзя.
    """
    flat = np.zeros((300, 300))
    hit = dl.insert_fault(flat, (150, 150), axis_deg, length_m=20000.0,
                          width_m=400.0, amplitude=4.0, sigma=1.0, res_m=RES,
                          profile="ridge")
    st = fz.structure_tensor(hit, 1000.0, res_m=RES)
    line, _ = dl.fault_masks(flat.shape, (150, 150), axis_deg, 20000.0, 400.0, RES)
    got = np.degrees(np.arctan2(np.nanmean(st["dir_sin"][line]),
                                np.nanmean(st["dir_cos"][line]))) / 2.0 % 180.0
    tol = 180.0 / config.LIN_SAT_ROSE_BINS      # один лепесток розы азимутов
    assert dl.axis_difference_deg(got, axis_deg) < tol


# --- Детектор пороговой ветки ---

def test_hough_finds_strong_scarp_and_reports_segment_count():
    pytest.importorskip("skimage")
    img = _noise(shape=(300, 300), seed=6, scale=0.2)
    hit = dl.insert_fault(img, (150, 150), 0.0, length_m=12000.0, width_m=200.0,
                          amplitude=8.0, sigma=1.0, res_m=RES, profile="scarp")
    res = dl.hough_response(hit, (150, 150), 0.0, length_m=12000.0, res_m=RES)
    assert res["hough_n_match"] >= 1
    assert res["hough_n_segments"] >= res["hough_n_match"]


def test_hough_rejects_segment_of_wrong_azimuth():
    """Совпадение по положению без совпадения по азимуту попаданием не считается."""
    pytest.importorskip("skimage")
    img = _noise(shape=(300, 300), seed=7, scale=0.2)
    hit = dl.insert_fault(img, (150, 150), 0.0, length_m=12000.0, width_m=200.0,
                          amplitude=8.0, sigma=1.0, res_m=RES, profile="scarp")
    # Та же вставка, но спрашиваем про перпендикулярную ось.
    res = dl.hough_response(hit, (150, 150), 90.0, length_m=12000.0, res_m=RES)
    assert res["hough_n_match"] == 0


def test_hough_rejects_collinear_segment_far_along_the_axis():
    """Отрезок ТОЙ ЖЕ оси, но за километры вдоль неё, — не попадание.

    Условие на продольное смещение (``|s| <= half + tol``) отдельно от
    поперечного: без него засчитывалась бы любая линия, лежащая на бесконечном
    продолжении вставки, хоть на другом краю листа.
    """
    pytest.importorskip("skimage")
    img = _noise(shape=(400, 400), seed=11, scale=0.2)
    # Уступ длиной 6 км стоит у столбца 200; спрашиваем про ту же ось сначала
    # на его месте, потом в 10 км вдоль неё, где уступа уже нет.
    hit = dl.insert_fault(img, (200, 200), 0.0, length_m=6000.0, width_m=200.0,
                          amplitude=8.0, sigma=1.0, res_m=RES, profile="scarp")
    near = dl.hough_response(hit, (200, 200), 0.0, length_m=6000.0, res_m=RES)
    far = dl.hough_response(hit, (200, 300), 0.0, length_m=6000.0, res_m=RES)
    assert near["hough_n_match"] >= 1
    assert far["hough_n_match"] == 0


# --- Порог по нуль-модели и предел ---

def test_detection_threshold_is_null_quantile():
    null = np.arange(100, dtype=float)
    assert dl.detection_threshold(null, q=0.95) == pytest.approx(np.quantile(null, 0.95))


def test_detection_threshold_all_nan_is_nan():
    assert np.isnan(dl.detection_threshold(np.full(10, np.nan)))


def test_detection_rate_counts_strict_excess():
    assert dl.detection_rate(np.array([1.0, 2.0, 3.0, 4.0]), 2.0) == pytest.approx(0.5)


def test_detection_rate_with_nan_threshold_is_nan():
    assert np.isnan(dl.detection_rate(np.array([1.0, 2.0]), float("nan")))


def test_limit_interpolates_between_sweep_nodes():
    """Сетка амплитуд грубая: без интерполяции предел округлялся бы вдвое."""
    got = dl.limit_from_curve(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.25, 0.75]))
    assert got == pytest.approx(1.0 + (0.5 - 0.25) / (0.75 - 0.25))


def test_limit_is_inf_when_target_rate_never_reached():
    got = dl.limit_from_curve(np.array([0.0, 1.0, 4.0]), np.array([0.02, 0.03, 0.1]))
    assert np.isinf(got)


def test_limit_is_first_node_when_already_detected_there():
    got = dl.limit_from_curve(np.array([0.5, 1.0]), np.array([0.9, 1.0]))
    assert got == pytest.approx(0.5)


def test_limit_ignores_nan_rates():
    got = dl.limit_from_curve(np.array([0.0, 1.0, 2.0]),
                              np.array([0.0, np.nan, 1.0]))
    assert np.isfinite(got) and 0.0 < got <= 2.0


# --- Калибровка по настоящим разломам ---

def test_corridor_contrast_recovers_known_trough_depth():
    """Контраст в долях sigma — та же ось, что у амплитуды свипа."""
    rng = np.random.default_rng(8)
    dist = np.concatenate([np.zeros(500), np.full(500, 4000.0)])
    values = rng.normal(size=1000)
    values[:500] -= 1.5                    # коридор темнее окружения на 1.5 sigma
    got = dl.corridor_contrast(values, dist, sigma=1.0)
    assert got["contrast_median"] == pytest.approx(1.5, abs=0.15)
    assert got["n_inside"] == 500 and got["n_ring"] == 500


def test_corridor_contrast_spread_catches_scarp_that_median_misses():
    """У уступа медиана внутри коридора не сдвигается, а разброс растёт."""
    rng = np.random.default_rng(9)
    dist = np.concatenate([np.zeros(600), np.full(600, 4000.0)])
    values = rng.normal(size=1200)
    values[:300] += 2.0                    # один борт уступа
    values[300:600] -= 2.0                 # другой борт
    got = dl.corridor_contrast(values, dist, sigma=1.0)
    assert got["contrast_median"] < 0.5
    assert got["contrast_spread"] > 1.5


def test_corridor_contrast_too_few_cells_is_nan_not_crash():
    got = dl.corridor_contrast(np.array([1.0, 2.0]), np.array([0.0, 0.0]), sigma=1.0)
    assert np.isnan(got["contrast_median"]) and got["n_ring"] == 0


# --- Сборка: одна врезка проходит оба детектора ---

def test_probe_once_returns_both_detectors():
    pytest.importorskip("skimage")
    img = _noise(shape=(500, 500), seed=10)
    row = dl.probe_once(img, (250, 250), 45.0, amplitude=2.0, length_m=3000.0,
                        width_m=200.0, res_m=RES, sigma=dl.robust_sigma(img),
                        scales_m=(1000.0,))
    assert set(row) == {"coh_1km", "hough_n_match", "hough_n_segments"}
    assert np.isfinite(row["coh_1km"])
