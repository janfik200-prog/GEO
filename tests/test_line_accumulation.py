"""Тесты накопления вдоль линии (src/line_accumulation.py).

Главный контракт здесь — НОРМИРОВКА согласованного фильтра. Вся ценность прогона
держится на том, что ``acc_matched`` на независимом фоне распределена как
``|N(0, 1)|``: только тогда отклонение её нуль-модели от табличного квантиля
можно читать как меру связности реального композита, а не как ошибку прибора.
Поэтому нормировка проверяется прямо — прогоном фильтра по белому шуму.

Второй контракт — система координат. Хаф возвращает ``(col, row)``, а вся
геометрия ветки живёт в ``(row, col)``; перепутанные оси дали бы правдоподобные,
но повёрнутые на 90° цепочки, и по числам это неотличимо.
"""
import numpy as np
import pytest

from src import config, detection_limit as dl, line_accumulation as acc


# --------------------------------------------------------------------------
# Согласованный фильтр: нормировка и поведение
# --------------------------------------------------------------------------

def _noise(shape=(160, 160), seed=0):
    return np.random.default_rng(seed).normal(size=shape)


def test_matched_on_white_noise_is_standard_normal():
    """Нуль-модель на белом шуме обязана лечь на квантиль |N(0, 1)|.

    Это калибровка самого прибора: если на независимом фоне статистика уже не
    стандартна, то измеренная на композите «доля независимости» мерила бы
    ошибку нормировки, а не текстуру листа.
    """
    vals = []
    for seed in range(200):
        img = _noise(seed=seed)
        vals.append(acc.matched_response(img, (80, 80), 30.0, length_m=3000.0,
                                         width_m=500.0, res_m=100.0,
                                         profile="ridge")["acc_matched"])
    got = np.quantile(vals, config.DL_NULL_Q)
    ideal = 1.9600                                   # квантиль 0.95 модуля N(0, 1)
    assert 0.75 * ideal < got < 1.30 * ideal, got
    assert 0.6 < acc.independence_ratio(got) < 1.8


def test_matched_grows_with_amplitude():
    img = _noise(seed=1)
    kw = dict(length_m=3000.0, width_m=500.0, res_m=100.0)
    got = [acc.matched_response(
        dl.insert_fault(img, (80, 80), 30.0, amplitude=a, sigma=1.0,
                        profile="ridge", **kw),
        (80, 80), 30.0, profile="ridge", **kw)["acc_matched"]
        for a in (0.0, 0.5, 2.0)]
    assert got[0] < got[1] < got[2]


def test_matched_ignores_sign_of_contrast():
    """Разлом может быть и светлее, и темнее фона — детектор берёт модуль."""
    img = _noise(seed=2)
    kw = dict(length_m=3000.0, width_m=500.0, res_m=100.0)
    plus = dl.insert_fault(img, (80, 80), 30.0, amplitude=2.0, sigma=1.0,
                           profile="ridge", **kw)
    minus = dl.insert_fault(img, (80, 80), 30.0, amplitude=-2.0, sigma=1.0,
                            profile="ridge", **kw)
    a = acc.matched_response(plus, (80, 80), 30.0, profile="ridge", **kw)["acc_matched"]
    b = acc.matched_response(minus, (80, 80), 30.0, profile="ridge", **kw)["acc_matched"]
    assert a == pytest.approx(b, rel=0.15)


def test_scarp_statistic_differs_from_ridge_on_a_step():
    """У уступа маска объекта накрывает оба борта, поэтому жёлобная формула слепа.

    Обратный тест к выбору ACC_PROFILE_PRIMARY: он показывает, ЗАЧЕМ формулы
    две и почему нуль-модель размечается по профилю, а не общая.
    """
    img = np.zeros((160, 160))
    kw = dict(length_m=3000.0, width_m=500.0, res_m=100.0)
    step = dl.insert_fault(img, (80, 80), 30.0, amplitude=3.0, sigma=1.0,
                           profile="scarp", **kw)
    step += _noise(seed=3) * 0.1
    as_scarp = acc.matched_response(step, (80, 80), 30.0, profile="scarp", **kw)
    as_ridge = acc.matched_response(step, (80, 80), 30.0, profile="ridge", **kw)
    assert as_scarp["acc_matched"] > 10 * as_ridge["acc_matched"]


def test_matched_reports_nan_instead_of_faking_a_value():
    """Вырезка, в которой полос сравнения не помещается, обязана дать nan."""
    tiny = np.zeros((6, 6))
    got = acc.matched_response(tiny, (3, 3), 0.0, length_m=3000.0,
                               width_m=500.0, res_m=100.0, profile="ridge")
    assert np.isnan(got["acc_matched"])


def test_matched_denominator_is_measured_on_background_only():
    """Сильная вставка не должна раздувать собственный знаменатель.

    Считай разброс по всему окну — амплитуда 4 sigma подняла бы его сама, и
    статистика росла бы медленнее сигнала. Проверяется тем, что отклик
    пропорционален амплитуде: при знаменателе «по всему окну» пропорциональность
    ломается.
    """
    img = _noise(seed=4)
    kw = dict(length_m=3000.0, width_m=500.0, res_m=100.0)
    got = []
    for a in (1.0, 2.0, 4.0):
        hit = dl.insert_fault(img, (80, 80), 20.0, amplitude=a, sigma=1.0,
                              profile="ridge", **kw)
        got.append(acc.matched_response(hit, (80, 80), 20.0, profile="ridge",
                                        **kw)["acc_matched"])
    assert got[1] / got[0] == pytest.approx(2.0, rel=0.2)
    assert got[2] / got[0] == pytest.approx(4.0, rel=0.2)


def test_nominal_n_matches_the_mask_it_claims():
    """acc_nominal_n обязана отвечать реальной маске, иначе доля независимости врёт."""
    img = _noise(seed=5)
    got = acc.matched_response(img, (80, 80), 0.0, length_m=3000.0,
                               width_m=500.0, res_m=100.0, profile="ridge")
    # Маска объекта: 31 пиксель вдоль x 7 поперёк; полосы шире, поэтому
    # номинальное число независимых отсчётов лежит между ними и их суммой.
    assert 100.0 < got["acc_nominal_n"] < 400.0


def test_independence_ratio_is_one_when_null_matches_the_table():
    assert acc.independence_ratio(1.9600) == pytest.approx(1.0, rel=0.01)
    # Вдвое более широкая нуль-модель = вчетверо меньше эффективных отсчётов.
    assert acc.independence_ratio(2 * 1.9600) == pytest.approx(0.25, rel=0.01)
    assert np.isnan(acc.independence_ratio(0.0))


# --------------------------------------------------------------------------
# Сращивание: геометрия
# --------------------------------------------------------------------------

def _seg(r0, c0, r1, c1):
    """Отрезок в формате Хафа — (col, row), как его возвращает skimage."""
    return ((c0, r0), (c1, r1))


def test_segment_arrays_keep_row_col_convention():
    """Хаф отдаёт (col, row); модуль обязан переставить, иначе оси развернутся на 90°."""
    p0, p1, mid, length, axis = acc._segment_arrays([_seg(10, 20, 10, 30)])
    np.testing.assert_allclose(p0[0], [10, 20])
    np.testing.assert_allclose(p1[0], [10, 30])
    assert length[0] == pytest.approx(10.0)
    # Строка не меняется, столбец растёт -> ось вдоль col, то есть a = 0.
    assert axis[0] == pytest.approx(0.0)


def test_mean_axis_survives_the_wraparound_at_zero():
    """179° и 1° — почти одно направление; наивное среднее дало бы 90°."""
    got = acc._mean_axis_deg(np.array([179.0, 1.0]), np.array([1.0, 1.0]))
    assert min(got, 180.0 - got) < 1.0
    assert 0.0 <= got < 180.0


def test_collinear_fragments_with_small_gap_become_one_chain():
    segs = [_seg(50, 10, 50, 20), _seg(50, 25, 50, 35)]     # разрыв 5 px = 500 м
    chains = acc.link_segments(segs, res_m=100.0)
    assert len(chains) == 1
    assert chains[0]["span_m"] == pytest.approx(2500.0)      # от col 10 до col 35
    assert chains[0]["cover_m"] == pytest.approx(2000.0)     # сами фрагменты
    assert chains[0]["n_segments"] == 2


def test_gap_beyond_tolerance_keeps_fragments_apart():
    far = int(config.ACC_LINK_GAP_M / 100.0) + 10
    segs = [_seg(50, 10, 50, 20), _seg(50, 20 + far, 50, 30 + far)]
    assert len(acc.link_segments(segs, res_m=100.0)) == 2


def test_perpendicular_offset_beyond_tolerance_keeps_fragments_apart():
    off = int(config.ACC_LINK_OFFSET_M / 100.0) + 3
    segs = [_seg(50, 10, 50, 20), _seg(50 + off, 25, 50 + off, 35)]
    assert len(acc.link_segments(segs, res_m=100.0)) == 2


def test_different_axes_are_not_linked():
    segs = [_seg(50, 10, 50, 20), _seg(45, 21, 55, 31)]      # 0° против 45°
    assert len(acc.link_segments(segs, res_m=100.0)) == 2


def test_linking_is_transitive_so_length_accumulates():
    """Концы цепочки могут не сцепляться напрямую — в этом и смысл накопления."""
    # Шаг равен допуску на разрыв, сам фрагмент 5 px — значит разрыв между
    # соседями чуть МЕНЬШЕ допуска, а между крайними фрагментами вчетверо больше.
    step = int(config.ACC_LINK_GAP_M / 100.0)
    segs = [_seg(50, i * step, 50, i * step + 5) for i in range(5)]
    chains = acc.link_segments(segs, res_m=100.0)
    assert len(chains) == 1
    assert chains[0]["n_segments"] == 5
    # Протяжённость много больше суммарной длины самих фрагментов.
    assert chains[0]["span_m"] > 3 * chains[0]["cover_m"]


def test_empty_input_gives_no_chains():
    assert acc.link_segments([], res_m=100.0) == []


# --------------------------------------------------------------------------
# Слепой детектор целиком
# --------------------------------------------------------------------------

def test_linked_response_finds_a_strong_inserted_line():
    pytest.importorskip("skimage")
    img = _noise((220, 220), seed=6)
    kw = dict(length_m=6000.0, width_m=300.0, res_m=100.0)
    hit = dl.insert_fault(img, (110, 110), 30.0, amplitude=6.0, sigma=1.0,
                          profile="ridge", **kw)
    got = acc.linked_response(hit, (110, 110), 30.0, length_m=6000.0, res_m=100.0)
    assert got["acc_link_n_match"] >= 1
    assert got["acc_link_span"] > 2000.0


def test_linked_response_on_pure_noise_is_not_a_guaranteed_hit():
    pytest.importorskip("skimage")
    hits = [acc.linked_response(_noise((160, 160), seed=s), (80, 80), 30.0,
                                length_m=3000.0, res_m=100.0)["acc_link_span"]
            for s in range(8)]
    assert min(hits) == 0.0 or np.median(hits) < 3000.0


def test_fragments_respect_the_cap():
    pytest.importorskip("skimage")
    segs = acc.fragments(_noise((300, 300), seed=7))
    assert len(segs) <= config.ACC_MAX_FRAGMENTS


def test_probe_once_returns_old_and_new_detectors_together():
    pytest.importorskip("skimage")
    img = _noise((220, 220), seed=8)
    row = acc.probe_once(img, (110, 110), 30.0, amplitude=2.0, length_m=3000.0,
                         width_m=500.0, res_m=100.0, sigma=1.0, profile="ridge",
                         scales_m=(300.0,))
    for key in ("coh_300", "hough_n_match", "acc_matched", "acc_link_span"):
        assert key in row, key


# --------------------------------------------------------------------------
# Дисциплина констант
# --------------------------------------------------------------------------

def test_relaxation_is_exactly_twofold_as_declared():
    """Ослабление объявлено правилом «ровно вдвое», а не подобрано по результату."""
    assert 100.0 - config.ACC_EDGE_PCTL == 2 * (100.0 - config.LIN_EDGE_PCTL)
    assert 100.0 - config.ACC_EDGE_LOW_PCTL == 2 * (100.0 - config.LIN_EDGE_LOW_PCTL)
    assert config.ACC_MIN_LEN_PX == config.LIN_MIN_LEN_PX // 2
    assert config.ACC_HOUGH_THRESHOLD == config.LIN_HOUGH_THRESHOLD // 2
    assert config.ACC_MAX_GAP_PX == config.LIN_MAX_GAP_PX


def test_link_tolerances_are_taken_from_published_thresholds():
    """Ни одного нового числа: иначе за каждое пришлось бы оправдываться отдельно."""
    assert config.ACC_LINK_TOL_DEG == config.DL_MATCH_TOL_DEG
    assert config.ACC_LINK_OFFSET_M == config.FZ_SCALES_M[0]
    assert config.ACC_LINK_GAP_M == config.LIN_SAT_MATCH_M[-1]


def test_flank_starts_beyond_the_tail_of_the_widest_insertion():
    """Полоса сравнения не должна содержать хвост самой вставки.

    У жёлоба гауссиана с сигмой в полуширину; полоса начинается на
    ACC_FLANK_INNER_M от кромки, то есть на (1 + inner/полуширина) сигмы.
    При самой широкой вставке шкалы это должно оставлять меньше 2% амплитуды.
    """
    half = config.DL_WIDTHS_M[-1] / 2.0
    sigmas = (half + config.ACC_FLANK_INNER_M) / half
    assert np.exp(-0.5 * sigmas ** 2) < 0.02


def test_flank_stays_inside_the_background_zone_of_the_tensor():
    """Внешний край полосы не выходит за DL_BACKGROUND_PAD_M — одна зона «фон» на ветку."""
    outer = config.DL_WIDTHS_M[-1] / 2.0 + config.ACC_FLANK_INNER_M + config.ACC_FLANK_M
    assert outer <= config.DL_BACKGROUND_PAD_M


def test_primary_profile_matches_what_corridor_contrast_measures():
    """Основной профиль — жёлоб, потому что calibration меряет разность медиан."""
    assert config.ACC_PROFILE_PRIMARY == "ridge"
    assert config.ACC_PROFILE_PRIMARY in config.DL_PROFILES
