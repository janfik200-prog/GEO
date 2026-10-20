"""Тесты ориентации трансформант как трассировщиков кромок (src/potential_fields.py).

Ориентация здесь не косметика: протокол задачи №2 берёт ВЕРХНИЕ FZ_AREA карты,
поэтому кандидат обязан расти к кромке источника. Подай tilt-derivative сырым —
и прогон молча ответит на другой вопрос («лежит ли разлом НАД телом» вместо
«лежит ли разлом НА КРОМКЕ тела»), причём по числам это неотличимо.
"""
import numpy as np
import pandas as pd
import pytest

from src import config, potential_fields as pf


def test_orient_raw_is_identity():
    v = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_array_equal(pf.orient_edge_score(v, "raw"), v)


def test_orient_abs_drops_sign_of_signed_transform():
    """У вертикальной производной знак задаёт контраст плотности, не геометрию."""
    v = np.array([-3.0, -1.0, 1.0, 3.0])
    np.testing.assert_array_equal(pf.orient_edge_score(v, "abs"),
                                  [3.0, 1.0, 1.0, 3.0])


def test_orient_neg_abs_puts_maximum_at_zero_of_tilt():
    """У tilt кромка — это ноль, поэтому близость к кромке есть -|tilt|."""
    tilt = np.array([-1.4, -0.2, 0.0, 0.3, 1.5])       # радианы, кромка в центре
    got = pf.orient_edge_score(tilt, "neg_abs")
    assert np.argmax(got) == 2                          # максимум ровно на нуле tilt
    assert got[0] < got[1] < got[2] and got[2] > got[3] > got[4]


def test_orient_unknown_rule_raises():
    with pytest.raises(ValueError, match="ориентация"):
        pf.orient_edge_score(np.zeros(3), "как-нибудь")


def test_raw_tilt_would_rank_body_above_edge():
    """Обратный тест: почему сырой tilt нельзя подавать в протокол.

    Верхние значения СЫРОГО tilt — это центр тела (tilt -> +pi/2), а кромка
    (tilt = 0) оказывается в середине рейтинга. Протокол, берущий верхние 10%,
    на сыром tilt мерил бы попадание разломов в ТЕЛА источников.
    """
    tilt = np.array([-1.5, 0.0, 1.5])                   # вне тела, кромка, над телом
    assert np.argmax(tilt) == 2                          # сырой: верх — тело
    assert np.argmax(pf.orient_edge_score(tilt, "neg_abs")) == 1   # ориентированный: кромка


# --- Реестр кандидатов ---

def test_every_tracer_has_a_known_orientation_rule():
    for name, (col, how) in pf.EDGE_TRACERS.items():
        assert how in ("raw", "abs", "neg_abs"), f"{name}: неизвестное правило {how}"
        assert isinstance(col, str) and col


def test_tracer_registry_matches_preregistered_candidate_list():
    """Реестр и пре-регистрация обязаны совпадать по составу.

    Разъедься они — знаменатель Бонферрони (FZ_N_TESTS_PF) перестал бы отвечать
    числу реально перебираемых карт, то есть порог поехал бы молча.
    """
    assert set(pf.EDGE_TRACERS) == set(config.PF_FAULT_PRIMARY["кандидаты"])


def test_bonferroni_counter_matches_two_checks_per_candidate():
    """16 = 8 кандидатов x 2 проверки (сырая и остаток), как в screen_candidate."""
    assert config.FZ_N_TESTS_PF == 2 * len(pf.EDGE_TRACERS)


def test_prior_counter_includes_every_series_run_against_the_same_targets():
    """Знаменатель обязан включать ВСЕ прогоны против тех же ячеек и той же цели.

    Регрессия на ошибку 25.08.2026: слагаемое блока 2 задачи №2 объявлено не в
    секции задачи №2, а в секции задачи №3 (FZ_N_TESTS_BLOCK2 рядом с
    FG_N_TESTS_PRIOR), и попытка «починить» FZ_N_TESTS_PRIOR сложением всего
    подряд молча сдвинула порог уже опубликованной задачи №3 с 92 тестов на 132.
    Цепочка теперь фиксируется тестом целиком.
    """
    assert config.FZ_N_TESTS_PRIOR == config.FZ_N_TESTS_BLOCK1 + config.FZ_N_TESTS_2C
    assert config.FG_N_TESTS_PRIOR == config.FZ_N_TESTS_PRIOR + config.FZ_N_TESTS_BLOCK2
    assert config.PF_N_TESTS_PRIOR == config.FG_N_TESTS_PRIOR + config.FG_N_TESTS
    # Опубликованные значения сводного отчёта по задачам №2/№2C/№3.
    assert config.FG_N_TESTS_PRIOR + config.FG_N_TESTS == 92
    assert config.PF_N_TESTS_PRIOR + config.FZ_N_TESTS_PF == 108


def test_threshold_is_reachable_with_configured_number_of_shifts():
    """Порог, недостижимый ни при каком результате, — это грабли этапа 7a.

    Минимальное p сдвигового нуля равно 1/(1+n_shifts); если оно не ниже alpha,
    тест не может быть пройден вообще никогда, и прогон бессмыслен.
    """
    alpha = 0.05 / (config.PF_N_TESTS_PRIOR + config.FZ_N_TESTS_PF)
    assert 1.0 / (1.0 + config.FZ_N_SHIFTS) < alpha


# --- Сборка скоров из таблицы ---

def _frame(cols):
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=50) for c in cols})


def test_edge_tracer_scores_reports_missing_instead_of_faking_them():
    """Ненайденная колонка обязана попасть в «не посчитано», а не в нули."""
    df = _frame(["gm_gr_1G_25", "pf_mag_asig"])
    scores, missing = pf.edge_tracer_scores(df)
    assert set(scores) == {"thdr_gr", "asig_mag"}
    assert set(missing) == set(pf.EDGE_TRACERS) - {"thdr_gr", "asig_mag"}


def test_edge_tracer_scores_applies_the_registry_orientation():
    df = _frame(["pf_mag_tdr"])
    scores, _ = pf.edge_tracer_scores(df)
    np.testing.assert_allclose(scores["tilt_mag"], -np.abs(df["pf_mag_tdr"]))


def test_edge_tracer_scores_on_empty_frame_is_all_missing():
    scores, missing = pf.edge_tracer_scores(pd.DataFrame())
    assert scores == {}
    assert set(missing) == set(pf.EDGE_TRACERS)
