"""Тесты src/assessment.py: P-A метрики, бутстрэп, планки, карта расхождений,
строгий дедуп точек, кластеры, realized_area, сдвиговый null."""
import numpy as np
import pandas as pd

from src import assessment


def _score_with_top_points(n=1000, n_pts=10):
    """Скор 0..1 по индексу; точки — в самых верхних ячейках."""
    score = np.linspace(0.0, 1.0, n)
    points = np.arange(n - n_pts, n)
    return score, points


def test_pa_curve_full_capture_on_top_score():
    score, points = _score_with_top_points()
    pa = assessment.pa_curve(score, points, areas=(0.05, 0.10))
    assert np.allclose(pa["coverage"], 1.0)
    assert np.allclose(pa["lift"], [1 / 0.05, 1 / 0.10])
    # монотонность: coverage не убывает с ростом площади
    pa_full = assessment.pa_curve(score, points)
    assert (np.diff(pa_full["coverage"]) >= -1e-12).all()


def test_capture_efficiency_random_points_near_one():
    rng = np.random.default_rng(1)
    score = rng.random(20_000)
    points = rng.choice(20_000, size=500, replace=False)
    lift = assessment.capture_efficiency(score, points, area=0.10)
    assert 0.7 < lift < 1.3


def test_bootstrap_diff_identical_scores_ci_covers_zero():
    score, points = _score_with_top_points()
    res = assessment.bootstrap_diff(score, score.copy(), points, n_boot=200, seed=0)
    assert res["delta"] == 0.0
    assert res["ci_lo"] <= 0.0 <= res["ci_hi"]


def test_bootstrap_diff_detects_dominant_method():
    score_good, points = _score_with_top_points()
    score_bad = score_good[::-1].copy()  # точки в хвосте
    res = assessment.bootstrap_diff(score_good, score_bad, points,
                                    area=0.10, n_boot=200, seed=0)
    assert res["delta"] > 0
    assert res["ci_lo"] > 0  # превосходство значимо


def test_pool_excludes_cells_from_threshold():
    # вне пула лежат ячейки с огромным скором — без пула они съели бы top-10%
    score = np.concatenate([np.linspace(0, 1, 900), np.full(100, 10.0)])
    points = np.arange(890, 900)             # верх пула, но не верх всей сетки
    pool = np.arange(900)
    assert assessment.capture_efficiency(score, points, 0.10, pool=pool) == 10.0
    assert assessment.capture_efficiency(score, points, 0.10, pool=None) == 0.0


def test_naive_hydro_score_orientation():
    df = pd.DataFrame({"dist_dnl": [10.0, 5000.0], "dist_dnara": [400.0, 9000.0]})
    s = assessment.naive_hydro_score(df)
    assert s[0] > s[1]  # ближе к реке = выше скор


def test_random_score_reproducible():
    a = assessment.random_score(100, seed=7)
    b = assessment.random_score(100, seed=7)
    np.testing.assert_array_equal(a, b)


def test_filter_points_drops_invalid_cells():
    valid = np.array([True, False, True, True])
    pts = np.array([0, 1, 3])
    np.testing.assert_array_equal(assessment.filter_points(pts, valid), [0, 3])


def test_map_agreement_codes_and_pool():
    n = 100
    a = np.zeros(n); b = np.zeros(n)
    a[:20] = 1.0                  # top-10% A: ячейки с максимумами A
    b[10:30] = 2.0                # top-10% B
    a[10:20] += 1.0               # пересечение 10..19 — верх обеих
    codes = assessment.map_agreement(a, b, area=0.10)
    assert set(np.unique(codes)) <= {0, 1, 2, 3}
    assert (codes[10:20] == 3).all()
    assert (codes == 0).sum() > 50
    # вне пула — всегда 0
    pool = np.arange(50)
    codes_p = assessment.map_agreement(a, b, area=0.10, pool=pool)
    assert (codes_p[50:] == 0).all()


def test_spearman_maps_extremes():
    x = np.linspace(0, 1, 200)
    assert assessment.spearman_maps(x, x * 2 + 1) == 1.0
    assert assessment.spearman_maps(x, -x) == -1.0


# --------------------------------------------------------------------------
# Строгий дедуп, кластеры, realized_area, сдвиговый null (пункт 13)
# --------------------------------------------------------------------------

def _toy_meta(prf=10, pic=100):
    from src.integro_grid import GridMeta
    return GridMeta(obj_count=prf * pic, prop_count=0, pic=pic, prf=prf,
                    dx=500.0, dy=500.0, x0=0.0, y0=0.0)


def test_aggregate_point_cells_strict_dedup():
    # ячейка 5: коды {1, 2} — НЕ несмещённая (есть рыхлая проба);
    # ячейка 7: только код 2 — строго несмещённая; ячейка 9: только код 5 — нет
    out = assessment.aggregate_point_cells(
        np.array([5, 5, 7, 9]), np.array([1, 2, 2, 5]))
    out = out.set_index("cell")
    assert not out.loc[5, "unbiased_strict"]
    assert out.loc[5, "code"] == 2          # приоритетный код сохранён для карт
    assert out.loc[7, "unbiased_strict"]
    assert not out.loc[9, "unbiased_strict"]
    np.testing.assert_array_equal(
        assessment.unbiased_cells(out.reset_index()), [7])


def test_point_clusters_single_linkage():
    meta = _toy_meta()
    # ячейки 0,1,2 — соседние (500 м < 2000 м), ячейка 50 — в 24 км
    labels = assessment.point_clusters(np.array([0, 1, 2, 50]), meta)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    assert len(np.unique(labels)) == 2
    # вырожденный случай: одна точка — один кластер
    assert assessment.point_clusters(np.array([3]), meta).size == 1


def test_bootstrap_diff_cluster_resampling():
    score, points = _score_with_top_points()
    clusters = np.repeat([1, 2], points.size // 2)
    res = assessment.bootstrap_diff(score, score.copy(), points,
                                    n_boot=100, seed=0, clusters=clusters)
    assert res["delta"] == 0.0
    assert res["ci_lo"] <= 0.0 <= res["ci_hi"]
    # кластерный ДИ доминирующего метода шире не делает вывод ложным
    res2 = assessment.bootstrap_diff(score, score[::-1].copy(), points,
                                     n_boot=100, seed=0, clusters=clusters)
    assert res2["delta"] > 0


def test_coverage_and_area_constant_score_ties():
    # константный скор: порог накрывает 100% пула -> realized_area=1, lift=1
    score = np.ones(100)
    cov, realized = assessment.coverage_and_area(score, np.array([0, 5]), 0.10)
    assert cov == 1.0 and realized == 1.0
    assert assessment.capture_efficiency(score, np.array([0, 5]), 0.10) == 1.0
    pa = assessment.pa_curve(score, np.array([0, 5]), areas=(0.10,))
    assert pa.loc[0, "lift"] == 1.0 and pa.loc[0, "realized_area"] == 1.0


def test_spatial_null_p_in_unit_interval_and_degenerate():
    meta = _toy_meta(prf=20, pic=30)
    rng = np.random.default_rng(0)
    score = rng.random(meta.prf * meta.pic)
    points = rng.choice(score.size, size=5, replace=False)
    res = assessment.spatial_null_pvalue(score, points, meta,
                                         area=0.10, n_shifts=99, seed=1)
    assert 0.0 < res["p"] <= 1.0
    # точки в самом низу скора: observed lift = 0 -> p = 1 (вырожденно)
    order = np.argsort(score)
    res0 = assessment.spatial_null_pvalue(score, order[:5], meta,
                                          area=0.10, n_shifts=99, seed=1)
    assert res0["observed"] == 0.0 and res0["p"] == 1.0


def test_spatial_null_detects_true_signal():
    meta = _toy_meta(prf=30, pic=40)
    rng = np.random.default_rng(2)
    score = rng.random(meta.prf * meta.pic) * 0.01
    points = np.array([100, 500, 900])
    score[points] = 10.0                      # карта пикует ровно на точках
    res = assessment.spatial_null_pvalue(score, points, meta,
                                         area=0.10, n_shifts=200, seed=3)
    assert res["p"] < 0.05
    assert res["observed"] > res["null_q95"]
