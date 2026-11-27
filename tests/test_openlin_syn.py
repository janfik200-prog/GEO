"""Тесты SYN1: вставка синтетических линий и общая фазовая подделка."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _коррелированные_поля(n=256, seed=0):
    rng = np.random.default_rng(seed)
    from scipy import ndimage as ndi
    a = ndi.gaussian_filter(rng.normal(size=(n, n)), 3)
    b = a + 0.3 * ndi.gaussian_filter(rng.normal(size=(n, n)), 3)
    return a.astype("float32"), b.astype("float32")


def test_joint_fake_keeps_cross_correlation():
    from experiments.openlin_syn_insert import joint_phase_randomize
    a, b = _коррелированные_поля()
    valid = np.ones(a.shape, bool)
    r0 = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    fa, fb = joint_phase_randomize([a, b], valid, np.random.default_rng(1))
    r1 = np.corrcoef(fa.ravel(), fb.ravel())[0, 1]
    assert abs(r1 - r0) <= 0.05, (r0, r1)
    # среднее и разброс по valid сохранены
    assert abs(fa.mean() - a.mean()) < 1e-3
    assert abs(fa.std() - a.std()) < 1e-3


def test_joint_fake_breaks_lines():
    from experiments.openlin_syn_insert import joint_phase_randomize
    from experiments.openlin_cand1 import ridge_dir
    from experiments.openlin_fam import RES_IN
    rng = np.random.default_rng(2)
    n = 256
    x = rng.normal(size=(n, n)).astype("float64") * 0.3
    row = 128
    x[row - 1:row + 2, 60:180] -= 5.0              # конечная прямая тёмная полоса (бесконечная полоса фазами не разрушается: спектр только на kx=0)
    sigs = [2 * RES_IN]
    valid = np.ones(x.shape, bool)
    resp0, _ = ridge_dir(x, sigs, +1)
    k0 = resp0[row].max() / np.median(resp0)
    assert k0 >= 3                                    # у оригинала линия видна
    (fx,) = joint_phase_randomize([x], valid, np.random.default_rng(3))
    resp1, _ = ridge_dir(fx.astype("float64"), sigs, +1)
    # порог «< 3× медианы» недостижим: максимум по любой строке шума ~5× медианы
    # (измерено); поэтому: отклик вдоль строки полосы упал в >4 раза против оригинала
    k1 = resp1[row].max() / np.median(resp1)
    assert k1 < 0.25 * k0, (k0, k1)


def test_joint_fake_seed():
    from experiments.openlin_syn_insert import joint_phase_randomize
    a, b = _коррелированные_поля()
    valid = np.ones(a.shape, bool)
    p = joint_phase_randomize([a, b], valid, np.random.default_rng(5))
    q = joint_phase_randomize([a, b], valid, np.random.default_rng(5))
    r = joint_phase_randomize([a, b], valid, np.random.default_rng(6))
    assert all(np.array_equal(u, v) for u, v in zip(p, q))
    assert not np.array_equal(p[0], r[0])


def test_joint_fake_shape_mismatch():
    from experiments.openlin_syn_insert import joint_phase_randomize
    a = np.zeros((64, 64), "float32")
    b = np.zeros((64, 32), "float32")
    with pytest.raises(ValueError):
        joint_phase_randomize([a, b], np.ones((64, 64), bool),
                              np.random.default_rng(0))
    with pytest.raises(ValueError):
        joint_phase_randomize([a], np.ones((32, 64), bool),
                              np.random.default_rng(0))


# ------------------------------------------------------------- задача 2: врезки
def _шум(n=400, seed=0):
    return np.random.default_rng(seed).normal(size=(n, n))


def test_plan_no_overlap():
    from scipy.spatial.distance import cdist
    from experiments.openlin_syn_insert import SEP_M, make_plan, trace_points
    res = 20.0
    valid = np.ones((1000, 1000), bool)
    plan = make_plan(valid.shape, valid, res, np.random.default_rng(7), 40)
    assert 2 <= len(plan) <= 40
    pts = [trace_points(p, res) for p in plan]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = cdist(pts[i], pts[j]).min() * res
            assert d >= SEP_M - 1e-6, (i, j, d)
    for p in plan:
        r, c = p["center"]
        assert valid[int(r), int(c)]


def test_plan_params_in_declared_ranges():
    from experiments.openlin_syn_insert import (AMP_SIG, KINDS, LEN_M,
                                                STRAIGHT_MIN, make_plan)
    res = 20.0
    valid = np.ones((3000, 3000), bool)
    plan = make_plan(valid.shape, valid, res, np.random.default_rng(1), 300)
    assert {p["kind"] for p in plan} <= set(KINDS)
    for p in plan:
        assert AMP_SIG[0] <= p["amp"] <= AMP_SIG[1]
        assert STRAIGHT_MIN <= p["bend"] <= 1.0
        pieces_len = sum(b - a for a, b in p["pieces"])
        assert LEN_M[0] - 1e-6 <= pieces_len <= LEN_M[1] + 1e-6


def test_straight_equals_insert_fault():
    """Прямая врезка без неравномерности = insert_fault (профиль ridge) + детренд."""
    from scipy.ndimage import gaussian_filter
    from experiments.openlin_syn_insert import apply_plan, insertion
    from experiments.openlin_winter import DETREND_M
    from src.detection_limit import axis_from_geographic, insert_fault
    res, n = 20.0, 400
    img = np.zeros((n, n))
    ins = insertion("светлая", (200.0, 200.0), 30.0, 2000.0, 2.0,
                    np.random.default_rng(0), width_m=80.0, cv=0.0)
    img2, _, _, _ = apply_plan(img, img.copy(), img.copy(), [ins], 1.5, 1.0, res)
    ref = insert_fault(img, (200.0, 200.0), axis_from_geographic(30.0),
                       length_m=2000.0, width_m=80.0, amplitude=2.0, sigma=1.5,
                       res_m=res, profile="ridge")
    ref = ref - gaussian_filter(ref, DETREND_M / res / 2, mode="nearest")
    # окно у нас меньше листа: сравниваем вблизи следа, где детренд одинаков
    assert np.abs(img2[100:300, 100:300] - ref[100:300, 100:300]).max() < 0.02 * 3.0


def test_dark_band_visible():
    from experiments.openlin_cand1 import response
    from experiments.openlin_cand2 import local_norm_nz
    from experiments.openlin_syn_insert import apply_plan, insertion
    from src.detection_limit import robust_sigma
    res = 20.0
    img = _шум()
    dem = _шум(seed=1)
    valid = np.ones(img.shape, bool)
    ins = insertion("тёмная", (200.0, 200.0), 60.0, 3000.0, 4.0,
                    np.random.default_rng(3), width_m=80.0)
    img2, _, _, truth = apply_plan(img, dem, dem.copy(), [ins],
                                   robust_sigma(img), 1.0, res)
    z = local_norm_nz(response(0, img2, dem)[0], valid)
    tr = truth[0]
    assert len(tr) > 100
    assert z[tr[:, 0], tr[:, 1]].mean() > 2


def test_relief_into_both():
    from experiments.openlin_syn_insert import apply_plan, insertion
    res = 20.0
    img, dem, alos = _шум(seed=0), _шум(seed=1), _шум(seed=2)
    ins = insertion("ложбина", (200.0, 200.0), 100.0, 2500.0, 3.0,
                    np.random.default_rng(4), bend=0.8)
    img2, dem2, alos2, _ = apply_plan(img, dem, alos, [ins], 1.0, 2.0, res)
    assert np.array_equal(img2, img)
    assert np.abs(dem2 - dem).max() > 1.0
    assert np.abs((dem2 - dem) - (alos2 - alos)).max() < 1e-9
    # ложбина ниже окружения
    _, _, _, tr = apply_plan(img, dem, alos, [ins], 1.0, 2.0, res)
    t = tr[0]
    assert (dem2 - dem)[t[:, 0], t[:, 1]].mean() < 0


def test_mixed_into_image_and_relief():
    from experiments.openlin_syn_insert import apply_plan, insertion
    res = 20.0
    img, dem = _шум(seed=0), _шум(seed=1)
    ins = insertion("смешанная", (200.0, 200.0), 10.0, 2500.0, 3.0,
                    np.random.default_rng(5))
    img2, dem2, alos2, _ = apply_plan(img, dem, dem.copy(), [ins], 1.0, 1.0, res)
    assert np.abs(img2 - img).max() > 1.0
    assert np.abs(dem2 - dem).max() > 1.0
    assert {p["kind"] for p in ins["parts"]} & {"тёмная", "светлая"}
    assert {p["kind"] for p in ins["parts"]} & {"ложбина", "гребень", "уступ"}


def test_composite_has_gaps():
    from scipy import ndimage as ndi
    from experiments.openlin_syn_insert import apply_plan, insertion
    res = 20.0
    img = np.zeros((400, 400))
    for seed in range(5):
        ins = insertion("составная", (200.0, 200.0), 45.0, 4000.0, 2.0,
                        np.random.default_rng(seed))
        assert 2 <= len(ins["pieces"]) <= 4
        _, _, _, truth = apply_plan(img, img, img, [ins], 1.0, 1.0, res)
        m = np.zeros(img.shape, bool)
        m[truth[0][:, 0], truth[0][:, 1]] = True
        _, n = ndi.label(m, np.ones((3, 3)))
        gaps = [b0 - a1 for (_, a1), (b0, _) in zip(ins["pieces"][:-1], ins["pieces"][1:])]
        # разрыв шире 3 ячеек обязан разделить куски (узкий может слиться 8-связностью)
        assert 1 + sum(g > 3 * res for g in gaps) <= n <= len(ins["pieces"]), (n, gaps)
        assert all(g >= 0 for g in gaps)


def test_truth_inside_valid():
    from experiments.openlin_syn_insert import apply_plan, insertion
    res = 20.0
    img = _шум()
    valid = np.ones(img.shape, bool)
    valid[:, 200:] = False
    ins = insertion("гребень", (200.0, 180.0), 90.0, 3000.0, 2.0,
                    np.random.default_rng(6))
    _, _, _, truth = apply_plan(img, img, img, [ins], 1.0, 1.0, res, valid=valid)
    t = truth[0]
    assert len(t) > 5
    assert valid[t[:, 0], t[:, 1]].all()
    _, _, _, full = apply_plan(img, img, img, [ins], 1.0, 1.0, res)
    assert len(full[0]) > len(t)


def test_relief_sigma():
    from experiments.openlin_syn_insert import relief_sigma
    rng = np.random.default_rng(0)
    from experiments.openlin_winter import detrend
    from src.detection_limit import robust_sigma
    yy, xx = np.mgrid[0:400, 0:400]
    # длинная волна (период 8 км) ослабляется детрендом: гаусс σ = 50 ячеек
    # пропускает exp(−2π²·50²/400²) ≈ 0,73 её амплитуды в фон, вычитая ≈ 73 %
    dem = 30 * np.sin(2 * np.pi * xx / 400) + rng.normal(size=(400, 400))
    valid = np.ones(dem.shape, bool)
    s = relief_sigma(dem, valid)
    assert s == pytest.approx(robust_sigma(detrend(dem, valid, 20.0)[valid]))
    assert s < 0.5 * robust_sigma(dem), (s, robust_sigma(dem))


# ------------------------------------------- исправление 1: ступень локализована
@pytest.mark.parametrize("bend", [1.0, 0.75])
def test_scarp_far_field_vanishes(bend):
    """Перепад: дальше 3σ детренда + 4 ширины от следа поле ≈ 0, без шва на краю окна
    и без поперечных линий у торцов (ревью задачи 2, критическое 1)."""
    from scipy.spatial import cKDTree
    from experiments.openlin_syn_insert import apply_plan, insertion, trace_points
    from experiments.openlin_winter import DETREND_M
    res, n, amp, w = 20.0, 800, 4.0, 80.0
    z = np.zeros((n, n))
    ins = insertion("перепад", (400.0, 400.0), 0.0, 3000.0, amp,
                    np.random.default_rng(0), bend=bend, width_m=w, cv=0.0)
    f, _, _, _ = apply_plan(z, z, z, [ins], 1.0, 1.0, res)
    rr, cc = np.mgrid[0:n, 0:n]
    d = cKDTree(trace_points(ins, res)).query(
        np.c_[rr.ravel(), cc.ravel()])[0].reshape(n, n) * res
    far = d > 3 * DETREND_M / 2 + 4 * w
    assert far.sum() > 0.3 * n * n
    assert np.abs(f[far]).max() < 0.01 * amp, np.abs(f[far]).max()
    # шов на границе окна: соседние ячейки вдали от следа не прыгают
    jump = max(np.abs(np.diff(f, axis=0))[far[1:]].max(),
               np.abs(np.diff(f, axis=1))[far[:, 1:]].max())
    assert jump < 0.001 * amp, jump
    # у линии ступень есть
    assert np.abs(f).max() > 0.3 * amp


def test_scarp_sign_flips_with_azimuth():
    """Перепад по азимуту 0 (на север): нормаль n = (sin a, cos a) смотрит на восток,
    поднят восточный борт; по азимуту 180 — западный. Перегиб — на следе."""
    from experiments.openlin_syn_insert import apply_plan, insertion
    res, n, amp = 20.0, 400, 4.0
    z = np.zeros((n, n))
    sides = {}
    for az in (0.0, 180.0):
        ins = insertion("перепад", (200.0, 200.0), az, 3000.0, amp,
                        np.random.default_rng(0), width_m=80.0, cv=0.0)
        f, _, _, tr = apply_plan(z, z, z, [ins], 1.0, 1.0, res)
        assert set(tr[0][:, 1]) == {200}                 # след — столбец центра
        prof = f[150:251].mean(axis=0)                   # средний профиль поперёк
        east, west = prof[203:210].mean(), prof[191:198].mean()
        sides[az] = east - west
        # перегиб — на следе: в самом столбце следа ≈ 0, по бортам разные знаки
        s = np.sign(sides[az])
        assert abs(prof[200]) < 0.05 * amp, prof[200]
        assert s * prof[201] > 0 > s * prof[199], prof[198:203]
        assert (s * prof[201:211] > 0).all() and (s * prof[190:200] < 0).all()
    assert sides[0.0] > 0.5 * amp and sides[180.0] < -0.5 * amp, sides


_GOLDEN = ROOT / ".superpowers" / "sdd" / "2026-11-23-anfisa-synthetic-teacher" / "golden_task3.npz"


def test_arr_equals_golden():
    """Тракт по массивам (candidates_arr + line_features_arr) даёт тот же пул и те же признаки,
    что прежний путь по листу до рефакторинга (эталон снят golden_task3.py на куске Утсйоки)."""
    if not _GOLDEN.exists():
        pytest.skip("нет эталона golden_task3.npz")
    import hashlib
    from experiments import openlin_anfl as A
    from experiments import openlin_anfl0b as B
    from experiments import openlin_cand1 as C1
    from experiments.openlin_fam import DEM
    g = np.load(_GOLDEN)
    r0, r1, c0, c1 = (int(x) for x in g["crop"])
    h0, h1, h2, h3 = (int(x) for x in g["hole"])
    a0, a1, a2, a3 = (int(x) for x in g["alos_gap"])
    alos_path = DEM / "dem_alos30_utsjoki_лист_20m.npy"
    if not alos_path.exists():
        pytest.skip("нет входов Утсйоки")
    img, dem, valid, _meta = C1.load_inputs("utsjoki")
    raw = np.load(alos_path)[:valid.shape[0], :valid.shape[1]]
    img, dem, valid = img[r0:r1, c0:c1].copy(), dem[r0:r1, c0:c1].copy(), valid[r0:r1, c0:c1].copy()
    alos = raw[r0:r1, c0:c1].astype("float64")
    valid[h0:h1, h2:h3] = False
    alos[a0:a1, a2:a3] = np.nan
    log = lambda *a: None
    pool = A.candidates_arr(img, dem, valid, log)
    h = hashlib.sha256()
    for p in pool:
        h.update(np.array([p["ch"]], "int32").tobytes())
        h.update(np.ascontiguousarray(p["cells"], "int32").tobytes())
    assert h.hexdigest() == str(g["sha"])
    pool = [p for p in pool if valid[p["cells"][:, 0], p["cells"][:, 1]].any()]   # как B.site_data
    assert len(pool) == int(g["n_pool"])
    X = B.line_features_arr(pool, img, dem, alos, valid, log)
    assert X.shape == g["X"].shape
    assert np.allclose(X, g["X"], rtol=0, atol=1e-6)


# ------------------------------------------------- задача 4: метки и сборка обучения
def _кандидат(ch, rows, cols):
    cells = np.stack([np.asarray(rows, int), np.asarray(cols, int)], 1)
    return {"ch": ch, "cells": cells, "poly": cells[[0, -1]].astype(float), "mz": 1.0}


def test_labels_from_truth():
    """Поправка 1 (registrar): ≥ 50 % ячеек в допуске → 1; 0 ячеек → 0; часть (>0, <50 %) → −1."""
    from experiments.openlin_syn1 import labels_from_truth
    from experiments.openlin_zone_lines_v3 import CONF_TOL_M
    shape = (200, 200)
    truth = [np.stack([np.full(100, 100), np.arange(50, 150)], 1)]     # след — строка 100
    col = np.arange(50, 150)
    pool = [
        _кандидат(0, np.full(100, 100), col),                          # на следе → 1
        _кандидат(1, np.full(100, 115), col),                          # параллельный в 300 м → 0
        _кандидат(2, np.full(100, 101), np.arange(100, 200)),          # наполовину на следе → 1
        # четверть на следе → −1: 25 ячеек у следа (строка 101), 75 — далеко (строка 160)
        _кандидат(3, np.r_[np.full(25, 101), np.full(75, 160)],
                  np.r_[np.arange(125, 150), np.arange(0, 75)]),
    ]
    y =labels_from_truth(pool, truth, shape, CONF_TOL_M)
    assert y.tolist() == [1, 0, 1, -1]
    assert y.dtype.kind == "i"


def test_empty_channel_pool():
    """Канал без кандидатов (и пустой след) не роняет метки и сборку признаков."""
    from experiments.openlin_anfl0b import line_features_arr
    from experiments.openlin_syn1 import labels_from_truth
    from experiments.openlin_zone_lines_v3 import CONF_TOL_M
    n = 160
    rng = np.random.default_rng(0)
    img, dem = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    alos = dem.copy()
    alos[:10, :10] = np.nan
    valid = np.ones((n, n), bool)
    pool = [_кандидат(0, np.full(60, 40), np.arange(20, 80)),          # каналов 1–4 нет вовсе
            _кандидат(3, np.arange(30, 90), np.full(60, 120))]
    assert labels_from_truth(pool, [], (n, n), CONF_TOL_M).tolist() == [0, 0]
    assert labels_from_truth([], [], (n, n), CONF_TOL_M).shape == (0,)
    truth = [np.stack([np.full(60, 40), np.arange(20, 80)], 1)]
    y = labels_from_truth(pool, truth, (n, n), CONF_TOL_M)
    assert y.tolist() == [1, 0]
    X = line_features_arr(pool, img, dem, alos, valid, lambda *a: None)
    assert X.shape == (2, 20) and np.isfinite(X).all()


# ------------------------------------------------- задача 5: порог по подделке Б
def _лист_игрушка(scores, n_rows=None):
    """Игрушечный лист: линия i — 10 ячеек в строке i; признак X[:,0] — готовая оценка."""
    n = len(scores)
    pool = [{"cells": np.stack([np.full(10, i), np.arange(10)], 1)} for i in range(n)]
    X = np.zeros((n, 2))
    X[:, 0] = scores
    return {"pool": pool, "X": X, "valid": np.ones((n + 5, 20), bool)}


def test_choose_threshold_toy():
    """Подделка ниже снимка → порог существует, доля в (0, 1]; подделка = снимок → SystemExit."""
    from experiments.openlin_syn1 import choose_threshold
    model = lambda X: X[:, 0]
    log = lambda *a: None
    real = [_лист_игрушка(np.linspace(0, 1, 40))]
    fake = [_лист_игрушка(np.linspace(0, 0.5, 40))]
    thr, share = choose_threshold(real, fake, model, log)
    assert 0.4 < thr <= 1.0           # ниже 0,4 подделка выбирает > 20 % км снимка
    assert 0 < share <= 1
    with pytest.raises(SystemExit):
        choose_threshold(real, [_лист_игрушка(np.linspace(0, 1, 40))], model, log)


def test_choose_threshold_empty_pool():
    """Пустой пул настоящих листов (0 км) → SystemExit, а не порог."""
    from experiments.openlin_syn1 import choose_threshold
    model = lambda X: X[:, 0]
    with pytest.raises(SystemExit):
        choose_threshold([_лист_игрушка(np.zeros(0))], [_лист_игрушка(np.zeros(0))], model, lambda *a: None)


def test_recall_ignores_parallel_real_line():
    """Задача 6, решение контролёра: врезка найдена, только если ≥ 50 % её следа в 60 м от выбранных
    ячеек с направлением в ±TAU от местного направления врезки. Настоящая линия рядом, но другого
    направления («лесенка» поперёк следа), и параллельная линия дальше допуска — не находят врезку."""
    from experiments.openlin_acon1 import TAU
    from experiments.openlin_fam import RES_IN
    from experiments.openlin_syn1 import recall_of_insertions
    from experiments.openlin_syn_insert import insertion, truth_cells
    from experiments.openlin_zone_lines_v3 import CONF_TOL_M
    shape = (300, 300)
    valid = np.ones(shape, bool)
    # азимут 90° (на восток): след — строка 150, направление в соглашении ядер a = π/2
    ins = insertion("тёмная", (150.0, 150.0), 90.0, 2000.0, 2.0, np.random.default_rng(0), cv=0.0)
    tr = truth_cells(ins, shape, valid, RES_IN)
    assert set(tr[:, 0]) == {150}
    col = np.arange(100, 201)
    # 1) своя линия вдоль следа, 1 ячейка в стороне, того же направления → найдена
    cells = np.stack([np.full(col.size, 151), col], 1)
    cov = recall_of_insertions(cells, np.full(col.size, np.pi / 2), [tr], [ins], CONF_TOL_M, TAU)
    assert cov[0] >= 0.99
    # 2) «лесенка»: короткие поперечные линии через 3 ячейки накрывают весь след в 60 м, но поперёк → нет
    lad = np.array([(r, c) for c in range(100, 201, 3) for r in range(146, 155)])
    cov = recall_of_insertions(lad, np.zeros(len(lad)), [tr], [ins], CONF_TOL_M, TAU)
    assert cov[0] == 0.0
    # 3) та же линия, что в 1), но направление на 30° (> 22,5°) в сторону → нет; на 20° → да
    cov30 = recall_of_insertions(cells, np.full(col.size, np.pi / 2 + np.radians(30)), [tr], [ins], CONF_TOL_M, TAU)
    cov20 = recall_of_insertions(cells, np.full(col.size, np.pi / 2 - np.radians(20)), [tr], [ins], CONF_TOL_M, TAU)
    assert cov30[0] == 0.0 and cov20[0] >= 0.99
    # 4) параллельная настоящая линия в 300 м (15 ячеек) — дальше допуска 60 м → нет
    far = np.stack([np.full(col.size, 165), col], 1)
    assert recall_of_insertions(far, np.full(col.size, np.pi / 2), [tr], [ins], CONF_TOL_M, TAU)[0] == 0.0
    # 5) пустой выбор → нули
    assert recall_of_insertions(np.zeros((0, 2), int), np.zeros(0), [tr], [ins], CONF_TOL_M, TAU).tolist() == [0.0]


def test_insertion_dirs_follow_arc():
    """Местное направление врезки — касательная дуги: у изогнутой врезки на концах оно отличается
    от направления в центре на ±θ/2 (θ — центральный угол дуги)."""
    from experiments.openlin_acon1 import angdiff
    from experiments.openlin_fam import RES_IN
    from experiments.openlin_syn1 import insertion_dirs
    from experiments.openlin_syn_insert import _theta_from_ratio, curve_points, insertion
    ins = insertion("гребень", (200.0, 200.0), 30.0, 3000.0, 2.0, np.random.default_rng(1), bend=0.8, cv=0.0)
    s = np.array([-1400.0, 0.0, 1400.0])
    cells = np.rint(curve_points(ins, s, RES_IN)).astype(int)
    a = insertion_dirs(ins, cells, RES_IN)
    a0 = np.arctan2(np.sin(np.radians(30)), np.cos(np.radians(30))) % np.pi     # u = (−cos a, sin a)
    assert angdiff(a[1], a0) < 0.03
    half = _theta_from_ratio(0.8) / 2 * 1400 / 1500
    assert abs(angdiff(a[0], a0) - half) < 0.05 and abs(angdiff(a[2], a0) - half) < 0.05


@pytest.mark.parametrize("kind,az", [("тёмная", 30.0), ("перепад", 120.0), ("гребень", 75.0)])
def test_cand_dirs_match_insertion_dirs(kind, az):
    """Соглашения совпадают: направление ядра канала (cand_dirs, θ отклика) на следе врезки
    в ±TAU от местного направления врезки (insertion_dirs)."""
    from experiments.openlin_acon1 import TAU, angdiff
    from experiments.openlin_fam import RES_IN
    from experiments.openlin_syn1 import cand_dirs, insertion_dirs
    from experiments.openlin_syn_insert import apply_plan, insertion
    img, dem = _шум(seed=0) * 0.2, _шум(seed=1) * 0.2
    ins = insertion(kind, (200.0, 200.0), az, 3000.0, 6.0, np.random.default_rng(2), cv=0.0)
    img2, dem2, _a, truth = apply_plan(img, dem, dem.copy(), [ins], 1.0, 1.0, RES_IN)
    tr = truth[0]
    ch = {"тёмная": 0, "перепад": 2, "гребень": 4}[kind]
    d = cand_dirs([{"ch": ch, "cells": tr}], img2, dem2)[0]
    ok = angdiff(d, insertion_dirs(ins, tr, RES_IN)) <= TAU
    assert ok.mean() > 0.9, ok.mean()


def test_verdict_a():
    """Вердикт (а): старшая корзина > f, старшая > младшей, ρ > 0; пустая крайняя корзина —
    «не определено», засчитывается как НЕ выполнено (решение контролёра)."""
    from experiments.openlin_syn1 import verdict_a
    nan = float("nan")
    ok, txt, det, rho = verdict_a([0.0, 0.1, 0.2, 0.3, 0.6], 0.1)
    assert ok is True and det is True and rho > 0 and txt == "ВЫПОЛНЕНО"
    ok, txt, det, _ = verdict_a([0.0, 0.1, 0.2, 0.3, 0.05], 0.1)          # старшая ≤ f
    assert ok is False and det is True and txt.startswith("ПРОВАЛ")
    ok, txt, det, _ = verdict_a([0.7, 0.1, 0.2, 0.3, 0.6], 0.1)           # старшая ≤ младшей
    assert ok is False and det is True
    ok, txt, det, _ = verdict_a([0.5, 0.8, 0.9, 0.1, 0.6], 0.1)           # ρ ≤ 0
    assert ok is False and det is True
    for r in ([nan, 0.1, 0.2, 0.3, 0.6], [0.0, 0.1, 0.2, 0.3, nan]):      # пустая крайняя корзина
        ok, txt, det, _ = verdict_a(r, 0.1)
        assert ok is False and det is False and txt.startswith("НЕ ВЫПОЛНЕНО (не определено")


@pytest.mark.slow
def test_run_on_crop(tmp_path, monkeypatch):
    """Задача 6, шаг 1: весь run() + windows() на кусках 600×600 всех листов, 200 предложений врезок —
    цепочка доходит до конца, файлы syn1_* пишутся во временный OUT (настоящие выходы не трогаются).
    Идёт ~4–6 мин на процессоре."""
    from experiments.openlin_fam import DEM
    if not (DEM / "dem_alos30_utsjoki_лист_20m.npy").exists():
        pytest.skip("нет входов")
    import json
    from experiments import openlin_syn1 as S1
    monkeypatch.setattr(S1, "OUT", tmp_path)
    monkeypatch.setattr(S1, "CROP", 600)
    lines = []
    log = lambda *a: lines.append(" ".join(str(x) for x in a))
    try:
        S1.run(log, n_ins=200)
    except SystemExit as e:                     # законный стоп по правилу объявления — тоже конец цепочки
        pytest.fail(f"стоп на куске: {e}\n" + "\n".join(lines[-30:]))
    for site in ("utsjoki", "anabar"):
        for ext in ("npz", "gpkg"):
            assert (tmp_path / f"syn1_{site}.{ext}").exists(), (site, ext)
        z = np.load(tmp_path / f"syn1_{site}.npz")
        assert len(z["score"]) == len(z["sel"]) == len(z["ch"]) == z["X"].shape[0]
        assert str(z["teacher"]) == S1.TEACHER
    res = json.loads((tmp_path / "syn1_итог.json").read_text(encoding="utf-8"))
    assert {"а", "б", "в", "код", "входы"} <= set(res)
    assert res["SYN1_N_TESTS"] == 0 and res["SYN1_N_TESTS_PRIOR"] == 160
    S1.windows(log)
    assert len(list((tmp_path / "syn1_utsjoki").glob("окно_*.png"))) == 9
    assert len(list((tmp_path / "syn1_anabar").glob("окно_*.png"))) == 9
    import geopandas as gpd
    g = gpd.read_file(tmp_path / "syn1_utsjoki.gpkg")
    if len(g):
        assert set(g["учитель"]) == {S1.TEACHER}


def test_sample_arrays_on_crop():
    """Сборка обучения на куске листа (600×600): формы, метки в {−1, 0, 1}, есть «да»."""
    from experiments import openlin_cand1 as C1
    from experiments.openlin_fam import DEM
    from experiments.openlin_syn1 import sample_arrays
    alos_path = DEM / "dem_alos30_utsjoki_лист_20m.npy"
    if not alos_path.exists():
        pytest.skip("нет входов Утсйоки")
    img, dem, valid, _meta = C1.load_inputs("utsjoki")
    raw = np.load(alos_path)[:valid.shape[0], :valid.shape[1]].astype("float64")
    H, W = valid.shape
    r0, c0 = H // 2 - 300, W // 2 - 300
    sl = (slice(r0, r0 + 600), slice(c0, c0 + 600))
    d = sample_arrays(img[sl].copy(), dem[sl].copy(), raw[sl].copy(), valid[sl].copy(),
                      seed=3104, n_ins=60, log=lambda *a: None, site="utsjoki-кусок")
    assert set(d) >= {"X", "y", "pool", "truth", "plan", "valid"}
    assert d["X"].shape == (len(d["pool"]), 20)
    assert d["y"].shape == (len(d["pool"]),)
    assert set(np.unique(d["y"]).tolist()) <= {-1, 0, 1}
    assert (d["y"] == 1).sum() >= 1
    assert len(d["truth"]) == len(d["plan"]) >= 1
