"""ANFIS по ЛИНИЯМ: можно ли предсказать подтверждение по геометрии линии.

ОТКУДА ВОПРОС. Поячеечный ANFIS выигрывает внутри площадки и проваливается на
переносе (1.72 против 0.85): спектр ячейки у каждой площадки свой. Здесь тот же
ANFIS работает не с ячейками, а с ЛИНИЯМИ, и признаки у них другие по природе —
длина, прямолинейность, контраст относительно локального фона, однородность силы
вдоль линии. Это относительные и безразмерные величины, и именно поэтому у них
есть шанс перенестись туда, где спектр не перенёсся.

ЗАЧЕМ ЭТО НУЖНО ПРАКТИЧЕСКИ. Сейчас подтверждение линии требует ДВУХ приборов:
оптики и радара по отдельности. Если модель научится предсказывать подтверждение
по геометрии, то на площадке с одной съёмкой линии можно будет отранжировать по
уверенности, а не отдавать вповалку. Правила ANFIS при этом скажут словами, чем
настоящая линия отличается от случайной.

ЧЕСТНОСТЬ ПРОВЕРКИ. Метка «подтверждена» получена на каждой площадке независимо
и одинаково (``openlin_unsup``). Модель учится на одной площадке и ни в каком
виде не видит другую. Первичная метрика — AUC на ПЕРЕНОСЕ; внутри площадки
число приводится для сравнения, но ответом на вопрос не является.

Запуск из корня репозитория:
    python -m experiments.openlin_line_rank --train anabar --test nevada
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import SEED, fit_anfis, anfis_predict
from experiments.openlin_domains import edges_from_components
from experiments.openlin_gate import components
from experiments.openlin_lines import _axis, _branches, trace_lines
from experiments.openlin_unsup import (CONFIRM_ANGLE_DEG, CONFIRM_FRAC, OPTIC,
                                       RADAR, azimuth_field)
from experiments.openlin_anfis import TOL_M

M = Path("outputs/metrics")

#: Радиус кольца, в котором меряется локальный фон силы границы. 1 км — заметно
#: больше ширины линии и заметно меньше масштаба, на котором меняется тип
#: поверхности, то есть кольцо описывает окружение именно этой линии.
RING_M = 1000.0

#: Пре-регистрация. Своё слагаемое в своей секции.
#: Кандидаты: 2 направления переноса x 3 модели = 6.
#: ПЕРВИЧНАЯ МЕТРИКА — AUC на переносе; значимость — перестановочный нуль по
#: меткам (2000 перестановок), потому что объекты здесь линии, а не ячейки, и
#: сдвиговый нуль к ним неприменим.
RANK_N_TESTS_PRIOR = 219
RANK_N_TESTS = 6
RANK_ALPHA = 0.05 / (RANK_N_TESTS_PRIOR + RANK_N_TESTS)
#: 2000 перестановок дали бы минимальное p = 0.0005 при alpha = 0.00022 —
#: порог был бы недостижим при любом результате. 10000 дают 0.0001, запас 2.2x;
#: перестановки здесь дешёвые, объекты — линии, а не ячейки.
N_PERM = 10_000
assert 1.0 / (1 + N_PERM) < RANK_ALPHA, "порог недостижим при таком числе перестановок"

FEATURES = ("длина_км", "прямолинейность", "сила_средняя", "сила_макс",
            "сила_над_фоном", "однородность", "согласие_с_полем",
            "изломов_на_км", "соседей_рядом")


def line_table(site: str, density: str, with_coords: bool = False):
    """Признаки и метка подтверждения для каждой прослеженной линии.

    При ``with_coords`` возвращает ещё и координаты веток в том же порядке —
    нужно, чтобы раскрасить линии на карте предсказанной уверенностью.
    """
    from scipy.ndimage import distance_transform_edt, uniform_filter

    fields = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        z, valid, res_m, comp = components(site, chans)
        s, ang = edges_from_components(comp, valid, res_m)
        s = np.nan_to_num(s, nan=0.0)
        fields[name] = (s, ang, trace_lines(s, valid, valid, res_m, density)[0])

    s_all, ang_all, l_all = fields["оптика+радар"]
    l_opt, l_rad = fields["оптика"][2], fields["радар"][2]

    # Локальный фон силы: среднее в окне 2*RING, чтобы «сила над фоном» была
    # относительной величиной и не зависела от общей яркости площадки.
    k = int(round(2 * RING_M / res_m))
    bg = uniform_filter(s_all.astype("float64"), size=k, mode="nearest")
    bg_sd = np.sqrt(np.maximum(
        uniform_filter(s_all.astype("float64") ** 2, size=k, mode="nearest")
        - bg ** 2, 1e-12))

    dens = uniform_filter(l_all.astype("float64"), size=k, mode="nearest")

    conf = {}
    for tag, other in (("opt", l_opt), ("rad", l_rad)):
        dt, idx = distance_transform_edt(~other, sampling=(res_m, res_m),
                                         return_indices=True)
        conf[tag] = (dt, idx, azimuth_field(other, res_m))

    rows, coords = [], []
    for pts in _branches(l_all, res_m):
        if len(pts) < 3:
            continue
        coords.append(pts)
        r, c = pts[:, 0], pts[:, 1]
        ax, _a, _b = _axis(pts)
        proj = (pts - pts.mean(0)) @ ax
        length = float(proj.max() - proj.min()) * res_m
        path = len(pts) * res_m
        my_az = np.degrees(np.arctan2(ax[1], -ax[0])) % 180

        sv = s_all[r, c]
        d_field = np.abs(ang_all[r, c] - my_az)
        d_field = np.minimum(d_field, 180.0 - d_field)

        ok = {}
        for tag in ("opt", "rad"):
            dt, idx, az_o = conf[tag]
            near = dt[r, c] <= TOL_M
            oth = az_o[idx[0][r, c], idx[1][r, c]]
            dd = np.abs(oth - my_az)
            same = np.minimum(dd, 180.0 - dd) <= CONFIRM_ANGLE_DEG
            ok[tag] = float((near & same & np.isfinite(oth)).mean())

        rows.append({
            "длина_км": length / 1000.0,
            "прямолинейность": min(length / path, 1.0) if path else 0.0,
            "сила_средняя": float(sv.mean()),
            "сила_макс": float(sv.max()),
            "сила_над_фоном": float(((sv - bg[r, c]) / bg_sd[r, c]).mean()),
            "однородность": float(1.0 - sv.std() / max(sv.mean(), 1e-6)),
            "согласие_с_полем": float(1.0 - d_field.mean() / 90.0),
            "изломов_на_км": float(np.abs(np.diff(np.diff(proj))).sum()
                                   / max(length / 1000.0, 1e-6)),
            "соседей_рядом": float(dens[r, c].mean()),
            "подтверждена": int(ok["opt"] >= CONFIRM_FRAC
                                and ok["rad"] >= CONFIRM_FRAC),
        })
    df = pd.DataFrame(rows)
    return (df, coords, valid, res_m) if with_coords else df


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC через ранги: доля правильно упорядоченных пар."""
    from scipy.stats import rankdata

    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="anabar")
    ap.add_argument("--test", default="nevada")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--rules", type=int, default=8)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"обучение {a.train}, перенос {a.test}, плотность '{a.density}'")
    log(f"пре-регистрация: первичная метрика — AUC на переносе; тестов "
        f"{RANK_N_TESTS_PRIOR + RANK_N_TESTS}, alpha = {RANK_ALPHA:.6f}")

    tr = line_table(a.train, a.density)
    te = line_table(a.test, a.density)
    for nm, d in ((a.train, tr), (a.test, te)):
        log(f"  {nm}: линий {len(d)}, подтверждённых "
            f"{int(d['подтверждена'].sum())} ({100 * d['подтверждена'].mean():.0f}%)")

    xtr = tr[list(FEATURES)].to_numpy("float32")
    xte = te[list(FEATURES)].to_numpy("float32")
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd          # нормировка ПО ОБУЧЕНИЮ, не по проверке
    ytr = tr["подтверждена"].to_numpy("float32")
    yte = te["подтверждена"].to_numpy()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    models = {
        "ANFIS": None,
        "логрегрессия": LogisticRegression(max_iter=1000,
                                           class_weight="balanced").fit(xtr, ytr > 0.5),
        "бустинг": HistGradientBoostingClassifier(
            random_state=SEED, max_iter=200,
            class_weight="balanced").fit(xtr, ytr > 0.5),
    }
    # 8 правил, а не 16: линий шестьсот, а не полмиллиона, и на таком объёме
    # шестнадцать правил дали бы по три десятка линий на правило.
    anfis = fit_anfis(xtr, ytr, a.rules, soft=False, log=lambda *_: None)

    rng = np.random.default_rng(SEED)
    log("\n=== AUC (первичная метрика — перенос) " + "=" * 26)
    out = []
    for name in models:
        s_tr = (anfis_predict(anfis, xtr) if name == "ANFIS"
                else models[name].predict_proba(xtr)[:, 1])
        s_te = (anfis_predict(anfis, xte) if name == "ANFIS"
                else models[name].predict_proba(xte)[:, 1])
        a_tr, a_te = auc(ytr > 0.5, s_tr), auc(yte, s_te)
        null = [auc(rng.permutation(yte), s_te) for _ in range(N_PERM)]
        p = (1 + sum(v >= a_te for v in null)) / (1 + N_PERM)
        out.append({"модель": name, "AUC_обучение": a_tr, "AUC_перенос": a_te,
                    "p": p, "значим": p < RANK_ALPHA})
        log(f"  {name:<14} своя {a_tr:.3f}   перенос {a_te:.3f}   p = {p:.4f}")

    base = auc(yte, te["длина_км"].to_numpy())
    log(f"\n  планка «только длина»: AUC переноса {base:.3f}")
    log(f"  планка «случайный порядок»: 0.500")

    df = pd.DataFrame(out)
    df.to_csv(M / f"openlin_line_rank_{a.train}_to_{a.test}.csv", index=False,
              encoding="utf-8-sig")
    tr.to_csv(M / f"openlin_line_feats_{a.train}.csv", index=False,
              encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_line_rank_{a.train}_to_{a.test}.csv'}")


if __name__ == "__main__":
    main()
