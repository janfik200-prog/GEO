"""Форма цели обучения: порог верхних 10% против непрерывного балла.

ПОВОД. Все модели проекта учатся отличать верхние 10% критериального прогноза от
остальных. Но сама метрика сравнивает ВЕРХНИЕ 10% двух карт, то есть зависит от
порядка ячеек вблизи границы отсечения. Двоичный порог этот порядок стирает:
ячейка на 11-м процентиле и ячейка на 99-м для модели одинаково «не цель».
Непрерывный балл его сохраняет. Это единственный неопробованный рычаг, который
не требует новых данных.

ЧТО СРАВНИВАЕТСЯ (признаки, полосы, сглаживание и метрика одни и те же):

* верхние 10% — как сейчас, опорная точка;
* верхние 20% и 30% — шире положительный класс: границе отсечения достаётся
  больше примеров с обеих сторон;
* регрессия на ранг критериального балла — порядок сохраняется полностью,
  шкала равномерная;
* регрессия на сам балл — то же, но в исходной шкале формулы, где различия
  между высокими значениями крупнее.

Метрика во всех случаях считается одинаково: верхние 10% предсказания против
верхних 10% критериальной карты, обе после снятия собственного наклона. Меняется
только то, чему модель учится.

РЕЗУЛЬТАТ (07.09.2026) — небольшой выигрыш у регрессии, в пределах разброса.

| цель | фолды ROC | без тренда | с трендом |
|---|---|---|---|
| верхние 10% (как сейчас) | 0.739 | 0.362 | 0.199 |
| верхние 20% | 0.746 | 0.283 | 0.155 |
| верхние 30% | 0.756 | 0.320 | 0.167 |
| регрессия на ранг балла | 0.759 | 0.367 | 0.184 |
| регрессия на сам балл | **0.760** | **0.384** | **0.201** |

Расширение положительного класса до 20% и 30% ухудшает: −0.080 [−0.141; −0.025]
и −0.044 [−0.121; +0.021]. Регрессия на сам критериальный балл лучше нынешней
цели по всем трём числам, но разность +0.020 [−0.042; +0.087] накрывает ноль —
предъявлять как улучшение нельзя. Устойчиво лучше только ранжирование по фолдам
(0.760 против 0.739), и это не стоит смены рабочей точки само по себе.

Выход: `outputs/metrics/target_form.csv`.
Запуск из корня: ``python -m experiments.target_form``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               make_model, poly_design)
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.spectral_exposure import score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (agree_detrended,  # noqa: E402
                                                     boot_diff, smooth_strips)
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def make_reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.HGB_MAX_ITER, learning_rate=config.HGB_LEARNING_RATE,
        max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
        min_samples_leaf=config.HGB_MIN_SAMPLES_LEAF,
        l2_regularization=config.HGB_L2, early_stopping=False,
        random_state=config.CRIT_SEED)


def strip_map(X, target, y, pool, cx, cy, kind: str):
    """Карта по полосам. `kind`: «класс» — классификация, «число» — регрессия."""
    in_pool = np.zeros(y.size, dtype=bool)
    in_pool[pool] = True
    per_ax, aucs = {}, []
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        cal = np.full(y.size, np.nan)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            if te.size == 0 or y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            if kind == "класс":
                m = make_model().fit(X[tr], target[tr])
                p_tr = m.predict_proba(X[tr])[:, 1]
                p_te = m.predict_proba(X[te])[:, 1]
            else:
                m = make_reg().fit(X[tr], target[tr])
                p_tr, p_te = m.predict(X[tr]), m.predict(X[te])
            ref = np.sort(p_tr)
            cal[te] = np.searchsorted(ref, p_te, side="left") / max(ref.size, 1)
            aucs.append(roc_auc_score(y[te], p_te))
        per_ax[ax] = cal
    return per_ax, aucs


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    blocks = crit_reference.block_labels(meta)

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    X = spectral_mask.apply(feat[cols]).to_numpy(dtype=float)

    rank = np.full(y.size, np.nan)
    order = np.argsort(np.argsort(crit[pool]))
    rank[pool] = order / max(pool.size - 1, 1)
    log("признаков %d, меток верхних 10%% — %d" % (X.shape[1], int(y.sum())))

    targets = [("верхние 10% (как сейчас)", y, "класс")]
    for share in (0.20, 0.30):
        t = np.zeros(y.size, dtype=int)
        t[pool] = (crit[pool] >= np.quantile(crit[pool], 1 - share)).astype(int)
        targets.append(("верхние %d%%" % int(share * 100), t, "класс"))
    targets.append(("регрессия на ранг балла", np.nan_to_num(rank), "число"))
    targets.append(("регрессия на сам балл",
                    np.where(np.isfinite(crit), crit, np.nanmin(crit[pool])), "число"))

    rows, maps = [], {}
    for name, target, kind in targets:
        per_ax, aucs = strip_map(X, target, y, pool, cx, cy, kind)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        maps[name] = sm
        m = score_map(sm, y, crit, pool, valid, D)
        rows.append({"цель": name, "тип": kind,
                     "фолды_roc_auc": float(np.mean(aucs)), **m})
        log("  %-30s без тренда=%.3f  с трендом=%.3f  фолды=%.3f"
            % (name, m["совпадение_без_тренда"], m["совпадение_верх_10"],
               rows[-1]["фолды_roc_auc"]))

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "target_form.csv"
    res.to_csv(out, index=False)
    print()
    print("=== Форма цели обучения ===")
    print(res.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print()
    print("Потолок метрики без тренда — 0.622; 70%% от него это 0.435.")

    ref = "верхние 10% (как сейчас)"
    ta_ref, tc = agree_detrended(maps[ref], crit, pool, valid, D)
    print()
    print("=== Разность с нынешней целью (бутстрэп 10 км) ===")
    for name in maps:
        if name == ref:
            continue
        ta, _ = agree_detrended(maps[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print("  %-30s %+.3f [%+.3f; %+.3f]"
              % (name, b["разность_средняя"], b["разность_ниж_5"],
                 b["разность_верх_95"]))
    print()
    print("OK: %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
