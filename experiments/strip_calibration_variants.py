"""Сколько метрики съедает способ сшивки полос и можно ли сшивать иначе.

ПОВОД. `ceiling_decomposition.py` показал, что потолок совпадения без тренда —
0.622, и упирается в него уже модель на настоящих факторах формулы (0.621).
Но потолок этот создан не природой данных, а нашей сборкой карты: идеальный
предсказатель, пропущенный через конвейер, теряет треть метрики ДО того, как
в дело вступает какая-либо модель. Значит, прежде чем улучшать модель, надо
проверить саму сборку.

ГДЕ ТЕРЯЕТСЯ. Карта склеивается из отложенных полос, и баллы разных полос надо
привести к общей шкале, иначе верхние 10% соберутся из одной полосы. Сейчас
балл ячейки — это её квантиль относительно распределения ТОЙ ЖЕ модели на её
ОБУЧАЮЩИХ ячейках. Но обучающие ячейки лежат в другой части листа, где уровень
перспективности другой из-за регионального наклона. Квантиль поэтому
одновременно приводит шкалу и подмешивает разницу уровней между частями листа —
ровно то, что метрика без тренда потом пытается снять.

ЧТО СРАВНИВАЕТСЯ (модели одни и те же, отличается только сшивка):

* «по обучающим ячейкам полосы» — как сейчас;
* «ранг внутри отложенной полосы» — балл это ранг ячейки среди ячеек своей же
  полосы. Меток не касается, утечки нет; разница уровней между полосами
  снимается полностью, но вместе с ней теряется и реальная разница
  перспективности между полосами;
* «без приведения шкалы» — сырая вероятность модели, для контроля.

Каждая сшивка проверяется дважды: на идеальном предсказателе (какой потолок она
задаёт) и на рабочей точке (что даёт на деле). Полезной сшивка считается,
только если поднимает и то и другое.

РЕЗУЛЬТАТ (07.09.2026) — нынешняя сшивка остаётся лучшей.

Потолок действительно создан сборкой, и обе её части стоят дорого. Идеальный
предсказатель: без сглаживания и без приведения шкалы 1.000; только приведение
шкалы 0.667; только сглаживание 0.728; вместе 0.622. Ранг внутри отложенной
полосы поднимает потолок до 0.644.

Но на реальной модели всё наоборот: сглаживание не отнимает, а ДОБАВЛЯЕТ
(0.260 -> 0.362), ранг внутри полосы даёт 0.306 (−0.051 [−0.149; +0.036]),
без приведения шкалы 0.151 (−0.210 [−0.362; −0.055]). То есть сборка режет
потолок, но столько же и больше отыгрывает на шуме реальных предсказаний.
Свободного улучшения здесь нет: ни один из двух вариантов сшивки не обыгрывает
нынешний, а отказ от приведения шкалы разрушает карту.

Выход: `outputs/metrics/strip_calibration_variants.csv`.
Запуск из корня: ``python -m experiments.strip_calibration_variants``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

MODES = ("по обучающим ячейкам полосы", "ранг внутри отложенной полосы",
         "без приведения шкалы")
T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def _rank(v: np.ndarray) -> np.ndarray:
    """Ранг в пределах массива, приведённый к 0..1."""
    order = np.argsort(np.argsort(v))
    return order / max(v.size - 1, 1)


def strip_maps(score, y, pool, cx, cy):
    """По каждой сшивке — свой набор карт по осям. `score(tr, te) -> (p_tr, p_te)`."""
    in_pool = np.zeros(y.size, dtype=bool)
    in_pool[pool] = True
    out = {m: {} for m in MODES}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        cal = {m: np.full(y.size, np.nan) for m in MODES}
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            if te.size == 0 or y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            p_tr, p_te = score(tr, te)
            ref = np.sort(p_tr)
            cal[MODES[0]][te] = np.searchsorted(ref, p_te,
                                                side="left") / max(ref.size, 1)
            cal[MODES[1]][te] = _rank(p_te)
            cal[MODES[2]][te] = p_te
        for m in MODES:
            out[m][ax] = cal[m]
    return out


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
    log("признаков %d, ячеек в работе %d" % (X.shape[1], pool.size))

    def oracle(tr, te):
        return crit[tr], crit[te]

    def model(tr, te):
        m = make_model().fit(X[tr], y[tr])
        return m.predict_proba(X[tr])[:, 1], m.predict_proba(X[te])[:, 1]

    rows, maps = [], {}
    for kind, score in (("идеальный предсказатель (потолок)", oracle),
                        ("рабочая точка", model)):
        per_mode = strip_maps(score, y, pool, cx, cy)
        log("%s: карты собраны" % kind)
        for mode in MODES:
            for smooth in (False, True):
                per = per_mode[mode]
                s = (smooth_strips(per, meta, valid, cx, cy) if smooth
                     else mean_axes(per["X"], per["Y"]))
                name = "%s | %s | %s" % (kind, mode,
                                         "со сглаживанием" if smooth
                                         else "без сглаживания")
                maps[name] = s
                m = score_map(s, y, crit, pool, valid, D)
                rows.append({"что_собирается": kind, "сшивка": mode,
                             "сглаживание": "да" if smooth else "нет", **m})
                log("  %-58s без тренда=%.3f  с трендом=%.3f"
                    % (mode + (" +сглаж." if smooth else ""),
                       m["совпадение_без_тренда"], m["совпадение_верх_10"]))

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "strip_calibration_variants.csv"
    res.to_csv(out, index=False)
    print()
    print("=== Способ сшивки полос: потолок и рабочая точка ===")
    print(res.to_string(index=False, float_format=lambda v: "%.4f" % v))

    ref = "рабочая точка | %s | со сглаживанием" % MODES[0]
    ta_ref, tc = agree_detrended(maps[ref], crit, pool, valid, D)
    print()
    print("=== Разность с нынешней сшивкой рабочей точки (бутстрэп 10 км) ===")
    for name in maps:
        if name == ref or not name.startswith("рабочая точка"):
            continue
        ta, _ = agree_detrended(maps[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print("  %-58s %+.3f [%+.3f; %+.3f]"
              % (name.replace("рабочая точка | ", ""), b["разность_средняя"],
                 b["разность_ниж_5"], b["разность_верх_95"]))
    print()
    print("OK: %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
