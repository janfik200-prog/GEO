"""Потолок совпадения с критериальным прогнозом и из чего складывается разрыв.

ПОСТАНОВКА. Рабочая точка на дистанционных данных даёт совпадение верхних 10%
без тренда 0.362, модель на настоящих факторах формулы — 0.621. Прежде чем
улучшать модель, надо понять, что означают эти два числа.

Разрыв до единицы состоит из трёх слагаемых, и их надо разделить:

1. **Цена протокола.** Наша карта собирается из отложенных полос с разрывом
   15 км, калибруется по квантилю внутри полосы и сглаживается ядром
   5 x 1.25 км. Даже ИДЕАЛЬНЫЙ предсказатель, пропущенный через этот же
   конвейер, не совпадёт с исходной картой на 100%: сглаживание сдвигает
   границы верхних 10%, калибровка по полосам меняет ранги. Считается прямо:
   на вход конвейера подаётся сам критериальный балл. Ни одна карта в проекте
   не может превысить это число — это потолок метрики, а не модели.
2. **Цена отсутствия факторных слоёв.** Разница между идеальной картой и
   моделью на настоящих восьми факторах: сколько теряется оттого, что формулу
   приходится восстанавливать обучением, а не считать.
3. **Цена перехода на дистанционку.** Разница между моделью на настоящих
   факторах и рабочей точкой: сколько теряется оттого, что факторных слоёв нет.

Дальше третье слагаемое раскладывается по факторам: к дистанционным признакам
добавляется по одному настоящему фактору, и смотрится, какой из них закрывает
разрыв. Отдельно проверяется утверждение, ради которого всё затевалось: правда
ли узкое место — разломные слои. Для этого считаются два набора факторов —
пять факторов без разломных и только три разломных.

ЗАЧЕМ ИМЕННО ТАК. Утверждение «разломы не восстанавливаются дистанционкой»
(`directional_relief_faults.py`) объясняет разрыв только если разломные слои
вообще что-то дают карте. Если пять факторов без разломов дают почти тот же
потолок, то узкое место в другом месте, и работу надо вести там.

Метрика и сборка карты везде одинаковы: полосы с разрывом 15 км, сглаживание
ядром 5 x 1.25 км на северо-запад, совпадение верхних 10% после снятия
собственного наклона у обеих карт. Разности с рабочей точкой — блочный
бутстрэп по блокам 10 км.

РЕЗУЛЬТАТ (07.09.2026).

* **Потолок метрики — 0.622.** Идеальный предсказатель в нашем конвейере
  совпадает с критериальной картой без тренда на 0.622 (с трендом 0.699). Это
  верхняя граница для любой карты проекта. Рабочая точка 0.362 — это 58% от
  достижимого, а не 36% от единицы. По метрике С трендом картина хуже: 0.199
  против потолка 0.699, то есть 28%.
* **Настоящие 8 факторов формулы дают 0.621 — ровно потолок.** Значит, в
  критериальной карте нет ничего сверх её восьми входов, и весь разрыв
  объясняется не «формулу трудно выучить», а «факторных слоёв нет».
* **Ни один фактор по отдельности разрыв не закрывает.** Пять факторов без
  разломных дают 0.421, три разломных — 0.230, вместе 0.621. Сумма частей
  сильно меньше целого: формула держится на пересечениях факторов, что
  согласуется с её надбавками за пересечения (`config.LOCAL_BONUS_WEIGHTS`).
* **Добавление одного настоящего фактора к дистанционке НЕ помогает.**
  Расстояние до магматических тел +0.011, до разломов 1-й системы +0.006 —
  оба в пределах разброса; остальные шесть ухудшают, расстояние до
  структурного несогласия −0.165 [−0.276; −0.058], интервал не накрывает ноль.
  При этом метрика С трендом от тех же добавок растёт (плотность разломов
  0.199 -> 0.473, каппа 0.110 -> 0.415): одиночный фактор входит в модель как
  региональный наклон. Это та же ловушка, что в
  [[project-elevation-dependence]], и она же — причина не смотреть на
  совпадение с трендом отдельно от поправки.

Вывод: улучшать нечего добавлением по одному — разрыв 0.362 -> 0.622 закрывается
только всем набором факторных слоёв сразу, которых на новой территории нет.

Выход: `outputs/metrics/ceiling_decomposition.csv`.
Запуск из корня: ``python -m experiments.ceiling_decomposition``
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
                                               poly_design)
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.factor_from_remote import FACTORS, FACTOR_TITLE  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (agree_detrended,  # noqa: E402
                                                     boot_diff, smooth_strips)
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAULT_FACTORS = ("dist_tect1", "dist_tect2", "dens_tect")
T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def oracle_axes(value: np.ndarray, y: np.ndarray, pool: np.ndarray,
                cx: np.ndarray, cy: np.ndarray) -> dict:
    """Идеальный предсказатель в конвейере: на выходе полосы — сам критерий.

    Повторяет `blind_map` во всём, кроме модели: вместо обученного предсказания
    в отложенную полосу кладётся истинный критериальный балл, откалиброванный
    по квантилю обучающих ячеек той же полосы. Отбор полос — по тому же условию
    на число меток, иначе покрытие карт будет разным и числа несравнимы.
    """
    in_pool = np.zeros(y.size, dtype=bool)
    in_pool[pool] = True
    per_ax = {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        cal = np.full(y.size, np.nan)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            if te.size == 0 or y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            ref = np.sort(value[tr])
            cal[te] = np.searchsorted(ref, value[te], side="left") / max(ref.size, 1)
        per_ax[ax] = cal
    return per_ax


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
    work = spectral_mask.apply(feat[cols])
    X = work.to_numpy(dtype=float)
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    log("признаков дистанционки %d, факторов %d, ячеек в работе %d"
        % (X.shape[1], len(FACTORS), pool.size))

    rows, maps = [], {}

    def add(name: str, per_ax: dict, n_feat, auc=float("nan")) -> None:
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        maps[name] = sm
        m = score_map(sm, y, crit, pool, valid, D)
        rows.append({"вариант": name, "признаков": n_feat,
                     "фолды_roc_auc": auc, **m})
        log("  %-46s без тренда=%.3f  с трендом=%.3f"
            % (name, m["совпадение_без_тренда"], m["совпадение_верх_10"]))

    # 1. Потолок метрики: идеальный предсказатель в том же конвейере
    per_or = oracle_axes(crit, y, pool, cx, cy)
    s_or = mean_axes(per_or["X"], per_or["Y"])
    m_or = score_map(s_or, y, crit, pool, valid, D)
    rows.append({"вариант": "идеальная карта, БЕЗ сглаживания", "признаков": 0,
                 "фолды_roc_auc": float("nan"), **m_or})
    log("  %-46s без тренда=%.3f  с трендом=%.3f"
        % ("идеальная карта, БЕЗ сглаживания", m_or["совпадение_без_тренда"],
           m_or["совпадение_верх_10"]))
    add("идеальная карта через конвейер (потолок метрики)", per_or, 0)

    # 2-3. Настоящие факторы и дистанционка
    variants = [("только дистанционка (рабочая точка)", X),
                ("настоящие 8 факторов формулы",
                 raw[FACTORS].to_numpy(dtype=float)),
                ("5 факторов без разломных",
                 raw[[c for c in FACTORS if c not in FAULT_FACTORS]]
                 .to_numpy(dtype=float)),
                ("только 3 разломных фактора",
                 raw[list(FAULT_FACTORS)].to_numpy(dtype=float))]

    # 4. Вклад каждого фактора поверх дистанционки
    for col in FACTORS:
        variants.append(("дистанционка + %s" % FACTOR_TITLE[col],
                         np.column_stack([X, raw[col].to_numpy(dtype=float)])))

    for name, Xv in variants:
        _, aucs, per_ax = blind_map(Xv, y, pool, cx, cy)
        add(name, per_ax, Xv.shape[1], float(np.mean(aucs)))

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "ceiling_decomposition.csv"
    res.to_csv(out, index=False)
    print()
    print("=== Потолок и его разложение ===")
    print(res.to_string(index=False, float_format=lambda v: "%.4f" % v))

    ref = "только дистанционка (рабочая точка)"
    ta_ref, tc = agree_detrended(maps[ref], crit, pool, valid, D)
    print()
    print("=== Разность с рабочей точкой (блочный бутстрэп 10 км) ===")
    for name in maps:
        if name == ref:
            continue
        ta, _ = agree_detrended(maps[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print("  %-46s %+.3f [%+.3f; %+.3f]"
              % (name, b["разность_средняя"], b["разность_ниж_5"],
                 b["разность_верх_95"]))
    print()
    print("OK: %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
