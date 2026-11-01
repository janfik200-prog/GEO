"""Простые модели сравнения по той же метрике O₁₀ в том же конвейере.

Замечание рецензента: в статье географическое запоминание показано через
ROC-AUC на отдельном наборе признаков, но нет ответа на прямой вопрос —
какое совпадение верхних децилей (O₁₀) дала бы заведомо простая модель,
прогнанная через **тот же** конвейер. Без этого остаточное совпадение
дистанционной модели не с чем сравнить.

Здесь считаются простые модели сравнения полностью той же сборкой, что и
рабочая точка: обучение по восьми отложенным полосам с разрывом 15 км,
квантильная калибровка внутри полосы, сглаживание ядром 5 × 1,25 км на
северо-запад, те же маска валидных ячеек, эталон и определение O₁₀.
Отличается только набор входных столбцов:

* только координаты ячейки (x, y) — чистая география;
* только высота рельефа (``dem_elev``);
* координаты и высота вместе;
* рельеф без спектров и линеаментов (``dem_*`` и ``relief2_*``);
* случайный признак с записанным зерном — якорь случайного уровня.

Строка рабочей точки (38 признаков с маской обнажённости) берётся из
первичной таблицы заморозки и в прогоне не пересчитывается.

Нужны закрытые сетки критериального расчёта: прогон возможен только там, где
они есть. Результат пишется в каталог заморозки, файлы репозитория не
меняются.

Запуск::

    python -m experiments.ipuz_baseline_o10 --freeze <каталог_заморозки>
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import smooth_strips  # noqa: E402

СТОЛБЦЫ = ("вариант", "признаков", "фолды_roc_auc", "совпадение_верх_10",
           "совпадение_без_тренда", "каппа", "карта_roc_auc")


def параметры_модели() -> dict:
    return {"алгоритм": "HistGradientBoostingClassifier",
            "max_iter": config.HGB_MAX_ITER,
            "learning_rate": config.HGB_LEARNING_RATE,
            "max_leaf_nodes": config.HGB_MAX_LEAF_NODES,
            "min_samples_leaf": config.HGB_MIN_SAMPLES_LEAF,
            "l2_regularization": config.HGB_L2,
            "early_stopping": False, "class_weight": "balanced",
            "random_state": config.CRIT_SEED}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    a = ap.parse_args()
    freeze = a.freeze.resolve()

    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    base = feat[cols]
    рельеф = [c for c in cols if c.startswith(("dem_", "relief2_"))]
    print("валидных ячеек: %d, признаков переноса: %d, из них рельефных: %d"
          % (pool.size, len(cols), len(рельеф)))

    варианты = [
        ("только координаты ячейки", np.column_stack([cx, cy])),
        ("только высота рельефа", base[["dem_elev"]].to_numpy(float)),
        ("координаты и высота",
         np.column_stack([cx, cy, base["dem_elev"].to_numpy(float)])),
        ("рельеф без спектров и линеаментов",
         base[рельеф].to_numpy(float)),
        ("случайный признак, зерно 42",
         np.random.default_rng(config.CRIT_SEED)
         .standard_normal((y.size, 1))),
    ]

    строки = []
    for имя, X in варианты:
        t0 = time.time()
        s, aucs, per_ax = blind_map(np.asarray(X, dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        m = score_map(sm, y, crit, pool, valid, D)
        строки.append({"вариант": имя, "признаков": int(np.shape(X)[1]),
                       "фолды_roc_auc": float(np.mean(aucs)), **m})
        print("  %-34s O10 %.4f, без тренда %.4f, ROC карты %.3f  (%.0f с)"
              % (имя, m["совпадение_верх_10"], m["совпадение_без_тренда"],
                 m["карта_roc_auc"], time.time() - t0))

    # Рабочая точка — из первичной таблицы, без пересчёта.
    главная = pd.read_csv(freeze / "04_metrics" / "main_metrics.csv")
    гл = главная[главная["вариант"].str.contains("обнажённых")].iloc[0]
    строки.append({"вариант": "рабочая точка: 38 признаков с маской "
                              "(из заморозки)",
                   "признаков": int(гл["признаков"]),
                   "фолды_roc_auc": float(гл["фолды_roc_auc"]),
                   "совпадение_верх_10": float(гл["совпадение_верх_10"]),
                   "совпадение_без_тренда": float(гл["совпадение_без_тренда"]),
                   "каппа": float(гл["каппа"]),
                   "карта_roc_auc": float(гл["карта_roc_auc"])})

    df = pd.DataFrame(строки)[list(СТОЛБЦЫ)]
    out = freeze / "04_metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "baseline_o10.csv", index=False)

    простые = df[~df["вариант"].str.startswith("рабочая точка")]
    мета = {
        "что_это": "простые модели сравнения в идентичном конвейере рабочей "
                   "точки: те же полосы, калибровка, сглаживание, маска "
                   "валидных ячеек, эталон и определение O₁₀",
        "зачем": "замечание рецензента: контроль географического запоминания "
                 "в статье был только по ROC-AUC на другом наборе признаков; "
                 "здесь простые модели измерены той же метрикой O₁₀",
        "дата_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "валидных_ячеек": int(pool.size),
        "доля_площади_верхнего_дециля": config.CREF_AREA,
        "полос": 8, "разрыв_м": 15000,
        "модель": параметры_модели(),
        "сглаживание": "ядро 5 × 1,25 км, вытянуто на северо-запад, внутри "
                       "каждой полосы отдельно",
        "максимум_простых_без_тренда": float(
            простые["совпадение_без_тренда"].max()),
        "рабочая_точка_без_тренда": float(
            df[df["вариант"].str.startswith("рабочая точка")]
            ["совпадение_без_тренда"].iloc[0]),
        "окружение": {"python": platform.python_version(),
                      "numpy": np.__version__, "pandas": pd.__version__},
    }
    import sklearn
    мета["окружение"]["scikit-learn"] = sklearn.__version__
    (out / "baseline_o10_meta.json").write_text(
        json.dumps(мета, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nлучшая простая модель без тренда: %.4f, рабочая точка: %.4f"
          % (мета["максимум_простых_без_тренда"],
             мета["рабочая_точка_без_тренда"]))
    print("записано:", out / "baseline_o10.csv")


if __name__ == "__main__":
    main()
