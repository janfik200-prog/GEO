"""Совпадает ли модель, которую мы измеряли, с моделью, которой рисуем карты.

Расхождение, ради которого написан скрипт. Вся диагностика переносимости
(`honest_feature_signal.py`, `honest_feature_selection.py`,
`honest_coverage_variants.py`) считалась ОДИНОЧНЫМ
`HistGradientBoostingClassifier` — он быстрый, и для сравнения наборов
признаков между собой этого достаточно. А прогнозные карты
(`honest_forecast_main.py`, `forecast_wide.py`) строит
`HistBackgroundEnsemble`: восемь моделей, каждая на позитивах плюс случайной
подвыборке `ENS_BG_FRAC` = 70% фона, с усреднением вероятностей.
Гиперпараметры у них одни и те же (`config.HGB_*`), различается только схема
усреднения по фону. Значит цифра 0.7483, которой описан набор переноса,
формально характеризует не ту модель, что стоит за картой.

Отдельная причина проверить именно сейчас: подвыборка фона вводилась под
прежнюю разметку в 385 позитивов, где фон подавлял позитивы в 50 раз. После
перехода на top-`CREF_AREA` (2295 позитивов против 20651 фона, 1:9) смысл
этого приёма мог просто исчезнуть.

Протокол — тот же и без изменений: полосы 4 по X + 4 по Y, разрыв 15 км между
обучением и тестом (`honest_feature_signal.strip_folds`), мишень —
top-`CREF_AREA` критериального прогноза. Сравниваются одиночный бустинг и
ансамбль на обоих рабочих наборах: 58 признаков (основной лист) и 37 (перенос
на смежную территорию).

Что означает исход. Если разница в пределах разброса по фолдам — расхождение
безвредно, измеренные цифры можно относить к прод-картам, и вопрос закрыт.
Если ансамбль хуже — карты надо строить одиночной моделью. Если ансамбль
заметно лучше — переизмерять придётся весь цикл отбора признаков, потому что
отбор шёл по другой модели.

Выход: `outputs/metrics/honest_model_form.csv`.

Запуск из корня репозитория: ``python -m experiments.honest_model_form``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src import config, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (  # noqa: E402
    STRIP_BUFFER_M, log, make_model, strip_folds,
)


def run(X, y, folds, tag: str, factory) -> dict:
    t0 = time.time()
    au, pr = [], []
    for tr, te in folds:
        p = factory().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        au.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
    res = {"вариант": tag, "n_фолдов": len(au),
           "roc_auc": float(np.mean(au)), "roc_auc_std": float(np.std(au)),
           "pr_auc": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
           "roc_min": float(np.min(au)), "roc_max": float(np.max(au)),
           "секунд": round(time.time() - t0, 1)}
    log(f"  {tag:44s} ROC-AUC={res['roc_auc']:.4f}±{res['roc_auc_std']:.3f} "
        f"PR-AUC={res['pr_auc']:.4f}")
    return res


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    df = criterial_target.training_features("v5_rebuilt")
    cx, cy = meta.cell_centers()
    folds = strip_folds(cx.ravel(), cy.ravel(), y, STRIP_BUFFER_M)
    log(f"позитивов {int(y.sum())}, фона {int((y == 0).sum())}, "
        f"фолдов {len(folds)} (полосы + разрыв {STRIP_BUFFER_M/1000:.0f} км)")

    rows = []
    for kind, label in (("основной", "основной лист"), ("перенос", "перенос")):
        feat = honest_features(kind)
        X = df[feat].to_numpy(dtype=float)
        n = len(feat)
        rows.append(run(X, y, folds, f"{n} призн. ({label}): одиночный бустинг",
                        make_model))
        rows.append(run(X, y, folds, f"{n} призн. ({label}): ансамбль по фону (прод)",
                        lambda: HistBackgroundEnsemble(random_state=config.CRIT_SEED)))

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "honest_model_form.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n=== Одиночная модель против ансамбля (полосы + разрыв 15 км) ===")
    print(out[["вариант", "roc_auc", "roc_auc_std", "pr_auc", "секунд"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    for i in (0, 2):
        d = out.loc[i + 1, "roc_auc"] - out.loc[i, "roc_auc"]
        spread = max(out.loc[i, "roc_auc_std"], out.loc[i + 1, "roc_auc_std"])
        verdict = ("в пределах разброса по фолдам" if abs(d) < spread
                   else "БОЛЬШЕ разброса — расхождение значимо")
        print(f"\n{out.loc[i, 'вариант'].split(':')[0]}: ансамбль минус одиночный "
              f"= {d:+.4f} при разбросе ±{spread:.3f} — {verdict}")
    print(f"\nOK: {out_path}")


if __name__ == "__main__":
    main()
