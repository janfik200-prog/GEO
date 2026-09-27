"""Прогноз, а не копия формулы: оценка по точкам заверки, а не по согласию.

Цель уточнена 02.09.2026: нужен прогноз, а не воспроизведение критериального
метода. Это меняет метрику. Совпадение верхних 10% с критериальной картой
измеряет ПОХОЖЕСТЬ на старый метод: 100% там означало бы копию, ради которой
машинное обучение не нужно — формулу можно просто пересчитать. Ориентиром
согласие остаётся (если карта расходится с формулой полностью, это повод
разбираться), но целевой метрикой быть перестаёт.

Целевая метрика — попадание в точки заверки. Протокол взят готовым
(`src/assessment.run_protocol`) именно потому, что сравнение методов имеет
смысл только когда выполнено одним и тем же кодом: P-A кривые, нулевая
модель, бутстрэп по кластерам маршрутов и вердикт с поправкой на число
опробованных конфигураций.

Ограничение метрики, которое нельзя забывать при чтении результата. Точек
мало, и они косвенные: строго несмещённое подмножество — только коренные
пробы и проявления U/TR (`config.VER_UNBIASED_CODES`), шлихи Au в него не
входят (шлиховой ореол указывает на источник выше по потоку, а не на ячейку).
Точки собраны маршрутами, поэтому независимая единица — кластер маршрута, а
не точка; отсюда широкие доверительные интервалы. Заверить прогноз по ним
можно только в форме «не хуже критериального» — заявлять превосходство на
таком числе независимых единиц нельзя.

Сравниваются карты, построенные одинаково — сплошной прогноз по полосам с
разрывом 15 км, калибровка по обучающему распределению, среднее схем X и Y:

* `база_58` — рабочий набор переносимых признаков;
* `масштабы_232` — те же 58 плюс их сглаженные версии на 2/5/10 км;
* `масштабы_сглаж_10км` — то же, с постобработкой прогноза на 10 км (лучший
  вариант по согласию: 44.1% против 27.9% у базы, `honest_raise_agreement.py`);
* `criterial` — критериальный прогноз как база сравнения.

Выход: `outputs/metrics/honest_points_verdict.csv`,
`outputs/metrics/honest_points_pa.csv`.

Запуск из корня репозитория: ``python -m experiments.honest_forecast_points``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, criterial_target  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_raise_agreement import (  # noqa: E402
    score_variant, smooth_field,
)

#: Масштабы сглаживания признаков, км (20 км проверен и хуже — см. raise_agreement).
SCALES_KM = (2.0, 5.0, 10.0)
#: Постобработка прогноза, км.
POST_KM = 10.0

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    meta, prognoz_grid = criterial_target.load_prognoz_grid()
    prognoz = prognoz_grid.ravel()
    y = criterial_target.training_positives(prognoz_grid).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in honest_features() if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(feat_all)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    shape = (meta.prf, meta.pic)
    log(f"признаков {len(cols)}, позитивов {int(y.sum())}, пул {int(valid.sum())}")

    s_base = score_variant(X, y, cx, cy)
    log("база 58 посчитана")

    parts = [X]
    for km in SCALES_KM:
        sig = km * 1000.0 / meta.dx
        parts.append(np.column_stack([smooth_field(X[:, j], shape, sig)
                                      for j in range(X.shape[1])]))
    X_ms = np.column_stack(parts)
    s_ms = score_variant(X_ms, y, cx, cy)
    s_sm = smooth_field(s_ms, shape, POST_KM * 1000.0 / meta.dx)
    log(f"многомасштабный набор {X_ms.shape[1]} столбцов посчитан")

    scores = {"база_58": s_base, "масштабы_232": s_ms,
              "масштабы_сглаж_10км": s_sm, "criterial": -prognoz}
    own = ["база_58", "масштабы_232", "масштабы_сглаж_10км"]
    # Поправка на число опробованных карт: заявка на «лучше базы» проверяется
    # трижды, поэтому порог делится на три.
    alpha = 0.05 / len(own)
    prot = assessment.run_protocol(scores, own, meta, valid, alpha,
                                   baseline="criterial", log=log)

    print("\n=== Заверка точками: прогноз против критериального ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "honest_points_verdict.csv",
                           index=False, encoding="utf-8-sig")
    prot["pa"].to_csv(ROOT / "outputs" / "metrics" / "honest_points_pa.csv",
                      index=False, encoding="utf-8-sig")
    print(f"\nпорог значимости с поправкой на {len(own)} карты: {alpha:.4f}")
    print("OK: outputs/metrics/honest_points_verdict.csv")


if __name__ == "__main__":
    main()
