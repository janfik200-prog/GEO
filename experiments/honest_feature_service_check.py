"""Не держится ли переносимый сигнал на служебных признаках съёмки.

В набор, отобранный `honest_feature_selection.py` по переносимости, попали
шесть служебных столбцов — `s2_n_obs`, `s2_valid_frac`, `s1_n_obs`,
`s1_valid_frac`, `l8_n_obs`, `l8_valid_frac`. Это не геология, а свойства
самой съёмки: сколько снимков попало в ячейку и какая доля пикселей прошла
маску облаков. Опасность конкретная: границы витков и полос покрытия — это
география в чистом виде, `s1_n_obs` объясняется координатами на 64%,
`s2_n_obs` на 48%. Если переносимость набора держится на них, то честной она
только выглядит.

Проверка прямая: тот же протокол (полосы + буфер 15 км, опора — голые
координаты), три набора — отобранный целиком, он же без служебных, и одни
служебные. Если без служебных результат сохраняется, а на одних служебных
сигнала нет, набор геологический.

Выход: `outputs/metrics/honest_service_check.csv`; при подтверждении
обновляет `data/processed/honest_feature_set.json`.
Запуск из корня репозитория: ``python -m experiments.honest_feature_service_check``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.honest_feature_signal import (  # noqa: E402
    STRIP_BUFFER_M, evaluate, poly_design, strip_folds,
)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    spec_path = config.PROCESSED_DIR / "honest_feature_set.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cols = [c for c in spec["признаки"] if c in feat.columns]

    service = [c for c in cols if cell_mask.is_service(c)]
    geo = [c for c in cols if not cell_mask.is_service(c)]
    X = feat[cols].to_numpy(dtype=float)
    Xg = feat[geo].to_numpy(dtype=float)
    Xs = feat[service].to_numpy(dtype=float)

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    XY = np.column_stack([cx, cy])
    folds = strip_folds(cx, cy, y, STRIP_BUFFER_M)
    print(f"отобрано {len(cols)}: геологических {len(geo)}, служебных "
          f"{len(service)} ({', '.join(service)}); фолдов {len(folds)}", flush=True)

    rows = [evaluate(XY, y, folds, "ТОЛЬКО координаты (опора)"),
            evaluate(X, y, folds, f"отобранный набор ({len(cols)})"),
            evaluate(Xg, y, folds, f"он же без служебных ({len(geo)})"),
            evaluate(Xs, y, folds, f"ТОЛЬКО служебные ({len(service)})")]
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "honest_service_check.csv", index=False)
    print("\n=== Держится ли сигнал без служебных признаков ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    base = res.loc[0, "roc_auc"]
    full, without, only = res.loc[1, "roc_auc"], res.loc[2, "roc_auc"], res.loc[3, "roc_auc"]
    print(f"\nОпора (координаты) {base:.4f}; случайный прогноз 0.5000.")
    print(f"Потеря от снятия служебных: {without - full:+.4f}; "
          f"одни служебные дают {only:.4f}.")
    if without > base and without > 0.5:
        spec["признаки"] = geo
        spec["служебные_исключены"] = service
        spec["набор_roc_auc"] = float(without)
        spec["комментарий"] = ("служебные столбцы съёмки исключены: это география "
                               "покрытия, а не геология; сигнал без них сохраняется")
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\nНабор обновлён: {len(geo)} геологических признаков -> {spec_path}")
    else:
        print("\nНабор НЕ обновлён: без служебных признаков переносимость теряется — "
              "это и есть содержательный отрицательный результат.")


if __name__ == "__main__":
    main()
