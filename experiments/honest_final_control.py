"""Решающий контроль: сигнал геологический или это гладкая география.

Зачем понадобился. `honest_feature_service_check.py` показал две вещи сразу.
Хорошую: отобранный набор без служебных столбцов работает даже лучше
(ROC-AUC 0.753 против 0.744), значит на них он не держится. И тревожную: одни
служебные столбцы — «сколько снимков попало в ячейку» и «какая доля пикселей
прошла маску облаков» — сами дают 0.734. Геологии в них нет вовсе; это
границы витков и полос покрытия, то есть плавное поле, заданное географией.
Раз такое поле набирает 0.73, планка «обыграть голые координаты» (0.557)
оказывается слишком низкой: дереву решений сырые X и Y неудобны, а любая
гладкая функция от них — удобна.

Отсюда правильная опора: не координаты, а МАКСИМАЛЬНО СИЛЬНАЯ гладкая
пространственная модель. Здесь это решётка гауссовых базисов
(`RBF_GRID`×`RBF_GRID` центров), которой можно приблизить любое плавное поле
на листе. Она не знает ни одного признака — только «где». Сколько она наберёт
на отложенной полосе, столько География даёт даром, и настоящий вклад
геологии — это превышение над ней, а не над 0.5.

Второй инструмент — локальная аномалия (признак минус его сглаженное поле
20 км, сглаживание только по обучающим ячейкам). Она вырезает из признака всё
крупномасштабное, то есть всё, чем признак может подменять координаты. Если
геологический набор держится и после этого, а служебные столбцы обваливаются,
вопрос закрыт: сигнал местный и предметный.

Выход: `outputs/metrics/honest_final_control.csv`.
Запуск из корня репозитория: ``python -m experiments.honest_final_control``.
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
    STRIP_BUFFER_M, evaluate, strip_folds,
)
from experiments.honest_feature_selection import HIGHPASS_KM  # noqa: E402
from experiments.honest_feature_selection import evaluate as evaluate_hp  # noqa: E402

RBF_GRID = 8  # 8x8 = 64 базиса на лист ~74x76 км, то есть центр каждые ~10 км


def rbf_basis(cx: np.ndarray, cy: np.ndarray, n: int) -> np.ndarray:
    """Решётка гауссовых базисов — как угодно гибкое, но чисто плавное поле."""
    gx = np.linspace(cx.min(), cx.max(), n)
    gy = np.linspace(cy.min(), cy.max(), n)
    sigma = (gx[1] - gx[0])
    cols = []
    for x0 in gx:
        for y0 in gy:
            d2 = (cx - x0) ** 2 + (cy - y0) ** 2
            cols.append(np.exp(-d2 / (2 * sigma ** 2)))
    return np.column_stack(cols)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    geo = [c for c in spec["признаки"] if c in feat.columns]
    service = [c for c in spec.get("служебные_исключены", []) if c in feat.columns]

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    shape = (meta.prf, meta.pic)
    sigma = HIGHPASS_KM * 1000.0 / meta.dx
    folds = strip_folds(cx, cy, y, STRIP_BUFFER_M)

    XY = np.column_stack([cx, cy])
    RB = rbf_basis(cx, cy, RBF_GRID)
    Xg = feat[geo].to_numpy(dtype=float)
    Xs = feat[service].to_numpy(dtype=float)
    print(f"геологических {len(geo)}, служебных {len(service)}, "
          f"базисов {RB.shape[1]} (шаг ~{(cx.max()-cx.min())/(RBF_GRID-1)/1000:.0f} км); "
          f"фолдов {len(folds)}", flush=True)

    print("\n=== Опоры: сколько даёт одна география ===")
    rows = [evaluate(XY, y, folds, "координаты X,Y"),
            evaluate(RB, y, folds, f"гладкое поле, {RB.shape[1]} базисов")]

    print("\n=== Признаки против этих опор ===")
    rows.append(evaluate(Xg, y, folds, f"геологические ({len(geo)}), сырые"))
    rows.append(evaluate(Xs, y, folds, f"служебные ({len(service)}), сырые"))

    print(f"\n=== То же, но без крупного масштаба (аномалия {HIGHPASS_KM:.0f} км) ===")
    rows.append(evaluate_hp(Xg, y, folds, f"геологические ({len(geo)}), аномалия",
                            shape=shape, sigma=sigma))
    rows.append(evaluate_hp(Xs, y, folds, f"служебные ({len(service)}), аномалия",
                            shape=shape, sigma=sigma))

    # Служебные столбцы разной природы, и трактовка зависит от того, какие из них
    # работают. `*_n_obs` — сколько снимков накрыло ячейку, это геометрия витков,
    # чистый артефакт. `*_valid_frac` — доля пикселей, прошедших маску облаков и
    # теней; она зависит от облачности и затенения склонов, то есть опосредованно
    # от рельефа, и предметный смысл у неё всё-таки есть.
    print("\n=== Какие именно служебные столбцы работают ===")
    n_obs = [c for c in service if c.endswith("_n_obs")]
    vfrac = [c for c in service if c.endswith("_valid_frac")]
    rows.append(evaluate(feat[n_obs].to_numpy(dtype=float), y, folds,
                         f"витки съёмки, n_obs ({len(n_obs)})"))
    rows.append(evaluate(feat[vfrac].to_numpy(dtype=float), y, folds,
                         f"доля валидных, valid_frac ({len(vfrac)})"))

    res = pd.DataFrame(rows)
    for col in ("n_признаков", "фолдов_выше_0.5"):
        if col in res.columns:
            res[col] = res[col].astype("Int64")
    res.to_csv(ROOT / "outputs" / "metrics" / "honest_final_control.csv", index=False)
    print("\n=== ИТОГ ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    geo_raw = res.loc[2, "roc_auc"]
    geo_hp = res.loc[4, "roc_auc"]
    srv_hp = res.loc[5, "roc_auc"]
    spatial = res.loc[1, "roc_auc"]
    print(f"\nГеография даром (гладкое поле): {spatial:.4f}. "
          f"Геологические сырые: {geo_raw:.4f} ({geo_raw - spatial:+.4f}).")
    print(f"После вырезания крупного масштаба: геологические {geo_hp:.4f}, "
          f"служебные {srv_hp:.4f}.")
    print("\nЧитать так: собственный вклад признаков — это превышение над гладким "
          "полем, а не над 0.5;")
    print(f"он равен {geo_raw - spatial:+.4f} ROC-AUC. Локальная (мельче "
          f"{HIGHPASS_KM:.0f} км) составляющая признаков даёт {geo_hp:.4f} "
          f"при случайном 0.5.")


if __name__ == "__main__":
    main()
