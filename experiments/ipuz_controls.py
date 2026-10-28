"""WP04/G6 ТЗ ИПУЗ: четыре контрольных ROC-AUC в основном окружении.

Замороженная таблица `controls.csv` содержит четыре величины — по две схемы
проверки на двух наборах входов:

    только координаты          случайная 5-кратная 0,996839 / полосы 0,539869
    набор v5_rebuilt (135)     случайная 5-кратная 0,997466 / полосы 0,414580

Пересчёт 26.10.2026 делался на временной сборке scikit-learn 1.7.2 и на
наборе `dataset_v6` (105 признаков) — то есть мерил не то, что в таблице.
Здесь считаются ровно те же четыре числа: тот же набор `v5_rebuilt`, те же
две схемы, основное окружение.

Контроль диагностический. Эти ROC-AUC нельзя переносить в таблицу основной
38-признаковой карты: другой набор входов, другая целевая метрика и другой
протокол агрегации (гейт G6).

Запуск::

    python -m experiments.ipuz_controls --out <каталог>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS,  # noqa: E402
                                               STRIP_BUFFER_M, make_model)
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()

# Что показывает замороженная таблица — для строки «разница».
ЗАМОРОЖЕНО = {
    ("координаты", "случайная"): 0.9968391087094173,
    ("координаты", "полосы"): 0.5398692272378223,
    ("признаки_v5_rebuilt(135)", "случайная"): 0.9974655152912424,
    ("признаки_v5_rebuilt(135)", "полосы"): 0.4145797009072901,
}
СХЕМЫ = {"случайная": "случайная 5-кратная по ячейкам",
         "полосы": "полосы 4X+4Y, разрыв 15 км (среднее по фолдам)"}


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def random_split_auc(X: np.ndarray, y: np.ndarray,
                     valid: np.ndarray) -> tuple[float, float, list[float]]:
    """Случайная 5-кратная по ячейкам — контрпример географической памяти."""
    Xp, yp = X[valid], y[valid]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.CRIT_SEED)
    aucs = [roc_auc_score(yp[te], make_model().fit(Xp[tr], yp[tr])
                          .predict_proba(Xp[te])[:, 1])
            for tr, te in skf.split(Xp, yp)]
    return float(np.mean(aucs)), float(np.std(aucs)), [float(a) for a in aucs]


def strip_split_auc(X: np.ndarray, y: np.ndarray, valid: np.ndarray,
                    cx: np.ndarray, cy: np.ndarray
                    ) -> tuple[float, float, list[float]]:
    """Восемь отложенных полос с разрывом 15 км — тот же протокол, что у карты."""
    aucs: list[float] = []
    for coord in (cx, cy):
        sid, edges = strip_ids(coord)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & valid)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & valid)
            if te.size == 0 or y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            m = make_model().fit(X[tr], y[tr])
            aucs.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return float(np.mean(aucs)), float(np.std(aucs)), aucs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="каталог для controls_recount_v2.csv")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    log("пул %d ячеек, позитивов %d" % (int(valid.sum()), int(y[valid].sum())))

    наборы = {
        "координаты": np.column_stack([cx, cy]),
        "признаки_v5_rebuilt(135)": feat.to_numpy(dtype=float),
    }
    log("набор v5_rebuilt: %d признаков" % feat.shape[1])

    rows = []
    for имя, X in наборы.items():
        for ключ, fn in (("случайная", lambda X: random_split_auc(X, y, valid)),
                         ("полосы", lambda X: strip_split_auc(X, y, valid, cx, cy))):
            mean, sd, folds = fn(X)
            frozen = ЗАМОРОЖЕНО[(имя, ключ)]
            log("%-26s %-9s ROC-AUC %.6f ± %.6f  (заморожено %.6f, разница %+.6f)"
                % (имя, ключ, mean, sd, frozen, mean - frozen))
            rows.append({"схема": СХЕМЫ[ключ], "набор": имя, "roc_auc": mean,
                         "sd": sd if ключ == "полосы" else "",
                         "n": len(folds) if ключ == "полосы" else "",
                         "заморожено_27_10": frozen,
                         "разница": round(mean - frozen, 9),
                         "фолды": json.dumps([round(v, 6) for v in folds])})

    df = pd.DataFrame(rows)
    df.to_csv(out / "controls_recount_v2.csv", index=False, encoding="utf-8")
    print()
    print(df[["схема", "набор", "roc_auc", "заморожено_27_10",
              "разница"]].to_string(index=False))
    худшая = df["разница"].abs().max()
    print("\nмаксимальное отклонение от замороженной таблицы: %.2e" % худшая)
    print("записано:", out / "controls_recount_v2.csv")


if __name__ == "__main__":
    main()
