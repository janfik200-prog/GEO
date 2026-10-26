"""Пересчёт трёх величин, у которых не было первичного файла (ТЗ, EVIDENCE GATE).

1. ROC-AUC при СЛУЧАЙНОМ разбиении ячеек, только координаты (в отчёте 0,995);
2. то же на расширенном наборе признаков (в отчёте 0,997);
3. контроль случайной маской той же доли (в отчёте 0,328).

Пишет только в каталог заморозки. Файлы репозитория не трогаются, в статью
результат не вносится.

Запуск::

    python -m experiments.ipuz_recount_controls <каталог_заморозки>
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, poly_design  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import smooth_strips  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def random_split_auc(X: np.ndarray, y: np.ndarray, pool: np.ndarray,
                     n_splits: int = 5) -> tuple[float, float, list[float]]:
    """ROC-AUC при случайном разбиении ячеек — диагностический контрпример."""
    Xp, yp = X[pool], y[pool]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=config.CRIT_SEED)
    aucs = []
    for tr, te in skf.split(Xp, yp):
        m = make_model().fit(Xp[tr], yp[tr])
        aucs.append(roc_auc_score(yp[te], m.predict_proba(Xp[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), [float(a) for a in aucs]


def main() -> None:
    freeze = Path(sys.argv[1])
    out_dir = freeze / "04_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool_idx = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    log("ячеек в пуле %d, позитивов %d" % (valid.sum(), int(y[valid].sum())))

    rows = []

    # --- 1-2. Случайное разбиение -----------------------------------------
    coords = np.column_stack([cx, cy])
    mean_c, std_c, folds_c = random_split_auc(coords, y, valid)
    log("случайное разбиение, только координаты: ROC-AUC %.4f ± %.4f"
        % (mean_c, std_c))
    rows.append({"величина": "ROC-AUC, случайное разбиение, только координаты",
                 "в_отчёте": 0.995, "пересчёт": round(mean_c, 6),
                 "разброс_по_фолдам": round(std_c, 6), "признаков": 2,
                 "фолды": json.dumps([round(a, 4) for a in folds_c])})

    d6 = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    wide_cols = [c for c in d6.columns if c not in ("row", "col", "x", "y")]
    Xw = d6[wide_cols].to_numpy(dtype=float)
    mean_w, std_w, folds_w = random_split_auc(Xw, y, valid)
    log("случайное разбиение, %d признаков: ROC-AUC %.4f ± %.4f"
        % (len(wide_cols), mean_w, std_w))
    rows.append({"величина": "ROC-AUC, случайное разбиение, расширенный набор",
                 "в_отчёте": 0.997, "пересчёт": round(mean_w, 6),
                 "разброс_по_фолдам": round(std_w, 6),
                 "признаков": len(wide_cols),
                 "фолды": json.dumps([round(a, 4) for a in folds_w])})

    # --- 3. Контроль случайной маской --------------------------------------
    cols = [c for c in honest_features("перенос") if c in feat.columns]
    base = feat[cols]
    keep = spectral_mask.exposure_mask(base)
    share = float(keep.mean())
    spec = spectral_mask.spectral_columns(base.columns)
    log("доля обнажённых %.4f, спектральных столбцов %d" % (share, len(spec)))

    def run(mask: np.ndarray, tag: str) -> float:
        work = base.copy()
        work.loc[~mask, spec] = np.nan
        work[spectral_mask.FLAG_COL] = mask.astype(float)
        X = work.to_numpy(dtype=float)
        _, _, per_ax = blind_map(X, y, pool_idx, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        m = score_map(sm, y, crit, pool_idx, valid, D)
        log("  %-44s без тренда=%.4f" % (tag, m["совпадение_без_тренда"]))
        return float(m["совпадение_без_тренда"])

    real = run(keep, "маска обнажённости (порог 0,02)")
    rnd_vals = []
    for seed in (11, 23, 47):
        rng = np.random.default_rng(seed)
        rm = np.zeros(len(base), dtype=bool)
        rm[rng.choice(len(base), size=int(round(share * len(base))),
                      replace=False)] = True
        rnd_vals.append(run(rm, "случайная маска, зерно %d" % seed))

    rnd_mean = float(np.mean(rnd_vals))
    rows.append({"величина": "O10 без тренда, маска обнажённости",
                 "в_отчёте": 0.362287, "пересчёт": round(real, 6),
                 "разброс_по_фолдам": "", "признаков": len(cols) + 1,
                 "фолды": ""})
    rows.append({"величина": "O10 без тренда, случайная маска (среднее 3 зёрен)",
                 "в_отчёте": 0.328, "пересчёт": round(rnd_mean, 6),
                 "разброс_по_фолдам": round(float(np.std(rnd_vals)), 6),
                 "признаков": len(cols) + 1,
                 "фолды": json.dumps([round(v, 4) for v in rnd_vals])})

    df = pd.DataFrame(rows)
    df["разница"] = [round(float(r["пересчёт"]) - float(r["в_отчёте"]), 6)
                     for r in rows]
    df.to_csv(out_dir / "controls_recount.csv", index=False)

    log("")
    log("ИТОГ")
    print(df[["величина", "в_отчёте", "пересчёт", "разница"]].to_string(index=False))
    print()
    print("прибавка географии обнажённости сверх случайной маски: %+.4f"
          % (real - rnd_mean))
    print("записано:", out_dir / "controls_recount.csv")


if __name__ == "__main__":
    main()
