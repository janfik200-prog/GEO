"""Переносимы ли градиентные трансформанты гравики и магнитки (в отличие от полей).

ПОВОД. Группы `gm` и `pf` исключены из всех рабочих наборов признаков, потому
что на протоколе полос с разрывом 15 км дают перевёрнутый прогноз: `gm` целиком
ROC-AUC 0.319, локальная аномалия «поле минус сглаживание 20 км» — 0.361
(`honest_feature_signal.csv`, `honest_feature_selection.csv`). Объяснение —
региональный тренд: `gm_gr_all` объясняется координатами на 87%, `pf_gr_up1` на
80%, и бустинг берёт поле вместо координаты.

Но замер делался по группе целиком, а группа неоднородна. В ней есть величины,
которые по построению не несут регионального уровня:

* горизонтальные производные `1GX`, `1GY`, их модуль `1G` и направление `1GFI`,
  вторая производная `2G`;
* аналитический сигнал `asig` (модуль полного градиента), тилт-угол `tdr`
  (нормированная производная), вертикальная производная `vz`;
* остаточные поля после фильтрации (`ost_35`, `flt_35`).

И есть противоположное — сами поля (`gm_gr_all`, `gm_mg_all`) и пересчёт вверх
(`up1`...`up10`), который региональный уровень как раз усиливает.

Этот прогон меряет переносимость подгрупп по отдельности на том же протоколе,
что и все остальные наборы: 4 полосы по X + 4 по Y, разрыв 15 км, тот же
бустинг, та же мишень (верхние 10% критериального прогноза). Дополнительно —
практический вопрос: даёт ли добавление переносимых трансформант что-нибудь
поверх набора из 37 признаков, на котором обучается лучшая модель проекта.

Выход: `outputs/metrics/gm_gradient_transferability.csv`,
       `outputs/metrics/gm_gradient_trend.csv`.
Запуск из корня: ``python -m experiments.gm_gradient_transferability``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target  # noqa: E402
from experiments.honest_feature_signal import (STRIP_BUFFER_M, detrend,  # noqa: E402
                                               log, make_model, poly_design,
                                               quantile_norm, strip_folds,
                                               trend_share)

GRAD_GR = ["gm_gr_1GX_25", "gm_gr_1GY_25", "gm_gr_1G_25", "gm_gr_1GFI_25", "gm_gr_2G_25"]
GRAD_MAG = ["gm_mag_1GX_25", "gm_mag_1GY_25", "gm_mag_1G_25", "gm_mag_1GFI_25",
            "gm_mag_2G_25"]
EDGE = ["pf_gr_asig", "pf_gr_tdr", "pf_gr_vz", "pf_mag_asig", "pf_mag_tdr", "pf_mag_vz"]
RESID = ["gm_gr_ost_35", "gm_mag_ost_35", "gm_mag_flt_35"]
LEVEL = ["gm_gr_all", "gm_mg_all", "pf_gr_up1", "pf_mag_up1", "pf_mag_up2",
         "pf_mag_up5", "pf_mag_up10"]
LOCAL = GRAD_GR + GRAD_MAG + EDGE + RESID

SUBGROUPS = {
    "уровни полей и пересчёт вверх": LEVEL,
    "градиенты гравики": GRAD_GR,
    "градиенты магнитки": GRAD_MAG,
    "градиенты обеих съёмок": GRAD_GR + GRAD_MAG,
    "аналит. сигнал, тилт, верт. производная": EDGE,
    "остаточные поля после фильтрации": RESID,
    "ВСЁ локальное (градиенты+края+остатки)": LOCAL,
}


def evaluate(X: np.ndarray, y: np.ndarray, folds, tag: str,
             D: np.ndarray | None = None, mode: str = "raw"):
    """То же, что в `honest_feature_signal`, плюс счёт полос выше случайного."""
    au, pr = [], []
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        if mode == "detrend":
            Xtr, Xte = detrend(Xtr, Xte, D[tr], D[te])
        elif mode == "qnorm":
            Xtr, Xte = quantile_norm(Xtr, Xte)
        p = make_model().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        au.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
    au = np.array(au)
    row = {"набор": tag, "n_признаков": X.shape[1], "roc_auc": float(au.mean()),
           "roc_auc_std": float(au.std()), "pr_auc": float(np.mean(pr)),
           "roc_min": float(au.min()), "roc_max": float(au.max()),
           "полос_выше_0.5": f"{int((au > 0.5).sum())}/{len(au)}"}
    log(f"  {tag:42s} ROC-AUC={row['roc_auc']:.4f}±{row['roc_auc_std']:.3f} "
        f"({row['полос_выше_0.5']})")
    return row, au


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    df = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    keep37 = [c for c in spec["признаки_перенос"] if c in df.columns]
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    XY = np.column_stack([cx, cy])
    folds = strip_folds(cx, cy, y, STRIP_BUFFER_M)
    log(f"позитивов {int(y.sum())}, фолдов {len(folds)} "
        f"(полосы + разрыв {STRIP_BUFFER_M / 1000:.0f} км)")

    # --- трендовость подгрупп: гипотеза в том, что у градиентов её нет ---
    allcols = sorted(set(LEVEL + LOCAL))
    tr = trend_share(df[allcols].to_numpy(dtype=float), D)
    role = {c: n for n, cols in SUBGROUPS.items() for c in cols
            if n != "ВСЁ локальное (градиенты+края+остатки)"}
    tdf = pd.DataFrame({"признак": allcols, "подгруппа": [role[c] for c in allcols],
                        "тренд_R2": tr}).sort_values("тренд_R2", ascending=False)
    tdf.to_csv(ROOT / "outputs" / "metrics" / "gm_gradient_trend.csv", index=False)
    print("\n=== Доля дисперсии, объяснимая координатами (тренд) ===")
    print(tdf.groupby("подгруппа")["тренд_R2"]
          .agg(["mean", "max", "count"]).sort_values("mean", ascending=False)
          .to_string(float_format=lambda v: f"{v:.3f}"))

    rows, per_fold = [], {}
    print("\n=== Планки ===")
    for tag, cols in (("ТОЛЬКО координаты (планка)", None),
                      ("переносимый набор 37 (планка)", keep37)):
        Xv = XY if cols is None else df[cols].to_numpy(dtype=float)
        r, au = evaluate(Xv, y, folds, tag)
        rows.append(r)
        per_fold[tag] = au

    print("\n=== Подгруппы гравики и магнитки, сырые значения ===")
    for name, cols in SUBGROUPS.items():
        r, au = evaluate(df[cols].to_numpy(dtype=float), y, folds, name)
        rows.append(r)
        per_fold[name] = au

    print("\n=== Средства против тренда и сдвига для локальных трансформант ===")
    Xloc = df[LOCAL].to_numpy(dtype=float)
    for tag, mode in (("локальное, детренд по обучению", "detrend"),
                      ("локальное, квантильная нормировка", "qnorm")):
        r, au = evaluate(Xloc, y, folds, tag, D=D, mode=mode)
        rows.append(r)
        per_fold[tag] = au

    print("\n=== Практический вопрос: даёт ли это что-то поверх 37 признаков ===")
    for tag, cols in (("37 + градиенты обеих съёмок", keep37 + GRAD_GR + GRAD_MAG),
                      ("37 + всё локальное", keep37 + LOCAL)):
        r, au = evaluate(df[cols].to_numpy(dtype=float), y, folds, tag)
        rows.append(r)
        per_fold[tag] = au

    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    res.to_csv(ROOT / "outputs" / "metrics" / "gm_gradient_transferability.csv",
               index=False)
    print("\n=== ИТОГ: переносимость через разрыв 15 км ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Парные сравнения по полосам ===")
    pairs = [("градиенты обеих съёмок", "ТОЛЬКО координаты (планка)"),
             ("ВСЁ локальное (градиенты+края+остатки)", "ТОЛЬКО координаты (планка)"),
             ("градиенты обеих съёмок", "уровни полей и пересчёт вверх"),
             ("37 + градиенты обеих съёмок", "переносимый набор 37 (планка)"),
             ("37 + всё локальное", "переносимый набор 37 (планка)")]
    for a, b in pairs:
        d = per_fold[a] - per_fold[b]
        print(f"{a} − {b}: {d.mean():+.4f}, побед {(d > 0).sum()}/{len(d)}, "
              f"p={sps.wilcoxon(d).pvalue:.4f}")
    print("\nНапоминание: ROC-AUC ниже 0.5 на полосах — это не «нет сигнала», "
          "а систематически перевёрнутый прогноз.")


if __name__ == "__main__":
    main()
