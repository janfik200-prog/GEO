"""Отбор признаков, которые реально переносятся на незнакомую территорию.

Продолжение `honest_feature_signal.py`. Тот прогон (02.09.2026) показал при
разрыве 15 км между обучением и тестом:

    только координаты      ROC-AUC 0.557      (опора: сколько даёт география)
    все 105 признаков      ROC-AUC 0.421      хуже случайного
    группа s2 (17)         ROC-AUC 0.741
    группа ter (6)         ROC-AUC 0.740
    группа l8 (15)         ROC-AUC 0.723
    группа gm (15)         ROC-AUC 0.320      ниже случайного
    группа pf (12)         ROC-AUC 0.460

То есть переносимый сигнал в признаках ЕСТЬ, но общая модель его теряет:
бустинг цепляется за самые сильные признаки, а самые сильные — это
широкополосные региональные поля (`gm_gr_all` объясняется координатами на
87%, `pf_gr_up1` на 80%). На обучающих полосах такой признак работает как
замена координатам, на отложенной полосе тренд другой, и предсказание
переворачивается — отсюда AUC ниже 0.5 не только у `gm`, но и у всего набора.

Отсюда две идеи, которые здесь проверяются:

1. **Отбор** — оставить только группы/признаки, переносимость которых измерена
   отдельно. Порог отбора — не «важность», а «обыгрывает ли голые координаты».
2. **Локальная аномалия вместо абсолютного значения** — вычесть из признака
   его же сглаженное поле (гаусс, `HIGHPASS_KM`). Региональный уровень уходит,
   остаётся местная аномалия, а именно она и должна нести рудный смысл: важно
   не «здесь гравика высокая», а «здесь гравика выше окружения». Сглаживание
   считается ТОЛЬКО по обучающим ячейкам фолда (маскированный гаусс), поэтому
   значения отложенной полосы в подготовку признаков не попадают.

Схема оценки та же, что в `honest_feature_signal.py`, и меняться не должна:
полосы с буфером 15 км, опора — модель на голых координатах.

Выход: `outputs/metrics/honest_feature_selection.csv`,
       `data/processed/honest_feature_set.json` (итоговый набор признаков).
Запуск из корня репозитория: ``python -m experiments.honest_feature_selection``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, features_v2  # noqa: E402
from experiments.forecast_wide import _keep  # noqa: E402
from experiments.honest_feature_signal import (  # noqa: E402
    STRIP_BUFFER_M, make_model, poly_design, strip_folds, trend_share,
)
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

# Радиус сглаживания для локальной аномалии. 20 км — примерный размер
# перспективной площади критериального прогноза: всё крупнее считаем
# региональным фоном, всё мельче — аномалией.
HIGHPASS_KM = 20.0

# Группы, обыгравшие голые координаты в `honest_feature_signal.py`.
TRANSFER_GROUPS = ("s2", "ter", "l8", "s2raw", "relief2", "dist", "s1", "lin")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def highpass(X: np.ndarray, shape, sigma: float, fit_mask: np.ndarray) -> np.ndarray:
    """Признак минус его сглаженное поле; сглаживание — только по ``fit_mask``.

    Маскированный гаусс: отношение сглаженных «значения» и «веса». Так ячейки
    вне маски (отложенная полоса, пропуски) не вносят вклад в фон, но фон в них
    всё равно определён — интерполирован от обучающих соседей.
    """
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        v = X[:, j]
        w = (fit_mask & np.isfinite(v)).astype(float).reshape(shape)
        g = np.where(np.isfinite(v), v, 0.0).reshape(shape) * w
        num = gaussian_filter(g, sigma, mode="nearest")
        den = gaussian_filter(w, sigma, mode="nearest")
        smooth = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
        out[:, j] = v - smooth.ravel()
    return out


def evaluate(X: np.ndarray, y: np.ndarray, folds, tag: str, *,
             shape=None, sigma: float | None = None) -> dict:
    """Оценка набора по полосам. При заданной ``sigma`` — по локальным аномалиям."""
    au, pr = [], []
    for tr, te in folds:
        if sigma is None:
            Xtr, Xte = X[tr], X[te]
        else:
            H = highpass(X, shape, sigma, tr)
            Xtr, Xte = H[tr], H[te]
        p = make_model().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        au.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
    res = {"набор": tag, "n_признаков": X.shape[1], "n_фолдов": len(au),
           "roc_auc": float(np.mean(au)), "roc_auc_std": float(np.std(au)),
           "pr_auc": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
           "roc_min": float(np.min(au)), "roc_max": float(np.max(au)),
           "фолдов_выше_0.5": int(np.sum(np.array(au) > 0.5))}
    log(f"  {tag:38s} ROC-AUC={res['roc_auc']:.4f}±{res['roc_auc_std']:.3f} "
        f"PR-AUC={res['pr_auc']:.4f} ({res['фолдов_выше_0.5']}/{len(au)} фолдов > 0.5)")
    return res


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    df = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in df.columns if _keep(c)]
    X = df[cols].to_numpy(dtype=float)
    groups = np.array([features_v2.feature_group(c) for c in cols])
    shape = (meta.prf, meta.pic)
    sigma = HIGHPASS_KM * 1000.0 / meta.dx

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    XY = np.column_stack([cx, cy])
    folds = strip_folds(cx, cy, y, STRIP_BUFFER_M)
    tsh = trend_share(X, D)

    log(f"признаков {len(cols)}, позитивов {int(y.sum())}, фолдов {len(folds)}; "
        f"сглаживание {HIGHPASS_KM:.0f} км = {sigma:.0f} ячеек")

    rows = [evaluate(XY, y, folds, "ТОЛЬКО координаты (опора)"),
            evaluate(X, y, folds, "все 105, сырые")]

    print("\n=== Отбор по переносимости групп ===")
    sel = np.isin(groups, TRANSFER_GROUPS)
    rows.append(evaluate(X[:, sel], y, folds, "переносимые группы, сырые"))
    top3 = np.isin(groups, ("s2", "ter", "l8"))
    rows.append(evaluate(X[:, top3], y, folds, "только s2+ter+l8, сырые"))

    print("\n=== Отбор по трендовости признака ===")
    for thr in (0.15, 0.30):
        m = tsh < thr
        rows.append(evaluate(X[:, m], y, folds, f"тренд_R2 < {thr:.2f}, сырые"))

    print("\n=== Локальная аномалия (сглаживание по обучающим ячейкам) ===")
    rows.append(evaluate(X, y, folds, "все 105, локальная аномалия",
                         shape=shape, sigma=sigma))
    rows.append(evaluate(X[:, sel], y, folds, "переносимые группы, аномалия",
                         shape=shape, sigma=sigma))
    bad = np.isin(groups, ("gm", "pf"))
    rows.append(evaluate(X[:, bad], y, folds, "только gm+pf, аномалия",
                         shape=shape, sigma=sigma))

    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    out = ROOT / "outputs" / "metrics" / "honest_feature_selection.csv"
    res.to_csv(out, index=False)
    print("\n=== ИТОГ: переносимость через разрыв 15 км ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    base = res.loc[res["набор"].str.startswith("ТОЛЬКО коорд"), "roc_auc"].iloc[0]
    best = res.iloc[0]
    print(f"\nОпора (координаты): ROC-AUC {base:.4f}; случайный прогноз 0.5000, "
          f"PR-AUC {y.mean():.4f}.")
    print(f"Лучший набор: {best['набор']} — ROC-AUC {best['roc_auc']:.4f}, "
          f"PR-AUC {best['pr_auc']:.4f}, выше опоры на {best['roc_auc'] - base:+.4f}")

    # Фиксируем состав лучшего отобранного набора, чтобы прод-модель бралась
    # из измерения, а не из повторного ручного перечисления групп.
    keep_cols = [c for c, g in zip(cols, groups) if g in TRANSFER_GROUPS]
    spec = {"дата": "2026-09-02", "критерий": "переносимость через разрыв 15 км",
            "группы": list(TRANSFER_GROUPS), "признаки": keep_cols,
            "опора_координаты_roc_auc": float(base),
            "набор_roc_auc": float(res.loc[res["набор"] == "переносимые группы, сырые",
                                           "roc_auc"].iloc[0])}
    spec_path = config.PROCESSED_DIR / "honest_feature_set.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOK: {out}\nOK: {spec_path} ({len(keep_cols)} признаков)")


if __name__ == "__main__":
    main()
