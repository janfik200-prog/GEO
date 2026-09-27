"""Есть ли в признаках сигнал, переносимый за пределы окрестности обучения.

Повод (02.09.2026): прогон `train_test_main_sheet.py` дал ROC-AUC 0.997 при
случайной CV по ячейкам, но проверка показала, что столько же даёт модель на
ОДНИХ КООРДИНАТАХ (0.995) — то есть измерялась пространственная интерполяция,
а не признаки. При разрыве 15 км между обучением и тестом модель на 105
признаках упала до ROC-AUC 0.421 — хуже случайного. Этот скрипт выясняет,
почему, и есть ли вообще переносимый сигнал.

Схема оценки везде одна: полосы (4 по X + 4 по Y) с буфером
`STRIP_BUFFER_M` — ячейки ближе буфера к тестовой полосе выбрасываются из
обучения. Это единственная схема, в которой модель не может подсмотреть ответ
у соседа, а значит единственная, отвечающая на вопрос «переносится ли на новую
территорию». Опорная точка сравнения — модель на голых координатах: любой
набор признаков обязан её обыгрывать, иначе он не несёт ничего сверх географии.

Что проверяется:

1. **Тренд** — доля дисперсии каждого признака, объяснимая плавной поверхностью
   от координат (полином 2-й степени). Гипотеза инверсии: модель учит
   региональный тренд («севернее — перспективнее»), на отложенной полосе тренд
   другой, и предсказание переворачивается.
2. **По группам** — какие группы признаков переносятся, а какие только мешают.
   Считается отдельно по каждой группе: широкополосная региональная гравика и
   локальные трансформанты потенциальных полей ведут себя по-разному, в общей
   куче это не видно.
3. **Детренд** — те же признаки за вычетом плавной поверхности, подогнанной
   ТОЛЬКО на обучающей части фолда (иначе снова утечка). Остаток — локальная
   аномалия, она обязана переноситься лучше сырого значения, если гипотеза
   тренда верна.
4. **Квантильная нормировка** по обучающей части — против ковариантного сдвига
   (одинаковые породы на разных полосах дают разные абсолютные значения
   яркости/поля).

Выход: `outputs/metrics/honest_feature_signal.csv`.
Запуск из корня репозитория: ``python -m experiments.honest_feature_signal``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, features_v2  # noqa: E402
from experiments.forecast_wide import _keep  # noqa: E402

N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config.HGB_MAX_ITER, learning_rate=config.HGB_LEARNING_RATE,
        max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
        min_samples_leaf=config.HGB_MIN_SAMPLES_LEAF,
        l2_regularization=config.HGB_L2, early_stopping=False,
        class_weight="balanced", random_state=config.CRIT_SEED)


def poly_design(cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Полином 2-й степени от координат (нормированных) — база для детренда."""
    x = (cx - cx.mean()) / cx.std()
    y = (cy - cy.mean()) / cy.std()
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def trend_share(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Доля дисперсии каждого признака, объяснимая координатным полиномом (R^2)."""
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        v = X[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 50 or np.nanstd(v[ok]) == 0:
            continue
        beta, *_ = np.linalg.lstsq(D[ok], v[ok], rcond=None)
        resid = v[ok] - D[ok] @ beta
        out[j] = max(0.0, 1.0 - resid.var() / v[ok].var())
    return out


def detrend(Xtr: np.ndarray, Xte: np.ndarray, Dtr: np.ndarray, Dte: np.ndarray):
    """Вычесть координатный тренд; коэффициенты подгоняются ТОЛЬКО на обучении."""
    Rtr, Rte = Xtr.copy(), Xte.copy()
    for j in range(Xtr.shape[1]):
        v = Xtr[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 50 or np.nanstd(v[ok]) == 0:
            continue
        beta, *_ = np.linalg.lstsq(Dtr[ok], v[ok], rcond=None)
        Rtr[:, j] = v - Dtr @ beta
        Rte[:, j] = Xte[:, j] - Dte @ beta
    return Rtr, Rte


def quantile_norm(Xtr: np.ndarray, Xte: np.ndarray):
    """Ранговая нормировка по обучающей части (против ковариантного сдвига).

    ``QuantileTransformer`` пропускает NaN насквозь, заполнять их не нужно.
    """
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=min(1000, Xtr.shape[0]),
                             output_distribution="normal",
                             random_state=config.CRIT_SEED)
    return qt.fit_transform(Xtr), qt.transform(Xte)


def strip_folds(cx: np.ndarray, cy: np.ndarray, y: np.ndarray, buffer_m: float):
    """Список (train_mask, test_mask) по полосам X и Y с буфером."""
    folds = []
    for coord in (cx, cy):
        edges = np.linspace(coord.min(), coord.max(), N_STRIPS + 1)
        sid = np.clip(np.digitize(coord, edges[1:-1]), 0, N_STRIPS - 1)
        for k in range(N_STRIPS):
            te = sid == k
            tr = (coord < edges[k] - buffer_m) | (coord > edges[k + 1] + buffer_m)
            if y[te].sum() >= 5 and y[tr].sum() >= 20:
                folds.append((tr, te))
    return folds


def evaluate(X: np.ndarray, y: np.ndarray, folds, tag: str,
             D: np.ndarray | None = None, mode: str = "raw") -> dict:
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
    res = {"набор": tag, "n_фолдов": len(au),
           "roc_auc": float(np.mean(au)), "roc_auc_std": float(np.std(au)),
           "pr_auc": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
           "roc_min": float(np.min(au)), "roc_max": float(np.max(au))}
    log(f"  {tag:34s} ROC-AUC={res['roc_auc']:.4f}±{res['roc_auc_std']:.3f} "
        f"PR-AUC={res['pr_auc']:.4f}")
    return res


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    df = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in df.columns if _keep(c)]
    X = df[cols].to_numpy(dtype=float)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    XY = np.column_stack([cx, cy])

    folds = strip_folds(cx, cy, y, STRIP_BUFFER_M)
    log(f"признаков {len(cols)}, позитивов {int(y.sum())}, "
        f"фолдов {len(folds)} (полосы + буфер {STRIP_BUFFER_M/1000:.0f} км)")

    # --- 1. Тренд ---
    tr_share = trend_share(X, D)
    groups = np.array([features_v2.feature_group(c) for c in cols])
    tdf = (pd.DataFrame({"признак": cols, "группа": groups, "тренд_R2": tr_share})
           .sort_values("тренд_R2", ascending=False))
    tdf.to_csv(ROOT / "outputs" / "metrics" / "honest_trend_share.csv", index=False)
    print("\n=== Доля дисперсии признака, объяснимая координатами (тренд) ===")
    print(tdf.groupby("группа")["тренд_R2"].agg(["mean", "max", "count"])
          .sort_values("mean", ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nСамые трендовые признаки:")
    print(tdf.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rows = []
    print("\n=== Опорные точки ===")
    rows.append(evaluate(XY, y, folds, "ТОЛЬКО координаты"))
    rows.append(evaluate(X, y, folds, "все признаки, сырые"))

    print("\n=== Средства против тренда и сдвига (все признаки) ===")
    rows.append(evaluate(X, y, folds, "все, детренд по обучению", D=D, mode="detrend"))
    rows.append(evaluate(X, y, folds, "все, квантильная нормировка", mode="qnorm"))

    print("\n=== По группам признаков (сырые) ===")
    for g in sorted(set(groups)):
        idx = np.flatnonzero(groups == g)
        rows.append(evaluate(X[:, idx], y, folds, f"группа {g} ({idx.size})"))

    res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    out = ROOT / "outputs" / "metrics" / "honest_feature_signal.csv"
    res.to_csv(out, index=False)
    print("\n=== ИТОГ: что переносится через разрыв 15 км ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    base = res.loc[res["набор"] == "ТОЛЬКО координаты", "roc_auc"].iloc[0]
    print(f"\nОпора — координаты: ROC-AUC {base:.4f}. Случайный прогноз: 0.5000, "
          f"PR-AUC {y.mean():.4f}.")
    better = res[(res["roc_auc"] > max(base, 0.5)) & (res["набор"] != "ТОЛЬКО координаты")]
    print("Обыгрывают и координаты, и случайность:")
    print(better[["набор", "roc_auc", "pr_auc"]].to_string(index=False,
          float_format=lambda v: f"{v:.4f}") if len(better) else "  НИ ОДИН НАБОР")
    print(f"\nOK: {out}")


if __name__ == "__main__":
    main()
