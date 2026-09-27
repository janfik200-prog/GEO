"""Диагностика п.1 после `nested_feature_elimination.py`: есть ли сигнал сверх тренда.

Nested-прогон (27.08.2026) показал, что честная (без double dipping) модель на
признаках нигде не превосходит базу «только координаты» (xy-only=0.3235,
1-NN=0.3345) при воспроизведении `prognoz` — при этом полный честный пул из 91
признака сам по себе даёт лишь 0.188. Открытый вопрос: признаки просто не несут
ничего сверх гладкого регионального тренда критериальной формулы, или тренд
маскирует более мелкий, но реальный вклад признаков, который тонет на фоне
тренда при прямой оценке на `prognoz`?

Здесь `prognoz` раскладывается на двух уровнях на каждой из 8 внешних полос
(leave-one-strip-out, буфер 15 км — та же схема, что в nested-прогоне):

1. **тренд** — RF только на координатах (x, y), обучен на внешнем train,
   предсказан на train (для остатка обучающей выборки) и на test (для остатка
   held-out);
2. **остаток** = prognoz - предсказание тренда;
3. модель на всех 91 признаках обучается предсказывать ОСТАТОК train (не сырой
   prognoz) и оценивается на остатке test (честно — то, что не увидено при
   обучении, включая обучение тренд-модели);
4. отдельно проверяется реконструкция (тренд + предсказанный остаток) —
   сверка с уже известным честным числом для K=91 (0.188) как проверка, что
   декомпозиция не сломана.

Здесь без переранжирования/отбора (K=91 фиксирован) — double dipping в этом
скрипте невозможен по конструкции, задача только диагностическая, не замена
основного nested-прогона.

Запуск из корня репозитория: ``python -m experiments.trend_residual_diagnostic``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.feature_batch_elimination import strip_id_and_edges  # noqa: E402
from experiments.nested_feature_elimination import _spearman, fit_rf  # noqa: E402

N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0
SEEDS = [config.CRIT_SEED, 7]


def run_outer_fold(axis_name: str, s: int, coord: np.ndarray, xy: np.ndarray,
                    X: np.ndarray, y: np.ndarray, seed: int) -> dict | None:
    strip_id, edges = strip_id_and_edges(coord, N_STRIPS)
    lo, hi = edges[s], edges[s + 1]
    test_mask = strip_id == s
    train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
    if test_mask.sum() == 0 or train_mask.sum() < 200:
        return None

    # ---- тренд: RF только на координатах ----
    model_trend = fit_rf(xy[train_mask], y[train_mask], seed)
    trend_train = model_trend.predict(xy[train_mask])
    trend_test = model_trend.predict(xy[test_mask])
    rho_trend = _spearman(y[test_mask], trend_test)

    resid_train = y[train_mask] - trend_train
    resid_test = y[test_mask] - trend_test

    # ---- модель на признаках, обученная предсказывать ОСТАТОК ----
    model_resid = fit_rf(X[train_mask], resid_train, seed)
    pred_resid_test = model_resid.predict(X[test_mask])
    rho_resid = _spearman(resid_test, pred_resid_test)

    # ---- сверка: тренд + предсказанный остаток против сырого prognoz ----
    rho_combo = _spearman(y[test_mask], trend_test + pred_resid_test)

    # ---- для сравнения: модель на признаках прямо на сыром prognoz (K=91, известно ~0.188) ----
    model_raw = fit_rf(X[train_mask], y[train_mask], seed)
    rho_raw = _spearman(y[test_mask], model_raw.predict(X[test_mask]))

    print(f"    [{axis_name}{s} seed={seed}] rho_trend={rho_trend:.4f} "
          f"rho_resid={rho_resid:.4f} rho_combo={rho_combo:.4f} rho_raw={rho_raw:.4f} "
          f"(train={train_mask.sum()}, test={test_mask.sum()})", flush=True)

    return {
        "axis": axis_name, "strip": s, "seed": seed,
        "rho_trend": rho_trend, "rho_resid": rho_resid,
        "rho_combo": rho_combo, "rho_raw": rho_raw,
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
    }


def main() -> None:
    t0 = time.time()
    meta, prognoz = criterial_target.load_prognoz_grid()
    y_full = -prognoz.ravel().astype(float)

    feat_df = criterial_target.training_features(dataset="v6")
    pool = [c for c in feat_df.columns if not cell_mask.is_service(c)]
    valid = cell_mask.build_valid_mask(feat_df)
    print(f"Пул признаков: {len(pool)}. Валидных ячеек: {valid.sum()}/{len(valid)}", flush=True)

    X = feat_df[pool].fillna(0).to_numpy()[valid]
    y = y_full[valid]
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel()[valid], cy.ravel()[valid]
    xy = np.column_stack([cx, cy])

    rows = []
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===", flush=True)
        for axis_name, coord in [("X", cx), ("Y", cy)]:
            for s in range(N_STRIPS):
                row = run_outer_fold(axis_name, s, coord, xy, X, y, seed)
                if row is not None:
                    rows.append(row)

    df = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "trend_residual_diagnostic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path}")

    means = df[["rho_trend", "rho_resid", "rho_combo", "rho_raw"]].mean()
    stds = df[["rho_trend", "rho_resid", "rho_combo", "rho_raw"]].std()
    print(f"\nСреднее по {len(df)} внешним фолдам (8 полос x {len(SEEDS)} сида):")
    for name, label in [("rho_trend", "тренд (xy-only, сверка с базой)"),
                         ("rho_resid", "признаки -> ОСТАТОК (ключевой результат)"),
                         ("rho_combo", "тренд+остаток -> сырой prognoz (сверка с K=91=0.188)"),
                         ("rho_raw", "признаки -> сырой prognoz напрямую (сверка)")]:
        print(f"  {label}: {means[name]:.4f} (std {stds[name]:.4f})")

    print("\nПо осям (rho_resid, среднее по сидам):")
    print(df.groupby("axis")["rho_resid"].mean().to_string(float_format=lambda v: f"{v:.4f}"))

    print(f"\nВсего времени: {(time.time()-t0)/60:.1f} мин")


if __name__ == "__main__":
    main()
