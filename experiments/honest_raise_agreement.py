"""Можно ли поднять совпадение с критериальной картой, не возвращая утечку.

Задача поставлена как «довести совпадение верхних 10% до 90% на одном листе».
Текущее значение при обучении не ближе 15 км от предсказываемой точки — 27.8%
(случайно 10%, по одной географии 14.6%). Прежние 91.9% получались при
случайном разбиении по ячейкам 500 м, где модель копировала ответ у соседа:
модель на одних координатах давала те же 91.9%. Поэтому здесь проверяются
только те способы, которые не возвращают эту утечку.

Что меряется (каждый вариант — сплошной прогноз по полосам с разрывом 15 км,
калибровка по обучающему распределению, среднее схем X и Y — то же, что в
`honest_forecast_main.py`, но одиночным бустингом: его эквивалентность
ансамблю измерена в `honest_model_form.py`, +0.002 при разбросе ±0.09):

1. **58 признаков** — контроль, должен воспроизвести 27.8%.
2. **Сглаживание готового прогноза** (2/5/10 км) — постобработка без
   переобучения. Мишень крупноблочная и плавная, а признаки попиксельные;
   возможно, значительная часть расхождения — просто масштаб.
3. **Многомасштабные признаки** — к каждому признаку добавлены его гауссово
   сглаженные версии на 2/5/10 км. Это не утечка: признаки известны на всей
   территории и в момент применения на смежном листе тоже, сглаживается
   признак, а не мишень.
4. **Регрессия на непрерывный `prognoz`** вместо бинарной мишени
   «верхние 10%» — ранжирование может быть точнее, чем классификация с
   произвольным порогом.
5. **Потолок: 58 + 13 слоёв самой формулы** (`geo`, `geo2`: разломы, фации,
   палеодолины, узлы пересечения) — это заведомая циркулярность, модель
   получает прямые входы критериального расчёта. Смысл варианта не в том,
   чтобы его применять, а в том, чтобы измерить верхнюю границу: если даже с
   прямыми входами формулы прогноз через разрыв 15 км не дотягивает до 90%,
   значит 90% недостижимы в принципе при этом протоколе, и это будет
   измерение, а не мнение.
6. **Только 13 слоёв формулы** — та же граница без наших признаков.

Выход: `outputs/metrics/honest_raise_agreement.csv`.

Запуск из корня репозитория: ``python -m experiments.honest_raise_agreement``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (  # noqa: E402
    N_STRIPS, STRIP_BUFFER_M, make_model,
)
from experiments.honest_forecast_main import strip_ids  # noqa: E402

#: Масштабы сглаживания признаков, км.
SCALES_KM = (2.0, 5.0, 10.0, 20.0)
#: Масштабы сглаживания уже готового прогноза, км.
POST_KM = (5.0, 10.0, 15.0, 20.0, 25.0)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def smooth_field(v: np.ndarray, shape, sigma_cells: float) -> np.ndarray:
    """Гауссово сглаживание с игнорированием пропусков (нормировка по маске)."""
    w = np.isfinite(v).astype(float).reshape(shape)
    g = np.where(np.isfinite(v), v, 0.0).reshape(shape)
    num = gaussian_filter(g * w, sigma_cells, mode="nearest")
    den = gaussian_filter(w, sigma_cells, mode="nearest")
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).ravel()


def blind(X: np.ndarray, y: np.ndarray, coord: np.ndarray, regression: bool):
    """Калиброванный прогноз по полосам одной оси (полоса предсказана извне)."""
    sid, edges = strip_ids(coord)
    cal = np.full(len(y), np.nan)
    for k in range(N_STRIPS):
        te = sid == k
        tr = (coord < edges[k] - STRIP_BUFFER_M) | (coord > edges[k + 1] + STRIP_BUFFER_M)
        if regression:
            m = HistGradientBoostingRegressor(
                max_iter=config.HGB_MAX_ITER, learning_rate=config.HGB_LEARNING_RATE,
                max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
                min_samples_leaf=config.HGB_MIN_SAMPLES_LEAF,
                l2_regularization=config.HGB_L2, early_stopping=False,
                random_state=config.CRIT_SEED).fit(X[tr], y[tr])
            p_te, p_tr = m.predict(X[te]), np.sort(m.predict(X[tr]))
        else:
            m = make_model().fit(X[tr], y[tr])
            p_te = m.predict_proba(X[te])[:, 1]
            p_tr = np.sort(m.predict_proba(X[tr])[:, 1])
        cal[te] = np.searchsorted(p_tr, p_te, side="left") / p_tr.size
    return cal


def score_variant(X, y, cx, cy, regression=False):
    return (blind(X, y, cx, regression) + blind(X, y, cy, regression)) / 2


def metrics(s, y, prognoz, pool, tag) -> dict:
    top_c = prognoz <= np.quantile(prognoz[pool], config.CREF_AREA)
    top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
    res = {"вариант": tag,
           "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
           "roc_auc": roc_auc_score(y[pool], s[pool]),
           "pr_auc": average_precision_score(y[pool], s[pool])}
    log(f"  {tag:44s} совпадение={res['совпадение_верх_10']:.1%} "
        f"ROC-AUC={res['roc_auc']:.4f}")
    return res


def main() -> None:
    meta, prognoz_grid = criterial_target.load_prognoz_grid()
    prognoz = prognoz_grid.ravel()
    y = criterial_target.training_positives(prognoz_grid).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in honest_features() if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    shape = (meta.prf, meta.pic)
    log(f"признаков {len(cols)}, позитивов {int(y.sum())}, пул {pool.size}, "
        f"полосы {N_STRIPS}/ось, разрыв {STRIP_BUFFER_M/1000:.0f} км")

    rows = []

    # 1. Контроль.
    s_base = score_variant(X, y, cx, cy)
    rows.append(metrics(s_base, y, prognoz, pool, "58 признаков (текущий)"))

    # 2. Сглаживание готового прогноза — без переобучения.
    for km in SCALES_KM:
        s = smooth_field(s_base, shape, km * 1000.0 / meta.dx)
        rows.append(metrics(s, y, prognoz, pool, f"тот же прогноз, сглажен {km:.0f} км"))

    # 3. Многомасштабные признаки.
    extra = [X]
    for km in SCALES_KM:
        sig = km * 1000.0 / meta.dx
        extra.append(np.column_stack([smooth_field(X[:, j], shape, sig)
                                      for j in range(X.shape[1])]))
    X_ms = np.column_stack(extra)
    log(f"многомасштабный набор: {X_ms.shape[1]} столбцов")
    s_ms = score_variant(X_ms, y, cx, cy)
    rows.append(metrics(s_ms, y, prognoz, pool,
                        f"{X_ms.shape[1]} признаков (много масштабов)"))
    for km in POST_KM:
        s = smooth_field(s_ms, shape, km * 1000.0 / meta.dx)
        rows.append(metrics(s, y, prognoz, pool,
                            f"много масштабов + сглажен {km:.0f} км"))

    # Координаты как признаки: на одном листе это законно (обучение всё равно
    # через разрыв 15 км), но такой набор НЕПЕРЕНОСИМ — на смежной территории
    # координаты другие, а выученная по ним поверхность останется здешней.
    from experiments.honest_final_control import RBF_GRID, rbf_basis
    X_geo = np.column_stack([X_ms, rbf_basis(cx, cy, RBF_GRID)])
    s_geo = score_variant(X_geo, y, cx, cy)
    rows.append(metrics(s_geo, y, prognoz, pool,
                        "много масштабов + география (НЕПЕРЕНОСИМ)"))
    s_geo_sm = smooth_field(s_geo, shape, 10.0 * 1000.0 / meta.dx)
    rows.append(metrics(s_geo_sm, y, prognoz, pool,
                        "то же + сглажен 10 км (НЕПЕРЕНОСИМ)"))

    # 4. Регрессия на непрерывный prognoz (меньше — перспективнее, поэтому знак).
    s_reg = score_variant(X, -prognoz, cx, cy, regression=True)
    rows.append(metrics(s_reg, y, prognoz, pool, "58 признаков, регрессия на prognoz"))
    s_reg_ms = score_variant(X_ms, -prognoz, cx, cy, regression=True)
    rows.append(metrics(s_reg_ms, y, prognoz, pool, "много масштабов, регрессия"))

    # 5-6. Потолок: слои самой формулы (циркулярно, только для верхней границы).
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    fac = [c for c in raw.columns
           if c not in feat_all.columns and c not in ("row", "col", "x", "y", "mask_svita")]
    F = raw[fac].to_numpy(dtype=float)
    log(f"слои формулы: {len(fac)} — {', '.join(fac)}")
    s_ceil = score_variant(np.column_stack([X, F]), y, cx, cy)
    rows.append(metrics(s_ceil, y, prognoz, pool,
                        f"ПОТОЛОК: 58 + {len(fac)} слоёв формулы"))
    s_fac = score_variant(F, y, cx, cy)
    rows.append(metrics(s_fac, y, prognoz, pool,
                        f"ПОТОЛОК: только {len(fac)} слоёв формулы"))

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "honest_raise_agreement.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n=== Подъём совпадения без возврата утечки (разрыв 15 км) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nСправочно: случайно 10.0%, по одной географии 14.6%, цель 90%.")
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
