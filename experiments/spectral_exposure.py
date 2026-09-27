"""Спектральный блок: измеряет ли он породу — и что даёт маска «измерено».

ПОВОД (07.09.2026). Разбор дерева признаков показал два обстоятельства.
Первое: спектральные индексы посчитаны во всех ячейках листа, но открытый
грунт есть только в 17% из них — на остальных индекс всё равно имеет число, и
модель учит на нём растительность, а не породу. Второе: ни в рабочем наборе 58
признаков, ни в наборе переноса 37 нет НИ ОДНОГО канала ASTER — блок,
единственный способный увидеть гидротермально изменённые породы, из моделей
выпал целиком (исключён в ветке 4 из-за контура съёмки на 70% листа).

ЧТО ПРОВЕРЯЕТСЯ. Три независимых вопроса, каждый своим вариантом входа:

1. **Форма спектрального признака.** Отношение соседних каналов (`ast_aloh` и
   прочие) против глубины полосы поглощения после удаления континуума — так
   минеральную полосу меряют в спектроскопии. Отношение растёт и от наклона
   спектра, глубина — только от самой полосы.
2. **Маска «измерено / не измерено».** Спектр оставляется только там, где есть
   открытый грунт, в остальных ячейках ставится пропуск (бустинг обрабатывает
   пропуски сам) плюс отдельный признак-флаг. Гипотеза: пропуск честнее нуля —
   модель перестаёт принимать растительность за минерал.
3. **Стоит ли вообще возвращать ASTER** при его контуре в 70% площади.

СХЕМА ОЦЕНКИ прежняя и единственная допустимая: полосы (4 по X + 4 по Y) с
разрывом 15 км между обучением и тестом. Кроме ROC по фолдам считается то, что
на самом деле является целью проекта, — сплошная карта из предсказаний
отложенных полос и её совпадение с критериальным прогнозом по верхним 10%,
до и после снятия регионального наклона.

Все входы строго дистанционные: съёмки и рельеф, ни одного слоя с карт.

Выход: `outputs/metrics/spectral_exposure.csv`.
Запуск из корня: ``python -m experiments.spectral_exposure``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               make_model, poly_design)
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Доля открытого грунта в ячейке, начиная с которой считаем, что спектр
# действительно измерил породу. 0.20 — та же граница, по которой в дереве
# признаков посчитаны 17% площади.
BARE_MIN = 0.20

# Центры каналов ASTER SWIR, мкм — нужны для линейного континуума.
WL = {4: 1.650, 5: 2.165, 6: 2.205, 7: 2.260, 8: 2.330, 9: 2.395}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def band_depth(b: dict, lo: int, mid: int, hi: int) -> np.ndarray:
    """Глубина полосы поглощения после удаления континуума.

    Континуум — прямая через каналы `lo` и `hi`, глубина — насколько канал
    `mid` ниже неё. 0 — полосы нет, больше — полоса глубже. В отличие от
    отношения каналов не зависит от общего наклона спектра.
    """
    t = (WL[mid] - WL[lo]) / (WL[hi] - WL[lo])
    cont = b[lo] * (1 - t) + b[hi] * t
    return 1.0 - b[mid] / np.where(cont > 0, cont, np.nan)


def aster_blocks() -> dict[str, pd.DataFrame]:
    """Спектральный блок ASTER в двух формах: отношения и глубины полос."""
    a = pd.read_parquet(config.PROCESSED_DIR / "ast_features.parquet")
    inside = a["ast_valid_frac"].to_numpy() > 0          # контур съёмки
    b = {k: a[f"ast_b0{k}"].to_numpy(dtype=float) for k in range(1, 10)}

    ratios = a[["ast_aloh", "ast_kaolin", "ast_alter", "ast_carb",
                "ast_ferric", "ast_ferrous"]].copy()
    depths = pd.DataFrame({
        "ast_d2205": band_depth(b, 5, 6, 7),    # Al-OH: серицит, мусковит
        "ast_d2330": band_depth(b, 7, 8, 9),    # Mg-OH и карбонат: хлорит, доломит
        "ast_d2165": band_depth(b, 4, 5, 7),    # каолинитовый дублет
        "ast_d1650": band_depth(b, 4, 5, 6),    # контроль наклона в SWIR
    })
    for df in (ratios, depths):
        df[~inside] = np.nan                    # вне контура съёмки — пропуск
    return {"отношения": ratios, "глубины": depths}


def exposure() -> tuple[np.ndarray, np.ndarray]:
    """Маска «спектр измерил породу» и она же как признак-флаг."""
    s2 = pd.read_parquet(config.PROCESSED_DIR / "s2_features.parquet")
    bare = s2["s2_bare_frac"].to_numpy(dtype=float)
    return bare >= BARE_MIN, bare


def masked(df: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    """Оставить значения только там, где спектр измерен; иначе пропуск."""
    out = df.copy()
    out[~keep] = np.nan
    return out


def blind_map(X: np.ndarray, y: np.ndarray, pool: np.ndarray, cx: np.ndarray,
              cy: np.ndarray) -> tuple[np.ndarray, list[float], dict]:
    """Сплошная карта из предсказаний отложенных полос + ROC по фолдам.

    Калибровка внутри полосы — квантиль относительно распределения этой же
    модели на её обучающих ячейках, иначе полосы несопоставимы и верхние 10%
    соберутся из одной полосы (как в `honest_forecast_main`).
    """
    in_pool = np.zeros(y.size, dtype=bool)
    in_pool[pool] = True
    per_ax, aucs = {}, []
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        cal = np.full(y.size, np.nan)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            if te.size == 0 or y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            m = make_model().fit(X[tr], y[tr])
            p_tr = np.sort(m.predict_proba(X[tr])[:, 1])
            p_te = m.predict_proba(X[te])[:, 1]
            cal[te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
            aucs.append(roc_auc_score(y[te], p_te))
        per_ax[ax] = cal
    s = mean_axes(per_ax["X"], per_ax["Y"])
    return s, aucs, per_ax


def score_map(s: np.ndarray, y: np.ndarray, crit: np.ndarray, pool: np.ndarray,
              valid: np.ndarray, D: np.ndarray) -> dict:
    """Метрики сплошной карты: совпадение верхних 10% и качество ранжирования."""
    area = config.CREF_AREA
    s = np.where(np.isfinite(s), s, np.nanmin(s[pool]))
    top_m = (s >= np.quantile(s[pool], 1 - area)) & valid
    top_c = crit_reference.reference_mask(crit, pool, area) & valid
    sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
    tpr = (sr >= np.quantile(sr[pool], 1 - area)) & valid
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid
    agr = crit_reference.agreement(s, crit, pool, area)
    return {"совпадение_верх_10": float((top_m & top_c).sum() / top_c.sum()),
            "совпадение_без_тренда": float((tpr & tcr).sum() / tcr.sum()),
            "каппа": float(agr["kappa"]),
            "карта_roc_auc": float(roc_auc_score(y[pool], s[pool])),
            "карта_pr_auc": float(average_precision_score(y[pool], s[pool]))}


def landscape_link(df: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    """Связь каждого спектрального признака с растительностью и уклоном.

    Признак, который меряет минерал, не обязан коррелировать с ландшафтом;
    если корреляция сильная, признак меряет поверхность, а не породу.
    """
    s2 = pd.read_parquet(config.PROCESSED_DIR / "s2_features.parquet")
    d = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    ndvi, slope = s2["s2_ndvi"].to_numpy(), d["dem_slope"].to_numpy()
    rows = []
    for c in df.columns:
        v = df[c].to_numpy(dtype=float)
        m = np.isfinite(v) & np.isfinite(ndvi) & np.isfinite(slope)
        row = {"признак": c, "n_ячеек": int(m.sum()),
               "растительность": sps.spearmanr(v[m], ndvi[m]).statistic,
               "уклон": sps.spearmanr(v[m], slope[m]).statistic}
        mk = m & keep
        row["растительность_на_обнажённых"] = (
            sps.spearmanr(v[mk], ndvi[mk]).statistic if mk.sum() > 100 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)

    base_cols = [c for c in honest_features("перенос") if c in feat.columns]
    base = feat[base_cols]
    blocks = aster_blocks()
    keep, bare = exposure()
    flag = pd.DataFrame({"спектр_измерен": keep.astype(float)})

    log(f"база: {len(base_cols)} дистанционных признаков, позитивов {int(y.sum())}, "
        f"ячеек в работе {pool.size}")
    log(f"открытый грунт >= {BARE_MIN:.0%}: {int(keep.sum())} ячеек "
        f"({keep.mean():.1%}); контур ASTER: "
        f"{int(np.isfinite(blocks['глубины']['ast_d2205']).sum())} ячеек")

    # --- Меряет ли спектральный признак породу или ландшафт ---
    link = pd.concat([landscape_link(blocks["отношения"], keep),
                      landscape_link(blocks["глубины"], keep)], ignore_index=True)
    link.to_csv(ROOT / "outputs" / "metrics" / "spectral_landscape_link.csv",
                index=False)
    print("\n=== Связь спектрального признака с ландшафтом (ранговая) ===")
    print(link.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # --- Варианты входа ---
    s2_spec = [c for c in base_cols
               if c.startswith("s2_") and c not in ("s2_bare_frac",)]
    variants = {
        "база 37, как сейчас": base,
        "база + отношения ASTER": pd.concat([base, blocks["отношения"]], axis=1),
        "база + глубины полос ASTER": pd.concat([base, blocks["глубины"]], axis=1),
        "база + глубины по маске обнажённости": pd.concat(
            [base, masked(blocks["глубины"], keep), flag], axis=1),
        "база с маской на спектрах S2": pd.concat(
            [base.drop(columns=s2_spec), masked(base[s2_spec], keep), flag], axis=1),
        "спектр только измеренный (ASTER+S2 по маске)": pd.concat(
            [base.drop(columns=s2_spec), masked(base[s2_spec], keep),
             masked(blocks["глубины"], keep), flag], axis=1),
        "база + флаг обнажённости (контроль)": pd.concat([base, flag], axis=1),
    }

    rows = []
    for name, df in variants.items():
        X = df.to_numpy(dtype=float)
        s, aucs, _ = blind_map(X, y, pool, cx, cy)
        r = {"вариант": name, "признаков": X.shape[1],
             "фолды_roc_auc": float(np.mean(aucs)),
             "фолды_roc_std": float(np.std(aucs))}
        r.update(score_map(s, y, crit, pool, valid, D))
        rows.append(r)
        log(f"  {name:46s} фолды={r['фолды_roc_auc']:.4f} "
            f"карта={r['карта_roc_auc']:.4f} совпадение={r['совпадение_верх_10']:.3f} "
            f"без тренда={r['совпадение_без_тренда']:.3f}")

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "spectral_exposure.csv"
    res.to_csv(out, index=False)
    print("\n=== ИТОГ: обработка спектра против протокола полос ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nOK: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
