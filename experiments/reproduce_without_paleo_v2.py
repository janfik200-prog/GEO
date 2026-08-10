"""Задача 5, вариант 2: подставить воспроизведённый фактор палеодолин (задача 6)
в задачу восстановления прогноза без paleo (задача 5), вместо/вместе с сырыми
признаками полей и снимков.

ИДЕЯ (по итогам обсуждения). Задача 6 показала, что фактор «долины и впадины»
неплохо воспроизводится (AUC 0.90) — но ПОЛНОЙ моделью, которая использует
геологию и минерагению в придачу к полям/снимкам. Задача 5 по постановке не
имеет права использовать геологические факторные слои (иначе это дублирование
пяти факторов, что уже остаются в формуле-без-палео, — см. докстринг
``reproduce_without_paleo.py``). Вопрос: если воспроизвести фактор палеодолин
ЧЕСТНО В ТЕХ ЖЕ ГРАНИЦАХ, что разрешены задаче 5 (только поля+снимки, БЕЗ
геологии/минерагении), а затем подставить эту реконструкцию как признак в
регрессию невязки задачи 5 — улучшится ли объяснение невязки по сравнению с
сырыми 138 признаками поодиночке?

МЕТОДИЧЕСКИ ЭТО НЕ ОБХОД ОГРАНИЧЕНИЯ: реконструкция паледолин здесь считается
на том же признаковом пуле (``SAT_PF_GROUPS`` из ``reproduce_without_paleo.py``,
без geo/geo2/mingeo), честной OOF-блочной CV — то есть это просто НЕЛИНЕЙНОЕ
СЖАТИЕ тех же 138 признаков в один производный, обученное на честно
пространственно-раздельной цели (палеодолина), а не подсказка от геологии.

Запуск из корня: ``python -X utf8 -m experiments.reproduce_without_paleo_v2``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target, features, features_v2  # noqa: E402

SAT_PF_GROUPS = ("gm", "pf", "ls", "s2", "s2raw", "s1", "psr", "ast", "astir", "l8", "opt")
CV_SEEDS = (1, 7, 13, 21, 42)
AREAS = (0.05, 0.10, 0.20)


def _is_technical(c: str) -> bool:
    return c.endswith("_n_obs") or "valid_frac" in c


def _pct_rank(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    r = np.full(x.shape, np.nan)
    r[pool] = pd.Series(x[pool]).rank(pct=True).to_numpy()
    return r


def _topk_overlap(score_a: np.ndarray, score_b: np.ndarray, pool: np.ndarray, area: float) -> float:
    a, b = score_a[pool], score_b[pool]
    top_a = a >= np.quantile(a, 1 - area)
    top_b = b >= np.quantile(b, 1 - area)
    return float((top_a & top_b).sum() / top_a.sum())


def _oof(model_cls, X: np.ndarray, y: np.ndarray, blocks: np.ndarray, seeds, **kw) -> np.ndarray:
    gkf = GroupKFold(n_splits=5)
    stack = np.zeros((len(seeds), y.size))
    for si, seed in enumerate(seeds):
        for tr, te in gkf.split(X, y, blocks):
            model = model_cls(random_state=seed, n_jobs=-1, **kw)
            model.fit(X[tr], y[tr])
            pred = model.predict_proba(X[te])[:, 1] if hasattr(model, "predict_proba") else model.predict(X[te])
            stack[si, te] = pred
    return stack.mean(axis=0)


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz.ravel())
    print(f"ячеек {len(df)}, валидных {pool.size}")

    sat_cols = [c for c in df.columns
                if features_v2.feature_group(c) in SAT_PF_GROUPS and not _is_technical(c)]
    print(f"признаков полей+снимков (без geo/mingeo): {len(sat_cols)}")
    X_sat = df[sat_cols].fillna(0).to_numpy(dtype=float)

    x_c, y_c = df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    bx = np.floor((x_c - x_c.min()) / config.VAL_BLOCK_SIZE).astype(int)
    by = np.floor((y_c - y_c.min()) / config.VAL_BLOCK_SIZE).astype(int)
    blocks = bx * (by.max() + 1) + by

    # --- 1. Честная реконструкция палеодолин ТОЛЬКО по полям+снимкам (без геологии) ---
    y_paleo_full = (df["dist_paleo"].to_numpy(dtype=float) == 0).astype(int)
    Xp, yp, bp = X_sat[pool], y_paleo_full[pool], blocks[pool]
    rf_kw = dict(n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
                 min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                 class_weight="balanced")
    paleo_hat_pool = _oof(RandomForestClassifier, Xp, yp, bp, CV_SEEDS, **rf_kw)
    auc_sat_only = roc_auc_score(yp, paleo_hat_pool)
    print(f"\n--- 1. Реконструкция палеодолин ТОЛЬКО полями/снимками (без геологии/минерагении) ---")
    print(f"OOF AUC = {auc_sat_only:.4f} (для сравнения: полная модель задачи 6 с геологией = 0.8965, "
          f"только рельеф = 0.7206)")

    paleo_hat = np.full(len(df), np.nan)
    paleo_hat[pool] = paleo_hat_pool

    # --- 2. Цена вопроса и цель для ML — как в задаче 5 ---
    dist_by_role = {role: df[f"dist_{role}"].to_numpy(dtype=float) for role in config.TAXONOMY_TRANSFORMS}
    score_full = -features.taxonomy_weighted_distance(dist_by_role)
    score_no_paleo = -features.taxonomy_weighted_distance(dist_by_role, exclude={"paleo"})
    r_crit = _pct_rank(crit, pool)
    r_no_paleo = _pct_rank(score_no_paleo, pool)
    residual = np.zeros_like(crit)
    residual[pool] = r_crit[pool] - r_no_paleo[pool]

    # --- 3. baseline (сырые 138 признаков) — воспроизведено для честного сравнения ---
    rf_reg_kw = dict(n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
                      min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT)
    yr = residual[pool]
    oof_baseline = _oof(RandomForestRegressor, Xp, yr, bp, CV_SEEDS, **rf_reg_kw)
    rho_base, p_base = stats.spearmanr(oof_baseline, yr)
    print(f"\n--- 2. Baseline (сырые {len(sat_cols)} признаков, без производного паледолинного) ---")
    print(f"Spearman(OOF-прогноз невязки, факт) = {rho_base:.4f} (p={p_base:.2e})")

    # --- 4. + производный признак paleo_hat (реконструкция задачи 6, честная OOF) ---
    X_aug = np.column_stack([Xp, paleo_hat_pool])
    oof_aug = _oof(RandomForestRegressor, X_aug, yr, bp, CV_SEEDS, **rf_reg_kw)
    rho_aug, p_aug = stats.spearmanr(oof_aug, yr)
    print(f"\n--- 3. + производный признак «реконструкция палеодолин» (139 признаков) ---")
    print(f"Spearman(OOF-прогноз невязки, факт) = {rho_aug:.4f} (p={p_aug:.2e})")

    # --- 5. Итоговые карты-кандидаты: без paleo + поправка (baseline vs +paleo_hat) ---
    score_recovered_base = np.zeros_like(crit)
    score_recovered_base[pool] = r_no_paleo[pool] + oof_baseline
    score_recovered_aug = np.zeros_like(crit)
    score_recovered_aug[pool] = r_no_paleo[pool] + oof_aug

    ag_no_paleo = crit_reference.agreement(score_no_paleo, crit, pool)
    ag_base = crit_reference.agreement(score_recovered_base, crit, pool)
    ag_aug = crit_reference.agreement(score_recovered_aug, crit, pool)
    print("\n--- 4. Согласие с нативным prognoz: без paleo / +сырая поправка / +поправка с paleo_hat ---")
    print(pd.DataFrame([{"вариант": "без paleo (формула)", **ag_no_paleo},
                         {"вариант": "+OOF-поправка (сырые признаки)", **ag_base},
                         {"вариант": "+OOF-поправка (+ paleo_hat)", **ag_aug}]
                       ).round(4).to_string(index=False))

    boot = crit_reference.block_bootstrap(score_recovered_aug, crit, meta, pool, score_b=score_recovered_base)
    print(f"\nблочный бутстрэп AUC, дельта (+paleo_hat - сырые признаки): {boot['delta']:.4f}, "
          f"95% ДИ [{boot['delta_ci_lo']:.4f}, {boot['delta_ci_hi']:.4f}], блоков {boot['n_blocks']}")

    overlap = pd.DataFrame([{"площадь": f"{int(a * 100)}%",
                              "без_paleo": _topk_overlap(crit, score_no_paleo, pool, a),
                              "сырые_признаки": _topk_overlap(crit, score_recovered_base, pool, a),
                              "плюс_paleo_hat": _topk_overlap(crit, score_recovered_aug, pool, a)}
                             for a in AREAS])
    print("\nперекрытие топ-k% с нативным prognoz:")
    print(overlap.round(4).to_string(index=False))

    # --- 6. Важность признака paleo_hat внутри дополненной модели (среднее по фолдам/сидам) ---
    gkf = GroupKFold(n_splits=5)
    imps = []
    for seed in CV_SEEDS:
        for tr, te in gkf.split(X_aug, yr, bp):
            m = RandomForestRegressor(random_state=seed, n_jobs=-1, **rf_reg_kw)
            m.fit(X_aug[tr], yr[tr])
            imps.append(m.feature_importances_)
    imp = np.mean(imps, axis=0)
    rank = int((-imp).argsort().tolist().index(len(sat_cols)) + 1)
    print(f"\n--- 5. Важность paleo_hat в дополненной модели ---")
    print(f"важность (MDI) = {imp[-1]:.4f}, ранг среди {len(sat_cols) + 1} признаков = {rank}")

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"вариант": "без paleo (формула)", **ag_no_paleo},
                  {"вариант": "+OOF-поправка (сырые признаки)", **ag_base},
                  {"вариант": "+OOF-поправка (+ paleo_hat)", **ag_aug}]
                ).to_csv(out_dir / "no_paleo_v2_agreement.csv", index=False)
    print(f"\nсохранено: {out_dir / 'no_paleo_v2_agreement.csv'}")


if __name__ == "__main__":
    main()
