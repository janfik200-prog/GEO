"""Задача 5, повтор на dataset_v6: воспроизвести критериальный прогноз без
фактора «долины и впадины» на актуальном (очищенном) наборе признаков вместо
dataset_v5. Протокол, метод, цель для ML (невязка, не сам prognoz — см.
методическое предупреждение в докстринге ``reproduce_without_paleo.py``) — те
же; отличие только в источнике признаков потенциальных полей/космоснимков.

Запуск из корня: ``python -X utf8 -m experiments.reproduce_without_paleo_v6``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target, features, features_v2  # noqa: E402

# Потенциальные поля + космоснимки и их производные (НЕ геологические факторные
# слои: те уже в формуле-без-палео, повторный ввод — дублирование, не новизна).
SAT_PF_GROUPS = ("gm", "pf", "ls", "s2", "s2raw", "s1", "psr", "ast", "astir", "l8", "opt")
CV_SEEDS = (1, 7, 13, 21, 42)
AREAS = (0.05, 0.10, 0.20)


def _pct_rank(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Процентильный ранг [0, 1] внутри пула, NaN вне пула."""
    r = np.full(x.shape, np.nan)
    r[pool] = pd.Series(x[pool]).rank(pct=True).to_numpy()
    return r


def _topk_overlap(score_a: np.ndarray, score_b: np.ndarray, pool: np.ndarray, area: float) -> float:
    """Доля топ-``area`` карты A, совпавшая с топ-``area`` карты B (по пулу, равные площади)."""
    a, b = score_a[pool], score_b[pool]
    top_a = a >= np.quantile(a, 1 - area)
    top_b = b >= np.quantile(b, 1 - area)
    return float((top_a & top_b).sum() / top_a.sum())


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz.ravel())
    print(f"ячеек {len(df)}, валидных {pool.size}")

    # --- 1. Цена вопроса: реконструкция формулы с палео и без ---
    dist_by_role = {role: df[f"dist_{role}"].to_numpy(dtype=float) for role in config.TAXONOMY_TRANSFORMS}
    score_full = -features.taxonomy_weighted_distance(dist_by_role)
    score_no_paleo = -features.taxonomy_weighted_distance(dist_by_role, exclude={"paleo"})

    ag_full = crit_reference.agreement(score_full, crit, pool)
    ag_no_paleo = crit_reference.agreement(score_no_paleo, crit, pool)
    print("\n--- 1. Реконструкция: полная формула (6 факторов) vs без paleo (5 факторов) ---")
    print(pd.DataFrame([{"вариант": "полная (6 факторов)", **ag_full},
                         {"вариант": "без paleo (5 факторов)", **ag_no_paleo}]
                       ).round(4).to_string(index=False))

    overlap0 = pd.DataFrame([{"площадь": f"{int(a * 100)}%",
                               "перекрытие_с_нативным": _topk_overlap(crit, score_no_paleo, pool, a)}
                              for a in AREAS])
    print("\nперекрытие топ-k% (нативный prognoz vs формула без paleo):")
    print(overlap0.round(4).to_string(index=False))

    # --- 2. Цель для ML: невязка (не сам prognoz — методическое предупреждение) ---
    r_crit = _pct_rank(crit, pool)
    r_no_paleo = _pct_rank(score_no_paleo, pool)
    residual = np.zeros_like(crit)
    residual[pool] = r_crit[pool] - r_no_paleo[pool]

    def _is_technical(c: str) -> bool:
        return c.endswith("_n_obs") or "valid_frac" in c

    sat_cols = [c for c in df.columns
                if features_v2.feature_group(c) in SAT_PF_GROUPS and not _is_technical(c)]
    print(f"\nпризнаков потенциальных полей + космоснимков: {len(sat_cols)} "
          f"(группы {SAT_PF_GROUPS}, dataset_v6)")
    X = df[sat_cols].fillna(0).to_numpy(dtype=float)

    x_c = df["x"].to_numpy(dtype=float)
    y_c = df["y"].to_numpy(dtype=float)
    bx = np.floor((x_c - x_c.min()) / config.VAL_BLOCK_SIZE).astype(int)
    by = np.floor((y_c - y_c.min()) / config.VAL_BLOCK_SIZE).astype(int)
    blocks = bx * (by.max() + 1) + by

    Xp, yp, gp = X[pool], residual[pool], blocks[pool]
    gkf = GroupKFold(n_splits=5)
    oof_stack = np.zeros((len(CV_SEEDS), pool.size))
    for si, seed in enumerate(CV_SEEDS):
        oof = np.zeros(pool.size)
        for tr, te in gkf.split(Xp, yp, gp):
            model = RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                random_state=seed, n_jobs=-1,
            )
            model.fit(Xp[tr], yp[tr])
            oof[te] = model.predict(Xp[te])
        oof_stack[si] = oof
    oof_pred = oof_stack.mean(axis=0)

    rho_skill, p_skill = stats.spearmanr(oof_pred, yp)
    print("\n--- 2. OOF-регрессия невязки по потенциальным полям + космоснимкам ---")
    print(f"Spearman(OOF-прогноз невязки, факт. невязка) = {rho_skill:.4f} (p={p_skill:.2e}), "
          f"{len(CV_SEEDS)} сидов x 5 блочных фолдов ({config.VAL_BLOCK_SIZE / 1000:.0f} км)")

    # --- 3. Формула-без-paleo + OOF-поправка, сравнение с самой формулой-без-paleo ---
    score_recovered = np.zeros_like(crit)
    score_recovered[pool] = r_no_paleo[pool] + oof_pred
    ag_recovered = crit_reference.agreement(score_recovered, crit, pool)
    boot = crit_reference.block_bootstrap(score_recovered, crit, meta, pool, score_b=score_no_paleo)

    print("\n--- 3. Без paleo (формула) vs без paleo + OOF-поправка (спутник+поля) ---")
    print(pd.DataFrame([{"вариант": "без paleo (формула, baseline)", **ag_no_paleo},
                         {"вариант": "без paleo + OOF-поправка", **ag_recovered}]
                       ).round(4).to_string(index=False))
    print(f"\nблочный бутстрэп AUC, дельта (с поправкой - без): {boot['delta']:.4f}, "
          f"95% ДИ [{boot['delta_ci_lo']:.4f}, {boot['delta_ci_hi']:.4f}], блоков {boot['n_blocks']}")

    overlap1 = pd.DataFrame([{"площадь": f"{int(a * 100)}%",
                               "перекрытие_без_paleo": _topk_overlap(crit, score_no_paleo, pool, a),
                               "перекрытие_с_поправкой": _topk_overlap(crit, score_recovered, pool, a)}
                              for a in AREAS])
    print("\nперекрытие топ-k% с нативным prognoz — до/после OOF-поправки:")
    print(overlap1.round(4).to_string(index=False))

    # --- 4. Интерпретируемость: какие каналы несут сигнал невязки ---
    rows = []
    for c in sat_cols:
        v = df[c].to_numpy(dtype=float)
        rho, p = stats.spearmanr(v[pool], residual[pool])
        rows.append({"признак": c, "spearman_с_невязкой": rho, "p": p})
    feat_tbl = pd.DataFrame(rows)
    top_feat = feat_tbl.reindex(feat_tbl["spearman_с_невязкой"].abs().sort_values(ascending=False).index).head(15)
    print("\n--- 4. Топ-15 признаков потенциальных полей/космоснимков по |Spearman| с невязкой ---")
    print(top_feat.round(4).to_string(index=False))

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"вариант": "полная (6 факторов)", **ag_full},
                  {"вариант": "без paleo (5 факторов)", **ag_no_paleo},
                  {"вариант": "без paleo + OOF-поправка", **ag_recovered}]
                ).to_csv(out_dir / "no_paleo_agreement_v6.csv", index=False)
    feat_tbl.to_csv(out_dir / "no_paleo_gap_features_v6.csv", index=False)
    print(f"\nсохранено: {out_dir / 'no_paleo_agreement_v6.csv'}, {out_dir / 'no_paleo_gap_features_v6.csv'}")

    # --- 5. Карта листа: нативный prognoz vs без paleo vs без paleo + OOF-поправка ---
    import matplotlib.pyplot as plt

    labels = criterial_target.label_ore_objects(prognoz).ravel()
    obj_rows, obj_cols = [], []
    for oid in range(1, int(labels.max()) + 1):
        r, c = np.unravel_index(np.flatnonzero(labels == oid), (meta.prf, meta.pic))
        obj_rows.append(r.mean())
        obj_cols.append(c.mean())
    invalid = ~np.isfinite(prognoz.ravel())

    def _panel(ax, arr, title):
        img = arr.reshape(meta.prf, meta.pic).astype(float).copy()
        img.ravel()[invalid] = np.nan
        im = ax.imshow(img, cmap="viridis")
        ax.scatter(obj_cols, obj_rows, marker="x", color="red", s=70, linewidths=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    _panel(axes[0], crit, "Нативный prognoz\n(инвертированный crit)")
    _panel(axes[1], score_no_paleo, "Без paleo\n(5 факторов формулы)")
    im = _panel(axes[2], score_recovered, "Без paleo + OOF-поправка\n(потенц. поля + космоснимки)")
    fig.colorbar(im, ax=axes, shrink=0.75, label="скор (ранг по пулу), больше = перспективнее")
    fig.suptitle("Задача 5 (dataset_v6): воспроизведение прогноза без фактора палеодолин "
                f"(красный x — рудные объекты, capture {config.CREF_AREA:.0%}: "
                f"{ag_full['capture']:.2f}x -> {ag_no_paleo['capture']:.2f}x -> {ag_recovered['capture']:.2f}x)",
                fontsize=10)
    map_path = ROOT / "outputs" / "no_paleo_map_v6.png"
    fig.savefig(map_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {map_path}")


if __name__ == "__main__":
    main()
