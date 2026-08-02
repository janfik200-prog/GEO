"""Лестница нетипичности на признаках v1.1 — прогон после ревью model-critic.

Отличия от :mod:`experiments.atypicality_v1` (все — по обязательным пунктам
ревью, см. `Идеи-и-вопросы.md`):

* признаки — :func:`src.features_v11.ladder_features` (единый источник списка):
  без алгебраически избыточных градиентных колонок, Landsat отношениями,
  ``relief_m`` залит из Copernicus DEM; первичный набор БЕЗ ``dist_*``
  (иначе наивная планка «минус расстояние до рек» не контроль), с ними —
  абляция;
* точки заверки — СТРОГО несмещённое подмножество (ячейка несмещённая, только
  если все её пробы коренные/проявления);
* lift = coverage / realized_area (ties критериального не завышают площадь);
* значимость — сдвиговый (тороидальный) null на 500 сдвигов, а не только
  бутстрэп; бутстрэп — точечный И кластерный (маршруты опробования);
* два пула порога: валидные ячейки и валидные внутри свит;
* диагностики: число обусловленности до/после, калибровка relief~DEM,
  сходимость AE (``n_iter_``), сид-свип детекторов, разложение скора по
  признакам.

Первичная метрика зафиксирована ДО прогона: ``config.VER_PRIMARY``.

Выход: ``outputs/metrics/atypicality_v1_1{,_boot,_null,_seeds,_contrib}.csv``,
``outputs/atypicality_v1_1_map.png``.

Запуск из корня: ``python -m experiments.atypicality_v1_1``.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

from src import (assessment, atypicality, cell_mask, config,  # noqa: E402
                 criterial_target, features_v11)

LADDER = [
    ("mahalanobis", atypicality.robust_mahalanobis),
    ("pca_residual", atypicality.pca_residual),
    ("isoforest", atypicality.isoforest_score),
    ("ocsvm", atypicality.ocsvm_score),
    ("shallow_ae", atypicality.shallow_ae_score),
]
SEED_SWEEP_FNS = [("mahalanobis", atypicality.robust_mahalanobis),
                  ("isoforest", atypicality.isoforest_score),
                  ("ocsvm", atypicality.ocsvm_score),
                  ("shallow_ae", atypicality.shallow_ae_score)]


def _run_ladder(X, valid, tag=""):
    """Все ступени + ранговый ансамбль на матрице X -> словарь скоров сетки."""
    out = {}
    for name, fn in LADDER:
        s = fn(X)
        out[f"{name}{tag}"] = atypicality.expand_to_grid(s, valid)
    members = {k: v[valid] for k, v in out.items()}
    out[f"rank_ensemble{tag}"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble(members), valid)
    return out


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:4.0f}s] {msg}", flush=True)

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool_valid = np.flatnonzero(valid)
    in_svita = valid & (np.nan_to_num(df["mask_svita"].to_numpy(), nan=0.0) > 0)
    pool_svita = np.flatnonzero(in_svita)
    log(f"валидных ячеек {valid.sum()}/{len(df)}, из них внутри свит {in_svita.sum()}")

    # --- Признаки: старый набор (только диагностика) и v1.1 ---
    feat_old = criterial_target.training_features()
    cond_old = features_v11.condition_number(feat_old, valid)
    feat_nodist = features_v11.ladder_features(df, include_dist=False)
    feat_dist = features_v11.ladder_features(df, include_dist=True)
    cond_new = features_v11.condition_number(feat_nodist, valid)
    log(f"признаков: v1 {feat_old.shape[1]} (cond={cond_old:.3g}), "
        f"v1.1 без dist {feat_nodist.shape[1]} (cond={cond_new:.3g}), "
        f"v1.1 с dist {feat_dist.shape[1]}")

    X = atypicality.prepare_matrix(feat_nodist, valid)
    X_dist = atypicality.prepare_matrix(feat_dist, valid)

    # --- Диагностика AE: сходимость (в v1 lift 0.41 - подозрение на баг) ---
    ae_score, ae_info = atypicality.shallow_ae_score(X, return_info=True)
    log(f"AE: n_iter_={ae_info['n_iter']} из max_iter={config.ANOM_AE_MAX_ITER}")

    # --- Лестница на обоих наборах ---
    scores = _run_ladder(X, valid)
    log("лестница v1.1 (без dist) готова")
    scores.update(_run_ladder(X_dist, valid, tag="|dist"))
    log("лестница v1.1 (с dist, абляция) готова")

    # --- Планки и равноправный критериальный ---
    meta, prognoz = criterial_target.load_prognoz_grid()
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    # Диагностика ties критериального вне свит
    outside = valid & ~in_svita
    pg_out = prognoz.ravel()[outside]
    vals, cnt = np.unique(pg_out, return_counts=True)
    log(f"prognoz вне свит: {outside.sum()} валидных ячеек "
        f"({outside.sum() / valid.sum():.1%} пула), уникальных значений {vals.size}, "
        f"доля самого частого {cnt.max() / cnt.sum():.1%}")

    # --- Точки заверки: строгая дедупликация + кластеры маршрутов ---
    pts = assessment.load_verification_points(meta)
    all_cells = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb_strict = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    loose = pts.loc[pts["code"].isin(list(config.VER_UNBIASED_CODES)), "cell"].to_numpy()
    unb_loose = assessment.filter_points(loose, valid)
    cl_all = assessment.point_clusters(all_cells, meta)
    cl_unb = assessment.point_clusters(unb_strict, meta)
    log(f"точки: все {all_cells.size} ({np.unique(cl_all).size} кластеров), "
        f"несмещённые строго {unb_strict.size} ({np.unique(cl_unb).size} кластеров), "
        f"по старому мягкому правилу было {unb_loose.size}")

    POINT_SETS = [("all", all_cells, cl_all), ("unbiased_strict", unb_strict, cl_unb)]
    POOLS = [("valid", pool_valid), ("valid_svita", pool_svita)]

    # --- P-A таблица: метод x набор точек x пул x площадь ---
    rows = []
    for name, score in scores.items():
        for pts_name, cells, _ in POINT_SETS:
            for pool_name, pool in POOLS:
                pa = assessment.pa_curve(score, cells, pool=pool)
                pa.insert(0, "method", name)
                pa.insert(1, "points", pts_name)
                pa.insert(2, "pool", pool_name)
                rows.append(pa)
    pa_df = pd.concat(rows, ignore_index=True)
    log("P-A таблица готова")

    # --- Бутстрэп разностей vs критериальный: точечный и кластерный ---
    primary = [n for n, _ in LADDER] + ["rank_ensemble", "naive_hydro"]
    boot_rows = []
    for name in primary:
        for pts_name, cells, clusters in POINT_SETS:
            for kind, cl in [("point", None), ("cluster", clusters)]:
                r = assessment.bootstrap_diff(scores[name], scores["criterial"],
                                              cells, pool=pool_valid, clusters=cl)
                boot_rows.append({"a": name, "b": "criterial", "points": pts_name,
                                  "resample": kind, **r})
    boot_df = pd.DataFrame(boot_rows)
    log("бутстрэп (точечный и кластерный) готов")

    # --- Сдвиговый null на первичной конфигурации ---
    null_rows = []
    for name in primary + ["criterial", "random"]:
        for pts_name, cells, _ in POINT_SETS:
            r = assessment.spatial_null_pvalue(scores[name], cells, meta,
                                               pool=pool_valid)
            null_rows.append({"method": name, "points": pts_name, **r})
    null_df = pd.DataFrame(null_rows)
    log(f"сдвиговый null ({config.VER_N_SHIFTS} сдвигов) готов")

    # --- Сид-свип стохастичных ступеней (диагностика стабильности) ---
    seed_rows = []
    for name, fn in SEED_SWEEP_FNS:
        maps, lifts = [], []
        for seed in config.ANOM_SEEDS:
            s = fn(X, seed=seed)
            full = atypicality.expand_to_grid(s, valid)
            maps.append(s)
            lifts.append(assessment.capture_efficiency(full, unb_strict,
                                                       pool=pool_valid))
        rho = [sps.spearmanr(maps[i], maps[j]).statistic
               for i in range(len(maps)) for j in range(i + 1, len(maps))]
        seed_rows.append({"method": name, "lift_median": float(np.median(lifts)),
                          "lift_min": float(np.min(lifts)),
                          "lift_max": float(np.max(lifts)),
                          "spearman_median": float(np.median(rho)),
                          "n_seeds": len(config.ANOM_SEEDS)})
        log(f"сид-свип {name}: lift медиана {np.median(lifts):.2f} "
            f"[{np.min(lifts):.2f}, {np.max(lifts):.2f}], "
            f"Spearman карт {np.median(rho):.3f}")
    seed_df = pd.DataFrame(seed_rows)

    # --- Разложение скора по признакам ---
    names = list(feat_nodist.columns)
    ladder_only = [n for n, _ in LADDER]
    best = max(ladder_only, key=lambda n: assessment.capture_efficiency(
        scores[n], unb_strict, pool=pool_valid))
    contrib_rows = []
    for name in [best, "rank_ensemble"]:
        c = atypicality.feature_contributions(X, scores[name][valid], names)
        c.insert(0, "method", name)
        contrib_rows.append(c)
    contrib_df = pd.concat(contrib_rows, ignore_index=True)

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out_dir / "metrics" / "atypicality_v1_1"
    pa_df.to_csv(f"{stem}.csv", index=False)
    boot_df.to_csv(f"{stem}_boot.csv", index=False)
    null_df.to_csv(f"{stem}_null.csv", index=False)
    seed_df.to_csv(f"{stem}_seeds.csv", index=False)
    contrib_df.to_csv(f"{stem}_contrib.csv", index=False)

    # --- Карта-панель: ступени + ансамбль + критериальный (нормированные ранги) ---
    plot_names = ladder_only + ["rank_ensemble", "criterial"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    prow = df["row"].to_numpy()[unb_strict]
    pcol = df["col"].to_numpy()[unb_strict]
    for ax, name in zip(axes.ravel(), plot_names):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        im = ax.imshow(grid.reshape(meta.prf, meta.pic), cmap="magma", vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^",
                   label="несмещённые точки заверки")
        ax.set_title(name)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot_names):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Лестница нетипичности (признаки v1.1): нормированные ранги, "
                 "север сверху", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "atypicality_v1_1_map.png", dpi=130, bbox_inches="tight")

    # --- Сводка: первичная конфигурация ---
    prim = pa_df[(pa_df["area"] == config.VER_AREA) & (pa_df["pool"] == "valid")]
    summary = (prim.pivot(index="method", columns="points", values="lift")
               .reindex([n for n in scores]).round(2))
    print(f"\nПервичная метрика: {config.VER_PRIMARY['metric']}, "
          f"точки {config.VER_PRIMARY['points']}, пул {config.VER_PRIMARY['pool']}")
    print(summary.to_string())
    print("\np сдвигового null (первичные точки):")
    print(null_df[null_df["points"] == "unbiased_strict"]
          [["method", "observed", "null_mean", "p"]].round(3).to_string(index=False))
    print(f"\nТоп-5 признаков-драйверов ({best}):")
    print(contrib_df[contrib_df["method"] == best].head(5)
          [["feature", "abs_dz"]].round(3).to_string(index=False))
    log("сохранено: outputs/atypicality_v1_1_map.png, outputs/metrics/atypicality_v1_1*.csv")


if __name__ == "__main__":
    main()
