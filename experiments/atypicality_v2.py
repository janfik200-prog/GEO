"""Этап 4: лестница нетипичности на датасете v2 + домены + абляции по группам.

Что нового по сравнению с :mod:`experiments.atypicality_v1_1` (тот же протокол
заверки, другой пул признаков):

* признаки — :func:`src.features_v2.pool_features`: ядро v1 (гравика/магнитка,
  отношения Landsat) плюс группы этапа 2 — рельеф v2 (``ter``), линеаменты
  (``lin``), геоботанический композит Sentinel-2 (``s2``);
* АБЛЯЦИИ ПО ГРУППАМ — главный вопрос этапа: что дал каждый источник. Считаются
  оба направления, потому что они отвечают на разные вопросы: «только группа»
  (что источник может сам) и «всё без группы» (что теряется при его удалении —
  единственный честный ответ при коррелированных группах, ср. линеаменты,
  которые на 0.7-0.8 пересказывают расчленённость рельефа);
* доменная нормировка (:func:`src.atypicality.per_domain` по
  :func:`src.cell_mask.build_domains`): выброс ищется внутри своей литологии,
  иначе «нетипично» = «другая свита»;
* значимость — тот же сдвиговый (тороидальный) null на 500 сдвигов; порог
  значимости берётся ДО прогона из ``config.VER_PRIMARY``.

Метрика зафиксирована ДО прогона: lift@10% = coverage/realized_area на строго
несмещённых точках заверки, пул — валидные ячейки.

Выход: ``outputs/metrics/atypicality_v2{,_abl,_null,_boot,_contrib}.csv``,
``outputs/atypicality_v2_map.png``, ``outputs/atypicality_v2_ablation.png``.

Запуск из корня: ``python -m experiments.atypicality_v2``.
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
                 criterial_target, features_v2)

LADDER = [
    ("mahalanobis", atypicality.robust_mahalanobis),
    ("pca_residual", atypicality.pca_residual),
    ("isoforest", atypicality.isoforest_score),
    ("ocsvm", atypicality.ocsvm_score),
    ("shallow_ae", atypicality.shallow_ae_score),
    ("lof", atypicality.lof_score),
    ("knn_dist", atypicality.knn_distance_score),
    ("gmm_nll", atypicality.gmm_nll_score),
]


def run_ladder(X, valid, tag=""):
    """Все ступени + ранговый ансамбль на матрице X -> словарь скоров сетки."""
    out = {}
    for name, fn in LADDER:
        out[f"{name}{tag}"] = atypicality.expand_to_grid(fn(X), valid)
    members = {k: v[valid] for k, v in out.items()}
    out[f"rank_ensemble{tag}"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble(members), valid)
    return out


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:4.0f}s] {msg}", flush=True)

    path = config.PROCESSED_DIR / "dataset_v2.parquet"
    if not path.exists():
        raise SystemExit("нет dataset_v2.parquet — сначала "
                         "python -m experiments.build_dataset_v2")
    df = pd.read_parquet(path)
    valid = cell_mask.build_valid_mask(df)
    pool_valid = np.flatnonzero(valid)
    domains = cell_mask.build_domains(df)
    log(f"датасет v2: {df.shape[0]} ячеек, {df.shape[1]} колонок; "
        f"валидных {int(valid.sum())}")

    feat = features_v2.pool_features(df)
    sizes = features_v2.group_sizes(feat)
    cond = features_v2.condition_number(feat, valid)
    log(f"пул v2: {feat.shape[1]} признаков {sizes}, cond={cond:.3g}")
    for col, why in features_v2.dropped_columns(df).items():
        log(f"  исключён {col}: {why}")

    X = atypicality.prepare_matrix(feat, valid)

    # --- Лестница на полном пуле ---
    scores = run_ladder(X, valid)
    log("лестница v2 готова")

    # --- Доменная нормировка: выброс внутри своей литологии ---
    for name, fn in LADDER:
        s = atypicality.per_domain(fn, X, domains[valid])
        scores[f"{name}|dom"] = atypicality.expand_to_grid(s, valid)
    members = {k: v[valid] for k, v in scores.items() if k.endswith("|dom")}
    scores["rank_ensemble|dom"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble(members), valid)
    log(f"доменная нормировка готова (доменов: {np.unique(domains[valid]).size})")

    # --- Планки и равноправный критериальный ---
    meta, prognoz = criterial_target.load_prognoz_grid()
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    # --- Точки заверки ---
    pts = assessment.load_verification_points(meta)
    all_cells = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    cl_all = assessment.point_clusters(all_cells, meta)
    cl_unb = assessment.point_clusters(unb, meta)
    POINT_SETS = [("all", all_cells, cl_all), ("unbiased_strict", unb, cl_unb)]
    log(f"точки: все {all_cells.size} ({np.unique(cl_all).size} кластеров), "
        f"несмещённые {unb.size} ({np.unique(cl_unb).size} кластеров)")

    def lift(score, cells=None):
        return assessment.capture_efficiency(score, unb if cells is None else cells,
                                             pool=pool_valid)

    # --- P-A таблица ---
    rows = []
    for name, score in scores.items():
        for pts_name, cells, _ in POINT_SETS:
            pa = assessment.pa_curve(score, cells, pool=pool_valid)
            pa.insert(0, "method", name)
            pa.insert(1, "points", pts_name)
            rows.append(pa)
    pa_df = pd.concat(rows, ignore_index=True)
    log("P-A таблица готова")

    # --- Абляции по группам: «только группа» и «всё без группы» ---
    groups = [g for g in config.V2_FEATURE_GROUPS if g in sizes]
    abl_rows = []
    base_lift = {n: lift(scores[n]) for n, _ in LADDER}
    base_lift["rank_ensemble"] = lift(scores["rank_ensemble"])
    for g in groups:
        for mode, keep in [("only", (g,)),
                           ("without", tuple(x for x in groups if x != g))]:
            sub = features_v2.pool_features(df, groups=keep)
            if sub.shape[1] < 2:
                continue
            Xg = atypicality.prepare_matrix(sub, valid)
            sc = run_ladder(Xg, valid)
            for name in list(base_lift):
                abl_rows.append({
                    "group": g, "mode": mode, "n_features": sub.shape[1],
                    "method": name, "lift": lift(sc[name]),
                    "lift_full_pool": base_lift[name],
                    "delta": lift(sc[name]) - base_lift[name]})
            log(f"абляция {mode} {g} ({sub.shape[1]} призн.): "
                f"ансамбль lift {lift(sc['rank_ensemble']):.2f} "
                f"(полный пул {base_lift['rank_ensemble']:.2f})")
    abl_df = pd.DataFrame(abl_rows)

    # --- Значимость: сдвиговый null ---
    primary = [n for n, _ in LADDER] + ["rank_ensemble", "rank_ensemble|dom",
                                        "criterial", "naive_hydro", "random"]
    null_rows = []
    for name in primary:
        for pts_name, cells, _ in POINT_SETS:
            r = assessment.spatial_null_pvalue(scores[name], cells, meta,
                                               pool=pool_valid)
            null_rows.append({"method": name, "points": pts_name, **r})
    null_df = pd.DataFrame(null_rows)
    log(f"сдвиговый null ({config.VER_N_SHIFTS} сдвигов) готов")

    # --- Бутстрэп разности с критериальным ---
    boot_rows = []
    for name in [n for n, _ in LADDER] + ["rank_ensemble", "rank_ensemble|dom"]:
        for pts_name, cells, clusters in POINT_SETS:
            for kind, cl in [("point", None), ("cluster", clusters)]:
                r = assessment.bootstrap_diff(scores[name], scores["criterial"],
                                              cells, pool=pool_valid, clusters=cl)
                boot_rows.append({"a": name, "b": "criterial", "points": pts_name,
                                  "resample": kind, **r})
    boot_df = pd.DataFrame(boot_rows)
    log("бутстрэп готов")

    # --- Разложение лучшего скора по признакам ---
    best = max([n for n, _ in LADDER], key=lambda n: lift(scores[n]))
    contrib_rows = []
    for name in [best, "rank_ensemble"]:
        c = atypicality.feature_contributions(X, scores[name][valid],
                                              list(feat.columns))
        c.insert(0, "method", name)
        contrib_rows.append(c)
    contrib_df = pd.concat(contrib_rows, ignore_index=True)

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out_dir / "metrics" / "atypicality_v2"
    pa_df.to_csv(f"{stem}.csv", index=False)
    abl_df.to_csv(f"{stem}_abl.csv", index=False)
    null_df.to_csv(f"{stem}_null.csv", index=False)
    boot_df.to_csv(f"{stem}_boot.csv", index=False)
    contrib_df.to_csv(f"{stem}_contrib.csv", index=False)

    # --- Карта-панель ---
    plot_names = [n for n, _ in LADDER] + ["rank_ensemble", "rank_ensemble|dom",
                                           "criterial"]
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    prow, pcol = df["row"].to_numpy()[unb], df["col"].to_numpy()[unb]
    for ax, name in zip(axes.ravel(), plot_names):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        im = ax.imshow(grid.reshape(meta.prf, meta.pic), cmap="magma", vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^",
                   label="несмещённые точки заверки")
        ax.set_title(f"{name} (lift@10% = {lift(scores[name]):.2f})", fontsize=10)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot_names):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Лестница нетипичности на датасете v2 ({feat.shape[1]} признаков): "
                 "нормированные ранги, север сверху", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "atypicality_v2_map.png", dpi=130, bbox_inches="tight")

    # --- График абляций ---
    if not abl_df.empty:
        ens = abl_df[abl_df["method"] == "rank_ensemble"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, mode, title in [(axes[0], "only", "только эта группа"),
                                (axes[1], "without", "весь пул без этой группы")]:
            sub = ens[ens["mode"] == mode].sort_values("lift")
            ax.barh(sub["group"], sub["lift"], color="steelblue")
            ax.axvline(base_lift["rank_ensemble"], color="crimson", ls="--",
                       label=f"полный пул ({base_lift['rank_ensemble']:.2f})")
            ax.axvline(1.0, color="gray", ls=":", label="случайный скор (1.0)")
            ax.set_title(title)
            ax.set_xlabel("lift@10% на несмещённых точках")
            ax.legend(fontsize=8)
        fig.suptitle("Абляции по группам признаков, ранговый ансамбль")
        fig.tight_layout()
        fig.savefig(out_dir / "atypicality_v2_ablation.png", dpi=130,
                    bbox_inches="tight")

    # --- Сводка ---
    prim = pa_df[(pa_df["area"] == config.VER_AREA)]
    summary = (prim.pivot(index="method", columns="points", values="lift")
               .reindex(list(scores)).round(2))
    print(f"\nПервичная метрика: {config.VER_PRIMARY['metric']}, "
          f"точки {config.VER_PRIMARY['points']}")
    print(summary.to_string())
    print("\np сдвигового null (несмещённые точки):")
    print(null_df[null_df["points"] == "unbiased_strict"]
          [["method", "observed", "null_mean", "p"]].round(3).to_string(index=False))
    if not abl_df.empty:
        print("\nАбляции (ранговый ансамбль, lift@10% на несмещённых точках):")
        print(abl_df[abl_df["method"] == "rank_ensemble"]
              [["group", "mode", "n_features", "lift", "delta"]]
              .round(2).to_string(index=False))
    print(f"\nТоп-5 признаков-драйверов ({best}):")
    print(contrib_df[contrib_df["method"] == best].head(5)
          [["feature", "abs_dz"]].round(3).to_string(index=False))
    log("сохранено: outputs/atypicality_v2*.png, outputs/metrics/atypicality_v2*.csv")


if __name__ == "__main__":
    main()
