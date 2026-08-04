"""Этап 5: обучение TinyMAE на патчах и заверка нейросетевых скоров.

ЧТО ПРОВЕРЯЕТСЯ. Может ли нейросетевой метод БЕЗ УЧИТЕЛЯ превзойти
критериальный анализ. Отличие от этапов 4/4b — не в данных (пул тот же v3,
52 признака, включая 8 геологических слоёв критериального метода), а в том,
что модель видит ОКРЕСТНОСТЬ ячейки, а не только её собственный вектор:
self-supervised masked-автоэнкодер учится достраивать закрытые куски патча
5.5 x 5.5 км и вынужден выучить локальный рисунок поля.

Из модели берутся два разных скора (см. :mod:`src.mae_model`):

* ``mae_err`` — ошибка восстановления ЦЕНТРА патча при принудительно закрытом
  центре: «насколько ячейка непредсказуема по своему окружению»;
* лестница :mod:`src.atypicality` поверх 16-мерного эмбеддинга — «насколько
  нетипична ОКРЕСТНОСТЬ ячейки», в двух режимах (глобальный и по литодоменам).

КРИТЕРИЙ ПРЕВОСХОДСТВА — ЗАФИКСИРОВАН ДО ПРОГОНА (тот же, что в этапе 4b,
менять его под результат нельзя). Первичная метрика ``config.VER_PRIMARY``:
lift@10% на строго несмещённых точках заверки. Нейросетевой метод считается
превзошедшим критериальный, если выполнены ОБА условия:

1. нижняя граница 90% бутстрэп-интервала разности (нейросетевой минус
   критериальный) при ресэмплинге ПО ПРОСТРАНСТВЕННЫМ КЛАСТЕРАМ точек строго
   больше нуля;
2. сдвиговый (тороидальный) null самого скора даёт p ниже порога Бонферрони
   0.05/``config.MAE_N_CONFIGS`` = 0.0025, где 20 конфигураций = ошибка
   реконструкции + лестница (8 детекторов + ансамбль) x 2 режима нормировки.

При 19 точках и площади 10% одна точка стоит 0.53 lift, поэтому условие 1
требует перевеса примерно в 5-6 точек из 19. Планка высокая намеренно. Если она
не взята, корректная формулировка — «методы неразличимы на имеющемся объёме
заверки», а не «критериальный лучше».

Обучение честно валидируется ПРОСТРАНСТВЕННЫМИ БЛОКАМИ (патчи соседних ячеек
перекрываются на 10/11, случайный holdout мерил бы память): число эпох
выбирается по блочному val-loss.

Выход: ``outputs/metrics/mae{,_hist,_null,_boot,_contrib,_verdict}.csv``,
``outputs/mae_map.png``, ``outputs/mae_training.png``,
``outputs/metrics/mae_scores.npz`` (скоры и эмбеддинг для этапа 7).

Запуск из корня: ``python -m experiments.mae_train``.
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
                 criterial_target, features_v2, mae_model)

from experiments.atypicality_v2 import LADDER  # noqa: E402


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    path = config.PROCESSED_DIR / "dataset_v2.parquet"
    if not path.exists():
        raise SystemExit("нет dataset_v2.parquet — сначала "
                         "python -m experiments.build_dataset_v2")
    df = pd.read_parquet(path)
    valid = cell_mask.build_valid_mask(df)
    pool_valid = np.flatnonzero(valid)
    domains = cell_mask.build_domains(df)
    meta, prognoz = criterial_target.load_prognoz_grid()

    rows = df["row"].to_numpy().astype(np.int64)
    cols = df["col"].to_numpy().astype(np.int64)
    # Порядок строк датасета обязан совпадать с разворачиванием сетки, иначе
    # растровый стек соберётся с перепутанными ячейками и всё дальнейшее —
    # мусор, который ничем себя не выдаст.
    assert np.array_equal(rows * meta.pic + cols, np.arange(len(df))), \
        "порядок ячеек датасета не совпадает с сеткой (row*pic + col)"
    log(f"датасет v2: {df.shape[0]} ячеек ({meta.prf}x{meta.pic}), "
        f"валидных {int(valid.sum())}")

    # Пул v3: тот же, что в этапе 4b, — сравнение должно отличаться ТОЛЬКО
    # алгоритмом. Обучения на метках нет, циркулярность невозможна.
    feat = features_v2.pool_features(df, include_factors=True)
    log(f"пул v3: {feat.shape[1]} признаков {features_v2.group_sizes(feat)}")

    stack, vld = mae_model.build_stack(feat, valid, rows, cols,
                                       (meta.prf, meta.pic))
    centers = np.stack([rows[valid], cols[valid]], axis=1)
    is_val = mae_model.block_split(centers[:, 0], centers[:, 1],
                                   config.MAE_PATCH)
    log(f"патчи {config.MAE_PATCH}x{config.MAE_PATCH} "
        f"({config.MAE_PATCH * config.CELL_SIZE / 1000:.1f} км): "
        f"центров {len(centers)}, блочный holdout {int(is_val.sum())} "
        f"({is_val.mean():.0%}), маскируется {config.MAE_MASK_RATIO:.0%} позиций")

    model, hist = mae_model.train_mae(stack, vld, centers, is_val, log=log)
    best = hist.loc[hist["val_loss"].idxmin()]
    log(f"обучение готово: {mae_model.n_params(model)} параметров, "
        f"лучшая эпоха {int(best['epoch'])} из {len(hist)}, "
        f"val-loss {best['val_loss']:.4f} (старт {hist['val_loss'].iloc[0]:.4f})")

    emb = mae_model.embed(model, stack, vld, centers)
    err = mae_model.reconstruction_error(model, stack, vld, centers)
    log(f"эмбеддинг {emb.shape}, ошибка реконструкции посчитана "
        f"({config.MAE_ERR_REPEATS} реализаций маски)")

    # --- Скоры на сетке ---
    scores = {"mae_err": atypicality.expand_to_grid(err, valid)}
    Z = atypicality.prepare_matrix(pd.DataFrame(emb, columns=[
        f"emb{i}" for i in range(emb.shape[1])]), np.ones(len(emb), dtype=bool))
    for name, fn in LADDER:
        scores[f"emb_{name}"] = atypicality.expand_to_grid(fn(Z), valid)
    members = {k: v[valid] for k, v in scores.items() if k.startswith("emb_")}
    scores["emb_rank_ensemble"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble(members), valid)
    log("лестница на эмбеддинге готова")

    for name, fn in LADDER:
        s = atypicality.per_domain(fn, Z, domains[valid])
        scores[f"emb_{name}|dom"] = atypicality.expand_to_grid(s, valid)
    members = {k: v[valid] for k, v in scores.items() if k.endswith("|dom")}
    scores["emb_rank_ensemble|dom"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble(members), valid)
    log("доменная нормировка готова")

    # --- Опорные методы: критериальный и лучшая ступень этапа 4b на том же пуле ---
    X = atypicality.prepare_matrix(feat, valid)
    scores["pool_pca_residual"] = atypicality.expand_to_grid(
        atypicality.pca_residual(X), valid)
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
        f"несмещённые {unb.size} ({np.unique(cl_unb).size} кластеров); "
        f"шаг метрики 1 точка = {1.0 / unb.size / config.VER_AREA:.2f} lift")

    def lift(score, cells=None):
        return assessment.capture_efficiency(score, unb if cells is None else cells,
                                             pool=pool_valid)

    rows_pa = []
    for name, score in scores.items():
        for pts_name, cells, _ in POINT_SETS:
            pa = assessment.pa_curve(score, cells, pool=pool_valid)
            pa.insert(0, "method", name)
            pa.insert(1, "points", pts_name)
            rows_pa.append(pa)
    pa_df = pd.concat(rows_pa, ignore_index=True)

    # --- Значимость и бутстрэп ---
    alpha = 0.05 / config.MAE_N_CONFIGS
    own = [k for k in scores if k.startswith(("mae_", "emb_"))]
    null_rows = []
    for name in own + ["pool_pca_residual", "criterial", "naive_hydro", "random"]:
        for pts_name, cells, _ in POINT_SETS:
            r = assessment.spatial_null_pvalue(scores[name], cells, meta,
                                               pool=pool_valid)
            null_rows.append({"method": name, "points": pts_name, **r})
    null_df = pd.DataFrame(null_rows)
    log(f"сдвиговый null ({config.VER_N_SHIFTS} сдвигов) готов; "
        f"порог Бонферрони 0.05/{config.MAE_N_CONFIGS} = {alpha:.4f}")

    boot_rows = []
    for name in own:
        for base in ["criterial", "pool_pca_residual"]:
            for pts_name, cells, clusters in POINT_SETS:
                for kind, cl in [("point", None), ("cluster", clusters)]:
                    r = assessment.bootstrap_diff(scores[name], scores[base],
                                                  cells, pool=pool_valid,
                                                  clusters=cl)
                    boot_rows.append({"a": name, "b": base, "points": pts_name,
                                      "resample": kind, **r})
    boot_df = pd.DataFrame(boot_rows)
    log("бутстрэп готов")

    # --- Вердикт по предзарегистрированному критерию ---
    ver_rows = []
    for name in own:
        b = boot_df[(boot_df["a"] == name) & (boot_df["b"] == "criterial")
                    & (boot_df["points"] == "unbiased_strict")
                    & (boot_df["resample"] == "cluster")].iloc[0]
        p = null_df[(null_df["method"] == name)
                    & (null_df["points"] == "unbiased_strict")]["p"].iloc[0]
        ver_rows.append({"method": name, "lift": lift(scores[name]),
                         "lift_criterial": lift(scores["criterial"]),
                         "delta": b["delta"], "ci_lo": b["ci_lo"],
                         "ci_hi": b["ci_hi"], "p_shift": p, "alpha": alpha,
                         "cond1_ci_lo_gt_0": bool(b["ci_lo"] > 0),
                         "cond2_p_lt_alpha": bool(p < alpha),
                         "superior": bool(b["ci_lo"] > 0 and p < alpha)})
    ver_df = pd.DataFrame(ver_rows).sort_values("lift", ascending=False)

    # --- Драйверы: какие признаки пула стоят за нейросетевым скором ---
    contrib_rows = []
    for name in ["mae_err", "emb_rank_ensemble"]:
        c = atypicality.feature_contributions(X, scores[name][valid],
                                              list(feat.columns))
        c.insert(0, "method", name)
        contrib_rows.append(c)
    contrib_df = pd.concat(contrib_rows, ignore_index=True)

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out_dir / "metrics" / "mae"
    pa_df.to_csv(f"{stem}.csv", index=False)
    hist.to_csv(f"{stem}_hist.csv", index=False)
    null_df.to_csv(f"{stem}_null.csv", index=False)
    boot_df.to_csv(f"{stem}_boot.csv", index=False)
    contrib_df.to_csv(f"{stem}_contrib.csv", index=False)
    ver_df.to_csv(f"{stem}_verdict.csv", index=False)
    np.savez_compressed(f"{stem}_scores.npz", emb=emb, mae_err=err,
                        centers=centers, valid=valid)

    # --- График обучения ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hist["epoch"], hist["train_loss"], label="обучение")
    ax.plot(hist["epoch"], hist["val_loss"], label="блочный holdout (20x20 ячеек)")
    ax.axvline(best["epoch"], color="crimson", ls="--",
               label=f"выбранная эпоха {int(best['epoch'])}")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("MSE на закрытых позициях, робастные z^2")
    ax.set_title("Обучение TinyMAE: восстановление закрытых участков патча")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_training.png", dpi=130, bbox_inches="tight")

    # --- Карта-панель ---
    plot_names = ["mae_err", "emb_rank_ensemble", "emb_rank_ensemble|dom",
                  "emb_pca_residual", "emb_mahalanobis", "emb_isoforest",
                  "pool_pca_residual", "criterial"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    prow, pcol = df["row"].to_numpy()[unb], df["col"].to_numpy()[unb]
    for ax, name in zip(axes.ravel(), plot_names):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        im = ax.imshow(grid.reshape(meta.prf, meta.pic), cmap="magma",
                       vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^",
                   label="несмещённые точки заверки")
        ax.set_title(f"{name} (lift@10% = {lift(scores[name]):.2f})", fontsize=10)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Нейросетевой метод без учителя (TinyMAE на патчах 5.5 км) "
                 "против критериального прогноза: нормированные ранги, север сверху",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_map.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    prim = pa_df[pa_df["area"] == config.VER_AREA]
    summary = (prim.pivot(index="method", columns="points", values="lift")
               .reindex(list(scores)).round(2))
    print(f"\nПервичная метрика: {config.VER_PRIMARY['metric']}, "
          f"точки {config.VER_PRIMARY['points']}")
    print(summary.to_string())
    print("\nВердикт по предзарегистрированному критерию превосходства:")
    print(ver_df[["method", "lift", "lift_criterial", "delta", "ci_lo", "ci_hi",
                  "p_shift", "superior"]].round(3).to_string(index=False))
    print(f"\nТоп-5 признаков-драйверов (mae_err):")
    print(contrib_df[contrib_df["method"] == "mae_err"].head(5)
          [["feature", "abs_dz"]].round(3).to_string(index=False))
    if ver_df["superior"].any():
        won = ", ".join(ver_df.loc[ver_df["superior"], "method"])
        log(f"КРИТЕРИЙ ВЗЯТ: {won}")
    else:
        log("критерий не взят ни одним скором: методы неразличимы на "
            "имеющемся объёме заверки")
    log("сохранено: outputs/mae_map.png, outputs/mae_training.png, "
        "outputs/metrics/mae*.csv")


if __name__ == "__main__":
    main()
