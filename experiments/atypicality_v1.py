"""Лестница нетипичности на датасете v1 — первая карта фазы «собственный алгоритм».

Цель: карта перспективности БЕЗ обучения на метках (критериальный прогноз и
точки опробования — только заверка) на уже собранных 29 независимых признаках
(гравика/магнитка, Landsat 7, рельеф, гидросеть; факторные слои исключены).

Ступени: Махаланобис (MinCovDet) -> PCA-невязка -> IsolationForest -> OC-SVM ->
неглубокий AE -> ранговый ансамбль. Заверка единым протоколом
:mod:`src.assessment`: P-A/lift на 73 ячейках точек и несмещённом
подмножестве n=19, планки — критериальный прогноз (равноправный метод),
наивный «минус расстояние до гидросети», случайный скор.

Выход: ``outputs/atypicality_v1_map.png`` (карты ступеней),
``outputs/metrics/atypicality_v1.csv`` (P-A таблица),
``outputs/metrics/atypicality_v1_boot.csv`` (бутстрэп разностей).

Запуск из корня: ``python -m experiments.atypicality_v1``.
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

from src import assessment, atypicality, cell_mask, config, criterial_target  # noqa: E402

LADDER = [
    ("mahalanobis", atypicality.robust_mahalanobis),
    ("pca_residual", atypicality.pca_residual),
    ("isoforest", atypicality.isoforest_score),
    ("ocsvm", atypicality.ocsvm_score),
    ("shallow_ae", atypicality.shallow_ae_score),
]


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    feat_df = criterial_target.training_features()
    X = atypicality.prepare_matrix(feat_df, valid)
    print(f"[{time.time()-t0:3.0f}s] X: {X.shape}, валидных ячеек {valid.sum()}/{len(df)}")

    scores: dict[str, np.ndarray] = {}
    for name, fn in LADDER:
        scores[name] = atypicality.expand_to_grid(fn(X), valid)
        print(f"[{time.time()-t0:3.0f}s] {name} готов")
    members = {k: v[valid] for k, v in scores.items()}
    scores["rank_ensemble"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble(members), valid)

    # Планки и равноправный критериальный
    meta, prognoz = criterial_target.load_prognoz_grid()
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    pts = assessment.load_verification_points(meta)
    all_cells = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb_cells = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    print(f"[{time.time()-t0:3.0f}s] точки: все {all_cells.size}, несмещённые {unb_cells.size}")

    # P-A таблица: метод x набор точек x площадь
    rows = []
    for name, score in scores.items():
        for pts_name, cells in [("all", all_cells), ("unbiased", unb_cells)]:
            pa = assessment.pa_curve(score, cells, pool=pool)
            pa.insert(0, "method", name)
            pa.insert(1, "points", pts_name)
            rows.append(pa)
    pa_df = pd.concat(rows, ignore_index=True)

    # Бутстрэп: каждая ступень против предыдущей и против критериального
    ladder_names = [n for n, _ in LADDER] + ["rank_ensemble"]
    boot_rows = []
    for prev, cur in zip(ladder_names, ladder_names[1:]):
        for pts_name, cells in [("all", all_cells), ("unbiased", unb_cells)]:
            r = assessment.bootstrap_diff(scores[cur], scores[prev], cells, pool=pool)
            boot_rows.append({"a": cur, "b": prev, "points": pts_name, **r})
    for name in ladder_names:
        for pts_name, cells in [("all", all_cells), ("unbiased", unb_cells)]:
            r = assessment.bootstrap_diff(scores[name], scores["criterial"], cells, pool=pool)
            boot_rows.append({"a": name, "b": "criterial", "points": pts_name, **r})
    boot_df = pd.DataFrame(boot_rows)

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    pa_df.to_csv(out_dir / "metrics" / "atypicality_v1.csv", index=False)
    boot_df.to_csv(out_dir / "metrics" / "atypicality_v1_boot.csv", index=False)

    # Карты: ступени + ансамбль + критериальный (ранги для сопоставимой шкалы)
    from scipy import stats as sps
    plot_names = ladder_names + ["criterial"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    prow = df["row"].to_numpy()[all_cells]
    pcol = df["col"].to_numpy()[all_cells]
    for ax, name in zip(axes.ravel(), plot_names):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        img = grid.reshape(meta.prf, meta.pic)
        im = ax.imshow(img, cmap="magma", vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=12, c="cyan", marker="^", label="точки заверки")
        ax.set_title(name)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot_names):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Лестница нетипичности (v1): нормированные ранги скора, север сверху", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "atypicality_v1_map.png", dpi=130, bbox_inches="tight")

    # Сводка в консоль: lift@10%
    summary = (pa_df[pa_df["area"] == config.VER_AREA]
               .pivot(index="method", columns="points", values="lift")
               .loc[[n for n in scores]]
               .round(2))
    print(f"\nlift@{config.VER_AREA:.0%} (пул = валидные ячейки):")
    print(summary.to_string())
    print(f"\n[{time.time()-t0:3.0f}s] сохранено: outputs/atypicality_v1_map.png, "
          "outputs/metrics/atypicality_v1{,_boot}.csv")


if __name__ == "__main__":
    main()
