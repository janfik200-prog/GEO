"""Предел обнаружения лестницы нетипичности: вживление синтетического рудного тела.

Зачем: заверка на 19 точках не различает методы (сдвиговый null не значим ни у
кого, включая критериальный). Прежде чем наращивать признаки, надо ответить на
более базовый вопрос — СПОСОБНЫ ли детекторы без учителя в принципе найти
компактную аномалию в этих данных, и какой контраст для этого нужен.

Метод (positive control): в реальную матрицу признаков вживляется компактная
группа ячеек (``POW_BODY_CELLS`` штук, соседних по сетке), у которых
``POW_N_FEATURES`` случайных признаков сдвинуты на заданный контраст в
робастных единицах (IQR). Детекторы обучаются на загрязнённых данных (как в
реальности — руда внутри выборки, а не снаружи) и оцениваются по двум метрикам:

* ``rank_median`` — медианный нормированный ранг ячеек тела (1.0 = все в самом
  верху карты);
* ``det10`` — доля ячеек тела, попавших в top-10% скора.

Для привязки к реальности печатается наблюдаемый контраст: насколько по тем же
робастным единицам отличаются от фона объекты критериального прогноза и ячейки
несмещённых точек заверки.

Проверяются и другие семейства детекторов (LOF, расстояние до k-го соседа,
смесь гауссиан) — чтобы отличить «не работает семейство» от «нет сигнала».

Выход: ``outputs/metrics/atypicality_power.csv``, ``outputs/atypicality_power.png``.
Запуск из корня: ``python -m experiments.atypicality_power``.
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

DETECTORS = [
    ("mahalanobis", atypicality.robust_mahalanobis),
    ("pca_residual", atypicality.pca_residual),
    ("isoforest", atypicality.isoforest_score),
    ("ocsvm", atypicality.ocsvm_score),
    ("shallow_ae", atypicality.shallow_ae_score),
    ("lof", atypicality.lof_score),
    ("knn_dist", atypicality.knn_distance_score),
    ("gmm_nll", atypicality.gmm_nll_score),
    # Масштаб соседства против размера тела: при k < размера тела локальные
    # методы видят внутри тела «своих» соседей и объявляют его нормой.
    # k = 100 > POW_BODY_CELLS = 30 — прямая проверка этого объяснения.
    ("lof_k100", lambda X: atypicality.lof_score(X, k=100)),
    ("knn_k100", lambda X: atypicality.knn_distance_score(X, k=100)),
]


def plant_body(rows: np.ndarray, cols: np.ndarray, rng: np.random.Generator,
               n_cells: int) -> np.ndarray:
    """Индексы (в пространстве валидных ячеек) компактного тела из n_cells ячеек."""
    c = rng.integers(rows.size)
    d2 = (rows - rows[c]) ** 2 + (cols - cols[c]) ** 2
    return np.argsort(d2)[:n_cells]


def observed_contrast(X: np.ndarray, cells: np.ndarray) -> float:
    """Наблюдаемый контраст группы ячеек: медиана по признакам |median z|."""
    if cells.size == 0:
        return float("nan")
    return float(np.median(np.abs(np.median(X[cells], axis=0))))


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:4.0f}s] {msg}", flush=True)

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    valid = cell_mask.build_valid_mask(df)
    feat = features_v11.ladder_features(df, include_dist=False)
    X0 = atypicality.prepare_matrix(feat, valid)
    rows = df["row"].to_numpy()[valid]
    cols = df["col"].to_numpy()[valid]
    n, d = X0.shape
    log(f"матрица {n} x {d} (валидные ячейки, признаки v1.1)")

    # --- Наблюдаемый контраст реальных целей (привязка шкалы) ---
    pos_in_valid = np.full(len(df), -1)
    pos_in_valid[np.flatnonzero(valid)] = np.arange(n)
    meta, prognoz = criterial_target.load_prognoz_grid()
    labels = criterial_target.label_ore_objects(prognoz).ravel()
    for obj in range(1, config.CRIT_N_OBJECTS + 1):
        cells = pos_in_valid[np.flatnonzero(labels == obj)]
        cells = cells[cells >= 0]
        log(f"объект критериального #{obj}: {cells.size} ячеек, "
            f"наблюдаемый контраст {observed_contrast(X0, cells):.2f} IQR")
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    unb_v = pos_in_valid[unb]
    log(f"несмещённые точки заверки: {unb_v.size} ячеек, "
        f"наблюдаемый контраст {observed_contrast(X0, unb_v):.2f} IQR")

    # --- Вживление тела и прогон детекторов ---
    rng = np.random.default_rng(config.POW_SEED)
    rows_out = []
    for seed_i in range(config.POW_N_SEEDS):
        body = plant_body(rows, cols, rng, config.POW_BODY_CELLS)
        feats = rng.choice(d, size=config.POW_N_FEATURES, replace=False)
        signs = rng.choice([-1.0, 1.0], size=config.POW_N_FEATURES)
        for contrast in config.POW_CONTRASTS:
            X = X0.copy()
            X[np.ix_(body, feats)] += contrast * signs
            for name, fn in DETECTORS:
                s = fn(X)
                r = sps.rankdata(s) / s.size
                thr = np.quantile(s, 0.90)
                rows_out.append({
                    "seed": seed_i, "contrast": contrast, "detector": name,
                    "rank_median": float(np.median(r[body])),
                    "det10": float((s[body] >= thr).mean()),
                })
            log(f"размещение {seed_i}, контраст {contrast:.0f} IQR — готово")
    res = pd.DataFrame(rows_out)

    agg = (res.groupby(["detector", "contrast"])[["rank_median", "det10"]]
           .median().reset_index())
    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "metrics" / "atypicality_power.csv", index=False)

    # --- График: доля тела в top-10% против контраста ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, _ in DETECTORS:
        g = agg[agg["detector"] == name]
        axes[0].plot(g["contrast"], g["det10"], marker="o", label=name)
        axes[1].plot(g["contrast"], g["rank_median"], marker="o", label=name)
    for ax, ylab, ref in [(axes[0], "доля тела в top-10% карты", 0.10),
                          (axes[1], "медианный нормированный ранг тела", 0.90)]:
        ax.axhline(ref, color="gray", ls="--", lw=1,
                   label="случайный уровень" if ref == 0.10 else "порог уверенного детекта")
        ax.set_xlabel("контраст сигнатуры, робастных IQR")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Предел обнаружения детекторов без учителя: тело "
                 f"{config.POW_BODY_CELLS} ячеек, сигнатура по "
                 f"{config.POW_N_FEATURES} признакам из {d} "
                 f"(медиана по {config.POW_N_SEEDS} размещениям)")
    fig.tight_layout()
    fig.savefig(out_dir / "atypicality_power.png", dpi=130, bbox_inches="tight")

    print("\nДоля тела в top-10% карты (медиана по размещениям):")
    print(agg.pivot(index="detector", columns="contrast", values="det10")
          .round(2).to_string())
    print("\nМедианный нормированный ранг тела:")
    print(agg.pivot(index="detector", columns="contrast", values="rank_median")
          .round(2).to_string())
    log("сохранено: outputs/atypicality_power.png, outputs/metrics/atypicality_power.csv")


if __name__ == "__main__":
    main()
