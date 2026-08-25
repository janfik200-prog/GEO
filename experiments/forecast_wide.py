"""Продакшн-модель на всех 3 объектах (без held-out) и её применение к смежной территории.

В отличие от :func:`src.criterial_target.leave_one_object_out` (объект-level
CV, нужна для проверки обобщения), здесь модель обучается на ВСЕХ трёх
объектах сразу — это не валидация, а получение единственной прод-модели для
скоринга сетки, на которой критериальный расчёт не проводился (нет `prognoz`,
нет способа проверить обобщение честно). Прогноз стоит читать с той же
оговоркой, что и вывод задачи 3 (`CLAUDE.md`): при пространственно честном
буфере lift = 0 на held-out объектах на исходном листе — модель здесь
экстраполирует паттерн обучающих объектов, а не проверенный вне выборки сигнал.

Признаки — `training_features(dataset="v5_rebuilt")` БЕЗ групп `ast`/`ls`:
на широкой сетке `ast_*` покрыт на ~14% ячеек (редкие пролёты ASTER),
`ls_*` — только северная полоса (легаси Landsat 7 fragmenta), с ними
скоринг терял бы >99% площади смежной территории (пересечение NaN по всем
признакам). Это ослабляет модель (обучающий пул был 135 признаков, здесь —
подмножество без ast(26)/ls(6)), поэтому это ОТДЕЛЬНАЯ модель, не то же
дерево, что в `dataset_v5_rebuilt`-прогонах задачи 3.

Выход: `data/processed/forecast_wide.parquet` (row/col/x/y/score),
`outputs/forecast_wide_map.png`.

Запуск из корня репозитория: ``python -m experiments.forecast_wide``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, criterial_target, features_v2  # noqa: E402
from src.model import BackgroundEnsemble  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

DROP_GROUPS = ("ast", "ls")


def main() -> None:
    X, labels_flat, coords, feat_names, meta = criterial_target.build_dataset(dataset="v5_rebuilt")
    keep_idx = [i for i, c in enumerate(feat_names) if features_v2.feature_group(c) not in DROP_GROUPS]
    feat_names = [feat_names[i] for i in keep_idx]
    X = X[:, keep_idx]
    print(f"Обучающих признаков (без {DROP_GROUPS}): {len(feat_names)}")

    pos_idx = np.where(labels_flat > 0)[0]
    bg_candidates = np.where(labels_flat == 0)[0]
    rng = np.random.default_rng(config.CRIT_SEED)
    n_bg = min(config.CRIT_N_BACKGROUND, len(bg_candidates))
    bg_idx = rng.choice(bg_candidates, size=n_bg, replace=False)
    train_idx = np.concatenate([pos_idx, bg_idx])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(bg_idx))]).astype(int)
    print(f"Обучение: {len(pos_idx)} позитивов (все 3 объекта), {len(bg_idx)} фона")

    model = BackgroundEnsemble(random_state=config.CRIT_SEED)
    model.fit(X[train_idx], y)

    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
    print("Топ-10 признаков по важности:")
    print(imp.head(10).to_string())

    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    missing = [c for c in feat_names if c not in wide.columns]
    if missing:
        raise RuntimeError(f"нет в dataset_wide.parquet: {missing}")
    X_wide = wide[feat_names].fillna(0).to_numpy(dtype=float)
    score = model.predict_proba(X_wide)[:, 1]

    out = wide[["row", "col", "x", "y"]].copy()
    out["score"] = score.astype(np.float32)
    out_path = config.PROCESSED_DIR / "forecast_wide.parquet"
    out.to_parquet(out_path, index=False)
    print(f"OK: {out_path} {out.shape}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = score.reshape(WIDE_META.shape)
    fig, ax = plt.subplots(figsize=(9, 14))
    im = ax.imshow(grid, origin="upper", cmap="viridis",
                    extent=[WIDE_META.x0, WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx,
                            WIDE_META.y0, WIDE_META.y_top])
    ax.set_title("Прогноз модели (без ast/ls), обучение на всех 3 объектах\n"
                  "смежная территория — экстраполяция, не проверена вне выборки")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    fig.colorbar(im, ax=ax, label="score (вероятность класса «объект»)", fraction=0.03)
    fig.tight_layout()
    fig_path = config.PROJECT_ROOT / "outputs" / "forecast_wide_map.png"
    fig.savefig(fig_path, dpi=130)
    print(f"OK: {fig_path}")


if __name__ == "__main__":
    main()
