"""Задача 3 на dataset_v5_rebuilt: тот же протокол, пересобранный пул признаков.

``experiments/criterial_target_v5.py`` (167 признаков, ``dataset_v5.parquet``)
показал первый результат на v5. Отдельно колхоз-коллега прислал свою чистку
(``dataset_v5_final.parquet``, 79 признаков) — ревью ``model-critic``
(14.08.2026) не подтвердило доверие к её шумовому фильтру (Boruta на
leave-one-object-out БЕЗ пространственного буфера — том же CV, где объект 3
«ловится» только за счёт близости <15 км к объекту 1: при буфере ≥20 км
lift объекта 3 = 0.0× во всех 5 сидах, ``outputs/metrics/crit_buffer_sensitivity.csv``).

``dataset_v5_rebuilt.parquet`` (``data/processed/dataset_v5_rebuilt_notes.md``)
восстанавливает только target-free корреляционную чистку (пересчитана
самостоятельно, с проверкой прямой, а не транзитивной корреляции) и
возвращает все признаки, отсеянные ненадёжным шумовым фильтром.

Протокол не меняется (тот же ``src.criterial_target``, тот же буфер, тот же
перестановочный тест и сид-свип) — меняется только признаковое пространство,
это прямая проверка, повлиял ли объём чистки коллеги на результат.

Запуск из корня: ``python -X utf8 -m experiments.criterial_target_v5_rebuilt``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target  # noqa: E402

DATASET = "v5_rebuilt"


def main() -> None:
    X, labels_flat, coords, feature_names, meta = criterial_target.build_dataset(DATASET)
    print(f"признаков: {len(feature_names)} (dataset_{DATASET}, группы geo/geo2 исключены)")
    for obj in range(1, config.CRIT_N_OBJECTS + 1):
        print(f"объект {obj}: {(labels_flat == obj).sum()} ячеек")

    # --- 1. Буферная чувствительность (5..30 км) ---
    buf_df = criterial_target.buffer_sensitivity(dataset=DATASET)
    buf_summary = criterial_target.summarize_buffer_sensitivity(buf_df)
    print("\n--- 1. Буферная чувствительность (медиана lift@10%, 5 сидов) ---")
    print(buf_summary.to_string(index=False))

    # --- 2. Честный протокол: LOO-CV на буфере по умолчанию (10 км) + permutation ---
    summary_df, loo_results, perm_df = criterial_target.run_leave_one_object_out_cv(
        seed=config.CRIT_SEED, dataset=DATASET,
    )
    print("\n--- 2. LOO-CV, буфер 10 км (протокольное значение) ---")
    print(summary_df[["object", "n_cells", "n_train_pos", "coverage@10%", "lift@10%"]]
          .to_string(index=False))
    print("\nПерестановочный тест (форм-сохраняющие размещения, area=10%):")
    print(perm_df[["object", "observed_lift", "null_mean", "null_q95", "p_value"]]
          .round(3).to_string(index=False))

    # --- 3. Сид-свип (20 сидов) ---
    sweep_df = criterial_target.seed_sweep(dataset=DATASET)
    sweep_summary = criterial_target.summarize_sweep(sweep_df)
    print(f"\n--- 3. Сид-свип ({config.CRIT_SWEEP_N_SEEDS} сидов), lift@10% ---")
    print(sweep_summary.to_string(index=False))

    # --- 4. Важность признаков по фолдам (5 сидов, буфер 10 км) ---
    imp_df = criterial_target.fold_feature_importances(n_seeds=5, dataset=DATASET)
    top10_per_fold = {c: set(imp_df[c].sort_values(ascending=False).head(10).index)
                       for c in imp_df.columns}
    common = set.intersection(*top10_per_fold.values())
    print(f"\n--- 4. Топ-10 важных признаков по фолдам: общих для всех трёх — {len(common)} ---")
    print(imp_df.head(15).round(4).to_string())
    print(f"общие признаки: {sorted(common)}")

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    buf_summary.to_csv(out_dir / "criterial_target_v5_rebuilt_buffer.csv", index=False)
    summary_df.to_csv(out_dir / "criterial_target_v5_rebuilt_loo.csv", index=False)
    perm_df.to_csv(out_dir / "criterial_target_v5_rebuilt_perm.csv", index=False)
    sweep_summary.to_csv(out_dir / "criterial_target_v5_rebuilt_sweep.csv", index=False)
    imp_df.to_csv(out_dir / "criterial_target_v5_rebuilt_importance.csv")
    print(f"\nсохранено: {out_dir}/criterial_target_v5_rebuilt_*.csv")

    # --- 5. Карта листа: скор каждого LOO-фолда (обучен на 2 объектах, третий — контроль) ---
    import matplotlib.pyplot as plt

    invalid = ~np.isfinite(criterial_target.load_prognoz_grid()[1].ravel())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, r in zip(axes, loo_results):
        obj = r["summary"]["object"]
        img = r["score_all"].reshape(meta.prf, meta.pic).astype(float).copy()
        img.ravel()[invalid] = np.nan
        im = ax.imshow(img, cmap="viridis", vmin=0, vmax=1)
        held_rows, held_cols = np.divmod(r["held_idx"], meta.pic)
        ax.scatter(held_cols, held_rows, marker="x", color="red", s=10, linewidths=1,
                   label="held-out объект (контроль)")
        ax.set_title(f"Объект {obj} held-out\nlift@10% = {r['summary']['lift@10%']:.2f}",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75, label="скор модели (вероятность «рудный объект»)")
    fig.suptitle(f"Задача 3 на dataset_v5_rebuilt ({len(feature_names)} независимых признаков): "
                f"leave-one-object-out, буфер {config.CRIT_HOLDOUT_BUFFER_M / 1000:.0f} км "
                "(красные точки — held-out объект)", fontsize=10)
    map_path = ROOT / "outputs" / "criterial_target_v5_rebuilt_map.png"
    fig.savefig(map_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {map_path}")


if __name__ == "__main__":
    main()
