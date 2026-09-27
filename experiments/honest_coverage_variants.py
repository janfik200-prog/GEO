"""Цена неполного покрытия: что делать с ячейками смежной территории без L8/S1.

Задача. Набор с измеренной переносимостью (58 признаков,
`honest_feature_set.json`) отбирался на основном листе, где покрытие сплошное:
`s1` не пропущен нигде, `l8` — в 0.25% ячеек. На широкой сетке всё иначе —
`l8` отсутствует в 38.1% ячеек, `s1` в 27.8%; ячеек без единого пропуска
только 61.5%. То есть 13 признаков из 58 (группа `l8`) и ещё 6 (`s1`) больше
чем на трети территории — пустые.

Почему это нельзя просто отдать бустингу. Гистограммный бустинг умеет
обрабатывать пропуски, но правило «куда отправлять NaN» он ВЫУЧИВАЕТ по
пропускам в обучающей выборке. Здесь их там практически нет, поэтому правило
не выучено, а взято по умолчанию — и на трети смежной территории треть
признаков пошла бы в произвольную ветку. Это ровно тот сорт молчаливой
поломки, из-за которого прежний перенос давал перевёрнутый прогноз: цифра
считается, карта рисуется, а под ней ничего нет.

Что меряется. Тот же протокол, что и во всём цикле: полосы 4 по X + 4 по Y с
разрывом 15 км между обучением и тестом (`honest_feature_signal.strip_folds`),
мишень — top-`CREF_AREA` критериального прогноза. Сравниваются:

* `58 признаков` — верхняя планка, полное покрытие;
* `58, но l8/s1 стёрты в тесте` — наивный перенос как есть: модель обучена на
  полных данных, а предсказывает там, где их нет;
* `45 (без l8)`, `52 (без s1)`, `39 (без l8+s1)` — модели, честно обученные
  без тех групп, которых на месте применения не будет;
* `география` — планка «модель знает только где» (64 гауссовых базиса), чтобы
  видеть, остаётся ли у обеднённого набора собственный вклад.

Вывод скрипта — не одна модель, а решение о том, сколькими моделями крыть
широкую сетку и как подписать зоны разного покрытия.

Выход: `outputs/metrics/honest_coverage_variants.csv`,
`outputs/honest_wide_coverage_map.png` (где на смежной территории какие данные
есть).

Запуск из корня репозитория: ``python -m experiments.honest_coverage_variants``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, features_v2  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (  # noqa: E402
    STRIP_BUFFER_M, evaluate, log, strip_folds,
)
from experiments.honest_final_control import rbf_basis  # noqa: E402

#: Группы, покрытие которых на широкой сетке неполное (доля пропусков там).
PATCHY = ("l8", "s1")


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = honest_features()
    df = criterial_target.training_features("v5_rebuilt")[feat]
    groups = np.array([features_v2.feature_group(c) for c in feat])
    X = df.to_numpy(dtype=float)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    folds = strip_folds(cx, cy, y, STRIP_BUFFER_M)
    log(f"основной лист: {len(feat)} признаков, {int(y.sum())} позитивов, "
        f"{len(folds)} фолдов (полосы + разрыв {STRIP_BUFFER_M/1000:.0f} км)")

    # --- покрытие широкой сетки ---
    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    have = {g: wide[[c for c in feat if features_v2.feature_group(c) == g]]
            .notna().any(axis=1).to_numpy() for g in PATCHY}
    both = have["l8"] & have["s1"]
    log(f"широкая сетка ({len(wide)} ячеек): l8 есть у {100*have['l8'].mean():.1f}%, "
        f"s1 у {100*have['s1'].mean():.1f}%, обе группы у {100*both.mean():.1f}%")

    rows = []
    rows.append(evaluate(X, y, folds, "58 признаков (полное покрытие)"))

    # Наивный перенос: обучение на полных данных, тест — там, где групп нет.
    drop = np.isin(groups, PATCHY)
    X_erased = X.copy()
    X_erased[:, drop] = np.nan
    au, pr = [], []
    from sklearn.metrics import average_precision_score, roc_auc_score
    from experiments.honest_feature_signal import make_model
    for tr, te in folds:
        p = make_model().fit(X[tr], y[tr]).predict_proba(X_erased[te])[:, 1]
        au.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
    rows.append({"набор": "58, но l8/s1 стёрты в тесте", "n_фолдов": len(au),
                 "roc_auc": float(np.mean(au)), "roc_auc_std": float(np.std(au)),
                 "pr_auc": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
                 "roc_min": float(np.min(au)), "roc_max": float(np.max(au))})
    log(f"  {'58, но l8/s1 стёрты в тесте':34s} ROC-AUC={np.mean(au):.4f} "
        f"PR-AUC={np.mean(pr):.4f}")

    # Модели, честно обученные без недостающих групп.
    for tag, drop_groups in (("45 (без l8)", ("l8",)),
                             ("52 (без s1)", ("s1",)),
                             ("39 (без l8+s1)", PATCHY)):
        keep = ~np.isin(groups, drop_groups)
        rows.append(evaluate(X[:, keep], y, folds,
                             f"{int(keep.sum())} признаков ({tag.split('(')[1][:-1]})"))

    rows.append(evaluate(rbf_basis(cx, cy, 8), y, folds, "география (64 базиса)"))

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "honest_coverage_variants.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n=== Цена неполного покрытия (полосы + разрыв 15 км) ===")
    print(out[["набор", "roc_auc", "roc_auc_std", "pr_auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nOK: {out_path}")

    # --- карта покрытия смежной территории ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    code = np.where(both, 3, np.where(have["s1"], 2, np.where(have["l8"], 1, 0)))
    fig, ax = plt.subplots(figsize=(7, 11))
    cmap = ListedColormap(["#d9d9d9", "#fdd0a2", "#9ecae1", "#31a354"])
    ax.imshow(code.reshape(WIDE_META.shape), origin="upper", cmap=cmap, vmin=0, vmax=3,
              extent=[WIDE_META.x0, WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx,
                      WIDE_META.y0, WIDE_META.y_top])
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#31a354", label=f"обе съёмки — {100*(code==3).mean():.0f}%"),
        Patch(color="#9ecae1", label=f"только радар — {100*(code==2).mean():.0f}%"),
        Patch(color="#fdd0a2", label=f"только Landsat-8 — {100*(code==1).mean():.0f}%"),
        Patch(color="#d9d9d9", label=f"ни той, ни другой — {100*(code==0).mean():.0f}%"),
    ], loc="lower left", fontsize=9)
    ax.set_title("Покрытие смежной территории съёмками,\nот которых зависит 19 признаков из 58")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    fig.tight_layout()
    fig_path = ROOT / "outputs" / "honest_wide_coverage_map.png"
    fig.savefig(fig_path, dpi=130)
    print(f"OK: {fig_path}")


if __name__ == "__main__":
    main()
