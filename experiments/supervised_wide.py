"""Перенос лучшего метода с учителем на широкую сетку ГГК-200 (303 x 299).

ЗАЧЕМ. В прогоне `supervised_sweep` лучшей по обеим метрикам оказалась одна и та
же карта: голосование по рангам случайного леса, гистограммного бустинга и
перцептрона на входах «сырые 37 признаков + скользящие средние в окнах 5x5 и
15x15» (ROC-AUC 0.769 на разрыве 15 км, совпадение с критериальной картой по
верхним 10% — 33.4%). Здесь эта же модель обучается на всём основном листе
сразу и применяется к смежной территории.

ОГОВОРКА. Обучение на всём листе — это не валидация, а получение рабочей модели
для площади, где критериального расчёта нет и проверить обобщение нечем (та же
оговорка, что в `forecast_wide.py` и `nn_geo_mpm.py::wide_extend`). Качество
модели меряется прогоном `supervised_sweep`, а не этой картой.

Скользящие средние на широкой сетке считаются по самой широкой сетке (окно
физически то же — 2.5 и 7.5 км), а нормировки признаков модель не использует:
деревья к шкале нечувствительны, перцептрон получает квантильное преобразование,
подогнанное на листе.

Выход: `data/processed/forecast_wide_supervised.parquet`,
       `outputs/supervised_wide.png`.
Запуск из корня: ``python -m experiments.supervised_wide``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target  # noqa: E402
from experiments.nn_geo_mpm import log  # noqa: E402
from experiments.nn_geo_mpm_control import WINDOWS, smoothed  # noqa: E402
from experiments import supervised_sweep as sw  # noqa: E402


def main() -> None:
    from experiments.fetch_wide_area import WIDE_META

    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    sw.Y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    cols = [c for c in spec["признаки_перенос"] if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)
    pool = np.flatnonzero(cell_mask.build_valid_mask(feat_all))

    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    Xw = wide[cols].to_numpy(dtype=float)
    log(f"лист: {X.shape}, обучающих ячеек {pool.size} ({sw.Y[pool].sum()} позитивов); "
        f"широкая сетка: {Xw.shape}")

    ctx = np.hstack([X] + [smoothed(X, shape, w) for w in WINDOWS])
    ctx_w = np.hstack([Xw] + [smoothed(Xw, WIDE_META.shape, w) for w in WINDOWS])
    log(f"входы с окрестностью: {ctx.shape[1]} столбцов")

    # Одна общая матрица: строки листа (обучение) + строки широкой сетки (прогноз).
    both = np.vstack([ctx, ctx_w])
    allx = np.arange(len(both))
    score = sw.fit_vote(pool, allx, X=both)[len(ctx):]
    log("голосование трёх методов обучено на всём листе и применено к широкой сетке")

    out = wide[["row", "col", "x", "y"]].copy()
    out["score"] = score.astype(np.float32)
    out.to_parquet(config.PROCESSED_DIR / "forecast_wide_supervised.parquet", index=False)
    log(f"OK: data/processed/forecast_wide_supervised.parquet {out.shape}")

    inside = ((out["x"] >= meta.x0) & (out["x"] <= meta.x0 + meta.pic * meta.dx)
              & (out["y"] >= meta.y0) & (out["y"] <= meta.y_top)).to_numpy()
    thr = np.quantile(score, 1 - config.CREF_AREA)
    log(f"доля верхних 10% внутри контура листа {(score[inside] >= thr).mean():.1%}, "
        f"вне контура {(score[~inside] >= thr).mean():.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pct = (pd.Series(score).rank(pct=True).to_numpy() * 100).reshape(WIDE_META.shape)
    ext = [WIDE_META.x0, WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx,
           WIDE_META.y0, WIDE_META.y_top]
    fig, ax = plt.subplots(figsize=(9, 9.5))
    im = ax.imshow(pct, origin="upper", cmap="viridis", extent=ext)
    ax.contour(pct, levels=[90], colors="cyan", linewidths=1.2, extent=ext,
               origin="upper")
    ax.plot([meta.x0, meta.x0 + meta.pic * meta.dx, meta.x0 + meta.pic * meta.dx,
             meta.x0, meta.x0],
            [meta.y0, meta.y0, meta.y_top, meta.y_top, meta.y0],
            color="white", lw=1.6, label="контур основного листа (обучение)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Перенос лучшего метода с учителем на смежную территорию\n"
                 "голосование леса, бустинга и перцептрона; "
                 f"{len(cols)} признаков набора переноса + окрестность\n"
                 "за контуром листа критериального расчёта нет: "
                 "это экстраполяция, а не проверенный прогноз")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    fig.colorbar(im, ax=ax, label="место ячейки в ряду всех ячеек, %", fraction=0.03)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "supervised_wide.png", dpi=130)
    log("OK: outputs/supervised_wide.png")


if __name__ == "__main__":
    main()
