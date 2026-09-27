"""Прогон минерагенического набора по протоколу полос 4+4.

Что считается. Датасет `dataset_mineragenic_full.parquet` (54 признака:
7 организации + 13 факторных слоёв + 34 нашей докачки) прогоняется той же
схемой, что и все предыдущие сравнения: 4 полосы по X и 4 по Y, разрыв
`STRIP_BUFFER_M` = 15 км между обучением и тестовой полосой, калибровка
предсказаний по обучающему распределению, итог — среднее схем X и Y. Ничего
в протоколе под этот набор не меняется, иначе цифры перестанут быть
сопоставимыми с уже посчитанными.

Обе формы модели считаются рядом: одиночный бустинг (им посчитаны все прежние
сравнения наборов) и производственный `HistBackgroundEnsemble` — их
эквивалентность измерена в `honest_model_form.py` (+0.002 при разбросе по
фолдам ±0.09), и здесь это проверяется ещё раз уже на другом наборе.

Как читать результат. В наборе есть факторные слои — прямые входы критериальной
формулы, а мишень взята из её результата. Поэтому высокое согласие здесь НЕ
означает удачный прогноз: измеренный потолок такого рода наборов — 70.9%
(`honest_raise_agreement.py`). Значимы три вещи: (1) доля тренда R2 — не
выродилась ли карта в пятно, (2) заверка по точкам, где факторные слои
преимущества не дают автоматически, (3) сравнение с 58 признаками без
факторных слоёв (27.94%, ROC-AUC 0.8034).

Выход: `outputs/metrics/honest_mineragenic_run.csv`,
`outputs/metrics/honest_mineragenic_pa.csv`,
`outputs/metrics/honest_mineragenic_verdict.csv`,
`outputs/honest_mineragenic_map.png`.

Запуск из корня репозитория: ``python -m experiments.honest_mineragenic_run [full|org]``
(по умолчанию `full`; вариант `org` считает набор без нашей докачки).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import (  # noqa: E402
    N_STRIPS, STRIP_BUFFER_M, make_model,
)
from experiments.honest_forecast_final import trend_share  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402
from experiments.honest_forecast_main import strip_ids  # noqa: E402

#: Какую версию минерагенического набора считать: `full` — 7 признаков
#: организации + 13 факторных слоёв + 34 нашей докачки, `org` — те же 7 + 13
#: без докачки. Разница между прогонами и есть вклад самостоятельной докачки.
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "full"
DATASET = config.PROCESSED_DIR / f"dataset_mineragenic_{VARIANT}.parquet"
TAG = "honest_mineragenic" + ("" if VARIANT == "full" else f"_{VARIANT}")
ID_COLS = ("row", "col", "x", "y")

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def blind(X, y, coord, factory):
    """Калиброванный прогноз по полосам одной оси: полоса предсказана извне."""
    sid, edges = strip_ids(coord)
    cal = np.full(len(y), np.nan)
    au = []
    for k in range(N_STRIPS):
        te = sid == k
        tr = (coord < edges[k] - STRIP_BUFFER_M) | (coord > edges[k + 1] + STRIP_BUFFER_M)
        m = factory().fit(X[tr], y[tr])
        p_te = m.predict_proba(X[te])[:, 1]
        p_tr = np.sort(m.predict_proba(X[tr])[:, 1])
        cal[te] = np.searchsorted(p_tr, p_te, side="left") / p_tr.size
        au.append(roc_auc_score(y[te], p_te))
    return cal, au


def main() -> None:
    meta, prognoz_grid = criterial_target.load_prognoz_grid()
    prognoz = prognoz_grid.ravel()
    y = criterial_target.training_positives(prognoz_grid).astype(int)
    df = pd.read_parquet(DATASET)
    cols = [c for c in df.columns if c not in ID_COLS]
    X = df[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(criterial_target.training_features("v5_rebuilt"))
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    basis = np.column_stack([np.ones(len(prognoz)), rbf_basis(cx, cy, RBF_GRID)])
    top_c = prognoz <= np.quantile(prognoz[pool], config.CREF_AREA)
    log(f"{DATASET.name}: {len(cols)} признаков, позитивов {int(y.sum())}, "
        f"полосы {N_STRIPS}/ось, разрыв {STRIP_BUFFER_M / 1000:.0f} км")

    forms = {"одиночный бустинг": make_model,
             "ансамбль по фону (прод)": lambda: HistBackgroundEnsemble(
                 random_state=config.CRIT_SEED)}
    rows, scores = [], {"criterial": -prognoz}
    for tag, factory in forms.items():
        t0 = time.time()
        sx, aux = blind(X, y, cx, factory)
        sy, auy = blind(X, y, cy, factory)
        s = (sx + sy) / 2
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        agr = crit_reference.agreement(s, crit_reference.criterial_score(prognoz),
                                      pool, config.CREF_AREA)
        au = aux + auy
        rows.append({
            "модель": tag, "признаков": len(cols),
            "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
            "roc_auc_сплошной": roc_auc_score(y[pool], s[pool]),
            "pr_auc_сплошной": average_precision_score(y[pool], s[pool]),
            "roc_auc_фолды": float(np.mean(au)), "roc_auc_std": float(np.std(au)),
            "доля_тренда_R2": trend_share(s, basis, valid),
            "capture": agr["capture"], "kappa": agr["kappa"], "rho": agr["rho"],
            "ячеек_новых": int((top_m & ~top_c & valid).sum()),
            "секунд": round(time.time() - t0, 1),
        })
        scores[f"минерагенический_{tag.split()[0]}"] = s
        log(f"  {tag:26s} совпадение={rows[-1]['совпадение_верх_10']:.1%} "
            f"ROC-AUC={rows[-1]['roc_auc_сплошной']:.4f} "
            f"тренд R2={rows[-1]['доля_тренда_R2']:.4f} ({rows[-1]['секунд']:.0f} c)")

    out = pd.DataFrame({"row": df["row"], "col": df["col"], "x": cx, "y": cy,
                        "prognoz": prognoz, "valid": valid,
                        **{k: v for k, v in scores.items() if k != "criterial"}})
    out.to_parquet(config.PROCESSED_DIR / f"{TAG}_scores.parquet", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_run.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Минерагенический набор, полосы 4+4 с разрывом 15 км ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("справочно: 58 признаков без факторных слоёв — 27.94%, ROC-AUC 0.8034; "
          "потолок только на 13 факторных слоях — 70.93%, ROC-AUC 0.9672")

    own = [k for k in scores if k != "criterial"]
    prot = assessment.run_protocol(scores, own, meta, valid, 0.05 / len(own),
                                   baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_verdict.csv",
                           index=False, encoding="utf-8-sig")
    prot["pa"].to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_pa.csv",
                      index=False, encoding="utf-8-sig")
    print("\n=== Заверка по точкам ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = res.loc[res["roc_auc_сплошной"].idxmax(), "модель"]
    s = scores[f"минерагенический_{best.split()[0]}"]
    shape = (meta.prf, meta.pic)
    ext = [meta.x0, meta.x0 + meta.pic * meta.dx, meta.y0, meta.y_top]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    axes[0].imshow(np.where(valid, prognoz, np.nan).reshape(shape), origin="upper",
                   cmap="magma_r", extent=ext)
    axes[0].set_title("Критериальный метод")
    axes[1].imshow(np.where(valid, s, np.nan).reshape(shape), origin="upper",
                   cmap="magma", extent=ext)
    axes[1].set_title(f"Минерагенические признаки, {len(cols)} шт. ({best})\n"
                      "обучение не ближе 15 км от каждой точки")
    for ax in axes:
        ax.set_xlabel("X, м")
    axes[0].set_ylabel("Y, м")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"{TAG}_map.png", dpi=130)
    log(f"OK: outputs/{TAG}_map.png")


if __name__ == "__main__":
    main()
