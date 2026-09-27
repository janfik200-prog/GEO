"""Шумовой пол `nn_geo_mpm`: устойчивы ли выводы к сиду инициализации сети.

ЗАЧЕМ. Вывод прогона `nn_geo_mpm` держится на одном обучении каждой сети. У
нейросети случайны и начальные веса, и порядок примеров, и прореживание
(dropout), а обучающая выборка мала (2295 позитивов). Разница «графовая сеть
0.799 против бустинга 0.749» имеет смысл, только если она больше разброса от
одного лишь сида — иначе это не свойство архитектуры, а рябь.

Считаются те же две сети на тех же полосах с тем же разрывом 15 км, но с
четырьмя сидами. Отчёт даёт по каждой архитектуре: среднее ROC-AUC по фолдам
для каждого сида, разброс между сидами и то же для совпадения верх-10%.

Выход: `outputs/metrics/nn_geo_mpm_seeds.csv`.
Запуск из корня: ``python -m experiments.nn_geo_mpm_seeds``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target  # noqa: E402
from experiments.nn_geo_mpm import (blind, grid_edges, log, norm_adj,  # noqa: E402
                                    normalize, predict_cnn, robust_stats,
                                    stack_image, train_cnn, train_gcn)

SEEDS = (42, 0, 1, 7)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    cols = [c for c in spec["признаки_перенос"] if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    top_c = prognoz.ravel() <= np.quantile(prognoz.ravel()[pool], config.CREF_AREA)
    A = norm_adj(*grid_edges(shape, valid), y.size)

    rows = []
    for seed in SEEDS:
        for arch in ("граф GCN, соседство на сетке", "свёрточная, патч 15x15"):
            def fp(tr, allx, seed=seed, arch=arch):
                med, iqr = robust_stats(X, tr)
                Z = normalize(X, med, iqr)
                if arch.startswith("граф"):
                    return train_gcn(Z, y, tr, A, seed=seed)[allx]
                img = stack_image(Z, shape)
                m = train_cnn(img, y, tr, 7, meta.pic, seed=seed)
                return predict_cnn(m, img, allx, 7, meta.pic)

            cals, aucs = [], []
            for ax, coord in (("X", cx), ("Y", cy)):
                log(f"сид {seed}, {arch}, ось {ax}")
                _, cal, fr = blind(fp, coord, y, pool, f"{arch}|{seed}", ax)
                cals.append(cal)
                aucs += [r["roc_auc"] for r in fr]
            s = np.nanmean(np.vstack(cals), axis=0)
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
            top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
            rows.append({"архитектура": arch, "сид": seed,
                         "фолды_roc_auc": float(np.mean(aucs)),
                         "фолды_roc_std": float(np.std(aucs)),
                         "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                         "согласие_каппа": agr["kappa"],
                         "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1)})
            log(f"  итог: фолды {rows[-1]['фолды_roc_auc']:.4f}, "
                f"совпадение {rows[-1]['совпадение_верх_10']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm_seeds.csv", index=False)
    print("\n=== Разброс по сидам (тот же протокол полос, разрыв 15 км) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    agg = res.groupby("архитектура").agg(
        roc_среднее=("фолды_roc_auc", "mean"), roc_разброс=("фолды_roc_auc", "std"),
        roc_мин=("фолды_roc_auc", "min"), roc_макс=("фолды_roc_auc", "max"),
        совп_среднее=("совпадение_верх_10", "mean"),
        совп_разброс=("совпадение_верх_10", "std"))
    print("\n=== Сводка ===")
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nДля сравнения планки из nn_geo_mpm.csv: бустинг 0.7488, "
          "ТОЛЬКО география 0.6753 (по фолдам)")


if __name__ == "__main__":
    main()
