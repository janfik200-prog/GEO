"""Почему деления на обучение и проверку мало при 47 точках.

ЗАЧЕМ. Возник законный вопрос: выборку ведь можно просто поделить на
обучающую и проверочную — в чём тогда трудность? Здесь это проверяется
измерением, тремя счётами:

1. Одно и то же обучение при случайном делении ячеек и при делении по
   площади с разрывом. Если случайное деление даёт заметно выше, значит оно
   меряет не прогноз, а доучивание по соседней ячейке того же пятна.
2. Сколько среди 47 точек независимых скоплений. Точки опробования
   сгруппированы по маршрутам, поэтому число независимых наблюдений меньше
   числа точек, и проверка «одна точка против остальных» частью опирается на
   соседей той же группы.
3. Какой ширины доверительный интервал у доли точек, попавших в верхние 10%
   площади, при таком числе наблюдений (интервал Уилсона). Если интервалы
   наборов перекрываются, разница между наборами не измерена, сколько ни дели
   выборку.

ВХОД: сетка листа, dataset_v5_rebuilt.parquet, слой опробования
ВЫХОД: outputs/metrics/gold_points_split_power.log и .csv
Запуск из корня: python -X utf8 -m experiments.gold_points_split_power
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.gold_points_branches import fill, gold_cells, point_folds  # noqa: E402
from experiments.honest_feature_signal import make_model  # noqa: E402
from experiments.shap_two_branches import GEOLOGY  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLUSTER_KM = 5.0
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Интервал Уилсона для доли — честные границы при малом числе наблюдений."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def top_share(p: np.ndarray, y: np.ndarray, area: float = 0.10) -> float:
    r = (np.argsort(np.argsort(p)) + 0.5) / p.size
    return float((r[y == 1] >= 1 - area).mean())


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    idx, _ = gold_cells(meta)
    y_full = np.zeros(cx.size, int)
    y_full[idx] = 1
    y = y_full[pool]
    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    Xa = fill(spectral_mask.apply(feat[dzz]).to_numpy(float)[pool])
    geo = [c for c in GEOLOGY if c in raw.columns]
    Xb = np.column_stack([Xa, fill(raw[geo].to_numpy(float)[pool])])
    log("ячеек с золотом %d из %d" % (int(y.sum()), y.size))

    # --- 1. случайное деление против деления по площади
    rows = []
    folds_sp = point_folds(cx[pool], cy[pool], y)
    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=config.CRIT_SEED)
    log("")
    log("ДЕЛЕНИЕ СЛУЧАЙНОЕ ПРОТИВ ДЕЛЕНИЯ ПО ПЛОЩАДИ (разрыв 15 км)")
    for tag, X in (("только дистанционка", Xa),
                   ("дистанционка + геология", Xb)):
        rnd = [(roc_auc_score(y[te], make_model().fit(X[tr], y[tr])
                              .predict_proba(X[te])[:, 1]),
                top_share(make_model().fit(X[tr], y[tr])
                          .predict_proba(X[te])[:, 1], y[te]))
               for tr, te in skf.split(X, y)]
        sp = [(roc_auc_score(y[te], make_model().fit(X[tr], y[tr])
                             .predict_proba(X[te])[:, 1]),
               top_share(make_model().fit(X[tr], y[tr])
                         .predict_proba(X[te])[:, 1], y[te]))
              for tr, te in folds_sp]
        ar, cr = np.mean([a for a, _ in rnd]), np.mean([c for _, c in rnd])
        asp, csp = np.mean([a for a, _ in sp]), np.mean([c for _, c in sp])
        log("   %-26s случайное: площадь под кривой %.3f, точек в верхних "
            "10%% %.0f%% | по площади: %.3f, %.0f%%"
            % (tag, ar, 100 * cr, asp, 100 * csp))
        rows.append({"набор": tag, "случайное_auc": ar, "случайное_верх10": cr,
                     "по_площади_auc": asp, "по_площади_верх10": csp})

    # --- 2. сколько независимых скоплений среди точек
    pts = np.column_stack([cx[idx], cy[idx]])
    link = sch.linkage(pdist(pts), method="single")
    lab = sch.fcluster(link, t=CLUSTER_KM * 1000, criterion="distance")
    sizes = np.bincount(lab)[1:]
    log("")
    log("СКОПЛЕНИЯ ТОЧЕК: при слиянии ближе %.0f км — %d групп из %d точек, "
        "крупнейшая %d точек, одиночек %d"
        % (CLUSTER_KM, lab.max(), idx.size, sizes.max(), int((sizes == 1).sum())))

    # --- 3. ширина интервала доли при таком числе точек
    log("")
    log("ТОЧНОСТЬ ОЦЕНКИ: доля точек в верхних 10%% площади, интервал Уилсона "
        "95%% при n=%d" % idx.size)
    for name, share in (("критериальный балл", 0.26),
                        ("дистанционка + геология", 0.21),
                        ("только дистанционка", 0.15),
                        ("случайный уровень", 0.10)):
        k = int(round(share * idx.size))
        lo, hi = wilson(k, idx.size)
        log("   %-26s %2.0f%% (%.0f–%.0f%%)" % (name, 100 * share, 100 * lo,
                                                100 * hi))
        rows.append({"набор": name, "доля_верх10": share, "низ": lo,
                     "верх": hi})
    lo_eff, hi_eff = wilson(int(round(0.26 * lab.max())), lab.max())
    log("   то же при %d независимых группах: %2.0f%% (%.0f–%.0f%%)"
        % (lab.max(), 26, 100 * lo_eff, 100 * hi_eff))

    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics"
                              / "gold_points_split_power.csv", index=False,
                              encoding="utf-8-sig")
    (ROOT / "outputs" / "metrics" / "gold_points_split_power.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
