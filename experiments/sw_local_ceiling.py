"""Потолок юго-запада: берут ли дистанционные признаки цель, если учить ЗДЕСЬ.

ЗАЧЕМ. Перенос модели с листа на юго-запад не работает (согласие 0.02 при
случайном уровне 0.10), и аналоговый поиск вслепую тоже не работает. Остаётся
отделить две причины: модель не переносится — или сведений о цели в наших
признаках на юго-западе нет вообще. Проверка прямая: учим НА САМОМ
юго-западе, полосами с разрывом 15 км, и смотрим, берут ли признаки цель
лучше, чем одна голая география. Если не лучше — значит на юго-западе
критериальную разметку дистанционкой не воспроизвести, и мерить там согласие
с формулой бессмысленно.

ЧТО СЧИТАЕТСЯ. Цель — верхние 10% нашего воссоздания критериальной формулы
внутри юго-западной зоны (эталон с 04.10.2026). Наборы:
  * 37 признаков переноса (вся дистанционка);
  * 15 признаков рельефа и линеаментов (работают на всей площади);
  * только география — решётка гауссовых базисов, планка «одна география»;
  * признаки вместе с географией.
Для масштаба те же наборы считаются на листе той же цепочкой.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet,
      data/processed/criterial_reconstruct_multisheet.parquet
ВЫХОД: outputs/metrics/sw_local_ceiling.log и .csv
Запуск из корня: python -m experiments.sw_local_ceiling
"""

import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, spectral_mask  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402
from experiments.honest_final_control import rbf_basis  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
RBF_N = 10
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def evaluate(X: np.ndarray, y: np.ndarray, folds, tag: str) -> dict:
    au, pr = [], []
    for tr, te in folds:
        p = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        au.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
    res = {"набор": tag, "фолдов": len(au), "roc_auc": float(np.mean(au)),
           "roc_std": float(np.std(au)), "pr_auc": float(np.mean(pr))}
    log("   %-34s ROC-AUC %.3f±%.3f, PR-AUC %.3f (%d фолдов)"
        % (tag, res["roc_auc"], res["roc_std"], res["pr_auc"], len(au)))
    return res


def main() -> None:
    wide = pd.read_parquet(config.PROCESSED_DIR / "wide_v1_archive"
                           / "dataset_wide.parquet")
    feat = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    Xdf = spectral_mask.apply(wide[cols])
    names = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)
    terr = [n for n in names if not n.startswith("s2_") and n != "спектр_измерен"]
    jt = [names.index(n) for n in terr]
    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)

    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    dist, idx = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                         ref["y"].to_numpy()[rmsk]])).query(
        np.column_stack([wx, wy]))
    common = dist <= 375.0
    crit = np.full(wx.size, np.nan)
    crit[common] = -ref["воссоздание_вся_область"].to_numpy()[rmsk][idx[common]]
    zone = zone_of(wx, wy)

    rows = []
    for area_tag, cells in (("ЮГО-ЗАПАД", common & (zone == "юго-запад")),
                            ("ЛИСТ (для масштаба)", common & (zone == "лист"))):
        sel = np.flatnonzero(cells)
        y = np.zeros(sel.size, dtype=int)
        y[crit[sel] >= np.quantile(crit[sel], 1 - AREA)] = 1
        sx, sy = wx[sel], wy[sel]
        folds = strip_folds(sx, sy, y, GAP_M)
        G = rbf_basis(sx, sy, RBF_N)
        log("%s: ячеек %d, целевых %d, полос с разрывом %.0f км — %d"
            % (area_tag, sel.size, int(y.sum()), GAP_M / 1000, len(folds)))
        full = X[sel]
        ok = np.isfinite(full).all(axis=1)
        log("   ячеек с полным набором всех признаков: %d (%.0f%%)"
            % (ok.sum(), 100 * ok.mean()))
        for tag, M, sub in (("все признаки переноса (37)", full, ok),
                            ("рельеф и линеаменты (15)", X[sel][:, jt], None),
                            ("только география", G, None),
                            ("рельеф, линеаменты + география",
                             np.column_stack([X[sel][:, jt], G]), None)):
            if sub is None:
                r = evaluate(M, y, folds, tag)
            else:
                sub_folds = [(tr & sub, te & sub) for tr, te in folds
                             if y[tr & sub].sum() >= 20 and y[te & sub].sum() >= 5]
                if not sub_folds:
                    log("   %-34s пропущен: мало ячеек с полным набором" % tag)
                    continue
                r = evaluate(M, y, sub_folds, tag)
            r["площадь"] = area_tag
            rows.append(r)

    out = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out / "sw_local_ceiling.csv", index=False,
                              encoding="utf-8-sig")
    (out / "sw_local_ceiling.log").write_text("\n".join(LOG), encoding="utf-8")
    log("готово")


if __name__ == "__main__":
    main()
