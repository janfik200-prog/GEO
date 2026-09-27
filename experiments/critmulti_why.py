"""Почему наш перенос расходится с критериальной картой за рамкой листа.

ЗАЧЕМ. 24.09.2026 впервые появилась критериальная карта за пределами основного
листа (`by_crt`, `criterial_multisheet_compare.py`). Сравнение идёт по всей
области критериального расчёта, включая южную часть (критериальный выход там
есть, территории должны быть сопоставимы), поэтому признаки берутся с архивной
сетки `wide_v1_archive/dataset_wide.parquet` — она накрывает область расчёта
целиком. Смежная площадь — всё вне габаритов листа, 14.5 тыс. ячеек. Согласие
верхних 10% там 0.015 как есть и 0.067 после снятия наклона при случайном
уровне 0.10, ранговая корреляция -0.149, тогда как внутри листа 0.232/0.340
при потолке самого метода 0.422/0.430. Расхождение может значить три разные
вещи, и лечатся они по-разному:

1. **Переноса нет** — нужная информация в наших признаках есть, но модель,
   обученная на листе, её за рамкой не применяет (сдвиг шкал, новизна).
2. **Информации нет** — критериальная карта на юго-западе держится на факторах
   (разломы, гранитоиды, коры выветривания, палеодолины), которых в нашем
   наборе дистанционных признаков нет даже косвенно.
3. **Сдвиг, а не промах** — обе карты показывают одни и те же структуры, но со
   смещением, и метрика «совпадение ячеек» этого не прощает.

ЧТО СЧИТАЕТСЯ.

* Тест 1 (различает 1 и 2). Цель — верхние 10% критериальной карты ВНЕ листа.
  Три модели на одной полосовой проверке с разрывом 15 км прямо на
  смежной площади: наши 37 признаков переноса, только география (решётка гауссовых
  базисов) и признаки вместе с географией. Если признаки, обученные ЗДЕСЬ,
  цель не берут — дело не в переносе, а в отсутствии информации.
* Тест 2. Перенос как он есть: ROC-AUC готовой карты (`score_сглаженный`)
  против той же цели, без всякого дообучения.
* Тест 3 (различает 3). Расстояние от ядер одной карты до ближайшего ядра
  другой: если медиана порядка ширины ядра сглаживания — это смещение, если
  десятки километров — карты говорят о разном.
* Тест 4. Новизна ячеек (расстояние до листа в пространстве признаков) там,
  где методы согласны, и там, где расходятся.

ВХОД: data/SBORKA_NESKOLKO_LISTOV/ПРОГНОЗ_ВОСПРОИЗВ/, data/processed/
      wide_v1_archive/dataset_wide.parquet, forecast_arch_ядро5км.parquet
ВЫХОД: outputs/metrics/critmulti_why.log, outputs/metrics/critmulti_why.csv
Запуск из корня: ``python -m experiments.critmulti_why``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, spectral_mask  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402
from experiments.honest_final_control import rbf_basis  # noqa: E402
from experiments.criterial_multisheet_compare import (SCORE_COL,  # noqa: E402
                                                      load_criterial, zone_of)
from experiments.transfer_map import novelty  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0       # разрыв между обучающей и проверочной полосой
RBF_N = 10             # решётка базисов на южную площадь (~115x150 км)
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def criterial_on_wide(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """Критериальная перспективность (больше = лучше) в ячейках широкой сетки."""
    crt, msk, cx, cy = load_criterial()
    crt, msk = crt.ravel(), msk.ravel()
    cx, cy = cx.ravel(), cy.ravel()
    tree = cKDTree(np.column_stack([cx[msk], cy[msk]]))
    dist, idx = tree.query(np.column_stack([wx, wy]))
    return np.where(dist <= 375.0, -crt[msk][idx], np.nan)


def evaluate(X: np.ndarray, y: np.ndarray, folds, tag: str) -> dict:
    au, pr = [], []
    for tr, te in folds:
        p = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        au.append(roc_auc_score(y[te], p))
        pr.append(average_precision_score(y[te], p))
    res = {"набор": tag, "фолдов": len(au), "roc_auc": float(np.mean(au)),
           "roc_std": float(np.std(au)), "pr_auc": float(np.mean(pr)),
           "roc_min": float(np.min(au)), "roc_max": float(np.max(au))}
    log("  %-38s ROC-AUC %.4f±%.3f (от %.3f до %.3f), PR-AUC %.4f"
        % (tag, res["roc_auc"], res["roc_std"], res["roc_min"], res["roc_max"],
           res["pr_auc"]))
    return res


def main() -> None:
    wide = pd.read_parquet(config.PROCESSED_DIR / "wide_v1_archive"
                           / "dataset_wide.parquet")
    fw = pd.read_parquet(config.PROCESSED_DIR / "forecast_arch_ядро5км.parquet")
    wx = wide["x"].to_numpy(float)
    wy = wide["y"].to_numpy(float)
    assert np.allclose(wx, fw["x"].to_numpy(float)), "сетки не совпали"

    crit = criterial_on_wide(wx, wy)
    in_sheet = fw["в_габаритах_листа"].to_numpy(bool)
    common = np.isfinite(crit)
    zone = zone_of(wx, wy)
    south = common & ~in_sheet          # вся смежная часть области расчёта
    log("общая площадь %d ячеек; смежная %d, из них юго-запад %d, юг %d"
        % (int(common.sum()), int(south.sum()),
           int((south & (zone == "юго-запад")).sum()),
           int((south & (zone == "юг")).sum())))

    # Цель: верхние 10% критериальной карты на южной площади
    y = np.zeros(len(wide), dtype=int)
    thr = np.quantile(crit[south], 1 - AREA)
    y[south] = (crit[south] >= thr).astype(int)
    log("позитивов на смежной площади: %d (%.1f%%)"
        % (int(y[south].sum()), 100 * y[south].mean()))

    # --- Тест 2: перенос как есть, без дообучения
    ours = fw[SCORE_COL].to_numpy(float)
    log("ТЕСТ 2. Готовая карта переноса против критериальной цели:")
    log("  смежная   ROC-AUC %.4f, ранговая корреляция %+.3f"
        % (roc_auc_score(y[south], ours[south]),
           sps.spearmanr(ours[south], crit[south]).statistic))
    y_in = np.zeros(len(wide), dtype=int)
    sheet_common = common & in_sheet
    y_in[sheet_common] = (crit[sheet_common]
                          >= np.quantile(crit[sheet_common], 1 - AREA)).astype(int)
    log("  на листе   ROC-AUC %.4f, ранговая корреляция %+.3f"
        % (roc_auc_score(y_in[sheet_common], ours[sheet_common]),
           sps.spearmanr(ours[sheet_common], crit[sheet_common]).statistic))

    # --- Тест 1: а можно ли взять эту цель нашими признаками, обучаясь ЗДЕСЬ
    feat = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    Xw = spectral_mask.apply(wide[cols]).to_numpy(dtype=float)
    log("признаков переноса: %d" % len(cols))

    sx, sy = wx[south], wy[south]
    folds = strip_folds(sx, sy, y[south], GAP_M)
    G = rbf_basis(sx, sy, RBF_N)
    log("ТЕСТ 1. Обучение на смежной площади, полосы с разрывом %.0f км, %d фолдов:"
        % (GAP_M / 1000, len(folds)))
    rows = [evaluate(Xw[south], y[south], folds, "признаки переноса (37)"),
            evaluate(G, y[south], folds, "только география (%d базисов)" % G.shape[1]),
            evaluate(np.column_stack([Xw[south], G]), y[south], folds,
                     "признаки + география")]

    # то же на листе — для масштаба, той же цепочкой
    folds_in = strip_folds(wx[sheet_common], wy[sheet_common],
                           y_in[sheet_common], GAP_M)
    G_in = rbf_basis(wx[sheet_common], wy[sheet_common], RBF_N)
    log("  для масштаба — то же на листе, %d фолдов:" % len(folds_in))
    rows += [evaluate(Xw[sheet_common], y_in[sheet_common], folds_in,
                      "лист: признаки переноса (37)"),
             evaluate(G_in, y_in[sheet_common], folds_in,
                      "лист: только география")]

    # --- Тест 3: сдвиг или промах
    log("ТЕСТ 3. Расстояние между ядрами двух карт (смежная площадь):")
    top_c = south & (y == 1)
    top_o = np.zeros(len(wide), bool)
    top_o[south] = ours[south] >= np.quantile(ours[south], 1 - AREA)
    tree_c = cKDTree(np.column_stack([wx[top_c], wy[top_c]]))
    tree_o = cKDTree(np.column_stack([wx[top_o], wy[top_o]]))
    d_oc, _ = tree_c.query(np.column_stack([wx[top_o], wy[top_o]]))
    d_co, _ = tree_o.query(np.column_stack([wx[top_c], wy[top_c]]))
    for tag, d in (("от наших ядер до критериальных", d_oc),
                   ("от критериальных ядер до наших", d_co)):
        log("  %-32s медиана %5.1f км, 90-й процентиль %5.1f км, максимум %5.1f км"
            % (tag, np.median(d) / 1000, np.quantile(d, 0.9) / 1000, d.max() / 1000))

    # --- Тест 4: новизна в согласных и несогласных ячейках
    Xs = spectral_mask.apply(feat[cols]).to_numpy(dtype=float)
    plain = [j for j, c in enumerate(cols) if not c.startswith("s2_")]
    ok_s = np.isfinite(Xs[:, plain]).all(axis=1)
    nov = novelty(Xs[ok_s][:, plain], Xw[:, plain])
    log("ТЕСТ 4. Новизна ячейки (1.0 = типична для листа):")
    for tag, m in (("оба метода", top_c & top_o),
                   ("только критериальный", top_c & ~top_o),
                   ("только наш прогноз", top_o & ~top_c),
                   ("вся смежная", south),
                   ("общая площадь на листе", sheet_common)):
        if m.sum():
            log("  %-24s ячеек %6d, медиана новизны %.2f"
                % (tag, int(m.sum()), float(np.median(nov[m]))))

    out = ROOT / "outputs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "critmulti_why.csv", index=False,
                              encoding="utf-8-sig")
    (out / "critmulti_why.log").write_text("\n".join(LOG), encoding="utf-8")
    log("готово: %s" % (out / "critmulti_why.csv"))


if __name__ == "__main__":
    main()
