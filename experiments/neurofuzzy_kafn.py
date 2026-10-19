"""Нечёткие сети КАФН и ДФФЛ на наших данных, по нашему протоколу проверки.

ЗАЧЕМ. Репозиторий `deep-neuro-fuzzy` (fims9000) даёт интерпретируемые
нейро-нечёткие модели: КАФН — маршрутизируемая нечёткая сеть
Колмогорова–Арнольда (аддитивные одномерные предикаты LOW/MEDIUM/HIGH с
маршрутизацией и суперпозицией), ДФФЛ — глубокое обучение нечётких признаков.
Их заявленная сила не в точности, а в устойчивости структуры правил между
запусками, то есть ровно в том, чего нам не хватает: прогноз, который можно
показать геологу набором правил, а не весами дерева.

ЗАЧЕМ ОТДЕЛЬНЫЙ ЗАПУСК. Авторский бенчмарк считает на своих наборах
(covtype, SUSY) со случайным разбиением. Для наших данных случайное разбиение
меряет интерполяцию между соседними ячейками, поэтому модель подключается
напрямую к нашему протоколу: полосы листа с разрывом 15 км, те же фолды, что
и у рабочего бустинга, и сравнение на одних и тех же разбиениях.

ЧТО СЧИТАЕТСЯ. Две ветки признаков (A — только дистанционка, B — с
геологическими слоями формулы), на каждой: КАФН, бустинг (рабочая точка) и,
если доступен, глубокий вариант. Метрика — площадь под кривой на отложенной
полосе, среднее и разброс по фолдам.

ВХОД: сетка листа, data/processed/dataset_v5_rebuilt.parquet
ВЫХОД: outputs/metrics/neurofuzzy_kafn.log и .csv
Запуск из корня: python -X utf8 -m experiments.neurofuzzy_kafn
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NF_ROOT = Path.home() / "deep-neuro-fuzzy"
sys.path.insert(0, str(NF_ROOT / "src"))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402
from experiments.shap_two_branches import GEOLOGY  # noqa: E402

from ruanfis.kanfis import KANFISModel  # noqa: E402
from ruanfis.trainer import FuzzyTrainer, TrainingConfig  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
EPOCHS = 60
BATCH = 256
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def to_unit(train: np.ndarray, other: np.ndarray) -> tuple:
    """Шкалирование в [0,1] по квантилям обучающей части: термы сети ждут долей."""
    lo = np.nanpercentile(train, 2, axis=0)
    hi = np.nanpercentile(train, 98, axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    f = lambda z: np.clip((z - lo) / span, 0.0, 1.0)
    return f(train), f(other)


def fill(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = X.copy()
    med = np.nanmedian(ref, axis=0)
    bad = ~np.isfinite(out)
    out[bad] = np.take(med, np.where(bad)[1])
    return out


def run_kafn(Xtr, ytr, Xte, yte, seed: int) -> float:
    torch.manual_seed(seed)
    model = KANFISModel(input_dim=Xtr.shape[1], superposition_terms=12,
                        train_rule_gates=True)
    cfg = TrainingConfig(task_type="binary_classification", max_epochs=EPOCHS,
                         learning_rate=5e-3, batch_size=BATCH, patience=15,
                         binary_auto_pos_weight=True, device="cpu")
    trainer = FuzzyTrainer(model, cfg)
    tt = lambda z: torch.tensor(z, dtype=torch.float32)
    trainer.fit(tt(Xtr), tt(ytr).view(-1, 1), tt(Xte), tt(yte).view(-1, 1))
    model.eval()
    with torch.no_grad():
        out = model(tt(Xte))
        logits = out[0] if isinstance(out, tuple) else out
        p = torch.sigmoid(logits).numpy().ravel()
    return roc_auc_score(yte, p)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    S = spectral_mask.apply(feat[dzz])
    Xa = S.to_numpy(float)
    geo = [c for c in GEOLOGY if c in raw.columns]
    Xb = np.column_stack([Xa, raw[geo].to_numpy(float)])
    y = (crit >= np.nanquantile(crit[pool], 1 - AREA)).astype(int)
    folds = strip_folds(cx[pool], cy[pool], y[pool], GAP_M)
    log("ячеек %d, целевых %d, полос %d; КАФН: эпох %d, размер пакета %d"
        % (pool.size, int(y[pool].sum()), len(folds), EPOCHS, BATCH))

    rows = []
    for tag, X in (("A: только дистанционка", Xa),
                   ("B: дистанционка + геология", Xb)):
        Xf = fill(X, X[pool])
        kafn, boost = [], []
        for k, (tr, te) in enumerate(folds):
            Xtr, Xte = Xf[pool][tr], Xf[pool][te]
            ytr, yte = y[pool][tr], y[pool][te]
            Utr, Ute = to_unit(Xtr, Xte)
            kafn.append(run_kafn(Utr, ytr, Ute, yte, config.CRIT_SEED + k))
            m = make_model().fit(Xtr, ytr)
            boost.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            log("   %s, полоса %d: КАФН %.3f, бустинг %.3f"
                % (tag, k + 1, kafn[-1], boost[-1]))
        log("%-28s КАФН %.3f ± %.3f | бустинг %.3f ± %.3f"
            % (tag, np.mean(kafn), np.std(kafn), np.mean(boost), np.std(boost)))
        rows.append({"ветка": tag, "КАФН": np.mean(kafn),
                     "КАФН_разброс": np.std(kafn), "бустинг": np.mean(boost),
                     "бустинг_разброс": np.std(boost), "фолдов": len(folds)})

    out = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out / "neurofuzzy_kafn.csv", index=False,
                              encoding="utf-8-sig")
    (out / "neurofuzzy_kafn.log").write_text("\n".join(LOG), encoding="utf-8")
    log("готово")


if __name__ == "__main__":
    main()
