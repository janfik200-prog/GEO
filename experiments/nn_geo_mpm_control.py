"""Контроль к `nn_geo_mpm`: а что из выигрыша нейросетей даёт простое усреднение?

ЗАЧЕМ. В прогоне `nn_geo_mpm` графовая сеть по соседству на сетке дала на
разрыве 15 км ROC-AUC 0.799 против 0.749 у гистограммного бустинга на тех же
37 признаках и 0.675 у планки «только география». Соблазнительно записать это
как «нейросеть работает лучше». Но обе архитектуры отличаются от бустинга ровно
одним: они видят ОКРЕСТНОСТЬ ячейки, а не только её собственный вектор. Графовая
свёртка в два слоя по 8-связности — это, по сути, взвешенное среднее признаков
в окне 5x5; свёрточная сеть с патчем 15x15 — набор фильтров в окне 15x15.

Значит, честный вопрос не «нейросеть против бустинга», а «нейросеть против
бустинга, которому дали ту же окрестность». Если бустинг на усреднённых
признаках догоняет графовую сеть, то весь выигрыш — это сглаживание, доступное
одной строкой `uniform_filter`, и никакой нейросети для него не нужно.
Если не догоняет — разница действительно приходится на нелинейную обработку
рисунка поля, и тогда архитектура что-то добавляет.

Усреднение признаков метками не пользуется, поэтому считается один раз по всей
сетке до разбиения на полосы — это операция над входами, а не над мишенью.

Варианты:
  * `бустинг, сырые` — воспроизведение планки из `nn_geo_mpm` (сверка чисел);
  * `бустинг, среднее 5x5` — окрестность графовой сети;
  * `бустинг, среднее 15x15` — окрестность свёрточной сети;
  * `бустинг, сырые + среднее 5x5` — и значение в точке, и контекст (74 столбца);
  * `бустинг, сырые + оба окна` — полный контекст (111 столбцов).

Выход: `outputs/metrics/nn_geo_mpm_control.csv`.
Запуск из корня: ``python -m experiments.nn_geo_mpm_control``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.nn_geo_mpm import SEED, blind, log, stack_image  # noqa: E402

WINDOWS = (5, 15)


def smoothed(X: np.ndarray, shape: tuple[int, int], w: int) -> np.ndarray:
    """Скользящее среднее каждого признака в окне w x w (края — отражением)."""
    img = stack_image(np.nan_to_num(X, nan=np.nan), shape).astype(np.float64)
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        ch = img[c]
        m = np.isfinite(ch)
        num = ndimage.uniform_filter(np.where(m, ch, 0.0), w, mode="nearest")
        den = ndimage.uniform_filter(m.astype(float), w, mode="nearest")
        out[c] = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(img.shape[0], -1).T


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

    log(f"признаков {len(cols)}, сетка {shape}, пул {pool.size}")
    sm = {w: smoothed(X, shape, w) for w in WINDOWS}
    log(f"скользящие средние посчитаны: окна {WINDOWS} (в км: "
        f"{', '.join(f'{w * meta.dx / 1000:.1f}' for w in WINDOWS)})")

    variants = {
        "бустинг, сырые": X,
        "бустинг, среднее 5x5 (2.5 км)": sm[5],
        "бустинг, среднее 15x15 (7.5 км)": sm[15],
        "бустинг, сырые + среднее 5x5": np.hstack([X, sm[5]]),
        "бустинг, сырые + оба окна": np.hstack([X, sm[5], sm[15]]),
    }

    rows, fold_rows = [], []
    for name, Xv in variants.items():
        def fp(tr, allx, Xv=Xv):
            m = HistBackgroundEnsemble(random_state=SEED).fit(Xv[tr], y[tr])
            return m.predict_proba(Xv[allx])[:, 1]

        cals = []
        for ax, coord in (("X", cx), ("Y", cy)):
            log(f"{name}, ось {ax}")
            _, cal, fr = blind(fp, coord, y, pool, name, ax)
            cals.append(cal)
            fold_rows += fr
        s = np.nanmean(np.vstack(cals), axis=0)
        s = np.where(np.isfinite(s), s, np.nanmin(s))
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        f_ = pd.DataFrame(fold_rows)
        f_ = f_[f_["ключ"] == name]
        rows.append({"вариант": name, "столбцов": Xv.shape[1],
                     "фолды_roc_auc": float(f_["roc_auc"].mean()),
                     "фолды_roc_std": float(f_["roc_auc"].std()),
                     "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                     "карта_pr_auc": average_precision_score(y[pool], s[pool]),
                     "согласие_каппа": agr["kappa"],
                     "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1)})

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm_control.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        ROOT / "outputs" / "metrics" / "nn_geo_mpm_control_folds.csv", index=False)
    print("\n=== Контроль: даёт ли окрестность то же, что нейросеть ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nДля сравнения из nn_geo_mpm.csv (тот же протокол, те же 37 признаков):")
    prev = pd.read_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm.csv")
    print(prev[["метод", "фолды_roc_auc", "карта_roc_auc", "совпадение_верх_10"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
