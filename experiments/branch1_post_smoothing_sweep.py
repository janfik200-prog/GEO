"""Сколько сглаживать прогноз: совпадение против сохранности рисунка.

ПОВОД. В прогоне `branch1_multiscale_ensemble` лучшие по совпадению верхних 10%
варианты получаются после гауссова сглаживания готовой карты с сигмой 10 км
(20 ячеек). На картинке такая карта — почти одно гладкое пятно: ядро с сигмой
20 ячеек захватывает ±60 ячеек, то есть больше трети листа. Совпадение при этом
растёт (43-47%), и вопрос, чем именно оно оплачено: реальным укрупнением
прогноза до масштаба мишени или сползанием карты в региональный тренд, который
случайно похож на критериальный.

ЧТО СЧИТАЕТСЯ. Ничего не переобучается: берутся уже посчитанные прогнозы по
полосам из кэша `branch1_multiscale_cache.npz`, и для набора сигм от 0 до 15 км
считаются три вещи вместе:

* совпадение верхних 10% с критериальной картой и каппа — что мы максимизируем;
* совпадение после снятия тренда: из обеих карт вычитается своя квадратичная
  поверхность по координатам, и верхние 10% остатков сравниваются заново — это
  показывает, сколько совпадения приходится на местный рисунок, а не на общий
  наклон «в середине листа лучше»;
* доля дисперсии карты, объяснимая квадратичной поверхностью по координатам
  (`trend_share`) — насколько карта выродилась в «в середине листа хорошо»;
* ROC-AUC по полосам на сглаживании внутри полосы — сохраняется ли различающая
  способность по ячейкам.

Два способа сглаживания, как и в основном прогоне: по всей карте (соседние
полосы видны ядру — краевая утечка через разрыв) и внутри полосы (ядро видит
только прогнозы одной модели).

Выход: `outputs/metrics/branch1_post_smoothing.csv`,
       `outputs/branch1_post_smoothing.png`.
Запуск из корня: ``python -m experiments.branch1_post_smoothing_sweep``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target  # noqa: E402
from experiments.honest_feature_signal import N_STRIPS, poly_design, trend_share  # noqa: E402
from experiments.branch1_multiscale_ensemble import (log, mean_axes,  # noqa: E402
                                                     smooth_field)

SIGMA_KM = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0)
KEYS = ["бустинг + графовая сеть | 58 много масштабов",
        "голосование трёх (лес+бустинг+перцептрон) | 37 много масштабов (перенос)",
        "бустинг | 58 много масштабов",
        "бустинг | 58 сырые (планка)",
        "бустинг | география (планка)"]
SHORT = {KEYS[0]: "бустинг+граф, 58 масштабов", KEYS[1]: "голосование трёх, 37 масштабов",
         KEYS[2]: "бустинг, 58 масштабов", KEYS[3]: "бустинг, 58 сырых",
         KEYS[4]: "только география"}


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    crit = crit_reference.criterial_score(prognoz)
    pr_flat = prognoz.ravel()
    top_c = pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)
    cx, cy = meta.cell_centers()
    D = poly_design(cx.ravel(), cy.ravel())

    def detrend(v):
        beta, *_ = np.linalg.lstsq(D[pool], v[pool], rcond=None)
        return v - D @ beta

    crit_r = detrend(crit)
    top_c_r = (crit_r >= np.quantile(crit_r[pool], 1 - config.CREF_AREA))

    z = np.load(config.PROCESSED_DIR / "branch1_multiscale_cache.npz", allow_pickle=True)
    maps = z["maps"].item()
    strips = z["strips"].item()
    log(f"карт в кэше {len(maps)}, сигмы {SIGMA_KM} км")

    rows, gallery = [], {}
    for key in KEYS:
        per_ax = maps[key]
        for km in SIGMA_KM:
            sigma = km * 1000.0 / meta.dx
            variants = {}
            if km == 0:
                s = mean_axes(per_ax["X"], per_ax["Y"])
                variants["без сглаживания"] = np.where(np.isfinite(s), s, np.nanmin(s))
                fold_roc = np.nan
            else:
                s0 = mean_axes(per_ax["X"], per_ax["Y"])
                s0 = np.where(np.isfinite(s0), s0, np.nanmin(s0))
                variants["по всей карте"] = smooth_field(s0, shape, sigma)
                per_ax_sm, aus = {}, []
                for ax in ("X", "Y"):
                    acc = np.full(y.size, np.nan)
                    for k in range(N_STRIPS):
                        mask = strips[ax] == k
                        acc[mask] = smooth_field(per_ax[ax], shape, sigma, mask=mask)[mask]
                        te = np.flatnonzero(mask)
                        aus.append(roc_auc_score(y[te], acc[te]))
                    per_ax_sm[ax] = acc
                s2 = mean_axes(per_ax_sm["X"], per_ax_sm["Y"])
                variants["внутри полосы"] = np.where(np.isfinite(s2), s2, np.nanmin(s2))
                fold_roc = float(np.mean(aus))
            for how, s in variants.items():
                top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
                s_r = detrend(s)
                top_m_r = s_r >= np.quantile(s_r[pool], 1 - config.CREF_AREA)
                agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
                rows.append({
                    "карта": SHORT[key], "сглаживание": how, "сигма_км": km,
                    "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                    "совпадение_без_тренда": ((top_c_r & top_m_r & valid).sum()
                                              / max((top_c_r & valid).sum(), 1)),
                    "согласие_каппа": agr["kappa"],
                    "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                    "фолды_roc_auc": fold_roc,
                    "тренд_R2": float(trend_share(s[:, None], D)[0]),
                    "новых_ячеек": int((top_m & ~top_c & valid).sum())})
                if key == KEYS[0] and how != "внутри полосы":
                    gallery[km] = s
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "branch1_post_smoothing.csv",
               index=False, encoding="utf-8-sig")
    for how in ("по всей карте", "внутри полосы"):
        sub = res[(res["сглаживание"] == how) | (res["сигма_км"] == 0)]
        print(f"\n=== Сглаживание {how}: совпадение против трендовости ===")
        print(sub.pivot_table(index="сигма_км", columns="карта",
                              values=["совпадение_верх_10", "совпадение_без_тренда",
                                      "тренд_R2"])
              .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nТренд_R2 — доля дисперсии карты, объяснимая квадратичной поверхностью "
          "по координатам: 1.0 = карта стала гладким пятном «в середине хорошо».")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size  # noqa: E731
    show = [k for k in (0.0, 2.0, 3.0, 5.0, 10.0) if k in gallery]
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(3.6 * (len(show) + 1), 4.4))
    axes[0].imshow(rank(crit), cmap="magma")
    axes[0].set_title("Критериальный прогноз", fontsize=10)
    for ax, km in zip(axes[1:], show):
        ax.imshow(rank(gallery[km]), cmap="magma")
        r_ = res[(res["карта"] == SHORT[KEYS[0]]) & (res["сигма_км"] == km)].iloc[0]
        ax.set_title(f"сглаживание {km:.0f} км\nсовпадение {r_['совпадение_верх_10']:.0%}, "
                     f"без тренда {r_['совпадение_без_тренда']:.0%}, "
                     f"тренд {r_['тренд_R2']:.2f}", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Цена сглаживания: совпадение с критериальной картой растёт, "
                 "но за счёт общего наклона, а не местного рисунка "
                 "(бустинг + графовая сеть, 58 признаков с масштабами)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch1_post_smoothing.png", dpi=130)
    log("OK: outputs/branch1_post_smoothing.png")


if __name__ == "__main__":
    main()
