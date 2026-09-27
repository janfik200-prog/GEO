"""Ветка 5, шаг 1: чем смежная территория отличается от листа — по каждому признаку.

ЗАЧЕМ. Перенос модели на широкую сетку до сих пор держался на одной проверке:
разрыв 15 км между обучающей и тестовой полосой внутри листа. Это проверка
«работает ли признак вдали от обучения», но не «одинаково ли он устроен по обе
стороны границы листа». Доля ячеек вне обучающего диапазона считалась ровно для
одной группы — `dist` (гидросеть оцифрована только на листе). Здесь то же самое
считается для всех признаков набора переноса, плюс два измерения, которых не
было вовсе.

ТРИ ИЗМЕРЕНИЯ.

1. **Одно место, две сборки.** Широкая сетка (299x303, шаг 500 м) целиком
   накрывает габариты листа, но её ячейки смещены относительно ячеек листа на
   272 м по X и 134 м по Y, а мозаики снимков и рельефа собирались отдельно.
   Значит для 22.9 тыс. ячеек листа есть вторая, независимая сборка тех же
   признаков. Если признак в двух сборках расходится, перенос ломается ещё до
   всякой геологии: модель обучена на одной шкале, а применяется к другой.
   Сравнение идёт по ближайшей ячейке (смещение меньше половины ячейки).

2. **Сдвиг распределения.** Для каждого признака: доля ячеек смежной территории
   вне обучающего диапазона (min-max и 1-99 процентиль), сдвиг среднего в
   единицах обучающего стандартного отклонения, расстояние Колмогорова-Смирнова,
   доля пропусков по обе стороны.

3. **Насколько домены вообще различимы.** Однопризнаковая ROC-AUC «лист против
   смежной территории»: 0.5 — признак не отличает домены, ближе к 0 или 1 —
   признак кодирует «где», а не «что». Плюс тот же показатель для всего набора
   сразу (бустинг, обучение на случайной половине ячеек, проверка на второй).
   Полная различимость доменов ожидаема и сама по себе не порок — важно, какие
   признаки её создают.

Выход: `outputs/metrics/branch5_same_place.csv`, `branch5_feature_shift.csv`,
       `outputs/branch5_shift.png`.
Запуск из корня: ``python -m experiments.branch5_transfer_shift``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target, features_v2  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def single_auc(v_sheet: np.ndarray, v_adj: np.ndarray) -> float:
    """ROC-AUC различения доменов по одному признаку (пропуски отброшены)."""
    a, b = v_sheet[np.isfinite(v_sheet)], v_adj[np.isfinite(v_adj)]
    if a.size < 50 or b.size < 50:
        return float("nan")
    y = np.concatenate([np.zeros(a.size), np.ones(b.size)])
    return float(roc_auc_score(y, np.concatenate([a, b])))


def neighbour_corr(v: np.ndarray, shape, valid: np.ndarray) -> float:
    """Корреляция признака с самим собой, сдвинутым на одну ячейку.

    Планка для сравнения сборок: ячейки двух сеток разъехались на 264 м, то есть
    больше половины ячейки, и часть расхождения объясняется одним этим смещением,
    а не разной сборкой. Признак, который и внутри своей сетки слабо похож на
    соседнюю ячейку, не мог совпасть и между сборками.
    """
    g = np.where(valid, v, np.nan).reshape(shape)
    out = []
    for a, b in ((g[:-1, :], g[1:, :]), (g[:, :-1], g[:, 1:])):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 100:
            out.append(sps.spearmanr(a[ok], b[ok]).statistic)
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    log(f"набор переноса: {len(cols)} признаков; лист {int(valid.sum())} рабочих "
        f"ячеек из {valid.size}, широкая сетка {len(wide)} ячеек")

    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)
    in_sheet = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
                (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    log(f"широкая сетка: в габаритах листа {int(in_sheet.sum())}, "
        f"смежная территория {int((~in_sheet).sum())} ячеек "
        f"({(~in_sheet).mean():.1%} площади)")

    # --- 1. одно место, две сборки -----------------------------------------
    tree = cKDTree(np.column_stack([wx[in_sheet], wy[in_sheet]]))
    dist, idx = tree.query(np.column_stack([cx[valid], cy[valid]]), k=1)
    w_in = wide.loc[in_sheet].reset_index(drop=True)
    log(f"сопоставление ячеек: медиана смещения {np.median(dist):.0f} м, "
        f"максимум {dist.max():.0f} м")
    shape = (meta.prf, meta.pic)
    rows = []
    for c in cols:
        a = feat.loc[valid, c].to_numpy(float)
        b = w_in.loc[idx, c].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        nb_r = neighbour_corr(feat[c].to_numpy(float), shape, valid)
        sd = np.nanstd(a[ok]) if ok.sum() else np.nan
        rows.append({
            "признак": c, "группа": features_v2.feature_group(c),
            "пар": int(ok.sum()),
            "спирмен": (float(sps.spearmanr(a[ok], b[ok]).statistic)
                        if ok.sum() > 100 else np.nan),
            "спирмен_соседняя_ячейка": nb_r,
            "сдвиг_ср_в_сигмах": (float((np.mean(b[ok]) - np.mean(a[ok])) / sd)
                                  if sd and np.isfinite(sd) and sd > 0 else np.nan),
            "отношение_разбросов": (float(np.std(b[ok]) / sd)
                                    if sd and np.isfinite(sd) and sd > 0 else np.nan),
            "пропусков_лист": float(np.mean(~np.isfinite(a))),
            "пропусков_широкая": float(np.mean(~np.isfinite(b)))})
    same = pd.DataFrame(rows).sort_values("спирмен")
    same.to_csv(ROOT / "outputs" / "metrics" / "branch5_same_place.csv",
                index=False, encoding="utf-8-sig")
    print("\n=== Одно место, две сборки: согласие признаков листа и широкой сетки ===")
    print(same.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- 2-3. сдвиг распределения и различимость доменов --------------------
    Xs = feat.loc[valid, cols].to_numpy(float)              # обучающая сборка листа
    Xa = wide.loc[~in_sheet, cols].to_numpy(float)          # смежная территория
    rows = []
    for j, c in enumerate(cols):
        a, b = Xs[:, j], Xa[:, j]
        af = a[np.isfinite(a)]
        lo, hi = (af.min(), af.max()) if af.size else (np.nan, np.nan)
        p1, p99 = (np.percentile(af, [1, 99]) if af.size else (np.nan, np.nan))
        bf = b[np.isfinite(b)]
        sd = af.std() if af.size else np.nan
        rows.append({
            "признак": c, "группа": features_v2.feature_group(c),
            "вне_минмакс": float(np.mean((bf < lo) | (bf > hi))) if bf.size else np.nan,
            "вне_1_99": float(np.mean((bf < p1) | (bf > p99))) if bf.size else np.nan,
            "сдвиг_ср_в_сигмах": (float((bf.mean() - af.mean()) / sd)
                                  if bf.size and sd > 0 else np.nan),
            "колмогоров": (float(sps.ks_2samp(af, bf).statistic)
                           if bf.size and af.size else np.nan),
            "пропусков_смежная": float(np.mean(~np.isfinite(b))),
            "auc_домена": single_auc(a, b)})
    shift = pd.DataFrame(rows)
    shift["различимость"] = (shift["auc_домена"] - 0.5).abs()
    shift = shift.sort_values("различимость", ascending=False)
    shift.to_csv(ROOT / "outputs" / "metrics" / "branch5_feature_shift.csv",
                 index=False, encoding="utf-8-sig")
    print("\n=== Сдвиг распределений лист -> смежная территория ===")
    print(shift.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nПо группам (среднее):")
    print(shift.groupby("группа")[["вне_минмакс", "вне_1_99", "колмогоров",
                                   "различимость", "пропусков_смежная"]]
          .mean().round(3).to_string())

    y_dom = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xa))])
    Xd = np.vstack([Xs, Xa])
    rng = np.random.default_rng(config.CRIT_SEED)
    half = rng.random(len(Xd)) < 0.5
    clf = HistGradientBoostingClassifier(random_state=config.CRIT_SEED)
    clf.fit(Xd[half], y_dom[half])
    auc_dom = roc_auc_score(y_dom[~half], clf.predict_proba(Xd[~half])[:, 1])
    log(f"различимость доменов всем набором сразу: ROC-AUC {auc_dom:.4f} "
        "(1.0 — домены различаются полностью)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = shift.head(6)["признак"].tolist()
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axs.ravel(), worst):
        j = cols.index(c)
        a, b = Xs[:, j], Xa[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        lo = min(np.percentile(a, 0.5), np.percentile(b, 0.5))
        hi = max(np.percentile(a, 99.5), np.percentile(b, 99.5))
        bins = np.linspace(lo, hi, 60)
        ax.hist(a, bins=bins, density=True, alpha=0.55, label="лист (обучение)")
        ax.hist(b, bins=bins, density=True, alpha=0.55, label="смежная территория")
        r = shift[shift["признак"] == c].iloc[0]
        ax.set_title(f"{c}: вне диапазона {r['вне_минмакс']:.0%}, "
                     f"различимость {r['auc_домена']:.2f}", fontsize=9)
        ax.set_ylabel("плотность")
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Ветка 5: признаки набора переноса с самым сильным сдвигом "
                 "распределения", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch5_shift.png", dpi=130)
    log("OK: outputs/branch5_shift.png")


if __name__ == "__main__":
    main()
