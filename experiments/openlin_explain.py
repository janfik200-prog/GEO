"""Что на самом деле решает судьбу линии: разбор модели по SHAP.

ЗАЧЕМ. К этому моменту у нас есть отбор, который работает (линия против фазовой
подделки, AUC 0.84), но неясно, ЧТО именно он использует и почему целые
семейства линий не выделяются. Линейные веса такой вопрос не закрывают: они
усредняют, а нас интересует поведение на конкретных линиях.

ЧТО СЧИТАЕТСЯ:

1. Вклад каждого признака в оценку каждой линии (SHAP по градиентному
   бустингу). Глобально — что вообще определяет решение; отдельно — как вклад
   зависит от значения признака.
2. Из чего сделаны компоненты сжатия: какие из 160 признаков окрестности в них
   вошли. Это показывает, что оператор физически способен увидеть.
3. Чем отличаются линии, которые метод находит, от прямых, которые он
   пропускает (пропуски берутся из ``outputs/metrics/_miss_segs.npy``).

Запуск из корня репозитория:
    python -m experiments.openlin_explain --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = Path("outputs/metrics")
OUT = Path("outputs/steps")


def shap_report(df: pd.DataFrame, log) -> None:
    """Вклад признаков в решение «линия против артефакта»."""
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    feats = [c for c in df.columns
             if c not in ("настоящая", "оператор", "оценка")]
    X = np.nan_to_num(df[feats].to_numpy("float64"), nan=0.0, posinf=0.0,
                      neginf=0.0)
    y = df["настоящая"].to_numpy()
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    model.fit(xtr, ytr)

    from experiments.openlin_line_rank import auc
    log(f"бустинг на отложенной части: AUC {auc(yte, model.predict_proba(xte)[:, 1]):.3f}")

    rng = np.random.default_rng(0)
    sample = xte[rng.choice(len(xte), min(1500, len(xte)), replace=False)]
    bg = xtr[rng.choice(len(xtr), 200, replace=False)]
    # HistGradientBoosting: точная аддитивность деревьев не гарантируется,
    # поэтому берём интервентный explainer и отключаем её проверку
    expl = shap.TreeExplainer(model, data=bg,
                              feature_perturbation="interventional",
                              feature_names=feats)
    sv = shap.Explanation(
        values=expl.shap_values(sample, check_additivity=False),
        base_values=np.repeat(expl.expected_value, len(sample)),
        data=sample, feature_names=feats)

    imp = np.abs(sv.values).mean(0)
    order = np.argsort(-imp)
    log("\nвклад признаков в решение (средний |SHAP|):")
    for i in order:
        # знак: в какую сторону тянет высокое значение признака
        hi = sample[:, i] > np.median(sample[:, i])
        direction = "+" if sv.values[hi, i].mean() > sv.values[~hi, i].mean() else "-"
        log(f"  {feats[i]:20s} {imp[i]:6.3f}  высокое значение тянет "
            f"{'к линии' if direction == '+' else 'к артефакту'}")

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, sample, feature_names=feats, show=False,
                      max_display=len(feats))
    plt.tight_layout()
    fig.savefig(OUT / "shap_сводка.png", dpi=140)
    plt.close(fig)
    log(f"сохранено: {OUT / 'shap_сводка.png'}")


def pca_loadings(site: str, log) -> None:
    """Из каких исходных признаков собраны компоненты сжатия."""
    from sklearn.decomposition import PCA

    from experiments.openlin_neigh import neigh_stack
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    x, _y, valid, _u = build_stack(z, use)
    res_m = float(z["res_m"])
    nx, names = neigh_stack(x, use, valid, res_m, base=use)
    all_names = list(use) + list(names)
    stack = np.concatenate([x, nx]).astype("float32")
    rows = stack[:, valid].T
    sel = np.random.default_rng(42).choice(len(rows), 50_000, replace=False)
    pca = PCA(n_components=8, whiten=True, random_state=42).fit(rows[sel])

    log("\nиз чего сделаны компоненты (топ-5 признаков на компоненту):")
    for k in range(3):
        w = pca.components_[k]
        top = np.argsort(-np.abs(w))[:5]
        share = 100 * pca.explained_variance_ratio_[k]
        log(f"  компонента {k + 1} ({share:.0f}% разброса): "
            + ", ".join(f"{all_names[i]} {w[i]:+.2f}" for i in top))

    # сколько веса приходится на масштабы окрестности
    log("\nна что приходится вес первых трёх компонент:")
    groups = {"исходные каналы": lambda n: "_" not in n or n in use,
              "окрестность 500 м": lambda n: n.endswith("500"),
              "окрестность 1 км": lambda n: n.endswith("1000"),
              "окрестность 2 км": lambda n: n.endswith("2000")}
    w3 = np.abs(pca.components_[:3]).sum(0)
    for gname, test in groups.items():
        idx = [i for i, n in enumerate(all_names) if test(n)]
        if idx:
            log(f"  {gname:20s} {100 * w3[idx].sum() / w3.sum():5.1f} %")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--table", default=None)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    src = Path(a.table) if a.table else M / f"openlin_fdr_{a.site}_hessian,old.csv"
    df = pd.read_csv(src, encoding="utf-8-sig")
    log(f"таблица {src.name}: {len(df)} линий "
        f"({int(df['настоящая'].sum())} с настоящего входа)")
    shap_report(df, log)
    pca_loadings(a.site, log)


if __name__ == "__main__":
    main()
