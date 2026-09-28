"""SHAP для «лес + Кора» и для чистого леса: на что опирается различение.

ЗАЧЕМ. Карты двух прогонов ансамбля (полосы обучения и проверки поменяны
местами, подделки новые) совпали на 0.02. Это можно объяснить двумя разными
способами, и SHAP их различает:

* модель в обоих прогонах опирается на ОДНИ И ТЕ ЖЕ признаки, но пороги
  подобраны под выборку по-разному — тогда ранжирование признаков по SHAP
  совпадёт, а карты нет;
* модель в каждом прогоне цепляется за СВОИ признаки — тогда не совпадёт и
  ранжирование, и это значит, что устойчивого различия, которое она могла бы
  выучить, в признаках попросту нет.

КАК СЧИТАЕТСЯ. Чистый лес — точно, ``TreeExplainer`` (вклад в вероятность
класса «снимок»). Ансамбль — оценка у него не дерево, а доля сработавших
правил, поэтому объяснитель перестановочный (``PermutationExplainer``) на
подвыборке проверочных ячеек, фон — отдельная подвыборка тех же ячеек.

Запуск из корня репозитория:
    python -m experiments.openlin_forest_kora_shap --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_forest_kora import RUNS, fit_runs, score

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

#: Сколько проверочных ячеек объясняется (поровну снимка и подделки) и сколько
#: берётся фоном. Перестановочный объяснитель вызывает оценку ~41 раз на
#: строку при 20 признаках, больше строк — только дольше, картина та же.
N_EXPLAIN = 1000
N_BACKGROUND = 200


def short(n: str) -> str:
    n = n.replace("coh_summer_vv_", "лет.ког ").replace("coh_winter_vv_",
                                                         "зим.ког ")
    for k, w in (("eng", " энерг "), ("coh", " когер "), ("cos", " cos "),
                 ("sin", " sin "), ("rdg", " гребень ")):
        if n.endswith(tuple(f"{k}{s}" for s in ("500", "1000", "2000"))):
            head, tail = n.rsplit(k, 1)
            return f"{head.rstrip('_')}{w}{int(tail) / 1000:g} км"
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import pandas as pd
    import shap
    from scipy.stats import spearmanr

    res, names, _z, _valid, _full160 = fit_runs(a.site, log)
    labels = [short(n) for n in names]
    rng = np.random.default_rng(5)
    imp, sv_store = {}, {}
    for run, *_ in RUNS:
        r, f = res[run]["r_te"], res[run]["f_te"]
        i1 = rng.choice(len(r), N_EXPLAIN // 2, replace=False)
        i2 = rng.choice(len(f), N_EXPLAIN // 2, replace=False)
        X = np.vstack([r[i1], f[i2]])
        bg = np.vstack([r[rng.choice(len(r), N_BACKGROUND // 2, replace=False)],
                        f[rng.choice(len(f), N_BACKGROUND // 2, replace=False)]])

        tree = shap.TreeExplainer(res[run]["rf"])
        sv = tree.shap_values(X, check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:                     # (строки, признаки, классы)
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
        imp[("лес", run)] = np.abs(sv).mean(0)

        rules = res[run]["rules"]
        ens = shap.PermutationExplainer(lambda Z: score(Z, rules), bg)
        se = ens(X).values
        imp[("лес+Кора", run)] = np.abs(se).mean(0)
        sv_store[run] = (se, X)
        log(f"{run}: SHAP посчитан")

    tab = pd.DataFrame({f"{m} / {r}": v for (m, r), v in imp.items()},
                       index=labels)
    tab.to_csv(M / f"openlin_forest_kora_shap_{a.site}.csv",
               encoding="utf-8-sig")
    log("\nсредний |SHAP|, пять главных признаков:")
    for col in tab.columns:
        top = tab[col].sort_values(ascending=False).head(5)
        log(f"  {col:22s} " + "; ".join(f"{k} {v:.3f}" for k, v in top.items()))

    log("\nсовпадает ли ранжирование признаков (Спирмен по среднему |SHAP|):")
    for m in ("лес", "лес+Кора"):
        rho = spearmanr(imp[(m, "исходный")], imp[(m, "повтор")]).statistic
        t1 = set(np.argsort(-imp[(m, "исходный")])[:5])
        t2 = set(np.argsort(-imp[(m, "повтор")])[:5])
        log(f"  {m:9s} между прогонами: {rho:+.2f}; общих в пятёрке "
            f"главных {len(t1 & t2)} из 5")
    for run, *_ in RUNS:
        rho = spearmanr(imp[("лес", run)], imp[("лес+Кора", run)]).statistic
        log(f"  лес против лес+Кора, прогон «{run}»: {rho:+.2f}")

    plot(sv_store, labels, tab, a.site, log)


def plot(sv_store, labels, tab, site, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    fig = plt.figure(figsize=(21, 9.5), facecolor="#fcfcfb")
    for k, (run, (se, X)) in enumerate(sv_store.items(), start=1):
        ax = fig.add_subplot(1, 3, k)
        plt.sca(ax)
        shap.summary_plot(se, X, feature_names=labels, show=False,
                          plot_size=None, max_display=12, color_bar=False)
        ax.set_title(f"лес + «Кора», прогон «{run}»", fontsize=14, loc="left")
        ax.set_xlabel("SHAP: вклад в оценку «снимок»", fontsize=11)

    ax = fig.add_subplot(1, 3, 3)
    cols = [c for c in tab.columns if c.startswith("лес+Кора")]
    order = tab[cols].mean(1).sort_values().index[-12:]
    y = np.arange(len(order))
    for off, c, col in ((-0.2, cols[0], "#2a78d6"), (0.2, cols[1], "#e34948")):
        ax.barh(y + off, tab.loc[order, c], height=0.38, color=col,
                label=c.split(" / ")[1])
    ax.set_yticks(y, order, fontsize=10)
    ax.set_xlabel("средний |SHAP|", fontsize=11)
    ax.set_title("одни ли признаки в двух прогонах", fontsize=14, loc="left")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = OUT / f"лес_кора_shap_{site}.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
