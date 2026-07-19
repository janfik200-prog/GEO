"""Чистый head-to-head: наша ML vs критериальный (ГИС-Интегро-аналог) на
коммодити-правильных данных США-Запад — NURE радиометрия (eU/K/eTh) + глобальная
геофизика, метки MRDS. Числа — из прогона experiments.nure_uranium (spatial CV,
StratifiedGroupKFold x3 сида). Рисуем AUC и lift@10% для УРАНА и ЗОЛОТА.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# результаты experiments.nure_uranium (пространственная CV)
RES = {
    "УРАН (9397 рудопроявл.)": {
        "Критериальный\n(геофизика)":  (0.647, 1.44),
        "ML геофизика":               (0.813, 5.49),
        "ML +радиометрия":            (0.822, 5.75),
        "ML только\nрадиометрия":     (0.672, 3.07),
    },
    "ЗОЛОТО (56709 рудопроявл.)": {
        "Критериальный\n(геофизика)":  (0.634, 1.70),
        "ML геофизика":               (0.755, 3.77),
        "ML +радиометрия":            (0.741, 2.99),
        "ML только\nрадиометрия":     (0.532, 1.31),
    },
}
COL = {"Критериальный\n(геофизика)": "#888888", "ML геофизика": "#1f77b4",
       "ML +радиометрия": "#2ca02c", "ML только\nрадиометрия": "#9467bd"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for j, (target, models) in enumerate(RES.items()):
    names = list(models)
    auc = [models[n][0] for n in names]
    lift = [models[n][1] for n in names]
    cols = [COL[n] for n in names]
    x = np.arange(len(names))
    for i, (vals, ttl, base) in enumerate([(auc, "ROC-AUC", 0.5), (lift, "lift@10%", 1.0)]):
        ax = axes[i, j]
        bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.6)
        ax.axhline(base, color="red", ls="--", lw=1, alpha=0.7,
                   label=("случайно" if i == 0 else "случайно (=1)"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}" if i == 0 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(ttl, fontsize=10)
        if i == 0:
            ax.set_title(target, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

fig.suptitle("Наша ML vs критериальный (ГИС-Интегро) — США-Запад, коммодити-правильные данные\n"
             "NURE радиометрия (eU/K/eTh) + глобальная геофизика · метки MRDS · пространственная CV",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("outputs/nure_ml_vs_criterial.png", dpi=140, bbox_inches="tight")
print("saved outputs/nure_ml_vs_criterial.png")
