"""Кривая «точность против плотности»: сколько линий и какой ценой.

Плотность трассировки задаёт пользователь, а не метод: можно выдать полтора
десятка отрезков, в которых модель уверена, а можно две сотни. Здесь эта ручка
прокручивается целиком, и для каждого положения считается, какая доля длины
выделенных линий легла в допуск от карты. Горизонталь — доля площади у карты,
то есть то, что даст линия, брошенная наугад.

Скор берётся из уже сохранённых прогонов ``openlin_lines``, обучение заново не
запускается.

Запуск из корня репозитория:
    python -m experiments.openlin_lines_density
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.openlin_anfis import TOL_M, block_split
from experiments.openlin_lines import line_metrics, trace_lines
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")
OUT = Path("outputs/openlin_lines_density.png")

#: Развёртка ручки плотности: (sigma, HI, LO, мин. длина, прямолинейность).
#: Порядок — от «только уверенное» к «густо»; шаг подобран так, чтобы число
#: линий росло примерно вдвое за две ступени.
SWEEP = ((300.0, 0.05, 0.15, 3000.0, 0.80),
         (300.0, 0.08, 0.20, 2500.0, 0.78),
         (250.0, 0.10, 0.25, 2000.0, 0.75),
         (200.0, 0.15, 0.35, 1500.0, 0.70),
         (175.0, 0.20, 0.42, 1200.0, 0.65),
         (150.0, 0.25, 0.50, 1000.0, 0.60),
         (125.0, 0.30, 0.58, 1000.0, 0.55),
         (100.0, 0.35, 0.65,  800.0, 0.50))

#: Какие сохранённые прогоны разворачиваем: (файл, площадка, режим, подпись).
CASES = (("openlin_lines_nevada_бустинг_hard_1_nevada.npz", "nevada", "проверка",
          "Невада, своя проверочная часть"),
         ("openlin_lines_nevada_логрегрессия_smooth_1_anabar.npz", "anabar",
          "перенос", "Анабар, обучение на Неваде"),
         ("openlin_lines_anabar_dykes_бустинг_smooth_1_anabar.npz", "anabar",
          "перенос", "Анабар, обучение на дайках"))


def main() -> None:
    rows = []
    for fname, site, mode, cap in CASES:
        p = M / fname
        if not p.exists():
            print(f"пропуск: нет {p}")
            continue
        z = load_site(site)
        valid = np.isfinite(z["gray"])
        res_m = float(z["res_m"])
        if mode == "проверка":
            _tr, area = block_split(valid.shape, res_m)
            area &= valid
        else:
            area = valid
        base = float(((z["dist"] <= TOL_M) & area).sum() / area.sum())
        sc = np.load(p)["score"]
        for params in SWEEP:
            pred, br = trace_lines(sc, valid, area, res_m, params)
            r = line_metrics(pred, z["dist"], area, res_m)
            rows.append({"случай": cap, "площадка": site, "случайная": base,
                         "линий": len(br), "длина_км": r["длина_км"],
                         "точность": r["точность"], "полнота": r["полнота"],
                         "мин_длина_км": params[3] / 1000})
            print(f"{cap:<34} линий {len(br):>4}  {r['длина_км']:>5.0f} км  "
                  f"точность {r['точность']:.3f}  (случайная {base:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(M / "openlin_lines_density.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    colors = {c: k for c, k in zip(df["случай"].unique(),
                                   ("#12a150", "#1f66c1", "#c98b3a"))}
    for cap, g in df.groupby("случай", sort=False):
        g = g.sort_values("линий")
        axes[0].plot(g["линий"], g["точность"], "o-", color=colors[cap], label=cap)
        axes[0].axhline(g["случайная"].iloc[0], color=colors[cap], ls="--", lw=1,
                        alpha=0.7)
        axes[1].plot(g["линий"], g["полнота"], "o-", color=colors[cap], label=cap)
    axes[0].set_xlabel("сколько линий выдано")
    axes[0].set_ylabel("точность (доля длины в допуске 250 м)")
    axes[0].set_title("Чем гуще, тем грязнее\n"
                      "пунктир — уровень случайной линии для этой площадки",
                      fontsize=11)
    axes[1].set_xlabel("сколько линий выдано")
    axes[1].set_ylabel("полнота (доля длины карты, покрытая линиями)")
    axes[1].set_title("Полнота растёт с плотностью, но остаётся низкой",
                      fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Плотность трассировки — ручка пользователя, а не свойство метода",
                 fontsize=14)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"\nсохранено: {OUT}")


if __name__ == "__main__":
    main()
