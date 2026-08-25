"""График для слайда dataset_v6: откуда взялся шум и сколько его в каждой группе.

Зачем. Голый список из 44 убранных колонок неинформативен на слайде — интереснее
показать долю шума ВНУТРИ каждой группы признаков (сколько % группы оказалось
шумом на честной permutation importance), это вскрывает конкретный вывод:
группа ``pf`` (доп. трансформанты потенциальных полей, задача add-on к gm) —
11 из 12 колонок (92%) оказались шумом, то есть сам приём "нагенерить трансформант
потенциального поля" почти не работает на этом листе, в отличие от базовых gm.

Данные берутся из уже посчитанных списков (`experiments.build_dataset_v6`),
без переобучения — просто группировка по `features_v2.feature_group`.

Запуск из корня: ``python -m experiments.plot_dataset_v6_noise``.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v2  # noqa: E402
from experiments.build_dataset_v6 import NOISE_FEATURES, SERVICE_COLS, RESIDUAL_DUP_COLS  # noqa: E402

OUT = config.PROJECT_ROOT / "outputs" / "dataset_v6_noise_breakdown.png"

GOLD = "#C89B2C"
RED = "#B03A2E"
GREY = "#999999"
DARK = "#2B2B2B"


GROUP_LABELS_RU = {
    "pf": "потен. поля (доп.)",
    "opt": "опт. индекс (доп.)",
    "psr": "PALSAR-2",
    "gm": "грав-магнитка",
    "l8": "Landsat 8/9",
    "s2": "Sentinel-2",
    "s2raw": "Sentinel-2 (исходные каналы)",
}


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    total = Counter(features_v2.feature_group(c) for c in df.columns)
    noise = Counter(features_v2.feature_group(c) for c in NOISE_FEATURES)

    rows = [(g, noise[g], total[g]) for g in noise if g is not None]
    rows.sort(key=lambda r: -(r[1] / r[2]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Левая панель: доля шума внутри каждой группы (главный вывод) ---
    ax = axes[0]
    labels = [GROUP_LABELS_RU.get(g, g) for g, n, t in rows]
    fracs = [100 * n / t for g, n, t in rows]
    colors = [RED if f >= 60 else GOLD for f in fracs]
    bars = ax.barh(labels, fracs, color=colors, edgecolor="white")
    for bar, (g, n, t) in zip(bars, rows):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{n} из {t}", va="center", fontsize=10, color=DARK)
    ax.invert_yaxis()
    ax.set_xlabel("доля признаков группы, признанных шумом, %")
    ax.set_xlim(0, 100)
    ax.set_title("Доля шума внутри каждой группы признаков\n"
                  "(перестановочная проверка значимости, среднее ≤ 0 на 8 полосах)",
                  fontsize=11)
    ax.annotate("pf (доп. трансформанты\nпотенциальных полей):\n92% группы — шум,\nсам приём почти не работает",
                xy=(fracs[0] - 2, 0.3), xytext=(48, 5.15),
                fontsize=9.5, color=RED, style="italic", ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    # --- Правая панель: что вообще убрали из 153 колонок ---
    ax2 = axes[1]
    cats = ["служебные\n(*_valid_frac,\n*_n_obs)", "шумовые\nпризнаки", "остаточный\nдубль"]
    counts = [len(SERVICE_COLS), len(NOISE_FEATURES), len(RESIDUAL_DUP_COLS)]
    bars2 = ax2.bar(cats, counts, color=[GREY, RED, GOLD], edgecolor="white")
    for bar, c in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(c),
                  ha="center", fontsize=12, fontweight="bold", color=DARK)
    ax2.set_ylabel("число колонок")
    ax2.set_ylim(0, max(counts) + 6)
    ax2.set_title(f"Всего убрано навсегда: {sum(counts)} из 153\n"
                   f"(153 -> 109 столбцов, 91 признак для обучения)", fontsize=11)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
