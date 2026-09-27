"""Две карты рядом: расчёт ВНИГНИ на области переноса и наше воссоздание.

ЗАЧЕМ. Показать одним рисунком, что именно сравнивается за рамкой основного
листа: слева критериальная карта организации (by_crt из сборки
ПРОГНОЗ_ВОСПРОИЗВ, 24.09.2026), справа наша питоновская реконструкция той же
формулы, посчитанная на той же площади по тем же шести факторным слоям.
Обе панели в одной шкале классов и в одних границах.

ВХОД: data/processed/criterial_reconstruct_multisheet.parquet
      (пишет experiments/criterial_reconstruct_multisheet.py)
ВЫХОД: outputs/02_критериальный_метод/criterial_side_by_side.png
Запуск из корня: python -m experiments.criterial_side_by_side
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, mapstyle, outputs  # noqa: E402
from experiments.criterial_reconstruct_multisheet import panel  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PIC, PRF = 230, 452


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR
                         / "criterial_reconstruct_multisheet.parquet")
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    msk = df["в_маске"].to_numpy(bool)
    in_sheet = df["в_габаритах_листа"].to_numpy(bool)
    shape = (PRF, PIC)
    flip = y.reshape(shape)[0, 0] < y.reshape(shape)[-1, 0]

    def grid(v):
        out = np.where(msk, v, np.nan).reshape(shape)
        return out[::-1] if flip else out

    def classes(v):
        cuts = np.percentile(v[msk], mapstyle.CLASS_EDGES[1:-1])
        return grid(np.digitize(v, cuts).astype(float))

    ext = (x.min() - 250, x.max() + 250, y.min() - 250, y.max() + 250)
    sx, sy = x[msk & in_sheet], y[msk & in_sheet]
    rect = (sx.min() - 250, sy.min() - 250,
            sx.max() - sx.min() + 500, sy.max() - sy.min() + 500)

    theirs = -df["by_crt"].to_numpy(float)
    ours = -df["воссоздание_вся_область"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 10.2), constrained_layout=True)
    panel(axes[0], classes(theirs), ext,
          "Критериальный метод ВНИГНИ, расчёт организации\n"
          "(ГИС Интегро, свойство by_crt)", rect)
    panel(axes[1], classes(ours), ext,
          "Наше воссоздание того же метода\n(Python, мера Плюты)", rect)

    handles = [Patch(facecolor=c, edgecolor="#777777", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    handles.append(plt.Line2D([], [], color="#101010", ls="--",
                              label="габариты основного листа"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6,
               frameon=False, fontsize=10, title="Классы перспективности")
    fig.suptitle("Область расчёта 7557 кв.км: за рамкой листа — территория "
                 "переноса", fontsize=13)
    path = outputs.png("criterial_side_by_side.png")
    mapstyle.save_fig(fig, path, dpi=180)
    print("Карта:", path)


if __name__ == "__main__":
    main()
