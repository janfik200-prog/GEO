"""Карта квадрантов «оценка × обеспеченность» и распределение по ним.

Зачем: главное утверждение представления — что два выхода ячейки нельзя
сводить в один. Эта пара панелей показывает, чем они различаются на реальном
листе: где высокая оценка подкреплена наблюдениями, а где она означает
запрос на съёмку.

Данные — пробная свёртка варианта А (`knowledge_map_v0_2.parquet`), порог
оценки берётся по верхнему децилю, порог обеспеченности — 0,8, то есть
четыре наблюдаемых звена из пяти.

Запуск из корня репозитория::

    python -m experiments.kr_quadrants
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ИСХОД = ROOT / "data" / "processed" / "knowledge_map_v0_2.parquet"
ВЫХОД = ROOT / "outputs" / "14_статья_знания" / "kr7_квадранты.png"

ШИРИНА_СМ, ВЫСОТА_СМ, DPI = 16.0, 8.4, 400
ПОРОГ_C = 0.8

ТЁМНЫЙ = "#13242d"
ТЕКСТ = "#3a4d59"

#: (имя, цвет) в порядке кода квадранта 0..3
КВАДРАНТЫ = [
    ("оценка низкая, обеспеченность низкая", "#e8eaec"),
    ("оценка низкая, обеспеченность высокая", "#9fb8c9"),
    ("оценка высокая, обеспеченность высокая", "#1f6f4a"),
    ("оценка высокая, обеспеченность низкая", "#d98a2b"),
]


def main() -> None:
    d = pd.read_parquet(ИСХОД)
    порог_p = float(d["итог"].quantile(0.90))
    выс_p = d["итог"] >= порог_p
    выс_c = d["уверенность"] >= ПОРОГ_C
    код = np.where(выс_p & выс_c, 2,
                   np.where(выс_p & ~выс_c, 3,
                            np.where(~выс_p & выс_c, 1, 0)))

    строк = int(d["row"].max()) + 1
    столбцов = int(d["col"].max()) + 1
    поле = np.full((строк, столбцов), np.nan)
    поле[d["row"].to_numpy(int), d["col"].to_numpy(int)] = код

    fig = plt.figure(figsize=(ШИРИНА_СМ / 2.54, ВЫСОТА_СМ / 2.54), dpi=DPI)
    ось_к = fig.add_axes([0.02, 0.07, 0.40, 0.76])
    ось_д = fig.add_axes([0.68, 0.20, 0.29, 0.58])

    ось_к.imshow(поле, origin="upper", interpolation="nearest",
                 cmap=ListedColormap([ц for _, ц in КВАДРАНТЫ]),
                 vmin=-0.5, vmax=3.5)
    ось_к.set_xticks([])
    ось_к.set_yticks([])
    for сторона in ось_к.spines.values():
        сторона.set_edgecolor("#9aa7b0")
    ось_к.set_title("Лист R-48-XI,XII: ячейки по квадрантам",
                    fontsize=7.6, color=ТЁМНЫЙ, fontweight="bold", pad=4)

    доли = [float((код == i).mean()) * 100 for i in range(4)]
    счёт = [int((код == i).sum()) for i in range(4)]
    порядок = [2, 3, 1, 0]
    y = np.arange(len(порядок))
    ось_д.barh(y, [доли[i] for i in порядок],
               color=[КВАДРАНТЫ[i][1] for i in порядок],
               edgecolor="#6b7d8a", linewidth=0.6, height=0.62)
    ось_д.set_yticks(y)
    ось_д.set_yticklabels([КВАДРАНТЫ[i][0].replace(", ", ",\n")
                           for i in порядок], fontsize=6.0, color=ТЕКСТ)
    ось_д.invert_yaxis()
    ось_д.set_xlabel("доля ячеек листа, %", fontsize=6.8, color=ТЕКСТ)
    ось_д.tick_params(axis="x", labelsize=6.4, colors=ТЕКСТ)
    for сторона in ("top", "right"):
        ось_д.spines[сторона].set_visible(False)
    for сторона in ("left", "bottom"):
        ось_д.spines[сторона].set_color("#9aa7b0")
    for i, кв in enumerate(порядок):
        подпись = ("%.1f %% (%s)" % (доли[кв], "{:,}".format(счёт[кв])
                   .replace(",", " "))).replace(".", ",")
        внутри = доли[кв] > 50.0
        ось_д.text(доли[кв] + (-1.5 if внутри else 1.2), i, подпись,
                   va="center", ha="right" if внутри else "left",
                   fontsize=5.9, color=ТЁМНЫЙ, fontweight="bold")
    ось_д.set_xlim(0, max(доли) * 1.12)
    ось_д.set_title("Распределение ячеек", fontsize=7.6, color=ТЁМНЫЙ,
                    fontweight="bold", pad=4)

    верх = счёт[2] + счёт[3]
    fig.text(0.02, 0.985, "Два выхода ячейки нельзя свести в один",
             ha="left", va="top", fontsize=8.4, color=ТЁМНЫЙ,
             fontweight="bold")
    fig.text(0.02, 0.945, "Из %s ячеек верхнего дециля оценки наблюдениями "
             "обеспечены только %d — остальные означают запрос на съёмку"
             % ("{:,}".format(верх).replace(",", " "), счёт[2]), ha="left", va="top", fontsize=6.6,
             color=ТЕКСТ)
    fig.text(0.44, 0.045, "Порог оценки — верхний дециль (%s);\n"
             "порог обеспеченности — 0,8, то есть\n"
             "четыре наблюдаемых звена из пяти."
             % ("%.2f" % порог_p).replace(".", ","),
             ha="left", va="bottom", fontsize=6.0, color=ТЕКСТ,
             linespacing=1.35)

    ВЫХОД.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ВЫХОД, facecolor="white")
    plt.close(fig)
    print("порог оценки: %.3f" % порог_p)
    for и, (имя, _) in enumerate(КВАДРАНТЫ):
        print("  %-40s %6d  %5.1f %%" % (имя, счёт[и], доли[и]))
    print("нарисовано:", ВЫХОД)


if __name__ == "__main__":
    main()
