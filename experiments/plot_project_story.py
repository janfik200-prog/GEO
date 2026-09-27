"""Одна иллюстрация для короткого рассказа о проекте: вход → модель → совпадение.

Задача картинки — за один взгляд показать суть: критериальный метод геологов
строится на геологических слоях (разломы, фации, палеодолины), наша модель их
не видит вообще, она смотрит только на спутник, рельеф и геофизику — и
восстанавливает тот же результат. Текста минимум, всё несут карты.

Нижний ряд рисуется по `score_blind` из `experiments/honest_forecast_main.py` —
прогнозу, где каждая ячейка предсказана моделью, обучавшейся не ближе 15 км от
неё. Прежняя версия картинки брала прогноз со случайной перекрёстной проверки
по ячейкам и показывала совпадение 92%; проверка 02.09.2026 установила, что при
шаге сетки 500 м отложенная ячейка почти всегда окружена обучающими, и такая
цифра измеряет пространственную интерполяцию, а не работу признаков — столько
же даёт модель, знающая одни координаты. Цифра на картинке заменена.

Третья опорная величина, без которой совпадение нечитаемо: сколько даёт одна
география. Мишень крупноблочная и плавная, поэтому даже модель без единого
признака, знающая только «где», угадывает часть перспективной площади. Она
показана рядом со случайным уровнем как планка, которую результат обязан
превышать.

Оси намеренно убраны (в них номера столбцов сетки, для рассказа это шум) —
вместо них масштабная линейка и стрелка на север, чего для чтения карты
достаточно.

Вход: `data/processed/forecast_main.parquet` (создаётся
`experiments/train_test_main_sheet.py`).
Выход: `outputs/project_story.png`.

Запуск из корня репозитория: ``python -m experiments.plot_project_story``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target  # noqa: E402

BG = "#101219"
FG = "#e8e6e3"
DIM = "#9aa0ad"

# Превью входных данных: (колонка, подпись, палитра). Подобраны так, чтобы
# показать РАЗНОРОДНОСТЬ источников, а не лучшие по важности признаки.
# Все шесть входят в итоговый набор: показывать в превью признак, который
# модель не использует (гравику, магнитное поле), было бы прямой неправдой.
PREVIEW = [
    ("dem_elev", "рельеф", "terrain"),
    ("relief2_catch_log", "водосборы", "YlGnBu"),
    ("s2_b11", "спутник", "bone"),
    ("l8_swir16", "тепловая съёмка", "inferno"),
    ("s1_vv", "радар", "pink"),
    ("lin_dens", "линеаменты", "afmhot"),
]


def norm(a: np.ndarray) -> np.ndarray:
    """Растяжка по 2-98 перцентилю — иначе выбросы съедают весь контраст."""
    a = np.asarray(a, dtype=float)
    lo, hi = np.nanpercentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo) if hi > lo else a * 0, 0, 1)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import FancyArrow, Rectangle
    from scipy import stats as sps

    meta, prognoz = criterial_target.load_prognoz_grid()
    d = pd.read_parquet(config.PROCESSED_DIR / "forecast_main_honest.parquet")
    feat = criterial_target.training_features("v5_rebuilt")

    pool = np.flatnonzero(d["valid"].to_numpy())
    crit = -d["prognoz"].to_numpy()
    model = d["score_blind"].to_numpy()        # обучение не ближе 15 км, см. докстринг
    geo = d["score_geo"].to_numpy()            # планка: модель без признаков
    shape = (meta.prf, meta.pic)

    def rank(v):
        return (sps.rankdata(v) / v.size).reshape(shape)

    g_crit, g_model = rank(crit), rank(model)
    top_c = (crit >= np.quantile(crit[pool], 1 - config.CREF_AREA)).reshape(shape)
    top_m = (model >= np.quantile(model[pool], 1 - config.CREF_AREA)).reshape(shape)
    top_g = (geo >= np.quantile(geo[pool], 1 - config.CREF_AREA)).reshape(shape)
    hit = 100 * (top_c & top_m).sum() / max(top_c.sum(), 1)
    hit_geo = 100 * (top_c & top_g).sum() / max(top_c.sum(), 1)

    # Ровно тот набор, на котором обучалась модель, а не всё, что есть в файле.
    n_feat = len([c for c in json.loads(
        (config.PROCESSED_DIR / "honest_feature_set.json").read_text(encoding="utf-8")
    )["признаки"] if c in feat.columns])
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 3.1], hspace=0.10, wspace=0.06,
                          left=0.03, right=0.97, top=0.86, bottom=0.06)

    # --- верх: разнородные входные данные ---
    for i, (col, label, cmap) in enumerate(PREVIEW):
        ax = fig.add_subplot(gs[0, i])
        img = norm(feat[col].to_numpy()).reshape(shape) if col in feat.columns \
            else np.zeros(shape)
        ax.imshow(img, cmap=cmap)
        ax.set_title(label, color=DIM, fontsize=11, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2c3040")

    fig.text(0.5, 0.965, f"{n_feat} признаков — спутник, рельеф, радар",
             ha="center", color=FG, fontsize=18, fontweight="bold")
    fig.text(0.5, 0.935, "ни одного геологического слоя из формулы геологов",
             ha="center", color=DIM, fontsize=12.5)

    # --- низ: эталон, модель, совпадение ---
    def big(ax, grid, title, sub, sub_color=DIM):
        ax.imshow(grid, cmap="magma", vmin=0, vmax=1)
        ax.contour(top_c.astype(float), levels=[0.5], colors="#4dd7ff", linewidths=1.1)
        ax.set_title(title, color=FG, fontsize=15, pad=9, fontweight="bold")
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center",
                color=sub_color, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#2c3040")

    ax1 = fig.add_subplot(gs[1, 0:2])
    big(ax1, g_crit, "Метод геологов", "разломы · фации · палеодолины")

    ax2 = fig.add_subplot(gs[1, 2:4])
    big(ax2, g_model, "Наша модель", "обучение не ближе 15 км от каждой точки")

    # Масштабная линейка и север — вместо осей.
    # Линейка на тёмной подложке: под ней светлый участок карты, без неё не видна.
    km20 = 20_000 / meta.dx
    ax1.add_patch(Rectangle((4, meta.prf - 20), km20 + 4, 18, color=BG, alpha=0.75))
    ax1.add_patch(Rectangle((6, meta.prf - 7), km20, 2.6, color=FG))
    ax1.text(6 + km20 / 2, meta.prf - 9, "20 км", ha="center", va="bottom",
             color=FG, fontsize=10)
    ax1.add_patch(FancyArrow(meta.pic - 12, 22, 0, -13, width=1.4, head_width=5,
                             head_length=6, color=FG))
    ax1.text(meta.pic - 12, 27, "С", ha="center", color=FG, fontsize=11)

    # Третья панель — совпадение топ-зон, самая содержательная.
    agree = np.zeros(shape, dtype=int)
    agree[top_c & ~top_m] = 1
    agree[~top_c & top_m] = 2
    agree[top_c & top_m] = 3
    ax3 = fig.add_subplot(gs[1, 4:6])
    ax3.imshow(agree, cmap=ListedColormap(["#16181f", "#3b6fa0", "#a04a3b", "#ffd34d"]),
               vmin=0, vmax=3, interpolation="nearest")
    ax3.set_title("Совпадение", color=FG, fontsize=15, pad=9, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_color("#2c3040")
    # Цифра — в верхней трети панели, там свободно; подложка, чтобы не спорить
    # с жёлтыми пятнами, если разметка сдвинется на других данных.
    ax3.text(0.5, 0.80, f"{hit:.0f}%", transform=ax3.transAxes, ha="center", va="center",
             color=FG, fontsize=76, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec="none", alpha=0.72))
    ax3.text(0.5, -0.045, "перспективной площади найдено заново",
             transform=ax3.transAxes, ha="center", color=DIM, fontsize=11)
    # Без этих двух планок цифра нечитаема: часть площади угадывается вообще
    # без данных, просто потому что перспективные зоны крупные и плавные.
    ax3.text(0.5, 0.655, f"случайно — 10%    ·    по одной географии — {hit_geo:.0f}%",
             transform=ax3.transAxes, ha="center", va="center", color=DIM, fontsize=11.5,
             bbox=dict(boxstyle="round,pad=0.22", fc=BG, ec="none", alpha=0.72))

    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ("#ffd34d", "#3b6fa0", "#a04a3b")]
    ax3.legend(handles, ["оба метода", "только геологи", "только модель"],
               loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3,
               fontsize=10, framealpha=0.0, labelcolor=FG,
               handlelength=1.1, columnspacing=1.4, borderpad=0.2)

    # Стрелка «вход -> результат», между рядами.
    fig.text(0.5, 0.625, "▼", ha="center", color="#4dd7ff", fontsize=24)

    out = ROOT / "outputs" / "project_story.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"совпадение top-{config.CREF_AREA:.0%}: модель {hit:.1f}% "
          f"({int((top_c & top_m).sum())} из {int(top_c.sum())} ячеек), "
          f"одна география {hit_geo:.1f}%, случайно {config.CREF_AREA:.0%}")
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
