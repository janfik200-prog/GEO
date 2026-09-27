"""Четыре картинки для дорожной карты: по одной на пункт, без подписей.

ЗАЧЕМ. На слайде дорожной карты подписи уже стоят рядом с картинкой, поэтому
внутри самих картинок текста быть не должно — иначе он дублируется и мельчит.
Здесь собраны четыре иллюстрации одинакового размера и с прозрачным фоном,
которые ложатся на любую подложку:

1. `roadmap_1_геопризнаки` — пространственные геопризнаки: четыре реальных слоя
   листа стопкой (рельеф, линеаменты, спектр окисления, гравика);
2. `roadmap_2_взаимосвязи` — поиск скрытых взаимосвязей: граф связей между 58
   признаками, ребро проводится при |r| >= 0.55, цвет узла — группа съёмки;
3. `roadmap_3_перспективность` — карта перспективных зон: итоговая карта листа
   в пяти классах с контуром верхних 10%;
4. `roadmap_4_приоритет` — приоритет для изучения: пять самых сильных связных
   зон из верхних 10%, цвет по рангу от первого к пятому.

Данные не пересчитываются: карта берётся из `full_sheet_map_ядро5км.parquet`,
признаки — из `dataset_v5_rebuilt`. Ни одной подписи, ни одной оси.

Запуск из корня: ``python -m experiments.roadmap_icons``
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402
from scipy import ndimage, stats as sps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, criterial_target, mapstyle,  # noqa: E402
                 outputs as outdir)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLD = "#c89b2c"
DARK = "#2b2b2b"
GREY = "#9a9a9a"

#: Сторона картинки в дюймах: все четыре одного размера, чтобы на слайде
#: вставали в ряд без подгонки.
SIDE = 3.2
DPI = 300

#: Слой -> своя шкала цвета. Признаки взяты разных природ, чтобы стопка читалась
#: как «разные съёмки об одной территории», а не как четыре копии рельефа.
LAYERS = (
    ("dem_elev", "#f5f2ea", "#6b6257"),        # рельеф
    ("lin_dens", "#eef2f7", "#2f6fa8"),        # линеаменты
    ("s2_iron_ox", "#fbf2de", "#c2661f"),      # окисление по спектру
    ("gm_gr_ost_35", "#eef4ee", "#3f8a45"),    # гравика
)

#: Цвет узла графа по группе признака.
GROUP_COLOR = {
    "dem": "#6b6257", "relief2": "#8a7f6d", "lin": "#2f6fa8",
    "s2": "#c2661f", "l8": "#a3122b", "s1": "#6a4c9c", "dist": "#3f8a45",
}

#: Пять рангов приоритета: от первого (тёмный) к пятому (светлый).
RANK_COLORS = ("#7d0d20", "#a3122b", "#d1462a", "#e2662c", "#f2b544")
N_ZONES = len(RANK_COLORS)

#: Зона меньше этого числа ячеек 500 м (2.5 км²) — не участок для изучения,
#: а обрывок контура на границе класса.
MIN_ZONE_CELLS = 12


def save(fig, name: str) -> None:
    """Запись с прозрачным фоном и повтором: файл может быть занят."""
    import time

    path = outdir.png(name + ".png")
    for k in range(4):
        try:
            fig.savefig(path, dpi=DPI, transparent=True,
                        bbox_inches="tight", pad_inches=0.02)
            break
        except OSError:
            if k == 3:
                raise
            time.sleep(1.5)
    plt.close(fig)
    print("OK: %s" % path.relative_to(ROOT))


def ranked(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Признак в долях 0-1 по рангу: слой читается независимо от единиц."""
    out = np.full(v.size, np.nan)
    good = mask & np.isfinite(v)
    out[good] = sps.rankdata(v[good]) / max(int(good.sum()), 1)
    return out


def ramp(low: str, high: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("", [low, high])


def fig_features(feat: pd.DataFrame, valid: np.ndarray, shape) -> None:
    """Пункт 1: стопка слоёв листа в изометрии."""
    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    step = 0.62
    for i, (col, low, high) in enumerate(LAYERS):
        v = ranked(feat[col].to_numpy(dtype=float), valid).reshape(shape)
        # Плоскость кладётся набок: сжатие по высоте и сдвиг по горизонтали
        # дают привычный вид «слоёв ГИС», наложенных друг на друга.
        tr = (Affine2D().scale(1.0, 0.46).skew_deg(26, 0)
              .translate(0.0, i * step) + ax.transData)
        ax.imshow(np.ma.masked_invalid(v), origin="upper", extent=(0, 1, 0, 1),
                  transform=tr, cmap=ramp(low, high), interpolation="nearest",
                  zorder=i * 2)
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        ax.add_patch(Polygon(corners, closed=True, transform=tr, fill=False,
                             edgecolor=DARK, linewidth=0.9, zorder=i * 2 + 1))
    ax.set_xlim(-0.06, 1.42)
    ax.set_ylim(-0.08, 4 * step + 0.42)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "roadmap_1_геопризнаки")


def fig_links(feat: pd.DataFrame, valid: np.ndarray) -> None:
    """Пункт 2: граф связей между признаками рабочего набора."""
    names = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                       .read_text(encoding="utf-8"))["признаки"]
    cols = [c for c in names if c in feat.columns]
    X = feat.loc[valid, cols]
    r = np.array(X.corr(method="spearman").to_numpy(), dtype=float)
    np.fill_diagonal(r, 0.0)

    g = nx.Graph()
    g.add_nodes_from(range(len(cols)))
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(r[i, j]) >= 0.55:
                g.add_edge(i, j, weight=abs(r[i, j]))
    g.remove_nodes_from([n for n, d in g.degree() if d == 0])
    pos = nx.spring_layout(g, seed=config.RANDOM_STATE, weight="weight",
                           iterations=400)

    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    segs, widths, alphas = [], [], []
    for i, j, d in g.edges(data=True):
        segs.append([pos[i], pos[j]])
        widths.append(0.35 + 1.5 * (d["weight"] - 0.55) / 0.45)
        alphas.append(0.18 + 0.5 * (d["weight"] - 0.55) / 0.45)
    ax.add_collection(LineCollection(segs, linewidths=widths, colors=GREY,
                                     alpha=None, zorder=1,
                                     capstyle="round",
                                     array=None))
    # LineCollection одной альфой красит все рёбра одинаково; прозрачность
    # задаётся поцветно, поэтому цвет собирается из серого и своей альфы.
    ax.collections[-1].set_color([(0.6, 0.6, 0.6, a) for a in alphas])

    deg = np.array([g.degree(n) for n in g.nodes()], dtype=float)
    size = 26 + 34 * deg / max(deg.max(), 1.0)
    color = [GROUP_COLOR.get(cols[n].split("_")[0], GREY) for n in g.nodes()]
    xy = np.array([pos[n] for n in g.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, linewidths=0.6,
               edgecolors="white", zorder=3)
    ax.set_aspect("equal")
    ax.margins(0.06)
    ax.axis("off")
    save(fig, "roadmap_2_взаимосвязи")


def load_map(shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Итоговая карта листа: балл, верхние 10%, рабочая маска."""
    df = pd.read_parquet(config.PROCESSED_DIR / "full_sheet_map_ядро5км.parquet")
    score = df["score"].to_numpy(dtype=float)
    top = df["верхние_10"].to_numpy(dtype=bool)
    valid = df["валидная"].to_numpy(dtype=bool)
    assert score.size == shape[0] * shape[1]
    return score, top, valid


def fig_prospect(score, top, valid, shape) -> None:
    """Пункт 3: карта перспективных зон в классах оформления проекта."""
    ext = [0, shape[1], 0, shape[0]]
    pct = mapstyle.percentile(score, valid)
    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    ax.imshow(np.ma.masked_invalid(pct.reshape(shape)), origin="upper",
              extent=ext, cmap=mapstyle.class_cmap(),
              norm=mapstyle.class_norm(), interpolation="nearest")
    mapstyle.outline(ax, top, shape, ext, flip=True, color="#101010", lw=0.7)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "roadmap_3_перспективность")


def fig_priority(score, top, valid, shape) -> None:
    """Пункт 4: пять сильнейших зон верхних 10%, цвет по рангу."""
    lab, n = ndimage.label(top.reshape(shape),
                           structure=np.ones((3, 3), dtype=int))
    flat = lab.ravel()
    rows = []
    for k in range(1, n + 1):
        cells = flat == k
        if cells.sum() < MIN_ZONE_CELLS:
            continue
        rows.append((k, int(cells.sum()), float(score[cells].mean())))
    zones = sorted(rows, key=lambda t: t[2], reverse=True)[:N_ZONES]
    print("зон в верхних 10%: %d, крупнее %d ячеек: %d, показаны %d"
          % (n, MIN_ZONE_CELLS, len(rows), len(zones)))

    rank = np.full(flat.size, np.nan)
    for i, (k, _, _) in enumerate(zones):
        rank[flat == k] = i

    ext = [0, shape[1], 0, shape[0]]
    fig, ax = plt.subplots(figsize=(SIDE, SIDE))
    # Фон — сам лист одним тоном: приоритет читается как «эти пять из всей
    # площади», а не как отдельно висящие пятна.
    back = np.where(valid, 1.0, np.nan).reshape(shape)
    ax.imshow(np.ma.masked_invalid(back), origin="upper", extent=ext,
              cmap=ListedColormap(["#ece9e4"]), interpolation="nearest")
    ax.imshow(np.ma.masked_invalid(rank.reshape(shape)), origin="upper",
              extent=ext, cmap=ListedColormap(RANK_COLORS[:len(zones)]),
              norm=BoundaryNorm(np.arange(len(zones) + 1) - 0.5, len(zones)),
              interpolation="nearest")
    for i, (k, _, _) in enumerate(zones):
        mapstyle.outline(ax, flat == k, shape, ext, flip=True,
                         color="#101010", lw=0.6)
    mapstyle.outline(ax, valid, shape, ext, flip=True, color=DARK, lw=0.8)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "roadmap_4_приоритет")


def main() -> None:
    meta, _ = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    score, top, vmap = load_map(shape)

    fig_features(feat, valid, shape)
    fig_links(feat, valid)
    fig_prospect(score, top, vmap, shape)
    fig_priority(score, top, vmap, shape)


if __name__ == "__main__":
    main()
