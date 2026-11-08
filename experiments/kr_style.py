"""Единый стиль схем статьи: шапка, цветные карточки, выноски, панели.

Сделано по образцам журнальных иллюстраций: заголовок в рамке-шапке, карточки
с цветной рамкой и цветным заголовком, выноски с указателем, сноски в рамке,
палитра по смыслу уровня.

Важное правило: схема рисуется в той ширине, в которой вставляется в текст
(``ШИРИНА_СМ``), поэтому ``fontsize`` — настоящие пункты на бумаге. Раньше
схемы рисовались шире места вставки, и кегль внутри падал почти в полтора
раза.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ШИРИНА_СМ = 11.5
ДЮЙМ = ШИРИНА_СМ / 2.54
DPI = 400

ТЁМНЫЙ = "#16262f"
ТЕКСТ = "#43555f"
СЕРЫЙ = "#6d7d87"

#: (заливка, рамка, цвет заголовка)
ТОНА = {
    "синий": ("#e6eff8", "#3f7bb0", "#1f4f78"),
    "зелёный": ("#e7f3ea", "#4e9b63", "#2c6b3d"),
    "жёлтый": ("#fdf5e1", "#c79a2a", "#8a680f"),
    "красный": ("#fbeae8", "#c0594f", "#8d3a32"),
    "сирень": ("#efe9f8", "#7a5aa6", "#533774"),
    "охра": ("#fdf0e2", "#c87f2a", "#8a5311"),
    "серый": ("#f0f1f2", "#9aa5ac", "#4f5a61"),
}

ЗАГ = 8.2
ОСН = 7.0
МЕЛК = 6.2


def холст(высота_см: float):
    fig = plt.figure(figsize=(ДЮЙМ, высота_см / 2.54), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def шапка(ax, текст: str, тон: str = "синий", высота: float = 9.0) -> float:
    """Заголовок в рамке во всю ширину; возвращает низ шапки."""
    з, р, ц = ТОНА[тон]
    низ = 100.0 - высота
    ax.add_patch(FancyBboxPatch(
        (1.0, низ), 98.0, высота - 1.0,
        boxstyle="round,pad=0,rounding_size=1.2", linewidth=1.3,
        edgecolor=р, facecolor=з, zorder=3))
    ax.text(50.0, низ + (высота - 1.0) / 2, текст, ha="center", va="center",
            fontsize=ЗАГ, color=ц, fontweight="bold", zorder=4)
    return низ


def карточка(ax, x, y, w, h, заголовок, строки=(), *, тон="синий",
             кегль=ОСН, центр=False, жирно_строки=False) -> None:
    """Карточка с цветной рамкой, цветным заголовком и списком строк."""
    з, р, ц = ТОНА[тон]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.4, edgecolor=р, facecolor=з, zorder=3))
    if центр and len(строки) == 1:
        # Короткая карточка: заголовок и одна строка делят высоту поровну.
        ax.text(x + w / 2, y + h * 0.63, заголовок, ha="center",
                va="center", fontsize=кегль, color=ц, fontweight="bold",
                zorder=4, linespacing=1.1)
        ax.text(x + w / 2, y + h * 0.26, строки[0], ha="center",
                va="center", fontsize=МЕЛК, color=ТЕКСТ, zorder=4)
    elif строки:
        ax.text(x + w / 2, y + h - 1.4, заголовок, ha="center", va="top",
                fontsize=кегль, color=ц, fontweight="bold", zorder=4)
        шаг = (h - 5.2) / max(len(строки), 1)
        yy = y + h - 5.2
        for с in строки:
            ax.text(x + w / 2 if центр else x + 2.0, yy, с,
                    ha="center" if центр else "left", va="top",
                    fontsize=МЕЛК, color=ТЕКСТ, zorder=4,
                    fontweight="bold" if жирно_строки else "normal")
            yy -= шаг
    else:
        ax.text(x + w / 2, y + h / 2, заголовок, ha="center", va="center",
                fontsize=кегль, color=ц, fontweight="bold", zorder=4,
                linespacing=1.2)


def стрелка(ax, x1, y1, x2, y2, *, цвет=None, стиль="solid",
            толщина=1.3) -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
        linewidth=толщина, color=цвет or СЕРЫЙ, linestyle=стиль,
        shrinkA=0, shrinkB=0, zorder=2))


def выноска(ax, x_точки, y_точки, x_текста, y_текста, текст, *,
            тон="охра", кегль=МЕЛК) -> None:
    """Подпись с указателем, как на журнальных картах."""
    _, р, ц = ТОНА[тон]
    ax.annotate(текст, xy=(x_точки, y_точки), xytext=(x_текста, y_текста),
                fontsize=кегль, color=ц, fontweight="bold", zorder=7,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=р, linewidth=0.9,
                                shrinkA=2, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor=р, linewidth=0.9))


def панель(ax, x, y, w, h, заголовок, строки, *, тон="серый") -> None:
    """Сноска в рамке: пояснение под схемой."""
    з, р, ц = ТОНА[тон]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=1.1, edgecolor=р, facecolor=з, zorder=3))
    ax.text(x + 1.8, y + h - 1.5, заголовок, ha="left", va="top",
            fontsize=МЕЛК + 0.4, color=ц, fontweight="bold", zorder=4)
    yy = y + h - 5.8
    for с in строки:
        ax.text(x + 1.8, yy, с, ha="left", va="top", fontsize=МЕЛК,
                color=ТЕКСТ, zorder=4)
        yy -= 4.2


def сохранить(fig, путь) -> None:
    путь.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(путь, facecolor="white")
    plt.close(fig)


def многоугольник(ax, точки, текст, *, тон="синий", кегль=ОСН, низ="",
                  жирный=True):
    """Произвольная фигура с подписью по центру: трапеция, ромб, шестиугольник."""
    from matplotlib.patches import Polygon
    з, р, ц = ТОНА[тон]
    ax.add_patch(Polygon(точки, closed=True, linewidth=1.4, edgecolor=р,
                         facecolor=з, zorder=3))
    xs = [т[0] for т in точки]
    ys = [т[1] for т in точки]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    сдвиг = 1.6 if низ else 0.0
    ax.text(cx, cy + сдвиг, текст, ha="center", va="center", fontsize=кегль,
            color=ц, fontweight="bold" if жирный else "normal", zorder=4,
            linespacing=1.15)
    if низ:
        ax.text(cx, cy - 2.6, низ, ha="center", va="center", fontsize=МЕЛК,
                color=ТЕКСТ, zorder=4)
    return cx, cy


def трапеция(ax, y, h, w_верх, w_низ, текст, *, низ="", тон="синий",
             центр_x=50.0, кегль=ОСН):
    """Ярус пирамиды: ширина кодирует объём уровня."""
    точки = [(центр_x - w_верх / 2, y + h), (центр_x + w_верх / 2, y + h),
             (центр_x + w_низ / 2, y), (центр_x - w_низ / 2, y)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль, низ=низ)


def ромб(ax, cx, cy, w, h, текст, *, тон="жёлтый", кегль=ОСН):
    точки = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
             (cx - w / 2, cy)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль)


def параллелограмм(ax, x, y, w, h, текст, *, наклон=3.0, тон="серый",
                   кегль=ОСН, низ=""):
    точки = [(x + наклон, y + h), (x + w, y + h), (x + w - наклон, y),
             (x, y)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль, низ=низ)
