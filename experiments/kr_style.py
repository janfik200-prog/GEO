"""Единый стиль схем статьи: шапка, цветные карточки, выноски, панели.

Оформление журнальное и нарочито контрастное: насыщенные заголовочные
плашки с белым текстом, светлое тело карточки, цветная рамка. Палитра
различает уровни представления, а не украшает.

Важное правило: схема рисуется в той ширине, в которой вставляется в текст
(``ШИРИНА_СМ`` — полоса набора 16 см), поэтому ``fontsize`` — настоящие
пункты на бумаге.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    FancyArrowPatch, FancyBboxPatch, Polygon)

ШИРИНА_СМ = 16.0
ДЮЙМ = ШИРИНА_СМ / 2.54
DPI = 400

ТЁМНЫЙ = "#13222b"
ТЕКСТ = "#384a55"
СЕРЫЙ = "#5f7079"

#: (светлая заливка, насыщенный цвет, тёмный цвет надписи)
ТОНА = {
    "синий": ("#dcebfb", "#1f7ac4", "#0d4576"),
    "зелёный": ("#d9f2e0", "#1f9e58", "#0e5e33"),
    "жёлтый": ("#fdeec6", "#e2a50f", "#875f00"),
    "красный": ("#fde0da", "#e0503a", "#8e2616"),
    "сирень": ("#e9ddfa", "#7b45c9", "#45227d"),
    "охра": ("#ffe5cd", "#f07d1c", "#9a4603"),
    "бирюза": ("#d4f1f1", "#0fa3a3", "#06595a"),
    "серый": ("#eceff1", "#6f8089", "#36464f"),
}

ЗАГ = 8.6
ОСН = 7.2
МЕЛК = 6.4


def холст(высота_см: float):
    fig = plt.figure(figsize=(ДЮЙМ, высота_см / 2.54), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def шапка(ax, текст: str, тон: str = "синий", высота: float = 9.0) -> float:
    """Заголовок: сплошная насыщенная плашка во всю ширину, текст белый."""
    _, р, _ = ТОНА[тон]
    низ = 100.0 - высота
    ax.add_patch(FancyBboxPatch(
        (1.0, низ), 98.0, высота - 1.0,
        boxstyle="round,pad=0,rounding_size=1.2", linewidth=0,
        facecolor=р, zorder=3))
    ax.text(50.0, низ + (высота - 1.0) / 2, текст, ha="center", va="center",
            fontsize=ЗАГ, color="white", fontweight="bold", zorder=4)
    return низ


def карточка(ax, x, y, w, h, заголовок, строки=(), *, тон="синий",
             кегль=ОСН, центр=False, жирно_строки=False,
             полоса: float = 0.0) -> None:
    """Карточка: цветная рамка, заголовок, список строк.

    ``полоса`` > 0 рисует сплошную заголовочную плашку этой высоты с белым
    текстом — так карточка читается издалека.
    """
    з, р, ц = ТОНА[тон]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
        linewidth=1.5, edgecolor=р, facecolor=з, zorder=3))
    if полоса > 0:
        ax.add_patch(FancyBboxPatch(
            (x, y + h - полоса), w, полоса,
            boxstyle="round,pad=0,rounding_size=1.0", linewidth=0,
            facecolor=р, zorder=4))
        ax.text(x + w / 2, y + h - полоса / 2, заголовок, ha="center",
                va="center", fontsize=кегль, color="white",
                fontweight="bold", zorder=5, linespacing=1.1)
        шаг = (h - полоса - 1.2) / max(len(строки), 1)
        yy = y + h - полоса - шаг / 2 - 0.4
        for с in строки:
            ax.text(x + w / 2 if центр else x + 2.2, yy, с,
                    ha="center" if центр else "left", va="center",
                    fontsize=МЕЛК, color=ТЕКСТ, zorder=5,
                    fontweight="bold" if жирно_строки else "normal")
            yy -= шаг
        return
    if центр and len(строки) == 1:
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
            ax.text(x + w / 2 if центр else x + 2.2, yy, с,
                    ha="center" if центр else "left", va="top",
                    fontsize=МЕЛК, color=ТЕКСТ, zorder=4,
                    fontweight="bold" if жирно_строки else "normal")
            yy -= шаг
    else:
        ax.text(x + w / 2, y + h / 2, заголовок, ha="center", va="center",
                fontsize=кегль, color=ц, fontweight="bold", zorder=4,
                linespacing=1.2)


def стрелка(ax, x1, y1, x2, y2, *, цвет=None, стиль="solid",
            толщина=1.6) -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=толщина, color=цвет or СЕРЫЙ, linestyle=стиль,
        shrinkA=0, shrinkB=0, zorder=2))


def выноска(ax, x_точки, y_точки, x_текста, y_текста, текст, *,
            тон="охра", кегль=МЕЛК) -> None:
    """Подпись с указателем: насыщенная плашка, белый текст."""
    _, р, _ = ТОНА[тон]
    ax.annotate(текст, xy=(x_точки, y_точки), xytext=(x_текста, y_текста),
                fontsize=кегль, color="white", fontweight="bold", zorder=7,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=р, linewidth=1.1,
                                shrinkA=2, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=р,
                          edgecolor=р, linewidth=0.9))


def панель(ax, x, y, w, h, заголовок, строки, *, тон="серый") -> None:
    """Сноска в рамке: пояснение под схемой, заголовок цветной."""
    з, р, ц = ТОНА[тон]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=1.3, edgecolor=р, facecolor=з, zorder=3))
    ax.text(x + 2.0, y + h - 1.5, заголовок, ha="left", va="top",
            fontsize=МЕЛК + 0.5, color=ц, fontweight="bold", zorder=4)
    yy = y + h - 5.8
    for с in строки:
        ax.text(x + 2.0, yy, с, ha="left", va="top", fontsize=МЕЛК,
                color=ТЕКСТ, zorder=4)
        yy -= 4.2


def сохранить(fig, путь) -> None:
    путь.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(путь, facecolor="white")
    plt.close(fig)


def многоугольник(ax, точки, текст, *, тон="синий", кегль=ОСН, низ="",
                  жирный=True, залить=False):
    """Фигура с подписью по центру: трапеция, ромб, шестиугольник.

    ``залить`` даёт насыщенную заливку и белый текст.
    """
    з, р, ц = ТОНА[тон]
    ax.add_patch(Polygon(точки, closed=True, linewidth=1.6, edgecolor=р,
                         facecolor=р if залить else з, zorder=3))
    xs = [т[0] for т in точки]
    ys = [т[1] for т in точки]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    сдвиг = 1.7 if низ else 0.0
    ax.text(cx, cy + сдвиг, текст, ha="center", va="center", fontsize=кегль,
            color="white" if залить else ц,
            fontweight="bold" if жирный else "normal", zorder=4,
            linespacing=1.15)
    if низ:
        ax.text(cx, cy - 2.8, низ, ha="center", va="center", fontsize=МЕЛК,
                color="#f2f6f8" if залить else ТЕКСТ, zorder=4)
    return cx, cy


def трапеция(ax, y, h, w_верх, w_низ, текст, *, низ="", тон="синий",
             центр_x=50.0, кегль=ОСН, залить=False):
    """Ярус пирамиды: ширина кодирует объём уровня."""
    точки = [(центр_x - w_верх / 2, y + h), (центр_x + w_верх / 2, y + h),
             (центр_x + w_низ / 2, y), (центр_x - w_низ / 2, y)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль, низ=низ,
                         залить=залить)


def ромб(ax, cx, cy, w, h, текст, *, тон="жёлтый", кегль=ОСН, залить=False):
    точки = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
             (cx - w / 2, cy)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль,
                         залить=залить)


def параллелограмм(ax, x, y, w, h, текст, *, наклон=2.2, тон="серый",
                   кегль=ОСН, низ="", залить=False):
    точки = [(x + наклон, y + h), (x + w, y + h), (x + w - наклон, y),
             (x, y)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль, низ=низ,
                         залить=залить)


def шестиугольник(ax, x, y, w, h, текст, *, низ="", тон="зелёный",
                  кегль=ОСН, залить=True):
    скос = h * 0.42
    точки = [(x + скос, y + h), (x + w - скос, y + h), (x + w, y + h / 2),
             (x + w - скос, y), (x + скос, y), (x, y + h / 2)]
    return многоугольник(ax, точки, текст, тон=тон, кегль=кегль, низ=низ,
                         залить=залить)
