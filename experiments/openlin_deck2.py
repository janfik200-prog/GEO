"""Сборка презентации заново: одна мысль — один слайд, весь счёт — формулами.

Принципы вёрстки:

* на слайде заголовок, одна крупная картинка и максимум одна строка подписи;
* каждая расчётная величина показана формулой (карточки
  :mod:`experiments.openlin_formulas`), а не пересказом словами;
* схемы идут по одной на слайд (:mod:`experiments.openlin_slides2`), поэтому
  картинка занимает почти всю площадь, а подписи на ней читаются из зала.

Прогон пересобирает файл целиком из резервной копии, поэтому его можно
запускать сколько угодно раз подряд — результат один и тот же.

Запуск из корня репозитория (после openlin_slides2 и openlin_formulas):
    python -m experiments.openlin_deck2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

SRC = Path("presentation/Линеаменты по космоснимкам.pptx")
FIG = Path("outputs/slides")
OUTPUTS = Path("outputs")

INK = RGBColor(0x1B, 0x2A, 0x32)
INK2 = RGBColor(0x6E, 0x7F, 0x88)
ACC = RGBColor(0xC1, 0x5B, 0x33)
CARD = RGBColor(0xF4, 0xEF, 0xEA)
EDGE = RGBColor(0xD8, 0xD2, 0xC8)

W, H = 13.333, 7.5
MARGIN = 0.55
TITLE_Y, SUB_Y = 0.32, 1.20

#: Размеры шрифтов. Нижняя граница — 16 pt: всё, что мельче, из зала не читают.
T_TITLE, T_SUB, T_CAP, T_BIG, T_LBL = 34, 17, 16, 54, 17


# --------------------------------------------------------------------------
# Помощники
# --------------------------------------------------------------------------

def text(slide, x, y, w, h, s, size=T_CAP, color=INK, bold=False,
         align=PP_ALIGN.LEFT, font="Calibri"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.name = font
    return box


def card(slide, x, y, w, h, fill=CARD):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = EDGE
    sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def head(slide, title, sub=None):
    # Кегль заголовка под его длину: в 34 pt Cambria в строку помещается около
    # 44 знаков, а перенос на вторую строку наезжает на подзаголовок.
    size = T_TITLE if len(title) <= 42 else (30 if len(title) <= 52 else 27)
    text(slide, MARGIN, TITLE_Y, W - 2 * MARGIN, 0.9, title, size=size,
         bold=True, font="Cambria")
    if sub:
        text(slide, MARGIN, SUB_Y, W - 2 * MARGIN, 0.5, sub, size=T_SUB,
             color=INK2)


def picture(slide, path, top, bottom):
    """Картинка по центру, вписанная в свободную полосу слайда."""
    iw, ih = Image.open(path).size
    box_w, box_h = W - 2 * MARGIN, bottom - top
    w = min(box_w, box_h * iw / ih)
    h = w * ih / iw
    return slide.shapes.add_picture(str(path), Inches((W - w) / 2),
                                    Inches(top + (box_h - h) / 2),
                                    Inches(w), Inches(h))


def caption(slide, s, y=6.72):
    text(slide, MARGIN, y, W - 2 * MARGIN, 0.5, s, size=T_CAP, color=INK2,
         align=PP_ALIGN.CENTER)


def numbers(slide, items, y=2.6, h=2.3):
    """Ряд крупных чисел с подписями."""
    n = len(items)
    gap = 0.3
    w = (W - 2 * MARGIN - gap * (n - 1)) / n
    for i, (value, label) in enumerate(items):
        x = MARGIN + i * (w + gap)
        card(slide, x, y, w, h)
        # Кегль под длину: «86 × 77 км» в 54 pt шире карточки и переносится.
        size = T_BIG if len(value) <= 4 else (40 if len(value) <= 7 else 30)
        text(slide, x, y + 0.42 + (54 - size) * 0.006, w, 1.0, value,
             size=size, bold=True, color=ACC, align=PP_ALIGN.CENTER,
             font="Cambria")
        text(slide, x + 0.2, y + 1.48, w - 0.4, 0.7, label, size=T_LBL,
             color=INK, align=PP_ALIGN.CENTER)


def pipeline(slide, y=2.5):
    """Пять шагов конвейера одной полосой."""
    steps = [("1", "признаки\nокрестности"), ("2", "сжатие\n160 → 8"),
             ("3", "оператор\nграницы"), ("4", "трассировка"),
             ("5", "отбор по\nподтверждению")]
    gap, arrow = 0.22, 0.42
    w = (W - 2 * MARGIN - (gap + arrow) * (len(steps) - 1)) / len(steps)
    for i, (num, name) in enumerate(steps):
        x = MARGIN + i * (w + gap + arrow)
        card(slide, x, y, w, 2.0)
        text(slide, x, y + 0.18, w, 0.6, num, size=30, bold=True, color=ACC,
             align=PP_ALIGN.CENTER, font="Cambria")
        text(slide, x + 0.12, y + 0.92, w - 0.24, 1.0, name, size=T_LBL,
             color=INK, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            text(slide, x + w + gap * 0.3, y + 0.75, arrow + 0.3, 0.5, "→",
                 size=26, color=INK2, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Сборка
# --------------------------------------------------------------------------

def blank(prs):
    """Пустой слайд: макет без плейсхолдеров, иначе они лезут поверх вёрстки."""
    master = prs.slide_masters[0]
    layout = None
    for lay in master.slide_layouts:
        if not lay.placeholders._element.findall(
                ".//{http://schemas.openxmlformats.org/presentationml/2006/main}sp"):
            layout = lay
            break
    slide = prs.slides.add_slide(layout or master.slide_layouts[-1])
    for sh in list(slide.shapes):
        sh._element.getparent().remove(sh._element)
    return slide


def drop_all(prs) -> None:
    ids = prs.slides._sldIdLst
    rel = ("{http://schemas.openxmlformats.org/officeDocument/2006/"
           "relationships}id")
    for sld in list(ids):
        prs.part.drop_rel(sld.get(rel))
        ids.remove(sld)


def fig_slide(prs, title, sub, name, cap=None):
    """Слайд «заголовок + крупная картинка + одна строка подписи»."""
    s = blank(prs)
    head(s, title, sub)
    bottom = 6.62 if cap else 7.05
    picture(s, (FIG if not str(name).startswith("outputs") else Path("."))
            / name, 1.82, bottom)
    if cap:
        caption(s, cap)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(SRC),
                    help="куда сохранить (по умолчанию сама презентация)")
    args = ap.parse_args()
    out = Path(args.out)

    backup = SRC.with_name(SRC.stem + " (до правки).pptx")
    if not backup.exists():
        raise SystemExit(f"нет резервной копии {backup.name}")
    prev = SRC.with_name(SRC.stem + " (версия 1).pptx")
    if SRC.exists() and not prev.exists():
        shutil.copy2(SRC, prev)
        print(f"прежняя версия сохранена: {prev.name}")

    prs = Presentation(str(backup))
    drop_all(prs)

    # --- 1. титул ----------------------------------------------------------
    s = blank(prs)
    text(s, MARGIN, 2.45, W - 2 * MARGIN, 1.2, "Линеаменты по космоснимкам",
         size=48, bold=True, font="Cambria")
    text(s, MARGIN, 3.62, W - 2 * MARGIN, 0.7,
         "Метод, формулы, результат по листу R-48-XI,XII (Анабарский щит)",
         size=21, color=INK2)
    text(s, MARGIN, 6.35, W - 2 * MARGIN, 0.8,
         "ФГБУ «ВНИГНИ», Отделение геоинформатики (ГИС Интегро)\n"
         "Открытые съёмки Sentinel-1 / Sentinel-2", size=15, color=INK2)

    # --- 2. дано -----------------------------------------------------------
    s = blank(prs)
    head(s, "Что дано", "Композит открытых съёмок Sentinel-1 и Sentinel-2 "
         "по листу")
    numbers(s, [("10", "каналов съёмки"),
                ("100 м", "шаг растра"),
                ("86 × 77 км", "площадка"),
                ("662 200", "ячеек")])
    caption(s, "Оптика: яркость и NDVI. Радар: VV, VH и шесть мер "
            "интерферометрической когерентности.")

    # --- 3-4. результат сразу, до разбора метода ---------------------------
    s = blank(prs)
    head(s, "Результат по Анабару",
         "Каждая линия подтверждена оптикой и радаром независимо")
    numbers(s, [("255", "подтверждённых линий"),
                ("1010 км", "суммарная длина"),
                ("2.1", "ожидаемо случайных"),
                ("×121", "чистота отбора")])
    caption(s, "Взаимное покрытие карт оптики и радара 0.422 против 0.250 "
            "у случайного уровня.")

    s = blank(prs)
    head(s, "Итоговая карта линеаментов Анабара",
         "Слева вся площадка, справа фрагмент в рамке крупным планом")
    picture(s, FIG / "p_final_map.png", 1.72, 6.85)

    # --- 5. конвейер -------------------------------------------------------
    s = blank(prs)
    head(s, "Метод: пять шагов",
         "Каждый шаг — детерминированная функция входа: повторный прогон даёт "
         "тот же результат")
    pipeline(s, y=2.7)
    caption(s, "Шаги 1–3 считают поле линейности, шаг 4 переводит его в "
            "отрезки, шаг 5 отбирает подтверждённые.", y=5.6)

    # --- шаг 1 -------------------------------------------------------------
    fig_slide(prs, "Шаг 1. Ячейка описывается окрестностью",
              "Один и тот же фрагмент: яркость и две величины, посчитанные "
              "по окрестности 500 м", "p_features.png")
    fig_slide(prs, "Шаг 1. Формулы: структурный тензор",
              "Усредняются произведения градиентов — иначе встречные "
              "градиенты по краям борозды гасят друг друга", "f_tensor.png")
    fig_slide(prs, "Шаг 1. Что означает эллипс направлений",
              "Три случая и настоящие участки листа под каждый из них",
              "p_ellipse.png")
    fig_slide(prs, "Шаг 1. Формулы: пять величин",
              "Считаются на трёх масштабах окрестности: 500 м, 1 км, 2 км",
              "f_five.png")

    # --- итог шага 1 -------------------------------------------------------
    s = blank(prs)
    head(s, "Что есть после шага 1",
         "Вместо одного числа в ячейке — описание её окрестности")
    numbers(s, [("10", "каналов съёмки"),
                ("3", "масштаба окрестности"),
                ("5", "величин на канал и масштаб"),
                ("160", "чисел в каждой ячейке")])
    caption(s, "106 миллионов чисел на площадку, и всё это посчитано прямо "
            "из снимка: никакой разметки на этом шаге нет.")

    # --- шаг 2 -------------------------------------------------------------
    fig_slide(prs, "Шаг 2. Восьми компонент хватает",
              "Доля разброса, объяснённая каждой компонентой", "p_scree.png")
    fig_slide(prs, "Шаг 2. Формулы: главные компоненты и отбеливание",
              "После отбеливания расстояние между участками измеряется "
              "в сигмах", "f_pca.png")
    fig_slide(prs, "Шаг 2. Компоненты как карты",
              "Первые две из восьми на том же фрагменте", "p_components.png")
    fig_slide(prs, "Шаг 2. Формулы: свёрточный автоэнкодер",
              "Второй способ сжатия: обучается на самом снимке, разметка "
              "не нужна", "f_ae.png")
    fig_slide(prs, "Шаг 2. Оба способа идут в карту",
              "Списки линий пересекаются лишь наполовину, поэтому берётся "
              "объединение", "p_two_ways.png")

    # --- шаг 3 -------------------------------------------------------------
    fig_slide(prs, "Шаг 3. Два окна по разные стороны линии",
              "Сравниваются средние портреты участков, а не яркость",
              "p_windows.png")
    fig_slide(prs, "Шаг 3. Формулы: геометрия окон",
              "Вдоль линии окно длинное, поперёк — узкое", "f_window.png")
    fig_slide(prs, "Шаг 3. Перебор двенадцати направлений",
              "Отклик одной ячейки: максимум даёт силу границы и азимут",
              "p_angles.png")
    fig_slide(prs, "Шаг 3. Формулы: сила границы и азимут",
              "Расстояние между восьмимерными портретами участков",
              "f_edge.png")
    fig_slide(prs, "Шаг 3. Готовое поле силы границы",
              "Тот же фрагмент Анабара", "p_field.png")

    # --- шаг 4 -------------------------------------------------------------
    fig_slide(prs, "Шаг 4. От поля к линиям: этапы 1–3",
              "Сглаживание, гребень, два порога", "p_trace_a.png")
    fig_slide(prs, "Шаг 4. Формулы: гребень и гистерезис",
              "Немаксимумы подавляются поперёк линии, пороги задаются "
              "квантилями площади", "f_ridge.png")
    fig_slide(prs, "Шаг 4. От поля к линиям: этапы 4–5",
              "Скелет, сшивка разрывов и отсев", "p_trace_b.png")
    fig_slide(prs, "Шаг 4. Формулы: геометрия ветки",
              "Длина, прямолинейность и правило сшивки", "f_geom.png")

    # --- шаг 5 -------------------------------------------------------------
    fig_slide(prs, "Шаг 5. Что видит каждый прибор",
              "Три независимых прогона: только оптика, только радар, "
              "пересечение", "p_confirm.png")
    fig_slide(prs, "Шаг 5. Формулы: правило подтверждения",
              "Рядом, в том же направлении и не меньше чем на половине длины",
              "f_confirm.png")
    fig_slide(prs, "Шаг 5. Цена отбора",
              "Сколько линий остаётся на каждом условии", "p_waterfall.png")
    fig_slide(prs, "Шаг 5. Формулы: случайный уровень",
              "Тороидальный сдвиг сохраняет геометрию карты и рвёт привязку "
              "к местности", "f_null.png")

    # --- плотность ---------------------------------------------------------
    fig_slide(prs, "Плотность задаёт пользователь",
              "Пороги трассировки ослабляются, пока подтверждение держится "
              "выше случайного уровня", "p_density.png")
    fig_slide(prs, "Плотность: формула стоп-правила",
              "Взаимное покрытие карт двух приборов и его запас над "
              "случайным", "f_density.png")

    # --- перенос -----------------------------------------------------------
    fig_slide(prs, "Ранжирование линий переносится между площадками",
              "Модель обучена на одной площадке и применена к другой, "
              "которую не видела", "p_transfer.png")
    fig_slide(prs, "Перенос: метрика и значимость",
              "Девять признаков линии: длина, прямолинейность, сила над "
              "фоном, однородность, согласие с полем", "f_auc.png")

    # --- итоги -------------------------------------------------------------
    s = blank(prs)
    head(s, "Итоги", "Что метод даёт прямо сейчас")
    items = [("Карта", "255 линий, 1010 км, каждая подтверждена двумя "
              "независимыми съёмками"),
             ("Воспроизводимость", "обучения в построении карты нет: тот же "
              "вход даёт тот же результат"),
             ("Переносимость", "ранжирование линий работает на площадке, "
              "где есть только одна съёмка")]
    y = 2.05
    for name, body in items:
        card(s, MARGIN, y, W - 2 * MARGIN, 1.32)
        text(s, MARGIN + 0.35, y + 0.16, 3.4, 0.5, name, size=20, bold=True,
             color=ACC, font="Cambria")
        text(s, MARGIN + 0.35, y + 0.68, W - 2 * MARGIN - 0.7, 0.6, body,
             size=T_LBL, color=INK)
        y += 1.55
    caption(s, "Следующий шаг — разметка фрагмента 20 × 20 км геофизиком: "
            "она калибрует порог отбора под вашу манеру выделения.", y=6.9)

    prs.save(str(out))
    print(f"сохранено: {out}, слайдов {len(prs.slides)}")


if __name__ == "__main__":
    main()
