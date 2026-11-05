"""Презентация к докладу 05.11.2026 (ветка линеаментов): дополняет рассказ, не пересказывает.

Цифры — из логов прогонов 05.11.2026 (UTS1, RELU1–4, openlin_ref_match,
openlin_missed_ref, openlin_missed_why, XFER, проверки на артефакты).
    python -m experiments.openlin_deck_0511
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

IMG = Path("outputs/deck_img")        # картинки к слайдам (outputs/ не в git)
OUT = Path("presentation/Доклад_линеаменты_05_11.pptx")

DARK = RGBColor(0x1E, 0x2B, 0x2F)
LIGHT = RGBColor(0xFF, 0xFF, 0xFF)
ALT = RGBColor(0xEE, 0xF1, 0xF1)
MUTED = RGBColor(0x5A, 0x64, 0x68)
SOFT = RGBColor(0xBF, 0xCA, 0xCC)
OURS = RGBColor(0xC8, 0x64, 0x1E)      # наши линии — оранжевый (на картах — красный)
REF = RGBColor(0x2E, 0x6A, 0xA8)       # эталон — синий (на картах — синий)
NULLC = RGBColor(0xA9, 0xB3, 0xB6)     # случайный уровень — серый
HEAD, BODY = "Cambria", "Calibri"

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=18, color=DARK, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    """runs: строка или список абзацев; абзац — строка или список (текст, {опции})."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for s in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, s, 0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.name = o.get("font", font)
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            r.font.color.rgb = o.get("color", color)
        if i < len(paras) - 1:
            p.space_after = Pt(o.get("after", 8) if parts else 8)
    return tb


def title(slide, t, color=DARK):
    text(slide, M, Inches(0.45), W - 2 * M, Inches(0.9), t, size=36, color=color,
         font=HEAD, bold=True, name="Заголовок")


def card(slide, x, y, w, h, fill=ALT):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def stat(slide, x, y, w, num, label, color=OURS, numsize=60, labsize=16, labcolor=MUTED):
    hn = Pt(numsize * 1.3)
    text(slide, x, y, w, hn, num, size=numsize, color=color, font=HEAD, bold=True)
    text(slide, x, y + hn + Pt(6), w, Inches(0.9), label, size=labsize, color=labcolor)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def picture(slide, path, x, y, w=None, h=None, name=None):
    p = slide.shapes.add_picture(str(path), x, y, w, h)
    if name:
        p.name = name
    return p


def bar_chart(slide, x, y, w, h, cats, series, colors, stacked=False, maxv=None,
              legend=True, fmt='0"%"'):
    cd = CategoryChartData()
    cd.categories = cats
    for n, v in series:
        cd.add_series(n, v)
    kind = XL_CHART_TYPE.BAR_STACKED if stacked else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, x, y, w, h, cd)
    ch = gf.chart
    ch.font.name = BODY
    ch.font.size = Pt(14)
    ch.font.color.rgb = MUTED
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(14)
    va = ch.value_axis
    va.has_major_gridlines = not stacked
    if va.has_major_gridlines:
        va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE2, 0xE3)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(12)
    if maxv:
        va.maximum_scale = maxv
    va.minimum_scale = 0
    if stacked:
        va.visible = False
    ca = ch.category_axis
    ca.reverse_order = stacked
    ca.format.line.color.rgb = RGBColor(0xC8, 0xCF, 0xD1)
    ca.tick_labels.font.size = Pt(16)
    ca.tick_labels.font.color.rgb = DARK
    plot = ch.plots[0]
    plot.gap_width = 60 if stacked else 80
    if stacked:
        plot.overlap = 100
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
        s.format.line.fill.background()
        s.data_labels.show_value = True
        s.data_labels.number_format = fmt
        s.data_labels.number_format_is_linked = False
        s.data_labels.font.size = Pt(14)
        s.data_labels.font.bold = True
        if stacked:
            s.data_labels.position = XL_LABEL_POSITION.CENTER
            s.data_labels.font.color.rgb = LIGHT
        else:
            s.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
            s.data_labels.font.color.rgb = DARK
    return ch


def main() -> None:
    pres = Presentation()
    pres.slide_width, pres.slide_height = W, H
    blank = pres.slide_layouts[6]
    new = lambda: pres.slides.add_slide(blank)
    ACC = RGBColor(0xE3, 0x9A, 0x5C)

    # 1. Обложка
    s = new(); bg(s, DARK)
    text(s, M, Inches(2.0), Inches(11), Inches(0.5), "ЛИНЕАМЕНТЫ · 05.11.2026",
         size=18, color=ACC, bold=True)
    text(s, M, Inches(2.6), Inches(11.5), Inches(2.2), ["Площадка с эталоном", "и рельеф"],
         size=54, color=LIGHT, font=HEAD, bold=True)
    notes(s, "Три вопроса с прошлой встречи: почему расходятся рельеф и снимок, можно ли "
             "выделять впадины, на чём проверять метод. Начинаю с третьего.")

    # 2. Площадка
    s = new(); bg(s, LIGHT)
    title(s, "Утсйоки: тундра с готовым эталоном")
    pw = Inches(7.6)
    ph = pw * 882 / 1980
    picture(s, IMG / "utsjoki_gtk.png", M, Inches(1.5), pw, ph, name="Карта Утсйоки")
    text(s, M, Inches(1.5) + ph + Inches(0.12), pw, Inches(0.8),
         [[("Слева — зимний снимок. Справа: ", {}), ("синее", {"color": REF, "bold": True}),
           (" — эталон, ", {}), ("красное", {"color": OURS, "bold": True}),
           (" — наши линии", {})]],
         size=15, color=MUTED)
    # Паспорт: лист TILE_KM openlin_probe; высоты — GLO-30 на листе, 5–95-й процентили;
    # сцены и солнце — кэш s2_lowsun/utsjoki и лог UTS1; эталон — GTK 292 внутри листа.
    facts = [("Площадь", "86 × 77 км, ~6 600 км² — как лист Анабара"),
             ("Где", "север Финляндии, 69.6° с. ш., 27.1° в. д., щит, докембрий"),
             ("Высоты", "155–420 м, перепад ~260 м (Анабар 280–600 м, ~320 м)"),
             ("Снимки", "Sentinel-2, 12 зимних сцен 2019–2024, солнце 6.5–9.6°, шаг 20 м"),
             ("Рельеф", "Copernicus GLO-30, шаг 30 м; проверка — ALOS 30 м"),
             ("Эталон", "Геологическая служба Финляндии, 1:500 000, вручную по LiDAR 2 м; "
                        "73 линии, 1 119 км")]
    x = M + pw + Inches(0.45)
    fw = W - M - x
    y = Inches(1.5)
    for k, v in facts:
        text(s, x, y, fw, Inches(0.3), k, size=14, color=REF, bold=True)
        text(s, x, y + Inches(0.3), fw, Inches(0.6), v, size=15, color=DARK)
        y += Inches(0.92)
    notes(s, "Эталон нарисован по рельефу — это ДЗЗ, а не карта разломов. "
             "На Утсйоки метод настраиваем, Анабар — цель.")

    # 3. Результат
    s = new(); bg(s, LIGHT)
    title(s, "40 % эталона — втрое больше случайного")
    bar_chart(s, M, Inches(1.5), Inches(8.0), Inches(5.4),
              ["Снимок", "Рельеф", "Снимок + рельеф"],
              [("Найдено эталона", (15, 38, 40)), ("Случайные линии той же длины", (4, 11, 12))],
              [OURS, NULLC], maxv=50)
    x = M + Inches(8.5)
    stat(s, x, Inches(1.9), Inches(3.6), "20 из 57", "длинных линий эталона найдено",
         color=REF, numsize=48, labsize=18, labcolor=DARK)
    text(s, x, Inches(4.3), Inches(3.6), Inches(2.0),
         "Случайный уровень вычитаем: без этого рельеф «находил» 52 %, а случайные линии — 32 %",
         size=16, color=MUTED)
    notes(s, "Поправка: на Анабаре по рельефу 1568 линий, не 800; новых около 60 %, не 70.")

    # 4. Почему 40 %
    s = new(); bg(s, LIGHT)
    title(s, "Почему только 40 %")
    items = [("LiDAR 2 м", "Эталон рисовали по рельефу с шагом 2 м, у нас — 30 м. "
                           "Неглубокие ложбины, по которым проведена остальная часть, "
                           "на нашем рельефе не видны"),
             ("Десятки км", "Геолог соединял короткие слабые признаки в одну длинную линию. "
                            "Наш метод видит только отрезки, где признак есть"),
             ("1:500 000", "Масштаб карты: часть линий эталона проведена в 300–600 м "
                           "от формы на местности")]
    y = Inches(1.6)
    for k, v in items:
        card(s, M, y, W - 2 * M, Inches(1.45))
        text(s, M + Inches(0.4), y, Inches(3.0), Inches(1.45), k, size=30, color=REF,
             font=HEAD, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, M + Inches(3.6), y, Inches(8.1), Inches(1.45), v, size=19, color=DARK,
             anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(1.65)
    text(s, M, Inches(6.6), W - 2 * M, Inches(0.5),
         "То, что видно в наших данных, метод находит. Дальше упираемся в детальность данных",
         size=20, color=OURS, bold=True)
    notes(s, "Проверка: пропущенную часть эталона разобрали по причинам и сравнили со случайными "
             "местами — сверх случайного только найденное и небольшой сдвиг 300–600 м. "
             "Отсев и сборка линий эталон не выбрасывают.")

    # 5. Пропущенное
    s = new(); bg(s, LIGHT)
    title(s, "Под пропущенной линией почти ничего нет")
    ih = Inches(5.5)
    iw = ih * 1350 / 900
    picture(s, IMG / "missed.png", M, Inches(1.5), iw, ih, name="Пропущенный эталон")
    x = M + iw + Inches(0.5)
    text(s, x, Inches(1.6), Inches(3.4), Inches(0.4), "Глубина ложбины", size=18, color=MUTED,
         bold=True)
    stat(s, x, Inches(2.1), Inches(3.4), "13 м", "под найденными линиями", color=OURS,
         numsize=56, labsize=17, labcolor=DARK)
    stat(s, x, Inches(3.8), Inches(3.4), "2 м", "под пропущенными", color=REF,
         numsize=56, labsize=17, labcolor=DARK)
    text(s, x, Inches(5.5), Inches(3.4), Inches(1.4),
         "Эталон рисовали по рельефу 2 м, у нас 30 м", size=18, color=DARK, bold=True)
    notes(s, "Пропущенное разобрали по причинам: сверх случайного только найденное. "
             "Метод не теряет то, что видно в данных; потолок задаёт детальность.")

    # 6. Впадины
    s = new(); bg(s, LIGHT)
    title(s, "Впадины находим, но подтвердить пока нечем")
    stat(s, M, Inches(1.9), Inches(3.8), "79 %",
         "мест под найденным эталоном — ложбины", color=REF, numsize=80, labsize=20,
         labcolor=DARK)
    tries = [("Вторая модель рельефа", "повторяет первую — пропускает всё"),
             ("Зимний снимок", "подтверждает только то, что видит сам"),
             ("Радар", "даёт линию сбоку от ложбины, не по оси")]
    x0 = M + Inches(4.5)
    y = Inches(1.9)
    for k, v in tries:
        card(s, x0, y, Inches(7.6), Inches(1.3))
        text(s, x0 + Inches(0.35), y + Inches(0.18), Inches(7.0), Inches(0.45), k,
             size=20, color=DARK, bold=True)
        text(s, x0 + Inches(0.35), y + Inches(0.68), Inches(7.0), Inches(0.5), v,
             size=17, color=MUTED)
        y += Inches(1.5)
    notes(s, "Идея с прошлой встречи верна: линеаменты эталона в основном ложбины. "
             "Нужен независимый источник, который видит именно оси понижений.")

    # 7. Выводы и дальше
    s = new(); bg(s, DARK)
    title(s, "Выводы и дальше", color=LIGHT)
    concl = ["Есть площадка с эталоном и честная мера",
             "Снимок + рельеф: 40 % эталона, втрое выше случайного",
             "Линеаменты — в основном ложбины; подтверждать их пока нечем",
             "Потолок — детальность данных, не метод"]
    y = Inches(1.6)
    for t in concl:
        text(s, M, y, Inches(6.2), Inches(1.0), t, size=22, color=LIGHT)
        y += Inches(1.15)
    x = M + Inches(6.8)
    text(s, x, Inches(1.6), Inches(5.3), Inches(0.5), "ДАЛЬШЕ", size=18, color=ACC, bold=True)
    steps = ["Рельеф 2 м (ArcticDEM)", "Длинные линии из слабых признаков",
             "Чем подтверждать ложбины", "Слой рельефа — на Анабар"]
    y = Inches(2.2)
    for i, t in enumerate(steps):
        text(s, x, y, Inches(0.5), Inches(0.6), str(i + 1), size=26, color=ACC, font=HEAD,
             bold=True)
        text(s, x + Inches(0.6), y + Inches(0.05), Inches(4.7), Inches(0.8), t, size=20,
             color=LIGHT)
        y += Inches(1.0)
    notes(s, "Покрытие Анабара ArcticDEM ещё надо проверить.")

    OUT.parent.mkdir(exist_ok=True)
    pres.save(OUT)
    print(f"сохранено: {OUT} ({len(pres.slides)} слайдов)")


if __name__ == "__main__":
    main()
