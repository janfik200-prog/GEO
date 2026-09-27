# -*- coding: utf-8 -*-
"""Сборка презентации «Технологическая схема: золото-уран» — пересборка
скачанного шаблона-инфографики (`gold_uranium_template.pptx`) в стиле проекта
(тёмная шапка + золотой акцент, Calibri, палитра из
`build_feature_space_presentation.py`), с текстом по содержанию пояснительного
документа организации (`gold_uranium_notes.docx`, 28.08.2026).

Текст — сплошные абзацы (не тезисы). Цитаты из обсуждения убраны: вместо них
на каждом узле, где задача решена или в работе, — картинка из уже посчитанных
outputs/ с подписью; для узлов без результата — декоративная плашка с номером
узла и коротким статусом, без текста "из обсуждения".

Цветовая кодировка узлов схемы (серый/фиолетовый/зелёный/красный/охра)
сохранена из исходного шаблона — она несёт смысл (последовательный шаг /
точка ветвления-слияния / ветка данных / ветка литературы / финал).

Запуск из корня: ``python -X utf8 -m experiments.build_tech_scheme_presentation``.
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs II presentation"
OUT = OUT_DIR / "Презентация-Технологическая-схема-золото-уран-28.08.2026.pptx"
IMG = ROOT / "outputs"

GOLD = RGBColor(0xC8, 0x9B, 0x2C)
DARK = RGBColor(0x2B, 0x2B, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF7, 0xF2, 0xE3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB0, 0x3A, 0x2E)
GREEN = RGBColor(0x3E, 0x7A, 0x3E)

# Категории узлов схемы v2 (28.08.2026, по новой технологической карте —
# 11 блоков, старт/эталон-контроль/БД/БЗ/эксперимент/валидация-итог)
CAT = {
    "seq": RGBColor(0x4A, 0x4A, 0x48),    # старт
    "merge": RGBColor(0x1F, 0x52, 0x94),  # эталон / контроль / точка слияния
    "data": RGBColor(0x0F, 0x6E, 0x56),   # ветка данных (БД)
    "lit": RGBColor(0x6A, 0x3D, 0x9E),    # ветка знаний (БЗ)
    "final": RGBColor(0x1E, 0x7A, 0x74),  # валидация и итог
    "gold": GOLD,                          # экспериментальная ветка
}
TINT = {
    "seq": RGBColor(0xEA, 0xEA, 0xE9),
    "merge": RGBColor(0xE4, 0xEB, 0xF6),
    "data": RGBColor(0xE1, 0xF5, 0xEE),
    "lit": RGBColor(0xEF, 0xE6, 0xF7),
    "final": RGBColor(0xE1, 0xF1, 0xEF),
    "gold": RGBColor(0xFB, 0xF3, 0xE1),
}
CAT_LABEL = {
    "seq": "СТАРТ",
    "merge": "ЭТАЛОН / КОНТРОЛЬ",
    "data": "ВЕТКА ДАННЫХ (БД)",
    "lit": "ВЕТКА ЗНАНИЙ (БЗ)",
    "final": "ВАЛИДАЦИЯ И ИТОГ",
    "gold": "ЭКСПЕРИМЕНТАЛЬНАЯ ВЕТКА",
}

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(slide, x, y, w, h, color):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def round_rect(slide, x, y, w, h, color, radius=0.09):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sp.adjustments[0] = radius
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def oval(slide, x, y, d, color):
    sp = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def connector(slide, x1, y1, x2, y2, color, weight=1.75, arrow=True):
    cxn = slide.shapes.add_connector(MSO_CONNECTOR.ELBOW, x1, y1, x2, y2)
    cxn.line.color.rgb = color
    cxn.line.width = Pt(weight)
    if arrow:
        ln = cxn.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return cxn


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def set_run(r, text, size, color=DARK, bold=False, italic=False):
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = "Calibri"


def header(slide, title, num, node=None, cat=None):
    """Тёмная шапка (house style) + опционально цветной кружок-номер узла
    и цветная метка категории справа (точка слияния/ветка данных/...)."""
    rect(slide, 0, 0, SW, Inches(1.0), DARK)
    rect(slide, 0, Inches(1.0), SW, Pt(4), GOLD)
    title_x = Inches(0.55)
    if node is not None:
        d = Inches(0.62)
        oval(slide, Inches(0.55), Inches(0.19), d, CAT[cat])
        tfn = textbox(slide, Inches(0.55), Inches(0.19), d, d, MSO_ANCHOR.MIDDLE)
        pn = tfn.paragraphs[0]
        pn.alignment = PP_ALIGN.CENTER
        set_run(pn.add_run(), str(node), 15, WHITE, bold=True)
        title_x = Inches(1.35)
    tf = textbox(slide, title_x, 0, Inches(9.6), Inches(1.0), MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    set_run(p.add_run(), title, 24, WHITE, bold=True)
    if cat is not None:
        tfc = textbox(slide, Inches(9.7), 0, Inches(2.55), Inches(1.0), MSO_ANCHOR.MIDDLE)
        pc = tfc.paragraphs[0]
        pc.alignment = PP_ALIGN.RIGHT
        set_run(pc.add_run(), CAT_LABEL[cat], 11, GOLD, bold=True)
    tf2 = textbox(slide, Inches(12.2), 0, Inches(0.9), Inches(1.0), MSO_ANCHOR.MIDDLE)
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.RIGHT
    set_run(p2.add_run(), str(num), 18, GOLD, bold=True)


def prose(slide, x, y, w, h, paragraphs, size=15):
    """Сплошные абзацы текста — без маркеров списка."""
    tf = textbox(slide, x, y, w, h)
    first = True
    for text in paragraphs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(14)
        p.line_spacing = 1.15
        set_run(p.add_run(), text, size, DARK)
    return tf


def labeled_block(slide, x, y, w, h, label, paragraphs, cat="seq", size=14.5):
    """Подписанный смысловой блок (ЦЕЛЬ / ЗАДАЧИ / КРИТЕРИИ...) — заголовок-метка
    цветом категории + сплошной текст абзацем(ами) под ней, без маркеров списка."""
    tfl = textbox(slide, x, y, w, Inches(0.32))
    pl = tfl.paragraphs[0]
    set_run(pl.add_run(), label, 13, CAT[cat], bold=True)
    prose(slide, x, y + Inches(0.4), w, h - Inches(0.4), paragraphs, size=size)


def fit_image(slide, path, x, y, max_w, max_h, caption=None):
    iw, ih = Image.open(path).size
    ratio = min(max_w / iw, max_h / ih)
    w = Emu(int(iw * ratio))
    h = Emu(int(ih * ratio))
    px = x + Emu(int((max_w - int(iw * ratio)) / 2))
    pic = slide.shapes.add_picture(str(path), px, y, width=w, height=h)
    if caption:
        cap_y = y + h + Pt(4)
        tf = textbox(slide, x, cap_y, max_w, Inches(0.5))
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), caption, 10.5, GREY, italic=True)
    return pic


def image_panel(slide, x, y, w, h, cat, image_name, caption):
    round_rect(slide, x, y, w, h, LIGHT, radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT[cat])
    fit_image(slide, IMG / image_name, x + Inches(0.22), y + Inches(0.18), w - Inches(0.44), h - Inches(0.85), caption=caption)


def status_panel(slide, x, y, w, h, cat, big_label, caption):
    """Плашка без картинки: номер узла крупно + короткий статус, без цитат."""
    round_rect(slide, x, y, w, h, TINT[cat], radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT[cat])
    tf = textbox(slide, x, y + h * 0.28, w, h * 0.34, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), str(big_label), 64, CAT[cat], bold=True)
    tf2 = textbox(slide, x + Inches(0.4), y + h * 0.64, w - Inches(0.8), h * 0.3)
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    set_run(p2.add_run(), caption, 12.5, GREY, italic=True)


def step_slide(num, node, cat, title, paragraphs, image_name=None, image_caption=None, status_caption=None):
    s = add_slide()
    header(s, title, num, node=node, cat=cat)
    has_side = image_name or status_caption
    text_w = Inches(6.85) if has_side else Inches(11.9)
    prose(s, Inches(0.55), Inches(1.35), text_w, Inches(5.75), paragraphs, size=15.5)
    side_x, side_w, side_y, side_h = Inches(7.65), Inches(5.1), Inches(1.35), Inches(5.75)
    if image_name:
        image_panel(s, side_x, side_y, side_w, side_h, cat, image_name, image_caption)
    elif status_caption:
        status_panel(s, side_x, side_y, side_w, side_h, cat, node, status_caption)
    return s


def branch_column(slide, x, y, w, cat, label, paragraphs):
    round_rect(slide, x, y, w, Inches(0.5), CAT[cat], radius=0.12)
    tfh = textbox(slide, x, y, w, Inches(0.5), MSO_ANCHOR.MIDDLE)
    ph = tfh.paragraphs[0]
    ph.alignment = PP_ALIGN.CENTER
    set_run(ph.add_run(), label, 15, WHITE, bold=True)
    prose(slide, x + Inches(0.1), y + Inches(0.65), w - Inches(0.2), Inches(2.15), paragraphs, size=13)


def flow_box(slide, cx, cy, w, h, cat, lines, size=10.5):
    """Прямоугольный узел векторной блок-схемы (не картинка) с центрированным текстом."""
    x0, y0 = cx - w // 2, cy - h // 2
    round_rect(slide, x0, y0, w, h, CAT[cat], radius=0.14)
    tf = textbox(slide, x0 + Inches(0.05), y0, w - Inches(0.1), h, MSO_ANCHOR.MIDDLE)
    first = True
    for line in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), line, size, WHITE, bold=True)
    return cx, cy


def verification_diagram(slide, x, y, w, h):
    """Узел 10: векторная схема верификации вместо картинки — устойчивость,
    геологическая логика и отложенные данные, процесс ещё предстоит пройти."""
    round_rect(slide, x, y, w, h, LIGHT, radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT["final"])
    cx = x + w // 2
    bh = Inches(0.62)
    top_y = y + Inches(0.55)
    flow_box(slide, cx, top_y, Inches(3.9), bh, "gold", ["Кандидат из узла 9"], size=11)
    gap = Inches(0.55)
    y2 = top_y + bh + gap
    flow_box(slide, cx, y2, Inches(3.9), bh, "merge", ["Устойчивость", "(сиды, перенос на смежный лист)"], size=10.5)
    y3 = y2 + bh + gap
    flow_box(slide, cx, y3, Inches(3.9), bh, "lit", ["Сверка с геологической логикой"], size=10.5)
    y4 = y3 + bh + gap
    flow_box(slide, cx, y4, Inches(3.9), bh, "final", ["Проверка на отложенных данных"], size=10.5)
    connector(slide, cx, top_y + bh // 2, cx, y2 - bh // 2, CAT["gold"], weight=1.5)
    connector(slide, cx, y2 + bh // 2, cx, y3 - bh // 2, CAT["merge"], weight=1.5)
    connector(slide, cx, y3 + bh // 2, cx, y4 - bh // 2, CAT["lit"], weight=1.5)
    cap_y = y4 + bh // 2 + Inches(0.3)
    tf = textbox(slide, x + Inches(0.25), cap_y, w - Inches(0.5), Inches(0.85))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "Кандидат из узла 9 проверяется на устойчивость, сверяется с "
                          "геологической логикой и проверяется на данных вне конвейера обучения", 10.5, GREY, italic=True)


def zonation_diagram(slide, x, y, w, h):
    """Узел 10: макет итоговой карты районирования — иллюстрация приёма
    (ячейка сетки → зона перспективности), а не расчётный прогноз."""
    round_rect(slide, x, y, w, h, LIGHT, radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT["final"])
    pattern = [
        "0000001",
        "0001112",
        "0011222",
        "0112221",
        "0011110",
        "0000000",
    ]
    rows, cols = len(pattern), len(pattern[0])
    grid_x = x + Inches(0.35)
    grid_y = y + Inches(0.3)
    grid_w = w - Inches(0.7)
    grid_h = h - Inches(2.15)
    cell_w = grid_w // cols
    cell_h = grid_h // rows
    zone_color = {"0": RGBColor(0xE3, 0xDF, 0xD2), "1": TINT["final"], "2": RGBColor(0x8F, 0xC6, 0x9E)}
    star_cells = []
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            cx0 = grid_x + c * cell_w
            cy0 = grid_y + r * cell_h
            rect(slide, cx0, cy0, cell_w - Emu(9525), cell_h - Emu(9525), zone_color[ch])
            if ch == "2" and (r, c) in ((2, 5), (3, 4)):
                star_cells.append((cx0 + cell_w // 2, cy0 + cell_h // 2))
    for mx, my in star_cells:
        oval(slide, mx - Inches(0.09), my - Inches(0.09), Inches(0.18), WHITE)
        oval(slide, mx - Inches(0.05), my - Inches(0.05), Inches(0.10), GOLD)

    leg_y = grid_y + grid_h + Inches(0.2)
    legend = [("2", "высокая перспективность"), ("1", "средняя"), ("0", "низкая / неопределённая")]
    lx = x + Inches(0.35)
    for key, label in legend:
        rect(slide, lx, leg_y, Inches(0.22), Inches(0.16), zone_color[key])
        tfl = textbox(slide, lx + Inches(0.3), leg_y - Inches(0.07), Inches(2.6), Inches(0.3))
        pl = tfl.paragraphs[0]
        set_run(pl.add_run(), label, 10, GREY)
        leg_y += Inches(0.26)
    oval(slide, lx + Inches(0.04), leg_y - Inches(0.05), Inches(0.14), GOLD)
    tfl = textbox(slide, lx + Inches(0.3), leg_y - Inches(0.11), Inches(2.9), Inches(0.3))
    pl = tfl.paragraphs[0]
    set_run(pl.add_run(), "кандидат золото-уранового узла", 10, GREY)

    cap_y = y + h - Inches(0.55)
    tf = textbox(slide, x + Inches(0.25), cap_y, w - Inches(0.5), Inches(0.5))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "Макет итогового приёма (зоны по ячейкам сетки листа + кандидаты узлов) — "
                          "иллюстрация, не расчётный результат", 10, GREY, italic=True)


def sufficiency_loop_diagram(slide, x, y, w, h):
    """Узел 4: контур возврата при недостатке данных — векторная мини-схема
    вместо картинки (этап без расчётного результата)."""
    round_rect(slide, x, y, w, h, TINT["data"], radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT["data"])
    cx = x + w // 2
    bw = Inches(3.6)
    bh = Inches(0.62)
    y1 = y + Inches(0.55)
    flow_box(slide, cx, y1, bw, bh, "data", ["Подготовка и", "паспортизация слоёв"], size=11)
    gap = Inches(0.55)
    y2 = y1 + bh + gap
    flow_box(slide, cx, y2, bw, bh, "merge", ["Проверка", "достаточности данных"], size=11)
    connector(slide, cx, y1 + bh // 2, cx, y2 - bh // 2, CAT["data"], weight=1.5)
    y3 = y2 + bh + gap
    flow_box(slide, cx, y3, bw, bh, "final", ["Да — к узлу 7", "(соотнесение с БЗ)"], size=10.5)
    connector(slide, cx, y2 + bh // 2, cx, y3 - bh // 2, CAT["final"], weight=1.5)
    loop_x = cx + bw // 2 + Inches(0.55)
    connector(slide, loop_x, y2, loop_x, y1, RGBColor(0x6F, 0xA8, 0x82), weight=1.4, arrow=False)
    connector(slide, cx + bw // 2, y2, loop_x, y2, RGBColor(0x6F, 0xA8, 0x82), weight=1.4, arrow=False)
    connector(slide, loop_x, y1, cx + bw // 2, y1, RGBColor(0x6F, 0xA8, 0x82), weight=1.4)
    tfl = textbox(slide, loop_x - Inches(0.05), (y1 + y2) // 2 - Inches(0.5), Inches(1.0), Inches(1.0))
    pl = tfl.paragraphs[0]
    pl.alignment = PP_ALIGN.CENTER
    set_run(pl.add_run(), "нет — доп. сбор (узел 3)", 9, RGBColor(0x4A, 0x7C, 0x5E), italic=True)
    cap_y = y3 + bh // 2 + Inches(0.3)
    tf = textbox(slide, x + Inches(0.25), cap_y, w - Inches(0.5), Inches(0.6))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "Контур при недостатке данных — паспортизация возвращает процесс к сбору, "
                          "а не идёт дальше с пробелами по территории или признаку", 10.5, GREY, italic=True)


def concept_graph_diagram(slide, x, y, w, h, cat, center_label, satellites, caption):
    """Общий вид — граф связей узел-в-центре + спутники (узел 6: граф знаний)."""
    round_rect(slide, x, y, w, h, TINT[cat], radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT[cat])
    cx_c = x + w // 2
    cy_c = y + int(h * 0.30)
    d_c = Inches(1.15)
    oval(slide, cx_c - d_c // 2, cy_c - d_c // 2, d_c, CAT[cat])
    tf = textbox(slide, cx_c - d_c // 2, cy_c - d_c // 2, d_c, d_c, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), center_label, 10.5, WHITE, bold=True)

    n = len(satellites)
    d_s = Inches(0.92)
    start_x = x + int(w * 0.14)
    span = int(w * 0.72)
    for i, label in enumerate(satellites):
        sx = start_x + (span * i // max(n - 1, 1) if n > 1 else span // 2)
        sy = y + int(h * 0.68)
        oval(slide, sx - d_s // 2, sy - d_s // 2, d_s, RGBColor(0xD8, 0xC6, 0xE8))
        tfs = textbox(slide, sx - d_s // 2 - Inches(0.1), sy - d_s // 2, d_s + Inches(0.2), d_s, MSO_ANCHOR.MIDDLE)
        ps = tfs.paragraphs[0]
        ps.alignment = PP_ALIGN.CENTER
        set_run(ps.add_run(), label, 8, DARK, bold=True)
        connector(slide, sx, sy - d_s // 2, cx_c, cy_c + d_c // 2, CAT[cat], weight=1.25)

    cap_y = y + h - Inches(0.42)
    tf2 = textbox(slide, x + Inches(0.15), cap_y, w - Inches(0.3), Inches(0.4))
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    set_run(p2.add_run(), caption, 9.5, GREY, italic=True)


def knowledge_base_diagram(slide, x, y, w, h):
    """Узел 4B: агентная база знаний уже начата — векторная схема источников
    вместо картинки-скриншота (метрик по этой ветке ещё нет)."""
    round_rect(slide, x, y, w, h, TINT["lit"], radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT["lit"])
    cx_c = x + int(w * 0.66)
    cy_c = y + int(h * 0.40)
    d_c = Inches(1.15)
    oval(slide, cx_c - d_c // 2, cy_c - d_c // 2, d_c, CAT["lit"])
    tf = textbox(slide, cx_c - d_c // 2, cy_c - d_c // 2, d_c, d_c, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "База знаний", 11, WHITE, bold=True)

    sats = [("Методы: RF / SVM / SOM", 0.22), ("Кейсы + ссылки на публикации", 0.58)]
    d_s = Inches(0.98)
    sx = x + int(w * 0.18)
    for label, frac in sats:
        sy = y + int(h * frac)
        oval(slide, sx - d_s // 2, sy - d_s // 2, d_s, RGBColor(0xD8, 0xB0, 0x9E))
        tfs = textbox(slide, sx - d_s // 2 - Inches(0.1), sy - d_s // 2, d_s + Inches(0.2), d_s, MSO_ANCHOR.MIDDLE)
        ps = tfs.paragraphs[0]
        ps.alignment = PP_ALIGN.CENTER
        set_run(ps.add_run(), label, 8.5, DARK, bold=True)
        connector(slide, sx + d_s // 2, sy, cx_c - d_c // 2, cy_c, CAT["lit"], weight=1.25)

    cap_y = y + h - Inches(0.5)
    tf2 = textbox(slide, x + Inches(0.15), cap_y, w - Inches(0.3), Inches(0.46))
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    set_run(p2.add_run(), "Уже начато — растёт со ссылками на публикации", 10, GREY, italic=True)


OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================ Слайд 1 — Титул
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, 0, Inches(4.55), SW, Pt(5), GOLD)
tf = textbox(s, Inches(1.0), Inches(1.5), Inches(11.3), Inches(0.6), MSO_ANCHOR.BOTTOM)
p = tf.paragraphs[0]
set_run(p.add_run(), "АНАБАРСКИЙ ЩИТ · ЛИСТ R-48-XI, XII", 13, RGBColor(0x3F, 0xAE, 0x8C), bold=True)
tf = textbox(s, Inches(1.0), Inches(2.2), Inches(11.3), Inches(2.2), MSO_ANCHOR.BOTTOM)
p = tf.paragraphs[0]
set_run(p.add_run(), "Технологическая схема: золото-уран", 36, WHITE, bold=True)
p2 = tf.add_paragraph()
set_run(p2.add_run(), "как устроена разработка нового ГИС-метода: этапы, ветки, точки сравнения", 20, GOLD, bold=True)
tf2 = textbox(s, Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.8))
p = tf2.paragraphs[0]
set_run(p.add_run(), "Не линейная инструкция «делай раз, делай два», а разработка новой технологии — "
                      "с ветками, параллельным перебором подходов и подбором наилучшего решения в каждом узле", 16, WHITE)
p = tf2.add_paragraph(); p.space_before = Pt(10)
set_run(p.add_run(), "Организация · разработка метода · рабочая схема", 14, GOLD, italic=True)
p = tf2.add_paragraph()
set_run(p.add_run(), "28.08.2026", 12, RGBColor(0xBB, 0xBB, 0xBB))

# ============================================================ Слайд 2 — Обзор схемы
s = add_slide()
header(s, "Обзор: как устроена технологическая схема", 2)
tf = textbox(s, Inches(0.55), Inches(1.05), Inches(12.2), Inches(0.35))
p = tf.paragraphs[0]
set_run(p.add_run(), "11 узлов, две параллельные ветки сбора (данные / знания), пять параллельных экспериментальных веток и два контура возврата", 13, GREY, italic=True)

# Полотно-подложка под всю схему
CANVAS_X, CANVAS_Y = Inches(0.4), Inches(1.5)
CANVAS_W, CANVAS_H = Inches(12.55), Inches(4.5)
round_rect(s, CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H, RGBColor(0xFB, 0xFA, 0xF7), radius=0.02)

BOX_W, BOX_H = Inches(1.12), Inches(0.8)
N_COLS = 9
COL_X0 = Inches(1.05)
COL_X8 = Inches(11.9)
PITCH = (COL_X8 - COL_X0) // (N_COLS - 1)
BACKBONE_Y = Inches(3.75)
UPPER_Y = Inches(2.25)
LOWER_Y = Inches(5.25)

col_x = [COL_X0 + i * PITCH for i in range(N_COLS)]

NODES = [
    ("1", ["Постановка", "задачи"], col_x[0], BACKBONE_Y, "seq"),
    ("2", ["Эталон:", "критериальный"], col_x[1], BACKBONE_Y, "merge"),
    ("3", ["Агентная", "сборка БД"], col_x[2], LOWER_Y, "data"),
    ("5", ["Агентная", "сборка БЗ"], col_x[2], UPPER_Y, "lit"),
    ("4", ["Подготовка,", "паспортизация"], col_x[3], LOWER_Y, "data"),
    ("6", ["Граф знаний,", "связи"], col_x[3], UPPER_Y, "lit"),
    ("7", ["Соотнесение", "БД и БЗ"], col_x[4], BACKBONE_Y, "merge"),
    ("8A", ["Методы из", "литературы"], col_x[5], Inches(1.85), "gold"),
    ("8B", ["Геопростр.", "дата-майнинг"], col_x[5], Inches(2.65), "gold"),
    ("8C", ["Призн. простр.", "— инструмент"], col_x[5], Inches(3.45), "gold"),
    ("8D", ["Призн. простр.", "— объяснитель"], col_x[5], Inches(4.25), "gold"),
    ("8E", ["Без учителя", "+ граф"], col_x[5], Inches(5.05), "gold"),
    ("9", ["Сравнение", "кандидатов"], col_x[6], BACKBONE_Y, "final"),
    ("10", ["Верифи-", "кация"], col_x[7], BACKBONE_Y, "final"),
    ("11", ["Итог:", "метод + карта"], col_x[8], BACKBONE_Y, "final"),
]

NODE_XY = {}
for node_id, lines, cx, cy, cat in NODES:
    bw, bh = (Inches(1.32), Inches(0.62)) if node_id.startswith("8") and len(node_id) == 2 else (BOX_W, BOX_H)
    x0, y0 = cx - bw // 2, cy - bh // 2
    NODE_XY[node_id] = (cx, cy)
    round_rect(s, x0, y0, bw, bh, CAT[cat], radius=0.16)
    tf = textbox(s, x0, y0 + Inches(0.03), bw, bh - Inches(0.06), MSO_ANCHOR.TOP)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), node_id, 11.5 if bh < BOX_H else 13, WHITE, bold=True)
    for line in lines:
        pl = tf.add_paragraph()
        pl.alignment = PP_ALIGN.CENTER
        pl.space_before = Pt(0)
        set_run(pl.add_run(), line, 7.5 if bh < BOX_H else 9, WHITE)

EDGES = [
    ("1", "2", "seq"), ("1", "3", "data"), ("1", "5", "lit"),
    ("3", "4", "data"), ("5", "6", "lit"),
    ("4", "7", "data"), ("6", "7", "lit"), ("2", "9", "merge"),
    ("7", "8A", "gold"), ("7", "8B", "gold"), ("7", "8C", "gold"), ("7", "8D", "gold"), ("7", "8E", "gold"),
    ("8A", "9", "gold"), ("8B", "9", "gold"), ("8C", "9", "gold"), ("8D", "9", "gold"), ("8E", "9", "gold"),
    ("9", "10", "final"), ("10", "11", "final"),
]
for a, b, cat in EDGES:
    ax, ay = NODE_XY[a]
    bx, by = NODE_XY[b]
    aw = Inches(1.32) if a.startswith("8") and len(a) == 2 else BOX_W
    bw = Inches(1.32) if b.startswith("8") and len(b) == 2 else BOX_W
    if a == "2" and b == "9":
        # эталон подключается к сравнению кандидатов отдельной дугой сверху канвы
        x1, y1 = ax, ay - BOX_H // 2
        x2, y2 = bx, by - BOX_H // 2
        connector(s, x1, y1 - Inches(0.15), x2, y2 - Inches(0.15), CAT[cat], weight=1.6)
        continue
    x1, y1 = ax + aw // 2, ay
    x2, y2 = bx - bw // 2, by
    connector(s, x1, y1, x2, y2, CAT[cat], weight=1.6 if (a.startswith("8") or b.startswith("8")) else 2.0)

# контур 1: возврат при недостатке данных (4 -> 3)
loop_y = LOWER_Y + Inches(0.55)
lx1, ly1 = NODE_XY["4"][0], NODE_XY["4"][1] + BOX_H // 2
lx2, ly2 = NODE_XY["3"][0], NODE_XY["3"][1] + BOX_H // 2
cxn = connector(s, lx1, loop_y, lx2, loop_y, RGBColor(0x6F, 0xA8, 0x82), weight=1.4)
connector(s, lx1, ly1, lx1, loop_y, RGBColor(0x6F, 0xA8, 0x82), weight=1.4, arrow=False)
connector(s, lx2, loop_y, lx2, ly2, RGBColor(0x6F, 0xA8, 0x82), weight=1.4)
tf = textbox(s, (lx1 + lx2) // 2 - Inches(0.65), loop_y + Inches(0.02), Inches(1.3), Inches(0.24))
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
set_run(p.add_run(), "данных мало", 8, RGBColor(0x4A, 0x7C, 0x5E), italic=True)

# контур 2: очистка и проверка признаков внутри экспериментальных веток (8 -> 7)
loop_x = col_x[5] + Inches(0.95)
ly1 = NODE_XY["8A"][1] - Inches(0.05)
ly2 = NODE_XY["8E"][1] + Inches(0.05)
connector(s, loop_x, ly1, loop_x, ly2, RGBColor(0x6F, 0x93, 0xC2), weight=1.4, arrow=False)
tf = textbox(s, loop_x - Inches(0.08), (ly1 + ly2) // 2 - Inches(0.55), Inches(0.7), Inches(1.1))
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
set_run(p.add_run(), "контур очистки и проверки", 7.5, RGBColor(0x3E, 0x64, 0x8F), italic=True)

LEGEND = [
    ("seq", "старт"),
    ("merge", "эталон / контроль"),
    ("data", "ветка данных (БД)"),
    ("lit", "ветка знаний (БЗ)"),
    ("gold", "эксперимент. ветка"),
    ("final", "валидация и итог"),
]
lx = Inches(0.55)
ly = Inches(6.25)
for cat, label in LEGEND:
    oval(s, lx, ly, Inches(0.16), CAT[cat])
    tfl = textbox(s, lx + Inches(0.24), ly - Inches(0.07), Inches(1.85), Inches(0.32))
    pl = tfl.paragraphs[0]
    set_run(pl.add_run(), label, 10, GREY)
    lx += Inches(2.05)

tf = textbox(s, Inches(0.55), Inches(6.7), Inches(12.2), Inches(0.55))
p = tf.paragraphs[0]
set_run(p.add_run(), "Критерий оценки на каждом узле — сравнение подходов между собой, с критериальным методом ВНИГНИ (узел 2) и с экспертной оценкой геологов. Пунктирные линии — контуры возврата: при нехватке данных и при очистке/проверке признаков внутри узла 8", 11.5, GREY, italic=True)

# ============================================================ Слайд 3 — Узел 1
s = add_slide()
header(s, "Постановка задачи", 3, node="1", cat="seq")
text_x, text_w = Inches(0.55), Inches(6.85)
labeled_block(
    s, text_x, Inches(1.35), text_w, Inches(1.55), "ЦЕЛЬ",
    [
        "Разработать новый ГИС-метод прогнозирования золото-урановых узлов на основе "
        "машинного обучения и нейросетей для Анабарского щита (лист R-48-XI, XII) — "
        "переносимый метод, а не разовая карта под конкретный участок.",
    ], cat="seq",
)
labeled_block(
    s, text_x, Inches(3.05), text_w, Inches(2.35), "ЗАДАЧИ",
    [
        "Собрать данные территории независимо от признаков критериального анализа "
        "(узел 2 — эталон, а не источник признаков); двумя параллельными ветками "
        "(данные и знания) выйти на пять экспериментальных направлений обучения; в "
        "каждом подобрать свою лучшую модель; сравнить всех кандидатов между собой, "
        "с критериальным методом ВНИГНИ и с геологической логикой; построить итоговую "
        "карту районирования зон перспективности.",
    ], cat="data",
)
labeled_block(
    s, text_x, Inches(5.55), text_w, Inches(1.55), "КРИТЕРИИ РЕЗУЛЬТАТА",
    [
        "Прямое сравнение с уже опробованным методом — критериальным анализом ВНИГНИ — "
        "и экспертное заключение геологов-предметников, оба критерия равноправны: "
        "статистическое превосходство без согласия эксперта не считается успехом, как и наоборот.",
    ], cat="final",
)
status_panel(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75), "seq", "1",
             "не вычислительный этап — постановка цели, задач и критериев оценки")

# ============================================================ Слайд 4 — Узел 2
step_slide(
    4, "2", "merge", "Эталон: критериальный метод",
    [
        "Прежде чем начинать сбор данных, потребовалось решить отдельную большую задачу — "
        "переписать критериальный анализ ВНИГНИ с нуля на Python. Он реализован в "
        "проприетарной ГИС Интегро, поэтому просто «взять формулу» было нельзя: пришлось "
        "воспроизвести весь набор факторных слоёв и саму логику объединяющей формулы, шаг "
        "за шагом сверяя результат с оригиналом. Объём работы оказался достаточным, чтобы "
        "выделить её в отдельный репозиторий и отдельную презентацию точности воссоздания, "
        "вне рамок этого проекта.",
        "Значение этого шага не в самом факте переписывания, а в том, что без него сравнивать "
        "новый метод машинного обучения со старым было бы не с чем: любое «мы лучше критериального анализа» "
        "повисло бы в воздухе. Узел работает как эталон и контроль на всём протяжении схемы — "
        "именно с ним в итоге сравниваются кандидаты из узла 9, а не только в самом конце.",
    ],
    image_name="criterial_prognoz_map.png",
    image_caption="Нативный результат ГИС Интегро (prognoz.pgrid, меньше = перспективнее), контур — 3 выделенных рудных объекта",
)

# ============================================================ Слайд 5 — Узел 3
step_slide(
    5, "3", "data", "Агентная сборка базы данных",
    [
        "Для территории собирается максимум доступных данных — намеренно без признаков, "
        "которые использует сам критериальный анализ. Если включить их в модель, метод "
        "перестанет переноситься на другие территории, а сравнение с критериальным анализом "
        "превратится в подгонку под уже известный ответ.",
        "Заранее неизвестно, какие именно данные понадобятся — сам метод ещё предстоит "
        "разработать, поэтому собирается по возможности весь доступный объём: карты, "
        "геофизика, данные дистанционного зондирования. Сбор ведут агентные системы "
        "(разработка — Александр Лебедев); пересечение с данными, которые уже использует "
        "ГИС Интегро, фиксируется отдельно.",
    ],
    image_name="dataset_v6_preview.png",
    image_caption="dataset_v6 — представители разных групп признаков на сетке листа",
)

# ============================================================ Слайд 6 — Узел 4 (+ контур)
s = add_slide()
header(s, "Подготовка и паспортизация данных", 6, node="4", cat="data")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "Собранные слои приводятся к единой сетке и системе координат, проверяются на "
    "покрытие и качество; для каждого источника документируются разрешение, охват и "
    "дата съёмки — без этого признаки из разных систем нельзя честно сопоставлять "
    "друг с другом.",
    "На этом же шаге проверяется, достаточно ли данных для дальнейшей работы. Если по "
    "территории или по конкретному признаку данных не хватает, процесс возвращается к "
    "агентной сборке (узел 3) за доп. сбором, а не идёт дальше с пробелами — это первый "
    "из двух контуров возврата в схеме.",
], size=15.5)
sufficiency_loop_diagram(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75))

# ============================================================ Слайд 7 — Узел 5
s = add_slide()
header(s, "Агентная сборка базы знаний", 7, node="5", cat="lit")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "Параллельно сбору сырых данных запущена вторая, независимая линия работы — "
    "агентная сборка базы знаний по мировым публикациям: кто уже искал похожие рудные "
    "объекты и какими данными при этом пользовался. Сбор и здесь ведут агентные системы "
    "(разработка — Александр Лебедев).",
    "Работа уже начата: база растёт со ссылками на публикации и аннотациями по трём "
    "семействам методов — Random Forest, SVM и самоорганизующиеся карты Кохонена, — "
    "чтобы позже соединиться с уже собранными данными (узел 7), а через литературу "
    "обосновывается геологическая значимость отдельных признаков.",
], size=15.5)
knowledge_base_diagram(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75))

# ============================================================ Слайд 8 — Узел 6
s = add_slide()
header(s, "Граф знаний и предметные связи", 8, node="6", cat="lit")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "База знаний пополняется не просто списком публикаций, а со связями: какой признак "
    "в какой работе оказался значим, через какой геологический механизм и в связке с "
    "каким методом. Это отдельный шаг, а не побочный эффект накопления ссылок.",
    "Именно этот граф понадобится дальше в узле 8E — обучении без учителя: прошлая "
    "попытка такого обучения без предметного графа дала статистически рабочий, но "
    "геологически необъяснимый результат (признаки просто сильно коррелировали). Граф "
    "нужен, чтобы объяснять «почему так, а не иначе», а не только фиксировать «что "
    "сработало».",
], size=15.5)
concept_graph_diagram(
    s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75), "lit",
    "Граф знаний",
    ["Публикации (БЗ)", "Признаки БД", "Геологический механизм", "Метод / алгоритм"],
    "Связи признак ↔ механизм ↔ метод — основа для узла 8E",
)

# ============================================================ Слайд 9 — Узел 7
step_slide(
    9, "7", "merge", "Соотнесение базы данных и базы знаний",
    [
        "Данных собрано много, но неясно, чему и как учить модель. Здесь мировой опыт из "
        "базы знаний впервые сверяется с реальным собранным датасетом территории.",
        "Результат узла — первые направления обучения, а не готовая модель. После сверки "
        "процесс расходится на пять параллельных экспериментальных веток (узел 8) — "
        "каждая из них подбирает свой лучший подход независимо от остальных.",
    ],
    status_caption="направление не подтверждено — сверка с базой знаний ещё предстоит",
)

# ============================================================ Слайд 10 — Узел 8 (обзор веток)
s = add_slide()
header(s, "Развилка: пять экспериментальных веток", 10, node="8", cat="gold")
tf = textbox(s, Inches(0.55), Inches(1.12), Inches(12.2), Inches(0.5))
p = tf.paragraphs[0]
set_run(p.add_run(), "Подбор модели — не общий шаг после слияния, а отдельный внутренний этап внутри каждой ветки; сравнение (узел 9) идёт уже после того, как каждая ветка выбрала свой лучший вариант.", 13.5, GREY, italic=True)

BRANCHES = [
    ("8A", "Методы из литературы",
     "Мини-датасет под каждую найденную статью, метод воспроизводится на наших данных, "
     "результат сравнивается с критериальным тестом — большая ветка, много итераций.",
     "не начато"),
    ("8B", "Геопространственный дата-майнинг",
     "Весь датасет картируется целиком без отбора; каждый признак объясняется — что это "
     "и на каком геологическом основании может указывать на золото.",
     "в работе"),
    ("8C · 8D", "Признаковое пространство — инструмент и объяснитель",
     "Поиск «облаков» золота через корреляцию с узлами критериального анализа + перенос "
     "модели, обученной на клетках месторождений, на смежный лист со сверкой весов сети.",
     "в работе"),
    ("8E", "Обучение без учителя + предметный граф",
     "Автономный перебор методов без учителя, объяснение результата через граф знаний "
     "(узел 6), а не только через статистическую корреляцию.",
     "не начато"),
]
row_y = Inches(1.85)
row_h = Inches(1.12)
for code, title_txt, desc, status in BRANCHES:
    round_rect(s, Inches(0.55), row_y, Inches(12.2), row_h - Inches(0.14), LIGHT, radius=0.06)
    rect(s, Inches(0.55), row_y, Pt(4), row_h - Inches(0.14), CAT["gold"])
    tfc = textbox(s, Inches(0.8), row_y + Inches(0.08), Inches(1.55), row_h - Inches(0.3), MSO_ANCHOR.MIDDLE)
    pc = tfc.paragraphs[0]
    set_run(pc.add_run(), code, 15, CAT["gold"], bold=True)
    tft = textbox(s, Inches(2.45), row_y + Inches(0.08), Inches(7.7), row_h - Inches(0.3), MSO_ANCHOR.MIDDLE)
    pt = tft.paragraphs[0]
    set_run(pt.add_run(), title_txt, 12.5, DARK, bold=True)
    pd = tft.add_paragraph()
    pd.space_before = Pt(2)
    set_run(pd.add_run(), desc, 10.5, GREY)
    tfs = textbox(s, Inches(10.35), row_y + Inches(0.08), Inches(2.25), row_h - Inches(0.3), MSO_ANCHOR.MIDDLE)
    ps = tfs.paragraphs[0]
    ps.alignment = PP_ALIGN.RIGHT
    set_run(ps.add_run(), status, 11.5, GREEN if status == "в работе" else GREY, bold=True, italic=(status != "в работе"))
    row_y += row_h

tf = textbox(s, Inches(0.55), row_y + Inches(0.05), Inches(12.2), Inches(0.5))
p = tf.paragraphs[0]
set_run(p.add_run(), "Контур очистки и проверки: внутри каждой ветки признаки и промежуточные результаты вычищаются от геологически бессмысленных корреляций до того, как кандидат попадёт в узел 9", 11.5, GREY, italic=True)

# ============================================================ Слайд 11 — Узел 8A
step_slide(
    11, "8A", "gold", "8A · Методы из литературы",
    [
        "Если кто-то уже искал золото по признакам, похожим на те, что есть в нашем "
        "датасете, под эту статью собирается собственный мини-датасет — только с её "
        "признаками. Метод статьи воспроизводится на наших данных, а результат "
        "сравнивается с критериальным тестом.",
        "Каждый признак картируется до обучения, а результат — после, чтобы было видно, "
        "что именно изменилось. Ожидается много таких итераций: это большая и трудоёмкая "
        "ветка, требующая отдельного прогона под каждую найденную работу.",
    ],
    status_caption="не начато — предметно за пределами разовых конспектов литературы",
)

# ============================================================ Слайд 12 — Узел 8B
step_slide(
    12, "8B", "gold", "8B · Геопространственный дата-майнинг",
    [
        "Весь собранный датасет картируется целиком, без предварительного отбора. Для "
        "каждого признака фиксируется, что это такое, зачем он добавлен и на каком "
        "геологическом основании он может указывать на золото — обоснование ищется через "
        "литературу и объяснения из базы знаний (узел 6).",
        "Признаки, которые пока не удаётся объяснить геологически — например, "
        "вегетационные индексы, — на этом этапе не отбрасываются: причина может найтись "
        "позже, а раннее исключение рискует выбросить рабочий сигнал вместе с шумом.",
    ],
    image_name="feature_space_corr_heatmap_slide.png",
    image_caption="Корреляционная матрица 104 признаков dataset_v6, сгруппированных в 20 кластеров (Spearman)",
)

# ============================================================ Слайд 13 — Узел 8C/8D
step_slide(
    13, "8C", "gold", "8C · 8D — Признаковое пространство",
    [
        "Признаковое пространство работает одновременно в двух ролях. Как инструмент (8C) — "
        "поиск «облаков» золота через корреляцию с известными узлами критериального "
        "анализа, взятыми как якорь; сильно скоррелированные признаки при этом не всегда "
        "стоит вычищать. Как объяснитель (8D) — модель обучается на клетках известных "
        "месторождений, результат переносится на смежный лист, а большие веса сети "
        "сверяются с геологической логикой.",
        "Главный риск — сеть может выучить геологическую бессмыслицу вроде «озеро рядом — "
        "значит золото»: такой шум отслеживается и вычищается через контур очистки "
        "внутри узла 8.",
    ],
    image_name="feature_space_map.png",
    image_caption="Признаковое пространство dataset_v6 (MDS-проекция по корреляционному расстоянию), цвет — геологическая категория, ★ — значимость подтверждена статистически",
)

# ============================================================ Слайд 14 — Узел 8E
step_slide(
    14, "8E", "gold", "8E · Обучение без учителя + предметный граф",
    [
        "Отдельный автономный конвейер на основе крупных языковых моделей перебирает "
        "разные методы обучения без учителя на всём собранном датасете и сверяется с уже "
        "размеченной территорией — без готового ответа, куда должна вести модель.",
        "Ключевое условие, добавленное после прошлой неудачной попытки: параллельно "
        "обязательно строится предметный граф признаков (узел 6), который объясняет "
        "«почему так, а не иначе». Без него результат может быть статистически рабочим, "
        "но геологически необъяснимым — предметник не примет ответ «признаки просто "
        "сильно скоррелировались».",
    ],
    status_caption="не начато — зависит от готовности графа знаний в узле 6",
)

# ============================================================ Слайд 15 — Узел 9
step_slide(
    15, "9", "final", "Сравнение кандидатов",
    [
        "Каждая из пяти экспериментальных веток (узел 8) выходит со своим лучшим "
        "кандидатом. Здесь кандидаты сравниваются друг с другом и с эталоном — "
        "критериальным методом ВНИГНИ (узел 2), который подключён к этому сравнению "
        "напрямую, а не только на старте схемы.",
        "Это ещё не финальная приёмка: пройти сравнение — необходимое, но не "
        "достаточное условие, дальше кандидат должен пройти верификацию (узел 10).",
    ],
    image_name="compare_methods_map.png",
    image_caption="Карта: лучший метод каждого семейства vs критериальный эталон vs расхождения",
)

# ============================================================ Слайд 16 — Узел 10
s = add_slide()
header(s, "Верификация", 16, node="10", cat="final")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "Кандидат, прошедший сравнение (узел 9), проверяется на устойчивость: повторный "
    "прогон с другими сидами, перенос на смежный лист и, отдельно, согласие признаков "
    "и результата с геологической логикой, а не только со статистикой.",
    "Финальная проверка — на данных, не участвовавших в конвейере обучения: только так "
    "можно отличить реальную обобщающую способность метода от подгонки под обучающую "
    "выборку. Этап описывает будущий порядок действий, а не уже полученный результат, "
    "поэтому справа — схема самого порядка сверки, а не карта.",
], size=15.5)
verification_diagram(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75))

# ============================================================ Слайд 17 — Узел 11
s = add_slide()
header(s, "Итог: метод и карта районирования", 17, node="11", cat="final")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "Два итоговых результата схемы. Первый — сам метод: переносимый ГИС-метод "
    "прогнозирования золото-урановых узлов, применимый за пределами конкретного "
    "участка, где он разработан.",
    "Второй — практическое применение метода: карта территории с выделенными зонами "
    "перспективности и отмеченными кандидатами золото-урановых узлов. Она становится "
    "базой для сравнения с экспертной оценкой геологов — тем же критерием, что был "
    "задан на первом этапе схемы (узел 1). Итоговая карта ещё не готова — она зависит "
    "от результатов всех предыдущих узлов, поэтому справа показан макет самого приёма "
    "построения, а не промежуточный рабочий вариант.",
], size=15.5)
zonation_diagram(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75))

prs.save(OUT)
print("сохранено:", OUT)
print("слайдов:", len(prs.slides._sldIdLst))
