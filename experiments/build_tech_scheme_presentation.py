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

# Категории узлов схемы (цвета — из скачанного шаблона-инфографики)
CAT = {
    "seq": RGBColor(0x5F, 0x5E, 0x5A),    # последовательный шаг
    "merge": RGBColor(0x3C, 0x34, 0x89),  # точка слияния / разветвление
    "data": RGBColor(0x0F, 0x6E, 0x56),   # ветка данных
    "lit": RGBColor(0xA3, 0x39, 0x1B),    # ветка литературы
    "final": RGBColor(0x85, 0x4F, 0x0B),  # финал
    "gold": GOLD,
}
TINT = {
    "seq": RGBColor(0xED, 0xEA, 0xE0),
    "merge": RGBColor(0xEE, 0xED, 0xFE),
    "data": RGBColor(0xE1, 0xF5, 0xEE),
    "lit": RGBColor(0xFA, 0xEC, 0xE7),
    "final": RGBColor(0xFA, 0xEE, 0xDA),
}
CAT_LABEL = {
    "seq": "ПОСЛЕДОВАТЕЛЬНЫЙ ШАГ",
    "merge": "ТОЧКА ВЕТВЛЕНИЯ / СЛИЯНИЯ",
    "data": "ВЕТКА ДАННЫХ",
    "lit": "ВЕТКА ЛИТЕРАТУРЫ",
    "final": "ФИНАЛ",
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
    """Узел 9: векторная схема верификации вместо картинки — сравнение веток
    само по себе не результат, а процесс, который ещё предстоит пройти."""
    round_rect(slide, x, y, w, h, LIGHT, radius=0.04)
    rect(slide, x, y, Pt(4), h, CAT["seq"])
    cx = x + w // 2
    bh = Inches(0.62)
    top_y = y + Inches(0.5)
    left_cx = x + int(w * 0.28)
    right_cx = x + int(w * 0.72)
    bw = Inches(1.95)
    flow_box(slide, left_cx, top_y, bw, bh, "data", ["Ветка", "данных"])
    flow_box(slide, right_cx, top_y, bw, bh, "lit", ["Ветка", "литературы"])
    gap = Inches(0.55)
    y2 = top_y + bh + gap
    flow_box(slide, cx, y2, Inches(3.9), bh, "merge", ["Сравнение с критериальным", "методом ВНИГНИ"], size=10.5)
    y3 = y2 + bh + gap
    flow_box(slide, cx, y3, Inches(3.9), bh, "seq", ["Сверка с геологической логикой"], size=10.5)
    y4 = y3 + bh + gap
    flow_box(slide, cx, y4, Inches(3.9), bh, "final", ["Проверка на отложенных данных"], size=10.5)
    connector(slide, left_cx, top_y + bh // 2, cx, y2 - bh // 2, CAT["data"], weight=1.5)
    connector(slide, right_cx, top_y + bh // 2, cx, y2 - bh // 2, CAT["lit"], weight=1.5)
    connector(slide, cx, y2 + bh // 2, cx, y3 - bh // 2, CAT["merge"], weight=1.5)
    connector(slide, cx, y3 + bh // 2, cx, y4 - bh // 2, CAT["seq"], weight=1.5)
    cap_y = y4 + bh // 2 + Inches(0.3)
    tf = textbox(slide, x + Inches(0.25), cap_y, w - Inches(0.5), Inches(0.85))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), "Результаты обеих веток сравниваются с критериальным методом, сверяются с "
                          "геологической логикой и проверяются на данных вне конвейера обучения", 10.5, GREY, italic=True)


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
set_run(p.add_run(), "Ветки идут параллельно; узел завершается сравнением результатов, а не единственным правильным шагом", 13, GREY, italic=True)

# Полотно-подложка под всю схему
CANVAS_X, CANVAS_Y = Inches(0.4), Inches(1.5)
CANVAS_W, CANVAS_H = Inches(12.55), Inches(4.5)
round_rect(s, CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H, RGBColor(0xFB, 0xFA, 0xF7), radius=0.02)

BOX_W, BOX_H = Inches(1.08), Inches(0.82)
N_COLS = 10
COL_X0 = Inches(1.05)
COL_X9 = Inches(11.9)
PITCH = (COL_X9 - COL_X0) // (N_COLS - 1)
BACKBONE_Y = Inches(3.75)
UPPER_Y = Inches(2.25)
LOWER_Y = Inches(5.25)

col_x = [COL_X0 + i * PITCH for i in range(N_COLS)]

NODES = [
    ("1", ["Постановка", "задачи"], col_x[0], BACKBONE_Y, "seq"),
    ("2", ["Критериальный", "метод"], col_x[1], BACKBONE_Y, "seq"),
    ("3", ["Сбор сырых", "данных"], col_x[2], BACKBONE_Y, "seq"),
    ("4A", ["Открытые", "данные"], col_x[3], UPPER_Y, "data"),
    ("4B", ["Агенты: БЗ"], col_x[3], LOWER_Y, "lit"),
    ("5", ["Соотнесение", "БЗ и данных"], col_x[4], BACKBONE_Y, "merge"),
    ("6A", ["Дата-", "майнинг"], col_x[5], UPPER_Y, "data"),
    ("6B", ["Методы из", "литературы"], col_x[5], LOWER_Y, "lit"),
    ("7A", ["Признаковый", "анализ"], col_x[6], UPPER_Y, "data"),
    ("8A", ["Подбор", "модели"], col_x[7], UPPER_Y, "data"),
    ("8B", ["Подбор", "модели"], col_x[7], LOWER_Y, "lit"),
    ("9", ["Верифи-", "кация"], col_x[8], BACKBONE_Y, "seq"),
    ("10", ["Райониро-", "вание зон"], col_x[9], BACKBONE_Y, "final"),
]

NODE_XY = {}
for node_id, lines, cx, cy, cat in NODES:
    x0, y0 = cx - BOX_W // 2, cy - BOX_H // 2
    NODE_XY[node_id] = (cx, cy)
    round_rect(s, x0, y0, BOX_W, BOX_H, CAT[cat], radius=0.16)
    tf = textbox(s, x0, y0 + Inches(0.06), BOX_W, BOX_H - Inches(0.1), MSO_ANCHOR.TOP)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), node_id, 13, WHITE, bold=True)
    for line in lines:
        pl = tf.add_paragraph()
        pl.alignment = PP_ALIGN.CENTER
        pl.space_before = Pt(1)
        set_run(pl.add_run(), line, 9, WHITE)

EDGES = [
    ("1", "2", "gold"), ("2", "3", "gold"),
    ("3", "4A", "data"), ("3", "4B", "lit"),
    ("4A", "5", "data"), ("4B", "5", "lit"),
    ("5", "6A", "data"), ("5", "6B", "lit"),
    ("6A", "7A", "data"), ("7A", "8A", "data"),
    ("6B", "8B", "lit"),
    ("8A", "9", "data"), ("8B", "9", "lit"),
    ("9", "10", "gold"),
]
for a, b, cat in EDGES:
    ax, ay = NODE_XY[a]
    bx, by = NODE_XY[b]
    x1, y1 = ax + BOX_W // 2, ay
    x2, y2 = bx - BOX_W // 2, by
    connector(s, x1, y1, x2, y2, CAT[cat], weight=2.0)

LEGEND = [
    ("seq", "последовательный шаг"),
    ("merge", "точка слияния / развилка"),
    ("data", "ветка данных"),
    ("lit", "ветка литературы"),
    ("final", "финал"),
]
lx = Inches(0.75)
ly = Inches(6.28)
for cat, label in LEGEND:
    oval(s, lx, ly, Inches(0.18), CAT[cat])
    tfl = textbox(s, lx + Inches(0.28), ly - Inches(0.06), Inches(2.15), Inches(0.32))
    pl = tfl.paragraphs[0]
    set_run(pl.add_run(), label, 11, GREY)
    lx += Inches(2.45)

tf = textbox(s, Inches(0.55), Inches(6.78), Inches(12.2), Inches(0.5))
p = tf.paragraphs[0]
set_run(p.add_run(), "Критерий оценки на каждом узле — сравнение подходов между собой, с критериальным методом ВНИГНИ и с экспертной оценкой геологов", 12, GREY, italic=True)

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
        "Собрать данные территории независимо от признаков критериального анализа; "
        "исследовать два параллельных направления — геопространственный дата-майнинг "
        "всего датасета и воспроизведение методов из мировой литературы; в каждом "
        "направлении подобрать свою лучшую модель; сравнить оба результата между собой, "
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
    4, "2", "seq", "Воссоздание критериального метода",
    [
        "Прежде чем начинать сбор данных, потребовалось решить отдельную большую задачу — "
        "переписать критериальный анализ ВНИГНИ с нуля на Python. Он реализован в "
        "проприетарной ГИС Интегро, поэтому просто «взять формулу» было нельзя: пришлось "
        "воспроизвести весь набор факторных слоёв и саму логику объединяющей формулы, шаг "
        "за шагом сверяя результат с оригиналом. Объём работы оказался достаточным, чтобы "
        "выделить её в отдельный репозиторий и отдельную презентацию точности воссоздания, "
        "вне рамок этого проекта.",
        "Значение этого шага не в самом факте переписывания, а в том, что без него сравнивать "
        "новый ML-метод со старым было бы не с чем: любое «мы лучше критериального анализа» "
        "повисло бы в воздухе. На сегодня задача решена, точность воссоздания подтверждена, "
        "и база для честного сравнения готова.",
    ],
    image_name="criterial_prognoz_map.png",
    image_caption="Нативный результат ГИС Интегро (prognoz.pgrid, меньше = перспективнее), контур — 3 выделенных рудных объекта",
)

# ============================================================ Слайд 5 — Узел 3
step_slide(
    5, "3", "seq", "Сбор сырых данных",
    [
        "Для территории собирается максимум доступных данных — намеренно без признаков, "
        "которые использует сам критериальный анализ. Если включить их в модель, метод "
        "перестанет переноситься на другие территории, а сравнение с критериальным анализом "
        "превратится в подгонку под уже известный ответ.",
        "Заранее неизвестно, какие именно данные понадобятся — сам метод ещё предстоит "
        "разработать, поэтому собирается и описывается по возможности весь доступный объём, "
        "а пересечение с данными, которые уже использует ГИС Интегро, фиксируется отдельно.",
    ],
    image_name="dataset_v6_preview.png",
    image_caption="dataset_v6 — представители разных групп признаков на сетке листа",
)

# ============================================================ Слайд 6 — Узел 4 (развилка)
s = add_slide()
header(s, "Развилка: открытые данные и база знаний", 6, node="4", cat="merge")
tf = textbox(s, Inches(0.55), Inches(1.12), Inches(12.2), Inches(0.45))
p = tf.paragraphs[0]
set_run(p.add_run(), "Параллельно сбору сырых данных запускается вторая, независимая линия работы — агентная сборка базы знаний по мировым публикациям.", 13.5, GREY, italic=True)
colL_x, colR_x = Inches(0.55), Inches(6.85)
col_w = Inches(5.95)
col_top = Inches(1.7)
branch_column(s, colL_x, col_top, col_w, "data", "4A · Открытые данные", [
    "Максимально возможный сбор карт, геофизики и данных дистанционного зондирования по "
    "территории. Каждый источник описывается непосредственно в момент сбора — что это, "
    "какое разрешение и какой охват.",
])
branch_column(s, colR_x, col_top, col_w, "lit", "4B · Агенты: база знаний", [
    "Агенты собирают базу знаний по мировым публикациям: кто уже искал похожие рудные "
    "объекты и какие данные при этом использовал. База растёт со ссылками, чтобы позже "
    "соединиться с уже собранными данными, а через литературу обосновывается геологическая "
    "значимость отдельных признаков.",
])
foot_y = Inches(4.55)
foot_h = Inches(2.55)
image_panel(s, colL_x, foot_y, col_w, foot_h, "data", "ggk200_overview_map.png",
            "Обзорная карта: 8 листов ГГК-200 + контур расчётной сетки (открытые источники)")
knowledge_base_diagram(s, colR_x, foot_y, col_w, foot_h)

# ============================================================ Слайд 7 — Узел 5
step_slide(
    7, "5", "merge", "Точка слияния: соотнесение базы знаний и данных",
    [
        "Данных собрано много, но неясно, чему и как учить модель. Здесь мировой опыт из "
        "базы знаний впервые сверяется с реальным собранным датасетом территории.",
        "Результат этапа — первые направления обучения, а не готовая модель. После сверки "
        "процесс снова расходится на две независимые ветки: геопространственный "
        "дата-майнинг и воспроизведение методов из литературы.",
    ],
    status_caption="направление не подтверждено — сверка с базой знаний ещё предстоит",
)

# ============================================================ Слайд 8 — Узел 6A
step_slide(
    8, "6A", "data", "Геопространственный дата-майнинг",
    [
        "Весь собранный датасет картируется целиком, без предварительного отбора. Для "
        "каждого признака фиксируется, что это такое, зачем он добавлен и на каком "
        "геологическом основании он может указывать на золото — обоснование ищется через "
        "литературу и объяснения из базы знаний.",
        "Признаки, которые пока не удаётся объяснить геологически — например, "
        "вегетационные индексы, — на этом этапе не отбрасываются: причина может найтись "
        "позже, а раннее исключение рискует выбросить рабочий сигнал вместе с шумом.",
    ],
    image_name="feature_space_corr_heatmap_slide.png",
    image_caption="Корреляционная матрица 104 признаков dataset_v6, сгруппированных в 20 кластеров (Spearman)",
)

# ============================================================ Слайд 9 — Узел 7A
step_slide(
    9, "7A", "data", "Признаковое пространство: инструмент и объяснитель",
    [
        "Признаковое пространство работает одновременно в двух ролях. Как инструмент — "
        "поиск «облаков» золота через корреляцию с известными узлами критериального "
        "анализа, взятыми как якорь; сильно скоррелированные признаки при этом не всегда "
        "стоит вычищать. Как объяснитель — модель обучается на клетках известных "
        "месторождений, результат переносится на смежный лист, а большие веса сети "
        "сверяются с геологической логикой.",
        "Главный риск — сеть может выучить геологическую бессмыслицу вроде «озеро рядом — "
        "значит золото»: такой шум нужно отслеживать и вычищать. Обучение без учителя "
        "требует отдельного предметного графа признаков — без него результат статистически "
        "рабочий, но геологически необъясним, и его нельзя показать эксперту.",
    ],
    image_name="feature_space_map.png",
    image_caption="Признаковое пространство dataset_v6 (MDS-проекция по корреляционному расстоянию), цвет — геологическая категория, ★ — значимость подтверждена статистически",
)

# ============================================================ Слайд 10 — Узел 6B
step_slide(
    10, "6B", "lit", "Методы из литературы",
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

# ============================================================ Слайд 11 — Узел 8 (развилка)
s = add_slide()
header(s, "Подбор наилучшей модели — в каждой ветке", 11, node="8", cat="merge")
tf = textbox(s, Inches(0.55), Inches(1.12), Inches(12.2), Inches(0.45))
p = tf.paragraphs[0]
set_run(p.add_run(), "Подбор модели — не общий шаг после слияния веток, а отдельный внутренний этап внутри каждой из них; сравнение результатов идёт уже после того, как обе ветки выбрали свой лучший вариант.", 13.5, GREY, italic=True)
branch_column(s, colL_x, col_top, col_w, "data", "8A · Ветка данных", [
    "Ветка признакового анализа подбирает свою лучшую модель или алгоритм — этот шаг "
    "раньше в схеме отсутствовал и добавлен отдельно, вместе с шагом 8B.",
])
branch_column(s, colR_x, col_top, col_w, "lit", "8B · Ветка литературы", [
    "Ветка сырых данных и воспроизведённых литературных методов подбирает свой лучший "
    "способ обучения независимо от ветки признакового анализа.",
])
image_panel(s, colL_x, foot_y, Inches(12.25), foot_h, "merge", "compare_methods_map.png",
            "Карта: лучший метод каждого семейства vs критериальный эталон vs расхождения — общий прогон для обеих веток")

# ============================================================ Слайд 12 — Узел 9
s = add_slide()
header(s, "Верификация и соотнесение с предметной областью", 12, node="9", cat="seq")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "Результаты обеих веток сравниваются между собой и с критериальным методом ВНИГНИ. "
    "Отдельно проверяется, согласуются ли признаки и результаты ИИ с геологической "
    "логикой, а не только со статистикой.",
    "Финальная проверка проводится на данных, не участвовавших в конвейере обучения — "
    "только так можно отличить реальную обобщающую способность метода от подгонки под "
    "обучающую выборку. Этот этап описывает будущий порядок действий, а не уже "
    "полученный результат, поэтому вместо карты — схема самого порядка сверки.",
], size=15.5)
verification_diagram(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75))

# ============================================================ Слайд 13 — Узел 10
s = add_slide()
header(s, "Районирование зон", 13, node="10", cat="final")
prose(s, Inches(0.55), Inches(1.35), Inches(6.85), Inches(5.75), [
    "Итоговый практический результат — карта территории с выделенными зонами "
    "перспективности: где искать в первую очередь, а где вероятность находки "
    "минимальна.",
    "На карте отмечаются кандидаты золото-урановых узлов, а сам результат становится "
    "базой для сравнения с экспертной оценкой геологов — тем же критерием, что был "
    "задан на первом этапе схемы (узел 1). Итоговая карта ещё не готова — она зависит "
    "от результатов всех предыдущих узлов, поэтому справа показан макет самого приёма "
    "построения, а не промежуточный рабочий вариант.",
], size=15.5)
zonation_diagram(s, Inches(7.65), Inches(1.35), Inches(5.1), Inches(5.75))

prs.save(OUT)
print("сохранено:", OUT)
print("слайдов:", len(prs.slides._sldIdLst))
