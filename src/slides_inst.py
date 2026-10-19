# -*- coding: utf-8 -*-
"""Светлый институтский стиль презентаций: холст 10 x 5.625", синий акцент.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. В проекте два стиля колод. `src/slides.py` — прежний
(холст 13.333 x 7.5", тёмная шапка, золотая линейка). Серия отчётов для
организации с 28.09.2026 свёрстана иначе: белый лист 10 x 5.625", синий акцент,
тонкие линии-разделители, верхний колонтитул «01 • РАЗДЕЛ … ПОДЗАГОЛОВОК
СПРАВА», нижний «ФГБУ ВНИГНИ · РАЗВИТИЕ МЕТОДОВ ИИ … 02 / 06», рубрика синими
капителями над заголовком, врезка «ИТОГ» тёмной полосой внизу. Раньше эти
примитивы жили копией в `experiments/fix_ai_vnigni_v2.py`; здесь они собраны
в одном месте, чтобы следующая колода той же серии не переписывала их заново.

Палитра и размеры сняты с файла
`docs II presentation/Презентация-ИИ-во-ВНИГНИ-06.10.2026.pptx` — он остаётся
эталоном внешнего вида серии.

    from src.slides_inst import Deck, bullets, card, metric, picture, table
    deck = Deck(total=6)
    deck.title_slide(...)
    sl = deck.slide("АНАЛИЗ", "ЧЕМ НАБРАНЫ ЯДРА", "Заголовок", "подзаголовок")
    bullets(sl, 5.35, 2.12, 4.15, 1.45, [("Лид: ", "остальной текст")])
    deck.note(sl, "Одна мысль, одно предложение.")
    deck.save(OUT)

Проверка готового файла — `python -X utf8 -m src.slide_check "<файл>.pptx"`.
Проверка настроена на холст 13.3": на 10" она даёт предупреждения о кегле
меньше 10 пт и о близости к краю — здесь это норма стиля, смотреть надо
выгрузку в PNG.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "docs II presentation"
IMG = ROOT / "outputs"

# --- палитра ---------------------------------------------------------------
ACCENT = RGBColor(0x25, 0x63, 0xEB)    # синий акцент
ACCENT_D = RGBColor(0x1D, 0x4E, 0xD8)  # текст на бледно-голубом
ACCENT_L = RGBColor(0x60, 0xA5, 0xFA)  # метка на тёмной полосе
PALE_BLUE = RGBColor(0xDB, 0xEA, 0xFE)  # бледно-голубая заливка
ON_BLUE = RGBColor(0xDB, 0xEA, 0xFE)   # подпись на синем
DARK = RGBColor(0x0F, 0x17, 0x2A)      # заголовки
TEXT = RGBColor(0x1E, 0x29, 0x3B)      # основной текст
DARK_BOX = RGBColor(0x1E, 0x29, 0x3B)  # тёмный блок и врезка
ON_DARK = RGBColor(0xCB, 0xD5, 0xE1)   # подпись на тёмном
GREY = RGBColor(0x47, 0x55, 0x69)
GREY_L = RGBColor(0x94, 0xA3, 0xB8)
BORDER = RGBColor(0xE2, 0xE8, 0xF0)
BORDER_G = RGBColor(0xCB, 0xD5, 0xE1)
PALE = RGBColor(0xF8, 0xFA, 0xFC)      # заливка карточек
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"

# --- цвета разделов большой колоды (решение 08.10.2026) ---------------------
# (основной, светлый для подписей на тёмном)
SECTION_ACCENTS = {
    "общая": (RGBColor(0x25, 0x63, 0xEB), RGBColor(0x60, 0xA5, 0xFA)),       # синий
    "бд": (RGBColor(0x0D, 0x94, 0x88), RGBColor(0x5E, 0xEA, 0xD4)),          # бирюзовый
    "бз": (RGBColor(0x7C, 0x3A, 0xED), RGBColor(0xC4, 0xB5, 0xFD)),          # фиолетовый
    "линеаменты": (RGBColor(0xEA, 0x58, 0x0C), RGBColor(0xFD, 0xBA, 0x74)),  # оранжевый
    "прогноз": (RGBColor(0x1E, 0x3A, 0x8A), RGBColor(0x93, 0xC5, 0xFD)),     # тёмно-синий
}

# --- сетка листа -----------------------------------------------------------
SLIDE_W = 10.0
SLIDE_H = 5.625
MARGIN = 0.50
CONTENT_W = 9.00
HEAD_RULE_Y = 0.62     # линия под верхним колонтитулом
BODY_Y = 1.88          # верх содержимого при заголовке с подзаголовком
NOTE_Y = 4.50          # верх врезки «ИТОГ»
FOOT_RULE_Y = 5.20
FOOTER = "ФГБУ ВНИГНИ · РАЗВИТИЕ МЕТОДОВ ИИ"

__all__ = [
    "Deck", "SECTION_ACCENTS", "rect", "line", "oval", "textbox", "run", "para", "caps", "hline",
    "node", "card", "metric", "bullets", "picture", "table", "chip",
    "Inches", "Pt", "RGBColor", "MSO_ANCHOR", "PP_ALIGN",
    "ACCENT", "ACCENT_D", "ACCENT_L", "PALE_BLUE", "ON_BLUE", "DARK", "TEXT",
    "DARK_BOX", "ON_DARK", "GREY", "GREY_L", "BORDER", "BORDER_G", "PALE",
    "WHITE", "MARGIN", "CONTENT_W", "BODY_Y", "NOTE_Y", "IMG", "DECK_DIR",
    "ROOT",
]


# --- примитивы -------------------------------------------------------------
def rect(sl, x, y, w, h, fill=None, border=None, lw=0.75):
    """Прямоугольник без тени: заливка и/или тонкая рамка."""
    sh = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                             Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if border is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = border
        sh.line.width = Pt(lw)
    sh.text_frame.word_wrap = True
    return sh


def line(sl, x1, y1, x2, y2, color=GREY_L, lw=1.0, arrow=True, dash=None):
    """Прямая линия (prstGeom line), при необходимости со стрелкой."""
    sh = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(min(x1, x2)),
                             Inches(min(y1, y2)),
                             Inches(abs(x2 - x1)), Inches(abs(y2 - y1)))
    sh.shadow.inherit = False
    sh._element.spPr.find(qn("a:prstGeom")).set("prst", "line")
    sh.fill.background()
    sh.line.color.rgb = color
    sh.line.width = Pt(lw)
    if dash:
        sh.line.dash_style = dash
    if arrow:
        ln = sh._element.spPr.find(qn("a:ln"))
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle"}))
    if x2 < x1 or y2 < y1:   # разворот рамки для линий справа налево / вверх
        xfrm = sh._element.spPr.find(qn("a:xfrm"))
        if x2 < x1:
            xfrm.set("flipH", "1")
        if y2 < y1:
            xfrm.set("flipV", "1")
    return sh


def oval(sl, x, y, d, fill=ACCENT, border=None):
    """Кружок-маркер (узел дорожки плана)."""
    sh = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y),
                             Inches(d), Inches(d))
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if border is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = border
    return sh


def hline(sl, x, y, w, color=BORDER, lw=0.75):
    """Тонкая линия-разделитель во всю ширину содержимого."""
    return line(sl, x, y, x + w, y, color=color, lw=lw, arrow=False)


def textbox(sl, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    sh = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    return tf


def run(par, text, size, color, bold=False, italic=False, spacing=None):
    r = par.add_run()
    r.text = text
    f = r.font
    f.name = FONT
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    if spacing:
        r._r.get_or_add_rPr().set("spc", str(int(spacing)))
    return r


def para(tf, text, size, color, bold=False, italic=False,
         align=PP_ALIGN.LEFT, first=False, space_before=None, spacing=None):
    """Абзац в текстовой рамке; `first` — занять пустой первый абзац."""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    run(p, text, size, color, bold=bold, italic=italic, spacing=spacing)
    return p


def caps(sl, x, y, w, text, color=GREY, size=8.5, align=PP_ALIGN.LEFT):
    """Надпись-рубрика: заглавные, разреженные, жирные."""
    tf = textbox(sl, x, y, w, 0.16)
    p = tf.paragraphs[0]
    p.alignment = align
    run(p, text, size, color, bold=True, spacing=100)
    return tf


def chip(sl, x, y, w, text, size=8.5):
    """Бледно-голубая метка-таблетка (ряд под подзаголовком титула)."""
    rect(sl, x, y, w, 0.22, PALE_BLUE)
    tf = textbox(sl, x, y, w, 0.22, MSO_ANCHOR.MIDDLE)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    run(tf.paragraphs[0], text, size, ACCENT_D, bold=True, spacing=60)
    return tf


# --- блоки -----------------------------------------------------------------
def node(sl, x, y, w, h, title, sub=None, fill=ACCENT, border=None,
         tcolor=WHITE, scolor=ON_BLUE, num=None, ncolor=None,
         align=PP_ALIGN.CENTER, tsize=8.5, ssize=7.5, pad=0.05):
    """Блок схемы: плашка с жирным заголовком и курсивным пояснением."""
    rect(sl, x, y, w, h, fill, border)
    tf = textbox(sl, x + pad, y, w - 2 * pad, h, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = align
    if num:
        run(p, num + "  ", tsize, ncolor or scolor, bold=True)
    run(p, title, tsize, tcolor, bold=True)
    if sub:
        ps = tf.add_paragraph()
        ps.alignment = align
        run(ps, sub, ssize, scolor, italic=True)
    return tf


def card(sl, x, y, w, h, title, body=None, items=None, tsize=10.5,
         bsize=9.0, fill=PALE, border=BORDER, pad=0.14):
    """Карточка: светлая плашка, жирный заголовок, текст или пункты."""
    rect(sl, x, y, w, h, fill, border)
    # высота заголовка считается по числу строк, иначе двухстрочный заголовок
    # наезжает на первый пункт
    cpl = max(int((w - 2 * pad) / (0.50 * tsize / 72.0)), 8)
    lines = max(1, -(-len(title) // cpl))
    th = lines * tsize / 72.0 * 1.28 + 0.10
    tf = textbox(sl, x + pad, y + pad, w - 2 * pad, th)
    run(tf.paragraphs[0], title, tsize, DARK, bold=True)
    top = y + pad + th
    if body:
        t2 = textbox(sl, x + pad, top, w - 2 * pad, h - (top - y) - pad)
        run(t2.paragraphs[0], body, bsize, GREY)
    if items:
        bullets(sl, x + pad, top, w - 2 * pad, h - (top - y) - pad, items,
                size=bsize)
    return tf


def metric(sl, x, y, w, h, value, label, vsize=26.0, lsize=8.0):
    """Карточка с крупным числом и подписью под ним.

    Высота строки с числом считается от кегля, иначе подпись наезжает на
    число (так это выглядит в исходной колоде серии).
    """
    rect(sl, x, y, w, h, PALE, BORDER)
    vh = vsize / 72.0 * 1.38
    tv = textbox(sl, x + 0.15, y + 0.07, w - 0.30, vh)
    run(tv.paragraphs[0], value, vsize, ACCENT, bold=True)
    top = y + 0.07 + vh + 0.04
    tl = textbox(sl, x + 0.15, top, w - 0.25, max(y + h - 0.05 - top, 0.14))
    run(tl.paragraphs[0], label, lsize, GREY)
    return tv


def bullets(sl, x, y, w, h, items, size=9.0, lead_color=DARK,
            text_color=TEXT, dot=ACCENT, space=3):
    """Пункты с синей точкой. Элемент — строка или пара (лид, остальное)."""
    tf = textbox(sl, x, y, w, h)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if i:
            p.space_before = Pt(space)
        run(p, "●  ", size * 0.67, dot)
        if isinstance(it, (tuple, list)):
            lead, rest = it
            if lead:
                run(p, lead, size, lead_color, bold=True)
            if rest:
                run(p, rest, size, text_color)
        else:
            run(p, it, size, text_color)
    return tf


def picture(sl, x, y, w, h, path, caption=None, sub=None, frame=True):
    """Картинка в тонкой рамке с подписью-капителью и пояснением под ней."""
    if frame:
        rect(sl, x - 0.02, y - 0.02, w + 0.04, h + 0.04, WHITE, BORDER)
    sl.shapes.add_picture(str(path), Inches(x), Inches(y),
                          Inches(w), Inches(h))
    if caption:
        caps(sl, x, y + h + 0.08, w, caption, GREY, 7.5)
    if sub:
        tf = textbox(sl, x, y + h + 0.25, w, 0.16)
        run(tf.paragraphs[0], sub, 8.0, GREY)
    return None


def table(sl, x, y, w, rows, widths, row_h=0.26, head_h=0.24, size=8.5,
          accent_row=None):
    """Простая таблица фигурами: тёмная шапка, чередование строк.

    `rows[0]` — шапка; `widths` — доли ширины, в сумме 1.
    `accent_row` — номер строки тела (с 0), выделяемой бледно-голубым.
    """
    cols = [w * f for f in widths]
    xs = [x]
    for c in cols[:-1]:
        xs.append(xs[-1] + c)
    rect(sl, x, y, w, head_h, DARK_BOX)
    for j, cell in enumerate(rows[0]):
        tf = textbox(sl, xs[j] + 0.08, y, cols[j] - 0.16, head_h,
                     MSO_ANCHOR.MIDDLE)
        tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        run(tf.paragraphs[0], cell, size - 0.5, WHITE, bold=True, spacing=40)
    yy = y + head_h
    for i, rowv in enumerate(rows[1:]):
        fill = PALE_BLUE if accent_row == i else (PALE if i % 2 == 0 else WHITE)
        rect(sl, x, yy, w, row_h, fill, BORDER)
        for j, cell in enumerate(rowv):
            tf = textbox(sl, xs[j] + 0.08, yy, cols[j] - 0.16, row_h,
                         MSO_ANCHOR.MIDDLE)
            tf.paragraphs[0].alignment = (PP_ALIGN.LEFT if j == 0
                                          else PP_ALIGN.CENTER)
            run(tf.paragraphs[0], cell, size,
                ACCENT_D if accent_row == i else TEXT,
                bold=(j == 0 or accent_row == i))
        yy += row_h
    return yy


# --- колода ----------------------------------------------------------------
class Deck:
    """Колода в светлом институтском стиле: холст 10 x 5.625", нумерация.

    `total` — сколько слайдов будет всего (для «02 / 06» в подвале).
    Титул — страница 01 без номера раздела; дальше разделы 01, 02, …
    """

    def __init__(self, total, footer=FOOTER):
        self.prs = Presentation()
        self.prs.slide_width = Inches(SLIDE_W)
        self.prs.slide_height = Inches(SLIDE_H)
        self._layout = self.prs.slide_layouts[6]
        self.total = total
        self.footer = footer
        self.page = 0
        self.section = 0
        self.accent = ACCENT
        self.accent_l = ACCENT_L

    def set_accent(self, accent, accent_light=None):
        """Цвет раздела: полоса сверху, номера, рубрики; дальше действует до смены.

        Для примитивов вне `Deck` (card, metric, node, chip) цвет передают явно
        из `deck.accent`.
        """
        self.accent = accent
        self.accent_l = accent_light or accent

    def divider(self, number, title, lead, facts=None):
        """Разделитель раздела: крупный номер, название, строка-лид, факты."""
        sl = self._blank()
        rect(sl, 0, 0, SLIDE_W, SLIDE_H - 0.50, DARK_BOX)
        rect(sl, 0.80, 1.55, 0.50, 0.05, self.accent_l)
        tf = textbox(sl, 0.80, 1.00, 2.00, 0.50)
        run(tf.paragraphs[0], number, 22.0, self.accent_l, bold=True)
        tf = textbox(sl, 0.80, 1.78, 8.40, 1.10)
        run(tf.paragraphs[0], title, 28.0, WHITE, bold=True)
        tf = textbox(sl, 0.80, 3.00, 8.00, 0.60)
        run(tf.paragraphs[0], lead, 12.0, ON_DARK)
        for k, (lab, val) in enumerate(facts or []):
            caps(sl, 0.80 + k * 2.9, 4.05, 2.7, lab, GREY_L, 8.0)
            tf = textbox(sl, 0.80 + k * 2.9, 4.22, 2.7, 0.26)
            run(tf.paragraphs[0], val, 12.0, WHITE, bold=True)
        return sl

    # -- служебное ----------------------------------------------------------
    def _blank(self):
        sl = self.prs.slides.add_slide(self._layout)
        rect(sl, 0, 0, SLIDE_W, 0.02, self.accent)      # верхняя акцентная полоса
        self.page += 1
        hline(sl, MARGIN, FOOT_RULE_Y, CONTENT_W)
        tf = textbox(sl, MARGIN, 5.30, 7.00, 0.15)
        run(tf.paragraphs[0], self.footer, 8.5, GREY, bold=True, spacing=100)
        tn = textbox(sl, 8.70, 5.30, 0.80, 0.15)
        tn.paragraphs[0].alignment = PP_ALIGN.RIGHT
        run(tn.paragraphs[0], "%02d / %02d" % (self.page, self.total), 8.0,
            self.accent)
        return sl

    # -- слайды -------------------------------------------------------------
    def title_slide(self, rubric, title, lead, tagline, chips, facts, date,
                    picture_path=None, pic_caption=None, pic_sub=None,
                    city="МОСКВА", org="ФГБУ ВНИГНИ",
                    org_full="Всероссийский научно-исследовательский "
                             "геологический нефтяной институт"):
        """Титул: шапка института, заголовок, лид, метки, строка фактов."""
        sl = self._blank()
        rect(sl, 0.50, 0.42, 0.06, 0.06, self.accent)
        tf = textbox(sl, 0.64, 0.37, 3.50, 0.18)
        run(tf.paragraphs[0], org, 8.5, self.accent, bold=True, spacing=100)
        tf = textbox(sl, 0.50, 0.57, 5.50, 0.16)
        run(tf.paragraphs[0], org_full, 7.0, GREY)

        caps(sl, 0.80, 1.28, 5.00, rubric, self.accent)
        tf = textbox(sl, 0.80, 1.44, 5.20, 1.44)
        run(tf.paragraphs[0], title, 25.0, DARK, bold=True)
        hline(sl, 0.80, 2.96, 0.25, GREY, lw=1.5)
        tf = textbox(sl, 0.80, 3.06, 5.20, 0.20)
        run(tf.paragraphs[0], lead, 10.0, GREY)
        tf = textbox(sl, 0.80, 3.30, 5.20, 0.32)
        run(tf.paragraphs[0], tagline, 9.0, GREY)

        xx = 0.80
        for text, wch in chips:
            chip(sl, xx, 3.70, wch, text)
            xx += wch + 0.07

        if picture_path:
            # картинка вписывается в окно 2.90 x 1.62" без искажения пропорций
            from PIL import Image
            with Image.open(str(picture_path)) as im:
                ar = im.width / im.height
            w, h = 2.90, 2.90 / ar
            if h > 1.62:
                h, w = 1.62, 1.62 * ar
            x = 6.30 + (2.90 - w) / 2
            rect(sl, x - 0.02, 1.38, w + 0.04, h + 0.04, PALE)
            sl.shapes.add_picture(str(picture_path), Inches(x), Inches(1.40),
                                  Inches(w), Inches(h))
            if pic_caption:
                caps(sl, 6.30, 1.40 + h + 0.12, 2.90, pic_caption, GREY, 7.5)
            if pic_sub:
                tf = textbox(sl, 6.30, 1.40 + h + 0.34, 2.90, 0.30)
                run(tf.paragraphs[0], pic_sub, 8.0, GREY)

        hline(sl, 0.80, 4.12, 8.40)
        xs = (0.80, 3.90)
        for (lab, val), x in zip(facts, xs):
            caps(sl, x, 4.25, 3.00, lab, GREY, 8.0)
            tf = textbox(sl, x, 4.44, 3.00, 0.20)
            run(tf.paragraphs[0], val, 10.0, DARK)
        caps(sl, 7.60, 4.25, 1.60, city, GREY, 8.0, PP_ALIGN.RIGHT)
        tf = textbox(sl, 7.60, 4.42, 1.60, 0.26)
        tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
        run(tf.paragraphs[0], date, 15.0, self.accent, bold=True)
        return sl

    def slide(self, section, right, title, subtitle=None, rubric=None):
        """Обычный слайд: колонтитул, рубрика, заголовок, подзаголовок."""
        sl = self._blank()
        self.section += 1
        tf = textbox(sl, MARGIN, 0.36, 0.22, 0.16)
        run(tf.paragraphs[0], "%02d" % self.section, 9.0, self.accent)
        rect(sl, 0.72, 0.42, 0.04, 0.04, self.accent)
        caps(sl, 0.82, 0.36, 3.00, section, GREY)
        caps(sl, 5.50, 0.36, 4.00, right, GREY_L, align=PP_ALIGN.RIGHT)
        hline(sl, MARGIN, HEAD_RULE_Y, CONTENT_W)

        caps(sl, MARGIN, 0.86, 5.00, rubric or section, self.accent)
        tf = textbox(sl, MARGIN, 1.08, CONTENT_W, 0.42)
        run(tf.paragraphs[0], title, 22.0, DARK, bold=True)
        if subtitle:
            tf = textbox(sl, MARGIN, 1.52, CONTENT_W, 0.20)
            run(tf.paragraphs[0], subtitle, 9.0, GREY)
        return sl

    def note(self, sl, text, label="ИТОГ", y=NOTE_Y, h=0.55, size=11.0):
        """Врезка-вывод: тёмная полоса с меткой слева и одной мыслью."""
        rect(sl, MARGIN, y, CONTENT_W, h, DARK_BOX)
        tf = textbox(sl, 0.70, y, 0.80, h, MSO_ANCHOR.MIDDLE)
        run(tf.paragraphs[0], label, 8.5, self.accent_l, bold=True, spacing=100)
        tf = textbox(sl, 1.45, y, 7.90, h, MSO_ANCHOR.MIDDLE)
        run(tf.paragraphs[0], text, size, WHITE)
        return sl

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.prs.save(str(path))
        except PermissionError:
            raise SystemExit(
                f"Файл занят (открыт в PowerPoint): {path}\n"
                "Закройте его и запустите сборку заново.")
        return path
