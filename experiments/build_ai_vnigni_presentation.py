# -*- coding: utf-8 -*-
"""Сборка презентации «ИИ во ВНИГНИ»: прогноз золото-урановых узлов по данным ДЗЗ.

Цель — короткий (5 слайдов) отчёт о том, что во ВНИГНИ развивают методы
искусственного интеллекта для прогноза рудных объектов. Колода неявно
отвечает на пять вопросов (что за проект, зачем, как идём, где мы сейчас,
что дальше); сами вопросы на слайдах не пишутся.

План — задача оркестратора от 28.09.2026 (текст задачи, файла плана нет).
Оформление — `src/slides.py` (тёмная шапка, золотая линейка, Calibri); на
каждом слайде схема фигурами или картинка.

Картинки в `outputs/11_презентации/ai_vnigni_*.png`:
* обзор листов ГГК-200 — обрезка `outputs/01_исходные_данные/ggk200_overview_map.png`
  через PIL, приглушённая до декоративного фона (мелкие служебные надписи
  исходной карты не должны читаться как информация); основной лист обведён
  заново золотой рамкой;
* пара карт «критериальный прогноз ВНИГНИ / итоговая карта листа» рисуется под
  слайд скриптом `experiments/plot_ai_vnigni_maps.py` (вызывается отсюда) по
  данным `data/processed/full_sheet_map_ядро5км.parquet`.

Правки 28.09.2026 (задача оркестратора): врезки слайдов 2 и 5, выравнивание
дорожек слайда 2, статусы и подписи узлов схемы слайда 3, числа совпадения с
поправкой на наклон и новая пара карт на слайде 4, шевроны без повтора в
карточках на слайде 5, приглушённый обзор листов на титуле.

Правки 28.09.2026, вторая волна (задача оркестратора): подача без
пессимистичных формулировок — подписи дорожек и врезка слайда 2, статус
«первые воспроизведены» на слайде 3, крупные числа 0.767 и 58% и итоговая
строка вместо ограничения на слайде 4, врезка слайда 5. Числа не менялись.

Запуск из корня: ``python -X utf8 -m experiments.build_ai_vnigni_presentation``.
"""
from PIL import Image, ImageDraw, ImageEnhance
from pptx.enum.shapes import MSO_SHAPE

from src.slides import (
    BLUE, CONTENT_W, DARK, GOLD, GREEN, GREY, IMG, LIGHT, MARGIN, NOTE_Y, PALE,
    WHITE, DECK_DIR, Deck, Inches, MSO_ANCHOR, PP_ALIGN, Pt, RGBColor, bullets,
    connector, note, oval, picture, rect, round_rect, set_run, textbox,
)

OUT = DECK_DIR / "Презентация-ИИ-во-ВНИГНИ-28.09.2026.pptx"
FIG_DIR = IMG / "11_презентации"

AHEAD = RGBColor(0xD9, 0xD9, 0xD6)  # «впереди» — светло-серый узел
GOLD_DARK = RGBColor(0xA8, 0x80, 0x20)

BLOCKS = {1: "ЗАДАЧА", 2: "ПОДХОД", 3: "ЭТАП", 4: "ПЛАН"}


# ------------------------------------------------------------- картинки
def crop_overview():
    """Обзор листов ГГК-200: только центральная и южная полосы листов, без осей.

    Верхний ряд отрезан: слева над ним пустое белое поле, на тёмном титуле
    оно выглядит дырой.

    Карта — декоративный фон: насыщенность снижена, сверху тёмная подложка
    (цвет DARK, непрозрачность 0.55), чтобы подписи листов, пометки и чужие
    рамки исходного рисунка не читались как информация. Основной лист
    обведён заново золотой рамкой (по месту красной рамки в обрезке).
    """
    src = IMG / "01_исходные_данные" / "ggk200_overview_map.png"
    dst = FIG_DIR / "ai_vnigni_ggk200_crop.png"
    im = Image.open(src).convert("RGB").crop((563, 457, 1839, 1336))
    im = ImageEnhance.Color(im).enhance(0.3)
    im = Image.blend(im, Image.new("RGB", im.size, (0x2B, 0x2B, 0x2B)), 0.55)
    ImageDraw.Draw(im).rectangle((436, 11, 858, 463),
                                 outline=(0xC8, 0x9B, 0x2C), width=7)
    im.save(dst)
    return dst


def draw_map_pair():
    """Пара карт под слайд — `experiments/plot_ai_vnigni_maps.py`."""
    from experiments.plot_ai_vnigni_maps import main as plot_maps
    return plot_maps()


# ------------------------------------------------------------- фигуры
def card(s, x, y, w, h, fill, text, size=14, color=WHITE, bold=True,
         sub=None, sub_size=11.5):
    """Карточка round_rect с текстом по центру; `sub` — вторая строка мельче."""
    sp = round_rect(s, x, y, w, h, fill, radius=0.12)
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.08))
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), text, size, color, bold=bold)
    if sub:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        set_run(p2.add_run(), sub, sub_size, color, italic=True)
    return sp


def input_card(s, x, y, w, h, stripe, title, body, size=13.5):
    """Карточка исходных данных: светлый фон, цветная полоса слева."""
    round_rect(s, x, y, w, h, LIGHT, radius=0.1)
    rect(s, x, y + Inches(0.12), Pt(5), h - Inches(0.24), stripe)
    tf = textbox(s, x + Inches(0.18), y, w - Inches(0.28), h, MSO_ANCHOR.MIDDLE)
    set_run(tf.paragraphs[0].add_run(), title, size + 0.5, stripe, bold=True)
    p = tf.add_paragraph()
    p.space_before = Pt(3)
    set_run(p.add_run(), body, size, DARK)


def arrow(s, x1, y1, x2, y2, color=GREY, weight=2.0):
    return connector(s, x1, y1, x2, y2, color, weight=weight)


def lane_label(s, x, y, text, color):
    tf = textbox(s, x, y, Inches(8), Inches(0.35))
    set_run(tf.paragraphs[0].add_run(), text, 12, color, bold=True)


def caption(s, x, y, w, text, size=12.5, align=PP_ALIGN.LEFT):
    tf = textbox(s, x, y, w, Inches(0.4))
    tf.paragraphs[0].alignment = align
    set_run(tf.paragraphs[0].add_run(), text, size, GREY, italic=True)


def chevron(s, x, y, w, h, fill, text, first=False):
    shape = MSO_SHAPE.PENTAGON if first else MSO_SHAPE.CHEVRON
    sp = s.shapes.add_shape(shape, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    sp.line.fill.background()
    sp.shadow.inherit = False
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    set_run(tf.paragraphs[0].add_run(), text, 15, WHITE, bold=True)
    return sp


# ============================================================== сборка
deck = Deck(BLOCKS)
FIG_DIR.mkdir(parents=True, exist_ok=True)
overview = crop_overview()
map_pair = draw_map_pair()

# --- Титул: собственная раскладка — текст слева, обзор листов справа -----
s = deck.blank()
rect(s, 0, 0, deck.sw, deck.sh, DARK)
tf = textbox(s, Inches(0.8), Inches(1.0), Inches(6.2), Inches(2.45),
             MSO_ANCHOR.BOTTOM)
set_run(tf.paragraphs[0].add_run(),
        "Прогноз золото-урановых узлов по данным дистанционного зондирования",
        32, WHITE, bold=True)
rect(s, Inches(0.8), Inches(3.62), Inches(6.2), Pt(5), GOLD)
tf = textbox(s, Inches(0.8), Inches(3.9), Inches(6.2), Inches(3.0))
rows = [
    ("ВНИГНИ · развитие методов искусственного интеллекта в прогнозе "
     "рудных объектов", 16, GOLD, True),
    ("Опытная территория — Анабарский щит, лист R-48-XI,XII", 15, WHITE, False),
    ("Эталон сравнения — критериальный метод ВНИГНИ", 15, WHITE, False),
    ("28 сентября 2026", 13, PALE, False),
]
for i, (text, size, color, bold) in enumerate(rows):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(12)
    set_run(p.add_run(), text, size, color, bold=bold)
# Картинка: обзор листов ГГК-200, обрезка без осей (crop_overview)
pic_bottom = picture(s, overview, Inches(7.45), Inches(1.55), Inches(5.4),
                     Inches(4.2))
rect(s, Inches(7.45), pic_bottom + Inches(0.08), Inches(5.4), Pt(3), GOLD)
tf = textbox(s, Inches(7.45), pic_bottom + Inches(0.18), Inches(5.4),
             Inches(0.6))
tf.paragraphs[0].alignment = PP_ALIGN.CENTER
set_run(tf.paragraphs[0].add_run(),
        "Анабарский щит: основной лист R-48-XI,XII (золотая рамка) "
        "и смежные листы ГГК-200", 11.5, PALE, italic=True)

# --- 1. Зачем нужен новый метод: две дорожки фигурами -------------------
s, top = deck.slide("Зачем нужен новый метод", 1)
MAP_X, MAP_W = Inches(10.05), Inches(2.7)
BH = Inches(1.3)
# Общая сетка обеих дорожек: входные данные | два средних блока | карта.
IN_W = Inches(3.7)                     # левые плашки одной ширины
C1_X, C1_W = Inches(4.8), Inches(2.2)
C2_X, C2_W = Inches(7.5), Inches(2.05)

# Верхняя дорожка — эталон (BLUE)
y1 = Inches(1.85)
lane_label(s, MARGIN, Inches(1.4), "КРИТЕРИАЛЬНЫЙ МЕТОД ВНИГНИ — ЭТАЛОН", BLUE)
input_card(s, MARGIN, y1, IN_W, BH, BLUE, "Геологические карты",
           "разломы двух систем, фации, палеоструктуры, коры выветривания, "
           "магматизм")
# формула занимает оба средних столбца сетки
card(s, C1_X, y1, C2_X + C2_W - C1_X, BH, BLUE, "Критериальная формула", 16)
card(s, MAP_X, y1, MAP_W, BH, BLUE, "Карта перспективности", 16)
ym = y1 + BH // 2
arrow(s, MARGIN + IN_W, ym, C1_X, ym)
arrow(s, C2_X + C2_W, ym, MAP_X, ym)
caption(s, MARGIN, y1 + BH + Inches(0.08), CONTENT_W,
        "Опирается на подробную геологическую основу листа")

# Нижняя дорожка — новый метод (GOLD, итог GREEN)
y2 = Inches(4.2)
lane_label(s, MARGIN, Inches(3.75), "НОВЫЙ МЕТОД", GOLD_DARK)
input_card(s, MARGIN, y2, IN_W, BH, GOLD_DARK, "Рельеф и спутниковая съёмка",
           "Sentinel-2, ASTER, Landsat, радар")
card(s, C1_X, y2, C1_W, BH, GOLD, "Признаковое пространство", 15)
card(s, C2_X, y2, C2_W, BH, GOLD, "Модель машинного обучения", 15)
card(s, MAP_X, y2, MAP_W, BH, GREEN, "Карта перспективности", 16)
ym = y2 + BH // 2
arrow(s, MARGIN + IN_W, ym, C1_X, ym)
arrow(s, C1_X + C1_W, ym, C2_X, ym)
arrow(s, C2_X + C2_W, ym, MAP_X, ym)
caption(s, MARGIN, y2 + BH + Inches(0.08), CONTENT_W,
        "Открытые данные есть на любую территорию")

note(s, MARGIN, NOTE_Y, CONTENT_W,
     "Метод помогает сужать площадь поиска на любой территории "
     "и служит опорой для эксперта.", color=GREEN)

# --- 2. Как построена работа: технологическая схема с ветвлениями --------
s, top = deck.slide("Как построена работа", 2)
# Статусы — по задаче оркестратора. «Слияние» и «Выбор лучшей модели» в
# задаче без статуса; помечены «в работе»: агентные ветки ещё идут, а в ветке
# признакового пространства рабочая точка уже выбрана, в остальных нет.
# 28.09.2026: сбор разделён — сбор по листу готов, агентный сбор на несколько
# листов в работе; «Методы из статей» — в работе (первые воспроизведены).
DONE, WORK = GREEN, GOLD
# Легенда
lx = Inches(8.35)
for i, (col, txt, tcol) in enumerate([(DONE, "готово", DARK),
                                      (WORK, "в работе", DARK),
                                      (AHEAD, "впереди", DARK)]):
    x = lx + Inches(1.5) * i
    round_rect(s, x, Inches(1.42), Inches(0.32), Inches(0.24), col, radius=0.2)
    tf = textbox(s, x + Inches(0.38), Inches(1.32), Inches(1.1), Inches(0.42),
                 MSO_ANCHOR.MIDDLE)
    set_run(tf.paragraphs[0].add_run(), txt, 12, tcol)

CY = Inches(4.08)          # центр схемы: поровну пусто сверху и снизу
SH_ = Inches(1.9)          # высота одиночного узла
GAP = Inches(0.28)
widths = [1.5, 1.75, 1.3, 1.7, 1.45, 1.15, 1.65]
xs = []
x = MARGIN
for w in widths:
    xs.append(x)
    x += Inches(w) + GAP
W = [Inches(w) for w in widths]
FS = 12.5


def single(i, fill, text, color=WHITE, sub=None):
    card(s, xs[i], CY - SH_ // 2, W[i], SH_, fill, text, FS, color, sub=sub,
         sub_size=10.5)


def stack(i, items):
    """Параллельные узлы столбца i; возвращает их вертикальные центры."""
    n = len(items)
    h = Inches(1.3) if n == 2 else Inches(1.22)
    g = Inches(0.4) if n == 2 else Inches(0.2)
    total = h * n + g * (n - 1)
    y = CY - total // 2
    centers = []
    for fill, text, color, sub in items:
        card(s, xs[i], y, W[i], h, fill, text, FS, color, sub=sub,
             sub_size=10.5)
        centers.append(y + h // 2)
        y += h + g
    return centers


single(0, DONE, "Эталон: воссоздание критериального метода")
c2 = stack(1, [(DONE, "Сбор данных по листу", WHITE, None),
               (WORK, "Агентный сбор и паспортизация на несколько листов",
                WHITE, "ИИ-агенты"),
               (WORK, "База знаний по мировым публикациям", WHITE,
                "сбор ИИ-агентами")])
single(2, WORK, "Слияние: выбор направлений обучения")
c4 = stack(3, [(DONE, "Признаковое пространство", WHITE, None),
               (WORK, "Методы из статей на наших данных", WHITE,
                "первые воспроизведены"),
               (DONE, "Нейросети", WHITE, "опробованы")])
single(4, WORK, "Выбор лучшей модели в каждой ветке")
single(5, AHEAD, "Заверка геологами", DARK)
single(6, AHEAD, "Районирование: зоны и узлы-кандидаты", DARK)


def right(i):
    return xs[i] + W[i]


for c in c2:
    arrow(s, right(0), CY, xs[1], c)
    arrow(s, right(1), c, xs[2], CY)
for c in c4:
    arrow(s, right(2), CY, xs[3], c)
    arrow(s, right(3), c, xs[4], CY)
for i in (4, 5):
    arrow(s, right(i), CY, xs[i + 1], CY)

note(s, MARGIN, NOTE_Y, CONTENT_W,
     "Работа идёт ветками: в каждом узле пробуем несколько подходов "
     "и оставляем лучший.")

# --- 3. Где мы сейчас: две карты слева, статус и числа справа -----------
s, top = deck.slide("Где мы сейчас", 3)
MAP_BOX_W, MAP_BOX_H = Inches(7.2), Inches(4.68)
# Картинка: experiments/plot_ai_vnigni_maps.py (данные
# data/processed/full_sheet_map_ядро5км.parquet, ядро 5 x 1.25 км, порог
# верхних 10%, обучение на всём листе; 73 точки опробования, 19 независимых)
pb = picture(s, map_pair, MARGIN, Inches(1.3), MAP_BOX_W, MAP_BOX_H)
tf = textbox(s, MARGIN, pb + Inches(0.04), MAP_BOX_W, Inches(0.5))
tf.paragraphs[0].alignment = PP_ALIGN.CENTER
set_run(tf.paragraphs[0].add_run(),
        "Слева критериальный прогноз ВНИГНИ, справа итоговая карта листа "
        "по 37 переносимым дистанционным признакам", 11, GREY, italic=True)

# Статус справа.
# 0.984 — outputs/metrics/criterial_reconstruct_multisheet.log, строка
#   «наше воссоздание vs нативный prognoz: ранговая +0.9842 (нормировка по
#   листу)»; prognoz — расчёт ГИС Интегро на листе.
# 58 и 37 — data/processed/honest_feature_set.json, поля «признаки» (58) и
#   «признаки_перенос» (37 из 58).
sx = Inches(8.1)
RW = Inches(12.78) - sx
rect(s, sx - Inches(0.22), Inches(1.4), Pt(2), Inches(5.45), PALE)
bullets(s, sx, Inches(1.25), RW, Inches(2.95), [
    ("Критериальный метод воссоздан на Python", ": ранговая корреляция "
     "с расчётом ГИС Интегро на листе 0.984"),
    ("Собрано признаковое пространство", ": 58 признаков на листе, из них "
     "37 переносимы на смежные листы"),
    ("Построены карта листа и карта переноса", ""),
    ("Разворачивается база данных с агентным сбором", ": несколько листов, паспорта "
     "слоёв и признаков"),
], size=14)

# Два числа совпадения верхних 10% площади с картой ВНИГНИ, крупно — с
# поправкой на наклон (одна шкала).
# 0.767 и 0.548 — outputs/metrics/full_sheet_map.csv, строка «5 x 1.25,
#   порог верхних 10%, обучение на всём листе» (совпадение_верх_10 0.7669,
#   совпадение_без_тренда 0.5478); то же в docstring
#   experiments/elevation_detrend.py.
# 0.362 — outputs/metrics/full_sheet_map.csv, строка «5 x 1.25, порог верхних
#   10%, обучение по полосам с разрывом 15 км» (совпадение_без_тренда 0.3623);
#   0.362 и 0.622 — outputs/metrics/ceiling_decomposition.csv (строки «только
#   дистанционка (рабочая точка)» 0.3623 и «идеальная карта через конвейер
#   (потолок метрики)» 0.6220, столбец совпадение_без_тренда);
#   58% = 0.3623/0.6220 — outputs/metrics/ceiling_decomposition.csv (те же
#   строки), расчёт в experiments/ceiling_decomposition.py, раздел РЕЗУЛЬТАТ.
# 28.09.2026: подача без оговорок — крупно 0.767 и 58%, числа не менялись.
ny = Inches(4.3)
tf = textbox(s, sx, ny, RW, Inches(0.55))
set_run(tf.paragraphs[0].add_run(),
        "Совпадение верхних 10% площади с картой ВНИГНИ",
        12.5, DARK, bold=True)
ch = Inches(0.8)
cy = ny + Inches(0.6)
for k, (num, col, head, body) in enumerate([
        ("0.767", BLUE, "Итоговая карта листа",
         "с поправкой на региональный наклон — 0.548"),
        ("58%", GREEN, "Проверка на незнакомых участках",
         "от максимально достижимого (0.362 при потолке 0.622)")]):
    y = cy + (ch + Inches(0.1)) * k
    round_rect(s, sx, y, RW, ch, LIGHT, radius=0.08)
    rect(s, sx, y + Inches(0.1), Pt(5), ch - Inches(0.2), col)
    tfn = textbox(s, sx + Inches(0.15), y, Inches(1.2), ch, MSO_ANCHOR.MIDDLE)
    set_run(tfn.paragraphs[0].add_run(), num, 28, col, bold=True)
    tfb = textbox(s, sx + Inches(1.35), y, RW - Inches(1.45), ch,
                  MSO_ANCHOR.MIDDLE)
    set_run(tfb.paragraphs[0].add_run(), head, 13, DARK, bold=True)
    p = tfb.add_paragraph()
    p.space_before = Pt(2)
    set_run(p.add_run(), body, 11.5, DARK)
# Итог блока чисел — одной строкой.
tf = textbox(s, sx, cy + ch * 2 + Inches(0.18), RW, Inches(0.72))
set_run(tf.paragraphs[0].add_run(),
        "Модель воспроизводит экспертную карту ВНИГНИ только по открытым "
        "дистанционным данным.",
        11, GREY, italic=True)

# --- 4. Что дальше: дорожная карта шевронами ---------------------------
s, top = deck.slide("Что дальше", 4)
# В шевронах — этапы, в карточках — что получится (без повтора заголовка).
steps = [
    ("База данных и база знаний",
     "Агентный сбор по нескольким листам: слои и признаки с паспортами, "
     "публикации со ссылками"),
    ("Перенос на смежные листы",
     "Прогнозные карты смежных листов и сверка с критериальным расчётом "
     "ВНИГНИ по нескольким листам"),
    ("Заверка геологами",
     "Экспертное заключение: какие зоны и признаки модели геологически "
     "обоснованы"),
    ("Районирование",
     "Итоговая карта: зоны перспективности и золото-урановые "
     "узлы-кандидаты"),
]
n = len(steps)
ov = Inches(0.25)                       # наложение шевронов
cw_ = (CONTENT_W + ov * (n - 1)) // n
fills = [GOLD, GOLD_DARK, GOLD, GOLD_DARK]
for i, (head, body) in enumerate(steps):
    x = MARGIN + (cw_ - ov) * i
    chevron(s, x, Inches(1.75), cw_, Inches(1.15), fills[i], head, first=i == 0)
    # карточка под шевроном: результат этапа
    kg = Inches(0.25)
    kw = (CONTENT_W - kg * (n - 1)) // n
    kx = MARGIN + (kw + kg) * i
    ky = Inches(3.3)
    round_rect(s, kx, ky, kw, Inches(2.45), LIGHT, radius=0.08)
    d = Inches(0.6)
    o = oval(s, kx + kw // 2 - d // 2, ky + Inches(0.3), d, WHITE)
    o.line.color.rgb = fills[i]
    o.line.width = Pt(2)
    tfo = o.text_frame
    tfo.margin_left = tfo.margin_right = 0
    tfo.vertical_anchor = MSO_ANCHOR.MIDDLE
    tfo.paragraphs[0].alignment = PP_ALIGN.CENTER
    set_run(tfo.paragraphs[0].add_run(), str(i + 1), 18, fills[i], bold=True)
    tf = textbox(s, kx + Inches(0.15), ky + Inches(1.05), kw - Inches(0.3),
                 Inches(1.4))
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    set_run(tf.paragraphs[0].add_run(), body, 14.5, DARK)

note(s, MARGIN, NOTE_Y, CONTENT_W,
     "Метод работает на открытых дистанционных данных и не требует "
     "дополнительных полевых съёмок.", color=GOLD)

deck.save(OUT)
