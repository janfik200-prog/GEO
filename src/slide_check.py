# -*- coding: utf-8 -*-
"""Проверка готовой презентации: вёрстка по данным PowerPoint и язык слайдов.

python-pptx записывает только рамки и строки — переносы строк делает
PowerPoint при показе, поэтому вылез ли текст из рамки, из файла не видно.
Здесь файл открывается в самом PowerPoint (COM, только чтение, без окна), и
для каждой надписи берутся реальные границы свёрстанного текста. По ним:

- текст за краем слайда или вплотную к краю (поле меньше 0.25 дюйма);
- текст вылез из своей рамки;
- текст наехал на другой текст, на картинку или на чужую плашку/строку
  таблицы (плашка, внутри которой стоит сама надпись, — не наезд);
- картинки наложены друг на друга;
- кегль меньше 10 пт;
- язык: слово «честный», «заказчик» вместо «организация», англоязычный
  жаргон (жёсткий список — ошибка; прочая строчная латиница вне кодов
  признаков — предупреждение, решает человек).

Выгрузка слайдов в PNG — только для визуальной сверки, во временную папку.
Сравнение `--compare` показывает, что в файле поправлено вручную относительно
свежей сборки скрипта (без PowerPoint, по самому файлу).

Запуск из корня:
    python -X utf8 -m src.slide_check "docs II presentation/X.pptx"
    python -X utf8 -m src.slide_check X.pptx --slides 3,5 --render <папка>
    python -X utf8 -m src.slide_check X.pptx --compare <свежая сборка>.pptx
Код возврата 1, если есть ошибки.
"""
import argparse
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

PT_PER_IN = 72.0
TOL = 1.5            # пт: допуск на округления вёрстки
EDGE = 0.25 * PT_PER_IN
MIN_FONT = 10.0

# Англоязычный жаргон, который в слайдах заменяется русским по смыслу
# (правило от 18.08.2026). AUC, OOF, CV, lift разрешены.
JARGON = [
    "permutation importance", "feature importance", "leave-one-out",
    "leave one out", "spatial cv", "buffer-cv", "buffer cv",
    "average precision", "held-out", "holdout", "hold-out", "baseline",
    "unsupervised", "supervised", "boruta", "pipeline", "dataset", "feature",
    "cross-validation", "overfitting", "train", "test split", "fold",
    "embedding", "cluster", "score", "recall", "precision", "threshold",
    "outlier", "ground truth", "benchmark", "workflow", "insight", "trade-off",
]
FORBIDDEN = [
    (re.compile(r"честн", re.I), "слово «честный» — заменить на «строгий», "
                                 "«объективный», «пространственно строгий»"),
    (re.compile(r"заказчик", re.I), "«заказчик» — во внешних документах "
                                    "«организация»"),
]
ALLOWED_LATIN = {"auc", "oof", "cv", "lift", "roc", "pr", "km", "nm", "mm",
                 "cm", "dem", "gis", "id"}


def _feature_tokens():
    """Части кодов признаков (ast, aloh, iron, ox...) — это не жаргон."""
    tokens = set()
    try:
        from src import feature_dict as fd
    except Exception:
        return tokens
    for name in dir(fd):
        val = getattr(fd, name)
        keys = val.keys() if isinstance(val, dict) else (
            val if isinstance(val, (set, list, tuple)) else ())
        for k in keys:
            if isinstance(k, str):
                tokens.update(t.lower() for t in re.split(r"[_\W\d]+", k) if t)
    return tokens


@dataclass
class Issue:
    slide: int
    level: str   # "ошибка" | "предупреждение"
    what: str

    def __str__(self):
        return "слайд %2d  %-14s %s" % (self.slide, self.level, self.what)


@dataclass
class Box:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    def overlap(self, o):
        """Площадь пересечения, пт² (с допуском по краям)."""
        w = min(self.right, o.right) - max(self.left, o.left) - TOL
        h = min(self.bottom, o.bottom) - max(self.top, o.top) - TOL
        return w * h if w > 0 and h > 0 else 0.0

    def contains_point(self, x, y):
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass
class Shape:
    name: str
    kind: str          # "text" | "picture" | "fill" | "table" | "line"
    frame: Box
    text_box: Box = None
    text: str = ""
    min_font: float = None


# ---------------------------------------------------------------- PowerPoint
@contextmanager
def powerpoint(pptx):
    """Открыть файл в PowerPoint только на чтение и без окна.

    Если PowerPoint уже был запущен пользователем, он не закрывается —
    закрывается только открытая здесь копия файла.
    """
    import pythoncom
    import win32com.client as win32
    pythoncom.CoInitialize()
    try:
        win32.GetActiveObject("PowerPoint.Application")
        was_running = True
    except Exception:
        was_running = False
    app = win32.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(str(Path(pptx).resolve()), True, False, False)
    try:
        yield pres
    finally:
        pres.Close()
        if not was_running and app.Presentations.Count == 0:
            app.Quit()


def _flatten(shapes):
    for sh in shapes:
        if sh.Type == 6:  # группа
            yield from _flatten(sh.GroupItems)
        else:
            yield sh


def _read_shape(sh, fonts):
    frame = Box(sh.Left, sh.Top, sh.Width, sh.Height)
    if sh.HasTable:
        cells = []
        tbl = sh.Table
        for i in range(1, tbl.Rows.Count + 1):
            for j in range(1, tbl.Columns.Count + 1):
                cells.append(tbl.Cell(i, j).Shape.TextFrame2.TextRange.Text)
        return Shape(sh.Name, "table", frame, text=" | ".join(cells))
    if sh.Type == 13 or (sh.Type == 14 and sh.PlaceholderFormat.ContainedType == 13):
        return Shape(sh.Name, "picture", frame)
    if sh.Connector or sh.Type == 9:
        return Shape(sh.Name, "line", frame)
    if sh.HasTextFrame and sh.TextFrame2.HasText:
        rng = sh.TextFrame2.TextRange
        return Shape(sh.Name, "text", frame,
                     Box(rng.BoundLeft, rng.BoundTop, rng.BoundWidth,
                         rng.BoundHeight),
                     rng.Text, fonts.get(sh.Id))
    filled = sh.Fill.Visible and sh.Fill.Transparency < 0.95
    return Shape(sh.Name, "fill" if filled else "line", frame)


def _min_fonts(pptx):
    """Наименьший явно заданный кегль по фигурам: {(слайд, id фигуры): пт}.

    Через COM фрагменты текста не перечисляются, поэтому кегль читается
    из файла python-pptx; фигуры сопоставляются по id.
    """
    from pptx import Presentation
    out = {}

    def walk(shapes, n):
        for sh in shapes:
            if sh.shape_type == 6:
                walk(sh.shapes, n)
            elif sh.has_text_frame:
                sizes = [r.font.size.pt for p in sh.text_frame.paragraphs
                         for r in p.runs if r.font.size and r.text.strip()]
                if sizes:
                    out[(n, sh.shape_id)] = min(sizes)

    for n, slide in enumerate(Presentation(str(pptx)).slides, 1):
        walk(slide.shapes, n)
    return out


def read_deck(pptx, slides=None):
    """Размер слайда (пт) и фигуры по слайдам: {номер: [Shape, ...]}."""
    out = {}
    fonts = _min_fonts(pptx)
    with powerpoint(pptx) as pres:
        size = (pres.PageSetup.SlideWidth, pres.PageSetup.SlideHeight)
        for i in range(1, pres.Slides.Count + 1):
            if slides and i not in slides:
                continue
            f = {sid: v for (n, sid), v in fonts.items() if n == i}
            out[i] = [_read_shape(sh, f)
                      for sh in _flatten(pres.Slides(i).Shapes)]
    return size, out


# ------------------------------------------------------------------ проверки
def _short(text, n=45):
    text = " ".join(text.split())
    return "«%s»" % (text if len(text) <= n else text[:n - 1] + "…")


def check_layout(size, shapes, slide_no):
    sw, sh = size
    issues = []
    add = lambda lvl, msg: issues.append(Issue(slide_no, lvl, msg))
    slide_box = Box(0, 0, sw, sh)
    texts = [s for s in shapes if s.kind == "text"]
    pics = [s for s in shapes if s.kind == "picture"]
    fills = [s for s in shapes if s.kind == "fill"]

    for s in shapes:
        f = s.frame
        if (f.left < -TOL or f.top < -TOL or f.right > sw + TOL
                or f.bottom > sh + TOL):
            add("ошибка", "фигура %s выходит за край слайда" % s.name)
        if s.kind == "picture" and not _full_bleed(f, slide_box):
            if min(f.left, f.top, sw - f.right, sh - f.bottom) < EDGE - TOL:
                add("предупреждение", "картинка %s ближе 0.25\" к краю слайда"
                    % s.name)

    for t in texts:
        tb, f = t.text_box, t.frame
        if tb.left < -TOL or tb.top < -TOL or tb.right > sw + TOL \
                or tb.bottom > sh + TOL:
            add("ошибка", "текст %s за краем слайда" % _short(t.text))
        elif tb.bottom > 76 and min(tb.left, sw - tb.right,
                                    sh - tb.bottom) < EDGE - TOL:
            add("предупреждение", "текст %s ближе 0.25\" к краю слайда"
                % _short(t.text))
        over = max(f.top - tb.top, tb.bottom - f.bottom,
                   f.left - tb.left, tb.right - f.right)
        if over > TOL:
            add("предупреждение", "текст %s вылез из своей рамки на %.0f пт"
                % (_short(t.text), over))
        if t.min_font and t.min_font < MIN_FONT:
            add("предупреждение", "кегль %.1f пт в %s — мелко для зала"
                % (t.min_font, _short(t.text)))

    cx = lambda b: (b.left + b.width / 2, b.top + b.height / 2)
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if a.text_box.overlap(b.text_box) > 0:
                add("ошибка", "текст %s наехал на текст %s"
                    % (_short(a.text, 30), _short(b.text, 30)))
        for p in pics:
            if a.text_box.overlap(p.frame) > 0:
                lvl = ("предупреждение" if p.frame.contains_point(*cx(a.frame))
                       else "ошибка")
                add(lvl, "текст %s наехал на картинку %s"
                    % (_short(a.text), p.name))
        for f in fills:
            # плашка, в которой стоит сама надпись, — её фон, не наезд
            if f.frame.contains_point(*cx(a.frame)):
                continue
            if a.text_box.overlap(f.frame) > 0:
                add("ошибка", "текст %s наехал на плашку %s"
                    % (_short(a.text), f.name))
    for i, a in enumerate(pics):
        for b in pics[i + 1:]:
            if a.frame.overlap(b.frame) > 0:
                add("предупреждение", "картинки %s и %s наложены"
                    % (a.name, b.name))
    return issues


def _full_bleed(box, slide_box):
    return box.width >= slide_box.width - TOL and box.height >= slide_box.height - TOL


def check_language(text, slide_no, feature_tokens):
    issues = []
    low = text.lower()
    for rx, msg in FORBIDDEN:
        if rx.search(text):
            issues.append(Issue(slide_no, "ошибка", msg))
    found = set()
    for term in JARGON:
        if re.search(r"(?<![a-z_])%s(?![a-z_])" % re.escape(term), low):
            found.add(term)
            issues.append(Issue(slide_no, "ошибка",
                                "англоязычный жаргон «%s» — заменить русским "
                                "по смыслу" % term))
    other = set()
    for m in re.finditer(r"(?<![\w\-])([a-z][a-z\-]{2,})(?![\w\-])", text):
        w = m.group(1)
        parts = w.split("-")
        if w in ALLOWED_LATIN or any(w in j for j in found):
            continue
        if all(p in feature_tokens or p in ALLOWED_LATIN for p in parts if p):
            continue
        other.add(w)
    if other:
        issues.append(Issue(slide_no, "предупреждение",
                            "латиница в тексте: %s — не жаргон ли?"
                            % ", ".join(sorted(other))))
    return issues


def check(pptx, slides=None):
    size, deck = read_deck(pptx, slides)
    tokens = _feature_tokens()
    issues = []
    for n, shapes in deck.items():
        issues += check_layout(size, shapes, n)
        text = "\n".join(s.text for s in shapes if s.text)
        issues += check_language(text, n, tokens)
    return issues


# ------------------------------------------------------------------ выгрузка
def render(pptx, outdir, slides=None, width=1600):
    """Слайды в PNG (ширина `width` пикселей); возвращает список путей."""
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    with powerpoint(pptx) as pres:
        w, h = pres.PageSetup.SlideWidth, pres.PageSetup.SlideHeight
        height = int(round(width * h / w))
        for i in range(1, pres.Slides.Count + 1):
            if slides and i not in slides:
                continue
            path = outdir / ("slide-%02d.png" % i)
            pres.Slides(i).Export(str(path), "PNG", width, height)
            paths.append(path)
    return paths


# ------------------------------------------------------------------ сравнение
def _signature(pptx):
    """Слайды как списки (вид, ключ, текст, рамка в дюймах) — по файлу, без COM.

    Ключ текста — сам текст, картинки — хеш содержимого; плашки без текста
    не сравниваются (фоны и строки таблиц дают только шум).
    """
    import hashlib
    from pptx import Presentation
    emu = 914400.0
    deck = []

    def walk(shapes, out):
        for sh in shapes:
            if sh.shape_type == 6:
                walk(sh.shapes, out)
                continue
            box = tuple(round((v or 0) / emu, 2) for v in
                        (sh.left, sh.top, sh.width, sh.height))
            if sh.shape_type == 13:
                key = hashlib.sha1(sh.image.blob).hexdigest()[:10]
                out.append(("picture", key, "", box))
            elif sh.has_text_frame and sh.text_frame.text.strip():
                text = " ".join(sh.text_frame.text.split())
                out.append(("text", text, text, box))

    for slide in Presentation(str(pptx)).slides:
        items = []
        walk(slide.shapes, items)
        deck.append(items)
    return deck


def _title(items):
    texts = [t for kind, _, t, box in items if kind == "text" and box[1] < 1.0]
    return _short(max(texts, key=len)) if texts else "(без заголовка)"


def _text_delta(old, new, limit=3):
    """Изменённые фрагменты текста: «было» -> «стало», с парой слов контекста."""
    from difflib import SequenceMatcher
    a, b = old.split(), new.split()
    parts = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b,
                                              autojunk=False).get_opcodes():
        if op == "equal":
            continue
        ctx = " ".join(a[max(0, i1 - 2):i1])
        was = " ".join(a[i1:i2]) or "∅"
        now = " ".join(b[j1:j2]) or "∅"
        parts.append("…%s [%s -> %s]" % (ctx, _short(was, 80), _short(now, 80)))
    more = len(parts) - limit
    return "; ".join(parts[:limit]) + (" (и ещё %d)" % more if more > 0 else "")


def compare(built, current, move_tol=0.03):
    """Чем файл `current` (правленный вручную) отличается от сборки `built`.

    Слайды сопоставляются по тексту (удалённые и вставленные вручную слайды
    сдвигают нумерацию), внутри слайда — надписи по тексту, картинки по
    содержимому. Возвращает строки отчёта; пустой список — отличий нет.
    """
    from difflib import SequenceMatcher
    a, b = _signature(built), _signature(current)
    sig = lambda items: " ".join(t for _, _, t, _ in items)[:400]
    sm = SequenceMatcher(None, [sig(x) for x in a], [sig(x) for x in b],
                         autojunk=False)
    pairs, out = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            pairs += list(zip(range(i1, i2), range(j1, j2)))
            continue
        for i in range(i1, i2):
            out.append("слайд сборки %d %s — в файле нет (удалён вручную)"
                       % (i + 1, _title(a[i])))
        for j in range(j1, j2):
            out.append("слайд файла %d %s — в сборке нет (добавлен вручную)"
                       % (j + 1, _title(b[j])))
    for i, j in pairs:
        where = ("слайд %d" % (j + 1) if i == j
                 else "слайд сборки %d = файла %d" % (i + 1, j + 1))
        rest = list(b[j])
        for kind, key, text, box in a[i]:
            hit = next((x for x in rest if x[0] == kind and x[1] == key), None)
            if hit is None and kind == "text":
                cand = [(SequenceMatcher(None, text, x[2]).ratio(), x)
                        for x in rest if x[0] == "text"]
                score, hit = max(cand, default=(0, None), key=lambda c: c[0])
                if hit is not None and score >= 0.6:
                    out.append("%s: текст %s изменён: %s"
                               % (where, _short(text, 30),
                                  _text_delta(text, hit[2])))
                else:
                    hit = None
            if hit is None:
                what = _short(text) if kind == "text" else "картинка"
                out.append("%s: %s — в файле нет" % (where, what))
                continue
            rest.remove(hit)
            d = [h - g for g, h in zip(box, hit[3])]
            if max(abs(v) for v in d) > move_tol:
                what = _short(text) if kind == "text" else "картинка"
                out.append("%s: %s сдвинут/изменён по размеру: "
                           "x%+.2f y%+.2f ш%+.2f в%+.2f дюйма"
                           % (where, what, *d))
        for kind, key, text, box in rest:
            what = _short(text) if kind == "text" else "картинка"
            out.append("%s: %s — добавлено вручную (x %.2f, y %.2f дюйма)"
                       % (where, what, box[0], box[1]))
    return out


def _parse_slides(arg):
    if not arg:
        return None
    out = set()
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pptx")
    ap.add_argument("--slides", help="номера слайдов: 3,5,7-9 (по умолчанию все)")
    ap.add_argument("--render", metavar="ПАПКА",
                    help="выгрузить слайды в PNG в эту папку")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--no-check", action="store_true",
                    help="только выгрузка, без проверок")
    ap.add_argument("--compare", metavar="СБОРКА",
                    help="сравнить файл со свежей сборкой скрипта: что "
                         "поправлено вручную")
    args = ap.parse_args(argv)
    slides = _parse_slides(args.slides)
    if args.compare:
        diff = compare(args.compare, args.pptx)
        for line in diff:
            print(line)
        print("итого: отличий %d" % len(diff))
        return 0
    errors = 0
    if not args.no_check:
        issues = check(args.pptx, slides)
        for it in issues:
            print(it)
        errors = sum(it.level == "ошибка" for it in issues)
        print("итого: ошибок %d, предупреждений %d"
              % (errors, len(issues) - errors))
    if args.render:
        for p in render(args.pptx, args.render, slides, args.width):
            print(p)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
