# -*- coding: utf-8 -*-
"""Отдельная презентация «Агентная база данных» (узлы 3-4 технологической схемы).

Выделена из большой колоды по решению пользователя 09.10.2026. Слайды — модуль
`experiments/report_sections/db.py` (числа читаются из живой базы gis_au_u, роль gis_read;
скриншоты — `experiments/make_db_screens.py`). Стиль — институтский (`src/slides_inst.py`),
бирюзовый акцент раздела.

Требуется запущенная база: F:/GIS-DB/start_db.bat (порт 5433).
Запуск:   python -X utf8 -m experiments.build_db_presentation
Проверка: python -X utf8 -m src.slide_check "<файл>.pptx"  (на холсте 10" предупреждения о
          кегле <10 пт и близости к краю — норма стиля; смотреть PNG)
Выход:    docs II presentation/Презентация-Агентная-база-данных-09.10.2026.pptx
"""
import os

from src.slides_inst import DECK_DIR, SECTION_ACCENTS, Deck
from experiments.report_sections import db as sec_db

OUT = DECK_DIR / "Презентация-Агентная-база-данных-09.10.2026.pptx"
DB_FIGS = sec_db.DB_FIGS


def main():
    d = sec_db.live()
    deck = Deck(total=sec_db.COUNT + 1)
    deck.set_accent(*SECTION_ACCENTS["бд"])
    deck.title_slide(
        rubric="АНАБАРСКИЙ ЩИТ · ЛИСТ R-48-XI, XII",
        title="Агентная база данных проекта",
        lead="Сбор данных, паспорта признаков и происхождение каждого числа",
        tagline="Узлы 3 и 4 технологической схемы",
        chips=[("УЗЕЛ 3 · СБОР", 1.1), ("УЗЕЛ 4 · ПАСПОРТИЗАЦИЯ", 1.8)],
        facts=[("В БАЗЕ", f"{sec_db._sp(d['cells'])} ячеек, {d['features']} признака"),
               ("ПАСПОРТОВ", f"{d['fp'] + d['lp']} подтверждено оператором")],
        # ручные правки 09.10.2026: схему с титула убрали, город — Дубна
        date="10.2026", city="Дубна")
    start = deck.page
    sec_db.build(deck)
    assert deck.page - start == sec_db.COUNT, (deck.page - start, sec_db.COUNT)
    path = deck.save(os.environ.get("DECK_OUT") or OUT)
    print(path, deck.page, "слайдов")


if __name__ == "__main__":
    main()
