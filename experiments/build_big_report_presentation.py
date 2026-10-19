# -*- coding: utf-8 -*-
"""Большая колода-отчёт по мини-проектам (08.10.2026): общая часть, база данных,
база знаний, линеаменты, прогнозирование.

Цель: переработанная и расширенная колода по технологической схеме, в институтском
стиле (`src/slides_inst.py`): общий каркас, свой акцентный цвет у каждого раздела.
План: диалог 08.10.2026; план раздела «Прогнозирование» —
`docs II presentation/План-колоды-Прогнозирование.md`.
Состав разделов — модули `experiments/report_sections/*.py` (у каждого COUNT и build(deck)).

Запуск:   python -X utf8 -m experiments.build_big_report_presentation
Проверка: python -X utf8 -m src.slide_check "<файл>.pptx"   (на холсте 10" предупреждения
          о кегле <10 пт и близости к краю — норма стиля; смотреть PNG)
Требуется запущенная база (F:\\GIS-DB\\start_db.bat) для раздела «База данных».
"""
import importlib
import os

from src.slides_inst import DECK_DIR, SECTION_ACCENTS, Deck
from experiments.report_sections import db as sec_db

OUT = DECK_DIR / "Презентация-Отчёт-по-мини-проектам-08.10.2026.pptx"

# (ключ цвета, модуль, разделитель или None)
PLAN = [
    ("общая", "general", None),
    ("бд", "db", ("03–04", "База данных",
                  "Агентный сбор данных и паспортизация: одна база, где у каждого числа есть происхождение и подтверждение оператора")),
    ("бз", "kb", ("05–06", "База знаний и граф",
                  "Агентный сбор мировых публикаций и граф связей «признак — геологический механизм — метод»")),
    ("линеаменты", "lineaments", ("ЛИН.", "Линеаменты", "Отдельный мини-проект: материал раздела будет добавлен")),
    ("прогноз", "forecast", ("07–11", "Прогнозирование",
                             "Метод из литературы, поиск по аналогам, сравнение с критериальной картой, перенос и рабочая точка")),
]


def main():
    mods = []
    for key, name, div in PLAN:
        try:
            mods.append((key, importlib.import_module(f"experiments.report_sections.{name}"), div))
        except ModuleNotFoundError:
            print("нет модуля", name, "— раздел пропущен")
    total = sum(m.COUNT + (1 if d else 0) for _, m, d in mods)
    deck = Deck(total=total)
    for key, m, div in mods:
        deck.set_accent(*SECTION_ACCENTS[key])
        if div:
            deck.divider(*div, facts=_facts(m))
        start = deck.page
        m.build(deck)
        assert deck.page - start == m.COUNT, (m.__name__, deck.page - start, m.COUNT)
    assert deck.page == total, (deck.page, total)
    path = deck.save(os.environ.get("DECK_OUT") or OUT)
    print(path, total, "слайдов")


def _facts(m):
    if m.__name__.endswith(".db"):
        d = sec_db.live()
        return [("ЯЧЕЕК СЕТКИ", sec_db._sp(d["cells"])), ("ПРИЗНАКОВ", str(d["features"])),
                ("ПАСПОРТОВ", str(d["fp"] + d["lp"]))]
    return None


if __name__ == "__main__":
    main()
