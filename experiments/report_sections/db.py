# -*- coding: utf-8 -*-
"""Раздел «База данных» (узлы 3-4: агентная сборка и паспортизация) большой колоды.

Числа — из живой базы gis_au_u (роль gis_read, порт 5433) на момент сборки; скриншоты —
`experiments/make_db_screens.py` (без названий моделей и журнала решений). Версии стека —
из `GIS-agent-db/db/README.md` и установленных пакетов (проверено 09.10.2026).
Про подтверждение паспортов пишем «подтверждены оператором»: очереди 30.09.2026 закрыты
пакетом, слово «проверены» не используем.
Запуск самопроверки: python -X utf8 -m experiments.report_sections.db
"""
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image
from pptx.dml.color import RGBColor

from src.slides_inst import (
    SECTION_ACCENTS, Deck, ROOT, bullets, line, metric, node, picture,
    table, PALE, PALE_BLUE, GREY, DARK,
)

COUNT = 11
SCREENS = ROOT / "outputs" / "db_screens"
DB_FIGS = Path(r"C:\Users\gstat\GIS-agent-db\outputs\рисунки")
DB_REPO = Path(r"C:\Users\gstat\GIS-agent-db")

ORANGE = RGBColor(0xFE, 0xE4, 0xD3)    # решает оператор
ORANGE_D = RGBColor(0xC2, 0x5A, 0x14)


def _fit(path, max_w, max_h):
    with Image.open(str(path)) as im:
        ar = im.width / im.height
    w, h = max_w, max_w / ar
    if h > max_h:
        h, w = max_h, max_h * ar
    return w, h


def _sp(n):
    return f"{n:,}".replace(",", "\u00a0")


def live():
    """Числа из базы; запрос только на чтение."""
    sys.path.insert(0, str(DB_REPO))
    from gisdb import db
    cur = db.connect("gis_read").cursor()

    def one(sql):
        cur.execute(sql)
        return cur.fetchone()[0]

    d = {
        "cells": one("select count(*) from data.cell"),
        "features": one("select count(*) from data.feature"),
        "values": one("select count(*) from data.feature_value"),
        "layers": one("select count(*) from data.source_layer"),
        "terr": one("select count(*) from data.territory"),
        "deriv": one("select count(*) from meta.derivation"),
        "ops": one("select count(*) from meta.operation"),
        "runs": one("select count(*) from runs.run"),
        "metrics": one("select count(*) from runs.metric"),
        "fp": one("select count(*) from meta.feature_passport where status='подтверждён'"),
        "lp": one("select count(*) from meta.layer_passport where status='подтверждён'"),
        "links": one("select count(*) from bridge.feature_concept"),
    }
    cur.execute("select title, holder, access from meta.gap order by priority, id")
    d["gap_rows"] = cur.fetchall()
    return d


def build(deck):
    d = live()
    A = deck.accent
    S, R = "УЗЛЫ 3–4 · БАЗА ДАННЫХ", "АГЕНТНАЯ БАЗА ДАННЫХ"
    plain = dict(fill=PALE, border=GREY, tcolor=DARK, scolor=GREY)
    agent = dict(fill=PALE_BLUE, border=A, tcolor=DARK, scolor=GREY)
    human = dict(fill=ORANGE, border=ORANGE_D, tcolor=DARK, scolor=GREY)
    mil = f"{d['values'] / 1e6:.1f} млн"

    # 1. Зачем база. Числа: live().
    sl = deck.slide(S, R, "Зачем нужна база",
                    "Раньше у каждого эксперимента был свой файл и свои пометки, теперь всё собрано вместе")
    bullets(sl, 0.50, 1.95, 5.3, 3.0, [
        ("", "В базе лежат слои-источники, признаки и их значения по ячейкам сетки 500 м. Рядом с каждым признаком его паспорт: откуда взят, где покрывает территорию, что значит геологически."),
        ("", "Территорий несколько: основной лист и смежная область. Новый лист добавляется без переделки базы."),
        ("", "Черновики паспортов пишет агент. Статус «подтверждён» может поставить только оператор, база не принимает его ни от кого другого."),
        ("", "Про каждое число известно, из какого файла и каким скриптом оно получено."),
    ], size=10.5, dot=A, space=8)
    for k, (v, lab) in enumerate([
        (_sp(d["cells"]), "ячеек сетки в двух территориях"),
        (str(d["features"]), "признаков с паспортами"),
        (mil, "значений признаков"),
    ]):
        metric(sl, 6.25, 1.95 + k * 1.02, 3.25, 0.92, v, lab, vsize=20, lsize=8.5)

    # 2. Схема работы.
    sl = deck.slide(S, R, "Как работает агентная база",
                    "Синее делает агент, оранжевое решает оператор, серое — таблицы и источники базы")
    y1, y2, h, w = 2.05, 3.55, 0.9, 1.85
    xs = [0.50, 2.55, 4.60, 6.65]
    node(sl, xs[0], y1, w, h, "Источники", "открытые данные, материалы организации, публикации", **plain)
    node(sl, xs[1], y1, w, h, "Агент-сборщик", "скачивает, считает хеш, определяет контур съёмки", **agent)
    node(sl, xs[2], y1, w, h, "Паспорт слоя", "источник, лицензия, контур, хеш", **plain)
    node(sl, xs[3], y1, 2.85, h, "Пересчёт в ячейки сетки 500 м", "одна система координат, версия признака", **agent)
    node(sl, xs[0], y2, w, h, "Признаки и значения",
         f"{d['features']} признака, {d['values'] / 1e6:.1f} млн значений", **plain)
    node(sl, xs[1], y2, w, h, "Паспорт признака", "покрытие, переносимость, геологический смысл", **plain)
    node(sl, xs[2], y2, w, h, "Очередь подтверждений", "черновики ждут решения", **human)
    node(sl, xs[3], y2, 2.85, h, "Оператор", "удаление, лицензия, геологический смысл", **human)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        line(sl, xs[a] + w, y1 + h / 2, xs[b], y1 + h / 2, color=GREY)
    line(sl, xs[3] + 1.4, y1 + h, xs[3] + 1.4, y2, color=GREY)
    for a, b in [(3, 2), (2, 1), (1, 0)]:
        line(sl, xs[a], y2 + h / 2, xs[b] + w, y2 + h / 2, color=GREY)

    # 3. Состав: скрин + схемы базы.
    sl = deck.slide(S, R, "Из чего состоит база",
                    "Данные, паспорта, журналы и справочники лежат в отдельных схемах")
    w3, h3 = _fit(SCREENS / "db_composition.png", 4.4, 2.9)
    picture(sl, 0.50, 1.85, w3, h3, SCREENS / "db_composition.png",
            caption="ЧИСЛО СТРОК ПО ТАБЛИЦАМ", sub="читается из самой базы на момент отчёта")
    table(sl, 5.20, 1.88, 4.30, [
        ["Схема", "Что хранит"],
        ["data", "территории, сетки, ячейки, признаки"],
        ["meta", "паспорта, лицензии, происхождение"],
        ["runs", "журнал запусков и метрики"],
        ["kb", "справочник понятий"],
        ["bridge", "связи данных и знаний"],
        ["staging", "приёмный буфер агента"],
    ], [0.28, 0.72], row_h=0.38, size=9)

    # 4. Стек технологий (по просьбе пользователя).
    sl = deck.slide(S, R, "На чём это сделано",
                    "Всё работает на одном ноутбуке, без облака и без контейнеров")
    table(sl, 0.50, 1.88, 9.0, [
        ["Часть", "Что используем", "Зачем"],
        ["СУБД", "PostgreSQL 18.3 с расширением PostGIS 3.6", "хранить признаки, геометрию и паспорта; своя система координат сетки ГИС Интегро"],
        ["Растры", "файлы COG на диске F:", "тяжёлые снимки остаются вне базы, в базе путь, паспорт и контур покрытия"],
        ["Код", "Python 3.12, psycopg 3, SQLAlchemy 2, pandas, geopandas, rasterio", "загрузка слоёв, пересчёт в ячейки, запись происхождения"],
        ["Снимки для опытов", "parquet с манифестом (pyarrow)", "каждый эксперимент берёт неизменяемую копию признаков"],
        ["Панель оператора", "FastAPI, Jinja2, uvicorn", "очередь черновиков, подтверждение и отклонение, журнал решений"],
        ["Агент-паспортизатор", "локальная языковая модель, ответы по жёсткой схеме JSON", "черновик геологического смысла признака, который затем подтверждает оператор"],
        ["Проверки и копии", "pytest, дамп базы с проверкой восстановления", "правила базы проверяются тестами, копия считается копией только после восстановления"],
    ], [0.20, 0.38, 0.42], row_h=0.40, size=8.5)

    # 5. Что загружено.
    sl = deck.slide(S, R, "Что уже загружено",
                    "Покрытие территории по группам признаков посчитано по самой базе")
    w4, h4 = _fit(DB_FIGS / "покрытие_по_группам.png", 6.5, 2.85)
    picture(sl, 3.0, 1.82, w4, h4, DB_FIGS / "покрытие_по_группам.png",
            caption="СРЕДНЕЕ ПОКРЫТИЕ ТЕРРИТОРИИ ПО ГРУППАМ ПРИЗНАКОВ, %",
            sub="основной лист и широкая территория")
    for k, (v, lab) in enumerate([
        (str(d["terr"]), "территории"), (str(d["layers"]), "слоёв-источников"),
        (_sp(d["cells"]), "ячеек сетки"), (str(d["features"]), "признаков"),
    ]):
        metric(sl, 0.50, 1.85 + k * 0.80, 2.2, 0.72, v, lab, vsize=16, lsize=8)

    # 6. Паспорт признака.
    sl = deck.slide(S, R, "Паспорт признака",
                    "Что это за признак, где он измерен и можно ли ему верить")
    w5, h5 = _fit(SCREENS / "db_card_feature.png", 4.6, 2.9)
    picture(sl, 4.90, 1.82, w5, h5, SCREENS / "db_card_feature.png",
            caption="ПАСПОРТ ПРИЗНАКА ast_kaolin", sub="карточка в панели оператора")
    bullets(sl, 0.50, 1.95, 4.2, 2.9, [
        ("Покрытие: ", "какую долю территории признак реально измерен"),
        ("Контур съёмки: ", "зависит ли значение от того, где была съёмка"),
        ("Формула ВНИГНИ: ", "входит ли признак в критериальную формулу, чтобы не получить круговой вывод"),
        ("Переносимость: ", "проверена ли она на другой территории"),
        ("Геологический смысл: ", "на каком основании признак может указывать на золото или уран"),
    ], size=10, dot=A, space=6)

    # 7. Паспорт слоя.
    sl = deck.slide(S, R, "Паспорт слоя-источника",
                    "Второй уровень паспорта: сам исходный слой, из которого посчитаны признаки")
    w6, h6 = _fit(SCREENS / "db_card_layer.png", 4.6, 2.9)
    picture(sl, 4.90, 1.82, w6, h6, SCREENS / "db_card_layer.png",
            caption="ПАСПОРТ СЛОЯ-ИСТОЧНИКА", sub="карточка в панели оператора")
    bullets(sl, 0.50, 1.95, 4.2, 2.9, [
        ("Откуда: ", "кто выпустил слой и на каких условиях"),
        ("Лицензия: ", "отдельная запись, подтверждает оператор"),
        ("Контур: ", "где слой измерен, а где пусто"),
        ("Хеш файла: ", "по нему видно, что файл с тех пор не менялся"),
    ], size=10, dot=A, space=6)

    # 8. Правила подтверждения.
    sl = deck.slide(S, R, "Что может агент, а что только оператор",
                    "Правило записано в правах роли и в триггерах базы, обойти его агент не может")
    table(sl, 0.50, 1.88, 9.0, [
        ["Действие", "Агент", "Оператор", "Как устроено"],
        ["Завести черновик паспорта или лицензии", "да", "да", "обычная работа агента"],
        ["Поставить статус «подтверждён»", "нет", "да", "срабатывает триггер базы"],
        ["Удалить строку из данных или паспортов", "нет", "да", "у роли агента нет права удаления"],
        ["Удалять в приёмнике", "да", "да", "это черновая площадка агента"],
    ], [0.44, 0.10, 0.12, 0.34], row_h=0.40, size=9)
    deck.note(sl, "Агент готовит черновики сам, а подтверждение статуса и удаление остаются за оператором.")

    # 9. Панель.
    sl = deck.slide(S, R, "Панель оператора", "Здесь оператор смотрит черновики и принимает решения")
    w8, h8 = _fit(SCREENS / "db_queue_features.png", 4.9, 2.9)
    picture(sl, 4.60, 1.82, w8, h8, SCREENS / "db_queue_features.png",
            caption="ОЧЕРЕДЬ ПАСПОРТОВ ПРИЗНАКОВ", sub="статус «подтверждён»")
    for k, (v, lab) in enumerate([
        (str(d["fp"]), "паспортов признаков подтверждено оператором"),
        (str(d["lp"]), "паспортов слоёв подтверждено оператором"),
        (str(d["links"]), "связей признаков с понятиями справочника"),
    ]):
        metric(sl, 0.50, 1.88 + k * 1.0, 3.7, 0.90, v, lab, vsize=20, lsize=8.5)

    # 10. Происхождение.
    sl = deck.slide(S, R, "У каждого числа есть история",
                    "Цепочка «источник → операция → результат» плюс журнал запусков")
    node(sl, 0.50, 2.05, 2.5, 1.0, "Файл-источник", "хеш SHA-256, путь в проекте", **plain)
    node(sl, 3.75, 2.05, 2.5, 1.0, "Операция", "скрипт и версия кода", **agent)
    node(sl, 7.00, 2.05, 2.5, 1.0, "Признак", "версия определяется хешем входов", **human)
    line(sl, 3.00, 2.55, 3.75, 2.55, color=GREY)
    line(sl, 6.25, 2.55, 7.00, 2.55, color=GREY)
    for k, (v, lab) in enumerate([
        (_sp(d["deriv"]), "связей происхождения"), (_sp(d["ops"]), "операций"),
        (str(d["runs"]), "запусков"), (str(d["metrics"]), "метрик запусков"),
    ]):
        metric(sl, 0.50 + k * 2.28, 3.50, 2.1, 0.95, v, lab, vsize=20, lsize=8.5)

    # 11. Реестр пробелов.
    sl = deck.slide(S, R, "Чего у нас нет",
                    "Недоступные данные записаны в базе: что нужно, у кого лежит и зачем")
    stat = {"отказано": "данные у держателя фондов", "закрыт": "съёмки нет"}
    rows = [["Чего нет", "У кого", "Состояние"]]
    for title, holder, access in d["gap_rows"]:
        title = title.replace("Pathfinder-элементы", "Элементы-спутники золота")
        rows.append([title, holder, stat.get(access, access)])
    table(sl, 0.50, 1.88, 9.0, rows, [0.42, 0.34, 0.24], row_h=0.46, size=9)
    deck.note(sl, "Если данных не хватает, процесс возвращается к сбору, а пробел остаётся записанным.")


if __name__ == "__main__":
    deck = Deck(total=COUNT)
    deck.set_accent(*SECTION_ACCENTS["бд"])
    start = deck.page
    build(deck)
    assert deck.page - start == COUNT, (deck.page - start, COUNT)
    out = os.environ.get("DECK_OUT") or str(Path(tempfile.gettempdir()) / "report_db.pptx")
    deck.save(out)
    print(out)
