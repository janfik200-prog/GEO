# -*- coding: utf-8 -*-
"""Снимки панели агентной базы данных для раздела «База данных» большой колоды.

Цель: получить картинки интерфейса и состава базы с ЖИВЫМИ числами (не нарисованными).
Вход: запущенная база `F:\\GIS-DB` (порт 5433, роль gis_read) и пакет `gisdb`
репозитория `C:\\Users\\gstat\\GIS-agent-db`.
Выход: `outputs/db_screens/*.png`.

Таблицу оценки моделей и журнал решений на снимки не берём (названия моделей,
пакетные подтверждения). Страницы панели рисуются её же шаблонами напрямую, без сервера и без входа:
база открывается только на чтение, никакие решения не записываются.
Снимает Edge в безголовом режиме.

Запуск (база должна работать: F:\\GIS-DB\\start_db.bat):
    python -X utf8 -m experiments.make_db_screens
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_REPO = Path(r"C:\Users\gstat\GIS-agent-db")
OUT = ROOT / "outputs" / "db_screens"
HTML = OUT / "_html"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

sys.path.insert(0, str(DB_REPO))
from gisdb import db  # noqa: E402
from gisdb.panel import ШАБЛОНЫ, queries as q  # noqa: E402

# Имя проверяющего в шапке панели на снимках не показываем.
КТО = ""


def render(name, **kw):
    return ШАБЛОНЫ.get_template(name).render(**kw)


def scrub(html):
    """На слайды не выносим названия моделей: в карточке они стоят в подзаголовке."""
    html = html.replace("автоматически моделью локальный:gemma4:12b.", "автоматически.")
    html = re.sub(r"(средство|локальный|облачный):[\w.\-:]+", "агент", html)
    html = re.sub(r"автоматически\s+моделью\s+агент\.?", "автоматически агентом.", html)
    html = html.replace("локальный:gemma4:12b", "")
    # история решений внизу карточки: имя проверяющего и причины пакетных решений
    i = html.find("Что решали раньше")
    if i != -1:
        a = html.rfind("<h2", 0, i)
        b = html.find("</main>", i)
        html = html[:a] + html[b:]
    html = html.replace("Ниже — те же факты, по которым его писала модель.",
                        "Ниже — те же факты, по которым он писался.")
    return html.replace("Что именно видела модель", "Исходные факты для черновика")


def shoot(html, png, w=1280, h=800):
    html = scrub(html)
    HTML.mkdir(parents=True, exist_ok=True)
    f = HTML / (Path(png).stem + ".html")
    f.write_text(html, encoding="utf-8")
    subprocess.run(
        [EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--window-size={w},{h}", f"--screenshot={OUT / png}", f.resolve().as_uri()],
        check=True, capture_output=True, timeout=90)
    print("снимок", png)


def composition_html(conn):
    """Состав базы: число строк по схемам и главным таблицам — из самой базы."""
    cur = conn.cursor()
    items = [
        ("data", "territory", "территории"), ("data", "grid", "сетки"),
        ("data", "cell", "ячейки сетки"), ("data", "source_layer", "слои-источники"),
        ("data", "feature", "признаки"), ("data", "feature_value", "значения признаков"),
        ("meta", "layer_passport", "паспорта слоёв"), ("meta", "feature_passport", "паспорта признаков"),
        ("meta", "derivation", "связи происхождения"), ("meta", "operation", "операции"),
        ("meta", "review", "решения проверяющего"), ("meta", "gap", "реестр пробелов"),
        ("runs", "run", "запуски"), ("runs", "metric", "метрики запусков"),
        ("bridge", "feature_concept", "связи признаков и понятий"), ("kb", "concept", "понятия справочника"),
    ]
    rows = ""
    for s, t, label in items:
        cur.execute(f"select count(*) from {s}.{t}")
        n = cur.fetchone()[0]
        rows += (f"<tr><td>{label}</td><td class='приглушённо'>{s}.{t}</td>"
                 f"<td class='число'>{n:,}</td></tr>".replace(",", "\u00a0"))
    body = ("<h1>Состав базы</h1><p class='подзаголовок'>Число строк читается из самой базы "
            "gis_au_u на момент снимка</p><table><tr><th>Что хранится</th><th>Таблица</th>"
            f"<th class='число'>Строк</th></tr>{rows}</table>")
    tpl = ШАБЛОНЫ.from_string("{% extends 'base.html' %}{% block content %}" + body + "{% endblock %}")
    return tpl.render(кто=КТО)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conn = db.connect("gis_read")
    shoot(composition_html(conn), "db_composition.png", w=760, h=720)
    shoot(render("index.html", кто=КТО, сводка=q.сводка(conn),
                 оценка=[], журнал=[]), "db_panel_index.png", h=440)
    shoot(render("queue.html", кто=КТО, вид=q.ВИДЫ["признаки"], статус="подтверждён",
                 статусы=q.СТАТУСЫ, записи=q.очередь(conn, q.ВИДЫ["признаки"], "подтверждён")[:14]),
          "db_queue_features.png", w=800, h=760)
    # паспорт признака ast_kaolin на основном листе (feature_id = 24) и слой-источник
    осталось = len(q.очередь(conn, q.ВИДЫ["признаки"], "черновик"))
    shoot(render("card_признаки.html", кто=КТО, вид=q.ВИДЫ["признаки"],
                 к=q.карточка(conn, q.ВИДЫ["признаки"], 24), подписи=q.ПОДПИСИ,
                 осталось=осталось, открыто=0), "db_card_feature.png", w=760, h=760)
    shoot(render("card_слои.html", кто=КТО, вид=q.ВИДЫ["слои"],
                 к=q.карточка(conn, q.ВИДЫ["слои"], 2), подписи=q.ПОДПИСИ,
                 осталось=осталось, открыто=0), "db_card_layer.png", w=760, h=760)


if __name__ == "__main__":
    main()
