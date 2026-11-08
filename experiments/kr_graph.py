"""Гибридный граф представления знаний: слой знаний и слой данных.

Форма по просьбе научного руководителя: то же содержание, что в описании
формализации, но не таблицей и не веером, а графом — типизированные узлы и
именованные рёбра.

Граф гибридный и читается снизу вверх: нижний слой — данные (канал,
признак, условие наблюдения), верхний — понятия предметной модели (критерий,
объект, звено, этап). Слои соединены единственным типом ребра «измеряет»;
условия наблюдения дают только C(x) и в оценку перспективности не входят.

Показан реальный фрагмент модели, а не абстрактный пример: разломная зона в
трёх ролях, вмещающая свита как ограничение, спектральный признак под
условием обнажённости.

Требуется d2 (``winget install Terrastruct.D2``).

Запуск из корня репозитория::

    python -m experiments.kr_graph
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ВЫХОД = ROOT / "outputs" / "14_статья_знания"
ИСХОДНИК = ВЫХОД / "kr_граф.d2"
PNG = ВЫХОД / "kr5_граф.png"
SVG = ВЫХОД / "kr5_граф.svg"

ИСТОЧНИК = r'''vars: { d2-config: { layout-engine: elk; theme-id: 0 } }
direction: up

classes: {
  этап:     { style: { fill: "#dfe9f3"; stroke: "#3f6d94"; stroke-width: 2; font-size: 20; bold: true; border-radius: 20 } }
  звено:    { style: { fill: "#cfe3d6"; stroke: "#3f7a57"; stroke-width: 2; font-size: 20; bold: true; border-radius: 20 } }
  объект:   { style: { fill: "#e6ddf0"; stroke: "#6a4f93"; stroke-width: 2; font-size: 20; bold: true; border-radius: 10 } }
  критерий: { style: { fill: "#fbeedc"; stroke: "#b8690f"; stroke-width: 2; font-size: 19; bold: true; border-radius: 10 } }
  признак:  { style: { fill: "#eef5fa"; stroke: "#7f9fb3"; stroke-width: 1; font-size: 18; border-radius: 6 } }
  канал:    { style: { fill: "#f1f1ef"; stroke: "#8b8b86"; stroke-width: 1; font-size: 18; border-radius: 6 } }
  условие:  { style: { fill: "#fdeaea"; stroke: "#a44a4a"; stroke-width: 2; font-size: 18; bold: true; border-radius: 10 } }
  правило:  { style: { fill: "#fff8e1"; stroke: "#8a6d1f"; stroke-width: 2; font-size: 18; bold: true; stroke-dash: 3; border-radius: 10 } }
  выход:    { style: { fill: "#c3dbea"; stroke: "#2f6b8f"; stroke-width: 3; font-size: 21; bold: true; border-radius: 20 } }
  источник: { style: { fill: "#ffffff"; stroke: "#9aa9b6"; stroke-width: 1; font-size: 17; border-radius: 6 } }
}

выходы: "ВЫХОД ЯЧЕЙКИ" {
  style: { fill: "#f7fbfd"; stroke: "#2f6b8f"; stroke-width: 2; font-size: 24; bold: true }
  p: "P(x)\nоценка перспективности" { class: выход }
  c: "C(x)\nобеспеченность наблюдениями" { class: выход }
}

знания: "СЛОЙ ЗНАНИЙ — понятия предметной модели" {
  style: { fill: "#fbfcfe"; stroke: "#3f6d94"; stroke-width: 2; font-size: 24; bold: true }

  этап3: "Этап III — рифей" { class: этап }

  зи: "Звено И\nисточник" { class: звено }
  зп: "Звено П\nпуть" { class: звено }
  зн: "Звено Н\nместо накопления" { class: звено }

  о1: "О1 Разломная зона" { class: объект }
  о9: "О9 Складки" { class: объект }
  о11: "О11 Свиты рифея" { class: объект }

  к2: "К2 близость к зоне" { class: критерий }
  к3: "К3 раздроблённость" { class: критерий }
  к1: "К1 узлы пересечения" { class: критерий }
  к21: "К21 согласованность линий" { class: критерий }
  к23: "К23 принадлежность свите" { class: критерий }

  ог1: "Ог1: объект закрывает\nне более одного звена" { class: правило }
  ист: "Минаев и др., 2026" { class: источник }

  к2 -> о1: проверяет
  к3 -> о1: проверяет
  к1 -> о1: проверяет
  к21 -> о9: проверяет
  к23 -> о11: проверяет

  о1 -> зи: играет роль
  о1 -> зп: играет роль
  о1 -> зн: играет роль
  о9 -> зн: играет роль
  о11 -> зн: ограничивает

  зи -> этап3: на этапе
  зп -> этап3: на этапе
  зн -> этап3: на этапе

  ог1 -> о1: ограничивает роли
  ист -> о1: обосновывает
}

данные: "СЛОЙ ДАННЫХ — признаки и каналы" {
  style: { fill: "#fbfcfe"; stroke: "#7f9fb3"; stroke-width: 2; font-size: 24; bold: true }

  сетка: "Канал на сетке 500 м — 113 543 ячейки" { class: канал }

  lin_dist: "lin_dist ↓" { class: признак }
  lin_dens: "lin_dens ↑" { class: признак }
  lin_node: "lin_node_dens ↑" { class: признак }
  lin_aniso: "lin_aniso_r ↑" { class: признак }
  svk: "svk_list — доля" { class: признак }
  ast: "ast_aloh ↑" { class: признак }

  в1: "В1 обнажённость\nпорог 2 %" { class: условие }
  в3: "В3 контур съёмки\n69,7 % листа" { class: условие }

  сетка -> lin_dist
  сетка -> lin_dens
  сетка -> lin_node
  сетка -> lin_aniso
  сетка -> svk
  сетка -> ast
  ast -> в1: наблюдается при
  ast -> в3: наблюдается при
}

данные.lin_dist -> знания.к2: измеряет { style: { stroke: "#b8690f"; stroke-width: 2; font-size: 18; font-color: "#8a4e09" } }
данные.lin_dens -> знания.к3: измеряет { style: { stroke: "#b8690f"; stroke-width: 2; font-size: 18; font-color: "#8a4e09" } }
данные.lin_node -> знания.к1: измеряет { style: { stroke: "#b8690f"; stroke-width: 2; font-size: 18; font-color: "#8a4e09" } }
данные.lin_aniso -> знания.к21: измеряет { style: { stroke: "#b8690f"; stroke-width: 2; font-size: 18; font-color: "#8a4e09" } }
данные.svk -> знания.к23: измеряет { style: { stroke: "#b8690f"; stroke-width: 2; font-size: 18; font-color: "#8a4e09" } }

знания.зи -> выходы.p: мягкое «и» { style: { stroke: "#2f6b8f"; stroke-width: 3; font-size: 18 } }
знания.зп -> выходы.p: мягкое «и» { style: { stroke: "#2f6b8f"; stroke-width: 3; font-size: 18 } }
знания.зн -> выходы.p: мягкое «и» { style: { stroke: "#2f6b8f"; stroke-width: 3; font-size: 18 } }
данные.в1 -> выходы.c: доля наблюдаемых звеньев { style: { stroke: "#a44a4a"; stroke-width: 3; stroke-dash: 3; font-size: 18; font-color: "#a44a4a" } }
данные.в3 -> выходы.c: доля наблюдаемых звеньев { style: { stroke: "#a44a4a"; stroke-width: 3; stroke-dash: 3; font-size: 18; font-color: "#a44a4a" } }

легенда: |md
  ### Как читать граф

  **Поток снизу вверх:** канал → признак → критерий → объект → звено → выход ячейки. Это и есть порядок вычисления.

  **Типы узлов:** этап · звено процесса · геологический объект · критерий · признак · условие наблюдения · правило · источник · выход.

  **Типы рёбер:** содержит · играет роль · ограничивает · проверяет · измеряет · наблюдается при · обосновывает · мягкое «и».

  **Почему граф, а не дерево:** разломная зона О1 связана ребром «играет роль» с тремя звеньями сразу. В дереве с одним родителем это либо потеря двух ролей, либо три копии объекта. Правило Ог1 ограничивает её до одной роли в каждой ячейке.

  **Почему гибридный:** слой знаний и слой данных живут в разных базах и соединены единственным типом ребра — «измеряет». Условия наблюдения подключены к слою данных и в P(x) не входят: они дают только C(x).
| { style: { font-size: 19; stroke-width: 0 } }

легенда -> данные: { style: { opacity: 0 } }
'''


def main() -> None:
    запасные = [r"C:\Program Files\D2\d2.exe",
                r"C:\Program Files\D2\bin\d2.exe"]
    d2 = shutil.which("d2") or next(
        (p for p in запасные if Path(p).exists()), "")
    if not d2:
        sys.exit("не найден d2: winget install Terrastruct.D2")
    ВЫХОД.mkdir(parents=True, exist_ok=True)
    ИСХОДНИК.write_text(ИСТОЧНИК, encoding="utf-8")
    for цель, доп in ((PNG, ["--scale", "0.9"]), (SVG, [])):
        р = subprocess.run(
            [d2, "--layout", "elk", "--theme", "0", "--pad", "50", *доп,
             str(ИСХОДНИК), str(цель)],
            capture_output=True, text=True, encoding="utf-8")
        if р.returncode:
            print(р.stdout, р.stderr, sep="\n")
            sys.exit("d2 завершился с кодом %d" % р.returncode)
    print("исходник:", ИСХОДНИК)
    print("png:", PNG)
    print("svg:", SVG)


if __name__ == "__main__":
    main()
