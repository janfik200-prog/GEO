"""Собирает два docx-файла по 104 признакам-кандидатам ТЕКУЩЕГО датасета
(data/processed/dataset_v6.parquet), сгруппированным по источнику съёмки/данных:

1. Дополнительные данные.docx — технический словарь (что за колонка); уже
   отправлен организации в data/share/dataset_v6_2026-08-19.zip, состав и
   текст не меняются этим скриптом относительно отправленной версии.
2. Признаки датасета — предметный смысл.docx — отдельный файл с переводом
   каждого признака на предметный (геологический/рудоконтролирующий) язык.

Запуск из корня: ``python -X utf8 -m experiments.build_v6_feature_dict_docx``.
"""
from __future__ import annotations

from pathlib import Path

import docx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset_v6.parquet"
OUT = ROOT / "data" / "SBORKA_DOP" / "Дополнительные данные.docx"
OUT_GEO = ROOT / "data" / "SBORKA_DOP" / "Признаки датасета — предметный смысл.docx"

ID_COLS = {"row", "col", "x", "y", "mask_svita"}

from src.feature_dict import (  # словарь признаков общий с агентом-паспортизатором
    GROUP_ORDER, GROUP_TITLE, classify, describe, geo_note,
)
def build_technical(groups: dict[str, list[str]]) -> None:
    """Технический словарь (что за колонка) — тот же файл, что уже отправлен
    организации в data/share/dataset_v6_2026-08-19.zip; текст не менять без
    отдельной причины пересылать заново."""
    d = docx.Document()
    d.add_heading("Признаки датасета — лист R-48-XI,XII", level=1)
    d.add_paragraph(
        "data/processed/dataset_v6.parquet: 22 946 ячеек сетки 500х500 м "
        f"(mask_svita маскирует территорию листа), 104 признака-кандидата "
        "(+ 5 служебных id/маска-колонок: row, col, x, y, mask_svita)."
    )

    for grp in GROUP_ORDER:
        feats = groups[grp]
        if not feats:
            continue
        heading = d.add_paragraph(f"{GROUP_TITLE[grp]} ({len(feats)} признак(ов))")
        heading.runs[0].bold = True
        for feat in feats:
            p = d.add_paragraph(style="List Bullet")
            p.add_run(f"{feat} – {describe(feat, grp)}.")

    d.save(OUT)
    print("сохранено:", OUT)


def build_geo(groups: dict[str, list[str]]) -> None:
    """Отдельный файл: предметный (геологический/рудоконтролирующий) смысл
    каждого признака — та же структура и группировка, что в техническом
    словаре, но не пересобирает уже отправленный файл заново."""
    d = docx.Document()
    d.add_heading("Признаки датасета — предметный (геологический) смысл", level=1)
    d.add_paragraph(
        "Дополнение к «Дополнительные данные.docx»: для каждого из 104 "
        "признаков-кандидатов dataset_v6 — что он означает с точки зрения "
        "рудного контроля (золото-урановая минерализация), а не только "
        "техническое описание съёмки/канала/индекса."
    )

    for grp in GROUP_ORDER:
        feats = groups[grp]
        if not feats:
            continue
        heading = d.add_paragraph(f"{GROUP_TITLE[grp]} ({len(feats)} признак(ов))")
        heading.runs[0].bold = True
        for feat in feats:
            p = d.add_paragraph(style="List Bullet")
            p.add_run(f"{feat} – {geo_note(feat, grp)}.")

    d.save(OUT_GEO)
    print("сохранено:", OUT_GEO)


def main() -> None:
    df = pd.read_parquet(DATASET)
    cols = [c for c in df.columns if c not in ID_COLS]
    assert len(cols) == 104, len(cols)

    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for c in cols:
        groups[classify(c)].append(c)

    build_technical(groups)
    build_geo(groups)
    for grp in GROUP_ORDER:
        print(f"  {grp}: {len(groups[grp])}")


if __name__ == "__main__":
    main()
