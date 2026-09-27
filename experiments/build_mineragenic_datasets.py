"""Две минерагенические версии датасета: с нашей докачкой и без неё.

Задача (03.09.2026): отделить вклад самостоятельно докачанных данных от того,
что дала организация. Признаки берутся только минерагенические — те, у которых
есть рудоконтролирующая трактовка в словаре
`experiments/build_v6_feature_dict_docx.py`; разбор и происхождение —
`data/processed/mineragenic_feature_set.json`.

Собираются два набора:

* **полный** — 7 признаков организации (гравика/магнитка, ареалы долин) +
  13 факторных слоёв генетической модели + 34 признака нашей докачки
  (ASTER, Sentinel-2, Landsat 8/9, производные Copernicus DEM);
* **только организация** — те же 7 + 13, без докачки;
* **только докачка** — 34 признака, собранных нами, без единого слоя организации.
  Это единственный из трёх наборов без факторных слоёв, а значит единственный,
  где согласие с критериальной картой не завышено её же входами и где вообще
  возможен перенос на смежную территорию.

Оговорка, обязательная при чтении любых метрик на этих наборах. Факторные слои
(`dist_tect*`, `dens_*`, `dist_magm`, `dist_struct`, `dist_facies`,
`dist_paleo`, `geo2_*`) — прямые входы критериальной формулы, а мишень обучения
берётся из её же результата. Модель на них воспроизводит формулу, а не
прогнозирует независимо: измеренный потолок такого набора — 70.9% совпадения
верхних 10% даже через разрыв 15 км (`honest_raise_agreement.py`). Поэтому
разница между двумя наборами читается как «сколько добавляет докачка сверх
данных организации», и только так; заверкой прогноза ни один из них не служит.

Оба набора существуют только на основном листе: факторные слои оцифрованы по
нему, на `WIDE_META` их нет, поэтому развитие на широкую территорию возможно
лишь для подмножества без факторных слоёв.

Выход: `data/processed/dataset_mineragenic_full.parquet`,
`data/processed/dataset_mineragenic_org.parquet`.

Запуск из корня репозитория: ``python -m experiments.build_mineragenic_datasets``.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

SPEC = config.PROCESSED_DIR / "mineragenic_feature_set.json"
SOURCE = config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet"
ID_COLS = ["row", "col", "x", "y"]


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    org = [c for v in spec["происхождение"]["от_организации"].values() for c in v]
    own = [c for v in spec["происхождение"]["докачано_нами"].values() for c in v]
    factor = spec["факторные_слои_организации"]["признаки"]

    src = pd.read_parquet(SOURCE)
    missing = [c for c in org + own + factor + ID_COLS if c not in src.columns]
    if missing:
        raise KeyError(f"нет в {SOURCE.name}: {missing}")

    sets = {
        "dataset_mineragenic_full": ID_COLS + org + factor + own,
        "dataset_mineragenic_org": ID_COLS + org + factor,
        "dataset_mineragenic_own": ID_COLS + own,
    }
    for name, cols in sets.items():
        out = src[cols].copy()
        path = config.PROCESSED_DIR / f"{name}.parquet"
        out.to_parquet(path, index=False)
        na = out.drop(columns=ID_COLS).isna().mean()
        print(f"{name}: {len(out)} строк, {len(cols) - len(ID_COLS)} признаков "
              f"(организация {len(org)}, факторные {len(factor)}, "
              f"докачка {len(cols) - len(ID_COLS) - len(org) - len(factor)})")
        print(f"  покрытие: полное у {int((na == 0).sum())} признаков, "
              f"худшее {1 - na.max():.3f} ({na.idxmax()})")
        print(f"  OK: {path}")


if __name__ == "__main__":
    main()
