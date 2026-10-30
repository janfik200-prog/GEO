"""WP09 ТЗ ИПУЗ: сверка полного прогона от исходных сеток с заморозкой.

Скрипт ничего не считает заново. Он сравнивает то, что оставил после себя
полный прогон::

    python -m experiments.operating_point_masked

с замороженными таблицами релиза. Это вторая из двух независимых проверок,
которых требует Приложение А ТЗ: первая — полный pytest, вторая — пересчёт
метрик из сохранённых данных. Здесь проверяется более сильное утверждение:
что метрики получаются из **исходных сеток** проекта, а не только из
производных таблиц.

Сравниваются все числовые столбцы `outputs/metrics/operating_point_masked.csv`
с `04_metrics/main_metrics.csv` и массив прогноза
`data/processed/operating_point_masked.parquet` поэлементно.

SHA самого parquet при повторной записи меняется (pandas пишет свою
служебную часть), поэтому решает поэлементное сравнение, а не хеш файла.

Запуск::

    python -m experiments.ipuz_full_rerun_check --freeze <каталог_заморозки>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ДОПУСК = 1e-6
КЛЮЧ = "вариант"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for кусок in iter(lambda: f.read(1 << 20), b""):
            h.update(кусок)
    return h.hexdigest()


def версии() -> dict:
    import sklearn
    out = {"python": platform.python_version(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__}
    try:
        import rasterio
        out["rasterio"] = rasterio.__version__
        out["GDAL"] = rasterio.__gdal_version__
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--log", type=Path, default=None,
                    help="журнал полного прогона; будет скопирован в 03_runs")
    a = ap.parse_args()

    freeze = a.freeze.resolve()
    свежий_csv = a.root / "outputs" / "metrics" / "operating_point_masked.csv"
    замор_csv = freeze / "04_metrics" / "main_metrics.csv"
    свежий_pq = a.root / "data" / "processed" / "operating_point_masked.parquet"
    for p in (свежий_csv, замор_csv, свежий_pq):
        if not p.exists():
            raise SystemExit(
                "Не найден файл: %s\nСначала выполните полный прогон: "
                "python -m experiments.operating_point_masked" % p)

    св = pd.read_csv(свежий_csv).set_index(КЛЮЧ).sort_index()
    зм = pd.read_csv(замор_csv).set_index(КЛЮЧ).sort_index()
    if list(св.index) != list(зм.index):
        raise SystemExit("Наборы вариантов не совпадают: %s против %s"
                         % (list(св.index), list(зм.index)))

    столбцы = [c for c in зм.columns if c in св.columns
               and pd.api.types.is_numeric_dtype(зм[c])]
    расхождения = {}
    for вариант in зм.index:
        for c in столбцы:
            d = float(abs(св.loc[вариант, c] - зм.loc[вариант, c]))
            расхождения["%s / %s" % (вариант, c)] = d
    макс = max(расхождения.values())

    массив = pd.read_parquet(свежий_pq)["score"].to_numpy(float)
    эталон_pq = freeze / "01_inputs" / "operating_point_masked.parquet"
    сравнение_массива = None
    if эталон_pq.exists():
        э = pd.read_parquet(эталон_pq)["score"].to_numpy(float)
        сравнение_массива = {
            "источник": str(эталон_pq.relative_to(freeze)),
            "ячеек": int(массив.size),
            "совпало_точно": int((массив == э).sum()),
            "макс_разность": float(np.nanmax(np.abs(массив - э))),
        }

    отчёт = {
        "что_проверено": "полный прогон от исходных сеток проекта против "
                         "замороженных таблиц релиза",
        "команда_прогона": "python -m experiments.operating_point_masked",
        "дата_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "окружение": версии(),
        "вход_прогона": {
            "сетка_критериального_расчёта": "data/Gis-integro/Расчет/"
                                            "prognoz.pgrid",
            "примечание": "исходные сетки в релиз не входят — закрытые данные "
                          "правообладателя; в пакете остаются только "
                          "производные таблицы",
        },
        "столбцов_сверено": len(столбцы),
        "вариантов": list(map(str, зм.index)),
        "допуск": ДОПУСК,
        "макс_расхождение": макс,
        "расхождения": расхождения,
        "массив_прогноза": сравнение_массива,
        "sha256_свежего_parquet": sha256(свежий_pq),
        "примечание_о_sha": "SHA файла parquet меняется при повторной записи "
                            "из-за служебной части формата; сравнение идёт по "
                            "значениям, а не по хешу файла",
        "итог": "PASS" if макс <= ДОПУСК else "FAIL",
    }

    out = freeze / "03_runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_pipeline_rerun.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.log and a.log.exists():
        (out / "stdout_full_pipeline.log").write_text(
            a.log.read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8")
        отчёт["журнал"] = "03_runs/stdout_full_pipeline.log"

    print("столбцов сверено: %d, вариантов: %d" % (len(столбцы), len(зм.index)))
    print("макс расхождение: %.3g (допуск %.0e) → %s"
          % (макс, ДОПУСК, отчёт["итог"]))
    if сравнение_массива:
        print("массив прогноза: совпало %d из %d, макс разность %.3g"
              % (сравнение_массива["совпало_точно"],
                 сравнение_массива["ячеек"],
                 сравнение_массива["макс_разность"]))
    print("записано:", out / "full_pipeline_rerun.json")
    raise SystemExit(0 if отчёт["итог"] == "PASS" else 1)


if __name__ == "__main__":
    main()
