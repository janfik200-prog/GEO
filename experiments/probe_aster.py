"""Зонд архива ASTER L1T над листом R-48-XI,XII (gate-скрипт ветки ASTER).

SWIR-детекторы ASTER вышли из строя 01.04.2008 — пригоден только архив
2000-03.2008. Скрипт анонимно опрашивает NASA CMR и печатает летние
(VI-IX) сцены с облачностью <= AST_MAX_CLOUD. Вердикт:

- есть сцены -> ветка открыта, скачивание через Earthdata Login (аккаунт);
- нет сцен -> ветку закрыть, зафиксировать в dataset_v2_sources.md.

Результат зонда 02.08.2026: 105 летних гранул, в т.ч. облачность 0-2%
(21.06.2001, 26.06.2002 x3) — ветка ОТКРЫТА.

Запуск из корня: ``python -m experiments.probe_aster``.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
BBOX = "105.85,70.6,108.21,71.4"          # лист R-48-XI,XII в WGS84
TEMPORAL = "2000-03-01T00:00:00Z,2008-04-01T00:00:00Z"
SUMMER_MONTHS = ("06", "07", "08", "09")
AST_MAX_CLOUD = 30.0


def main() -> None:
    params = {"short_name": "AST_L1T", "bounding_box": BBOX,
              "temporal": TEMPORAL, "page_size": "500"}
    with urllib.request.urlopen(f"{CMR_URL}?{urllib.parse.urlencode(params)}") as resp:
        entries = json.load(resp)["feed"]["entry"]
    summer = [e for e in entries if e["time_start"][5:7] in SUMMER_MONTHS]
    good = [e for e in summer
            if float(e.get("cloud_cover") or 100.0) <= AST_MAX_CLOUD]
    good.sort(key=lambda e: float(e.get("cloud_cover") or 100.0))

    print(f"Гранул всего: {len(entries)}, летних (VI-IX): {len(summer)}, "
          f"с облачностью <= {AST_MAX_CLOUD:.0f}%: {len(good)}")
    for e in good[:20]:
        print(f"  {e['time_start'][:10]}  cloud={e.get('cloud_cover', '?'):>4}  {e['title']}")
    print("\nВердикт:", "ветка ASTER ОТКРЫТА (скачивание требует Earthdata Login)"
          if good else "летних малооблачных SWIR-сцен нет — ветку ЗАКРЫТЬ")


if __name__ == "__main__":
    main()
