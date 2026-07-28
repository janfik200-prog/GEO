"""Единый ПОСТОЯННЫЙ кэш данных вне /tmp (переживает перезапуск сессии).

/tmp очищается при перезапуске окружения, поэтому все скачанные данные (NURE,
MRDS, CMMI, глобальная геофизика GMT) храним здесь. Каталог можно переопределить
переменной окружения GIS_CACHE.

Использование в скриптах:
    from cache_paths import NURE, MRDS_CSV, WORLD_GEO, CMMI
"""
import os
from pathlib import Path

CACHE = Path(os.environ.get("GIS_CACHE", Path(__file__).resolve().parent / "datacache"))
NURE = CACHE / "nure"
MRDS_CSV = CACHE / "nure" / "mrds" / "mrds.csv"
WORLD_GEO = CACHE / "world_geo"
ANABAR_GEO = CACHE / "anabar_geo"
ANABAR_DEM = CACHE / "anabar_dem"
CMMI = CACHE / "cmmi"
CMMI_AU = CACHE / "cmmi_au"
CMMI_OCC = CMMI / "cmmi_occ.csv"

for _p in (NURE, MRDS_CSV.parent, WORLD_GEO, ANABAR_GEO, ANABAR_DEM, CMMI, CMMI_AU):
    _p.mkdir(parents=True, exist_ok=True)
