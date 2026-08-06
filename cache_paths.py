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
S2_ANABAR = CACHE / "s2_anabar"
# Этап 8: остальные съёмки на тот же лист, каждая в своём каталоге. Разделены не
# ради порядка, а ради идемпотентности: прогон одного сенсора не должен видеть
# и перечитывать чужие кэши.
S1_ANABAR = CACHE / "s1_anabar"          # Sentinel-1 RTC
L8_ANABAR = CACHE / "l8_anabar"          # Landsat 8/9 Collection 2 L2
PSR_ANABAR = CACHE / "psr_anabar"        # ALOS PALSAR-2 годовые мозаики
AST_ANABAR = CACHE / "ast_anabar"        # ASTER L1T (архив до 04.2008)
GGK200 = CACHE / "ggk200"                # листы Госгеолкарты-200 (растры geokniga)

for _p in (NURE, MRDS_CSV.parent, WORLD_GEO, ANABAR_GEO, ANABAR_DEM, CMMI, CMMI_AU,
           S2_ANABAR, S1_ANABAR, L8_ANABAR, PSR_ANABAR, AST_ANABAR, GGK200):
    _p.mkdir(parents=True, exist_ok=True)
