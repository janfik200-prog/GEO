"""Рецепты четырёх съёмок на лист: Sentinel-1, Landsat 8/9, PALSAR-2, ASTER.

Механика общая и живёт в :mod:`src.stac_grid`; здесь — только то, чем сенсоры
действительно различаются: коллекция, ассеты, перевод чисел в физику и набор
индексов. Каждый сенсор отдаёт ``DataFrame`` на 22 946 ячеек со своим префиксом,
пригодный для прямой склейки в датасет.

ПОЧЕМУ ИМЕННО ЭТИ ЧЕТЫРЕ. На листе 98.3% пикселей — растительность, поэтому
пятая оптическая съёмка ничего нового не даст: она измерит ту же тундру. Взяты
три РАЗНЫХ физических принципа плюс архив:

* Sentinel-1 (C-band, 5.5 см) — геометрия и шероховатость поверхности; радар
  не зависит от освещения и облаков и почти не зависит от травяного покрова;
* ALOS PALSAR-2 (L-band, 24 см) — та же природа сигнала, но длинная волна
  проникает сквозь покров глубже и рисует структурный каркас чётче;
* Landsat 8/9 — единственный доступный ТЕРМАЛЬНЫЙ канал: тепловая инерция
  грунта отражает влажность и гранулометрию, то есть субстрат напрямую;
* ASTER — 6 каналов SWIR против двух у Sentinel-2, единственный шанс на прямые
  Al-OH и карбонатные индексы; съёмка мертва с апреля 2008, работаем с архивом.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stac_grid

# --- Sentinel-1 RTC ---------------------------------------------------------
S1_BANDS: dict[str, tuple[str, int]] = {p: (p, 1) for p in config.S1_POLARIZATIONS}

# --- Landsat 8/9 Collection 2 L2 -------------------------------------------
L8_BANDS: dict[str, tuple[str, int]] = {
    "blue": ("blue", 1), "green": ("green", 1), "red": ("red", 1),
    "nir08": ("nir08", 1), "swir16": ("swir16", 1), "swir22": ("swir22", 1),
    "lwir11": ("lwir11", 1),
}
# Числитель и знаменатель — КОРТЕЖИ каналов (как у ASTER): формат общий, потому
# что _ratio складывает каналы, а строка «red» при переборе даёт буквы.
L8_RATIOS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "iron_ox": (("red",), ("blue",)),          # оксиды железа
    "ferrous": (("swir16",), ("nir08",)),      # закисное железо
    "clay": (("swir16",), ("swir22",)),        # глинистые / Al-OH
}

# --- ALOS PALSAR-2 годовая мозаика -----------------------------------------
PSR_BANDS: dict[str, tuple[str, int]] = {p.lower(): (p, 1) for p in config.PSR_POLARIZATIONS}
PSR_BANDS["linci"] = ("linci", 1)      # локальный угол падения: контроль рельефного артефакта

# --- ASTER L1T --------------------------------------------------------------
# Ассет SWIR — шестиканальный (b04..b09), VNIR — трёхканальный (b01, b02, b03n).
AST_BANDS: dict[str, tuple[str, int]] = {
    "b01": ("VNIR", 1), "b02": ("VNIR", 2), "b03": ("VNIR", 3),
    "b04": ("SWIR", 1), "b05": ("SWIR", 2), "b06": ("SWIR", 3),
    "b07": ("SWIR", 4), "b08": ("SWIR", 5), "b09": ("SWIR", 6),
}
#: Классические ASTER-индексы. Числитель и знаменатель — суммы каналов.
AST_INDICES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "aloh": (("b05", "b07"), ("b06",)),        # Al-OH, серицит-мусковит (аргиллизит)
    "kaolin": (("b04",), ("b05",)),            # каолинит
    "alter": (("b04",), ("b06",)),             # общая гидротермальная переработка
    "carb": (("b06", "b09"), ("b07", "b08")),  # карбонат / хлорит / эпидот
    "ferric": (("b02",), ("b01",)),            # окисное железо
    "ferrous": (("b05",), ("b04",)),           # закисное железо
}


def _ratio(comp: dict[str, np.ndarray], num, den) -> np.ndarray:
    """Отношение (сумм) каналов; деление на ноль и NaN гасятся в NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        a = sum(comp[b] for b in num)
        b = sum(comp[d] for d in den)
        r = a / b
    return np.where(np.isfinite(r), r, np.nan)


def _composite(items, bands, cache_dir, meta, log=None, mask_fn=None,
               nodata=None, min_obs=None):
    """Скачать сцены на сетку (с кэшем) и свернуть в медианный композит."""
    scenes = []
    for i, it in enumerate(items):
        arr = stac_grid.scene_on_grid(it, bands, meta, cache_dir,
                                      mask_fn=mask_fn, nodata=nodata)
        scenes.append(arr)
        if log:
            log(f"  сцена {i + 1}/{len(items)}: {it.id}")
    return stac_grid.median_stack(scenes, min_obs=min_obs)


# ---------------------------------------------------------------- Sentinel-1
def fetch_s1(meta, log=None) -> tuple[pd.DataFrame, list]:
    """Медианный летний композит Sentinel-1 RTC: VV, VH и их отношение в дБ.

    Квота по относительной орбите: ракурс съёмки задаёт, какие склоны освещены
    радаром, а какие в радиотени, — выборка из одной орбиты дала бы карту
    ракурса, а не поверхности.
    """
    items = stac_grid.search(config.S1_COLLECTION, meta, config.S1_YEARS,
                             config.S1_MONTHS, group_key="sat:relative_orbit",
                             max_per_group=config.S1_MAX_SCENES_PER_ORBIT)
    if log:
        orb = sorted({str(i.properties.get("sat:relative_orbit")) for i in items})
        log(f"Sentinel-1: {len(items)} сцен, орбиты {orb}")
    from cache_paths import S1_ANABAR

    comp, n_obs = _composite(items, S1_BANDS, S1_ANABAR, meta, log,
                             nodata=config.S1_NODATA)
    lay = {p: stac_grid.to_db(comp[p]) for p in config.S1_POLARIZATIONS}
    # Разность в дБ = отношение мощностей: чем она меньше, тем выше вклад
    # объёмного рассеяния (кустарник, обломочный чехол) против поверхностного.
    lay["vv_vh"] = lay["vv"] - lay["vh"]
    return stac_grid.to_cells(lay, meta, n_obs=n_obs, prefix="s1_"), items


# ---------------------------------------------------------------- Landsat 8/9
def _l8_mask(item, meta, res_m):
    """Маска качества Landsat по битам ``qa_pixel`` (облако, тень, снег)."""
    qa = stac_grid.read_on_grid(item, "qa_pixel", meta, res_m, nearest=True)
    bad = np.zeros(qa.shape, dtype=bool)
    q = np.nan_to_num(qa, nan=0.0).astype(np.uint32)
    for bit in config.L8_QA_REJECT_BITS:
        bad |= (q >> bit) & 1 > 0
    return ~bad & np.isfinite(qa)


def fetch_l8(meta, log=None) -> tuple[pd.DataFrame, list]:
    """Композит Landsat 8/9: отражение, минералогические отношения и LST."""
    items = stac_grid.search(
        config.L8_COLLECTION, meta, config.L8_YEARS, config.L8_MONTHS,
        query={"eo:cloud_cover": {"lt": config.L8_MAX_CLOUD},
               "platform": {"in": list(config.L8_PLATFORMS)}},
        group_key="landsat:wrs_path", max_per_group=config.L8_MAX_SCENES_PER_PATH)
    if log:
        log(f"Landsat 8/9: {len(items)} сцен, "
            f"годы {sorted({i.datetime.year for i in items})}")
    from cache_paths import L8_ANABAR

    comp, n_obs = _composite(items, L8_BANDS, L8_ANABAR, meta, log, mask_fn=_l8_mask)
    a, b = config.L8_SR_SCALE
    lay = {k: comp[k] * a + b for k in L8_BANDS if k != "lwir11"}
    ratios = {n: _ratio(lay, num, den) for n, (num, den) in L8_RATIOS.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = (lay["nir08"] - lay["red"]) / (lay["nir08"] + lay["red"])
    ratios["ndvi"] = np.where(np.isfinite(nd), nd, np.nan)
    # Температура поверхности в кельвинах: тепловая инерция грунта — прокси
    # влажности и гранулометрии, то есть субстрата под задернованием.
    ca, cb = config.L8_ST_SCALE
    lay["lst"] = comp["lwir11"] * ca + cb
    lay.pop("lwir11", None)
    return stac_grid.to_cells({**lay, **ratios}, meta, n_obs=n_obs, prefix="l8_"), items


# ---------------------------------------------------------------- PALSAR-2
def fetch_psr(meta, log=None) -> tuple[pd.DataFrame, list]:
    """Годовые мозаики ALOS PALSAR-2: HH, HV в дБ и локальный угол падения.

    Мозаики — это тайлы 1x1 градус, они не перекрываются, поэтому «наблюдения»
    в медиане набираются по годам, а не по перекрытию сцен.
    """
    items = stac_grid.search(config.PSR_COLLECTION, meta, config.PSR_YEARS)
    if log:
        log(f"PALSAR-2: {len(items)} тайлов, "
            f"годы {sorted({i.datetime.year for i in items})}")
    from cache_paths import PSR_ANABAR

    comp, n_obs = _composite(items, PSR_BANDS, PSR_ANABAR, meta, log, nodata=0.0)
    lay = {}
    for p in ("hh", "hv"):
        dn = comp[p]
        # Калибровка JAXA: γ0 [дБ] = 10*log10(DN^2) + offset. DN <= 1 — служебные
        # значения мозаики (нет данных), их нельзя пускать в логарифм.
        with np.errstate(invalid="ignore", divide="ignore"):
            db = 20.0 * np.log10(np.where(dn > 1, dn, np.nan)) + config.PSR_CAL_OFFSET
        lay[p] = np.where(np.isfinite(db), db, np.nan)
    lay["hh_hv"] = lay["hh"] - lay["hv"]
    lay["linci"] = comp["linci"]
    return stac_grid.to_cells(lay, meta, n_obs=n_obs, prefix="psr_"), items


# ---------------------------------------------------------------- ASTER
def fetch_ast(meta, log=None) -> tuple[pd.DataFrame, list]:
    """Композит ASTER L1T и минералогические индексы по 6 каналам SWIR.

    Порог наблюдений снижен (``AST_MIN_OBS``): летних сцен над листом всего
    десяток, требовать три перекрытия — остаться без карты. Это осознанный
    размен: карта будет, но шумнее, и это обязано быть видно в заверке.
    """
    items = stac_grid.search(
        config.AST_COLLECTION, meta, config.AST_YEARS, config.AST_MONTHS,
        query={"eo:cloud_cover": {"lt": config.AST_MAX_CLOUD}},
        max_per_group=config.AST_MAX_SCENES)
    if log:
        log(f"ASTER: {len(items)} сцен, даты "
            f"{sorted({i.datetime.strftime('%Y-%m') for i in items})}")
    from cache_paths import AST_ANABAR

    # nodata=0: у L1T нет объявленного nodata, а нулём закодирован фон
    # повёрнутой сцены — без явного указания он попал бы в медиану как чернота.
    comp, n_obs = _composite(items, AST_BANDS, AST_ANABAR, meta, log, nodata=0.0,
                             min_obs=config.AST_MIN_OBS)
    idx = {n: _ratio(comp, num, den) for n, (num, den) in AST_INDICES.items()}
    return stac_grid.to_cells({**comp, **idx}, meta, n_obs=n_obs, prefix="ast_"), items


#: Реестр сенсоров для прогонов: имя -> (функция, человекочитаемое название).
SENSORS = {
    "s1": (fetch_s1, "Sentinel-1 RTC (C-band радар)"),
    "l8": (fetch_l8, "Landsat 8/9 Collection 2 L2"),
    "psr": (fetch_psr, "ALOS PALSAR-2 (L-band радар)"),
    "ast": (fetch_ast, "ASTER L1T (архив SWIR до 2008)"),
}
