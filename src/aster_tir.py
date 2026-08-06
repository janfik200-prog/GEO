"""ASTER TIR (каналы 10-14): признак окварцевания по архиву LP DAAC.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ, а не ещё пара каналов в :mod:`src.sat_sources`.

Кварц — главный минерал золото-кварцевой формации — не имеет диагностических
полос поглощения ни в видимом диапазоне, ни в SWIR. Единственная его
спектральная подпись это полосы остаточных лучей (Reststrahlen) на 8-9.5 мкм,
то есть тепловой диапазон. Из съёмок на территорию нужное спектральное
разрешение там есть только у ASTER: пять каналов 8.1-11.6 мкм с шагом 90 м.
У Landsat 8 тепловой канал один, по одному каналу отношение не построишь.

Два отличия от группы ``ast_*`` заставляют держать отдельный код:

1. ИСТОЧНИК. Каталог Planetary Computer обрывается на 31.12.2006, поэтому
   ``ast_*`` покрывает лист на 71.6%. Тепловой диапазон отказ SWIR в апреле
   2008 пережил (сломались детекторы SWIR, не TIR), и в архиве LP DAAC над
   листом 136 летних малооблачных гранул после 2008 против 17 до. Ветка тянет
   данные оттуда: версия AST_L1T.004 раздаётся поканальными GeoTIFF, поэтому
   HDF-EOS читать не нужно, а окно под лист вычитывается через ``/vsicurl``.
2. ФИЗИКА. TIR меряет не отражённый солнечный свет, а собственное излучение
   поверхности, и до индексов DN обязан стать радиансом: коэффициенты каналов
   различаются в 1.3 раза, на сырых DN индекс смещён. Дальше — нормировка
   Ниномии: радианс пересчитывается так, будто поверхность имеет фиксированную
   температуру 300 К. Без неё индексы читают температуру, а не минерал.

ИНДЕКСЫ (Ninomiya, Fu, Cudahy, 2005; определены как раз для радианса на
сенсоре, поэтому атмосферная коррекция не требуется):

* ``qi`` — кварцевый: у кварца эмиссия в канале 11 выше, чем в 10 и 12;
* ``ci`` — карбонатный: канал 14 против 13;
* ``mi`` — мафический, обратно связан с валовым содержанием SiO2.

Ограничение честности: индекс меряет кварц В ПОВЕРХНОСТНОМ СЛОЕ. На листе
98.3% пикселей — растительность, поэтому ожидание сигнала привязано к тем же
обнажениям, что и у остальной оптики; это проверяется покрытием и заверкой,
а не постулируется.

ВЕРДИКТ (06.08.2026, ``python -m experiments.quartz_check``, полный протокол
и цифры — в докстринге модуля). Покрытие подтвердилось: 100% валидных ячеек
против 69.7% у группы ``ast`` — гипотеза про выживший после 2008 года TIR
верна. Но содержательная проверка ОТРИЦАТЕЛЬНА: ``qi`` коррелирует с NDVI
(rho=-0.65) и уклоном (rho=0.53) сильнее порога артефакта (|rho|>0.5) —
нормировка на 300 К снимает объёмную температуру, но не снимает эффекты
субпиксельной растительности и геометрии освещения на склонах, которые
для отношений каналов остаются доминирующим сигналом. После линейного
выноса NDVI и t13 связь с геологией не восстанавливается (|rho_част|<=0.25
по всем 10 геологическим слоям). В критериальном протоколе ни один из трёх
индексов Ниномии не проходит порог AUC>=0.75 (лучший — ``ci`` 0.644,
95% ДИ [0.573, 0.716]) и не проходит бутстрэп-воспроизводимость
(``reproduces=False`` у всех). Признаки ``astir_qi/ci/mi`` в датасет как
геологический сигнал НЕ добавлены: то, что они измеряют, уже представлено
в датасете напрямую ``s2_ndvi`` и ``dem_slope`` без потери информации.
Ветка закрыта.
"""
from __future__ import annotations

import json
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, integro_grid, stac_grid

#: Короткие имена каналов -> суффикс файла у LP DAAC.
BANDS: tuple[str, ...] = ("b10", "b11", "b12", "b13", "b14")


@dataclass
class Granule:
    """Гранула LP DAAC в интерфейсе, совместимом с item'ом STAC.

    Поля ``id``/``datetime``/``properties`` названы как у pystac.Item намеренно:
    их читают отбор сцен (:func:`stac_grid._pick`) и контроль качества прогона.
    """

    id: str
    datetime: datetime
    cloud: float
    urls: dict[str, str]
    center: tuple[float, float] = (0.0, 0.0)
    properties: dict = field(default_factory=dict)


# ------------------------------------------------------------------ каталог
def _polygon_center(entry: dict) -> tuple[float, float]:
    """Центр гранулы по её полигону CMR (широта, долгота)."""
    polys = entry.get("polygons") or []
    if not polys:
        return (0.0, 0.0)
    nums = [float(v) for v in polys[0][0].split()]
    lat, lon = nums[0::2], nums[1::2]
    return (sum(lat) / len(lat), sum(lon) / len(lon))


def _query_cmr(bbox: list[float], years: tuple[int, ...]) -> list[dict]:
    """Все гранулы AST_L1T над охватом за годы; постранично, до исчерпания."""
    t0, t1 = min(years), max(years) + 1
    out: list[dict] = []
    for page in range(1, 21):
        params = {"short_name": config.ASTIR_SHORT_NAME,
                  "bounding_box": ",".join(f"{v:.4f}" for v in bbox),
                  "temporal": f"{t0}-01-01T00:00:00Z,{t1}-01-01T00:00:00Z",
                  "page_size": "2000", "page_num": str(page)}
        url = f"{config.ASTIR_CMR_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            entries = json.load(resp)["feed"]["entry"]
        out += entries
        if len(entries) < 2000:
            break
    return out


def _band_urls(entry: dict) -> dict[str, str]:
    """Ссылки на поканальные GeoTIFF TIR; пусто, если гранула их не раздаёт."""
    urls: dict[str, str] = {}
    for link in entry.get("links", []):
        href = link.get("href", "")
        if not href.startswith("https://"):
            continue
        for b in BANDS:
            if href.endswith(f"_TIR_B{b[1:]}.tif"):
                urls[b] = href
    return urls


def search(meta: integro_grid.GridMeta,
           years: tuple[int, ...] | None = None,
           months: tuple[int, ...] | None = None,
           max_cloud: float | None = None,
           max_per_group: int | None = None,
           group_deg: float | None = None) -> list[Granule]:
    """Летние малооблачные дневные гранулы над листом, с квотой на группу.

    Квота — по округлённому центру гранулы, а не общий лимит: сцена ASTER
    60x60 км, лист 86x77 км, и без квоты вся выборка уходит в тот угол, где
    чаще всего ясно, а остальная площадь остаётся с одной-двумя датами.
    """
    years = years or config.ASTIR_YEARS
    months = months or config.ASTIR_MONTHS
    max_cloud = config.ASTIR_MAX_CLOUD if max_cloud is None else max_cloud
    max_per_group = config.ASTIR_MAX_PER_GROUP if max_per_group is None else max_per_group
    group_deg = config.ASTIR_GROUP_DEG if group_deg is None else group_deg

    granules: list[Granule] = []
    for e in _query_cmr(stac_grid.bbox_wgs84(meta), years):
        if config.ASTIR_DAY_ONLY and e.get("day_night_flag") != "DAY":
            continue
        cloud = float(e.get("cloud_cover") or 100.0)
        if cloud > max_cloud:
            continue
        dt = datetime.strptime(e["time_start"][:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc)
        if dt.year not in years or dt.month not in months:
            continue
        urls = _band_urls(e)
        if len(urls) < len(BANDS):
            continue
        granules.append(Granule(id=e["title"], datetime=dt, cloud=cloud, urls=urls,
                                center=_polygon_center(e),
                                properties={"eo:cloud_cover": cloud}))

    out: list[Granule] = []
    keys = sorted({(round(g.center[0] / group_deg), round(g.center[1] / group_deg))
                   for g in granules})
    for k in keys:
        grp = [g for g in granules
               if (round(g.center[0] / group_deg), round(g.center[1] / group_deg)) == k]
        out += stac_grid._pick(grp, max_per_group)
    return out


# ------------------------------------------------------------------ чтение
def edl_env() -> dict[str, str]:
    """Переменные GDAL для чтения защищённых файлов LP DAAC.

    Авторизация Earthdata живёт в ``~/_netrc`` (вне репозитория) и подставляется
    самим curl; редирект на urs.earthdata.nasa.gov отдаёт cookie, ради которой
    и нужен общий файл cookies — иначе каждый канал каждой сцены логинился бы
    заново.
    """
    ck = str(Path(tempfile.gettempdir()) / "gis_edl_cookies.txt")
    return {"GDAL_HTTP_NETRC": "YES", "GDAL_HTTP_COOKIEFILE": ck,
            "GDAL_HTTP_COOKIEJAR": ck, "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_USE_HEAD": "NO", "GDAL_HTTP_MAX_RETRY": "3",
            "GDAL_HTTP_RETRY_DELAY": "2"}


def granule_on_grid(gr: Granule, meta: integro_grid.GridMeta, cache_dir,
                    res_m: float | None = None) -> dict[str, np.ndarray]:
    """Каналы гранулы на сетке листа, с кэшем npz (идемпотентно).

    Нулём в L1T закодирован фон повёрнутой сцены, объявленного nodata у файла
    нет — без явного указания чернота попала бы в медиану как холодная
    поверхность.
    """
    res_m = res_m or config.SAT_RES_M
    cache = Path(cache_dir) / f"{gr.id}_{res_m:.0f}m.npz"
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as z:
            have = {k: z[k] for k in z.files}
    need = [b for b in BANDS if b not in have]
    if need:
        env = edl_env()
        for b in need:
            have[b] = stac_grid.href_on_grid(f"/vsicurl/{gr.urls[b]}", meta, res_m,
                                             nodata=0.0, env=env)
        np.savez_compressed(cache, **have)
    return {b: have[b] for b in BANDS}


# ------------------------------------------------------------------ физика
def radiance(dn: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """DN -> радианс на сенсоре, Вт/(м^2 * ср * мкм): L = (DN - 1) * UCC.

    DN = 0 — не «холодно», а «нет данных», поэтому ноль уходит в NaN, а не
    в отрицательный радианс.
    """
    out = {}
    for b, a in dn.items():
        v = np.where(a > 0, (a - 1.0) * config.ASTIR_UCC[b], np.nan)
        out[b] = np.where(v > 0, v, np.nan)
    return out


def brightness_temp(rad13: np.ndarray) -> np.ndarray:
    """Яркостная температура по каналу 13, К (обращение формулы Планка)."""
    lam = config.ASTIR_LAMBDA_UM["b13"]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = config.ASTIR_C1 / (np.pi * lam ** 5 * rad13) + 1.0
        t = config.ASTIR_C2 / (lam * np.log(x))
    return np.where(np.isfinite(t) & (t > 0), t, np.nan)


def normalize(rad: dict[str, np.ndarray],
              norm_t: float | None = None) -> dict[str, np.ndarray]:
    """Нормировка Ниномии: радианс приводится к фиксированной температуре.

    Температура поверхности входит во все каналы сразу, но по-разному (закон
    Планка нелинеен по длине волны), поэтому в отношении она НЕ сокращается:
    без нормировки карта кварцевого индекса — это в первую очередь карта
    прогрева склонов. Опорным берётся канал 13: в нём эмиссия почти всех
    силикатов близка к единице, поэтому его яркостная температура — лучшая
    доступная оценка температуры поверхности.
    """
    norm_t = config.ASTIR_NORM_T_K if norm_t is None else norm_t
    t13 = brightness_temp(rad["b13"])
    out = {}
    for b, a in rad.items():
        lam = config.ASTIR_LAMBDA_UM[b]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            num = np.exp(config.ASTIR_C2 / (lam * t13)) - 1.0
            den = np.exp(config.ASTIR_C2 / (lam * norm_t)) - 1.0
            v = a * num / den
        out[b] = np.where(np.isfinite(v), v, np.nan)
    return out


def indices(nl: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Индексы Ниномии по нормированному радиансу.

    ``qi`` — кварцевый, ``ci`` — карбонатный, ``mi`` — мафический. Формулы взяты
    как опубликованы, без «упрощений»: степени в ``mi`` не сокращаются, потому
    что каналы разные.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        qi = nl["b11"] * nl["b11"] / (nl["b10"] * nl["b12"])
        ci = nl["b13"] / nl["b14"]
        mi = nl["b12"] * nl["b14"] ** 3 / nl["b13"] ** 4
    return {"qi": np.where(np.isfinite(qi), qi, np.nan),
            "ci": np.where(np.isfinite(ci), ci, np.nan),
            "mi": np.where(np.isfinite(mi), mi, np.nan)}


def scene_layers(dn: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Полный расчёт по одной сцене: DN -> нормированный радианс + индексы.

    Индексы считаются ПОСЦЕННО, а не по медианному композиту DN: медиана по
    сценам с разной температурой поверхности смешала бы разные тепловые режимы,
    и нормировка потеряла бы смысл.
    """
    nl = normalize(radiance(dn))
    lay = {f"n{b}": v for b, v in nl.items()}
    lay["t13"] = brightness_temp(radiance(dn)["b13"])
    return {**lay, **indices(nl)}


# ------------------------------------------------------------------ признаки
def fetch(meta: integro_grid.GridMeta, log=None) -> tuple[pd.DataFrame, list[Granule]]:
    """Композит ASTER TIR на сетку листа: нормированные каналы и индексы."""
    from cache_paths import ASTIR_ANABAR

    grans = search(meta)
    if not grans:
        raise RuntimeError("над листом нет подходящих гранул ASTER TIR")
    if log:
        yrs = sorted({g.datetime.year for g in grans})
        log(f"ASTER TIR: {len(grans)} гранул, годы {yrs}, "
            f"облачность {min(g.cloud for g in grans):.0f}-"
            f"{max(g.cloud for g in grans):.0f}%")
    scenes = []
    for i, g in enumerate(grans):
        scenes.append(scene_layers(granule_on_grid(g, meta, ASTIR_ANABAR)))
        if log:
            log(f"  гранула {i + 1}/{len(grans)}: {g.id} "
                f"({g.datetime.date()}, облачность {g.cloud:.0f}%)")
    comp, n_obs = stac_grid.median_stack(scenes, min_obs=config.ASTIR_MIN_OBS)
    return stac_grid.to_cells(comp, meta, n_obs=n_obs, prefix="astir_"), grans
