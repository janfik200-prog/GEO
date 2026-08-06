"""Общий слой чтения спутниковых каталогов STAC на сетку листа (этап 8).

Один модуль на все сенсоры, потому что различаются они только рецептом (какая
коллекция, какие ассеты, как перевести числа в физику), а механика одна и та же:

1. :func:`search` — поиск сцен над листом с квотой на группу;
2. :func:`read_on_grid` — чтение ассета окном сразу в проекцию сетки;
3. :func:`scene_on_grid` — сцена целиком в кэш npz (идемпотентно, с дочиткой
   недостающих каналов);
4. :func:`median_stack` — попиксельная медиана по сценам;
5. :func:`to_cells` — агрегация субпикселей в ячейки 500 м: среднее И разброс.

Каталог — Microsoft Planetary Computer: там в одном месте лежат Sentinel-1 RTC,
Landsat Collection 2, годовые мозаики ALOS PALSAR и архив ASTER, а подписывание
ссылок анонимное. Полные продукты не качаются: каждый COG читается через
``WarpedVRT`` окном под сетку, поэтому трафик измеряется мегабайтами.

Квота на ГРУППУ, а не общий лимит сцен, — принципиальна. Без неё вся выборка
уходит в самый ясный (или самый часто снимаемый) угол листа, и композит на
остальной площади строится по одной-двум датам. Что считать группой, решает
сенсор: для оптики это тайл/path, для радара — относительная орбита, потому что
ракурс съёмки задаёт геометрию теней и наложений.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, integro_grid

_CATALOG = None


def catalog():
    """Клиент Planetary Computer с автоподписью ссылок (создаётся один раз)."""
    global _CATALOG
    if _CATALOG is None:
        import planetary_computer as pc
        from pystac_client import Client

        _CATALOG = Client.open(config.SAT_STAC_URL, modifier=pc.sign_inplace)
    return _CATALOG


def bbox_wgs84(meta: integro_grid.GridMeta, pad_m: float = 5000.0) -> list[float]:
    """Охват сетки в градусах для STAC-поиска (тот же, что у Sentinel-2)."""
    from .s2_composite import bbox_wgs84 as _bbox

    return _bbox(meta, pad_m)


def _pick(items: list, n: int) -> list:
    """Отобрать ``n`` сцен из группы: по облачности, а без неё — по разбросу дат.

    У радара облачности нет, и «лучших» сцен не существует; там важно не сгустить
    выборку в один сезон одного года, поэтому берётся равномерная выборка по
    времени.
    """
    if not items or len(items) <= n:
        return list(items)
    if items[0].properties.get("eo:cloud_cover") is not None:
        return sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:n]
    order = sorted(items, key=lambda i: i.datetime)
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [order[i] for i in sorted(set(idx.tolist()))]


def search(collection: str, meta: integro_grid.GridMeta,
           years: tuple[int, ...] | None = None,
           months: tuple[int, ...] | None = None,
           query: dict | None = None,
           group_key: str | None = None,
           max_per_group: int | None = None,
           max_items: int = 2000) -> list:
    """Сцены коллекции над листом: фильтр по годам/месяцам, квота на группу.

    ``group_key`` — имя свойства STAC, по которому делятся группы (``sat:
    relative_orbit`` для Sentinel-1, ``landsat:wrs_path`` для Landsat). Если
    ``None``, квота применяется ко всей выборке.
    """
    y0 = min(years) if years else 1990
    y1 = max(years) if years else 2100
    res = catalog().search(collections=[collection], bbox=bbox_wgs84(meta),
                           datetime=f"{y0}-01-01T00:00:00Z/{y1}-12-31T23:59:59Z",
                           query=query, max_items=max_items)
    items = [i for i in res.items()
             if (years is None or i.datetime.year in years)
             and (months is None or i.datetime.month in months)]
    if max_per_group is None:
        return items
    if group_key is None:
        return _pick(items, max_per_group)
    out: list = []
    keys = sorted({str(i.properties.get(group_key)) for i in items})
    for k in keys:
        out += _pick([i for i in items if str(i.properties.get(group_key)) == k],
                     max_per_group)
    return out


def grid_transform(meta: integro_grid.GridMeta, res_m: float):
    """Аффинное преобразование рабочего растра и его размеры в пикселях."""
    from rasterio.transform import Affine

    k = int(round(meta.dx / res_m))
    return Affine(res_m, 0.0, meta.x0, 0.0, -res_m, meta.y_top), meta.prf * k, meta.pic * k


def href_on_grid(href: str, meta: integro_grid.GridMeta,
                 res_m: float | None = None, band: int = 1,
                 nodata: float | None = None, nearest: bool = False,
                 env: dict | None = None) -> np.ndarray:
    """Растр по ссылке, посаженный на сетку листа; вне данных — NaN.

    Отделено от :func:`read_on_grid` ради источников без STAC: у LP DAAC каналы
    ASTER TIR лежат обычными GeoTIFF по прямым ссылкам, а посадка на сетку нужна
    ровно та же. ``env`` — переменные GDAL на время чтения (у LP DAAC это
    авторизация Earthdata через ``_netrc`` и файл cookies).
    """
    import rasterio
    from pyproj import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    res_m = res_m or config.SAT_RES_M
    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    if proj4 is None:
        raise RuntimeError("proj4 целевой сетки не найден (sidecar .pj4)")
    transform, height, width = grid_transform(meta, res_m)
    with rasterio.Env(**(env or {})):
        with rasterio.open(href) as src:
            nd = src.nodata if nodata is None else nodata
            with WarpedVRT(src, crs=CRS.from_proj4(proj4), transform=transform,
                           height=height, width=width, dtype="float32",
                           resampling=Resampling.nearest if nearest else Resampling.average,
                           src_nodata=nd, nodata=np.nan) as vrt:
                return vrt.read(band).astype("float32")


def read_on_grid(item, asset: str, meta: integro_grid.GridMeta,
                 res_m: float | None = None, band: int = 1,
                 nodata: float | None = None, nearest: bool = False) -> np.ndarray:
    """Один канал ассета STAC, посаженный на сетку листа; вне данных — NaN.

    ``nearest`` обязателен для масок качества: усреднять номера классов нельзя,
    среднее между «облако» и «чисто» смысла не имеет.
    """
    import planetary_computer as pc

    # Подпись обновляется прямо перед чтением: у ссылок MPC короткий срок жизни,
    # а прогон по сотне сцен идёт часами.
    return href_on_grid(pc.sign(item.assets[asset].href), meta, res_m,
                        band=band, nodata=nodata, nearest=nearest)


def scene_on_grid(item, bands: dict[str, tuple[str, int]],
                  meta: integro_grid.GridMeta, cache_dir,
                  res_m: float | None = None,
                  mask_fn=None, nodata: float | None = None) -> dict[str, np.ndarray]:
    """Сцена целиком на сетке, с кэшем npz и дочиткой недостающих каналов.

    ``bands`` — ``{короткое имя: (ассет, номер канала)}``. Недостающие каналы
    дочитываются в уже существующий кэш, а не перекачивают сцену заново: список
    каналов со временем растёт (так добавился красный край у Sentinel-2), и
    полный перезалив стоил бы часов трафика.

    ``mask_fn(item, meta, res_m)`` возвращает булеву маску «пиксель годен»;
    результат применяется ко всем каналам и в кэш попадает уже маскированным.
    """
    res_m = res_m or config.SAT_RES_M
    cache = cache_dir / f"{item.id}_{res_m:.0f}m.npz"
    have: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as z:
            have = {k: z[k] for k in z.files}
    need = [b for b in bands if b not in have]
    if not need:
        return {b: have[b] for b in bands}

    keep = mask_fn(item, meta, res_m) if mask_fn is not None else None
    for short in need:
        asset, band = bands[short]
        arr = read_on_grid(item, asset, meta, res_m, band=band, nodata=nodata)
        have[short] = arr if keep is None else np.where(keep, arr, np.nan)
    np.savez_compressed(cache, **have)
    return {b: have[b] for b in bands}


def median_stack(scenes: list[dict[str, np.ndarray]],
                 min_obs: int | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Попиксельная медиана по сценам и карта числа валидных наблюдений.

    Медиана, а не среднее: одиночное облако, снежник или всплеск обратного
    рассеяния смещают среднее, но не медиану. Пиксели, где наблюдений меньше
    ``min_obs``, объявляются пустыми — по двум наблюдениям медиана не фильтрует
    ничего.
    """
    import warnings

    min_obs = config.SAT_MIN_OBS if min_obs is None else min_obs
    if not scenes:
        raise ValueError("нет сцен для композита")
    comp, n_obs = {}, None
    for b in scenes[0]:
        cube = np.stack([s[b] for s in scenes])
        with warnings.catch_warnings():     # пиксель вне всех сцен -> NaN, это норма
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(cube, axis=0)
        cnt = np.isfinite(cube).sum(axis=0)
        n_obs = cnt if n_obs is None else np.minimum(n_obs, cnt)
        comp[b] = med
    for b in comp:
        comp[b] = np.where(n_obs >= min_obs, comp[b], np.nan)
    return comp, n_obs


def to_cells(layers: dict[str, np.ndarray], meta: integro_grid.GridMeta,
             res_m: float | None = None, n_obs: np.ndarray | None = None,
             prefix: str = "") -> pd.DataFrame:
    """Субпиксели -> ячейки 500 м: среднее по ячейке и разброс внутри неё.

    Разброс — самостоятельный признак, а не служебная величина: на 500 м среднее
    сглаживает пятнистость (обнажения, курумы, микрорельеф), а std её сохраняет.
    """
    import warnings

    res_m = res_m or config.SAT_RES_M
    k = int(round(meta.dx / res_m))

    def blocks(a: np.ndarray) -> np.ndarray:
        return a[:meta.prf * k, :meta.pic * k].reshape(meta.prf, k, meta.pic, k)

    out: dict[str, np.ndarray] = {}
    first = None
    for name, arr in layers.items():
        bl = blocks(arr)
        with warnings.catch_warnings():      # ячейка целиком без данных -> NaN, это норма
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"{prefix}{name}"] = np.nanmean(bl, axis=(1, 3)).ravel()
            out[f"{prefix}{name}_std"] = np.nanstd(bl, axis=(1, 3)).ravel()
        first = arr if first is None else first
    if first is not None:
        out[f"{prefix}valid_frac"] = blocks(
            np.isfinite(first).astype("float32")).mean(axis=(1, 3)).ravel()
    if n_obs is not None:
        out[f"{prefix}n_obs"] = blocks(n_obs.astype("float32")).mean(axis=(1, 3)).ravel()
    return pd.DataFrame(out)


def to_db(power: np.ndarray, floor_db: float | None = None) -> np.ndarray:
    """Линейная мощность обратного рассеяния -> дБ.

    В дБ, а не в линейной шкале: распределение γ0 логнормально, и в линейной
    шкале среднее по ячейке определяется единичными яркими пикселями (уступ,
    валун), а не типичным уровнем поверхности.
    """
    floor_db = config.S1_FLOOR_DB if floor_db is None else floor_db
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10.0 * np.log10(np.where(power > 0, power, np.nan))
    return np.where(np.isfinite(db), np.maximum(db, floor_db), np.nan)
