"""Композит Sentinel-2 L2A на сетку листа: поиск сцен, маски, медиана, признаки.

Зачем свои снимки, если в v1 уже есть Landsat 7: Landsat 7 после отказа SLC
(2003) идёт полосами, у него нет узких SWIR-каналов 20 м и нет пиксельной маски
качества уровня SCL, а главное — в v1 он лежит одиночной сценой одной даты, то
есть несёт погоду и освещение конкретного дня. Здесь строится МЕДИАННЫЙ композит
по многим летним датам: то, что не повторяется от года к году (облака, тени,
снежники, сезонная вода), медианой вырезается, а остаётся устойчивый спектр
поверхности.

Источник — публичные COG Sentinel-2 L2A (bucket ``sentinel-cogs``, каталог STAC
earth-search), без авторизации. Полные продукты .SAFE не скачиваются: каждый COG
читается окном сразу в проекцию сетки с шагом ``S2_RES_M`` через ``WarpedVRT``,
поэтому объём трафика измеряется мегабайтами, а не гигабайтами.

Порядок работы:

1. :func:`search_scenes` — STAC-поиск по охвату листа, летним месяцам и
   облачности; отбор самых чистых сцен на тайл;
2. :func:`scene_on_grid` — чтение каналов и SCL сцены на сетку, применение
   :func:`scl_mask` и :func:`veg_mask` (кэш ``datacache/s2_anabar``);
3. :func:`median_composite` — попиксельная медиана по сценам + число наблюдений;
4. :func:`band_ratios` — минералогические отношения каналов;
5. :func:`to_grid_features` — среднее И стандартное отклонение по ячейке 500 м.

Отношения каналов, а не сами каналы, — потому что при низком солнце (71° с.ш.)
абсолютная яркость сильно зависит от освещённости склона, а отношение двух
каналов от неё почти свободно.

ВАЖНОЕ ОГРАНИЧЕНИЕ (измерено, а не предположено). Лист покрыт сплошной тундрой:
на безоблачной сцене 06.08.2025 SCL относит 98.3% пикселей к растительности,
NDVI по снимку 0.56-0.66, ниже 0.35 — 0.3% пикселей. Значит прямое
минералогическое дешифрирование по оптике здесь невозможно: маска «оставить
только обнажённую породу» оставляет пустую карту. Поэтому вегетационная маска
выключена (``S2_APPLY_VEG_MASK``), а группа признаков ``s2_*`` честно называется
геоботанической: растительность отражает субстрат (влажность, гранулометрию,
химизм), и это косвенный, но не нулевой геологический сигнал. Полезность группы
решается абляцией на этапе 4, а не верой в спутник.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import config, integro_grid

#: Минералогические отношения каналов (числитель, знаменатель) и смысл.
RATIOS: dict[str, tuple[str, str]] = {
    "s2_iron_ox": ("b04", "b02"),    # оксиды железа (гематит/гётит): красный/синий
    "s2_ferrous": ("b11", "b8a"),    # закисное железо: SWIR1/NIR
    "s2_clay": ("b11", "b12"),       # глинистые/Al-OH (гидротермальный аргиллизит)
    "s2_ferric": ("b04", "b8a"),     # окисное железо
}


def bbox_wgs84(meta: integro_grid.GridMeta, pad_m: float = 5000.0) -> list[float]:
    """Охват сетки в градусах (для STAC-поиска), с полем ``pad_m``."""
    from pyproj import CRS, Transformer

    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    if proj4 is None:
        raise RuntimeError("proj4 целевой сетки не найден (sidecar .pj4)")
    tr = Transformer.from_crs(CRS.from_proj4(proj4), CRS.from_epsg(4326), always_xy=True)
    xs = [meta.x0 - pad_m, meta.x0 + meta.pic * meta.dx + pad_m]
    ys = [meta.y_top + pad_m, meta.y_top - meta.prf * meta.dy - pad_m]
    lons, lats = zip(*[tr.transform(x, y) for x in xs for y in ys])
    return [min(lons), min(lats), max(lons), max(lats)]


def search_scenes(meta: integro_grid.GridMeta, max_per_tile: int | None = None) -> list[dict]:
    """Летние сцены S2 L2A над листом, самые чистые — по ``max_per_tile`` на тайл.

    Ограничение на тайл, а не общее: тайлы покрывают лист кусками, и без квоты
    вся выборка ушла бы в один самый ясный угол листа.
    """
    max_per_tile = max_per_tile or config.S2_MAX_SCENES_PER_TILE
    y0, y1 = min(config.S2_YEARS), max(config.S2_YEARS)
    query = {
        "collections": [config.S2_COLLECTION],
        "bbox": bbox_wgs84(meta),
        "datetime": f"{y0}-01-01T00:00:00Z/{y1}-12-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": config.S2_MAX_CLOUD}},
        "limit": 100,
    }
    feats, url, body = [], config.S2_STAC_URL, dict(query)
    while url:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            page = json.load(resp)
        feats += page["features"]
        nxt = [l for l in page.get("links", []) if l.get("rel") == "next"]
        if not nxt or len(feats) >= 2000:
            break
        url, body = nxt[0]["href"], nxt[0].get("body", body)

    scenes = []
    for f in feats:
        p = f["properties"]
        month = int(p["datetime"][5:7])
        if month not in config.S2_MONTHS or int(p["datetime"][:4]) not in config.S2_YEARS:
            continue
        scenes.append({"id": f["id"], "tile": f["id"].split("_")[1],
                       "datetime": p["datetime"], "cloud": float(p["eo:cloud_cover"]),
                       "assets": {k: v["href"] for k, v in f["assets"].items()}})
    out: list[dict] = []
    for tile in sorted({s["tile"] for s in scenes}):
        got = sorted((s for s in scenes if s["tile"] == tile), key=lambda s: s["cloud"])
        out += got[:max_per_tile]
    return out


def scl_mask(scl: np.ndarray) -> np.ndarray:
    """True там, где класс сцены пригоден для геологии (``S2_SCL_KEEP``)."""
    return np.isin(np.rint(scl), np.asarray(config.S2_SCL_KEEP))


def veg_mask(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """True там, где NDVI ниже порога: спектр породы, а не тундрового покрова."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    return np.nan_to_num(ndvi, nan=1.0) < config.S2_NDVI_VEG_THR


def _grid_transform(meta: integro_grid.GridMeta, res_m: float):
    from rasterio.transform import Affine

    k = int(round(meta.dx / res_m))
    return Affine(res_m, 0.0, meta.x0, 0.0, -res_m, meta.y_top), meta.prf * k, meta.pic * k


def scene_on_grid(scene: dict, meta: integro_grid.GridMeta,
                  res_m: float | None = None, use_cache: bool = True,
                  apply_veg: bool | None = None) -> dict[str, np.ndarray]:
    """Каналы одной сцены на сетке листа, вне маски — NaN. Кэш на диск (npz).

    Читается через ``WarpedVRT``: перепроекция COG в метрику сетки идёт лениво,
    с диска тянутся только нужные тайлы нужного уровня обзора.
    """
    from cache_paths import S2_ANABAR

    res_m = res_m or config.S2_RES_M
    cache = S2_ANABAR / f"{scene['id']}_{res_m:.0f}m.npz"
    want = list(config.S2_BANDS.values())
    have: dict[str, np.ndarray] = {}
    if use_cache and cache.exists():
        with np.load(cache) as z:
            have = {k: z[k] for k in z.files}
        # Список каналов растёт со временем (так добавился красный край): сцена
        # не перекачивается целиком, дочитываются только недостающие каналы.
        if all(b in have for b in want):
            return {b: have[b] for b in want}

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from pyproj import CRS

    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    dst_crs = CRS.from_proj4(proj4)
    transform, height, width = _grid_transform(meta, res_m)

    def read(asset: str) -> np.ndarray:
        with rasterio.open(scene["assets"][asset]) as src:
            with WarpedVRT(src, crs=dst_crs, transform=transform,
                           height=height, width=width,
                           resampling=Resampling.average,
                           src_nodata=0, nodata=0) as vrt:
                return vrt.read(1).astype("float32")

    need = {asset: short for asset, short in config.S2_BANDS.items() if short not in have}
    bands = {short: read(asset) for asset, short in need.items()}
    scl = read("scl")
    keep = scl_mask(scl)
    if config.S2_APPLY_VEG_MASK if apply_veg is None else apply_veg:
        nir = bands.get("b8a", have.get("b8a"))
        red = bands.get("b04", have.get("b04"))
        keep = keep & veg_mask(nir, red)
    for k, v in bands.items():
        v[v <= 0] = np.nan                     # 0 = вне снимка / нет данных
        bands[k] = np.where(keep, v, np.nan)
    have.update(bands)
    if use_cache:
        np.savez_compressed(cache, **have)
    return {b: have[b] for b in want}


def median_composite(scenes: list[dict], meta: integro_grid.GridMeta,
                     res_m: float | None = None,
                     progress=None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Попиксельная медиана по сценам и карта числа валидных наблюдений.

    Медиана, а не среднее: одиночное облако или снежник смещают среднее, но не
    медиану. Пиксели с числом наблюдений меньше ``S2_MIN_OBS`` объявляются
    невалидными — по двум наблюдениям медиана ничего не фильтрует.
    """
    res_m = res_m or config.S2_RES_M
    stack: dict[str, list[np.ndarray]] = {b: [] for b in config.S2_BANDS.values()}
    for i, sc in enumerate(scenes):
        arr = scene_on_grid(sc, meta, res_m)
        for b in stack:
            stack[b].append(arr[b])
        if progress:
            progress(i + 1, len(scenes), sc)
    comp, n_obs = {}, None
    for b, layers in stack.items():
        cube = np.stack(layers)
        with np.errstate(all="ignore"):
            med = np.nanmedian(cube, axis=0)
        cnt = np.isfinite(cube).sum(axis=0)
        n_obs = cnt if n_obs is None else np.minimum(n_obs, cnt)
        comp[b] = med
    for b in comp:
        comp[b] = np.where(n_obs >= config.S2_MIN_OBS, comp[b], np.nan)
    return comp, n_obs


def band_ratios(comp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Минералогические отношения каналов + NDVI (как индикатор задернованности)."""
    out = {}
    for name, (num, den) in RATIOS.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            r = comp[num] / comp[den]
        out[name] = np.where(np.isfinite(r), r, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = (comp["b8a"] - comp["b04"]) / (comp["b8a"] + comp["b04"])
    out["s2_ndvi"] = np.where(np.isfinite(nd), nd, np.nan)
    return out


def to_grid_features(comp: dict[str, np.ndarray], ratios: dict[str, np.ndarray],
                     n_obs: np.ndarray, meta: integro_grid.GridMeta,
                     res_m: float | None = None) -> pd.DataFrame:
    """Агрегация на ячейки 500 м: среднее по ячейке + разброс внутри ячейки.

    Разброс (std субпикселей) — самостоятельный признак: на 500 м среднее
    сглаживает пятнистость обнажений, а разброс её сохраняет. Плюс
    ``s2_valid_frac`` — доля субпикселей с данными: без неё ячейка с одним
    случайным просветом среди облаков выглядит как полноценное измерение.
    """
    res_m = res_m or config.S2_RES_M
    k = int(round(meta.dx / res_m))

    def blocks(a: np.ndarray) -> np.ndarray:
        return a[:meta.prf * k, :meta.pic * k].reshape(meta.prf, k, meta.pic, k)

    import warnings

    out: dict[str, np.ndarray] = {}
    for name, arr in {**{f"s2_{b}": v for b, v in comp.items()}, **ratios}.items():
        bl = blocks(arr)
        with warnings.catch_warnings():          # ячейка целиком под облаком -> NaN, это норма
            warnings.simplefilter("ignore", RuntimeWarning)
            out[name] = np.nanmean(bl, axis=(1, 3)).ravel()
            out[f"{name}_std"] = np.nanstd(bl, axis=(1, 3)).ravel()
    valid = blocks(np.isfinite(comp["b04"]).astype("float32"))
    out["s2_valid_frac"] = valid.mean(axis=(1, 3)).ravel()
    out["s2_n_obs"] = blocks(n_obs.astype("float32")).mean(axis=(1, 3)).ravel()
    # Доля наименее задернованных субпикселей: под сплошной тундрой это
    # единственный доступный аналог «обнажённости» — не абсолютный порог NDVI
    # (ниже него на листе почти нет пикселей), а нижний квантиль по самому листу.
    ndvi = ratios["s2_ndvi"]
    if np.isfinite(ndvi).any():
        thr = float(np.nanquantile(ndvi, config.S2_BARE_QUANTILE))
        out["s2_bare_frac"] = blocks((ndvi < thr).astype("float32")).mean(axis=(1, 3)).ravel()
    else:
        out["s2_bare_frac"] = np.full(meta.prf * meta.pic, np.nan)
    return pd.DataFrame(out)
