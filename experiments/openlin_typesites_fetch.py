"""Загрузка входов для участков по видам линеаментов (audit/typesites.json).

Участки — небольшие окна (20–40 км), на каждом много линий одного вида: гряды
над дайками, уступы сбросов, раздвиги, полигоны мерзлоты, друмлины и т. д. Роль —
проверка полноты и мусора метода по видам и поиск его слепых мест; пороги общего
метода на них не подбираются. Роль каждого прогона объявляется отдельно.

Два шага:
  1. --preview: только GLO-30 и одна безоблачная сцена S2 (малый объём) ->
     картинка outputs/typesites/<участок>_preview.png. Глазами проверить, что вид
     линий в окне действительно есть (у координат 'уточнить' — обязательно).
  2. --full: всё для метода — S2 лето (медиана 6 каналов), S2 низкое солнце
     (только |широта| > 55°), S1 RTC VV/VH по орбитам, GLO-30, ALOS AW3D30.
     --extra добавляет проверочные слои (не входы метода): arcticdem (2 м, широта
     > 60°), 3dep (лидар США), tir (Landsat, тепловой канал).

Каждый слой — отдельный файл в datacache/typesites/<участок>/; готовый слой не
качается повторно, поэтому прерванную загрузку можно просто запустить снова.
Сетка — UTM участка, шаг --res (по умолчанию 20 м, как у листов ветки).

Запуск из корня репозитория:
    python -m experiments.openlin_typesites_fetch --list
    python -m experiments.openlin_typesites_fetch --preview ts_dike_shiprock ts_scarp_lostriver
    python -m experiments.openlin_typesites_fetch --full ts_dike_shiprock --extra 3dep
    python -m experiments.openlin_typesites_fetch --full all
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from cache_paths import CACHE

SITES_JSON = Path("audit/typesites.json")
DEST = CACHE / "typesites"
OUT = Path("outputs/typesites")
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PGC_STAC = "https://stac.pgc.umn.edu/api/v1"
YEARS = "2019-01-01/2025-10-31"
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
S2_BAD_SCL = (0, 1, 3, 8, 9, 10)     # нет данных, дефект, тени облаков, облака, перистые
LOWSUN_MAX_DEG = 12.0                # как openlin_lowsun.MAX_SUN_DEG
LOWSUN_MIN_ABS_LAT = 55.0            # южнее зимнего солнца ниже 12° не бывает
S1_NODATA = -32768.0


def load_sites() -> dict:
    return {s["name"]: s for s in json.loads(SITES_JSON.read_text(encoding="utf-8"))["sites"]}


def make_grid(site: dict, res_m: float):
    """Квадрат site['km'] вокруг центра в зоне UTM центра."""
    from pyproj import Transformer

    from experiments.openlin_probe import Grid
    lat, lon = site["lat"], site["lon"]
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = site["km"] * 500.0
    n = int(round(site["km"] * 1000 / res_m))
    return Grid(x0=x - half, y_top=y + half, res_m=res_m, height=n, width=n, epsg=epsg)


def read_on_grid(href: str, grid, resampling) -> np.ndarray:
    """Растр на сетку участка; нет данных -> NaN (нодата источника, а не 0)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.vrt import WarpedVRT
    if "blob.core.windows.net" in href:
        import planetary_computer as pc
        href = pc.sign(href.split("?")[0])
    with rasterio.open(href) as src:
        nd = src.nodata
        with WarpedVRT(src, crs=CRS.from_epsg(grid.epsg), transform=grid.transform,
                       height=grid.height, width=grid.width, resampling=resampling,
                       src_nodata=nd, nodata=np.nan if nd is None else nd,
                       dtype="float32") as vrt:
            a = vrt.read(1).astype("float32")
    if nd is not None and np.isfinite(nd):
        a[a == nd] = np.nan
    return a


def search(url, collections, grid, query=None, datetime=YEARS):
    from pystac_client import Client
    kw = {}
    if url == PC_STAC:
        import planetary_computer as pc
        kw["modifier"] = pc.sign_inplace
    for attempt in range(4):                           # обрыв связи на поиске — повтор
        try:
            cat = Client.open(url, **kw)
            return list(cat.search(collections=collections, bbox=grid.bbox_wgs84(pad_m=0),
                                   datetime=datetime, query=query).items())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def _window(grid):
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.ops import transform
    tr = Transformer.from_crs(grid.epsg, 4326, always_xy=True).transform
    return transform(tr, box(grid.x0, grid.y_top - grid.height * grid.res_m,
                             grid.x0 + grid.width * grid.res_m, grid.y_top))


def footprint_share(it, grid) -> float:
    from shapely.geometry import shape
    g = _window(grid)
    return shape(it.geometry).intersection(g).area / g.area


def date_groups(items, grid, n_max, min_cover=0.95):
    """Сцены S2 одной даты (соседние плитки одного витка) — одна съёмка.

    Окно 20–40 км часто лежит на стыке плиток (Spanish Peaks: 13SDB/13SEB, ни одна
    не накрывает больше 77 %), поэтому плитки одной даты склеиваются, а годной
    считается дата, у которой объединение плиток накрывает окно на min_cover.
    Даты — по возрастанию наибольшей облачности их плиток.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union
    g = _window(grid)
    by = {}
    for it in items:
        if shape(it.geometry).intersects(g):
            by.setdefault(it.datetime.date(), {})[it.properties.get("s2:mgrs_tile", it.id)] = it
    good = []
    for d, tiles in by.items():
        grp = list(tiles.values())
        cov = unary_union([shape(i.geometry) for i in grp]).intersection(g).area / g.area
        if cov >= min_cover:
            good.append((max(i.properties["eo:cloud_cover"] for i in grp), str(d), grp))
    good.sort(key=lambda r: (r[0], r[1]))
    return [grp for _c, _d, grp in good[:n_max]]


def months_for(lat: float):
    """Бесснежные месяцы: север 6–9, юг 12–3, между 50° ю. и 50° с. — все."""
    if lat > 50:
        return {6, 7, 8, 9}
    if lat < -50:
        return {12, 1, 2, 3}
    return set(range(1, 13))


# ------------------------------------------------------------------ слои
def layer_dem(collection: str):
    def run(site, grid, log):
        from rasterio.enums import Resampling
        items = search(PC_STAC, [collection], grid, datetime=None)
        if not items:
            raise RuntimeError(f"{collection}: плиток нет")
        acc = np.full(grid.shape, np.nan, "float32")
        for it in items:
            try:
                a = read_on_grid(it.assets["data"].href, grid, Resampling.bilinear)
            except Exception as exc:                   # битая плитка не валит слой
                log(f"    {collection} {it.id} ошибка {exc!r:.100}")
                continue
            acc = np.where(np.isnan(acc), a, acc)
        log(f"    {collection}: плиток {len(items)}, пусто {100 * np.isnan(acc).mean():.1f} %")
        return {"dem": acc}, [it.id for it in items]
    return run


def _s2_stack(groups, grid, bands, log, tag, bad_scl=S2_BAD_SCL):
    """Медиана по датам; внутри даты плитки склеиваются (первая годная ячейка)."""
    from rasterio.enums import Resampling
    stacks = {b: [] for b in bands}
    used = []
    for i, grp in enumerate(groups, 1):
        day = {b: np.full(grid.shape, np.nan, "float32") for b in bands}
        n_ok = 0
        for it in grp:
            try:
                scl = read_on_grid(it.assets["SCL"].href, grid, Resampling.nearest)
                bad = np.isnan(scl) | np.isin(scl, bad_scl)
                for b in bands:
                    a = np.where(bad, np.nan, read_on_grid(it.assets[b].href, grid, Resampling.average))
                    day[b] = np.where(np.isnan(day[b]), a, day[b])
                used.append(it.id)
                n_ok += 1
            except Exception as exc:                   # плитка пропускается, не весь слой
                log(f"    {tag}: {it.id} ошибка {exc!r:.100}")
        if n_ok:
            for b in bands:
                stacks[b].append(day[b])
            log(f"    {tag}: дата {i}/{len(groups)} {grp[0].datetime.date()}, плиток {n_ok}/{len(grp)}")
    if not used:
        raise RuntimeError(f"{tag}: ни одной сцены не прочитано")
    out = {b: np.nanmedian(np.stack(v), 0).astype("float32") for b, v in stacks.items()}
    log(f"    {tag}: сцен {len(used)}, пусто {100 * np.isnan(out[bands[0]]).mean():.1f} %")
    return out, used


def _distinct_dates(items, n):
    seen, out = set(), []
    for it in items:
        d = it.datetime.date()
        if d not in seen:
            seen.add(d)
            out.append(it)
        if len(out) == n:
            break
    return out


def layer_s2_summer(n_max: int):
    def run(site, grid, log):
        ms = months_for(site["lat"])
        items = [it for it in search(PC_STAC, ["sentinel-2-l2a"], grid,
                                     query={"eo:cloud_cover": {"lt": 10}})
                 if it.datetime.month in ms]
        items = date_groups(items, grid, n_max)
        log(f"    S2 лето: годных дат {len(items)} (месяцы {sorted(ms)})")
        # летом снег и лёд (SCL 11) — помеха, отсеиваются; на низком солнце снег — фон снимка
        return _s2_stack(items, grid, S2_BANDS, log, "S2 лето", S2_BAD_SCL + (11,))
    return run


def layer_s2_lowsun(n_max: int):
    def run(site, grid, log):
        if abs(site["lat"]) < LOWSUN_MIN_ABS_LAT:
            raise RuntimeError("низкого солнца на этой широте нет — слой не нужен")
        items = []
        for it in search(PC_STAC, ["sentinel-2-l2a"], grid, query={"eo:cloud_cover": {"lt": 10}}):
            sz = it.properties.get("s2:mean_solar_zenith")
            if sz is None or not 0 < 90 - float(sz) <= LOWSUN_MAX_DEG:
                continue
            items.append(it)
        items = date_groups(items, grid, n_max)
        az = [float(np.mean([float(i.properties.get("s2:mean_solar_azimuth", np.nan)) for i in g]))
              for g in items]
        log(f"    S2 низкое солнце: дат {len(items)}, азимуты солнца {np.round(az).tolist()}")
        out, used = _s2_stack(items, grid, ("B04", "B08"), log, "S2 низкое солнце")
        out["sun_azimuth"] = np.array(az, "float32")
        return out, used
    return run


def layer_s2_img(n_max: int):
    """Красный канал по датам -> композит и две половины для тракта (TS1).

    Как openlin_winter / openlin_zone_lines_v3.build_halves: каждая дата
    нормируется (openlin_winter.normalize), композит — медиана при >= MIN_SCENES
    датах, половины — даты через одну (>= 1 даты), всё минус фон 2 км (detrend).
    Север (|широта| >= 55° и >= 2 дат с солнцем ниже 12°) — низкое солнце, тени;
    иначе — лето (подмена, объявлена в audit/TS1_DECLARATION.md).
    """
    def run(site, grid, log):
        from experiments.openlin_winter import MIN_SCENES, detrend, normalize
        all_s2 = search(PC_STAC, ["sentinel-2-l2a"], grid, query={"eo:cloud_cover": {"lt": 10}})
        groups, kind, bad = [], "лето", S2_BAD_SCL + (11,)
        if abs(site["lat"]) >= LOWSUN_MIN_ABS_LAT:
            low = [it for it in all_s2 if it.properties.get("s2:mean_solar_zenith") is not None
                   and 0 < 90 - float(it.properties["s2:mean_solar_zenith"]) <= LOWSUN_MAX_DEG]
            groups = date_groups(low, grid, n_max)
            if len(groups) >= 2:
                kind, bad = "низкое солнце", S2_BAD_SCL
        if kind == "лето":
            ms = months_for(site["lat"])
            groups = date_groups([it for it in all_s2 if it.datetime.month in ms], grid, n_max)
        groups = sorted(groups, key=lambda g: g[0].datetime)          # половины — даты через одну
        log(f"    S2 тракт: {kind}, дат {len(groups)}")
        z, used = _s2_stack_dates(groups, grid, log, bad)
        layers = [normalize(b, np.isfinite(b) & (b > 0)) for b in z]
        cube = np.stack(layers)
        n = np.isfinite(cube).sum(0)

        def med(c, nmin):
            k = np.isfinite(c).sum(0)
            with np.errstate(invalid="ignore"):
                m = np.nanmedian(c, 0)
            m = np.where(k >= nmin, m, np.nan).astype("float32")
            return detrend(m, np.isfinite(m), grid.res_m)
        out = {"shade": med(cube, MIN_SCENES), "half_A": med(cube[0::2], 1),
               "half_B": med(cube[1::2], 1), "count": n.astype("int16"),
               "kind": np.array(kind),
               "sun_elev": np.array([90 - float(np.mean([float(i.properties.get("s2:mean_solar_zenith", np.nan))
                                                          for i in g])) for g in groups], "float32"),
               "sun_az": np.array([float(np.mean([float(i.properties.get("s2:mean_solar_azimuth", np.nan))
                                                  for i in g])) for g in groups], "float32"),
               "dates": np.array([str(g[0].datetime.date()) for g in groups])}
        log(f"    S2 тракт: композит пусто {100 * np.isnan(out['shade']).mean():.1f} %, "
            f"половины {100 * np.isnan(out['half_A']).mean():.1f} / {100 * np.isnan(out['half_B']).mean():.1f} %")
        return out, used
    return run


def _s2_stack_dates(groups, grid, log, bad_scl):
    """B04 по датам (плитки даты склеены), без медианы."""
    from rasterio.enums import Resampling
    out, used = [], []
    for grp in groups:
        day = np.full(grid.shape, np.nan, "float32")
        for it in grp:
            try:
                scl = read_on_grid(it.assets["SCL"].href, grid, Resampling.nearest)
                a = read_on_grid(it.assets["B04"].href, grid, Resampling.average)
                a = np.where(np.isnan(scl) | np.isin(scl, bad_scl), np.nan, a)
                day = np.where(np.isnan(day), a, day)
                used.append(it.id)
            except Exception as exc:
                log(f"    S2 тракт: {it.id} ошибка {exc!r:.100}")
        if np.isfinite(day).any():
            out.append(day)
        log(f"    S2 тракт: дата {len(out)}/{len(groups)} {grp[0].datetime.date()}, плиток {len(grp)}")
    if len(out) < 2:
        raise RuntimeError("S2 тракт: меньше двух дат — половин не будет")
    return out, used


def layer_s1(per_orbit: int):
    def run(site, grid, log):
        from rasterio.enums import Resampling
        ms = months_for(site["lat"])
        items = [it for it in search(PC_STAC, ["sentinel-1-rtc"], grid)
                 if it.datetime.month in ms and footprint_share(it, grid) > 0.95]
        by = {}
        for it in items:
            key = (it.properties.get("sat:relative_orbit", -1), it.properties.get("sat:orbit_state", "?"))
            by.setdefault(key, []).append(it)
        log(f"    S1: сцен {len(items)}, орбит {len(by)}: {sorted(by)}")
        out, used = {}, []
        for pol in ("vv", "vh"):
            for (orb, state), group in sorted(by.items()):
                stack = []
                for it in sorted(group, key=lambda x: x.datetime)[-per_orbit:]:
                    if pol not in it.assets:
                        continue
                    try:
                        a = read_on_grid(it.assets[pol].href, grid, Resampling.average)
                    except Exception as exc:
                        log(f"    S1 {it.id} ошибка {exc!r:.100}")
                        continue
                    stack.append(np.where((a > 0) & (a != S1_NODATA), a, np.nan))
                    used.append(it.id)
                if stack:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        out[f"{pol}_o{orb}_{state}"] = (10 * np.log10(np.nanmedian(np.stack(stack), 0))).astype("float32")
                    log(f"    S1 {pol} орбита {orb} {state}: сцен {len(stack)}")
        if not out:
            raise RuntimeError("S1: ни одной сцены не прочитано")
        return out, used
    return run


def layer_arcticdem(site, grid, log):
    from rasterio.enums import Resampling
    if site["lat"] < 60:
        raise RuntimeError("ArcticDEM только севернее 60°")
    items = search(PGC_STAC, ["arcticdem-mosaics-v4.1-2m"], grid, datetime=None)
    if not items:
        raise RuntimeError("ArcticDEM: плиток нет")
    acc = np.full(grid.shape, np.nan, "float32")
    for it in items:
        a = read_on_grid(it.assets["dem"].href, grid, Resampling.average)
        acc = np.where(np.isnan(acc), a, acc)
    log(f"    ArcticDEM: плиток {len(items)}, пусто {100 * np.isnan(acc).mean():.1f} %")
    return {"dem": acc}, [it.id for it in items]


def layer_3dep(site, grid, log):
    from rasterio.enums import Resampling
    items = search(PC_STAC, ["3dep-lidar-dtm"], grid, datetime=None)
    if not items:
        raise RuntimeError("3DEP: лидара здесь нет (только США)")
    acc = np.full(grid.shape, np.nan, "float32")
    for it in items:
        try:
            a = read_on_grid(it.assets["data"].href, grid, Resampling.average)
        except Exception as exc:
            log(f"    3DEP {it.id} ошибка {exc!r:.100}")
            continue
        acc = np.where(np.isnan(acc), a, acc)
    log(f"    3DEP: плиток {len(items)}, пусто {100 * np.isnan(acc).mean():.1f} %")
    return {"dem": acc}, [it.id for it in items]


def layer_tir(n_max: int):
    def run(site, grid, log):
        from rasterio.enums import Resampling
        ms = months_for(site["lat"])
        items = [it for it in search(PC_STAC, ["landsat-c2-l2"], grid,
                                     query={"eo:cloud_cover": {"lt": 10},
                                            "platform": {"in": ["landsat-8", "landsat-9"]}})
                 if it.datetime.month in ms and "lwir11" in it.assets]
        items = _distinct_dates(sorted(items, key=lambda it: it.properties["eo:cloud_cover"]), n_max)
        stack, used = [], []
        for it in items:
            try:
                a = read_on_grid(it.assets["lwir11"].href, grid, Resampling.nearest)
            except Exception as exc:
                log(f"    TIR {it.id} ошибка {exc!r:.100}")
                continue
            a = np.where((a >= 1) & (a < 65535), a, np.nan)  # 0 — нет данных, 65535 — насыщение
            stack.append(a * 0.00341802 + 149.0 - 273.15)   # коэффициенты Collection 2 L2 ST, °C
            used.append(it.id)
        if not stack:
            raise RuntimeError("TIR: ни одной сцены не прочитано")
        log(f"    TIR: сцен {len(used)}")
        return {"st_c": np.nanmedian(np.stack(stack), 0).astype("float32")}, used
    return run


# ------------------------------------------------------------------ шаги
def save_layer(d: Path, name: str, fn, site, grid, log, meta):
    f = d / f"{name}.npz"
    if f.exists():
        log(f"  {name}: уже есть — пропуск")
        return
    t0 = time.time()
    try:
        arrs, ids = fn(site, grid, log)
    except Exception as exc:
        log(f"  {name}: НЕ ПОЛУЧЕН — {exc!r:.200}")
        meta["layers"][name] = {"ошибка": repr(exc)[:300]}
        return
    tmp = d / f"{name}.part.npz"                  # обрыв при записи не оставит «готовый» битый слой
    np.savez_compressed(tmp, **arrs)
    tmp.replace(f)
    meta["layers"][name] = {"сцены": ids, "секунд": round(time.time() - t0)}
    log(f"  {name}: сохранено {f} ({time.time() - t0:.0f} с)")


def hillshade(dem, res, az, alt=30.0):
    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmedian(dem)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(90 - alt)
    return np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)


def preview(name, site, grid, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.enums import Resampling
    d = DEST / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"site": site, "res_m": grid.res_m, "epsg": grid.epsg, "layers": {}}
    save_layer(d, "glo30", layer_dem("cop-dem-glo-30"), site, grid, log, meta)
    groups = date_groups([it for it in search(PC_STAC, ["sentinel-2-l2a"], grid,
                                              query={"eo:cloud_cover": {"lt": 5}})
                          if it.datetime.month in months_for(site["lat"])], grid, 1)
    rgb = None
    if groups:
        rgb = np.full(grid.shape + (3,), np.nan, "float32")
        for it in groups[0]:
            a = np.dstack([read_on_grid(it.assets[b].href, grid, Resampling.average)
                           for b in ("B04", "B03", "B02")])
            rgb = np.where(np.isnan(rgb), a, rgb)
        lo, hi = np.nanpercentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
        log(f"  S2 для просмотра: {groups[0][0].datetime.date()}, плиток {len(groups[0])}")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    if (d / "glo30.npz").exists():
        dem = np.load(d / "glo30.npz")["dem"]
        for k, az in enumerate((315, 45)):
            ax[k].imshow(hillshade(dem, grid.res_m, az), cmap="gray")
            ax[k].set_title(f"GLO-30, отмывка, солнце {az}°")
    if rgb is not None:
        ax[2].imshow(np.nan_to_num(rgb))
        ax[2].set_title("Sentinel-2, цвет")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{name}: {site['type']} — {site['km']} км, координаты: {site['coord']}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}_preview.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"  картинка: {p}")


def full(name, site, grid, log, extras, a):
    d = DEST / name
    d.mkdir(parents=True, exist_ok=True)
    mp = d / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"layers": {}}
    meta.update({"site": site, "res_m": grid.res_m, "epsg": grid.epsg, "x0": grid.x0,
                 "y_top": grid.y_top, "shape": list(grid.shape)})
    layers = {"glo30": layer_dem("cop-dem-glo-30"), "alos": layer_dem("alos-dem"),
              "s2_summer": layer_s2_summer(a.s2_max), "s2_lowsun": layer_s2_lowsun(a.s2_max),
              "s1": layer_s1(a.s1_per_orbit), "s2_img": layer_s2_img(a.s2_max)}
    extra_fns = {"arcticdem": layer_arcticdem, "3dep": layer_3dep, "tir": layer_tir(a.s2_max)}
    for e in extras:
        layers[e] = extra_fns[e]
    for ln, fn in layers.items():
        save_layer(d, ln, fn, site, grid, log, meta)
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sites", nargs="*", help="имена участков или all")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--list", action="store_true")
    g.add_argument("--preview", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], choices=["arcticdem", "3dep", "tir"])
    ap.add_argument("--res", type=float, default=20.0)
    ap.add_argument("--s2-max", type=int, default=8, help="сцен S2 на медиану")
    ap.add_argument("--s1-per-orbit", type=int, default=3, help="сцен S1 на орбиту")
    ap.add_argument("--utsjoki-control", action="store_true",
                    help="только слой s2_img на сетке листа Утсйоки (контроль (б) TS1)")
    a = ap.parse_args()
    for k, v in (("GDAL_HTTP_TIMEOUT", "60"), ("GDAL_HTTP_CONNECTTIMEOUT", "20"),
                 ("GDAL_HTTP_MAX_RETRY", "3"), ("GDAL_HTTP_RETRY_DELAY", "5")):
        os.environ.setdefault(k, v)
    sites = load_sites()
    if a.utsjoki_control:                              # audit/TS1_DECLARATION.md, поправка 1 п. 4(б)
        from experiments.openlin_winter import site_grid
        grid = site_grid("utsjoki", 20.0)
        d = DEST / "_utsjoki"
        d.mkdir(parents=True, exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
        meta = {"site": "utsjoki", "res_m": 20.0, "epsg": grid.epsg, "x0": grid.x0,
                "y_top": grid.y_top, "shape": list(grid.shape), "layers": {}}
        save_layer(d, "s2_img", layer_s2_img(a.s2_max), {"lat": 69.551, "lon": 27.142},
                   grid, print, meta)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if not (a.list or a.preview or a.full):
        raise SystemExit("нужен один из ключей: --list, --preview, --full, --utsjoki-control")
    if a.list:
        for n, s in sites.items():
            print(f"{n:28s} {s['kind']:22s} {s['coord']:9s} {s['km']:3d} км  {s['type']}")
        return
    names = list(sites) if a.sites == ["all"] else a.sites
    if not names:
        raise SystemExit("укажите участки или all")
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "_fetch.log").open("a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()
    for n in names:
        if n not in sites:
            log(f"{n}: нет в {SITES_JSON} — пропуск")
            continue
        s = sites[n]
        if s["lat"] is None:
            log(f"{n}: координаты не заданы ({s['src']}) — пропуск")
            continue
        grid = make_grid(s, a.res)
        log(f"=== {n}: {s['type']}; {s['lat']}, {s['lon']}; EPSG {grid.epsg}; {grid.shape}")
        try:
            if a.preview:
                preview(n, s, grid, log)
            else:
                full(n, s, grid, log, a.extra, a)
        except Exception as exc:                       # один участок не валит остальные
            log(f"  {n}: СБОЙ — {exc!r:.300}")


if __name__ == "__main__":
    main()
