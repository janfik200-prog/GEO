"""Радарный композит с сохранённым ракурсом: недостающие азимуты подсветки.

ОТКУДА ПОСТАНОВКА. Детектор уступов по зимнему композиту (:mod:`openlin_shade`)
нашёл 227 настоящих элементов, но из них 189 широтных и РОВНО НОЛЬ
меридиональных. Это не свойство местности: на широте 71 низкое солнце бывает
только около полудня и только с юга, поэтому структура, идущая на солнце, тени
не отбрасывает вовсе. Две трети углового пространства оптике недоступны в
принципе, и никакой обработкой это не чинится.

Радар подсвечивает местность СБОКУ, почти перпендикулярно витку. Sentinel-1
смотрит вправо по ходу: на восходящем витке аппарат идёт на север и смотрит на
восток, то есть подсветка приходит с запада (азимут около 257°); на нисходящем
— наоборот, с востока (около 103°). Это ровно те азимуты, которых нет у солнца,
и это по-прежнему космоснимок, а не сторонний материал.

ПОЧЕМУ ВИТКИ НЕ СМЕШИВАЮТСЯ. Боевой радарный композит проекта (``S1_*`` в
``src/config.py``) берёт медиану по всем орбитам сразу — и правильно делает,
потому что там нужна устойчивая мера шероховатости. Здесь нужна ровно обратная
вещь: складывать только снимки с ОДНИМ ракурсом. Смешать восходящие с
нисходящими — это та же ошибка, что медиана сорока по-разному освещённых летних
сцен: подсветка с запада и подсветка с востока взаимно гасят друг друга, и
рельефный рисунок исчезает.

ОГОВОРКА ПРО RTC. Коллекция ``sentinel-1-rtc`` радиометрически выровнена по
рельефу с помощью ЦМР в 30 м, то есть крупный рельефный вклад из неё уже
вычтен. Для нашей задачи это работает ПРОТИВ нас, а не за: уступы в первые
десятки метров мельче ЦМР и выравниванием не затрагиваются, но всё, что крупнее,
ослаблено. Если сигнала не окажется, следующая попытка — GRD без выравнивания.

Запуск из корня репозитория:
    python -m experiments.openlin_radar --site anabar --res 20
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from experiments.openlin_winter import detrend, site_grid

CACHE = Path("datacache/s1_look")
OUT = CACHE / "composite"

#: Азимут подсветки, градусы: откуда радар освещает местность. Sentinel-1
#: смотрит вправо по ходу, курс на восходящем витке около 347°, на нисходящем
#: около 193°, отсюда луч на 77° и 283°, а подсветка — с противоположной
#: стороны. Ошибка в несколько градусов роли не играет: она меньше шага
#: перебора простираний.
ILLUM_AZ = {"ascending": 257.0, "descending": 103.0}

#: Месяцы съёмки. Те же, что в боевом конфиге: снег искажает обратное
#: рассеяние сильнее, чем облака — оптику.
MONTHS = (6, 7, 8, 9)

#: Сколько сцен брать на каждый ракурс и сколько минимум должно накрыть ячейку.
N_SCENES = 10
MIN_SCENES = 3

#: Поляризация. Для рельефного рисунка хватает VV: кросс-поляризация сильнее
#: реагирует на объёмное рассеяние растительностью, а не на геометрию склона.
POL = "vv"

NODATA = -32768.0
FLOOR_DB = -35.0


def find_scenes(grid, log) -> dict[str, list]:
    """Сцены Sentinel-1 RTC, разложенные по направлению витка."""
    import planetary_computer as pc
    from pystac_client import Client
    from shapely.geometry import shape

    from experiments.openlin_probe import STAC_URL
    from experiments.openlin_winter import frag_polygon

    fp = frag_polygon(grid)
    cat = Client.open(STAC_URL, modifier=pc.sign_inplace)
    res = cat.search(collections=["sentinel-1-rtc"], bbox=grid.bbox_wgs84(),
                     datetime="2018-01-01/2025-12-31")
    out: dict[str, list] = {"ascending": [], "descending": []}
    for it in res.items():
        if it.datetime.month not in MONTHS:
            continue
        state = str(it.properties.get("sat:orbit_state", "")).lower()
        if state not in out or POL not in it.assets:
            continue
        cover = shape(it.geometry).intersection(fp).area / fp.area
        if cover < 0.30:
            continue
        out[state].append((cover, it))
    for k, v in out.items():
        v.sort(key=lambda r: -r[0])
        log(f"  {k}: сцен {len(v)}")
    return out


def fetch(items: list, grid, log, n_max: int) -> list[np.ndarray]:
    """Скачать сцены одного ракурса на сетку листа."""
    from rasterio.enums import Resampling

    from experiments.openlin_probe import _read_on_grid

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "90")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
    CACHE.mkdir(parents=True, exist_ok=True)

    out, seen = [], set()
    for _cov, it in items:
        d = it.datetime.date()
        if d in seen:
            continue
        seen.add(d)
        p = CACHE / f"{it.id}_{int(grid.res_m)}m.npy"
        if p.exists():
            out.append(np.load(p))
            log(f"    {it.datetime:%Y-%m-%d} — из кэша")
        else:
            t0 = time.time()
            try:
                a = _read_on_grid(it.assets[POL].href, grid,
                                  Resampling.average)
            except Exception as e:                       # сеть или битый тайл
                log(f"    {it.datetime:%Y-%m-%d} — пропуск ({type(e).__name__})")
                continue
            a = np.where((a > 0) & (a != NODATA), a, np.nan).astype("float32")
            np.save(p, a)
            out.append(a)
            log(f"    {it.datetime:%Y-%m-%d} — скачано за "
                f"{time.time() - t0:.0f} с")
        if len(out) >= n_max:
            break
    return out


def composite(layers: list[np.ndarray], log) -> tuple[np.ndarray, np.ndarray]:
    """Медиана в линейной мощности с переводом в дБ в конце.

    Медиана децибелов — это медиана логарифмов, физически не та величина,
    поэтому усреднение идёт в линейной шкале, как и в боевом конвейере.
    """
    cube = np.stack(layers)
    n = np.isfinite(cube).sum(0)
    with np.errstate(invalid="ignore"):
        lin = np.nanmedian(cube, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(lin)
    db = np.where(n >= MIN_SCENES, np.maximum(db, FLOOR_DB), np.nan)
    log(f"    сцен {len(layers)}, ячеек с медианой "
        f"{100 * np.isfinite(db).mean():.1f} %")
    return db.astype("float32"), n.astype("int16")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--res", type=float, default=20.0)
    ap.add_argument("--scenes", type=int, default=N_SCENES)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    grid = site_grid(a.site, a.res)
    log(f"площадка {a.site}: весь лист {grid.width * a.res / 1000:.0f} x "
        f"{grid.height * a.res / 1000:.0f} км, шаг {a.res:.0f} м")
    found = find_scenes(grid, log)

    OUT.mkdir(parents=True, exist_ok=True)
    for state, items in found.items():
        if not items:
            log(f"\n{state}: сцен нет")
            continue
        log(f"\n{state} (подсветка с {ILLUM_AZ[state]:.0f}°):")
        layers = fetch(items, grid, log, a.scenes)
        if len(layers) < MIN_SCENES:
            log("  сцен меньше минимума — ракурс пропущен")
            continue
        db, n = composite(layers, log)
        valid = np.isfinite(db)
        sh = detrend(db, valid, a.res)
        p = OUT / f"radar_{a.site}_{int(a.res)}m_{state}.npz"
        np.savez_compressed(p, **{POL: db}, shade=sh, count=n,
                            res_m=a.res, x0=grid.x0, y_top=grid.y_top,
                            epsg=grid.epsg,
                            azimuth=np.array([ILLUM_AZ[state]]),
                            sun=np.array([np.nan]))
        log(f"  сохранено: {p}")


if __name__ == "__main__":
    main()
