"""Докачка ВСЕХ годных зимних сцен листа — чтобы половины стали равными.

ОТКУДА ПОСТАНОВКА. Композит и половины собраны из 12 сцен, по три на плитку:
отбор для листа (:func:`openlin_winter.cover_pick`) останавливался, как только
каждую точку накрывало минимальное число сцен. Композиту этого хватало, двум
независимым половинам — нет: деление 8 против 4, на 77 % листа половина «Б» —
одна сцена, и подтверждение спотыкается о неё. Запрос к каталогу (05.10.2026)
показал, что при ТЕХ ЖЕ фильтрах годных сцен 61: 11 / 10 / 18 / 22 по плиткам
48WWD / WWE / WXD / WXE. Три попытки ослабить подтверждение статистикой
закрыты; это улучшение — данными, ни одно правило метода не меняется.

ФИЛЬТРЫ — прежние (:mod:`openlin_lowsun`): солнце не выше 12°, облачность
сцены меньше 10 %, пустых пикселей не больше 5 %. Из переобработок одной
съёмки (та же дата и плитка) берётся одна, самая поздняя.

Сцены кладутся в ОТДЕЛЬНУЮ папку, прежние 12 и всё, что на них посчитано,
не трогаются. Уже скачанные сцены копируются, а не качаются заново.

Запуск из корня репозитория:
    python -m experiments.openlin_fetch_all --site anabar
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np

from experiments.openlin_lowsun import (BANDS, CACHE, MAX_CLOUD, MAX_NODATA,
                                        MAX_SUN_DEG, frag_polygon)

DEST = CACHE / "лист_все"


def eligible(grid) -> list:
    """Годные сцены: (высота солнца, азимут, элемент каталога), без дублей."""
    from pystac_client import Client
    from shapely.geometry import shape

    from experiments.openlin_probe import STAC_URL

    fp = frag_polygon(grid)
    res = Client.open(STAC_URL).search(
        collections=["sentinel-2-l2a"], bbox=grid.bbox_wgs84(),
        datetime="2019-01-01/2026-10-01",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}})
    best = {}
    for it in res.items():
        sz = it.properties.get("s2:mean_solar_zenith")
        if sz is None:
            continue
        sun = 90.0 - float(sz)
        if not 0.0 < sun <= MAX_SUN_DEG:
            continue
        if float(it.properties.get("s2:nodata_pixel_percentage", 0.0)) > MAX_NODATA:
            continue
        if shape(it.geometry).intersection(fp).area / fp.area < 0.02:
            continue
        key = (it.datetime.date(), it.properties.get("s2:mgrs_tile"))
        if key not in best or it.id > best[key][2].id:
            best[key] = (sun, float(it.properties.get("s2:mean_solar_azimuth", "nan")), it)
    return sorted(best.values(), key=lambda r: r[2].id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--res", type=float, default=20.0)
    a = ap.parse_args()

    import os

    import planetary_computer as pc
    from rasterio.enums import Resampling

    from experiments.openlin_probe import _read_on_grid
    from experiments.openlin_winter import site_grid

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
    t0 = time.time()
    grid = site_grid(a.site, a.res)
    items = eligible(grid)
    DEST.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for _s, _a, it in items:
        t = it.properties.get("s2:mgrs_tile")
        tiles[t] = tiles.get(t, 0) + 1
    print(f"годных сцен {len(items)}: " + ", ".join(f"{t} {n}" for t, n in sorted(tiles.items())),
          flush=True)

    done = failed = 0
    for i, (sun, az, it) in enumerate(items, 1):
        name = f"{it.id}_лист_{int(a.res)}m.npz"
        path = DEST / name
        if path.exists():
            done += 1
            continue
        old = CACHE / name
        if old.exists():                      # одна из прежних 12 — копия, не загрузка
            z = dict(np.load(old))
            np.savez_compressed(path, **z, sun=sun, az=az)
            done += 1
            print(f"  {i}/{len(items)} {it.datetime:%Y-%m-%d} — из прежних", flush=True)
            continue
        t1 = time.time()
        for attempt in range(3):
            try:
                bands = {b: _read_on_grid(pc.sign(it.assets[b].href), grid,
                                          Resampling.bilinear)
                         for b in BANDS if b in it.assets}
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **bands, sun=sun, az=az)
                shutil.move(tmp, path)
                done += 1
                print(f"  {i}/{len(items)} {it.datetime:%Y-%m-%d} "
                      f"{it.properties.get('s2:mgrs_tile')} солнце {sun:.1f}° — "
                      f"{time.time() - t1:.0f} с (всего {time.time() - t0:.0f} с)",
                      flush=True)
                break
            except Exception as exc:          # сеть: повтор, потом пропуск сцены
                if attempt == 2:
                    failed += 1
                    print(f"  {i}/{len(items)} {it.id} — НЕ СКАЧАНА ({exc!r:.120})",
                          flush=True)
                else:
                    time.sleep(10 * (attempt + 1))
    print(f"готово: в папке {done} сцен из {len(items)}, не скачано {failed} "
          f"({time.time() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
