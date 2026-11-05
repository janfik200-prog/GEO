"""Радар Sentinel-1 RTC на сетке листа 20 м: две половины по орбитам (VV).

Подготовка входа для подтверждения рельефа радаром. Отбор сцен — тот же, что
openlin_probe.s1_composite (config.S1_*: месяцы, годы, коллекция), но на
сетке 20 м держать 120 сцен в памяти нельзя (120 x 16.5 млн x 4 байта ~ 8 ГБ),
поэтому берётся по PER_ORBIT сцен на относительную орбиту. Половины — чётные и
нечётные орбиты по порядку номера: разные ракурсы съёмки.
    python -m experiments.openlin_s1_halves utsjoki
"""
import sys
import time

import numpy as np


PER_ORBIT = 2


def main(site):
    import planetary_computer as pc
    from pystac_client import Client
    from rasterio.enums import Resampling

    from cache_paths import CACHE
    from experiments.openlin_probe import STAC_URL, _read_on_grid
    from experiments.openlin_winter import site_grid
    from src import config

    grid = site_grid(site, 20.0)
    cat = Client.open(STAC_URL, modifier=pc.sign_inplace)
    res = cat.search(collections=[config.S1_COLLECTION], bbox=grid.bbox_wgs84(),
                     datetime=f"{min(config.S1_YEARS)}-01-01/{max(config.S1_YEARS)}-12-31")
    items = [it for it in res.items() if it.datetime.month in config.S1_MONTHS]
    by_orbit = {}
    for it in items:
        by_orbit.setdefault(it.properties.get("sat:relative_orbit", -1), []).append(it)
    orbits = sorted(by_orbit)
    print(f"сцен {len(items)}, орбит {len(orbits)}: {orbits}", flush=True)
    halves = {"А": [], "Б": []}
    for i, o in enumerate(orbits):
        g = sorted(by_orbit[o], key=lambda x: x.datetime)[:PER_ORBIT]
        halves["А" if i % 2 == 0 else "Б"] += g
    out = {}
    t0 = time.time()
    for name, group in halves.items():
        stack = []
        for it in group:
            if "vv" not in it.assets:
                continue
            try:
                a = _read_on_grid(it.assets["vv"].href, grid, Resampling.average)
            except Exception as exc:
                print(f"  {it.id}: ошибка {exc!r:.80}", flush=True)
                continue
            a = np.where((a > 0) & (a != config.S1_NODATA), a, np.nan)
            stack.append(a.astype("float32"))
            print(f"  половина {name}: {len(stack)}/{len(group)} ({time.time() - t0:.0f} с)", flush=True)
        lin = np.nanmedian(np.stack(stack), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = (10 * np.log10(lin)).astype("float32")
        print(f"  половина {name}: сцен {len(stack)}, пусто {100 * np.isnan(out[name]).mean():.1f} %", flush=True)
    p = CACHE / "s1_look" / f"s1_vv_{site}_лист_20m_половины.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, А=out["А"], Б=out["Б"],
                        orbits_A=np.array([o for i, o in enumerate(orbits) if i % 2 == 0]),
                        orbits_B=np.array([o for i, o in enumerate(orbits) if i % 2 == 1]))
    print(f"сохранено: {p}")


if __name__ == "__main__":
    main(sys.argv[1])
