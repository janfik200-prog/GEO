"""Загрузка лидарного рельефа MML 2 м (зеркало Funet) на лист — для проверки «линия настоящая».

Лидар — не вход метода, только проверка (память reproduce-reference-without-lidar,
extract-all-real-lines). Тайлы 3000×3000 (6×6 км, EPSG:3067) кладутся как есть в
datacache/mml_dem2m/<путь тайла>; готовые не качаются повторно.
Пилот 16.11.2026: на bbox Утсйоки 218 тайлов (покрытие 99 %), тайл ~25 МБ, ~2 с.
    python -m experiments.openlin_mml_fetch --site utsjoki
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://www.nic.funet.fi/index/geodata/mml/dem2m/"
VRT = Path("datacache/mml_dem2m/dem2m_direct.vrt")
DEST = Path("datacache/mml_dem2m")
X0, Y_TOP, RES = 62000, 7782000, 2.0          # из GeoTransform dem2m_direct.vrt


def tiles_for(site: str) -> list[str]:
    from pyproj import Transformer

    from experiments.openlin_winter import site_grid
    g = site_grid(site, 20.0)
    tr = Transformer.from_crs(g.epsg, 3067, always_xy=True)
    xs = [g.x0, g.x0 + g.width * g.res_m]
    ys = [g.y_top - g.height * g.res_m, g.y_top]
    X, Y = zip(*[tr.transform(x, y) for x in xs for y in ys])
    bx = (min(X), min(Y), max(X), max(Y))
    v = VRT.read_text(encoding="utf-8")
    src = re.findall(r'<SourceFilename relativeToVRT="1">([^<]+)</SourceFilename>.*?'
                     r'<DstRect xOff="(\d+)" yOff="(\d+)" xSize="(\d+)" ySize="(\d+)"', v, re.S)
    out = []
    for f, xo, yo, xs_, ys_ in src:
        a, b = X0 + int(xo) * RES, Y_TOP - int(yo) * RES
        c, d = a + int(xs_) * RES, b - int(ys_) * RES
        if a < bx[2] and c > bx[0] and d < bx[3] and b > bx[1]:
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="utsjoki")
    a = ap.parse_args()
    if a.site in ("lapland",):
        raise SystemExit("Лапландия — контроль, лидар для неё не качается без объявления")
    tl = tiles_for(a.site)
    print(f"{a.site}: тайлов {len(tl)}", flush=True)
    t0, got = time.time(), 0
    for i, f in enumerate(tl, 1):
        p = DEST / f
        if p.exists() and p.stat().st_size > 0:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(BASE + f, timeout=300).read()
                tmp = p.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(p)
                got += len(data)
                break
            except Exception as exc:
                print(f"  {f}: попытка {attempt + 1} — {exc!r:.100}", flush=True)
                time.sleep(10 * (attempt + 1))
        print(f"  {i}/{len(tl)} {f} ({got / 1e9:.2f} ГБ, {time.time() - t0:.0f} с)", flush=True)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
