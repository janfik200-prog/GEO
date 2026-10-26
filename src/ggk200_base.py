"""Подложка из геологических карт ГГК-200 для карт проекта.

8 листов ГГК-200 (основной R-48-XI,XII и 7 смежных) лежат в
`datacache/ggk200/<лист>/*.tif` — сканы с geokniga, привязанные в WGS 84
(см. `experiments/fetch_ggk200.py`). Совмещение с нашими векторами проверено
21.10.2026 на основном листе: границы `svita_new` и буферы даек ложатся на
растр с точностью до толщины линии, — то есть `svita_new` оцифрованы с этого
же листа.

Рисование идёт в долготе-широте; наши слои (Гаусс–Крюгер, осевой 105°)
переводятся функцией :func:`to_lonlat`.

Использование::

    from src import ggk200_base
    ggk200_base.draw(ax, bounds=(106, 70.66, 108, 71.34), decimate=2)
    ggk200_base.to_lonlat(gdf).plot(ax=ax, ...)
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from cache_paths import GGK200
from src.anabar_inset import SHEET_CRS

SHEETS = ("R-48-V,VI", "R-48-IX,X", "R-48-XI,XII", "R-48-XV,XVI",
          "R-48-XVII,XVIII", "R-49-I,II", "R-49-VII,VIII", "R-49-XIII,XIV")
LAT0 = 71.0


def sheet_tif(sheet: str):
    folder = GGK200 / sheet.replace(",", "_")
    tifs = sorted(folder.glob("*.tif"))
    return tifs[0] if tifs else None


def to_lonlat(gdf, crs: str = SHEET_CRS):
    """Слой проекта -> WGS 84. Шейпы ГИС Интегро без .prj — СК задаётся явно."""
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    return gdf.to_crs(4326)


def draw(ax, bounds=None, decimate: int = 4, alpha: float = 1.0,
         sheets=SHEETS, frames: bool = True) -> None:
    """Растры листов, попадающих в `bounds` (lon0, lat0, lon1, lat1)."""
    for sheet in sheets:
        tif = sheet_tif(sheet)
        if tif is None:
            continue
        with rasterio.open(tif) as src:
            b = src.bounds
            if bounds is not None:
                l, bo = max(b.left, bounds[0]), max(b.bottom, bounds[1])
                r, t = min(b.right, bounds[2]), min(b.top, bounds[3])
                if l >= r or bo >= t:
                    continue
            else:
                l, bo, r, t = b.left, b.bottom, b.right, b.top
            win = from_bounds(l, bo, r, t, src.transform).round_offsets().round_lengths()
            shape = (src.count, max(1, int(win.height) // decimate),
                     max(1, int(win.width) // decimate))
            img = src.read(window=win, out_shape=shape).transpose(1, 2, 0)
            wl, wb, wr, wt = src.window_bounds(win)
        ax.imshow(img, extent=[wl, wr, wb, wt], origin="upper", alpha=alpha,
                  zorder=0, interpolation="bilinear")
        if frames:
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (b.left, b.bottom), b.right - b.left, b.top - b.bottom,
                fill=False, edgecolor="0.4", linewidth=0.6, zorder=1))
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect(1 / np.cos(np.radians(LAT0)))
