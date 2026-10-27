"""Подложка из геологических карт ГГК-200 для карт проекта.

8 листов ГГК-200 (основной R-48-XI,XII и 7 смежных) лежат в
`datacache/ggk200/<лист>/*.tif` — сканы с geokniga, привязанные в WGS 84
(см. `experiments/fetch_ggk200.py`). Совмещение с нашими векторами проверено
21.10.2026 на основном листе: границы `svita_new` и буферы даек ложатся на
растр с точностью до толщины линии, — то есть `svita_new` оцифрованы с этого
же листа.

Рисование идёт в долготе-широте; наши слои (Гаусс–Крюгер, осевой 105°)
переводятся функцией :func:`to_lonlat`.

Если rasterio не загружается (27.10.2026 его библиотеку GDAL заблокировала
политика управления приложениями Windows), сканы читаются через PIL по тегам
привязки — картинка та же с точностью до округления окна.

Использование::

    from src import ggk200_base
    ggk200_base.draw(ax, bounds=(106, 70.66, 108, 71.34), decimate=2)
    ggk200_base.to_lonlat(gdf).plot(ax=ax, ...)
"""
from __future__ import annotations

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:  # на машине без рабочего rasterio — чтение через PIL
    rasterio = None

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


class _Bounds(tuple):
    """Границы листа с полями как у rasterio: left, bottom, right, top."""
    left = property(lambda self: self[0])
    bottom = property(lambda self: self[1])
    right = property(lambda self: self[2])
    top = property(lambda self: self[3])


def _read_rasterio(tif, bounds, decimate: int):
    """Окно скана через rasterio: (границы листа, картинка, границы окна)
    или None, если лист не попадает в `bounds`."""
    with rasterio.open(tif) as src:
        b = src.bounds
        if bounds is not None:
            l, bo = max(b.left, bounds[0]), max(b.bottom, bounds[1])
            r, t = min(b.right, bounds[2]), min(b.top, bounds[3])
            if l >= r or bo >= t:
                return None
        else:
            l, bo, r, t = b.left, b.bottom, b.right, b.top
        win = from_bounds(l, bo, r, t, src.transform).round_offsets().round_lengths()
        shape = (src.count, max(1, int(win.height) // decimate),
                 max(1, int(win.width) // decimate))
        img = src.read(window=win, out_shape=shape).transpose(1, 2, 0)
        return b, img, src.window_bounds(win)


def _read_pil(tif, bounds, decimate: int):
    """Окно скана без rasterio: сканы ГГК-200 — несжатые RGB GeoTIFF в
    долготе-широте с привязкой тегами 33922 (опорная точка, левый верхний
    угол) и 33550 (размер пикселя). Возвращает (границы листа, картинка,
    границы окна) или None, если лист не попадает в `bounds`."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(tif) as im:
        sx, sy = im.tag_v2[33550][:2]
        x0, y0 = im.tag_v2[33922][3:5]
        w, h = im.size
        b = _Bounds((x0, y0 - h * sy, x0 + w * sx, y0))
        if bounds is not None:
            l, bo = max(b.left, bounds[0]), max(b.bottom, bounds[1])
            r, t = min(b.right, bounds[2]), min(b.top, bounds[3])
            if l >= r or bo >= t:
                return None
        else:
            l, bo, r, t = b
        c0, c1 = int(round((l - x0) / sx)), int(round((r - x0) / sx))
        r0, r1 = int(round((y0 - t) / sy)), int(round((y0 - bo) / sy))
        crop = im.convert("RGB").crop((c0, r0, c1, r1))
        crop = crop.resize((max(1, (c1 - c0) // decimate),
                            max(1, (r1 - r0) // decimate)), Image.BILINEAR)
        img = np.asarray(crop)
    return b, img, (x0 + c0 * sx, y0 - r1 * sy, x0 + c1 * sx, y0 - r0 * sy)


def draw(ax, bounds=None, decimate: int = 4, alpha: float = 1.0,
         sheets=SHEETS, frames: bool = True) -> None:
    """Растры листов, попадающих в `bounds` (lon0, lat0, lon1, lat1)."""
    for sheet in sheets:
        tif = sheet_tif(sheet)
        if tif is None:
            continue
        got = (_read_pil if rasterio is None else _read_rasterio)(
            tif, bounds, decimate)
        if got is None:
            continue
        b, img, (wl, wb, wr, wt) = got
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
