"""Перенос линеаментных полей с рабочего растра 100 м на сетку прогноза 500 м.

Ветка линеаментов (``experiments/openlin_*``) считает свои поля на собственном
растре площадки: 770x860 ячеек по 100 м в EPSG:32647. Сетка прогноза — 154x149
ячеек по 500 м в EPSG:32648. Здесь одно кладётся на другое.

ПОЧЕМУ УСРЕДНЕНИЕ, А НЕ ИНТЕРПОЛЯЦИЯ. Ячейка сетки в 25 раз крупнее пикселя
растра по площади. Брать значение в центре ячейки — значит выбросить 24 из 25
измерений и получить шум вместо признака. Каждый пиксель растра приписывается
своей ячейке сетки, и по ячейке берётся среднее — так же, как ``stac_grid.to_cells``
кладёт на сетку спутниковые каналы.

ПОЧЕМУ ЧЕРЕЗ ПЕРЕПРОЕЦИРОВАНИЕ, А НЕ ПО СМЕЩЕНИЮ. Растр и сетка в РАЗНЫХ зонах
UTM (47 и 48), и разница между ними не сводится к сдвигу: на широте 71 градус
сетка размером 75 км разъезжается по краям на километры. Координаты пикселей
переводятся ``pyproj``, поэтому CRS обеих сторон берётся из данных, а не
предполагается.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src import integro_grid

#: CRS сетки прогноза. В исходных гридах ГИС Интегро проекция не записана, а
#: ``integro_grid`` подставляет плоскую CRS по умолчанию; фактическая зона
#: определена по координатам центра листа (107 в.д., 71 с.ш. -> UTM 48N).
GRID_EPSG = 32648


def cell_of_pixels(z: dict, meta: integro_grid.GridMeta
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Номер ячейки сетки для каждого пикселя растра.

    Возвращает ``(индексы ячеек, маска попавших в сетку)``; индекс — как в
    датасете, то есть по строкам с севера: ``row * pic + col``.
    """
    from pyproj import Transformer

    h, w = z["gray"].shape
    res = float(z["res_m"])
    xs = float(z["x0"]) + (np.arange(w) + 0.5) * res
    ys = float(z["y_top"]) - (np.arange(h) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)

    tr = Transformer.from_crs(int(z["epsg"]), GRID_EPSG, always_xy=True)
    gx, gy = tr.transform(xx.ravel(), yy.ravel())

    col = np.floor((gx - meta.x0) / meta.dx).astype(np.int64)
    row = np.floor((meta.y_top - gy) / meta.dy).astype(np.int64)
    ok = (col >= 0) & (col < meta.pic) & (row >= 0) & (row < meta.prf)
    return row * meta.pic + col, ok


def to_grid(field: np.ndarray, z: dict, meta: integro_grid.GridMeta,
            how: str = "mean") -> np.ndarray:
    """Поле на растре 100 м -> вектор по ячейкам сетки (длина ``prf * pic``).

    ``how``: ``mean`` — среднее по пикселям ячейки (для непрерывных полей),
    ``max`` — максимум (для «есть ли здесь линия»), ``frac`` — доля пикселей с
    ненулевым значением (для бинарных масок: получается длина линий на ячейку,
    нормированная на её площадь).
    """
    idx, ok = cell_of_pixels(z, meta)
    val = np.asarray(field, dtype="float64").ravel()
    good = ok & np.isfinite(val)
    n = meta.prf * meta.pic
    out = np.full(n, np.nan)
    if not good.any():
        return out

    cnt = np.bincount(idx[good], minlength=n)
    if how == "max":
        acc = np.full(n, -np.inf)
        np.maximum.at(acc, idx[good], val[good])
        out[cnt > 0] = acc[cnt > 0]
        return out
    src = (val[good] != 0).astype("float64") if how == "frac" else val[good]
    tot = np.bincount(idx[good], weights=src, minlength=n)
    out[cnt > 0] = tot[cnt > 0] / cnt[cnt > 0]
    return out


def distance_to(mask: np.ndarray, z: dict, meta: integro_grid.GridMeta
                ) -> np.ndarray:
    """Расстояние от центра ячейки сетки до ближайшей ячейки маски, м.

    Считается на растре (там разрешение 100 м), а на сетку кладётся средним —
    иначе расстояние округлялось бы до полукилометра.
    """
    from scipy.ndimage import distance_transform_edt

    res = float(z["res_m"])
    if not mask.any():
        return np.full(meta.prf * meta.pic, np.nan)
    dt = distance_transform_edt(~mask, sampling=(res, res))
    return to_grid(dt, z, meta, how="mean")


def load_site_raster(path: str | Path) -> dict:
    """Растр площадки из ``outputs/metrics/openlin_<site>.npz``."""
    return dict(np.load(Path(path)))
