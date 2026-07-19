"""Чтение нативных сеток ГИС Интегро: `.pgrid` (заголовок) и `.property` (значения).

Вопреки названию, `.pgrid` и `._rg` — это открытый XML (`Windows-1251`), а не
бинарный формат: `.pgrid` описывает геометрию сетки (число точек по X/Y, шаг,
начало координат) и список свойств (`Properties/TGrDocProp`), `._rg` — только
легенда отображения (диапазоны цветовой шкалы), данных в нём нет.

Бинарен только `.property`: плоский массив без заголовка, длиной `ObjCount`
из `.pgrid`. Соответствие `DataType` -> тип подтверждено размером файла и
сверкой со статистикой `Min/Max/Average/StDev`, зашитой прямо в `.pgrid`
(см. тесты в ``tests/test_integro_grid.py`` — совпадение до точности float32
для lyth/tect_nw/magm, включая StDev с поправкой Бесселя, ``ddof=1``):

- ``dtSingle``   -> ``float32`` little-endian, ``count = ObjCount``;
- ``dtByte``     -> ``uint8`` (каналы Landsat в СБОРКА_ДОП, размер файла = ObjCount);
- ``dtBitmask``  -> упакованные биты, ``ceil(ObjCount / 8)`` байт.

Порядок бит внутри байта для ``dtBitmask`` — LSB-first (конвенция Delphi
``TBits``): подтверждён связностью маски ``prognoz.sed`` (пространственно
цельная область, без полос с периодом 8).

Ориентация сетки (подтверждена численно, 2026-07-18): столбцы = ``Pic`` (X,
запад->восток), строки = ``Prf`` (Y), причём **строка 0 — северный край**
(массив хранится с севера на юг, как в GeoTIFF), а ``X0``/``Y0`` — юго-западный
угол. Доказательства: corr(topo5_new, Copernicus DEM) = +0.998 против -0.34 у
перевёрнутой гипотезы; corr(prognoz.lyth, расстояние до fasii.shp) = +0.9998
против +0.38. Для отрисовки использовать ``origin='upper'``; геопривязка —
:attr:`GridMeta.transform`.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GridProperty:
    name: str
    caption: str
    kind: str
    dtype: str
    stat_exists: bool
    vmin: float | None = None
    vmax: float | None = None
    average: float | None = None
    stdev: float | None = None


@dataclass
class GridMeta:
    obj_count: int
    prop_count: int
    pic: int   # число точек (пикетов) по профилю -> столбцы, X
    prf: int   # число профилей -> строки, Y
    dx: float
    dy: float
    x0: float
    y0: float
    properties: list[GridProperty] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        """``(строки, столбцы)`` = ``(Prf, Pic)`` для :func:`read_property`."""
        return self.prf, self.pic

    @property
    def y_top(self) -> float:
        """Y северного края сетки (``Y0`` — южный край, строки идут с севера)."""
        return self.y0 + self.prf * self.dy

    @property
    def transform(self) -> tuple[float, float, float, float, float, float]:
        """GDAL-геотрансформация ``(x0, dx, 0, y_top, 0, -dy)`` — north-up растр."""
        return (self.x0, self.dx, 0.0, self.y_top, 0.0, -self.dy)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Координаты центров ячеек ``(X, Y)`` формы :attr:`shape` (строка 0 — север)."""
        cols, rows = np.meshgrid(np.arange(self.pic), np.arange(self.prf))
        x = self.x0 + (cols + 0.5) * self.dx
        y = self.y_top - (rows + 0.5) * self.dy
        return x, y

    def property_by_name(self, name: str) -> GridProperty:
        for prop in self.properties:
            if prop.name == name:
                return prop
        raise KeyError(f"Свойство {name!r} не найдено в .pgrid (есть: {[p.name for p in self.properties]})")


def read_pgrid(path: Path) -> GridMeta:
    """Разобрать XML-заголовок `.pgrid` (ГИС Интегро)."""
    text = Path(path).read_text(encoding="cp1251", errors="ignore")
    root = ET.fromstring(text)

    def _f(tag: str, node=root) -> float | None:
        el = node.find(tag)
        return float(el.text) if el is not None and el.text is not None else None

    def _i(tag: str, node=root) -> int | None:
        val = _f(tag, node)
        return int(val) if val is not None else None

    properties = []
    props_node = root.find("Properties")
    if props_node is not None:
        for node in props_node:
            properties.append(GridProperty(
                name=node.findtext("PropName"),
                caption=node.findtext("Caption"),
                kind=node.findtext("PropKind"),
                dtype=node.findtext("DataType"),
                stat_exists=node.findtext("StatExists") == "True",
                vmin=_f("Min", node),
                vmax=_f("Max", node),
                average=_f("Average", node),
                stdev=_f("StDev", node),
            ))

    return GridMeta(
        obj_count=_i("ObjCount"),
        prop_count=_i("PropCount"),
        pic=_i("Pic"),
        prf=_i("Prf"),
        dx=_f("DX"),
        dy=_f("DY"),
        x0=_f("X0"),
        y0=_f("Y0"),
        properties=properties,
    )


# Соответствие DataType из .pgrid -> numpy dtype (плоские массивы фиксированной ширины)
_DTYPE_MAP: dict[str, str] = {
    "dtSingle": "<f4",
    "dtByte": "u1",
}


def read_property(path: Path, meta: GridMeta, prop: GridProperty) -> np.ndarray:
    """Прочитать `.property` в сетку формы :attr:`GridMeta.shape` согласно ``prop.dtype``."""
    raw = Path(path).read_bytes()
    if prop.dtype in _DTYPE_MAP:
        arr = np.frombuffer(raw, dtype=_DTYPE_MAP[prop.dtype], count=meta.obj_count)
    elif prop.dtype == "dtBitmask":
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
        arr = bits[: meta.obj_count].astype(np.uint8)
    else:
        raise ValueError(f"Неизвестный DataType: {prop.dtype!r}")
    return arr.reshape(meta.shape)


def read_grid_proj4(pgrid_path: Path) -> str | None:
    """Прочитать proj4 из sidecar `<stem>_pgrid.pj4` (тот же формат, что у шейпов).

    Импортирует :mod:`src.data_loader` лениво — этот модуль (и его зависимость
    от geopandas) не нужен, если требуется только бинарное чтение сетки.
    """
    from . import data_loader
    return data_loader.read_sidecar_proj4(Path(pgrid_path), suffix="_pgrid.pj4")


def resample_to_grid(
    arr: np.ndarray,
    src_meta: GridMeta,
    dst_meta: GridMeta,
    method: str = "bilinear",
    proj4: str | None = None,
) -> np.ndarray:
    """Пересчитать массив с сетки ``src_meta`` на сетку ``dst_meta`` (одна CRS).

    ``method``: ``bilinear`` — непрерывные поля того же/близкого шага;
    ``average`` — агрегация мелкой сетки в крупную (Landsat 30 м -> 500 м);
    ``nearest`` — категориальные значения. NoData = NaN сохраняется.

    Для углов в градусах (направление градиента ``*_1GFI_*``) использовать
    :func:`resample_angle_to_grid` — прямая интерполяция ломается на ±180°.
    """
    import rasterio.warp
    from rasterio.crs import CRS as RioCRS
    from rasterio.transform import Affine

    crs = RioCRS.from_proj4(proj4) if proj4 else RioCRS.from_epsg(32648)  # фиктивная планарная CRS: сетки в одной системе
    resampling = getattr(rasterio.warp.Resampling, method)

    def affine(meta: GridMeta) -> Affine:
        x0, dx, _, y_top, _, neg_dy = meta.transform
        return Affine(dx, 0.0, x0, 0.0, neg_dy, y_top)

    src = arr.astype(np.float32, copy=False)
    dst = np.full(dst_meta.shape, np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=src, destination=dst,
        src_transform=affine(src_meta), dst_transform=affine(dst_meta),
        src_crs=crs, dst_crs=crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def resample_angle_to_grid(
    arr_deg: np.ndarray, src_meta: GridMeta, dst_meta: GridMeta, proj4: str | None = None
) -> np.ndarray:
    """Пересчитать поле углов (градусы) через sin/cos — без разрыва на ±180°."""
    rad = np.deg2rad(arr_deg.astype(np.float32))
    s = resample_to_grid(np.sin(rad), src_meta, dst_meta, "bilinear", proj4)
    c = resample_to_grid(np.cos(rad), src_meta, dst_meta, "bilinear", proj4)
    return np.rad2deg(np.arctan2(s, c)).astype(np.float32)


def to_geotiff(out_path: Path, arr: np.ndarray, meta: GridMeta, proj4: str | None = None) -> None:
    """Сохранить массив (строка 0 = север) в GeoTIFF c привязкой из ``meta``.

    Импортирует rasterio лениво — по паттерну необязательных тяжёлых
    зависимостей (см. ``src/dem_features.py``).
    """
    import rasterio
    from rasterio.transform import Affine

    x0, dx, _, y_top, _, neg_dy = meta.transform
    transform = Affine(dx, 0.0, x0, 0.0, neg_dy, y_top)
    data = arr
    kwargs = dict(
        driver="GTiff", height=meta.prf, width=meta.pic, count=1,
        dtype=data.dtype.name, transform=transform, compress="deflate",
    )
    if proj4:
        kwargs["crs"] = rasterio.crs.CRS.from_proj4(proj4)
    if np.issubdtype(data.dtype, np.floating):
        kwargs["nodata"] = np.nan
    with rasterio.open(out_path, "w", **kwargs) as dst:
        dst.write(data, 1)


def to_common_grid(
    pgrid_path: Path,
    dst_meta: GridMeta,
    *,
    method: str = "bilinear",
    prefix: str = "",
    include: list[str] | None = None,
    angle_props: tuple[str, ...] = (),
    nodata: float | None = None,
    scale: float | None = None,
    proj4: str | None = None,
) -> dict[str, np.ndarray]:
    """Прочитать все свойства сетки и пересчитать их на общую сетку ``dst_meta``.

    ``method`` — способ ресемплинга для обычных свойств (см. :func:`resample_to_grid`);
    свойства из ``angle_props`` идут через :func:`resample_angle_to_grid` (поля
    направлений в градусах). ``include`` ограничивает набор свойств; ``nodata``
    (например, 0 для Landsat) переводится в NaN до ресемплинга; ``scale``
    домножает результат (пересчёт единиц). Ключи результата — ``prefix + PropName``.
    """
    meta, arrays = load_pgrid_dataset(Path(pgrid_path))
    out: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        if include is not None and name not in include:
            continue
        src = arr.astype(np.float32)
        if nodata is not None:
            src[src == nodata] = np.nan
        if name in angle_props:
            dst = resample_angle_to_grid(src, meta, dst_meta, proj4)
        else:
            dst = resample_to_grid(src, meta, dst_meta, method, proj4)
        if scale is not None:
            dst = dst * np.float32(scale)
        out[prefix + name] = dst
    return out


def load_pgrid_dataset(pgrid_path: Path) -> tuple[GridMeta, dict[str, np.ndarray]]:
    """Прочитать `.pgrid` + все существующие рядом `.property`, по именам из заголовка.

    Возвращает ``(meta, arrays)``, где ``arrays`` — словарь ``PropName -> ndarray``
    формы ``meta.shape`` (только для свойств, чей файл реально найден на диске).
    """
    pgrid_path = Path(pgrid_path)
    meta = read_pgrid(pgrid_path)
    stem = pgrid_path.stem
    arrays: dict[str, np.ndarray] = {}
    for prop in meta.properties:
        prop_path = pgrid_path.with_name(f"{stem}.{prop.name}.property")
        if prop_path.exists():
            arrays[prop.name] = read_property(prop_path, meta, prop)
    return meta, arrays
