"""Шаг 0 на открытых данных: есть ли в снимке контраст закартированных разломов.

ЗАЧЕМ. Ветка линеаментов на Анабаре закрыта измерением: порог обнаружения
0.60 sigma (:mod:`src.line_accumulation`) против контраста коридоров карты
0.086-0.194 sigma (:mod:`src.detection_limit`). Прежде чем обучать сеть
воспроизводить разметку геолога, надо проверить на площадке с ОТКРЫТОЙ разметкой,
что там сигнал вообще есть — иначе обучение меряет не метод, а отсутствие входа.

Прибор тот же, что и на Анабаре: :func:`src.detection_limit.corridor_contrast`
меряет, насколько яркость внутри коридора разлома отличается от яркости в кольце
вокруг него, в долях устойчивой сигмы композита. Число выходит прямо сравнимым с
анабарскими 0.086-0.194 sigma, потому что считается той же функцией.

ЧЕМ ЭТО НЕ ЯВЛЯЕТСЯ. Это ОПИСАНИЕ площадки, а не проверка гипотезы: ни p-value,
ни alpha, знаменатель Бонферрони структурной ветки (``config.FZ_N_TESTS_PRIOR``
и далее) НЕ растёт. Ровно тот же статус, что у шага 0.

ПЛОЩАДКИ (выбраны по плотности разметки, см. --help):
* ``abitibi``  — Abitibi greenstone belt, Superior craton: архейский щит и
  золоторудный пояс мирового класса, то есть ближайший доступный аналог Анабара
  по геологии. Разметка OGS MRD126, 1:250 000;
* ``nevada``   — Basin & Range: не щит, но обнажённая аридная территория, где
  сигнал заведомо есть. Контрольная площадка: если прибор не видит разломы и
  здесь, дело в приборе, а не в местности. Разметка USGS SGMC.

Запуск из корня репозитория:
    python -m experiments.openlin_probe --site abitibi
    python -m experiments.openlin_probe --site nevada --res 100
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import detection_limit as dl

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Площадки
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Site:
    """Площадка сверки: центр, размер и слой разломов к ней."""
    name: str
    lat: float
    lon: float
    faults: Path | tuple[Path, ...]
    #: Поле, по которому отбираются разломы среди прочих линий слоя. ``None`` —
    #: когда слой целиком состоит из разломов (так у анабарских tect1/tect2).
    fault_query: str | None
    epsg: int                 # метрическая проекция площадки (UTM)
    comment: str


#: Размер площадки — как у листа R-48-XI,XII, чтобы площадь совпадала с Анабаром
#: и плотность разметки была сравнима напрямую.
TILE_KM = (86.0, 77.0)

SITES: dict[str, Site] = {
    "abitibi": Site(
        name="abitibi", lat=48.362, lon=-79.884,
        faults=Path("datacache/ontario/MRD126"),
        fault_query="FEATURE", epsg=32617,
        comment="Abitibi, Superior craton (архей, золото); OGS MRD126 1:250k"),
    "nevada": Site(
        name="nevada", lat=38.669, lon=-116.023,
        faults=Path("datacache/sgmc/NV/NV_structure.shp"),
        fault_query="DESCRIPT", epsg=32611,
        comment="Basin & Range, обнажённая аридная; USGS SGMC"),
    # Центр листа R-48-XI,XII — тот же, что у расчёта IGRF в config.PF_*.
    # Разломы: два полигональных слоя зон, целиком (фильтровать нечего).
    "anabar": Site(
        name="anabar", lat=71.0030, lon=107.0235,
        faults=(Path("data/Gis-integro/shp_dbf/glub_raz_nw.shp"),
                Path("data/Gis-integro/shp_dbf/glub_r_nw.shp")),
        fault_query=None, epsg=32647,
        comment="Анабарский щит, лист R-48-XI,XII; зоны разломов ГИС Интегро"),
}

# --------------------------------------------------------------------------
# Композит: те же принципы, что у src/s2_composite.py, но CRS площадки
# --------------------------------------------------------------------------

#: Годы и месяцы. Лето берётся коротким окном: цель — минимум снега и максимум
#: освещённости, а не разнообразие фенологии.
S2_YEARS = (2019, 2020, 2021, 2022, 2023)
S2_MONTHS = (7, 8)
S2_MAX_CLOUD = 20
S2_MAX_SCENES = 40
#: Классы SCL, которые оставляем: 4 растительность, 5 открытый грунт, 6 вода,
#: 7 облако низкой вероятности. Тот же набор, что config.S2_SCL_KEEP.
SCL_KEEP = (4, 5, 6, 7)
#: Минимум наблюдений на пиксель — иначе медиана считается по одной-двум сценам
#: и меряет конкретную дату, а не поверхность.
MIN_OBS = 3
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class Grid:
    """Регулярная сетка площадки в метрической проекции (north-up растр)."""
    x0: float
    y_top: float
    res_m: float
    height: int
    width: int
    epsg: int

    @property
    def transform(self):
        from rasterio.transform import Affine
        return Affine(self.res_m, 0, self.x0, 0, -self.res_m, self.y_top)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def bbox_wgs84(self, pad_m: float = 5000.0) -> list[float]:
        from pyproj import CRS, Transformer
        tr = Transformer.from_crs(CRS.from_epsg(self.epsg), CRS.from_epsg(4326),
                                  always_xy=True)
        xs = [self.x0 - pad_m, self.x0 + self.width * self.res_m + pad_m]
        ys = [self.y_top + pad_m, self.y_top - self.height * self.res_m - pad_m]
        lons, lats = zip(*[tr.transform(x, y) for x in xs for y in ys])
        return [min(lons), min(lats), max(lons), max(lats)]


def build_grid(site: Site, res_m: float) -> Grid:
    """Сетка ``TILE_KM`` вокруг центра площадки, шаг ``res_m``."""
    from pyproj import CRS, Transformer

    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(site.epsg),
                              always_xy=True)
    cx, cy = tr.transform(site.lon, site.lat)
    w_m, h_m = TILE_KM[0] * 1000, TILE_KM[1] * 1000
    width = int(round(w_m / res_m))
    height = int(round(h_m / res_m))
    return Grid(x0=cx - w_m / 2, y_top=cy + h_m / 2, res_m=res_m,
                height=height, width=width, epsg=site.epsg)


def search_scenes(grid: Grid, log) -> list:
    """Летние сцены Sentinel-2 L2A над площадкой, самые чистые."""
    import planetary_computer as pc
    from pystac_client import Client

    cat = Client.open(STAC_URL, modifier=pc.sign_inplace)
    res = cat.search(collections=["sentinel-2-l2a"], bbox=grid.bbox_wgs84(),
                     datetime=f"{min(S2_YEARS)}-01-01/{max(S2_YEARS)}-12-31",
                     query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD}})
    items = [it for it in res.items() if it.datetime.month in S2_MONTHS]
    items.sort(key=lambda it: it.properties["eo:cloud_cover"])
    log(f"  сцен найдено {len(items)}, беру {min(len(items), S2_MAX_SCENES)}")
    return items[:S2_MAX_SCENES]


def _read_on_grid(href: str, grid: Grid, resampling) -> np.ndarray:
    """Один растр, перепроецированный сразу на сетку площадки (окном, не целиком)."""
    import rasterio
    from rasterio.crs import CRS as RCRS
    from rasterio.vrt import WarpedVRT

    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=RCRS.from_epsg(grid.epsg),
                       transform=grid.transform, height=grid.height,
                       width=grid.width, resampling=resampling,
                       src_nodata=0, nodata=0) as vrt:
            return vrt.read(1).astype("float32")


def grayscale(grid: Grid, items: list, log) -> tuple[np.ndarray, np.ndarray]:
    """Серый композит (медиана красного канала) и медианный NDVI по сценам.

    Серый — тот же принцип, что в :mod:`src.lineaments_sat`: один канал, а не
    RGB, потому что детектор линий работает по яркости. NDVI считается заодно —
    он нужен как мера задернованности, то есть как причина, по которой контраст
    может оказаться низким.
    """
    import os

    from rasterio.enums import Resampling

    # Те же сетевые пороги, что у проекта: без них /vsicurl висит бесконечно.
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")
    os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "60")
    os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "1000")

    reds, ndvis = [], []
    for i, it in enumerate(items, 1):
        try:
            scl = _read_on_grid(it.assets["SCL"].href, grid, Resampling.nearest)
            keep = np.isin(scl.astype(int), SCL_KEEP)
            if keep.mean() < 0.2:
                continue
            red = _read_on_grid(it.assets["B04"].href, grid, Resampling.average)
            nir = _read_on_grid(it.assets["B08"].href, grid, Resampling.average)
        except Exception as e:
            log(f"    сцена {i} пропущена: {type(e).__name__}")
            continue
        red = np.where(keep & (red > 0), red, np.nan)
        nir = np.where(keep & (nir > 0), nir, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (nir - red) / (nir + red)
        reds.append(red)
        ndvis.append(ndvi)
        if i % 5 == 0:
            log(f"    прочитано {i}/{len(items)} сцен")

    if not reds:
        raise RuntimeError("ни одной пригодной сцены")
    stack = np.stack(reds)
    n_obs = np.sum(np.isfinite(stack), axis=0)
    gray = np.nanmedian(stack, axis=0)
    gray = np.where(n_obs >= MIN_OBS, gray, np.nan)
    ndvi = np.nanmedian(np.stack(ndvis), axis=0)
    log(f"  композит из {len(reds)} сцен; ячеек с >= {MIN_OBS} наблюдений: "
        f"{100 * np.mean(n_obs >= MIN_OBS):.1f}%")
    return gray, ndvi


# --------------------------------------------------------------------------
# Разломы -> расстояние по сетке
# --------------------------------------------------------------------------

#: Метрики интерферометрической когерентности, которые берём как каналы.
#: COH12 лето/зима — сама когерентность на 12-суточной базе в двух режимах
#: поверхности (в мерзлоте зимой она заметно выше: нет ни листвы, ни талой воды).
#: COH24/COH48 — насколько когерентность держится с ростом базы; поверхность,
#: сохраняющая фазу месяцами, это не полог, а грунт или камень.
#: rho/tau — параметры экспоненциальной модели затухания когерентности
#: (уровень насыщения и характерное время декорреляции), то есть сжатое описание
#: устойчивости поверхности одним числом вместо серии баз.
COH_METRICS: tuple[str, ...] = (
    "winter_vv_COH12", "summer_vv_COH12", "summer_vv_COH24", "summer_vv_COH48",
    "summer_vv_rho", "summer_vv_tau",
)
COH_BASE = ("https://sentinel-1-global-coherence-earthbigdata.s3.us-west-2."
            "amazonaws.com/data/tiles")


def _coh_tiles(bbox: list[float]) -> list[str]:
    """Имена 1x1-градусных тайлов, покрывающих охват (тайл назван верхним углом)."""
    lon0, lat0, lon1, lat1 = bbox
    names = []
    for lat in range(int(np.floor(lat0)), int(np.ceil(lat1)) + 1):
        for lon in range(int(np.floor(lon0)), int(np.ceil(lon1)) + 1):
            ns = f"N{lat + 1:02d}" if lat + 1 >= 0 else f"S{abs(lat + 1):02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
            names.append(f"{ns}{ew}")
    return names


def coherence(grid: Grid, log) -> dict[str, np.ndarray]:
    """Интерферометрическая когерентность S1 из готового глобального набора.

    ЗАЧЕМ. Амплитуда радара (и тем более оптика) меряет ЯРКОСТЬ поверхности, а
    когерентность — сохраняет ли поверхность фазу между пролётами. Это другая
    физическая величина: полог растительности декоррелирует за дни, а грунт и
    коренной выход держат фазу неделями. В мерзлоте зимой когерентность высокая,
    и по литературе InSAR устойчиво работает именно за пределами лесной зоны.

    Берётся ГОТОВЫЙ продукт (Kellndorfer et al. 2022, AWS Open Data): 205 тыс.
    сцен SLC уже обработаны в сезонные метрики с шагом 3 угловые секунды (~90 м,
    почти наш шаг). Своя обработка SLC дала бы то же самое ценой недели счёта.
    """
    import os

    from rasterio.enums import Resampling

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    tiles = _coh_tiles(grid.bbox_wgs84())
    log(f"  тайлов когерентности: {len(tiles)} ({', '.join(tiles[:4])}...)")

    out: dict[str, np.ndarray] = {}
    for metric in COH_METRICS:
        parts = []
        for t in tiles:
            url = f"/vsicurl/{COH_BASE}/{t}/{t}_{metric}.tif"
            try:
                parts.append(_read_on_grid(url, grid, Resampling.average))
            except Exception:
                continue                      # тайла над водой/льдом может не быть
        if not parts:
            log(f"  {metric}: ни одного тайла — пропущен")
            continue
        # Тайлы не перекрываются: собираем мозаику, беря везде ненулевое значение.
        stack = np.stack(parts)
        stack = np.where(stack > 0, stack, np.nan)
        merged = np.nanmax(stack, axis=0)
        out[f"coh_{metric}"] = merged.astype("float32")
        log(f"  {metric}: покрыто {100 * np.mean(np.isfinite(merged)):.1f}%, "
            f"медиана {np.nanmedian(merged):.0f}")
    return out


def _read_pj4(shp: Path) -> str:
    """proj4-строка из sidecar ``<имя>_shp.pj4`` рядом с шейпом."""
    side = shp.with_name(shp.stem + "_shp.pj4")
    if not side.exists():
        raise SystemExit(f"нет ни .prj, ни {side.name} — проекция слоя неизвестна")
    for line in side.read_text(encoding="cp1251").splitlines():
        if line.strip().startswith("pj4="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"в {side.name} нет строки pj4=")


def s1_composite(grid: Grid, log) -> dict[str, np.ndarray]:
    """Радарный композит Sentinel-1 RTC: медиана gamma0 по VV и VH, в дБ.

    ЗАЧЕМ РЯДОМ С ОПТИКОЙ. Радар меряет шероховатость и геометрию поверхности, а
    не отражение полога, и не зависит ни от облаков, ни от освещённости. Это
    физически другой канал, а не ещё одна версия того же снимка.

    Параметры взяты из боевого конфига проекта (``config.S1_*``), а не подобраны
    заново: месяцы 6-9 — снег искажает обратное рассеяние сильнее, чем облака
    оптику; квота на относительную орбиту — ракурс съёмки задаёт тени и layover,
    и без квоты вся выборка ушла бы в один ракурс.

    Усреднение идёт в ЛИНЕЙНОЙ мощности с переводом в дБ в конце: медиана
    децибелов — это медиана логарифмов, физически не та величина.
    """
    import os

    import planetary_computer as pc
    from pystac_client import Client
    from rasterio.enums import Resampling

    from src import config

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "20")

    cat = Client.open(STAC_URL, modifier=pc.sign_inplace)
    res = cat.search(collections=[config.S1_COLLECTION], bbox=grid.bbox_wgs84(),
                     datetime=f"{min(config.S1_YEARS)}-01-01/"
                              f"{max(config.S1_YEARS)}-12-31")
    items = [it for it in res.items() if it.datetime.month in config.S1_MONTHS]

    # Квота на относительную орбиту (config.S1_MAX_SCENES_PER_ORBIT)
    by_orbit: dict[int, list] = {}
    for it in items:
        by_orbit.setdefault(it.properties.get("sat:relative_orbit", -1), []).append(it)
    picked = []
    for orbit, group in by_orbit.items():
        group.sort(key=lambda i: i.datetime)
        picked.extend(group[:config.S1_MAX_SCENES_PER_ORBIT])
    log(f"  сцен S1 {len(items)} на {len(by_orbit)} орбитах, беру {len(picked)}")

    out: dict[str, np.ndarray] = {}
    for pol in config.S1_POLARIZATIONS:
        stack = []
        for it in picked:
            if pol not in it.assets:
                continue
            try:
                a = _read_on_grid(it.assets[pol].href, grid, Resampling.average)
            except Exception:
                continue
            a = np.where((a > 0) & (a != config.S1_NODATA), a, np.nan)
            stack.append(a)
        if not stack:
            log(f"  {pol}: ни одной сцены — канал пропущен")
            continue
        lin = np.nanmedian(np.stack(stack), axis=0)          # линейная мощность
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(lin)
        out[pol] = np.maximum(db, config.S1_FLOOR_DB).astype("float32")
        log(f"  {pol}: композит из {len(stack)} сцен, "
            f"медиана {np.nanmedian(out[pol]):.1f} дБ")
    return out


def fault_distance(grid: Grid, site: Site, log) -> tuple[np.ndarray, int]:
    """Расстояние от центра каждой ячейки до ближайшего разлома, м.

    Разломы отбираются по текстовому полю (``FEATURE``/``DESCRIPT``) и, отдельно,
    ОТБРАСЫВАЮТСЯ скрытые (``concealed``): разлом под перекрытием на снимке не
    виден по определению, и его присутствие в цели занижает измеряемый контраст,
    не сообщая ничего о методе.
    """
    import geopandas as gpd
    import pandas as pd
    from scipy.ndimage import distance_transform_edt
    from rasterio.features import rasterize

    paths = site.faults if isinstance(site.faults, tuple) else (site.faults,)
    parts = []
    for p in paths:
        shp = next(p.rglob("Geolines.shp")) if p.is_dir() else p
        g = gpd.read_file(shp)
        if g.crs is None:
            # Слои ГИС Интегро идут без .prj: проекция лежит в sidecar
            # `<имя>_shp.pj4` (тот же приём, что в src/integro_grid.py).
            g = g.set_crs(_read_pj4(shp))
        if site.fault_query is None:
            # Слой целиком разломный: границы полигонов зон и есть цель сверки.
            parts.append(g.to_crs(f"EPSG:{site.epsg}").boundary)
            log(f"  {shp.name}: {len(g)} объектов, взяты целиком (границы зон)")
            continue
        col = site.fault_query
        is_fault = g[col].str.contains("fault", case=False, na=False)
        concealed = g[col].str.contains("conceal", case=False, na=False)
        parts.append(g[is_fault & ~concealed].to_crs(f"EPSG:{site.epsg}").geometry)
        log(f"  {shp.name}: разломов {int(is_fault.sum())}, из них скрытых "
            f"{int((is_fault & concealed).sum())} — отброшены")
    f = gpd.GeoDataFrame(geometry=pd.concat(parts, ignore_index=True),
                         crs=f"EPSG:{site.epsg}")

    minx = grid.x0
    maxx = grid.x0 + grid.width * grid.res_m
    miny = grid.y_top - grid.height * grid.res_m
    f = f.cx[minx:maxx, miny:grid.y_top]
    log(f"  на площадке: {len(f)} отрезков, {f.length.sum() / 1000:.0f} км")
    if len(f) == 0:
        raise RuntimeError("на площадке нет разломов")

    mask = rasterize(((geom, 1) for geom in f.geometry), out_shape=grid.shape,
                     transform=grid.transform, fill=0, dtype="uint8")
    dist = distance_transform_edt(mask == 0, sampling=grid.res_m)
    return dist.astype("float32"), len(f)


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=sorted(SITES), default="abitibi")
    ap.add_argument("--res", type=float, default=100.0,
                    help="шаг рабочей сетки, м (по умолчанию 100 — как в шаге 0)")
    args = ap.parse_args()

    t0 = time.time()

    def log(msg=""):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    site = SITES[args.site]
    log(f"площадка «{site.name}»: {site.comment}")
    grid = build_grid(site, args.res)
    log(f"сетка {grid.height}x{grid.width}, шаг {grid.res_m:.0f} м, "
        f"EPSG:{grid.epsg}; bbox {[round(v, 3) for v in grid.bbox_wgs84()]}")

    log("разломы:")
    dist, n_faults = fault_distance(grid, site, log)

    log("снимок:")
    items = search_scenes(grid, log)
    gray, ndvi = grayscale(grid, items, log)

    log("радар:")
    radar = s1_composite(grid, log)

    log("когерентность:")
    radar.update(coherence(grid, log))

    sigma = dl.robust_sigma(gray)
    veg = float(np.nanmean(ndvi > 0.35))
    log(f"  sigma композита (MAD*1.4826) = {sigma:.4g}; "
        f"доля пикселей с NDVI > 0.35: {100 * veg:.1f}%")

    # Ширина коридора перебирается, а не назначается. На Анабаре разломы —
    # ПОЛИГОНЫ зон (8-21 км), и «внутри» там определено самой геометрией. Здесь
    # разломы — ЛИНИИ, ширины у них нет, поэтому вместо произвольного выбора
    # показывается зависимость: если контраст есть, он должен падать с ростом
    # коридора (полоса размывается фоном), а не колебаться вокруг нуля.
    log("")
    log(f"КОНТРАСТ КОРИДОРОВ на «{site.name}»:")
    log("  полуширина | медиан | разброс |  внутри |  кольцо")
    rows = []
    for corridor_m in (0.0, 250.0, 500.0, 1000.0):
        got = dl.corridor_contrast(gray.ravel(), dist.ravel(), sigma,
                                   corridor_m=corridor_m)
        rows.append({"полуширина_м": corridor_m, **got})
        log(f"  {corridor_m:9.0f} | {got['contrast_median']:+6.3f} | "
            f"{got['contrast_spread']:+7.3f} | {got['n_inside']:7.0f} | "
            f"{got['n_ring']:7.0f}")
    log("")
    log("для сравнения (Анабар, тот же прибор):")
    log("  коридоры карты 0.086-0.194 sigma; порог обнаружения 0.60 sigma")

    import pandas as pd

    out = Path("outputs/metrics")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "площадка", site.name)
    df["sigma"] = sigma
    df["доля_NDVI>0.35"] = veg
    df["отрезков"] = n_faults
    df.to_csv(out / f"openlin_{site.name}.csv", index=False, encoding="utf-8-sig")

    # Композит сохраняется целиком: он же вход обучения на следующем этапе,
    # а сборка из 40 сцен идёт минутами.
    np.savez_compressed(out / f"openlin_{site.name}.npz", gray=gray, ndvi=ndvi,
                        dist=dist, sigma=sigma, res_m=grid.res_m,
                        x0=grid.x0, y_top=grid.y_top, epsg=grid.epsg, **radar)
    log(f"сохранено: outputs/metrics/openlin_{site.name}.csv и .npz "
        f"(композит, NDVI, расстояния — вход обучения сети)")


if __name__ == "__main__":
    main()
