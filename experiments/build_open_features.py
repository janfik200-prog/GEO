"""Признаки из открытых слоёв (``experiments/fetch_open_layers.py``) на обеих сетках.

Группы и смысл:

* ``osm_*`` — расстояние до рек, ручьёв и озёр OpenStreetMap, км. Замысел —
  заменить ``dist_dnl``/``dist_dnara``, которые за рамкой листа меряют
  расстояние до края оцифровки. На листе печатается ранговая связь с ними:
  без неё нельзя решить, тот же это признак или другой;
* ``wc_*`` — доли классов ESA WorldCover 2021 в ячейке: голый грунт и редкая
  растительность, мхи и лишайники, кустарник, травы, вода, болота;
* ``ged_*`` — ASTER GED AG100: средняя излучательная способность в каналах
  10-14 по всему архиву ASTER и число наблюдений. Отношение b10/b12 растёт с
  кремнезёмом — тот самый тепловой индикатор, который мозаике ``ast_*``
  одной даты недоступен;
* ``gmt_*`` — глобальная геофизика GMT: магнитка EMAG2 4 км, гравика в
  свободном воздухе, вертикальный градиент, геоид. Региональный фон, не
  замена съёмке организации (разрешение 4 км против 500 м). Единицы условные:
  ``common.sample_global`` умножает коды JP2 на масштаб, но не вычитает
  смещение тайла, поэтому порядок значений верный, а абсолютный уровень нет.

Выход: ``data/processed/open_main_v3.parquet`` (22 946 ячеек листа) и
``open_wide_v3.parquet`` (90 597 ячеек широкой сетки), порядок ячеек тот же,
что у остальных ``*_wide.parquet`` (строка 0 — север). В датасеты и базу не
вливается: решение принимает сверка.

Запуск из корня: ``python -u -m experiments.build_open_features``.
"""
import glob
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from pyproj import Transformer  # noqa: E402

from cache_paths import CACHE  # noqa: E402
from src import config, integro_grid, stac_grid  # noqa: E402
from src.vector_features import distance_raster  # noqa: E402
from experiments import common  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

MAIN_META = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
PROJ4 = integro_grid.read_grid_proj4(config.GRAVMAG_PGRID)

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 60: "bare", 70: "snow", 80: "water",
              90: "wetland", 100: "moss"}
GED_BANDS = ("b10", "b11", "b12", "b13", "b14")


def blocks_mean(a: np.ndarray, meta, k: int) -> np.ndarray:
    """Субпиксели k x k -> ячейка 500 м: среднее без NaN."""
    a = a.reshape(meta.prf, k, meta.pic, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=(1, 3)).ravel()


# --------------------------------------------------------------------- OSM
def osm_geoms():
    data = json.loads((CACHE / "osm_hydro" / "waterways.json").read_text(encoding="utf-8"))
    to_grid = Transformer.from_crs("EPSG:4326", PROJ4, always_xy=True)
    rivers, streams, water = [], [], []

    def line(pts):
        xs, ys = to_grid.transform([p["lon"] for p in pts], [p["lat"] for p in pts])
        return shapely.LineString(np.column_stack([xs, ys]))

    for el in data["elements"]:
        tags = el.get("tags", {})
        parts = [el["geometry"]] if "geometry" in el else \
            [m["geometry"] for m in el.get("members", []) if "geometry" in m]
        for pts in parts:
            if len(pts) < 2:
                continue
            g = line(pts)
            if tags.get("waterway") == "river":
                rivers.append(g)
            elif tags.get("waterway") in ("stream", "canal"):
                streams.append(g)
            elif tags.get("natural") == "water":
                # Контур озера; ячейка внутри получит расстояние до берега, а
                # не 0 — для признака «близость к водоёму» это допустимо.
                water.append(g)
    return rivers, streams, water


def osm_features(meta) -> pd.DataFrame:
    rivers, streams, water = osm_geoms()
    out = {"osm_dist_river": distance_raster(meta, rivers).ravel() / 1000.0,
           "osm_dist_stream": distance_raster(meta, rivers + streams).ravel() / 1000.0,
           "osm_dist_lake": distance_raster(meta, water).ravel() / 1000.0}
    return pd.DataFrame(out)


# --------------------------------------------------------------------- WorldCover
def worldcover_features(meta) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import Resampling, reproject

    k = 10                                   # 50 м внутри ячейки 500 м
    tr, h, w = stac_grid.grid_transform(meta, meta.dx / k)
    dst = np.zeros((h, w), np.uint8)
    for f in sorted(glob.glob(str(CACHE / "worldcover" / "*.tif"))):
        with rasterio.open(f) as src:
            reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs=PROJ4,
                      resampling=Resampling.mode, src_nodata=0, dst_nodata=0,
                      init_dest_nodata=False)
    blk = dst.reshape(meta.prf, k, meta.pic, k)
    valid = (blk > 0).sum(axis=(1, 3)).ravel().astype(float)
    out = {}
    for code, name in WC_CLASSES.items():
        n = (blk == code).sum(axis=(1, 3)).ravel()
        out[f"wc_{name}_frac"] = np.where(valid > 0, n / np.maximum(valid, 1), np.nan)
    return pd.DataFrame(out)


# --------------------------------------------------------------------- ASTER GED
def ged_features(meta) -> pd.DataFrame:
    import h5py
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    k = 5                                    # 100 м
    tr, h, w = stac_grid.grid_transform(meta, meta.dx / k)
    layers = {b: np.full((h, w), np.nan, np.float32) for b in GED_BANDS}
    layers["nobs"] = np.full((h, w), np.nan, np.float32)
    for f in sorted(glob.glob(str(CACHE / "aster_ged" / "*.h5"))):
        with h5py.File(f) as hf:
            lat = hf["Geolocation/Latitude"][...]
            lon = hf["Geolocation/Longitude"][...]
            em = hf["Emissivity/Mean"][...].astype(np.float32)
            nobs = hf["Observations/NumObs"][...].astype(np.float32)
        step = abs(float(lon[0, 1] - lon[0, 0]))
        src_tr = from_origin(float(lon[0, 0]) - step / 2, float(lat[0, 0]) + step / 2, step, step)
        em[em <= 0] = np.nan                 # заполнитель -9999
        em *= 0.001                          # масштаб продукта
        bad = ~np.isfinite(em[3])
        nobs[bad] = np.nan
        for i, b in enumerate(GED_BANDS):
            tmp = np.full((h, w), np.nan, np.float32)
            reproject(em[i], tmp, src_transform=src_tr, src_crs="EPSG:4326",
                      dst_transform=tr, dst_crs=PROJ4, resampling=Resampling.average,
                      src_nodata=np.nan, dst_nodata=np.nan)
            layers[b] = np.where(np.isfinite(tmp), tmp, layers[b])
        tmp = np.full((h, w), np.nan, np.float32)
        reproject(nobs, tmp, src_transform=src_tr, src_crs="EPSG:4326", dst_transform=tr,
                  dst_crs=PROJ4, resampling=Resampling.average, src_nodata=np.nan,
                  dst_nodata=np.nan)
        layers["nobs"] = np.where(np.isfinite(tmp), tmp, layers["nobs"])
    out = {f"ged_{b}": blocks_mean(layers[b], meta, k) for b in GED_BANDS}
    out["ged_n_obs"] = blocks_mean(layers["nobs"], meta, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Кварц даёт провал излучательности около 8.6-9.1 мкм (b10-b12) при
        # высокой в b13-b14: отношение растёт с кремнезёмом.
        out["ged_silica"] = out["ged_b13"] / out["ged_b12"]
    return pd.DataFrame(out)


# --------------------------------------------------------------------- GMT
def gmt_features(meta) -> pd.DataFrame:
    x, y = meta.cell_centers()
    lon, lat = Transformer.from_crs(PROJ4, "EPSG:4326", always_xy=True).transform(x.ravel(), y.ravel())
    return pd.DataFrame({f"gmt_{ds}": common.sample_global(ds, lon, lat)
                         for ds in ("mag4km", "faa", "vgg", "geoid")})


def build(meta, name: str) -> pd.DataFrame:
    parts = []
    for title, fn in (("OSM", osm_features), ("WorldCover", worldcover_features),
                      ("ASTER GED", ged_features), ("GMT", gmt_features)):
        df = fn(meta)
        cov = df.notna().all(axis=1).mean() * 100
        print(f"  {name} {title}: {df.shape[1]} признаков, полных ячеек {cov:.1f}%", flush=True)
        parts.append(df)
    out = pd.concat(parts, axis=1)
    path = config.PROCESSED_DIR / f"open_{name}_v3.parquet"
    out.to_parquet(path, index=False)
    print(f"  -> {path.name} {out.shape}", flush=True)
    return out


def check_against_org_rivers(main: pd.DataFrame) -> None:
    """Ранговая связь OSM с реками организации на листе — тот же признак или нет."""
    src = config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet"
    ds = pd.read_parquet(src)
    if len(ds) != len(main) or "dist_dnl" not in ds:
        print(f"  сверка пропущена: {src.name} не совпадает с сеткой листа")
        return
    for org in ("dist_dnl", "dist_dnara"):
        for osm in ("osm_dist_river", "osm_dist_stream", "osm_dist_lake"):
            r = pd.Series(main[osm].to_numpy()).corr(pd.Series(ds[org].to_numpy()), method="spearman")
            print(f"  {org} ~ {osm}: Spearman {r:+.3f}")


def main() -> None:
    print("--- основной лист ---", flush=True)
    main_df = build(MAIN_META, "main")
    check_against_org_rivers(main_df)
    print("--- широкая сетка ---", flush=True)
    build(WIDE_META, "wide")


if __name__ == "__main__":
    main()
