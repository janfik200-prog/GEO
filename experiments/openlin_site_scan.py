"""Выбор площадки для настройки тракта: квадраты листа по Финляндии.

ЗАЧЕМ. Тракт ветки настраивался на Анабаре, где разметки нет, — только глаз
и проверки против нуля. Лапландия (``SITES["lapland"]``) ушла на разовую
проверку переноса и подстраиваться под неё нельзя. Нужна третья площадка —
для настройки: похожая на Анабар по ландшафту и с разметкой по ДЗЗ (слой
GTK 292, линии по отмывке LiDAR). Только замеры, тракт здесь не запускается.

ЧТО МЕРИТСЯ. Квадраты ``TILE_KM`` (размер листа R-48-XI,XII) с шагом 20 км в
TM35FIN; для каждого — доля в Финляндии (эталон GTK есть только там), длина
линий GTK 292 и 291 внутри квадрата, близость к контрольной Лапландии, состав
покрова по ESA WorldCover 2021 и его отличие от Анабара: сумма модулей
разниц долей классов суши (вода исключена, доли перенормированы); 0 — тот же
состав, 2 — ничего общего.

ОТБОР (объявлен 05.11.2026 до просмотра результатов):
  1) квадрат в Финляндии не меньше чем на 95 % — иначе часть листа без эталона;
  2) не задевает контрольную Лапландию с запасом 20 км;
  3) длина GTK 292 не меньше половины лапландской (947 км из 1894) — чтобы было
     на чём мерить; по длине НЕ ранжируем: центр Лапландии брали по максимуму
     длины, и это смещение выбора повторять не нужно;
  4) ранжирование — только по отличию покрова от Анабара.

РЕЗУЛЬТАТ (05.11.2026): квадратов 548, проходят 1–3 — 184. Анабар: лес 3 %,
трава 34 %, мох и лишайник 62 %. Контрольная Лапландия: лес 76 %, отличие
1.64 — перенос проверен на ландшафте, почти противоположном Анабару. Отличие
меньше 1.2 — только у пяти квадратов, все у Утсйоки (69.4–69.6° с. ш.);
лучший, прошедший отбор, — 69.551° с. ш., 27.142° в. д.: лес 21 %, трава 54 %,
мох 15 %, GTK 292 1119 км, отличие 0.94 (→ ``SITES["utsjoki"]``). Остальные
квадраты — лес 60–85 %, отличие 1.29 и больше. Тундровые сопки северо-запада
(Энонтекиё) не проходят условие 1: финская часть там уже листа.

    python -m experiments.openlin_site_scan
"""
import json

import numpy as np

from cache_paths import CACHE
from experiments.openlin_probe import SITES, TILE_KM, build_grid

WC_CLASSES = {10: "лес", 20: "кустарник", 30: "трава", 40: "пашня", 50: "застройка",
              60: "голый", 70: "снег", 80: "вода", 90: "болото", 100: "мох_лишайник"}
#: Шаг мозаики покрова, ~900 м по широте: долю класса на квадрате 86 x 77 км
#: это не сдвигает, а читаются только обзоры COG, а не 10-метровые плитки.
RES_DEG = 1 / 120
STEP_M = 20_000
EPSG_FI = 3067
#: Финляндия — по Natural Earth 1:10m (naciscdn.org, ne_10m_admin_0_countries).
NE_ZIP = CACHE / "natural_earth" / "ne_10m_admin_0_countries.zip"
WC_DIR = CACHE / "worldcover"
GTK_DIR = CACHE / "gtk_lineaments"
FIN_MIN, LAP_GAP_M, L292_MIN_KM = 0.95, 20_000, 947.0


def worldcover_mosaic(bbox, cache):
    """Мозаика WorldCover 2021 (v200) в WGS84 с шагом RES_DEG, по обзорам COG."""
    import rasterio
    if cache.exists():
        z = np.load(cache)
        return z["a"], tuple(z["t"])
    import planetary_computer as pc
    from pystac_client import Client
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=bbox,
                            query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    print(f"  WorldCover: {len(items)} плиток", flush=True)
    lon0, lat0, lon1, lat1 = bbox
    W = int(np.ceil((lon1 - lon0) / RES_DEG))
    H = int(np.ceil((lat1 - lat0) / RES_DEG))
    a = np.zeros((H, W), np.uint8)
    for it in items:
        with rasterio.open(it.assets["map"].href) as src:
            b = src.bounds
            n = int(round((b.right - b.left) / RES_DEG))
            arr = src.read(1, out_shape=(n, n),
                           resampling=rasterio.enums.Resampling.nearest)
        c0 = int(round((b.left - lon0) / RES_DEG))
        r0 = int(round((lat1 - b.top) / RES_DEG))
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + n, H), min(c0 + n, W)
        if re <= rs or ce <= cs:
            continue
        sub = arr[rs - r0:re - r0, cs - c0:ce - c0]
        dst = a[rs:re, cs:ce]
        dst[sub > 0] = sub[sub > 0]
    t = (lon0, lat1, RES_DEG)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, a=a, t=np.array(t))
    return a, t


def composition(a, t, geom4326) -> dict[str, float]:
    """Доли классов WorldCover внутри многоугольника (без пустых ячеек)."""
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    m = ~geometry_mask([geom4326], a.shape, from_origin(t[0], t[1], t[2], t[2]))
    v = a[m]
    v = v[v > 0]
    return {name: float((v == k).mean()) if v.size else np.nan
            for k, name in WC_CLASSES.items()}


def land_l1(c1, c2, skip=("вода",)) -> float:
    """Отличие состава суши: сумма |разниц долей|, вода исключена и доли перенормированы."""
    k = [n for n in WC_CLASSES.values() if n not in skip]
    s1 = sum(c1[n] for n in k) or 1
    s2 = sum(c2[n] for n in k) or 1
    return sum(abs(c1[n] / s1 - c2[n] / s2) for n in k)


def site_box(name):
    import geopandas as gpd
    from shapely.geometry import box
    g = build_grid(SITES[name], 1000.0)
    b = box(g.x0, g.y_top - g.height * g.res_m, g.x0 + g.width * g.res_m, g.y_top)
    return gpd.GeoSeries([b], crs=g.epsg)


def main() -> None:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    fin = gpd.read_file(NE_ZIP)
    fin = fin[fin["ADMIN"] == "Finland"].to_crs(EPSG_FI).union_all()
    lap = site_box("lapland").to_crs(EPSG_FI).iloc[0]
    ana = site_box("anabar").to_crs(4326).iloc[0]
    gtk = {l: gpd.read_file(GTK_DIR / f"lin500k_{l}.gpkg").to_crs(EPSG_FI)
           for l in (291, 292)}

    wc_fi, t_fi = worldcover_mosaic([19.0, 59.5, 32.0, 70.2], WC_DIR / "wc_finland.npz")
    ab = list(ana.bounds)
    wc_an, t_an = worldcover_mosaic([ab[0] - 0.1, ab[1] - 0.1, ab[2] + 0.1, ab[3] + 0.1],
                                    WC_DIR / "wc_anabar.npz")
    c_an = composition(wc_an, t_an, ana)
    c_lap = composition(wc_fi, t_fi, site_box("lapland").to_crs(4326).iloc[0])

    w, h = TILE_KM[0] * 1000, TILE_KM[1] * 1000
    x0, y0, x1, y1 = fin.bounds
    rows = []
    for cy in np.arange(y0 + h / 2, y1 - h / 2 + 1, STEP_M):
        for cx in np.arange(x0 + w / 2, x1 - w / 2 + 1, STEP_M):
            t = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            fin_frac = t.intersection(fin).area / t.area
            if fin_frac < FIN_MIN:
                continue
            g4326 = gpd.GeoSeries([t], crs=EPSG_FI).to_crs(4326).iloc[0]
            c = composition(wc_fi, t_fi, g4326)
            L = {l: float(gtk[l].clip(t).length.sum() / 1000) for l in gtk}
            rows.append(dict(lat=round(g4326.centroid.y, 3), lon=round(g4326.centroid.x, 3),
                             fin=round(fin_frac, 3),
                             near_lap=bool(t.buffer(LAP_GAP_M).intersects(lap)),
                             L292=round(L[292], 1), L291=round(L[291], 1),
                             d_anabar=round(land_l1(c, c_an), 3),
                             **{k: round(v, 3) for k, v in c.items()}))
    df = pd.DataFrame(rows)
    df.to_csv("outputs/metrics/openlin_site_scan_finland.csv", index=False)
    ok = df[(~df.near_lap) & (df.L292 >= L292_MIN_KM)].sort_values("d_anabar")
    with open("outputs/metrics/openlin_site_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(anabar=c_an, lapland=c_lap, d_lap_anabar=land_l1(c_lap, c_an),
                       n_tiles=len(df), n_ok=len(ok), best=ok.iloc[0].to_dict()),
                  f, ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250)
    print("Анабар:", {k: round(v, 3) for k, v in c_an.items()})
    print("Лапландия (контроль):", {k: round(v, 3) for k, v in c_lap.items()},
          "отличие", round(land_l1(c_lap, c_an), 3))
    print(f"квадратов {len(df)}, проходят условия {len(ok)}")
    print(ok.head(15)[["lat", "lon", "L292", "L291", "d_anabar", "лес", "трава",
                       "мох_лишайник", "голый", "болото", "вода"]].to_string())


if __name__ == "__main__":
    main()
