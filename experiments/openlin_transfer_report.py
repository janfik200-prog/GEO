"""Пакет 09 ТЗ К1: различия площадок и где тракт отказывает (Анабар, Лапландия, finland65).

Объявление — ``audit/PACKET09_DECLARATION.md``. Описание, без условий:
вода, лес (NDVI > 0.35), облачность лета, зимние сцены, сцены на ячейку у
полного композита и половин, рельеф (ArcticDEM); подтверждение по углу к
солнцу; полнота против эталона v2 по углу к солнцу; связь F1 участка с долей
одиночной сцены в половине «Б».

Запуск из корня (после прогонов всех площадок):
    python -m experiments.openlin_transfer_report
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import REF, TOL_M, grid, prf, rast

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
OUT = Path("outputs/k1/transfer")
FIG = Path("outputs/k1/figures")
SITES = ("anabar", "lapland", "finland65")
STEM = {"anabar": "openlin_zone_lines_anabar_v3_бл"}
BANDS_DEG = ((0, 15), (15, 30), (30, 60), (60, 90.01))


def stem(site):
    return STEM.get(site, f"openlin_zone_lines_{site}_v3_перенос")


def water_share(site, g):
    import geopandas as gpd
    from shapely.ops import polygonize, unary_union
    osm = gpd.read_file(REF / f"{site}_osm.gpkg")
    polys = list(polygonize(unary_union(list(osm[osm.cls == "shore"].geometry.values)))) \
        if (osm.cls == "shore").any() else []
    m = rast(polys, g) if polys else np.zeros((g["H"], g["W"]), bool)
    if site == "anabar":
        import geopandas as gpd2
        from src import config
        from experiments.build_dataset_v1 import read_sidecar_proj4
        p = Path(config.TOPO_SHP_DIR) / "dnara.shp"
        d = gpd2.read_file(p)
        if d.crs is None:
            d = d.set_crs(read_sidecar_proj4(p), allow_override=True)
        m |= rast(d.to_crs(epsg=g["epsg"]).geometry, g)
    return float(m[g["valid"]].mean())


def log_number(path, pattern, cast=float):
    t = Path(path).read_text(encoding="utf-8", errors="ignore") if Path(path).exists() else ""
    m = re.search(pattern, t)
    return cast(m.group(1)) if m else None


def characteristics(site):
    g = grid(site)
    z = np.load(M / f"openlin_{site}.npz")
    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    hb = np.load(COMP / f"winter_{site}_20m_лист_Б.npz")["count"]
    ha = np.load(COMP / f"winter_{site}_20m_лист_А.npz")["count"]
    dem = np.load(f"datacache/arcticdem/{site}_10m.npy")
    H, W = g["H"], g["W"]
    d20 = np.nanmean(dem.reshape(H, 2, W, 2), axis=(1, 3))
    gy, gx = np.gradient(np.where(np.isfinite(d20), d20, np.nanmean(d20)), 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    tiles = sorted({next(p for p in s.split("_") if p.startswith("T")) for s in w["scenes"]})
    dates = sorted(s.split("_")[2][:8] for s in w["scenes"])
    v = g["valid"]
    plog = M / f"_probe_{site}.log"
    return {
        "площадь_км2": round(float(v.sum()) * 0.0004),
        "вода_доля": water_share(site, g),
        "лес_NDVI>0.35": float(np.nanmean(z["ndvi"] > 0.35)),
        "лето_сцен_годно": log_number(plog, r"композит из (\d+) сцен", int),
        "лето_ячеек_3+": log_number(plog, r"наблюдений: ([\d.]+)%"),
        "S1_сцен": log_number(plog, r"vv: композит из (\d+) сцен", int),
        "зима_сцен": int(len(w["scenes"])), "зима_даты": f"{dates[0]}..{dates[-1]}",
        "зима_плитки": tiles, "солнце_высота": [float(w["sun"].min()), float(w["sun"].max())],
        "солнце_азимут_медиана": float(np.median(w["azimuth"])),
        "сцен_в_ячейке_медиана": float(np.median(w["count"][v])),
        "половина_А_сцен_медиана": float(np.median(ha[v])),
        "половина_Б_сцен_медиана": float(np.median(hb[v])),
        "половина_Б_одна_сцена_доля": float((hb[v] == 1).mean()),
        "уклон_медиана_ArcticDEM": float(np.median(slope[v])),
    }, g, hb


def confirm_by_sun(site):
    txt = (M / (f"_zone_lines_{site}_v3_перенос.log" if site != "anabar" else "_zone_lines_anabar_v3_бл.log")
           ).read_text(encoding="utf-8", errors="ignore")
    out = {}
    for lo, hi, n, p in re.findall(r"\s(\d+)-\s?(\d+)°: линий\s+(\d+), подтверждено\s+(\d+) %", txt):
        out[f"{lo}-{hi}"] = {"линий": int(n), "подтверждено": int(p) / 100}
    return out


def main() -> None:
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt as edt
    from scipy.stats import spearmanr

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_baselines import eval_area
    res, pool = {}, []
    for site in SITES:
        ch, g, hb = characteristics(site)
        lines = np.load(M / f"{stem(site)}.npz")["C_all"][:g["H"], :g["W"]]
        ok = eval_area(site, g)
        ref = gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg")
        refm = rast(ref.geometry, g) & ok
        ours = lines & ok
        d_o, d_r = edt(~ours, sampling=20.0), edt(~refm, sampling=20.0)
        xy = np.array([list(gm.coords) for gm in ref.geometry])
        az = np.degrees(np.arctan2(xy[:, 1, 0] - xy[:, 0, 0], xy[:, 1, 1] - xy[:, 0, 1])) % 180
        sun = (184.0 if site == "anabar" else ch["солнце_азимут_медиана"]) % 180
        ang = np.abs(((az - sun) + 90) % 180 - 90)
        band = {}
        for lo_, hi_ in BANDS_DEG:
            rm = rast(ref.geometry[(ang >= lo_) & (ang < hi_)], g) & ok
            band[f"{lo_}-{int(hi_)}"] = float((d_o[rm] <= TOL_M).mean()) if rm.any() else None
        blocks = []
        for i in range(g["H"] // 1000):
            for j in range(g["W"] // 1000):
                sl = (slice(i * 1000, (i + 1) * 1000), slice(j * 1000, (j + 1) * 1000))
                if ok[sl].mean() < 0.95:
                    continue
                f = prf(ours[sl], refm[sl], d_o[sl], d_r[sl], TOL_M)[2]
                one = float((hb[sl][g["valid"][sl]] == 1).mean())
                blocks.append({"участок": f"{i}_{j}", "F1_v2": f, "Б_одна_сцена": one})
                pool.append((site, f, one))
        res[site] = {"характеристики": ch, "подтверждение_по_углу_к_солнцу": confirm_by_sun(site),
                     "полнота_v2_по_углу_к_солнцу": band, "участки": blocks}
        log(f"\n{site}: " + "; ".join(f"{k} {v}" for k, v in ch.items()))
        log("  подтверждение по углу к солнцу: " + ", ".join(
            f"{k}° {100 * v['подтверждено']:.0f} % ({v['линий']})" for k, v in res[site]["подтверждение_по_углу_к_солнцу"].items()))
        log("  полнота против v2 по углу к солнцу: " + ", ".join(
            f"{k}° {100 * v:.1f} %" for k, v in band.items() if v is not None))
        if len(blocks) >= 4:
            rho = spearmanr([b["F1_v2"] for b in blocks], [b["Б_одна_сцена"] for b in blocks]).statistic
            res[site]["Спирмен_F1_против_одной_сцены_Б"] = float(rho)
            log(f"  F1 участка против доли «одна сцена в половине Б»: Спирмен {rho:+.2f} по {len(blocks)} участкам")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "site_characteristics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\nсохранено: {OUT / 'site_characteristics.json'}")


if __name__ == "__main__":
    main()
