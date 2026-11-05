"""Проверочный (не тракт): приборные и ложные прямые в наборе линий ПО РЕЛЬЕФУ (площадка настройки).

Набор: outputs/metrics/openlin_zone_lines_<site>_v3_рельеф1.npz (вход — уклон GLO-30,
подтверждение — уклон ALOS; код experiments/openlin_relief_lines.py). Только замер: пороги тракта
не трогаются. Слои снимка (OSM, вода, лес, граница ЦМР) берутся из openlin_artifact_site;
швы композита к рельефу не относятся и из объяснений убраны. Добавлено для рельефа:
  * граница плитки ЦМР — целые градусы широты и долготы; плитки у GLO-30 и ALOS одинаковые
    (1° x 1°), поэтому слой общий; дополнительно замер скачка расхождения моделей на самих границах;
  * «края расхождения GLO-30 - ALOS»: полосы и ступеньки ОДНОЙ модели (TanDEM-X / PRISM) видны как
    линейные перепады в разности моделей, а настоящий рельеф в обеих моделях общий и в разности
    гасится. Слой = ячейки, где модуль градиента сглаженной разности >= 95-го процентиля (мера
    по данным, значение печатается); угол линии — по градиенту разности, допуск 20 °;
  * «плоско (квантование ALOS)»: ALOS отдаётся целыми метрами (доля целых 100 %), значит на плоском месте
    поле — ступеньки по 1 м. Слой = ячейки, где перепад ALOS в окне 9 x 9 (180 м) <= 1 м
    (один шаг квантования), без 3 ячеек от края области;
  * азимут съёмки: у солнечно-синхронного радара наклонение i = 97.44 ° и след на широте phi
    отклонён от меридиана на asin(|cos i| / cos phi); полосы вытянуты по следу. Печатается доля
    длины набора в секторе след ± 20 ° (допуск тракта) против изотропной доли сектора;
  * высота стены леса: перепад поверхности через край леса (средняя над лесом минус над открытым
    в окне 11 x 11) против той же величины на сдвинутых краях (рельеф без леса).
Запуск:  python -m experiments.openlin_artifact_relief --site utsjoki
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.ndimage import (distance_transform_edt, gaussian_filter, maximum_filter,
                           minimum_filter, uniform_filter)

import experiments.openlin_artifact_site as m
from experiments.openlin_relief_lines import ALOS_BAD_M

NPZ = "outputs/metrics/openlin_zone_lines_{site}_v3_рельеф1.npz"
DEM = "datacache/anabar_dem/dem_{kind}_{site}_лист_20m.npy"
INCL_DEG = 97.44            # наклонение орбиты TerraSAR-X / TanDEM-X
EXPLAIN_REL = ("граница ЦМР", "края расхождения GLO-ALOS", "плоско (квантование ALOS)",
               "OSM техногенное (всё)", "вода: берег или русло", "внутри воды")
GROUPS_REL = {"техногенное": ["OSM техногенное (всё)"], "вода": ["вода: берег или русло"],
              "край леса": ["край леса"], "граница плитки ЦМР": ["граница ЦМР"],
              "полосы/ступени модели": ["края расхождения GLO-ALOS"],
              "плоско (квантование)": ["плоско (квантование ALOS)"]}


def load_dems(d):
    r0, c0 = d["crop"]
    H, W = d["shape"]
    sl = (slice(r0, r0 + H), slice(c0, c0 + W))
    g = np.load(DEM.format(kind="glo30", site=d["site"]))[sl].astype("float64")
    a = np.load(DEM.format(kind="alos", site=d["site"]))[sl].astype("float64")
    a = np.where(np.abs(a - g) > ALOS_BAD_M, np.nan, a)
    return g, a


def add_relief_layers(d, log):
    g, a = load_dems(d)
    layers = d["layers"]
    # --- расхождение моделей
    D = np.where(np.isfinite(a), g - a, np.nan)
    Df = np.where(np.isfinite(D), D, np.nanmedian(D))
    Ds = gaussian_filter(Df, 1.0)
    gy, gx = np.gradient(Ds)
    gm = np.hypot(gx, gy)
    q95 = float(np.percentile(gm, 95))
    mask = gm >= q95
    ang = m.grad_ang(Df, mask)
    near, nang = m.make_near(mask, ang)
    layers["края расхождения GLO-ALOS"] = {"near": near, "nang": nang}
    log(f"расхождение GLO-30 - ALOS: медиана {np.nanmedian(D):+.2f} м, 5/95 % {np.nanpercentile(D, 5):+.1f}/"
        f"{np.nanpercentile(D, 95):+.1f} м; край расхождения: модуль градиента >= {q95:.3f} м/ячейку (95-й процентиль), "
        f"ячеек {100 * mask.mean():.2f} %")
    # --- квантование ALOS
    v = a[np.isfinite(a)]
    log(f"ALOS: доля целых метров {100 * np.mean(v == np.round(v)):.1f} %; GLO-30: "
        f"{100 * np.mean(g == np.round(g)):.3f} %")
    af = np.where(np.isfinite(a), a, g)
    rng9 = maximum_filter(af, 9) - minimum_filter(af, 9)
    flat = rng9 <= 1.0
    flat_in = flat & (distance_transform_edt(flat) > m.NEAR_C)
    layers["плоско (квантование ALOS)"] = {"near": flat_in}
    log(f"плоско (перепад ALOS в 180 м <= 1 м): {100 * flat.mean():.2f} % площади, внутри {100 * flat_in.mean():.2f} %")
    # --- скачок на границах плиток
    edge = layers["граница ЦМР"]["near"]
    band = edge
    if band.any():
        log(f"граница плитки: модуль градиента расхождения у границы {gm[band].mean():.3f} против "
            f"{gm[~band].mean():.3f} вдали (отношение {gm[band].mean() / gm[~band].mean():.2f}); "
            f"уклон GLO-30 у границы {np.hypot(*np.gradient(g, 20.0))[band].mean():.4f} против "
            f"{np.hypot(*np.gradient(g, 20.0))[~band].mean():.4f}")
    d["dems"] = (g, a, gm)
    return d


def forest_step(d, log, n_sh=300):
    """Высота стены леса: перепад поверхности через край леса, с нулём на сдвинутых краях."""
    g, a, _ = d["dems"]
    wc = d["wc"]
    water = wc == 80
    tree = (wc == 10).astype("float64")
    openm = ((wc != 10) & ~water).astype("float64")
    edge, _ea = d["forest_edge"]
    shp = d["shape"]
    rng = np.random.default_rng(m.SEED)
    H, W = shp
    er, ec = np.nonzero(edge)
    sub = rng.choice(len(er), min(len(er), 200000), replace=False)
    er, ec = er[sub], ec[sub]
    for nm, dem in (("GLO-30", g), ("ALOS", np.where(np.isfinite(a), a, g))):
        k = 11 * 11
        st = uniform_filter(tree, 11) * k
        so = uniform_filter(openm, 11) * k
        mt = uniform_filter(dem * tree, 11) * k / np.maximum(st, 1)
        mo = uniform_filter(dem * openm, 11) * k / np.maximum(so, 1)
        ok = (st >= 20) & (so >= 20)
        diff = np.where(ok, mt - mo, np.nan)
        obs = np.nanmean(diff[er, ec])
        nul = []
        for _ in range(n_sh):
            dr, dc = int(rng.integers(200, H - 200)), int(rng.integers(200, W - 200))
            nul.append(np.nanmean(diff[(er + dr) % H, (ec + dc) % W]))
        nul = np.array(nul)
        log(f"стена леса {nm}: поверхность над лесом минус над открытым через край {obs:+.2f} м "
            f"(нуль на сдвинутых краях {nul.mean():+.2f} м, 95-й {np.quantile(nul, .95):+.2f}; сдвигов {n_sh}); "
            f"p = {(1 + (nul >= obs).sum()) / (1 + n_sh):.4f}")


def heading_share(cells, lat_deg, log, tag):
    off = np.degrees(np.arcsin(abs(np.cos(np.radians(INCL_DEG))) / np.cos(np.radians(lat_deg))))
    sec = (m.angdiff(cells.az, off) <= m.ALIGN_DEG) | (m.angdiff(cells.az, 180 - off) <= m.ALIGN_DEG)
    ln = cells.cnt
    share = ln[sec].sum() / ln.sum()
    iso = 2 * 2 * m.ALIGN_DEG / 180.0
    log(f"азимут съёмки ({tag}): след отклонён от меридиана на {off:.1f} ° (широта {lat_deg:.2f}); сектор "
        f"+-{m.ALIGN_DEG:.0f} ° вокруг {off:.0f} ° и {180 - off:.0f} °: {100 * share:.1f} % длины набора против "
        f"{100 * iso:.0f} % изотропной доли")
    return share


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--n-shift", type=int, default=m.N_SHIFT)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s="": print(s, flush=True)
    d = m.load_site(a.site, NPZ.format(site=a.site), log)
    add_relief_layers(d, log)
    forest_step(d, log)
    # широта центра сетки
    from pyproj import Transformer
    H, W = d["shape"]
    lon, lat = Transformer.from_crs(f"EPSG:{d['epsg']}", "EPSG:4326", always_xy=True).transform(
        d["x0"] + W * m.RES / 2, d["yt"] - H * m.RES / 2)
    for sn in ("C_all", "C", "A_all"):
        heading_share(m.Cells(d["z"][sn]), lat, log, f"рельеф {sn}")
    opt = np.load(m.NPZ_BY_SITE[a.site])
    r0, c0 = d["crop"]
    oc = opt["C_all"][r0:r0 + H, c0:c0 + W]
    heading_share(m.Cells(oc), lat, log, "снимок C_all, для сравнения")
    m.analyse(d, explain=EXPLAIN_REL, groups=GROUPS_REL, log=log, t0=t0, n_shift=a.n_shift)


if __name__ == "__main__":
    main()
