r"""ВЕРСИЯ К1 (этот файл, 26.10.2026): сверка с GTK, ВОССТАНОВЛЕННАЯ по отчёту
25.10 на компьютере, где не было исходного кода, — зерно 2510, плюс площадка
finland65 и слои GTK по площадкам (``gtk_path``). По ней посчитаны числа отчёта
К1 и статьи ИПУЗ (20.3 % / 3.4 %, p = 0.0063). Исходный код, объявленный
24.10.2026 (зерно 11), — ``openlin_transfer_check``; при сведении двух
компьютеров 05.11.2026 файлы разведены по именам, выходы у обоих — в
``outputs/metrics/openlin_transfer_check_<site>*.csv`` (для Лапландии имена
совпадают: перезапуск одного затрёт выход другого).

Сверка перенесённых линий с независимым эталоном линеаментов GTK.

ЗАЧЕМ. На Анабаре эталона линеаментов нет, и тракт судится только
безэталонными условиями (подделка, сдвиговый нуль, устойчивость, длина). На
Лапландии эталон есть: GTK Lineaments 1:500 000 (Engström et al. 2025,
Norwegian Journal of Geology 105, doi 10.17850/njg105-1-3, CC BY 4.0). Сверка
отвечает на вопрос, связаны ли линии, построенные без эталона, с ручной
картой — и насколько.

ИСТОРИЯ. Модуль впервые написан и прогнан 25.10.2026 в копии D:\projects\GEO;
сюда та копия не перенесена. Здесь он ВОССТАНОВЛЕН по отчёту
(audit/lapland_report_25.10_REPORT_ONLY.md) 26.10.2026. Объявление ниже
повторяет объявление отчёта; места, которые отчёт не уточнял, помечены
«ВОССТАНОВЛЕНО» — они выбраны до прогона этой копии и после не меняются.

ЭТАЛОН НА ВХОД ТРАКТА НЕ ИДЁТ. Он скачивается и читается только здесь, после
того как линии построены (openlin_zone_lines_v3 --перенос) и сохранены.

ОБЪЯВЛЕНИЕ (отчёт 25.10.2026; повторено до прогона 26.10.2026):
* слой — 292 (lineament_500k_lidar: ручные линеаменты по отмывке 2-м ЦМР
  LiDAR, ближе всего к нашему входу); 291, 293, 294 — для сведения, без p;
* метрика — ПОЛНОТА: доля ячеек эталона на сетке 20 м, лежащих в пределах
  ``TOL_M`` = 300 м от ячеек наших линий. 300 м — масштаб дешифрирования
  эталона (растр около 300 м), поэтому точность по нему не показательна;
* нуль — тороидальный сдвиг растра эталона по листу, ``N_SHIFTS`` = 159;
  p = (1 + число сдвигов с полнотой >= наблюдаемой) / (1 + 159);
* проверок 2 (два набора: «в зонах» C и «весь лист» C_all), alpha = 0.05/2 =
  0.025; пол p 1/160 = 0.00625, запас alpha/пол = 4x. Свой счётчик, в цепочку
  ветки (OPENLIN_RULES.md) не входит;
* условие: p <= alpha; подъём над средним нуля — размер эффекта, не условие.
ВОССТАНОВЛЕНО (отчёт не уточнял; закреплено до прогона этой копии):
* сдвиг по каждой оси — целое число ячеек из [m, n - m], m = min(200, n // 8)
  («поля min(200, H/8)» отчёта): сдвиг не меньше 4 км и не почти полный круг;
  зерно ``SEED`` = 2510 (дата прогона отчёта). Исходных 159 сдвигов нет — этот
  нуль новая реализация того же правила, а не повтор (ТЗ К1, Пакет 05);
* ячейки эталона учитываются только внутри валидной области композита (там,
  где линии в принципе могли быть найдены) — и в наблюдении, и после сдвига;
* эталон обрезается по рамке листа, растеризация без all_touched.

РЕЗУЛЬТАТ (26.10.2026, лог ``_transfer_check_lapland.log``) — оба набора
ПРОЙДЕНЫ, вывод отчёта подтверждён; числа отличаются от отчёта на 0.1-0.3 п.п.:
* в зонах: полнота 3.4 % (отчёт 3.3), нуль 1.4 / 2.2 % (1.5 / 2.2), подъём
  2.41x (2.21), p = 0.0063;
* весь лист: полнота 20.3 % (20.1), нуль 13.4 / 15.6 % (13.4 / 15.7), подъём
  1.52x (1.50), p = 0.0063;
* для сведения: наших линий у эталона 41.4 % / 27.7 % (41.8 / 28.1) при
  полосе 16.8 % площади (17.1); полнота 150/300/600/1000 м: в зонах
  1.9/3.4/6.7/10.5 %, весь лист 10.9/20.3/37.0/53.0 %; слои 291 — 3.0 % /
  18.5 %, 293 — 4.6 % / 15.1 %, 294 — 2.6 % / 14.7 %.
Длина эталона 292: 100 отрезков, 2690 км целиком (это и есть «около 2690»
отчёта — отрезки, задевающие лист, БЕЗ обрезки), 1894 км в рамке листа,
1593 км на сетке 20 м. Число отчёта 2529 км не воспроизведено ни одной
обрезкой (лист 1894, лист + 5 км 2164, рамка WGS84 с полями 2183).
Наши длины 135 / 1293 км — это ячейки x 20 м, по трассам те же наборы дают
140 / 1339 км (audit/DISCREPANCIES.md, D03-D04). Причина остаточной разницы
полноты не установлена: исходный код сверки (копия D:) недоступен.

Запуск из корня:
    python -m experiments.openlin_transfer_check_k1 --site lapland
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
COMP = Path("datacache/s2_lowsun/composite")
GTK_DIR = Path("datacache/gtk_lineaments")
GTK_URL = ("https://gtkdata.gtk.fi/arcgis/rest/services/Rajapinnat/"
           "kokoavaWMS/MapServer/{layer}/query")
LAYERS = {291: "объединённый", 292: "LiDAR (основной)", 293: "магнитка",
          294: "электроразведка"}
MAIN_LAYER = 292

# ---- объявление (см. шапку) -------------------------------------------------
TOL_M = 300.0
N_SHIFTS = 159
N_TESTS = 2
ALPHA = 0.05 / N_TESTS                    # 0.025
SEED = 2510
TOL_SWEEP_M = (150.0, 300.0, 600.0, 1000.0)   # для сведения
# ----------------------------------------------------------------------------
assert 1.0 / (N_SHIFTS + 1) <= ALPHA / 2, "порог недостижим при этом числе сдвигов"


def gtk_path(site: str, layer: int) -> Path:
    """Кэш слоя GTK по площадке; у Лапландии — прежнее имя без префикса."""
    return GTK_DIR / (f"lin500k_{layer}.gpkg" if site == "lapland" else f"{site}_lin500k_{layer}.gpkg")


def fetch_layer(layer: int, bbox_m, epsg: int, log, site: str = "lapland") -> Path:
    """Слой GTK в рамке листа через ArcGIS REST (GeoJSON, постранично)."""
    import geopandas as gpd

    GTK_DIR.mkdir(parents=True, exist_ok=True)
    p = gtk_path(site, layer)
    if p.exists():
        return p
    x0, y0, x1, y1 = bbox_m
    feats, off = [], 0
    while True:
        q = {"where": "1=1", "outFields": "*", "f": "geojson",
             "geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope",
             "inSR": epsg, "outSR": epsg, "spatialRel": "esriSpatialRelIntersects",
             "resultOffset": off, "resultRecordCount": 1000}
        url = GTK_URL.format(layer=layer) + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=120) as r:
            js = json.load(r)
        got = js.get("features", [])
        feats += got
        if len(got) < 1000:
            break
        off += len(got)
    g = gpd.GeoDataFrame.from_features(feats, crs=f"EPSG:{epsg}")
    g.to_file(p, driver="GPKG")
    log(f"  слой {layer}: скачано объектов {len(g)} -> {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="lapland")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import box

    t0 = time.time()
    w = np.load(COMP / f"winter_{a.site}_20m_лист.npz")
    res, x0, ytop, epsg = float(w["res_m"]), float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    lines = np.load(M / f"openlin_zone_lines_{a.site}_v3_перенос.npz")
    H, W = lines["C"].shape
    valid = np.isfinite(w["shade"])[:H, :W]
    bbox = (x0, ytop - H * res, x0 + W * res, ytop)
    tr = Affine(res, 0, x0, 0, -res, ytop)
    log(f"площадка {a.site}: сетка {H} x {W}, шаг {res:.0f} м, EPSG:{epsg}, "
        f"валидно {100 * valid.mean():.1f} %")
    log(f"объявление: допуск {TOL_M:.0f} м, {N_SHIFTS} сдвигов эталона, "
        f"alpha = 0.05/{N_TESTS} = {ALPHA}, зерно {SEED}")

    m = min(200, H // 8), min(200, W // 8)
    rng = np.random.default_rng(SEED)
    shifts = [(int(rng.integers(m[0], H - m[0] + 1)), int(rng.integers(m[1], W - m[1] + 1)))
              for _ in range(N_SHIFTS)]
    np.savetxt(M / f"openlin_transfer_check_{a.site}_shifts.csv", np.array(shifts),
               fmt="%d", delimiter=",", header="dy_cells,dx_cells", comments="")

    refs = {}
    for layer in LAYERS:
        g = gpd.read_file(fetch_layer(layer, bbox, epsg, log, a.site)).to_crs(epsg=epsg)
        g = g.clip(box(*bbox))
        g = g[~g.geometry.is_empty]
        r = rasterize(((geom, 1) for geom in g.geometry), out_shape=(H, W),
                      transform=tr, fill=0, dtype="uint8").astype(bool)
        refs[layer] = r
        log(f"  слой {layer} ({LAYERS[layer]}): отрезков {len(g)}, длина по "
            f"вектору {g.length.sum() / 1000:.0f} км; ячеек {int(r.sum())} "
            f"(x {res:.0f} м = {r.sum() * res / 1000:.0f} км), из них в валидной "
            f"области {int((r & valid).sum())} ({(r & valid).sum() * res / 1000:.0f} км)")

    def coverage(dist, ref):
        rv = ref & valid
        return float((dist[rv] <= TOL_M).mean()) if rv.any() else np.nan

    rows = []
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    sets = (("в зонах", "C"), ("весь лист", "C_all"))
    log("")
    for name, key in sets:
        ours = lines[key][:H, :W]
        dist = distance_transform_edt(~ours, sampling=(res, res))
        log(f"набор «{name}»: ячеек линий {int(ours.sum())} "
            f"(x {res:.0f} м = {ours.sum() * res / 1000:.0f} км)")
        for layer in LAYERS:
            ref = refs[layer]
            obs = coverage(dist, ref)
            null = np.array([coverage(dist, np.roll(np.roll(ref, dy, 0), dx, 1))
                             for dy, dx in shifts])
            p = (1 + int((null >= obs).sum())) / (N_SHIFTS + 1)
            lift = obs / max(null.mean(), 1e-12)
            row = {"набор": name, "слой": layer, "допуск_м": TOL_M, "полнота": obs,
                   "нуль_ср": null.mean(), "нуль_макс": null.max(), "подъём": lift,
                   "p": p if layer == MAIN_LAYER else np.nan}
            if layer == MAIN_LAYER:
                band = valid & (distance_transform_edt(~(ref & valid), sampling=(res, res)) <= TOL_M)
                near = float(band[ours & valid].mean()) if ours.any() else 0.0
                share = float(band.sum() / valid.sum())
                row |= {"наших_у_эталона": near, "доля_полосы": share}
                log(f"  слой {layer}: полнота {100 * obs:.1f} %, нуль {100 * null.mean():.1f} "
                    f"/ {100 * null.max():.1f} % (сред./макс.), подъём {lift:.2f}x, "
                    f"p = {p:.4f} (порог {ALPHA}) -> {mark[p <= ALPHA]}")
                log(f"    для сведения: наших линий у эталона {100 * near:.1f} % при полосе "
                    f"{100 * share:.1f} % площади ({near / max(share, 1e-12):.2f}x)")
                sweep = []
                for tol in TOL_SWEEP_M:
                    rv = ref & valid
                    sweep.append(float((dist[rv] <= tol).mean()))
                    rows.append({"набор": name, "слой": layer, "допуск_м": tol,
                                 "полнота": sweep[-1], "для_сведения": True})
                log("    полнота при " + " / ".join(f"{t:.0f}" for t in TOL_SWEEP_M)
                    + " м: " + " / ".join(f"{100 * s:.1f}" for s in sweep) + " %")
            else:
                log(f"  слой {layer} ({LAYERS[layer]}), для сведения: полнота "
                    f"{100 * obs:.1f} % ({lift:.2f}x)")
            rows.append(row)
            np.savetxt(M / f"openlin_transfer_check_{a.site}_null_{key}_{layer}.csv",
                       null, fmt="%.6f", header="coverage", comments="")
    import pandas as pd
    pd.DataFrame(rows).to_csv(M / f"openlin_transfer_check_{a.site}.csv", index=False,
                              encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_transfer_check_{a.site}.csv'}, сдвиги и нули "
        f"по каждому набору и слою ({time.time() - t0:.0f} с)")
    plot(a.site, w, lines, refs[MAIN_LAYER], valid, log)


def plot(site, w, lines, ref, valid, log) -> None:
    """Весь лист: зимний снимок, наши линии (весь лист) и эталон 292."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    H, W = valid.shape
    base = np.nan_to_num(_stretch(np.where(valid, w["shade"][:H, :W], np.nan)), nan=1.0)
    k = np.ones((5, 5), bool)
    s = slice(None, None, 4)
    ov = np.zeros(base[s, s].shape + (4,), "float32")
    ov[binary_dilation(ref, k)[s, s]] = (0.1, 0.4, 0.95, 0.9)
    ov[binary_dilation(lines["C_all"][:H, :W], k)[s, s]] = (0.88, 0.05, 0.05, 0.9)
    fig, ax = plt.subplots(figsize=(13, 12), facecolor="#fcfcfb")
    ax.imshow(base[s, s], cmap="gray", vmin=0, vmax=1)
    ax.imshow(ov)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{site}: красные — наши линии (весь лист), синие — GTK 292 (LiDAR), "
                 "север сверху", loc="left", fontsize=13)
    fig.tight_layout()
    p = Path("outputs/steps") / f"перенос_{site}_сверка_GTK.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
