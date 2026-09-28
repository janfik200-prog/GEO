"""Зимний композит при низком солнце: вход, в котором тени не стёрты.

ОТКУДА ПОСТАНОВКА. Сравнение на фрагменте 22 x 22 км показало, что рисунок
теряется дважды и по разным причинам:

* наш летний композит — МЕДИАНА сорока сцен, снятых при разном положении
  солнца. Медиана по построению убирает всё, что зависит от освещения, то есть
  ровно тени. Плато на ней распадается на светлые и тёмные кляксы;
* загрубление до 100 м стирает штриховой рисунок: прямые полосы шириной в
  десятки метров и длиной в километры видны в родных 10 м и исчезают при
  усреднении блоком 10 x 10.

ЧТО ЗДЕСЬ. Обе потери устраняются без смены источника — это тот же Sentinel-2.
Берутся только сцены с солнцем ниже 12 градусов. На широте 71 такое солнце
бывает только около полудня в октябре и феврале, поэтому азимут освещения у
всех сцен одинаков (177-190 градусов). Именно это и делает осмысленным
усреднение: складываются снимки с ОДНОЙ геометрией освещения, тени не гасят
друг друга, а шум, облака и случайные перемёты снега — гасят.

Цена одинакового азимута: солнце всегда с юга, тени ложатся на север, и
структуры, идущие строго север-юг, тенью не выражаются вовсе. Это ограничение
самого приёма, и его надо держать в голове при разборе азимутов найденного.

КАНАЛЫ НА ВЫХОДЕ. Помимо самих B04 и B08 считается ``shade`` — снимок за
вычетом крупномасштабной составляющей. Отнимается сглаженное на 2 км поле,
то есть убирается собственная яркость поверхности (снег против леса, разные
породы), и остаётся только то, что создано наклоном склонов.

Запуск из корня репозитория:
    python -m experiments.openlin_winter --site anabar --scenes 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments import openlin_lowsun
from experiments.openlin_lowsun import (BANDS, CACHE, FRAG_KM, RES_M, fetch,
                                        find_scenes, frag_of_composite,
                                        frag_polygon, scene_grid, _stretch)

OUT = CACHE / "composite"

#: Масштаб, на котором снимается собственная яркость поверхности, м. Крупнее
#: этого рельефная тень не бывает: склон длиной в два километра уже сам по
#: себе форма рельефа, а не уступ.
DETREND_M = 2000.0

#: Сколько сцен обязано быть в ячейке, чтобы медиана считалась. При десяти
#: сценах это отсекает только края и провалы облачной маски.
MIN_SCENES = 3


def site_grid(site: str, res_m: float):
    """Сетка всего листа в его проекции, но с нужным шагом."""
    from experiments.openlin_probe import Grid

    z = dict(np.load(Path("outputs/metrics") / f"openlin_{site}.npz"))
    big = float(z["res_m"])
    h, w = z["gray"].shape
    k = big / res_m
    return Grid(x0=float(z["x0"]), y_top=float(z["y_top"]), res_m=res_m,
                height=int(round(h * k)), width=int(round(w * k)),
                epsg=int(z["epsg"]))


def cover_pick(items, grid, n_max: int, log, n_probe: int = 120) -> list:
    """Отбор сцен на весь лист: по отдельности они его не накрывают.

    Один снимок Sentinel-2 — плитка 110 км, и лист 77 x 86 км попадает на
    стык нескольких плиток и витков. Требовать полного покрытия одной сценой,
    как на фрагменте, здесь нельзя: подходящих не будет ни одной. Зато медиана
    считается по каждой ячейке отдельно, поэтому достаточно, чтобы КАЖДУЮ
    ячейку накрыло не меньше ``MIN_SCENES`` сцен.

    Сцены добавляются жадно: сначала самые низкие солнца, но следующей берётся
    та, что закрывает больше всего ещё недобранных мест.
    """
    from shapely.geometry import Point, shape
    from shapely.prepared import prep

    fp = frag_polygon(grid)
    x0, y0, x1, y1 = fp.bounds
    n = int(np.sqrt(n_probe))
    probes = [Point(x, y)
              for x in np.linspace(x0, x1, n) for y in np.linspace(y0, y1, n)]
    need = np.zeros(len(probes), dtype=int)
    geoms = [prep(shape(it.geometry)) for _s, _a, it in items]
    hits = np.array([[g.contains(pt) for pt in probes] for g in geoms])

    picked, used_day = [], set()
    order = list(range(len(items)))
    while len(picked) < n_max:
        gain, best = -1, None
        for i in order:
            d = items[i][2].datetime.date()
            if d in used_day:
                continue
            g = int((hits[i] & (need < MIN_SCENES)).sum())
            # при равной пользе выигрывает более низкое солнце: список уже
            # отсортирован по нему, поэтому строгое сравнение
            if g > gain:
                gain, best = g, i
        if best is None or gain <= 0:
            break
        need += hits[best]
        used_day.add(items[best][2].datetime.date())
        picked.append(items[best])
    done = 100 * float((need >= MIN_SCENES).mean())
    log(f"  отобрано сцен {len(picked)}; мест с {MIN_SCENES}+ сценами "
        f"{done:.0f} %")
    return picked


def normalize(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Робастная нормировка сцены: разная дымка и разное солнце дают разную
    общую яркость, и без этого одна сцена перетянула бы медиану на себя."""
    v = a[valid]
    if v.size == 0:
        return np.full_like(a, np.nan, dtype="float32")
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    sd = max(float(q3 - q1) / 1.349, 1e-6)
    out = (a - med) / sd
    return np.where(valid, out, np.nan).astype("float32")


def detrend(a: np.ndarray, valid: np.ndarray, res_m: float,
            scale_m: float = DETREND_M) -> np.ndarray:
    """Снимок минус собственная яркость поверхности."""
    from scipy.ndimage import gaussian_filter

    s = scale_m / res_m / 2.0
    x = np.where(valid, np.nan_to_num(a), 0.0).astype("float64")
    w = valid.astype("float64")
    num = gaussian_filter(x, s, mode="nearest")
    den = gaussian_filter(w, s, mode="nearest")
    base = num / np.maximum(den, 1e-6)
    return np.where(valid, a - base, np.nan).astype("float32")


def _ndvi(z: dict) -> np.ndarray | None:
    """Зимний NDVI одной сцены: считается ДО нормировки, по сырым яркостям."""
    if "B04" not in z or "B08" not in z:
        return None
    r = z["B04"].astype("float32")
    n = z["B08"].astype("float32")
    good = np.isfinite(r) & np.isfinite(n) & (r > 0) & (n > 0)
    d = np.where(good, (n - r) / np.maximum(n + r, 1e-6), np.nan)
    return d.astype("float32")


def stack_scenes(got: dict, log) -> tuple[dict, np.ndarray, list]:
    """Медиана по сценам, отдельно по каждому каналу.

    Медиана здесь уместна ровно потому, что складываются сцены с ОДНОЙ
    геометрией освещения: она убирает облака и перемёты снега, а тень, общую
    для всех сцен, оставляет. У летнего композита то же действие убивало
    сигнал, потому что освещение там у каждой сцены своё.
    """
    keys = sorted(got)
    meta = []
    out = {}
    count = None
    for b in BANDS:
        layers = []
        for k in keys:
            if b not in got[k]:
                continue
            a = got[k][b].astype("float32")
            good = np.isfinite(a) & (a > 0)
            layers.append(normalize(a, good))
        if not layers:
            continue
        cube = np.stack(layers)
        n = np.isfinite(cube).sum(0)
        count = n if count is None else np.minimum(count, n)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(cube, axis=0)
        out[b] = np.where(n >= MIN_SCENES, med, np.nan).astype("float32")
        log(f"  {b}: сцен {len(layers)}, ячеек с медианой "
            f"{100 * np.isfinite(out[b]).mean():.1f} %")
    nd = [_ndvi(got[k]) for k in keys]
    nd = [a for a in nd if a is not None]
    if nd:
        with np.errstate(invalid="ignore"):
            out["ndvi"] = np.nanmedian(np.stack(nd), axis=0).astype("float32")
        log(f"  ndvi: сцен {len(nd)}")
    for k in keys:
        meta.append((k, float(got[k]["sun"]), float(got[k]["az"])))
    return out, count, meta


def write_site(site: str, comp: dict, shade: np.ndarray, grid, log) -> Path:
    """Сохранить композит в формате площадки, чтобы шаги 1-5 пошли по нему.

    Весь конвейер читает площадку через ``openlin_unet.load_site``, поэтому
    новый вход подкладывается тем же файлом, а не отдельной веткой кода: так
    зимний композит проходит ровно ту же обработку и ту же фазовую подделку,
    что и летний, и сравнение остаётся честным.
    """
    from experiments.openlin_unet import load_site

    old = load_site(site)
    k = int(round(float(old["res_m"]) / grid.res_m))
    h, w = comp["B04"].shape
    dist = np.repeat(np.repeat(old["dist"], k, 0), k, 1)[:h, :w]
    if dist.shape != (h, w):
        pad = np.full((h, w), np.nan, dtype="float32")
        pad[:dist.shape[0], :dist.shape[1]] = dist
        dist = pad
    out = {"gray": comp["B04"], "shade": shade, "dist": dist.astype("float32"),
           "res_m": grid.res_m, "x0": grid.x0, "y_top": grid.y_top,
           "epsg": grid.epsg, "sigma": old["sigma"]}
    if "ndvi" in comp:
        out["ndvi"] = comp["ndvi"]
    p = Path("outputs/metrics") / f"openlin_{site}_zima.npz"
    np.savez_compressed(p, **out)
    log(f"площадка для конвейера: {p} "
        f"(каналы: {', '.join(c for c in out if np.ndim(out[c]) == 2)})")
    return p


def plot(comp: dict, shade: np.ndarray, site: str, res_m: float, log,
         full: bool = False) -> None:
    """Летний композит, зимний композит и его теневая составляющая."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if full:
        from experiments.openlin_unet import load_site
        summer, res_s = load_site(site)["gray"], 100.0
    else:
        summer, res_s = frag_of_composite(site, FRAG_KM)
    panels = [(summer, f"летний композит, {res_s:.0f} м"),
              (comp["B04"], f"зимний композит, низкое солнце, {res_m:.0f} м"),
              (shade, f"он же за вычетом яркости поверхности, {res_m:.0f} м")]
    fig, ax = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.8))
    for a_, (img, ttl) in zip(ax, panels):
        m = np.isfinite(img)
        a_.imshow(_stretch(np.nan_to_num(img), m), cmap="gray", vmin=0, vmax=1)
        a_.set_title(ttl, fontsize=13)
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    p = Path("outputs/steps/зимний_композит"
             + ("_лист" if full else "") + ".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110)
    log(f"сохранено: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--frag", type=float, default=FRAG_KM)
    ap.add_argument("--res", type=float, default=RES_M)
    ap.add_argument("--full", action="store_true",
                    help="весь лист, а не фрагмент 22 км")
    ap.add_argument("--sun-max", type=float, default=None,
                    help="верхняя граница высоты солнца, градусы")
    ap.add_argument("--sun-min", type=float, default=None,
                    help="нижняя граница: для высокосолнечного композита")
    ap.add_argument("--months", default=None,
                    help="месяцы через запятую: состояние поверхности должно "
                         "совпадать у обеих подсветок (снег со снегом)")
    ap.add_argument("--center", default=None,
                    help="центр фрагмента «строка,столбец» в ячейках 100-м "
                         "сетки площадки; по умолчанию — участок настройки")
    ap.add_argument("--tag", default="",
                    help="суффикс имён, чтобы подсветки не затирали друг друга")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    # границы по высоте солнца и месяцам задаются на модуле поиска: он один
    # и тот же для обеих подсветок, чтобы отбор отличался только ими
    if a.sun_max is not None:
        openlin_lowsun.MAX_SUN_DEG = a.sun_max
    sun_min = a.sun_min
    months = (tuple(int(m) for m in a.months.split(",")) if a.months else None)
    if sun_min is not None or months is not None:
        base_find = openlin_lowsun.find_scenes

        def find_scenes(grid, log, **kw):          # noqa: F811
            out = base_find(grid, log, **kw)
            out = [r for r in out
                   if (sun_min is None or r[0] >= sun_min)
                   and (months is None or r[2].datetime.month in months)]
            log(f"  после отбора по высоте и месяцам: {len(out)}")
            return out
    else:
        find_scenes = openlin_lowsun.find_scenes

    if a.full:
        grid = site_grid(a.site, a.res)
        log(f"площадка {a.site}: весь лист "
            f"{grid.width * a.res / 1000:.0f} x "
            f"{grid.height * a.res / 1000:.0f} км, шаг {a.res:.0f} м "
            f"({grid.height} x {grid.width} ячеек)")
        items = find_scenes(grid, log, min_cover=0.02)
        if not items:
            raise SystemExit("подходящих сцен нет")
        items = cover_pick(items, grid, a.scenes, log)
    else:
        center = (tuple(int(v) for v in a.center.split(","))
                  if a.center else None)
        grid = scene_grid(a.site, a.frag, a.res, center)
        log(f"площадка {a.site}: фрагмент {a.frag:.0f} x {a.frag:.0f} км, "
            f"шаг {a.res:.0f} м ({grid.height} x {grid.width} ячеек)")
        items = find_scenes(grid, log)
        if not items:
            raise SystemExit("подходящих сцен нет")
    got = fetch(items, grid, log, a.scenes,
                tag=("лист" if a.full else "фраг") + a.tag)
    if not got:
        raise SystemExit("ничего не скачалось")

    log("\nсборка медианы по сценам:")
    comp, count, meta = stack_scenes(got, log)
    valid = np.isfinite(comp["B04"])
    shade = detrend(comp["B04"], valid, a.res)

    suns = np.array([m[1] for m in meta])
    azs = np.array([m[2] for m in meta])
    log(f"\nсцен в композите {len(meta)}, солнце {suns.min():.1f}-"
        f"{suns.max():.1f}°, азимут {azs.min():.0f}-{azs.max():.0f}°")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"winter_{a.site}_{int(a.res)}m"
               + ("_лист" if a.full else "") + a.tag + ".npz")
    if a.center:
        # центр пишется в сам файл, чтобы проверка потом знала, что это за кусок
        np.savez_compressed(OUT / f"_center_{a.site}{a.tag}.npz",
                            center=np.array([int(v) for v in a.center.split(",")]))
    np.savez_compressed(p, shade=shade, count=count.astype("int16"),
                        res_m=a.res, x0=grid.x0, y_top=grid.y_top,
                        epsg=grid.epsg,
                        sun=suns, azimuth=azs,
                        scenes=np.array([m[0] for m in meta]),
                        **comp)
    log(f"сохранено: {p}")
    if not a.tag:
        write_site(a.site, comp, shade, grid, log)
    if not a.tag:
        plot(comp, shade, a.site, a.res, log, full=a.full)


if __name__ == "__main__":
    main()
