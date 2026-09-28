"""Снимки при низком солнце: тот же Sentinel-2, но с длинными тенями.

ОТКУДА ПОСТАНОВКА. Геолог размечает линеаменты по космоснимку, значит контраст
в снимке есть. Наш композит его теряет по двум причинам сразу:

* он собран как МЕДИАНА сорока летних сцен, а медиана по построению убирает
  всё, что зависит от освещения, — в первую очередь тени от рельефа;
* лето на широте 71° — худшее время: солнце стоит на 40 градусах, тени
  короткие.

В каталоге по этому листу есть сцены, снятые в октябре и феврале, когда солнце
поднимается всего на 5-8 градусов над горизонтом. Тени в такой съёмке в пять-шесть
раз длиннее, и уступы читаются так же, как на теневой отмывке рельефа — только
это по-прежнему космоснимок, никаких сторонних материалов.

Здесь такие сцены скачиваются в родном разрешении на небольшой фрагмент и
складываются в набор с РАЗНЫМИ азимутами освещения: линия, перпендикулярная
солнцу, видна отлично, а параллельная ему не видна вовсе, поэтому один азимут
всегда показывает только часть структур.

Запуск из корня репозитория:
    python -m experiments.openlin_lowsun --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

CACHE = Path("datacache/s2_lowsun")
M = Path("outputs/metrics")

#: Максимальная высота солнца, градусы: выше — тени уже короткие.
MAX_SUN_DEG = 12.0

#: Максимальная облачность сцены, %.
MAX_CLOUD = 10.0

#: Какую долю фрагмента контур сцены обязан накрывать. Первый заход выбирал
#: сцены по одному bbox листа и получил покрытие 15-20 %: на широте 71 граница
#: витка идёт наискось, и «подходящая» сцена задевала фрагмент углом. Контур
#: сцены лежит в самом каталоге, поэтому проверяется до скачивания.
MIN_COVER = 0.995

#: Доля пустых ячеек в самой сцене (свойство каталога): даже накрывающая
#: фрагмент сцена может быть наполовину заполнена нулями.
MAX_NODATA = 5.0

#: Размер фрагмента, км, и рабочее разрешение, м.
FRAG_KM = 22.0
RES_M = 10.0

#: Какие каналы брать. Красный — основной для рельефного рисунка, ближний ИК
#: лучше пробивает дымку и сильнее реагирует на растительность вдоль зон
#: трещиноватости.
BANDS = ("B04", "B08")


#: Центр фрагмента по умолчанию, в ячейках 100-метровой сетки площадки: тот же
#: участок, на котором шла вся настройка.
FRAG_CENTER = (340, 410)


def scene_grid(site: str, frag_km: float, res_m: float,
               center: tuple[int, int] | None = None):
    """Сетка фрагмента в проекции площадки; центр — в ячейках её сетки."""
    from experiments.openlin_probe import Grid

    z = dict(np.load(M / f"openlin_{site}.npz"))
    big = float(z["res_m"])
    x0, ytop, epsg = float(z["x0"]), float(z["y_top"]), int(z["epsg"])
    cr, cc = center or FRAG_CENTER
    cx = x0 + cc * big
    cy = ytop - cr * big
    n = int(round(frag_km * 1000 / res_m))
    return Grid(x0=cx - frag_km * 500, y_top=cy + frag_km * 500, res_m=res_m,
                height=n, width=n, epsg=epsg)


def frag_polygon(grid):
    """Контур фрагмента в широтах-долготах — для проверки покрытия."""
    from pyproj import CRS, Transformer
    from shapely.geometry import Polygon

    tr = Transformer.from_crs(CRS.from_epsg(grid.epsg), CRS.from_epsg(4326),
                              always_xy=True)
    x1 = grid.x0 + grid.width * grid.res_m
    y1 = grid.y_top - grid.height * grid.res_m
    return Polygon([tr.transform(x, y) for x, y in
                    ((grid.x0, grid.y_top), (x1, grid.y_top),
                     (x1, y1), (grid.x0, y1))])


def find_scenes(grid, log, min_cover: float = MIN_COVER,
                max_nodata: float = MAX_NODATA) -> list:
    """Сцены с низким солнцем, малой облачностью и полным покрытием."""
    import planetary_computer as pc
    from pystac_client import Client
    from shapely.geometry import shape

    from experiments.openlin_probe import STAC_URL

    fp = frag_polygon(grid)
    cat = Client.open(STAC_URL, modifier=pc.sign_inplace)
    res = cat.search(collections=["sentinel-2-l2a"], bbox=grid.bbox_wgs84(),
                     datetime="2019-01-01/2025-12-31",
                     query={"eo:cloud_cover": {"lt": MAX_CLOUD}})
    low, out = 0, []
    for it in res.items():
        sz = it.properties.get("s2:mean_solar_zenith")
        az = it.properties.get("s2:mean_solar_azimuth")
        if sz is None:
            continue
        sun = 90.0 - float(sz)
        if not 0.0 < sun <= MAX_SUN_DEG:
            continue
        low += 1
        cover = shape(it.geometry).intersection(fp).area / fp.area
        nodata = float(it.properties.get("s2:nodata_pixel_percentage", 0.0))
        if cover < min_cover or nodata > max_nodata:
            continue
        out.append((sun, float(az) if az is not None else float("nan"), it))
    out.sort(key=lambda r: r[0])
    log(f"  сцен с солнцем ниже {MAX_SUN_DEG:.0f}°: {low}, "
        f"из них накрывают фрагмент целиком: {len(out)}")
    return out


def fetch(items, grid, log, n_max: int = 4, tag: str = "фраг") -> dict:
    """Скачать выбранные сцены на сетку фрагмента."""
    import os

    from rasterio.enums import Resampling

    from experiments.openlin_probe import _read_on_grid

    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
    CACHE.mkdir(parents=True, exist_ok=True)

    # Азимут освещения на этой широте почти не гуляет: низкое солнце бывает
    # только около полудня, и все сцены сняты с юга (177-190°). Поэтому
    # разнообразия по азимуту набрать неоткуда, и берутся просто самые низкие
    # солнца с разных дат — повтор даёт независимую проверку, что найденное
    # не случайность конкретной съёмки.
    picked, seen_day = [], set()
    for sun, az, it in items:
        d = it.datetime.date()
        if d in seen_day:
            continue
        seen_day.add(d)
        picked.append((sun, az, it))
        if len(picked) >= n_max:
            break

    out = {}
    for sun, az, it in picked:
        key = f"{it.id}"
        path = CACHE / f"{key}_{tag}_{int(grid.res_m)}m.npz"
        if path.exists():
            out[key] = dict(np.load(path)) | {"sun": sun, "az": az}
            log(f"  {it.datetime:%Y-%m-%d} солнце {sun:.1f}°, азимут "
                f"{az:.0f}° — из кэша")
            continue
        t0 = time.time()
        bands = {}
        for b in BANDS:
            if b not in it.assets:
                continue
            bands[b] = _read_on_grid(it.assets[b].href, grid,
                                     Resampling.bilinear)
        if not bands:
            continue
        np.savez_compressed(path, **bands)
        out[key] = bands | {"sun": sun, "az": az}
        log(f"  {it.datetime:%Y-%m-%d} солнце {sun:.1f}°, азимут {az:.0f}° — "
            f"скачано за {time.time() - t0:.0f} с")
    return out


def _stretch(a, valid, lo=2.0, hi=98.0):
    """Растяжка по процентилям: без неё снег выбивает всю шкалу."""
    v = a[valid]
    if v.size == 0:
        return np.zeros_like(a)
    p_lo, p_hi = np.percentile(v, [lo, hi])
    return np.clip((a - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)


def frag_of_composite(site: str, frag_km: float,
                      center: tuple[int, int] | None = None):
    """Тот же фрагмент из нашего летнего композита, в его 100 м."""
    from experiments.openlin_unet import load_site

    z = load_site(site)
    res = float(z["res_m"])
    cr, cc = center or FRAG_CENTER
    half = int(round(frag_km * 500 / res))
    return z["gray"][cr - half:cr + half, cc - half:cc + half], res


def plot(site: str, log) -> None:
    """Летний композит против сцен с низким солнцем."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(CACHE.glob(f"*_{int(RES_M)}m.npz"))
    if not files:
        raise SystemExit("сначала скачайте сцены")
    comp, res_c = frag_of_composite(site, FRAG_KM)
    n = len(files) + 1
    fig, ax = plt.subplots(1, n, figsize=(5.2 * n, 5.6))
    m = np.isfinite(comp)
    ax[0].imshow(_stretch(np.nan_to_num(comp), m), cmap="gray",
                 vmin=0, vmax=1)
    ax[0].set_title(f"летний композит, {res_c:.0f} м", fontsize=13)
    for k, f in enumerate(files, start=1):
        z = dict(np.load(f))
        b = z["B04"].astype("float32")
        good = np.isfinite(b) & (b > 0)
        cov = 100 * good.mean()
        ax[k].imshow(_stretch(np.nan_to_num(b), good), cmap="gray",
                     vmin=0, vmax=1)
        d = f.name[11:19]
        ax[k].set_title(f"{d[6:8]}.{d[4:6]}.{d[0:4]}, {RES_M:.0f} м",
                        fontsize=13)
        ax[k].set_xlabel(f"заполнено {cov:.0f} %", fontsize=11)
        log(f"  {f.name}: заполнено {cov:.0f} %, "
            f"разброс {np.nanstd(b[good]):.0f}")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    out = Path("outputs/steps/низкое_солнце.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    log(f"сохранено: {out}")



def plot_zoom(site: str, log, win_km: float = 6.0) -> None:
    """Два окна крупно: композит, низкое солнце и оно же, загрублённое до 100 м.

    Третья колонка отделяет одну причину от другой. Если новое видно и в ней —
    дело в освещении, и наш композит теряет структуру из-за медианы по сорока
    по-разному освещённым сценам. Если новое пропадает при загрублении — дело
    в разрешении, 10 м против 100 м, а не в тенях.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(CACHE.glob(f"*_{int(RES_M)}m.npz"))
    best = [f for f in files if "20251021" in f.name] or files[-1:]
    z = dict(np.load(best[0]))
    low = z["B04"].astype("float32")
    comp, res_c = frag_of_composite(site, FRAG_KM)
    k = int(round(res_c / RES_M))                  # 10 ячеек низкого солнца
    n_c = int(round(win_km * 1000 / res_c))
    # два окна: водораздельное плато и участок с густой сетью долин
    wins = [(int(0.18 * comp.shape[0]), int(0.18 * comp.shape[1])),
            (int(0.52 * comp.shape[0]), int(0.45 * comp.shape[1]))]

    fig, ax = plt.subplots(len(wins), 3, figsize=(15.5, 5.3 * len(wins)))
    for i, (r, c) in enumerate(wins):
        cw = comp[r:r + n_c, c:c + n_c]
        lw = low[r * k:(r + n_c) * k, c * k:(c + n_c) * k]
        # загрубление именно усреднением по блоку 10x10 — так же, как
        # строился бы наш композит, если бы в него попала эта сцена
        h = lw.shape[0] // k * k
        blk = lw[:h, :h].reshape(h // k, k, h // k, k).mean((1, 3))
        for j, (img, ttl) in enumerate((
                (cw, f"летний композит, {res_c:.0f} м"),
                (lw, f"21.10.2025, солнце 8.4°, {RES_M:.0f} м"),
                (blk, f"она же, загрублена до {res_c:.0f} м"))):
            m = np.isfinite(img) & (img != 0)
            ax[i, j].imshow(_stretch(np.nan_to_num(img), m), cmap="gray",
                            vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(ttl, fontsize=13)
        ax[i, 0].set_ylabel(f"окно {win_km:.0f} x {win_km:.0f} км",
                            fontsize=12)
    fig.tight_layout()
    out = Path("outputs/steps/низкое_солнце_крупно.png")
    fig.savefig(out, dpi=110)
    log(f"сохранено: {out}")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--plot", action="store_true",
                    help="только картинка по уже скачанному")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    if a.plot:
        plot(a.site, log)
        plot_zoom(a.site, log)
        return

    grid = scene_grid(a.site, FRAG_KM, RES_M)
    log(f"фрагмент {FRAG_KM:.0f} x {FRAG_KM:.0f} км, шаг {RES_M:.0f} м "
        f"({grid.height} x {grid.width} ячеек)")
    items = find_scenes(grid, log)
    if not items:
        raise SystemExit("подходящих сцен нет")
    got = fetch(items, grid, log, a.scenes)
    log(f"\nготово: {len(got)} сцен в {CACHE}")


if __name__ == "__main__":
    main()
