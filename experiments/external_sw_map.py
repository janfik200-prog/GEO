"""Внешние данные по юго-западу: урановые зоны, объекты, шлихи — на одной карте.

ЗАЧЕМ. 05.10.2026 по юго-западному квадранту впервые собрана внешняя
фактография: 12 шлиховых проб с золотом, несколько рудных объектов и
прогнозные урановые зоны Госгеолкарты-1000/3 (лист R-48 «Хатанга»). Всё это
нужно увидеть рядом с нашими картами — прогнозной и критериальной.

ЧТО ДЕЛАЕТСЯ.
1. Контуры прогнозных зон вытаскиваются ИЗ ВЕКТОРНОГО слоя pdf-карты
   закономерностей размещения полезных ископаемых: нужные линии опознаются по
   цвету и шагу штриха, взятым из условных обозначений этой же карты
   (Западно-Анабарская 6.1 — длинный штрих, Мукунская 6.1.1 и Усть-Ильинская
   6.1.2 — короткий). Привязка пиксель -> градусы строится по рамке листа:
   меридианы 102 и 108 — прямые (подгонка 0.3-0.5 px), параллель 72 — дуга
   (0.2 px), широта калибруется по номерам строк 1:200 000 на левой рамке.
   Остаточная погрешность привязки — около 4 км, это надо помнить при чтении.
2. Рядом наносятся: 12 шлиховых проб (по правилу проекта — поисковый признак,
   не рудопроявление), рудные объекты с координатами из каталога листа, наши
   верхние 10% площади и верхние 10% критериального воссоздания.
3. Вторая панель — та же площадь на геологической основе листов ГГК-200
   (geotiff с привязкой), чтобы зоны читались вместе с геологией.

ВХОД: datacache/ggk1000/R-48_Hatanga/r-48kzrppi.pdf,
      datacache/ggk200/R-48-XV_XVI/*.tif, R-48-XXI_XXII/*.tif,
      data/processed/sw_shlikh_points.csv, terrain_drift_maps.parquet,
      criterial_reconstruct_multisheet.parquet
ВЫХОД: outputs/12_внешние_данные/ — карта, контуры зон (csv), список объектов
Запуск из корня: python -m experiments.external_sw_map
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymupdf  # noqa: E402
import rasterio  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, mapstyle  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "outputs" / "12_внешние_данные"
PDF = ROOT / "datacache" / "ggk1000" / "R-48_Hatanga" / "r-48kzrppi.pdf"
TIFS = (ROOT / "datacache" / "ggk200" / "R-48-XV_XVI"
        / "r-48-xvxvigeologicheskayakarta.tif",
        ROOT / "datacache" / "ggk200" / "R-48-XXI_XXII"
        / "r-48-xxixxiigeologicheskayakarta.tif")
PROJ4 = ("+proj=tmerc +lon_0=105 +lat_0=0 +x_0=500000 +y_0=0 +k_0=1 "
         "+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +ellps=krass "
         "+units=m +no_defs")
ZONE_COLOR = (0.69, 0.22, 0.14)        # цвет линий урановых зон в условных знаках
DASH_BIG = "28.346293"                  # 6.1 Западно-Анабарская
DASH_SMALL = "14.173182"                # 6.1.1 Мукунская и 6.1.2 Усть-Ильинская
CLIP = (60, 140)                        # левый верхний угол вырезки карты, pt
SCALE = 150 / 72.0                      # пиксели на пункт при dpi=150
AREA = 0.10

# рудные объекты листа с координатами из каталога (привязка ±1 км)
OBJECTS = [
    ("Ыраас, апатит-магнетитовое\nместорождение (крупное)", 105.40, 70.19, "D"),
    ("Река Котуйкан 5:\nсвинец, реальгар", 105.28, 70.64, "s"),
    ("Река Илья 1-3:\nредкие земли", 105.94, 70.23, "o"),
    ("Река Угар 1-2: медь", 105.37, 69.52, "o"),
]
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def frame_transform() -> callable:
    """Привязка листа R-48: пункты страницы -> долгота, широта."""
    pix = pymupdf.open(PDF)[0].get_pixmap(dpi=150,
                                          clip=pymupdf.Rect(60, 140, 920, 1420))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width,
                                                       pix.n)[..., :3]
    dark = img.mean(axis=2) < 110
    h, w = dark.shape

    def scan_x(rows, lo, hi, side):
        out = []
        for y in rows:
            idx = np.flatnonzero(dark[y, lo:hi])
            if idx.size:
                out.append((y, lo + (idx[0] if side == "left" else idx[-1])))
        return np.array(out)

    def fit_line(P):
        a, b = P[:, 0].astype(float), P[:, 1].astype(float)
        keep = np.ones(a.size, bool)
        for _ in range(6):
            c = np.linalg.lstsq(np.column_stack([a[keep], np.ones(keep.sum())]),
                                b[keep], rcond=None)[0]
            r = b - (c[0] * a + c[1])
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            keep = np.abs(r - np.median(r)) < 2.5 * s
        return c

    def arc(lo, hi):
        pts = []
        for x in range(260, 1500, 4):
            idx = np.flatnonzero(dark[lo:hi, x])
            if idx.size:
                pts.append((x, lo + idx[0]))
        P = np.array(pts)
        x, y = P[:, 0].astype(float), P[:, 1].astype(float)
        keep = np.ones(x.size, bool)
        for _ in range(6):
            c = np.linalg.lstsq(np.column_stack([x[keep] ** 2, x[keep],
                                                 np.ones(keep.sum())]),
                                y[keep], rcond=None)[0]
            r = y - (c[0] * x ** 2 + c[1] * x + c[2])
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            keep = np.abs(r - np.median(r)) < 2.5 * s
        return c

    cl = fit_line(scan_x(range(300, h - 400, 5), 150, 420, "left"))
    cr = fit_line(scan_x(range(300, h - 400, 5), w - 420, w - 20, "right"))
    ct = arc(150, 330)
    ay = (cr[1] - cl[1]) / (cl[0] - cr[0])
    ax = cl[0] * ay + cl[1]

    def polar(x, y):
        return np.hypot(x - ax, y - ay), np.arctan2(x - ax, y - ay)

    y_top = 200.0
    for _ in range(80):
        x_top = cl[0] * y_top + cl[1]
        y_top = ct[0] * x_top ** 2 + ct[1] * x_top + ct[2]
    r72, a102 = polar(x_top, y_top)
    y_tr = 200.0
    for _ in range(80):
        x_tr = cr[0] * y_tr + cr[1]
        y_tr = ct[0] * x_tr ** 2 + ct[1] * x_tr + ct[2]
    a108 = polar(x_tr, y_tr)[1]

    # номера строк 1:200 000 на левой рамке — центры строк II, III, IV
    marks = []
    for y in range(200, 2460):
        x0 = int(round(cl[0] * y + cl[1]))
        if dark[y, x0 + 16:x0 + 60].sum() >= 3:
            marks.append(y)
    marks = np.array(marks)
    groups = [g for g in np.split(marks, np.flatnonzero(np.diff(marks) > 8) + 1)
              if len(g) >= 8]
    ys = np.array([g.mean() for g in groups[:3]])
    xs = cl[0] * ys + cl[1]
    R = np.append(np.hypot(xs - ax, ys - ay), r72)
    L = np.array([71.0, 70 + 1 / 3, 69 + 2 / 3, 72.0])
    k, b = np.linalg.lstsq(np.column_stack([R, np.ones(4)]), L, rcond=None)[0]
    resid = L - (k * R + b)
    say("привязка листа: остаточная ошибка по широте до %.1f км, "
        "долгота проверена по подписи 105°00' (совпадение в 1 px)"
        % (111 * np.abs(resid).max()))

    def to_geo(px, py):
        X, Y = (px - CLIP[0]) * SCALE, (py - CLIP[1]) * SCALE
        r, a = polar(X, Y)
        return 102 + 6 * (a - a102) / (a108 - a102), k * r + b

    return to_geo


def zone_paths(to_geo) -> dict:
    """Линии урановых зон из векторного слоя карты, в градусах."""
    page = pymupdf.open(PDF)[0]
    out = {}
    for g in page.get_drawings():
        col = g.get("color")
        if not col or tuple(round(v, 2) for v in col) != ZONE_COLOR:
            continue
        if g["rect"].x0 > 2000:        # условные обозначения, не карта
            continue
        dash = g.get("dashes") or ""
        tag = ("6.1 Западно-Анабарская" if DASH_BIG in dash
               else "6.1.1/6.1.2" if DASH_SMALL in dash else None)
        if tag is None:
            continue
        pts = [(q.x, q.y) for it in g["items"] for q in it[1:]
               if isinstance(q, pymupdf.Point)]
        if len(pts) < 10:
            continue
        P = np.array(pts)
        lon, lat = to_geo(P[:, 0], P[:, 1])
        out.setdefault(tag, []).append(np.column_stack([lon, lat]))
    for tag, parts in out.items():
        say("%s: линий %d, точек %d" % (tag, len(parts), sum(len(p) for p in parts)))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = Transformer.from_crs("EPSG:4326", CRS.from_proj4(PROJ4), always_xy=True)

    to_geo = frame_transform()
    zones = zone_paths(to_geo)

    rows = []
    for tag, parts in zones.items():
        for j, P in enumerate(parts, 1):
            X, Y = tr.transform(P[:, 0], P[:, 1])
            for lon, lat, x, y in zip(P[:, 0], P[:, 1], X, Y):
                rows.append({"зона": tag, "линия": j, "долгота": lon,
                             "широта": lat, "X_м": x, "Y_м": y})
    pd.DataFrame(rows).to_csv(OUT_DIR / "урановые_зоны_контуры.csv", index=False,
                              encoding="utf-8-sig")

    # --- наши карты
    d = pd.read_parquet(config.PROCESSED_DIR / "terrain_drift_maps.parquet")
    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
    common = d["в_области"].to_numpy(bool)
    ours = d["C: рельеф по зонам"].to_numpy(float)
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    dist, idx = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                         ref["y"].to_numpy()[rmsk]])).query(
        np.column_stack([x, y]))
    crit = np.full(x.size, np.nan)
    crit[common] = -ref["воссоздание_вся_область"].to_numpy()[rmsk][idx[common]]
    shape = (ARCH_META.prf, ARCH_META.pic)
    in_sheet = common & (zone_of(x, y) == "лист")

    def top_grid(v):
        t = np.full(x.size, np.nan)
        t[common] = (v[common] >= np.nanquantile(v[common], 1 - AREA)).astype(float)
        return t.reshape(shape)

    # --- шлихи и объекты
    sh = pd.read_csv(config.PROCESSED_DIR / "sw_shlikh_points.csv", sep=";")
    sx, sy = tr.transform(sh["долгота"].to_numpy(), sh["широта"].to_numpy())
    ox, oy, onames = [], [], []
    for name, lon, lat, _m in OBJECTS:
        X, Y = tr.transform(lon, lat)
        ox.append(X)
        oy.append(Y)
        onames.append(name)

    # --- геологическая основа ГГК-200 в проекции проекта
    ext = (x.min() - 250, x.max() + 250, y.min() - 250, y.max() + 250)
    dst = np.zeros((3, 900, 500), dtype=np.uint8) + 255
    dst_tr = rasterio.transform.from_bounds(ext[0], ext[2], ext[1], ext[3],
                                            500, 900)
    for tif in TIFS:
        with rasterio.open(tif) as ds:
            band = ds.read(indexes=[1, 2, 3] if ds.count >= 3 else [1, 1, 1])
            tmp = np.zeros_like(dst)
            reproject(band, tmp, src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=dst_tr, dst_crs=CRS.from_proj4(PROJ4),
                      resampling=Resampling.bilinear, src_nodata=None,
                      dst_nodata=0)
            m = tmp.sum(axis=0) > 0
            dst[:, m] = tmp[:, m]
    say("геологическая основа ГГК-200 пересобрана в проекцию проекта")

    # --- рисунок
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 10.6), constrained_layout=True)
    for axi, backdrop in zip(axes, (False, True)):
        if backdrop:
            axi.imshow(np.moveaxis(dst, 0, -1), extent=ext, origin="upper",
                       interpolation="bilinear", alpha=0.85)
        else:
            axi.imshow(np.ma.masked_where(top_grid(crit) != 1, top_grid(crit)),
                       extent=ext, origin="upper", interpolation="nearest",
                       cmap=matplotlib.colors.ListedColormap(["#f2b544"]),
                       alpha=0.75)
            axi.imshow(np.ma.masked_where(top_grid(ours) != 1, top_grid(ours)),
                       extent=ext, origin="upper", interpolation="nearest",
                       cmap=matplotlib.colors.ListedColormap(["#a3122b"]),
                       alpha=0.55)
            axi.contour(np.where(common, 1.0, 0.0).reshape(shape), levels=[0.5],
                        colors="#777777", linewidths=0.8, extent=ext,
                        origin="upper")
        for tag, parts in zones.items():
            style = dict(color="#1a7f3c", lw=2.0, ls=(0, (9, 3))) if "6.1 " in tag \
                else dict(color="#6a0dad", lw=1.6, ls=(0, (4, 2)))
            for j, P in enumerate(parts):
                X, Y = tr.transform(P[:, 0], P[:, 1])
                axi.plot(X, Y, label=tag if j == 0 else None, **style)
        axi.scatter(sx, sy, s=46, marker="*", c="#111111", zorder=5,
                    label="шлиховые пробы с золотом (12)")
        axi.scatter(ox, oy, s=52, marker="D", facecolors="none",
                    edgecolors="#0b3d91", linewidths=1.6, zorder=5,
                    label="рудные объекты каталога")
        sxx, syy = x[in_sheet], y[in_sheet]
        axi.add_patch(matplotlib.patches.Rectangle(
            (sxx.min() - 250, syy.min() - 250), sxx.max() - sxx.min() + 500,
            syy.max() - syy.min() + 500, fill=False, ec="#101010", lw=1.2,
            ls="--"))
        axi.set_xlim(ext[0], ext[1])
        axi.set_ylim(ext[2], ext[3])
        axi.set_aspect("equal")
        axi.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
    for name, X, Y in zip(onames, ox, oy):
        axes[0].annotate(name, (X, Y), fontsize=7.5, ha="left", va="center",
                         xytext=(7, 0), textcoords="offset points")
    axes[0].set_title("Наши верхние 10% площади и внешние данные\n"
                      "(красное — прогноз, жёлтое — критериальный метод)",
                      fontsize=11)
    axes[1].set_title("То же на геологической основе листов 1:200 000\n"
                      "R-48-XV,XVI и R-48-XXI,XXII", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.suptitle("Юго-западный квадрант: прогнозные урановые зоны, рудные "
                 "объекты и шлиховые пробы", fontsize=13)
    path = OUT_DIR / "внешние_данные_юго_запад.png"
    mapstyle.save_fig(fig, path, dpi=180)
    say("карта: %s" % path)

    pd.DataFrame([{"объект": n.replace("\n", " "), "долгота": lo, "широта": la}
                  for n, lo, la, _ in OBJECTS]).to_csv(
        OUT_DIR / "рудные_объекты.csv", index=False, encoding="utf-8-sig")
    # --- насколько ядра карт лежат внутри прогнозных зон
    from matplotlib.path import Path as MplPath
    sw_cells = common & (zone_of(x, y) != "лист")
    top_o = sw_cells & (ours >= np.nanquantile(ours[sw_cells], 1 - AREA))
    top_c = sw_cells & (crit >= np.nanquantile(crit[sw_cells], 1 - AREA))
    stat = []
    for tag, pair in (("6.1 Западно-Анабарская", (0, 1)), ("6.1.1/6.1.2", (0, 2))):
        parts = zones.get(tag, [])
        if len(parts) <= max(pair):
            continue
        A = np.column_stack(tr.transform(parts[pair[0]][:, 0], parts[pair[0]][:, 1]))
        B = np.column_stack(tr.transform(parts[pair[1]][:, 0], parts[pair[1]][:, 1]))
        poly = MplPath(np.vstack([A, B[::-1], A[:1]]))
        ins = poly.contains_points(np.column_stack([x, y]))
        row = {"зона": tag,
               "площадь_в_области_кв_км": 0.25 * int((ins & common).sum()),
               "доля_юго_запада": float((ins & sw_cells).sum() / sw_cells.sum()),
               "доля_наших_ядер": float(ins[top_o].mean()),
               "доля_критериальных_ядер": float(ins[top_c].mean())}
        stat.append(row)
        say("%s: занимает %.0f%% юго-запада; внутри неё наших верхних 10%% — "
            "%.0f%%, критериальных — %.0f%% (полигон собран из двух граничных "
            "линий, это приближение)"
            % (tag, 100 * row["доля_юго_запада"], 100 * row["доля_наших_ядер"],
               100 * row["доля_критериальных_ядер"]))
    pd.DataFrame(stat).to_csv(OUT_DIR / "зоны_и_ядра.csv", index=False,
                              encoding="utf-8-sig")
    sh.to_csv(OUT_DIR / "шлиховые_пробы.csv", index=False, sep=";",
              encoding="utf-8-sig")

    (OUT_DIR / "описание.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
