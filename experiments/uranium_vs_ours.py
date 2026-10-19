"""Урановый прогноз Госгеолкарты и наш прогноз — рядом, на одной площади.

ЗАЧЕМ. Цель проекта золото-урановая, а по юго-западу внешняя фактография
говорит разное: по золоту территория проверена бурением и признана
малоперспективной, по урану — наоборот, целиком накрыта Западно-Анабарской
ураноносной прогнозной зоной с ресурсами P3. Поэтому наш прогноз нужно
посмотреть рядом именно с урановым прогнозом, а не только с критериальной
картой.

ЧТО НА КАРТЕ.
* Слева — урановый прогноз листа R-48: Западно-Анабарская зона 6.1 и две
  потенциальные урановорудные зоны внутри неё (Мукунская 6.1.1 и
  Усть-Ильинская 6.1.2). Контуры взяты из векторного слоя карты
  закономерностей (`experiments/external_sw_map.py`), заливка — приближение:
  полигон собран из двух граничных линий, обрезанных рамкой листа.
* Справа — наш прогноз в текущей версии (рельеф нормирован по зонам).
* На обеих панелях — точки заверки по урану и торию (категория U/TR) и
  шлиховые пробы с золотом, найденные по листам 1:200 000.

ВХОД: outputs/12_внешние_данные/урановые_зоны_контуры.csv,
      data/processed/terrain_drift_maps.parquet, sw_shlikh_points.csv
ВЫХОД: outputs/12_внешние_данные/уран_и_наш_прогноз.png и .log
Запуск из корня: python -m experiments.uranium_vs_ours
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, config, criterial_target, mapstyle  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = ROOT / "outputs" / "12_внешние_данные"
PROJ4 = ("+proj=tmerc +lon_0=105 +lat_0=0 +x_0=500000 +y_0=0 +k_0=1 "
         "+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +ellps=krass "
         "+units=m +no_defs")
AREA = 0.10
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def main() -> None:
    shape = (ARCH_META.prf, ARCH_META.pic)
    d = pd.read_parquet(config.PROCESSED_DIR / "terrain_drift_maps.parquet")
    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
    common = d["в_области"].to_numpy(bool)
    ours = d["C: рельеф по зонам"].to_numpy(float)
    in_sheet = common & (zone_of(x, y) == "лист")
    sw = common & ~in_sheet

    z = pd.read_csv(OUT_DIR / "урановые_зоны_контуры.csv")
    zones = {}
    for tag, pair in (("6.1 Западно-Анабарская", (1, 2)), ("6.1.1/6.1.2", (1, 3))):
        g = z[z["зона"] == tag]
        A = g[g["линия"] == pair[0]][["X_м", "Y_м"]].to_numpy()
        B = g[g["линия"] == pair[1]][["X_м", "Y_м"]].to_numpy()
        poly = MplPath(np.vstack([A, B[::-1], A[:1]]))
        zones[tag] = (poly, poly.contains_points(np.column_stack([x, y])),
                      [g[g["линия"] == k][["X_м", "Y_м"]].to_numpy()
                       for k in sorted(g["линия"].unique())])

    tr = Transformer.from_crs("EPSG:4326", CRS.from_proj4(PROJ4), always_xy=True)
    sh = pd.read_csv(config.PROCESSED_DIR / "sw_shlikh_points.csv", sep=";")
    sx, sy = tr.transform(sh["долгота"].to_numpy(), sh["широта"].to_numpy())

    meta, _ = criterial_target.load_prognoz_grid()
    pts = assessment.load_verification_points(meta)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    u = pts["code"].to_numpy() == 4
    ux, uy = cx[pts["cell"].to_numpy()][u], cy[pts["cell"].to_numpy()][u]
    say("точек заверки по урану и торию на листе: %d" % u.sum())

    top = np.zeros(x.size, bool)
    top[common] = ours[common] >= np.nanquantile(ours[common], 1 - AREA)
    for tag, (_, ins, _) in zones.items():
        say("%s: доля наших верхних 10%% внутри зоны — вся область %.0f%%, "
            "юго-запад %.0f%%, лист %.0f%% (площадь зоны: вся область %.0f%%, "
            "юго-запад %.0f%%)"
            % (tag, 100 * ins[top].mean(), 100 * ins[top & sw].mean(),
               100 * ins[top & in_sheet].mean(),
               100 * (ins & common).sum() / common.sum(),
               100 * (ins & sw).sum() / sw.sum()))
    for tag, (_, ins, _) in zones.items():
        hit = ins[np.argmin(np.hypot(x[:, None] - ux[None, :],
                                     y[:, None] - uy[None, :]), axis=0)]
        say("%s: из %d точек U/TR внутри зоны %d" % (tag, u.sum(), int(hit.sum())))

    ext = (x.min() - 250, x.max() + 250, y.min() - 250, y.max() + 250)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 10.4), constrained_layout=True)

    # --- левая панель: урановый прогноз
    base = np.where(common, 0.0, np.nan)
    img = base.copy()
    img[zones["6.1 Западно-Анабарская"][1] & common] = 1.0
    img[zones["6.1.1/6.1.2"][1] & common] = 2.0
    axes[0].imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                   origin="upper", interpolation="nearest", vmin=-0.5, vmax=2.5,
                   cmap=matplotlib.colors.ListedColormap(
                       ["#eeeef2", "#cfe3c8", "#7fb069"]))
    for tag, style in (("6.1 Западно-Анабарская",
                        dict(color="#1a7f3c", lw=2.0, ls=(0, (9, 3)))),
                       ("6.1.1/6.1.2",
                        dict(color="#4b0082", lw=1.5, ls=(0, (4, 2))))):
        for j, P in enumerate(zones[tag][2]):
            axes[0].plot(P[:, 0], P[:, 1], label=tag if j == 0 else None, **style)
    axes[0].set_title("Урановый прогноз Госгеолкарты листа R-48\n"
                      "(светлое — Западно-Анабарская зона 6.1, тёмное — "
                      "потенциальные рудные 6.1.1 и 6.1.2)", fontsize=10.5)

    # --- правая панель: наш прогноз
    cuts = np.nanpercentile(ours[common], mapstyle.CLASS_EDGES[1:-1])
    cls = np.where(common, np.digitize(ours, cuts).astype(float), np.nan)
    axes[1].imshow(np.ma.masked_invalid(cls.reshape(shape)), extent=ext,
                   origin="upper", interpolation="nearest", vmin=-0.5, vmax=4.5,
                   cmap=matplotlib.colors.ListedColormap(mapstyle.CLASS_COLORS))
    for tag, style in (("6.1 Западно-Анабарская",
                        dict(color="#1a7f3c", lw=1.6, ls=(0, (9, 3)))),
                       ("6.1.1/6.1.2",
                        dict(color="#4b0082", lw=1.2, ls=(0, (4, 2))))):
        for P in zones[tag][2]:
            axes[1].plot(P[:, 0], P[:, 1], **style)
    axes[1].set_title("Наш прогноз, текущая версия\n(классы по доле площади "
                      "всей области; контуры зон для сравнения)", fontsize=10.5)

    sheet_x, sheet_y = x[in_sheet], y[in_sheet]
    for axi in axes:
        axi.scatter(ux, uy, s=70, marker="^", facecolors="#ffffff",
                    edgecolors="#101010", linewidths=1.4, zorder=6,
                    label="заверка по урану и торию (4)")
        axi.scatter(sx, sy, s=42, marker="*", c="#101010", zorder=6,
                    label="шлиховые пробы с золотом (12)")
        axi.add_patch(matplotlib.patches.Rectangle(
            (sheet_x.min() - 250, sheet_y.min() - 250),
            sheet_x.max() - sheet_x.min() + 500,
            sheet_y.max() - sheet_y.min() + 500, fill=False, ec="#101010",
            lw=1.2, ls="--"))
        axi.set_xlim(ext[0], ext[1])
        axi.set_ylim(ext[2], ext[3])
        axi.set_aspect("equal")
        axi.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    handles = [matplotlib.patches.Patch(facecolor=c, edgecolor="#777777", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    axes[1].legend(handles=handles, loc="lower left", fontsize=8,
                   framealpha=0.95, title="классы прогноза")
    fig.suptitle("Уран: государственный прогноз и наш прогноз на одной площади",
                 fontsize=13)
    path = OUT_DIR / "уран_и_наш_прогноз.png"
    mapstyle.save_fig(fig, path, dpi=180)
    say("карта: %s" % path)
    (OUT_DIR / "уран_и_наш_прогноз.log").write_text("\n".join(LOG),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
