# -*- coding: utf-8 -*-
"""Рисунки раздела «Прогнозирование» (модуль report_sections/forecast.py).

ЗАЧЕМ. Карты раздела строятся заново под холст 10" и крупно: готовые PNG из
`outputs/` сделаны под 13.3", подписи на них не читаются. Размер рисунка равен
размеру на слайде, кегль 8.5-10 пт.

ЧТО СТРОИТСЯ (всё в outputs/report_figs/, префикс fz_):
  * fz_transfer.png   - карта переноса: вариант «как есть» и принятый вариант
    «рельеф по зонам» (data/processed/terrain_drift_maps.parquet от 04.10.2026,
    скрипт experiments/terrain_drift_fix.py), стиль src/mapstyle.py, рамка
    основного листа, точки заверки (73, из них 19 несмещённых);
  * fz_wide.png       - принятая карта переноса и критериальная формула
    (воссоздание на всей области, criterial_reconstruct_multisheet.parquet);
  * fz_analog.png     - поиск по аналогам: две версии сходства
    (data/processed/analog_search_v2.parquet);
  * fz_analog_nov.png - сходство по рельефу с верхними 10% и новизна ячеек;
  * fz_uranium.png    - урановый прогноз Госгеолкарты и наша карта, 12 шлихов
    (как experiments/uranium_vs_ours.py, но крупнее);
  * fz_hamissana.png  - три панели из готового lit6b_hamissana.png (карты
    метода статьи на признаках без входов формулы) со своими подписями;
  * fz_pair.png, fz_sheet.png - обрезки готового ai_vnigni_map_pair.png;
  * fz_numbers.png    - столбики: итоговая карта, слепая проверка, потолок
    (full_sheet_map.csv, ceiling_decomposition.csv), пары «до / после поправки
    на наклон».
Запуск из корня: python -X utf8 -m experiments.make_report_figures_forecast
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, config, criterial_target, mapstyle  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402

OUT = ROOT / "outputs"
FIG = OUT / "report_figs"
MET = OUT / "metrics"
SHAPE = (ARCH_META.prf, ARCH_META.pic)
PROJ4 = ("+proj=tmerc +lon_0=105 +lat_0=0 +x_0=500000 +y_0=0 +k_0=1 "
         "+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +ellps=krass "
         "+units=m +no_defs")
TEAL = "#14B8A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    mapstyle.save_fig(fig, FIG / name, dpi=220)
    plt.close(fig)
    print("OK", name)


# --- общая подложка -----------------------------------------------------------
class Area:
    def __init__(self):
        d = pd.read_parquet(config.PROCESSED_DIR / "terrain_drift_maps.parquet")
        self.d = d
        self.x = d["x"].to_numpy(float)
        self.y = d["y"].to_numpy(float)
        self.common = d["в_области"].to_numpy(bool)
        self.sheet = self.common & (zone_of(self.x, self.y) == "лист")
        self.ext = (self.x.min() - 250, self.x.max() + 250,
                    self.y.min() - 250, self.y.max() + 250)
        meta_s, _ = criterial_target.load_prognoz_grid()
        pts = assessment.load_verification_points(meta_s)
        sx, sy = meta_s.cell_centers()
        sx, sy = sx.ravel(), sy.ravel()
        self.px = sx[pts["cell"].to_numpy()]
        self.py = sy[pts["cell"].to_numpy()]
        self.unb = pts["unbiased_strict"].to_numpy(bool)

    def frame(self, ax, ylabel=True):
        sx, sy = self.x[self.sheet], self.y[self.sheet]
        ax.add_patch(Rectangle((sx.min() - 250, sy.min() - 250),
                               sx.max() - sx.min() + 500,
                               sy.max() - sy.min() + 500, fill=False,
                               ec="#1E293B", lw=0.9, ls="--"))
        ax.set_xlim(self.ext[0], self.ext[1])
        ax.set_ylim(self.ext[2], self.ext[3])
        ax.set_aspect("equal")
        ax.set_xticks([520000, 590000])
        ax.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        ax.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        ax.set_xlabel("X, км")
        if ylabel:
            ax.set_ylabel("Y, км")
        else:
            ax.set_yticklabels([])

    def classes(self, ax, score, mask=None, top=True):
        m = self.common if mask is None else mask
        pct = mapstyle.percentile(score, m)
        cls = np.where(m, np.digitize(pct, mapstyle.CLASS_EDGES[1:-1]), np.nan)
        ax.imshow(np.ma.masked_invalid(cls.reshape(SHAPE).astype(float)),
                  extent=self.ext, origin="upper", interpolation="nearest",
                  vmin=-0.5, vmax=4.5, cmap=mapstyle.class_cmap())
        if top:
            hi = np.nan_to_num(pct, nan=0) >= 90
            mapstyle.outline(ax, hi, SHAPE, self.ext, flip=True, lw=0.8)
        return pct

    def points(self, ax, size=13):
        ax.scatter(self.px[~self.unb], self.py[~self.unb], s=size, marker="o",
                   facecolors="none", edgecolors="#101010", lw=0.7, zorder=5)
        ax.scatter(self.px[self.unb], self.py[self.unb], s=size * 1.9,
                   marker="^", c=TEAL, edgecolors="#0F3D3A", lw=0.6, zorder=6)


def class_handles(labels=None):
    labels = labels or mapstyle.CLASS_LABELS
    return [Patch(fc=c, ec="#555", lw=0.4, label=l)
            for c, l in zip(mapstyle.CLASS_COLORS, labels)]


def pts_handles():
    return [Line2D([], [], color="#101010", lw=1, label="граница верх. 10%"),
            Line2D([], [], marker="o", mfc="none", mec="#101010", ls="",
                   label="точки опроб., 73"),
            Line2D([], [], marker="^", mfc=TEAL, mec="#0F3D3A", ls="",
                   label="из них независимые\nот формулы, 19")]


PW, PH = 1.25, 2.45          # панель карты, дюймы (область вытянута на север)


def panels(A, n, width, legend_handles, draw, titles, name, height=3.4,
           legend_title=None):
    """n панелей карты в ряд и колонка условных знаков справа."""
    fig = plt.figure(figsize=(width, height))
    x0, bot = 0.66, 0.45
    for i in range(n):
        ax = fig.add_axes([(x0 + i * (PW + 0.07)) / width, bot / height,
                           PW / width, PH / height])
        draw(i, ax)
        A.frame(ax, ylabel=(i == 0))
        ax.set_title(titles[i], fontsize=8.5, fontweight="bold", pad=3)
    fig.legend(handles=legend_handles, loc="center left", fontsize=8,
               frameon=False, bbox_to_anchor=((x0 + n * (PW + 0.07) + 0.05) / width,
                                              0.52),
               handlelength=1.3, labelspacing=0.55, title=legend_title,
               title_fontsize=8)
    save(fig, name)


def formula_score(A):
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    from scipy.spatial import cKDTree
    rm = ref["в_маске"].to_numpy(bool)
    _, idx = cKDTree(np.column_stack([ref["x"].to_numpy()[rm],
                                      ref["y"].to_numpy()[rm]])).query(
        np.column_stack([A.x, A.y]))
    crit = np.full(A.x.size, np.nan)
    crit[A.common] = -ref["воссоздание_вся_область"].to_numpy()[rm][idx[A.common]]
    return crit


def fig_transfer(A):
    crit = formula_score(A)
    sc = [A.d["A: как есть"].to_numpy(float),
          A.d["C: рельеф по зонам"].to_numpy(float), crit]

    def draw(i, ax):
        A.classes(ax, sc[i])
        A.points(ax, size=9)

    panels(A, 3, 6.3, class_handles() + pts_handles(), draw,
           ["Как есть", "Рельеф по\nзонам (принята)", "Критериальная\nформула"],
           "fz_transfer.png")


def fig_analog(A):
    d = pd.read_parquet(config.PROCESSED_DIR / "analog_search_v2.parquet")
    v1 = d["сходство: все признаки (обнажённые ячейки)"].to_numpy(float)
    v2 = d["сходство: рельеф и линеаменты (вся площадь)"].to_numpy(float)
    nov = d["новизна"].to_numpy(float)
    ok1 = A.common & np.isfinite(v1)
    ok2 = A.common & np.isfinite(v2)

    def draw(i, ax):
        if i == 0:
            A.classes(ax, v1, ok1, top=False)
        elif i == 1:
            A.classes(ax, v2, ok2, top=False)
        else:
            img = np.where(ok2, nov, np.nan).reshape(SHAPE)
            ax.imshow(np.ma.masked_invalid(img), extent=A.ext, origin="upper",
                      interpolation="nearest", cmap="viridis_r", vmin=0, vmax=4)
        if i < 2:
            A.points(ax, size=8)

    labs = ("нижние 50%", "50-75%", "75-90%", "90-97%", "верхние 3%")
    h = class_handles(labs) + [
        Line2D([], [], marker="^", mfc=TEAL, mec="#0F3D3A", ls="",
               label="независимые от\nформулы точки, 19")]
    panels(A, 3, 6.3, h, draw,
           ["Все признаки,\nобнажённые ячейки", "Рельеф и\nлинеаменты",
            "Новизна ячейки\n(жёлтый: как на листе)"], "fz_analog.png",
           legend_title="Классы сходства,\nдоля площади")

    # верхние 10% двух версий на одной карте
    fig = plt.figure(figsize=(3.6, 3.4))
    ax = fig.add_axes([0.66 / 3.6, 0.45 / 3.4, 1.25 / 3.6, 2.45 / 3.4])
    base = np.where(A.common, 0.0, np.nan)
    ax.imshow(np.ma.masked_invalid(base.reshape(SHAPE)), extent=A.ext,
              origin="upper", cmap=matplotlib.colors.ListedColormap(["#EEEEF2"]))
    for v, ok, col in ((v1, ok1, "#2563EB"), (v2, ok2, "#DC2626")):
        pct = mapstyle.percentile(v, ok)
        hi = np.nan_to_num(pct, nan=0) >= 90
        im = np.where(hi, 1.0, np.nan).reshape(SHAPE)
        ax.imshow(np.ma.masked_invalid(im), extent=A.ext, origin="upper",
                  interpolation="nearest", alpha=0.8,
                  cmap=matplotlib.colors.ListedColormap([col]))
    A.points(ax, size=8)
    A.frame(ax)
    ax.set_title("Верхние 10% сходства", fontsize=9, fontweight="bold", pad=3)
    fig.legend(handles=[Patch(fc="#DC2626", alpha=0.8, label="рельеф и\nлинеаменты"),
                        Patch(fc="#2563EB", alpha=0.8, label="все признаки"),
                        Line2D([], [], marker="^", mfc=TEAL, mec="#0F3D3A",
                               ls="", label="точки заверки")],
               loc="center left", fontsize=8, frameon=False,
               bbox_to_anchor=(1.95 / 3.6, 0.5), handlelength=1.2)
    save(fig, "fz_analog_nov.png")


def fig_uranium(A):
    ext_dir = OUT / "12_внешние_данные"
    z = pd.read_csv(ext_dir / "урановые_зоны_контуры.csv")
    zones = {}
    for tag, pair in (("6.1 Западно-Анабарская", (1, 2)),
                      ("6.1.1/6.1.2", (1, 3))):
        g = z[z["зона"] == tag]
        a = g[g["линия"] == pair[0]][["X_м", "Y_м"]].to_numpy()
        b = g[g["линия"] == pair[1]][["X_м", "Y_м"]].to_numpy()
        poly = MplPath(np.vstack([a, b[::-1], a[:1]]))
        zones[tag] = (poly.contains_points(np.column_stack([A.x, A.y])),
                      [g[g["линия"] == k][["X_м", "Y_м"]].to_numpy()
                       for k in sorted(g["линия"].unique())])
    tr = Transformer.from_crs("EPSG:4326", CRS.from_proj4(PROJ4), always_xy=True)
    sh = pd.read_csv(config.PROCESSED_DIR / "sw_shlikh_points.csv", sep=";")
    sx, sy = tr.transform(sh["долгота"].to_numpy(), sh["широта"].to_numpy())
    from matplotlib.colors import ListedColormap
    img = np.where(A.common, 0.0, np.nan)
    img[zones["6.1 Западно-Анабарская"][0] & A.common] = 1.0
    img[zones["6.1.1/6.1.2"][0] & A.common] = 2.0

    def draw(i, ax):
        if i == 0:
            ax.imshow(np.ma.masked_invalid(img.reshape(SHAPE)), extent=A.ext,
                      origin="upper", interpolation="nearest", vmin=-0.5,
                      vmax=2.5,
                      cmap=ListedColormap(["#EEEEF2", "#CFE3C8", "#7FB069"]))
        else:
            A.classes(ax, A.d["C: рельеф по зонам"].to_numpy(float), top=False)
            for p in zones["6.1 Западно-Анабарская"][1]:
                ax.plot(p[:, 0], p[:, 1], color="#1A7F3C", lw=1.0,
                        ls=(0, (6, 2)))
        ax.scatter(sx, sy, s=34, marker="*", c="#101010", zorder=6)

    labs = ("нижние 50%", "50-75%", "75-90%", "90-97%", "верхние 3%")
    h = [Patch(fc="#7FB069", label="зона 6.1.1/6.1.2"),
         Patch(fc="#CFE3C8", label="зона 6.1\nЗападно-Анабарская"),
         Line2D([], [], color="#1A7F3C", ls="--",
                label="контур зоны 6.1\nна нашей карте"),
         Line2D([], [], marker="*", color="#101010", ls="",
                label="шлиховые пробы, 12")] + class_handles(labs)
    panels(A, 2, 5.3, h, draw, ["Урановый прогноз\nГосгеолкарты",
                                "Наш прогноз,\nклассы по площади"],
           "fz_uranium.png")


def fig_formula(A):
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    from scipy.spatial import cKDTree
    from matplotlib.colors import ListedColormap
    rm = ref["в_маске"].to_numpy(bool)
    _, idx = cKDTree(np.column_stack([ref["x"].to_numpy()[rm],
                                      ref["y"].to_numpy()[rm]])).query(
        np.column_stack([A.x, A.y]))
    ours = np.full(A.x.size, np.nan)
    theirs = np.full(A.x.size, np.nan)
    ours[A.common] = -ref["воссоздание_вся_область"].to_numpy()[rm][idx[A.common]]
    theirs[A.common] = -ref["by_crt"].to_numpy()[rm][idx[A.common]]
    top_o = A.common & (ours >= np.nanquantile(ours[A.common], 0.9))
    top_t = A.common & (theirs >= np.nanquantile(theirs[A.common], 0.9))
    both = np.where(A.common, top_o + 2.0 * top_t, np.nan)
    cm = ListedColormap(["#F3F3F3", "#4A6FB5", "#F2B544", "#A3122B"])

    def draw(i, ax):
        if i < 2:
            A.classes(ax, (ours, theirs)[i], top=False)
        else:
            ax.imshow(np.ma.masked_invalid(both.reshape(SHAPE)), extent=A.ext,
                      origin="upper", interpolation="nearest", vmin=-0.5,
                      vmax=3.5, cmap=cm)

    h = class_handles(("нижние 50%", "50-75%", "75-90%", "90-97%",
                       "верхние 3%")) + [
        Patch(fc="#4A6FB5", label="только наше"),
        Patch(fc="#F2B544", label="только организации"),
        Patch(fc="#A3122B", label="оба расчёта")]
    panels(A, 3, 6.4, h, draw, ["Наше\nвоссоздание", "Расчёт\nорганизации",
                                "Верхние 10%:\nсогласие"], "fz_formula.png")


def fig_cores():
    """Обрезка готового cores_anatomy.png (experiments/cores_anatomy.py)."""
    im = Image.open(OUT / "02_критериальный_метод" / "cores_anatomy.png").convert("RGB")
    f = 1.30
    boxes = (("Критериальная карта", (160, 60, 830, 1385)),
             ("Наш прогноз", (1165, 60, 1835, 1385)))
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 3.4))
    for ax, (t, b) in zip(axes, boxes):
        ax.imshow(np.asarray(im.crop(tuple(int(v * f) for v in b))))
        ax.axis("off")
        ax.set_title(t, fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04)
    save(fig, "fz_cores.png")


def fig_hamissana():
    im = Image.open(OUT / "10_методы_из_литературы" / "lit6b_hamissana.png").convert("RGB")
    boxes = {"crit": (24, 128, 537, 660), "l8": (564, 128, 1077, 660),
             "s2": (1104, 128, 1617, 660)}
    titles = {"crit": "Критериальный\nпрогноз",
              "l8": "Landsat-8: AUC по статье\n0.84, по полосам 0.71",
              "s2": "Sentinel-2: AUC по статье\n0.88, по полосам 0.68"}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.95))
    for ax, k in zip(axes, ("crit", "l8", "s2")):
        ax.imshow(np.asarray(im.crop(boxes[k])))
        ax.axis("off")
        ax.set_title(titles[k], fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.04)
    save(fig, "fz_hamissana.png")


def crops():
    src = Image.open(OUT / "11_презентации" / "ai_vnigni_map_pair.png").convert("RGB")
    src.crop((0, 0, src.width, int(src.height * 0.765))).save(FIG / "fz_pair.png")
    src.crop((1030, 15, 1880, 930)).save(FIG / "fz_sheet.png")
    print("OK fz_pair, fz_sheet")


def fig_numbers():
    f = pd.read_csv(MET / "full_sheet_map.csv", encoding="utf-8-sig")
    r = f[(f["цель"] == "порог верхних 10%") & (f["ядро_км"] == "5 x 1.25")]
    full = r[r["сборка"] == "обучение на всём листе"].iloc[0]
    blind = r[r["сборка"].str.contains("полосам")].iloc[0]
    c = pd.read_csv(MET / "ceiling_decomposition.csv", encoding="utf-8-sig")
    ceil = c[c["вариант"].str.contains("потолок")].iloc[0]
    before = [full["совпадение_верх_10"], blind["совпадение_верх_10"],
              ceil["совпадение_верх_10"]]
    after = [full["совпадение_без_тренда"], blind["совпадение_без_тренда"],
             ceil["совпадение_без_тренда"]]
    names = ["итоговая карта\n(модель видела\nвесь лист)",
             "слепая проверка\nпо полосам\nс разрывом 15 км", "потолок\nслепой проверки"]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    x = np.arange(3)
    b1 = ax.bar(x - 0.19, before, 0.36, color="#93C5FD", label="до поправки на наклон")
    b2 = ax.bar(x + 0.19, after, 0.36, color="#1E3A8A", label="после поправки на наклон")
    for bars in (b1, b2):
        for rr in bars:
            ax.text(rr.get_x() + rr.get_width() / 2, rr.get_height() + 0.012,
                    "%.3f" % rr.get_height(), ha="center", fontsize=9,
                    fontweight="bold")
    ax.axhline(0.10, color="#DC2626", lw=1, ls="--", label="случайный уровень 0.10")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("совпадение верхних 10% площади", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fz_numbers.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    crops()
    fig_hamissana()
    fig_cores()
    fig_numbers()
    A = Area()
    fig_transfer(A)
    fig_analog(A)
    fig_uranium(A)
    fig_formula(A)
