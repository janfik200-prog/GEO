"""Очистка линеаментов от долин: свои каналы звеньев И, П, Н без речной сети.

Зачем. Линеаменты проекта (`src/lineaments.py`) — края многоазимутной отмывки
рельефа, а края отмывки — прежде всего бровки долин. Пробная карта на знаниях
v0 показала, что поэтому звенья «источник», «путь» и «место накопления»
повторяют современную речную сеть (память project_knowledge_map_v0). Здесь
отрезки делятся на два класса:

* **долинный** — идёт вдоль тальвега: больше половины длины отрезка ближе
  ``W_M`` к тальвегу и под углом к нему не больше ``ANGLE_DEG``;
* **вне долин** — всё остальное, включая отрезки, пересекающие долины.

Долинные отрезки не выбрасываются, а идут в отдельный канал ``linv_*``
«спрямлённые долины»: отрезок Хафа прямой по построению, поэтому долинный
отрезок — это прямой участок долины, кандидат в К26 «долины разломного
заложения» вместо `dist_dnara` (полигоны гидросети). Каналы вне долин —
``linc_*``, на замену ``lin_*`` в звеньях И, П, Н.

Тальвеги — по самому рельефу (D8 по заполненному рельефу 100 м, водосбор не
меньше ``A0_KM2``), не по рекам организации: метод должен переноситься за
лист. Реки организации (`ТОПО/dnl.shp`) — только сверка сети тальвегов.

Записано до расчёта (критерии успеха):
1. сеть тальвегов воспроизводит реки организации: не меньше 70% длины рек
   ближе 300 м к тальвегу;
2. долинных отрезков заметно больше, чем при случайном повороте тех же
   отрезков вокруг их середины (иначе правило ловит не долины, а густоту сети);
3. после очистки связь плотности и расстояния до линеаментов с
   расчленённостью рельефа и с расстоянием до тальвегов падает до |ρ| < 0.3
   (до очистки ``lin_dist`` ~ расчленённость ρ ≈ −0.8);
4. очищенные линии не становятся артефактом освещения: согласие карт по
   чётным и нечётным азимутам отмывки остаётся не ниже стоп-правила ветки
   (``config.LIN_MIN_SPEARMAN`` = 0.5).
Сверка с разломными зонами формулы (⚠ `dist_tect1/2`) — только справочно:
слои — зоны по ~2000 км², а не линии (память project_fault_layers_are_zones).

Чувствительность — те же проверки при W 300 и 800 м и водосборе 0.5 и 5 км².

Выход: `data/processed/lineament_clean.parquet` (row, col, linc_*, linv_*,
dist_thalweg) и `outputs/13_генетическая_модель/gmo_lineaments_clean.png`.

Запуск::

    python -X utf8 -m experiments.gmo_lineaments_clean
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import catchments, config, integro_grid, lineaments, mapstyle, outputs, terrain_v2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
OUT = ROOT / "data/processed/lineament_clean.parquet"
RES = config.TER_RES_M          # 100 м
PAD = config.TER_PAD_PX
A0_KM2, W_M, ANGLE_DEG, FRAC = 2.0, 500.0, 30.0, 0.5
SENS = [(2.0, 300.0), (2.0, 800.0), (0.5, 500.0), (5.0, 500.0)]
rng = np.random.default_rng(7)
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:5.0f} с] {msg}", flush=True)


# ---------- тальвеги ----------
def thalwegs(elev: np.ndarray, valid: np.ndarray):
    """Накопленная площадь (пикс.) и направление стока (угол оси в кадре строка/столбец)."""
    rec = catchments.flow_dirs_d8(np.where(valid, elev, np.nan), valid)
    acc = catchments.flow_accumulation(rec).reshape(elev.shape)
    pic = elev.shape[1]
    i = np.arange(rec.size)
    dr = np.where(rec >= 0, rec // pic - i // pic, 0).reshape(elev.shape)
    dc = np.where(rec >= 0, rec % pic - i % pic, 0).reshape(elev.shape)
    return acc, np.arctan2(dr, dc)


def orientation_field(mask: np.ndarray, ang: np.ndarray, sigma: float = 3.0):
    """Сглаженная ось тальвега (удвоенный угол) — у отдельного шага D8 всего 8 направлений."""
    m = mask.astype(float)
    c = ndimage.gaussian_filter(m * np.cos(2 * ang), sigma)
    s = ndimage.gaussian_filter(m * np.sin(2 * ang), sigma)
    return 0.5 * np.arctan2(s, c)


class Valleys:
    def __init__(self, acc, ang, a0_km2: float):
        self.mask = acc >= a0_km2 * 1e6 / RES ** 2
        self.dist, idx = ndimage.distance_transform_edt(~self.mask, return_indices=True)
        self.dist = self.dist * RES
        axis = orientation_field(self.mask, ang)
        self.near_axis = axis[idx[0], idx[1]]   # ось ближайшего тальвега в каждом пикселе


def seg_valley_frac(seg, v: Valleys, w_m: float) -> float:
    from skimage.draw import line as draw_line
    (x0, y0), (x1, y1) = seg
    rr, cc = draw_line(int(y0), int(x0), int(y1), int(x1))
    ok = (rr >= 0) & (rr < v.dist.shape[0]) & (cc >= 0) & (cc < v.dist.shape[1])
    rr, cc = rr[ok], cc[ok]
    if len(rr) == 0:
        return 0.0
    a = np.arctan2(y1 - y0, x1 - x0)
    d = np.abs(np.angle(np.exp(2j * (a - v.near_axis[rr, cc])))) / 2   # 0..π/2
    return float(np.mean((v.dist[rr, cc] <= w_m) & (d <= np.radians(ANGLE_DEG))))


def classify(segs, v: Valleys, w_m: float) -> np.ndarray:
    return np.array([seg_valley_frac(s, v, w_m) > FRAC for s in segs], dtype=bool)


def rotated(segs):
    """Те же отрезки, повёрнутые на случайный угол вокруг середины."""
    out = []
    for (x0, y0), (x1, y1) in segs:
        mx, my, h = (x0 + x1) / 2, (y0 + y1) / 2, np.hypot(x1 - x0, y1 - y0) / 2
        a = rng.uniform(0, np.pi)
        out.append(((mx - h * np.cos(a), my - h * np.sin(a)),
                    (mx + h * np.cos(a), my + h * np.sin(a))))
    return out


# ---------- проверки ----------
def _capture(riv, v: Valleys, meta) -> float:
    """Доля длины рек организации ближе 300 м к тальвегам (сверка, не вход)."""
    from rasterio import features
    T = _transform(meta)
    r = features.rasterize(((g, 1) for g in riv.geometry), out_shape=v.dist.shape,
                           transform=T, fill=0, dtype="uint8").astype(bool)
    return float((v.dist[r] <= 300).mean())


def _transform(meta):
    from rasterio.transform import from_origin
    left, top = integro_grid_origin(meta)
    return from_origin(left - PAD * RES, top + PAD * RES, RES, RES)


def integro_grid_origin(meta):
    ds = pd.read_parquet(DATA, columns=["row", "col", "x", "y"]).iloc[0]
    return ds.x - (ds.col + 0.5) * meta.dx, ds.y + (ds.row + 0.5) * meta.dy


def half_reproducibility(elev, v: Valleys, w_m: float) -> dict[str, float]:
    az = list(config.LIN_AZIMUTHS)
    smooth = max(3, config.LIN_MIN_LEN_PX | 1)
    maps = {"все": [], "вне долин": []}
    for half in (az[::2], az[1::2]):
        segs = lineaments.extract_lines(lineaments.edge_map(elev, azimuths=half, res_m=RES))
        keep = ~classify(segs, v, w_m)
        for k, ss in (("все", segs), ("вне долин", [s for s, q in zip(segs, keep) if q])):
            maps[k].append(ndimage.uniform_filter(lineaments.density_raster(ss, elev.shape),
                                                  size=smooth))
    out = {}
    for k, (a, b) in maps.items():
        a, b = a.ravel(), b.ravel()
        both = (a > 0) | (b > 0)
        out[k] = float(spearmanr(a[both], b[both]).statistic)
    return out


def relief_links(df: pd.DataFrame, prefixes) -> pd.DataFrame:
    rows = []
    for p in prefixes:
        for f in ("dens", "dist", "node_dens"):
            r = {"канал": f"{p}{f}"}
            for ref in ("relief2_tri_std", "dem_slope", "dist_thalweg", "dist_dnl",
                        "dist_tect1", "dist_tect2"):
                ok = df[f"{p}{f}"].notna() & df[ref].notna()
                r[ref] = spearmanr(df.loc[ok, f"{p}{f}"], df.loc[ok, ref]).statistic
            rows.append(r)
    return pd.DataFrame(rows).set_index("канал").round(2)


def draw(elev, segs, valley, v: Valleys, df: pd.DataFrame, meta) -> None:
    fig = plt.figure(figsize=(17, 10.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(lineaments.hillshade(elev, 315.0), cmap="gray", alpha=0.8)
    ax.imshow(np.ma.masked_where(~v.mask, v.mask), cmap="Blues", vmin=0, vmax=1.5, alpha=0.6)
    for flag, col, lab in ((False, "#c0392b", "вне долин"), (True, "#1f6fb2", "вдоль долин")):
        ss = [[a, b] for (a, b), q in zip(segs, valley) if q == flag]
        ax.add_collection(LineCollection(ss, colors=col, linewidths=0.7,
                                         label=f"{lab}: {len(ss)} отрезков"))
    ax.set_xlim(PAD, PAD + meta.pic * 5); ax.set_ylim(PAD + meta.prf * 5, PAD)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title(f"Отрезки поверх отмывки 315°, голубое — тальвеги (водосбор ≥ {A0_KM2:g} км²)",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    nrow, ncol = meta.prf, meta.pic
    for i, (col, ttl) in enumerate((("lin_dens", "до очистки: плотность линий"),
                                     ("linc_dens", "вне долин: плотность линий"),
                                     ("linv_dens", "спрямлённые долины: плотность"),
                                     ("dist_thalweg", "расстояние до тальвега, м"))):
        a = np.full((nrow, ncol), np.nan)
        a[df.row, df.col] = df[col]
        axi = fig.add_subplot(2, 4, 3 + (i % 2) + 4 * (i // 2))
        lo, hi = np.nanpercentile(a, [2, 98])
        im = axi.imshow(a, cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        axi.set_title(ttl, fontsize=10); axi.set_xticks([]); axi.set_yticks([])
        fig.colorbar(im, ax=axi, shrink=0.7, label="м" if "dist" in col else "км/км²")
    fig.suptitle("Лист R-48-XI,XII: линеаменты по рельефу 100 м, очистка от долин "
                 f"(ближе {W_M:.0f} м к тальвегу и под углом ≤ {ANGLE_DEG:.0f}°)", fontsize=12)
    fig.tight_layout()
    out = outputs.png("gmo_lineaments_clean.png")
    mapstyle.save_fig(fig, out, dpi=130)
    log(f"рисунок: {out}")


def main() -> None:
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    raw = terrain_v2.projected_dem(meta)
    valid = np.isfinite(raw)
    elev = np.where(valid, raw, np.nanmedian(raw))
    log(f"рельеф {elev.shape}, шаг {RES:.0f} м")
    acc, ang = thalwegs(raw, valid)
    log("тальвеги посчитаны")
    segs = lineaments.extract_lines(lineaments.edge_map(elev, res_m=RES))
    log(f"отрезков {len(segs)}")

    ds = pd.read_parquet(DATA, columns=["row", "col", "lin_dens", "lin_dist", "lin_node_dens",
                                        "relief2_tri_std", "dem_slope", "dist_dnl",
                                        "dist_tect1", "dist_tect2"])
    ref = lineaments.rasters_to_cell_features(meta, lineaments.line_rasters(segs, elev.shape),
                                              "lin_", pad=PAD)
    rho = spearmanr(ref.lin_dens.to_numpy()[ds.row * meta.pic + ds.col], ds.lin_dens).statistic
    log(f"пересчёт совпадает с датасетом: lin_dens ρ = {rho:.3f}")

    def run(a0, w, report: bool):
        v = Valleys(acc, ang, a0)
        valley = classify(segs, v, w)
        null = np.mean([classify(rotated(segs), v, w).mean() for _ in range(10)])
        lens = np.array([np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in segs]) * RES / 1e3
        res = {"водосбор, км²": a0, "W, м": w, "тальвеги, % пл.": 100 * v.mask.mean(),
               "реки орг. у тальвега, %": 100 * _capture(
                   gpd.read_file(config.TOPO_SHP_DIR / "dnl.shp"), v, meta),
               "долинных, %": 100 * valley.mean(), "при повороте, %": 100 * null,
               "длина долинных, %": 100 * lens[valley].sum() / lens.sum()}
        if not report:
            return res
        rep = half_reproducibility(elev, v, w)
        res |= {"воспр. все": rep["все"], "воспр. вне долин": rep["вне долин"]}
        cells = []
        for pre, ss in (("linc_", [s for s, q in zip(segs, valley) if not q]),
                        ("linv_", [s for s, q in zip(segs, valley) if q])):
            cells.append(lineaments.rasters_to_cell_features(
                meta, lineaments.line_rasters(ss, elev.shape), pre, pad=PAD))
        dt = v.dist[PAD:PAD + meta.prf * 5, PAD:PAD + meta.pic * 5]
        cells.append(pd.DataFrame({"dist_thalweg": dt.reshape(meta.prf, 5, meta.pic, 5)
                                   .mean(axis=(1, 3)).ravel()}))
        g = pd.concat(cells, axis=1)
        g["row"], g["col"] = np.divmod(np.arange(len(g)), meta.pic)
        return res, g, v, valley

    res, g, v, valley = run(A0_KM2, W_M, report=True)
    log("основной вариант посчитан")
    df = ds.merge(g, on=["row", "col"], how="left")
    keep = ["row", "col", "dist_thalweg"] + [c for c in g if c.startswith(("linc_", "linv_"))]
    df[keep].to_parquet(OUT, index=False)
    log(f"сохранено: {OUT}")

    print("\nОсновной вариант:")
    for k, val in res.items():
        print(f"   {k}: {val:.2f}" if isinstance(val, float) else f"   {k}: {val}")
    print("\nСвязь каналов с рельефом и разломными зонами (ρ Спирмена, ячейки листа):")
    print(relief_links(df, ("lin_", "linc_", "linv_")).to_string())
    print("\nЧувствительность:")
    print(pd.DataFrame([run(a0, w, report=False) for a0, w in SENS]).round(1).to_string(index=False))
    draw(elev, segs, valley, v, df, meta)


if __name__ == "__main__":
    main()
