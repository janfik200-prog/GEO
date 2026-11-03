"""Гипотеза Г11: ячейки в пяти свитах из списка ВНИГНИ богаче золотом при равной обнажённости.

Цель: первая проверка предметной модели на данных. Критерий К23 (объект О11,
звено «место накопления» этапа III): золото рифея накапливается в ильинской,
бурдурской, лабастахской, усть-ильинской и котуйканской свитах (ответ 4 ВНИГНИ
от 02.11.2026). Если это так, золото на листе должно тяготеть к ним сильнее,
чем к остальной площади — четвертичным долинам, долеритам, кембрию и архею.

Входные данные:
* канал К23 — `data/processed/k23_svity_grid.parquet` (`experiments/gmo_k23_channel.py`);
* обнажённость `s2_bare_frac` — `data/processed/dataset_v5_rebuilt.parquet`;
* заверка — только золото, в обучение не входит:
  - 5 золотых ореолов из 14 контуров `геохимические ореолы.shp`;
  - пробы с золотом из `геохимическое_опробование.shp`: литохимия по коренным
    (код 2, 7 проб) и по рыхлым отложениям (код 1, 38 проб). Шлиховые (код 5)
    не берутся: их золото принесено по долинам, а россыпи модель не
    объясняет (ограничение Ог6).

Как проверяется:
1. Ячейки 500 м, покрытые картой не меньше чем наполовину. Ячейка «в списке»,
   если свиты из списка занимают в ней не меньше половины открытой части.
2. Слои по обнажённости — пороги `s2_bare_frac` 0 / 0.02 / 0.1 / 0.3 и слой без
   данных: сравнение «в списке / вне» идёт внутри слоя, итог — отношение
   долей Мантеля — Хензеля.
3. Неопределённость. Ореолов пять, поэтому для них — перестановка объектов:
   каждый ореол целиком поворачивается и сдвигается в случайное место листа,
   2000 раз. Пробы стоят кучно, поэтому для них — бутстрэп по блокам 10 км.
4. Сравнение с пробами на другие элементы — не контроль места опробования (в слое только аномальные пробы): доля «в списке» у проб с золотом против проб
   тех же видов на другие элементы (точный критерий Фишера).
5. По свитам: сколько положительных ячеек в каждой свите против ожидаемого
   при той же обнажённости.

Ожидаемый результат: вердикт Г11 — «работает», «не работает» или «не
проверяема», в `outputs/13_генетическая_модель/gmo_h11_svity_halos.png` и в
консоли.

Запуск::

    python -X utf8 -m experiments.gmo_h11_svity_halos
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely import affinity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, mapstyle, outputs, spectral_mask  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

K23 = ROOT / "data/processed/k23_svity_grid.parquet"
BARE = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
SHP = ROOT / "data/Gis-integro/shp_dbf"
CELL, SUB = 500.0, 10
N_PERM, N_BOOT, BLOCK = 2000, 2000, 10_000.0
SVK = {"svk_kt": "котуйканская", "svk_ul": "усть-ильинская", "svk_lb": "лабастахская",
       "svk_br": "бурдурская", "svk_il": "ильинская"}
rng = np.random.default_rng(11)


def load_cells() -> pd.DataFrame:
    df = pd.read_parquet(K23)
    bare = pd.read_parquet(BARE, columns=["row", "col", spectral_mask.BARE_COL])
    df = df.merge(bare, on=["row", "col"], how="left")
    df = df[df.svk_cover >= 0.5].reset_index(drop=True)
    df["in_list"] = df.svk_list >= 0.5
    b = df[spectral_mask.BARE_COL]
    # у 65% ячеек открытой породы нет вовсе — квартили слипаются в ноль,
    # поэтому пороги фиксированные: нет / порог спектра 0.02 / 0.1 / 0.3
    df["stratum"] = np.where(b.isna(), -1, np.digitize(b, [1e-9, 0.02, 0.1, 0.3]))
    return df


def halo_fraction(polys, df: pd.DataFrame) -> np.ndarray:
    """Доля каждой ячейки под ореолами — растр 50 м, среднее по ячейке."""
    from rasterio import features
    from rasterio.transform import from_origin
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    x0, ytop = (df.x - (df.col + 0.5) * CELL).iloc[0], (df.y + (df.row + 0.5) * CELL).iloc[0]
    T = from_origin(x0, ytop, CELL / SUB, CELL / SUB)
    r = features.rasterize(((p, 1) for p in polys), out_shape=(nrow * SUB, ncol * SUB),
                           transform=T, fill=0, dtype="uint8")
    f = r.reshape(nrow, SUB, ncol, SUB).mean(axis=(1, 3))
    return f[df.row.to_numpy(), df.col.to_numpy()]


def point_cells(pts: gpd.GeoDataFrame, df: pd.DataFrame) -> np.ndarray:
    key = dict(zip(zip(df.row, df.col), range(len(df))))
    x0, ytop = (df.x - (df.col + 0.5) * CELL).iloc[0], (df.y + (df.row + 0.5) * CELL).iloc[0]
    col = np.floor((pts.geometry.x - x0) / CELL).astype(int)
    row = np.floor((ytop - pts.geometry.y) / CELL).astype(int)
    pos = np.zeros(len(df), bool)
    for r, c in zip(row, col):
        if (r, c) in key:
            pos[key[(r, c)]] = True
    return pos


def mh_ratio(pos: np.ndarray, df: pd.DataFrame) -> float:
    """Отношение долей положительных «в списке / вне» по Мантелю — Хензелю."""
    num = den = 0.0
    for _, s in df.groupby("stratum").groups.items():
        s = np.asarray(s)
        e = df.in_list.to_numpy()[s]
        p = pos[s]
        n1, n0, N = e.sum(), (~e).sum(), len(s)
        if n1 == 0 or n0 == 0:
            continue
        num += p[e].sum() * n0 / N
        den += p[~e].sum() * n1 / N
    if den == 0:
        return np.inf if num > 0 else np.nan
    return num / den


def expected_by_svita(pos: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Положительные ячейки по свитам против ожидаемого при той же обнажённости."""
    rows = []
    groups = list(SVK) + ["вне списка"]
    main = np.where(df.in_list, df[list(SVK)].idxmax(axis=1), "вне списка")
    for gname in groups:
        m = main == gname
        exp = 0.0
        for _, s in df.groupby("stratum").groups.items():
            s = np.asarray(s)
            exp += pos[s].sum() * m[s].mean()
        rows.append((SVK.get(gname, gname), int(m.sum()), int(pos[m].sum()), round(exp, 1)))
    return pd.DataFrame(rows, columns=["свита", "ячеек", "положительных", "ожидалось"])


def block_boot(pos: np.ndarray, df: pd.DataFrame) -> tuple[float, float]:
    bid = (np.floor(df.x / BLOCK).astype(int) * 1000 + np.floor(df.y / BLOCK).astype(int)).to_numpy()
    ub = np.unique(bid)
    idx = {b: np.flatnonzero(bid == b) for b in ub}
    vals = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        sub = df.iloc[take].reset_index(drop=True)
        v = mh_ratio(pos[take], sub)
        if not np.isnan(v):
            vals.append(v)
    # без интерполяции: при нуле положительных вне списка отношение бесконечно
    return (float(np.percentile(vals, 2.5, method="lower")),
            float(np.percentile(vals, 97.5, method="higher")))


def halo_perm(polys, df: pd.DataFrame, area) -> tuple[np.ndarray, int]:
    """Каждый ореол — в случайное место `area` с поворотом, целиком внутри неё.

    Раскладка, где хотя бы один ореол не уложился за 200 попыток, не
    урезается, а пересоздаётся целиком: иначе крупные ореолы, которые хуже
    входят в площадь со щелями, тихо выпадали бы из нулевого распределения.
    Возвращает отношения М-Х и число пересозданных раскладок.
    """
    minx, miny, maxx, maxy = area.bounds
    out, redo = [], 0
    while len(out) < N_PERM:
        moved = []
        for p in polys:
            for _try in range(200):
                q = affinity.rotate(p, rng.uniform(0, 360), origin="centroid")
                c = q.centroid
                q = affinity.translate(q, rng.uniform(minx, maxx) - c.x, rng.uniform(miny, maxy) - c.y)
                if area.contains(q):
                    moved.append(q)
                    break
            else:
                break
        if len(moved) < len(polys):
            redo += 1
            continue
        out.append(mh_ratio(halo_fraction(moved, df) >= 0.1, df))
    return np.array(out), redo


def perm_p(perm: np.ndarray, obs: float) -> float:
    """Бесконечное отношение (все ореолы в свитах списка) не меньше любого
    наблюдаемого — оно считается в пользу случайности, а не выбрасывается."""
    perm = perm[~np.isnan(perm)]
    return (np.sum(perm >= obs) + 1) / (len(perm) + 1)


def min_detectable(n: int, share: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Наименьшее отношение долей, которое n положительных ячеек обнаружат.

    Под нулём положительная ячейка вне списка с вероятностью 1 - share;
    критерий — мало положительных вне списка (односторонний). Ищется
    отношение r, при котором мощность достигает `power`.
    """
    from scipy.stats import binom
    p0 = 1 - share
    k = int(binom.ppf(alpha, n, p0))
    while k >= 0 and binom.cdf(k, n, p0) > alpha:
        k -= 1
    if k < 0:
        return np.inf
    for r in np.arange(1.0, 200.0, 0.1):
        pr = (1 - share) / (1 - share + share * r)
        if binom.cdf(k, n, pr) >= power:
            return round(float(r), 1)
    return np.inf


def main() -> None:
    df = load_cells()
    crs = anabar_inset.SHEET_CRS
    halos = gpd.read_file(SHP / "геохимические ореолы.shp").set_crs(crs, allow_override=True)
    au_h = [shapely.Polygon(g.coords) for g in halos[halos.label == "Au"].geometry]
    pts = gpd.read_file(SHP / "геохимическое_опробование.shp").set_crs(crs, allow_override=True)
    pts["au"] = pts.label.astype(str).str.contains("Au")

    print(f"Ячеек под картой {len(df)}, в списке {int(df.in_list.sum())} "
          f"({df.in_list.mean():.1%}); слои обнажённости: "
          f"{df.stratum.value_counts().sort_index().to_dict()}")

    sets = {
        "ореолы Au (5)": halo_fraction(au_h, df) >= 0.1,
        "пробы Au по коренным (7)": point_cells(pts[(pts.code == 2) & pts.au], df),
        "пробы Au по рыхлым (38)": point_cells(pts[(pts.code == 1) & pts.au], df),
    }
    sets["пробы Au: коренные + рыхлые"] = sets["пробы Au по коренным (7)"] | sets["пробы Au по рыхлым (38)"]

    rows = []
    for name, pos in sets.items():
        rr = mh_ratio(pos, df)
        share = df.in_list[pos].mean() if pos.any() else np.nan
        if name.startswith("ореолы"):
            cover = shapely.union_all(gpd.read_file(K23.with_name("svita_labeled.gpkg"))
                                      .to_crs(crs).geometry).buffer(0)
            # опробованная площадь: оболочка всех проб с запасом 5 км — так
            # нуль не смешивает «где золото» с «где вообще искали»
            sampled = shapely.convex_hull(shapely.MultiPoint(list(pts.geometry))).buffer(5000)
            parts = []
            for nm, area in (("весь лист", cover), ("опробованная площадь", cover.intersection(sampled))):
                perm, redo = halo_perm(au_h, df, area)
                parts.append(f"{nm}: p={perm_p(perm, rr):.3f} (пересоздано {redo})")
            unc = "перестановка, " + "; ".join(parts)
        else:
            lo, hi = block_boot(pos, df)
            unc = f"бутстрэп блоков 10 км: 95% [{lo:.2f}; {hi:.2f}]"
        rows.append((name, int(pos.sum()), f"{share:.0%}", f"{df.in_list.mean():.0%}",
                     round(rr, 2), unc))
    res = pd.DataFrame(rows, columns=["заверка", "ячеек", "доля в списке",
                                      "доля площади", "отношение М-Х", "неопределённость"])
    print("\n" + res.to_string(index=False))

    share = df.in_list.mean()
    n_all = int(sets["пробы Au: коренные + рыхлые"].sum())
    print(f"\nМинимальный обнаружимый эффект (мощность 0.8, α=0.05, доля списка {share:.1%}): "
          f"{n_all} ячеек с пробами — отношение ≥ {min_detectable(n_all, share)}; "
          f"22 независимых скопления — ≥ {min_detectable(22, share)}; "
          f"5 ореолов — ≥ {min_detectable(5, share)}")

    print("\nЧувствительность отношения М-Х к порогам:")
    base = df.copy()
    for thr in (0.3, 0.5, 0.7):
        base["in_list"] = base.svk_list >= thr
        line = [f"свиты ≥ {thr}: пробы {mh_ratio(sets['пробы Au: коренные + рыхлые'], base):.2f}"]
        for ht in (0.05, 0.1, 0.3):
            line.append(f"ореолы ≥ {ht}: {mh_ratio(halo_fraction(au_h, base) >= ht, base):.2f}")
        print("  " + "; ".join(line))

    # Сравнение золота с другими элементами тех же видов проб. Это НЕ контроль
    # места опробования: в слое только аномальные пробы (пустых нет), а у
    # урана своя литологическая приуроченность. Усилие опробования неизвестно.
    from scipy.stats import fisher_exact
    cells_pts = pts[pts.code.isin([1, 2])].copy()
    key = point_cells  # те же ячейки
    au_pos = key(cells_pts[cells_pts.au], df)
    ot_pos = key(cells_pts[~cells_pts.au], df) & ~au_pos
    a, b = int(df.in_list[au_pos].sum()), int((~df.in_list[au_pos]).sum())
    c, d = int(df.in_list[ot_pos].sum()), int((~df.in_list[ot_pos]).sum())
    orr, pf = fisher_exact([[a, b], [c, d]])
    print(f"\nЗолото против других элементов (не контроль опробования): золото в списке {a}, вне {b}; "
          f"другие элементы в списке {c}, вне {d}; отношение шансов {orr:.2f}, p={pf:.3f}")

    print("\nПо свитам (все пробы Au, коренные + рыхлые):")
    print(expected_by_svita(sets["пробы Au: коренные + рыхлые"], df).to_string(index=False))
    print("\nПо свитам (ореолы Au):")
    print(expected_by_svita(sets["ореолы Au (5)"], df).to_string(index=False))

    draw(df, au_h, pts)


def draw(df, au_h, pts) -> None:
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    x0 = (df.x - (df.col + 0.5) * CELL).iloc[0]
    ytop = (df.y + (df.row + 0.5) * CELL).iloc[0]
    ext = [x0, x0 + ncol * CELL, ytop - nrow * CELL, ytop]
    a = np.full((nrow, ncol), np.nan)
    a[df.row, df.col] = df.in_list.astype(float)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.subplots_adjust(left=0.07, right=0.66)
    from matplotlib.colors import ListedColormap
    ax.imshow(a, extent=ext, cmap=ListedColormap(["#bdbdbd", "#f6d7a7"]), vmin=0, vmax=1,
              interpolation="nearest")
    gpd.GeoSeries(au_h).boundary.plot(ax=ax, color="#b8860b", linewidth=2.2)
    for code, mk, lab in [(2, "s", "проба Au по коренным"), (1, "o", "проба Au по рыхлым")]:
        p = pts[(pts.code == code) & pts.au]
        ax.scatter(p.geometry.x, p.geometry.y, marker=mk, s=28, color="#c0392b",
                   edgecolor="k", linewidth=0.4, label=lab, zorder=5)
    p = pts[pts.code.isin([1, 2]) & ~pts.au]
    ax.scatter(p.geometry.x, p.geometry.y, marker="x", s=24, color="#1f3a93",
               label="пробы тех же съёмок на другие элементы", zorder=5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h = [Patch(color="#f6d7a7", label="ячейка в свитах из списка ВНИГНИ"),
         Patch(color="#bdbdbd", label="вне списка: долины, долериты, кембрий, архей"),
         Line2D([0], [0], color="#b8860b", lw=2.2, label="ореол Au")] + h
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              framealpha=0.92)
    ax.set_xlabel("X, м (Гаусс–Крюгер, осевой 105°)")
    ax.set_ylabel("Y, м")
    ax.ticklabel_format(style="plain")
    ax.tick_params(labelsize=7)
    ax.set_title("Г11: золото и свиты из списка ВНИГНИ, лист R-48-XI,XII, сетка 500 м",
                 fontsize=11)
    out = outputs.png("gmo_h11_svity_halos.png")
    mapstyle.save_fig(fig, out, dpi=140)
    print(f"\nКарта: {out}")


if __name__ == "__main__":
    main()
