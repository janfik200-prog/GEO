"""WP6 ТЗ ИПУЗ: три настоящие карты вместо макета рис. 4.

Собирает:
  * ``map_pair.png``    — критериальная карта и слепая карта по дистанционке
                          в одной сетке, палитре и шкале рангов;
  * ``map_overlap.png`` — пересечение верхних 10 % обеих поверхностей
                          с раздельной окраской;
  * ``map_detail.png``  — увеличенный фрагмент с рельефом, контуром съёмки
                          ASTER и маской обнажённости.

Источники карт — уже посчитанные файлы, модель не обучается:
  * критериальный балл  — ``prognoz.pgrid`` через ``target_mask.csv``;
  * слепая карта (OOF)  — ``data/processed/operating_point_masked.parquet``,
    та самая, что даёт O10 = 0,1986 и 0,3623 без тренда.

Экспорт только в PNG: ``rasterio`` закрыт политикой Smart App Control, поэтому
GeoTIFF из ТЗ (``crit.tif``, ``oof_remote.tif``) этим скриптом не пишется.

Запуск::

    python -m experiments.ipuz_maps <каталог_заморозки>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mapstyle  # noqa: E402

SHAPE = (154, 149)
CELL_M = 500.0
PROJ4_SHORT = ("попереч. Меркатора, осевой меридиан 105° в.д., "
               "эллипсоид Красовского (Пулково-1942)")
PLAIN = "#f8fafc"
INK = "#1E293B"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


# ------------------------------------------------------------------ подготовка
def load(freeze: Path) -> dict:
    """Собрать карты, маски и геометрию в один словарь."""
    t = pd.read_csv(freeze / "01_inputs/target_mask.csv")
    x = t["x"].to_numpy(float)
    y = t["y"].to_numpy(float)
    valid = t["valid"].to_numpy(bool)
    # Перспективность по возрастанию: в формуле меньший балл = перспективнее.
    crit = -t["crit_score"].to_numpy(float)

    op = pd.read_parquet(ROOT / "data/processed/operating_point_masked.parquet")
    remote = op["score"].to_numpy(float)

    feat = pd.read_parquet(ROOT / "data/processed/dataset_v6.parquet")
    ast_cols = [c for c in feat.columns if c.startswith("ast_")]
    ast_cov = feat[ast_cols].notna().any(axis=1).to_numpy(bool)
    dem = feat["dem_elev"].to_numpy(float)

    bare = pd.read_parquet(
        ROOT / "data/processed/s2_features.parquet")["s2_bare_frac"].to_numpy(float)
    bare_mask = np.isfinite(bare) & (bare >= 0.20)

    pool = valid & np.isfinite(remote)
    return {
        "x": x, "y": y, "valid": valid, "pool": pool,
        "crit": crit, "remote": remote, "dem": dem,
        "ast_cov": ast_cov, "bare": bare_mask,
        "ext": (x.min() - CELL_M / 2, x.max() + CELL_M / 2,
                y.min() - CELL_M / 2, y.max() + CELL_M / 2),
    }


def top_decile(v: np.ndarray, pool: np.ndarray, frac: float = 0.10) -> np.ndarray:
    m = pool & np.isfinite(v)
    thr = np.quantile(v[m], 1.0 - frac)
    out = np.zeros(v.size, dtype=bool)
    out[m] = v[m] >= thr
    return out


def grid(v: np.ndarray) -> np.ndarray:
    return np.ma.masked_invalid(np.asarray(v, dtype=float).reshape(SHAPE))


# -------------------------------------------------------------- оформление оси
def frame(ax, d: dict, *, ylabel: bool = True) -> None:
    """Рамка, оси в километрах, равный масштаб."""
    ax.set_xlim(d["ext"][0], d["ext"][1])
    ax.set_ylim(d["ext"][2], d["ext"][3])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
    ax.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
    ax.set_xlabel("X, км")
    if ylabel:
        ax.set_ylabel("Y, км")
    else:
        ax.set_yticklabels([])
    for s in ax.spines.values():
        s.set_edgecolor(INK)
        s.set_linewidth(0.8)


def scalebar(ax, d: dict, length_m: float = 20_000.0) -> None:
    """Масштабная линейка с подписью в километрах."""
    x0, x1, y0, y1 = d["ext"]
    pad = (x1 - x0) * 0.06
    bx = x0 + pad
    by = y0 + (y1 - y0) * 0.045
    h = (y1 - y0) * 0.011
    ax.add_patch(Rectangle((bx, by), length_m / 2, h, fc=INK, ec=INK, lw=0.5,
                           zorder=6))
    ax.add_patch(Rectangle((bx + length_m / 2, by), length_m / 2, h,
                           fc="white", ec=INK, lw=0.5, zorder=6))
    ax.text(bx + length_m / 2, by + h * 3.2, "%.0f км" % (length_m / 1000),
            ha="center", va="bottom", fontsize=7.5, color=INK, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def north(ax, d: dict) -> None:
    """Указатель севера."""
    x0, x1, y0, y1 = d["ext"]
    nx = x1 - (x1 - x0) * 0.065
    ny = y1 - (y1 - y0) * 0.075
    ax.annotate("", xy=(nx, ny), xytext=(nx, ny - (y1 - y0) * 0.065),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1),
                zorder=6)
    ax.text(nx, ny + (y1 - y0) * 0.008, "С", ha="center", va="bottom",
            fontsize=8.5, color=INK, fontweight="bold", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))


def footer(fig, text: str) -> None:
    """Подвал в две строки: длинная строка не влезает в габарит фигуры."""
    fig.text(0.5, 0.012, text + "\nСистема координат: " + PROJ4_SHORT,
             ha="center", va="bottom", fontsize=7, color="#475569",
             linespacing=1.5)


def classes(ax, d: dict, score: np.ndarray) -> None:
    """Пятиклассовая картограмма по процентилю внутри области оценки."""
    pct = mapstyle.percentile(score, d["pool"])
    cls = np.where(d["pool"], np.digitize(pct, mapstyle.CLASS_EDGES[1:-1]), np.nan)
    ax.imshow(grid(cls), extent=d["ext"], origin="upper",
              interpolation="nearest", vmin=-0.5, vmax=4.5,
              cmap=mapstyle.class_cmap(), zorder=2)


# ----------------------------------------------------------------- рис. 4 (а,б)
def fig_pair(d: dict, out: Path, o10_raw: float, o10_det: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 6.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.235,
                        wspace=0.06)
    for ax, (score, title) in zip(axes, [
            (d["crit"], "а) критериальная карта метода организации"),
            (d["remote"], "б) карта по дистанционным данным,\n"
                          "слепая проверка по полосам с разрывом 15 км")]):
        ax.set_facecolor(PLAIN)
        classes(ax, d, score)
        frame(ax, d, ylabel=ax is axes[0])
        scalebar(ax, d)
        ax.set_title(title, fontsize=9, color=INK)
    north(axes[0], d)

    handles = [Patch(fc=c, ec="#94a3b8", lw=0.4, label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8, title="ранг перспективности, процентиль внутри листа",
               title_fontsize=8.5, bbox_to_anchor=(0.5, 0.062))
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м, %d оцениваемых ячеек\n"
                 "совпадение верхних децилей O₁₀ = %.3f; "
                 "после снятия регионального тренда — %.3f"
                 % (int(d["pool"].sum()), o10_raw, o10_det),
                 fontsize=9.5, color=INK, y=0.975)
    footer(fig, "Обе карты в одной сетке, палитре и шкале рангов.")
    mapstyle.save_fig(fig, out, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------- карта пересечения
def fig_overlap(d: dict, out: Path, a10: np.ndarray, b10: np.ndarray) -> float:
    both = a10 & b10
    only_a = a10 & ~b10
    only_b = b10 & ~a10
    code = np.full(a10.size, np.nan)
    code[d["pool"]] = 0.0
    code[only_a] = 1.0
    code[only_b] = 2.0
    code[both] = 3.0

    colors = ["#eef2f7", "#8ea9c9", "#f2b544", "#a3122b"]
    labels = [
        "остальная площадь листа",
        "только критериальная карта — %d яч." % int(only_a.sum()),
        "только дистанционка — %d яч." % int(only_b.sum()),
        "совпадение — %d яч." % int(both.sum()),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 6.9))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.895, bottom=0.185)
    ax.set_facecolor(PLAIN)
    ax.imshow(grid(code), extent=d["ext"], origin="upper",
              interpolation="nearest", vmin=-0.5, vmax=3.5,
              cmap=matplotlib.colors.ListedColormap(colors), zorder=2)
    frame(ax, d)
    scalebar(ax, d)
    north(ax, d)
    share = float(both.sum() / a10.sum())
    ax.set_title("Пересечение верхних 10 %% двух карт\n"
                 "доля ядра критериальной карты, воспроизведённая "
                 "дистанционкой: %.3f" % share, fontsize=9.5, color=INK)
    fig.legend(handles=[Patch(fc=c, ec="#94a3b8", lw=0.4, label=l)
                        for c, l in zip(colors, labels)],
               loc="lower center", bbox_to_anchor=(0.5, 0.055), frameon=False,
               fontsize=8, ncol=2)
    footer(fig, "Лист R-48-XI,XII, сетка 500 м. "
                "Ядра равны по площади: по 2291 ячейке.")
    mapstyle.save_fig(fig, out, dpi=300)
    plt.close(fig)
    return share


# ------------------------------------------------------------ увеличенный фрагмент
def pick_window(d: dict, a10: np.ndarray, b10: np.ndarray,
                size: int = 40) -> tuple[int, int]:
    """Окно ``size``×``size`` ячеек, где видны оба ядра и их расхождение.

    Максимум расхождения сам по себе даёт окно, в котором одно из ядер почти
    отсутствует и читателю нечего сравнивать. Поэтому в оценке участвует
    меньшее из двух ядер, а расхождение добавляется с меньшим весом.
    """
    ga = a10.reshape(SHAPE).astype(float)
    gb = b10.reshape(SHAPE).astype(float)
    gx = (a10 ^ b10).reshape(SHAPE).astype(float)
    best, pos = -1.0, (0, 0)
    for r in range(0, SHAPE[0] - size + 1, 4):
        for c in range(0, SHAPE[1] - size + 1, 4):
            sl = (slice(r, r + size), slice(c, c + size))
            na, nb = ga[sl].sum(), gb[sl].sum()
            s = 2.0 * min(na, nb) + 0.5 * gx[sl].sum()
            if s > best:
                best, pos = s, (r, c)
    return pos


def fig_detail(d: dict, out: Path, a10: np.ndarray, b10: np.ndarray,
               size: int = 40) -> dict:
    r0, c0 = pick_window(d, a10, b10, size)
    x = d["x"].reshape(SHAPE)
    y = d["y"].reshape(SHAPE)
    ext = (x[r0, c0] - CELL_M / 2, x[r0, c0 + size - 1] + CELL_M / 2,
           y[r0 + size - 1, c0] - CELL_M / 2, y[r0, c0] + CELL_M / 2)
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    sub_ext = {"ext": ext}

    def sub(v):
        return np.ma.masked_invalid(
            np.asarray(v, dtype=float).reshape(SHAPE)[sl])

    fig, ax = plt.subplots(figsize=(7.0, 7.2))
    fig.subplots_adjust(left=0.105, right=0.885, top=0.875, bottom=0.215)
    ax.set_facecolor(PLAIN)
    im = ax.imshow(sub(d["dem"]), extent=ext, origin="upper",
                   interpolation="nearest", cmap="Greys", zorder=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("высота, м", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # Ядра — полупрозрачные заливки: бинарную маску контуром читать нельзя.
    for mask, color in ((a10, "#1d4ed8"), (b10, "#a3122b")):
        m2 = np.asarray(mask, dtype=bool).reshape(SHAPE)[sl]
        ax.imshow(np.ma.masked_where(~m2, m2.astype(float)), extent=ext,
                  origin="upper", interpolation="nearest", alpha=0.42,
                  cmap=matplotlib.colors.ListedColormap([color]),
                  vmin=0, vmax=1, zorder=3)

    # Обнажённость — редкий точечный крап, а не контур по каждой ячейке.
    bm = np.asarray(d["bare"], dtype=bool).reshape(SHAPE)[sl]
    bx, by = x[sl][bm], y[sl][bm]
    ax.plot(bx, by, ls="none", marker="o", ms=1.9, mfc="#b45309",
            mec="none", alpha=0.75, zorder=5)

    # Контур съёмки ASTER — только если он пересекает окно.
    am = np.asarray(d["ast_cov"], dtype=bool).reshape(SHAPE)[sl]
    ast_crosses = bool(am.any() and not am.all())
    if ast_crosses:
        ax.contour(np.linspace(ext[0] + CELL_M / 2, ext[1] - CELL_M / 2, size),
                   np.linspace(ext[3] - CELL_M / 2, ext[2] + CELL_M / 2, size),
                   am.astype(float), levels=[0.5], colors=["#0f766e"],
                   linewidths=1.8, zorder=6)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
    ax.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
    ax.set_xlabel("X, км")
    ax.set_ylabel("Y, км")
    scalebar(ax, sub_ext, length_m=5000.0)
    north(ax, sub_ext)
    ax.set_title("Увеличенный фрагмент %g×%g км: рельеф, ядра двух карт\n"
                 "и границы наблюдаемости" % (size * 0.5, size * 0.5),
                 fontsize=9.5, color=INK)
    ast_label = ("контур съёмки ASTER" if ast_crosses
                 else "съёмка ASTER покрывает окно целиком")
    fig.legend(handles=[
        Patch(fc="#1d4ed8", alpha=0.42, ec="none",
              label="ядро критериальной карты"),
        Patch(fc="#a3122b", alpha=0.42, ec="none",
              label="ядро дистанционной карты"),
        Patch(fc="#7a4a86", alpha=0.62, ec="none",
              label="перекрытие ядер — совпадение"),
        Line2D([], [], color="#b45309", ls="none", marker="o", ms=3.4,
               label="открытый грунт ≥ 20 % площади ячейки"),
        Line2D([], [], color="#0f766e", lw=1.8,
               ls="-" if ast_crosses else "none", label=ast_label),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.035), frameon=False,
        fontsize=8, ncol=2)
    footer(fig, "Окно выбрано там, где присутствуют оба ядра. "
                "Лист R-48-XI,XII, сетка 500 м.")
    mapstyle.save_fig(fig, out, dpi=300)
    plt.close(fig)
    return {"строка": int(r0), "столбец": int(c0), "размер_ячеек": size,
            "охват_м": [float(v) for v in ext]}


# ----------------------------------------------------------------------- запуск
def main() -> None:
    freeze = Path(sys.argv[1])
    maps_dir = freeze / "05_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    d = load(freeze)
    a10 = top_decile(d["crit"], d["pool"])
    b10 = top_decile(d["remote"], d["pool"])
    o10_raw = float((a10 & b10).sum() / a10.sum())

    # метрика без тренда — та же процедура, что в WP3
    xs = (d["x"] - d["x"].mean()) / d["x"].std()
    ys = (d["y"] - d["y"].mean()) / d["y"].std()
    D = np.column_stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys])

    def detrend(v):
        out = np.full(v.size, np.nan)
        ok = d["pool"] & np.isfinite(v)
        coef, *_ = np.linalg.lstsq(D[ok], v[ok], rcond=None)
        out[ok] = v[ok] - D[ok] @ coef
        return out

    a10d = top_decile(detrend(d["crit"]), d["pool"])
    b10d = top_decile(detrend(d["remote"]), d["pool"])
    o10_det = float((a10d & b10d).sum() / a10d.sum())

    fig_pair(d, maps_dir / "map_pair.png", o10_raw, o10_det)
    share = fig_overlap(d, maps_dir / "map_overlap.png", a10, b10)
    win = fig_detail(d, maps_dir / "map_detail.png", a10, b10)

    sources = {
        "лист": "R-48-XI,XII",
        "сетка": {"ячеек": 22946, "оцениваемых": int(d["pool"].sum()),
                  "шаг_м": CELL_M, "форма": list(SHAPE)},
        "crs_proj4": ("+proj=tmerc +lon_0=105 +lat_0=0 +x_0=500000 +y_0=0 "
                      "+k_0=1 +ellps=krass +units=m +no_defs"),
        "критериальная_карта": {
            "источник": "data/Gis-integro/Расчет/prognoz.pgrid",
            "через": "01_inputs/target_mask.csv :: crit_score",
            "шкала": "инвертирована: в формуле меньший балл = перспективнее",
        },
        "дистанционная_карта": {
            "источник": "data/processed/operating_point_masked.parquet",
            "сборка": "слепая, полосы 4×X и 4×Y, буфер 15 км",
            "проверка": "даёт O10 0,198603 и 0,362287 без тренда",
        },
        "контур_ASTER": {"источник": "dataset_v6.parquet, колонки ast_*",
                         "ячеек": 16014, "доля": 0.698},
        "маска_обнажённости": {"источник": "s2_features.parquet :: s2_bare_frac",
                               "порог": 0.20},
        "рельеф_на_фрагменте": "dataset_v6.parquet :: dem_elev",
        "метрики_на_рисунках": {"O10": round(o10_raw, 6),
                                "O10_без_тренда": round(o10_det, 6),
                                "доля_ядра_воспроизведена": round(share, 6)},
        "окно_фрагмента": win,
        "dpi": 300,
        "ограничение": ("500 м — размер расчётной ячейки, а не точность "
                        "исходных геологических границ"),
        "не_выполнено": ("экспорт в GeoTIFF: rasterio закрыт политикой "
                         "Smart App Control"),
    }
    (maps_dir / "map_sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")

    print("O10 =", round(o10_raw, 6), "| без тренда =", round(o10_det, 6))
    print("ядро критериальной карты:", int(a10.sum()), "яч.")
    print("ядро дистанционной карты:", int(b10.sum()), "яч.")
    print("совпадение:", int((a10 & b10).sum()), "яч. =", round(share, 4))
    print("окно фрагмента: строка", win["строка"], "столбец", win["столбец"])
    for n in ("map_pair.png", "map_overlap.png", "map_detail.png",
              "map_sources.json"):
        p = maps_dir / n
        print("  %-20s %s" % (n, "OK %d КБ" % (p.stat().st_size // 1024)
                              if p.exists() else "НЕ СОЗДАН"))


if __name__ == "__main__":
    main()
