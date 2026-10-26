"""WP5 ТЗ ИПУЗ: диагностические карты наблюдаемости.

Собирает:
  * ``coverage_aster.png``  — контур съёмки ASTER и ядро критериальной карты:
    видно, совпадает ли граница съёмки с границей перспективной зоны;
  * ``bare_soil_mask.png``  — доля открытого грунта с обоими порогами, 2 % и
    20 %, и ядро критериальной карты поверх.

Обе карты отвечают на один вопрос ТЗ: не является ли прибавка метрики
отражением доступности наблюдений, а не геологии.

Запуск::

    python -m experiments.ipuz_wp5_maps <каталог_заморозки>
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mapstyle  # noqa: E402
from experiments.ipuz_maps import (CELL_M, INK, PLAIN, PROJ4_SHORT, SHAPE,
                                   footer, frame, grid, load, north,
                                   scalebar, top_decile)  # noqa: E402


def fig_coverage(d: dict, a10: np.ndarray, out: Path) -> dict:
    """Контур съёмки ASTER против ядра критериальной карты."""
    cov = np.asarray(d["ast_cov"], dtype=bool)
    code = np.full(cov.size, np.nan)
    code[d["valid"]] = 0.0
    code[d["valid"] & cov] = 1.0
    code[d["valid"] & a10 & ~cov] = 2.0
    code[d["valid"] & a10 & cov] = 3.0

    colors = ["#eef2f7", "#cfe3e0", "#a3122b", "#0f766e"]
    inside = int((a10 & cov).sum())
    outside = int((a10 & ~cov).sum())
    labels = [
        "вне съёмки ASTER",
        "съёмка ASTER — %d яч., %.1f %%" % (int((cov & d["valid"]).sum()),
                                            100 * cov[d["valid"]].mean()),
        "ядро критерия вне съёмки — %d яч." % outside,
        "ядро критерия внутри съёмки — %d яч." % inside,
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
    share = inside / max(a10.sum(), 1)
    ax.set_title("Контур съёмки ASTER и ядро критериальной карты\n"
                 "внутри съёмки %.1f %% ядра при %.1f %% площади листа"
                 % (100 * share, 100 * cov[d["valid"]].mean()),
                 fontsize=9.5, color=INK)
    fig.legend(handles=[Patch(fc=c, ec="#94a3b8", lw=0.4, label=l)
                        for c, l in zip(colors, labels)],
               loc="lower center", bbox_to_anchor=(0.5, 0.05), frameon=False,
               fontsize=8, ncol=2)
    footer(fig, "Лист R-48-XI,XII, сетка 500 м.")
    mapstyle.save_fig(fig, out, dpi=300)
    plt.close(fig)
    return {"ядро_внутри_съёмки": inside, "ядро_вне_съёмки": outside,
            "доля_ядра_внутри": round(float(share), 4),
            "доля_площади_съёмки": round(float(cov[d["valid"]].mean()), 4)}


def fig_bare(d: dict, a10: np.ndarray, out: Path) -> dict:
    """Доля открытого грунта и оба порога."""
    bare = pd.read_parquet(
        ROOT / "data/processed/s2_features.parquet")["s2_bare_frac"].to_numpy(float)
    show = np.where(d["valid"], bare, np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 7.1))
    fig.subplots_adjust(left=0.105, right=0.885, top=0.885, bottom=0.195)
    ax.set_facecolor(PLAIN)
    im = ax.imshow(grid(show), extent=d["ext"], origin="upper",
                   interpolation="nearest", cmap="YlOrBr", vmin=0, vmax=0.6,
                   zorder=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("доля открытого грунта в ячейке", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    xs = np.linspace(d["ext"][0] + CELL_M / 2, d["ext"][1] - CELL_M / 2, SHAPE[1])
    ys = np.linspace(d["ext"][3] - CELL_M / 2, d["ext"][2] + CELL_M / 2, SHAPE[0])
    stats = {}
    for thr, color, lw in ((0.02, "#1d4ed8", 1.3), (0.20, "#111827", 1.3)):
        m = (np.isfinite(bare) & (bare >= thr) & d["valid"])
        ax.contour(xs, ys, m.reshape(SHAPE).astype(float), levels=[0.5],
                   colors=[color], linewidths=lw, zorder=4)
        stats["порог_%.2f" % thr] = {
            "ячеек": int(m.sum()),
            "доля_листа": round(float(m[d["valid"]].mean()), 4),
            "доля_ядра_покрыта": round(float((m & a10).sum() / max(a10.sum(), 1)), 4),
        }
    ax.contour(xs, ys, a10.reshape(SHAPE).astype(float), levels=[0.5],
               colors=["#a3122b"], linewidths=1.6, zorder=5)

    frame(ax, d)
    scalebar(ax, d)
    north(ax, d)
    ax.set_title("Открытый грунт и два порога маски\n"
                 "рабочий порог 2 %% покрывает %.1f %% листа, порог 20 %% — %.1f %%"
                 % (100 * stats["порог_0.02"]["доля_листа"],
                    100 * stats["порог_0.20"]["доля_листа"]),
                 fontsize=9.5, color=INK)
    fig.legend(handles=[
        Line2D([], [], color="#1d4ed8", lw=1.3, label="порог 2 % — рабочий"),
        Line2D([], [], color="#111827", lw=1.3, label="порог 20 %"),
        Line2D([], [], color="#a3122b", lw=1.6, label="ядро критериальной карты"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.05), frameon=False,
        fontsize=8, ncol=3)
    footer(fig, "Лист R-48-XI,XII, сетка 500 м. Источник — s2_features.parquet.")
    mapstyle.save_fig(fig, out, dpi=300)
    plt.close(fig)
    return stats


def main() -> None:
    freeze = Path(sys.argv[1])
    maps = freeze / "05_maps"
    maps.mkdir(parents=True, exist_ok=True)

    d = load(freeze)
    a10 = top_decile(d["crit"], d["pool"])

    cov = fig_coverage(d, a10, maps / "coverage_aster.png")
    bare = fig_bare(d, a10, maps / "bare_soil_mask.png")

    res = {"покрытие_ASTER": cov, "обнажённость": bare}
    (maps / "wp5_coverage.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(res, ensure_ascii=False, indent=2))
    for n in ("coverage_aster.png", "bare_soil_mask.png"):
        p = maps / n
        print("  %-22s %s" % (n, "OK %d КБ" % (p.stat().st_size // 1024)
                              if p.exists() else "НЕ СОЗДАН"))


if __name__ == "__main__":
    main()
