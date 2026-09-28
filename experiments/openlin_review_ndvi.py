"""Окна для глазной проверки НОВЫХ линий NDVI — тех, что красный канал не нашёл.

ЗАЧЕМ. Прогон по NDVI (25.09.2026) прошёл все четыре условия, но 57–62 %
его подтверждённых линий повторяют красный канал, а новые (480 кусков по
всему листу) короче и вдвое реже лежат у воды. По цифрам не различить, это
тоновые границы на водоразделах или снеговой и растительный шум: мусор у NDVI
в 2.5 раза выше, чем у красного. Решает взгляд.

ЧТО НА КАРТИНКЕ. Окно 5 x 5 км в четырёх панелях: сверху чистые снимки —
обычный зимний (красный канал) и NDVI; снизу те же снимки с разметкой.
Синие — новые линии NDVI, красные тонкие — подтверждённые линии красного
канала, для ориентира. «Новая» линия — та, у которой меньше 30 % длины лежит
в 60 м от линий красного канала (то же правило совпадения, что в сверке).

Запуск из корня репозитория:
    python -m experiments.openlin_review_ndvi --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_review_tiles import COVER_TARGET, MAX_WINDOWS, WIN_KM, pick_windows
from experiments.openlin_zone_lines import COMP, M, OUT
from experiments.openlin_zone_lines_v3 import CONF_FRAC, CONF_TOL_M

OUT_DIR = OUT.parent / "линеаменты_проверка_ndvi"
NEW = (0.12, 0.42, 0.85, 1.0)      # синие — новые линии NDVI
RED = (0.85, 0.05, 0.05, 0.55)     # красные — линии красного канала, для ориентира


def new_lines_mask(ndvi_all, red_all, res_m: float) -> np.ndarray:
    """Куски NDVI, не повторяющие красный канал по правилу совпадения."""
    from scipy.ndimage import distance_transform_edt, label

    dred = distance_transform_edt(~red_all, sampling=(res_m, res_m))
    lab, n = label(ndvi_all, np.ones((3, 3), bool))
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        r, c = np.where(lab == i)
        keep[i] = (dred[r, c] <= CONF_TOL_M).mean() < CONF_FRAC
    return keep[lab] & ndvi_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch
    from experiments.openlin_winter import detrend

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    red = np.load(M / f"openlin_zone_lines_{a.site}_v3.npz")
    nd = np.load(M / f"openlin_zone_lines_{a.site}_v3_ndvi.npz")
    shape = red["C_all"].shape
    red_all = red["C_all"]
    new = new_lines_mask(nd["C_all"], red_all, res_m)

    img = w["shade"].astype("float32")[:shape[0], :shape[1]]
    ndv = w["ndvi"].astype("float32")
    ndv = detrend(ndv, np.isfinite(ndv), res_m)[:shape[0], :shape[1]]
    base_r = _stretch(np.where(np.isfinite(img), img, np.nan))
    base_n = _stretch(np.where(np.isfinite(ndv), ndv, np.nan))

    n_cell = int(WIN_KM * 1000 / res_m)
    half = n_cell // 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    wins = pick_windows(new, n_cell, MAX_WINDOWS, COVER_TARGET)
    print(f"ячеек новых линий NDVI {int(new.sum())}; окон {len(wins)} "
          f"({time.time() - t0:.0f} с)", flush=True)

    from pyproj import Transformer
    tr = Transformer.from_crs(f"EPSG:{int(w['epsg'])}", "EPSG:4326", always_xy=True)
    x0, y_top = float(w["x0"]), float(w["y_top"])
    k3 = np.ones((3, 3), bool)
    px = 1000.0 / res_m

    for i, (r, c, n) in enumerate(wins, 1):
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[red_all[sl]] = RED
        ov[new[sl]] = NEW
        fig, ax = plt.subplots(2, 2, figsize=(19, 19.6), facecolor="#fcfcfb")
        panels = ((0, 0, base_r, "обычный зимний снимок — смотрите сами", False),
                  (0, 1, base_n, "NDVI — смотрите сами", False),
                  (1, 0, base_r, "обычный снимок с разметкой", True),
                  (1, 1, base_n, "NDVI с разметкой", True))
        for rr, cc, b, title, marked in panels:
            axx = ax[rr, cc]
            axx.imshow(b[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if marked:
                axx.imshow(ov)
            axx.plot([10, 10 + px], [n_cell - 10] * 2, color="#111", lw=3)
            axx.text(10, n_cell - 16, "1 км", color="#111", fontsize=12)
            axx.set_title(title, loc="left", fontsize=15)
            axx.set_xticks([]); axx.set_yticks([])
        lon, lat = tr.transform(x0 + c * res_m, y_top - r * res_m)
        fig.suptitle(f"Окно {i} из {len(wins)} · {WIN_KM:.0f} × {WIN_KM:.0f} км · "
                     f"{lat:.3f}° с.ш., {lon:.3f}° в.д. · север сверху   |   "
                     f"синие — новые линии NDVI, красные — линии красного канала",
                     x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = OUT_DIR / f"{i:02d}_окно_{r}_{c}.png"
        fig.savefig(p, dpi=80, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  {p.name}: ячеек новых линий {n}", flush=True)

    fig, ax = plt.subplots(figsize=(13, 12), facecolor="#fcfcfb")
    ax.imshow(base_r, cmap="gray", vmin=0, vmax=1)
    ov = np.zeros(shape + (4,), "float32")
    ov[binary_dilation(red_all, np.ones((3, 3), bool))] = (0.85, 0.05, 0.05, 0.6)
    ov[binary_dilation(new, np.ones((5, 5), bool))] = NEW
    ax.imshow(ov)
    for i, (r, c, _n) in enumerate(wins, 1):
        ax.add_patch(plt.Rectangle((c - half, r - half), n_cell, n_cell,
                                   fill=False, ec="#1d7a3c", lw=1.6))
        ax.text(c - half, r - half - 25, str(i), color="#1d7a3c", fontsize=13,
                weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Весь лист: синие — новые линии NDVI, красные — красный канал; "
                 "зелёные рамки — окна", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "00_обзор.png", dpi=100, facecolor=fig.get_facecolor())
    print(f"\nготово: {OUT_DIR} ({time.time() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
