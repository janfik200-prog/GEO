"""LINB2 «как есть» на Анабаре (лист R-48-XI,XII) — карта линий и окна для глаза.

Описательно, без нового вывода и без сверки с разломами: замороженный build (frozen_fam3.pkl),
отбор по рангу среднего P до 47,0 % км LINB1 (правило переноса LEN1; на Лапландии порог длины
не подтверждён — длина здесь только раскраска: < 620 м жёлтым, ≥ 620 м красным).
Входы — только зимний композит S2 и GLO-30 (разрешённые). Геология ГИС Интегро не используется.
    python -m experiments.openlin_anabar_linb2
"""
from __future__ import annotations

import numpy as np

from experiments.openlin_len1 import BUDGET_SHARE, sha
from experiments.openlin_real1 import lines_from_pts

SITE = "anabar"
L_MARK_M = 620                   # LEN1 Утсйоки — только раскраска
WIN_KM = 8
N_WIN = (3, 3)                   # окна по сетке листа 3×3 (без выбора по линиям)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments import openlin_anfisa_lines as A
    from experiments.openlin_fam import DEM, OUT, log_to
    from experiments.openlin_transfer_check import COMP
    out = OUT / "anabar_linb2"
    out.mkdir(exist_ok=True)
    log = log_to(OUT / "_anabar_linb2.log")
    log("LINB2 на Анабаре как есть (описательно); входы: зимний S2 + GLO-30")
    log(f"  sha256: openlin_anfisa_lines {sha(A.__file__)}, openlin_fam {sha('experiments/openlin_fam.py')}")
    w = dict(np.load(COMP / f"winter_{SITE}_20m_лист.npz"))
    img = w["shade"].astype("float32")
    shape = img.shape
    res = float(w["res_m"])
    lines, P, _D, _th, _valid, _meta = A.build(SITE, log)
    valid20 = np.isfinite(img)
    k = int(round(A.CELL_M / A.RES_IN))
    ids_all = np.where(valid20, A.rasterize(lines, shape, k), 0)
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    mp = np.array([A.line_mean_p(pl, P) for pl in lines])
    lenm = np.array([A.length_m(pl) for pl in lines])
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    total = cells.sum() * res / 1000
    cum = np.cumsum(cells[order]) * res / 1000
    n = int(np.searchsorted(cum, BUDGET_SHARE * total, side="right"))
    keep = np.zeros(len(lines) + 1, bool)
    keep[order[:n] + 1] = True
    ids = np.where(keep[ids_all], ids_all, 0)
    np.savez_compressed(out / "linb2_anabar.npz", ids=ids, res_m=res, x0=w["x0"], y_top=w["y_top"], epsg=w["epsg"])
    rr, cc = np.nonzero(ids)
    L = [x for x in lines_from_pts(np.stack([rr, cc], 1), ids[rr, cc]) if len(x) >= 2]
    lens = np.array([len(x) for x in L]) * res
    area = valid20.sum() * res * res / 1e6
    log(f"  LINB1: линий {len(lines)}, {total:.0f} км; LINB2 (47,0 %): линий {len(L)}, {lens.sum() / 1000:.0f} км, "
        f"порог P {mp[order[n - 1]]:.3f}; плотность {lens.sum() / 1000 / area:.2f} км/км²; "
        f"< {L_MARK_M} м: {100 * np.mean(lens < L_MARK_M):.0f} % линий")
    dem = np.load(DEM / f"dem_glo30_{SITE}_лист_20m.npy").astype("float32")[:shape[0], :shape[1]]
    d = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    gy, gx = np.gradient(d, res)
    az = np.radians(315)
    shade = np.clip(0.6 + 0.4 * (-(gx * np.sin(az) - gy * np.cos(az)) / np.maximum(np.hypot(gx, gy), 1e-3)
                                 * np.tanh(np.hypot(gx, gy) * 10)), 0, 1)

    def show(ax, a, sl, title):
        b = a[sl]
        lo, hi = np.nanpercentile(b, [2, 98])
        ax.imshow(b, cmap="gray", vmin=lo, vmax=hi,
                  extent=(0, b.shape[1] * res / 1000, b.shape[0] * res / 1000, 0))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("км")

    def draw(ax, r0, c0, h, wd, ms):
        for x, ln in zip(L, lens):
            y = x - [r0, c0]
            m = (y[:, 0] >= 0) & (y[:, 0] < h) & (y[:, 1] >= 0) & (y[:, 1] < wd)
            if m.any():
                y = y[m] * res / 1000
                ax.plot(y[:, 1], y[:, 0], ".", ms=ms, color="red" if ln >= L_MARK_M else "gold")

    # обзор листа
    fig, axs = plt.subplots(1, 2, figsize=(22, 11))
    full = (slice(None), slice(None))
    show(axs[0], img, full, "Анабар, зимний S2 (тени), без линий")
    show(axs[1], img, full, f"LINB2: {len(L)} линий, {lens.sum() / 1000:.0f} км (красные ≥ {L_MARK_M} м, жёлтые короче)")
    draw(axs[1], 0, 0, *shape, 0.3)
    H, Wd = shape
    wp = int(WIN_KM * 1000 / res)
    wins = []
    for i in range(N_WIN[0]):
        for j in range(N_WIN[1]):
            r0 = int((i + 0.5) * H / N_WIN[0] - wp / 2)
            c0 = int((j + 0.5) * Wd / N_WIN[1] - wp / 2)
            wins.append((r0, c0))
            for ax in axs:
                ax.add_patch(plt.Rectangle((c0 * res / 1000, r0 * res / 1000), WIN_KM, WIN_KM,
                                           fill=False, ec="cyan", lw=1.2))
                ax.text(c0 * res / 1000 + 0.3, r0 * res / 1000 + 1.5, str(len(wins)), color="cyan", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "обзор.png", dpi=90)
    plt.close(fig)
    log(f"  {out / 'обзор.png'}")
    for nwin, (r0, c0) in enumerate(wins, 1):
        sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        show(axs[0, 0], img, sl, "зимний S2, без линий")
        show(axs[0, 1], img, sl, "зимний S2 + LINB2")
        show(axs[1, 0], shade, sl, "рельеф GLO-30 (отмывка), без линий")
        show(axs[1, 1], shade, sl, "рельеф GLO-30 + LINB2")
        for ax in (axs[0, 1], axs[1, 1]):
            draw(ax, r0, c0, wp, wp, 2.0)
        nn = sum(((x[:, 0] >= r0) & (x[:, 0] < r0 + wp) & (x[:, 1] >= c0) & (x[:, 1] < c0 + wp)).any() for x in L)
        fig.suptitle(f"Окно {nwin} ({WIN_KM}×{WIN_KM} км), строка {r0}, столбец {c0}; линий в окне {nn}. "
                     f"Красные ≥ {L_MARK_M} м, жёлтые короче", fontsize=14)
        fig.tight_layout()
        p = out / f"окно_{nwin}.png"
        fig.savefig(p, dpi=75)
        plt.close(fig)
        log(f"  {p}")


if __name__ == "__main__":
    main()
