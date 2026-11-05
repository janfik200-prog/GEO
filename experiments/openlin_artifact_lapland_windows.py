"""Проверочный (не тракт): парные окна для глазной проверки артефактов Лапландии.

Слева зимний композит shade без разметки, справа он же с линиями C_all
(красные — вне зон; синие — подтверждённые в зонах, набор C; голубой пунктир —
береговая линия воды WorldCover). Эталон GTK не накладывается. Оформление как в
openlin_review_tiles. Кластеры: берег (в зонах), дороги, ЛЭП, край леса,
«не объяснено». Окно 3.5 км. Папка outputs/линеаменты_проверка_lapland_артефакты/.
Запуск: python -m experiments.openlin_artifact_lapland_windows
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experiments.openlin_artifact_lapland as m
from experiments.openlin_shade import _stretch

OUTD = Path("outputs/линеаменты_проверка_lapland_артефакты")
WIN_KM = 3.5
PLAN = [("берег", 4), ("дороги", 4), ("ЛЭП", 3), ("край леса", 3), ("не объяснено", 3)]
MIN_SEP_KM = 6.0


def layer(mask, ang):
    nr, na = m.make_near(mask, ang)
    return {"near": nr, "nang": na}


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    for f in OUTD.glob("*.png"):
        f.unlink()
    z = np.load(m.NPZ); w = np.load(m.COMP)
    x0, yt, epsg = float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    C, Cz = z["C_all"], z["C"]
    shape = C.shape; H, W = shape
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    wc = np.load(m.WC)["wc"]
    layers = {}
    for n, key in (("roads", "дороги"), ("power", "ЛЭП")):
        L = m.osm_layer(n, shape, x0, yt, epsg, tr)
        layers[key] = layer(L["mask"], L["ang"])
    water = wc == 80
    shore = water & ~binary_erosion(water, np.ones((3, 3), bool))
    g = gaussian_filter(water.astype("float32"), 2.0); gr, gc = np.gradient(g)
    layers["берег"] = layer(shore, np.where(shore, np.round(m.az_of(-gc, gr)).astype(np.int16) % 180, -1).astype(np.int16))
    land = ~water; tree = wc == 10
    edge = (tree & binary_dilation(land & ~tree, np.ones((3, 3), bool))) | \
           (land & ~tree & binary_dilation(tree, np.ones((3, 3), bool)))
    g = gaussian_filter(np.where(land, tree, 0.5).astype("float32"), 2.0); gr, gc = np.gradient(g)
    layers["край леса"] = layer(edge, np.where(edge, np.round(m.az_of(-gc, gr)).astype(np.int16) % 180, -1).astype(np.int16))
    cells = m.Cells(C)
    _, fl = m.stats(cells, shape, layers)
    inz = np.bincount(cells.comp, Cz[cells.r, cells.c], minlength=cells.n) / cells.cnt >= 0.5
    anyf = fl["берег"] | fl["дороги"] | fl["ЛЭП"] | fl["край леса"]
    cand = {
        "берег": fl["берег"] & inz,
        "дороги": fl["дороги"] & ~fl["берег"],
        "ЛЭП": fl["ЛЭП"] & ~fl["берег"],
        "край леса": fl["край леса"] & ~fl["берег"] & ~fl["дороги"] & ~fl["ЛЭП"],
        "не объяснено": ~anyf,
    }
    img = _stretch(np.where(np.isfinite(w["shade"]), w["shade"], np.nan))[:H, :W]
    n = int(WIN_KM * 1000 / m.RES); half = n // 2
    cen = np.zeros((cells.n, 2))
    for i in range(cells.n):
        pass
    cr = np.bincount(cells.comp, cells.r, minlength=cells.n) / cells.cnt
    cc = np.bincount(cells.comp, cells.c, minlength=cells.n) / cells.cnt
    sep = MIN_SEP_KM * 1000 / m.RES
    items = []
    for kind, k in PLAN:
        idx = np.where(cand[kind])[0]
        idx = idx[np.argsort(-cells.cnt[idx])]
        taken = []
        for i in idx:
            if not (half <= cr[i] < H - half and half <= cc[i] < W - half):
                continue
            if any(np.hypot(cr[i] - a, cc[i] - b) < sep for a, b in taken):
                continue
            taken.append((cr[i], cc[i])); items.append((kind, int(i)))
            if len(taken) == k:
                break
    thumbs = []
    for j, (kind, i) in enumerate(items, 1):
        r, c = int(cr[i]), int(cc[i])
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        base = img[sl]
        ov = np.zeros(base.shape + (4,), "float32")
        ov[C[sl]] = (0.85, 0.05, 0.05, 0.95)
        ov[Cz[sl]] = (0.10, 0.30, 0.95, 0.95)
        sh = shore[sl]
        ov[sh & ~C[sl]] = (0.0, 0.8, 0.8, 0.35)
        mine = (cells.comp == i)
        E, N = x0 + c * m.RES, yt - r * m.RES
        title = (f"{kind}: линия {cells.cnt[i] * m.RES / 1000:.1f} км, азимут {cells.az[i]:.0f}°, "
                 f"центр E {E:.0f} N {N:.0f} (UTM 35N), окно {WIN_KM} км; "
                 f"{'в зоне' if inz[i] else 'вне зоны'}")
        fig, ax = plt.subplots(1, 2, figsize=(17, 8.6), facecolor="#fcfcfb")
        for k in (0, 1):
            ax[k].imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[k].set_xticks([]); ax[k].set_yticks([])
            px = 1000.0 / m.RES
            ax[k].plot([10, 10 + px], [base.shape[0] - 10] * 2, color="#111", lw=3)
            ax[k].text(10, base.shape[0] - 16, "1 км", color="#111", fontsize=12)
        ax[1].imshow(ov)
        # отмеченная линия: контур-подсветка
        rr = cells.r[mine] - (r - half); cc2 = cells.c[mine] - (c - half)
        okp = (rr >= 0) & (rr < n) & (cc2 >= 0) & (cc2 < n)
        ax[1].scatter(cc2[okp], rr[okp], s=2, facecolors="none", edgecolors="#ffd400", linewidths=0.3)
        ax[0].set_title("без разметки — смотрите сами", loc="left", fontsize=13)
        ax[1].set_title("красные — C_all вне зон, синие — в зонах (C), "
                        "голубая заливка — берег воды; жёлтый ореол — линия из подписи",
                        loc="left", fontsize=10)
        fig.suptitle(f"{j:02d}. {title}", x=0.01, ha="left", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = OUTD / f"{j:02d}_{kind.replace(' ', '_')}_E{E:.0f}_N{N:.0f}.png"
        fig.savefig(p, dpi=90, facecolor=fig.get_facecolor()); plt.close(fig)
        thumbs.append((j, kind, title, base, ov))
    # сводный лист
    cols = 4; rows = int(np.ceil(len(thumbs) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 5.4 * rows), facecolor="#fcfcfb")
    ax = np.atleast_2d(ax)
    for a in ax.ravel():
        a.axis("off")
    for (j, kind, title, base, ov), a in zip(thumbs, ax.ravel()):
        a.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); a.imshow(ov)
        a.set_title(f"{j:02d} {kind}\n" + title.split(": ", 1)[1].split(", окно")[0], fontsize=7.5, loc="left")
    fig.suptitle("Лапландия: окна по кластерам артефактов (красные — C_all, синие — в зонах)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTD / "00_сводный_лист.png", dpi=90, facecolor=fig.get_facecolor())
    print(len(thumbs), "окон +сводный лист")
    for j, kind, title, *_ in thumbs:
        print(j, title)


if __name__ == "__main__":
    main()
