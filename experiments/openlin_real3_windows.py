"""Окна для глаза: короткие линии LINB2 и их вердикт по мере REAL3 (S2 лето + Landsat лето).

Описательная картинка, без нового вывода. Окна 4×4 км, где больше всего коротких (нижняя
терциль длины) неподтверждённых линий. Цвет линии: красный — w = 0 (обе соседние копии сильнее),
жёлтый — 0,5, зелёный — 1; длинные линии — тонкие серые.
    python -m experiments.openlin_real3_windows
"""
from __future__ import annotations

import numpy as np

from experiments.openlin_real1 import RES, geometry, load_sets, witnesses
from experiments.openlin_real2 import MAIN_PX, prep_witness, wins
from experiments.openlin_real3 import admissible, t_stats

WIN_PX = int(4000 / RES)
N_WIN = 6


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_fam import OUT
    from experiments.openlin_real1 import M
    out = OUT / "real3_windows"
    out.mkdir(exist_ok=True)
    shape = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
    lines = load_sets(shape)["Анфиса LINB2"]
    W = witnesses(shape)
    P, idx, T, N = geometry(lines)
    nl = len(lines)
    ok = admissible(P, idx, N, nl, shape)
    allst = None
    for wn in ("S2 лето", "Landsat лето"):
        hs, valid = prep_witness(W[wn], shape)
        st = t_stats(P, idx, T, N, nl, hs, valid, shape, pairs=(MAIN_PX,))
        allst = st if allst is None else np.fmax(allst, st)
    fin = ok & np.all(np.isfinite(allst), 1)
    w = np.full(nl, np.nan)
    w[fin] = wins(allst[fin], MAIN_PX)
    lens = np.array([len(x) for x in lines]) * RES / 1000
    short = lens <= np.quantile(lens[fin], 1 / 3)
    cen = np.array([x.mean(0) for x in lines])
    print(f"LINB2: линий {nl}, в счёт {int(fin.sum())}; короткие ≤ {np.quantile(lens[fin], 1 / 3):.2f} км; "
          f"у коротких w=0 {np.mean(w[fin & short] == 0):.0%}, w=1 {np.mean(w[fin & short] == 1):.0%}")
    # окна с наибольшим числом коротких w = 0, без перекрытия
    bad = fin & short & (w == 0)
    H, Wd = shape
    grid = np.zeros((H // (WIN_PX // 2) + 1, Wd // (WIN_PX // 2) + 1))
    for c in cen[bad]:
        grid[int(c[0]) // (WIN_PX // 2), int(c[1]) // (WIN_PX // 2)] += 1
    wins_xy = []
    for flat in np.argsort(grid, axis=None)[::-1]:
        gi, gj = np.unravel_index(flat, grid.shape)
        r0 = int(np.clip(gi * (WIN_PX // 2) - WIN_PX // 4, 0, H - WIN_PX))
        c0 = int(np.clip(gj * (WIN_PX // 2) - WIN_PX // 4, 0, Wd - WIN_PX))
        if all(abs(r0 - a) >= WIN_PX or abs(c0 - b) >= WIN_PX for a, b in wins_xy):
            wins_xy.append((r0, c0))
        if len(wins_xy) == N_WIN:
            break
    g = W["GLO-30"]["высота"]
    gy, gx = np.gradient(np.where(np.isfinite(g), g, np.nanmedian(g)), RES)
    shade = np.clip(0.7 - 0.7 * (gx * np.cos(np.radians(315)) - gy * np.sin(np.radians(315))) * 5, 0, 1)
    bases = (("S2 лето, красный", W["S2 лето"]["красный"]), ("Landsat лето, красный", W["Landsat лето"]["красный"]),
             ("рельеф GLO-30, отмывка", shade))
    colors = {0.0: "red", 0.5: "gold", 1.0: "lime"}
    for n, (r0, c0) in enumerate(wins_xy, 1):
        fig, axs = plt.subplots(1, 4, figsize=(24, 6.4))
        sl = (slice(r0, r0 + WIN_PX), slice(c0, c0 + WIN_PX))
        for ax, (title, f) in zip(axs, bases + (("без линий: S2 лето", W["S2 лето"]["красный"]),)):
            a = f[sl]
            lo, hi = np.nanpercentile(a, [2, 98])
            ax.imshow(a, cmap="gray", vmin=lo, vmax=hi, extent=(0, WIN_PX * RES / 1000, WIN_PX * RES / 1000, 0))
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("км")
            if title.startswith("без линий"):
                continue
            for i, x in enumerate(lines):
                y = x - [r0, c0]
                m = (y[:, 0] >= 0) & (y[:, 0] < WIN_PX) & (y[:, 1] >= 0) & (y[:, 1] < WIN_PX)
                if not m.any():
                    continue
                y = y[m] * RES / 1000
                if short[i] and np.isfinite(w[i]):
                    ax.plot(y[:, 1], y[:, 0], ".", ms=2.5, color=colors[float(w[i])])
                else:
                    ax.plot(y[:, 1], y[:, 0], ".", ms=0.8, color="deepskyblue", alpha=0.6)
        k = [i for i in range(nl) if short[i] and np.isfinite(w[i])
             and r0 <= cen[i, 0] < r0 + WIN_PX and c0 <= cen[i, 1] < c0 + WIN_PX]
        fig.suptitle(f"Окно {n}: строка {r0}, столбец {c0} (сетка 20 м). Короткие линии LINB2: "
                     f"красные {sum(w[i] == 0 for i in k)}, жёлтые {sum(w[i] == 0.5 for i in k)}, "
                     f"зелёные {sum(w[i] == 1 for i in k)}; голубые — длинные", fontsize=13)
        fig.tight_layout()
        p = out / f"окно_{n}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        print(p)


if __name__ == "__main__":
    main()
