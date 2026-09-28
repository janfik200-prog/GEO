"""Рисунок для доклада: снимок и его двойник рядом, с линиями на обоих.

Двойник (:func:`openlin_kora.surrogate`) сохраняет гистограмму, зернистость и
связь каналов, но разрушает расположение. Показывается одно окно 5 x 5 км:
слева зимний композит, справа его двойник, и под каждым — сколько линий длиннее
500 м нашёл на нём детектор. Двойник строится на вырезке 20 x 20 км вокруг
окна (все три канала: композит и обе половины, как в проверках), чтобы не
гонять разложение по всему листу.

Запуск из корня репозитория:
    python -m experiments.openlin_twin_figure --site anabar
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.openlin_zone_lines import COMP, OUT, mask_of, trace

CENTER = (813, 3534)         # окно 1 из папки проверки глазом
CUT = 1000                   # вырезка 20 x 20 км для двойника
WIN = 250                    # показываемое окно 5 x 5 км
SEED = 900                   # то же зерно, что у первой подделки в проверках


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_kora import surrogate
    from experiments.openlin_shade import _stretch
    from experiments.openlin_zone_lines_v3 import build_halves

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    halves = build_halves(a.site, lambda *_: None)
    r0 = max(CENTER[0] - CUT // 2, 0)
    c0 = max(CENTER[1] - CUT // 2, 0)
    sl = (slice(r0, r0 + CUT), slice(c0, c0 + CUT))
    stack = np.stack([w["shade"][sl], halves["А"][sl], halves["Б"][sl]]
                     ).astype("float32")
    valid = np.all(np.isfinite(stack), axis=0)
    twin = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(SEED))

    real, fake = stack[0], np.where(valid, twin[0], np.nan)
    Lr = trace(np.where(valid, real, np.nan), valid, res_m, local=True)
    Lf = trace(fake.astype("float32"), valid, res_m, local=True)
    mr, mf = mask_of(Lr, valid.shape), mask_of(Lf, valid.shape)

    cr, cc = CENTER[0] - r0, CENTER[1] - c0
    ws = (slice(cr - WIN // 2, cr + WIN // 2), slice(cc - WIN // 2, cc + WIN // 2))
    lo, hi = np.nanpercentile(real[valid], [2, 98])
    k3 = np.ones((3, 3), bool)

    fig, ax = plt.subplots(2, 2, figsize=(16, 16.6), facecolor="#ffffff")
    for j, (img, m, name, n) in enumerate((
            (real, mr, "снимок", len(Lr)), (fake, mf, "двойник", len(Lf)))):
        g = np.clip((img[ws] - lo) / (hi - lo), 0, 1)
        for i in (0, 1):
            ax[i, j].imshow(g, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ov = np.zeros(g.shape + (4,), "float32")
        ov[binary_dilation(m[ws], k3)] = (0.76, 0.36, 0.2, 0.95)
        ax[1, j].imshow(ov)
        ax[0, j].set_title(name, loc="left", fontsize=22, color="#1B2A32")
        ax[1, j].set_title(f"линии детектора: {n} на вырезке 20 × 20 км",
                           loc="left", fontsize=18, color="#6E7F88")
    fig.tight_layout()
    p = OUT / f"снимок_и_двойник_{a.site}.png"
    fig.savefig(p, dpi=80, facecolor=fig.get_facecolor())
    print(f"сохранено: {p}; линий на снимке {len(Lr)}, на двойнике {len(Lf)}")


if __name__ == "__main__":
    main()
