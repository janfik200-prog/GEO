"""Окна-диагностика CAND3: на каком шаге пропадает линия (картинка, без нового вывода; пул не меняется).

Слои по шагам тракта (каналы снимка — на снимке, рельефа — на отмывке):
  серый   — ось черты (nms) НИЖЕ порога: z между 75 % и нижним уровнем 90 % (уровень 75 % — только для показа);
  оранж.  — прошло порог (гистерезис 97/90 %), но кусок короче 200 м — выброшен по длине;
  фиолет. — линия ≥ 200 м, снята как дубль другого канала (правило 127 м / 50 %);
  красный — пул CAND3.
    python -m experiments.openlin_cand3_diag --site anabar     (или utsjoki)
"""
from __future__ import annotations

import sys

import numpy as np

from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_iso import Q_HIGH, Q_LOW

SHOW_Q = 0.75                      # только для показа «ниже порога», не порог тракта


def layers(site, log):
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches, nms, rasterize
    from experiments.openlin_iso import hysteresis8
    from experiments.openlin_linb5 import length_px_m, stitch20
    img, dem, valid, _meta = C1.load_inputs(site)
    pool_z = np.load(OUT / f"cand3_{site}.npz")
    out = {}
    for ch in range(5):
        r, th = C1.response(ch, img, dem)
        z = C2.local_norm_nz(r, valid)
        v = z[valid]
        hi, lo, sq = (float(np.quantile(v, q)) for q in (Q_HIGH, Q_LOW, SHOW_Q))
        axis = nms(z, th) & valid
        thin = np.where(axis, z, float(v.min()))
        keep = hysteresis8(thin, lo, hi) & valid
        sk = skeletonize(keep)
        lines = [pl for pl in stitch20(branches(sk), lambda *a: None) if length_px_m(pl) >= C1.LINK_MIN_LEN_M]
        full = rasterize(lines, valid.shape, 1) > 0
        pool = pool_z[f"ids_{ch + 1}"][:valid.shape[0], :valid.shape[1]] > 0
        out[ch] = {"weak": axis & (z >= sq) & (z < lo) & ~keep, "short": sk & ~full,
                   "dup": full & ~pool, "pool": pool}
        log(f"  канал {ch + 1} {C1.CHANNELS[ch]}: ниже порога {out[ch]['weak'].sum() * RES_IN / 1000:.0f} км, "
            f"куски < 200 м {out[ch]['short'].sum() * RES_IN / 1000:.0f} км, дубли {out[ch]['dup'].sum() * RES_IN / 1000:.0f} км, "
            f"пул {pool.sum() * RES_IN / 1000:.0f} км")
    return out, img, dem


def windows(site, L, img, dem, log, win_km):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dst = OUT / f"cand3_{site}_diag"
    dst.mkdir(exist_ok=True)
    gy, gx = np.gradient(dem, RES_IN)
    az = np.radians(315)
    shade = np.clip(0.6 + 0.4 * (-(gx * np.sin(az) - gy * np.cos(az)) / np.maximum(np.hypot(gx, gy), 1e-3)
                                 * np.tanh(np.hypot(gx, gy) * 10)), 0, 1)
    style = (("weak", "#bbbbbb", 0.9, "ниже порога"), ("short", "orange", 1.3, "короче 200 м"),
             ("dup", "magenta", 1.3, "снят как дубль"), ("pool", "red", 1.5, "пул CAND3"))
    H, Wd = img.shape
    wp = int(win_km * 1000 / RES_IN)
    k = 0
    for i in range(3):
        for j in range(3):
            k += 1
            r0 = int((i + 0.5) * H / 3 - wp / 2)
            c0 = int((j + 0.5) * Wd / 3 - wp / 2)
            sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
            fig, axs = plt.subplots(2, 2, figsize=(18, 18))
            panels = (("зимний S2, без линий", img, ()), ("S2: каналы снимка по шагам", img, (0, 1, 2)),
                      ("рельеф GLO-30, без линий", shade, ()), ("рельеф: каналы рельефа по шагам", shade, (3, 4)))
            for ax, (title, a, chs) in zip(axs.ravel(), panels):
                b = a[sl]
                lo_, hi_ = np.nanpercentile(b, [2, 98])
                ax.imshow(b, cmap="gray", vmin=lo_, vmax=hi_, extent=(0, win_km, win_km, 0))
                ax.set_title(title, fontsize=13)
                for key, col, ms, lab in style:
                    if not chs:
                        continue
                    m = np.zeros((wp, wp), bool)
                    for ch in chs:
                        m |= L[ch][key][sl]
                    rr, cc = np.nonzero(m)
                    ax.plot(cc * RES_IN / 1000, rr * RES_IN / 1000, ".", ms=ms, color=col, label=lab, lw=0)
                if chs:
                    ax.legend(loc="lower right", fontsize=11, markerscale=8, framealpha=0.8)
                ax.set_xlim(0, win_km)
                ax.set_ylim(win_km, 0)
            fig.suptitle(f"CAND3 {site}, диагностика, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}. "
                         f"Серый — ось черты ниже порога (z между 75 % и 90 %)", fontsize=14)
            fig.tight_layout()
            p = dst / f"окно_{k}.png"
            fig.savefig(p, dpi=75)
            plt.close(fig)
            log(f"  {p}")


def main():
    from experiments.openlin_fam import log_to
    site = sys.argv[sys.argv.index("--site") + 1]
    log = log_to(OUT / f"_cand3_{site}_diag.log")
    log(f"CAND3 диагностика {site} (картинка, без вывода)")
    L, img, dem = layers(site, log)
    windows(site, L, img, dem, log, 8 if site == "anabar" else 6)


if __name__ == "__main__":
    main()
