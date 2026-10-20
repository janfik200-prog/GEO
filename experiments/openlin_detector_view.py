"""Что видит детектор: снимок, карта отклика, все вытянутые линии, итог.

ЗАЧЕМ. Проверка глазом показала пропуски, и неясно, где линеамент теряется:
детектор его не видит вовсе, видит, но не вытягивает в линию, или вытягивает,
а дальше его отсеивают продолжение, сшивка, длина или подтверждение. Окно
5 x 5 км в четырёх панелях отвечает на это по стадиям:

  1. чистый зимний снимок;
  2. карта отклика — «на что реагирует» детектор ещё до порогов: сила узкой
     тени или кромки в сигмах своей окрестности (светлее — сильнее);
  3. все линии, вытянутые из отклика (утончение + два порога), любой длины —
     до продолжения, сшивки и отбора;
  4. итог: красные — подтверждены двумя съёмками, жёлтые — найдены, но не
     подтверждены.

Окна по умолчанию — контрольные и те, где при просмотре находились пропуски.

Запуск из корня репозитория:
    python -m experiments.openlin_detector_view --site anabar
    python -m experiments.openlin_detector_view --окна "1176,733;813,3534"
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, M, OUT, mask_of, trace

OUT_DIR = OUT.parent / "линеаменты_детектор"
WIN_KM = 5.0
OVERLAY_ALPHA = 0.6          # непрозрачность наложений: снимок под разметкой остаётся виден
#: контрольные окна 1-4 текущего набора, прежние контрольные 1 и 6 (там геолог
#: нашёл пропущенные линеаменты) и два обычных окна для сравнения
DEFAULT_WINDOWS = ((1176, 733), (2977, 206), (1684, 526), (2069, 988),
                   (1088, 767), (1681, 2208), (813, 3534), (959, 2707))


def whole_sheet(img, valid, z, R, A, C) -> None:
    """Весь лист: четыре стадии по отдельности (1 ячейка = 1 пиксель) и обзор 2x2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = np.nan_to_num(_stretch(np.where(valid, img, np.nan)), nan=1.0)
    gray = np.repeat(g[..., None], 3, axis=2).astype("float32")
    zmax = float(np.quantile(z[valid], 0.99))
    resp = cm.magma(np.clip(np.where(valid, z, 0.0), 0, zmax) / zmax)[..., :3]

    def paint(masks_colors, grow):
        out = gray.copy()
        k = np.ones((grow, grow), bool)
        for m, col in masks_colors:
            out[binary_dilation(m, k) if grow > 1 else m] = col
        return out

    stages = {
        "1_снимок": gray,
        "2_отклик_детектора": resp,
        "3_все_вытянутые_линии": paint([(R, (0.15, 0.75, 0.95))], 3),
        "4_итог": paint([(A & ~C, (0.95, 0.60, 0.0)), (C, (0.85, 0.05, 0.05))], 3),
    }
    # наложения: детектор полупрозрачно поверх снимка, чтобы под разметкой был
    # виден сам объект. Отклик — прозрачность растёт с силой (слабый не красит
    # снимок вовсе), линии — ровные полупрозрачные
    al = (OVERLAY_ALPHA * np.clip(np.where(valid, z, 0.0), 0, zmax) / zmax)[..., None]
    stages["наложение_отклик"] = gray * (1 - al) + resp * al
    lm = binary_dilation(R, np.ones((3, 3), bool))[..., None] * OVERLAY_ALPHA
    stages["наложение_линии"] = gray * (1 - lm) + np.array((0.15, 0.75, 0.95)) * lm
    for name, arr in stages.items():
        try:
            plt.imsave(OUT_DIR / f"весь_лист_{name}.png", np.clip(arr, 0, 1))
        except OSError as exc:          # файл открыт в просмотрщике — не перезаписать
            print(f"  весь_лист_{name}.png: НЕ записан ({exc.strerror})", flush=True)
            continue
        print(f"  весь_лист_{name}.png", flush=True)

    # обзор: линии утолщены сильнее, иначе при уменьшении листа они исчезают
    over = {
        "1. снимок": gray,
        "2. на что реагирует детектор (светлее — сильнее)": resp,
        "3. все линии, вытянутые из отклика": paint([(R, (0.15, 0.75, 0.95))], 7),
        "4. итог: красные подтверждены, жёлтые — нет":
            paint([(A & ~C, (0.95, 0.60, 0.0)), (C, (0.85, 0.05, 0.05))], 7),
    }
    fig, ax = plt.subplots(2, 2, figsize=(26, 24), facecolor="#fcfcfb")
    for axx, (title, arr) in zip(ax.ravel(), over.items()):
        axx.imshow(np.clip(arr, 0, 1), interpolation="antialiased")
        axx.set_title(title, loc="left", fontsize=20)
        axx.set_xticks([]); axx.set_yticks([])
    fig.suptitle("Анабар, лист R-48-XI,XII · 77 × 86 км · север сверху",
                 x=0.01, ha="left", fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "весь_лист_обзор.png", dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  весь_лист_обзор.png", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--окна", dest="wins", default=None,
                    help='центры окон в ячейках: "строка,столбец;строка,столбец"')
    ap.add_argument("--весь-лист", dest="sheet", action="store_true",
                    help="весь лист: обзор 2x2 и каждая стадия в полном разрешении")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_iso import Q_HIGH, Q_LOW, local_norm, ridge_response
    from experiments.openlin_shade import _stretch
    from experiments.openlin_shadow import standardize

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    valid = np.isfinite(img)
    from experiments.openlin_zone_lines_v3 import work_npz
    d = np.load(work_npz(a.site))        # рабочий набор — «цепочки»
    A, C = d["A_all"], d["C_all"]
    shape = C.shape
    img, valid = img[:shape[0], :shape[1]], valid[:shape[0], :shape[1]]

    x = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    z = local_norm(np.where(valid, ridge_response(x), 0.0), valid, res_m)
    lo, hi = np.quantile(z[valid], Q_LOW), np.quantile(z[valid], Q_HIGH)
    raw = trace(np.where(valid, img, np.nan).astype("float32"), valid, res_m,
                local=True, min_len_m=0)
    R = mask_of(raw, shape)
    print(f"отклик и трассировка готовы: линий любой длины {len(raw)} "
          f"({time.time() - t0:.0f} с)", flush=True)

    if a.sheet:
        whole_sheet(img, valid, z, R, A, C)
        print(f"готово: {OUT_DIR} ({time.time() - t0:.0f} с)", flush=True)
        return

    wins = (DEFAULT_WINDOWS if a.wins is None else
            tuple(tuple(int(v) for v in p.split(",")) for p in a.wins.split(";")))
    n_cell = int(WIN_KM * 1000 / res_m)
    half = n_cell // 2
    base = _stretch(np.where(valid, img, np.nan))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    px = 1000.0 / res_m

    from pyproj import Transformer
    tr = Transformer.from_crs(f"EPSG:{int(w['epsg'])}", "EPSG:4326", always_xy=True)

    for i, (r, c) in enumerate(wins, 1):
        r = min(max(r, half), shape[0] - half)
        c = min(max(c, half), shape[1] - half)
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        fig, ax = plt.subplots(2, 2, figsize=(19, 19.6), facecolor="#fcfcfb")
        ax[0, 0].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[0, 0].set_title("1. снимок — смотрите сами", loc="left", fontsize=15)
        ax[0, 1].imshow(np.clip(z[sl], 0, None), cmap="magma", vmin=0,
                        vmax=float(np.quantile(z[valid], 0.99)), interpolation="nearest")
        ax[0, 1].set_title("2. на что реагирует детектор (светлее — сильнее)",
                           loc="left", fontsize=15)
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[R[sl]] = (0.15, 0.75, 0.95, 1.0)
        ax[1, 0].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[1, 0].imshow(ov)
        ax[1, 0].set_title("3. все линии, вытянутые из отклика, любой длины",
                           loc="left", fontsize=15)
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[A[sl] & ~C[sl]] = (0.95, 0.60, 0.0, 0.95)
        ov[C[sl]] = (0.85, 0.05, 0.05, 0.95)
        ax[1, 1].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[1, 1].imshow(ov)
        ax[1, 1].set_title("4. итог: красные подтверждены, жёлтые — нет",
                           loc="left", fontsize=15)
        for axx in ax.ravel():
            axx.plot([10, 10 + px], [n_cell - 10] * 2, color="#eee", lw=3)
            axx.text(10, n_cell - 16, "1 км", color="#eee", fontsize=12)
            axx.set_xticks([]); axx.set_yticks([])
        lon, lat = tr.transform(float(w["x0"]) + c * res_m,
                                float(w["y_top"]) - r * res_m)
        fig.suptitle(f"Окно {i} · {WIN_KM:.0f} × {WIN_KM:.0f} км · {lat:.3f}° с.ш., "
                     f"{lon:.3f}° в.д. · север сверху · центр ({r}, {c})",
                     x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = OUT_DIR / f"{i:02d}_окно_{r}_{c}.png"
        fig.savefig(p, dpi=80, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  {p.name}", flush=True)
    print(f"готово: {OUT_DIR} ({time.time() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
