"""Линеамент как цепочка выстроенных уступов, а не как сплошная линия.

ОТКУДА ПОСТАНОВКА. Оператор уступов (:mod:`openlin_shade`) после подбора
геометрии находит объекты длиной около километра — это и есть его окно
накопления. Требовать от него сплошную линию в десятки километров
бессмысленно: при окне в 4 км и больше не выживает ничего.

Но линеамент и не обязан быть сплошным. В определении O'Leary и др. (1976) это
«составной линейный элемент, ЧАСТИ КОТОРОГО ВЫСТРОЕНЫ в прямую или слабо
изогнутую линию»: спрямлённый участок долины, уступ, изгиб другой долины — и
все на одной прямой. Поэтому цепочки собираются из километровых элементов,
каждый из которых уже прошёл фазовую подделку.

ДВА НУЛЯ. Цепочка — утверждение о СОВПАДЕНИИ положения и направления, поэтому
нулей нужно два, и они разные:

* элементы с фазовой подделки: там нет ни одного настоящего уступа, а значит
  все собранные из них цепочки — ложные;
* перемешивание азимутов при сохранённых положениях: плотность элементов, их
  расположение и набор направлений остаются теми же, рушится ровно связь «где
  элемент» с «куда он смотрит». Без этого нуля длинная полоса через весь лист
  набирает «цепочку» просто по плотности.

Запуск из корня репозитория:
    python -m experiments.openlin_shade_chains --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_align import chains, elements
from experiments.openlin_shade import fdr_curve, lines_from_field, load_winter

M = Path("outputs/metrics")

#: Порог оценки и минимальная длина элемента, м: взяты по кривой доли ложных
#: (14 % на уровне ячеек, 19 % на уровне элементов — 227 против 44).
THR = 3.5
MIN_LEN_M = 500.0

#: Сколько раз перемешиваются азимуты для нуля выстроенности.
N_SHUFFLE = 20


def report(name: str, cent, az, ln, res_m, area_km2, log) -> list:
    t0 = time.time()
    ch = chains(cent, az, ln, res_m, area_km2)
    log(f"  {name:22s} элементов {len(cent):5d}, цепочек {len(ch):4d} "
        f"({time.time() - t0:.0f} с)")
    return ch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--thr", type=float, default=THR)
    ap.add_argument("--min-len", type=float, default=MIN_LEN_M)
    ap.add_argument("--shuffles", type=int, default=N_SHUFFLE)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    z = dict(np.load(M / f"openlin_shade_{a.site}.npz"))
    zf = dict(np.load(M / f"openlin_shade_{a.site}_подделка.npz"))
    res_m = float(z["res_m"])
    sc, scf = z["оценка"], zf["оценка"]
    valid = np.isfinite(sc)
    area_km2 = float(valid.sum()) * (res_m / 1000.0) ** 2

    tab = fdr_curve(sc, scf)
    junk = float(np.interp(a.thr, tab["порог"], tab["доля мусора, %"]))
    log(f"площадка {a.site}: порог {a.thr:.1f} (мусора на уровне ячеек "
        f"{junk:.0f} %), элементы от {a.min_len:.0f} м, площадь "
        f"{area_km2:.0f} км²")

    mask, df = lines_from_field(sc, res_m, a.thr, a.min_len)
    mask_f, df_f = lines_from_field(scf, res_m, a.thr, a.min_len)
    cent, az, ln = elements(mask, res_m)
    cent_f, az_f, ln_f = elements(mask_f, res_m)

    log("\nцепочки:")
    ch = report("настоящий вход", cent, az, ln, res_m, area_km2, log)
    ch_f = report("подделка", cent_f, az_f, ln_f, res_m, area_km2, log)

    rng = np.random.default_rng(7)
    n_sh = []
    for i in range(a.shuffles):
        az_s = rng.permutation(az)
        n_sh.append(len(chains(cent, az_s, ln, res_m, area_km2)))
    n_sh = np.asarray(n_sh)
    log(f"  {'перемешанные азимуты':22s} цепочек {n_sh.mean():.1f} "
        f"в среднем (разброс {n_sh.min()}-{n_sh.max()}, {a.shuffles} повторов)")

    n = len(ch)
    p = (1 + int((n_sh >= n).sum())) / (1 + a.shuffles)
    log(f"\nцепочек на настоящем входе {n}, на подделке {len(ch_f)}, "
        f"при перемешанных азимутах {n_sh.mean():.1f}")
    log(f"p по нулю выстроенности: {p:.3f} "
        f"(минимально достижимое {1 / (1 + a.shuffles):.3f})")

    if not ch:
        log("цепочек нет — собирать нечего")
        return

    import pandas as pd
    tabc = pd.DataFrame([{k: v for k, v in c.items() if k != "индексы"}
                         for c in ch])
    tabc.to_csv(M / f"openlin_shade_chains_{a.site}.csv", index=False,
                encoding="utf-8-sig")
    sp = tabc["протяжённость_км"]
    log(f"протяжённость цепочек: медиана {sp.median():.0f} км, максимум "
        f"{sp.max():.0f} км; элементов в цепочке медиана "
        f"{tabc['элементов'].median():.0f}")
    log(f"сохранено: {M / f'openlin_shade_chains_{a.site}.csv'}")

    # карта: элементы и собранные из них цепочки поверх снимка
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    img = load_winter(a.site, res_m)["shade"].astype("float32")
    base = _stretch(img)
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.6))
    wide = binary_dilation(mask, np.ones((3, 3), dtype=bool))
    for a_ in ax:
        a_.imshow(base, cmap="gray", vmin=0, vmax=1)
        ov = np.zeros(wide.shape + (4,), dtype="float32")
        ov[wide] = (1.0, 0.25, 0.1, 1.0)
        a_.imshow(ov)
        a_.set_xticks([]); a_.set_yticks([])
    ax[0].set_title(f"элементы: {len(df)} шт., "
                    f"{df['длина_км'].sum():.0f} км", fontsize=13)
    for c in ch:
        idx = c["индексы"]
        u = np.radians(c["азимут"])
        d = np.array([np.sin(u), -np.cos(u)])
        pts = cent[idx]
        t = (pts - pts.mean(0)) @ d
        p0, p1 = pts.mean(0) + d * t.min(), pts.mean(0) + d * t.max()
        ax[1].plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color="#2ec4ff",
                   lw=1.6, alpha=0.9)
    ax[1].set_title(f"цепочки: {len(ch)} шт., медиана "
                    f"{sp.median():.0f} км", fontsize=13)
    fig.tight_layout()
    out = Path("outputs/steps") / f"уступы_цепочки_{a.site}.png"
    fig.savefig(out, dpi=110)
    log(f"сохранено: {out}")


if __name__ == "__main__":
    main()
