"""Карта уступов: линии выше порога, откалиброванного по фазовой подделке.

Порог не подбирается на глаз. Для каждого значения оценки известно, сколько
ячеек даёт настоящий вход и сколько — подделка, в которой протяжённых объектов
нет по построению. Их отношение и есть доля ложных среди оставленных, поэтому
«столько-то линий при десяти процентах мусора» — проверяемое утверждение.

Запуск из корня репозитория:
    python -m experiments.openlin_shade_map --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_shade import (fdr_curve, lines_from_field,
                                       load_winter, _stretch)

M = Path("outputs/metrics")

#: Пороги для карты: рабочий и строгий, оба взяты по измеренной доле мусора.
THRESHOLDS = (3.5, 4.5, 6.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--min-len", type=float, default=1000.0)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def log(m=""):
        print(m, flush=True)

    z = dict(np.load(M / f"openlin_shade_{a.site}.npz"))
    zf = dict(np.load(M / f"openlin_shade_{a.site}_подделка.npz"))
    res_m = float(z["res_m"])
    sc, scf = z["оценка"], zf["оценка"]
    img = load_winter(a.site, res_m)["shade"].astype("float32")

    tab = fdr_curve(sc, scf)
    tab.to_csv(M / f"openlin_shade_fdr_{a.site}.csv", index=False,
               encoding="utf-8-sig")
    log(f"кривая доли ложных сохранена: "
        f"{M / f'openlin_shade_fdr_{a.site}.csv'}")

    base = _stretch(img)
    fig, ax = plt.subplots(1, len(THRESHOLDS) + 1, figsize=(6.0 * (len(THRESHOLDS) + 1), 6.4))
    ax[0].imshow(base, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("зимний композит", fontsize=14)
    for k, thr in enumerate(THRESHOLDS, start=1):
        mask, df = lines_from_field(sc, res_m, thr, a.min_len)
        junk = float(np.interp(thr, tab["порог"], tab["доля мусора, %"]))
        km = float(df["длина_км"].sum()) if len(df) else 0.0
        log(f"порог {thr:.1f} (мусора {junk:.0f} %): линий {len(df)}, "
            f"суммарно {km:.0f} км, медианная длина "
            f"{df['длина_км'].median() if len(df) else 0:.1f} км, "
            f"прямолинейность "
            f"{df['прямолинейность'].median() if len(df) else 0:.2f}")
        if len(df):
            df.to_csv(M / f"openlin_shade_lines_{a.site}_{thr:.1f}.csv",
                      index=False, encoding="utf-8-sig")
        from scipy.ndimage import binary_dilation
        wide = binary_dilation(mask, np.ones((3, 3), dtype=bool))
        ax[k].imshow(base, cmap="gray", vmin=0, vmax=1)
        ov = np.zeros(wide.shape + (4,), dtype="float32")
        ov[wide] = (1.0, 0.15, 0.1, 1.0)
        ax[k].imshow(ov)
        ax[k].set_title(f"порог {thr:.1f}: {len(df)} линий, {km:.0f} км\n"
                        f"мусора {junk:.0f} %", fontsize=14)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    p = Path("outputs/steps") / f"уступы_карта_{a.site}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=105)
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
