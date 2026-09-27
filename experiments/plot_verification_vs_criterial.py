"""Точки заверки на карте листа рядом с картой критериального прогноза.

Чисто визуализация, без новых расчётов: точки — `src.assessment.load_verification_points`
(76 точек геохимического опробования -> 73 ячейки, коды 1/2/4/5, см. докстринг
`criterial_target.load_real_points`), критериальный прогноз — `prognoz.pgrid`
(`criterial_target.load_prognoz_grid`, ниже = перспективнее, на карте показан
как ранг `-prognoz` для интуитивности «выше = перспективнее», тот же приём,
что в `experiments/atypicality_v6.py`).

Выход: `outputs/verification_points_vs_criterial.png`.
Запуск из корня репозитория: ``python -m experiments.plot_verification_vs_criterial``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, config, criterial_target  # noqa: E402

CODE_LABELS = {
    1: "литопробы, рыхлые (аномальные)",
    2: "литопробы, коренные (аномальные)",
    4: "проявления U/TR",
    5: "шлихи Au (россыпь)",
}
CODE_MARKERS = {1: "o", 2: "s", 4: "*", 5: "^"}
CODE_COLORS = {1: "#8ecae6", 2: "#219ebc", 4: "#ffb703", 5: "#fb8500"}
CODE_SIZES = {1: 22, 2: 30, 4: 110, 5: 40}


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    pts = assessment.load_verification_points(meta)
    pts["row"] = pts["cell"] // meta.pic
    pts["col"] = pts["cell"] % meta.pic
    print(f"Точки заверки: {len(pts)} ячеек, по кодам:\n{pts['code'].value_counts().sort_index()}")

    score = -prognoz.ravel()
    rank = sps.rankdata(score) / score.size
    grid = rank.reshape(meta.prf, meta.pic)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_pts, ax_crit) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    ax_pts.set_facecolor("#f0ede5")
    ax_pts.set_xlim(0, meta.pic)
    ax_pts.set_ylim(meta.prf, 0)
    for code in sorted(CODE_LABELS):
        sub = pts[pts["code"] == code]
        if sub.empty:
            continue
        ax_pts.scatter(sub["col"] + 0.5, sub["row"] + 0.5,
                       marker=CODE_MARKERS[code], c=CODE_COLORS[code],
                       s=CODE_SIZES[code], edgecolors="black", linewidths=0.4,
                       label=f"{CODE_LABELS[code]} (n={len(sub)})", zorder=3)
    ax_pts.set_title("Точки заверки (геохимическое опробование)")
    ax_pts.set_xlabel("столбец (x500 м)")
    ax_pts.set_ylabel("строка (x500 м)")
    ax_pts.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_pts.set_aspect("equal")

    im = ax_crit.imshow(grid, cmap="magma", vmin=0, vmax=1)
    for code in sorted(CODE_LABELS):
        sub = pts[pts["code"] == code]
        if sub.empty:
            continue
        ax_crit.scatter(sub["col"] + 0.5, sub["row"] + 0.5,
                        marker=CODE_MARKERS[code], c="cyan",
                        s=CODE_SIZES[code] * 0.6, edgecolors="black", linewidths=0.4,
                        zorder=3)
    ax_crit.set_title("Критериальный прогноз (ранг −prognoz, выше = перспективнее)")
    ax_crit.set_xlabel("столбец (x500 м)")
    fig.colorbar(im, ax=ax_crit, shrink=0.8, label="нормированный ранг")

    fig.suptitle("Точки заверки vs критериальный прогноз — главный лист")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "verification_points_vs_criterial.png"
    fig.savefig(out_path, dpi=140)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
