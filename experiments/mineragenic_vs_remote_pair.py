"""Карта-пара для доклада: минерагения против дистанционки в одном оформлении.

ПОВОД. В таблице «два обучения рядом» стоят два числа — 0.455 у 20
минерагенических признаков и 0.362 у 37 дистанционных. Числа сопоставимы
(один протокол, одна сборка, `experiments/mineragenic_current.py`), но карты
до сих пор жили в разных картинках с разными подписями и разным масштабом,
и сравнить их глазом было нельзя.

Здесь обе карты рисуются одной функцией: одинаковые классы перспективности,
один и тот же контур верхних 10% критериальной карты после снятия наклона,
одни и те же точки заверки. Разница на картинке — только разница наборов.

Считается ровно то, что показано: обучение по полосам (4 по X + 4 по Y,
разрыв 15 км), сглаживание рабочим ядром 5 x 1.25 км на северо-запад.

Выход: `outputs/09_минерагения/mineragenic_vs_remote_pair.png`.
Запуск из корня: ``python -m experiments.mineragenic_vs_remote_pair``
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, mapstyle, neighbourhood as nb,
                 outputs as outdir, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, poly_design  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH, smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ID_COLS = ("row", "col", "x", "y")
OUT_PNG = "mineragenic_vs_remote_pair.png"

#: Два набора доклада: подпись, файл датасета (None — дистанционный набор),
#: подпись состава признаков для заголовка панели.
PAIR = (
    ("Минерагения: слои организации + факторные",
     "dataset_mineragenic_org.parquet", "20 признаков"),
    ("Дистанционное зондирование: набор переноса", None,
     "37 признаков и признак «спектр измерен»"),
)

T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    area = config.CREF_AREA

    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    panels = []
    for title, fname, n_label in PAIR:
        if fname is None:
            cols = [c for c in honest_features("перенос") if c in feat.columns]
            X = spectral_mask.apply(feat[cols]).to_numpy(dtype=float)
        else:
            df = pd.read_parquet(config.PROCESSED_DIR / fname)
            cols = [c for c in df.columns if c not in ID_COLS]
            X = df[cols].to_numpy(dtype=float)

        _, aucs, per_ax = blind_map(X, y, pool, cx, cy)
        s = smooth_strips(per_ax, meta, valid, cx, cy, 1.0)
        s = np.where(np.isfinite(s), s, np.nanmin(s[pool]))
        sc = score_map(s, y, crit, pool, valid, D)
        top = (s >= np.quantile(s[pool], 1 - area)) & valid
        panels.append({"title": title, "n": X.shape[1], "подпись": n_label,
                       "map": s,
                       "точек_из_19": int(top[unb].sum()), **sc})
        log("%-44s признаков %2d, совпадение %.3f, без наклона %.3f, точек %d"
            % (title, X.shape[1], sc["совпадение_верх_10"],
               sc["совпадение_без_тренда"], panels[-1]["точек_из_19"]))

    # --- Карта-пара ---
    x_ext = (cx.max() - cx.min()) / 1000.0
    y_ext = (cy.max() - cy.min()) / 1000.0
    ext = [0, x_ext, 0, y_ext]
    cr = detrend(crit, pool, D)
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid
    px = cx[unb] / 1000.0 - cx.min() / 1000.0
    py = cy[unb] / 1000.0 - cy.min() / 1000.0

    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.4),
                            gridspec_kw={"width_ratios": [1, 1]})
    for ax, pan in zip(axs, panels):
        pct = mapstyle.percentile(pan["map"], valid)
        ax.imshow(pct.reshape(shape), origin="upper", extent=ext,
                  cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                  interpolation="nearest")
        mapstyle.outline(ax, tcr, shape, ext, flip=True, color="#101010",
                         lw=0.9)
        ax.plot(px, py, linestyle="none", marker="^", markersize=5,
                markerfacecolor="none", markeredgecolor="#0b3d2e",
                markeredgewidth=1.2)
        # Три строки, а не одна: длинная подпись состава признаков в одну
        # строку налезает на соседнюю панель.
        ax.set_title("%s\n%s\nсовпадение без наклона %.3f, "
                     "точек заверки %d из 19"
                     % (pan["title"], pan["подпись"],
                        pan["совпадение_без_тренда"], pan["точек_из_19"]),
                     fontsize=10, linespacing=1.4)
        ax.set_xlabel("восток, км от края листа", fontsize=9)
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("север, км от края листа", fontsize=9)

    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    handles += [
        Line2D([], [], color="#101010", lw=1.2,
               label="верхние 10% критериальной карты без наклона"),
        Line2D([], [], linestyle="none", marker="^", markersize=6,
               markerfacecolor="none", markeredgecolor="#0b3d2e",
               label="точки заверки по коренным пробам, 19"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Обучение по полосам с разрывом 15 км, ядро 5 x 1.25 км: "
                 "оба набора на одних метках, в одном протоколе и одной шкале",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    mapstyle.save_fig(fig, outdir.png(OUT_PNG), dpi=170)
    log("OK: %s" % outdir.png(OUT_PNG).relative_to(ROOT))


if __name__ == "__main__":
    main()
