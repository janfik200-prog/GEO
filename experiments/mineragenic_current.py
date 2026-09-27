"""Минерагенические наборы, пересчитанные в текущей сборке карты.

ПОВОД. Прогоны `honest_mineragenic_run.py` (03.09.2026) считались до того, как
рабочей стала сборка с двумя обязательными элементами: сглаживанием вытянутым
ядром 5 x 1.25 км на северо-запад и поправкой на наклон (из каждой карты
вычитается её собственная координатная поверхность). Их числа нельзя ставить в
одну таблицу с прогоном по 37 дистанционным признакам, который посчитан уже в
новой сборке. Здесь все наборы считаются одинаково.

Что считается для каждого набора:

* обучение на всём листе — как собирается итоговая карта;
* обучение по полосам (4 по X + 4 по Y, разрыв 15 км) — слепая проверка;
* обе карты сглаживаются рабочим ядром (и вдвое меньшим — второй файл метрик);
* совпадение верхних 10% с критериальной картой, оно же с поправкой на наклон,
  каппа, доля наклона в самой карте, попадание точек опробования.

Четвёртый набор — 37 дистанционных признаков с маской обнажённости — считается
тем же кодом, чтобы строка сравнения бралась из одного прогона, а не сшивалась
из разных.

Выход: `outputs/metrics/mineragenic_current.csv`,
`outputs/09_минерагения/mineragenic_current_map.png`.
Запуск из корня: ``python -m experiments.mineragenic_current``
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, mapstyle, neighbourhood as nb,
                 outputs as outdir, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, poly_design  # noqa: E402
from experiments.honest_forecast_final import trend_share  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH, smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KERNELS = (1.0, 0.5)
ID_COLS = ("row", "col", "x", "y")

#: Наборы: имя для таблицы -> файл датасета (или None для дистанционного).
VARIANTS = (
    ("слои организации + факторные", "dataset_mineragenic_org.parquet"),
    ("всё вместе", "dataset_mineragenic_full.parquet"),
    ("только наша докачка", "dataset_mineragenic_own.parquet"),
    ("дистанционные 37 (для сравнения)", None),
)

T0 = time.time()


def log(msg: str) -> None:
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


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
    basis = np.column_stack([np.ones(y.size), rbf_basis(cx, cy, RBF_GRID)])
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    area = config.CREF_AREA

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    def fill(v):
        return np.where(np.isfinite(v), v, np.nanmin(v[pool]))

    rows, maps = [], {}
    for name, fname in VARIANTS:
        if fname is None:
            cols = [c for c in honest_features("перенос") if c in feat.columns]
            X = spectral_mask.apply(feat[cols]).to_numpy(dtype=float)
            n_feat = X.shape[1]
        else:
            df = pd.read_parquet(config.PROCESSED_DIR / fname)
            cols = [c for c in df.columns if c not in ID_COLS]
            X = df[cols].to_numpy(dtype=float)
            n_feat = len(cols)

        p = np.full(y.size, np.nan)
        p[pool] = make_model().fit(X[pool], y[pool]).predict_proba(X[pool])[:, 1]
        _, aucs, per_ax = blind_map(X, y, pool, cx, cy)
        log("%-34s признаков %2d, площадь под кривой по полосам %.3f"
            % (name, n_feat, float(np.mean(aucs))))

        for scale in KERNELS:
            km = lambda v: v * scale * 1000.0 / meta.dx                # noqa: E731
            full = fill(nb.aniso_smooth(p, shape, km(ALONG_KM), km(ACROSS_KM),
                                        theta, mask=valid))
            bl = fill(smooth_strips(per_ax, meta, valid, cx, cy, scale))
            for how, s in (("обучение на всём листе", full),
                           ("обучение по полосам с разрывом 15 км", bl)):
                top = (s >= np.quantile(s[pool], 1 - area)) & valid
                maps[(name, scale, how)] = s
                rows.append({
                    "набор": name, "признаков": n_feat,
                    "ядро_км": "%g x %g" % (ALONG_KM * scale, ACROSS_KM * scale),
                    "сборка": how,
                    **score_map(s, y, crit, pool, valid, D),
                    "доля_наклона_R2": trend_share(s, basis, valid),
                    "площадь_под_кривой_полосы": float(np.mean(aucs)),
                    "точек_из_73": int(top[allp].sum()),
                    "непредвзятых_из_19": int(top[unb].sum()),
                })
                log("   ядро %-11s %-38s совпадение %.3f, без наклона %.3f, "
                    "каппа %.3f, наклон %.2f"
                    % (rows[-1]["ядро_км"], how, rows[-1]["совпадение_верх_10"],
                       rows[-1]["совпадение_без_тренда"], rows[-1]["каппа"],
                       rows[-1]["доля_наклона_R2"]))

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "mineragenic_current.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n=== Наборы в текущей сборке (ядро 5 x 1.25 км) ===")
    show = res[res["ядро_км"] == "5 x 1.25"]
    print(show[["набор", "признаков", "сборка", "совпадение_верх_10",
                "совпадение_без_тренда", "каппа", "доля_наклона_R2",
                "непредвзятых_из_19"]]
          .to_string(index=False, float_format=lambda v: "%.3f" % v))

    # --- Карта: слепая проверка, рабочее ядро, четыре набора рядом ---
    x_ext = (cx.max() - cx.min()) / 1000.0
    y_ext = (cy.max() - cy.min()) / 1000.0
    ext = [0, x_ext, 0, y_ext]
    cr = detrend(crit, pool, D)
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid

    fig, axs = plt.subplots(1, 4, figsize=(19, 5.6))
    for ax, (name, _) in zip(axs, VARIANTS):
        s = maps[(name, 1.0, "обучение по полосам с разрывом 15 км")]
        pct = mapstyle.percentile(s, valid)
        ax.imshow(pct.reshape(shape), origin="upper", extent=ext,
                  cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                  interpolation="nearest")
        mapstyle.outline(ax, tcr, shape, ext, flip=True,
                         color="#101010", lw=0.8)
        r = res[(res["набор"] == name) & (res["ядро_км"] == "5 x 1.25") &
                (res["сборка"] == "обучение по полосам с разрывом 15 км")].iloc[0]
        ax.set_title("%s\n%d признаков, без наклона %.3f"
                     % (name, r["признаков"], r["совпадение_без_тренда"]),
                     fontsize=10)
        ax.set_xlabel("км")
    axs[0].set_ylabel("км")
    fig.suptitle("Слепая проверка по полосам с разрывом 15 км, ядро 5 x 1.25 км; "
                 "чёрная линия — верхние 10% критериальной карты без наклона",
                 fontsize=11)
    fig.tight_layout()
    mapstyle.save_fig(fig, outdir.png("mineragenic_current_map.png"), dpi=150)
    log("OK: %s, %s" % (outdir.png("mineragenic_current_map.png").relative_to(ROOT),
                        out.relative_to(ROOT)))


if __name__ == "__main__":
    main()
