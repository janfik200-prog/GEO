"""Карта рабочей точки проекта — то, что на сегодня считается лучшим результатом.

ЗАЧЕМ. Числа рабочей точки разбросаны по метрикам веток 2 и 3, а показать нужно
саму карту: где прогноз ставит перспективные площади, где он согласен с
критериальным методом, где расходится, и как выглядит тот же прогноз после
снятия регионального наклона — потому что именно совпадение без тренда взято
главной метрикой.

ЧТО ЗА МОДЕЛЬ. Бустинг на 58 сырых признаках, обученный по протоколу полос с
разрывом 15 км (каждая ячейка предсказана моделью, не видевшей ни её саму, ни
её окрестность в 15 км), плюс сглаживание готовой карты ядром, вытянутым на
северо-запад: 5 км вдоль простирания и 1.25 км поперёк. Форма выбрана на
внутренних фолдах обучающих полос, а не по метрике отчёта.

ПЕРЕОБУЧЕНИЯ НЕТ: прогнозы берутся из кэша ветки 1
(`branch1_multiscale_cache.npz`), сглаживание применяется внутри каждой полосы,
схемы по X и по Y усредняются — ровно как в `branch3_shapes.py`.

Выход: `outputs/operating_point_map.png`.
Запуск из корня: ``python -m experiments.operating_point_map``
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, neighbourhood as nb)
from experiments.honest_feature_signal import N_STRIPS, poly_design  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTER_KEY = "бустинг | 58 сырые (планка)"
ALONG_KM, ACROSS_KM, NW_AZIMUTH = 5.0, 1.25, 315.0


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)

    # сборка карты рабочей точки: своё сглаживание внутри каждой полосы
    z = np.load(config.PROCESSED_DIR / "branch1_multiscale_cache.npz", allow_pickle=True)
    per_ax = {ax: np.array(z["maps"].item()[OUTER_KEY][ax], dtype=float)
              for ax in ("X", "Y")}
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    km = lambda v: v * 1000.0 / meta.dx                                # noqa: E731
    acc_ax = {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, _ = strip_ids(coord)
        sid = np.where(valid, sid, -1)
        acc = np.full(valid.size, np.nan)
        for k in range(N_STRIPS):
            m = sid == k
            acc[m] = nb.aniso_smooth(per_ax[ax], shape, km(ALONG_KM),
                                     km(ACROSS_KM), theta, mask=m)[m]
        acc_ax[ax] = acc
    s = mean_axes(acc_ax["X"], acc_ax["Y"])
    s = np.where(np.isfinite(s), s, np.nanmin(s))

    # метрики — для подписи под картой и для сверки с метриками ветки 3
    area = config.CREF_AREA
    top_m = (s >= np.quantile(s[pool], 1 - area)) & valid
    top_c = crit_reference.reference_mask(crit, pool, area) & valid
    sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
    tpr = (sr >= np.quantile(sr[pool], 1 - area)) & valid
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid
    agr = crit_reference.agreement(s, crit, pool, area)
    agree = float((top_m & top_c).sum() / top_c.sum())
    agree_dt = float((tpr & tcr).sum() / tcr.sum())

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    hit_all = int(top_m[allp].sum())
    hit_unb = int(top_m[unb].sum())
    print(f"совпадение верхних 10% {agree:.3f}, без тренда {agree_dt:.3f}, "
          f"каппа {agr['kappa']:.3f}, ROC карты {agr['auc']:.3f}; "
          f"точек {hit_unb} из {unb.size} непредвзятых и {hit_all} из {allp.size}")
    print(f"площадь верхних 10%: {int(top_m.sum())} ячеек = "
          f"{top_m.sum() * (meta.dx / 1000) ** 2:.0f} кв. км")

    # ------------------------------------------------------------------
    # Карта
    # ------------------------------------------------------------------
    x0, y0 = cx.min() / 1000.0, cy.min() / 1000.0
    ext = [0, (cx.max() - cx.min()) / 1000.0, 0, (cy.max() - cy.min()) / 1000.0]
    px, py = (cx - cx.min()) / 1000.0, (cy - cy.min()) / 1000.0

    # Строка 0 сетки — север: imshow с origin по умолчанию рисует её
    # сверху, поэтому переворот массива не нужен — он ставил карту
    # зеркально к точкам опробования, которые кладутся по координатам.
    def img(v, m):
        out = np.full(v.size, np.nan)
        out[m] = sps.rankdata(v[m]) / m.sum() * 100.0
        return out.reshape(shape)

    def cat_img(a, b):
        out = np.full(valid.size, np.nan)
        out[valid] = 0
        out[a & ~b] = 1
        out[b & ~a] = 2
        out[a & b] = 3
        return out.reshape(shape)

    cat_cmap = ListedColormap(["#e9e9ec", "#4c78a8", "#e4a11b", "#c0392b"])
    fig, axs = plt.subplots(1, 4, figsize=(21.5, 6.4))

    for ax, v, ttl in (
            (axs[0], crit, "Критериальный прогноз ВНИГНИ\n(эталон сравнения)"),
            (axs[1], s, "Прогноз рабочей точки\n(бустинг, 58 признаков, "
                        "ядро на северо-запад 5×1.25 км)")):
        h = ax.imshow(img(v, valid), extent=ext, cmap="magma", vmin=0, vmax=100)
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02,
                     label="перспективность, процентиль по листу")

    axs[2].imshow(cat_img(top_m, top_c), extent=ext, cmap=cat_cmap, vmin=0, vmax=3)
    axs[2].set_title(f"Верхние 10% площади: где сходятся\n"
                     f"совпадение {agree:.0%}, каппа {agr['kappa']:.2f}, "
                     f"ROC {agr['auc']:.3f}", fontsize=10)
    axs[2].legend(handles=[Patch(color="#c0392b", label="оба метода"),
                           Patch(color="#4c78a8", label="только прогноз"),
                           Patch(color="#e4a11b", label="только критериальный"),
                           Patch(color="#e9e9ec", label="ни один")],
                  loc="lower left", fontsize=8, framealpha=0.9)

    axs[3].imshow(cat_img(tpr, tcr), extent=ext, cmap=cat_cmap, vmin=0, vmax=3)
    axs[3].set_title(f"То же после снятия наклона\n"
                     f"совпадение {agree_dt:.0%} при случайном уровне 10%",
                     fontsize=10)

    for ax in axs:
        ax.scatter(px[allp], py[allp], s=16, facecolors="none",
                   edgecolors="#101010", linewidths=0.8, zorder=3)
        ax.scatter(px[unb], py[unb], s=34, marker="^", facecolors="#22d3a7",
                   edgecolors="#0b3b32", linewidths=0.7, zorder=4)
        ax.set_xlabel("восток, км от края листа")
        ax.set_ylabel("север, км от края листа")
    axs[0].legend(handles=[
        Line2D([], [], marker="o", color="none", markeredgecolor="#101010",
               label=f"точки заверки, {allp.size}"),
        Line2D([], [], marker="^", color="none", markerfacecolor="#22d3a7",
               markeredgecolor="#0b3b32", label=f"из них непредвзятые, {unb.size}")],
        loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Лучший результат проекта: прогноз, обученный по полосам с разрывом 15 км "
        "(ни одна ячейка не предсказана моделью, видевшей её окрестность)  |  "
        f"в верхние 10% попало {hit_unb} из {unb.size} непредвзятых точек заверки "
        f"и {hit_all} из {allp.size} всех", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ROOT / "outputs" / "operating_point_map.png", dpi=140)
    print("OK: outputs/operating_point_map.png")


if __name__ == "__main__":
    main()
