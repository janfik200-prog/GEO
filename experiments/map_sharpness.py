"""Резкость карты: чем платим за то, чтобы контуры были чётче.

ПОВОД. Сдаваемая карта строится по сглаженному баллу — ядро рабочей точки
вытянуто вдоль северо-запада, 5 x 1.25 км (ветка 3). При ячейке 500 м это
усреднение по десятку ячеек вдоль зоны, поэтому карта выглядит размытой: зоны
перспективности расплываются, границы читаются плохо.

Размытость складывается из двух разных вещей, и их надо разделять:

1. **Оформление.** Непрерывная заливка процентилей + сглаживание изображения
   самим matplotlib. Это лечится бесплатно: резкие классы перспективности,
   `interpolation="nearest"`, отдельная линия по границе верхних 10%
   (`src/mapstyle.py`). Числа при этом не меняются вообще.
2. **Само сглаживание балла.** Оно не украшение, а часть рабочей точки: без
   него совпадение с критериальным прогнозом падает. Здесь измеряется, сколько
   именно стоит каждая ступень уменьшения ядра.

ЧТО СЧИТАЕТСЯ. Ядро масштабируется множителем 0 (не сглаживать), 0.25, 0.5,
0.75, 1 (рабочая точка), 1.5, 2. Для каждого — совпадение верхних 10% с
критериальной картой, оно же после снятия наклона, каппа; и три меры резкости:
доля ячеек границы в верхних 10% (чем больше, тем изрезаннее контур), число
отдельных пятен и площадь самого крупного из них.

Две схемы обучения, как на итоговых картах: на всех метках листа (мера
выразимости) и по полосам с разрывом 15 км (слепая проверка).

РЕЗУЛЬТАТ (08.09.2026), совпадение верхних 10% / оно же без наклона:

| ядро | на всём листе | по полосам | пятен | доля границы |
|---|---|---|---|---|
| без сглаживания | 0.725 / 0.461 | 0.245 / 0.260 | 158 | 76% |
| 1.25 x 0.31 км | 0.828 / 0.548 | 0.215 / 0.320 | 7 | 29% |
| 2.50 x 0.62 км | 0.803 / 0.560 | 0.192 / 0.350 | 3 | 24% |
| 3.75 x 0.94 км | 0.785 / 0.554 | 0.196 / 0.364 | 4 | 22% |
| 5.00 x 1.25 км (рабочая точка) | 0.767 / 0.548 | 0.199 / 0.362 | 4 | 20% |
| 10.0 x 2.50 км | 0.675 / 0.543 | 0.192 / 0.333 | 4 | 19% |

Явно плохо только полное отсутствие сглаживания: -0.082 [-0.153; -0.013] на
листе и -0.098 [-0.178; -0.003] вслепую (блочный бутстрэп по блокам 10 км), и
карта рассыпается на 158 пятен вместо четырёх зон. Всё между половиной и
полуторным ядром неразличимо: половинное ядро даёт +0.014 [-0.013; +0.042] на
листе и -0.010 [-0.051; +0.033] вслепую. То есть карту можно сделать вдвое
резче бесплатно, но и назвать это улучшением нельзя — выбор всё равно попадает
в разброс, а мерился бы он тем же числом, которым отчитываемся.

Решение (08.09.2026): сдаваемые карты переведены на ядро 2.5 x 0.62 км
(`KERNEL_SCALE = 0.5` в `full_sheet_map.py` и `transfer_map.py`) — контуры
вдвое резче при качестве, неотличимом от прежнего. Предъявлять это как
улучшение нельзя: выбор сделан тем же числом, которым отчитываемся, и обе
разности накрывают ноль. Ядро 5 x 1.25 км, выбранное вложенной схемой ветки 3,
остаётся опорной точкой сравнения.

Выход: `outputs/metrics/map_sharpness.csv`, `outputs/map_sharpness.png`.
Запуск из корня: ``python -m experiments.map_sharpness``
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, neighbourhood as nb, outputs as outdir,
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               make_model, poly_design)
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.spectral_exposure import score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH, agree_detrended,
                                                     boot_diff)
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def smooth(v, shape, dx, scale, mask=None):
    """Ядро рабочей точки, уменьшенное или увеличенное в `scale` раз."""
    if scale <= 0:
        return v.copy()
    km = 1000.0 / dx
    return nb.aniso_smooth(v, shape, ALONG_KM * scale * km, ACROSS_KM * scale * km,
                           nb.theta_from_azimuth(NW_AZIMUTH), mask=mask)


def sharpness(top, shape, valid, dx):
    """Меры изрезанности контура верхних 10%."""
    m = top.reshape(shape)
    er = ndimage.binary_erosion(m, np.ones((3, 3), bool))
    lab, n = ndimage.label(m, np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    cell_km2 = (dx / 1000.0) ** 2
    return {"доля_границы": float((m & ~er).sum() / max(m.sum(), 1)),
            "число_пятен": int(n),
            "крупнейшее_пятно_км2": float(sizes.max() * cell_km2)}


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
    area = config.CREF_AREA

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    X = spectral_mask.apply(feat[cols]).to_numpy(dtype=float)
    log("признаков %d, рабочих ячеек %d" % (X.shape[1], pool.size))

    # Сырые баллы считаются один раз, меняется только сглаживание.
    p_full = np.full(y.size, np.nan)
    p_full[pool] = make_model().fit(X[pool], y[pool]).predict_proba(X[pool])[:, 1]

    in_pool = np.zeros(y.size, dtype=bool)
    in_pool[pool] = True
    raw_ax, sid_ax = {}, {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        cal = np.full(y.size, np.nan)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            if te.size == 0 or y[tr].sum() < 20 or y[te].sum() < 5:
                continue
            m = make_model().fit(X[tr], y[tr])
            ref = np.sort(m.predict_proba(X[tr])[:, 1])
            p_te = m.predict_proba(X[te])[:, 1]
            cal[te] = np.searchsorted(ref, p_te, side="left") / max(ref.size, 1)
        raw_ax[ax] = cal
        sid_ax[ax] = np.where(valid, sid, -1)
    log("сырые баллы посчитаны: карта листа и слепая по полосам")

    rows, maps = [], {}
    for scale in SCALES:
        s_full = smooth(p_full, shape, meta.dx, scale, mask=valid)
        s_full = np.where(np.isfinite(s_full), s_full, np.nanmin(s_full[pool]))

        acc = {}
        for ax in ("X", "Y"):
            v = np.where(np.isfinite(raw_ax[ax]), raw_ax[ax],
                         np.nanmin(raw_ax[ax][valid]))
            a = np.full(valid.size, np.nan)
            for k in range(N_STRIPS):
                m = sid_ax[ax] == k
                if m.any():
                    a[m] = smooth(v, shape, meta.dx, scale, mask=m)[m]
            acc[ax] = a
        s_blind = mean_axes(acc["X"], acc["Y"])
        s_blind = np.where(np.isfinite(s_blind), s_blind, np.nanmin(s_blind[pool]))

        for how, s in (("обучение на всём листе", s_full),
                       ("обучение по полосам с разрывом 15 км", s_blind)):
            top = (s >= np.quantile(s[pool], 1 - area)) & valid
            rows.append({"ядро_доля_от_рабочего": scale,
                         "ядро_км": "%.2f x %.2f" % (ALONG_KM * scale,
                                                     ACROSS_KM * scale),
                         "схема": how,
                         **score_map(s, y, crit, pool, valid, D),
                         **sharpness(top, shape, valid, meta.dx)})
            maps[(scale, how)] = s
        a = rows[-2]
        b = rows[-1]
        log("ядро x%.2f: лист совпадение %.3f / без наклона %.3f; слепая %.3f / "
            "%.3f; пятен %d, граница %.0f%%"
            % (scale, a["совпадение_верх_10"], a["совпадение_без_тренда"],
               b["совпадение_верх_10"], b["совпадение_без_тренда"],
               a["число_пятен"], 100 * a["доля_границы"]))

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "map_sharpness.csv", index=False)

    # Разность с рабочей точкой: блочный бутстрэп по блокам 10 км.
    blocks = crit_reference.block_labels(meta)
    print()
    print("=== Разность с рабочим ядром (5 x 1.25 км), совпадение без наклона ===")
    for how in ("обучение на всём листе", "обучение по полосам с разрывом 15 км"):
        ta_ref, tc = agree_detrended(maps[(1.0, how)], crit, pool, valid, D)
        for scale in SCALES:
            if scale == 1.0:
                continue
            ta, _ = agree_detrended(maps[(scale, how)], crit, pool, valid, D)
            b = boot_diff(ta, ta_ref, tc, pool, blocks)
            print("  %-38s ядро x%.2f  %+.3f [%+.3f; %+.3f]"
                  % (how, scale, b["разность_средняя"], b["разность_ниж_5"],
                     b["разность_верх_95"]))

    ext = [0, (cx.max() - cx.min()) / 1000.0, 0, (cy.max() - cy.min()) / 1000.0]
    show = [s for s in SCALES if s in (0.0, 0.5, 1.0, 2.0)]
    fig, axs = plt.subplots(1, len(show), figsize=(5.4 * len(show), 6.0))
    for ax, scale in zip(axs, show):
        s = maps[(scale, "обучение на всём листе")]
        top = (s >= np.quantile(s[pool], 1 - area)) & valid
        img = np.flipud(mapstyle.percentile(s, valid).reshape(shape))
        h = ax.imshow(img, extent=ext, cmap=mapstyle.class_cmap(),
                      norm=mapstyle.class_norm(), interpolation="nearest")
        mapstyle.outline(ax, top, shape, ext)
        r = res[(res["ядро_доля_от_рабочего"] == scale) &
                (res["схема"] == "обучение на всём листе")].iloc[0]
        ax.set_title(("%s%s"
                      "совпадение %.0f%%, без наклона %.0f%%%s"
                      "пятен %d, крупнейшее %.0f кв. км")
                     % ("ядро %s км" % r["ядро_км"] if scale
                        else "без сглаживания", chr(10),
                        r["совпадение_верх_10"] * 100,
                        r["совпадение_без_тренда"] * 100, chr(10),
                        r["число_пятен"], r["крупнейшее_пятно_км2"]), fontsize=10)
        ax.set_xlabel("восток, км от края листа")
        ax.set_ylabel("север, км от края листа")
    fig.colorbar(h, ax=axs, fraction=0.02, ticks=mapstyle.CLASS_EDGES,
                 label="класс перспективности, процентиль по листу")
    fig.suptitle("Резкость карты и цена резкости: слева направо ядро сглаживания "
                 "растёт от нуля до двойного. Обводка — сдаваемые верхние 10%. "
                 "Числа считаны по одной и той же модели, меняется только "
                 "сглаживание", fontsize=11)
    mapstyle.save_fig(fig, outdir.png("map_sharpness.png"))
    print()
    print(res.to_string(index=False, float_format=lambda v: "%.3f" % v))
    log("OK: %s, outputs/metrics/map_sharpness.csv"
        % outdir.png("map_sharpness.png").relative_to(ROOT))


if __name__ == "__main__":
    main()
