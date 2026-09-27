"""Карта модели, обученной на всём листе: совпадение с прогнозом ВНИГНИ.

ЧТО ЭТО. Модель на 37 дистанционных признаках с маской обнажённости, обученная
на ВСЕХ ячейках листа и применённая к ним же. Это итоговая карта — то, что
организация видит как результат воспроизведения критериального прогноза
дистанционными данными.

ЧТО ЭТО НЕ ЕСТЬ. Не проверка обобщения. Модель видела разметку всех ячеек, и
высокое совпадение здесь ожидаемо: она пересказывает карту, которую ей показали,
а не предсказывает её на незнакомой территории. Цифру обобщения даёт другая
сборка — обучение по полосам с разрывом 15 км, где ни одна ячейка не оценивается
моделью, видевшей её окрестность (`operating_point_masked.py`, совпадение 20%,
без наклона 36%). Обе карты показаны рядом именно для того, чтобы их не путать.

ЗАЧЕМ ОБЕ. Первая отвечает на вопрос «можно ли вообще выразить критериальный
прогноз через съёмку и рельеф». Вторая отвечает на вопрос «перенесётся ли это
на территорию, где формулу не считали» — там результат много скромнее. Первая
цифра не доказывает вторую.

ЦЕЛЬ ОБУЧЕНИЯ. Обе карты считаются в двух вариантах: порог верхних 10% (как
было до 08.09.2026) и регрессия на сам критериальный балл. Второй вариант
сохраняет порядок ячеек вблизи границы отсечения, по которому метрика и
спрашивает, и на слепой проверке даёт 0.384 против 0.362
(`experiments/target_form.py`). Рисуется регрессия, обе цифры пишутся в CSV.

ЯДРО СГЛАЖИВАНИЯ. Под каждое ядро из `KERNELS` — свой файл: рабочая точка
ветки 3 (5 x 1.25 км) и половинное ядро (2.5 x 0.62 км) с вдвое более резкими
контурами. Разность между ними внутри разброса (+0.014 [-0.013; +0.042] на
листе и -0.010 [-0.051; +0.033] вслепую, `map_sharpness.py`) — это выбор
оформления при равном качестве, а не улучшение. Обучение считается один раз на
цель, ядро меняет только сглаживание.

РЕЗУЛЬТАТ СМЕНЫ ЦЕЛИ (08.09.2026), половинное ядро.

| цель | обучение на всём листе | по полосам с разрывом 15 км |
|---|---|---|
| порог верхних 10% | 0.803 / без наклона 0.560 | 0.192 / 0.350 |
| сам критериальный балл | 0.569 / 0.495 | 0.175 / 0.383 |

При рабочем ядре 5 x 1.25 км было соответственно 0.767 / 0.548 и 0.199 / 0.362
для порога, 0.591 / 0.460 и 0.201 / 0.384 для балла.

Регрессия выигрывает там, где модель не видела территорию (слепая проверка,
+0.022 без наклона), и заметно проигрывает там, где видела: порог верхних 10%
учит ровно ту границу, по которой потом считается совпадение, поэтому на
собственных ячейках воспроизводит её точнее. Рабочей точкой остаётся порог:
разность на слепой проверке лежит внутри разброса (+0.020 [-0.042; +0.087],
`target_form.py`), а на сдаваемой карте листа цена смены велика и измерена.
Обе карты рисуются, обе цифры пишутся в CSV.

РЕЗКОСТЬ. Карта строится по сглаженному баллу, поэтому выглядит размытой. Само
сглаживание — часть рабочей точки, но оформление (резкие классы
перспективности, отрисовка без интерполяции, отдельная линия по границе
верхних 10%) ничего не стоит и вынесено в `src/mapstyle.py`. Цена уменьшения
ядра измерена отдельно: `experiments/map_sharpness.py`.

Выход: по файлу на каждое сочетание ядра и цели —
`outputs/full_sheet_map_ядро{5|2.5}км[_регрессия].png` и такие же
`data/processed/full_sheet_map_*.parquet`; все восемь строк метрик —
в `outputs/metrics/full_sheet_map.csv`.
Запуск из корня: ``python -m experiments.full_sheet_map``
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats as sps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference, mapstyle,  # noqa: E402
                 criterial_target, neighbourhood as nb, outputs as outdir,
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, poly_design  # noqa: E402
from experiments.target_form import make_reg, strip_map  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH, smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ядра сглаживания, под каждое — свой файл карты. 1.0 — рабочая точка ветки 3
# (5 x 1.25 км), 0.5 — вдвое более резкие контуры при неотличимом качестве;
# цена измерена в `experiments/map_sharpness.py`.
KERNELS = (1.0, 0.5)


def kernel_tag(scale: float) -> str:
    return "_ядро%gкм" % (ALONG_KM * scale)


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
    base = feat[cols]
    keep = spectral_mask.exposure_mask(base)
    X = spectral_mask.apply(base).to_numpy(dtype=float)
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    target = np.where(np.isfinite(crit), crit, np.nanmin(crit[pool]))

    def fill(v):
        return np.where(np.isfinite(v), v, np.nanmin(v[pool]))

    # Обучение не зависит от ядра, поэтому баллы считаются один раз на цель, а
    # ядро меняет только сглаживание.
    raw = {}
    for kind in ("класс", "число"):
        p = np.full(y.size, np.nan)
        if kind == "класс":
            p[pool] = make_model().fit(X[pool], y[pool]).predict_proba(X[pool])[:, 1]
            _, _, per_ax = blind_map(X, y, pool, cx, cy)
        else:
            p[pool] = make_reg().fit(X[pool], target[pool]).predict(X[pool])
            per_ax, _ = strip_map(X, target, y, pool, cx, cy, "число")
        raw[kind] = (p, per_ax)

    maps, rows = {}, []
    for scale in KERNELS:
        km = lambda v: v * scale * 1000.0 / meta.dx                    # noqa: E731
        for kind, ttl in (("число", "регрессия на критериальный балл"),
                          ("класс", "порог верхних 10%")):
            p, per_ax = raw[kind]
            full = fill(nb.aniso_smooth(p, shape, km(ALONG_KM), km(ACROSS_KM),
                                        theta, mask=valid))
            bl = fill(smooth_strips(per_ax, meta, valid, cx, cy, scale))
            for how, s in (("обучение на всём листе", full),
                           ("обучение по полосам с разрывом 15 км", bl)):
                maps[(scale, kind, how)] = s
                rows.append({"ядро_км": "%g x %g" % (ALONG_KM * scale,
                                                     ACROSS_KM * scale),
                             "цель": ttl, "сборка": how,
                             **score_map(s, y, crit, pool, valid, D)})
                print("  ядро %-11s %-32s %-38s совпадение %.3f, без наклона "
                      "%.3f, каппа %.3f"
                      % (rows[-1]["ядро_км"], ttl, how,
                         rows[-1]["совпадение_верх_10"],
                         rows[-1]["совпадение_без_тренда"], rows[-1]["каппа"]))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "full_sheet_map.csv", index=False)

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    x_ext = (cx.max() - cx.min()) / 1000.0
    y_ext = (cy.max() - cy.min()) / 1000.0
    ext = [0, x_ext, 0, y_ext]
    px, py = (cx - cx.min()) / 1000.0, (cy - cy.min()) / 1000.0

    # Строка 0 сетки — север, поэтому массив идёт в imshow как есть
    # (origin="upper"), а контуры `mapstyle.outline` берутся с flip=True.
    # Прежний np.flipud переворачивал растр относительно точек опробования,
    # которые ставятся по координатам.
    def img(v, m):
        o = np.full(v.size, np.nan)
        o[m] = sps.rankdata(v[m]) / m.sum() * 100.0
        return o.reshape(shape)

    def cat_img(a, b):
        o = np.full(valid.size, np.nan)
        o[valid] = 0
        o[a & ~b] = 1
        o[b & ~a] = 2
        o[a & b] = 3
        return o.reshape(shape)

    cat_cmap = ListedColormap(["#e9e9ec", "#4c78a8", "#e4a11b", "#c0392b"])
    top_c = crit_reference.reference_mask(crit, pool, area) & valid
    cr = detrend(crit, pool, D)
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid

    for scale in KERNELS:
      for kind, ttl, tail in (("класс", "цель — порог верхних 10%", ""),
                              ("число", "цель — сам критериальный балл",
                               "_регрессия")):
        suffix = kernel_tag(scale) + tail
        ttl = "%s, ядро %g x %g км" % (ttl, ALONG_KM * scale, ACROSS_KM * scale)
        s_full = maps[(scale, kind, "обучение на всём листе")]
        s_blind = maps[(scale, kind, "обучение по полосам с разрывом 15 км")]
        top_f = (s_full >= np.quantile(s_full[pool], 1 - area)) & valid
        top_b = (s_blind >= np.quantile(s_blind[pool], 1 - area)) & valid
        fr = detrend(s_full, pool, D)
        tfr = (fr >= np.quantile(fr[pool], 1 - area)) & valid
        agr = crit_reference.agreement(s_full, crit, pool, area)
        agree = float((top_f & top_c).sum() / top_c.sum())
        agree_dt = float((tfr & tcr).sum() / tcr.sum())
        blind_row = [r for r in rows
                     if r["сборка"] != "обучение на всём листе"
                     and r["ядро_км"].startswith("%g " % (ALONG_KM * scale))
                     and (r["цель"] == "порог верхних 10%") == (kind == "класс")][0]

        pd.DataFrame({"row": np.repeat(np.arange(shape[0]), shape[1]),
                      "col": np.tile(np.arange(shape[1]), shape[0]),
                      "x": cx, "y": cy, "score": s_full.astype(np.float32),
                      "верхние_10": top_f, "валидная": valid}).to_parquet(
            config.PROCESSED_DIR / ("full_sheet_map%s.parquet" % suffix), index=False)
        print("%s: верхние 10%% — %d ячеек = %.0f кв. км; точек опробования внутри "
              "%d из %d, непредвзятых %d из %d"
              % (ttl, int(top_f.sum()), top_f.sum() * (meta.dx / 1000) ** 2,
                 int(top_f[allp].sum()), allp.size, int(top_f[unb].sum()), unb.size))

        fig, axs = plt.subplots(1, 4, figsize=(21.5, 6.4))
        for ax, v, t in (
                (axs[0], crit, "Критериальный прогноз ВНИГНИ" + chr(10) +
                 "(эталон сравнения)"),
                (axs[1], s_full,
                 "Итоговая карта: обучение на всём листе," + chr(10) +
                 "37 дистанционных признаков," + chr(10) + ttl)):
            h = ax.imshow(img(v, valid), extent=ext, cmap=mapstyle.class_cmap(),
                          norm=mapstyle.class_norm(), interpolation="nearest")
            ax.set_title(t, fontsize=10)
            cb = plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02,
                              ticks=mapstyle.CLASS_EDGES,
                              label="класс перспективности, процентиль по листу")
            cb.ax.tick_params(labelsize=8)
        mapstyle.outline(axs[0], top_c, shape, ext, flip=True)
        mapstyle.outline(axs[1], top_f, shape, ext, flip=True)

        axs[2].imshow(cat_img(top_f, top_c), extent=ext, cmap=cat_cmap,
                      vmin=0, vmax=3, interpolation="nearest")
        axs[2].set_title(("Верхние 10%% площади: где сходятся\n"
                          "совпадение %.0f%%, каппа %.2f, ROC %.3f\n"
                          "после снятия наклона %.0f%%")
                         % (agree * 100, agr["kappa"], agr["auc"],
                            agree_dt * 100), fontsize=10)
        axs[2].legend(handles=[Patch(color="#c0392b", label="оба метода"),
                               Patch(color="#4c78a8", label="только наша карта"),
                               Patch(color="#e4a11b", label="только критериальный"),
                               Patch(color="#e9e9ec", label="ни один")],
                      loc="lower left", fontsize=8, framealpha=0.9)

        axs[3].imshow(cat_img(top_b, top_c), extent=ext, cmap=cat_cmap,
                      vmin=0, vmax=3, interpolation="nearest")
        axs[3].set_title(("Для сравнения: обучение по полосам\n"
                          "с разрывом 15 км — совпадение %.0f%%,\n"
                          "после снятия наклона %.0f%%")
                         % (blind_row["совпадение_верх_10"] * 100,
                            blind_row["совпадение_без_тренда"] * 100), fontsize=10)

        for ax in (axs[2], axs[3]):
            mapstyle.outline(ax, valid & keep, shape, ext, flip=True,
                             color="#2b2b2b", lw=0.35)

        for ax in axs:
            ax.scatter(px[allp], py[allp], s=16, facecolors="none",
                       edgecolors="#101010", linewidths=0.8, zorder=3)
            ax.scatter(px[unb], py[unb], s=34, marker="^", facecolors="#22d3a7",
                       edgecolors="#0b3b32", linewidths=0.7, zorder=4)
            ax.set_xlabel("восток, км от края листа")
            ax.set_ylabel("север, км от края листа")
        axs[0].legend(handles=[
            Line2D([], [], marker="o", color="none", markeredgecolor="#101010",
                   label="точки опробования, %d" % allp.size),
            Line2D([], [], marker="^", color="none", markerfacecolor="#22d3a7",
                   markeredgecolor="#0b3b32",
                   label="из них непредвзятые, %d" % unb.size)],
            loc="lower left", fontsize=8, framealpha=0.9)

        fig.suptitle(
            "Воспроизведение критериального прогноза дистанционными данными, %s: "
            "карта листа совпадает с прогнозом ВНИГНИ" % ttl + chr(10) +
            "на %.0f%% площади верхних 10%%, после снятия наклона %.0f%%. "
            "ВАЖНО: модель обучена на разметке всего листа —" % (
                agree * 100, agree_dt * 100) + chr(10) +
            "это мера выразимости прогноза через съёмку, а не проверка переноса; "
            "правая карта показывает, что остаётся при обучении вслепую",
            fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.87))
        mapstyle.save_fig(fig, outdir.png("full_sheet_map%s.png" % suffix))
        plt.close(fig)
        print("OK: %s" % outdir.png("full_sheet_map%s.png" % suffix)
              .relative_to(ROOT))


if __name__ == "__main__":
    main()
