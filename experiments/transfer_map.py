"""Перенос итоговой модели на смежную территорию и карта широкой сетки.

ЗАЧЕМ. Карта переноса собиралась 05.09.2026 (`branch5_alignment.py`) моделью,
которая тогда была рабочей: 37 признаков без разбора, ансамбль по фону,
постобработка ветки 3. С тех пор рабочая точка изменилась — спектры считаются
измерением породы только на обнажённых ячейках, остальным ставится пропуск плюс
признак-флаг (`src/spectral_mask.py`), и это подняло совпадение на самом листе
с 0.279 до 0.362. Карту переноса надо пересобрать тем же, чем считается лист,
иначе рядом лежат две карты, посчитанные разными методами.

ЧТО ДЕЛАЕТСЯ.

1. Модель обучается на ВСЕХ ячейках листа — как итоговая карта листа
   (`full_sheet_map.py`), потому что на смежной территории отложенных полос
   быть не может: там нет разметки вообще. Постобработка ровно та же, что у
   карты листа, чтобы две карты можно было класть рядом.
2. Широкая сборка приводится к шкале сборки листа поквантильно, по парам «одно
   место, две сборки» на перекрытии (широкая сетка накрывает лист целиком).
   Это измеренная поправка, а не допущение о равенстве территорий. Проверено
   ранее: поправка безвредна, убирает выход за обучающий диапазон, прироста
   качества не доказано.
3. Маска обнажённости считается ПОСЛЕ приведения шкал — модель училась на
   пороге в шкале листа, и флаг «спектр измерен» должен означать на смежной
   территории ровно то же, что на листе.
4. Балл сглаживается тем же ядром рабочей точки (5 x 1.25 км вдоль
   северо-запада).

МЕРА ДОВЕРИЯ. Два очевидных кандидата не годятся. Разброс между зёрнами
обучения равен нулю: гистограммный бустинг без ранней остановки и подвыборок
детерминирован. Доля признаков вне обучающего диапазона тоже равна нулю, но по
другой причине — поквантильное приведение шкал загоняет значения в диапазон
листа по построению, так что мерить им нечего.

Поэтому считается новизна: расстояние в пространстве признаков до десятой
ближайшей ячейки листа, в единицах медианного такого расстояния внутри самого
листа. Единица означает «ячейка окружена знакомым не хуже, чем типичная ячейка
листа», три — «втрое дальше от всего, что модель видела». Берутся только
признаки рельефа и линеаментов: спектральные под маской обнажённости содержат
пропуски, и расстояние по ним считать нечестно.

ЧЕГО ЗДЕСЬ НЕТ. Проверки. На смежной территории нет ни критериального расчёта,
ни точек опробования, поэтому качество переноса измерить нечем — совпадение
считается только на перекрытии с листом, где модель обучалась, и служит
проверкой сборки, а не оценкой прогноза. Единственное, что говорит о переносе
по существу: доля смежной территории, попавшая в верхние 10%, и новизна
признаков.

ДВЕ ЦЕЛИ ОБУЧЕНИЯ. Считается и рисуется два варианта: порог верхних 10% (как
было) и регрессия на сам критериальный балл — тот же перебор, что на карте
листа.

РЕЗУЛЬТАТ (08.09.2026). Считаются оба ядра из `KERNELS` и обе цели, под
каждое сочетание — свой файл.

| ядро / цель | верхние 10% | балл |
|---|---|---|
| 5 x 1.25: сборка листа | 0.767 / без наклона 0.548 | 0.591 / 0.460 |
| 5 x 1.25: широкая сборка тех же мест | 0.660 / 0.469 | 0.462 / 0.394 |
| 5 x 1.25: смежная территория в верхних 10% | 5.1% площади | 6.7% |
| 2.5 x 0.62: сборка листа | 0.803 / 0.560 | 0.569 / 0.495 |
| 2.5 x 0.62: широкая сборка тех же мест | 0.649 / 0.483 | 0.429 / 0.396 |
| 2.5 x 0.62: смежная территория в верхних 10% | 5.8% площади | 7.1% |

Смена сборки стоит около 0.11 совпадения и 0.08 без наклона — цена того, что
мозаики собирались отдельно, а ячейки смещены на 264 м. Новизна признаков на
смежной территории (медиана 2.68) почти не отличается от новизны внутри
габаритов листа, пересчитанного второй сборкой (2.55): различие создаёт смена
сборки, а не география.

Выход: `outputs/transfer_map_ядро{5|2.5}км[_регрессия].png`,
`data/processed/forecast_wide_v3_ядро*.parquet`,
`outputs/metrics/transfer_map.csv`. Имя без суффикса не занимается: его пишет
другой скрипт, `experiments/transfer_nn.py`.
Запуск из корня: ``python -m experiments.transfer_map``
"""

import sys
import time
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
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference, mapstyle,  # noqa: E402
                 criterial_target, neighbourhood as nb, outputs as outdir,
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, poly_design  # noqa: E402
from experiments.target_form import make_reg  # noqa: E402
from experiments.spectral_exposure import score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)
from experiments.branch5_alignment import align  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ядра сглаживания, под каждое — свой файл карты (те же, что у карты листа).
KERNELS = (1.0, 0.5)


def kernel_tag(scale: float) -> str:
    return "_ядро%gкм" % (ALONG_KM * scale)

T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def novelty(Fs: np.ndarray, Fw: np.ndarray, k: int = 10) -> np.ndarray:
    """Новизна ячейки: расстояние до k-й ближайшей ячейки листа, в единицах
    медианного такого расстояния внутри самого листа."""
    mu, sd = Fs.mean(axis=0), Fs.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    zs, zw = (Fs - mu) / sd, (Fw - mu) / sd
    tree = cKDTree(zs)
    d_in, _ = tree.query(zs, k=k + 1)
    d_out, _ = tree.query(zw, k=k)
    return d_out[:, -1] / max(np.median(d_in[:, -1]), 1e-9)


def smooth_sheet(v, meta, valid, scale: float = 1.0):
    """Ядро рабочей точки на сетке листа, уменьшенное в `scale` раз."""
    return nb.aniso_smooth(v, (meta.prf, meta.pic),
                           ALONG_KM * scale * 1000.0 / meta.dx,
                           ACROSS_KM * scale * 1000.0 / meta.dx,
                           nb.theta_from_azimuth(NW_AZIMUTH), mask=valid)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    area = config.CREF_AREA

    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)
    in_sheet = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
                (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    tree = cKDTree(np.column_stack([wx[in_sheet], wy[in_sheet]]))
    _, idx = tree.query(np.column_stack([cx, cy]), k=1)
    w_in = wide.loc[in_sheet].reset_index(drop=True)
    log("признаков %d, лист %d рабочих ячеек, широкая сетка %d, перекрытие %d"
        % (len(cols), pool.size, len(wide), int(in_sheet.sum())))

    # 1. Приведение широкой сборки к шкале листа по парам «одно место, две сборки»
    raw_sheet = feat[cols]
    pair_w = w_in.loc[idx, cols].to_numpy(dtype=float)[valid]
    pair_s = raw_sheet.to_numpy(dtype=float)[valid]
    wide_al = pd.DataFrame(align(pair_w, pair_s, wide[cols].to_numpy(dtype=float)),
                           columns=cols)
    log("широкая сборка приведена к шкале листа")

    # 2. Маска обнажённости — после приведения, в общей шкале
    Xs = spectral_mask.apply(raw_sheet).to_numpy(dtype=float)
    keep_w = spectral_mask.exposure_mask(wide_al)
    keep_s = spectral_mask.exposure_mask(raw_sheet)
    Xw = spectral_mask.apply(wide_al).to_numpy(dtype=float)
    log("обнажённых ячеек: лист %.1f%%, широкая сетка %.1f%%"
        % (100 * keep_s[valid].mean(), 100 * keep_w.mean()))

    plain = [j for j, c in enumerate(cols) if not c.startswith("s2_")]
    outside = novelty(Xs[pool][:, plain], Xw[:, plain])
    log("новизна по %d непропускным признакам: в габаритах листа медиана %.2f, "
        "на смежной территории медиана %.2f, 90-й процентиль %.2f"
        % (len(plain), np.median(outside[in_sheet]),
           np.median(outside[~in_sheet]), np.quantile(outside[~in_sheet], 0.9)))

    target = np.where(np.isfinite(crit), crit, np.nanmin(crit[pool]))
    Xw_pair = Xw[in_sheet][idx]

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    ext = [WIDE_META.x0 / 1000.0,
           (WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx) / 1000.0,
           WIDE_META.y0 / 1000.0, WIDE_META.y_top / 1000.0]
    box_x = np.array([cx.min(), cx.max(), cx.max(), cx.min(), cx.min()]) / 1000.0
    box_y = np.array([cy.min(), cy.min(), cy.max(), cy.max(), cy.min()]) / 1000.0

    def pct(v):
        return (sps.rankdata(v) / v.size * 100.0).reshape(WIDE_META.shape)

    ct = crit_reference.reference_mask(crit, pool, area) & valid
    crit_top = np.zeros(wide.shape[0])
    where = np.flatnonzero(in_sheet)
    crit_top[where[idx]] = ct.astype(float)
    old = pd.read_parquet(config.PROCESSED_DIR / "forecast_wide_v2.parquet")
    t_old = old["score_сглаженный"].to_numpy()

    # Обучение не зависит от ядра: модель на цель считается один раз.
    fitted = {}
    for kind in ("класс", "число"):
        if kind == "класс":
            mdl = make_model().fit(Xs[pool], y[pool])
            predict = lambda M, m=mdl: m.predict_proba(M)[:, 1]  # noqa: E731
        else:
            predict = make_reg().fit(Xs[pool], target[pool]).predict
        fitted[kind] = (predict, predict(Xw))

    rows = []
    for scale in KERNELS:
      for kind, ttl, tail in (("класс", "цель — порог верхних 10%", ""),
                              ("число", "цель — сам критериальный балл",
                               "_регрессия")):
        suffix = kernel_tag(scale) + tail
        ttl = "%s, ядро %g x %g км" % (ttl, ALONG_KM * scale, ACROSS_KM * scale)
        log("=== %s ===" % ttl)
        predict, score = fitted[kind]

        # 4. Сглаживание: ядро рабочей точки, уменьшенное в `scale` раз
        km = scale * 1000.0 / WIDE_META.dx
        score_sm = nb.aniso_smooth(score, WIDE_META.shape, ALONG_KM * km,
                                   ACROSS_KM * km,
                                   nb.theta_from_azimuth(NW_AZIMUTH))
        log("сглаживание: спирмен с несглаженной картой %.3f"
            % sps.spearmanr(score, score_sm).statistic)

        top_w = score_sm >= np.quantile(score_sm, 1 - area)
        out_share = float(top_w[~in_sheet].mean())
        log("верхние 10%% широкой сетки: %d ячеек, из них на смежной территории "
            "%d (%.0f%%), это %.1f%% её площади"
            % (int(top_w.sum()), int(top_w[~in_sheet].sum()),
               100 * top_w[~in_sheet].sum() / max(top_w.sum(), 1),
               100 * out_share))

        # 5. Контроль сборки на перекрытии: те же места, признаки широкой сборки
        s_pair = np.full(y.size, np.nan)
        s_pair[pool] = predict(Xw_pair[pool])
        s_pair = np.where(np.isfinite(s_pair), s_pair, np.nanmin(s_pair[pool]))
        s_own = np.full(y.size, np.nan)
        s_own[pool] = predict(Xs[pool])
        s_own = np.where(np.isfinite(s_own), s_own, np.nanmin(s_own[pool]))

        for name, v in (("сборка листа (как итоговая карта)", s_own),
                        ("широкая сборка тех же мест, приведена", s_pair)):
            sm = smooth_sheet(v, meta, valid, scale)
            rows.append({"цель": ttl, "вариант": name,
                         **score_map(sm, y, crit, pool, valid, D)})
            log("  %-40s совпадение %.3f, без наклона %.3f, каппа %.3f"
                % (name, rows[-1]["совпадение_верх_10"],
                   rows[-1]["совпадение_без_тренда"], rows[-1]["каппа"]))
        rows.append({"цель": ttl,
                     "вариант": "смежная территория: доля площади в верхних 10%",
                     "совпадение_верх_10": out_share})

        out = wide[["row", "col", "x", "y"]].copy()
        out["score"] = score.astype(np.float32)
        out["score_сглаженный"] = score_sm.astype(np.float32)
        out["новизна"] = outside.astype(np.float32)
        out["верхние_10"] = top_w
        out["спектр_измерен"] = keep_w
        out["в_габаритах_листа"] = in_sheet
        out.to_parquet(
            config.PROCESSED_DIR / ("forecast_wide_v3%s.parquet" % suffix),
            index=False)

        log("против карты 05.09 (forecast_wide_v2): спирмен %.3f, общих ячеек в "
            "верхних 10%% %.0f%%"
            % (sps.spearmanr(t_old, score_sm).statistic,
               100 * (top_w & (t_old >= np.quantile(t_old, 1 - area))).sum()
               / max(top_w.sum(), 1)))

        # --------------------------------------------------------------
        fig, axs = plt.subplots(1, 4, figsize=(22.5, 6.6))
        im = axs[0].imshow(pct(score_sm), origin="upper", extent=ext,
                           cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                           interpolation="nearest")
        mapstyle.outline(axs[0], top_w, WIDE_META.shape, ext, flip=True)
        axs[0].set_title("Перенос на смежную территорию: модель,\n"
                         "обученная на всех метках листа,\n%s" % ttl, fontsize=10)
        cb = fig.colorbar(im, ax=axs[0], fraction=0.045,
                          ticks=mapstyle.CLASS_EDGES,
                          label="класс перспективности, процентиль по сетке")
        cb.ax.tick_params(labelsize=8)

        cat = np.zeros(wide.shape[0])
        cat[top_w] = 1
        cat[crit_top > 0.5] = 2
        cat[top_w & (crit_top > 0.5)] = 3
        axs[1].imshow(cat.reshape(WIDE_META.shape), origin="upper", extent=ext,
                      cmap=ListedColormap(["#e9e9ec", "#4c78a8", "#e4a11b",
                                           "#c0392b"]), vmin=0, vmax=3,
                      interpolation="nearest")
        axs[1].set_title("Верхние 10%% широкой сетки и критериальные\n"
                         "верхние 10%% листа: вне листа занято %.1f%% площади"
                         % (100 * out_share), fontsize=10)
        axs[1].legend(handles=[Patch(color="#c0392b", label="оба метода"),
                               Patch(color="#4c78a8", label="только перенос"),
                               Patch(color="#e4a11b", label="только критериальный"),
                               Patch(color="#e9e9ec", label="ни один")],
                      loc="lower left", fontsize=8, framealpha=0.9)

        im = axs[2].imshow(outside.reshape(WIDE_META.shape), origin="upper",
                           extent=ext, cmap="magma", vmin=0,
                           vmax=float(np.quantile(outside, 0.99)),
                           interpolation="nearest")
        axs[2].set_title("Новизна признаков: расстояние до знакомого\n"
                         "по листу, вне рамки медиана %.2f"
                         % np.median(outside[~in_sheet]), fontsize=10)
        fig.colorbar(im, ax=axs[2], fraction=0.045,
                     label="в единицах типичного расстояния внутри листа")

        axs[3].imshow(keep_w.reshape(WIDE_META.shape).astype(float),
                      origin="upper", extent=ext,
                      cmap=ListedColormap(["#e9e9ec", "#4c9a6a"]), vmin=0, vmax=1,
                      interpolation="nearest")
        axs[3].set_title("Где спектр измерил породу (открытый\n"
                         "грунт от 2%%): %.0f%% площади" % (100 * keep_w.mean()),
                         fontsize=10)

        for ax in axs:
            ax.plot(box_x, box_y, color="#101010", lw=1.4)
            ax.scatter(cx[allp] / 1000.0, cy[allp] / 1000.0, s=13,
                       facecolors="none", edgecolors="#101010", linewidths=0.7,
                       zorder=3)
            ax.scatter(cx[unb] / 1000.0, cy[unb] / 1000.0, s=28, marker="^",
                       facecolors="#22d3a7", edgecolors="#0b3b32",
                       linewidths=0.6, zorder=4)
            ax.set_xlabel("X, км")
            ax.set_ylabel("Y, км")
        axs[0].legend(handles=[
            Line2D([], [], color="#101010", lw=1.4,
                   label="границы основного листа"),
            Line2D([], [], marker="o", color="none", markeredgecolor="#101010",
                   label="точки опробования, %d" % allp.size),
            Line2D([], [], marker="^", color="none", markerfacecolor="#22d3a7",
                   markeredgecolor="#0b3b32",
                   label="из них непредвзятые, %d" % unb.size)],
            loc="lower left", fontsize=8, framealpha=0.9)

        fig.suptitle(
            "Перенос итоговой модели на смежную территорию, %s: обучение на "
            "всех метках основного листа, широкая сборка приведена\n"
            "к шкале листа поквантильно. Внутри чёрной рамки есть критериальный "
            "расчёт, снаружи проверять прогноз нечем — там показана\n"
            "только осторожность модели" % ttl, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        mapstyle.save_fig(fig, outdir.png("transfer_map%s.png" % suffix))
        plt.close(fig)
        log("OK: %s" % outdir.png("transfer_map%s.png" % suffix)
            .relative_to(ROOT))

    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics" / "transfer_map.csv",
                              index=False)


if __name__ == "__main__":
    main()
