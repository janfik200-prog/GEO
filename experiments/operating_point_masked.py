"""Рабочая точка проекта на маскированных спектрах — новая версия карты.

ЧТО ИЗМЕНИЛОСЬ по сравнению с `operating_point_map.py`. Прежняя рабочая точка
считала спектральные признаки Sentinel-2 во всех ячейках листа, включая те, где
поверхность закрыта растительностью и никакой породы спектр не видит. Замер
07.09.2026 показал, что это не мелочь: совпадение с критериальным прогнозом без
тренда 0.279 при спектрах везде против 0.362, если оставить их только на
обнажённых ячейках (`src/spectral_mask.py`, обоснование порога и контроли — там
же в докстринге).

ЧТО ЗА МОДЕЛЬ. Гистограммный бустинг на 37 признаках с измеренной
переносимостью — строго дистанционных, ни одного слоя с геологических карт, —
из которых спектральные оставлены только там, где есть открытый грунт; плюс
признак-флаг «спектр измерен». Обучение по полосам с разрывом 15 км: каждая
ячейка предсказана моделью, не видевшей ни её саму, ни её окрестность. Готовая
карта сглаживается ядром, вытянутым на северо-запад (5 км вдоль простирания,
1.25 км поперёк) — форма из ветки 3, внутри каждой полосы отдельно.

Прежний вариант входа считается здесь же, чтобы обе карты и обе пары чисел
были получены в одном прогоне и одинаковой сборкой.

Выход: `outputs/operating_point_masked.png`,
`outputs/metrics/operating_point_masked.csv`,
`data/processed/operating_point_masked.parquet`.
Запуск из корня: ``python -m experiments.operating_point_masked``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
                 criterial_target, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import smooth_strips  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    base = feat[cols]
    masked = spectral_mask.apply(base)
    keep = spectral_mask.exposure_mask(base)
    print(f"признаков: {base.shape[1]} без маски, {masked.shape[1]} с маской и флагом; "
          f"спектр измерен в {int(keep.sum())} ячейках ({keep.mean():.1%})")

    rows, maps = [], {}
    for name, df in (("спектры везде (прежняя сборка)", base),
                     ("спектры только на обнажённых", masked)):
        s, aucs, per_ax = blind_map(df.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        m = score_map(sm, y, crit, pool, valid, D)
        maps[name] = sm
        rows.append({"вариант": name, "признаков": df.shape[1],
                     "фолды_roc_auc": float(np.mean(aucs)), **m})
        print(f"  {name:32s} совпадение {m['совпадение_верх_10']:.3f}, "
              f"без тренда {m['совпадение_без_тренда']:.3f}, "
              f"каппа {m['каппа']:.3f}, ROC карты {m['карта_roc_auc']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "operating_point_masked.csv", index=False)

    # ------------------------------------------------------------------
    # Новая рабочая точка: карта, метрики, точки заверки
    # ------------------------------------------------------------------
    s = maps["спектры только на обнажённых"]
    s = np.where(np.isfinite(s), s, np.nanmin(s[pool]))
    area = config.CREF_AREA
    top_m = (s >= np.quantile(s[pool], 1 - area)) & valid
    top_c = crit_reference.reference_mask(crit, pool, area) & valid
    sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
    tpr = (sr >= np.quantile(sr[pool], 1 - area)) & valid
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid
    agr = crit_reference.agreement(s, crit, pool, area)
    agree = float((top_m & top_c).sum() / top_c.sum())
    agree_dt = float((tpr & tcr).sum() / tcr.sum())

    out = pd.DataFrame({"row": np.repeat(np.arange(shape[0]), shape[1]),
                        "col": np.tile(np.arange(shape[1]), shape[0]),
                        "x": cx, "y": cy, "score": s.astype(np.float32),
                        "верхние_10": top_m, "валидная": valid})
    out.to_parquet(config.PROCESSED_DIR / "operating_point_masked.parquet", index=False)

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    hit_all, hit_unb = int(top_m[allp].sum()), int(top_m[unb].sum())
    print(f"\nверхние 10%: {int(top_m.sum())} ячеек = "
          f"{top_m.sum() * (meta.dx / 1000) ** 2:.0f} кв. км; "
          f"точек опробования внутри {hit_all} из {allp.size}, "
          f"непредвзятых {hit_unb} из {unb.size}")

    x_ext = (cx.max() - cx.min()) / 1000.0
    y_ext = (cy.max() - cy.min()) / 1000.0
    ext = [0, x_ext, 0, y_ext]
    px, py = (cx - cx.min()) / 1000.0, (cy - cy.min()) / 1000.0

    # Строка 0 сетки — север: imshow с origin по умолчанию рисует её
    # сверху, поэтому переворот массива не нужен — он ставил карту
    # зеркально к точкам опробования, которые кладутся по координатам.
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
    fig, axs = plt.subplots(1, 5, figsize=(26.5, 6.4))

    for ax, v, ttl in (
            (axs[0], crit, "Критериальный прогноз ВНИГНИ\n(эталон сравнения)"),
            (axs[1], maps["спектры везде (прежняя сборка)"],
             "Прежняя сборка:\nспектры посчитаны во всех ячейках"),
            (axs[2], s, "Новая рабочая точка:\nспектры только там, где видна порода")):
        h = ax.imshow(img(v, valid), extent=ext, cmap="magma", vmin=0, vmax=100)
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(h, ax=ax, fraction=0.046, pad=0.02,
                     label="перспективность, процентиль по листу")

    axs[3].imshow(cat_img(top_m, top_c), extent=ext, cmap=cat_cmap, vmin=0, vmax=3)
    axs[3].set_title(f"Верхние 10% площади: где сходятся\n"
                     f"совпадение {agree:.0%}, каппа {agr['kappa']:.2f}, "
                     f"ROC {agr['auc']:.3f}", fontsize=10)
    axs[3].legend(handles=[Patch(color="#c0392b", label="оба метода"),
                           Patch(color="#4c78a8", label="только прогноз"),
                           Patch(color="#e4a11b", label="только критериальный"),
                           Patch(color="#e9e9ec", label="ни один")],
                  loc="lower left", fontsize=8, framealpha=0.9)

    axs[4].imshow(cat_img(tpr, tcr), extent=ext, cmap=cat_cmap, vmin=0, vmax=3)
    axs[4].set_title(f"То же после снятия наклона\n"
                     f"совпадение {agree_dt:.0%} при случайном уровне 10%\n"
                     f"(прежняя сборка давала "
                     f"{rows[0]['совпадение_без_тренда']:.0%})", fontsize=10)

    # Область, где спектр измерен, — контекст к обеим картам верхних 10%
    exp_img = np.where(valid, keep, np.nan).astype(float).reshape(shape)
    for ax in (axs[3], axs[4]):
        ax.contour(np.linspace(0, x_ext, shape[1]), np.linspace(0, y_ext, shape[0]),
                   np.nan_to_num(exp_img), levels=[0.5], colors="#2b2b2b",
                   linewidths=0.35, alpha=0.55)

    for ax in axs:
        ax.scatter(px[allp], py[allp], s=16, facecolors="none",
                   edgecolors="#101010", linewidths=0.8, zorder=3)
        ax.scatter(px[unb], py[unb], s=34, marker="^", facecolors="#22d3a7",
                   edgecolors="#0b3b32", linewidths=0.7, zorder=4)
        ax.set_xlabel("восток, км от края листа")
        ax.set_ylabel("север, км от края листа")
    axs[0].legend(handles=[
        Line2D([], [], marker="o", color="none", markeredgecolor="#101010",
               label=f"точки опробования, {allp.size}"),
        Line2D([], [], marker="^", color="none", markerfacecolor="#22d3a7",
               markeredgecolor="#0b3b32", label=f"из них непредвзятые, {unb.size}")],
        loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Рабочая точка на маскированных спектрах: 37 дистанционных признаков, "
        "обучение по полосам с разрывом 15 км, сглаживание ядром 5×1.25 км на "
        "северо-запад  |  совпадение без тренда "
        f"{rows[0]['совпадение_без_тренда']:.0%} → {agree_dt:.0%}; "
        "тонкий контур на двух правых картах — граница площадей, где спектр "
        "измерил породу", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ROOT / "outputs" / "operating_point_masked.png", dpi=140)
    print("OK: outputs/operating_point_masked.png")


if __name__ == "__main__":
    main()
