"""Иллюстрации к докладу об обучении: одна картинка на каждый второй слайд.

ЗАЧЕМ. Числа доклада уже посчитаны (`mineragenic_current.py`, `full_sheet_map.py`,
прогоны веток), но лежат в таблицах и в широких четырёхпанельных картах, где
подписи нечитаемы после уменьшения до размера слайда. Здесь собраны компактные
рисунки под конкретные места презентации: каждый рисуется под свой размер на
слайде и не требует переобучения — берутся готовые метрики и дешёвые расчёты по
сетке.

Что получается (`outputs/11_презентации/`):

* `deck_labels_map` — что такое метки: критериальный балл, его верхние 10% и
  точки заверки;
* `deck_trend_correction` — карта, её координатная поверхность и остаток;
* `deck_missing` — доля пропусков по группам признаков (зачем бустинг);
* `deck_branch_intervals` — ветки и интервалы разности с критериальным методом;
* `deck_no_effect` — размер ядра и форма цели: разницы нет;
* `deck_levers` — три рычага, которые двигают метрику;
* `deck_sets_bars` — наборы признаков: на всём листе против слепой проверки;
* `deck_feature_groups` — состав 58 признаков и что из них переносимо;
* `deck_exposure` — где спектр измерил породу;
* `deck_tree` — из чего складывается обучение (схема одного дерева);
* `deck_kernel` — форма ядра: рабочее вытянутое против круглого;
* `deck_smoothing` — балл по ячейкам и он же после сглаживания.

Запуск из корня: ``python -m experiments.training_deck_figures``
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, mapstyle, outputs as outdir, spectral_mask)
from src import neighbourhood as nb  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (make_model,  # noqa: E402
                                               poly_design)
from experiments.spectral_exposure_threshold import (ACROSS_KM,  # noqa: E402
                                                     ALONG_KM,
                                                     NW_AZIMUTH)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

METRICS = ROOT / "outputs" / "metrics"

#: Ориентация карт. В сетке листа строка 0 — север (`GridMeta.cell_centers`,
#: проверено сэмплингом `relief_m.tif` в центрах ячеек: r=0.999 с `dem_elev`).
#: Поэтому растр рисуется с ``origin="upper"``, а контур `mapstyle.outline`
#: берётся с ``flip=True`` — иначе карта выходит зеркальной по вертикали и не
#: сходится с точками заверки, которые ставятся по координатам.
AREA = config.CREF_AREA

GOLD = "#c89b2c"
DARK = "#2b2b2b"
RED = "#b03a2e"
GREEN = "#2f6b4f"
BLUE = "#1f5294"
GREY = "#9a9a9a"

#: Человеческие имена групп признаков (префикс столбца -> подпись).
GROUP_NAMES = {
    "s2": "Sentinel-2",
    "l8": "Landsat 8/9",
    "s1": "Sentinel-1",
    "dem": "рельеф",
    "lin": "линеаменты",
    "relief2": "производные рельефа",
    "dist": "расстояния до гидросети",
    "спектр": "признак «спектр измерен»",
}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 9.5, "figure.dpi": 170})


def save(fig, name: str) -> None:
    mapstyle.save_fig(fig, outdir.png(name), dpi=170)
    print("  %s" % outdir.png(name).relative_to(ROOT), flush=True)


def group_of(col: str) -> str:
    return col.split("_")[0]


def bar_labels(ax, bars, fmt="%.3f", size=9, dy=0.004):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt % b.get_height(), ha="center", va="bottom", fontsize=size)


# ---------------------------------------------------------------- рисунки
def fig_labels(crit, labels, valid, shape, ext, pts_all, pts_unb, cxk, cyk) -> None:
    """Слайд 1: метки — это верхние 10% критериального балла, а не находки."""
    # Контур — ровно те ячейки, на которых учится модель (`training_positives`),
    # а не пересчитанный верхний дециль: у балла есть повторы, и границы
    # разошлись бы с числом меток в докладе.
    top = labels & valid
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    pct = mapstyle.percentile(crit, valid)
    im = ax.imshow(pct.reshape(shape), origin="upper", extent=ext,
                   cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                   interpolation="nearest")
    mapstyle.outline(ax, top, shape, ext, flip=True, color="#101010", lw=1.0)
    ax.plot(cxk[pts_all], cyk[pts_all], "o", markersize=3.4, mfc="none",
            mec="#404040", mew=0.9)
    ax.plot(cxk[pts_unb], cyk[pts_unb], "^", markersize=5.5, mfc="none",
            mec=GREEN, mew=1.3)
    ax.set_title("Критериальный прогноз и метки обучения\n"
                 "чёрная линия — верхние 10%%, это %d ячеек из %d"
                 % (int(top.sum()), int(valid.sum())), fontsize=10.5)
    ax.set_xlabel("восток, км от края листа")
    ax.set_ylabel("север, км от края листа")
    ax.legend(handles=[
        Line2D([], [], color="#101010", lw=1.2, label="верхние 10% — метки"),
        Line2D([], [], ls="none", marker="o", mfc="none", mec="#404040",
               label="точки опробования, 73"),
        Line2D([], [], ls="none", marker="^", mfc="none", mec=GREEN,
               label="непредвзятые, 19")],
        loc="upper left", fontsize=7.5, framealpha=0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                      ticks=list(mapstyle.CLASS_EDGES))
    cb.set_label("класс перспективности, процентиль по листу", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    save(fig, "deck_labels_map.png")


def fig_trend(crit, valid, shape, ext, D) -> None:
    """Слайд 4: карта = координатная поверхность + остаток."""
    pool = np.flatnonzero(valid)
    coef, *_ = np.linalg.lstsq(D[pool], crit[pool], rcond=None)
    surf = D @ coef
    resid = crit - surf
    r2 = 1.0 - np.var(resid[pool]) / np.var(crit[pool])
    tr = (resid >= np.quantile(resid[pool], 1 - AREA)) & valid

    fig, axs = plt.subplots(1, 3, figsize=(11.0, 4.7))
    for ax, (v, title) in zip(axs, [
            (crit, "Критериальная карта"),
            (surf, "Её координатная поверхность\n"
                   "второго порядка, доля объяснения %.2f" % r2),
            (resid, "Остаток: то, что сравнивается")]):
        pct = mapstyle.percentile(v, valid)
        ax.imshow(pct.reshape(shape), origin="upper", extent=ext,
                  cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                  interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("восток, км")
        ax.tick_params(labelsize=8)
    mapstyle.outline(axs[2], tr, shape, ext, flip=True, color="#101010",
                     lw=0.9)
    axs[0].set_ylabel("север, км")
    fig.suptitle("Наклон снимается у каждой карты отдельно: сравниваются "
                 "верхние 10% остатка, а не общий уклон территории",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "deck_trend_correction.png")


def fig_missing(feat, valid) -> None:
    """Слайд 6: сколько в наборе пропусков и откуда они берутся."""
    cols = [c for c in honest_features("перенос") if c in feat.columns]
    X = spectral_mask.apply(feat[cols])
    share = {}
    for c in X.columns:
        g = GROUP_NAMES.get(group_of(c), group_of(c))
        share.setdefault(g, []).append(
            float(np.isnan(X[c].to_numpy(dtype=float)[valid]).mean()))
    names = sorted(share, key=lambda k: -np.mean(share[k]))
    vals = [100 * float(np.mean(share[k])) for k in names]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.barh(names, vals,
                   color=[RED if v > 1 else GREY for v in vals])
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, "%.0f%%" % v,
                va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25 + 2)
    ax.set_xlabel("доля ячеек листа с пропуском, %")
    ax.set_title("Пропуски в наборе переноса:\n"
                 "спектр вне открытого грунта — не ноль,\n"
                 "а «не измеряли»", fontsize=10, linespacing=1.3)
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "deck_missing.png")


def fig_branches() -> None:
    """Слайд 8: интервалы разности с критериальным методом по веткам."""
    src = [
        ("branch4_points_verdict.csv", 0, "Набор переноса 37,\nрабочая точка"),
        ("branch3_points_verdict.csv", 0, "Форма окрестности:\nвытянутое ядро"),
        ("branch1_multiscale_verdict.csv", 0, "Много масштабов +\nграфовая сеть"),
        ("honest_points_verdict.csv", 1, "Много масштабов,\nбез сглаживания"),
    ]
    rows = []
    for fname, i, label in src:
        r = pd.read_csv(METRICS / fname).iloc[i]
        rows.append((label, float(r["delta"]), float(r["ci_lo"]),
                     float(r["ci_hi"])))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ys = np.arange(len(rows))[::-1]
    for y, (label, d, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=DARK, lw=2)
        ax.plot([lo, hi], [y, y], "|", color=DARK, ms=9)
        ax.plot([d], [y], "o", color=GOLD, ms=9, mec=DARK, mew=0.8)
    ax.axvline(0, color=RED, lw=1.4, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("превышение над критериальным методом по 19 точкам заверки")
    ax.set_title("Каждая ветка — свой интервал разности;\n"
                 "ни один не отходит от нуля", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "deck_branches.png")


def fig_no_effect() -> None:
    """Слайд 9: размер ядра и форма цели ничего не меняют."""
    mc = pd.read_csv(METRICS / "mineragenic_current.csv")
    fs = pd.read_csv(METRICS / "full_sheet_map.csv")
    blind = "обучение по полосам с разрывом 15 км"

    ker = mc[mc["сборка"] == blind]
    sets = ["слои организации + факторные", "всё вместе", "только наша докачка",
            "дистанционные 37 (для сравнения)"]
    short = ["организация\n+ факторные", "всё\nвместе", "только наша\nдокачка",
             "дистанционные\n37"]
    a = [float(ker[(ker["набор"] == s) & (ker["ядро_км"] == "5 x 1.25")]
               ["совпадение_без_тренда"].iloc[0]) for s in sets]
    b = [float(ker[(ker["набор"] == s) & (ker["ядро_км"] == "2.5 x 0.625")]
               ["совпадение_без_тренда"].iloc[0]) for s in sets]

    tgt = fs[fs["сборка"] == blind]
    t_names, t_vals = [], []
    for cel, lab in (("порог верхних 10%", "порог\nверхних 10%"),
                     ("регрессия на критериальный балл", "регрессия\nна балл")):
        for ker_km, suff in (("5 x 1.25", "ядро 5 км"),
                             ("2.5 x 0.625", "ядро 2.5 км")):
            row = tgt[(tgt["цель"] == cel) & (tgt["ядро_км"] == ker_km)]
            t_names.append(lab + "\n" + suff)
            t_vals.append(float(row["совпадение_без_тренда"].iloc[0]))

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.8),
                            gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(sets))
    w = 0.38
    bars1 = axs[0].bar(x - w / 2, a, w, color=GOLD, label="ядро 5 x 1.25 км")
    bars2 = axs[0].bar(x + w / 2, b, w, color=GREY, label="ядро вдвое меньше")
    bar_labels(axs[0], list(bars1) + list(bars2), size=8)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(short, fontsize=8)
    axs[0].set_title("Размер сглаживающего ядра", fontsize=10)
    axs[0].legend(fontsize=8, frameon=False)

    bars3 = axs[1].bar(np.arange(len(t_vals)), t_vals, 0.6,
                       color=[GOLD, GREY, GOLD, GREY])
    bar_labels(axs[1], bars3, size=8)
    axs[1].set_xticks(np.arange(len(t_vals)))
    axs[1].set_xticklabels(t_names, fontsize=7.5)
    axs[1].set_title("Форма цели обучения", fontsize=10)

    for ax in axs:
        ax.axhline(0.10, color=RED, ls="--", lw=1.1)
        ax.set_ylim(0, 0.56)
        ax.tick_params(labelsize=8)
    # Подпись оси только слева: у панелей одна и та же шкала.
    axs[0].set_ylabel("совпадение с поправкой на наклон", fontsize=9)
    axs[0].text(3.45, 0.115, "случайный уровень", color=RED, fontsize=7.5,
                ha="right")
    fig.suptitle("Слепая проверка: обе настройки внутри разброса", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "deck_no_effect.png")


def fig_levers() -> None:
    """Слайд 10: три рычага, которые метрику двигают."""
    items = [
        ("Состав признаков", "только наша\nдокачка, 34", 0.108,
         "минерагения\nорганизации, 20", 0.455),
        ("Маска обнажённости", "спектры\nвезде", 0.279, "спектры только\n"
         "на открытом грунте", 0.362),
        ("Сборка сетки", "широкая\nсборка", 0.469, "сборка\nлиста", 0.548),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, (title, l1, v1, l2, v2) in zip(axs, items):
        bars = ax.bar([0, 1], [v1, v2], 0.55, color=[GREY, GOLD])
        bar_labels(ax, bars, size=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([l1, l2], fontsize=8)
        ax.set_title("%s\n+%.3f" % (title, v2 - v1), fontsize=10)
        ax.set_ylim(0, 0.62)
        ax.axhline(0.10, color=RED, ls="--", lw=1.1)
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("совпадение с поправкой на наклон", fontsize=9)
    fig.suptitle("Все три рычага — про данные и сборку карты, "
                 "ни один не про устройство модели", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "deck_levers.png")


def fig_sets() -> None:
    """Слайд 13: на всём листе наборы неразличимы, слепо — расходятся."""
    mc = pd.read_csv(METRICS / "mineragenic_current.csv")
    mc = mc[mc["ядро_км"] == "5 x 1.25"]
    sets = ["слои организации + факторные", "всё вместе", "только наша докачка",
            "дистанционные 37 (для сравнения)"]
    short = ["организация +\nфакторные, 20", "всё вместе, 54",
             "только наша\nдокачка, 34", "дистанционные,\n37 + флаг"]
    full = [float(mc[(mc["набор"] == s) &
                     (mc["сборка"] == "обучение на всём листе")]
                  ["совпадение_без_тренда"].iloc[0]) for s in sets]
    blind = [float(mc[(mc["набор"] == s) &
                      (mc["сборка"] == "обучение по полосам с разрывом 15 км")]
                   ["совпадение_без_тренда"].iloc[0]) for s in sets]
    fig, ax = plt.subplots(figsize=(9.4, 2.9))
    x = np.arange(len(sets))
    w = 0.38
    b1 = ax.bar(x - w / 2, full, w, color=GREY, label="обучение на всём листе")
    b2 = ax.bar(x + w / 2, blind, w, color=GOLD,
                label="слепая проверка по полосам")
    bar_labels(ax, list(b1) + list(b2), size=8.5)
    ax.axhline(0.10, color=RED, ls="--", lw=1.1)
    ax.text(3.45, 0.115, "случайный уровень", color=RED, fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("совпадение\nс поправкой на наклон", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right")
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    save(fig, "deck_sets_bars.png")


def fig_groups() -> None:
    """Слайд 16: из чего состоят 58 признаков и что доживает до переноса."""
    js = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                    .read_text(encoding="utf-8"))
    base, tr = js["признаки"], set(js["признаки_перенос"])
    groups = {}
    for c in base:
        g = GROUP_NAMES.get(group_of(c), group_of(c))
        keep, drop = groups.setdefault(g, [0, 0])
        groups[g] = [keep + int(c in tr), drop + int(c not in tr)]
    names = sorted(groups, key=lambda k: -(groups[k][0] + groups[k][1]))
    keep = [groups[k][0] for k in names]
    drop = [groups[k][1] for k in names]
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    y = np.arange(len(names))
    ax.barh(y, keep, color=GREEN, label="остаётся при переносе")
    ax.barh(y, drop, left=keep, color=GREY, label="убрано: считается не везде")
    for i, (k, d) in enumerate(zip(keep, drop)):
        ax.text(k + d + 0.4, i, str(k + d), va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(np.array(keep) + np.array(drop)) + 3)
    ax.set_xlabel("признаков в группе")
    ax.set_title("58 признаков листа, из них 37 переносимых", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "deck_feature_groups.png")


def fig_wide_coverage() -> None:
    """Слайд 17: готовая карта покрытия съёмками, обрезанная по содержимому.

    Исходный файл нарисован под книжный лист и на слайде отдал бы половину
    поля белым полям, поэтому здесь он только подрезается по краям рисунка.
    """
    src = (ROOT / "outputs" / "07_перенос_на_смежные" /
           "honest_wide_coverage_map.png")
    im = Image.open(src).convert("RGB")
    ink = np.any(np.asarray(im) < 245, axis=2)
    rows, cols = np.where(ink)
    pad = 10
    box = (max(int(cols.min()) - pad, 0), max(int(rows.min()) - pad, 0),
           min(int(cols.max()) + pad, im.width),
           min(int(rows.max()) + pad, im.height))
    out = outdir.png("deck_wide_coverage.png")
    im.crop(box).save(out)
    print("  %s" % out.relative_to(ROOT), flush=True)


def fig_exposure(feat, valid, shape, ext, cx, cy, dx) -> None:
    """Слайд 17: где спектр действительно измерил породу."""
    m = spectral_mask.exposure_mask(feat) & valid
    sheet = 100.0 * m.sum() / valid.sum()

    # Смежная территория — широкая сетка за вычетом рамки основного листа:
    # доля обнажённости внутри листа уже посчитана и её нельзя учитывать дважды.
    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet",
                           columns=["x", "y", "s2_bare_frac"])
    wx, wy = wide["x"].to_numpy(), wide["y"].to_numpy()
    outside = ~((wx > cx.min() - dx / 2) & (wx < cx.max() + dx / 2) &
                (wy > cy.min() - dx / 2) & (wy < cy.max() + dx / 2))
    bare = wide["s2_bare_frac"].to_numpy(dtype=float)[outside]
    wide_share = 100.0 * float(np.mean(np.isfinite(bare) &
                                       (bare >= spectral_mask.BARE_MIN)))

    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    img = np.where(valid, np.where(m, 2.0, 1.0), np.nan)
    ax.imshow(img.reshape(shape), origin="upper", extent=ext,
              cmap=matplotlib.colors.ListedColormap(["#e9e6df", GREEN]),
              vmin=1, vmax=2, interpolation="nearest")
    ax.set_title("Спектр берётся только там, где открыт грунт:\n"
                 "%.1f%% ячеек листа, %.1f%% широкой сетки"
                 % (sheet, wide_share), fontsize=10)
    ax.set_xlabel("восток, км от края листа")
    ax.set_ylabel("север, км от края листа")
    ax.legend(handles=[Patch(facecolor=GREEN, label="спектр измерен"),
                       Patch(facecolor="#e9e6df", label="пропуск: закрыто "
                             "растительностью")],
              loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    save(fig, "deck_exposure.png")
    print("    доля обнажённых: лист %.1f%%, смежная территория %.1f%%"
          % (sheet, wide_share))


def fig_tree(feat) -> None:
    """Слайд «Как устроено обучение»: одно дерево и порядок сложения деревьев.

    Схема, а не расчёт: вопросы в узлах взяты из реальных признаков набора, но
    пороги подписаны условно — задача рисунка объяснить механику, а не показать
    конкретное обученное дерево.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 6.3)
    ax.axis("off")

    def box(x, y, w, h, text, color, fc="#ffffff", size=9.0, bold=False):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                                   edgecolor=color, linewidth=1.3, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=4,
                color=DARK, fontweight="bold" if bold else "normal")

    def edge(x1, y1, x2, y2, label):
        ax.plot([x1, x2], [y1, y2], color=GREY, linewidth=1.1, zorder=2)
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2 + 0.12, label, fontsize=8.5,
                color=GREY, ha="center", va="bottom")

    box(5.0, 5.5, 4.6, 0.72, "доля открытого грунта > 0.31 ?", BLUE)
    edge(3.6, 5.14, 2.2, 4.0, "нет")
    edge(6.4, 5.14, 7.8, 4.0, "да")
    box(2.2, 3.65, 3.4, 0.72, "высота > 480 м ?", BLUE)
    box(7.8, 3.65, 3.6, 0.72, "спектр глин выше нормы ?", BLUE)
    edge(1.3, 3.29, 0.9, 2.2, "нет")
    edge(3.1, 3.29, 3.5, 2.2, "да")
    edge(6.9, 3.29, 6.5, 2.2, "нет")
    edge(8.7, 3.29, 9.1, 2.2, "да")
    for x, val in ((0.9, "0.06"), (3.5, "0.41"), (6.5, "0.23"), (9.1, "0.78")):
        box(x, 1.85, 1.5, 0.62, val, GOLD, fc="#faf5e6", bold=True)
    ax.text(5.0, 1.05, "лист дерева — балл ячейки", fontsize=9, color=GREY,
            ha="center")
    ax.annotate("", xy=(9.3, 0.62), xytext=(0.7, 0.62),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, linewidth=2.0))
    ax.text(5.0, 0.28, "300 деревьев подряд: каждое следующее учится на "
            "ошибке предыдущих,", fontsize=8.5, color=DARK, ha="center")
    ax.text(5.0, -0.05, "вклад каждого умножается на шаг обучения 0.06",
            fontsize=8.5, color=DARK, ha="center")
    ax.set_title("Одно дерево: вопросы о числах признаков, в конце ветви — "
                 "балл\nПризнаков в рабочем наборе %d, строк-ячеек 22 905"
                 % (len([c for c in honest_features("перенос")
                         if c in feat.columns]) + 1), fontsize=10.5,
                 linespacing=1.4)
    save(fig, "deck_tree.png")


def fig_kernel(meta, shape) -> None:
    """Слайд «Ядро»: вес соседей рабочего ядра против круглого той же площади.

    Ядро не рисуется формулой заново — по сетке пропускается единичный импульс
    тем же кодом `nb.aniso_smooth`, что и в производственных картах, поэтому на
    картинке ровно то ядро, которым сглаживаются все карты доклада.
    """
    nr, nc = shape
    r0, c0 = nr // 2, nc // 2
    imp = np.zeros(nr * nc)
    imp[r0 * nc + c0] = 1.0
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    cells = 1000.0 / meta.dx
    round_km = float(np.sqrt(ALONG_KM * ACROSS_KM))
    variants = (
        (ALONG_KM, ACROSS_KM, theta,
         "рабочее ядро: 5 км вдоль северо-запада, 1.25 км поперёк"),
        (round_km, round_km, 0.0,
         "круглое ядро той же площади: %.1f км во все стороны" % round_km),
    )
    half = 20                                   # ячеек = 10 км в каждую сторону
    hk = half * meta.dx / 1000.0

    fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, (along, across, th, title) in zip(axs, variants):
        k = nb.aniso_smooth(imp, shape, along * cells, across * cells,
                            th).reshape(shape)
        win = k[r0 - half:r0 + half + 1, c0 - half:c0 + half + 1]
        win = win / win.max()
        g = np.linspace(-hk, hk, win.shape[0])
        ax.imshow(win, origin="upper", extent=[-hk, hk, -hk, hk],
                  cmap="magma_r", vmin=0, vmax=1, interpolation="bilinear")
        # contour кладёт строку 0 вниз, imshow с origin="upper" — вверх,
        # поэтому для линий уровня массив переворачивается.
        ax.contour(g, g, np.flipud(win), levels=[0.14, 0.37, 0.61], colors=DARK,
                   linewidths=0.7)
        # Белый квадрат с тёмным кантом: на тёмном ядре зелёная
        # метка сливалась с центром и была не видна.
        ax.plot(0, 0, marker="s", markersize=3.5, color="white",
                markeredgecolor=DARK, markeredgewidth=0.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("восток, км")
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("север, км")
    axs[0].text(-9.2, -9.2, "белый квадрат — сама ячейка 500 м",
                fontsize=8.5, color=DARK)
    fig.suptitle("Ядро — это вес соседей: чем темнее, тем сильнее сосед влияет "
                 "на балл ячейки", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "deck_kernel.png")


def fig_smoothing(feat, labels, valid, shape, ext, meta) -> None:
    """Слайд «Сглаживание»: балл модели по ячейкам и он же после ядра.

    Обе карты — из одной обученной модели (обучение на всех метках листа, как
    в итоговой карте), различие на картинке даёт только сглаживание.
    """
    lab = labels.astype(int)
    cols = [c for c in honest_features("перенос") if c in feat.columns]
    X = spectral_mask.apply(feat[cols]).to_numpy(dtype=float)
    pool = np.flatnonzero(valid)
    p = np.full(lab.size, np.nan)
    p[pool] = make_model().fit(X[pool], lab[pool]).predict_proba(X[pool])[:, 1]
    cells = 1000.0 / meta.dx
    sm = nb.aniso_smooth(p, shape, ALONG_KM * cells, ACROSS_KM * cells,
                         nb.theta_from_azimuth(NW_AZIMUTH), mask=valid)
    sm = np.where(np.isfinite(sm), sm, np.nanmin(sm[pool]))

    # Панели друг под другом: на слайде рисунок стоит в колонке справа, и в
    # такой раскладке карты получаются вдвое крупнее, чем при развороте вширь.
    fig, axs = plt.subplots(2, 1, figsize=(4.3, 8.2))
    for ax, (v, title) in zip(axs, (
            (p, "балл по ячейкам, как выходит из модели"),
            (sm, "он же после сглаживания ядром 5 x 1.25 км"))):
        pct = mapstyle.percentile(v, valid)
        ax.imshow(pct.reshape(shape), origin="upper", extent=ext,
                  cmap=mapstyle.class_cmap(), norm=mapstyle.class_norm(),
                  interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("восток, км от края листа", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axs:
        ax.set_ylabel("север, км от края листа", fontsize=9)
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    # Заголовка у рисунка нет намеренно: его роль играет заголовок слайда,
    # а узкой колонке каждая лишняя строка стоит высоты самих карт.
    fig.tight_layout(rect=(0, 0.055, 1, 1.0))
    save(fig, "deck_smoothing.png")


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    crit = crit_reference.criterial_score(prognoz)
    labels = criterial_target.training_positives(prognoz).astype(bool)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    ext = [0, (cx.max() - cx.min()) / 1000.0, 0, (cy.max() - cy.min()) / 1000.0]
    cxk = cx / 1000.0 - cx.min() / 1000.0
    cyk = cy / 1000.0 - cy.min() / 1000.0

    pts = assessment.load_verification_points(meta)
    p_all = assessment.filter_points(pts["cell"].to_numpy(), valid)
    p_unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    print("Рисунки доклада:")
    fig_labels(crit, labels, valid, shape, ext, p_all, p_unb, cxk, cyk)
    fig_trend(crit, valid, shape, ext, D)
    fig_missing(feat, valid)
    fig_branches()
    fig_no_effect()
    fig_levers()
    fig_sets()
    fig_groups()
    fig_exposure(feat, valid, shape, ext, cx, cy, meta.dx)
    fig_wide_coverage()
    fig_tree(feat)
    fig_kernel(meta, shape)
    fig_smoothing(feat, labels, valid, shape, ext, meta)


if __name__ == "__main__":
    main()
