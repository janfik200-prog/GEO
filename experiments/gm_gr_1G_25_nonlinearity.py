"""Где именно нелинейна связь gm_gr_1G_25 <-> целевая переменная.

Повод: ElasticNet (эксперимент `elastic_net_batch_elimination.py`) счёл
`gm_gr_1G_25` худшим из 91 признака (importance -0.055, вылетел в первом же
раунде), а RF (`feature_elimination_precise_21_to_11.py`) наоборот — обрыв
честной метрики на 0.07 при его удалении (13 -> 12 признаков). Расхождение
говорит о нелинейной/пороговой зависимости, которую линейная модель не может
использовать в принципе, а RF — может. Здесь: (1) находим форму зависимости —
data-level (бинированное среднее) и model-level (partial dependence RF) против
линейной аппроксимации; (2) находим диапазон значений признака, где кривая
PDP меняется резче всего (порог); (3) показываем НА КАРТЕ, где на территории
листа лежат ячейки в этом пороговом диапазоне, с точками заверки для
ориентира.

Запуск из корня репозитория: ``python -m experiments.gm_gr_1G_25_nonlinearity``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, criterial_target  # noqa: E402

FEATURE = "gm_gr_1G_25"

CODE_LABELS = {1: "литопробы (рыхлые)", 2: "литопробы (коренные)",
               4: "проявление U/TR", 5: "шлихи Au"}
CODE_COLORS = {1: "#999999", 2: "#1c8fd6", 4: "#c23c6b", 5: "#e08a1e"}


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = -prognoz.ravel().astype(float)

    feat_df = criterial_target.training_features(dataset="v6")
    pool = [c for c in feat_df.columns if not cell_mask.is_service(c)]
    X = feat_df[pool].fillna(0).to_numpy()
    gm = feat_df[FEATURE].fillna(0).to_numpy()
    gm_idx = pool.index(FEATURE)

    print(f"Обучаю RF на всём листе (91 признак) для partial dependence по {FEATURE}...", flush=True)
    model = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.CRIT_SEED, n_jobs=-1,
    ).fit(X, y)

    pdp = partial_dependence(model, X, [gm_idx], kind="average", grid_resolution=60)
    grid_vals = pdp["grid_values"][0]
    pdp_avg = pdp["average"][0]

    # линейная аппроксимация (то, что "видит" ElasticNet)
    slope, intercept = np.polyfit(gm, y, 1)
    lin_line = slope * grid_vals + intercept

    # data-level нелинейность (независимо от любой модели): бинированное среднее
    bins = pd.qcut(gm, 30, duplicates="drop")
    binned = pd.DataFrame({"gm": gm, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        gm_mean=("gm", "mean"), y_mean=("y", "mean"), n=("y", "size"))
    binned["gm_lo"] = [iv.left for iv in binned.index]
    binned["gm_hi"] = [iv.right for iv in binned.index]

    # PDP (RF) оказывается заметно более гладким/плоским, чем сырая
    # data-level зависимость (см. ниже) — сам по себе как индикатор порога
    # ненадёжен (ловит численный шум сглаживания, а не форму). Полная 30-бинная
    # таблица показывает не одну колоколообразную форму, а ДВУГОРБУЮ: целевая
    # переменная растёт от минимума gm_gr_1G_25 к локальному пику (~0.00045),
    # затем проваливается (~0.0007-0.00097) и снова растёт ко второму,
    # глобальному пику (~0.0014), после чего падает на самом хвосте (>0.0016,
    # где мало ячеек и возможен шум выбросов). Поэтому "благоприятный диапазон"
    # — не один контур вокруг argmax, а ВСЕ бины с средним выше медианы
    # целевой переменной по листу (может быть несвязным).
    deriv = np.gradient(pdp_avg, grid_vals)
    median_y = float(np.median(y))
    favorable_bins = binned[binned["y_mean"] > median_y]
    peak_pos = int(binned["y_mean"].to_numpy().argmax())
    peak_val = float(binned["gm_mean"].iloc[peak_pos])
    intervals = []
    lo = None
    prev_hi = None
    for _, row in binned.iterrows():
        if row["y_mean"] > median_y:
            if lo is None:
                lo = row["gm_lo"]
            prev_hi = row["gm_hi"]
        else:
            if lo is not None:
                intervals.append((float(lo), float(prev_hi)))
                lo = None
    if lo is not None:
        intervals.append((float(lo), float(prev_hi)))
    print(f"Глобальный пик данных (лучшее среднее): {FEATURE} ~ {peak_val:.4g}", flush=True)
    print(f"Благоприятные диапазоны (выше медианы целевой переменной по листу, "
          f"{len(intervals)} шт.): {[(round(a, 6), round(b, 6)) for a, b in intervals]}", flush=True)

    band_cells = np.zeros(len(gm), dtype=bool)
    for a, b in intervals:
        band_cells |= (gm >= a) & (gm <= b)
    band_lo, band_hi = intervals[0][0], intervals[-1][1]
    print(f"Ячеек в благоприятных диапазонах: {band_cells.sum()} из {len(gm)} "
          f"({band_cells.mean():.1%})", flush=True)

    pts = assessment.load_verification_points(meta)
    pts_row = pts["cell"].to_numpy() // meta.pic
    pts_col = pts["cell"].to_numpy() % meta.pic

    prognoz_favorable = (prognoz.ravel() <= config.CRIT_TARGET_THRESHOLD).reshape(meta.prf, meta.pic)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_map1 = fig.add_subplot(gs[1, 0])
    ax_map2 = fig.add_subplot(gs[1, 1])

    # --- панель 1: форма зависимости ---
    rng = np.random.RandomState(0)
    sub = rng.choice(len(gm), size=min(6000, len(gm)), replace=False)
    ax_scatter.scatter(gm[sub], y[sub], s=4, alpha=0.15, color="#999999", label="ячейки (подвыборка)")
    ax_scatter.plot(binned["gm_mean"], binned["y_mean"], color="#1c8fd6", linewidth=2.2,
                     marker="o", markersize=4, label="бинированное среднее (данные, 30 бинов)")
    ax_scatter.plot(grid_vals, pdp_avg, color="#c23c6b", linewidth=2.2,
                     label="partial dependence (RF)")
    ax_scatter.plot(grid_vals, lin_line, color="#2f6b34", linewidth=2, linestyle="--",
                     label="линейная аппроксимация (то, что видит ElasticNet)")
    for i, (a, b) in enumerate(intervals):
        ax_scatter.axvspan(a, b, color="#c23c6b", alpha=0.12,
                            label="благоприятные диапазоны (выше медианы)" if i == 0 else None)
    ax_scatter.set_xlabel(f"{FEATURE} (сырое значение)")
    ax_scatter.set_ylabel("целевая переменная (-prognoz, больше = перспективнее)")
    ax_scatter.set_title(f"Форма зависимости {FEATURE} <-> целевая переменная: "
                          "двугорбая (данные), почти плоская (PDP RF), линейная аппроксимация")
    ax_scatter.legend(fontsize=9, loc="best")
    ax_scatter.grid(alpha=0.25)

    # --- панель 2: сырая карта признака ---
    im1 = ax_map1.imshow(gm.reshape(meta.prf, meta.pic), cmap="cividis")
    ax_map1.contour(prognoz_favorable, levels=[0.5], colors="white", linewidths=1.0)
    ax_map1.set_title(f"{FEATURE}, сырые значения\n(белый контур — критериально перспективные ячейки)")
    ax_map1.set_xlabel("столбец сетки (500 м)")
    ax_map1.set_ylabel("строка сетки (500 м)")
    fig.colorbar(im1, ax=ax_map1, fraction=0.04, pad=0.02, label=FEATURE)

    # --- панель 3: карта пороговой полосы + точки заверки ---
    band_img = band_cells.reshape(meta.prf, meta.pic).astype(float)
    im2 = ax_map2.imshow(band_img, cmap="Reds", vmin=0, vmax=1)
    ax_map2.contour(prognoz_favorable, levels=[0.5], colors="#1c8fd6", linewidths=1.2)
    for code in sorted(pts["code"].unique()):
        m = pts["code"].to_numpy() == code
        ax_map2.scatter(pts_col[m], pts_row[m], s=45, color=CODE_COLORS.get(code, "black"),
                        edgecolors="white", linewidths=0.6, label=CODE_LABELS.get(code, str(code)), zorder=5)
    ax_map2.set_title(f"Где лежат благоприятные диапазоны {FEATURE}\n"
                       "(красный фон — выше медианы целевой переменной; синий контур — критериально перспективные)")
    ax_map2.set_xlabel("столбец сетки (500 м)")
    ax_map2.set_ylabel("строка сетки (500 м)")
    ax_map2.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = ROOT / "outputs" / "gm_gr_1G_25_nonlinearity.png"
    fig.savefig(out_path, dpi=150)
    print(f"Сохранено: {out_path}")

    out_csv = ROOT / "outputs" / "metrics" / "gm_gr_1G_25_pdp.csv"
    pd.DataFrame({"gm_gr_1G_25": grid_vals, "pdp_average": pdp_avg,
                  "linear_approx": lin_line, "abs_derivative": np.abs(deriv)}).to_csv(out_csv, index=False)
    print(f"Сохранено: {out_csv}")


if __name__ == "__main__":
    main()
