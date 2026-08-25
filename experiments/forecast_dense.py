"""Плотная (поячеечная) регрессия критериального `prognoz` вместо 3-объектной схемы.

Пользователь явно отошёл от object-level LOO (`src/criterial_target.py`,
3 псевдо-объекта): вместо этого учим модель на ВСЕЙ сетке `prognoz.pgrid`
(22 946 ячеек) как на непрерывном таргете. Первая версия скрипта (блочный
сплит 10×10 км, случайные 64 блока) получила ревью `model-critic` и была
отклонена по трём причинам — все исправлены здесь:

1. **Служебные признаки-утечка.** `*_n_obs`/`*_valid_frac` (число сцен и доля
   валидных пикселей в композите — геометрия чужих орбит и тайлов съёмки, не
   свойство площади) не были отфильтрованы в `criterial_target.training_features`
   (фильтр `cell_mask.is_service` там не применяется, в отличие от
   `features_v2.feature_columns`). `s1_n_obs` вышел топ-признаком (важность
   0.43) с r=-0.637 к таргету — оба поля имеют широкий гладкий тренд по
   площади листа, модель частично «предсказывала» prognoz по карте покрытия
   съёмкой, а не по геологии. Исправлено: `cell_mask.is_service` применяется
   явно, наравне с `DROP_GROUPS`.
2. **Блок 10×10 км меньше носителя сглаживающих фильтров признаков** (окна
   `_25`/`_35` = 12.5/17.5 км) и не имитирует перенос на смежный лист —
   тестовая ячейка всегда оставалась ближе 5 км к обучающей. Исправлено:
   основной протокол — leave-one-strip-out (полосы 4 шт по оси X и отдельно
   по оси Y листа 74.5×77 км, буферная дедзона 15 км между train/test) —
   это репетиция ИМЕННО экстраполяции на соседнюю площадь, а не интерполяции
   внутри известного паттерна.
3. **Impurity-важности RF/GB смещены и посчитаны in-sample.** Исправлено:
   отчёт по важности — permutation importance на honest held-out полосе.

Также добавлены baseline'ы, без которых Spearman/lift не интерпретируемы:
модель только на координатах (x, y) — если она почти не хуже полной модели,
результат объясняется трендом, а не признаками; ближайший обучающий сосед
(1-NN по координатам) — сила чистой пространственной автокорреляции, полная
модель обязана его превосходить, а не совпадать с ним.

Признаки: `training_features(dataset="v5_rebuilt")` минус группы `ast`/`ls`
(совместимость с `dataset_wide.parquet`) минус служебные (`cell_mask.is_service`).
`geo`/`geo2` уже исключены внутри `training_features`.

ВАЖНАЯ ОГОВОРКА, не решаемая методом (см. ревью model-critic): таргет —
экспертная карта `prognoz`, а не факт оруденения. Высокий Spearman означает
«критериальный прогноз воспроизводим по дистанционке и геофизике», а не
«модель нашла рудный узел» — заверка возможна только через пункт 5 (реальные
точки) и смежные листы, которых критериальный анализ не видел.

Выход: `outputs/forecast_dense_map.png` (сшитая карта основного листа: каждая
X-полоса — прогноз модели, обученной на остальных X-полосах вне буфера),
`data/processed/forecast_dense_wide.parquet` + `outputs/forecast_dense_wide_map.png`
(прод-модель, обученная на всём листе, применена к смежной территории —
экстраполяция, не проверена вне листа).

Запуск из корня репозитория: ``python -m experiments.forecast_dense``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cell_mask, config, criterial_target, features_v2  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

DROP_GROUPS = ("ast", "ls")
N_STRIPS = 4
STRIP_BUFFER_M = 15_000.0

#: 33 признака с неположительной честной permutation importance на 8 честных
#: leave-one-strip-out фолдах (`experiments/feature_relevance_check.py`,
#: `outputs/metrics/feature_relevance_v5_rebuilt.csv`), несколько — с явно
#: отрицательным вкладом. Совпадают почти один в один с парами |r|>0.9 из
#: корреляционной проверки. Подтверждено `experiments/feature_pruned_check.py`:
#: удаление этих 33 не портит ось X (-0.102->-0.103) и разворачивает ось Y
#: из отрицательной (-0.116) в слабоположительную (+0.093), R2 лучше на обеих.
NOISE_FEATURES = [
    "pf_gr_up1", "gm_mag_flt_35", "psr_linci", "pf_mag_up10", "gm_gr_1GY_25",
    "l8_lst", "s2_b8a", "l8_ferrous", "pf_mag_up5", "pf_mag_up2", "pf_gr_asig",
    "gm_mag_1GFI_25", "pf_gr_vz", "gm_mg_all", "pf_mag_up1", "s2_ferrous",
    "pf_gr_tdr", "s2_bare_frac", "s2_ireci", "pf_mag_tdr", "pf_mag_vz",
    "l8_lst_std", "pf_gm_corr", "l8_iron_ox_std", "psr_hv_std", "s2_b05_std",
    "psr_hh_hv_std", "psr_linci_std", "s2_b03", "l8_swir16_std", "s2_b04_std",
    "s2_b12_std", "gm_mag_2G_25",
]

#: Два пула признаков для сравнения (см. `main`): "full" — стандартный
#: "разрешённый" набор (v5_rebuilt минус geo/geo2 циркулярные минус ast/ls
#: минус служебные), "pruned" — тот же минус NOISE_FEATURES ("наши отсортированные").
POOLS = {
    "full": (),
    "pruned": tuple(NOISE_FEATURES),
}


class SimpleRegEnsemble(RegressorMixin, BaseEstimator):
    """Бэггинг RF+GB по нескольким сидам — регрессионный аналог `BackgroundEnsemble`.

    В отличие от `BackgroundEnsemble` (классификация presence/background) здесь
    нет фона для подвыборки — таргет `prognoz` непрерывный и определён на каждой
    ячейке. Но усреднение по `config.ENS_N_ROUNDS` (=8) сидам RF+GB, как и там,
    снижает дисперсию прогноза за счёт декорреляции членов ансамбля: 8 раундов
    x (RF+GB) = 16 моделей, а не одна пара RF+GB на весь фолд.

    Наследуется от sklearn `BaseEstimator`/`RegressorMixin` для совместимости с
    `permutation_importance` (нужны `__sklearn_tags__`/`get_params`, иначе падает
    на проверке `is_regressor`).
    """

    def __init__(self, random_state: int = 42, n_rounds: int = config.ENS_N_ROUNDS):
        self.random_state = random_state
        self.n_rounds = n_rounds

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.rfs_, self.gbs_ = [], []
        for _ in range(self.n_rounds):
            seed = int(rng.integers(np.iinfo(np.int32).max))
            self.rfs_.append(RandomForestRegressor(
                n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                random_state=seed, n_jobs=-1,
            ).fit(X, y))
            self.gbs_.append(GradientBoostingRegressor(
                n_estimators=config.GB_N_ESTIMATORS, max_depth=config.GB_MAX_DEPTH,
                learning_rate=config.GB_LEARNING_RATE, subsample=config.GB_SUBSAMPLE,
                random_state=seed,
            ).fit(X, y))
        return self

    def predict(self, X):
        preds = [m.predict(X) for m in self.rfs_ + self.gbs_]
        return np.mean(preds, axis=0)

    @property
    def feature_importances_(self):
        imps = [m.feature_importances_ for m in self.rfs_ + self.gbs_]
        return np.mean(imps, axis=0)


def strip_id_and_edges(coord: np.ndarray, n_strips: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(coord.min(), coord.max(), n_strips + 1)
    strip_id = np.clip(np.digitize(coord, edges[1:-1]), 0, n_strips - 1)
    return strip_id, edges


def lift_at(score: np.ndarray, perspective_mask: np.ndarray, pool: np.ndarray, area: float) -> float:
    """Lift охвата перспективных ячеек (`prognoz<=0.15`) в top-`area` по `score`, внутри `pool`."""
    if pool.sum() == 0:
        return float("nan")
    thr = np.quantile(score[pool], 1.0 - area)
    hit = perspective_mask[pool] & (score[pool] >= thr)
    denom = max(1, perspective_mask[pool].sum())
    return (hit.sum() / denom) / area


def nn_baseline_predict(coord_train: np.ndarray, y_train: np.ndarray, coord_test: np.ndarray) -> np.ndarray:
    """1-NN по координатам: сила чистой пространственной автокорреляции без единого признака."""
    tree = cKDTree(coord_train)
    _, idx = tree.query(coord_test, k=1)
    return y_train[idx]


def leave_one_strip_out(axis_name: str, coord: np.ndarray, xy: np.ndarray, X: np.ndarray, y: np.ndarray,
                         perspective: np.ndarray, feat_names: list[str],
                         seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прогон leave-one-strip-out по одной оси; печатает метрики модели и baseline'ов.

    Возвращает сшитые честные скоры трёх кандидатов: дерево-ансамбль (RF+GB,
    как раньше), Ridge (проверка гипотезы "переобучение сложной модели на
    локальный шум мешает экстраполяции") и их среднее ("blend" — хедж между
    режимами, где выигрывает то одна, то другая модель).
    """
    strip_id, edges = strip_id_and_edges(coord, N_STRIPS)
    stitched = np.full(y.size, np.nan)
    stitched_ridge = np.full(y.size, np.nan)
    stitched_blend = np.full(y.size, np.nan)
    print(f"\n=== Leave-one-strip-out по оси {axis_name} ({N_STRIPS} полосы, буфер {STRIP_BUFFER_M/1000:.0f} км) ===")
    rows = []
    for s in range(N_STRIPS):
        lo, hi = edges[s], edges[s + 1]
        test_mask = strip_id == s
        train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
        if test_mask.sum() == 0 or train_mask.sum() < 50:
            continue

        model = SimpleRegEnsemble(random_state=seed)
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        stitched[test_mask] = pred

        coord_model = SimpleRegEnsemble(random_state=seed)
        coord_model.fit(xy[train_mask], y[train_mask])
        pred_coord = coord_model.predict(xy[test_mask])

        pred_nn = nn_baseline_predict(xy[train_mask], y[train_mask], xy[test_mask])

        # Ridge на тех же 95 признаках, что и полная модель: если линейная
        # регуляризованная модель обгоняет RF+GB-ансамбль на честном held-out,
        # значит проблема в переобучении сложной модели на локальный шум, а
        # не в отсутствии сигнала в данных как таковых.
        scaler = StandardScaler().fit(X[train_mask])
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(scaler.transform(X[train_mask]), y[train_mask])
        pred_ridge = ridge.predict(scaler.transform(X[test_mask]))
        stitched_ridge[test_mask] = pred_ridge

        pred_blend = (pred + pred_ridge) / 2.0
        stitched_blend[test_mask] = pred_blend

        rho, _ = spearmanr(pred, y[test_mask])
        rho_coord, _ = spearmanr(pred_coord, y[test_mask])
        rho_nn, _ = spearmanr(pred_nn, y[test_mask])
        rho_ridge, _ = spearmanr(pred_ridge, y[test_mask])
        rho_blend, _ = spearmanr(pred_blend, y[test_mask])
        test_idx = np.where(test_mask)[0]
        lift10 = lift_at(stitched, perspective, test_idx, 0.10)
        n_persp = perspective[test_mask].sum()
        rows.append((s, test_mask.sum(), train_mask.sum(), n_persp,
                      rho, rho_coord, rho_nn, rho_ridge, rho_blend, lift10))

    df = pd.DataFrame(rows, columns=["strip", "n_test", "n_train", "n_perspective",
                                      "spearman_full", "spearman_xy_only", "spearman_1nn",
                                      "spearman_ridge", "spearman_blend", "lift@10%"])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for name, arr in [("дерево (RF+GB)", stitched), ("ridge", stitched_ridge), ("blend", stitched_blend)]:
        valid = ~np.isnan(arr)
        rho_all, _ = spearmanr(arr[valid], y[valid])
        print(f"Сшитая карта [{name}] (все полосы честно held-out): Spearman={rho_all:.3f}, "
              f"R2={r2_score(y[valid], arr[valid]):.3f}")

    # интерпретируемость: permutation importance на held-out полосе 0 (не in-sample impurity)
    lo, hi = edges[0], edges[1]
    test_mask = strip_id == 0
    train_mask = (coord < lo - STRIP_BUFFER_M) | (coord > hi + STRIP_BUFFER_M)
    model = SimpleRegEnsemble(random_state=seed)
    model.fit(X[train_mask], y[train_mask])
    perm = permutation_importance(model, X[test_mask], y[test_mask], n_repeats=10,
                                   random_state=seed, scoring="r2")
    imp = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
    print(f"Permutation importance (на held-out полосе 0, ось {axis_name}), топ-10:")
    print(imp.head(10).to_string())

    # Диагностика: какие признаки нестационарны по площади (сдвиг среднего
    # held-out полосы относительно train, в std train) - кандидаты на причину
    # блочного смещения между полосами (модель применяет "средний" паттерн
    # train там, где реальный уровень признака систематически иной).
    train_mean = X[train_mask].mean(axis=0)
    train_std = X[train_mask].std(axis=0) + 1e-9
    test_mean = X[test_mask].mean(axis=0)
    shift = np.abs(test_mean - train_mean) / train_std
    shift_s = pd.Series(shift, index=feat_names).sort_values(ascending=False)
    print(f"Сдвиг среднего (held-out полоса 0 относительно train, в std train), топ-10 ({axis_name}):")
    print(shift_s.head(10).to_string())

    return stitched, stitched_ridge, stitched_blend


def run(pool: str) -> None:
    exclude = POOLS[pool]
    suffix = "" if pool == "full" else f"_{pool}"
    print(f"\n{'='*70}\nПул признаков: '{pool}'"
          + (f" (доп. исключены {len(exclude)}: {exclude[:3]}...)" if exclude else " (стандартный, без доп. исключений)")
          + f"\n{'='*70}")

    meta, prognoz = criterial_target.load_prognoz_grid()
    prognoz_flat = prognoz.ravel().astype(float)
    y = -prognoz_flat
    perspective = prognoz_flat <= config.CRIT_TARGET_THRESHOLD
    print(f"Основной лист: {y.size} ячеек, перспективных (prognoz<={config.CRIT_TARGET_THRESHOLD}): "
          f"{perspective.sum()}")

    feat_df = criterial_target.training_features(dataset="v5_rebuilt")
    feat_names = [c for c in feat_df.columns
                  if features_v2.feature_group(c) not in DROP_GROUPS and not cell_mask.is_service(c)
                  and c not in exclude]
    n_service_dropped = sum(1 for c in feat_df.columns if cell_mask.is_service(c))
    print(f"Признаков: {len(feat_names)} (исключены группы {DROP_GROUPS}, "
          f"{n_service_dropped} служебных *_n_obs/*_valid_frac, geo/geo2 — уже вне training_features, "
          f"{len(exclude)} доп. по пулу '{pool}')")
    X = feat_df[feat_names].fillna(0).to_numpy()

    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    xy = np.column_stack([cx, cy])

    stitched_x, ridge_x, blend_x = leave_one_strip_out("X", cx, xy, X, y, perspective, feat_names, config.CRIT_SEED)
    stitched_y, ridge_y, blend_y = leave_one_strip_out("Y", cy, xy, X, y, perspective, feat_names, config.CRIT_SEED)

    # Заверка на точках, никогда не участвовавших ни в prognoz, ни в обучении
    # (73 ячейки: геохим. аномалии Au/U + 4 проявления, см. criterial_target.load_real_points)
    point_idx = criterial_target.load_real_points(meta)
    if point_idx.size > 0:
        print(f"\n=== Заверка на {point_idx.size} точках внешнего опробования (не в prognoz, не в обучении) ===")
        baseline_score = -prognoz_flat
        candidates = [
            ("dense_x", stitched_x), ("dense_y", stitched_y),
            ("ridge_x", ridge_x), ("ridge_y", ridge_y),
            ("blend_x", blend_x), ("blend_y", blend_y),
            ("criterial", baseline_score),
        ]
        rows = []
        for area in config.CRIT_AREAS:
            row = {"area": area}
            for name, score in candidates:
                thr = np.quantile(score, 1.0 - area)
                cov = float((score[point_idx] >= thr).mean())
                row[f"lift_{name}"] = cov / area
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("\nТочки внешнего опробования не найдены (load_real_points вернул пусто)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def extent(m):
        return [m.x0, m.x0 + m.pic * m.dx, m.y0, m.y_top]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, stitched, name in zip(axes, [stitched_x, stitched_y], ["X", "Y"]):
        im = ax.imshow(stitched.reshape(meta.shape), origin="upper", cmap="viridis", extent=extent(meta))
        ax.set_title(f"Leave-one-strip-out по оси {name}\n(полосы {N_STRIPS}, буфер {STRIP_BUFFER_M/1000:.0f} км)")
        ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
        fig.colorbar(im, ax=ax, label="score (-prognoz)", fraction=0.045)
    fig.suptitle("Плотная регрессия prognoz — честный out-of-sample прогноз (без служебных признаков)")
    fig.tight_layout()
    fig.savefig(config.PROJECT_ROOT / "outputs" / f"forecast_dense{suffix}_map.png", dpi=130)
    print(f"\nOK: outputs/forecast_dense{suffix}_map.png")

    if point_idx.size > 0:
        px, py = cx[point_idx], cy[point_idx]
        fig2, axes2 = plt.subplots(1, 3, figsize=(21, 7))
        panels = [("dense_x", stitched_x), ("dense_y", stitched_y), ("criterial", baseline_score)]
        for ax, (name, score) in zip(axes2, panels):
            im = ax.imshow(score.reshape(meta.shape), origin="upper", cmap="viridis", extent=extent(meta))
            ax.scatter(px, py, s=22, facecolors="none", edgecolors="red", linewidths=1.2,
                       label=f"точки заверки (n={point_idx.size})")
            ax.set_title(name)
            ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
            ax.legend(loc="upper right", fontsize=8)
            fig2.colorbar(im, ax=ax, fraction=0.045)
        fig2.suptitle("Точки внешней заверки (геохим. аномалии Au/U + 4 проявления) поверх карт score")
        fig2.tight_layout()
        fig2.savefig(config.PROJECT_ROOT / "outputs" / f"forecast_dense{suffix}_points_map.png", dpi=130)
        print(f"OK: outputs/forecast_dense{suffix}_points_map.png")

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        lift_df = pd.DataFrame(rows).set_index("area")
        lift_df.plot(kind="bar", ax=ax3)
        ax3.axhline(1.0, color="k", linewidth=0.8, linestyle="--", label="случайный уровень")
        ax3.set_ylabel("lift (покрытие точек / area)")
        ax3.set_xlabel("area (доля топ-N% score)")
        ax3.set_title(f"Lift на {point_idx.size} точках внешней заверки")
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        fig3.savefig(config.PROJECT_ROOT / "outputs" / f"forecast_dense{suffix}_points_lift.png", dpi=130)
        print(f"OK: outputs/forecast_dense{suffix}_points_lift.png")

        # Прямое сравнение с критериальной картой: сырые score не сопоставимы
        # (регрессия сжимает диапазон -0.8..-0.2, критериальный -0.9..-0.1, свои
        # цветовые шкалы на каждой панели искажают восприятие) -> переводим оба
        # в перцентильный ранг (0-100%, "топ-N% листа") на общей шкале, и
        # объединяем dense_x/dense_y в одну ML-карту, чтобы сравнивать 1:1.
        def pct_rank(arr: np.ndarray) -> np.ndarray:
            valid = ~np.isnan(arr)
            out = np.full(arr.shape, np.nan)
            out[valid] = rankdata(arr[valid]) / valid.sum() * 100.0
            return out

        dense_combined = np.nanmean(np.vstack([stitched_x, stitched_y]), axis=0)
        rank_dense = pct_rank(dense_combined)
        rank_crit = pct_rank(baseline_score)
        diff_rank = rank_dense - rank_crit

        fig4, axes4 = plt.subplots(1, 3, figsize=(21, 7))
        panels4 = [
            ("dense (X+Y, перцентиль)", rank_dense, "viridis", 0, 100),
            ("criterial (перцентиль)", rank_crit, "viridis", 0, 100),
            ("разница рангов: dense − criterial", diff_rank, "RdBu_r", -100, 100),
        ]
        for ax, (name, arr, cmap, vmin, vmax) in zip(axes4, panels4):
            im = ax.imshow(arr.reshape(meta.shape), origin="upper", cmap=cmap,
                            extent=extent(meta), vmin=vmin, vmax=vmax)
            ax.scatter(px, py, s=22, facecolors="none",
                       edgecolors="red" if cmap == "viridis" else "black", linewidths=1.2)
            ax.set_title(name)
            ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
            fig4.colorbar(im, ax=ax, fraction=0.045)
        fig4.suptitle("Прямое сравнение на общей шкале (перцентильный ранг 0-100%): "
                       "объединённая dense-карта (X+Y) vs критериальная")
        fig4.tight_layout()
        fig4.savefig(config.PROJECT_ROOT / "outputs" / f"forecast_dense{suffix}_compare_rank_map.png", dpi=130)
        print(f"OK: outputs/forecast_dense{suffix}_compare_rank_map.png")

        # Тот же формат для blend (RF+GB усреднённый с Ridge) — кандидат на
        # замену дерево-ансамбля после диагностики: Ridge устойчивее при
        # экстраполяции через полосы с сильным сдвигом признаков (см. отчёт
        # "Сдвиг среднего" выше), blend — хедж между режимами, где каждая
        # из моделей выигрывает.
        blend_combined = np.nanmean(np.vstack([blend_x, blend_y]), axis=0)
        rank_blend = pct_rank(blend_combined)
        diff_rank_blend = rank_blend - rank_crit

        fig5, axes5 = plt.subplots(1, 3, figsize=(21, 7))
        panels5 = [
            ("blend RF+GB/Ridge (X+Y, перцентиль)", rank_blend, "viridis", 0, 100),
            ("criterial (перцентиль)", rank_crit, "viridis", 0, 100),
            ("разница рангов: blend − criterial", diff_rank_blend, "RdBu_r", -100, 100),
        ]
        for ax, (name, arr, cmap, vmin, vmax) in zip(axes5, panels5):
            im = ax.imshow(arr.reshape(meta.shape), origin="upper", cmap=cmap,
                            extent=extent(meta), vmin=vmin, vmax=vmax)
            ax.scatter(px, py, s=22, facecolors="none",
                       edgecolors="red" if cmap == "viridis" else "black", linewidths=1.2)
            ax.set_title(name)
            ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
            fig5.colorbar(im, ax=ax, fraction=0.045)
        fig5.suptitle("Прямое сравнение на общей шкале (перцентильный ранг 0-100%): "
                       "blend-карта (RF+GB усреднённый с Ridge) vs критериальная")
        fig5.tight_layout()
        fig5.savefig(config.PROJECT_ROOT / "outputs" / f"forecast_dense{suffix}_blend_compare_rank_map.png", dpi=130)
        print(f"OK: outputs/forecast_dense{suffix}_blend_compare_rank_map.png")

    # прод-модель: весь основной лист -> применение к смежной территории (экстраполяция, не проверена)
    print("\nОбучение прод-модели на всём основном листе -> смежная территория...")
    prod_model = SimpleRegEnsemble(random_state=config.CRIT_SEED)
    prod_model.fit(X, y)

    prod_scaler = StandardScaler().fit(X)
    prod_ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(prod_scaler.transform(X), y)

    wide_path = config.PROCESSED_DIR / "dataset_wide.parquet"
    if not wide_path.exists():
        print(f"\n{wide_path.name} ещё не пересобран под новую WIDE_META (4 листа, "
              f"experiments/fetch_wide_area.py докачивает данные) — пропускаю проекцию "
              f"на смежную территорию, честная часть (strip-CV + заверка) выше уже готова.")
        return
    wide = pd.read_parquet(wide_path)
    missing = [c for c in feat_names if c not in wide.columns]
    if missing:
        raise RuntimeError(f"нет в dataset_wide.parquet: {missing}")

    # лёгкая проверка домена применимости: доля признаков, где диапазон wide
    # выходит далеко за диапазон обучения (лист) - не маска по ячейкам, а
    # общий сигнал "насколько экстраполяция выходит за обучающее облако"
    print("Проверка домена применимости (диапазон признаков: лист vs смежная территория):")
    out_of_range = []
    for c in feat_names:
        lo, hi = feat_df[c].min(), feat_df[c].max()
        span = max(hi - lo, 1e-9)
        w = wide[c].dropna()
        if w.empty:
            continue
        frac_outside = ((w < lo - 0.1 * span) | (w > hi + 0.1 * span)).mean()
        if frac_outside > 0.3:
            out_of_range.append((c, round(100 * frac_outside, 1)))
    if out_of_range:
        print(f"  {len(out_of_range)} признаков из {len(feat_names)} имеют >30% ячеек wide вне "
              f"диапазона (±10%) обучения на листе — прогноз там менее надёжен:")
        for c, pct in sorted(out_of_range, key=lambda t: -t[1])[:10]:
            print(f"    {c}: {pct}% вне диапазона")
    else:
        print("  все признаки в пределах диапазона обучения (±10%)")

    X_wide = wide[feat_names].fillna(0).to_numpy(dtype=float)
    score_wide = prod_model.predict(X_wide)
    score_wide_ridge = prod_ridge.predict(prod_scaler.transform(X_wide))
    score_wide_blend = (score_wide + score_wide_ridge) / 2.0

    out = wide[["row", "col", "x", "y"]].copy()
    out["score"] = score_wide.astype(np.float32)
    out["score_ridge"] = score_wide_ridge.astype(np.float32)
    out["score_blend"] = score_wide_blend.astype(np.float32)
    out_path = config.PROCESSED_DIR / f"forecast_dense{suffix}_wide.parquet"
    out.to_parquet(out_path, index=False)
    print(f"OK: {out_path} {out.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 14))
    for ax, (name, arr) in zip(axes, [("дерево (RF+GB)", score_wide),
                                       ("ridge", score_wide_ridge),
                                       ("blend", score_wide_blend)]):
        im = ax.imshow(arr.reshape(WIDE_META.shape), origin="upper", cmap="viridis",
                        extent=extent(WIDE_META))
        ax.set_title(name)
        ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
        fig.colorbar(im, ax=ax, label="score (-prognoz, сырой)", fraction=0.03)
    fig.suptitle("Плотная регрессия prognoz — прод-модели на смежной территории\n"
                  "(экстраполяция, без служебных признаков, не проверена вне листа)")
    fig.tight_layout()
    fig.savefig(config.PROJECT_ROOT / "outputs" / f"forecast_dense{suffix}_wide_map.png", dpi=130)
    print(f"OK: outputs/forecast_dense{suffix}_wide_map.png")


def main() -> None:
    """Прогон обоих пулов признаков подряд: 'full' (разрешённый, стандартный
    v5_rebuilt минус ast/ls/служебные) и 'pruned' (тот же минус 33 признака с
    неположительной честной важностью, см. NOISE_FEATURES)."""
    pools = sys.argv[1:] or list(POOLS)
    for pool in pools:
        run(pool)


if __name__ == "__main__":
    main()
