"""P1+P2: честная валидация на НЕЗАВИСИМЫХ геохимических аномалиях.

Тест-сет — ячейки, пересечённые слоями `геохимические ореолы` и `привнос урана`,
которых НЕ видела ни одна модель (обучение только на точках опробования Au/U).
Сравниваются критериальный baseline ГИС Интегро (geo_score) и модели под малые
данные. Метрика — lift попадания независимых аномалий в top-N% площади, с
bootstrap-95%% ДИ и разницей к критериальному.

Запуск из корня: python3 -m experiments.p1p2
"""

import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import OneClassSVM

from src import config
from src.data_loader import find_base_dir, load_all_layers, load_layer, to_crs_safe
from src.features import build_grid, build_features, compute_geo_score
from src.model import BackgroundEnsemble, mark_presence, sample_presence_background

SEEDS = (1, 7, 13)
AREAS = (0.10, 0.20)
N_BOOT = 3000
SHP = Path("data/Gis-integro/shp_dbf")


def cells_intersecting(grid: gpd.GeoDataFrame, gdf: gpd.GeoDataFrame) -> set[int]:
    """cell_id ячеек сетки, пересечённых геометриями слоя."""
    j = gpd.sjoin(grid[["cell_id", "geometry"]], gdf[["geometry"]], predicate="intersects", how="inner")
    return set(j["cell_id"].astype(int).unique())


def lift(score: np.ndarray, test_pos: np.ndarray, area: float) -> float:
    thr = np.quantile(score, 1.0 - area)
    return float((score[test_pos] >= thr).mean()) / area


def train_scores(grid, pos_cells) -> dict[str, np.ndarray]:
    """Обучить все модели на presence-background и вернуть score по всей сетке.

    Для ML усредняем по нескольким фонам (SEEDS) — снижает шум выбора фона.
    """
    X_all = grid[config.FEATURE_COLS].fillna(0).to_numpy()
    scores: dict[str, list] = {name: [] for name in
                               ("Логистическая L2", "MaxEnt-аппрокс (poly2+L2)",
                                "One-Class SVM", "Gradient Boosting", "Ансамбль RF+GB")}
    pos_mask = grid["cell_id"].isin(pos_cells).to_numpy()
    X_pos = X_all[pos_mask]
    for seed in SEEDS:
        sample_pos, y = sample_presence_background(grid, pos_cells, config.TRAIN_N_BACKGROUND, seed)
        Xs = X_all[sample_pos]

        lr = Pipeline([("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))])
        lr.fit(Xs, y)
        scores["Логистическая L2"].append(lr.predict_proba(X_all)[:, 1])

        me = Pipeline([("poly", PolynomialFeatures(2, interaction_only=False, include_bias=False)),
                       ("sc", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.3))])
        me.fit(Xs, y)
        scores["MaxEnt-аппрокс (poly2+L2)"].append(me.predict_proba(X_all)[:, 1])

        oc = Pipeline([("sc", StandardScaler()), ("clf", OneClassSVM(nu=0.3, gamma="scale"))])
        oc.fit(X_pos)  # presence-only
        scores["One-Class SVM"].append(oc.decision_function(X_all))

        gb = GradientBoostingClassifier(n_estimators=config.GB_N_ESTIMATORS, max_depth=config.GB_MAX_DEPTH,
                                        learning_rate=config.GB_LEARNING_RATE, subsample=config.GB_SUBSAMPLE,
                                        random_state=seed)
        gb.fit(Xs, y)
        scores["Gradient Boosting"].append(gb.predict_proba(X_all)[:, 1])

        ens = BackgroundEnsemble(random_state=seed)
        ens.fit(Xs, y)
        scores["Ансамбль RF+GB"].append(ens.predict_proba(X_all)[:, 1])

    return {k: np.mean(v, axis=0) for k, v in scores.items()}


def main() -> None:
    t0 = time.time()
    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, points = load_all_layers(base / config.SHP_SUBDIR, Path(ad))
    grid, _mu, shape = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = compute_geo_score(grid, shape)
    grid, pos = mark_presence(grid, points)
    pos_pos = np.where(grid["cell_id"].isin(pos).to_numpy())[0]
    print(f"[{time.time()-t0:4.0f}s] сетка {len(grid)} ячеек, обучающих точек (опробование): {len(pos)}")

    # независимые тест-сеты (ячейки, пересечённые аномалиями), без обучающих ячеек
    tests = {}
    for label, fn in [("геохим. ореолы (независим.)", "геохимические ореолы.shp"),
                      ("привнос урана (независим.)", "привнос урана.shp"),
                      ("опробование (IN-SAMPLE!)", "геохимическое_опробование.shp")]:
        gdf = to_crs_safe(load_layer(SHP / fn), layers["mask"].crs)
        cell_ids = cells_intersecting(grid, gdf)
        tp = np.where(grid["cell_id"].isin(cell_ids).to_numpy())[0]
        if "IN-SAMPLE" not in label:
            tp = np.array([i for i in tp if i not in set(pos_pos)])  # убрать обучающие ячейки
        tests[label] = tp
        print(f"           {label:32s}: {len(tp)} тест-ячеек")

    model_scores = {"ГИС Интегро (критериальный)": grid["geo_score"].to_numpy()}
    model_scores.update(train_scores(grid, pos))
    crit = model_scores["ГИС Интегро (критериальный)"]
    rng = np.random.default_rng(42)

    for test_label, tp in tests.items():
        print(f"\n=== {test_label}  (n={len(tp)}) ===")
        print(f"{'модель':30s} " + "  ".join(f"lift@{int(a*100)}% [95% ДИ]" for a in AREAS) + "   Δ vs критериальный@10%")
        for name, sc in model_scores.items():
            row = f"{name:30s} "
            for a in AREAS:
                l = lift(sc, tp, a)
                boot = np.array([lift(sc, tp[rng.integers(0, len(tp), len(tp))], a) for _ in range(N_BOOT)])
                row += f"{l:4.2f} [{np.quantile(boot,0.025):4.2f},{np.quantile(boot,0.975):4.2f}]  "
            if name != "ГИС Интегро (критериальный)":
                idx = np.arange(len(tp))
                dboot = np.array([(lambda s: lift(sc, tp[s], 0.10) - lift(crit, tp[s], 0.10))
                                  (rng.integers(0, len(tp), len(tp))) for _ in range(N_BOOT)])
                lo, hi = np.quantile(dboot, [0.025, 0.975])
                sig = "ЗНАЧИМ" if (lo > 0 or hi < 0) else "—"
                row += f"  {dboot.mean():+4.2f} [{lo:+4.2f},{hi:+4.2f}] {sig}"
            print(row)
    print(f"\n[{time.time()-t0:4.0f}s] готово.")


if __name__ == "__main__":
    main()
