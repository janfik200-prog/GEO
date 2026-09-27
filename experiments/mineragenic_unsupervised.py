"""Подбор метода обучения без учителя для минерагенического набора докачки.

Зачем именно здесь. Набор `dataset_mineragenic_own.parquet` (34 признака,
собранных нами, без единого слоя организации) — единственный из трёх, где нет
входов критериальной формулы, а значит единственный, где карта не может
оказаться пересборкой формулы. С учителем он дал согласие с критериальной
картой 15% и при этом лучшее попадание в точки заверки (7 из 19 против 4 у
формулы) — то есть сигнал в нём есть, но он не про формулу. Обучение без
учителя снимает последнюю зависимость: мишень (результат формулы) в обучении
больше не участвует вообще, точки заверки остаются целиком на заверку.

Что сравнивается (список зафиксирован ДО прогона, 14 кандидатов на наборе 34):

* лестница нетипичности `src.atypicality` как есть — махаланобис, невязка PCA,
  изолирующий лес, одноклассовый SVM, автоэнкодер, LOF, расстояние до k-го
  соседа, правдоподобие смеси гауссиан, ранговый ансамбль над ними;
* кластеризация как источник карты — k-средних (расстояние до ближайшего
  центроида и редкость кластера) и пространственно-связный Уорд, который даёт
  СВЯЗНЫЕ зоны, а не рассыпанные ячейки (блок 10 техсхемы, районирование);
* предметный граф — два варианта, где вклад задаётся не числом колонок, а
  геологическим классом из `mineragenic_feature_set.json` (изменения
  околорудные 9, окисление сульфидов 14, структурный контроль 8, ловушка
  палеодолины 3). Без этого 13 колонок ASTER автоматически перевешивают
  3 колонки палеодолины, и «нетипичность» оказывается нетипичностью спектра.
  Это прямой ответ на записанный в CLAUDE.md итог прошлой попытки: результат
  без предметного графа был статистически рабочим, но необъяснимым.

Как выбирается победитель. НЕ по согласию с критериальной картой (цель —
прогноз, а не копия формулы) и не по ROC-AUC против мишени формулы, они
считаются справочно. Решают три вещи: (1) lift на 19 непредвзятых точках
заверки, (2) доля тренда R2 — не выродилась ли карта в пятно посреди листа,
(3) устойчивость при переобучении на половине листа — прокси переносимости на
широкую территорию, ради которой этот набор и собирался.

Оговорки, которые нельзя терять при чтении таблицы.

* ASTER покрывает 70% листа, пропуски импутируются медианой
  (`atypicality.prepare_matrix`), то есть ячейки вне съёмки автоматически
  «типичны». Поэтому те же 14 кандидатов считаются и на ядре из 18 признаков
  со сплошным покрытием; там этого искажения нет.
* все 19 непредвзятых точек заверки лежат ВНУТРИ съёмки ASTER, занимающей
  69.7% площади. Значит карта, которая просто предпочитает контур съёмки,
  получает lift около 1.43 даром. Поэтому считается и `lift_в_ASTER` — тот же
  lift, но с порогом по ячейкам внутри съёмки, где это преимущество снято, —
  и в протокол добавлен контроль `контроль_контур_ASTER`: случайный скор
  внутри съёмки, минус бесконечность снаружи. Кандидат, не бьющий этот
  контроль, не нашёл ничего, кроме границы чужой сцены.
* формальный вердикт (`superior`) выносится только по кандидатам набора 34 с
  поправкой alpha = 0.05/14; для ядра 18 вердикт не выносится — разрешение
  сдвигового теста (1/501) не позволяет опустить порог ниже 0.002.
* нетипичность != рудоносность. Карта имеет смысл только вместе с заверкой по
  точкам и разбором, какие признаки гонят топ (`feature_contributions`).

Выход: `outputs/metrics/mineragenic_unsup_run.csv`, `..._pa.csv`,
`..._verdict.csv`, `..._contrib.csv`, `outputs/mineragenic_unsup_map.png`,
`data/processed/mineragenic_unsup_scores.parquet`.

Второй раунд (`... mineragenic_unsupervised rank`). Первый раунд показал, что
у ВСЕХ детекторов топ-10% гонит один столбец `lin_dens`: сдвиг медиан 7-12
робастных z против 0.8-1.0 у остальных признаков, то есть карта нетипичности
оказалась картой плотности линеаментов. Причина известна и описана в
`prepare_matrix`: нуль-раздутые колонки с вырожденным IQR масштабируются по
СКО, и редкие большие значения остаются доминирующими. Ранговое представление
(нормальные оценки по обучающей выборке) уравнивает вклад колонок по
построению; список кандидатов при этом не меняется, чтобы раунды сравнивались.

Запуск из корня репозитория: ``python -m experiments.mineragenic_unsupervised``
(второй раунд — с аргументом ``rank``).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy import stats as sps  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.feature_extraction.image import grid_to_graph  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from src import (assessment, atypicality, cell_mask, config,  # noqa: E402
                 crit_reference, criterial_target)
from experiments.honest_forecast_final import trend_share  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402

#: Число кластеров у всех кластерных кандидатов. 8 — компромисс: меньше даёт
#: зоны крупнее рудного узла, больше дробит лист мельче разрешения признаков.
K_CLUSTERS = 8
#: Множитель добивки при равенстве редкости: порядок внутри кластера задаётся
#: удалённостью от его центроида, но никогда не перебивает саму редкость.
TIE_EPS = 1e-6
#: Раунд: без аргумента — сырые робастные z, `rank` — нормальные оценки рангов.
MODE = sys.argv[1] if len(sys.argv) > 1 else "z"
TAG = "mineragenic_unsup" + ("_rank" if MODE == "rank" else "")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rarity_score(labels: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Редкость кластера как мера перспективности (руда — редкое событие).

    Внутри кластера все ячейки равны по редкости, поэтому порядок доопределяет
    удалённость от центроида — иначе порог «верхние 10%» резал бы кластер
    произвольно, по порядку строк.
    """
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    freq = cnt[inv] / labels.size
    return -np.log(freq) + TIE_EPS * sps.rankdata(dist) / dist.size


def kmeans_maps(X: np.ndarray, seed: int):
    """Две карты из k-средних: расстояние до центроида и редкость кластера."""
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=seed).fit(X)
    d = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    return d, rarity_score(km.labels_, d), km.labels_


def ward_map(X: np.ndarray, valid: np.ndarray, shape: tuple[int, int]):
    """Пространственно-связный Уорд: кластеры обязаны быть связными зонами."""
    conn = grid_to_graph(shape[0], shape[1], mask=valid.reshape(shape))
    lab = AgglomerativeClustering(n_clusters=K_CLUSTERS, connectivity=conn,
                                  linkage="ward").fit_predict(X)
    cen = np.stack([X[lab == c].mean(axis=0) for c in np.unique(lab)])
    d = np.linalg.norm(X - cen[lab], axis=1)
    return rarity_score(lab, d), lab


def block_maps(X: np.ndarray, blocks: dict, seed: int):
    """Предметный граф: каждый геологический класс голосует с равным весом.

    `блоки_ранги` — свой изолирующий лес на каждый класс, объединение по
    средним рангам: класс из 3 колонок весит столько же, сколько класс из 14.
    `блоки_pca_kmeans` — каждый класс сжимается до <=3 главных компонент, и уже
    на этом сжатом представлении ищутся кластеры; редкость кластера — карта.
    """
    ranks, parts = {}, []
    for name, idx in blocks.items():
        s = IsolationForest(n_estimators=config.ANOM_IF_N_ESTIMATORS,
                            random_state=seed, n_jobs=-1).fit(X[:, idx])
        ranks[name] = -s.score_samples(X[:, idx])
        z = PCA(n_components=min(3, idx.size), random_state=seed).fit_transform(X[:, idx])
        parts.append(z / (z.std(axis=0, keepdims=True) + 1e-9))
    _, rar, _ = kmeans_maps(np.column_stack(parts), seed)
    return atypicality.rank_ensemble(ranks), rar


def induced(name: str, Xfit: np.ndarray, Xall: np.ndarray, seed: int) -> np.ndarray:
    """Та же модель, но обученная на подмножестве ячеек и применённая ко всем.

    Нужна только для проверки устойчивости: в `src.atypicality` детекторы
    обучаются на всей сетке, а вопрос «переносится ли карта на территорию,
    которую модель не видела» требует индуктивной формы. Константы берутся из
    тех же `src.config`, чтобы формы не разъезжались.
    """
    if name == "махаланобис":
        from sklearn.covariance import MinCovDet
        return MinCovDet(random_state=seed).fit(Xfit).mahalanobis(Xall)
    if name == "автоэнкодер":
        from sklearn.neural_network import MLPRegressor
        c = np.clip(Xfit, -config.ANOM_CLIP_Z, config.ANOM_CLIP_Z)
        ae = MLPRegressor(hidden_layer_sizes=(config.ANOM_AE_HIDDEN,
                                              config.ANOM_AE_BOTTLENECK,
                                              config.ANOM_AE_HIDDEN),
                          max_iter=config.ANOM_AE_MAX_ITER, tol=config.ANOM_AE_TOL,
                          early_stopping=False, random_state=seed).fit(c, c)
        a = np.clip(Xall, -config.ANOM_CLIP_Z, config.ANOM_CLIP_Z)
        return ((a - ae.predict(a)) ** 2).mean(axis=1)
    if name == "k_й_сосед":
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=config.ANOM_KNN_K, n_jobs=-1).fit(Xfit)
        return nn.kneighbors(Xall)[0][:, -1]
    if name == "невязка_pca":
        p = PCA(n_components=config.ANOM_PCA_VAR, svd_solver="full").fit(Xfit)
        return ((Xall - p.inverse_transform(p.transform(Xall))) ** 2).mean(axis=1)
    if name == "изолирующий_лес":
        f = IsolationForest(n_estimators=config.ANOM_IF_N_ESTIMATORS,
                            random_state=seed, n_jobs=-1).fit(Xfit)
        return -f.score_samples(Xall)
    if name == "смесь_гауссиан":
        g = GaussianMixture(n_components=config.ANOM_GMM_COMPONENTS,
                            covariance_type="full", reg_covar=config.ANOM_GMM_REG,
                            random_state=seed).fit(Xfit)
        return -g.score_samples(Xall)
    if name in ("kmeans_расстояние", "kmeans_редкость"):
        km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=seed).fit(Xfit)
        lab = km.predict(Xall)
        d = np.linalg.norm(Xall - km.cluster_centers_[lab], axis=1)
        return d if name.endswith("расстояние") else rarity_score(lab, d)
    raise KeyError(name)


#: Кандидаты, для которых устойчивость измерима: остальные (LOF, k-й сосед,
#: Уорд, ранговые ансамбли над ними) трансдуктивны по построению — модели,
#: которую можно применить к новой территории, у них не существует.
STABLE = ("махаланобис", "невязка_pca", "изолирующий_лес", "автоэнкодер",
          "k_й_сосед", "смесь_гауссиан", "kmeans_расстояние", "kmeans_редкость")


def build_maps(X: np.ndarray) -> dict:
    """Карты лестницы нетипичности как есть (обучение на всей сетке)."""
    out = {}
    t = time.time()
    for name, fn in (("махаланобис", atypicality.robust_mahalanobis),
                     ("невязка_pca", atypicality.pca_residual),
                     ("изолирующий_лес", atypicality.isoforest_score),
                     ("одноклассовый_svm", atypicality.ocsvm_score),
                     ("автоэнкодер", atypicality.shallow_ae_score),
                     ("lof", atypicality.lof_score),
                     ("k_й_сосед", atypicality.knn_distance_score),
                     ("смесь_гауссиан", atypicality.gmm_nll_score)):
        out[name] = np.asarray(fn(X))
        log(f"    {name:20s} ({time.time() - t:.0f} c)")
        t = time.time()
    out["ранговый_ансамбль"] = atypicality.rank_ensemble(dict(out))
    return out


def main() -> None:
    meta, prognoz_grid = criterial_target.load_prognoz_grid()
    prognoz = prognoz_grid.ravel()
    y = criterial_target.training_positives(prognoz_grid).astype(int)
    shape = (meta.prf, meta.pic)
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_mineragenic_own.parquet")
    spec = json.loads((config.PROCESSED_DIR / "mineragenic_feature_set.json")
                      .read_text(encoding="utf-8"))
    valid = cell_mask.build_valid_mask(criterial_target.training_features("v5_rebuilt"))
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    basis = np.column_stack([np.ones(prognoz.size), rbf_basis(cx, cy, RBF_GRID)])
    top_c = prognoz <= np.quantile(prognoz[pool], config.CREF_AREA)
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    ast_cov = valid & df["ast_aloh"].notna().to_numpy()
    pool_ast = np.flatnonzero(ast_cov)

    all_cols = [c for c in df.columns if c not in ("row", "col", "x", "y")]
    core = [c for c in spec["ядро_полное_покрытие"] if c in all_cols]
    sets = {"34": all_cols, f"ядро_{len(core)}": core}
    west = valid & (cx < np.median(cx[valid]))
    log(f"раунд {MODE}: валидных ячеек {int(valid.sum())}, непредвзятых точек {unb.size}; "
        f"наборы: 34 признака и ядро {len(core)} со сплошным покрытием")

    rows, contrib = [], []
    scores = {"criterial": crit_reference.criterial_score(prognoz)}
    wards = {}
    for tag, cols in sets.items():
        X = atypicality.prepare_matrix(df[cols], valid)
        if MODE == "rank":
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                    random_state=config.ANOM_SEED).fit_transform(X)
        blocks = {}
        for cls, names in spec["классы"].items():
            idx = np.array([cols.index(n) for n in names if n in cols], dtype=int)
            if idx.size:
                blocks[cls] = idx
        log(f"набор {tag}: {X.shape[1]} признаков, классов {len(blocks)} "
            f"({', '.join(f'{k}:{v.size}' for k, v in blocks.items())})")
        maps = build_maps(X)
        d_km, rar_km, lab_km = kmeans_maps(X, config.ANOM_SEED)
        maps["kmeans_расстояние"], maps["kmeans_редкость"] = d_km, rar_km
        maps["ward_связный"], wards[tag] = ward_map(X, valid, shape)
        maps["блоки_ранги"], maps["блоки_pca_kmeans"] = block_maps(
            X, blocks, config.ANOM_SEED)
        log(f"    кластерные и блочные карты готовы, кандидатов {len(maps)}")

        Xw = X[west[valid]]
        east = valid & ~west
        for name, s_valid in maps.items():
            s = atypicality.expand_to_grid(s_valid, valid)
            top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
            agr = crit_reference.agreement(s, scores["criterial"], pool,
                                           config.CREF_AREA)
            cov, realized = assessment.coverage_and_area(
                s, unb, config.VER_AREA, pool=pool)
            cov_a, real_a = assessment.coverage_and_area(
                s, unb, config.VER_AREA, pool=pool_ast)
            rho_st = np.nan
            if name in STABLE:
                s_half = induced(name, Xw, X, config.ANOM_SEED)
                e = east[valid]
                rho_st = float(sps.spearmanr(s_half[e], s_valid[e]).statistic)
            rows.append({
                "набор": tag, "метод": name, "признаков": len(cols),
                "lift_10": cov / realized if realized > 0 else 0.0,
                "точек_из_19": int(round(cov * unb.size)),
                "lift_в_ASTER": cov_a / real_a if real_a > 0 else 0.0,
                "доля_топа_в_ASTER": float(ast_cov[s >= np.quantile(s[pool], 0.9)].mean()),
                "доля_тренда_R2": trend_share(s, basis, valid),
                "устойчивость_ро": rho_st,
                "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                "roc_auc_мишень": roc_auc_score(y[pool], s[pool]),
                "kappa": agr["kappa"], "rho_с_формулой": agr["rho"],
            })
            scores[f"{name}_{tag}"] = s
            contrib.append(atypicality.feature_contributions(X, s_valid, cols)
                           .head(5).assign(набор=tag, метод=name))
            log(f"  {tag:8s} {name:20s} lift={rows[-1]['lift_10']:.2f} "
                f"({rows[-1]['точек_из_19']}/19) "
                f"тренд={rows[-1]['доля_тренда_R2']:.3f} уст={rho_st:.3f} "
                f"согл={rows[-1]['совпадение_верх_10']:.1%}")

    rng = np.random.default_rng(config.VER_SEED)
    scores["случайный_контроль"] = np.where(valid, rng.random(valid.size), -np.inf)
    scores["контроль_контур_ASTER"] = np.where(ast_cov, rng.random(valid.size), -np.inf)

    res = pd.DataFrame(rows).sort_values(["набор", "lift_10"], ascending=[True, False])
    res.to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_run.csv",
               index=False, encoding="utf-8-sig")
    pd.concat(contrib).to_csv(
        ROOT / "outputs" / "metrics" / f"{TAG}_contrib.csv",
        index=False, encoding="utf-8-sig")
    print("\n=== Обучение без учителя на минерагеническом наборе докачки ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("справочно: с учителем на этом же наборе lift 3.68 (7/19), согласие 15.2%; "
          "критериальный метод lift 2.10 (4/19)")

    own = [k for k in scores if k.endswith("_34")]
    sub = {k: v for k, v in scores.items()
           if k.endswith("_34") or k in ("criterial", "случайный_контроль",
                                         "контроль_контур_ASTER")}
    prot = assessment.run_protocol(sub, own + ["случайный_контроль",
                                               "контроль_контур_ASTER"], meta, valid,
                                   0.05 / len(own), baseline="criterial", log=log)
    prot["verdict"].to_csv(
        ROOT / "outputs" / "metrics" / f"{TAG}_verdict.csv",
        index=False, encoding="utf-8-sig")
    prot["pa"].to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_pa.csv",
                      index=False, encoding="utf-8-sig")
    print("\n=== Заверка по точкам (вердикт только для набора 34) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = pd.DataFrame({"row": df["row"], "col": df["col"], "x": cx, "y": cy,
                        "prognoz": prognoz, "valid": valid,
                        **{k: v for k, v in scores.items() if k != "criterial"}})
    out["кластер_ward_34"] = np.nan
    out.loc[valid, "кластер_ward_34"] = wards["34"]
    out.to_parquet(config.PROCESSED_DIR / f"{TAG}_scores.parquet",
                   index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = res[(res["набор"] == "34") & (res["доля_тренда_R2"] <= 0.93)]
    best = (sel.iloc[0]["метод"] if len(sel) else res.iloc[0]["метод"])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    axes[0].imshow(np.where(valid, prognoz, np.nan).reshape(shape), origin="upper",
                   cmap="magma_r")
    axes[0].set_title("Критериальный метод")
    # Показ в перцентилях: у махаланобиса и невязок хвост тяжёлый, на линейной
    # шкале видны только несколько экстремальных ячеек и карта выглядит пустой.
    disp = np.full(valid.size, np.nan)
    disp[valid] = sps.rankdata(scores[f"{best}_34"][valid]) / int(valid.sum())
    axes[1].imshow(disp.reshape(shape), origin="upper", cmap="magma")
    axes[1].contour(np.nan_to_num(disp >= 0.9).reshape(shape).astype(float),
                    levels=[0.5], colors="cyan", linewidths=0.6)
    axes[1].set_title(f"Без учителя: {best}, перцентили\n"
                      "34 признака докачки; голубым — верхние 10%")
    axes[2].imshow(np.where(valid, atypicality.expand_to_grid(wards["34"], valid),
                            np.nan).reshape(shape), origin="upper", cmap="tab10")
    axes[2].set_title(f"Районирование: связный Уорд, {K_CLUSTERS} зон\n"
                      "те же 34 признака")
    pr, pc = np.divmod(unb, meta.pic)
    for ax in axes:
        ax.scatter(pc, pr, s=16, c="white", edgecolors="black", linewidths=0.5,
                   label="точки заверки (коренные, U/TR)")
        ax.set_xlabel("столбец сетки (шаг 500 м)")
    axes[0].set_ylabel("строка сетки")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"{TAG}_map.png", dpi=130)
    log(f"OK: outputs/{TAG}_map.png")


if __name__ == "__main__":
    main()
