"""Методы с учителем из литературы по прогнозной оценке — все на одном протоколе.

ЗАЧЕМ. Прогон `nn_geo_mpm` показал, что нейросети выигрывают не архитектурой, а
тем, что видят окрестность ячейки: бустинг на скользящем среднем признаков
обошёл обе сети по согласию с критериальной картой. Отсюда два вопроса, на
которые этот прогон отвечает разом:

1. какой метод С УЧИТЕЛЕМ лучше на наших данных, если сравнивать их честно —
   одним протоколом, на одних признаках, с одними планками;
2. сохраняется ли выигрыш от окрестности у ВСЕХ семейств методов или это
   свойство одного бустинга.

ЧТО ГОВОРИТ ЛИТЕРАТУРА. Сравнительных работ по прогнозной оценке много, и вывод
у них устойчиво один: универсального победителя нет, но случайный лес — самый
надёжный выбор по умолчанию.

* эпитермальное золото, Нью-Брансуик (Earth Sci Inform, 2025): случайный лес
  ROC-AUC 0.93, опорные векторы 0.92, XGBoost 0.91 — НО по кривой
  «площадь-захват» эффективнее оказался XGBoost, то есть порядок методов
  зависит от метрики;
* лес против опорных векторов при малой выборке (Minerals, 2023, 13(8):1073):
  все варианты опорных векторов точнее на обучении (>89% против 78%), но по
  предсказательной способности лучше лес — классическое переобучение на 20
  объектах;
* Восточная Ботсвана (наша база знаний): лес против CART и градиентного
  бустинга на трёх съёмках;
* Хамиссана, Судан (Minerals, 2023, 13(1):49): лес на 4 наборах признаков,
  негативы отбираются кластеризацией без учителя, а не случайно; главный вывод
  авторов — плотность линеаментов важнее спектральных признаков в любой
  конфигурации;
* обучение по положительным и неразмеченным примерам (positive-unlabeled) и
  полуобучение с учителем — ответ литературы на то, что достоверных «здесь руды
  нет» не существует.

Поэтому в сравнение включены: линейная модель (эквивалент MaxEnt), ближайшие
соседи, опорные векторы с ядром, случайный лес, экстремально рандомизированные
деревья, гистограммный бустинг (наш аналог XGBoost), многослойный перцептрон
(«ANN» этих статей), мешковое обучение по положительным и неразмеченным
примерам и мягкое голосование трёх сильнейших — то самое, что литература
рекомендует вместо выбора одного «лучшего».

ДВА НАБОРА ВХОДОВ. Каждый метод считается дважды: на сырых 37 признаках набора
переноса и на них же плюс скользящие средние в окнах 5x5 (2.5 км) и 15x15
(7.5 км) — 111 столбцов. Это прямая проверка вывода из `nn_geo_mpm_control`.

ПРОТОКОЛ. Полосы 4 по X + 4 по Y, разрыв 15 км между обучением и тестом; рядом
планка «только география». Метрики — и ROC-AUC, и согласие с критериальной
картой по верхним 10% (аналог кривой «площадь-захват» из статей), потому что
именно на этой паре порядок методов в литературе и расходится.

ОГОВОРКА ПО ОПОРНЫМ ВЕКТОРАМ: обучение подвыборкой 6000 ячеек (метод растёт
квадратично по числу примеров, полные 18 тыс. на 8 фолдов не считаются за
разумное время). Это ставит его в чуть худшие условия, чем остальных, и в
таблице помечено.

ЧТО ПОЛУЧИЛОСЬ (прогон 04.09.2026, 19 карт, 8 полос каждая).

По ROC-AUC на фолдах: голосование трёх сильнейших с окрестностью 0.7689,
положительные-неразмеченные с окрестностью 0.7640, голосование на сырых 0.7596,
бустинг с окрестностью 0.7584, экстремально рандомизированные деревья с
окрестностью 0.7571, перцептрон с окрестностью 0.7509, бустинг на сырых 0.7488,
случайный лес на сырых 0.7286, только география 0.6753, ближайшие соседи на
сырых 0.6394. По согласию с критериальной картой порядок другой: голосование с
окрестностью 33.4%, бустинг с окрестностью 31.7%, деревья с окрестностью 30.9%,
случайный лес с окрестностью 30.3%; логистическая модель на сырых — 6.6%, ХУЖЕ
случайного выбора (10%).

Вывод 1: случайный лес — не лучший выбор на этих данных, в отличие от
литературы. На сырых признаках он 0.7286 против 0.7488 у бустинга (парно
-0.0202, 3 победы из 8), а от окрестности он единственный, кто теряет
(-0.0116). Совпадение с рекомендацией литературы есть только в одном:
голосование ансамблей действительно лучше любого одиночного метода.

Вывод 2: окрестность помогает ВСЕМ девяти семействам по согласию с
критериальной картой (прирост от +1.7 до +8.1 процентного пункта, ни одного
минуса) — то есть эффект из `nn_geo_mpm_control` не свойство бустинга, а
свойство задачи. По ROC-AUC картина слабее: прирост положителен у 8 семейств
из 9, но ни у одного не проходит парный тест на 8 полосах (лучшее p=0.055 у
ближайших соседей).

Вывод 3: ни одна карта не бьёт планку «только география» с запасом. Лучший
метод против планки +0.0936 (6 побед из 8, p=0.0547) — на границе. Заверка 19
несмещёнными точками: победителей нет, бустинг на сырых 3.16 против 2.10 у
критериальной карты, p=0.0220 при пороге 0.0167.

Выход: `outputs/metrics/supervised_sweep.csv`,
       `outputs/metrics/supervised_sweep_folds.csv`,
       `outputs/metrics/supervised_sweep_verdict.csv`,
       `outputs/supervised_sweep.png`.
Запуск из корня: ``python -m experiments.supervised_sweep``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.nn_geo_mpm import SEED, blind, log  # noqa: E402
from experiments.nn_geo_mpm_control import WINDOWS, smoothed  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402

SVM_SUBSAMPLE = 6000
N_TREES = 300
PU_BAGS = 20


def qnorm(Xtr: np.ndarray, Xte: np.ndarray):
    """Квантильная нормировка в нормальную шкалу, подогнанная ТОЛЬКО на обучении."""
    qt = QuantileTransformer(n_quantiles=min(1000, len(Xtr)),
                            output_distribution="normal", random_state=SEED)
    return np.nan_to_num(qt.fit_transform(Xtr)), np.nan_to_num(qt.transform(Xte))


def fit_plain(model, scale: bool):
    """Обёртка «обучить и предсказать» под интерфейс `blind`."""
    def fp(tr, allx, X=None):
        Xtr, Xall = X[tr], X[allx]
        if scale:
            Xtr, Xall = qnorm(Xtr, Xall)
        else:
            Xtr, Xall = np.nan_to_num(Xtr), np.nan_to_num(Xall)
        return model().fit(Xtr, Y[tr]).predict_proba(Xall)[:, 1]
    return fp


def fit_svm(tr, allx, X=None):
    """Опорные векторы с ядром RBF на подвыборке обучения (см. оговорку в докстринге)."""
    rng = np.random.default_rng(SEED)
    sub = tr if len(tr) <= SVM_SUBSAMPLE else rng.choice(tr, SVM_SUBSAMPLE, replace=False)
    Xtr, Xall = qnorm(X[sub], X[allx])
    m = SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True,
            random_state=SEED).fit(Xtr, Y[sub])
    return m.predict_proba(Xall)[:, 1]


def fit_pu(tr, allx, X=None):
    """Мешковое обучение по положительным и неразмеченным примерам.

    Литературный ответ на то, что достоверных негативов не бывает: в каждом
    мешке к положительным добавляется случайная порция НЕРАЗМЕЧЕННЫХ ячеек как
    условные отрицательные, результат усредняется по мешкам. Ошибка отдельного
    мешка («здесь на самом деле руда, а мы назвали это фоном») усредняется, а
    не запоминается моделью.
    """
    rng = np.random.default_rng(SEED)
    pos = tr[Y[tr] == 1]
    unl = tr[Y[tr] == 0]
    Xtr_all = np.nan_to_num(X[allx])
    acc = np.zeros(len(allx))
    for b in range(PU_BAGS):
        neg = rng.choice(unl, len(pos), replace=False)
        idx = np.concatenate([pos, neg])
        yb = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        m = ExtraTreesClassifier(n_estimators=60, n_jobs=-1,
                                 random_state=SEED + b).fit(np.nan_to_num(X[idx]), yb)
        acc += m.predict_proba(Xtr_all)[:, 1]
    return acc / PU_BAGS


def fit_vote(tr, allx, X=None):
    """Мягкое голосование леса, бустинга и перцептрона — рекомендация обзоров.

    Голоса усредняются по РАНГАМ, а не по вероятностям: у трёх семейств шкалы
    вероятности несопоставимы, и среднее сырых вероятностей отдало бы вес самому
    «уверенному» методу, а не самому точному.
    """
    from scipy import stats as sps
    parts = []
    for fp in (fit_plain(lambda: RandomForestClassifier(
                   n_estimators=N_TREES, min_samples_leaf=3, n_jobs=-1,
                   class_weight="balanced", random_state=SEED), False),
               fit_plain(lambda: HistBackgroundEnsemble(random_state=SEED), False),
               fit_plain(lambda: MLPClassifier(hidden_layer_sizes=(64, 32),
                                               max_iter=300, random_state=SEED), True)):
        parts.append(sps.rankdata(fp(tr, allx, X=X)))
    return np.mean(parts, axis=0) / len(allx)


METHODS = {
    "логистическая (аналог MaxEnt)":
        (fit_plain(lambda: LogisticRegression(max_iter=1000, class_weight="balanced",
                                              random_state=SEED), True), None),
    "ближайшие соседи (k=25)":
        (fit_plain(lambda: KNeighborsClassifier(n_neighbors=25, n_jobs=-1), True), None),
    "опорные векторы, ядро RBF*": (fit_svm, None),
    "случайный лес":
        (fit_plain(lambda: RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=3,
                                                  n_jobs=-1, class_weight="balanced",
                                                  random_state=SEED), False), None),
    "экстремально рандомиз. деревья":
        (fit_plain(lambda: ExtraTreesClassifier(n_estimators=N_TREES, min_samples_leaf=3,
                                                n_jobs=-1, class_weight="balanced",
                                                random_state=SEED), False), None),
    "гистограммный бустинг":
        (fit_plain(lambda: HistBackgroundEnsemble(random_state=SEED), False), None),
    "перцептрон (ANN литературы)":
        (fit_plain(lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                         random_state=SEED), True), None),
    "положительные-неразмеченные": (fit_pu, None),
    "голосование трёх сильнейших": (fit_vote, None),
}

Y: np.ndarray = np.empty(0)


def main() -> None:
    global Y
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    Y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    cols = [c for c in spec["признаки_перенос"] if c in feat_all.columns]
    X_raw = feat_all[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    top_c = prognoz.ravel() <= np.quantile(prognoz.ravel()[pool], config.CREF_AREA)

    sm = {w: smoothed(X_raw, shape, w) for w in WINDOWS}
    X_ctx = np.hstack([X_raw] + [sm[w] for w in WINDOWS])
    INPUTS = {"сырые 37": X_raw, "с окрестностью 111": X_ctx}
    log(f"входы: сырые {X_raw.shape[1]} столбцов, с окрестностью {X_ctx.shape[1]} "
        f"(окна {WINDOWS} = {', '.join(f'{w * meta.dx / 1000:.1f}' for w in WINDOWS)} км)")

    cache = config.PROCESSED_DIR / "supervised_sweep_cache.npz"
    maps: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        names = list(z["names"])
        maps = {n: z[f"m{i}"] for i, n in enumerate(names)}
        fold_rows = list(z["folds"])
        log(f"карты взяты из кэша {cache.name}")
    else:
        RB = rbf_basis(cx, cy, RBF_GRID)
        todo = [(f"{name} | {inp}", fp, Xv)
                for inp, Xv in INPUTS.items() for name, (fp, _) in METHODS.items()]
        todo.append(("ТОЛЬКО география (планка) | сырые 37",
                     fit_plain(lambda: HistBackgroundEnsemble(random_state=SEED), False), RB))
        for key, fp, Xv in todo:
            cals = []
            for ax, coord in (("X", cx), ("Y", cy)):
                log(f"{key}, ось {ax}")
                _, cal, fr = blind(lambda tr, allx, Xv=Xv: fp(tr, allx, X=Xv),
                                   coord, Y, pool, key, ax)
                cals.append(cal)
                fold_rows += fr
            s = np.nanmean(np.vstack(cals), axis=0)
            maps[key] = np.where(np.isfinite(s), s, np.nanmin(s))
        np.savez(cache, folds=np.array(fold_rows, dtype=object),
                 names=np.array(list(maps), dtype=object),
                 **{f"m{i}": v for i, v in enumerate(maps.values())})
        log(f"кэш сохранён: {cache.name}")

    folds = pd.DataFrame(list(fold_rows))
    folds.to_csv(ROOT / "outputs" / "metrics" / "supervised_sweep_folds.csv", index=False)

    rows = []
    for key, s in maps.items():
        name, inp = key.split(" | ")
        f_ = folds[folds["ключ"] == key]
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        rows.append({"метод": name, "вход": inp,
                     "фолды_roc_auc": float(f_["roc_auc"].mean()),
                     "фолды_roc_std": float(f_["roc_auc"].std()),
                     "карта_roc_auc": roc_auc_score(Y[pool], s[pool]),
                     "карта_pr_auc": average_precision_score(Y[pool], s[pool]),
                     "согласие_каппа": agr["kappa"],
                     "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "supervised_sweep.csv", index=False)
    print("\n=== Методы с учителем, разрыв 15 км: по ROC-AUC на фолдах ===")
    print(res.sort_values("фолды_roc_auc", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Те же методы по согласию с критериальной картой ===")
    print(res.sort_values("совпадение_верх_10", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n* опорные векторы обучались на подвыборке 6000 ячеек — см. докстринг")
    print(f"Планки: случайный ROC-AUC 0.5000, совпадение верх-10% {config.CREF_AREA:.0%}")

    best_roc = res.loc[res["фолды_roc_auc"].idxmax()]
    best_agr = res.loc[res["совпадение_верх_10"].idxmax()]
    # Парные тесты по 8 полосам: вклад окрестности внутри каждого семейства.
    # Сравнение по полосам, а не по средним, — метод сам себе контроль, полосы
    # общие, поэтому разброс между полосами (он большой) не мешает сравнению.
    from scipy import stats as sps
    piv = folds.copy()
    piv[["м", "в"]] = piv["ключ"].str.split(" | ", regex=False, expand=True)
    piv = piv.pivot_table(index=["м", "полоса"], columns="в", values="roc_auc")
    prs = []
    for m, g in piv.groupby(level=0):
        if g.shape[1] < 2 or g.isna().any().any():
            continue
        d = g["с окрестностью 111"].to_numpy() - g["сырые 37"].to_numpy()
        prs.append({"метод": m, "прирост_roc_от_окрестности": d.mean(),
                    "полос_в_плюс": f"{(d > 0).sum()}/{len(d)}",
                    "p_вилкоксон": sps.wilcoxon(d).pvalue})
    pr = pd.DataFrame(prs).sort_values("прирост_roc_от_окрестности", ascending=False)
    pr.to_csv(ROOT / "outputs" / "metrics" / "supervised_sweep_paired.csv", index=False)
    print("\n=== Вклад окрестности внутри семейства (парно по 8 полосам) ===")
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    wide = folds.pivot_table(index="полоса", columns="ключ", values="roc_auc")
    base = "ТОЛЬКО география (планка) | сырые 37"
    best_key = f"{best_roc['метод']} | {best_roc['вход']}"
    print("\n=== Против планок (парно по 8 полосам) ===")
    for a, b in ((best_key, base), (best_key, "гистограммный бустинг | сырые 37"),
                 ("гистограммный бустинг | сырые 37", base),
                 ("случайный лес | сырые 37", "гистограммный бустинг | сырые 37")):
        d = wide[a].to_numpy() - wide[b].to_numpy()
        print(f"{a} − {b}: {d.mean():+.4f}, побед {(d > 0).sum()}/8, "
              f"p={sps.wilcoxon(d).pvalue:.4f}")

    # Заверка точками — только для лучших по каждой из двух метрик и планок,
    # иначе поправка на число опробованных карт съедает всю мощность.
    keys = {f"{best_roc['метод']} | {best_roc['вход']}",
            f"{best_agr['метод']} | {best_agr['вход']}",
            "гистограммный бустинг | сырые 37",
            "ТОЛЬКО география (планка) | сырые 37"}
    sel = {k: maps[k] for k in keys}
    alpha = 0.05 / len(sel)
    log(f"заверка точками для {len(sel)} карт, порог {alpha:.4f}")
    prot = assessment.run_protocol({**sel, "criterial": crit}, list(sel), meta, valid,
                                   alpha, baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "supervised_sweep_verdict.csv",
                           index=False)
    print("\n=== Точки заверки (косвенная проверка) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size  # noqa: E731
    # Панели: планка, тот же метод на сырых признаках, лучший по каждой метрике.
    # Если лучший по ROC-AUC и лучший по согласию совпали, дубль не рисуем.
    show = ["ТОЛЬКО география (планка) | сырые 37", "гистограммный бустинг | сырые 37"]
    for k in (f"{best_roc['метод']} | {best_roc['вход']}",
              f"{best_agr['метод']} | {best_agr['вход']}"):
        if k not in show:
            show.append(k)
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(4.8 * (len(show) + 1), 5.6))
    axes[0].imshow(rank(crit), cmap="magma")
    axes[0].set_title("Критериальный прогноз\n(эталон геологов)")
    for ax, key in zip(axes[1:], show):
        ax.imshow(rank(maps[key]), cmap="magma")
        r_ = res[(res["метод"] == key.split(" | ")[0]) & (res["вход"] == key.split(" | ")[1])].iloc[0]
        ax.set_title(f"{key}\nROC-AUC {r_['фолды_roc_auc']:.3f}, "
                     f"совпадение верх-10% {r_['совпадение_верх_10']:.0%}", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Методы с учителем: прогноз вслепую, обучение не ближе 15 км", fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "supervised_sweep.png", dpi=130)
    log("OK: outputs/supervised_sweep.png")

    out = pd.DataFrame({"row": np.repeat(np.arange(meta.prf), meta.pic),
                        "col": np.tile(np.arange(meta.pic), meta.prf),
                        "x": cx, "y": cy, "prognoz": prognoz.ravel(),
                        "target": Y, "valid": valid,
                        **{k: v.astype(np.float32) for k, v in maps.items()}})
    out.to_parquet(config.PROCESSED_DIR / "supervised_sweep_scores.parquet", index=False)
    log("OK: data/processed/supervised_sweep_scores.parquet")


if __name__ == "__main__":
    main()
