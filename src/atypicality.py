"""Лестница нетипичности: карта перспективности без меток.

Идея (линия Xiong & Zuo, 2016): рудная минерализация — редкое событие,
её геофизико-спектральный отклик плохо описывается моделью «типичного фона»;
чем хуже ячейка реконструируется/описывается моделью фона, тем она аномальнее.
Меток не требуется вообще — 3 критериальных объекта и точки опробования
уходят целиком в заверку (:mod:`src.assessment`), а не в обучение.

Ступени лестницы (по возрастанию сложности):

1. :func:`robust_mahalanobis` — эллиптический фон (MinCovDet);
2. :func:`pca_residual` — линейное многообразие (невязка реконструкции PCA);
3. :func:`isoforest_score` / :func:`ocsvm_score` — непараметрические детекторы;
4. :func:`shallow_ae_score` — нелинейное многообразие (неглубокий автоэнкодер
   16-3-16 на sklearn MLPRegressor: PyTorch на этой ступени не нужен);
5. :func:`rank_ensemble` — ранговое усреднение выбранных ступеней.

Каждая ступень обязана бить предыдущую на заверке — иначе берётся более
простая (стоп-правила плана фазы). Все функции принимают матрицу признаков
ВАЛИДНЫХ ячеек (см. :mod:`src.cell_mask`) после :func:`prepare_matrix`
и возвращают скор «выше = аномальнее» той же длины.

Осторожно: нетипичность != рудоносность. Вода/края уже сняты маской, но
интерпретация карты обязана идти вместе с заверкой по точкам и сравнением
с критериальным прогнозом как равноправным методом.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def prepare_matrix(feat_df: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
    """Матрица признаков валидных ячеек: импутация медианой + стандартизация.

    Медиана (а не 0, как в fillna(0) прошлой фазы) тянет пропуск к «типичному»
    значению и не создаёт искусственной аномалии — для детекторов нетипичности
    это принципиально. Стандартизация робастная: медиана/IQR, чтобы сами
    аномалии не растягивали шкалу.

    НУЛЬ-РАЗДУТЫЕ КОЛОНКИ. У признака, который равен нулю больше чем в 75%
    ячеек (плотность узлов линеаментов), IQR вырождается до машинного нуля, и
    деление на него давало z ~ 1e14: такая колонка в одиночку определяла и
    расстояние Махаланобиса, и isoforest. Если IQR меньше 1% СКО (у нормального
    распределения IQR ~ 1.35 СКО, так что порог срабатывает только на
    вырождении), масштаб берётся по СКО, у полностью постоянных колонок — 1.
    """
    X = feat_df.to_numpy(dtype=float)[valid]
    med = np.nanmedian(X, axis=0)
    # IQR — ДО импутации (nanpercentile): импутированные медианой значения
    # сжимали бы IQR и раздували масштаб колонок с большим числом пропусков.
    q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
    iqr, sd = q75 - q25, np.nanstd(X, axis=0)
    degenerate = ~(iqr > 0.01 * sd)
    if degenerate.any():
        names = np.asarray(feat_df.columns)[degenerate]
        warnings.warn("вырожденный IQR, масштаб по СКО: " + ", ".join(names))
    scale = np.where(degenerate, sd, iqr)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return (X - med) / scale


def robust_mahalanobis(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Ступень 1: расстояние Махаланобиса от робастного центра (MinCovDet)."""
    from sklearn.covariance import MinCovDet

    mcd = MinCovDet(random_state=config.ANOM_SEED if seed is None else seed)
    mcd.fit(X)
    return mcd.mahalanobis(X)


def pca_residual(X: np.ndarray, var: float | None = None) -> np.ndarray:
    """Ступень 2: невязка реконструкции PCA (фон = линейное многообразие).

    Число компонент — по доле объяснённой дисперсии ``ANOM_PCA_VAR``:
    признаки сильно скоррелированы (трансформанты одного поля), фиксированное
    k было бы произволом.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=var or config.ANOM_PCA_VAR, svd_solver="full")
    Z = pca.fit_transform(X)
    X_rec = pca.inverse_transform(Z)
    return ((X - X_rec) ** 2).mean(axis=1)


def isoforest_score(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Ступень 3а: Isolation Forest (выше = аномальнее)."""
    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(
        n_estimators=config.ANOM_IF_N_ESTIMATORS,
        random_state=config.ANOM_SEED if seed is None else seed,
        n_jobs=-1,
    )
    forest.fit(X)
    return -forest.score_samples(X)


def ocsvm_score(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Ступень 3б: One-Class SVM (rbf), обучение на случайной подвыборке.

    Полный fit на всех ячейках — O(n^2) по ядру; подвыборка
    ``ANOM_OCSVM_TRAIN_N`` статистически достаточна для модели фона
    (фон доминирует в любой случайной выборке), скорится вся сетка.
    """
    from sklearn.svm import OneClassSVM

    rng = np.random.default_rng(config.ANOM_SEED if seed is None else seed)
    n_train = min(config.ANOM_OCSVM_TRAIN_N, X.shape[0])
    train = rng.choice(X.shape[0], size=n_train, replace=False)
    svm = OneClassSVM(nu=config.ANOM_OCSVM_NU, gamma="scale")
    svm.fit(X[train])
    return -svm.score_samples(X)


def shallow_ae_score(
    X: np.ndarray, seed: int | None = None, return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Ступень 4: невязка неглубокого автоэнкодера (16-bottleneck-16).

    Узкое горлышко (``ANOM_AE_BOTTLENECK`` = 2-5) обязательно: широкое
    выучивает тождественное отображение, и невязка вырождается в шум.

    Против недообучения (в v1 lift 0.41 — хуже случайного): вход клипуется
    до ±``ANOM_CLIP_Z`` робастных единиц (хвосты не рвут градиенты) перед fit
    И predict; ``early_stopping=False`` — валидационный сплит 10% на
    мультивыходной регрессии рвал обучение на первых итерациях;
    ``tol=ANOM_AE_TOL``. При ``return_info=True`` возвращает также
    ``{"n_iter": ...}`` для диагностики сходимости.
    """
    from sklearn.neural_network import MLPRegressor

    Xc = np.clip(X, -config.ANOM_CLIP_Z, config.ANOM_CLIP_Z)
    ae = MLPRegressor(
        hidden_layer_sizes=(config.ANOM_AE_HIDDEN, config.ANOM_AE_BOTTLENECK,
                            config.ANOM_AE_HIDDEN),
        activation="tanh",
        max_iter=config.ANOM_AE_MAX_ITER,
        tol=config.ANOM_AE_TOL,
        random_state=config.ANOM_SEED if seed is None else seed,
        early_stopping=False,
    )
    ae.fit(Xc, Xc)
    score = ((Xc - ae.predict(Xc)) ** 2).mean(axis=1)
    if return_info:
        return score, {"n_iter": int(ae.n_iter_)}
    return score


def lof_score(X: np.ndarray, k: int | None = None) -> np.ndarray:
    """Локальная плотность (LOF): аномален тот, кто разрежённее своих соседей.

    Другое семейство, чем ступени 1-4: не «далеко от центра фона», а «локально
    менее плотно, чем окружение». Нужен для проверки, не является ли
    отрицательный результат артефактом одного семейства детекторов.
    """
    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor(n_neighbors=k or config.ANOM_LOF_K, n_jobs=-1)
    lof.fit(X)
    return -lof.negative_outlier_factor_


def knn_distance_score(X: np.ndarray, k: int | None = None) -> np.ndarray:
    """Расстояние до k-го ближайшего соседа — простейшая мера разрежённости.

    Непараметрическая планка внутри лестницы: если сложные детекторы не бьют
    её, вся сложность не оправдана.
    """
    from sklearn.neighbors import NearestNeighbors

    k = k or config.ANOM_KNN_K
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dist, _ = nn.kneighbors(X)
    return dist[:, -1]


def gmm_nll_score(X: np.ndarray, seed: int | None = None,
                  n_components: int | None = None) -> np.ndarray:
    """Минус лог-правдоподобие смеси гауссиан (многомодальный фон).

    Махаланобис предполагает ОДИН эллипсоид фона; при нескольких геологических
    доменах (свиты) это заведомо грубо. Смесь описывает такой фон явно.

    Смесь чувствительна к загрязнению: свободная компонента охотно садится на
    компактную группу выбросов и делает её «нормой». Против этого — пол
    ковариации ``ANOM_GMM_REG``.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components or config.ANOM_GMM_COMPONENTS,
        covariance_type="full",
        reg_covar=config.ANOM_GMM_REG,
        random_state=config.ANOM_SEED if seed is None else seed,
    )
    gmm.fit(X)
    return -gmm.score_samples(X)


def rank_ensemble(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Ступень 5: средний нормированный ранг по ступеням-членам.

    Ранги, а не сырые скоры: шкалы ступеней несопоставимы (расстояние^2,
    невязка, лесная глубина). Выход в [0, 1].
    """
    if not scores:
        raise ValueError("rank_ensemble: пустой словарь скоров")
    n = next(iter(scores.values())).size
    acc = np.zeros(n)
    for s in scores.values():
        acc += stats.rankdata(s) / n
    return acc / len(scores)


def per_domain(fn, X: np.ndarray, domains: np.ndarray, **kwargs) -> np.ndarray:
    """Доменная нормировка: детектор обучается внутри каждого литодомена.

    Снимает эффект «нетипично просто потому, что другая геология»: скор
    ячейки — нормированный ранг внутри своего домена, сопоставимый между
    доменами. ``domains`` — коды из :func:`src.cell_mask.build_domains`,
    уже сжатые до валидных ячеек.
    """
    out = np.empty(X.shape[0])
    for dom in np.unique(domains):
        sel = domains == dom
        s = fn(X[sel], **kwargs)
        out[sel] = stats.rankdata(s) / sel.sum()
    return out


def feature_contributions(
    X: np.ndarray, score: np.ndarray, feature_names: list[str],
    top_frac: float = 0.10,
) -> pd.DataFrame:
    """Какие признаки гонят аномальность: |Δмедиан| робастных z top vs остальные.

    Для ячеек из top-``top_frac`` скора против остальных считается модуль
    разницы медианных робастных z-значений по каждому признаку (вход ``X`` —
    матрица после :func:`prepare_matrix`). Сортировка по убыванию. Это
    артефакт-детектор: если топ тянут каналы Landsat/края покрытия, а не
    геофизика — карта ловит артефакты съёмки, а не геологию.
    """
    score = np.asarray(score, dtype=float)
    thr = np.quantile(score, 1.0 - top_frac)
    top = score >= thr
    diff = np.abs(np.median(X[top], axis=0) - np.median(X[~top], axis=0))
    out = pd.DataFrame({"feature": list(feature_names), "abs_dz": diff,
                        "n_top": int(top.sum())})
    return out.sort_values("abs_dz", ascending=False).reset_index(drop=True)


def expand_to_grid(score_valid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Скор валидных ячеек -> полная сетка; невалидные получают -inf
    (никогда не попадают в top-X% при пороге по пулу валидных)."""
    full = np.full(valid.size, -np.inf)
    full[valid] = score_valid
    return full
