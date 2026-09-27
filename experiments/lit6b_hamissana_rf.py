"""Ветка 6B, статья 1: Хамиссана (Судан) — Random Forest на трёх сенсорах.

ЗАЧЕМ. Блок 6B технологической схемы требует не «вообще посмотреть литературу»,
а под каждую статью собрать мини-датасет ТОЛЬКО из её признаков, воспроизвести
её метод на наших данных и сравнить результат с критериальным тестом. Первая
статья выбрана потому, что в ней всё выписано до формул:

Mohamed Taha A.M., Xi Y., He Q., Hu A., Wang S., Liu X. Investigating the
Capabilities of Various Multispectral Remote Sensors Data to Map Mineral
Prospectivity Based on Random Forest Predictive Model: A Case Study for Gold
Deposits in Hamissana Area, NE Sudan. Minerals, 2023, 13(1), 49.
Разбор — `knowledge_base/Детальный разбор/Хамиссана, Судан — Random Forest на
трёх сенсорах.md`.

ЧТО В СТАТЬЕ И ЧТО ИЗ ЭТОГО ВОСПРОИЗВОДИМО У НАС.

* Четыре набора признаков: Landsat-8 (12), Sentinel-2 (13), ASTER (24) и
  интеграция (41). Состав каждого — спектральные отношения и индексы по
  выписанным формулам, главные компоненты (PCA), компоненты максимального
  отношения сигнал/шум (MNF) и четыре геологических признака. Все формулы
  считаются по сырым каналам, которые в проекте есть: `ast_features.parquet`
  (9 каналов), `s2_features.parquet` (9), `l8_features.parquet` (6).
* Отбор отрицательных точек: кластеризация методом k-средних, ранжирование
  кластеров по частоте известных проявлений, отрицательные берутся из трёх
  худших классов не ближе 10 км к положительным. Воспроизведено — с поправками
  на добросовестность, см. отступления 3 и 4.
* Модель: случайный лес в режиме регрессии, гиперпараметры — перебором по
  сетке с кросс-проверкой, оценка — на отложенной части точек при случайном
  разбиении 70/30. Воспроизведено, и рядом посчитан тот же лес по нашему
  пространственному протоколу (полосы с разрывом 15 км).

ГЛАВНЫЙ ВОПРОС ПРОГОНА — не «сработает ли метод», а «что именно меряет число,
которое в статье называется качеством». Поэтому кроме самих наборов считаются
контроли, каждый снимает ровно одно предположение:

1. Отбор отрицательных: полный план 2×2 — {кластеры, случайно} × {буфер 10 км,
   без буфера}. Разнесение классов в пространстве и их разнесение по спектру —
   разные вещи, и разделить их можно только так.
2. Только координаты вместо признаков, на тех же точках и в тех же режимах
   отбора. Если координаты дают столько же, показатель меряет географию
   выборки, а не перспективность.
3. Пространственные полосы с разрывом 15 км против случайного разбиения 70/30 —
   на одних и тех же точках и признаках.
4. Геология «как в статье» против «без входов критериальной формулы».

ГДЕ ПРИШЛОСЬ ОТСТУПИТЬ ОТ СТАТЬИ (и почему).

1. Четыре геологических признака у авторов — расстояния до северо-восточных и
   северо-западных разломов, расстояние до контакта интрузий и плотность
   линеаментов. У нас первые три (`dist_tect1`, `dist_tect2`, `dist_magm`) —
   прямые входы самой критериальной формулы, и сравнивать с ней карту, в которую
   они заложены, значит мерить формулу самой формулой. Поэтому каждый набор
   считается в двух вариантах геологии: «как в статье» (с оговоркой о
   циркулярности) и «без входов формулы» (структурные признаки только по
   линеаментам). Разница между вариантами — и есть цена этих трёх столбцов.
2. Линеаменты у авторов выделены по пятой главной компоненте Sentinel-2; у нас
   рабочая ветка линеаментов — по рельефу, спутниковые не прошли порог
   воспроизводимости (`lineaments_sat_summary.csv`). Поэтому вывод авторов о
   первенстве плотности линеаментов проверяется НА НАШЕМ слое линеаментов, и
   отрицательный ответ говорит о нём, а не об их методе.
3. Кластеры для отбора отрицательных считаются по каналам ТОГО ЖЕ сенсора, что
   и набор признаков, а не всегда по ASTER, как у авторов: съёмка ASTER
   покрывает 70% листа, и кластеризация по ней сделала бы границу контура
   съёмки предиктором класса для наборов Landsat-8 и Sentinel-2.
4. Ранжирование кластеров и десятикилометровый буфер строятся ТОЛЬКО по
   обучающим положительным точкам и пересобираются в каждом разбиении. У
   авторов выборка строится один раз по всем меткам — при таком порядке
   отрицательные гарантированно лежат в кластерах, свободных в том числе от
   проверочных положительных, и проверка перестаёт быть слепой. Кластеры
   ранжируются по плотности проявлений (на ячейку кластера), а не по доле от
   всех точек: доля растёт с размером кластера и смещает отбор к мелким
   спектральным выбросам, которые отделяются тривиально.
5. Наборы с ASTER считаются только внутри контура съёмки: случайный лес
   пропусков не принимает, а заполнять их значило бы повторить ошибку ветки 4.
6. Наши каналы — медианные композиты в ячейке 500 м, поэтому отношения каналов
   считаются от медиан, а у авторов — попиксельно на нативном разрешении
   (медиана отношения не равна отношению медиан). По той же причине «шум» для
   MNF оценивается разностью соседних ячеек 500 м — это высокочастотный фильтр
   геологического поля, а не сенсорный шум статьи.
7. Главные компоненты и компоненты MNF подгоняются по всей сетке до разбиения
   на обучение и проверку. Меток они не видят, но проверочные ячейки в подгонку
   входят — это даёт моделям фору, то есть может завышать показатель по
   полосам, а не занижать его.
8. Положительные точки у авторов — 25 известных проявлений золота; у нас 73
   точки заверки. Порядок величины тот же, что и важно: метод статьи рассчитан
   на десятки меток, а не на тысячи.

Выход: `outputs/metrics/lit6b_hamissana_datasets.csv`, `..._importance.csv`,
       `..._vs_crit.csv`, `outputs/lit6b_hamissana.png`.
Запуск из корня: ``python -m experiments.lit6b_hamissana_rf``
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target)
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               poly_design)
from experiments.nn_geo_mpm import SEED  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()

N_CLUSTERS = 20          # k-средних, как в статье
N_CLASSES = 6            # шесть классов перспективности из 20 кластеров
NEG_BUFFER_M = 10_000.0  # отрицательные не ближе 10 км к обучающим положительным
N_REPEATS = 20           # повторов: своё разбиение И своя выборка отрицательных
N_PERM = 5               # перестановок на признак при оценке важности
GRID = {"n_estimators": [100, 300], "max_depth": [None, 5, 10],
        "max_features": ["sqrt", 0.5]}
GEO_ARTICLE = ("dist_tect2", "dist_tect1", "dist_magm", "lin_dens")
GEO_CLEAN = ("lin_dens", "lin_node_dens", "lin_dist", "lin_aniso_r")
NEG_MODES = (("кластеры", True), ("кластеры", False),
             ("случайно", True), ("случайно", False))
BASE_RF = dict(n_estimators=300, random_state=SEED, n_jobs=-1)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def ratio(a, b) -> np.ndarray:
    """Отношение каналов с защитой от неположительного знаменателя.

    В отражении, приведённом к физическим величинам, встречаются нули (тени,
    вода): деление даёт бесконечность или смену знака, оставаясь по виду
    числом. Такие ячейки честнее пометить пропуском.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = (y > 0) & (x > 0)
    out[ok] = x[ok] / y[ok]
    return out


# --------------------------------------------------------------------------
# Признаки по формулам статьи
# --------------------------------------------------------------------------

def spectral_l8(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Landsat-8: три отношения каналов (таблица формул статьи)."""
    return {
        "l8_BR_OH": ratio(d["l8_swir16"], d["l8_swir22"]),
        "l8_BR_ferric": ratio(d["l8_red"], d["l8_blue"]),
        "l8_BR_ferrous": ratio(d["l8_swir22"], d["l8_nir08"])
                         + ratio(d["l8_green"], d["l8_red"])}


def spectral_s2(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sentinel-2: четыре отношения каналов."""
    return {
        "s2_BR_OH": ratio(d["s2_b11"], d["s2_b12"]),
        "s2_BR_ferric": ratio(d["s2_b04"], d["s2_b03"]),
        "s2_BR_ferrous": ratio(d["s2_b12"], d["s2_b8a"])
                         + ratio(d["s2_b03"], d["s2_b04"]),
        "s2_BR_iron_ox": ratio(d["s2_b11"], d["s2_b8a"])}


def spectral_ast(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """ASTER: пять отношений, три глубины полос поглощения, четыре индекса."""
    b = {i: d[f"ast_b0{i}"] for i in range(1, 10)}
    return {
        "ast_BR_OH": ratio(b[4], b[6]),
        "ast_BR_ferric": ratio(b[2], b[1]),
        "ast_BR_ferrous": ratio(b[5], b[3]) + ratio(b[1], b[2]),
        "ast_BR_alunite": ratio(b[4], b[5]),
        "ast_BR_calcite": ratio(b[4], b[7]),
        "ast_RBD1_argill": ratio(b[4] + b[6], b[5]),
        "ast_RBD2_phyllic": ratio(b[5] + b[7], b[6]),
        "ast_RBD3_propyl": ratio(b[6] + b[9], b[7] + b[8]),
        "ast_OHI": ratio(b[7], b[6]) * ratio(b[4], b[6]),
        "ast_KLI": ratio(b[4], b[5]) * ratio(b[8], b[6]),
        "ast_ALI": ratio(b[7], b[5]) * ratio(b[7], b[8]),
        "ast_CLI": ratio(b[6], b[8]) * ratio(b[9], b[8])}


def components(X: np.ndarray, n_pca: int, n_mnf: int, shape, valid: np.ndarray,
               tag: str) -> dict[str, np.ndarray]:
    """Главные компоненты и компоненты максимального отношения сигнал/шум.

    PCA ищет направления наибольшей дисперсии, MNF — наибольшего отношения
    сигнал/шум: «шум» оценивается разностью соседних ячеек, данные отбеливаются
    его ковариацией, и только потом считаются компоненты. У авторов оба идут
    параллельно спектральным отношениям: физика ловит известное, статистика —
    неизвестное. Подгонка идёт по валидным ячейкам листа (вода и края
    исключены), но без разделения на обучение и проверку — см. отступление 7.
    """
    ok = np.isfinite(X).all(axis=1) & valid
    mu, sd = X[ok].mean(axis=0), X[ok].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    out = {}

    p = PCA(n_components=n_pca, random_state=SEED).fit(Z[ok])
    for j, v in enumerate(p.transform(np.nan_to_num(Z)).T):
        out[f"{tag}_pca{j + 1}"] = np.where(ok, v, np.nan)

    G = Z.reshape(shape + (Z.shape[1],))
    diff = (G[:, 1:] - G[:, :-1]).reshape(-1, Z.shape[1]) / np.sqrt(2.0)
    diff = diff[np.isfinite(diff).all(axis=1)]
    w, V = np.linalg.eigh(np.cov(diff, rowvar=False))
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    Zw = np.nan_to_num(Z) @ W
    m = PCA(n_components=n_mnf, random_state=SEED).fit(Zw[ok])
    for j, v in enumerate(m.transform(Zw).T):
        out[f"{tag}_mnf{j + 1}"] = np.where(ok, v, np.nan)
    return out


# --------------------------------------------------------------------------
# Отбор отрицательных точек
# --------------------------------------------------------------------------

def cluster_labels(bands: np.ndarray, work: np.ndarray) -> np.ndarray:
    """Разбиение ячеек на 20 спектральных кластеров (k-средних, как в статье)."""
    ok = work & np.isfinite(bands).all(axis=1)
    idx = np.flatnonzero(ok)
    Z = bands[idx]
    Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
    lab = np.full(work.size, -1)
    lab[idx] = KMeans(n_clusters=N_CLUSTERS, n_init=10,
                      random_state=SEED).fit_predict(Z)
    return lab


def build_negatives(labels: np.ndarray, pos_train: np.ndarray, work: np.ndarray,
                    cx: np.ndarray, cy: np.ndarray, n_neg: int,
                    rng: np.random.Generator, how: str,
                    buffer: bool) -> np.ndarray:
    """Отрицательные точки: достоверного «здесь золота нет» не бывает.

    ``how='кластеры'`` — как в статье: кластеры ранжируются по плотности
    проявлений, берутся три худших класса из шести. ``how='случайно'`` — из
    всего пула. ``buffer`` — требование удалённости на 10 км от ОБУЧАЮЩИХ
    положительных. Ранжирование и буфер видят только обучающие метки.
    """
    pool = work.copy()
    if how == "кластеры":
        size = np.array([max(int((labels == k).sum()), 1) for k in range(N_CLUSTERS)])
        cnt = np.array([int(np.isin(pos_train, np.flatnonzero(labels == k)).sum())
                        for k in range(N_CLUSTERS)])
        order = np.lexsort((np.arange(N_CLUSTERS), cnt / size))  # связки — по номеру
        per = int(np.ceil(N_CLUSTERS / N_CLASSES))
        pool &= np.isin(labels, order[:3 * per])
    cand = np.setdiff1d(np.flatnonzero(pool), pos_train)
    if buffer and cand.size and pos_train.size:
        d = np.min(np.hypot(cx[cand][:, None] - cx[pos_train][None, :],
                            cy[cand][:, None] - cy[pos_train][None, :]), axis=1)
        cand = cand[d >= NEG_BUFFER_M]
    if cand.size == 0:
        return cand
    return rng.choice(cand, size=min(n_neg, cand.size), replace=False)


# --------------------------------------------------------------------------
# Протоколы оценки
# --------------------------------------------------------------------------

def fit_eval(X: np.ndarray, tr: np.ndarray, te: np.ndarray, y_tr: np.ndarray,
             params: dict | None = None):
    kw = dict(BASE_RF)
    kw.update(params or {})
    m = RandomForestRegressor(**kw).fit(X[tr], y_tr)
    return m, m.predict(X[te])


def article_protocol(X: np.ndarray, pos: np.ndarray, labels: np.ndarray,
                     work: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                     how: str, buffer: bool) -> dict:
    """Случайное разбиение 70/30, как у авторов, повторённое N_REPEATS раз.

    В каждом повторе своя выборка отрицательных точек: главный источник
    разброса — какие именно отрицательные вытянулись, и мерить разброс при
    одной их реализации значит занижать неопределённость.
    """
    au = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(SEED + r)
        ptr, pte = train_test_split(pos, test_size=0.3, random_state=SEED + r)
        neg = build_negatives(labels, ptr, work, cx, cy, pos.size, rng, how, buffer)
        if neg.size < 10:
            continue
        ntr, nte = train_test_split(neg, test_size=0.3, random_state=SEED + r)
        tr = np.concatenate([ptr, ntr])
        te = np.concatenate([pte, nte])
        y_tr = np.r_[np.ones(ptr.size), np.zeros(ntr.size)]
        y_te = np.r_[np.ones(pte.size), np.zeros(nte.size)]
        ktr, kte = np.isfinite(X[tr]).all(axis=1), np.isfinite(X[te]).all(axis=1)
        if len(np.unique(y_te[kte])) < 2 or len(np.unique(y_tr[ktr])) < 2:
            continue
        _, p = fit_eval(X, tr[ktr], te[kte], y_tr[ktr])
        au.append(roc_auc_score(y_te[kte], p))
    if not au:
        return {"auc_статья": float("nan"), "auc_статья_разброс": float("nan"),
                "повторов": 0}
    return {"auc_статья": float(np.mean(au)), "auc_статья_разброс": float(np.std(au)),
            "повторов": len(au)}


def strip_folds(coord: np.ndarray, edges: np.ndarray):
    """Полосы по одной оси: маски проверки и обучения для каждой полосы.

    Границы полос заданы по СЕТКЕ ЛИСТА, а не по выбранным точкам: иначе у
    каждой конфигурации своя геометрия полос и показатели несопоставимы.
    """
    for k in range(N_STRIPS):
        te = (coord >= edges[k]) & (coord <= edges[k + 1])
        tr = ((coord < edges[k] - STRIP_BUFFER_M)
              | (coord > edges[k + 1] + STRIP_BUFFER_M))
        yield te, tr


def spatial_protocol(X: np.ndarray, pos: np.ndarray, labels: np.ndarray,
                     work: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                     edges_x: np.ndarray, edges_y: np.ndarray, how: str,
                     buffer: bool, cols: list[str] | None = None):
    """Тот же лес по полосам с разрывом 15 км — цена случайного разбиения.

    Отрицательные пересобираются в каждой полосе по обучающим положительным и
    делятся на обучающие и проверочные по той же геометрии полос. Заодно, если
    задан ``cols``, считается важность признаков ВНЕ обучающей выборки:
    перестановка столбца в отложенной полосе, падение показателя — важность.
    """
    au, imp, per_axis = [], {c: [] for c in (cols or [])}, []
    for ax, (coord, edges) in enumerate(((cx, edges_x), (cy, edges_y))):
        ax_au = []
        for f, (te_reg, tr_reg) in enumerate(strip_folds(coord, edges)):
            rng = np.random.default_rng(SEED + 100 * ax + f)
            ptr, pte = pos[tr_reg[pos]], pos[te_reg[pos]]
            if ptr.size < 10 or pte.size < 3:
                continue
            neg = build_negatives(labels, ptr, work, cx, cy, pos.size, rng,
                                  how, buffer)
            ntr, nte = neg[tr_reg[neg]], neg[te_reg[neg]]
            if ntr.size < 10 or nte.size < 3:
                continue
            tr = np.concatenate([ptr, ntr])
            te = np.concatenate([pte, nte])
            y_tr = np.r_[np.ones(ptr.size), np.zeros(ntr.size)]
            y_te = np.r_[np.ones(pte.size), np.zeros(nte.size)]
            ktr, kte = np.isfinite(X[tr]).all(axis=1), np.isfinite(X[te]).all(axis=1)
            if len(np.unique(y_te[kte])) < 2 or len(np.unique(y_tr[ktr])) < 2:
                continue
            m, p = fit_eval(X, tr[ktr], te[kte], y_tr[ktr])
            base = roc_auc_score(y_te[kte], p)
            au.append(base)
            ax_au.append(base)
            if cols:
                Xte, yte = X[te[kte]].copy(), y_te[kte]
                for j, c in enumerate(cols):
                    drops = []
                    for s in range(N_PERM):
                        Z = Xte.copy()
                        Z[:, j] = np.random.default_rng(SEED + s).permutation(Z[:, j])
                        drops.append(base - roc_auc_score(yte, m.predict(Z)))
                    imp[c].append(float(np.mean(drops)))
        per_axis.append(float(np.mean(ax_au)) if ax_au else float("nan"))
    out = {"auc_полосы": float(np.mean(au)) if au else float("nan"),
           "auc_полосы_разброс": float(np.std(au)) if au else float("nan"),
           "auc_полосы_x": per_axis[0], "auc_полосы_y": per_axis[1],
           "фолдов": len(au)}
    return out, {c: (float(np.mean(v)), float(np.std(v))) for c, v in imp.items() if v}


def tune(X: np.ndarray, cells: np.ndarray, y: np.ndarray, cx: np.ndarray,
         edges_x: np.ndarray) -> dict:
    """Гиперпараметры перебором с кросс-проверкой ПО ПОЛОСАМ, не случайной.

    Случайные фолды кладут пространственные почти-дубли по обе стороны
    разбиения и систематически выбирают самую глубокую модель; карта потом
    строится именно этими гиперпараметрами.
    """
    g = np.clip(np.digitize(cx[cells], edges_x[1:-1]), 0, N_STRIPS - 1)
    n = min(N_STRIPS, len(np.unique(g)))
    if n < 2:
        return {}
    gs = GridSearchCV(RandomForestRegressor(random_state=SEED), GRID,
                      cv=GroupKFold(n_splits=n), scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y, groups=g)
    return gs.best_params_


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    n_cells = meta.prf * meta.pic
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    edges_x = np.linspace(cx.min(), cx.max(), N_STRIPS + 1)
    edges_y = np.linspace(cy.min(), cy.max(), N_STRIPS + 1)
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)

    src = {"ast": pd.read_parquet(config.PROCESSED_DIR / "ast_features.parquet"),
           "s2": pd.read_parquet(config.PROCESSED_DIR / "s2_features.parquet"),
           "l8": pd.read_parquet(config.PROCESSED_DIR / "l8_features.parquet"),
           "lin": pd.read_parquet(config.PROCESSED_DIR / "lineament_features.parquet"),
           "raw": pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")}
    for k, d in src.items():
        assert len(d) == n_cells, f"{k}: {len(d)} строк вместо {n_cells}"
    geo = pd.concat([src["raw"][["dist_tect1", "dist_tect2", "dist_magm"]],
                     src["lin"]], axis=1)

    bands = {"ASTER": [f"ast_b0{i}" for i in range(1, 10)],
             "Sentinel-2": ["s2_b02", "s2_b03", "s2_b04", "s2_b05", "s2_b06",
                            "s2_b07", "s2_b8a", "s2_b11", "s2_b12"],
             "Landsat-8": ["l8_blue", "l8_green", "l8_red", "l8_nir08",
                           "l8_swir16", "l8_swir22"]}
    raw_bands = {n: src[{"ASTER": "ast", "Sentinel-2": "s2",
                         "Landsat-8": "l8"}[n]][c].to_numpy(float)
                 for n, c in bands.items()}
    raw_bands["интеграция"] = raw_bands["ASTER"]

    blocks = {
        "Landsat-8": {**spectral_l8(src["l8"]),
                      **components(raw_bands["Landsat-8"], 2, 3, shape, valid, "l8")},
        "Sentinel-2": {**spectral_s2(src["s2"]),
                       **components(raw_bands["Sentinel-2"], 2, 3, shape, valid, "s2")},
        "ASTER": {**spectral_ast(src["ast"]),
                  **components(raw_bands["ASTER"], 5, 3, shape, valid, "ast")}}
    inter = {}
    for b in ("Landsat-8", "Sentinel-2", "ASTER"):
        for k, v in blocks[b].items():
            assert k not in inter, f"столбец {k} задваивается в интеграции"
            inter[k] = v
    blocks["интеграция"] = inter
    for name, b in blocks.items():
        log(f"набор «{name}»: {len(b)} спектральных и статистических столбцов")

    pts = assessment.load_verification_points(meta)
    pos_all = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    ast_ok = np.isfinite(raw_bands["ASTER"]).all(axis=1)
    log(f"точек заверки {pos_all.size}, из них непредвзятых {unb.size}; "
        f"контур ASTER — {int((valid & ast_ok).sum())} ячеек из {int(valid.sum())}")

    labels = {}
    for name in ("Landsat-8", "Sentinel-2", "ASTER"):
        work = valid & ast_ok if name == "ASTER" else valid
        labels[name] = cluster_labels(raw_bands[name], work)
    labels["интеграция"] = labels["ASTER"]

    rows, imp_rows, maps = [], [], {}
    for bname, block in blocks.items():
        needs_ast = bname in ("ASTER", "интеграция")
        work = valid & ast_ok if needs_ast else valid
        pos = pos_all[work[pos_all]]
        lab = labels[bname]

        variants = [("как в статье", GEO_ARTICLE), ("без входов формулы", GEO_CLEAN),
                    ("только координаты", None)]
        for gname, gcols in variants:
            if gcols is None:
                cols = ["коорд_x", "коорд_y"]
                X = np.column_stack([cx, cy])
            else:
                cols = list(block) + list(gcols)
                X = np.column_stack(
                    [np.asarray(block[c], dtype=float) for c in block]
                    + [geo[c].to_numpy(float) for c in gcols])
            fin = np.isfinite(X).all(axis=1)
            wk = work & fin
            pl = np.flatnonzero(wk)
            posv = pos[wk[pos]]

            # план 2x2 по отбору отрицательных считается для чистой геометрии и
            # для координат; вариант «как в статье» и без того циркулярен,
            # множить его на четыре режима незачем
            modes = NEG_MODES if gname != "как в статье" else NEG_MODES[:1]
            for how, buf in modes:
                nm = f"{how}{', буфер 10 км' if buf else ', без буфера'}"
                main_mode = (how, buf) == NEG_MODES[0]
                res = {"набор": bname, "геология": gname, "отбор_отрицательных": nm,
                       "столбцов": X.shape[1], "ячеек": int(wk.sum()),
                       "точек": int(posv.size)}
                res.update(article_protocol(X, posv, lab, wk, cx, cy, how, buf))
                want_imp = main_mode and gcols is not None
                sp, imp = spatial_protocol(X, posv, lab, wk, cx, cy, edges_x,
                                           edges_y, how, buf,
                                           cols=cols if want_imp else None)
                res.update(sp)
                log(f"  {bname} | {gname} | {nm}: AUC статьи "
                    f"{res['auc_статья']:.3f}, по полосам {res['auc_полосы']:.3f}")
                for c, (m_, s_) in imp.items():
                    imp_rows.append({"набор": bname, "геология": gname,
                                     "признак": c, "важность": m_, "разброс": s_})

                # карта строится только для основного режима и настоящих признаков
                if want_imp:
                    rng = np.random.default_rng(SEED)
                    neg = build_negatives(lab, posv, wk, cx, cy, posv.size, rng,
                                          how, buf)
                    cells = np.concatenate([posv, neg])
                    y = np.r_[np.ones(posv.size), np.zeros(neg.size)]
                    params = tune(X[cells], cells, y, cx, edges_x)
                    m = RandomForestRegressor(**{**BASE_RF, **params}).fit(X[cells], y)
                    s = np.full(n_cells, np.nan)      # пропуски НЕ заполняются
                    s[pl] = m.predict(X[pl])
                    maps[f"{bname} | {gname}"] = (s, wk)

                    top_c = crit_reference.reference_mask(crit, pl, config.CREF_AREA) & wk
                    top_m = (s >= np.quantile(s[pl], 1 - config.CREF_AREA)) & wk
                    cr, sr = detrend(crit, pl, D), detrend(s, pl, D)
                    tcr = (cr >= np.quantile(cr[pl], 1 - config.CREF_AREA)) & wk
                    tpr = (sr >= np.quantile(sr[pl], 1 - config.CREF_AREA)) & wk
                    agr = crit_reference.agreement(s, crit, pl, config.CREF_AREA,
                                                   orient=1)
                    res.update({
                        "совпадение_верх_10": float((top_c & top_m).sum() / top_c.sum()),
                        "совпадение_без_тренда": float((tpr & tcr).sum() / tcr.sum()),
                        "реализованная_площадь": float(top_m.sum() / wk.sum()),
                        "карта_auc": agr["auc"], "карта_каппа": agr["kappa"],
                        "новых_ячеек": int((top_m & ~top_c).sum()),
                        "гиперпараметры": str(params)})
                rows.append(res)

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "lit6b_hamissana_datasets.csv",
               index=False, encoding="utf-8-sig")
    imp = pd.DataFrame(imp_rows)
    imp.to_csv(ROOT / "outputs" / "metrics" / "lit6b_hamissana_importance.csv",
               index=False, encoding="utf-8-sig")

    show = ["набор", "геология", "отбор_отрицательных", "столбцов", "auc_статья",
            "auc_статья_разброс", "auc_полосы", "auc_полосы_разброс",
            "совпадение_верх_10", "совпадение_без_тренда", "реализованная_площадь",
            "карта_auc"]
    print("\n=== Метод статьи на наших данных ===")
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.3f}",
                              na_rep="—"))
    print("\nУ авторов: Sentinel-2 AUC 0.875, ASTER 0.875, интеграция 0.938 — "
          "при случайном разбиении 70/30, то есть по столбцу «auc_статья».")
    print("Случайный уровень совпадения верхних 10% — 0.10; рабочая точка "
          "проекта (ветки 2/3) — 0.295 после снятия тренда при ROC карты 0.803.")

    # Сдвиговый null. Карты не заполняются: пропуски остаются пропусками, иначе
    # большой блок одинаковых значений катается по тору и нулевое распределение
    # становится вырожденным (та же ошибка, что закрыла ветку 6). Семьи
    # разделены: карты с прямыми входами формулы циркулярны по построению и не
    # должны разбавлять поправку для чистых карт.
    ver = []
    for family in ("как в статье", "без входов формулы"):
        sel = [k for k in maps if k.endswith(family)]
        alpha = 0.05 / len(sel)
        for k in sel:
            s, mask = maps[k]
            pool = np.flatnonzero(mask)
            a = crit_reference.agreement(s, crit, pool, config.CREF_AREA, orient=1)
            n = crit_reference.shift_null(s, crit, meta, pool, config.CREF_AREA,
                                          orient=1)
            b = crit_reference.block_bootstrap(s, crit, meta, pool,
                                               area=config.CREF_AREA)
            ver.append({"карта": k, "ячеек_в_пуле": pool.size, "auc": a["auc"],
                        "auc_ci_lo": b["auc_ci_lo"], "auc_ci_hi": b["auc_ci_hi"],
                        "capture": a["capture"], "kappa": a["kappa"],
                        "p_сдвиг": n["p"], "порог": alpha,
                        "воспроизводит": bool(a["auc"] >= config.CREF_AUC_MIN
                                              and b["auc_ci_lo"] > 0.5
                                              and n["p"] < alpha)})
        log(f"сдвиговый null и блочный бутстрэп готовы: семья «{family}»")
    ver = pd.DataFrame(ver)
    ver.to_csv(ROOT / "outputs" / "metrics" / "lit6b_hamissana_vs_crit.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Воспроизводит ли карта критериальный прогноз ===")
    print(ver.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Условия: AUC не ниже {config.CREF_AUC_MIN}, нижняя граница интервала "
          "выше 0.5, сдвиговый p ниже порога своей семьи.")

    print("\n=== Главный вывод авторов: важнее всех плотность линеаментов? ===")
    print("Важность считается вне обучающей выборки: перестановка столбца в "
          "отложенной полосе. Наш слой линеаментов — по рельефу, не по снимку.")
    for (b, g), grp in imp.groupby(["набор", "геология"]):
        rank = grp.sort_values("важность", ascending=False).reset_index(drop=True)
        w = rank.index[rank["признак"] == "lin_dens"]
        place = int(w[0]) + 1 if len(w) else 0
        top = ", ".join(f"{r['признак']} {r['важность']:+.3f}"
                        for _, r in rank.head(3).iterrows())
        print(f"{b:12s} | {g:18s}: место lin_dens {place} из {len(rank)}; "
              f"сильнейшие — {top}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rank_img(v, m):
        out = np.full(v.size, np.nan)
        out[m] = sps.rankdata(v[m]) / m.sum()
        return out.reshape(shape)

    keys = [k for k in maps if k.endswith("без входов формулы")]
    fig, axs = plt.subplots(1, len(keys) + 1, figsize=(4.2 * (len(keys) + 1), 5.2))
    axs[0].imshow(rank_img(crit, valid), cmap="magma")
    axs[0].set_title("Критериальный прогноз", fontsize=9)
    for ax, k in zip(axs[1:], keys):
        s, m = maps[k]
        r = res[(res["набор"] == k.split(" | ")[0])
                & (res["геология"] == "без входов формулы")
                & (res["отбор_отрицательных"].str.startswith("кластеры, буфер"))].iloc[0]
        ax.imshow(rank_img(s, m), cmap="magma")
        ax.set_title(f"{k.split(' | ')[0]} по методу статьи\n"
                     f"AUC по разбиению статьи {r['auc_статья']:.2f}, "
                     f"по полосам {r['auc_полосы']:.2f}\n"
                     f"совпадение верхних 10% {r['совпадение_верх_10']:.0%}, "
                     f"без тренда {r['совпадение_без_тренда']:.0%}", fontsize=9)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ветка 6B: метод Хамиссаны (Судан, 2023) на данных Анабара — "
                 "геология без прямых входов критериальной формулы", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ROOT / "outputs" / "lit6b_hamissana.png", dpi=130)
    log("OK: outputs/lit6b_hamissana.png")


if __name__ == "__main__":
    main()
