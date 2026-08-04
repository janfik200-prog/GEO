"""Критериальный прогноз как эталонная карта заверки.

ЗАЧЕМ ЭТО ЗАМЕНЯЕТ ТОЧКИ
------------------------
Этап 5e измерил разрешающую способность точечной заверки: 19 несмещённых точек
при площади 10% дают шаг метрики 0.53 lift, а одна и та же модель при разных
семенах обучения гуляет по этой шкале на 1-4 точки. То есть точечная метрика
измеряет случайность оптимизации, а не метод, и перебор архитектур на ней
бессмыслен: этап 6 (водосборы) это подтвердил — водосборы точек оказались
вырожденно мелкими (13 из 19 не больше двух ячеек), метрика схлопнулась обратно
в точечную.

Здесь единицей заверки становится ЯЧЕЙКА КАРТЫ, а эталоном — карта
критериального прогноза ГИС Интегро. Заверочная выборка вырастает с 19 точек до
22 905 ячеек, метрика становится непрерывной, и разброс по семенам перестаёт
доминировать над разницей между методами.

ЭТО ЗАКОННАЯ ФОРМА ЗАВЕРКИ, А НЕ ПОДМЕНА. В литературе по прогнозной оценке
ровно так и поступают, когда рудопроявлений мало или их нет: согласие
data-driven модели с knowledge-driven экспертной картой используется как
критерий (Porwal, Carranza & Hale 2006 — сравнение knowledge-driven и
data-driven нечётких моделей; Torppa, Nykanen & Molnar 2019 — кластеры
проверялись на соответствие экспертной интерпретации, а не только точкам).

ЧТО ЭТИМ ДОКАЗЫВАЕТСЯ, А ЧТО НЕТ
--------------------------------
НЕ доказывается превосходство над критериальным методом: он здесь сам эталон,
и потолок метрики — его точное воспроизведение. Заявлять «превзошли» по этой
метрике нельзя ни при каком результате, и в вердикте такой колонки нет.

Доказывается другое: воспроизводит ли нейросетевой метод БЕЗ УЧИТЕЛЯ, ни на
одном шаге не видевший ``prognoz``, перспективные площади критериального
прогноза. Если да — метод заверен независимым способом, а его расхождения с
эталоном становятся кандидатами в новые площади (карта расхождений этапа 7).

ЖЁСТКОЕ ТРЕБОВАНИЕ: карта-эталон не должна попадать в обучение НИ В КАКОМ ВИДЕ.
Модуль умышленно не содержит ни одной функции обучения — только оценку. Скоры
приходят готовыми из прогонов, которые про ``prognoz`` не знают.

ОРИЕНТАЦИЯ СКОРА
----------------
У детектора без учителя направление шкалы не определено: «необычно» не значит
«благоприятно», и знак согласия с эталоном заранее не известен. Знак выбирается
по эталону (по знаку ранговой корреляции), и это — выбор, сделанный по
заверочным данным, поэтому знаменатель поправки на множественность удваивается
(``config.CREF_ORIENT_PENALTY``). Молча брать модуль корреляции нельзя: это
скрытая подгонка.

ЗНАЧИМОСТЬ И ИНТЕРВАЛЫ
----------------------
Наивные p и ДИ по 22 905 ячейкам были бы фикцией: соседние ячейки сильно
автокоррелированы, независимых единиц здесь сотни, а не десятки тысяч. Поэтому:

* значимость — ТОРОИДАЛЬНЫЙ СДВИГОВЫЙ null (карта циклически сдвигается,
  эталон фиксирован): null сохраняет автокорреляцию карты-претендента;
* интервалы — БЛОЧНЫЙ БУТСТРЭП блоками ``config.CREF_BLOCK`` ячеек (10 км),
  блок берётся целиком.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import config, integro_grid


# --------------------------------------------------------------------------
# Эталон
# --------------------------------------------------------------------------

def criterial_score(prognoz: np.ndarray) -> np.ndarray:
    """Скор критериального прогноза в общей полярности «выше = перспективнее».

    Нативная шкала ``prognoz`` — взвешенное L1-расстояние до эталона (меньше =
    перспективнее), поэтому знак инвертируется.
    """
    return -np.asarray(prognoz, dtype=float).ravel()


def reference_mask(crit: np.ndarray, pool: np.ndarray,
                   area: float | None = None) -> np.ndarray:
    """Булев эталон «критериально перспективно» = top-``area`` карты по пулу.

    Порог берётся по пулу валидных ячеек, а не по всей сетке: вне пула лежат
    вода и края, у которых критериальный скор вырожден связками.
    """
    area = area if area is not None else config.CREF_AREA
    crit = np.asarray(crit, dtype=float)
    thr = np.quantile(crit[pool], 1.0 - area)
    return crit >= thr


def block_labels(meta: integro_grid.GridMeta,
                 block: int | None = None) -> np.ndarray:
    """Номер пространственного блока для каждой ячейки сетки (C-порядок)."""
    block = block or config.CREF_BLOCK
    rows = np.arange(meta.prf * meta.pic) // meta.pic
    cols = np.arange(meta.prf * meta.pic) % meta.pic
    n_bc = int(np.ceil(meta.pic / block))
    return (rows // block) * n_bc + (cols // block)


# --------------------------------------------------------------------------
# Метрики согласия с эталоном
# --------------------------------------------------------------------------

def orientation(score: np.ndarray, crit: np.ndarray,
                pool: np.ndarray) -> int:
    """+1 или -1: знак, при котором скор согласован с эталоном.

    Определяется по ранговой корреляции с НЕПРЕРЫВНЫМ критериальным скором, а
    не по бинарному эталону: непрерывная версия устойчивее и не зависит от
    выбранной площади.
    """
    rho, _ = stats.spearmanr(np.asarray(score)[pool], np.asarray(crit)[pool])
    if not np.isfinite(rho) or rho == 0:
        return 1
    return 1 if rho > 0 else -1


def agreement(score: np.ndarray, crit: np.ndarray, pool: np.ndarray,
              area: float | None = None,
              orient: int | None = None) -> dict[str, float]:
    """Согласие карты-претендента с критериальным эталоном.

    Возвращает:

    * ``auc`` — ПЕРВИЧНАЯ метрика этапа: площадь под ROC при предсказании
      класса «критериально перспективно» непрерывным скором. Не зависит от
      выбора порога у претендента, поэтому не награждает удачную калибровку;
    * ``capture`` — доля эталонной площади, накрытая top-``area`` претендента,
      делённая на реализованную площадь (аналог lift: 1 = случайно,
      1/``area`` = идеально);
    * ``kappa`` — каппа Коэна на бинаризованных картах (согласие с поправкой
      на случайное совпадение);
    * ``rho`` — ранговая корреляция с непрерывным критериальным скором;
    * ``orient`` — применённый знак.
    """
    area = area if area is not None else config.CREF_AREA
    score = np.asarray(score, dtype=float)
    crit = np.asarray(crit, dtype=float)
    if orient is None:
        orient = orientation(score, crit, pool)
    s = orient * score
    y = reference_mask(crit, pool, area)

    sp, yp = s[pool], y[pool]
    finite = np.isfinite(sp)
    if finite.sum() < 2 or yp[finite].all() or not yp[finite].any():
        return {"auc": 0.5, "capture": 0.0, "kappa": 0.0, "rho": 0.0,
                "orient": float(orient)}

    auc = float(roc_auc_score(yp[finite], sp[finite]))
    thr = np.quantile(sp[finite], 1.0 - area)
    top = sp >= thr
    realized = float(top[finite].mean())
    capture = float((top & yp).sum()) / max(int(yp.sum()), 1) / realized \
        if realized > 0 else 0.0
    kappa = _kappa(top[finite], yp[finite])
    rho, _ = stats.spearmanr(sp[finite], crit[pool][finite])
    return {"auc": auc, "capture": capture, "kappa": kappa,
            "rho": float(rho) if np.isfinite(rho) else 0.0,
            "orient": float(orient)}


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Каппа Коэна для двух булевых разметок одинаковой длины."""
    n = a.size
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if n == 0 or pe >= 1.0:
        return 0.0
    return (po - pe) / (1.0 - pe)


# --------------------------------------------------------------------------
# Значимость и интервалы с учётом пространственной автокорреляции
# --------------------------------------------------------------------------

def shift_null(score: np.ndarray, crit: np.ndarray,
               meta: integro_grid.GridMeta, pool: np.ndarray,
               area: float | None = None, n_shifts: int | None = None,
               seed: int | None = None,
               orient: int | None = None) -> dict[str, float]:
    """Тороидальный сдвиговый null для AUC согласия с эталоном.

    Карта-претендент циклически сдвигается и отражается, эталон и пул
    фиксированы. Null сохраняет пространственную структуру претендента,
    поэтому отвечает на правильный вопрос: «согласие выше, чем у карты с такой
    же автокорреляцией, положенной на лист случайно?» — а не на бессмысленный
    «выше, чем у белого шума?».

    Ориентация фиксируется ПО НАБЛЮДЁННОЙ карте и применяется ко всем сдвигам
    одинаково: если бы знак выбирался заново на каждом сдвиге, null получил бы
    ту же свободу выбора, что и наблюдение, и тест стал бы консервативным по
    неверной причине.
    """
    area = area if area is not None else config.CREF_AREA
    n_shifts = n_shifts or config.CREF_N_SHIFTS
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    score = np.asarray(score, dtype=float)
    crit = np.asarray(crit, dtype=float)
    if orient is None:
        orient = orientation(score, crit, pool)

    y = reference_mask(crit, pool, area)[pool]
    img = (orient * score).reshape(meta.prf, meta.pic)

    def auc_of(flat: np.ndarray) -> float:
        v = flat[pool]
        ok = np.isfinite(v)
        if ok.sum() < 2 or y[ok].all() or not y[ok].any():
            return 0.5
        return float(roc_auc_score(y[ok], v[ok]))

    observed = auc_of(img.ravel())
    null = np.empty(n_shifts)
    for i in range(n_shifts):
        a = img
        if rng.integers(2):
            a = np.flipud(a)
        if rng.integers(2):
            a = np.fliplr(a)
        a = np.roll(a, (int(rng.integers(meta.prf)), int(rng.integers(meta.pic))),
                    axis=(0, 1))
        null[i] = auc_of(a.ravel())
    p = float((1 + (null >= observed).sum()) / (1 + n_shifts))
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)), "p": p,
            "n_shifts": n_shifts}


def block_bootstrap(score_a: np.ndarray, crit: np.ndarray,
                    meta: integro_grid.GridMeta, pool: np.ndarray,
                    score_b: np.ndarray | None = None,
                    area: float | None = None, block: int | None = None,
                    n_boot: int | None = None, seed: int | None = None,
                    ci: tuple[float, float] = (5.0, 95.0)) -> dict[str, float]:
    """Блочный бутстрэп AUC (и разности AUC двух карт) по блокам 10 км.

    Ресемплируются БЛОКИ ячеек с возвращением, а не ячейки: соседние ячейки
    автокоррелированы, и поячеечный бутстрэп дал бы интервал в разы уже
    честного. Если задан ``score_b``, возвращается ещё и интервал разности
    (A - B) — им сравниваются между собой методы-претенденты (сравнение с самим
    эталоном смысла не имеет: у эталона AUC = 1 по построению).
    """
    area = area if area is not None else config.CREF_AREA
    n_boot = n_boot or config.CREF_N_BOOT
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)

    crit = np.asarray(crit, dtype=float)
    y = reference_mask(crit, pool, area)[pool]
    blocks = block_labels(meta, block)[pool]
    labels = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in labels}

    sa = (orientation(score_a, crit, pool) * np.asarray(score_a, float))[pool]
    sb = None if score_b is None else (
        orientation(score_b, crit, pool) * np.asarray(score_b, float))[pool]

    def auc(v: np.ndarray, yy: np.ndarray) -> float:
        ok = np.isfinite(v)
        if ok.sum() < 2 or yy[ok].all() or not yy[ok].any():
            return 0.5
        return float(roc_auc_score(yy[ok], v[ok]))

    obs_a, obs_b = auc(sa, y), (None if sb is None else auc(sb, y))
    aa = np.empty(n_boot)
    dd = np.empty(n_boot) if sb is not None else None
    for i in range(n_boot):
        chosen = rng.choice(labels, size=labels.size, replace=True)
        idx = np.concatenate([members[b] for b in chosen])
        aa[i] = auc(sa[idx], y[idx])
        if dd is not None:
            dd[i] = aa[i] - auc(sb[idx], y[idx])
    lo, hi = np.percentile(aa, ci)
    out = {"auc_a": obs_a, "auc_ci_lo": float(lo), "auc_ci_hi": float(hi),
           "n_blocks": int(labels.size), "n_boot": n_boot}
    if dd is not None:
        dlo, dhi = np.percentile(dd, ci)
        out.update({"auc_b": obs_b, "delta": obs_a - obs_b,
                    "delta_ci_lo": float(dlo), "delta_ci_hi": float(dhi)})
    return out


# --------------------------------------------------------------------------
# Протокол этапа
# --------------------------------------------------------------------------

def run_protocol(scores: dict[str, np.ndarray], crit: np.ndarray,
                 meta: integro_grid.GridMeta, valid: np.ndarray,
                 alpha: float, area: float | None = None,
                 log=None) -> dict[str, pd.DataFrame]:
    """Полная заверка набора карт по критериальному эталону.

    Возвращает ``agree`` (метрики согласия), ``null`` (сдвиговый тест),
    ``boot`` (блочные интервалы AUC) и ``verdict`` со столбцом
    ``reproduces``: карта считается воспроизводящей критериальный прогноз,
    если выполнены ВСЕ три условия — AUC не ниже ``config.CREF_AUC_MIN``,
    нижняя граница блочного интервала AUC выше 0.5 и сдвиговый p ниже
    ``alpha``. Столбца «превзошёл» здесь нет и быть не может: эталон и есть
    критериальный прогноз.
    """
    area = area if area is not None else config.CREF_AREA
    pool = np.flatnonzero(valid)

    rows = [{"method": n, **agreement(s, crit, pool, area)}
            for n, s in scores.items()]
    agree = pd.DataFrame(rows)
    if log:
        log(f"согласие с эталоном посчитано для {len(agree)} карт")

    null = pd.DataFrame([{"method": n,
                          **shift_null(s, crit, meta, pool, area)}
                         for n, s in scores.items()])
    if log:
        log(f"сдвиговый null ({config.CREF_N_SHIFTS} сдвигов) готов")
    boot = pd.DataFrame([{"method": n,
                          **block_bootstrap(s, crit, meta, pool, area=area)}
                         for n, s in scores.items()])
    if log:
        log(f"блочный бутстрэп ({config.CREF_N_BOOT} ресемплов) готов")

    verdict = (agree.merge(null[["method", "p"]].rename(columns={"p": "p_shift"}),
                           on="method")
               .merge(boot[["method", "auc_ci_lo", "auc_ci_hi", "n_blocks"]],
                      on="method"))
    verdict["alpha"] = alpha
    verdict["cond_auc"] = verdict["auc"] >= config.CREF_AUC_MIN
    verdict["cond_ci"] = verdict["auc_ci_lo"] > 0.5
    verdict["cond_p"] = verdict["p_shift"] < alpha
    verdict["reproduces"] = (verdict["cond_auc"] & verdict["cond_ci"]
                             & verdict["cond_p"])
    verdict = verdict.sort_values("auc", ascending=False).reset_index(drop=True)
    return {"agree": agree, "null": null, "boot": boot, "verdict": verdict}
