"""Единый протокол заверки и сравнения карт перспективности.

Все методы фазы «собственный алгоритм» (лестница нетипичности, TinyMAE)
и критериальный прогноз ГИС Интегро приводятся к одному интерфейсу — скор
длиной в сетку prognoz (22 946 ячеек, «выше = перспективнее») — и сравниваются
одним и тем же кодом. Критериальный здесь — равноправный метод, а не таргет:
ни один новый метод на нём (и вообще на метках) не обучается, поэтому
OOF-проблем прошлой фазы нет по построению.

Метрики: P-A кривая (prediction-area, Yousefi & Carranza) и её точечный срез
capture-efficiency (= coverage / area, он же lift@area). ROC/AUC не
используются: при 19–76 контрольных точках AUC неинформативен, P-A кривая
остаётся интерпретируемой. Обязательные планки: наивный скор «минус расстояние
до гидросети» (ловит речное смещение точек опробования: 75% точек «речные»
по дизайну съёмки) и случайный скор.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config, data_loader, integro_grid


# --------------------------------------------------------------------------
# Точки заверки
# --------------------------------------------------------------------------

def aggregate_point_cells(cells: np.ndarray, codes: np.ndarray) -> pd.DataFrame:
    """Агрегация точек до уникальных ячеек: code + строгий признак несмещённости.

    ``code`` — минимальный приоритетный код ячейки (2/4 приоритетнее 1/5) —
    для карт и отчётов. ``unbiased_strict`` — True, только если ВСЕ точки
    ячейки имеют код из ``config.VER_UNBIASED_CODES``: ячейка, где есть и
    рыхлая проба (1/5), и коренная (2/4), НЕ несмещённая — её положение всё
    равно продиктовано маршрутом вдоль гидросети.
    """
    df = pd.DataFrame({"cell": np.asarray(cells, dtype=int),
                       "code": np.asarray(codes, dtype=int)})
    if df.empty:
        return pd.DataFrame(columns=["cell", "code", "unbiased_strict"])
    unbiased = list(config.VER_UNBIASED_CODES)
    df["_unb"] = df["code"].isin(unbiased)
    df["_pri"] = np.where(df["_unb"], 0, 1)
    strict = df.groupby("cell")["_unb"].all()
    rep = (df.sort_values(["cell", "_pri", "code"]).drop_duplicates("cell")
             .set_index("cell")["code"])
    out = pd.DataFrame({"cell": strict.index.to_numpy(),
                        "code": rep.loc[strict.index].to_numpy(),
                        "unbiased_strict": strict.to_numpy()})
    return out.reset_index(drop=True)


def load_verification_points(meta: integro_grid.GridMeta) -> pd.DataFrame:
    """Точки «геохимическое_опробование» -> DataFrame(cell, code, unbiased_strict).

    В отличие от :func:`src.criterial_target.load_real_points` сохраняет тип
    точки (поле ``code``: 1 — литопробы в рыхлых, 2 — по коренным, 4 —
    проявления U/TR, 5 — шлихи Au), чтобы протокол мог выделять несмещённое
    подмножество. Ячейка может содержать несколько точек — строки агрегируются
    :func:`aggregate_point_cells`: ``code`` минимальный приоритетный (для
    карт), ``unbiased_strict`` — все точки ячейки с кодами 2/4.
    """
    base_dir = data_loader.find_base_dir()
    shp_dir = base_dir / config.SHP_SUBDIR
    alias_dir = config.PROJECT_ROOT / "data" / "processed" / "_shp_aliases_tmp"
    alias_dir.mkdir(parents=True, exist_ok=True)
    aliases = data_loader.prepare_ascii_aliases(shp_dir, alias_dir)
    mask_layer = data_loader.load_layer(aliases[config.LAYER_FILES["mask"]])
    points = data_loader.collect_points(mask_layer.crs, aliases)
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=["cell", "code", "unbiased_strict"])

    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    row = np.floor((meta.y_top - ys) / meta.dy).astype(int)
    col = np.floor((xs - meta.x0) / meta.dx).astype(int)
    inside = (row >= 0) & (row < meta.prf) & (col >= 0) & (col < meta.pic)
    cells = row[inside] * meta.pic + col[inside]
    codes = points["code"].to_numpy()[inside].astype(int)
    return aggregate_point_cells(cells, codes)


def unbiased_cells(points: pd.DataFrame) -> np.ndarray:
    """Ячейки СТРОГО несмещённого подмножества (все точки ячейки — коренные
    пробы/проявления, коды ``config.VER_UNBIASED_CODES``)."""
    return points.loc[points["unbiased_strict"], "cell"].to_numpy()


def point_clusters(point_idx: np.ndarray, meta: integro_grid.GridMeta) -> np.ndarray:
    """Метки маршрутных кластеров точек (single-linkage по центрам ячеек).

    Точки опробования кластеризованы маршрутами — независимая единица заверки
    не точка, а кластер. Порог объединения ``config.VER_CLUSTER_LINK_M`` (м),
    critерий 'distance'. При 0/1 точке — тривиальные метки.
    """
    point_idx = np.asarray(point_idx, dtype=int)
    if point_idx.size <= 1:
        return np.ones(point_idx.size, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage

    rows, cols = np.divmod(point_idx, meta.pic)
    xy = np.column_stack([meta.x0 + (cols + 0.5) * meta.dx,
                          meta.y_top - (rows + 0.5) * meta.dy])
    Z = linkage(xy, method="single")
    return fcluster(Z, t=config.VER_CLUSTER_LINK_M, criterion="distance")


def filter_points(point_idx: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Оставить только точки в валидных ячейках (вода/артефакты не заверяют)."""
    point_idx = np.asarray(point_idx, dtype=int)
    return point_idx[valid_mask[point_idx]]


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------

def coverage_and_area(
    score: np.ndarray, point_idx: np.ndarray, area: float,
    pool: np.ndarray | None = None,
) -> tuple[float, float]:
    """(coverage, realized_area) для порога top-``area`` по пулу.

    ``realized_area`` — ФАКТИЧЕСКАЯ доля пула со скором >= порога: при
    связках (ties, например у критериального prognoz вне свит) квантильный
    порог может накрывать заметно больше ``area`` — lift обязан делиться
    на реализованную площадь, а не на номинальную.
    """
    point_idx = np.asarray(point_idx, dtype=int)
    ref = score if pool is None else score[pool]
    thr = np.quantile(ref, 1.0 - area)
    coverage = float((score[point_idx] >= thr).mean()) if point_idx.size else 0.0
    realized = float((ref >= thr).mean())
    return coverage, realized


def pa_curve(
    score: np.ndarray, point_idx: np.ndarray,
    areas: tuple[float, ...] | None = None, pool: np.ndarray | None = None,
) -> pd.DataFrame:
    """P-A кривая: coverage, realized_area и lift по сетке площадей ``areas``.

    ``pool`` — индексы ячеек, по которым берётся порог top-X% (обычно валидные
    ячейки из :func:`src.cell_mask.build_valid_mask`); полярность скора —
    «выше = перспективнее». ``lift = coverage / realized_area`` (0, если
    реализованная площадь нулевая).
    """
    areas = areas or config.VER_AREAS
    rows = []
    for area in areas:
        cov, realized = coverage_and_area(score, point_idx, area, pool=pool)
        rows.append({"area": area, "coverage": cov, "realized_area": realized,
                     "lift": cov / realized if realized > 0 else 0.0,
                     "n_points": int(np.asarray(point_idx).size)})
    return pd.DataFrame(rows)


def capture_efficiency(
    score: np.ndarray, point_idx: np.ndarray,
    area: float | None = None, pool: np.ndarray | None = None,
) -> float:
    """lift@area = coverage / realized_area — точечный срез P-A кривой."""
    area = area or config.VER_AREA
    cov, realized = coverage_and_area(score, point_idx, area, pool=pool)
    return cov / realized if realized > 0 else 0.0


def bootstrap_diff(
    score_a: np.ndarray, score_b: np.ndarray, point_idx: np.ndarray,
    area: float | None = None, pool: np.ndarray | None = None,
    n_boot: int | None = None, seed: int | None = None,
    clusters: np.ndarray | None = None,
) -> dict[str, float]:
    """Бутстрэп по точкам (или кластерам): значима ли разность lift@area (A − B).

    Карты фиксированы — оценивается только неопределённость малого числа точек
    заверки. По умолчанию ресемплируются точки с возвращением; при заданных
    ``clusters`` (метки :func:`point_clusters`, длина = ``point_idx``)
    ресемплируются КЛАСТЕРЫ с возвращением: берутся все точки выбранных
    кластеров, размер выборки = число кластеров. Кластерный вариант честнее:
    точки одного маршрута не независимы. Возвращает наблюдаемую дельту,
    перцентильный 95% ДИ и долю ресемплов с дельтой <= 0.
    """
    area = area or config.VER_AREA
    n_boot = n_boot or config.VER_N_BOOT
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    point_idx = np.asarray(point_idx, dtype=int)

    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.size != point_idx.size:
            raise ValueError("bootstrap_diff: clusters должен быть той же длины, что point_idx")
        labels = np.unique(clusters)
        members = {c: point_idx[clusters == c] for c in labels}

        def resample():
            chosen = rng.choice(labels, size=labels.size, replace=True)
            return np.concatenate([members[c] for c in chosen])
    else:
        def resample():
            return rng.choice(point_idx, size=point_idx.size, replace=True)

    lift_a = capture_efficiency(score_a, point_idx, area, pool)
    lift_b = capture_efficiency(score_b, point_idx, area, pool)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sample = resample()
        deltas[i] = (capture_efficiency(score_a, sample, area, pool)
                     - capture_efficiency(score_b, sample, area, pool))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "lift_a": lift_a, "lift_b": lift_b, "delta": lift_a - lift_b,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "share_delta_le_0": float((deltas <= 0).mean()), "n_boot": n_boot,
    }


def spatial_null_pvalue(
    score: np.ndarray, point_idx: np.ndarray, meta: integro_grid.GridMeta,
    area: float | None = None, pool: np.ndarray | None = None,
    n_shifts: int | None = None, seed: int | None = None,
) -> dict[str, float]:
    """Сдвиговый (тороидальный) тест значимости lift@area.

    Главный тест значимости протокола: карта скора циклически сдвигается по
    обеим осям (плюс случайные отражения), точки и пул фиксированы — null
    сохраняет пространственную автокорреляцию карты, в отличие от перестановок
    точек. Невалидные ячейки (-inf/не конечные) становятся NaN и двигаются
    вместе с картой; порог — nanquantile по ячейкам пула, точка/ячейка с NaN —
    всегда промах. p — со сглаживанием Фипсона-Смита:
    ``(1 + #{null_lift >= observed}) / (1 + n_shifts)``.
    """
    area = area or config.VER_AREA
    n_shifts = n_shifts or config.VER_N_SHIFTS
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    point_idx = np.asarray(point_idx, dtype=int)

    img = np.asarray(score, dtype=float).copy()
    img[~np.isfinite(img)] = np.nan
    img = img.reshape(meta.prf, meta.pic)
    pool_idx = np.arange(img.size) if pool is None else np.asarray(pool, dtype=int)

    def lift_of(flat: np.ndarray) -> float:
        ref = flat[pool_idx]
        if not np.isfinite(ref).any():
            return 0.0
        thr = np.nanquantile(ref, 1.0 - area)
        ref_f = np.where(np.isnan(ref), -np.inf, ref)
        realized = float((ref_f >= thr).mean())
        if realized <= 0 or point_idx.size == 0:
            return 0.0
        vals = np.where(np.isnan(flat[point_idx]), -np.inf, flat[point_idx])
        return float((vals >= thr).mean()) / realized

    observed = lift_of(img.ravel())
    null = np.empty(n_shifts)
    for i in range(n_shifts):
        a = img
        if rng.integers(2):
            a = np.flipud(a)
        if rng.integers(2):
            a = np.fliplr(a)
        a = np.roll(a, (int(rng.integers(meta.prf)), int(rng.integers(meta.pic))),
                    axis=(0, 1))
        null[i] = lift_of(a.ravel())
    p = float((1 + (null >= observed).sum()) / (1 + n_shifts))
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)), "p": p}


def bootstrap_diff_catch(
    score_a: np.ndarray, score_b: np.ndarray, catchments: list,
    area: float | None = None, pool: np.ndarray | None = None,
    n_boot: int | None = None, seed: int | None = None,
    clusters: np.ndarray | None = None, ci: tuple = (5.0, 95.0),
) -> dict[str, float]:
    """То же, что :func:`bootstrap_diff`, но единица ресэмплинга — водосбор.

    Ресэмплируются целые водосборы (или их пространственные кластеры), карты
    фиксированы. Интервал по умолчанию 90% — как в предзарегистрированном
    критерии превосходства.
    """
    from src import catchments as _catch
    area = area or config.VER_AREA
    n_boot = n_boot or config.VER_N_BOOT
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    idx = np.arange(len(catchments))

    if clusters is not None:
        clusters = np.asarray(clusters)
        if clusters.size != idx.size:
            raise ValueError("bootstrap_diff_catch: clusters должен быть длины "
                             "списка водосборов")
        labels = np.unique(clusters)
        members = {c: idx[clusters == c] for c in labels}

        def resample():
            chosen = rng.choice(labels, size=labels.size, replace=True)
            return np.concatenate([members[c] for c in chosen])
    else:
        def resample():
            return rng.choice(idx, size=idx.size, replace=True)

    cap = _catch.catchment_capture
    lift_a = cap(score_a, catchments, area, pool)
    lift_b = cap(score_b, catchments, area, pool)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        s = [catchments[j] for j in resample()]
        deltas[i] = cap(score_a, s, area, pool) - cap(score_b, s, area, pool)
    lo, hi = np.percentile(deltas, list(ci))
    return {"lift_a": lift_a, "lift_b": lift_b, "delta": lift_a - lift_b,
            "ci_lo": float(lo), "ci_hi": float(hi),
            "share_delta_le_0": float((deltas <= 0).mean()), "n_boot": n_boot}


def spatial_null_catch(
    score: np.ndarray, catchments: list, meta: integro_grid.GridMeta,
    area: float | None = None, pool: np.ndarray | None = None,
    n_shifts: int | None = None, seed: int | None = None,
) -> dict[str, float]:
    """Сдвиговый null для водосборной метрики: двигается карта, не водосборы.

    Как и в точечном варианте, тороидальный сдвиг с отражениями сохраняет
    пространственную автокорреляцию карты — то есть null отвечает на вопрос
    «а не даёт ли такой результат любая карта с такой же зернистостью?».
    """
    from src import catchments as _catch
    area = area or config.VER_AREA
    n_shifts = n_shifts or config.VER_N_SHIFTS
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)

    img = np.asarray(score, dtype=float).copy()
    img[~np.isfinite(img)] = np.nan
    img = img.reshape(meta.prf, meta.pic)

    def cap_of(flat):
        f = np.where(np.isnan(flat), -np.inf, flat)
        return _catch.catchment_capture(f, catchments, area, pool)

    observed = cap_of(img.ravel())
    null = np.empty(n_shifts)
    for i in range(n_shifts):
        a = img
        if rng.integers(2):
            a = np.flipud(a)
        if rng.integers(2):
            a = np.fliplr(a)
        a = np.roll(a, (int(rng.integers(meta.prf)), int(rng.integers(meta.pic))),
                    axis=(0, 1))
        null[i] = cap_of(a.ravel())
    p = float((1 + (null >= observed).sum()) / (1 + n_shifts))
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)), "p": p}


# --------------------------------------------------------------------------
# Планки и сравнение карт
# --------------------------------------------------------------------------

def naive_hydro_score(df: pd.DataFrame) -> np.ndarray:
    """Наивная планка: минус расстояние до гидросети (линии и ареалы).

    Без геологического содержания; на 65 фоновых точках прошлой фазы давала
    lift@10% = 4.15 — выше обоих методов. Любой содержательный метод обязан
    сопоставляться с ней на смещённых точках.
    """
    d = np.minimum(df["dist_dnl"].to_numpy(), df["dist_dnara"].to_numpy())
    return -d


def random_score(n: int, seed: int | None = None) -> np.ndarray:
    """Случайная планка (lift ~= 1 в среднем)."""
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    return rng.random(n)


def spearman_maps(
    score_a: np.ndarray, score_b: np.ndarray, pool: np.ndarray | None = None,
) -> float:
    """Ранговая согласованность двух карт на пуле валидных ячеек."""
    if pool is not None:
        score_a, score_b = score_a[pool], score_b[pool]
    rho, _ = stats.spearmanr(score_a, score_b)
    return float(rho)


def run_protocol(
    scores: dict[str, np.ndarray], own: list[str], meta, valid: np.ndarray,
    alpha: float, baseline: str = "criterial", log=None,
) -> dict[str, pd.DataFrame]:
    """Полный протокол заверки для набора карт: P-A, null, бутстрэп, вердикт.

    Вынесено из прогонов (``experiments/mae_train``, ``transfer_nn``,
    ``transfer_cnn``), потому что сравнение методов имеет смысл только если оно
    выполнено ОДНИМ И ТЕМ ЖЕ кодом: любая правка протокола под конкретный
    прогон делает этапы несопоставимыми.

    ``own`` — карты-претенденты (сравниваются с ``baseline``), ``alpha`` —
    предзарегистрированный порог значимости с НАКОПИТЕЛЬНОЙ поправкой на число
    опробованных конфигураций. Возвращает словарь с ключами ``pa``, ``null``,
    ``boot``, ``verdict``.
    """
    pool = np.flatnonzero(valid)
    pts = load_verification_points(meta)
    all_cells = filter_points(pts["cell"].to_numpy(), valid)
    unb = filter_points(unbiased_cells(pts), valid)
    sets = [("all", all_cells, point_clusters(all_cells, meta)),
            ("unbiased_strict", unb, point_clusters(unb, meta))]
    if log:
        log(f"точки: все {all_cells.size}, несмещённые {unb.size}; "
            f"шаг метрики 1 точка = {1.0 / max(unb.size, 1) / config.VER_AREA:.2f} "
            f"lift; порог значимости {alpha:.5f}")

    pa = pd.concat([pa_curve(scores[n], c, pool=pool).assign(method=n, points=pn)
                    for n in scores for pn, c, _ in sets], ignore_index=True)
    null = pd.DataFrame(
        [{"method": n, "points": pn,
          **spatial_null_pvalue(scores[n], c, meta, pool=pool)}
         for n in scores for pn, c, _ in sets])
    if log:
        log(f"сдвиговый null ({config.VER_N_SHIFTS} сдвигов) готов")
    boot = pd.DataFrame(
        [{"a": n, "b": baseline, "points": pn, "resample": kind,
          **bootstrap_diff(scores[n], scores[baseline], c, pool=pool,
                           clusters=cl)}
         for n in own for pn, c, cls in sets
         for kind, cl in (("point", None), ("cluster", cls))])
    if log:
        log("бутстрэп готов")

    rows = []
    for n in own:
        bt = boot[(boot["a"] == n) & (boot["points"] == "unbiased_strict")
                  & (boot["resample"] == "cluster")].iloc[0]
        p = null[(null["method"] == n)
                 & (null["points"] == "unbiased_strict")]["p"].iloc[0]
        rows.append({"method": n,
                     "lift": capture_efficiency(scores[n], unb, pool=pool),
                     "lift_criterial": capture_efficiency(scores[baseline], unb,
                                                          pool=pool),
                     "delta": bt["delta"], "ci_lo": bt["ci_lo"],
                     "ci_hi": bt["ci_hi"], "p_shift": p, "alpha": alpha,
                     "cond1_ci_lo_gt_0": bool(bt["ci_lo"] > 0),
                     "cond2_p_lt_alpha": bool(p < alpha),
                     "superior": bool(bt["ci_lo"] > 0 and p < alpha)})
    verdict = pd.DataFrame(rows).sort_values("lift", ascending=False)
    return {"pa": pa, "null": null, "boot": boot, "verdict": verdict,
            "cells_all": all_cells, "cells_unbiased": unb}


def map_agreement(
    score_a: np.ndarray, score_b: np.ndarray,
    area: float | None = None, pool: np.ndarray | None = None,
) -> np.ndarray:
    """3-классная карта расхождений по top-``area`` каждой карты.

    Коды на ячейку: 0 — ни один метод, 1 — только A, 2 — только B,
    3 — оба (первоочередная заверка). Пороги — по пулу валидных ячеек;
    вне пула всегда 0.
    """
    area = area or config.VER_AREA
    ref_a = score_a if pool is None else score_a[pool]
    ref_b = score_b if pool is None else score_b[pool]
    top_a = score_a >= np.quantile(ref_a, 1.0 - area)
    top_b = score_b >= np.quantile(ref_b, 1.0 - area)
    out = top_a.astype(int) + 2 * top_b.astype(int)
    if pool is not None:
        in_pool = np.zeros(score_a.size, dtype=bool)
        in_pool[pool] = True
        out[~in_pool] = 0
    return out
