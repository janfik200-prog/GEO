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
from .validation import _coverage


# --------------------------------------------------------------------------
# Точки заверки
# --------------------------------------------------------------------------

def load_verification_points(meta: integro_grid.GridMeta) -> pd.DataFrame:
    """Точки «геохимическое_опробование» -> DataFrame(cell, code).

    В отличие от :func:`src.criterial_target.load_real_points` сохраняет тип
    точки (поле ``code``: 1 — литопробы в рыхлых, 2 — по коренным, 4 —
    проявления U/TR, 5 — шлихи Au), чтобы протокол мог выделять несмещённое
    подмножество ``config.VER_UNBIASED_CODES``. Ячейка может содержать
    несколько точек — строки агрегируются до уникальных ячеек, code берётся
    минимальный «несмещённый» при конфликте (2/4 приоритетнее 1/5: если в
    ячейке и рыхлая, и коренная проба — ячейка считается коренной).
    """
    base_dir = data_loader.find_base_dir()
    shp_dir = base_dir / config.SHP_SUBDIR
    alias_dir = config.PROJECT_ROOT / "data" / "processed" / "_shp_aliases_tmp"
    alias_dir.mkdir(parents=True, exist_ok=True)
    aliases = data_loader.prepare_ascii_aliases(shp_dir, alias_dir)
    mask_layer = data_loader.load_layer(aliases[config.LAYER_FILES["mask"]])
    points = data_loader.collect_points(mask_layer.crs, aliases)
    if points is None or len(points) == 0:
        return pd.DataFrame(columns=["cell", "code"])

    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    row = np.floor((meta.y_top - ys) / meta.dy).astype(int)
    col = np.floor((xs - meta.x0) / meta.dx).astype(int)
    inside = (row >= 0) & (row < meta.prf) & (col >= 0) & (col < meta.pic)
    cells = row[inside] * meta.pic + col[inside]
    codes = points["code"].to_numpy()[inside].astype(int)

    df = pd.DataFrame({"cell": cells, "code": codes})
    unbiased = set(config.VER_UNBIASED_CODES)
    df["_pri"] = np.where(df["code"].isin(list(unbiased)), 0, 1)
    df = (df.sort_values(["cell", "_pri"]).drop_duplicates("cell")
            .drop(columns="_pri").reset_index(drop=True))
    return df


def unbiased_cells(points: pd.DataFrame) -> np.ndarray:
    """Ячейки несмещённого к гидросети подмножества (коренные + проявления)."""
    sel = points["code"].isin(list(config.VER_UNBIASED_CODES))
    return points.loc[sel, "cell"].to_numpy()


def filter_points(point_idx: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Оставить только точки в валидных ячейках (вода/артефакты не заверяют)."""
    point_idx = np.asarray(point_idx, dtype=int)
    return point_idx[valid_mask[point_idx]]


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------

def pa_curve(
    score: np.ndarray, point_idx: np.ndarray,
    areas: tuple[float, ...] | None = None, pool: np.ndarray | None = None,
) -> pd.DataFrame:
    """P-A кривая: coverage и lift по сетке площадей ``areas``.

    ``pool`` — индексы ячеек, по которым берётся порог top-X% (обычно валидные
    ячейки из :func:`src.cell_mask.build_valid_mask`); полярность скора —
    «выше = перспективнее».
    """
    areas = areas or config.VER_AREAS
    rows = []
    for area in areas:
        cov = _coverage(score, point_idx, area, pool=pool)
        rows.append({"area": area, "coverage": cov, "lift": cov / area,
                     "n_points": int(np.asarray(point_idx).size)})
    return pd.DataFrame(rows)


def capture_efficiency(
    score: np.ndarray, point_idx: np.ndarray,
    area: float | None = None, pool: np.ndarray | None = None,
) -> float:
    """lift@area — точечный срез P-A кривой (основная скалярная метрика)."""
    area = area or config.VER_AREA
    return _coverage(score, point_idx, area, pool=pool) / area


def bootstrap_diff(
    score_a: np.ndarray, score_b: np.ndarray, point_idx: np.ndarray,
    area: float | None = None, pool: np.ndarray | None = None,
    n_boot: int | None = None, seed: int | None = None,
) -> dict[str, float]:
    """Бутстрэп по точкам: значима ли разность lift@area (A − B).

    Ресемплируются точки (с возвращением), карты фиксированы — оценивается
    только неопределённость малого числа точек заверки. Возвращает наблюдаемую
    дельту, перцентильный 95% ДИ и долю ресемплов с дельтой <= 0.
    """
    area = area or config.VER_AREA
    n_boot = n_boot or config.VER_N_BOOT
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    point_idx = np.asarray(point_idx, dtype=int)

    lift_a = capture_efficiency(score_a, point_idx, area, pool)
    lift_b = capture_efficiency(score_b, point_idx, area, pool)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(point_idx, size=point_idx.size, replace=True)
        deltas[i] = (capture_efficiency(score_a, sample, area, pool)
                     - capture_efficiency(score_b, sample, area, pool))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "lift_a": lift_a, "lift_b": lift_b, "delta": lift_a - lift_b,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "share_delta_le_0": float((deltas <= 0).mean()), "n_boot": n_boot,
    }


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
