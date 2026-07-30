"""Обучение на результате критериального анализа (пункт 3 постановки).

Постановка: «для обучения использовать результат критериального анализа,
в пределах территории выделяется три потенциальных рудных объекта, в обучение
можно взять два объекта из трёх, один оставить как контрольный».

Циркулярность (см. docstring :mod:`src.model`): если признаки, из которых
считается критериальный прогноз (``dist_tect*``, ``dist_magm`` и т.д.), остаются
в обучении, ML тривиально восстанавливает формулу критериального. Здесь
используются только НЕЗАВИСИМЫЕ признаки (геофизика, Landsat, рельеф, гидросеть —
``CRIT_EXCLUDE_FACTOR_FEATURES`` в :mod:`src.config`), а валидация — на
object-level leave-one-object-out (не на пиксельной блочной CV: блочная CV с
объектом, лежащим сразу в нескольких блоках, проверяет интерполяцию внутри
известной аномалии, а не обобщение на неизвестный объект).

Три объекта выделяются связными компонентами по АБСОЛЮТНОМУ порогу 0.15 на
нативной шкале ``prognoz`` (меньше = перспективнее), рекомендованному самой
методичкой ГИС Интегро — а не квантилем (квантиль даёт совсем другую, гораздо
более обширную выборку ячеек, не соответствующую «трём объектам» из постановки).

Порядок вызова: :func:`build_dataset` → :func:`leave_one_object_out` →
:func:`permutation_significance` (на каждый объект) → опционально
:func:`real_point_verification` (пункт 5, сверка с реальными рудопроявлениями,
которые нигде в конвейере не участвовали).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from . import config, data_loader, gold_features, integro_grid
from .model import BackgroundEnsemble
from .validation import _coverage


def load_prognoz_grid() -> tuple[integro_grid.GridMeta, np.ndarray]:
    """Прочитать сетку критериального прогноза (``meta``, массив ``prognoz``)."""
    meta, arrays = integro_grid.load_pgrid_dataset(config.GOLD_TARGET_PGRID)
    return meta, arrays[config.CRIT_TARGET_PROPERTY]


def label_ore_objects(
    prognoz: np.ndarray,
    threshold: float | None = None,
    min_cells: int | None = None,
    n_objects: int | None = None,
) -> np.ndarray:
    """Выделить связные компоненты наиболее перспективных ячеек — «рудные объекты».

    Ячейка перспективна при ``prognoz <= threshold`` (абсолютный порог, не квантиль).
    Компоненты меньше ``min_cells`` считаются шумом и отбрасываются. Возвращает
    массив формы ``prognoz.shape`` с метками ``0`` (фон) и ``1..n_objects``
    (ранг по убыванию размера — object 1 самый крупный).
    """
    threshold = config.CRIT_TARGET_THRESHOLD if threshold is None else threshold
    min_cells = config.CRIT_MIN_OBJECT_CELLS if min_cells is None else min_cells
    n_objects = config.CRIT_N_OBJECTS if n_objects is None else n_objects

    mask = prognoz <= threshold
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_lab = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    candidates = [lid for lid in range(1, n_lab + 1) if sizes[lid] >= min_cells]
    ranked = sorted(candidates, key=lambda lid: -sizes[lid])[:n_objects]

    out = np.zeros_like(labeled)
    for rank, lid in enumerate(ranked, start=1):
        out[labeled == lid] = rank
    return out


def training_features() -> pd.DataFrame:
    """Матрица признаков без факторных слоёв критериального анализа (независимая)."""
    df = gold_features.load_feature_matrix()
    drop = [c for c in config.CRIT_EXCLUDE_FACTOR_FEATURES if c in df.columns]
    return df.drop(columns=drop)


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], integro_grid.GridMeta]:
    """Собрать ``(X, labels_flat, coords, feature_names, meta)`` для LOO-CV.

    ``labels_flat`` — метка объекта на ячейку, в том же C-порядке (строка 0 —
    север), что и признаки/сетка prognoz: ``1..CRIT_N_OBJECTS`` — объекты,
    ``0`` — чистый фон, ``-1`` — критериально-перспективные ячейки
    (``prognoz <= CRIT_TARGET_THRESHOLD``), не вошедшие в топ-``CRIT_N_OBJECTS``
    компонент. Метка ``-1`` не даёт таким ячейкам попасть в фон обучения:
    фон из перспективных ячеек занижал бы контраст объект/фон.
    """
    meta, prognoz = load_prognoz_grid()
    labels = label_ore_objects(prognoz)
    labels[(labels == 0) & (prognoz <= config.CRIT_TARGET_THRESHOLD)] = -1
    labels_flat = labels.ravel()

    feat_df = training_features().fillna(0)
    if len(feat_df) != labels_flat.size:
        raise ValueError(
            f"Признаки ({len(feat_df)} ячеек) не совпадают по размеру со "
            f"сеткой критериального прогноза ({labels_flat.size} ячеек)"
        )

    x, y = meta.cell_centers()
    coords = np.column_stack([x.ravel(), y.ravel()])
    X = feat_df.to_numpy(dtype=float)
    return X, labels_flat, coords, list(feat_df.columns), meta


def leave_one_object_out(
    X: np.ndarray, labels_flat: np.ndarray, coords: np.ndarray, seed: int | None = None,
) -> list[dict]:
    """Leave-one-object-out: обучение на 2 из 3 объектов, третий — контроль.

    Фон для обучения берётся случайно из ячеек чистого фона (``labels_flat == 0``,
    т.е. без критериально-перспективных ячеек с меткой ``-1`` — см.
    :func:`build_dataset`), за вычетом буфера ``CRIT_HOLDOUT_BUFFER_M`` вокруг
    held-out объекта (иначе часть фона пространственно совпадает с окрестностью
    контроля и завышает lift). Тот же буфер применяется и к обучающим
    положительным ячейкам — ячейки обучающих объектов ближе буфера к контролю
    исключаются, чтобы модель не выучивала окрестность контроля напрямую.
    Метрики coverage/lift считаются по eval-пулу (контрольный объект + чистый
    фон вне обучающей выборки), а не по всей сетке: порог top-X% по всей сетке
    включает обучающие объекты и ``-1``-ячейки, которые съедают бюджет top-X%
    по-разному в разных фолдах и делают фолды несравнимыми.

    Возвращает список словарей по каждому объекту: сводка метрик (``summary``),
    прогноз по всей сетке (``score_all``), позиции held-out ячеек (``held_idx``),
    индексы обучения (``train_idx``) и eval-пул (``eval_pool``) — оба нужны
    permutation-тесту.
    """
    seed = config.CRIT_SEED if seed is None else seed
    all_idx = np.arange(len(labels_flat))
    results = []

    for obj in range(1, config.CRIT_N_OBJECTS + 1):
        held_idx = all_idx[labels_flat == obj]
        if held_idx.size == 0:
            continue

        tree = cKDTree(coords[held_idx])
        dist_to_held, _ = tree.query(coords)
        far_from_held = dist_to_held > config.CRIT_HOLDOUT_BUFFER_M

        train_pos_all = all_idx[(labels_flat != obj) & (labels_flat > 0)]
        train_obj_idx = train_pos_all[far_from_held[train_pos_all]]
        if train_obj_idx.size == 0:
            raise ValueError(
                f"Все ячейки обучающих объектов попали в буфер "
                f"{config.CRIT_HOLDOUT_BUFFER_M} м вокруг объекта {obj} — "
                f"уменьшите CRIT_HOLDOUT_BUFFER_M"
            )
        bg_candidates = all_idx[(labels_flat == 0) & far_from_held]

        rng = np.random.default_rng(seed)
        n_bg = min(config.CRIT_N_BACKGROUND, len(bg_candidates))
        bg_idx = rng.choice(bg_candidates, size=n_bg, replace=False)

        train_idx = np.concatenate([train_obj_idx, bg_idx])
        y = np.concatenate([np.ones(len(train_obj_idx)), np.zeros(len(bg_idx))]).astype(int)

        model = BackgroundEnsemble(random_state=seed)
        model.fit(X[train_idx], y)
        score_all = model.predict_proba(X)[:, 1]

        # eval-пул: контрольный объект + чистый фон, не участвовавший в обучении
        in_train = np.zeros(labels_flat.size, dtype=bool)
        in_train[train_idx] = True
        eval_pool = all_idx[((labels_flat == obj) | (labels_flat == 0)) & ~in_train]

        summary = {
            "object": obj,
            "n_cells": int(held_idx.size),
            "n_train_pos": int(train_obj_idx.size),
            "n_train_pos_total": int(train_pos_all.size),   # до вычета буфера — потеря видна в отчёте
            "n_bg_candidates": int(bg_candidates.size),
            "n_perspective_excluded": int((labels_flat == -1).sum()),
            "n_background": int(bg_idx.size),
            "n_eval_pool": int(eval_pool.size),
        }
        for area in config.CRIT_AREAS:
            cov = _coverage(score_all, held_idx, area, pool=eval_pool)
            summary[f"coverage@{int(round(area * 100))}%"] = cov
            summary[f"lift@{int(round(area * 100))}%"] = cov / area

        results.append({
            "summary": summary,
            "score_all": score_all,
            "held_idx": held_idx,
            "train_idx": train_idx,
            "eval_pool": eval_pool,
        })
    return results


def _random_object_placement(
    rel_r: np.ndarray, rel_c: np.ndarray, shape: tuple[int, int],
    valid_flat: np.ndarray, rng: np.random.Generator,
) -> np.ndarray | None:
    """Случайное форм-сохраняющее размещение объекта: сдвиг + поворот/отражение.

    ``rel_r``/``rel_c`` — координаты ячеек объекта относительно его bbox.
    Возвращает плоские индексы размещения или ``None``, если выпавшая позиция
    задевает недопустимые ячейки (``valid_flat`` == False) — вызывающий код
    просто пробует ещё раз.
    """
    n_rows, n_cols = shape
    h, w = int(rel_r.max()) + 1, int(rel_c.max()) + 1
    t = int(rng.integers(8))            # 8 элементов группы диэдра: повороты/отражения
    if t == 0:
        rr, cc = rel_r, rel_c
    elif t == 1:
        rr, cc = rel_r, w - 1 - rel_c
    elif t == 2:
        rr, cc = h - 1 - rel_r, rel_c
    elif t == 3:
        rr, cc = h - 1 - rel_r, w - 1 - rel_c
    elif t == 4:
        rr, cc = rel_c, rel_r
    elif t == 5:
        rr, cc = rel_c, h - 1 - rel_r
    elif t == 6:
        rr, cc = w - 1 - rel_c, rel_r
    else:
        rr, cc = w - 1 - rel_c, h - 1 - rel_r
    hh, ww = int(rr.max()) + 1, int(cc.max()) + 1
    if hh > n_rows or ww > n_cols:
        return None
    row0 = int(rng.integers(n_rows - hh + 1))
    col0 = int(rng.integers(n_cols - ww + 1))
    idx = (rr + row0) * n_cols + (cc + col0)
    return idx if valid_flat[idx].all() else None


def permutation_significance(
    score_all: np.ndarray,
    held_idx: np.ndarray,
    train_idx: np.ndarray,
    area: float | None = None,
    n_perm: int | None = None,
    seed: int | None = None,
    eval_pool: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
    placement_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Значим ли захват held-out объекта по сравнению со случайным участком той же площади.

    Нулевая гипотеза: модель, обученная на двух объектах, не выделяет held-out
    объект среди прочих непройденных ячеек. Порог top-``area`` — по ``eval_pool``
    (контрольный объект + чистый фон вне обучения, см. :func:`leave_one_object_out`).

    Null-распределение: при заданном ``shape`` (форма сетки, ``(prf, pic)``) —
    форм-сохраняющие размещения контура held-out объекта (случайный сдвиг +
    поворот/отражение) по ячейкам ``placement_mask``. Разрозненные случайные
    ячейки (режим без ``shape``, оставлен для обратной совместимости) занижают
    дисперсию null: у компактного автокоррелированного пятна разброс coverage
    много больше, чем у независимых ячеек, — тест выходит антиконсервативным.

    ``p_value`` — со сглаживанием ``(1 + #{null >= obs}) / (1 + n_perm)``
    (Phipson & Smyth), поэтому p = 0.000 невозможен по построению. При
    наблюдаемом lift = 0 значение p ~ 1 вырождено и означает «объект не
    захвачен», а не результат теста. При ``CRIT_N_OBJECTS == 3`` мощность
    ограничена малым n — трактовать индикативно.
    """
    area = config.CRIT_PERM_AREA if area is None else area
    n_perm = config.CRIT_PERM_N if n_perm is None else n_perm
    seed = config.CRIT_SEED if seed is None else seed

    observed = _coverage(score_all, held_idx, area, pool=eval_pool) / area
    rng = np.random.default_rng(seed)

    extra: dict[str, float] = {}
    if shape is not None:
        rows, cols = np.divmod(held_idx, shape[1])
        rel_r, rel_c = rows - rows.min(), cols - cols.min()
        valid = (
            np.ones(score_all.size, dtype=bool) if placement_mask is None else placement_mask
        )
        in_pool = None
        if eval_pool is not None:
            in_pool = np.zeros(score_all.size, dtype=bool)
            in_pool[eval_pool] = True
        null_list: list[float] = []
        unique_placements: set[bytes] = set()
        n_attempts = 0
        for _ in range(n_perm * 200):
            if len(null_list) >= n_perm:
                break
            n_attempts += 1
            idx = _random_object_placement(rel_r, rel_c, shape, valid, rng)
            if idx is None:
                continue
            # coverage нуля — только по ячейкам eval-пула: у наблюдаемого объекта
            # весь контур в пуле, а размещение может задеть train-фоновые ячейки
            # с прижатым обучением скором — иначе null разбавлен заведомыми
            # промахами и тест антиконсервативен
            eval_idx = idx if in_pool is None else idx[in_pool[idx]]
            if eval_idx.size < max(1, idx.size // 2):
                continue                      # контур лёг в основном мимо пула — неинформативен
            unique_placements.add(np.sort(idx).tobytes())
            null_list.append(_coverage(score_all, eval_idx, area, pool=eval_pool) / area)
        if len(null_list) < n_perm:
            raise ValueError(
                f"Удалось разместить контур объекта только {len(null_list)} раз из "
                f"{n_perm} — placement_mask слишком тесная для формы объекта"
            )
        null = np.array(null_list)
        # доля принятых попыток и уникальность размещений: при тесной маске
        # повторы снижают фактическое разрешение p-value ниже номинального
        extra = {
            "placement_accept_rate": float(len(null_list) / n_attempts),
            "n_unique_placements": float(len(unique_placements)),
        }
    else:
        pool = np.setdiff1d(np.arange(len(score_all)), train_idx) if eval_pool is None else eval_pool
        n = len(held_idx)
        null = np.array([
            _coverage(score_all, rng.choice(pool, size=n, replace=False), area, pool=eval_pool) / area
            for _ in range(n_perm)
        ])
    return {
        "observed_lift": float(observed),
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + (null >= observed).sum()) / (1 + null.size)),
        **extra,
    }


def run_leave_one_object_out_cv(seed: int | None = None) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """End-to-end: датасет → LOO-CV по 3 объектам → permutation-значимость.

    Возвращает ``(summary_df, loo_results, perm_df)``: сводную таблицу lift/coverage
    по объектам и площадям, сырые результаты LOO (для :func:`real_point_verification`)
    и таблицу значимости по объектам.
    """
    X, labels_flat, coords, _feature_names, meta = build_dataset()
    loo_results = leave_one_object_out(X, labels_flat, coords, seed=seed)

    perm_rows = []
    for r in loo_results:
        obj = r["summary"]["object"]
        # размещать контур можно по чистому фону и месту самого объекта;
        # обучающие объекты и -1-ячейки для null-размещений закрыты
        placement_mask = (labels_flat == 0) | (labels_flat == obj)
        perm = permutation_significance(
            r["score_all"], r["held_idx"], r["train_idx"], seed=seed,
            eval_pool=r["eval_pool"], shape=(meta.prf, meta.pic), placement_mask=placement_mask,
        )
        perm["object"] = obj
        perm_rows.append(perm)

    summary_df = pd.DataFrame([r["summary"] for r in loo_results])
    perm_df = pd.DataFrame(perm_rows)
    return summary_df, loo_results, perm_df


def seed_sweep(n_seeds: int | None = None) -> pd.DataFrame:
    """Сид-свип LOO: распределение lift по сидам вместо точечной оценки.

    Смена сида меняет подвыборку фона и сиды членов RF/GB; измеренный шумовой
    пол одного сида — порядка ±30% lift (фолд 1: 0.81 -> 1.05 только от смены
    сидов членов), то есть сопоставим с обсуждаемыми эффектами. Поэтому в отчёт
    идут медиана и разброс по ``CRIT_SWEEP_N_SEEDS`` прогонам, а не одно число.
    Возвращает «длинную» таблицу: строка = (seed, object) со всеми метриками
    из ``summary``. Агрегация — :func:`summarize_sweep`.
    """
    n_seeds = config.CRIT_SWEEP_N_SEEDS if n_seeds is None else n_seeds
    X, labels_flat, coords, _feature_names, _meta = build_dataset()
    rows = []
    for s in range(n_seeds):
        for r in leave_one_object_out(X, labels_flat, coords, seed=s):
            rows.append({"seed": s, **r["summary"]})
    return pd.DataFrame(rows)


def summarize_sweep(sweep_df: pd.DataFrame, area: float | None = None) -> pd.DataFrame:
    """Сводка сид-свипа по объектам: медиана, квартели, min/max lift и доля сидов с lift > 1."""
    area = config.CRIT_PERM_AREA if area is None else area
    col = f"lift@{int(round(area * 100))}%"
    g = sweep_df.groupby("object")[col]
    out = pd.DataFrame({
        "median": g.median(),
        "q25": g.quantile(0.25),
        "q75": g.quantile(0.75),
        "min": g.min(),
        "max": g.max(),
        "share_lift>1": g.apply(lambda s: float((s > 1.0).mean())),
        "n_seeds": g.size(),
    })
    return out.round(3).reset_index()


def load_real_points(meta: integro_grid.GridMeta) -> np.ndarray:
    """Плоские индексы ячеек сетки prognoz с реальными точками рудопроявлений.

    Эти точки не участвуют нигде в конвейере (ни в критериальном анализе, ни
    в обучении на нём) — независимая проверка пункта 5.
    """
    base_dir = data_loader.find_base_dir()
    shp_dir = base_dir / config.SHP_SUBDIR
    alias_dir = config.PROJECT_ROOT / "data" / "processed" / "_shp_aliases_tmp"
    alias_dir.mkdir(parents=True, exist_ok=True)
    aliases = data_loader.prepare_ascii_aliases(shp_dir, alias_dir)
    mask_layer = data_loader.load_layer(aliases[config.LAYER_FILES["mask"]])
    points = data_loader.collect_points(mask_layer.crs, aliases)
    if points is None or len(points) == 0:
        return np.array([], dtype=int)

    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    row = np.floor((meta.y_top - ys) / meta.dy).astype(int)
    col = np.floor((xs - meta.x0) / meta.dx).astype(int)
    valid = (row >= 0) & (row < meta.prf) & (col >= 0) & (col < meta.pic)
    flat_idx = row[valid] * meta.pic + col[valid]
    return np.unique(flat_idx)


def real_point_verification(
    loo_results: list[dict], prognoz: np.ndarray, meta: integro_grid.GridMeta,
    areas: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Пункт 5: ML (критериальный таргет) против baseline critериального на реальных точках.

    Модель, обученная лишь на критериальном прогнозе, в лучшем случае воспроизводит
    сам критериальный, но не превосходит его — если только не проверить обе стороны
    на данных, которые критериальный анализ вообще не видел (реальные точки).
    ``coverage_ml`` использует усреднение по всем LOO-моделям (каждая обучена без
    одного из трёх объектов) — приближение независимого от полного набора прогноза.
    """
    areas = areas or config.CRIT_AREAS
    point_idx = load_real_points(meta)
    if point_idx.size == 0:
        return pd.DataFrame()

    ml_score = np.mean([r["score_all"] for r in loo_results], axis=0)
    baseline_score = -prognoz.ravel()  # меньше prognoz = перспективнее -> инверсия для единой полярности

    rows = []
    for area in areas:
        rows.append({
            "area": area,
            "coverage_ml": _coverage(ml_score, point_idx, area),
            "lift_ml": _coverage(ml_score, point_idx, area) / area,
            "coverage_baseline": _coverage(baseline_score, point_idx, area),
            "lift_baseline": _coverage(baseline_score, point_idx, area) / area,
            "n_points": int(point_idx.size),
        })
    return pd.DataFrame(rows)
