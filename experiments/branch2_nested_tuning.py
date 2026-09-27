"""Ветка 2: радиус сглаживания и гиперпараметры подбираются ВНУТРИ обучения.

ЗАЧЕМ. Два методологических долга, оба видны после ветки 1.

1. Радиус постобработочного сглаживания (10 км) был выбран по той самой метрике,
   по которой мы отчитываемся, — это подгонка под тест. Здесь радиус выбирается
   вложенно: внутри обучающей части каждой полосы, на её собственных внутренних
   фолдах, а тестовая полоса при выборе не участвует ни одним значением.
2. Гиперпараметры бустинга не подбирались в проекте ни разу — везде значения из
   `config`. Здесь они подбираются тем же вложенным способом.

ЧЕМ ВЫБИРАЕМ. Не совпадением верхних 10% напрямую: ветка 1 показала, что эта
метрика берётся общим наклоном карты (рекорд 46.7% превращается в 15.5% после
снятия тренда при случайном уровне 10%). Поэтому критерий выбора — совпадение
ПОСЛЕ снятия тренда: из прогноза и из критериальной оценки вычитается своя
квадратичная поверхность по координатам, и сравниваются верхние 10% остатков.
Для контраста записывается и то, что выбрал бы старый критерий.

ПРОТОКОЛ. Внешний — прежний: 4 полосы по X + 4 по Y, разрыв 15 км. Внутренний —
обучающая часть внешнего фолда режется на 4 полосы ПЕРПЕНДИКУЛЯРНОЙ оси (внешний
фолд по X -> внутренние по Y) с тем же разрывом 15 км. Перпендикулярное деление
взято потому, что при делении по той же оси внутренняя полоса, соседняя с
тестовой, почти целиком съедается разрывом (924 ячейки из ~5700). При выборе
радиуса и параметров модель ни разу не видит меток ближе 15 км от оцениваемых
ячеек. Сглаживание —
только внутри полосы (ядро не заглядывает в соседнюю полосу, где училась другая
модель).

ЧТО СРАВНИВАЕТСЯ. Выбранный вложенно радиус против трёх фиксированных: 0 (без
сглаживания), 2 км и 10 км (та самая цифра из отчётов). Парные тесты — по 8
полосам.

Выход: `outputs/metrics/branch2_nested_folds.csv` (по полосам, что выбрано),
       `outputs/metrics/branch2_nested_selection.csv` (вся внутренняя поверхность
        выбора: фолд x радиус x набор параметров),
       `outputs/metrics/branch2_nested_maps.csv` (карты целиком),
       `outputs/metrics/branch2_nested_paired.csv` (парные тесты по полосам),
       `data/processed/branch2_inner_cache.npz`,
       `data/processed/branch2_nested_scores.parquet`,
       `outputs/branch2_nested.png`.
Запуск из корня: ``python -m experiments.branch2_nested_tuning``
(дымовой прогон: ``BRANCH2_SMOKE=1``, только ось X и вход 58 сырых).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               poly_design)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.branch1_multiscale_ensemble import (mean_axes, multiscale,  # noqa: E402
                                                     smooth_field)
from experiments.nn_geo_mpm import (SEED, grid_edges, norm_adj, normalize,  # noqa: E402
                                    robust_stats, strip_ids, train_gcn)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SMOKE = bool(os.environ.get("BRANCH2_SMOKE"))
#: Радиусы постобработочного сглаживания-кандидаты, км (0 = без сглаживания).
SIGMA_KM = (0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0) if not SMOKE else (0.0, 2.0, 10.0)
#: Радиусы, с которыми сравнивается вложенный выбор.
FIXED_KM = (0.0, 2.0, 10.0)
#: Наборы гиперпараметров бустинга; «по умолчанию» — значения из `config`.
HGB_GRID = {
    "по умолчанию": {},
    "мельче шаг, глубже": {"learning_rate": 0.03, "max_leaf_nodes": 31},
    "жёстче регуляризация": {"max_leaf_nodes": 7, "min_samples_leaf": 50,
                             "l2_regularization": 5.0},
    "крупный шаг, мелкие деревья": {"learning_rate": 0.10, "max_leaf_nodes": 7},
    "глубоко и зарегуляризовано": {"learning_rate": 0.03, "max_leaf_nodes": 31,
                                   "min_samples_leaf": 50, "l2_regularization": 5.0},
}
#: На каком входе подбираются гиперпараметры (290 столбцов — 224 с на обучение,
#: полная сетка по ним стоила бы более 7 часов, поэтому только сырые 58).
TUNE_INPUT = "58 сырые"

#: Конфигурации внешней оценки: имя -> (вход, модели, ключ в кэше ветки 1).
CONFIGS = {
    "бустинг | 58 сырые": ("58 сырые", ["бустинг"], "бустинг | 58 сырые (планка)"),
    "бустинг | 58 много масштабов": ("58 много масштабов", ["бустинг"],
                                     "бустинг | 58 много масштабов"),
    "бустинг + графовая сеть | 58 много масштабов":
        ("58 много масштабов", ["бустинг", "графовая сеть"],
         "бустинг + графовая сеть | 58 много масштабов"),
}

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fit_boost(X, y, tr, allx, params=None, **_):
    m = HistBackgroundEnsemble(random_state=SEED, hgb_params=params).fit(X[tr], y[tr])
    return m.predict_proba(X[allx])[:, 1]


def fit_gcn(X, y, tr, allx, adj=None, **_):
    med, iqr = robust_stats(X, tr)
    return train_gcn(normalize(X, med, iqr), y, tr, adj)[allx]


FIT = {"бустинг": fit_boost, "графовая сеть": fit_gcn}


def detrend(v: np.ndarray, cells: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Остаток после снятия квадратичной поверхности по координатам на `cells`."""
    beta, *_ = np.linalg.lstsq(D[cells], v[cells], rcond=None)
    return v - D @ beta


def score_cells(pred: np.ndarray, cells: np.ndarray, y: np.ndarray,
                crit: np.ndarray, D: np.ndarray) -> dict:
    """Совпадение верхних 10% на наборе ячеек — как есть и после снятия тренда.

    Мишень — верхние 10% критериального прогноза, поэтому «совпадение» здесь это
    доля попавших в прогноз положительных ячеек участка. Вариант без тренда
    сравнивает верхние 10% остатков прогноза с верхними 10% остатков непрерывной
    критериальной оценки на том же участке.
    """
    p, area = pred[cells], config.CREF_AREA
    ok = np.isfinite(p)
    if ok.sum() < 50 or y[cells].sum() == 0:
        return {"совпадение": np.nan, "совпадение_без_тренда": np.nan, "roc_auc": np.nan}
    top = p >= np.nanquantile(p, 1 - area)
    pr = detrend(np.where(np.isfinite(pred), pred, np.nanmedian(p)), cells, D)[cells]
    cr = detrend(crit, cells, D)[cells]
    tp, tc = pr >= np.quantile(pr, 1 - area), cr >= np.quantile(cr, 1 - area)
    return {"совпадение": float(y[cells][top].sum() / max(y[cells].sum(), 1)),
            "совпадение_без_тренда": float((tp & tc).sum() / max(tc.sum(), 1)),
            "roc_auc": float(roc_auc_score(y[cells], np.where(ok, p, np.nanmin(p))))}


def combine(raw: dict, models: list[str], n: int) -> np.ndarray:
    """Голосование по рангам, как в ветке 1 (ранги считаются до калибровки)."""
    if len(models) == 1:
        return raw[models[0]]
    return np.mean([sps.rankdata(raw[m]) for m in models], axis=0) / n


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    in_pool = valid.copy()
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)
    pr_flat = prognoz.ravel()
    top_c = pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)

    cols58 = [c for c in honest_features() if c in feat.columns]
    X58 = feat[cols58].to_numpy(dtype=float)
    INPUTS = {"58 сырые": X58}
    if not SMOKE:
        INPUTS["58 много масштабов"] = multiscale(X58, shape, meta.dx)
    log(f"признаков {len(cols58)}, пул {pool.size}, позитивов {int(y.sum())}; "
        f"входы {', '.join(f'{k} ({v.shape[1]} стлб)' for k, v in INPUTS.items())}")

    adj = norm_adj(*grid_edges(shape, valid), y.size)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - 1))

    configs = {k: v for k, v in CONFIGS.items() if v[0] in INPUTS}
    outer_ks = range(2) if SMOKE else range(N_STRIPS)
    axes = ("X",) if SMOKE else ("X", "Y")
    grid = {"по умолчанию": {}} if SMOKE else HGB_GRID

    # ------------------------------------------------------------------
    # Внутренние фолды: обучающая часть внешнего фолда делится тем же способом
    # ------------------------------------------------------------------
    cache_path = config.PROCESSED_DIR / ("branch2_inner_smoke.npz" if SMOKE
                                         else "branch2_inner_cache.npz")
    inner: dict[str, np.ndarray] = {}
    if cache_path.exists():
        inner = {k: v for k, v in np.load(cache_path, allow_pickle=True).items()}
        log(f"внутренние прогнозы взяты из кэша {cache_path.name} ({len(inner)} шт.)")
    strips, edges_by_ax = {}, {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        strips[ax] = np.where(in_pool, sid, -1)
        edges_by_ax[ax] = (sid, edges, coord)

    need_pairs = sorted({(inp, m) for inp, ms, _ in configs.values() for m in ms})
    todo = []
    for ax in axes:
        sid, edges, coord = edges_by_ax[ax]
        sid_in, edges_in, coord_in = edges_by_ax["Y" if ax == "X" else "X"]
        for k in outer_ks:
            outer_tr = ((coord < edges[k] - STRIP_BUFFER_M) |
                        (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool
            for j in range(N_STRIPS):
                te = np.flatnonzero(outer_tr & (sid_in == j))
                tr = np.flatnonzero(outer_tr &
                                    ((coord_in < edges_in[j] - STRIP_BUFFER_M) |
                                     (coord_in > edges_in[j + 1] + STRIP_BUFFER_M)))
                if te.size < 200 or tr.size < 500:
                    continue
                for inp, model in need_pairs:
                    combos = (grid if (model == "бустинг" and inp == TUNE_INPUT)
                              else {"по умолчанию": {}})
                    for cname, params in combos.items():
                        todo.append((ax, k, j, inp, model, cname, params, tr, te))
    log(f"внутренних обучений в плане {len(todo)}, из них уже в кэше "
        f"{sum(1 for t in todo if f'{t[0]}{t[1]}|{t[2]}|{t[3]}|{t[4]}|{t[5]}' in inner)}")

    done = 0
    for ax, k, j, inp, model, cname, params, tr, te in todo:
        key = f"{ax}{k}|{j}|{inp}|{model}|{cname}"
        if key in inner:
            continue
        t0 = time.time()
        allx = np.concatenate([tr, te])
        p = FIT[model](INPUTS[inp], y, tr, allx, adj=adj, params=params)
        buf = np.full(y.size, np.nan, dtype=np.float32)
        buf[te] = p[len(tr):]
        inner[key] = buf
        au = (roc_auc_score(y[te], p[len(tr):]) if y[te].sum() and (y[te] == 0).any()
              else float("nan"))
        log(f"  внутр. [{ax}{k}<-{j} | {inp} | {model} | {cname}] обучение {tr.size}, "
            f"оценка {te.size}, ROC={au:.4f} ({time.time() - t0:.0f}с)")
        done += 1
        if done % 4 == 0:          # кэш пишется пачками: пересжатие всего файла
            np.savez_compressed(cache_path, **inner)   # на каждом шаге дороже обучения
    np.savez_compressed(cache_path, **inner)
    log(f"внутренних прогнозов всего {len(inner)}")

    # ------------------------------------------------------------------
    # Выбор радиуса (и параметров) по внутренним фолдам
    # ------------------------------------------------------------------
    sel_rows, chosen = [], {}
    for cfg, (inp, models, _) in configs.items():
        combos = list(grid) if (models == ["бустинг"] and inp == TUNE_INPUT) \
            else ["по умолчанию"]
        for ax in axes:
            for k in outer_ks:
                for cname in combos:
                    for km in SIGMA_KM:
                        vals = {"совпадение": [], "совпадение_без_тренда": []}
                        for j in range(N_STRIPS):
                            keys = [f"{ax}{k}|{j}|{inp}|{m}|"
                                    f"{cname if m == 'бустинг' else 'по умолчанию'}"
                                    for m in models]
                            if not all(q in inner for q in keys):
                                continue
                            te = np.flatnonzero(np.isfinite(inner[keys[0]]))
                            raw = {m: inner[q][te].astype(float)
                                   for m, q in zip(models, keys)}
                            p = np.full(y.size, np.nan)
                            p[te] = combine(raw, models, te.size)
                            if km > 0:
                                mask = np.zeros(y.size, dtype=bool)
                                mask[te] = True
                                p = smooth_field(p, shape, km * 1000.0 / meta.dx,
                                                 mask=mask)
                            sc = score_cells(p, te, y, crit, D)
                            for m_ in vals:
                                vals[m_].append(sc[m_])
                        if not vals["совпадение"]:
                            continue
                        sel_rows.append({"конфигурация": cfg, "полоса": f"{ax}{k}",
                                         "параметры": cname, "сигма_км": km,
                                         "внутр_совпадение": np.nanmean(vals["совпадение"]),
                                         "внутр_без_тренда":
                                             np.nanmean(vals["совпадение_без_тренда"])})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(ROOT / "outputs" / "metrics" / "branch2_nested_selection.csv",
               index=False, encoding="utf-8-sig")
    for (cfg, fold), g in sel.groupby(["конфигурация", "полоса"]):
        best = g.loc[g["внутр_без_тренда"].idxmax()]
        alt = g.loc[g["внутр_совпадение"].idxmax()]
        chosen[(cfg, fold)] = {"сигма_км": float(best["сигма_км"]),
                               "параметры": best["параметры"],
                               "внутр_без_тренда": float(best["внутр_без_тренда"]),
                               "сигма_по_старому": float(alt["сигма_км"]),
                               "параметры_по_старому": alt["параметры"]}
    log("радиусы выбраны внутри обучения: " + ", ".join(
        f"{f}:{v['сигма_км']:.0f}км" for (c, f), v in chosen.items()
        if c == list(configs)[0]))

    # ------------------------------------------------------------------
    # Внешняя оценка: выбранный радиус против фиксированных
    # ------------------------------------------------------------------
    z = np.load(config.PROCESSED_DIR / "branch1_multiscale_cache.npz", allow_pickle=True)
    maps1 = z["maps"].item()
    outer_pred: dict[str, dict[str, np.ndarray]] = {}
    fold_rows, map_rows, assembled = [], [], {}

    for cfg, (inp, models, key1) in configs.items():
        need_refit = any(chosen[(cfg, f"{ax}{k}")]["параметры"] != "по умолчанию"
                         for ax in axes for k in outer_ks
                         if (cfg, f"{ax}{k}") in chosen)
        per_ax = {}
        for ax in axes:
            sid, edges, coord = edges_by_ax[ax]
            base = np.array(maps1[key1][ax], dtype=float)
            if need_refit:
                acc = np.full(y.size, np.nan)
                for k in outer_ks:
                    cn = chosen[(cfg, f"{ax}{k}")]["параметры"]
                    te = np.flatnonzero((sid == k) & in_pool)
                    if cn == "по умолчанию":
                        acc[te] = base[te]
                        continue
                    tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                         (coord > edges[k + 1] + STRIP_BUFFER_M))
                                        & in_pool)
                    t0 = time.time()
                    allx = np.concatenate([tr, te])
                    raw = {m: FIT[m](INPUTS[inp], y, tr, allx, adj=adj,
                                     params=grid[cn] if m == "бустинг" else None)
                           for m in models}
                    p = combine(raw, models, allx.size)
                    p_tr, p_te = np.sort(p[:len(tr)]), p[len(tr):]
                    acc[te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
                    log(f"  внешн. переобучение [{cfg} | {ax}{k} | {cn}] "
                        f"({time.time() - t0:.0f}с)")
                base = acc
            per_ax[ax] = base
        outer_pred[cfg] = per_ax

        variants = {f"фикс. {km:.0f} км": km for km in FIXED_KM}
        variants["выбрано внутри обучения"] = None
        for vname, km_fixed in variants.items():
            acc_ax = {}
            for ax in axes:
                acc = np.full(y.size, np.nan)
                for k in outer_ks:
                    mask = (strips[ax] == k)
                    te = np.flatnonzero(mask)
                    km = km_fixed if km_fixed is not None \
                        else chosen[(cfg, f"{ax}{k}")]["сигма_км"]
                    p = per_ax[ax]
                    if km > 0:
                        p = smooth_field(p, shape, km * 1000.0 / meta.dx, mask=mask)
                    acc[te] = p[te]
                    row = {"конфигурация": cfg, "вариант": vname, "полоса": f"{ax}{k}",
                           "сигма_км": km, "параметры":
                               chosen[(cfg, f"{ax}{k}")]["параметры"]}
                    row.update(score_cells(acc, te, y, crit, D))
                    fold_rows.append(row)
                acc_ax[ax] = acc
            s = mean_axes(*(acc_ax[a] for a in ("X", "Y"))) if len(axes) == 2 \
                else acc_ax["X"]
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            assembled[f"{cfg} | {vname}"] = s
            top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
            sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
            tpr = sr >= np.quantile(sr[pool], 1 - config.CREF_AREA)
            tcr = cr >= np.quantile(cr[pool], 1 - config.CREF_AREA)
            agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
            map_rows.append({
                "конфигурация": cfg, "вариант": vname,
                "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                "совпадение_без_тренда": ((tpr & tcr & valid).sum()
                                          / max((tcr & valid).sum(), 1)),
                "согласие_каппа": agr["kappa"],
                "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                "новых_ячеек": int((top_m & ~top_c & valid).sum())})

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ROOT / "outputs" / "metrics" / "branch2_nested_folds.csv",
                 index=False, encoding="utf-8-sig")
    mp = pd.DataFrame(map_rows)

    # точки заверки
    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    for i, r in mp.iterrows():
        s = assembled[f"{r['конфигурация']} | {r['вариант']}"]
        top = (s >= np.quantile(s[pool], 1 - config.CREF_AREA)) & valid
        mp.loc[i, "точек_19"] = int(top[unb].sum())
        mp.loc[i, "точек_73"] = int(top[allp].sum())
    mp.to_csv(ROOT / "outputs" / "metrics" / "branch2_nested_maps.csv",
              index=False, encoding="utf-8-sig")
    print("\n=== Карты целиком: вложенный выбор радиуса против фиксированных ===")
    print(mp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}; случайный уровень совпадения 0.10")

    print("\n=== Что выбрано внутри обучения (по полосам) ===")
    ch = pd.DataFrame([{"конфигурация": c, "полоса": f, **v}
                       for (c, f), v in chosen.items()])
    print(ch.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    ch.to_csv(ROOT / "outputs" / "metrics" / "branch2_nested_choice.csv",
              index=False, encoding="utf-8-sig")

    # парные тесты по полосам: выбранный радиус против каждого фиксированного
    pr_rows = []
    for cfg in configs:
        g = folds[folds["конфигурация"] == cfg]
        a = g[g["вариант"] == "выбрано внутри обучения"].set_index("полоса")
        for vname in (f"фикс. {km:.0f} км" for km in FIXED_KM):
            b = g[g["вариант"] == vname].set_index("полоса")
            for metric in ("совпадение", "совпадение_без_тренда", "roc_auc"):
                d = (a[metric] - b[metric]).dropna()
                if d.size < 3 or np.allclose(d, 0):
                    continue
                pr_rows.append({"конфигурация": cfg, "против": vname, "метрика": metric,
                                "разница": float(d.mean()),
                                "лучше_в": int((d > 0).sum()), "полос": int(d.size),
                                "p_уилкоксон": float(
                                    sps.wilcoxon(d, alternative="greater").pvalue)})
    paired = pd.DataFrame(pr_rows)
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch2_nested_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные тесты по полосам (вложенный выбор против фиксированного) ===")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if len(paired) else "нет сравнений")

    out = pd.DataFrame({k.replace(" | ", "__"): v for k, v in assembled.items()})
    out.to_parquet(config.PROCESSED_DIR / "branch2_nested_scores.parquet")
    log("OK: data/processed/branch2_nested_scores.parquet")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size  # noqa: E731
    show = [("Критериальный прогноз", crit, "")]
    for cfg in configs:
        for vname in ("фикс. 10 км", "выбрано внутри обучения"):
            k = f"{cfg} | {vname}"
            if k not in assembled:
                continue
            r = mp[(mp["конфигурация"] == cfg) & (mp["вариант"] == vname)].iloc[0]
            show.append((f"{cfg}\n{vname}", assembled[k],
                         f"совпадение {r['совпадение_верх_10']:.0%}, "
                         f"без тренда {r['совпадение_без_тренда']:.0%}"))
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 4.6))
    for ax, (title, v, sub) in zip(np.atleast_1d(axs), show):
        ax.imshow(rank(v), cmap="magma")
        ax.set_title(f"{title}\n{sub}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ветка 2: радиус сглаживания, выбранный внутри обучающих полос",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch2_nested.png", dpi=130)
    log("OK: outputs/branch2_nested.png")


if __name__ == "__main__":
    main()
