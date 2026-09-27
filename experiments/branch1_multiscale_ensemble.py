"""Ветка 1: многомасштабные признаки и ансамбли моделей — вместе, а не порознь.

ЗАЧЕМ. Два приёма подняли метрики по отдельности и ни разу не складывались:

* многомасштабные признаки (каждый признак плюс его сглаженные версии
  2/5/10/20 км) — `honest_raise_agreement`: одиночный бустинг даёт совпадение
  верхних 10% 34.6% против 27.9% на сырых 58, а с постобработочным сглаживанием
  прогноза 10 км — 42.5% при карте ROC-AUC 0.831;
* ансамбли и графовая сеть — `supervised_sweep` и `nn_geo_mpm`: голосование
  трёх семейств по рангам даёт лучший ROC-AUC на полосах (0.7689), графовая
  сеть по соседству на сетке — 0.799, но обе считались на сырых 37 признаках
  или на скользящих средних в двух окнах, а не на многомасштабном наборе.

Здесь оба приёма считаются на одном протоколе и одних входах: становится видно,
складываются ли выигрыши или это один и тот же выигрыш (окрестность ячейки),
измеренный дважды.

ЧТО МЕРЯЕТСЯ. Протокол с 02.09.2026 без изменений: 4 полосы по X + 4 по Y,
разрыв 15 км между обучением и тестом, калибровка квантилями обучающего
распределения внутри полосы, итоговая карта — среднее схем X и Y. Мишень —
верхние 10% критериального прогноза. Рядом две планки: бустинг на сырых 58
признаках (текущая рабочая модель) и «только география» (гауссовы базисы 8x8,
модель знает лишь «где»).

ЧЕСТНОЕ СГЛАЖИВАНИЕ ПРОГНОЗА. Постобработочное сглаживание готовой карты
считается двумя способами, и разница между ними — самостоятельный результат:

* «по всей карте» — как в `honest_raise_agreement` (оттуда цифра 42.5%). У
  этого способа есть щель: гауссово ядро 10 км шире, чем остаток разрыва до
  границы полосы, поэтому в значение ячейки подмешивается прогноз соседней
  полосы, а её модель обучалась в том числе на дальней части полосы этой
  ячейки. Утечка краевая и небольшая, но она есть;
* «внутри полосы» — сглаживание отдельно в каждой полосе каждой схемы, соседние
  полосы для ядра невидимы (пропуск). Тогда каждое значение собрано только из
  прогнозов ОДНОЙ модели, которая не видела ни одной метки ближе 15 км от своей
  полосы. Щели нет, и по этому варианту можно считать метрики по фолдам, а
  значит и парные тесты.

ЧТО СЧИТАЕТСЯ. На многомасштабном наборе 58x5=290 столбцов — бустинг,
случайный лес, перцептрон, графовая сеть по соседству на сетке, голосование по
рангам (три табличных семейства и все четыре). Отдельно тот же набор из
признаков переноса 37x5=185 — цена переносимости на смежную территорию.
Комбинации собираются из СЫРЫХ прогнозов внутри полосы до калибровки, поэтому
лишних обучений нет: каждая модель обучается один раз на полосу.

Выход: `outputs/metrics/branch1_multiscale_ensemble.csv` (карта целиком),
       `outputs/metrics/branch1_multiscale_folds.csv` (по полосам),
       `outputs/metrics/branch1_multiscale_paired.csv` (парные тесты),
       `outputs/metrics/branch1_multiscale_verdict.csv` (точки заверки),
       `data/processed/branch1_multiscale_scores.parquet`,
       `data/processed/branch1_multiscale_cache.npz`,
       `outputs/branch1_multiscale.png`.
Запуск из корня: ``python -m experiments.branch1_multiscale_ensemble``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import N_STRIPS, STRIP_BUFFER_M  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.nn_geo_mpm import (SEED, grid_edges, norm_adj, normalize,  # noqa: E402
                                    robust_stats, strip_ids, train_gcn)

#: Масштабы сглаживания признаков, км (как в `honest_raise_agreement`).
SCALES_KM = (2.0, 5.0, 10.0, 20.0)
#: Радиус постобработочного сглаживания готового прогноза, км.
POST_KM = 10.0
N_TREES = 300
#: Имя кэша сырых прогнозов по полосам (переопределяется в дымовом прогоне).
CACHE_NAME = "branch1_multiscale_cache.npz"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def smooth_field(v: np.ndarray, shape, sigma_cells: float,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """Гауссово сглаживание с игнорированием пропусков (нормировка по маске).

    ``mask`` ограничивает область, из которой берутся значения: для сглаживания
    внутри полосы за её пределами всё считается пропуском.
    """
    ok = np.isfinite(v)
    if mask is not None:
        ok = ok & mask
    w = ok.astype(float).reshape(shape)
    g = np.where(ok, v, 0.0).reshape(shape)
    num = gaussian_filter(g * w, sigma_cells, mode="nearest")
    den = gaussian_filter(w, sigma_cells, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).ravel()
    return np.where(ok, out, np.nan) if mask is not None else out


def mean_axes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Среднее схем X и Y по каждой ячейке; вне пула, где обе схемы пусты, — NaN."""
    st = np.vstack([a, b])
    ok = np.isfinite(st)
    cnt = ok.sum(axis=0)
    tot = np.where(ok, st, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def multiscale(X: np.ndarray, shape, dx: float) -> np.ndarray:
    """Признаки плюс их гауссово сглаженные версии на масштабах `SCALES_KM`."""
    parts = [X]
    for km in SCALES_KM:
        sig = km * 1000.0 / dx
        parts.append(np.column_stack([smooth_field(X[:, j], shape, sig)
                                      for j in range(X.shape[1])]))
    return np.column_stack(parts)


# --------------------------------------------------------------------------
# Базовые модели: обучить на `tr`, вернуть сырые оценки для `allx`
# --------------------------------------------------------------------------
def fit_boost(X, y, tr, allx, **_):
    m = HistBackgroundEnsemble(random_state=SEED).fit(X[tr], y[tr])
    return m.predict_proba(X[allx])[:, 1]


def fit_rf(X, y, tr, allx, **_):
    m = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=3, n_jobs=-1,
                               class_weight="balanced", random_state=SEED)
    m.fit(np.nan_to_num(X[tr]), y[tr])
    return m.predict_proba(np.nan_to_num(X[allx]))[:, 1]


def fit_mlp(X, y, tr, allx, **_):
    qt = QuantileTransformer(n_quantiles=min(1000, len(tr)),
                             output_distribution="normal", random_state=SEED)
    Xtr = np.nan_to_num(qt.fit_transform(X[tr]))
    Xall = np.nan_to_num(qt.transform(X[allx]))
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=SEED).fit(Xtr, y[tr])
    return m.predict_proba(Xall)[:, 1]


def fit_gcn(X, y, tr, allx, adj=None, **_):
    med, iqr = robust_stats(X, tr)
    return train_gcn(normalize(X, med, iqr), y, tr, adj)[allx]


BASE = {"бустинг": fit_boost, "случайный лес": fit_rf,
        "перцептрон": fit_mlp, "графовая сеть": fit_gcn}

#: Что считается на каком входе (комбинации собираются из этих же прогнозов).
PLAN = {
    "58 сырые (планка)": ["бустинг"],
    "58 много масштабов": ["бустинг", "случайный лес", "перцептрон", "графовая сеть"],
    "37 много масштабов (перенос)": ["бустинг", "случайный лес", "перцептрон"],
    "география (планка)": ["бустинг"],
}
#: Голосования по рангам — заданы заранее, не выбраны по результату.
VOTES = {
    "голосование трёх (лес+бустинг+перцептрон)": ["случайный лес", "бустинг", "перцептрон"],
    "голосование четырёх (+ графовая сеть)": ["случайный лес", "бустинг", "перцептрон",
                                              "графовая сеть"],
    "бустинг + графовая сеть": ["бустинг", "графовая сеть"],
}


def run_input(name: str, X: np.ndarray, y: np.ndarray, pool: np.ndarray,
              cx: np.ndarray, cy: np.ndarray, adj, models: list[str]):
    """Сплошной прогноз по полосам для всех моделей входа + их голосования.

    Возвращает ``{метод: {ось: (калиброванный прогноз, метки полос)}}`` и строки
    метрик по фолдам. Сырые прогнозы моделей внутри полосы переиспользуются для
    голосований, поэтому каждая модель обучается на полосу ровно один раз.
    """
    in_pool = np.zeros(y.size, dtype=bool)
    in_pool[pool] = True
    out: dict[str, dict[str, np.ndarray]] = {}
    strips: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    votes = {v: m for v, m in VOTES.items() if set(m) <= set(models)}

    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        strips[ax] = np.where(in_pool, sid, -1)
        cal = {m: np.full(y.size, np.nan) for m in models + list(votes)}
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            allx = np.concatenate([tr, te])
            raw = {}
            for m in models:
                t0 = time.time()
                raw[m] = BASE[m](X, y, tr, allx, adj=adj)
                au = roc_auc_score(y[te], raw[m][len(tr):])
                log(f"  [{name} | {m} | {ax}{k}] обучение {tr.size}, тест {te.size}, "
                    f"ROC-AUC={au:.4f} ({time.time() - t0:.0f}с)")
            for v, parts in votes.items():
                raw[v] = np.mean([sps.rankdata(raw[m]) for m in parts], axis=0) / allx.size
            for m, p in raw.items():
                p_tr, p_te = np.sort(p[:len(tr)]), p[len(tr):]
                cal[m][te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
                fold_rows.append({"ключ": f"{m} | {name}", "метод": m, "вход": name,
                                  "полоса": f"{ax}{k}", "n_обучение": int(tr.size),
                                  "n_тест": int(te.size),
                                  "roc_auc": roc_auc_score(y[te], p[len(tr):]),
                                  "pr_auc": average_precision_score(y[te], p[len(tr):])})
        for m, v in cal.items():
            out.setdefault(m, {})[ax] = v
    return out, strips, fold_rows


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    pr_flat = prognoz.ravel()
    top_c = pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)

    cols58 = [c for c in honest_features() if c in feat_all.columns]
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    cols37 = [c for c in spec["признаки_перенос"] if c in feat_all.columns]
    X58 = feat_all[cols58].to_numpy(dtype=float)
    X37 = feat_all[cols37].to_numpy(dtype=float)
    log(f"признаков {len(cols58)} (лист) и {len(cols37)} (перенос), пул {pool.size}, "
        f"позитивов {int(y.sum())}; полосы {N_STRIPS}/ось, разрыв "
        f"{STRIP_BUFFER_M / 1000:.0f} км")

    X_ms58 = multiscale(X58, shape, meta.dx)
    X_ms37 = multiscale(X37, shape, meta.dx)
    log(f"многомасштабные наборы: {X_ms58.shape[1]} и {X_ms37.shape[1]} столбцов "
        f"(масштабы {', '.join(f'{k:.0f}' for k in SCALES_KM)} км)")

    INPUTS = {"58 сырые (планка)": X58, "58 много масштабов": X_ms58,
              "37 много масштабов (перенос)": X_ms37,
              "география (планка)": rbf_basis(cx, cy, RBF_GRID)}
    adj = norm_adj(*grid_edges(shape, valid), y.size)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - 1))

    cache = config.PROCESSED_DIR / CACHE_NAME
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        maps = {k: v for k, v in z["maps"].item().items()}
        strips = z["strips"].item()
        fold_rows = list(z["folds"])
        log(f"прогнозы по полосам взяты из кэша {cache.name}")
    else:
        maps, fold_rows = {}, []
        strips = None
        for name, models in PLAN.items():
            got, st, fr = run_input(name, INPUTS[name], y, pool, cx, cy, adj, models)
            strips = strips or st
            fold_rows += fr
            for m, per_ax in got.items():
                maps[f"{m} | {name}"] = per_ax
        np.savez(cache, maps=np.array(maps, dtype=object), strips=np.array(strips, dtype=object),
                 folds=np.array(fold_rows, dtype=object))
        log(f"кэш сохранён: {cache.name}")

    folds = pd.DataFrame(list(fold_rows))
    folds.to_csv(ROOT / "outputs" / "metrics" / "branch1_multiscale_folds.csv",
                 index=False, encoding="utf-8-sig")

    sigma = POST_KM * 1000.0 / meta.dx
    final: dict[str, np.ndarray] = {}
    # Разбор ключа хранится явно: в названиях методов встречается « + »
    # («бустинг + графовая сеть»), и разбирать ключ строкой обратно нельзя.
    info: dict[str, dict] = {}
    fold_extra: list[dict] = []
    for key, per_ax in maps.items():
        m, name = key.split(" | ")
        # 1. без постобработки: среднее схем X и Y
        s = mean_axes(per_ax["X"], per_ax["Y"])
        final[key] = np.where(np.isfinite(s), s, np.nanmin(s))
        info[key] = {"метод": m, "вход": name, "постобработка": "нет", "фолды": key}
        # 2. сглаживание по всей карте (как в honest_raise_agreement)
        k_all = f"{key} + сглаж. {POST_KM:.0f} км (по всей карте)"
        final[k_all] = smooth_field(final[key], shape, sigma)
        info[k_all] = {"метод": m, "вход": name,
                       "постобработка": f"сглаж. {POST_KM:.0f} км (по всей карте)",
                       "фолды": f"{key} + сглаж. внутри полосы"}
        # 3. сглаживание внутри полосы: ядро не выходит за пределы одной модели
        per_ax_sm = {}
        for ax in ("X", "Y"):
            v = per_ax[ax]
            acc = np.full(y.size, np.nan)
            for k in range(N_STRIPS):
                mask = strips[ax] == k
                sv = smooth_field(v, shape, sigma, mask=mask)
                acc[mask] = sv[mask]
                te = np.flatnonzero(mask)
                fold_extra.append({"ключ": f"{key} + сглаж. внутри полосы",
                                   "метод": m, "вход": name, "полоса": f"{ax}{k}",
                                   "n_обучение": np.nan, "n_тест": int(te.size),
                                   "roc_auc": roc_auc_score(y[te], acc[te]),
                                   "pr_auc": average_precision_score(y[te], acc[te])})
            per_ax_sm[ax] = acc
        s2 = mean_axes(per_ax_sm["X"], per_ax_sm["Y"])
        k_in = f"{key} + сглаж. {POST_KM:.0f} км (внутри полосы)"
        final[k_in] = np.where(np.isfinite(s2), s2, np.nanmin(s2))
        info[k_in] = {"метод": m, "вход": name,
                      "постобработка": f"сглаж. {POST_KM:.0f} км (внутри полосы)",
                      "фолды": f"{key} + сглаж. внутри полосы"}
    folds_all = pd.concat([folds, pd.DataFrame(fold_extra)], ignore_index=True)
    folds_all.to_csv(ROOT / "outputs" / "metrics" / "branch1_multiscale_folds.csv",
                     index=False, encoding="utf-8-sig")

    rows = []
    for key, s in final.items():
        i = info[key]
        f_ = folds_all[folds_all["ключ"] == i["фолды"]]
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        rows.append({"карта": key, "метод": i["метод"], "вход": i["вход"],
                     "постобработка": i["постобработка"],
                     "фолды_roc_auc": float(f_["roc_auc"].mean()) if len(f_) else np.nan,
                     "фолды_roc_std": float(f_["roc_auc"].std()) if len(f_) else np.nan,
                     "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                     "карта_pr_auc": average_precision_score(y[pool], s[pool]),
                     "согласие_каппа": agr["kappa"],
                     "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "branch1_multiscale_ensemble.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Ветка 1: совпадение верхних 10% с критериальной картой ===")
    print(res.sort_values("совпадение_верх_10", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== То же по ROC-AUC на полосах (разрыв 15 км) ===")
    print(res.sort_values("фолды_roc_auc", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nПланки: случайно {config.CREF_AREA:.0%}, известные значения — бустинг "
          f"58 сырых 27.9%, много масштабов 34.6%, + сглаживание 10 км 42.5%")

    # --- парные тесты по 8 полосам: складываются ли приёмы -----------------
    wide = folds_all.pivot_table(index="полоса", columns="ключ", values="roc_auc")
    pairs = [("бустинг | 58 много масштабов", "бустинг | 58 сырые (планка)"),
             ("голосование четырёх (+ графовая сеть) | 58 много масштабов",
              "бустинг | 58 много масштабов"),
             ("голосование трёх (лес+бустинг+перцептрон) | 58 много масштабов",
              "бустинг | 58 много масштабов"),
             ("графовая сеть | 58 много масштабов", "бустинг | 58 много масштабов"),
             ("бустинг | 58 много масштабов + сглаж. внутри полосы",
              "бустинг | 58 много масштабов"),
             ("голосование четырёх (+ графовая сеть) | 58 много масштабов "
              "+ сглаж. внутри полосы", "бустинг | 58 много масштабов"),
             ("голосование трёх (лес+бустинг+перцептрон) | 37 много масштабов (перенос)",
              "голосование трёх (лес+бустинг+перцептрон) | 58 много масштабов"),
             ("бустинг | 58 много масштабов", "бустинг | география (планка)")]
    prs = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        d = wide[a].to_numpy() - wide[b].to_numpy()
        prs.append({"что": a, "против": b, "разница_roc": d.mean(),
                    "побед": f"{int((d > 0).sum())}/{len(d)}",
                    "p_вилкоксон": sps.wilcoxon(d).pvalue})
    paired = pd.DataFrame(prs)
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch1_multiscale_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные сравнения по 8 полосам ===")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- заверка точками: только лучшие по каждой метрике и планки ---------
    best_roc = res.loc[res["фолды_roc_auc"].idxmax()]
    best_agr = res.loc[res["совпадение_верх_10"].idxmax()]
    keys = {best_roc["карта"], best_agr["карта"], "бустинг | 58 сырые (планка)",
            "бустинг | география (планка)"}
    sel = {k: final[k] for k in keys if k in final}
    alpha = 0.05 / len(sel)
    log(f"заверка точками для {len(sel)} карт, порог {alpha:.4f}")
    prot = assessment.run_protocol({**sel, "criterial": crit}, list(sel), meta, valid,
                                   alpha, baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "branch1_multiscale_verdict.csv",
                           index=False, encoding="utf-8-sig")
    print("\n=== Точки заверки (19 непредвзятых, косвенная проверка) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = pd.DataFrame({"row": np.repeat(np.arange(meta.prf), meta.pic),
                        "col": np.tile(np.arange(meta.pic), meta.prf),
                        "x": cx, "y": cy, "prognoz": pr_flat, "target": y, "valid": valid,
                        **final})   # float64: у точечной метрики шаг 0.53 lift, float32 её сдвигает
    out.to_parquet(config.PROCESSED_DIR / "branch1_multiscale_scores.parquet", index=False)
    log("OK: data/processed/branch1_multiscale_scores.parquet")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size  # noqa: E731
    show = ["бустинг | 58 сырые (планка)", "бустинг | 58 много масштабов",
            best_roc["карта"], best_agr["карта"]]
    show = list(dict.fromkeys([k for k in show if k in final]))
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(4.6 * (len(show) + 1), 5.6))
    axes[0].imshow(rank(crit), cmap="magma")
    axes[0].set_title("Критериальный прогноз\n(эталон геологов)", fontsize=10)
    for ax, key in zip(axes[1:], show):
        ax.imshow(rank(final[key]), cmap="magma")
        r_ = res[res["карта"] == key].iloc[0]
        ax.set_title(f"{key}\nROC-AUC по полосам {r_['фолды_roc_auc']:.3f}, "
                     f"совпадение верх-10% {r_['совпадение_верх_10']:.0%}", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ветка 1: многомасштабные признаки и ансамбли, обучение не ближе 15 км",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch1_multiscale.png", dpi=130)
    log("OK: outputs/branch1_multiscale.png")


if __name__ == "__main__":
    main()
