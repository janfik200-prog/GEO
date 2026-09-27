"""Можно ли заменить минерагенические факторы формулы дистанционными данными.

ПОСТАНОВКА. Критериальный анализ ВНИГНИ считает перспективность по восьми
факторным слоям с геологических карт (`config.CRIT_EXCLUDE_FACTOR_FEATURES`):
расстояния до разломов двух систем и их плотность, расстояние до магматических
тел и их плотность, расстояние до структурного несогласия, до границ фаций и до
палеодолин. Эти слои нельзя использовать как признаки ML — тогда модель просто
восстановит формулу, — и их нет за пределами листа, потому что за его границей
карты не оцифрованы. Отсюда вопрос: сколько из каждого фактора видно
дистанционными данными и что получится, если подставить в формулу восстановленные
факторы вместо настоящих.

ЧТО СЧИТАЕТСЯ.

1. **Восстановимость каждого фактора по отдельности.** Регрессия «съёмки и
   рельеф → факторный слой», обученная на полосах с разрывом 15 км: ранговая
   связь предсказания с настоящим слоем и ROC для класса «фактор благоприятен»
   (верхние 10% благоприятности: для расстояний — ближние ячейки, для
   плотностей — высокие значения). Это прямая проверка «прокси против своего
   фактора», которой в проекте до сих пор не было ни для одного фактора, кроме
   палеодолины.
2. **Замена факторов в формуле.** Модель-заменитель формулы обучается на
   НАСТОЯЩИХ восьми факторах обучающей полосы, а применяется к
   ВОССТАНОВЛЕННЫМ факторам отложенной полосы — ровно так это работало бы на
   новой территории, где факторных слоёв нет вообще. Верхняя граница
   («потолок») — тот же заменитель на настоящих факторах отложенной полосы.
3. **Синергия.** Вход из дистанционных признаков ВМЕСТЕ с восстановленными
   факторами: добавляет ли факторный слой что-то сверх сырых признаков, из
   которых он же и восстановлен.

Сравнение идёт с прямой моделью «дистанционка → верхние 10% критериального
прогноза» (нынешняя рабочая точка, 0.362 без тренда). Метрика и сборка карты
везде одинаковы: полосы с разрывом 15 км, сглаживание ядром 5×1.25 км на
северо-запад, совпадение верхних 10% после снятия наклона у обеих карт.

ОГОВОРКА О ЦИРКУЛЯРНОСТИ. Восстановленный фактор — не независимое измерение
геологии, а предсказание карты по съёмке. Он годится для переноса формулы на
территорию без карт и для ответа «что из формулы вообще видно из космоса», но
не является проверкой самой формулы.

Выход: `outputs/metrics/factor_from_remote.csv` (восстановимость факторов),
`outputs/metrics/factor_from_remote_maps.csv` (карты и их метрики),
`data/processed/factor_from_remote.parquet` (восстановленные слои).
Запуск из корня: ``python -m experiments.factor_from_remote``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               make_model, poly_design)
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (agree_detrended,  # noqa: E402
                                                     boot_diff, smooth_strips)
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FACTORS = list(config.CRIT_EXCLUDE_FACTOR_FEATURES)
# Благоприятность: у расстояний — чем ближе, тем лучше; у плотностей наоборот.
FAVOURABLE_HIGH = ("dens_tect", "dens_magm")
FACTOR_TITLE = {
    "dist_tect1": "расстояние до разломов 1-й системы",
    "dist_tect2": "расстояние до разломов 2-й системы",
    "dens_tect": "плотность разломов",
    "dist_magm": "расстояние до магматических тел",
    "dens_magm": "плотность магматических тел",
    "dist_struct": "расстояние до структурного несогласия",
    "dist_facies": "расстояние до границ фаций",
    "dist_paleo": "расстояние до палеодолин",
}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def make_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config.HGB_MAX_ITER, learning_rate=config.HGB_LEARNING_RATE,
        max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
        min_samples_leaf=config.HGB_MIN_SAMPLES_LEAF,
        l2_regularization=config.HGB_L2, early_stopping=False,
        random_state=config.CRIT_SEED)


def favourability(col: str, v: np.ndarray) -> np.ndarray:
    """Слой в шкале «больше — благоприятнее», чтобы факторы были сравнимы."""
    return v if col in FAVOURABLE_HIGH else -v


def strip_plan(cx: np.ndarray, cy: np.ndarray, in_pool: np.ndarray):
    """Общий для всех прогонов список фолдов: (ось, обучение, тест)."""
    plan = []
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & in_pool)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & in_pool)
            if te.size and tr.size:
                plan.append((ax, tr, te))
    return plan


def rebuild_factors(X: np.ndarray, F: np.ndarray, plan) -> dict[str, np.ndarray]:
    """Восстановление каждого факторного слоя по дистанционным признакам.

    Возвращает по оси свой массив предсказаний: ячейка предсказана моделью,
    не видевшей ни её саму, ни её окрестность в 15 км.
    """
    out = {ax: np.full(F.shape, np.nan) for ax in ("X", "Y")}
    for ax, tr, te in plan:
        for j in range(F.shape[1]):
            m = make_regressor().fit(X[tr], F[tr, j])
            out[ax][te, j] = m.predict(X[te])
        log(f"  восстановление факторов: полоса {ax}, обучение {tr.size}, "
            f"тест {te.size}")
    return out


def factor_quality(F: np.ndarray, rec: dict, pool: np.ndarray) -> pd.DataFrame:
    """Ранговая связь и ROC «фактор благоприятен» для каждого слоя."""
    rows = []
    for j, col in enumerate(FACTORS):
        pred = mean_axes(rec["X"][:, j], rec["Y"][:, j])
        ok = pool[np.isfinite(pred[pool])]
        true_f = favourability(col, F[:, j])
        pred_f = favourability(col, pred)
        good = true_f >= np.quantile(true_f[pool], 0.90)
        rows.append({
            "фактор": col, "что это": FACTOR_TITLE[col],
            "ранговая_связь": float(sps.spearmanr(pred[ok], F[ok, j]).statistic),
            "roc_благоприятен": float(roc_auc_score(good[ok], pred_f[ok])),
            "доля_благоприятных": float(good[pool].mean())})
    return pd.DataFrame(rows)


def surrogate_map(F_tr_source: np.ndarray, F_te_source: np.ndarray,
                  y: np.ndarray, plan, size: int) -> dict[str, np.ndarray]:
    """Карта модели-заменителя формулы: обучение на одних слоях, счёт на других.

    `F_tr_source` — значения факторов, по которым учится заменитель (настоящие
    слои обучающей полосы), `F_te_source` — значения, к которым он применяется
    на отложенной полосе (настоящие или восстановленные).
    """
    per_ax = {ax: np.full(size, np.nan) for ax in ("X", "Y")}
    for ax, tr, te in plan:
        m = make_model().fit(F_tr_source[tr], y[tr])
        p_tr = np.sort(m.predict_proba(F_tr_source[tr])[:, 1])
        p_te = m.predict_proba(F_te_source[te])[:, 1]
        per_ax[ax][te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
    return per_ax


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    in_pool = valid.copy()
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    blocks = crit_reference.block_labels(meta)

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    work = spectral_mask.apply(feat[cols])
    X = work.to_numpy(dtype=float)
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    F = raw[FACTORS].to_numpy(dtype=float)

    plan = strip_plan(cx, cy, in_pool)
    log(f"признаков {X.shape[1]}, факторов {len(FACTORS)}, фолдов {len(plan)}")

    rec = rebuild_factors(X, F, plan)
    qual = factor_quality(F, rec, pool)
    qual.to_csv(ROOT / "outputs" / "metrics" / "factor_from_remote.csv", index=False)
    print("\n=== Сколько каждого фактора видно дистанционными данными ===")
    print(qual.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    rec_mean = np.column_stack([mean_axes(rec["X"][:, j], rec["Y"][:, j])
                                for j in range(F.shape[1])])
    out = pd.DataFrame(rec_mean, columns=[f"{c}_восст" for c in FACTORS])
    out.to_parquet(config.PROCESSED_DIR / "factor_from_remote.parquet", index=False)

    # ------------------------------------------------------------------
    # Карты: прямая модель, формула на настоящих факторах, формула на
    # восстановленных, синергия
    # ------------------------------------------------------------------
    maps, rows = {}, []

    s, aucs, per_ax = blind_map(X, y, pool, cx, cy)
    maps["дистанционка напрямую (рабочая точка)"] = (per_ax, float(np.mean(aucs)))

    rec_full = {ax: np.where(np.isfinite(rec[ax]), rec[ax], np.nan)
                for ax in ("X", "Y")}
    for name, te_src in (("формула на настоящих факторах (потолок)", None),
                         ("формула на восстановленных факторах", "rec")):
        per = {ax: np.full(y.size, np.nan) for ax in ("X", "Y")}
        for ax, tr, te in plan:
            src = F if te_src is None else rec_full[ax]
            m = make_model().fit(F[tr], y[tr])
            p_tr = np.sort(m.predict_proba(F[tr])[:, 1])
            p_te = m.predict_proba(src[te])[:, 1]
            per[ax][te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
        maps[name] = (per, float("nan"))

    # синергия: сырые признаки + восстановленные факторы (обучение тоже на
    # восстановленных, иначе вход обучения и вход счёта разной природы)
    per_syn = {ax: np.full(y.size, np.nan) for ax in ("X", "Y")}
    syn_auc = []
    for ax, tr, te in plan:
        Xs_tr = np.column_stack([X[tr], F[tr]])          # на обучении факторы есть
        Xs_te = np.column_stack([X[te], rec_full[ax][te]])  # на новой территории — нет
        m = make_model().fit(Xs_tr, y[tr])
        p_tr = np.sort(m.predict_proba(Xs_tr)[:, 1])
        p_te = m.predict_proba(Xs_te)[:, 1]
        per_syn[ax][te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
        syn_auc.append(roc_auc_score(y[te], p_te))
    maps["синергия: признаки + восстановленные факторы"] = (per_syn,
                                                            float(np.mean(syn_auc)))

    smoothed = {}
    for name, (per, auc) in maps.items():
        sm = smooth_strips(per, meta, valid, cx, cy)
        smoothed[name] = sm
        m = score_map(sm, y, crit, pool, valid, D)
        rows.append({"карта": name, "фолды_roc_auc": auc, **m})
        print(f"  {name:42s} совпадение {m['совпадение_верх_10']:.3f}, "
              f"без тренда {m['совпадение_без_тренда']:.3f}, "
              f"ROC карты {m['карта_roc_auc']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    outp = ROOT / "outputs" / "metrics" / "factor_from_remote_maps.csv"
    res.to_csv(outp, index=False)
    print("\n=== Карты: замена факторов дистанционкой ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ref = "дистанционка напрямую (рабочая точка)"
    ta_ref, tc = agree_detrended(smoothed[ref], crit, pool, valid, D)
    print(f"\n=== Разность с «{ref}» (блочный бутстрэп 10 км) ===")
    for name in maps:
        if name == ref:
            continue
        ta, _ = agree_detrended(smoothed[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print(f"  {name:42s} {b['разность_средняя']:+.3f} "
              f"[{b['разность_ниж_5']:+.3f}; {b['разность_верх_95']:+.3f}]")
    print(f"\nOK: {outp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
