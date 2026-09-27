"""Устойчива ли прибавка от маски «спектр измерен»: порог, сглаживание, интервал.

ПОВОД. `spectral_exposure.py` дал одиночное число: если спектральные признаки
Sentinel-2 оставить только в ячейках с открытым грунтом (доля от 20%), а в
остальных поставить пропуск, совпадение с критериальным прогнозом без тренда
растёт с 0.207 до 0.265 при почти неизменном ROC по фолдам. Одиночное число
ничего не значит, пока не проверено три вещи:

1. **Порог обнажённости** — не подобран ли он удачно. Прибавка обязана быть
   пологой по порогу, а не пиком в одной точке.
2. **Сглаживание рабочей точки** — не съедается ли прибавка ядром 5×1.25 км,
   вытянутым на северо-запад. Сравнение обязано идти на той сборке, которая
   считается рабочей, иначе улучшение нельзя предъявлять.
3. **Интервал** — блочный бутстрэп по блокам 10 км на РАЗНОСТИ совпадения двух
   карт. Ресемплируются блоки, а не ячейки: соседние ячейки автокоррелированы.

Карты фиксируются один раз (обучение по полосам с разрывом 15 км), бутстрэп
пересчитывает только метрику на переразмеченных блоках — иначе интервал мерил
бы ещё и случайность обучения.

Выход: `outputs/metrics/spectral_exposure_threshold.csv`.
Запуск из корня: ``python -m experiments.spectral_exposure_threshold``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 neighbourhood as nb)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import N_STRIPS, poly_design  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.nn_geo_mpm import strip_ids  # noqa: E402
from experiments.spectral_exposure import blind_map, masked, score_map  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.35]
ALONG_KM, ACROSS_KM, NW_AZIMUTH = 5.0, 1.25, 315.0     # ядро рабочей точки
N_BOOT = 400


def smooth_strips(per_ax: dict, meta, valid: np.ndarray, cx: np.ndarray,
                  cy: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Сглаживание рабочей точки: своё ядро внутри каждой полосы, потом оси.

    `scale` — множитель размеров ядра (1.0 — рабочая точка ветки 3; цена других
    значений измерена в `experiments/map_sharpness.py`).
    """
    shape = (meta.prf, meta.pic)
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    km = lambda v: v * scale * 1000.0 / meta.dx                        # noqa: E731
    acc_ax = {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, _ = strip_ids(coord)
        sid = np.where(valid, sid, -1)
        v = np.where(np.isfinite(per_ax[ax]), per_ax[ax],
                     np.nanmin(per_ax[ax][valid]))
        acc = np.full(valid.size, np.nan)
        for k in range(N_STRIPS):
            m = sid == k
            if not m.any():
                continue
            acc[m] = nb.aniso_smooth(v, shape, km(ALONG_KM), km(ACROSS_KM),
                                     theta, mask=m)[m]
        acc_ax[ax] = acc
    return mean_axes(acc_ax["X"], acc_ax["Y"])


def agree_detrended(s: np.ndarray, crit: np.ndarray, pool: np.ndarray,
                    valid: np.ndarray, D: np.ndarray):
    """Маски верхних 10% обеих карт после снятия собственного наклона каждой."""
    area = config.CREF_AREA
    s = np.where(np.isfinite(s), s, np.nanmin(s[pool]))
    sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
    tpr = (sr >= np.quantile(sr[pool], 1 - area)) & valid
    tcr = (cr >= np.quantile(cr[pool], 1 - area)) & valid
    return tpr, tcr


def boot_diff(tp_a, tp_b, tc, pool, blocks, seed: int = config.CRIT_SEED):
    """Блочный бутстрэп разности совпадения двух карт с эталоном.

    Блоки 10 км ресемплируются с возвращением; совпадение считается как доля
    эталонных ячеек выборки, накрытых верхними 10% карты.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks[pool])
    idx_by_block = {b: pool[blocks[pool] == b] for b in uniq}
    da, db = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        cells = np.concatenate([idx_by_block[b] for b in pick])
        c = tc[cells]
        if c.sum() < 20:
            continue
        da.append(tp_a[cells][c].mean())
        db.append(tp_b[cells][c].mean())
    d = np.array(da) - np.array(db)
    return {"разность_средняя": float(d.mean()),
            "разность_ниж_5": float(np.percentile(d, 5)),
            "разность_верх_95": float(np.percentile(d, 95)),
            "доля_положительных": float((d > 0).mean())}


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    blocks = crit_reference.block_labels(meta)

    base_cols = [c for c in honest_features("перенос") if c in feat.columns]
    base = feat[base_cols]
    s2_spec = [c for c in base_cols if c.startswith("s2_") and c != "s2_bare_frac"]
    bare = pd.read_parquet(config.PROCESSED_DIR /
                           "s2_features.parquet")["s2_bare_frac"].to_numpy(float)

    rows, keep_maps = [], {}
    for thr in [None] + THRESHOLDS:
        if thr is None:
            name, X = "база 37 (без маски)", base
        else:
            keep = bare >= thr
            name = f"маска на спектрах S2, порог {thr:.2f}"
            X = pd.concat([base.drop(columns=s2_spec), masked(base[s2_spec], keep),
                           pd.DataFrame({"спектр_измерен": keep.astype(float)})],
                          axis=1)
        s, aucs, per_ax = blind_map(X.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        r = {"вариант": name,
             "доля_обнажённых": float(1.0 if thr is None else (bare >= thr).mean()),
             "фолды_roc_auc": float(np.mean(aucs))}
        for tag, v in (("", s), ("_сглаж", sm)):
            m = score_map(v, y, crit, pool, valid, D)
            r[f"совпадение{tag}"] = m["совпадение_верх_10"]
            r[f"без_тренда{tag}"] = m["совпадение_без_тренда"]
            r[f"карта_roc{tag}"] = m["карта_roc_auc"]
        keep_maps[name] = (s, sm)
        rows.append(r)
        print(f"{name:36s} фолды={r['фолды_roc_auc']:.4f} "
              f"без тренда {r['без_тренда']:.3f} / со сглаживанием "
              f"{r['без_тренда_сглаж']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "spectral_exposure_threshold.csv"
    res.to_csv(out, index=False)
    print("\n=== Порог обнажённости против метрик ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Интервал на разности: лучший порог против базы ---
    best = res.loc[res["вариант"] != "база 37 (без маски)", "без_тренда_сглаж"].idxmax()
    best_name = res.loc[best, "вариант"]
    print(f"\n=== Блочный бутстрэп разности: «{best_name}» минус база ===")
    tcref = None
    for tag, i in (("без сглаживания", 0), ("со сглаживанием", 1)):
        ta, tcref = agree_detrended(keep_maps[best_name][i], crit, pool, valid, D)
        tb, _ = agree_detrended(keep_maps["база 37 (без маски)"][i], crit, pool,
                                valid, D)
        b = boot_diff(ta, tb, tcref, pool, blocks)
        print(f"  {tag:16s} разность {b['разность_средняя']:+.3f} "
              f"[{b['разность_ниж_5']:+.3f}; {b['разность_верх_95']:+.3f}], "
              f"положительна в {b['доля_положительных']:.0%} выборок")
    print(f"\nOK: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
