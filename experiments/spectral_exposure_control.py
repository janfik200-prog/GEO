"""Контроль прибавки от маски: география обнажённости или просто пропуски?

Маска «спектр измерен» подняла совпадение без тренда с 0.279 до 0.36
(`spectral_exposure_threshold.py`). Прежде чем считать это результатом, нужно
исключить два более скучных объяснения:

1. **Спектры S2 просто мешали.** Если убрать их совсем и получить то же самое,
   то ценность маски нулевая — это отбор признаков, а не измерение породы.
2. **Дело в самих пропусках, а не в их географии.** Если случайная маска той
   же доли ячеек даёт такую же прибавку, значит работает регуляризация
   пропусками, а не то, что мы оставили спектр именно там, где видна порода.
   Случайная маска гоняется на нескольких зерных, чтобы сравнивать с разбросом.

Схема оценки та же: полосы с разрывом 15 км, сглаживание рабочей точки,
метрика — совпадение верхних 10% с критериальным прогнозом после снятия
наклона у обеих карт.

Выход: `outputs/metrics/spectral_exposure_control.csv`.
Запуск из корня: ``python -m experiments.spectral_exposure_control``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402
from experiments.spectral_exposure import blind_map, masked, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import smooth_strips  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BARE_MIN = 0.02          # порог из прогона по порогам: прибавка полога, берём край
SEEDS = (11, 23, 47)     # зёрна случайной маски


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

    base_cols = [c for c in honest_features("перенос") if c in feat.columns]
    base = feat[base_cols]
    s2_spec = [c for c in base_cols if c.startswith("s2_") and c != "s2_bare_frac"]
    bare = pd.read_parquet(config.PROCESSED_DIR /
                           "s2_features.parquet")["s2_bare_frac"].to_numpy(float)
    keep = bare >= BARE_MIN
    share = float(keep.mean())

    def with_mask(m: np.ndarray) -> pd.DataFrame:
        return pd.concat([base.drop(columns=s2_spec), masked(base[s2_spec], m),
                          pd.DataFrame({"спектр_измерен": m.astype(float)})], axis=1)

    variants = {"база 37 (без маски)": base,
                "без спектров S2 совсем": base.drop(columns=s2_spec),
                f"маска обнажённости ({share:.0%} ячеек)": with_mask(keep)}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        rnd = rng.random(bare.size) < share
        variants[f"случайная маска той же доли, зерно {sd}"] = with_mask(rnd)

    rows = []
    for name, df in variants.items():
        s, aucs, per_ax = blind_map(df.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        m0 = score_map(s, y, crit, pool, valid, D)
        m1 = score_map(sm, y, crit, pool, valid, D)
        rows.append({"вариант": name, "признаков": df.shape[1],
                     "фолды_roc_auc": float(np.mean(aucs)),
                     "без_тренда": m0["совпадение_верх_10"] * 0 + m0["совпадение_без_тренда"],
                     "без_тренда_сглаж": m1["совпадение_без_тренда"],
                     "совпадение_сглаж": m1["совпадение_верх_10"],
                     "карта_roc_сглаж": m1["карта_roc_auc"]})
        print(f"{name:44s} фолды={rows[-1]['фолды_roc_auc']:.4f} "
              f"без тренда со сглаживанием {rows[-1]['без_тренда_сглаж']:.3f}",
              flush=True)

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "spectral_exposure_control.csv"
    res.to_csv(out, index=False)
    print("\n=== Контроль: география обнажённости против простых объяснений ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    rnd = res[res["вариант"].str.startswith("случайная")]["без_тренда_сглаж"]
    real = res[res["вариант"].str.startswith("маска")]["без_тренда_сглаж"].iloc[0]
    print(f"\nслучайная маска: {rnd.mean():.3f} ± {rnd.std():.3f} "
          f"({rnd.min():.3f}…{rnd.max():.3f}); маска обнажённости: {real:.3f}")
    print(f"OK: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
