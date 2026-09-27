"""Сколько в прогнозе абсолютной высоты и что останется, если её убрать.

ПОВОД (07.09.2026). Новая рабочая точка на маскированных спектрах связана с
высотой на +0.501 по рангу, тогда как критериальный прогноз — только на +0.237,
и `dem_elev` держит первое место по важности (0.287). Высота — самый трендовый
признак из всех: она почти целиком объясняется координатами, а значит модель,
опирающаяся на неё, воспроизводит региональный наклон, а не локальную геологию.
На новой территории наклон другой, и такой прогноз переносится вхолостую или с
обратным знаком (тот же механизм, что закрыл группы `gm` и `pf`, см.
`forecast_wide.py`).

ЧТО ПРОВЕРЯЕТСЯ:

1. Сколько даёт ОДНА высота — если она сама по себе воспроизводит карту не хуже
   всей модели, то модели, по сути, нет.
2. Что теряется при её удалении, и отдельно — при удалении широковолновой
   кривизны 5 км (единственного, кроме высоты, признака с сильной связью с ней,
   −0.46).
3. Опорная планка — только координаты: любой набор обязан её обыгрывать.

Схема оценки: полосы с разрывом 15 км, сглаживание рабочей точки, метрика —
совпадение верхних 10% с критериальным прогнозом после снятия наклона у обеих
карт. Разность с текущей рабочей точкой оценивается блочным бутстрэпом 10 км.

Выход: `outputs/metrics/relief_elevation_control.csv`.
Запуск из корня: ``python -m experiments.relief_elevation_control``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import poly_design, trend_share  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (agree_detrended,  # noqa: E402
                                                     boot_diff, smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ELEV = "dem_elev"
BROAD = "relief2_curv_5km"      # широковолновая кривизна, связь с высотой -0.46


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

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    work = spectral_mask.apply(feat[cols])

    # Насколько каждый признак — это просто координаты
    tr = trend_share(work.to_numpy(dtype=float), D)
    ts = pd.Series(tr, index=work.columns).sort_values(ascending=False)
    print("Доля дисперсии, объяснимая координатами (верх списка):")
    print(ts.head(6).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"…у критериального прогноза {trend_share(crit[:, None], D)[0]:.3f}\n")

    variants = {
        "рабочая точка (все 38)": work,
        "без высоты": work.drop(columns=[ELEV]),
        "без высоты и кривизны 5 км": work.drop(columns=[ELEV, BROAD]),
        "ТОЛЬКО высота": work[[ELEV]],
        "ТОЛЬКО координаты": pd.DataFrame({"x": cx, "y": cy}),
    }

    rows, maps = [], {}
    for name, df in variants.items():
        s, aucs, per_ax = blind_map(df.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        m = score_map(sm, y, crit, pool, valid, D)
        maps[name] = sm
        rows.append({"вариант": name, "признаков": df.shape[1],
                     "фолды_roc_auc": float(np.mean(aucs)), **m})
        print(f"  {name:28s} фолды={rows[-1]['фолды_roc_auc']:.4f} "
              f"совпадение={m['совпадение_верх_10']:.3f} "
              f"без тренда={m['совпадение_без_тренда']:.3f} "
              f"ROC карты={m['карта_roc_auc']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "relief_elevation_control.csv"
    res.to_csv(out, index=False)
    print("\n=== Что даёт высота и что остаётся без неё ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ref = "рабочая точка (все 38)"
    ta_ref, tc = agree_detrended(maps[ref], crit, pool, valid, D)
    print(f"\n=== Разность с «{ref}» (блочный бутстрэп 10 км) ===")
    for name in variants:
        if name == ref:
            continue
        ta, _ = agree_detrended(maps[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print(f"  {name:28s} {b['разность_средняя']:+.3f} "
              f"[{b['разность_ниж_5']:+.3f}; {b['разность_верх_95']:+.3f}]")
    print(f"\nOK: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
