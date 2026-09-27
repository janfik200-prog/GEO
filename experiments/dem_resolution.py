"""Разрешение рельефа: даёт ли Copernicus DEM в исходных 30 м больше, чем нынешние 100 м.

ПОВОД. Признаки рельефа проекта считаются не по исходному растру, а по копии,
перепроецированной в метрику листа с шагом 100 м (`config.TER_RES_M`). Исходное
разрешение Copernicus DEM GLO-30 на широте 71 градус — около 30 м по обеим осям
(1 угловая секунда по широте, 3 по долготе). То есть при сборке признаков
выбрасывается примерно десятикратный запас подробности, и делается это не по
результату проверки, а по умолчанию, унаследованному от общего рабочего шага
всех источников. Рельеф — единственный слой проекта, где подробность реально
есть: съёмка Sentinel-2 приходит на 10-20 м, но её признаки всё равно
осредняются по ячейке, а рельеф даёт производные (уклон, врез, положение в
окрестности), которые от шага растра зависят напрямую.

ЧТО ПРОВЕРЯЕТСЯ. Шаг рабочего растра берётся 100, 50 и 25 м; 25 м вместо 30
потому, что шаг должен нацело делить ячейку 500 м (500/25 = 20), а 25 м мельче
исходных 30 м и потому ничего не теряет. Производные считаются заново на каждом
шаге, окна оконных операций задаются в метрах и остаются теми же (500 м, 1 км,
2 км), так что меняется только подробность, а не масштаб признака. Дальше всё
одинаково: те же 37 признаков переноса, та же маска обнажённости, полосы с
разрывом 15 км, сглаживание 5 x 1.25 км, совпадение верхних 10% после снятия
собственного наклона.

Отдельный, более интересный вариант — «25 м + мелкие масштабы»: на шаге 25 м
становятся вычислимы признаки МЕЛЬЧЕ ячейки, которых на 100 м нет физически
(положение в окрестности 100 и 250 м, размах высот в окне 250 м, разброс высот
и уклонов внутри ячейки). Именно они, а не пересчёт старых признаков, могут
дать новое: 500-метровая ячейка перестаёт быть точкой и получает внутреннюю
текстуру.

РЕЗУЛЬТАТ (08.09.2026) — подробность рельефа ничего не решает.

| шаг растра | признаков | фолды ROC | без наклона | с наклоном |
|---|---|---|---|---|
| 100 м (как сейчас) | 38 | **0.739** | 0.362 | 0.199 |
| 50 м | 38 | 0.737 | **0.366** | **0.212** |
| 25 м | 38 | 0.736 | 0.364 | 0.207 |
| 25 м + мелкие масштабы | 43 | 0.733 | 0.338 | 0.207 |

Пересчёт тех же шести признаков рельефа на вчетверо более подробном растре даёт
+0.004 [-0.011; +0.018] на шаге 50 м и +0.002 [-0.009; +0.013] на 25 м —
интервалы накрывают ноль, выигрыша нет. Ранжирование по фолдам при этом даже
чуть падает (0.739 -> 0.736), то есть подробность добавляет шума, а не сигнала.

Добавление пяти признаков мельче ячейки — единственная часть проверки, где
подробность могла дать принципиально новое, — УХУДШАЕТ результат значимо:
-0.024 [-0.043; -0.004], интервал ноль не накрывает. Внутриячейковая текстура
(разброс высот и уклонов, положение в окрестности 100 и 250 м, размах высот в
окне 250 м) для критериального прогноза бесполезна и только оттягивает на себя
ёмкость модели. Это согласуется с устройством самого прогноза: он собран из
факторов масштаба километров (расстояния до разломов, магматических тел,
структурного несогласия), и подробностей ниже 500 м в нём просто нет.

Вывод: шаг 100 м оставлен как рабочий. Ограничение прогноза — не разрешение
исходных данных, а отсутствующие факторные слои
([[project-forecast-ceiling]]).

Выход: `outputs/metrics/dem_resolution.csv`.
Запуск из корня: ``python -m experiments.dem_resolution``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 spectral_mask, terrain_v2)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (agree_detrended,  # noqa: E402
                                                     boot_diff, smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESOLUTIONS = (100.0, 50.0, 25.0)
T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def _win(metres: float, res_m: float) -> int:
    return max(3, int(round(metres / res_m)) | 1)


def terrain_at(meta, res_m: float, fine: bool = False) -> pd.DataFrame:
    """Признаки рельефа на заданном шаге растра. `fine` — плюс масштабы мельче ячейки."""
    k = int(round(meta.dx / res_m))
    pad = terrain_v2.pad_px(res_m)
    dem = terrain_v2.projected_dem(meta, res_m)
    deriv = terrain_v2.terrain_derivatives(dem, res_m)
    out = {}
    for name in ("elev", "slope", "tpi_500", "tpi_2km", "relief_1km", "incision"):
        mean, std = terrain_v2._block_stats(deriv[name], meta, k, pad)
        out["dem_%s" % name] = mean
        if fine and name in ("elev", "slope"):
            out["dem_%s_разброс" % name] = std
    if fine:
        e = deriv["elev"]
        extra = {"dem_tpi_100": e - uniform_filter(e, size=_win(100, res_m)),
                 "dem_tpi_250": e - uniform_filter(e, size=_win(250, res_m)),
                 "dem_размах_250": (maximum_filter(e, size=_win(250, res_m))
                                    - minimum_filter(e, size=_win(250, res_m)))}
        for name, arr in extra.items():
            out[name], _ = terrain_v2._block_stats(arr, meta, k, pad)
    return pd.DataFrame(out)


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
    dem_cols = [c for c in cols if c.startswith("dem_")]
    log("признаков %d, из них рельефа %d: %s"
        % (len(cols), len(dem_cols), ", ".join(dem_cols)))

    variants = []
    for res in RESOLUTIONS:
        t = terrain_at(meta, res)
        log("рельеф на %.0f м собран, признаков %d" % (res, t.shape[1]))
        df = work.copy()
        for c in dem_cols:
            df[c] = t[c].to_numpy()
        variants.append(("рельеф %.0f м" % res, df))
    t_fine = terrain_at(meta, 25.0, fine=True)
    df = work.copy()
    for c in dem_cols:
        df[c] = t_fine[c].to_numpy()
    new = [c for c in t_fine.columns if c not in dem_cols]
    for c in new:
        df[c] = t_fine[c].to_numpy()
    log("мелкие масштабы: добавлено %d признаков: %s" % (len(new), ", ".join(new)))
    variants.append(("рельеф 25 м + мелкие масштабы (%d признаков)" % len(new), df))

    rows, maps = [], {}
    for name, df in variants:
        _, aucs, per_ax = blind_map(df.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        maps[name] = sm
        m = score_map(sm, y, crit, pool, valid, D)
        rows.append({"вариант": name, "признаков": df.shape[1],
                     "фолды_roc_auc": float(np.mean(aucs)), **m})
        log("  %-44s без наклона=%.3f  с наклоном=%.3f  фолды=%.3f"
            % (name, m["совпадение_без_тренда"], m["совпадение_верх_10"],
               rows[-1]["фолды_roc_auc"]))

    res_df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "dem_resolution.csv"
    res_df.to_csv(out, index=False)
    print()
    print("=== Разрешение растра рельефа ===")
    print(res_df.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print()
    print("Потолок совпадения без наклона — 0.622; 70%% от него это 0.435.")

    ref = "рельеф 100 м"
    ta_ref, tc = agree_detrended(maps[ref], crit, pool, valid, D)
    print()
    print("=== Разность с нынешним шагом 100 м (блочный бутстрэп 10 км) ===")
    for name in maps:
        if name == ref:
            continue
        ta, _ = agree_detrended(maps[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print("  %-44s %+.3f [%+.3f; %+.3f]"
              % (name, b["разность_средняя"], b["разность_ниж_5"],
                 b["разность_верх_95"]))
    print()
    print("OK: %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
