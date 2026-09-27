"""Восстанавливают ли направленные фильтры рельефа разломы критериальной формулы.

ПОВОД. `factor_from_remote.py` закрыл замену факторов формулы дистанционкой с
одним конкретным узким местом: разломные слои (`dist_tect1`, `dist_tect2`,
`dens_tect`) не восстанавливаются вообще — ROC 0.45–0.53 при случайном 0.5, при
том что в формуле пересечения разломов весят больше всего. Там же сформулирована
мерка: новая обработка имеет смысл, только если поднимает ROC восстановления
разломов выше 0.53, и обязательно с контролем «одни координаты» (для плотности
разломов он даёт 0.30).

ЧТО ДЕЛАЕТСЯ. Рельеф берётся в исходном разрешении 100 м (в пять раз мельче
рабочей сетки) и сворачивается с вытянутыми ядрами по 12 простираниям
(`src/directional_relief.py`): отклик долины/гребня и отклик уступа, две ширины
ядра. Отклики усредняются на ячейки 500 м и проверяются двумя способами.

1. **Что такое разломный слой на самом деле.** По исходным shapefile меряются
   число фигур, их площадь и простирание контуров, взвешенное по длине. Без
   этого непонятно, что вообще пытается найти фильтр.
2. **Совпадение простираний.** Для каждого азимута считается, насколько его
   отклик отличает окрестность разломов каждой системы от остальных — и на
   рабочей сетке 500 м, и на исходной сетке рельефа 100 м (чтобы отделить
   отсутствие сигнала от его потери при усреднении). Если рельеф действительно
   показывает разломы, максимум должен приходиться на простирание системы.
3. **Восстановление слоя.** Та же схема, что в `factor_from_remote.py`: полосы
   с разрывом 15 км, ранговая связь и ROC «фактор благоприятен». Входы
   сравниваются между собой: только координаты (планка), прежние признаки
   линеаментов, новые направленные, новые вместе с прежними и со всей
   дистанционкой.

РЕЗУЛЬТАТ (07.09.2026) — отрицательный, и причина не в фильтрах.

* Слои `tect1`/`tect2` — не линии разломов, а 4 и 2 полигона общей площадью
  2110 и 2023 км² при площади листа 5726 км². Внутри них лежит 37% и 35%
  валидных ячеек (расстояние ровно 0). Это широкие тектонические зоны, а не
  объекты, которые рельеф может очертить.
* Простирание контуров подтверждено измерением: 144° у первой системы
  (северо-западное) и 47° у второй (северо-восточное) — названия слоёв верны.
* Отклик вблизи зон действительно повышен (ROC 0.57–0.61 на любом азимуте), но
  ориентация ПРОТИВОПОЛОЖНА простиранию: у первой системы минимум приходится
  на 135–150° (0.548–0.560) при максимуме на 0–15° (0.603), у второй минимум на
  45° (0.541) при максимуме на 120° (0.607). То есть внутри зон рельеф линейно
  расчленён поперёк зоны, а не вдоль неё. Разброс по азимутам мал (0.005–0.020)
  и на 500 м, и на исходных 100 м — потеря при усреднении ни при чём.
* Восстановление слоёв не превышает планку координат нигде, кроме второй
  системы: 0.581 против 0.534, прибавка +0.047. Первая система 0.450 против
  0.475, плотность 0.488 против 0.303 (у координат она ниже случайного).
  Прежние признаки линеаментов хуже координат по всем трём слоям.

* Прогнозу сами признаки тоже не помогают: 38 -> 102 признака поднимают
  совпадение верхних 10% с 0.199 до 0.253 и каппу с 0.110 до 0.170, но
  совпадение БЕЗ ТРЕНДА падает с 0.362 до 0.330, разность −0.030
  [−0.117; +0.059] — ноль внутри интервала. Рост метрики с трендом при падении
  метрики без тренда — это добавка регионального наклона, а не локального
  сигнала: ровно та ловушка, ради которой метрика без тренда и введена.

Вывод: мерка «поднять ROC восстановления разломов выше 0.53» поставлена не по
адресу — она меряет способность повторить грубое деление листа на зоны, а не
найти разлом. Направленная фильтрация рельефа как способ заменить разломный
фактор формулы закрыта; признаки сохранены в parquet, но в рабочую точку не
входят.

Выход: `outputs/metrics/directional_relief_faults.csv` (восстановление),
`outputs/metrics/directional_relief_strike.csv` (совпадение простираний),
`data/processed/directional_relief_features.parquet` (признаки).
Запуск из корня: ``python -m experiments.directional_relief_faults``
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats as sps
from shapely.geometry import LineString
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 directional_relief, geology_v2, integro_grid,  # noqa: E402
                 spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.factor_from_remote import (favourability, make_regressor,  # noqa: E402
                                            strip_plan)
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (agree_detrended,  # noqa: E402
                                                     boot_diff, smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAULTS = ("dist_tect1", "dist_tect2", "dens_tect")
FAULT_TITLE = {"dist_tect1": "разломы 1-й системы (северо-западные)",
               "dist_tect2": "разломы 2-й системы (северо-восточные)",
               "dens_tect": "плотность разломов"}
NEAR_M = 1000.0          # «вблизи разлома» для теста совпадения простираний
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def build_features(meta):
    """Направленные признаки рельефа на ячейках рабочей сетки."""
    dem_meta, arrays = integro_grid.load_pgrid_dataset(config.TOPO5_PGRID)
    name = next(iter(arrays))
    dem = arrays[name].astype(float) * config.TOPO5_TO_METERS
    proj4 = integro_grid.read_grid_proj4(config.TOPO5_PGRID)
    log(f"рельеф: свойство «{name}», сетка {dem.shape}, шаг {dem_meta.dx:.0f} м")

    def to_cells(a: np.ndarray) -> np.ndarray:
        return integro_grid.resample_to_grid(a.astype(np.float32), dem_meta, meta,
                                             method="average", proj4=proj4).ravel()

    out: dict[str, np.ndarray] = {}
    for sigma in directional_relief.SCALES_CELLS:
        width_m = sigma * dem_meta.dx
        for order, kind in ((2, "долина"), (1, "уступ")):
            per_az = directional_relief.responses(dem, sigma, order)
            for az, r in per_az.items():
                out[f"dr_{kind}_{width_m:.0f}м_{az:.0f}гр"] = to_cells(np.abs(r))
            for key, v in directional_relief.summarize(per_az).items():
                out[f"dr_{kind}_{width_m:.0f}м_{key}"] = to_cells(v)
            log(f"  ядро {width_m:.0f} м, {kind}: {len(per_az)} простираний")
    return pd.DataFrame(out), dem, dem_meta


def _rings(geom):
    """Контуры фигуры как ломаные (слои разломов оцифрованы полигонами)."""
    if geom is None:
        return []
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        geom = geom.boundary
    if isinstance(geom, LineString):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]


def fault_geometry() -> pd.DataFrame:
    """Чем на самом деле являются разломные слои: число фигур, площадь, простирание."""
    rows = []
    for role, col in (("tect1", "dist_tect1"), ("tect2", "dist_tect2")):
        g = gpd.read_file(geology_v2._shp_dir() / f"{config.LAYER_FILES[role]}.shp")
        az, w = [], []
        for geom in g.geometry:
            for line in _rings(geom):
                c = np.asarray(line.coords)[:, :2]
                d = np.diff(c, axis=0)
                w.append(np.hypot(d[:, 0], d[:, 1]))
                az.append(np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 180.0)
        a, ln = np.concatenate(az), np.concatenate(w)
        ang = np.deg2rad(2.0 * a)                  # удвоенный угол: линия без начала
        mean = np.degrees(np.arctan2((ln * np.sin(ang)).sum(),
                                     (ln * np.cos(ang)).sum())) / 2.0 % 180.0
        rows.append({"слой": col, "что это": FAULT_TITLE[col], "фигур": len(g),
                     "площадь_км2": float(g.area.sum() / 1e6),
                     "простирание_гр": float(mean)})
    return pd.DataFrame(rows)


def strike_test_native(dem: np.ndarray, dem_meta, meta, raw: pd.DataFrame) -> pd.DataFrame:
    """Тот же тест простираний, но на исходной сетке рельефа 100 м."""
    rows = []
    for col in ("dist_tect1", "dist_tect2"):
        a = raw[col].to_numpy(dtype=float).reshape(meta.shape).astype(np.float32)
        d = integro_grid.resample_to_grid(a, meta, dem_meta, method="nearest")
        near = np.isfinite(d) & (d <= NEAR_M) & np.isfinite(dem)
        far = np.isfinite(d) & (d > 3 * NEAR_M) & np.isfinite(dem)
        sel = near | far
        for order, kind in ((2, "долина"), (1, "уступ")):
            per = {az: roc_auc_score(near[sel], np.abs(r)[sel])
                   for az, r in directional_relief.responses(
                       dem, directional_relief.SCALES_CELLS[0], order).items()}
            best = max(per, key=lambda k: abs(per[k] - 0.5))
            rows.append({"слой": col, "форма": kind, "лучшее_простирание": best,
                         "roc_на_нём": per[best],
                         "roc_разброс_по_азимутам": float(np.std(list(per.values()))),
                         **{f"roc_{a:.0f}гр": v for a, v in per.items()}})
    return pd.DataFrame(rows)


def strike_test(df: pd.DataFrame, raw: pd.DataFrame, pool: np.ndarray) -> pd.DataFrame:
    """На каком простирании отклик лучше всего отличает окрестность разлома."""
    rows = []
    for col in ("dist_tect1", "dist_tect2"):
        near = raw[col].to_numpy(dtype=float) <= NEAR_M
        for kind in ("долина", "уступ"):
            width = f"{directional_relief.SCALES_CELLS[0] * 100:.0f}м"
            best_az, best_auc = None, 0.5
            per = {}
            for az in directional_relief.azimuths():
                c = f"dr_{kind}_{width}_{az:.0f}гр"
                v = df[c].to_numpy()[pool]
                ok = np.isfinite(v)          # край рельефа: ~5% ячеек без данных
                auc = roc_auc_score(near[pool][ok], v[ok])
                per[az] = auc
                if abs(auc - 0.5) > abs(best_auc - 0.5):
                    best_az, best_auc = az, auc
            rows.append({"слой": col, "форма": kind, "лучшее_простирание": best_az,
                         "roc_на_нём": best_auc,
                         "roc_разброс_по_азимутам":
                             float(np.std(list(per.values()))),
                         **{f"roc_{a:.0f}гр": v for a, v in per.items()}})
    return pd.DataFrame(rows)


def rebuild(X: np.ndarray, target: np.ndarray, plan, pool: np.ndarray,
            col: str) -> dict:
    """Восстановление одного слоя: ранговая связь и ROC «фактор благоприятен»."""
    per_ax = {ax: np.full(target.size, np.nan) for ax in ("X", "Y")}
    for ax, tr, te in plan:
        per_ax[ax][te] = make_regressor().fit(X[tr], target[tr]).predict(X[te])
    p = mean_axes(per_ax["X"], per_ax["Y"])
    ok = pool[np.isfinite(p[pool])]
    good = favourability(col, target) >= np.quantile(
        favourability(col, target)[pool], 0.90)
    return {"ранговая_связь": float(sps.spearmanr(p[ok], target[ok]).statistic),
            "roc_благоприятен": float(roc_auc_score(good[ok],
                                                    favourability(col, p)[ok]))}


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")

    dr, dem, dem_meta = build_features(meta)
    dr.to_parquet(config.PROCESSED_DIR / "directional_relief_features.parquet",
                  index=False)
    log(f"направленных признаков: {dr.shape[1]}")

    fg = fault_geometry()
    print('\n=== Что такое разломный слой ===')
    print(fg.to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    for col in ("dist_tect1", "dist_tect2", "dens_tect"):
        inside = (raw[col].to_numpy(dtype=float)[pool] <= 0).mean()
        print(f"  {col}: ячеек ровно внутри зоны {inside:.0%}")

    st = strike_test(dr, raw, pool).assign(сетка="500 м (рабочая)")
    stn = strike_test_native(dem, dem_meta, meta, raw).assign(сетка="100 м (рельеф)")
    st = pd.concat([st, stn], ignore_index=True)
    st.to_csv(ROOT / "outputs" / "metrics" / "directional_relief_strike.csv",
              index=False)
    print('\n=== На каком простирании отклик отличает окрестность разлома ===')
    print(st[["сетка", "слой", "форма", "лучшее_простирание", "roc_на_нём",
              "roc_разброс_по_азимутам"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("Простирание самих зон: dist_tect1 "
          f"{fg.loc[0, 'простирание_гр']:.0f} гр, dist_tect2 "
          f"{fg.loc[1, 'простирание_гр']:.0f} гр")

    lin_cols = [c for c in raw.columns if c.startswith("lin_")]
    rs_cols = [c for c in honest_features("перенос") if c in feat.columns]
    rs = spectral_mask.apply(feat[rs_cols])
    inputs = {
        "только координаты": np.column_stack([cx, cy]),
        "прежние линеаменты": raw[lin_cols].to_numpy(dtype=float),
        "направленные фильтры": dr.to_numpy(dtype=float),
        "направленные + координаты": np.column_stack([dr.to_numpy(dtype=float),
                                                      cx, cy]),
        "направленные + вся дистанционка": np.column_stack(
            [dr.to_numpy(dtype=float), rs.to_numpy(dtype=float)]),
    }
    plan = strip_plan(cx, cy, valid)

    rows = []
    for name, X in inputs.items():
        for col in FAULTS:
            r = rebuild(X, raw[col].to_numpy(dtype=float), plan, pool, col)
            rows.append({"вход": name, "признаков": X.shape[1], "слой": col,
                         "что это": FAULT_TITLE[col], **r})
            log(f"  {name:32s} {col:11s} связь={r['ранговая_связь']:+.3f} "
                f"ROC={r['roc_благоприятен']:.3f}")

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "directional_relief_faults.csv"
    res.to_csv(out, index=False)
    print("\n=== Восстановление разломных слоёв ===")
    print(res.pivot(index="вход", columns="слой", values="roc_благоприятен")
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n(ROC «фактор благоприятен»; планка случайного — 0.500)")
    # Раз разломы не восстанавливаются — помогают ли новые признаки самому прогнозу
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)
    blocks = crit_reference.block_labels(meta)
    maps, rows2 = {}, []
    for name, df in (("рабочая точка", rs),
                     ("рабочая точка + направленные", pd.concat([rs, dr], axis=1))):
        sc, aucs, per_ax = blind_map(df.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        maps[name] = sm
        m = score_map(sm, y, crit, pool, valid, D)
        rows2.append({"вариант": name, "признаков": df.shape[1],
                      "фолды_roc_auc": float(np.mean(aucs)), **m})
        log("  %-30s без тренда=%.3f"
            % (name, m["совпадение_без_тренда"]))
    print('\n=== Помогают ли направленные признаки прогнозу ===')
    print(pd.DataFrame(rows2).to_string(index=False,
                                        float_format=lambda v: f"{v:.4f}"))
    ta, tc = agree_detrended(maps["рабочая точка + направленные"], crit, pool, valid, D)
    tb, _ = agree_detrended(maps["рабочая точка"], crit, pool, valid, D)
    b = boot_diff(ta, tb, tc, pool, blocks)
    print("Разность (блочный бутстрэп 10 км): %+.3f [%+.3f; %+.3f]"
          % (b["разность_средняя"], b["разность_ниж_5"],
             b["разность_верх_95"]))

    print(f"\nOK: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
