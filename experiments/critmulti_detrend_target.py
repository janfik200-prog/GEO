"""Проверка механизма: что будет, если убрать наклон из обучающей цели.

ЗАЧЕМ. 24.09.2026 выяснилось, почему наш перенос сходится с критериальной картой
на основном листе и расходится за его рамкой. Обе карты — в основном плавные
наклоны: координатный полином 2-й степени объясняет 0.70 разброса критериального
балла на листе и 0.75 на смежной площади, потому что ведущие факторы формулы там
сами наклоны (`kory_dist` 0.94, `gran_dist` 0.81 на листе — объектов этих типов
на листе почти нет: гранитоидов 2 кв.км из 366). На листе оба наклона направлены
на юг, поэтому карты согласны; юго-западнее критериальный наклон разворачивается
на восток, а наш продолжает идти на юго-запад.

Отсюда гипотеза: модель выучила наклон листа, а не признаки объектов. Если это
так, обучение на ОСТАТКЕ критериального балла (после снятия координатного
полинома листа) должно снять наклон с нашей карты и поднять согласие вне листа
по метрике «без наклона», которая наклоном не обманывается. Если не поднимет —
дело не в наклоне, а в отсутствии нужных сведений в признаках.

Метод при этом остаётся дистанционным: меняется только форма обучающей цели,
набор признаков прежний (37 переносимых, слои самой формулы в него не входят).

ПЛОЩАДЬ. Сравнение идёт по всей области критериального расчёта (30 229 ячеек,
7557 кв.км), включая южную часть: критериальный выход там есть, и территории
должны быть сопоставимы. Рабочая сетка признаков — архивная сборка
`wide_v1_archive/dataset_wide.parquet` (231 x 452, шаг 500 м, X 499750-614750,
Y 7694750-7920250): в отличие от текущей широкой сетки она накрывает область
критериального расчёта целиком (текущая обрывается на Y 7769250 и не достаёт
897 кв.км на юге). Набор из 37 переносимых признаков в ней тот же и без
пропусков. Южная часть разбирается и отдельно — там фундамент выходит из-под
чехла, обстановка другая.

ЧТО СЧИТАЕТСЯ. Четыре варианта обучения на одних и тех же признаках листа:
  * порог верхних 10% критериального балла (как в рабочей карте) — опора;
  * порог верхних 10% ОСТАТКА балла после снятия наклона листа;
  * сам балл числом (регрессия) — опора для числовой формы;
  * остаток балла числом.
Каждая карта сглаживается рабочим ядром 5 x 1.25 км и сравнивается с
критериальной картой организации на площадях: вся область расчёта, лист, вся
смежная часть, и по квадрантам (юго-запад, юг, запад). Метрики протокола:
совпадение верхних 10% как есть, после снятия наклона (полином снимается с
каждой карты отдельно внутри площади) и ранговая корреляция. Дополнительно —
направление наклона самой карты (ранговая корреляция балла с X и Y), чтобы
видеть, сработала ли операция вообще.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet, данные листа,
      data/SBORKA_NESKOLKO_LISTOV/ПРОГНОЗ_ВОСПРОИЗВ/
ВЫХОД: outputs/metrics/critmulti_detrend_target.log и .csv,
       data/processed/forecast_arch_ядро5км.parquet (четыре карты на архивной
       сетке — их использует `criterial_multisheet_compare`)
Запуск из корня: ``python -m experiments.critmulti_detrend_target``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 integro_grid, neighbourhood as nb, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, poly_design  # noqa: E402
from experiments.target_form import make_reg  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)
from experiments.branch5_alignment import align  # noqa: E402
from experiments.criterial_multisheet_compare import (EXCLUDED,  # noqa: E402
                                                      load_criterial, zone_of)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = config.CREF_AREA
WIDE_PARQUET = config.PROCESSED_DIR / "wide_v1_archive" / "dataset_wide.parquet"

# Архивная сетка: накрывает область критериального расчёта целиком.
ARCH_META = integro_grid.GridMeta(obj_count=231 * 452, prop_count=0,
                                  pic=231, prf=452, dx=500.0, dy=500.0,
                                  x0=499500.0, y0=7694500.0, properties=[])

T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def residual(v: np.ndarray, D: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Остаток после снятия координатного полинома, подогнанного по `cells`."""
    beta, *_ = np.linalg.lstsq(D[cells], v[cells], rcond=None)
    return v - D @ beta


def agree(ours: np.ndarray, crit: np.ndarray, x: np.ndarray, y: np.ndarray,
          cells: np.ndarray) -> dict:
    """Согласие двух карт на площади: как есть, без наклона, ранговая связь."""
    a, b = ours[cells], crit[cells]
    D = poly_design(x[cells], y[cells])
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    ta, tb = a >= np.quantile(a, 1 - AREA), b >= np.quantile(b, 1 - AREA)
    tra, trb = ra >= np.quantile(ra, 1 - AREA), rb >= np.quantile(rb, 1 - AREA)
    return {"ячеек": int(cells.sum()),
            "совпадение": float((ta & tb).sum() / tb.sum()),
            "без_наклона": float((tra & trb).sum() / trb.sum()),
            "ранговая": float(sps.spearmanr(a, b).statistic),
            "наклон_X": float(sps.spearmanr(x[cells], a).statistic),
            "наклон_Y": float(sps.spearmanr(y[cells], a).statistic)}


def criterial_on_grid(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """Критериальная перспективность (больше = лучше) в ячейках рабочей сетки."""
    crt, msk, cx, cy = load_criterial()
    crt, msk = crt.ravel(), msk.ravel()
    cx, cy = cx.ravel(), cy.ravel()
    tree = cKDTree(np.column_stack([cx[msk], cy[msk]]))
    dist, idx = tree.query(np.column_stack([wx, wy]))
    return np.where(dist <= 375.0, -crt[msk][idx], np.nan)


def main() -> None:
    # --- лист: признаки, цель, наклон
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit_sheet = crit_reference.criterial_score(prognoz)   # больше = перспективнее
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    Dz = poly_design(cx, cy)

    wide = pd.read_parquet(WIDE_PARQUET)
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)
    in_box = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
              (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    tree = cKDTree(np.column_stack([wx[in_box], wy[in_box]]))
    _, idx = tree.query(np.column_stack([cx, cy]), k=1)
    w_in = wide.loc[in_box].reset_index(drop=True)
    log("признаков %d, рабочих ячеек листа %d, архивная сетка %d ячеек"
        % (len(cols), pool.size, len(wide)))

    # приведение широкой сборки к шкале листа — как в рабочей карте переноса
    raw_sheet = feat[cols]
    pair_w = w_in.loc[idx, cols].to_numpy(dtype=float)[valid]
    pair_s = raw_sheet.to_numpy(dtype=float)[valid]
    wide_al = pd.DataFrame(align(pair_w, pair_s, wide[cols].to_numpy(dtype=float)),
                           columns=cols)
    Xs = spectral_mask.apply(raw_sheet).to_numpy(dtype=float)
    Xw = spectral_mask.apply(wide_al).to_numpy(dtype=float)
    log("сборки приведены к общей шкале, маска обнажённости применена")

    # цели: сам балл и его остаток после снятия наклона листа
    crit_f = np.where(np.isfinite(crit_sheet), crit_sheet,
                      np.nanmin(crit_sheet[pool]))
    res_f = residual(crit_f, Dz, pool)
    y_base = (crit_f >= np.quantile(crit_f[pool], 1 - AREA)).astype(int)
    y_res = (res_f >= np.quantile(res_f[pool], 1 - AREA)).astype(int)
    log("доля разброса критериального балла, объяснимая наклоном листа: %.3f"
        % (1 - res_f[pool].var() / crit_f[pool].var()))
    log("верхние 10%% листа: опора %d ячеек, остаток %d, общих %d"
        % (int(y_base[pool].sum()), int(y_res[pool].sum()),
           int((y_base & y_res)[pool].sum())))

    # --- критериальная карта организации на рабочей сетке
    crit_w = criterial_on_grid(wx, wy)
    common = np.isfinite(crit_w)
    zone = zone_of(wx, wy)
    in_sheet = zone == "лист"
    areas = {
        "вся область расчёта": common,
        "лист": common & in_sheet,
        "вся смежная": common & ~in_sheet,
        "  юго-запад": common & (zone == "юго-запад"),
        "  юг": common & (zone == EXCLUDED),
        "  запад": common & (zone == "запад"),
    }
    log("площади сравнения: " + ", ".join("%s %d" % (k.strip(), int(v.sum()))
                                          for k, v in areas.items()))

    variants = (
        ("порог верх 10%, сам балл (опора)", "класс", y_base),
        ("порог верх 10%, остаток без наклона", "класс", y_res),
        ("число: сам балл (опора)", "число", crit_f),
        ("число: остаток без наклона", "число", res_f),
    )
    km = 1000.0 / ARCH_META.dx
    rows = []
    maps = wide[["row", "col", "x", "y"]].copy()
    for tag, kind, target in variants:
        if kind == "класс":
            mdl = make_model().fit(Xs[pool], target[pool].astype(int))
            score = mdl.predict_proba(Xw)[:, 1]
        else:
            score = make_reg().fit(Xs[pool], target[pool]).predict(Xw)
        score_sm = nb.aniso_smooth(score, (ARCH_META.prf, ARCH_META.pic),
                                   ALONG_KM * km, ACROSS_KM * km,
                                   nb.theta_from_azimuth(NW_AZIMUTH))
        maps[tag] = score_sm.astype("float32")
        log("=== %s ===" % tag)
        for name, cells in areas.items():
            if cells.sum() < 50:
                continue
            r = agree(score_sm, crit_w, wx, wy, cells)
            rows.append({"цель": tag, "площадь": name.strip(), **r})
            log("  %-22s совпадение %.3f, без наклона %.3f, ранговая %+.3f, "
                "наклон карты X %+.3f Y %+.3f"
                % (name, r["совпадение"], r["без_наклона"], r["ранговая"],
                   r["наклон_X"], r["наклон_Y"]))

    out = ROOT / "outputs" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    maps["в_габаритах_листа"] = in_sheet
    maps.to_parquet(config.PROCESSED_DIR / "forecast_arch_ядро5км.parquet",
                    index=False)
    log("карты на архивной сетке: %s"
        % (config.PROCESSED_DIR / "forecast_arch_ядро5км.parquet"))
    pd.DataFrame(rows).to_csv(out / "critmulti_detrend_target.csv", index=False,
                              encoding="utf-8-sig")
    (out / "critmulti_detrend_target.log").write_text("\n".join(LOG),
                                                      encoding="utf-8")
    log("готово: %s" % (out / "critmulti_detrend_target.csv"))


if __name__ == "__main__":
    main()
