"""Заверка кварцевого индекса ASTER TIR: есть ли в нём геология.

Признак окварцевания добыт (``python -m experiments.fetch_sat astir``), и до
включения в датасет он обязан пройти три проверки, каждая из которых способна
его закрыть:

1. ПОКРЫТИЕ. Смысл ветки TIR был в том, что тепловые каналы пережили отказ
   SWIR 2008 года и потому закрывают дыру группы ``ast`` (71.6% валидных
   ячеек). Если покрытие не выше — ветка не дала того, ради чего заводилась.
2. НЕ АРТЕФАКТ ЛИ. Главные конкуренты за объяснение карты — растительность
   (98.3% пикселей листа) и прогрев склонов. Нормировка на 300 К снимает
   температуру по построению, но проверять надо измерением: считается
   корреляция с ``s2_ndvi`` и ``astir_t13``, а затем частная корреляция
   с геологией при вынесенных из индекса NDVI и температуре.
3. ЕСТЬ ЛИ СИГНАЛ. Мерится не по 19 точкам заверки (шаг метрики там 0.53
   lift на точку), а по критериальному эталону на 22 905 ячейках — тем же
   протоколом, что и все прочие карты этапов 7-9: AUC, сдвиговый нуль,
   блочный бутстрэп.

ВЕРДИКТ (прогон 06.08.2026, 116 гранул LP DAAC, 2001-2025, 22 905 валидных
ячеек). Прошла только проверка 1 — покрытие 100% против 69.7% у ``ast``,
подтверждение того, что TIR пережил отказ SWIR 2008 года. Проверки 2-4
провалены:

* ``astir_qi`` коррелирует с ``s2_ndvi`` rho=-0.65 и с ``dem_slope``
  rho=0.53 — выше принятого порога артефакта |rho|>0.5. Нормировка на 300 К
  снимает объёмную температуру, но не субпиксельную растительность и не
  геометрию освещения склонов — для отношений TIR-каналов это остаётся
  главным сигналом;
* после линейного выноса NDVI и t13 частная корреляция с геологией слабая
  (|rho_част| <= 0.25 по всем 10 слоям, критериальный прогноз 0.222);
* критериальный протокол (``crit_reference.run_protocol``): AUC 0.58-0.64
  у всех трёх индексов Ниномии, ниже порога 0.75; ``reproduces=False`` у
  всех трёх и у обеих планок (``naive_hydro``, ``random``).

Точечная заверка (вторична, шаг 0.53 lift/точка на 19 ячейках) не меняет
картину: ``astir_qi`` lift@10%=1.58 против 2.10 у критериального — разница
на грани шума одной точки.

Решение: ``astir_qi/ci/mi`` в датасет как геологический признак НЕ идут —
то, что они измеряют, это NDVI и уклон, уже присутствующие в датасете
напрямую. Ветка закрыта отрицательным результатом; в v4 не переходит.

Запуск из корня: ``python -m experiments.quartz_check``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, integro_grid)

#: Индексы Ниномии и то, что каждый из них означает геологически.
INDICES = {"astir_qi": "кварцевый (окварцевание)",
           "astir_ci": "карбонатный",
           "astir_mi": "мафический (обратен SiO2)"}

#: С чем сверяемся: конкуренты за объяснение карты и геологические слои.
CONFOUND = ("s2_ndvi", "astir_t13", "dem_elev", "dem_slope", "l8_lst",
            "astir_n_obs", "astir_valid_frac")
GEOLOGY = ("mask_svita", "dist_facies", "dist_struct", "dist_paleo",
           "dist_tect1", "dist_tect2", "dist_magm", "dens_tect",
           "ast_alter", "ast_aloh")


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    """Спирмен по ячейкам, где обе величины определены."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def _residual(y: np.ndarray, xs: list[np.ndarray]) -> np.ndarray:
    """Остаток ``y`` после линейного выноса ``xs`` (частная корреляция).

    Линейный вынос, а не ранговый: цель — не идеальная декорреляция, а ответ
    на вопрос «останется ли связь с геологией, если убрать очевидное».
    """
    ok = np.isfinite(y) & np.all([np.isfinite(x) for x in xs], axis=0)
    out = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < 100:
        return out
    A = np.column_stack([np.ones(ok.sum())] + [x[ok] for x in xs])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    out[ok] = y[ok] - A @ coef
    return out


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    astir_path = config.PROCESSED_DIR / "astir_features.parquet"
    if not astir_path.exists():
        raise SystemExit("нет astir_features.parquet — сначала "
                         "python -m experiments.fetch_sat astir")
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v3.parquet")
    tir = pd.read_parquet(astir_path)
    assert len(tir) == len(df), f"длина {len(tir)} != датасета {len(df)}"
    df = pd.concat([df, tir], axis=1)
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    log(f"ячеек {len(df)}, валидных {pool.size}")

    # --- 1. Покрытие: ради него ветка и заводилась -------------------------
    print("\n=== 1. Покрытие листа ===")
    rows = []
    for grp, col in (("ast (SWIR, MPC)", "ast_alter"),
                     ("astir (TIR, LP DAAC)", "astir_qi")):
        if col not in df.columns:
            continue
        has = np.isfinite(df[col].to_numpy()) & valid
        rows.append({"группа": grp, "валидных ячеек": int(has.sum()),
                     "доля": has.sum() / valid.sum()})
    cov = pd.DataFrame(rows)
    for _, r in cov.iterrows():
        print(f"  {r['группа']:<24} {r['валидных ячеек']:>7} ячеек  ({r['доля']:.1%})")
    if "astir_n_obs" in df.columns:
        no = df["astir_n_obs"].to_numpy()[valid]
        print(f"  наблюдений на ячейку: медиана {np.nanmedian(no):.1f}, "
              f"максимум {np.nanmax(no):.1f}")

    # --- 2. Не артефакт ли ------------------------------------------------
    print("\n=== 2. Конкуренты за объяснение карты (Spearman) ===")
    conf_tbl = []
    for name in INDICES:
        v = df[name].to_numpy()[valid]
        row = {"индекс": name}
        for c in CONFOUND:
            if c in df.columns:
                row[c] = _rho(v, df[c].to_numpy()[valid])
        conf_tbl.append(row)
    conf = pd.DataFrame(conf_tbl).set_index("индекс")
    print(conf.round(3).to_string())
    print("  (|rho| > 0.5 с NDVI или t13 = индекс меряет покров/прогрев, "
          "а не минерал)")

    print("\n=== 3. Связь с геологией: сырая и частная ===")
    print("   частная = после линейного выноса NDVI и температуры t13\n")
    ctrl = [df[c].to_numpy()[valid] for c in ("s2_ndvi", "astir_t13")
            if c in df.columns]
    geo_rows = []
    for name in INDICES:
        v = df[name].to_numpy()[valid]
        res = _residual(v, ctrl) if ctrl else v
        for g in GEOLOGY + ("критериальный прогноз",):
            gv = crit[valid] if g == "критериальный прогноз" else (
                df[g].to_numpy()[valid] if g in df.columns else None)
            if gv is None:
                continue
            geo_rows.append({"индекс": name, "слой": g,
                             "rho": _rho(v, gv), "rho_част": _rho(res, gv)})
    geo = pd.DataFrame(geo_rows)
    piv = geo.pivot(index="слой", columns="индекс", values="rho_част")
    print(piv.round(3).to_string())

    # --- 4. Заверка по критериальному эталону -----------------------------
    print("\n=== 4. Заверка по критериальному эталону (22 905 ячеек) ===")
    scores = {}
    for name in INDICES:
        v = df[name].to_numpy()
        # Пропуски заполняются медианой пула: протокол ранжирует все ячейки,
        # и NaN иначе молча уедут в конец ранга, изобразив «неперспективно».
        v = np.where(np.isfinite(v), v, np.nanmedian(v[valid]))
        scores[name] = v
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))
    own = list(INDICES)
    alpha = 0.05 / (config.CREF_ORIENT_PENALTY * len(own))
    res = crit_reference.run_protocol(scores, crit, meta, valid, alpha, log=log)
    v = res["verdict"]
    print(v[["method", "auc", "auc_ci_lo", "auc_ci_hi", "p_shift",
             "reproduces"]].round(4).to_string(index=False))

    # --- 5. Точечная заверка (вторична: шаг метрики 0.53) -----------------
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    print(f"\n=== 5. Точечная заверка ({unb.size} несмещённых ячеек) ===")
    print(f"   шаг метрики = {1 / (unb.size * config.VER_AREA):.2f} lift на точку")
    lift = {"критериальный": assessment.capture_efficiency(crit, unb, pool=pool)}
    for m in own:
        s = crit_reference.orientation(scores[m], crit, pool) * scores[m]
        lift[m] = assessment.capture_efficiency(s, unb, pool=pool)
    for k, val in lift.items():
        print(f"  {k:<16} lift@10% = {val:.2f}")

    # --- Выгрузка ---------------------------------------------------------
    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    cov.to_csv(out / "metrics" / "quartz_coverage.csv", index=False)
    conf.to_csv(out / "metrics" / "quartz_confounders.csv")
    geo.to_csv(out / "metrics" / "quartz_geology.csv", index=False)
    v.to_csv(out / "metrics" / "quartz_verdict.csv", index=False)

    shape = (meta.prf, meta.pic)
    mask2d = np.asarray(valid).reshape(shape)
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    panels = list(INDICES) + ["критериальный прогноз"]
    for ax, name in zip(axes, panels):
        arr = crit if name == "критериальный прогноз" else df[name].to_numpy()
        img = np.where(mask2d, np.asarray(arr).reshape(shape), np.nan)
        if np.isfinite(img).sum() < 10:
            ax.set_title(f"{name}: нет данных")
            ax.axis("off")
            continue
        lo, hi = np.nanpercentile(img, [2, 98])
        im = ax.imshow(img, cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(INDICES.get(name, name), fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("ASTER TIR: индексы Ниномии против критериального прогноза "
                 f"(сетка {meta.prf}x{meta.pic}, шаг {meta.dx:.0f} м)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "quartz_index.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"сохранено: {out / 'quartz_index.png'} и 4 таблицы в outputs/metrics")


if __name__ == "__main__":
    main()
