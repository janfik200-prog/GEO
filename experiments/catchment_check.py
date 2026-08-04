"""Этап 6: заверка по водосборам точек — непрерывная метрика вместо квантованной.

ЗАЧЕМ. Этап 5e измерил разрешающую способность точечной заверки и получил
приговор: на 19 точках одна точка = 0.53 lift, а одна и та же модель при разных
семенах обучения ловит от 1 до 5 точек. Разброс от случайности обучения шире,
чем вся разница между сравниваемыми методами, — то есть точечная метрика
измеряет не метод.

Водосборная заверка чинит это с двух сторон:

* СНИМАЕТ СИСТЕМАТИЧЕСКУЮ ОШИБКУ ПРИВЯЗКИ. Точка — ореол в русле, снесённый
  вниз по течению; коренной источник лежит выше. Метод, указавший источник в
  трёх километрах выше точки, сейчас считается промахнувшимся.
* ДЕЛАЕТ МЕТРИКУ НЕПРЕРЫВНОЙ. Единица заверки — доля площади водосбора,
  попавшей в перспективную площадь метода; она меняется плавно, а не скачком в
  1/19.

Нормировка на ``area`` делает метрику равной единице у случайной карты при
ЛЮБОМ размере водосбора, поэтому крупные водосборы не выигрывают автоматически.
Устойчивость к размеру проверяется явно на трёх ограничениях
(``config.CATCH_SIZES``): если порядок методов от размера водосбора зависит,
метрике нельзя верить, и это надо увидеть, а не замолчать.

ЧТО СРАВНИВАЕТСЯ. Все карты, накопленные этапами 5-5d и сохранённые в
``outputs/metrics/*_scores.npz``, плюс критериальный прогноз и планки (наивная
гидросеть, случайная карта). Ни одна карта здесь не обучается заново — этап
меняет ТОЛЬКО способ заверки, поэтому сравнение между этапами остаётся честным.

КРИТЕРИЙ ПРЕВОСХОДСТВА — тот же по форме, что и в точечном протоколе, но на
водосборной метрике: нижняя граница 90% кластерного бутстрэп-интервала разности
строго больше нуля И сдвиговый null даёт p ниже 0.05/``config.OWN_N_CONFIGS_TOTAL``.
Порог не смягчается: водосборная метрика — вторая предзарегистрированная
метрика того же протокола, а не повод пересчитать знаменатель.

Выход: ``outputs/metrics/catchment{,_sizes,_null,_boot,_verdict}.csv``,
``outputs/catchment_check.png``.

Запуск из корня: ``python -m experiments.catchment_check``.
"""
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

from src import (assessment, atypicality, catchments, cell_mask,  # noqa: E402
                 config, criterial_target)

NPZ = ["mae_scores.npz", "transfer_scores.npz", "transfer_cnn_scores.npz",
       "transfer_distil_scores.npz", "seed_stability_scores.npz"]
SKIP = {"valid", "emb", "centers", "criterial", "naive_hydro", "random"}


def load_maps(log) -> dict[str, np.ndarray]:
    """Все карты-претенденты из сохранённых прогонов (без пересчёта моделей)."""
    out: dict[str, np.ndarray] = {}
    for name in NPZ:
        p = ROOT / "outputs" / "metrics" / name
        if not p.exists():
            log(f"нет {name} — пропускаю")
            continue
        z = np.load(p)
        v = z["valid"].astype(bool)
        got = []
        for k in z.files:
            if k in SKIP or z[k].ndim != 1:
                continue
            a = z[k].astype(float)
            # Часть прогонов сохраняла скор только по валидным ячейкам
            # (mae_err), часть — по всей сетке. Приводим к сетке.
            out[k] = a if a.size == v.size else atypicality.expand_to_grid(a, v)
            got.append(k)
        log(f"{name}: {len(got)} карт")
    return out


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v2.parquet")
    valid = cell_mask.build_valid_mask(df)
    meta, prognoz = criterial_target.load_prognoz_grid()
    pool = np.flatnonzero(valid)

    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    cl_unb = assessment.point_clusters(unb, meta)
    log(f"точек: все {allp.size}, несмещённых {unb.size} "
        f"({np.unique(cl_unb).size} кластеров)")

    scores = load_maps(log)
    own = list(scores)
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))
    log(f"карт к сравнению: {len(scores)} (претендентов {len(own)})")

    # --- Водосборы при трёх ограничениях размера ---
    size_rows, catch_by_size = [], {}
    for size in config.CATCH_SIZES:
        cs = catchments.build_catchments(df, meta, unb, valid, max_cells=size)
        catch_by_size[size] = cs
        area_km2 = np.array([c.size for c in cs]) * 0.25
        log(f"водосборы (лимит {size} ячеек): медиана {np.median(area_km2):.1f} км2, "
            f"размах {area_km2.min():.1f}-{area_km2.max():.1f} км2")
        for n, s in scores.items():
            size_rows.append({"max_cells": size, "method": n,
                              "capture": catchments.catchment_capture(
                                  s, cs, config.VER_AREA, pool)})
    sizes_df = pd.DataFrame(size_rows)

    cs = catch_by_size[config.CATCH_MAX_CELLS]
    n_cells = np.array([c.size for c in cs])
    log(f"основной вариант: лимит {config.CATCH_MAX_CELLS} ячеек, "
        f"суммарно {n_cells.sum()} ячеек против {unb.size} при точечной заверке "
        f"(x{n_cells.sum() / unb.size:.0f})")

    # --- Протокол на водосборной метрике ---
    alpha = 0.05 / config.OWN_N_CONFIGS_TOTAL
    null = pd.DataFrame([{"method": n, **assessment.spatial_null_catch(
        s, cs, meta, config.VER_AREA, pool)} for n, s in scores.items()])
    log("сдвиговый null готов")
    boot = pd.DataFrame([{"a": n, **assessment.bootstrap_diff_catch(
        scores[n], scores["criterial"], cs, config.VER_AREA, pool,
        clusters=cl_unb)} for n in own])
    log("бутстрэп готов")

    verdict = boot.merge(null[["method", "p"]].rename(
        columns={"method": "a", "p": "p_shift"}), on="a")
    verdict = verdict.rename(columns={"a": "method", "lift_a": "capture",
                                      "lift_b": "capture_criterial"})
    verdict["alpha"] = alpha
    verdict["cond1_ci_lo_gt_0"] = verdict["ci_lo"] > 0
    verdict["cond2_p_lt_alpha"] = verdict["p_shift"] < alpha
    verdict["superior"] = verdict["cond1_ci_lo_gt_0"] & verdict["cond2_p_lt_alpha"]
    verdict = verdict.sort_values("capture", ascending=False)

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "catchment"
    sizes_df.to_csv(f"{stem}_sizes.csv", index=False)
    null.to_csv(f"{stem}_null.csv", index=False)
    boot.to_csv(f"{stem}_boot.csv", index=False)
    verdict.to_csv(f"{stem}_verdict.csv", index=False)
    pd.DataFrame({"cell": unb, "cluster": cl_unb,
                  "catch_cells": n_cells}).to_csv(f"{stem}.csv", index=False)

    # --- Рисунок ---
    cap_cr = float(verdict["capture_criterial"].iloc[0])
    top = verdict.head(12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    yy = np.arange(len(top))
    ax1.barh(yy, top["capture"], color="tab:blue")
    ax1.errorbar(top["capture"], yy,
                 xerr=[top["capture"] - (top["capture"] - (top["delta"] - top["ci_lo"])),
                       (top["ci_hi"] - top["delta"])],
                 fmt="none", ecolor="k", elinewidth=1, capsize=3)
    ax1.axvline(cap_cr, color="crimson", lw=2,
                label=f"критериальный прогноз ({cap_cr:.2f})")
    ax1.axvline(1.0, color="gray", ls=":", label="случайная карта (1.00)")
    ax1.set_yticks(yy)
    ax1.set_yticklabels(top["method"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("доля площади водосбора в top-10%, нормированная (безразм.)")
    ax1.set_title(f"Водосборная заверка, лимит {config.CATCH_MAX_CELLS} ячеек "
                  f"({config.CATCH_MAX_CELLS * 0.25:.0f} км2)\n"
                  f"{unb.size} водосборов несмещённых точек")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="x")

    piv = sizes_df.pivot(index="method", columns="max_cells", values="capture")
    piv = piv.loc[list(top["method"]) + ["criterial", "random"]]
    for name, row in piv.iterrows():
        st = dict(lw=2.5, color="crimson") if name == "criterial" else {}
        ax2.plot([c * 0.25 for c in piv.columns], row.to_numpy(), marker="o",
                 ms=4, label=name, **st)
    ax2.set_xlabel("ограничение размера водосбора, км2")
    ax2.set_ylabel("водосборная метрика, безразм.")
    ax2.set_title("Чувствительность к размеру водосбора\n"
                  "(порядок методов не должен зависеть от лимита)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)
    fig.suptitle("Заверка по площади сноса вместо ячейки точки", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "catchment_check.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    print("\nВодосборная метрика при разных ограничениях размера:")
    print(piv.round(3).to_string())
    print(f"\nВердикт (водосборная метрика, лимит {config.CATCH_MAX_CELLS} "
          f"ячеек, alpha = {alpha:.5f}):")
    print(verdict[["method", "capture", "capture_criterial", "delta", "ci_lo",
                   "ci_hi", "p_shift", "superior"]].round(3)
          .to_string(index=False))
    if verdict["superior"].any():
        log("КРИТЕРИЙ ВЗЯТ: "
            + ", ".join(verdict.loc[verdict["superior"], "method"]))
    else:
        log("критерий не взят и на водосборной метрике")
    log("сохранено: outputs/catchment_check.png, outputs/metrics/catchment*.csv")


if __name__ == "__main__":
    main()
