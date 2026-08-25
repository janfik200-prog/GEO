"""Задача 7, повтор на dataset_v6: существенность дельтовой/лагунной фаций и
обобщающий фактор. Протокол и метод — те же, что в ``facies_significance.py``
(dataset_v5); отличие только в источнике признаков.

МИНА, специфичная для этого повтора: ``geo2_facies1`` (дельтовая фация) не
входит в ``dataset_v6.parquet`` — при чистке v6 признак попал в число 33
«шумовых» (нулевая/отрицательная permutation importance для ОБЩЕЙ ML-фазы,
см. ``data/processed/dataset_v6_notes.md``). Для задачи 7 это не довод: здесь
цель — не ML-важность признака, а формульная верность (можно ли заменить вход
рецепта критериальной формулы), и без ``facies1`` сравнение с ``facies2`` не
провести. Признак не восстанавливается из dataset_v6, а пересчитывается заново
из тех же исходных шейп-файлов (``src.geology_v2.geology_v2_features`` —
детерминированная функция координат сетки, не требует переобучения ничего);
согласованность с уже сохранёнными в dataset_v6 ``geo2_facies2``/
``geo2_contact`` проверяется явным sanity-чеком ниже перед основным расчётом.

Запуск из корня: ``python -X utf8 -m experiments.facies_significance_v6``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target, features, geology_v2  # noqa: E402


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    meta, prognoz = criterial_target.load_prognoz_grid()
    prognoz = prognoz.ravel()
    crit = crit_reference.criterial_score(prognoz)

    dist_facies = df["dist_facies"].to_numpy(dtype=float)
    facies2 = df["geo2_facies2"].to_numpy(dtype=float)   # CODE_F=2, лагунная, 348 км²
    contact = df["geo2_contact"].to_numpy(dtype=float)

    print(f"ячеек {len(df)}, валидных {pool.size}")

    # --- geo2_facies1 не входит в dataset_v6 (убран как шумовой для общей ML-фазы) ---
    # пересчитывается напрямую из shp, не из датасета
    geo2_recalc = geology_v2.geology_v2_features(meta)
    facies1 = geo2_recalc["geo2_facies1"].to_numpy(dtype=float)

    dev_f2 = np.nanmax(np.abs(geo2_recalc["geo2_facies2"].to_numpy(dtype=float)[pool] - facies2[pool]))
    dev_ct = np.nanmax(np.abs(geo2_recalc["geo2_contact"].to_numpy(dtype=float)[pool] - contact[pool]))
    print(f"\nsanity: пересчитанные geo2_facies2/geo2_contact совпадают с dataset_v6 "
          f"(max|дельта|: facies2={dev_f2:.6f} м, contact={dev_ct:.6f} м)")
    assert dev_f2 < 1e-6 and dev_ct < 1e-6, "пересчёт geo2_v2 разошёлся с dataset_v6 — не sanity, а расхождение"

    # --- sanity: dist_facies действительно = min(facies1, facies2) ---
    dev = np.nanmax(np.abs(np.minimum(facies1, facies2)[pool] - dist_facies[pool]))
    print(f"sanity: max|dist_facies - min(facies1,facies2)| = {dev:.4f} м "
          "(тавтология по построению, не находка)")

    # --- 1. Существенность каждой фации ---
    print("\n--- 1. Насколько combined dist_facies определяется каждой фацией отдельно ---")
    rows = []
    for name, arr in (("geo2_facies1 (дельта)", facies1),
                      ("geo2_facies2 (лагуна)", facies2)):
        rho, p = stats.spearmanr(arr[pool], dist_facies[pool])
        rows.append({"фактор": name, "spearman_с_dist_facies": rho, "p": p})
    tbl1 = pd.DataFrame(rows)
    print(tbl1.round(4).to_string(index=False))

    dominant1 = facies1[pool] <= facies2[pool]
    print(f"\nдельта (facies1) ближе на {dominant1.mean():.1%} валидной площади, "
          f"лагуна (facies2) ближе на {(1 - dominant1.mean()):.1%}")

    # --- 2. Замена на обобщённый фактор: ablation критериальной формулы ---
    print("\n--- 2. Ablation: подмена фациального критерия в формуле ГИС Интегро ---")
    variants = {
        "baseline (dist_facies, нативный)": dist_facies,
        "только geo2_facies1 (дельта)": facies1,
        "только geo2_facies2 (лагуна)": facies2,
        "обобщённый geo2_contact": contact,
    }
    dist_by_role = {role: df[f"dist_{role}"].to_numpy(dtype=float)
                    for role in config.TAXONOMY_TRANSFORMS}
    rows2 = []
    for name, arr in variants.items():
        c = features.taxonomy_weighted_distance(dist_by_role, overrides={"facies": arr})
        score = -c  # больше = перспективнее, полярность как у crit
        ag = crit_reference.agreement(score, crit, pool)
        rows2.append({"вариант": name, **ag})
    tbl2 = pd.DataFrame(rows2)
    print(tbl2.round(4).to_string(index=False))

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl1.to_csv(out_dir / "facies_significance_individual_v6.csv", index=False)
    tbl2.to_csv(out_dir / "facies_significance_ablation_v6.csv", index=False)
    print(f"\nсохранено: {out_dir / 'facies_significance_individual_v6.csv'}, "
          f"{out_dir / 'facies_significance_ablation_v6.csv'}")

    # --- 3. Карта листа: нативные фации vs обобщённая зона geo2_contact ---
    import matplotlib.pyplot as plt

    invalid = ~np.isfinite(prognoz)

    def _score_for(arr: np.ndarray) -> np.ndarray:
        c = features.taxonomy_weighted_distance(dist_by_role, overrides={"facies": arr})
        return -c

    score_baseline = _score_for(dist_facies)
    score_contact = _score_for(contact)
    diff = score_contact - score_baseline

    obj_labels = criterial_target.label_ore_objects(prognoz.reshape(meta.prf, meta.pic)).ravel()
    obj_rows, obj_cols = [], []
    for oid in range(1, int(obj_labels.max()) + 1):
        r, c_ = np.unravel_index(np.flatnonzero(obj_labels == oid), (meta.prf, meta.pic))
        obj_rows.append(r.mean())
        obj_cols.append(c_.mean())

    def _panel(ax, arr, title, cmap="viridis", **kw):
        img = arr.reshape(meta.prf, meta.pic).astype(float).copy()
        img.ravel()[invalid] = np.nan
        im = ax.imshow(img, cmap=cmap, **kw)
        ax.scatter(obj_cols, obj_rows, marker="x", color="red", s=70, linewidths=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    im0 = _panel(axes[0], score_baseline, "Нативные фации\n(dist_facies)")
    _panel(axes[1], score_contact, "Обобщённая зона\n(geo2_contact)")
    lim = np.nanmax(np.abs(diff))
    im2 = _panel(axes[2], diff, "Разница\n(contact − нативные)", cmap="coolwarm", vmin=-lim, vmax=lim)
    fig.colorbar(im0, ax=axes[:2], shrink=0.75, label="скор формулы, больше = перспективнее")
    fig.colorbar(im2, ax=axes[2], shrink=0.75, label="разница скоров")
    fig.suptitle("Задача 7 (dataset_v6): замена фациального критерия на обобщённую зону контакта\n"
                 "(красный x — рудные объекты, capture "
                 f"{tbl2.loc[tbl2['вариант'].str.startswith('baseline'), 'capture'].iloc[0]:.2f}x -> "
                 f"{tbl2.loc[tbl2['вариант'].str.startswith('обобщённый'), 'capture'].iloc[0]:.2f}x)",
                 fontsize=10)
    map_path = ROOT / "outputs" / "facies_significance_map_v6.png"
    fig.savefig(map_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {map_path}")


if __name__ == "__main__":
    main()
