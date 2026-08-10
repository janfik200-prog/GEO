"""Задача 7: существенность дельтовой/лагунной фаций и обобщающий фактор.

Постановка («Решаемые задачи», доп. исследования, п. 1, с рис. 1): дельтовая и
лагунная фации ильинской/бурдурской свит — фактор эксперта-геолога без прямого
отражения на геологической карте (``dist_facies`` в критериальной формуле,
вес 0.891, см. ``config.TAXONOMY_WEIGHTS``). Вопрос — существенны ли обе фации
по отдельности, и можно ли заменить их обобщённым формализуемым фактором:
зоной вкрест простирания от контакта чехла с фундаментом (``geo2_contact`` /
``geo2_strike_*``, см. ``src/geology_v2.py``).

Тот же аппарат, что у задачи № 6 (см. реестр): Spearman-таблица «кандидат
против фактора» плюс явное обезвреживание циркулярности. Здесь циркулярность
устроена иначе, чем у задачи 6 (там мина — дубль таргета; здесь исходный
фактор ``dist_facies`` ПО ПОСТРОЕНИЮ равен ``min(geo2_facies1, geo2_facies2)``
— это не находка, а тавтология, и корреляция combined-с-любой-из-двух ничего
не доказывает). Поэтому:

1. Существенность каждой фации — Spearman ``geo2_facies1``/``geo2_facies2``
   с ``dist_facies`` (насколько комбинированный фактор фактически определяется
   одной фацией) + пространственное доминирование (доля площади, где ближе
   дельта против лагуны).
2. Замена на обобщённый фактор — ablation критериальной формулы ГИС Интегро
   (:func:`src.features.compute_geo_score`, тот же рецепт): фациальный критерий
   поочерёдно подменяется на facies1-only, facies2-only и ``geo2_contact``,
   и через :mod:`src.crit_reference` меряется согласие получившегося
   альтернативного прогноза с НАСТОЯЩИМ нативным ``prognoz`` (тот же аппарат,
   что у ``experiments/crit_agreement.py`` для глубоких моделей).

Запуск из корня: ``python -X utf8 -m experiments.facies_significance``.

РЕЗУЛЬТАТ (06.08.2026, dataset_v5, 22 905 валидных ячеек)
-----------------------------------------------------------
Обе фации существенны по отдельности, ни одна не избыточна: дельта (CODE_F=1)
ближе на 79 % площади и коррелирует с combined ``dist_facies`` сильнее
(Spearman 0.97 против 0.79 у лагуны) — ожидаемо, раз она доминирует чаще. Но
при подмене фациального критерия в формуле ГИС Интегро на facies1-only или
facies2-only реконструкция нативного ``prognoz`` теряет СОПОСТАВИМО —
capture top-10% падает с 8.94x (baseline, точная реконструкция) до 8.17x
(-8.7 %, дельта-only) и 8.24x (-7.8 %, лагуна-only). Лагунная фация, несмотря
на меньшее покрытие, несёт почти столько же информации, сколько дельтовая, —
обе нужны, заменять формулу на одну из них нельзя без потери точности.

Обобщённый фактор (``geo2_contact``, зона вкрест простирания от контакта
чехла с фундаментом) заменяет экспертный факт ХУЖЕ любой из фаций поодиночке:
AUC 0.946 против 0.995 baseline (0.988/0.984 у фаций-соло), capture падает
до 6.37x — потеря -28.7 % против -8 % у фаций-соло, втрое сильнее. **Вывод:
формализуемая зона НЕ является адекватной заменой экспертной оцифровки
дельтовой/лагунной фаций** — она ловит только грубую структуру (близость
к границе чехла), но теряет специфику контуров ``fasii``.

ОГОВОРКА. Это тест ФОРМУЛЬНОЙ ВЕРНОСТИ, а не независимой предсказательной
силы: ``prognoz`` — детерминированная функция тех же шести ``dist_*``, и
высокий AUC здесь означает «формула почти не изменилась после подмены входа», а
не «признак предсказывает золото». Это ровно то, о чём спрашивает заказчик
(можно ли заменить вход рецепта), но не аргумент в пользу геологической
значимости фаций для оруденения — тот вопрос остаётся открытым (см. общий
негативный результат фазы обучения на критериальном, задача № 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, crit_reference, criterial_target, features  # noqa: E402


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    meta, prognoz = criterial_target.load_prognoz_grid()
    prognoz = prognoz.ravel()
    crit = crit_reference.criterial_score(prognoz)

    dist_facies = df["dist_facies"].to_numpy(dtype=float)
    facies1 = df["geo2_facies1"].to_numpy(dtype=float)   # CODE_F=1, дельтовая, 509 км²
    facies2 = df["geo2_facies2"].to_numpy(dtype=float)   # CODE_F=2, лагунная, 348 км²
    contact = df["geo2_contact"].to_numpy(dtype=float)

    print(f"ячеек {len(df)}, валидных {pool.size}")

    # --- sanity: dist_facies действительно = min(facies1, facies2) ---
    dev = np.nanmax(np.abs(np.minimum(facies1, facies2)[pool] - dist_facies[pool]))
    print(f"\nsanity: max|dist_facies - min(facies1,facies2)| = {dev:.4f} м "
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
    tbl1.to_csv(out_dir / "facies_significance_individual.csv", index=False)
    tbl2.to_csv(out_dir / "facies_significance_ablation.csv", index=False)
    print(f"\nсохранено: {out_dir / 'facies_significance_individual.csv'}, "
          f"{out_dir / 'facies_significance_ablation.csv'}")

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
    fig.suptitle("Задача 7: замена фациального критерия на обобщённую зону контакта\n"
                 "(красный x — рудные объекты, capture "
                 f"{tbl2.loc[tbl2['вариант'].str.startswith('baseline'), 'capture'].iloc[0]:.2f}x -> "
                 f"{tbl2.loc[tbl2['вариант'].str.startswith('обобщённый'), 'capture'].iloc[0]:.2f}x)",
                 fontsize=10)
    map_path = ROOT / "outputs" / "facies_significance_map.png"
    fig.savefig(map_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {map_path}")


if __name__ == "__main__":
    main()
