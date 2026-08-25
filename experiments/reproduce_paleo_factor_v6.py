"""Задача 6, повтор на dataset_v6: воспроизвести фактор «долины и впадины»
на актуальном (очищенном) наборе признаков вместо dataset_v5.

Отличия от ``reproduce_paleo_factor.py`` (только источник данных, протокол и
метрики те же — честная блочная CV 10 км, GroupKFold, 5 сидов):

1. ``data/processed/dataset_v6.parquet`` вместо ``dataset_v5.parquet`` — 104
   признака-кандидата вместо 181: убраны служебные колонки покрытия
   (*_valid_frac/*_n_obs, риск утечки — уже нет нужды фильтровать явно,
   dataset_v6 их не содержит), 33 шумовых признака (permutation importance
   <=0 на 8 leave-one-strip-out фолдах) и дубль l8_green
   (``data/processed/dataset_v6_notes.md``).
2. Минерагенические признаки (геохимические ореолы Au/U, привнос урана)
   по-прежнему добавляются отдельно поверх — это независимый источник,
   не входящий ни в один из версионных датасетов (см. докстринг
   ``reproduce_paleo_factor.py`` про легитимность минерагенической карты
   именно для этой цели, не для золота).

Запуск из корня: ``python -X utf8 -m experiments.reproduce_paleo_factor_v6``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, criterial_target, features_v2, mineragenic_features  # noqa: E402
from src.data_loader import read_sidecar_proj4  # noqa: E402
from src.integro_grid import read_grid_proj4, read_pgrid  # noqa: E402

CV_SEEDS = (1, 7, 13, 21, 42)
AREAS = (0.05, 0.10, 0.20)
N_BOOT = config.CREF_N_BOOT
CI = (5.0, 95.0)

RELIEF_COLS_PREFIXES = ("dem_", "relief_m", "relief2_")


def _feature_group(col: str) -> str:
    if col.startswith("mingeo_"):
        return "mingeo"
    return features_v2.feature_group(col) or "other"


def _oof_proba(X: np.ndarray, y: np.ndarray, blocks: np.ndarray, seeds: tuple[int, ...]) -> tuple[np.ndarray, list]:
    """OOF-вероятность (GroupKFold по блокам, усреднение по сидам) + обученные модели."""
    gkf = GroupKFold(n_splits=5)
    stack = np.zeros((len(seeds), y.size))
    models = []
    for si, seed in enumerate(seeds):
        for tr, te in gkf.split(X, y, blocks):
            model = RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )
            model.fit(X[tr], y[tr])
            stack[si, te] = model.predict_proba(X[te])[:, 1]
            models.append(model)
    return stack.mean(axis=0), models


def _lift_at_area(y: np.ndarray, proba: np.ndarray, area: float) -> float:
    """lift@area = recall в топ-area / area (1 = случайно, 1/area = идеально)."""
    thr = np.quantile(proba, 1 - area)
    top = proba >= thr
    realized = top.mean()
    recall = (top & (y == 1)).sum() / max(int(y.sum()), 1)
    return recall / realized if realized > 0 else 0.0


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    out = {"auc": roc_auc_score(y, proba), "ap": average_precision_score(y, proba)}
    for a in AREAS:
        out[f"lift@{int(a * 100)}%"] = _lift_at_area(y, proba, a)
    return out


def _block_bootstrap_auc_delta(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray,
                                blocks: np.ndarray, seed: int = config.RANDOM_STATE) -> dict:
    """Блочный бутстрэп дельты AUC (a - b) по тем же 10-км блокам, что и CV."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            deltas[i] = np.nan
            continue
        deltas[i] = roc_auc_score(yb, proba_a[idx]) - roc_auc_score(yb, proba_b[idx])
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.percentile(deltas, CI)
    return {"delta": float(np.nanmean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": int(deltas.size)}


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    pool = np.flatnonzero(valid)
    print(f"ячеек {len(df)}, валидных {pool.size}")

    # --- 0. Минерагенические признаки (геохимические ореолы + привнос урана) ---
    target_meta = read_pgrid(config.GOLD_TARGET_PGRID)
    target_proj4 = read_grid_proj4(config.GOLD_TARGET_PGRID)
    shp_dir = config.GOLD_TARGET_PGRID.parents[1] / config.SHP_SUBDIR
    halo_path = shp_dir / "геохимические ореолы.shp"
    uran_path = shp_dir / "привнос урана.shp"
    halo, uran = gpd.read_file(halo_path), gpd.read_file(uran_path)
    for name, path in (("геохимические ореолы.shp", halo_path), ("привнос урана.shp", uran_path)):
        p = read_sidecar_proj4(path)
        tag = "совпадает" if p == target_proj4 else "ОТЛИЧАЕТСЯ (towgs84, некритично на сетке 500 м)"
        print(f"CRS {name}: {tag}")
    mingeo = mineragenic_features.geochem_distance_features(target_meta, halo, uran)
    rows, cols = np.indices(target_meta.shape)
    assert np.array_equal(rows.ravel(), df["row"].to_numpy()) and np.array_equal(cols.ravel(), df["col"].to_numpy()), \
        "сетка минерагенических признаков не совпадает по порядку с dataset_v6"
    for name, arr in mingeo.items():
        df[name] = arr.ravel()
    print(f"минерагенические признаки: {sorted(mingeo)}")

    # --- 1. Цель: бинарный факт «внутри полигона палеодолины» ---
    y_full = (df["dist_paleo"].to_numpy(dtype=float) == 0).astype(int)
    y = y_full[pool]
    print(f"\nцель: внутри палеодолины {y.sum()} из {pool.size} валидных ячеек ({y.mean():.1%})")

    # --- 2. Признаки: всё, КРОМЕ dist_paleo (dataset_v6 уже без служебных колонок) ---
    exclude = {"dist_paleo", "row", "col", "x", "y", "mask_svita"}
    feat_cols = [c for c in df.columns if c not in exclude]
    print(f"признаков в полной модели: {len(feat_cols)}")
    X = df[feat_cols].fillna(0).to_numpy(dtype=float)[pool]

    x_c, y_c = df["x"].to_numpy(dtype=float)[pool], df["y"].to_numpy(dtype=float)[pool]
    bx = np.floor((x_c - x_c.min()) / config.VAL_BLOCK_SIZE).astype(int)
    by = np.floor((y_c - y_c.min()) / config.VAL_BLOCK_SIZE).astype(int)
    blocks = bx * (by.max() + 1) + by

    # --- 3. Полная модель (OOF, блочная CV) ---
    proba_full, models_full = _oof_proba(X, y, blocks, CV_SEEDS)
    m_full = _metrics(y, proba_full)
    print("\n--- 1. Полная модель (dataset_v6 + минерагения) ---")
    print(pd.Series(m_full).round(4).to_string())

    # --- 4. Baseline А: наивная гипотеза «долина = понижение рельефа» ---
    relief_cols = [c for c in feat_cols if c.startswith(RELIEF_COLS_PREFIXES) or c.startswith("lin_")]
    Xr = df[relief_cols].fillna(0).to_numpy(dtype=float)[pool]
    proba_relief, _ = _oof_proba(Xr, y, blocks, CV_SEEDS)
    m_relief = _metrics(y, proba_relief)
    print(f"\n--- 2. Baseline: только рельеф/линеаменты ({len(relief_cols)} признаков, "
          "наивная гипотеза «долина=понижение») ---")
    print(pd.Series(m_relief).round(4).to_string())

    # --- 5. Baseline Б: mask_svita в одиночку (сильнейший univariate признак реестра) ---
    proba_svita = df["mask_svita"].to_numpy(dtype=float)[pool]
    m_svita = _metrics(y, proba_svita)
    print("\n--- 3. Baseline: только mask_svita (геологическая карта, univariate) ---")
    print(pd.Series(m_svita).round(4).to_string())

    # --- 6. Значимость: полная модель против baseline'ов (блочный бутстрэп дельты AUC) ---
    boot_relief = _block_bootstrap_auc_delta(y, proba_full, proba_relief, blocks)
    boot_svita = _block_bootstrap_auc_delta(y, proba_full, proba_svita, blocks)
    print("\n--- 4. Блочный бутстрэп дельты AUC (полная модель - baseline) ---")
    print(f"vs рельеф:      delta={boot_relief['delta']:.4f}, "
          f"95% ДИ [{boot_relief['ci_lo']:.4f}, {boot_relief['ci_hi']:.4f}], блоков {boot_relief['n_boot']}")
    print(f"vs mask_svita:  delta={boot_svita['delta']:.4f}, "
          f"95% ДИ [{boot_svita['ci_lo']:.4f}, {boot_svita['ci_hi']:.4f}], блоков {boot_svita['n_boot']}")

    # --- 7. Интерпретируемость: важность признаков полной модели, по признакам и по группам ---
    imp = np.mean([m.feature_importances_ for m in models_full], axis=0)
    imp_tbl = pd.DataFrame({"признак": feat_cols, "важность": imp}).sort_values("важность", ascending=False)
    print("\n--- 5. Топ-20 признаков полной модели по важности (MDI, усреднено по фолдам/сидам) ---")
    print(imp_tbl.head(20).round(4).to_string(index=False))

    imp_tbl["группа"] = imp_tbl["признак"].map(_feature_group)
    group_imp = imp_tbl.groupby("группа")["важность"].sum().sort_values(ascending=False)
    print("\nважность по группам источников данных:")
    print(group_imp.round(4).to_string())

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"вариант": "полная модель", **m_full},
                  {"вариант": "только рельеф", **m_relief},
                  {"вариант": "только mask_svita", **m_svita}]
                ).to_csv(out_dir / "paleo_factor_agreement_v6.csv", index=False)
    imp_tbl.to_csv(out_dir / "paleo_factor_importance_v6.csv", index=False)
    print(f"\nсохранено: {out_dir / 'paleo_factor_agreement_v6.csv'}, {out_dir / 'paleo_factor_importance_v6.csv'}")

    # --- 8. Карта листа: OOF-вероятность палеодолины (полная модель vs рельеф) ---
    import matplotlib.pyplot as plt

    prf, pic = target_meta.prf, target_meta.pic
    proba_full_grid = np.full(prf * pic, np.nan)
    proba_full_grid[pool] = proba_full
    proba_relief_grid = np.full(prf * pic, np.nan)
    proba_relief_grid[pool] = proba_relief
    actual = y_full.astype(float)
    actual[~valid] = np.nan

    _, prognoz = criterial_target.load_prognoz_grid()
    obj_labels = criterial_target.label_ore_objects(prognoz).ravel()
    obj_rows, obj_cols = [], []
    for oid in range(1, int(obj_labels.max()) + 1):
        r, c = np.unravel_index(np.flatnonzero(obj_labels == oid), (prf, pic))
        obj_rows.append(r.mean())
        obj_cols.append(c.mean())

    def _panel(ax, arr, title):
        im = ax.imshow(arr.reshape(prf, pic), cmap="viridis", vmin=0, vmax=1)
        ax.contour(actual.reshape(prf, pic), levels=[0.5], colors="red", linewidths=1.2)
        ax.scatter(obj_cols, obj_rows, marker="x", color="white", s=70, linewidths=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    im = _panel(axes[0], proba_full_grid, f"Полная модель (AUC {m_full['auc']:.2f})")
    _panel(axes[1], proba_relief_grid, f"Только рельеф/линеаменты (AUC {m_relief['auc']:.2f})")
    fig.colorbar(im, ax=axes, shrink=0.75, label="OOF-вероятность палеодолины")
    fig.suptitle("Задача 6 (dataset_v6): воспроизведение фактора «долины и впадины»\n"
                "(красный контур — факт. граница палеодолины, белый x — рудные объекты)",
                fontsize=10)
    map_path = ROOT / "outputs" / "paleo_factor_map_v6.png"
    fig.savefig(map_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {map_path}")


if __name__ == "__main__":
    main()
