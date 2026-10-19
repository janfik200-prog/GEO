"""Вклад признаков по Шепли: на чём держится модель на листе и на юго-западе.

ЗАЧЕМ. Перестановочная проверка значимости говорит, сколько теряет качество
без признака, но не говорит, КУДА признак толкает балл в конкретной ячейке.
Значения Шепли дают именно это: разложение балла каждой ячейки по признакам со
знаком. Нужно понять две вещи: на что модель опирается на обученной территории
и чем она пользуется за её пределами, где перенос не работает.

ЧТО СЧИТАЕТСЯ. Рабочая точка как есть: цель — верхние 10% критериального балла
листа, признаки — набор переноса с маской обнажённости, рельеф нормирован
внутри своей зоны (вариант C из `terrain_drift_fix.py`), модель — градиентный
бустинг. Значения Шепли считаются точным разложением по деревьям на двух
выборках: ячейки листа и ячейки юго-запада. Для каждого признака выводится
средний модуль вклада (сила) и знак связи вклада со значением признака
(направление: больше значение — выше или ниже балл).

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet,
      criterial_reconstruct_multisheet.parquet, сетка листа prognoz.pgrid
ВЫХОД: outputs/03_признаковое_пространство/shap_feature_contrib.png,
       outputs/metrics/shap_feature_contrib.log и .csv
Запуск из корня: python -X utf8 -m experiments.shap_feature_contrib
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, outputs, spectral_mask)
from experiments.branch5_alignment import align  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
SAMPLE = 4000            # ячеек в выборке для разложения
TERRAIN = ("dem_", "relief2_")
GROUP = {"dem": "рельеф", "relief2": "формы рельефа", "lin": "линеаменты",
         "s2": "спектры Sentinel-2", "спектр": "служебные"}
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit_sheet = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    wide = pd.read_parquet(config.PROCESSED_DIR / "wide_v1_archive"
                           / "dataset_wide.parquet")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)
    in_box = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
              (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    _, idx = cKDTree(np.column_stack([wx[in_box], wy[in_box]])).query(
        np.column_stack([cx, cy]), k=1)
    w_in = wide.loc[in_box].reset_index(drop=True)
    pair_w = w_in.loc[idx, cols].to_numpy(dtype=float)[valid]
    pair_s = feat[cols].to_numpy(dtype=float)[valid]
    wide_al = pd.DataFrame(align(pair_w, pair_s, wide[cols].to_numpy(dtype=float)),
                           columns=cols)
    S_df, W_df = spectral_mask.apply(feat[cols]), spectral_mask.apply(wide_al)
    names = list(S_df.columns)
    Xs, Xw = S_df.to_numpy(float), W_df.to_numpy(float)

    # область и зоны на широкой сетке
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    dist, _ = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                       ref["y"].to_numpy()[rmsk]])).query(
        np.column_stack([wx, wy]))
    common = dist <= 375.0
    zone = zone_of(wx, wy)
    in_sheet = common & (zone == "лист")
    sw = common & ~in_sheet

    # нормировка рельефа по зонам — как в рабочей версии карты
    for n in [c for c in names if c.startswith(TERRAIN)]:
        j = names.index(n)
        v = Xs[:, j]
        Xs[:, j] = (v - np.nanmean(v[valid])) / (np.nanstd(v[valid]) or 1.0)
        for sel in (in_sheet, sw):
            u = Xw[:, j]
            Xw[sel, j] = (u[sel] - np.nanmean(u[sel])) / (np.nanstd(u[sel]) or 1.0)

    crit_f = np.where(np.isfinite(crit_sheet), crit_sheet,
                      np.nanmin(crit_sheet[pool]))
    y = (crit_f >= np.quantile(crit_f[pool], 1 - AREA)).astype(int)
    model = make_model().fit(Xs[pool], y[pool])
    log("модель обучена: ячеек листа %d, целевых %d, признаков %d"
        % (pool.size, int(y[pool].sum()), len(names)))

    expl = shap.TreeExplainer(model)
    rng = np.random.default_rng(config.CRIT_SEED)
    rows = []
    sets = {}
    for tag, X, sel in (("лист", Xs, pool),
                        ("юго-запад", Xw, np.flatnonzero(sw))):
        take = rng.choice(sel, min(SAMPLE, sel.size), replace=False)
        sv = np.asarray(expl.shap_values(X[take]))
        if sv.ndim == 3:                      # (ячейки, признаки, классы)
            sv = sv[:, :, 1]
        strength = np.abs(sv).mean(axis=0)
        direction = []
        for j in range(len(names)):
            a, b = X[take][:, j], sv[:, j]
            ok = np.isfinite(a) & np.isfinite(b)      # спектры погашены маской
            direction.append(np.corrcoef(a[ok], b[ok])[0, 1]
                             if ok.sum() > 50 and np.std(a[ok]) > 1e-9 else np.nan)
        direction = np.array(direction)
        sets[tag] = (strength, direction)
        log("%s: разложено %d ячеек, суммарный вклад %.3f"
            % (tag, len(take), strength.sum()))
        for j, n in enumerate(names):
            rows.append({"площадь": tag, "признак": n,
                         "группа": GROUP.get(n.split("_")[0], "прочее"),
                         "сила": float(strength[j]),
                         "доля_%": float(100 * strength[j] / strength.sum()),
                         "направление": float(direction[j])})
    tab = pd.DataFrame(rows)

    log("")
    log("ВКЛАД ПО ГРУППАМ ПРИЗНАКОВ, доля от суммарного (%)")
    piv = tab.pivot_table(index="группа", columns="площадь", values="доля_%",
                          aggfunc="sum").round(1)
    for g, r in piv.iterrows():
        log("   %-20s лист %5.1f   юго-запад %5.1f" % (g, r.get("лист", 0),
                                                       r.get("юго-запад", 0)))
    log("")
    log("ДЕСЯТЬ СИЛЬНЕЙШИХ ПРИЗНАКОВ НА ЛИСТЕ (направление: + больше значение,"
        " выше балл)")
    top = tab[tab["площадь"] == "лист"].nlargest(10, "сила")
    for _, r in top.iterrows():
        other = tab[(tab["площадь"] == "юго-запад")
                    & (tab["признак"] == r["признак"])]["доля_%"].iloc[0]
        log("   %-18s лист %5.1f%%  юго-запад %5.1f%%  направление %+.2f"
            % (r["признак"], r["доля_%"], other, r["направление"]))

    # --- рисунок
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)
    for axi, tag in zip(axes, ("лист", "юго-запад")):
        sub = tab[tab["площадь"] == tag].nlargest(15, "сила").iloc[::-1]
        colors = ["#a3122b" if d > 0 else "#2b6cb0" if d < 0 else "#999999"
                  for d in sub["направление"]]
        axi.barh(sub["признак"], sub["доля_%"], color=colors)
        axi.set_xlabel("доля в суммарном вкладе, %")
        axi.set_title("Вклад признаков: %s" % tag, fontsize=12)
        axi.grid(axis="x", alpha=0.3)
    handles = [matplotlib.patches.Patch(color="#a3122b",
                                        label="больше значение — выше балл"),
               matplotlib.patches.Patch(color="#2b6cb0",
                                        label="больше значение — ниже балл")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle("Разложение балла модели по признакам (значения Шепли)",
                 fontsize=13)
    path = outputs.png("shap_feature_contrib.png")
    mapstyle.save_fig(fig, path, dpi=170)
    log("рисунок: %s" % path)

    out = ROOT / "outputs" / "metrics"
    tab.to_csv(out / "shap_feature_contrib.csv", index=False,
               encoding="utf-8-sig")
    (out / "shap_feature_contrib.log").write_text("\n".join(LOG),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
