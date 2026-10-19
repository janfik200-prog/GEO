"""Две ветки обучения — только дистанционка и дистанционка с геологией.

ЗАЧЕМ. 10.10.2026 разрешено использовать слои, из которых собрана
критериальная формула. Чтобы понять, что они дают, обучение разделено на две
ветки на одной и той же площади основного листа:
  A — только дистанционные признаки (как раньше);
  B — те же плюс геологические слои формулы (разломы северо-западные и
      северо-восточные, их плотность, дайки, коры выветривания, фации,
      палеодолины, узлы пересечения разломов, контакты, простирания).

ВАЖНО ПРИ ЧТЕНИИ. Цель обучения — верхние 10% критериального балла, а сам балл
собран ровно из этих геологических слоёв. Поэтому ветка B неизбежно
воспроизводит цель почти точно: её высокие показатели означают замыкание на
собственные входы формулы, а не геологическое открытие. Ветка нужна как мера
того, сколько в цели «геометрии формулы» и сколько остаётся дистанционке.

ЧТО СЧИТАЕТСЯ.
1. Полосовая проверка с разрывом 15 км для обеих веток — цена и выигрыш.
2. Значения Шепли для обеих веток: полный список вкладов всех признаков со
   знаком направления, плюс свод по группам.
3. Карты трёх ведущих признаков каждой ветки с контуром верхних 10% прогноза
   поверх — видно, совпадает ли рисунок признака с ядрами.

ВХОД: data/processed/dataset_v5_rebuilt.parquet (геология), сетка листа
ВЫХОД: outputs/03_признаковое_пространство/shap_two_branches_*.png,
       outputs/metrics/shap_two_branches.log и .csv
Запуск из корня: python -X utf8 -m experiments.shap_two_branches
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
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, neighbourhood as nb, outputs, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
SAMPLE = 4000
GEOLOGY = ["dist_tect1", "dist_tect2", "dens_tect", "dist_magm", "dens_magm",
           "dist_struct", "dist_facies", "dist_paleo", "geo2_fault_node",
           "geo2_contact", "geo2_facies2", "geo2_strike_cos", "geo2_strike_sin"]
RUS_GEO = {
    "dist_tect1": "расстояние до разломов СЗ",
    "dist_tect2": "расстояние до разломов СВ",
    "dens_tect": "плотность разломов",
    "dist_magm": "расстояние до даек",
    "dens_magm": "плотность даек",
    "dist_struct": "расстояние до кор выветривания",
    "dist_facies": "расстояние до благоприятных фаций",
    "dist_paleo": "расстояние до палеодолин",
    "geo2_fault_node": "расстояние до узлов пересечения разломов",
    "geo2_contact": "расстояние до геологических контактов",
    "geo2_facies2": "фациальная принадлежность",
    "geo2_strike_cos": "простирание структур, косинус",
    "geo2_strike_sin": "простирание структур, синус",
}
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def group_of(name: str) -> str:
    if name in GEOLOGY:
        return "геология формулы"
    pre = name.split("_")[0]
    return {"dem": "рельеф", "relief2": "формы рельефа", "lin": "линеаменты",
            "s2": "спектры"}.get(pre, "служебные")


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    S = spectral_mask.apply(feat[dzz])
    names_a = list(S.columns)
    Xa = S.to_numpy(float)
    geo = [c for c in GEOLOGY if c in raw.columns]
    Xg = raw[geo].to_numpy(float)
    names_b = names_a + geo
    Xb = np.column_stack([Xa, Xg])
    y = (crit >= np.nanquantile(crit[pool], 1 - AREA)).astype(int)
    log("ячеек листа %d, целевых %d; ветка A: %d признаков, ветка B: %d "
        "(из них геологических %d)"
        % (pool.size, int(y[pool].sum()), len(names_a), len(names_b), len(geo)))

    folds = strip_folds(cx[pool], cy[pool], y[pool], GAP_M)
    log("полос с разрывом %.0f км: %d" % (GAP_M / 1000, len(folds)))

    rows, models, samples = [], {}, {}
    rng = np.random.default_rng(config.CRIT_SEED)
    for tag, X, names in (("A: только дистанционка", Xa, names_a),
                          ("B: дистанционка + геология", Xb, names_b)):
        au = []
        for tr, te in folds:
            m = make_model().fit(X[pool][tr], y[pool][tr])
            au.append(roc_auc_score(y[pool][te], m.predict_proba(X[pool][te])[:, 1]))
        log("%-28s полосовая проверка: площадь под кривой %.3f ± %.3f"
            % (tag, np.mean(au), np.std(au)))
        model = make_model().fit(X[pool], y[pool])
        models[tag] = (model, X, names)
        take = rng.choice(pool, min(SAMPLE, pool.size), replace=False)
        sv = np.asarray(shap.TreeExplainer(model).shap_values(X[take]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        strength = np.abs(sv).mean(axis=0)
        for j, n in enumerate(names):
            a, b = X[take][:, j], sv[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            d = (np.corrcoef(a[ok], b[ok])[0, 1]
                 if ok.sum() > 50 and np.std(a[ok]) > 1e-9 else np.nan)
            rows.append({"ветка": tag, "признак": n, "группа": group_of(n),
                         "доля_%": 100 * strength[j] / strength.sum(),
                         "направление": d})
        samples[tag] = take
    tab = pd.DataFrame(rows)

    for tag in models:
        log("")
        log("=== %s — ВКЛАД ВСЕХ ПРИЗНАКОВ, %% от суммарного ===" % tag)
        sub = tab[tab["ветка"] == tag].sort_values("доля_%", ascending=False)
        for _, r in sub.iterrows():
            nm = RUS_GEO.get(r["признак"], "")
            log("   %-18s %5.2f%%  направление %+5.2f  %s"
                % (r["признак"], r["доля_%"], r["направление"], nm))
        log("   --- по группам: " + ", ".join(
            "%s %.1f%%" % (g, v) for g, v in
            sub.groupby("группа")["доля_%"].sum().sort_values(
                ascending=False).items()))

    # --- карты трёх ведущих признаков каждой ветки
    shape = (meta.prf, meta.pic)
    ext = (cx.min() - 250, cx.max() + 250, cy.min() - 250, cy.max() + 250)
    km = 1000.0 / meta.dx
    for tag, (model, X, names) in models.items():
        p = model.predict_proba(X)[:, 1]
        p = nb.aniso_smooth(np.nan_to_num(p), shape, ALONG_KM * km,
                            ACROSS_KM * km, nb.theta_from_azimuth(NW_AZIMUTH))
        top = (p >= np.quantile(p[valid], 1 - AREA)).astype(float)
        lead = tab[tab["ветка"] == tag].nlargest(3, "доля_%")["признак"].tolist()
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.4),
                                 constrained_layout=True)
        for axi, n in zip(axes, lead):
            v = X[:, names.index(n)].astype(float)
            vv = np.where(valid, v, np.nan)
            lo, hi = np.nanpercentile(vv[valid], [2, 98])
            im = axi.imshow(np.ma.masked_invalid(vv.reshape(shape)), extent=ext,
                            origin="upper", interpolation="nearest",
                            cmap="cividis", vmin=lo, vmax=hi)
            axi.contour(np.where(valid, top, 0).reshape(shape), levels=[0.5],
                        colors="#a3122b", linewidths=1.4, extent=ext,
                        origin="upper")
            fig.colorbar(im, ax=axi, shrink=0.6)
            share = tab[(tab["ветка"] == tag)
                        & (tab["признак"] == n)]["доля_%"].iloc[0]
            axi.set_title("%s — %.1f%% вклада\n%s" % (n, share,
                                                      RUS_GEO.get(n, "")),
                          fontsize=10.5)
            axi.set_xlabel("X, км")
            axi.set_ylabel("Y, км")
            axi.xaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
            axi.yaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
            axi.set_aspect("equal")
        fig.suptitle("%s: три ведущих признака; красный контур — верхние 10%% "
                     "прогноза" % tag, fontsize=12)
        stem = "shap_two_branches_%s" % ("dzz" if tag.startswith("A") else "geo")
        path = outputs.png(stem + ".png")
        mapstyle.save_fig(fig, path, dpi=150)
        log("карта: %s" % path)

    out = ROOT / "outputs" / "metrics"
    tab.to_csv(out / "shap_two_branches.csv", index=False, encoding="utf-8-sig")
    (out / "shap_two_branches.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
