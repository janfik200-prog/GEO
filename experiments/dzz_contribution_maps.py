"""Карта «дистанционка + геология» и доля дистанционки в ней.

ЗАЧЕМ. Геологические слои формулы разрешены, и сразу встал вопрос: что в
получившейся карте от дистанционного зондирования, а что от геологии. Здесь
это меряется тремя независимыми способами на двух мишенях сразу.

ДВЕ МИШЕНИ, потому что ответ у них разный:
  * верхние 10% критериального балла — мишень замкнута на геологические слои
    формулы (балл из них и собран), поэтому доля дистанционки здесь занижена
    по построению;
  * факт находки золота в пробе (47 ячеек) — мишень независимая, но меток
    мало, поэтому числа шумные.
Честный вывод складывается только из обеих.

ТРИ СПОСОБА ИЗМЕРИТЬ ВКЛАД.
  1. Доли Шепли — сколько процентов силы решения приходится на дистанционные
     признаки против геологических.
  2. Слепая метрика по полосам с разрывом 15 км: на сколько набор с
     дистанционкой лучше набора из одной геологии.
  3. Сдвиг карты: ранговая связь карты «геология» и карты «геология +
     дистанционка» и доля ячеек верхних 10%, которые дистанционка поменяла.

ВХОД: сетка листа, dataset_v5_rebuilt.parquet, слой опробования
ВЫХОД: outputs/06_прогнозные_карты/dzz_contribution_maps.png,
       outputs/metrics/dzz_contribution_maps.log и .csv
Запуск из корня: python -X utf8 -m experiments.dzz_contribution_maps
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
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, neighbourhood as nb, outputs, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.gold_points_branches import fill, gold_cells, point_folds  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402
from experiments.shap_two_branches import GEOLOGY  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
SAMPLE = 4000
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    shape = (meta.prf, meta.pic)
    km = 1000.0 / meta.dx
    smooth = lambda v: nb.aniso_smooth(np.nan_to_num(v), shape, ALONG_KM * km,
                                       ACROSS_KM * km,
                                       nb.theta_from_azimuth(NW_AZIMUTH))

    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    S = spectral_mask.apply(feat[dzz])
    names_d = list(S.columns)
    Xa = fill(S.to_numpy(float)[pool])
    geo = [c for c in GEOLOGY if c in raw.columns]
    Xg = fill(raw[geo].to_numpy(float)[pool])
    Xb = np.column_stack([Xa, Xg])
    names_b = names_d + geo

    idx, _ = gold_cells(meta)
    y_pts = np.zeros(cx.size, int)
    y_pts[idx] = 1
    y_crit = (crit >= np.nanquantile(crit[pool], 1 - AREA)).astype(int)
    log("ячеек %d | дистанционных признаков %d, геологических %d"
        % (pool.size, len(names_d), len(geo)))

    rng = np.random.default_rng(config.CRIT_SEED)
    rows, maps = [], {}
    targets = (("критериальный балл", y_crit[pool],
                strip_folds(cx[pool], cy[pool], y_crit[pool], GAP_M)),
               ("находки золота", y_pts[pool],
                point_folds(cx[pool], cy[pool], y_pts[pool])))
    for tag, y, folds in targets:
        log("")
        log("=== мишень: %s (положительных ячеек %d, полос %d) ==="
            % (tag, int(y.sum()), len(folds)))

        # 1. слепая метрика
        au = {}
        for nm, X in (("только геология", Xg), ("только дистанционка", Xa),
                      ("геология + дистанционка", Xb)):
            v = [roc_auc_score(y[te], make_model().fit(X[tr], y[tr])
                               .predict_proba(X[te])[:, 1]) for tr, te in folds]
            au[nm] = (float(np.mean(v)), float(np.std(v)))
            log("   %-26s площадь под кривой %.3f ± %.3f" % (nm, *au[nm]))
        gain = au["геология + дистанционка"][0] - au["только геология"][0]
        log("   прибавка дистанционки к геологии: %+.3f" % gain)

        # 2. доли Шепли
        model_b = make_model().fit(Xb, y)
        take = rng.choice(np.arange(pool.size), min(SAMPLE, pool.size),
                          replace=False)
        sv = np.asarray(shap.TreeExplainer(model_b).shap_values(Xb[take]))
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        strength = np.abs(sv).mean(axis=0)
        share_d = 100 * strength[:len(names_d)].sum() / strength.sum()
        log("   доли Шепли: дистанционка %.1f%%, геология %.1f%%"
            % (share_d, 100 - share_d))
        lead = [names_b[j] for j in np.argsort(strength)[::-1][:3]]
        lead_d = [names_b[j] for j in np.argsort(strength)[::-1]
                  if names_b[j] in names_d][:3]
        log("   ведущие признаки всего: %s" % ", ".join(lead))
        log("   ведущие среди дистанционных: %s" % ", ".join(lead_d))

        # 3. сдвиг карты
        pg = np.full(cx.size, np.nan)
        pb = np.full(cx.size, np.nan)
        pg[pool] = make_model().fit(Xg, y).predict_proba(Xg)[:, 1]
        pb[pool] = model_b.predict_proba(Xb)[:, 1]
        pg, pb = smooth(pg), smooth(pb)
        rho = sps.spearmanr(pg[pool], pb[pool]).statistic
        tg = pg >= np.quantile(pg[pool], 1 - AREA)
        tb = pb >= np.quantile(pb[pool], 1 - AREA)
        both = (tg & tb & valid).sum()
        moved = (tb & valid).sum() - both
        log("   карты: ранговая связь %+.3f; из верхних 10%% дистанционка "
            "перенесла %d ячеек (%.0f%% площади ядер, %.0f км²)"
            % (rho, moved, 100 * moved / max((tb & valid).sum(), 1),
               moved * 0.25))
        maps[tag] = (pg, pb, tg, tb)
        rows.append({"мишень": tag, "геология": au["только геология"][0],
                     "дистанционка": au["только дистанционка"][0],
                     "вместе": au["геология + дистанционка"][0],
                     "прибавка": gain, "доля_шепли_дзз_%": share_d,
                     "ранговая_карт": rho, "перенесено_ячеек": int(moved)})

    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics"
                              / "dzz_contribution_maps.csv", index=False,
                              encoding="utf-8-sig")

    # --- рисунок: по строке на мишень, три панели
    ext = (cx.min() - 250, cx.max() + 250, cy.min() - 250, cy.max() + 250)
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 11.6),
                             constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(mapstyle.CLASS_COLORS)
    for r, (tag, _, _) in enumerate(targets):
        pg, pb, tg, tb = maps[tag]
        for c, (ttl, v) in enumerate((("только геология формулы", pg),
                                      ("геология + дистанционка", pb))):
            axi = axes[r, c]
            vv = np.where(valid, v, np.nan)
            cuts = np.nanpercentile(vv[valid], mapstyle.CLASS_EDGES[1:-1])
            img = np.where(valid, np.digitize(vv, cuts).astype(float), np.nan)
            axi.imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                       origin="upper", interpolation="nearest", vmin=-0.5,
                       vmax=4.5, cmap=cmap)
            axi.set_title("%s\n%s" % (tag, ttl), fontsize=10.5)
        axi = axes[r, 2]
        diff = np.where(valid, tb.astype(float) - tg.astype(float), np.nan)
        axi.imshow(np.ma.masked_invalid(diff.reshape(shape)), extent=ext,
                   origin="upper", interpolation="nearest", vmin=-1, vmax=1,
                   cmap="coolwarm")
        axi.set_title("%s\nчто изменила дистанционка в верхних 10%%\n"
                      "красное — добавила, синее — убрала" % tag, fontsize=10.5)
        if r == 1:
            axi.scatter(cx[idx], cy[idx], s=14, facecolor="none",
                        edgecolor="#102a43", linewidths=0.8)
        for axi in axes[r]:
            axi.set_xlabel("X, км")
            axi.set_ylabel("Y, км")
            axi.xaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
            axi.yaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
            axi.set_aspect("equal")
    fig.suptitle("Вклад дистанционного зондирования: верхний ряд — мишень "
                 "критериального балла, нижний — находки золота", fontsize=12)
    path = outputs.png("dzz_contribution_maps.png")
    mapstyle.save_fig(fig, path, dpi=150)
    log("")
    log("карта: %s" % path)
    (ROOT / "outputs" / "metrics" / "dzz_contribution_maps.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
