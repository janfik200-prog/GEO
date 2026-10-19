"""Сдвиг рельефа между территориями: лечится ли перенос нормировкой.

ЗАЧЕМ. Проверка 04.10.2026 (`coverage_clean_transfer.py`) показала, что провал
переноса за рамку листа даёт не край съёмки и не признаки покрытия, а сдвиг
рельефа: смежная площадь расчленённее листа (dem_relief_1km +1.25 сигмы,
dem_slope +1.03, dem_incision +0.81, dem_elev +0.71), а модель на листе выучила
«перспективно там, где выположено». В итоге максимум механически уезжает на
западный фланг, где расчленение ближе к листовому (связь балла с координатой X
-0.30). Здесь проверяются лекарства.

ЧТО СЧИТАЕТСЯ. Четыре набора признаков на одной цепочке рабочей точки
(приведение широкой сборки к шкале листа, маска обнажённости, бустинг,
сглаживание вытянутым ядром):
  A — как есть (опора);
  B — рельеф нормирован по территории: лист по статистикам листа, широкая
      сборка по статистикам всей области расчёта (без шва на границе листа);
  C — рельеф нормирован по зонам: часть широкой сборки внутри габаритов листа
      по статистикам листа, смежная часть по своим (сдвиг снимается полностью,
      но на границе появляется шов);
  D — контроль: рельефные признаки убраны целиком (верхняя граница того, что
      может стоить отказ от рельефа).
Метрики: полосовая проверка на листе с разрывом 15 км; согласие с НАШИМ
воссозданием критериальной формулы по зонам; заверка по точкам; и две меры
артефакта — связь балла с координатой X на смежной площади и расчленённость
рельефа в верхних 10% вне листа.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet,
      data/processed/criterial_reconstruct_multisheet.parquet
ВЫХОД: outputs/metrics/terrain_drift_fix.log и .csv,
       outputs/07_перенос_на_смежные/terrain_drift_fix.png
Запуск из корня: python -m experiments.terrain_drift_fix
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, mapstyle, neighbourhood as nb, outputs,
                 spectral_mask)
from experiments.branch5_alignment import align  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (make_model, poly_design,  # noqa: E402
                                               strip_folds)
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
TERRAIN = ("dem_", "relief2_")      # рельефные признаки, дающие сдвиг
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def zscore(v: np.ndarray, cells: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(v[cells]), np.nanstd(v[cells])
    return (v - mu) / sd if sd > 1e-9 else v * 0.0


def agree(ours: np.ndarray, ref: np.ndarray, x: np.ndarray, y: np.ndarray,
          cells: np.ndarray) -> dict:
    a, b = ours[cells], ref[cells]
    D = poly_design(x[cells], y[cells])
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    ta, tb = a >= np.quantile(a, 1 - AREA), b >= np.quantile(b, 1 - AREA)
    sa, sb = ra >= np.quantile(ra, 1 - AREA), rb >= np.quantile(rb, 1 - AREA)
    return {"совпадение": float((ta & tb).sum() / tb.sum()),
            "без_наклона": float((sa & sb).sum() / sb.sum()),
            "ранговая": float(sps.spearmanr(a, b).statistic)}


def main() -> None:
    shape = (ARCH_META.prf, ARCH_META.pic)

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
    tree = cKDTree(np.column_stack([wx[in_box], wy[in_box]]))
    _, idx = tree.query(np.column_stack([cx, cy]), k=1)
    w_in = wide.loc[in_box].reset_index(drop=True)

    raw_sheet = feat[cols]
    pair_w = w_in.loc[idx, cols].to_numpy(dtype=float)[valid]
    pair_s = raw_sheet.to_numpy(dtype=float)[valid]
    wide_al = pd.DataFrame(align(pair_w, pair_s, wide[cols].to_numpy(dtype=float)),
                           columns=cols)
    S_df = spectral_mask.apply(raw_sheet)
    W_df = spectral_mask.apply(wide_al)
    names = list(S_df.columns)
    terr = [n for n in names if n.startswith(TERRAIN)]
    log("признаков %d, из них рельефных %d: %s"
        % (len(names), len(terr), ", ".join(terr)))

    crit_f = np.where(np.isfinite(crit_sheet), crit_sheet,
                      np.nanmin(crit_sheet[pool]))
    y = (crit_f >= np.quantile(crit_f[pool], 1 - AREA)).astype(int)

    # эталон и зоны на широкой сетке
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    dist, ridx = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                          ref["y"].to_numpy()[rmsk]])).query(
        np.column_stack([wx, wy]))
    common = dist <= 375.0
    crit_w = np.full(wx.size, np.nan)
    crit_w[common] = -ref["воссоздание_вся_область"].to_numpy()[rmsk][ridx[common]]
    zone = zone_of(wx, wy)
    in_sheet = common & (zone == "лист")
    out = common & ~in_sheet
    log("область расчёта %d ячеек: лист %d, смежная %d"
        % (common.sum(), in_sheet.sum(), out.sum()))

    pts = assessment.load_verification_points(meta)
    pt_x, pt_y = cx[pts["cell"].to_numpy()], cy[pts["cell"].to_numpy()]
    unb = pts["unbiased_strict"].to_numpy()
    d_pt, i_pt = cKDTree(np.column_stack([wx[common], wy[common]])).query(
        np.column_stack([pt_x, pt_y]))
    on_area = d_pt <= 375.0
    pt_cell = np.flatnonzero(common)[i_pt]

    relief = W_df["dem_relief_1km"].to_numpy(float)

    # --- четыре набора
    Xs0, Xw0 = S_df.to_numpy(float), W_df.to_numpy(float)
    sets = {}
    sets["A: как есть"] = (names, Xs0.copy(), Xw0.copy())

    Xs_b, Xw_b = Xs0.copy(), Xw0.copy()
    for n in terr:
        j = names.index(n)
        Xs_b[:, j] = zscore(Xs_b[:, j], valid)
        Xw_b[:, j] = zscore(Xw_b[:, j], common)
    sets["B: рельеф по территории"] = (names, Xs_b, Xw_b)

    Xs_c, Xw_c = Xs0.copy(), Xw0.copy()
    for n in terr:
        j = names.index(n)
        Xs_c[:, j] = zscore(Xs_c[:, j], valid)
        col = Xw_c[:, j].copy()
        new = np.full_like(col, np.nan)
        new[in_sheet] = zscore(col, in_sheet)[in_sheet]
        new[out] = zscore(col, out)[out]
        rest = ~(in_sheet | out)
        new[rest] = zscore(col, np.ones(col.size, bool))[rest]
        Xw_c[:, j] = new
    sets["C: рельеф по зонам"] = (names, Xs_c, Xw_c)

    keep = [n for n in names if n not in terr]
    jk = [names.index(n) for n in keep]
    sets["D: без рельефа"] = (keep, Xs0[:, jk], Xw0[:, jk])

    km = 1000.0 / ARCH_META.dx
    rows, maps = [], {}
    folds = strip_folds(cx[pool], cy[pool], y[pool], GAP_M)

    for tag, (use, Xs, Xw) in sets.items():
        au = []
        for tr, te in folds:
            p = make_model().fit(Xs[pool][tr], y[pool][tr]).predict_proba(
                Xs[pool][te])[:, 1]
            au.append(roc_auc_score(y[pool][te], p))
        log("%-26s признаков %2d, полосовая проверка на листе ROC-AUC %.4f±%.3f"
            % (tag, len(use), np.mean(au), np.std(au)))

        score = make_model().fit(Xs[pool], y[pool]).predict_proba(Xw)[:, 1]
        score = nb.aniso_smooth(score, shape, ALONG_KM * km, ACROSS_KM * km,
                                nb.theta_from_azimuth(NW_AZIMUTH))
        maps[tag] = score
        row = {"набор": tag, "признаков": len(use), "roc_auc": float(np.mean(au)),
               "roc_std": float(np.std(au))}
        for zn, cells in (("вся область", common), ("лист", in_sheet),
                          ("смежная", out)):
            r = agree(score, crit_w, wx, wy, cells)
            row.update({"%s_совпадение" % zn: r["совпадение"],
                        "%s_без_наклона" % zn: r["без_наклона"],
                        "%s_ранговая" % zn: r["ранговая"]})
            log("   %-26s %-12s совпадение %.3f, без наклона %.3f, ранговая %+.3f"
                % (tag, zn, r["совпадение"], r["без_наклона"], r["ранговая"]))

        top = score >= np.quantile(score[common], 1 - AREA)
        for who, sel in (("все", on_area), ("несмещённые", on_area & unb)):
            hit = top[pt_cell[sel]].mean()
            row["заверка_%s" % who] = float(hit)
            log("   %-26s заверка %-12s точек %2d, в верхних 10%% %.0f%% (x%.1f)"
                % (tag, who, int(sel.sum()), 100 * hit, hit / AREA))

        top_out = out & (score >= np.quantile(score[out], 1 - AREA))
        rx = sps.spearmanr(score[out], wx[out]).statistic
        rel = np.nanmedian(relief[top_out]) - np.nanmedian(relief[out])
        row["связь_с_X"] = float(rx)
        row["рельеф_ядер_минус_фон"] = float(rel)
        log("   %-26s артефакт: связь балла с X %+.3f, расчленённость верхних "
            "10%% вне листа %+.1f м к фону" % (tag, rx, rel))
        rows.append(row)

    # --- карта: четыре версии верхних 10%
    fig, axes = plt.subplots(1, 4, figsize=(22, 9.6), constrained_layout=True)
    ext = (wx.min() - 250, wx.max() + 250, wy.min() - 250, wy.max() + 250)
    for axi, (tag, score) in zip(axes, maps.items()):
        img = np.full(wx.size, np.nan)
        img[common] = (score[common] >= np.quantile(score[common], 1 - AREA)) + 1.0
        axi.imshow(np.ma.masked_invalid(img.reshape(shape)), origin="upper",
                   extent=ext, interpolation="nearest", vmin=0.5, vmax=2.5,
                   cmap=matplotlib.colors.ListedColormap(["#e9e9ef", "#a3122b"]))
        axi.scatter(pt_x, pt_y, s=8, c="#2b6cb0", marker="^", zorder=3)
        axi.set_title(tag, fontsize=11)
        axi.set_xlabel("X, км")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Верхние 10% площади при разных способах обойти сдвиг рельефа",
                 fontsize=13)
    path = outputs.png("terrain_drift_fix.png")
    mapstyle.save_fig(fig, path, dpi=170)
    log("карта: %s" % path)

    saved = wide[["row", "col", "x", "y"]].copy()
    saved["в_области"] = common
    saved["в_габаритах_листа"] = in_sheet
    for tag, score in maps.items():
        saved[tag] = score.astype("float32")
    saved.to_parquet(config.PROCESSED_DIR / "terrain_drift_maps.parquet", index=False)
    log("карты сохранены: %s" % (config.PROCESSED_DIR / "terrain_drift_maps.parquet"))

    out_dir = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out_dir / "terrain_drift_fix.csv", index=False,
                              encoding="utf-8-sig")
    (out_dir / "terrain_drift_fix.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
