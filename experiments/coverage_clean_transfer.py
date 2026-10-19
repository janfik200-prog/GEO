"""Чистка признаков покрытия и краевого буфера: что это даёт переносу.

ЗАЧЕМ. Разбор ядер (`experiments/cores_anatomy.py`, 04.10.2026) показал: все
пять ядер нашего прогноза вне основного листа прижаты к западной кромке
области расчёта, не перекрываются критериальными ядрами ни на процент и
выделены сырыми каналами съёмки, а в одном ядре на самом листе в топ
отклонений вышли признаки ПОКРЫТИЯ — доля открытого грунта и отметка «спектр
измерен». То есть модель частично учит контуры съёмки, а не геологию. Здесь
проверяется первое лекарство: убрать признаки покрытия из обучения и отрезать
краевой буфер области.

ЧТО СЧИТАЕТСЯ. Два набора признаков — как есть и без признаков покрытия —
прогоняются одной и той же цепочкой рабочей точки (приведение широкой сборки
к шкале листа, маска обнажённости, бустинг, сглаживание вытянутым ядром):
1. Полосовая проверка на листе с разрывом 15 км: цена чистки на обученной
   территории.
2. Карта на всю область расчёта и согласие с НАШИМ воссозданием критериальной
   формулы (эталон с 04.10.2026) по зонам.
3. Заверка по точкам: доля точек в верхних 10% площади.
4. Кромочность: какая доля верхних 10% вне листа лежит в краевом буфере
   области. Это прямая мера артефакта, ради которого всё затевалось.
Каждая метрика считается на всей области и на области без краевого буфера.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet,
      data/processed/criterial_reconstruct_multisheet.parquet
ВЫХОД: outputs/metrics/coverage_clean_transfer.log и .csv,
       outputs/03_прогноз/coverage_clean_transfer.png
Запуск из корня: python -m experiments.coverage_clean_transfer
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
from scipy.ndimage import distance_transform_edt  # noqa: E402
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
BUFFER_KM = 5.0          # ширина краевого буфера области расчёта
COVERAGE = ("s2_bare_frac", "спектр_измерен")   # признаки покрытия, не геологии
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


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

    # --- лист: признаки и цель
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
    all_cols = list(S_df.columns)
    log("признаков после маски: %d; из них покрытия: %s"
        % (len(all_cols), ", ".join(c for c in COVERAGE if c in all_cols)))

    crit_f = np.where(np.isfinite(crit_sheet), crit_sheet,
                      np.nanmin(crit_sheet[pool]))
    y = (crit_f >= np.quantile(crit_f[pool], 1 - AREA)).astype(int)

    # --- эталон: наше воссоздание формулы, перенесённое на архивную сетку
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    rtree = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                     ref["y"].to_numpy()[rmsk]]))
    dist, ridx = rtree.query(np.column_stack([wx, wy]))
    common = dist <= 375.0
    crit_w = np.full(wx.size, np.nan)
    crit_w[common] = -ref["воссоздание_вся_область"].to_numpy()[rmsk][ridx[common]]
    zone = zone_of(wx, wy)
    in_sheet = wide["x"].to_numpy() * 0 == 0  # заполнится ниже
    in_sheet = (zone == "лист") & common

    # --- краевой буфер области расчёта
    grid_common = common.reshape(shape)
    edge_km = distance_transform_edt(grid_common, sampling=ARCH_META.dx / 1000.0)
    edge_km = edge_km.ravel()
    inner = common & (edge_km >= BUFFER_KM)
    log("область расчёта %d ячеек; вне буфера %.0f км остаётся %d (%.0f%%)"
        % (common.sum(), BUFFER_KM, inner.sum(), 100 * inner.sum() / common.sum()))

    # --- точки заверки
    pts = assessment.load_verification_points(meta)
    pt_x, pt_y = cx[pts["cell"].to_numpy()], cy[pts["cell"].to_numpy()]
    unb = pts["unbiased_strict"].to_numpy()
    ptree = cKDTree(np.column_stack([wx[common], wy[common]]))
    d_pt, i_pt = ptree.query(np.column_stack([pt_x, pt_y]))
    on_area = d_pt <= 375.0
    pt_cell = np.flatnonzero(common)[i_pt]

    sets = {"как есть (%d)" % len(all_cols): all_cols,
            "без покрытия (%d)" % len([c for c in all_cols if c not in COVERAGE]):
                [c for c in all_cols if c not in COVERAGE]}
    km = 1000.0 / ARCH_META.dx
    rows, maps = [], {}

    for tag, use in sets.items():
        Xs = S_df[use].to_numpy(dtype=float)
        Xw = W_df[use].to_numpy(dtype=float)

        # 1. полосовая проверка на листе
        folds = strip_folds(cx[pool], cy[pool], y[pool], GAP_M)
        au = []
        for tr, te in folds:
            p = make_model().fit(Xs[pool][tr], y[pool][tr]).predict_proba(
                Xs[pool][te])[:, 1]
            au.append(roc_auc_score(y[pool][te], p))
        log("%-22s полосовая проверка на листе: ROC-AUC %.4f±%.3f (%d фолдов)"
            % (tag, np.mean(au), np.std(au), len(folds)))

        # 2. карта на всю область
        score = make_model().fit(Xs[pool], y[pool]).predict_proba(Xw)[:, 1]
        score = nb.aniso_smooth(score, shape, ALONG_KM * km, ACROSS_KM * km,
                                nb.theta_from_azimuth(NW_AZIMUTH))
        maps[tag] = score
        for area_tag, base in (("вся область", common), ("без кромки", inner)):
            for zn, cells in (("вся площадь", base),
                              ("лист", base & in_sheet),
                              ("смежная", base & ~in_sheet & common)):
                if cells.sum() < 300:
                    continue
                r = agree(score, crit_w, wx, wy, cells)
                rows.append({"набор": tag, "площадь": area_tag, "зона": zn, **r})
                log("  %-22s %-12s %-10s совпадение %.3f, без наклона %.3f, "
                    "ранговая %+.3f" % (tag, area_tag, zn, r["совпадение"],
                                        r["без_наклона"], r["ранговая"]))

        # 3. заверка по точкам
        thr = np.quantile(score[common], 1 - AREA)
        top = score >= thr
        for who, sel in (("все точки", on_area), ("несмещённые", on_area & unb)):
            hit = top[pt_cell[sel]].mean()
            log("  %-22s заверка, %-12s точек %2d, в верхних 10%% %.0f%% "
                "(выигрыш x%.1f)" % (tag, who, int(sel.sum()), 100 * hit, hit / AREA))
            rows.append({"набор": tag, "площадь": "заверка", "зона": who,
                         "совпадение": float(hit), "без_наклона": np.nan,
                         "ранговая": np.nan})

        # 4. кромочность верхних 10% вне листа
        out = common & ~in_sheet
        top_out = out & (score >= np.quantile(score[out], 1 - AREA))
        share = (edge_km[top_out] < BUFFER_KM).mean()
        log("  %-22s кромочность: %.0f%% верхних 10%% вне листа лежит ближе "
            "%.0f км к границе области (фон %.0f%%)"
            % (tag, 100 * share, BUFFER_KM,
               100 * (edge_km[out] < BUFFER_KM).mean()))
        rows.append({"набор": tag, "площадь": "кромочность", "зона": "вне листа",
                     "совпадение": float(share), "без_наклона": np.nan,
                     "ранговая": np.nan})

    # --- карта: было и стало
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 10.4), constrained_layout=True)
    ext = (wx.min() - 250, wx.max() + 250, wy.min() - 250, wy.max() + 250)
    for axi, (tag, score) in zip(axes, maps.items()):
        top = np.zeros(wx.size)
        top[common] = (score[common] >= np.quantile(score[common], 1 - AREA)) + 1.0
        img = np.where(common, top, np.nan).reshape(shape)
        axi.imshow(np.ma.masked_invalid(img), origin="upper", extent=ext,
                   cmap=matplotlib.colors.ListedColormap(["#e9e9ef", "#a3122b"]),
                   vmin=0.5, vmax=2.5, interpolation="nearest")
        axi.contour(np.where(common, edge_km, np.nan).reshape(shape),
                    levels=[BUFFER_KM], colors="#2b6cb0", linewidths=1.0,
                    extent=ext, origin="upper")
        axi.scatter(pt_x, pt_y, s=10, c="#101010", marker="^", zorder=3)
        axi.set_title("Верхние 10%%: %s" % tag, fontsize=12)
        axi.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Чистка признаков покрытия: синяя линия — граница краевого "
                 "буфера %.0f км, треугольники — точки заверки" % BUFFER_KM,
                 fontsize=12)
    path = outputs.png("coverage_clean_transfer.png")
    mapstyle.save_fig(fig, path, dpi=180)
    log("карта: %s" % path)

    out_dir = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out_dir / "coverage_clean_transfer.csv",
                              index=False, encoding="utf-8-sig")
    (out_dir / "coverage_clean_transfer.log").write_text("\n".join(LOG),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
