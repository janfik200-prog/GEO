"""Аналоговый поиск: где на юго-западе обстановка, как у известных проявлений.

ЗАЧЕМ. На юго-западе нет ни одной точки заверки (ближайшая в 36-61 км), а
критериальную формулу там по дистанционным данным воспроизвести нельзя
(обучение прямо на юго-западе берёт её цель на уровне чистой географии).
Значит задачу надо ставить иначе: не «повтори балл формулы», а «найди ячейки,
похожие на те, где проявления уже известны». Разметки на месте для этого не
нужно — нужен эталон с листа.

ЧТО СЧИТАЕТСЯ.
1. Эталон: ячейки несмещённых точек заверки — пробы по коренным и проявления
   U/TR. Шлихи и литопробы в рыхлых в эталон не входят: они указывают на
   источник выше по потоку, а не на само проявление.
2. Два пространства сравнения, потому что маска обнажённости гасит спектры под
   растительностью и оставляет полный набор лишь на трети площади:
   * «все признаки» — 38 столбцов, только обнажённые ячейки;
   * «рельеф и линеаменты» — 15 столбцов, вся площадь без пропусков.
   В обоих случаях признаки нормируются внутри своей зоны (лист и смежная
   площадь отдельно), затем берутся главные компоненты.
3. Сходство ячейки — минус среднее расстояние до k ближайших эталонных ячеек.
   Рядом считается новизна: расстояние до облака листа, чтобы отличить
   «похоже» от «не с чем сравнивать».
4. Проверка без разметки на месте: исключение по одной точке — сходство
   пересчитывается без самой эталонной ячейки, и смотрится, в какой процент
   площади она попадает. Плюс устойчивость к выбору k и к половине эталона.
5. Сравнение с критериальной картой на юго-западе и состав найденных ядер.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet,
      data/processed/criterial_reconstruct_multisheet.parquet
ВЫХОД: outputs/07_перенос_на_смежные/analog_search.png,
       outputs/metrics/analog_search.log и .csv,
       data/processed/analog_search.parquet
Запуск из корня: python -m experiments.analog_search
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import label  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, criterial_target,  # noqa: E402
                 mapstyle, neighbourhood as nb, outputs, spectral_mask)
from experiments.branch5_alignment import align  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
N_PC = 8
K_NN = 3
MIN_CELLS = 20
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def similarity(Z: np.ndarray, ref_idx: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(Z[ref_idx])
    kk = min(k, len(ref_idx))
    d, _ = tree.query(Z, k=kk)
    return -(d if kk == 1 else d.mean(axis=1))


def main() -> None:
    shape = (ARCH_META.prf, ARCH_META.pic)
    km = 1000.0 / ARCH_META.dx
    theta = nb.theta_from_azimuth(NW_AZIMUTH)

    meta, _ = criterial_target.load_prognoz_grid()
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
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
    Xdf = spectral_mask.apply(wide_al)
    names = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)

    ref_tab = pd.read_parquet(config.PROCESSED_DIR
                              / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref_tab["в_маске"].to_numpy(bool)
    dist, ridx = cKDTree(np.column_stack([ref_tab["x"].to_numpy()[rmsk],
                                          ref_tab["y"].to_numpy()[rmsk]])).query(
        np.column_stack([wx, wy]))
    common = dist <= 375.0
    crit = np.full(wx.size, np.nan)
    crit[common] = -ref_tab["воссоздание_вся_область"].to_numpy()[rmsk][ridx[common]]
    zone = zone_of(wx, wy)
    in_sheet = common & (zone == "лист")
    sw = common & (zone == "юго-запад")
    crit_top = np.zeros(wx.size, bool)
    crit_top[sw] = crit[sw] >= np.quantile(crit[sw], 1 - AREA)
    log("область %d ячеек: лист %d, юго-запад %d"
        % (common.sum(), in_sheet.sum(), sw.sum()))

    pts = assessment.load_verification_points(meta)
    unb = pts["unbiased_strict"].to_numpy()
    px, py = cx[pts["cell"].to_numpy()][unb], cy[pts["cell"].to_numpy()][unb]
    codes = pts["code"].to_numpy()[unb]
    log("несмещённых точек всего %d: по коренным %d, U/TR %d"
        % (unb.sum(), int((codes == 2).sum()), int((codes == 4).sum())))

    variants = {
        "все признаки (обнажённые ячейки)": names,
        "рельеф и линеаменты (вся площадь)":
            [c for c in names if not c.startswith("s2_") and c != "спектр_измерен"],
    }
    rows, maps, novs = [], {}, {}

    for tag, use in variants.items():
        jj = [names.index(c) for c in use]
        Z = np.full((wx.size, len(use)), np.nan)
        for sel in (in_sheet, common & ~in_sheet):
            mu = np.nanmean(X[sel][:, jj], axis=0)
            sd = np.nanstd(X[sel][:, jj], axis=0)
            sd[sd < 1e-9] = 1.0
            Z[sel] = (X[sel][:, jj] - mu) / sd
        ok = common & np.isfinite(Z).all(axis=1)
        pca = PCA(n_components=min(N_PC, len(use)), random_state=0).fit(Z[ok])
        P = pca.transform(Z[ok]) / np.sqrt(pca.explained_variance_)
        log("")
        log("=== %s: признаков %d, ячеек с полным набором %d (%.0f%% области), "
            "компонент %d, объяснено %.0f%% разброса"
            % (tag, len(use), ok.sum(), 100 * ok.sum() / common.sum(),
               P.shape[1], 100 * pca.explained_variance_ratio_.sum()))

        d_ref, i_ref = cKDTree(np.column_stack([wx[ok], wy[ok]])).query(
            np.column_stack([px, py]))
        keep = d_ref <= 375.0
        ref_loc = np.unique(i_ref[keep])
        log("   эталонных ячеек %d (точек в работе %d из %d)"
            % (len(ref_loc), int(keep.sum()), int(unb.sum())))

        sim_ok = similarity(P, ref_loc, K_NN)
        sim = np.full(wx.size, np.nan)
        sim[ok] = sim_ok
        base_fill = np.nanmin(sim_ok)
        sim_sm = np.where(common, nb.aniso_smooth(
            np.nan_to_num(sim, nan=base_fill), shape, ALONG_KM * km,
            ACROSS_KM * km, theta), np.nan)
        maps[tag] = sim_sm

        sheet_loc = np.flatnonzero(in_sheet[ok])
        d_nov, _ = cKDTree(P[sheet_loc]).query(P, k=10)
        nov = np.full(wx.size, np.nan)
        nov[ok] = d_nov.mean(axis=1)
        novs[tag] = nov
        log("   новизна: лист %.2f, юго-запад %.2f (медианы)"
            % (np.nanmedian(nov[in_sheet]), np.nanmedian(nov[sw])))

        rx, ry = wx[ok][ref_loc], wy[ok][ref_loc]
        for gap_km, tag2 in ((0.0, "исключение самой ячейки"),
                             (15.0, "исключение округи 15 км")):
            ranks = []
            for j, c in enumerate(ref_loc):
                far = np.hypot(rx - rx[j], ry - ry[j]) > gap_km * 1000
                rest = ref_loc[far] if gap_km > 0 else ref_loc[ref_loc != c]
                if len(rest) == 0:
                    continue
                s = similarity(P, rest, K_NN)
                ranks.append(100 * (s >= s[c]).mean())
            ranks = np.array(ranks)
            log("   %s: медиана ранга — верхние %.0f%% площади, в верхних 10%% "
                "%d из %d (у случайной карты медиана 50%%)"
                % (tag2, np.median(ranks), int((ranks <= 10).sum()), len(ranks)))

        top_a = sim_sm[common] >= np.nanquantile(sim_sm[common], 1 - AREA)
        for k in (1, 5, 10):
            s = similarity(P, ref_loc, k)
            full = np.full(wx.size, np.nan)
            full[ok] = s
            sm = nb.aniso_smooth(np.nan_to_num(full, nan=np.nanmin(s)), shape,
                                 ALONG_KM * km, ACROSS_KM * km, theta)
            top_b = sm[common] >= np.nanquantile(sm[common], 1 - AREA)
            log("   k=%2d: совпадение верхних 10%% с рабочим k=%d — %.0f%%"
                % (k, K_NN, 100 * (top_a & top_b).sum() / top_a.sum()))
        halves = []
        for seed in range(5):
            sub = np.random.default_rng(seed).choice(ref_loc, max(len(ref_loc) // 2, 1),
                                                     replace=False)
            s = similarity(P, sub, K_NN)
            full = np.full(wx.size, np.nan)
            full[ok] = s
            sm = nb.aniso_smooth(np.nan_to_num(full, nan=np.nanmin(s)), shape,
                                 ALONG_KM * km, ACROSS_KM * km, theta)
            top_b = sm[common] >= np.nanquantile(sm[common], 1 - AREA)
            halves.append((top_a & top_b).sum() / top_a.sum())
        log("   половина эталона (5 попыток): совпадение верхних 10%% %.0f±%.0f%%"
            % (100 * np.mean(halves), 100 * np.std(halves)))

        top = np.zeros(wx.size, bool)
        top[common] = top_a
        log("   верхние 10%% сходства: лист %.0f кв.км, юго-запад %.0f кв.км"
            % (0.25 * (top & in_sheet).sum(), 0.25 * (top & sw).sum()))
        if (top & sw).sum():
            log("   на юго-западе аналоги совпадают с критериальными верхними "
                "10%% на %.0f%% (случайный уровень 10%%)"
                % (100 * (top & sw & crit_top).sum() / (top & sw).sum()))

        lab, n = label((top & sw).reshape(shape), structure=np.ones((3, 3)))
        lab = lab.ravel()
        big = sorted([i for i in range(1, n + 1) if (lab == i).sum() >= MIN_CELLS],
                     key=lambda q: -(lab == q).sum())
        log("   ядер на юго-западе: %d (от 5 кв.км)" % len(big))
        for j, i in enumerate(big, 1):
            m = lab == i
            log("      ядро %d: %.0f кв.км, центр X %.0f Y %.0f, новизна %.2f, "
                "совпадение с критериальным %.0f%%"
                % (j, 0.25 * m.sum(), wx[m].mean() / 1000, wy[m].mean() / 1000,
                   np.nanmedian(nov[m]), 100 * crit_top[m].mean()))
            rows.append({"набор": tag, "ядро": j, "кв.км": 0.25 * m.sum(),
                         "X_км": wx[m].mean() / 1000, "Y_км": wy[m].mean() / 1000,
                         "новизна": float(np.nanmedian(nov[m])),
                         "совпадение_с_критериальным": float(crit_top[m].mean())})

    # согласие двух версий между собой
    a, b = maps[list(variants)[0]], maps[list(variants)[1]]
    sel = common & np.isfinite(a) & np.isfinite(b)
    ta = a[sel] >= np.nanquantile(a[sel], 1 - AREA)
    tb = b[sel] >= np.nanquantile(b[sel], 1 - AREA)
    log("")
    log("две версии между собой: совпадение верхних 10%% %.0f%%"
        % (100 * (ta & tb).sum() / ta.sum()))

    # --- карта
    fig, axes = plt.subplots(1, 3, figsize=(19, 9.8), constrained_layout=True)
    ext = (wx.min() - 250, wx.max() + 250, wy.min() - 250, wy.max() + 250)
    for axi, (tag, sim_sm) in zip(axes, maps.items()):
        cuts = np.nanpercentile(sim_sm[common], mapstyle.CLASS_EDGES[1:-1])
        img = np.where(common & np.isfinite(sim_sm),
                       np.digitize(sim_sm, cuts).astype(float), np.nan)
        axi.imshow(np.ma.masked_invalid(img.reshape(shape)), origin="upper",
                   extent=ext, interpolation="nearest", vmin=-0.5, vmax=4.5,
                   cmap=matplotlib.colors.ListedColormap(mapstyle.CLASS_COLORS))
        axi.set_title("Сходство с проявлениями:\n%s" % tag, fontsize=11)
    nv = novs[list(variants)[1]]
    im = axes[2].imshow(np.ma.masked_invalid(np.where(common, nv, np.nan).reshape(shape)),
                        origin="upper", extent=ext, cmap="viridis_r",
                        interpolation="nearest")
    axes[2].set_title("Новизна: насколько ячейка далека\nот обстановки листа",
                      fontsize=11)
    fig.colorbar(im, ax=axes[2], shrink=0.55, label="расстояние в признаках")
    for axi in axes:
        axi.scatter(px, py, s=14, c="#101010", marker="^", zorder=3)
        axi.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.set_aspect("equal")
    handles = [matplotlib.patches.Patch(facecolor=c, edgecolor="#777777", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               fontsize=10, title="Классы сходства (доли площади области)")
    fig.suptitle("Аналоговый поиск: эталон — известные проявления основного листа",
                 fontsize=13)
    path = outputs.png("analog_search.png")
    mapstyle.save_fig(fig, path, dpi=170)
    log("карта: %s" % path)

    saved = wide[["row", "col", "x", "y"]].copy()
    saved["в_области"] = common
    for tag, sim_sm in maps.items():
        saved["сходство: %s" % tag] = sim_sm.astype("float32")
    saved["новизна"] = novs[list(variants)[1]].astype("float32")
    saved.to_parquet(config.PROCESSED_DIR / "analog_search.parquet", index=False)
    out = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out / "analog_search.csv", index=False,
                              encoding="utf-8-sig")
    (out / "analog_search.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
