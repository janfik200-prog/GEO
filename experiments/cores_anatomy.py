"""Анатомия ядер двух карт: критериальной (наше воссоздание) и прогнозной.

ЗАЧЕМ. 04.10.2026 решено: эталоном служит НАШЕ воссоздание критериального
метода, а не пересчёт организации (`by_crt`) — специалисты подтвердили выбор.
Дальше обе карты нужно разобрать геологически: не «метрика такая-то», а что
за объекты выделены, чем именно каждый выделен и как они соотносятся с
известными точками заверки. Этот разбор — вход для корректировки обучения.

ЧТО СЧИТАЕТСЯ.
1. Согласие нашего прогноза с новым эталоном (наше воссоздание на всей области
   расчёта) по зонам: совпадение верхних 10%, то же после снятия координатного
   наклона, ранговая корреляция.
2. Ядра каждой карты: верхние 10% площади, связные области от 5 кв.км.
   Для каждого ядра — площадь, центр, зона, попадание в габариты листа.
3. Чем ядро выделено:
   * у критериальной карты — вклад каждого из шести факторов формулы в отрыв
     ядра от фона (в долях суммарного отрыва);
   * у прогнозной — признаки с наибольшим отклонением от фона (в сигмах фона),
     с расшифровкой из словаря признаков.
4. Точки заверки: сколько внутри ядра, расстояние до ближайшей, их тип.
5. Пересечение ядер двух карт.

ВХОД: data/processed/criterial_reconstruct_multisheet.parquet,
      data/processed/forecast_arch_ядро5км.parquet,
      data/processed/wide_v1_archive/dataset_wide.parquet
ВЫХОД: outputs/02_критериальный_метод/cores_anatomy.png,
       outputs/metrics/cores_anatomy.log, outputs/metrics/cores_anatomy.csv
Запуск из корня: python -m experiments.cores_anatomy
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.ndimage import label  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, config, criterial_target, feature_dict,  # noqa: E402
                 mapstyle, outputs, spectral_mask)
from experiments.criterial_multisheet_compare import SCORE_COL, zone_of  # noqa: E402
from experiments.criterial_reconstruct_multisheet import (ROLE_LAYER,  # noqa: E402
                                                          criteria, load_layers)
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
MIN_CELLS = 20          # 5 кв.км — меньшие пятна не обсуждаем
CELL_KM2 = 0.25
RUS_ROLE = {"tect1": "тектонический СЗ", "tect2": "тектонический СВ",
            "facies": "литолого-фациальный", "paleo": "палеодолины",
            "struct": "коры выветривания", "magm": "дайки"}
POINT_KIND = {1: "литопроба в рыхлых", 2: "проба по коренным",
              4: "проявление U/TR", 5: "шлих Au"}
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def to_archive(values: dict, px: np.ndarray, py: np.ndarray,
               ax: np.ndarray, ay: np.ndarray) -> dict:
    """Перенос значений с критериальной сетки на архивную по ближайшей ячейке."""
    tree = cKDTree(np.column_stack([px, py]))
    dist, idx = tree.query(np.column_stack([ax, ay]))
    ok = dist <= 375.0
    out = {"есть": ok}
    for name, v in values.items():
        col = np.full(ax.size, np.nan)
        col[ok] = v[idx[ok]]
        out[name] = col
    return out


def cores(score: np.ndarray, area_mask: np.ndarray, shape: tuple) -> np.ndarray:
    """Нумерация связных ядер: верхние AREA площади, от MIN_CELLS ячеек."""
    thr = np.quantile(score[area_mask], 1 - AREA)
    top = np.zeros(score.size, bool)
    top[area_mask] = score[area_mask] >= thr
    lab, n = label(top.reshape(shape), structure=np.ones((3, 3)))
    lab = lab.ravel()
    keep = {i for i in range(1, n + 1) if (lab == i).sum() >= MIN_CELLS}
    out = np.zeros(score.size, int)
    for new, old in enumerate(sorted(keep, key=lambda i: -(lab == i).sum()), 1):
        out[lab == old] = new
    return out


def agreement(ours: np.ndarray, ref: np.ndarray, x: np.ndarray, y: np.ndarray,
              cells: np.ndarray) -> tuple:
    a, b = ours[cells], ref[cells]
    D = poly_design(x[cells], y[cells])
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    ta, tb = a >= np.quantile(a, 1 - AREA), b >= np.quantile(b, 1 - AREA)
    sa, sb = ra >= np.quantile(ra, 1 - AREA), rb >= np.quantile(rb, 1 - AREA)
    return ((ta & tb).sum() / tb.sum(), (sa & sb).sum() / sb.sum(),
            sps.spearmanr(a, b).statistic)


def main() -> None:
    shape = (ARCH_META.prf, ARCH_META.pic)

    # --- входные карты на архивной сетке
    fw = pd.read_parquet(config.PROCESSED_DIR / "forecast_arch_ядро5км.parquet")
    ax = fw["x"].to_numpy(float)
    ay = fw["y"].to_numpy(float)
    ours = fw[SCORE_COL].to_numpy(float)
    in_sheet = fw["в_габаритах_листа"].to_numpy(bool)

    meta, layers, msk, by_crt, px_all, py_all = load_layers()
    z, names = criteria(layers, msk)
    w = np.array([config.TAXONOMY_WEIGHTS[n] for n in names])
    contrib = {n: z[:, j] * w[j] for j, n in enumerate(names)}
    crit_raw = np.column_stack(list(contrib.values())).sum(axis=1)

    vals = {"крит": crit_raw[msk]}
    vals.update({"вклад_" + n: contrib[n][msk] for n in names})
    moved = to_archive(vals, px_all[msk], py_all[msk], ax, ay)
    common = moved["есть"]
    crit = -moved["крит"]          # перспективность: больше = лучше
    zone = zone_of(ax, ay)

    say("ПЛОЩАДЬ РАЗБОРА")
    say("   общих ячеек %d (%.0f кв.км), из них на листе %d, вне листа %d"
        % (common.sum(), common.sum() * CELL_KM2, (common & in_sheet).sum(),
           (common & ~in_sheet).sum()))

    # --- 1. согласие с новым эталоном
    say("\n1. СОГЛАСИЕ НАШЕГО ПРОГНОЗА С НАШИМ ЖЕ ВОССОЗДАНИЕМ ФОРМУЛЫ")
    say("   %-16s %10s %12s %10s" % ("зона", "совпадение", "без наклона", "ранговая"))
    for tag, cells in (("вся область", common), ("лист", common & in_sheet),
                       ("вся смежная", common & ~in_sheet),
                       ("юго-запад", common & (zone == "юго-запад")),
                       ("юг", common & (zone == "юг"))):
        if cells.sum() < 300:
            continue
        a, b, r = agreement(ours, crit, ax, ay, cells)
        say("   %-16s %10.3f %12.3f %+10.3f" % (tag, a, b, r))
    say("   (случайный уровень совпадения 0.10)")

    # --- 2. ядра
    lab_c = cores(crit, common, shape)
    lab_o = cores(ours, common, shape)
    say("\n2. ЯДРА (верхние 10%% площади, связные области от 5 кв.км)")
    say("   критериальная карта: %d ядер, %.0f кв.км"
        % (lab_c.max(), (lab_c > 0).sum() * CELL_KM2))
    say("   прогнозная карта:    %d ядер, %.0f кв.км"
        % (lab_o.max(), (lab_o > 0).sum() * CELL_KM2))

    # --- точки заверки
    meta_s, _ = criterial_target.load_prognoz_grid()
    pts = assessment.load_verification_points(meta_s)
    sx, sy = meta_s.cell_centers()
    sx, sy = sx.ravel(), sy.ravel()
    pt_x = sx[pts["cell"].to_numpy()]
    pt_y = sy[pts["cell"].to_numpy()]
    pt_code = pts["code"].to_numpy()
    pt_tree = cKDTree(np.column_stack([pt_x, pt_y]))
    say("   точек заверки: %d (из них несмещённых %d)"
        % (len(pts), int(pts["unbiased_strict"].sum())))

    # --- признаки для прогнозной карты
    wide = pd.read_parquet(config.PROCESSED_DIR / "wide_v1_archive"
                           / "dataset_wide.parquet")
    feat = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    Xdf = spectral_mask.apply(wide[cols])
    cols = list(Xdf.columns)       # маска может переименовать/убрать столбцы
    X = Xdf.to_numpy(dtype=float)
    bg = common & (lab_o == 0)
    mu = np.nanmean(X[bg], axis=0)
    sd = np.nanstd(X[bg], axis=0)
    sd[sd < 1e-9] = np.nan

    rows = []
    for tag, lab in (("критериальная", lab_c), ("прогнозная", lab_o)):
        say("\n3. ЧЕМ ВЫДЕЛЕНО КАЖДОЕ ЯДРО — КАРТА: %s" % tag.upper())
        for k in range(1, lab.max() + 1):
            m = lab == k
            cx, cy = ax[m].mean(), ay[m].mean()
            inside = in_sheet[m].mean()
            zn = pd.Series(zone[m]).value_counts().idxmax()
            d_pt, i_pt = pt_tree.query([cx, cy])
            n_in = int(((pt_x >= ax[m].min() - 250) & (pt_x <= ax[m].max() + 250)
                        & (pt_y >= ay[m].min() - 250) & (pt_y <= ay[m].max() + 250)
                        ).sum())
            row = {"карта": tag, "ядро": k, "кв.км": m.sum() * CELL_KM2,
                   "X_км": cx / 1000, "Y_км": cy / 1000, "зона": zn,
                   "доля_на_листе": inside, "точек_рядом": n_in,
                   "до_точки_км": d_pt / 1000,
                   "тип_точки": POINT_KIND.get(int(pt_code[i_pt]), "?")}
            say("   ядро %-2d  %6.0f кв.км  центр X %.0f Y %.0f  %-10s  "
                "лист %3.0f%%  ближайшая точка %.1f км (%s)"
                % (k, row["кв.км"], cx / 1000, cy / 1000, zn, 100 * inside,
                   row["до_точки_км"], row["тип_точки"]))
            if tag == "критериальная":
                base = {n: np.nanmean(moved["вклад_" + n][common]) for n in names}
                dev = {n: base[n] - np.nanmean(moved["вклад_" + n][m]) for n in names}
                tot = sum(v for v in dev.values() if v > 0)
                share = sorted(((n, v / tot if tot > 0 else 0.0)
                                for n, v in dev.items()), key=lambda t: -t[1])
                say("           вклад факторов: " + ", ".join(
                    "%s %.0f%%" % (RUS_ROLE[n], 100 * s) for n, s in share[:4]))
                row.update({"фактор_%d" % (i + 1): "%s %.0f%%" % (RUS_ROLE[n], 100 * s)
                            for i, (n, s) in enumerate(share[:3])})
            else:
                dz = (np.nanmean(X[m], axis=0) - mu) / sd
                order = np.argsort(-np.abs(dz))[:5]
                say("           признаки: " + ", ".join(
                    "%s %+.1fσ" % (cols[j], dz[j]) for j in order))
                for i, j in enumerate(order[:3]):
                    row["признак_%d" % (i + 1)] = "%s %+.2f" % (cols[j], dz[j])
                    try:
                        row["смысл_%d" % (i + 1)] = feature_dict.describe(
                            cols[j], feature_dict.classify(cols[j]))
                    except ValueError:        # служебные столбцы маски
                        row["смысл_%d" % (i + 1)] = "служебный признак маски"
            rows.append(row)

    # --- 4. пересечение ядер
    say("\n4. ПЕРЕСЕЧЕНИЕ ЯДЕР ДВУХ КАРТ")
    both = (lab_c > 0) & (lab_o > 0)
    say("   общая площадь ядер: критериальных %.0f кв.км, прогнозных %.0f, "
        "общих %.0f (%.0f%% прогнозных)"
        % ((lab_c > 0).sum() * CELL_KM2, (lab_o > 0).sum() * CELL_KM2,
           both.sum() * CELL_KM2, 100 * both.sum() / max((lab_o > 0).sum(), 1)))
    for k in range(1, lab_o.max() + 1):
        m = lab_o == k
        hit = lab_c[m]
        share = (hit > 0).mean()
        who = pd.Series(hit[hit > 0]).value_counts().idxmax() if (hit > 0).any() else None
        say("   прогнозное ядро %-2d перекрыто критериальными на %3.0f%%%s"
            % (k, 100 * share,
               ("; главное — критериальное ядро %d" % who) if who else ""))

    # --- 5. заверка по точкам
    say("\n5. ЗАВЕРКА ПО ТОЧКАМ: доля точек в верхних 10% площади")
    tree_cell = cKDTree(np.column_stack([ax[common], ay[common]]))
    d_pt, i_pt = tree_cell.query(np.column_stack([pt_x, pt_y]))
    on_area = d_pt <= 375.0
    idx = np.flatnonzero(common)[i_pt]
    unb = pts["unbiased_strict"].to_numpy()
    for tag, lab in (("критериальная", lab_c), ("прогнозная", lab_o)):
        top = np.zeros(ax.size, bool)
        top[common] = (crit if tag == "критериальная" else ours)[common] >=             np.quantile((crit if tag == "критериальная" else ours)[common], 1 - AREA)
        for who, sel in (("все точки", on_area), ("несмещённые", on_area & unb)):
            if sel.sum() == 0:
                continue
            hit = top[idx[sel]].mean()
            say("   %-14s %-12s точек %2d, в ядрах %.0f%% (ожидание при случайной "
                "карте 10%%, выигрыш x%.1f)"
                % (tag, who, int(sel.sum()), 100 * hit, hit / AREA))

    # --- карта
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 10.4), constrained_layout=True)
    ext = (ax.min() - 250, ax.max() + 250, ay.min() - 250, ay.max() + 250)
    for axi, (tag, lab, score) in zip(axes, (("критериальная (наше воссоздание)",
                                              lab_c, crit),
                                             ("наш прогноз", lab_o, ours))):
        img = np.where(common, np.where(lab > 0, 2.0, 1.0), np.nan).reshape(shape)
        axi.imshow(np.ma.masked_invalid(img), origin="upper", extent=ext,
                   cmap=matplotlib.colors.ListedColormap(["#e9e9ef", "#a3122b"]),
                   vmin=0.5, vmax=2.5, interpolation="nearest")
        for k in range(1, lab.max() + 1):
            m = lab == k
            axi.annotate(str(k), (ax[m].mean(), ay[m].mean()), fontsize=9,
                         ha="center", va="center", color="#ffffff",
                         bbox=dict(boxstyle="circle,pad=0.15", fc="#101010",
                                   ec="none", alpha=0.75))
        axi.scatter(pt_x, pt_y, s=12, c="#2b6cb0", marker="^",
                    label="точки заверки", zorder=3)
        axi.set_title("Ядра: %s" % tag, fontsize=12)
        axi.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.yaxis.set_major_formatter(lambda v, _: "%.0f" % (v / 1000))
        axi.set_aspect("equal")
        axi.legend(loc="lower left", fontsize=9)
    fig.suptitle("Ядра верхних 10% площади: что выделяет каждая карта", fontsize=13)
    path = outputs.png("cores_anatomy.png")
    mapstyle.save_fig(fig, path, dpi=180)
    say("\nКарта: %s" % path)

    out = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out / "cores_anatomy.csv", index=False,
                              encoding="utf-8-sig")
    (out / "cores_anatomy.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
