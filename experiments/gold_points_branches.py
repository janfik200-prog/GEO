"""Обучение на факте: 1 там, где золото найдено, 0 везде остальное.

ЗАЧЕМ. До сих пор мишенью были верхние 10% критериального балла. Как только в
признаки разрешили геологические слои формулы, такая мишень замкнулась на свои
же входы: модель воспроизводит формулу, а не ищет золото. Единственный способ
убрать замыкание — сменить мишень на независимый факт. Здесь мишень —
опробование: ячейка получает 1, если в ней есть проба с золотом, и 0 иначе.
Критериальный анализ этих точек не видел ни на одном шаге, поэтому геологию
можно подавать свободно.

ЧТО ЗА ТОЧКИ. Слой «геохимическое_опробование» пакета организации, отобраны
метки, содержащие Au. Это НЕ рудопроявления: литохимические пробы по рыхлым
отложениям и по коренным породам плюс шлиховые пробы; состав печатается в
лог. Шлиховая проба — поисковый признак, а не объект, и в отчёте её нельзя
называть проявлением.

ОГОВОРКА О ГЕОМЕТРИИ ОПРОБОВАНИЯ. Пробы в рыхлых отложениях отбираются вдоль
водотоков, поэтому часть «успеха» любой модели может быть геометрией
маршрутов, а не геологией. Для контроля в сравнение включены две планки:
расстояние до гидросети как единственный признак и «одна география» (гладкое
поле без признаков). Модель обязана обойти обе, иначе результата нет.

ЧТО СЧИТАЕТСЯ. Пять наборов на одних и тех же полосах с разрывом 15 км:
критериальный балл формулы, одна география, гидросеть, только дистанционка,
дистанционка с геологией формулы, только геология формулы. Метрики — площадь
под кривой на отложенной полосе и доля точек, попавших в верхние 5/10/20%
площади по прогнозу вне обучения.

ВХОД: сетка листа, dataset_v5_rebuilt.parquet, слой опробования
ВЫХОД: outputs/02_прогноз/gold_points_branches.png,
       outputs/metrics/gold_points_branches.log и .csv
Запуск из корня: python -X utf8 -m experiments.gold_points_branches
"""

import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, neighbourhood as nb, outputs, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model  # noqa: E402
from experiments.honest_final_control import rbf_basis  # noqa: E402
from experiments.shap_two_branches import GEOLOGY  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GAP_M = 15_000.0
N_STRIPS = 4
MIN_TE_POS = 3          # минимум точек в отложенной полосе
MIN_TR_POS = 20         # минимум точек в обучающей части
AREAS = (0.05, 0.10, 0.20)
POINTS_SHP = "геохимическое_опробование.shp"
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def gold_cells(meta) -> tuple[np.ndarray, pd.DataFrame]:
    """Плоские индексы ячеек с золотосодержащими пробами."""
    g = gpd.read_file(ROOT / "data" / "Gis-integro" / config.SHP_SUBDIR
                      / POINTS_SHP)
    au = g[g["label"].astype(str).str.contains("Au")].copy()
    row = np.floor((meta.y_top - au.geometry.y.to_numpy()) / meta.dy).astype(int)
    col = np.floor((au.geometry.x.to_numpy() - meta.x0) / meta.dx).astype(int)
    ok = (row >= 0) & (row < meta.prf) & (col >= 0) & (col < meta.pic)
    return np.unique(row[ok] * meta.pic + col[ok]), au


def point_folds(cx, cy, y):
    """Полосы по X и по Y с разрывом; годятся только полосы с точками."""
    folds = []
    for coord in (cx, cy):
        edges = np.linspace(coord.min(), coord.max(), N_STRIPS + 1)
        sid = np.clip(np.digitize(coord, edges[1:-1]), 0, N_STRIPS - 1)
        for k in range(N_STRIPS):
            te = sid == k
            tr = (coord < edges[k] - GAP_M) | (coord > edges[k + 1] + GAP_M)
            if y[te].sum() >= MIN_TE_POS and y[tr].sum() >= MIN_TR_POS:
                folds.append((tr, te))
    return folds


def fill(X: np.ndarray) -> np.ndarray:
    out = X.copy()
    med = np.nanmedian(out, axis=0)
    bad = ~np.isfinite(out)
    out[bad] = np.take(med, np.where(bad)[1])
    return out


def evaluate(name, X, y, folds, fixed=None):
    """Площадь под кривой по полосам и доля точек в верхних долях площади.

    ``fixed`` — готовый балл (критериальный), его не обучают, а только ранжируют
    внутри отложенной полосы, как и прогнозы моделей.
    """
    au, cov = [], {a: [] for a in AREAS}
    for tr, te in folds:
        if fixed is None:
            p = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        else:
            p = fixed[te]
        au.append(roc_auc_score(y[te], p))
        r = (np.argsort(np.argsort(p)) + 0.5) / p.size      # ранг внутри полосы
        for a in AREAS:
            cov[a].append(float((r[y[te] == 1] >= 1 - a).mean()))
    row = {"набор": name, "полос": len(folds),
           "площадь_под_кривой": float(np.mean(au)),
           "разброс": float(np.std(au))}
    for a in AREAS:
        row["точек_в_верхних_%d%%" % (100 * a)] = float(np.mean(cov[a]))
    log("%-34s площадь под кривой %.3f ± %.3f | точек в верхних 5/10/20%%: "
        "%.0f%% / %.0f%% / %.0f%%"
        % (name, row["площадь_под_кривой"], row["разброс"],
           100 * row["точек_в_верхних_5%"], 100 * row["точек_в_верхних_10%"],
           100 * row["точек_в_верхних_20%"]))
    return row


def loo_eval(name, X, y, cxp, cyp, idx_pool, radius_m, fixed=None):
    """Проверка по одной точке: вокруг отложенной точки вырезается круг.

    Для каждой ячейки с золотом обучение идёт по всем остальным ячейкам, кроме
    попавших в круг радиусом ``radius_m`` вокруг неё, а ранг этой точки
    считается среди всех ячеек за пределами круга. Так используются все точки,
    а не только те полосы, где их набралось достаточно.
    """
    ranks = []
    for i in idx_pool:
        far = (cxp - cxp[i]) ** 2 + (cyp - cyp[i]) ** 2 > radius_m ** 2
        if fixed is None:
            m = make_model().fit(X[far], y[far])
            p = m.predict_proba(X)[:, 1]
        else:
            p = fixed
        sel = far.copy()
        sel[i] = True
        r = (np.argsort(np.argsort(p[sel])) + 0.5) / sel.sum()
        ranks.append(float(r[np.flatnonzero(sel) == i][0]))
    ranks = np.array(ranks)
    row = {"набор": name, "точек": ranks.size,
           "средний_ранг": float(ranks.mean())}
    for a in AREAS:
        row["точек_в_верхних_%d%%" % (100 * a)] = float((ranks >= 1 - a).mean())
    log("%-34s средний ранг %.3f | точек в верхних 5/10/20%%: "
        "%.0f%% / %.0f%% / %.0f%%"
        % (name, ranks.mean(), 100 * row["точек_в_верхних_5%"],
           100 * row["точек_в_верхних_10%"], 100 * row["точек_в_верхних_20%"]))
    return row


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    idx, au = gold_cells(meta)
    y_full = np.zeros(cx.size, int)
    y_full[idx] = 1
    y = y_full[pool]
    log("проб с золотом %d -> ячеек %d; в рабочей маске %d из %d ячеек листа"
        % (len(au), idx.size, int(y.sum()), pool.size))
    log("состав проб: " + ", ".join(
        "%s %d" % (k, v) for k, v in au["text"].astype(str).str.slice(0, 28)
        .value_counts().items()))

    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    S = spectral_mask.apply(feat[dzz])
    Xa = fill(S.to_numpy(float)[pool])
    geo = [c for c in GEOLOGY if c in raw.columns]
    Xg = fill(raw[geo].to_numpy(float)[pool])
    Xb = np.column_stack([Xa, Xg])
    Xgeog = rbf_basis(cx[pool], cy[pool], 6)
    Xhydro = fill(feat[["dist_dnara"]].to_numpy(float)[pool])

    folds = point_folds(cx[pool], cy[pool], y)
    log("полос с разрывом %.0f км: %d (точек в отложенной полосе не меньше %d)"
        % (GAP_M / 1000, len(folds), MIN_TE_POS))
    if not folds:
        log("полос не набралось — точек слишком мало")
        return

    rows = [
        evaluate("критериальный балл (формула)", None, y, folds,
                 fixed=crit[pool]),
        evaluate("одна география (гладкое поле)", Xgeog, y, folds),
        evaluate("расстояние до гидросети", Xhydro, y, folds),
        evaluate("A: только дистанционка", Xa, y, folds),
        evaluate("B: дистанционка + геология", Xb, y, folds),
        evaluate("только геология формулы", Xg, y, folds),
    ]
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics"
                              / "gold_points_branches.csv", index=False,
                              encoding="utf-8-sig")

    log("")
    log("ПРОВЕРКА ПО ОДНОЙ ТОЧКЕ, круг исключения %.0f км вокруг отложенной "
        "точки (ранг 0.5 — случайный уровень)" % (GAP_M / 1000))
    ipool = np.flatnonzero(y == 1)
    loo = [
        loo_eval("критериальный балл (формула)", None, y, cx[pool], cy[pool],
                 ipool, GAP_M, fixed=crit[pool]),
        loo_eval("одна география (гладкое поле)", Xgeog, y, cx[pool], cy[pool],
                 ipool, GAP_M),
        loo_eval("расстояние до гидросети", Xhydro, y, cx[pool], cy[pool],
                 ipool, GAP_M),
        loo_eval("A: только дистанционка", Xa, y, cx[pool], cy[pool], ipool,
                 GAP_M),
        loo_eval("B: дистанционка + геология", Xb, y, cx[pool], cy[pool],
                 ipool, GAP_M),
        loo_eval("только геология формулы", Xg, y, cx[pool], cy[pool], ipool,
                 GAP_M),
    ]
    pd.DataFrame(loo).to_csv(ROOT / "outputs" / "metrics"
                             / "gold_points_leave_one_out.csv", index=False,
                             encoding="utf-8-sig")

    # --- карта: лучший набор на всех точках против критериальной
    best = max(rows[3:], key=lambda r: r["площадь_под_кривой"])["набор"]
    Xbest = {"A: только дистанционка": Xa, "B: дистанционка + геология": Xb,
             "только геология формулы": Xg}[best]
    log("")
    log("карта по набору «%s», обучение на всех %d ячейках с золотом"
        % (best, int(y.sum())))
    m = make_model().fit(Xbest, y)
    p = np.full(cx.size, np.nan)
    p[pool] = m.predict_proba(Xbest)[:, 1]
    shape = (meta.prf, meta.pic)
    km = 1000.0 / meta.dx
    p = nb.aniso_smooth(np.nan_to_num(p), shape, ALONG_KM * km, ACROSS_KM * km,
                        nb.theta_from_azimuth(NW_AZIMUTH))
    for tag, score in (("прогноз по точкам", p), ("критериальный балл", crit)):
        r = np.full(cx.size, np.nan)
        r[pool] = (np.argsort(np.argsort(score[pool])) + 0.5) / pool.size
        log("   %-22s точек в верхних 5/10/20%%: %.0f%% / %.0f%% / %.0f%%"
            % (tag, 100 * (r[idx] >= 0.95).mean(), 100 * (r[idx] >= 0.90).mean(),
               100 * (r[idx] >= 0.80).mean()))

    ext = (cx.min() - 250, cx.max() + 250, cy.min() - 250, cy.max() + 250)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6),
                             constrained_layout=True)
    for axi, (ttl, v) in zip(axes, (("Обучение на пробах с золотом", p),
                                    ("Критериальный балл (формула)", crit))):
        vv = np.where(valid, v, np.nan)
        cuts = np.nanpercentile(vv[valid], mapstyle.CLASS_EDGES[1:-1])
        img = np.where(valid, np.digitize(vv, cuts).astype(float), np.nan)
        axi.imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                   origin="upper", interpolation="nearest", vmin=-0.5,
                   vmax=4.5, cmap=matplotlib.colors.ListedColormap(
                       mapstyle.CLASS_COLORS))
        axi.scatter(cx[idx], cy[idx], s=16, facecolor="none",
                    edgecolor="#102a43", linewidths=0.9,
                    label="пробы с золотом")
        axi.legend(loc="upper left", fontsize=8, framealpha=0.9)
        axi.set_title(ttl, fontsize=11)
        axi.set_xlabel("X, км")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.yaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Мишень — факт находки золота в пробе (классы по доле "
                 "площади)", fontsize=12)
    path = outputs.png("gold_points_branches.png")
    mapstyle.save_fig(fig, path, dpi=150)
    log("карта: %s" % path)
    (ROOT / "outputs" / "metrics" / "gold_points_branches.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
