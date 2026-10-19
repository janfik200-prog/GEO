"""Аппроксимация модели: можно ли записать прогноз простой формулой.

ЗАЧЕМ. Разложение по Шепли показало, что балл собирается из сочетаний
признаков, а не из порога по одному слою. Отсюда вопрос: насколько простой
может быть запись прогноза, чтобы он остался тем же? Если сумма одномерных
функций воспроизводит балл почти полностью, прогноз можно оформить формулой
по образцу критериальной — она переносится на новую территорию без
переобучения и объяснима геологу. Если нет, станет ясно, какая доля качества
держится именно на взаимодействиях.

ЧТО СЧИТАЕТСЯ. Обучается рабочая модель (бустинг, цель — верхние 10%
критериального балла листа). Её выход аппроксимируется четырьмя формами
возрастающей сложности:
  * линейная сумма признаков;
  * сумма одномерных функций (сплайны по каждому признаку) — это в точности
    устройство критериальной формулы, только функции подбираются по данным;
  * то же плюс парные произведения восьми ведущих признаков;
  * дерево решений глубины 4 — набор правил «если-то».
Проверка пространственно строгая: аппроксимация настраивается на обучающих
полосах листа с разрывом 15 км и оценивается на отложенной полосе — по
близости к баллу модели (доля объяснённого разброса, ранговая связь) и по
исходной цели (площадь под кривой). Отдельно считается, насколько форма
сохраняет карту на юго-западе.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet, сетка листа
ВЫХОД: outputs/03_признаковое_пространство/surrogate_approx.png,
       outputs/metrics/surrogate_approx.log и .csv
Запуск из корня: python -X utf8 -m experiments.surrogate_approx
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
from scipy import stats as sps  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import SplineTransformer, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, export_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, outputs, spectral_mask)
from experiments.branch5_alignment import align  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
TOP_PAIRS = 8          # сколько ведущих признаков перемножать попарно
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def fill(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Пропуски (погашенные маской спектры) — медианой обучающей части."""
    out = X.copy()
    med = np.nanmedian(ref, axis=0)
    bad = ~np.isfinite(out)
    out[bad] = np.take(med, np.where(bad)[1])
    return out


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
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
    box = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
           (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    _, idx = cKDTree(np.column_stack([wx[box], wy[box]])).query(
        np.column_stack([cx, cy]), k=1)
    w_in = wide.loc[box].reset_index(drop=True)
    wal = pd.DataFrame(align(w_in.loc[idx, cols].to_numpy(float)[valid],
                             feat[cols].to_numpy(float)[valid],
                             wide[cols].to_numpy(float)), columns=cols)
    S = spectral_mask.apply(feat[cols])
    names = list(S.columns)
    Xs = S.to_numpy(float)
    Xw = spectral_mask.apply(wal).to_numpy(float)

    y = (crit >= np.nanquantile(crit[pool], 1 - AREA)).astype(int)
    model = make_model().fit(fill(Xs[pool], Xs[pool]), y[pool])
    p_sheet = model.predict_proba(fill(Xs, Xs[pool]))[:, 1]
    log("модель обучена: ячеек %d, признаков %d" % (pool.size, len(names)))

    # ведущие признаки — по важности бустинга через перестановку на выборке
    rng = np.random.default_rng(config.CRIT_SEED)
    sub = rng.choice(pool, 3000, replace=False)
    base = roc_auc_score(y[sub], p_sheet[sub])
    imp = []
    Xf = fill(Xs, Xs[pool])
    for j in range(len(names)):
        Z = Xf[sub].copy()
        Z[:, j] = rng.permutation(Z[:, j])
        imp.append(base - roc_auc_score(y[sub], model.predict_proba(Z)[:, 1]))
    lead = [names[j] for j in np.argsort(imp)[::-1][:TOP_PAIRS]]
    log("ведущие признаки для парных произведений: %s" % ", ".join(lead))

    def build(tag):
        if tag == "линейная сумма":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0)), None
        if tag == "сумма одномерных функций":
            return make_pipeline(
                StandardScaler(),
                SplineTransformer(n_knots=6, degree=3, include_bias=False),
                Ridge(alpha=1.0)), None
        if tag == "одномерные + парные":
            jj = [names.index(n) for n in lead]
            return make_pipeline(
                StandardScaler(),
                SplineTransformer(n_knots=6, degree=3, include_bias=False),
                Ridge(alpha=1.0)), jj
        return DecisionTreeRegressor(max_depth=4,
                                     random_state=config.CRIT_SEED), None

    def design(X, jj):
        if jj is None:
            return X
        pairs = [X[:, a] * X[:, b] for i, a in enumerate(jj) for b in jj[i + 1:]]
        return np.column_stack([X] + pairs)

    folds = strip_folds(cx[pool], cy[pool], y[pool], GAP_M)
    log("полос с разрывом %.0f км: %d" % (GAP_M / 1000, len(folds)))
    rows = []
    for tag in ("линейная сумма", "сумма одномерных функций",
                "одномерные + парные", "дерево правил (глубина 4)"):
        r2, rho, auc = [], [], []
        for tr, te in folds:
            est, jj = build(tag)
            Xtr = design(Xf[pool][tr], jj)
            Xte = design(Xf[pool][te], jj)
            est.fit(Xtr, p_sheet[pool][tr])
            q = est.predict(Xte)
            t = p_sheet[pool][te]
            r2.append(1 - ((t - q) ** 2).sum() / ((t - t.mean()) ** 2).sum())
            rho.append(sps.spearmanr(q, t).statistic)
            auc.append(roc_auc_score(y[pool][te], q))
        auc_model = np.mean([roc_auc_score(y[pool][te], p_sheet[pool][te])
                             for _, te in folds])
        rows.append({"форма": tag, "объяснено_балла": float(np.mean(r2)),
                     "ранговая_с_баллом": float(np.mean(rho)),
                     "площадь_под_кривой": float(np.mean(auc)),
                     "у_модели": float(auc_model)})
        log("%-28s объяснено балла %.3f, ранговая %+.3f, по цели %.3f "
            "(у модели %.3f)" % (tag, np.mean(r2), np.mean(rho), np.mean(auc),
                                 auc_model))

    # --- карты на всей области: модель против лучшей простой формы
    est, jj = build("сумма одномерных функций")
    est.fit(design(Xf[pool], jj), p_sheet[pool])
    Xwf = fill(Xw, Xs[pool])
    q_wide = est.predict(design(Xwf, jj))
    p_wide = model.predict_proba(Xwf)[:, 1]
    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    dist, _ = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                       ref["y"].to_numpy()[rmsk]])).query(
        np.column_stack([wx, wy]))
    common = dist <= 375.0
    zone = zone_of(wx, wy)
    for tag, cells in (("лист", common & (zone == "лист")),
                       ("юго-запад", common & (zone != "лист"))):
        a = sps.spearmanr(q_wide[cells], p_wide[cells]).statistic
        ta = q_wide[cells] >= np.quantile(q_wide[cells], 1 - AREA)
        tb = p_wide[cells] >= np.quantile(p_wide[cells], 1 - AREA)
        log("перенос формы «сумма одномерных функций» на %s: ранговая с "
            "моделью %+.3f, общих верхних 10%% %.0f%%"
            % (tag, a, 100 * (ta & tb).sum() / tb.sum()))

    shape = (ARCH_META.prf, ARCH_META.pic)
    ext = (wx.min() - 250, wx.max() + 250, wy.min() - 250, wy.max() + 250)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 9.6), constrained_layout=True)
    for axi, (ttl, v) in zip(axes, (("Модель (бустинг)", p_wide),
                                    ("Аппроксимация суммой одномерных функций",
                                     q_wide))):
        vv = np.where(common, v, np.nan)
        cuts = np.nanpercentile(vv[common], mapstyle.CLASS_EDGES[1:-1])
        img = np.where(common, np.digitize(vv, cuts).astype(float), np.nan)
        axi.imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                   origin="upper", interpolation="nearest", vmin=-0.5, vmax=4.5,
                   cmap=matplotlib.colors.ListedColormap(mapstyle.CLASS_COLORS))
        axi.set_title(ttl, fontsize=11)
        axi.set_xlabel("X, км")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.yaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Насколько прогноз сводится к сумме одномерных функций "
                 "(классы по доле площади)", fontsize=12)
    path = outputs.png("surrogate_approx.png")
    mapstyle.save_fig(fig, path, dpi=150)
    log("карта: %s" % path)

    tree = DecisionTreeRegressor(max_depth=3, random_state=config.CRIT_SEED)
    tree.fit(Xf[pool], p_sheet[pool])
    log("")
    log("ДЕРЕВО ПРАВИЛ ГЛУБИНЫ 3 (для чтения, не рабочая версия):")
    for line in export_text(tree, feature_names=names, decimals=1).splitlines():
        log("   " + line)

    out = ROOT / "outputs" / "metrics"
    pd.DataFrame(rows).to_csv(out / "surrogate_approx.csv", index=False,
                              encoding="utf-8-sig")
    (out / "surrogate_approx.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
