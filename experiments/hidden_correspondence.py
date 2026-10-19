"""Скрытые соответствия: приравниваем дистанционку и геологию формулы.

ЗАЧЕМ. Два набора признаков описывают одну территорию: дистанционное
зондирование и геологические слои формулы. Вопрос не «какой из них лучше
предсказывает», а «насколько один выражается через другой» — то есть можно ли
приравнять два описания и что окажется общим. Если общее есть, дистанционка
частично измеряет ту же геологию, и это объясняет, почему она работает там,
где геологических карт нет.

ГЛАВНАЯ ЛОВУШКА. Любые два поля на одной территории связаны уже потому, что
оба меняются плавно с координатами. Поэтому каждое соответствие считается
дважды: как есть и после вычитания координатного тренда из обоих наборов.
Разница между двумя числами и есть настоящая связь, не сводимая к общей
географии. Веса подбираются только на обучающих полосах, значение связи
меряется на отложенной полосе.

ЧТО СЧИТАЕТСЯ.
1. Канонические связи: пары «взвешенная сумма дистанционных признаков» и
   «взвешенная сумма геологических», подобранные так, чтобы совпадать как
   можно сильнее. Это и есть приравнивание двух уравнений.
2. Послойно: какую долю разброса каждого геологического слоя дистанционка
   воспроизводит вне обучения (нелинейно, бустингом).
3. Нечёткие концепты: сеть ДФФЛ с раздельными блоками — три дистанционных,
   три геологических; какие концепты одной стороны совпадают с концептами
   другой и какие правила сводной стадии их связывают.

ВХОД: сетка листа, dataset_v5_rebuilt.parquet
ВЫХОД: outputs/03_признаковое_пространство/hidden_correspondence.png,
       outputs/metrics/hidden_correspondence.log и .csv
Запуск из корня: python -X utf8 -m experiments.hidden_correspondence
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
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NF_ROOT = Path.home() / "deep-neuro-fuzzy"
sys.path.insert(0, str(NF_ROOT / "src"))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, outputs, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.gold_points_branches import fill  # noqa: E402
from experiments.honest_feature_signal import (detrend, poly_design,  # noqa: E402
                                               strip_folds)
from experiments.neurofuzzy_dffl import fit_dffl, to_unit, tt  # noqa: E402
from experiments.shap_two_branches import GEOLOGY, RUS_GEO  # noqa: E402

from ruanfis import analyze_concept_flow  # noqa: E402

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
N_FOLDS = 4            # сколько полос брать на послойный счёт
N_PAIRS = 5            # сколько канонических пар показывать
RIDGE = 1e-3
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%5.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def zscore(train: np.ndarray, *others):
    mu = train.mean(axis=0)
    sd = np.where(train.std(axis=0) > 1e-9, train.std(axis=0), 1.0)
    return [(z - mu) / sd for z in (train,) + others]


def canon(Xtr, Ytr, Xte, Yte):
    """Канонические веса по обучающей части, связь — на отложенной полосе.

    Возвращает (связи на отложенной полосе, веса X, веса Y). Решение через
    отбеливание ковариаций и разложение по сингулярным числам: пары подбираются
    так, чтобы взвешенные суммы двух наборов совпадали как можно сильнее.
    """
    Xtr, Xte = zscore(Xtr, Xte)
    Ytr, Yte = zscore(Ytr, Yte)
    n = Xtr.shape[0]
    Cxx = Xtr.T @ Xtr / n + RIDGE * np.eye(Xtr.shape[1])
    Cyy = Ytr.T @ Ytr / n + RIDGE * np.eye(Ytr.shape[1])
    Cxy = Xtr.T @ Ytr / n

    def inv_sqrt(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-9, None)
        return V @ np.diag(w ** -0.5) @ V.T

    Rx, Ry = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Rx @ Cxy @ Ry)
    Wx, Wy = Rx @ U, Ry @ Vt.T
    u, v = Xte @ Wx, Yte @ Wy
    rho = [abs(np.corrcoef(u[:, k], v[:, k])[0, 1])
           for k in range(min(N_PAIRS, Wy.shape[1]))]
    return np.array(rho), Wx, Wy


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

    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    S = spectral_mask.apply(feat[dzz])
    names_d = list(S.columns)
    Xd = fill(S.to_numpy(float)[pool])
    geo = [c for c in GEOLOGY if c in raw.columns]
    Xg = fill(raw[geo].to_numpy(float)[pool])
    y = (crit >= np.nanquantile(crit[pool], 1 - AREA)).astype(int)[pool]
    D = poly_design(cx[pool], cy[pool])
    folds = strip_folds(cx[pool], cy[pool], y, GAP_M)[:N_FOLDS]
    log("ячеек %d | дистанционных признаков %d, геологических %d, полос %d"
        % (pool.size, len(names_d), len(geo), len(folds)))

    # --- 1. канонические связи, как есть и после вычитания тренда
    log("")
    log("КАНОНИЧЕСКИЕ СВЯЗИ ДВУХ ОПИСАНИЙ (на отложенной полосе)")
    res = {}
    for tag in ("как есть", "без координатного тренда"):
        rr, keepW = [], None
        for tr, te in folds:
            if tag == "как есть":
                A, Ate, B, Bte = Xd[tr], Xd[te], Xg[tr], Xg[te]
            else:
                A, Ate = detrend(Xd[tr], Xd[te], D[tr], D[te])
                B, Bte = detrend(Xg[tr], Xg[te], D[tr], D[te])
            rho, Wx, Wy = canon(A, B, Ate, Bte)
            rr.append(rho)
            keepW = keepW or (Wx, Wy)
        res[tag] = (np.mean(rr, axis=0), keepW)
        log("   %-26s пары 1-%d: %s" % (tag, N_PAIRS, ", ".join(
            "%.3f" % v for v in res[tag][0])))
    drop = res["как есть"][0][0] - res["без координатного тренда"][0][0]
    log("   первая пара теряет на вычитании тренда %.3f — столько в связи "
        "было общей географии" % drop)

    Wx, Wy = res["как есть"][1]
    ld = [names_d[j] for j in np.argsort(-np.abs(Wx[:, 0]))[:4]]
    lg = [RUS_GEO.get(geo[j], geo[j])
          for j in np.argsort(-np.abs(Wy[:, 0]))[:4]]
    log("   первая пара: со стороны съёмки %s" % ", ".join(ld))
    log("                со стороны геологии %s" % ", ".join(lg))

    # --- 2. послойно: что из геологии воспроизводится съёмкой
    log("")
    log("ПОСЛОЙНО: доля разброса геологического слоя, воспроизведённая "
        "съёмкой вне обучения (отрицательная — хуже среднего)")
    rows = []
    for j, col in enumerate(geo):
        r2, r2d = [], []
        for tr, te in folds:
            m = HistGradientBoostingRegressor(
                max_iter=config.HGB_MAX_ITER,
                learning_rate=config.HGB_LEARNING_RATE,
                max_leaf_nodes=config.HGB_MAX_LEAF_NODES,
                random_state=config.CRIT_SEED)
            t = Xg[:, j]
            p = m.fit(Xd[tr], t[tr]).predict(Xd[te])
            r2.append(1 - ((t[te] - p) ** 2).sum()
                      / ((t[te] - t[tr].mean()) ** 2).sum())
            Atr, Ate = detrend(Xd[tr], Xd[te], D[tr], D[te])
            td = t - D @ np.linalg.lstsq(D[tr], t[tr], rcond=None)[0]
            p2 = m.fit(Atr, td[tr]).predict(Ate)
            r2d.append(1 - ((td[te] - p2) ** 2).sum()
                       / ((td[te] - td[tr].mean()) ** 2).sum())
        rows.append({"слой": RUS_GEO.get(col, col), "столбец": col,
                     "как_есть": float(np.mean(r2)),
                     "без_тренда": float(np.mean(r2d))})
        log("   %-38s как есть %+.2f | без тренда %+.2f"
            % (rows[-1]["слой"], rows[-1]["как_есть"], rows[-1]["без_тренда"]))
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics"
                              / "hidden_correspondence.csv", index=False,
                              encoding="utf-8-sig")

    # --- 3. нечёткие концепты: что с чем совпало
    log("")
    log("НЕЧЁТКИЕ КОНЦЕПТЫ ДФФЛ: три блока по съёмке, три по геологии")
    import experiments.neurofuzzy_dffl as nd
    saved = nd.BLOCKS
    nd.BLOCKS = [
        ("рельеф", ("рельеф_1",),
         [("dem_elev", "высота"), ("dem_relief_1km", "расчленённость_1км"),
          ("relief2_curv_5km", "кривизна_5км"), ("dem_incision", "врезание")]),
        ("линеаменты", ("линеаменты_1",),
         [("lin_dist", "расст_линеамент"),
          ("lin_node_dens", "плотность_узлов"), ("lin_dens", "плотность_лин")]),
        ("спектры", ("спектры_1",),
         [("s2_iron_ox", "окислы_железа"), ("s2_ferrous", "закись_железа"),
          ("s2_b11_std", "разброс_канала_11")]),
        ("разломы", ("разломы_1",),
         [("dist_tect1", "расст_разломы_СЗ"), ("dist_tect2", "расст_разломы_СВ"),
          ("dens_tect", "плотность_разломов"),
          ("geo2_fault_node", "расст_узлы_разломов")]),
        ("среда", ("среда_1",),
         [("dist_paleo", "расст_палеодолины"), ("dist_facies", "расст_фации"),
          ("dist_struct", "расст_коры"), ("geo2_facies2", "фация")]),
        ("магматизм", ("магматизм_1",),
         [("dist_magm", "расст_дайки"), ("dens_magm", "плотность_даек"),
          ("geo2_contact", "расст_контакты")]),
    ]
    cols = []
    for _, _, fs in nd.BLOCKS:
        for col, _ in fs:
            src = S if col in S.columns else raw
            cols.append(src[col].to_numpy(float)[pool])
    Xf = fill(np.column_stack(cols))
    U, = to_unit(Xf)
    model = fit_dffl(U, y, U, y, config.CRIT_SEED)
    flows = analyze_concept_flow(model, tt(U[::6]), top_k_rules=1)
    vals: dict[str, list] = {}
    for f in flows:
        for c in f.hidden_concepts:
            if c.stage_name.startswith("предметные"):
                vals.setdefault(c.block_name, []).append(c.value)
    nd.BLOCKS = saved
    names = list(vals)
    M = np.column_stack([vals[n] for n in names])
    dzz_n, geo_n = names[:3], names[3:]
    log("   связь концептов съёмки и геологии (по значению концепта):")
    for i, a in enumerate(dzz_n):
        line = "   %-12s" % a
        for j, b in enumerate(geo_n):
            line += " | %s %+.2f" % (b, np.corrcoef(M[:, i],
                                                    M[:, 3 + j])[0, 1])
        log(line)
    cross = []
    for f in flows[:400]:
        for c in f.hidden_concepts:
            if c.stage_name.startswith("сводка"):
                for r in c.top_rule_contributions:
                    has_d = any(n in r.rule_text for n in dzz_n)
                    has_g = any(n in r.rule_text for n in geo_n)
                    if has_d and has_g:
                        cross.append(r.rule_text)
    if cross:
        log("   правила сводной стадии, связывающие съёмку с геологией:")
        for text, cnt in pd.Series(cross).value_counts().head(5).items():
            log("      %s  (ведущее в %d ячейках)" % (text, cnt))
    else:
        log("   правил, связывающих съёмку с геологией, среди ведущих нет")

    # --- рисунок
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6),
                             constrained_layout=True)
    k = np.arange(1, N_PAIRS + 1)
    axes[0].bar(k - 0.18, res["как есть"][0], width=0.36, label="как есть",
                color="#c44e52")
    axes[0].bar(k + 0.18, res["без координатного тренда"][0], width=0.36,
                label="без координатного тренда", color="#4c72b0")
    axes[0].set_xticks(k)
    axes[0].set_xlabel("номер канонической пары")
    axes[0].set_ylabel("связь на отложенной полосе")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Насколько два описания приравниваются", fontsize=11)
    ext = (cx.min() - 250, cx.max() + 250, cy.min() - 250, cy.max() + 250)
    Xdz, = zscore(Xd)
    Xgz, = zscore(Xg)
    for axi, (ttl, v) in zip(axes[1:], (
            ("Первая пара: сторона съёмки", Xdz @ Wx[:, 0]),
            ("Первая пара: сторона геологии", Xgz @ Wy[:, 0]))):
        img = np.full(cx.size, np.nan)
        img[pool] = v
        lo, hi = np.nanpercentile(img[pool], [2, 98])
        im = axi.imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                        origin="upper", interpolation="nearest",
                        cmap="cividis", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=axi, shrink=0.7)
        axi.set_title(ttl, fontsize=11)
        axi.set_xlabel("X, км")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.yaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Скрытые соответствия съёмки и геологии формулы", fontsize=12)
    path = outputs.png("hidden_correspondence.png")
    mapstyle.save_fig(fig, path, dpi=150)
    log("")
    log("карта: %s" % path)
    (ROOT / "outputs" / "metrics" / "hidden_correspondence.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
