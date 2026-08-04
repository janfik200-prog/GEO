"""Этап 5b: нейросетевой перенос направления с внешних меток на Анабар.

ЧТО ПРОВЕРЯЕТСЯ. Может ли НЕЙРОСЕТЬ превзойти критериальный анализ на нашем
объекте, если направление («какой конец шкалы благоприятен») она берёт не из
экспертных правил и не из наших точек, а из тысяч чужих рудопроявлений на
территории, где разведка проведена. Мотив — измеренный на этапе 5 структурный
предел обучения без учителя, см. докстринг :mod:`src.transfer_nn`.

Наш объект не участвует в обучении ни на одном шаге, поэтому 19 несмещённых
точек остаются независимой заверкой, как и на всех предыдущих этапах.

ПРОГОН СОСТОИТ ИЗ ДВУХ ЧАСТЕЙ:

A. Внешняя территория (США, MRDS). Пространственная блочная CV по ячейкам 2
   градуса: AUC и lift@10% у нейросети против линейного критериального индекса
   на тех же признаках. Это ВОРОТА: если сеть не бьёт критериальный индекс там,
   где меток тысячи, переносить нечего (``transfer_nn.MIN_CV_AUC``).
B. Анабар. Замороженная сеть применяется к 22 905 ячейкам в трёх режимах
   (``config.TRANSFER_MODES``) и оценивается тем же предзарегистрированным
   протоколом, что этапы 4b и 5.

КРИТЕРИЙ ПРЕВОСХОДСТВА — ТОТ ЖЕ, ЧТО В 4b И 5, менять его под результат нельзя.
Первичная метрика ``config.VER_PRIMARY``: lift@10% на строго несмещённых точках.
Метод считается превзошедшим критериальный, если выполнены ОБА условия:

1. нижняя граница 90% бутстрэп-интервала разности (наш минус критериальный) при
   ресэмплинге ПО ПРОСТРАНСТВЕННЫМ КЛАСТЕРАМ точек строго больше нуля;
2. сдвиговый (тороидальный) null даёт p ниже порога Бонферрони
   0.05/``config.OWN_N_CONFIGS_TOTAL``.

Знаменатель НАКОПИТЕЛЬНЫЙ (20 конфигураций этапа 5 + 10 этапа 5b = 30,
alpha = 0.00167). Мы перебираем архитектуры на одном и том же наборе из 19
точек, поэтому порог обязан учитывать все опробованные конфигурации: без этого
стратегия «перебирай, пока не выиграет» даёт ложную победу почти наверняка.

Выход: ``outputs/metrics/transfer{,_cv,_null,_boot,_verdict}.csv``,
``outputs/transfer_map.png``, ``outputs/metrics/transfer_scores.npz``.

Запуск из корня: ``python -m experiments.transfer_nn``.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src import (assessment, cell_mask, config,  # noqa: E402
                 criterial_target, integro_grid, transfer_nn)
from experiments.common import criterial, lift as ext_lift  # noqa: E402


def cell_lonlat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Центры ячеек сетки в WGS84 — глобальные гриды адресуются по lon/lat.

    CRS берётся из sidecar самой сетки (Красовский, tmerc lon_0=105), а не
    задаётся константой: подмена проекции сместила бы лист на десятки км и
    сэмплировала бы чужую геофизику, ничем себя не выдав.
    """
    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    if not proj4:
        raise SystemExit("не найден proj4 сетки — нельзя перевести ячейки в lon/lat")
    tr = Transformer.from_crs(proj4, 4326, always_xy=True)
    return tr.transform(df["x"].to_numpy(float), df["y"].to_numpy(float))


def external_cv(Z: np.ndarray, y: np.ndarray, groups: np.ndarray,
                log) -> pd.DataFrame:
    """Блочная CV на внешней территории: нейросеть против критериального индекса.

    Группы — блоки 2 градуса. Рудопроявления кучкуются в рудных районах, и
    случайная CV мерила бы память о районе, а не перенос на новую территорию.
    """
    rows = []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.TRANSFER_SEED)
    for k, (tr, te) in enumerate(cv.split(Z, y, groups), start=1):
        net = transfer_nn.FertilityNet(Z.shape[1], seed=config.TRANSFER_SEED)
        net.fit(Z[tr], y[tr], Z[te], y[te], epochs=config.TRANSFER_EPOCHS,
                batch=config.TRANSFER_BATCH, lr=config.TRANSFER_LR,
                seed=config.TRANSFER_SEED)
        s_nn = net.predict(Z[te])
        s_cr = criterial(Z[tr], y[tr], Z[te])
        rows.append({"fold": k, "n_test": len(te),
                     "auc_nn": roc_auc_score(y[te], s_nn),
                     "auc_criterial": roc_auc_score(y[te], s_cr),
                     "lift_nn": ext_lift(s_nn, y[te]),
                     "lift_criterial": ext_lift(s_cr, y[te])})
        log(f"  фолд {k}: AUC сеть {rows[-1]['auc_nn']:.3f} / "
            f"критериальный {rows[-1]['auc_criterial']:.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    path = config.PROCESSED_DIR / "dataset_v2.parquet"
    if not path.exists():
        raise SystemExit("нет dataset_v2.parquet — сначала "
                         "python -m experiments.build_dataset_v2")
    df = pd.read_parquet(path)
    valid = cell_mask.build_valid_mask(df)
    pool_valid = np.flatnonzero(valid)
    meta, prognoz = criterial_target.load_prognoz_grid()
    lon, lat = cell_lonlat(df)
    log(f"сетка: {len(df)} ячеек, валидных {int(valid.sum())}; "
        f"lon [{lon.min():.2f}, {lon.max():.2f}], lat [{lat.min():.2f}, {lat.max():.2f}]")

    scores: dict[str, np.ndarray] = {}
    cv_rows, gate = [], {}
    for tag, commods in config.TRANSFER_COMMODS.items():
        occ = transfer_nn.load_occurrences(commods, bbox=config.TRANSFER_BBOX)
        if len(occ) > config.TRANSFER_N_POS:
            occ = occ.sample(config.TRANSFER_N_POS,
                             random_state=config.TRANSFER_SEED)
        X, y, olon, olat = transfer_nn.build_training_set(
            occ, config.TRANSFER_BBOX, config.TRANSFER_N_BG,
            config.TRANSFER_SEED)
        med, sc = transfer_nn.robust_stats(X)
        Z = transfer_nn.apply_stats(X, med, sc)
        groups = transfer_nn.spatial_groups(olon, olat, config.TRANSFER_BLOCK_DEG)
        log(f"{tag}: меток {int(y.sum())}, фон {int((y == 0).sum())}, "
            f"блоков {np.unique(groups).size}")

        cv = external_cv(Z, y, groups, log)
        cv.insert(0, "labels", tag)
        cv_rows.append(cv)
        gate[tag] = (cv["auc_nn"].mean(), cv["auc_criterial"].mean())
        log(f"{tag}: блочная CV AUC сеть {gate[tag][0]:.3f} против "
            f"критериального {gate[tag][1]:.3f}")
        if gate[tag][0] < transfer_nn.MIN_CV_AUC:
            log(f"{tag}: ворота не пройдены (AUC < {transfer_nn.MIN_CV_AUC}), "
                "перенос пропущен")
            continue

        # Финальная сеть — на всей внешней выборке; наш объект её не видел.
        net = transfer_nn.FertilityNet(Z.shape[1], seed=config.TRANSFER_SEED)
        net.fit(Z, y, epochs=config.TRANSFER_EPOCHS, batch=config.TRANSFER_BATCH,
                lr=config.TRANSFER_LR, seed=config.TRANSFER_SEED)

        # --- Применение к Анабару в трёх режимах ---
        Xg = transfer_nn.feature_matrix(lon, lat)
        Xg_qm = np.column_stack([transfer_nn.quantile_match(Xg[:, j], X[:, j])
                                 for j in range(Xg.shape[1])])
        Xl_qm = transfer_nn.harmonize_local(df, X, lon, lat)
        for mode, M in (("global_raw", Xg), ("global_qm", Xg_qm),
                        ("local_qm", Xl_qm)):
            p = net.predict(transfer_nn.apply_stats(M, med, sc))
            scores[f"tr_{tag}_{mode}"] = np.nan_to_num(p, nan=np.nanmin(p))
        log(f"{tag}: скоры на сетке готовы (3 режима), "
            f"параметров сети {net.n_params()}")

    if not scores:
        raise SystemExit("ни один вариант меток не прошёл ворота — переносить нечего")

    # Ранговый ансамбль двух содержательных режимов основного варианта меток:
    # global_qm даёт крупную региональную рамку, local_qm — разрешение 500 м.
    a, b = scores.get("tr_AuU_global_qm"), scores.get("tr_AuU_local_qm")
    if a is not None and b is not None:
        scores["tr_AuU_ensemble"] = (sps.rankdata(a) + sps.rankdata(b)) / (2 * len(a))

    own = list(scores)
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    # --- Заверка тем же протоколом ---
    pts = assessment.load_verification_points(meta)
    all_cells = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    POINT_SETS = [("all", all_cells, assessment.point_clusters(all_cells, meta)),
                  ("unbiased_strict", unb, assessment.point_clusters(unb, meta))]
    alpha = 0.05 / config.OWN_N_CONFIGS_TOTAL
    log(f"точки: все {all_cells.size}, несмещённые {unb.size}; "
        f"порог Бонферрони 0.05/{config.OWN_N_CONFIGS_TOTAL} = {alpha:.5f}")

    def lift(score):
        return assessment.capture_efficiency(score, unb, pool=pool_valid)

    pa_df = pd.concat(
        [assessment.pa_curve(scores[n], c, pool=pool_valid).assign(
            method=n, points=pn)
         for n in scores for pn, c, _ in POINT_SETS], ignore_index=True)

    null_df = pd.DataFrame(
        [{"method": n, "points": pn,
          **assessment.spatial_null_pvalue(scores[n], c, meta, pool=pool_valid)}
         for n in scores for pn, c, _ in POINT_SETS])
    log(f"сдвиговый null ({config.VER_N_SHIFTS} сдвигов) готов")

    boot_df = pd.DataFrame(
        [{"a": n, "b": base, "points": pn, "resample": kind,
          **assessment.bootstrap_diff(scores[n], scores[base], c,
                                      pool=pool_valid, clusters=cl)}
         for n in own for base in ("criterial",)
         for pn, c, cls in POINT_SETS
         for kind, cl in (("point", None), ("cluster", cls))])
    log("бутстрэп готов")

    ver_rows = []
    for n in own:
        bt = boot_df[(boot_df["a"] == n) & (boot_df["points"] == "unbiased_strict")
                     & (boot_df["resample"] == "cluster")].iloc[0]
        p = null_df[(null_df["method"] == n)
                    & (null_df["points"] == "unbiased_strict")]["p"].iloc[0]
        ver_rows.append({"method": n, "lift": lift(scores[n]),
                         "lift_criterial": lift(scores["criterial"]),
                         "delta": bt["delta"], "ci_lo": bt["ci_lo"],
                         "ci_hi": bt["ci_hi"], "p_shift": p, "alpha": alpha,
                         "cond1_ci_lo_gt_0": bool(bt["ci_lo"] > 0),
                         "cond2_p_lt_alpha": bool(p < alpha),
                         "superior": bool(bt["ci_lo"] > 0 and p < alpha)})
    ver_df = pd.DataFrame(ver_rows).sort_values("lift", ascending=False)
    cv_df = pd.concat(cv_rows, ignore_index=True)

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "transfer"
    pa_df.to_csv(f"{stem}.csv", index=False)
    cv_df.to_csv(f"{stem}_cv.csv", index=False)
    null_df.to_csv(f"{stem}_null.csv", index=False)
    boot_df.to_csv(f"{stem}_boot.csv", index=False)
    ver_df.to_csv(f"{stem}_verdict.csv", index=False)
    np.savez_compressed(f"{stem}_scores.npz", valid=valid,
                        **{k: v for k, v in scores.items()})

    # --- Карта-панель ---
    plot = [n for n in own if n.startswith("tr_AuU")] + ["criterial"]
    plot = plot[:7] + ["criterial"] if len(plot) > 8 else plot
    ncol = min(4, len(plot))
    nrow = int(np.ceil(len(plot) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 5 * nrow),
                             squeeze=False)
    prow, pcol = df["row"].to_numpy()[unb], df["col"].to_numpy()[unb]
    for ax, name in zip(axes.ravel(), plot):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        im = ax.imshow(grid.reshape(meta.prf, meta.pic), cmap="magma",
                       vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^",
                   label="несмещённые точки заверки")
        ax.set_title(f"{name} (lift@10% = {lift(scores[name]):.2f})", fontsize=10)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Нейросеть, обученная на чужих метках (MRDS, США), перенесённая "
                 "на Анабар: нормированные ранги, север сверху", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "transfer_map.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    print("\nЧасть A — внешняя территория (США), блочная CV 2 градуса:")
    print(cv_df.groupby("labels")[["auc_nn", "auc_criterial", "lift_nn",
                                   "lift_criterial"]].mean().round(3).to_string())
    prim = pa_df[pa_df["area"] == config.VER_AREA]
    print(f"\nЧасть B — Анабар. Первичная метрика: "
          f"{config.VER_PRIMARY['metric']}, точки {config.VER_PRIMARY['points']}")
    print(prim.pivot(index="method", columns="points", values="lift")
          .reindex(list(scores)).round(2).to_string())
    print("\nВердикт по предзарегистрированному критерию превосходства:")
    print(ver_df[["method", "lift", "lift_criterial", "delta", "ci_lo", "ci_hi",
                  "p_shift", "superior"]].round(3).to_string(index=False))
    if ver_df["superior"].any():
        log("КРИТЕРИЙ ВЗЯТ: " + ", ".join(ver_df.loc[ver_df["superior"], "method"]))
    else:
        log("критерий не взят: методы неразличимы на имеющемся объёме заверки")
    log("сохранено: outputs/transfer_map.png, outputs/metrics/transfer*.csv")


if __name__ == "__main__":
    main()
