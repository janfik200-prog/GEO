"""Этап 7a/7b: глубокие модели без учителя, заверенные по критериальному эталону.

ПОЧЕМУ МЕНЯЕТСЯ ЗАВЕРКА, А НЕ ПОРОГ
-----------------------------------
Этапы 5e и 6 закрыли точечную заверку измерением, а не рассуждением: 19
несмещённых точек при площади 10% дают шаг метрики 0.53 lift, разброс одной и
той же модели по семенам обучения — 1-4 точки, а водосборы этих точек вышли
вырожденно мелкими (13 из 19 не крупнее двух ячеек), потому что точки стоят не
в руслах. На такой метрике сравнение методов невозможно в принципе.

Здесь эталоном становится карта критериального прогноза: 22 905 ячеек вместо 19
точек, метрика непрерывная. Обоснование, границы применимости и то, ЧЕГО такой
заверкой доказать нельзя, — в докстринге :mod:`src.crit_reference`. Коротко:
доказать превосходство над критериальным методом этой метрикой невозможно ни
при каком результате (он сам эталон), а доказать можно то, ради чего этап и
делается — что нейросетевой метод без учителя, ни разу не видевший ``prognoz``,
самостоятельно воспроизводит его перспективные площади.

ЧТО СРАВНИВАЕТСЯ
----------------
1. Четыре глубокие модели без учителя из :mod:`src.deep_unsup`, отобранные по
   обзорам MPM 2016-2025: VAE, DAGMM, DEC (два скора) и Deep SVDD. Каждая — на
   двух пулах признаков: v2 (без восьми факторных слоёв, из которых вычислен
   сам критериальный прогноз) и v3 (с ними). Разделение обязательно: согласие
   на пуле v3 частично тривиально — модель получает входы эталона; честный
   результат — это согласие на пуле v2.
2. Все карты, накопленные этапами 5-5e и сохранённые в ``*_scores.npz``.
3. Планки: наивная гидросеть и случайная карта.

Каждая глубокая модель обучается на ``config.DU_SEEDS`` семенах, претендентом
идёт РАНГОВЫЙ АНСАМБЛЬ (урок этапа 5e: одиночное семя — розыгрыш). Разброс AUC
по семенам сохраняется отдельной таблицей: это прямая проверка, что новая
метрика действительно перестала измерять шум оптимизации.

ПОРОГ ЗНАЧИМОСТИ фиксируется формулой до прогона: alpha = 0.05 / (2 * число
карт-претендентов). Двойка — плата за выбор ориентации скора по эталону
(``config.CREF_ORIENT_PENALTY``): у детектора без учителя направление шкалы не
определено, и знак выбирается на заверочных данных.

Выход: ``outputs/metrics/crit_agreement{,_seeds,_null,_boot,_verdict}.csv``,
``outputs/metrics/crit_agreement_scores.npz``, ``outputs/crit_agreement.png``,
``outputs/crit_agreement_map.png``.

Запуск из корня: ``python -m experiments.crit_agreement``.
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
from scipy import stats as sps  # noqa: E402

from src import (assessment, atypicality, cell_mask, config,  # noqa: E402
                 crit_reference, criterial_target, deep_unsup, features_v2)

NPZ = ["mae_scores.npz", "transfer_scores.npz", "transfer_cnn_scores.npz",
       "transfer_distil_scores.npz", "seed_stability_scores.npz"]
SKIP = {"valid", "emb", "centers", "criterial", "naive_hydro", "random"}


def load_saved(log) -> dict[str, np.ndarray]:
    """Карты прошлых этапов — без переобучения, меняется только заверка."""
    out: dict[str, np.ndarray] = {}
    for name in NPZ:
        p = ROOT / "outputs" / "metrics" / name
        if not p.exists():
            log(f"нет {name} — пропускаю")
            continue
        z = np.load(p)
        v = z["valid"].astype(bool)
        n = 0
        for k in z.files:
            if k in SKIP or z[k].ndim != 1:
                continue
            a = z[k].astype(float)
            out[f"old_{k}"] = a if a.size == v.size \
                else atypicality.expand_to_grid(a, v)
            n += 1
        log(f"{name}: {n} карт")
    return out


def mae_embedding(log) -> np.ndarray | None:
    """Эмбеддинг ячейки из self-supervised MAE этапа 5, робастно стандартизованный.

    ЗАЧЕМ ОН ЗДЕСЬ. Первый прогон этапа показал, что все четыре глубокие модели
    на векторе признаков ячейки стоят у случайного уровня (AUC 0.46-0.68). Это
    ровно тот дефицит, который был назван в этапе 5: детектор видит ячейку как
    набор чисел и ничего не знает о рисунке поля вокруг неё. Комбинация
    «self-supervised представление + детектор без учителя поверх него» — это
    схема GFM4MPM и линии Zhang et al. 2021 (свёрточный автоэнкодер без учителя
    как источник признаков), а не импровизация.
    """
    p = ROOT / "outputs" / "metrics" / "mae_scores.npz"
    if not p.exists():
        log("нет mae_scores.npz — пул с эмбеддингом MAE пропускается")
        return None
    z = np.load(p)
    E = z["emb"].astype(float)
    med = np.median(E, axis=0)
    q75, q25 = np.percentile(E, [75, 25], axis=0)
    scale = np.where(q75 - q25 > 0, q75 - q25, 1.0)
    log(f"эмбеддинг MAE: {E.shape}")
    return (E - med) / scale


def deep_maps(X: np.ndarray, valid: np.ndarray, tag: str,
              crit: np.ndarray, pool: np.ndarray,
              log) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Ансамбли по семенам для всех глубоких моделей на одном пуле признаков."""
    maps, rows = {}, []
    for name in deep_unsup.METHODS:
        t = time.time()
        ens, per_seed = deep_unsup.ensemble_score(name, X)
        for key, ens_score in ens.items():
            maps[f"{key}_{tag}"] = atypicality.expand_to_grid(ens_score, valid)
        # Разброс по семенам НА НОВОЙ МЕТРИКЕ — прямое сравнение с этапом 5e.
        for key, seeds in per_seed.items():
            for s, sc in zip(config.DU_SEEDS, seeds):
                g = atypicality.expand_to_grid(sc, valid)
                rows.append({"pool": tag, "method": key, "seed": s,
                             "auc": crit_reference.agreement(
                                 g, crit, pool)["auc"]})
        log(f"{name} на пуле {tag}: {len(ens)} карт за {time.time() - t:.0f}с")
    # Ранговый ансамбль всех глубоких моделей пула: карта «согласия семейств».
    members = {k: v for k, v in maps.items()}
    maps[f"deep_ensemble_{tag}"] = atypicality.expand_to_grid(
        atypicality.rank_ensemble({k: v[valid] for k, v in members.items()}),
        valid)
    return maps, rows


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
    meta, prognoz = criterial_target.load_prognoz_grid()
    pool = np.flatnonzero(valid)
    crit = crit_reference.criterial_score(prognoz)
    ref = crit_reference.reference_mask(crit, pool)
    log(f"ячеек {len(df)}, валидных {pool.size}; эталон "
        f"«критериально перспективно» = {int(ref[pool].sum())} ячеек "
        f"({ref[pool].mean():.1%} пула)")

    # --- Глубокие модели на двух пулах ---
    scores: dict[str, np.ndarray] = {}
    seed_rows: list[dict] = []
    pools: dict[str, np.ndarray] = {}
    for tag, with_factors in (("v2", False), ("v3", True)):
        feat = features_v2.pool_features(df, include_factors=with_factors)
        pools[tag] = atypicality.prepare_matrix(feat, valid)
        log(f"пул {tag}: {feat.shape[1]} признаков, матрица {pools[tag].shape}")
    E = mae_embedding(log)
    if E is not None:
        # Два пула с контекстом: только эмбеддинг (что даёт один рисунок поля)
        # и эмбеддинг вместе с признаками ячейки (контекст плюс значения).
        pools["mae"] = E
        pools["v3m"] = np.hstack([pools["v3"], E])
        log(f"пул v3m: {pools['v3m'].shape[1]} признаков "
            f"({pools['v3'].shape[1]} ячейки + {E.shape[1]} контекста)")

    for tag, X in pools.items():
        m, r = deep_maps(X, valid, tag, crit, pool, log)
        scores.update(m)
        seed_rows.extend(r)
    own_deep = list(scores)
    log(f"глубоких карт: {len(own_deep)}")

    # --- Карты прошлых этапов и планки ---
    scores.update(load_saved(log))
    own = list(scores)
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    # --- Протокол по эталону ---
    alpha = 0.05 / (config.CREF_ORIENT_PENALTY * len(own))
    log(f"карт к заверке {len(scores)} (претендентов {len(own)}); "
        f"alpha = 0.05 / (2 x {len(own)}) = {alpha:.6f}")
    res = crit_reference.run_protocol(scores, crit, meta, valid, alpha, log=log)
    seeds_df = pd.DataFrame(seed_rows)

    # --- Точечная заверка тех же карт: она никуда не делась, просто больше не
    # первичная. Показываем обе, чтобы не подменять метрику молча. ---
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    lift_cr = assessment.capture_efficiency(crit, unb, pool=pool)
    res["verdict"]["lift_points"] = [
        assessment.capture_efficiency(
            crit_reference.orientation(scores[m], crit, pool) * scores[m],
            unb, pool=pool) for m in res["verdict"]["method"]]
    res["verdict"]["lift_points_criterial"] = lift_cr

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "crit_agreement"
    res["agree"].to_csv(f"{stem}.csv", index=False)
    seeds_df.to_csv(f"{stem}_seeds.csv", index=False)
    res["null"].to_csv(f"{stem}_null.csv", index=False)
    res["boot"].to_csv(f"{stem}_boot.csv", index=False)
    res["verdict"].to_csv(f"{stem}_verdict.csv", index=False)
    np.savez_compressed(f"{stem}_scores.npz", valid=valid,
                        **{k: scores[k] for k in own_deep})

    # --- Рисунок 1: согласие с эталоном и разброс по семенам ---
    top = res["verdict"].head(14)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    yy = np.arange(len(top))
    colors = ["tab:green" if r else "tab:blue" for r in top["reproduces"]]
    ax1.barh(yy, top["auc"], color=colors)
    ax1.errorbar(top["auc"], yy,
                 xerr=[top["auc"] - top["auc_ci_lo"],
                       top["auc_ci_hi"] - top["auc"]],
                 fmt="none", ecolor="k", elinewidth=1, capsize=3)
    ax1.axvline(0.5, color="gray", ls=":", label="случайная карта (0.50)")
    ax1.axvline(config.CREF_AUC_MIN, color="crimson", lw=2,
                label=f"планка воспроизведения ({config.CREF_AUC_MIN})")
    ax1.set_yticks(yy)
    ax1.set_yticklabels(top["method"], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0.4, 1.0)
    ax1.set_xlabel("AUC против эталона «критериально перспективно», безразм.")
    ax1.set_title("Согласие карт с критериальным прогнозом\n"
                  f"эталон = top-{config.CREF_AREA:.0%} критериальной карты, "
                  f"{pool.size} ячеек; усы — блочный бутстрэп 10 км")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3, axis="x")

    if not seeds_df.empty:
        labels = [f"{m}|{p}" for p, m in
                  seeds_df[["pool", "method"]].drop_duplicates().to_numpy()]
        data = [seeds_df[(seeds_df["pool"] == lab.split("|")[1])
                         & (seeds_df["method"] == lab.split("|")[0])]["auc"]
                .to_numpy() for lab in labels]
        ax2.boxplot(data, tick_labels=labels, widths=0.5, vert=False)
        for i, d in enumerate(data, start=1):
            ax2.scatter(d, np.full(len(d), i)
                        + np.random.uniform(-.09, .09, len(d)),
                        s=16, c="tab:blue", alpha=0.7, zorder=3)
        ax2.tick_params(axis="y", labelsize=7)
    ax2.axvline(0.5, color="gray", ls=":")
    ax2.set_xlabel("AUC против критериального эталона, безразм.")
    ax2.set_title(f"Разброс метрики по {len(config.DU_SEEDS)} семенам\n"
                  "на этапе 5e тот же разброс перекрывал всю разницу методов")
    ax2.grid(alpha=0.3, axis="x")
    fig.suptitle("Заверка по критериальному прогнозу как эталонной карте", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "crit_agreement.png", dpi=130, bbox_inches="tight")

    # --- Рисунок 2: лучшая собственная карта, эталон и их расхождения ---
    best = res["verdict"][res["verdict"]["method"].isin(own_deep)].iloc[0]
    bname = best["method"]
    bscore = best["orient"] * scores[bname]
    agree_map = assessment.map_agreement(bscore, crit, config.CREF_AREA, pool)
    fig2, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    for ax, img, ttl in (
            (axes[0], sps.rankdata(bscore).reshape(meta.prf, meta.pic),
             f"{bname} (AUC {best['auc']:.3f})"),
            (axes[1], sps.rankdata(crit).reshape(meta.prf, meta.pic),
             f"критериальный прогноз (эталон)")):
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(ttl)
        ax.set_xlabel("столбец сетки (500 м)")
        ax.set_ylabel("строка сетки (500 м)")
        plt.colorbar(im, ax=ax, fraction=0.046, label="ранг скора")
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#eeeeee", "#1f77b4", "#d62728", "#2ca02c"])
    im = axes[2].imshow(agree_map.reshape(meta.prf, meta.pic), cmap=cmap,
                        vmin=-0.5, vmax=3.5)
    axes[2].set_title("Расхождения top-10%\n"
                      "синий — только нейросеть (новые площади), "
                      "красный — только критериальный")
    axes[2].set_xlabel("столбец сетки (500 м)")
    cb = plt.colorbar(im, ax=axes[2], fraction=0.046, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["ни один", "только НС", "только крит.", "оба"])
    fig2.suptitle("Собственная карта против критериального прогноза", y=1.0)
    fig2.tight_layout()
    fig2.savefig(out / "crit_agreement_map.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    cols = ["method", "auc", "auc_ci_lo", "auc_ci_hi", "capture", "kappa",
            "rho", "orient", "p_shift", "reproduces", "lift_points"]
    print("\nСогласие с критериальным эталоном (top-14):")
    print(res["verdict"][cols].head(14).round(3).to_string(index=False))
    if not seeds_df.empty:
        print("\nРазброс AUC по семенам (одна и та же конфигурация):")
        print(seeds_df.groupby(["pool", "method"])["auc"]
              .agg(["min", "median", "max", "std"]).round(3).to_string())
    ok = res["verdict"][res["verdict"]["reproduces"]
                        & res["verdict"]["method"].isin(own_deep)]
    if len(ok):
        log("ВОСПРОИЗВОДЯТ критериальный прогноз: "
            + ", ".join(ok["method"].head(8)))
    else:
        log("ни одна собственная карта не взяла планку воспроизведения")
    log("сохранено: outputs/crit_agreement.png, outputs/crit_agreement_map.png, "
        "outputs/metrics/crit_agreement*.csv")


if __name__ == "__main__":
    main()
