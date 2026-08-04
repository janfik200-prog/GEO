"""Этап 5c: контекст + направление. Патчевые модели переноса на Анабар.

ЧТО ПРОВЕРЯЕТСЯ. Два предыдущих этапа дали по половине решения:

* этап 5 (TinyMAE) — КОНТЕКСТ: сеть видела окрестность ячейки, но не знала,
  какой конец шкалы благоприятен, и подсвечивала «необычное» вместо «рудного»;
* этап 5b (перенос MLP) — НАПРАВЛЕНИЕ: сеть выучила благоприятность на тысячах
  чужих рудопроявлений, но видела всего пять чисел в точке, без окрестности.
  Точечная оценка впервые обошла критериальный прогноз (2.63 против 2.10), но
  планка значимости не была взята.

Здесь соединяется и то, и другое: классификатор учится на чужих метках (MRDS,
США), но на вход получает ПАТЧ пяти глобальных полей размером 56 x 56 км —
масштаб рудного района. Наш объект по-прежнему не участвует в обучении, поэтому
19 несмещённых точек остаются независимой заверкой.

Две архитектуры на одних и тех же патчах, чтобы отделить вклад контекста от
вклада свёрточной модели:

* ``ctx_mlp`` — тот же MLP, что в 5b, но на признаках окрестности (значение в
  центре + среднее и разброс каждого поля на трёх радиусах);
* ``cnn`` — свёрточная сеть, видящая патч целиком (``transfer_nn.FertilityCNN``).

Режим применения к Анабару — только ``qm`` (квантильное сопоставление каналов с
обучающим распределением): в 5b режим ``raw`` проиграл во всех вариантах, и это
уже измеренный факт, а не догадка.

КРИТЕРИЙ ПРЕВОСХОДСТВА — ТОТ ЖЕ, ЧТО В 4b, 5 И 5b, менять его под результат
нельзя. Первичная метрика ``config.VER_PRIMARY``: lift@10% на строго
несмещённых точках. Оба условия обязательны:

1. нижняя граница 90% бутстрэп-интервала разности (наш минус критериальный) при
   ресэмплинге по пространственным кластерам точек строго больше нуля;
2. сдвиговый null даёт p ниже 0.05/``config.OWN_N_CONFIGS_TOTAL``.

Знаменатель НАКОПИТЕЛЬНЫЙ (20 + 10 + 8 = 38 конфигураций, alpha = 0.00132):
все этапы меряются на одних и тех же 19 точках, поэтому порог обязан учитывать
весь перебор. Это цена стратегии «пробовать, пока не выиграет», и платить её
нужно честно.

Выход: ``outputs/metrics/transfer_cnn{,_cv,_null,_boot,_verdict}.csv``,
``outputs/transfer_cnn_map.png``, ``outputs/metrics/transfer_cnn_scores.npz``.

Запуск из корня: ``python -m experiments.transfer_cnn``.
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
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src import (assessment, cell_mask, config,  # noqa: E402
                 criterial_target, transfer_nn)
from experiments.common import criterial, lift as ext_lift  # noqa: E402
from experiments.transfer_nn import cell_lonlat  # noqa: E402

PATCH, STEP = config.TRANSFER_PATCH, config.TRANSFER_STEP_KM


def build_points(tag: str, commods, seed: int, log):
    """presence-background точки внешней территории + патчи вокруг них."""
    occ = transfer_nn.load_occurrences(commods, bbox=config.TRANSFER_BBOX)
    if len(occ) > config.TRANSFER_N_POS:
        occ = occ.sample(config.TRANSFER_N_POS, random_state=seed)
    rng = np.random.default_rng(seed)
    bb = config.TRANSFER_BBOX
    bl = rng.uniform(bb[0], bb[1], config.TRANSFER_N_BG)
    ba = rng.uniform(bb[2], bb[3], config.TRANSFER_N_BG)
    lon = np.r_[occ["lon"].to_numpy(), bl]
    lat = np.r_[occ["lat"].to_numpy(), ba]
    y = np.r_[np.ones(len(occ)), np.zeros(len(bl))].astype(np.int64)
    P = transfer_nn.extract_patches(lon, lat, PATCH, STEP)
    k = (PATCH - 1) // 2
    ok = np.isfinite(P[:, :, k, k]).all(axis=1)   # центр обязан быть измерен
    log(f"{tag}: точек {int(ok.sum())} (меток {int(y[ok].sum())}), "
        f"патчи {PATCH}x{PATCH} по {STEP:.0f} км = "
        f"{(PATCH - 1) * STEP:.0f} км, доля пропусков {np.isnan(P[ok]).mean():.3f}")
    return P[ok], y[ok], lon[ok], lat[ok]


def cv_scores(make, X, y, groups, epochs, batch, lr, log, name):
    """Блочная CV на внешней территории: модель против критериального индекса."""
    rows = []
    flat = X.reshape(len(X), -1)
    # Критериальный индекс — линейная свёртка со знаком корреляции; постоянные
    # столбцы (каналы валидности патча, где пропусков нет) дают деление на ноль
    # и NaN, поэтому в базовую линию они не идут.
    flat = flat[:, np.nanstd(flat, axis=0) > 1e-9]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.TRANSFER_SEED)
    for k, (tr, te) in enumerate(cv.split(flat, y, groups), start=1):
        net = make()
        net.fit(X[tr], y[tr], X[te], y[te], epochs=epochs, batch=batch, lr=lr,
                seed=config.TRANSFER_SEED)
        s = net.predict(X[te])
        s_cr = criterial(flat[tr], y[tr], flat[te])
        rows.append({"model": name, "fold": k, "auc": roc_auc_score(y[te], s),
                     "auc_criterial": roc_auc_score(y[te], s_cr),
                     "lift": ext_lift(s, y[te]),
                     "lift_criterial": ext_lift(s_cr, y[te])})
        log(f"  {name} фолд {k}: AUC {rows[-1]['auc']:.3f} / "
            f"критериальный {rows[-1]['auc_criterial']:.3f}")
    return pd.DataFrame(rows)


def qm_columns(M: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Поканальное квантильное сопоставление матрицы признаков с обучающей."""
    return np.column_stack([transfer_nn.quantile_match(M[:, j], ref[:, j])
                            for j in range(M.shape[1])])


def qm_patches(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Поканальное квантильное сопоставление патчей с обучающей выборкой.

    Единицы наших и обучающих полей несопоставимы по уровню, поэтому канал
    целиком (все узлы всех патчей) переводится в маргинал обучающего канала.
    Рисунок внутри патча при этом сохраняется — он и есть предмет свёртки.
    """
    out = np.empty_like(P)
    for j in range(P.shape[1]):
        src = P[:, j].ravel()
        rf = ref[:, j].ravel()
        out[:, j] = transfer_nn.quantile_match(src, rf).reshape(P[:, j].shape)
    return out


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
    lon, lat = cell_lonlat(df)

    # Патчи вокруг центров наших ячеек считаются один раз на все варианты меток.
    Pa = transfer_nn.extract_patches(lon, lat, PATCH, STEP)
    log(f"патчи Анабара: {Pa.shape}, доля пропусков {np.isnan(Pa).mean():.3f}")

    scores: dict[str, np.ndarray] = {}
    cv_rows = []
    for tag, commods in config.TRANSFER_COMMODS.items():
        P, y, olon, olat = build_points(tag, commods, config.TRANSFER_SEED, log)
        groups = transfer_nn.spatial_groups(olon, olat, config.TRANSFER_BLOCK_DEG)
        med, sc = transfer_nn.patch_stats(P)

        # --- ctx_mlp: признаки окрестности (центр + среднее и разброс на 3 радиусах)
        k = (PATCH - 1) // 2
        def ctx(P_):
            cols = [P_[:, :, k, k]]
            for rad in (max(1, k // 3), max(2, 2 * k // 3), k):
                sub = P_[:, :, k - rad:k + rad + 1, k - rad:k + rad + 1]
                cols.append(np.nanmean(sub, axis=(2, 3)))
                cols.append(np.nanstd(sub, axis=(2, 3)))
            return np.concatenate(cols, axis=1).astype(np.float64)

        Ctr = ctx(P)
        cmed, csc = transfer_nn.robust_stats(Ctr)
        Zc = transfer_nn.apply_stats(Ctr, cmed, csc)
        cv_rows.append(cv_scores(
            lambda: transfer_nn.FertilityNet(Zc.shape[1],
                                             seed=config.TRANSFER_SEED),
            Zc, y, groups, config.TRANSFER_EPOCHS, config.TRANSFER_BATCH,
            config.TRANSFER_LR, log, f"{tag}_ctx_mlp"))

        # --- cnn: патч целиком
        Xc = transfer_nn.prep_patches(P, med, sc)
        cv_rows.append(cv_scores(
            lambda: transfer_nn.FertilityCNN(Xc.shape[1], PATCH,
                                             hid=config.TRANSFER_CNN_HID,
                                             seed=config.TRANSFER_SEED),
            Xc, y, groups, 25, 256, 2e-3, log, f"{tag}_cnn"))

        # --- финальные модели на всей внешней выборке, применение к Анабару
        net_c = transfer_nn.FertilityNet(Zc.shape[1], seed=config.TRANSFER_SEED)
        net_c.fit(Zc, y, epochs=config.TRANSFER_EPOCHS,
                  batch=config.TRANSFER_BATCH, lr=config.TRANSFER_LR,
                  seed=config.TRANSFER_SEED)
        Ca = qm_columns(ctx(Pa), Ctr)
        p = net_c.predict(transfer_nn.apply_stats(Ca, cmed, csc))
        scores[f"cx_{tag}_ctx_mlp"] = np.nan_to_num(p, nan=float(np.nanmin(p)))

        net_n = transfer_nn.FertilityCNN(Xc.shape[1], PATCH,
                                         hid=config.TRANSFER_CNN_HID,
                                         seed=config.TRANSFER_SEED)
        net_n.fit(Xc, y, epochs=25, batch=256, lr=2e-3, seed=config.TRANSFER_SEED)
        p = net_n.predict(transfer_nn.prep_patches(qm_patches(Pa, P), med, sc))
        scores[f"cx_{tag}_cnn"] = np.nan_to_num(p, nan=float(np.nanmin(p)))
        log(f"{tag}: скоры на сетке готовы; параметров ctx_mlp "
            f"{net_c.n_params()}, cnn {net_n.n_params()}")

    def rank_mean(names):
        keys = [n for n in names if n in scores]
        return sum(sps.rankdata(scores[n]) for n in keys) / (len(keys) * len(df))

    scores["cx_AuU_ensemble"] = rank_mean(["cx_AuU_ctx_mlp", "cx_AuU_cnn"])
    scores["cx_all_ensemble"] = rank_mean(list(scores))

    own = list(scores)
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    alpha = 0.05 / config.OWN_N_CONFIGS_TOTAL
    res = assessment.run_protocol(scores, own, meta, valid, alpha, log=log)
    cv_df = pd.concat(cv_rows, ignore_index=True)

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "transfer_cnn"
    res["pa"].to_csv(f"{stem}.csv", index=False)
    cv_df.to_csv(f"{stem}_cv.csv", index=False)
    res["null"].to_csv(f"{stem}_null.csv", index=False)
    res["boot"].to_csv(f"{stem}_boot.csv", index=False)
    res["verdict"].to_csv(f"{stem}_verdict.csv", index=False)
    np.savez_compressed(f"{stem}_scores.npz", valid=valid, **scores)

    # --- Карта-панель ---
    unb = res["cells_unbiased"]
    plot = own[:7] + ["criterial"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    prow, pcol = df["row"].to_numpy()[unb], df["col"].to_numpy()[unb]
    pool = np.flatnonzero(valid)
    for ax, name in zip(axes.ravel(), plot):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        im = ax.imshow(grid.reshape(meta.prf, meta.pic), cmap="magma",
                       vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^",
                   label="несмещённые точки заверки")
        lf = assessment.capture_efficiency(scores[name], unb, pool=pool)
        ax.set_title(f"{name} (lift@10% = {lf:.2f})", fontsize=10)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Контекст + направление: сеть на патчах 56 км, обученная на "
                 "чужих метках (MRDS), против критериального прогноза", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "transfer_cnn_map.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    print("\nЧасть A — внешняя территория (США), блочная CV 2 градуса:")
    print(cv_df.groupby("model")[["auc", "auc_criterial", "lift",
                                  "lift_criterial"]].mean().round(3).to_string())
    prim = res["pa"][res["pa"]["area"] == config.VER_AREA]
    print(f"\nЧасть B — Анабар. Первичная метрика: "
          f"{config.VER_PRIMARY['metric']}, точки {config.VER_PRIMARY['points']}")
    print(prim.pivot(index="method", columns="points", values="lift")
          .reindex(list(scores)).round(2).to_string())
    print("\nВердикт по предзарегистрированному критерию превосходства:")
    print(res["verdict"][["method", "lift", "lift_criterial", "delta", "ci_lo",
                          "ci_hi", "p_shift", "superior"]].round(3)
          .to_string(index=False))
    if res["verdict"]["superior"].any():
        log("КРИТЕРИЙ ВЗЯТ: "
            + ", ".join(res["verdict"].loc[res["verdict"]["superior"], "method"]))
    else:
        log("критерий не взят: методы неразличимы на имеющемся объёме заверки")
    log("сохранено: outputs/transfer_cnn_map.png, "
        "outputs/metrics/transfer_cnn*.csv")


if __name__ == "__main__":
    main()
