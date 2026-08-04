"""Этап 5d: доменное перевзвешивание + дистилляция направления в локальный пул.

ЧТО ПРОВЕРЯЕТСЯ. Этап 5c показал предел переноса «как есть»: на чужой
территории сеть уверенно бьёт критериальный индекс (AUC 0.94 против 0.88), а на
нашем листе выигрывает всего одну точку из девятнадцати. Два узких места видны
прямо из постановки, и оба лечатся, не трогая заверку:

* УЧИТЕЛЬ УЧИТСЯ НЕ НА ТЕХ ОБСТАНОВКАХ. Обучающая выборка — вся территория США,
  включая кордильерские обстановки, которых на архейском щите нет. Правило,
  выученное на них, переносить не на что. Логистический дискриминатор
  «источник или цель» даёт отношение плотностей, им перевзвешивается обучение
  (``transfer_nn.domain_weights``), и сеть уделяет внимание той части
  признакового пространства, которая на листе реально встречается.
* УЧИТЕЛЬ ВИДИТ СЛИШКОМ ГРУБО. Пять глобальных полей с шагом 2' — на широте 71
  это около 1.2 км, вдвое грубее нашей сетки, тогда как на листе есть 52
  локальных признака с шагом 500 м. Сеть-ученик (``transfer_nn.DistilNet``)
  учится воспроизводить логит учителя по локальным признакам и выражает то же
  направление на том разрешении, на котором данные листа измерены.

ПОЧЕМУ ЭТО НЕ УТЕЧКА. Ни учитель, ни ученик не видят ни одной метки объекта:
учитель обучен на MRDS (США), цель ученика — выход учителя, а не заверочные
точки. Перевзвешивание использует только признаки ячеек листа, без меток.
19 несмещённых точек остаются полностью независимой проверкой.

Ученик проверяется блочным holdout'ом внутри листа (блоки 10 км): нужно знать,
воспроизводит ли он учителя за пределами обученных блоков или просто запоминает
карту. Это диагностика дистилляции, к заверке она отношения не имеет.

КРИТЕРИЙ ПРЕВОСХОДСТВА — ТОТ ЖЕ, ЧТО В 4b, 5, 5b И 5c. Первичная метрика
``config.VER_PRIMARY``: lift@10% на строго несмещённых точках. Оба условия
обязательны:

1. нижняя граница 90% бутстрэп-интервала разности (наш минус критериальный) при
   ресэмплинге по пространственным кластерам точек строго больше нуля;
2. сдвиговый null даёт p ниже 0.05/``config.OWN_N_CONFIGS_TOTAL``.

Знаменатель НАКОПИТЕЛЬНЫЙ (20 + 10 + 8 + 8 = 46 конфигураций, alpha = 0.00109):
все этапы меряются на одних и тех же 19 точках, поэтому порог обязан учитывать
весь перебор. Это цена стратегии «пробовать, пока не выиграет».

Выход: ``outputs/metrics/transfer_distil{,_cv,_null,_boot,_verdict}.csv``,
``outputs/transfer_distil_map.png``, ``outputs/metrics/transfer_distil_scores.npz``.

Запуск из корня: ``python -m experiments.transfer_distil``.
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

from src import (assessment, atypicality, cell_mask, config,  # noqa: E402
                 criterial_target, features_v2, transfer_nn)
from experiments.common import criterial, lift as ext_lift  # noqa: E402
from experiments.transfer_cnn import build_points, qm_columns  # noqa: E402
from experiments.transfer_nn import cell_lonlat  # noqa: E402

PATCH, STEP = config.TRANSFER_PATCH, config.TRANSFER_STEP_KM
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Вероятность -> логит; шкала логита линейна по вкладам и лучше для MSE."""
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def cv_weighted(Z, y, groups, w, log, name) -> pd.DataFrame:
    """Блочная CV на внешней территории для перевзвешенной модели.

    Веса берутся только из обучающей части фолда: вес объекта зависит от всей
    обучающей выборки, и подмешивать в него отложенные объекты нельзя.
    """
    rows = []
    flat = Z.reshape(len(Z), -1)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.TRANSFER_SEED)
    for k, (tr, te) in enumerate(cv.split(flat, y, groups), start=1):
        net = transfer_nn.FertilityNet(Z.shape[1], hidden=config.DISTIL_HID,
                                       seed=config.TRANSFER_SEED)
        net.fit(Z[tr], y[tr], Z[te], y[te], epochs=config.TRANSFER_EPOCHS,
                batch=config.TRANSFER_BATCH, lr=config.TRANSFER_LR,
                seed=config.TRANSFER_SEED, w=w[tr])
        s = net.predict(Z[te])
        s_cr = criterial(flat[tr], y[tr], flat[te])
        rows.append({"model": name, "fold": k, "auc": roc_auc_score(y[te], s),
                     "auc_criterial": roc_auc_score(y[te], s_cr),
                     "lift": ext_lift(s, y[te]),
                     "lift_criterial": ext_lift(s_cr, y[te])})
        log(f"  {name} фолд {k}: AUC {rows[-1]['auc']:.3f} / "
            f"критериальный {rows[-1]['auc_criterial']:.3f}")
    return pd.DataFrame(rows)


def block_groups(df: pd.DataFrame, side: int) -> np.ndarray:
    """Пространственные блоки ячеек листа (side x side ячеек = 10 км при 20)."""
    return ((df["row"].to_numpy() // side) * 10_000
            + df["col"].to_numpy() // side)


def distil(Xl: np.ndarray, target: np.ndarray, groups: np.ndarray,
           log, name) -> tuple[np.ndarray, float]:
    """Обучить ученика на логит учителя; вернуть скор и блочный R^2.

    R^2 считается по отложенным блокам 10 км: он отвечает на вопрос, переносим
    ли выученный пересчёт «локальные признаки -> направление» на не виденные
    участки листа. Итоговый скор берётся с модели, обученной на всём листе, —
    метки заверки в обучении не участвуют ни в одном из вариантов.
    """
    cv = StratifiedGroupKFold(4, shuffle=True, random_state=config.VER_SEED)
    strat = (target > np.median(target)).astype(int)
    r2 = []
    for tr, te in cv.split(Xl, strat, groups):
        st = transfer_nn.DistilNet(Xl.shape[1], hidden=config.DISTIL_HID,
                                   seed=config.TRANSFER_SEED)
        st.fit(Xl[tr], target[tr], Xl[te], target[te],
               epochs=config.DISTIL_EPOCHS, batch=config.DISTIL_BATCH,
               lr=config.DISTIL_LR, seed=config.TRANSFER_SEED)
        pred = st.predict(Xl[te])
        ss = ((target[te] - target[te].mean()) ** 2).sum()
        r2.append(1.0 - ((target[te] - pred) ** 2).sum() / max(ss, EPS))
    st = transfer_nn.DistilNet(Xl.shape[1], hidden=config.DISTIL_HID,
                              seed=config.TRANSFER_SEED)
    st.fit(Xl, target, epochs=config.DISTIL_EPOCHS, batch=config.DISTIL_BATCH,
           lr=config.DISTIL_LR, seed=config.TRANSFER_SEED)
    log(f"  {name}: ученик {st.n_params()} параметров, блочный R2 "
        f"{np.mean(r2):.3f} (по фолдам {np.round(r2, 3).tolist()})")
    return st.predict(Xl), float(np.mean(r2))


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

    # --- локальный пул v3: то, чего учитель не видит принципиально ---
    feat = features_v2.pool_features(df, include_factors=True)
    Xl = atypicality.prepare_matrix(feat, valid)
    groups_l = block_groups(df, config.DISTIL_BLOCK_CELLS)[valid]
    log(f"локальный пул: {Xl.shape[1]} признаков на {Xl.shape[0]} валидных "
        f"ячейках, блоков {np.unique(groups_l).size} по "
        f"{config.DISTIL_BLOCK_CELLS * 0.5:.0f} км")

    Pa = transfer_nn.extract_patches(lon, lat, PATCH, STEP)
    Ca_raw = transfer_nn.context_from_patches(Pa)
    log(f"патчи Анабара: {Pa.shape}, доля пропусков {np.isnan(Pa).mean():.3f}")

    scores: dict[str, np.ndarray] = {}
    cv_rows, diag = [], []
    for tag, commods in config.TRANSFER_COMMODS.items():
        P, y, olon, olat = build_points(tag, commods, config.TRANSFER_SEED, log)
        groups = transfer_nn.spatial_groups(olon, olat, config.TRANSFER_BLOCK_DEG)
        Ctr = transfer_nn.context_from_patches(P)
        cmed, csc = transfer_nn.robust_stats(Ctr)
        Zc = transfer_nn.apply_stats(Ctr, cmed, csc)
        # Цель приводится к маргиналу обучения ДО расчёта весов: сравнивать
        # плотности в разных единицах (pgrid против нТл) бессмысленно.
        Za = transfer_nn.apply_stats(qm_columns(Ca_raw, Ctr), cmed, csc)

        w = transfer_nn.domain_weights(Zc, Za[valid],
                                       clip=config.DISTIL_W_CLIP,
                                       seed=config.TRANSFER_SEED)
        eff = float(w.sum() ** 2 / (w ** 2).sum())
        log(f"{tag}: доменные веса, эффективный объём выборки {eff:.0f} из "
            f"{len(w)} ({eff / len(w):.2f}), доля меток в весе "
            f"{w[y == 1].sum() / w.sum():.3f}")

        cv_rows.append(cv_weighted(Zc, y, groups, w, log, f"{tag}_dm_ctx"))

        net = transfer_nn.FertilityNet(Zc.shape[1], hidden=config.DISTIL_HID,
                                       seed=config.TRANSFER_SEED)
        net.fit(Zc, y, epochs=config.TRANSFER_EPOCHS,
                batch=config.TRANSFER_BATCH, lr=config.TRANSFER_LR,
                seed=config.TRANSFER_SEED, w=w)
        p = net.predict(Za)
        p = np.nan_to_num(p, nan=float(np.nanmin(p)))
        scores[f"dm_{tag}_ctx"] = p

        # --- ученик: логит доменного учителя по локальным признакам 500 м ---
        s_st, r2 = distil(Xl, logit(p[valid]), groups_l, log, f"ds_{tag}")
        scores[f"ds_{tag}"] = atypicality.expand_to_grid(s_st, valid)
        rho = assessment.spearman_maps(scores[f"ds_{tag}"], scores[f"dm_{tag}_ctx"],
                                       pool=np.flatnonzero(valid))
        diag.append({"tag": tag, "n_eff": eff, "n_src": len(w),
                     "distil_block_r2": r2, "rho_student_teacher": rho,
                     "teacher_params": net.n_params()})
        log(f"{tag}: ученик против учителя, Spearman карт {rho:.3f}")

    def rank_mean(names):
        keys = [n for n in names if n in scores]
        return sum(sps.rankdata(scores[n]) for n in keys) / (len(keys) * len(df))

    scores["dm_all_ensemble"] = rank_mean([f"dm_{t}_ctx"
                                           for t in config.TRANSFER_COMMODS])
    scores["ds_all_ensemble"] = rank_mean([f"ds_{t}"
                                           for t in config.TRANSFER_COMMODS])

    own = list(scores)
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    alpha = 0.05 / config.OWN_N_CONFIGS_TOTAL
    res = assessment.run_protocol(scores, own, meta, valid, alpha, log=log)
    cv_df = pd.concat(cv_rows, ignore_index=True)
    diag_df = pd.DataFrame(diag)

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "transfer_distil"
    res["pa"].to_csv(f"{stem}.csv", index=False)
    cv_df.to_csv(f"{stem}_cv.csv", index=False)
    diag_df.to_csv(f"{stem}_diag.csv", index=False)
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
    fig.suptitle("Доменное перевзвешивание и дистилляция направления в "
                 "локальный пул 500 м против критериального прогноза", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "transfer_distil_map.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    print("\nЧасть A — внешняя территория (США), блочная CV 2 градуса, "
          "перевзвешенное обучение:")
    print(cv_df.groupby("model")[["auc", "auc_criterial", "lift",
                                  "lift_criterial"]].mean().round(3).to_string())
    print("\nДиагностика переноса и дистилляции:")
    print(diag_df.round(3).to_string(index=False))
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
    log("сохранено: outputs/transfer_distil_map.png, "
        "outputs/metrics/transfer_distil*.csv")


if __name__ == "__main__":
    main()
