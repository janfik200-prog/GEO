"""Этап 5e: сколько в разнице методов от метода, а сколько от случайного семени.

ЗАЧЕМ ЭТО НУЖНО. Этапы 5c и 5d обучали ОДНУ И ТУ ЖЕ модель (контекстный MLP на
патчах 56 км, обученный на MRDS) практически одинаково: разница только в том,
что в 5d градиент взвешивался весами доменного соответствия, а веса вышли почти
единичными — эффективный объём выборки 29 998 из 29 998. По любому разумному
счёту это одна и та же модель. Но lift@10% на 19 несмещённых точках дал 2.63 в
5c и 1.58 в 5d — ровно одну точку разницы, столько же, сколько разделяет наш
лучший результат и критериальный прогноз (2.63 против 2.10).

Отсюда прямой вопрос, который надо задать ДО того, как объявлять победителя:
какую долю наблюдаемых различий между методами создаёт не метод, а случайность
обучения — инициализация весов и порядок мини-батчей? Если разброс по семенам
сопоставим с разницей между методами, то на имеющемся объёме заверки методы
неразличимы в принципе, и никакой перебор архитектур этого не исправит.

ЧТО ДЕЛАЕТСЯ. Одна и та же конфигурация обучается ``config.SEED_N`` раз с
разными семенами; каждая обученная сеть применяется к листу, и для каждой
считается первичная метрика. Получается распределение lift по семенам при
полностью фиксированных данных, признаках и гиперпараметрах. Критериальный
прогноз детерминирован — у него ровно одно значение, это горизонтальная линия
сравнения.

АНСАМБЛЬ ПО СЕМЕНАМ — не просто диагностика, а осмысленное улучшение метода:
усреднение рангов по 20 обучениям убирает именно ту часть разброса, которая
порождена оптимизацией, и оставляет то, что модель действительно выучила.
Именно ансамбль (а не лучшее семя!) идёт в протокол как конфигурация-претендент.
Выбор лучшего семени по заверочной метрике был бы подгонкой под тест.

Знаменатель Бонферрони накопительный: 20 + 10 + 8 + 8 + 4 = 50 конфигураций,
alpha = 0.001. Таблица lift по отдельным семенам — диагностика, в знаменатель не
идёт, потому что ни одно отдельное семя не предъявляется как метод.

Выход: ``outputs/metrics/seed_stability{,_seeds,_null,_boot,_verdict}.csv``,
``outputs/seed_stability.png``.

Запуск из корня: ``python -m experiments.seed_stability``.
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

from src import (assessment, cell_mask, config,  # noqa: E402
                 criterial_target, transfer_nn)
from experiments.transfer_cnn import build_points, qm_columns  # noqa: E402
from experiments.transfer_nn import cell_lonlat  # noqa: E402

PATCH, STEP = config.TRANSFER_PATCH, config.TRANSFER_STEP_KM


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
    pool = np.flatnonzero(valid)

    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    step = 1.0 / unb.size / config.VER_AREA
    log(f"несмещённых точек {unb.size}; шаг метрики 1 точка = {step:.2f} lift")

    Pa = transfer_nn.extract_patches(lon, lat, PATCH, STEP)
    Ca_raw = transfer_nn.context_from_patches(Pa)
    log(f"патчи Анабара: {Pa.shape}")

    scores: dict[str, np.ndarray] = {}
    seed_rows = []
    for tag, commods in config.TRANSFER_COMMODS.items():
        P, y, olon, olat = build_points(tag, commods, config.TRANSFER_SEED, log)
        Ctr = transfer_nn.context_from_patches(P)
        cmed, csc = transfer_nn.robust_stats(Ctr)
        Zc = transfer_nn.apply_stats(Ctr, cmed, csc)
        Za = transfer_nn.apply_stats(qm_columns(Ca_raw, Ctr), cmed, csc)

        ranks = np.zeros(len(df))
        for s in config.SEED_LIST:
            net = transfer_nn.FertilityNet(Zc.shape[1], seed=s)
            net.fit(Zc, y, epochs=config.TRANSFER_EPOCHS,
                    batch=config.TRANSFER_BATCH, lr=config.TRANSFER_LR, seed=s)
            p = net.predict(Za)
            p = np.nan_to_num(p, nan=float(np.nanmin(p)))
            lf = assessment.capture_efficiency(p, unb, pool=pool)
            seed_rows.append({"tag": tag, "seed": s, "lift": lf,
                              "points_hit": int(round(lf * config.VER_AREA
                                                      * unb.size))})
            ranks += sps.rankdata(p) / len(df)
            log(f"  {tag} семя {s:2d}: lift@10% = {lf:.2f} "
                f"({seed_rows[-1]['points_hit']} точек из {unb.size})")
        scores[f"se_{tag}"] = ranks / len(config.SEED_LIST)
        v = [r["lift"] for r in seed_rows if r["tag"] == tag]
        log(f"{tag}: по семенам медиана {np.median(v):.2f}, "
            f"размах {min(v):.2f}-{max(v):.2f}, СКО {np.std(v):.2f}; "
            f"ансамбль {assessment.capture_efficiency(scores[f'se_{tag}'], unb, pool=pool):.2f}")

    scores["se_all_ensemble"] = (
        sum(sps.rankdata(scores[f"se_{t}"]) for t in config.TRANSFER_COMMODS)
        / (len(config.TRANSFER_COMMODS) * len(df)))

    own = list(scores)
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    alpha = 0.05 / config.OWN_N_CONFIGS_TOTAL
    res = assessment.run_protocol(scores, own, meta, valid, alpha, log=log)
    seeds_df = pd.DataFrame(seed_rows)

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "seed_stability"
    res["pa"].to_csv(f"{stem}.csv", index=False)
    seeds_df.to_csv(f"{stem}_seeds.csv", index=False)
    res["null"].to_csv(f"{stem}_null.csv", index=False)
    res["boot"].to_csv(f"{stem}_boot.csv", index=False)
    res["verdict"].to_csv(f"{stem}_verdict.csv", index=False)
    # Ансамбли по семенам — единственные претенденты этого этапа, поэтому их
    # карты сохраняются: этап 6 заверяет по водосборам уже готовые карты, и без
    # этого файла в водосборное сравнение попали бы только одиночные семена.
    np.savez_compressed(f"{stem}_scores.npz", valid=valid,
                        **{k: scores[k] for k in own})

    # --- Рисунок: разброс по семенам против разницы между методами ---
    lf_cr = assessment.capture_efficiency(scores["criterial"], unb, pool=pool)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    tags = list(config.TRANSFER_COMMODS)
    data = [seeds_df.loc[seeds_df["tag"] == t, "lift"].to_numpy() for t in tags]
    ax1.boxplot(data, tick_labels=tags, widths=0.5)
    for i, d in enumerate(data, start=1):
        ax1.scatter(np.full(len(d), i) + np.random.uniform(-.09, .09, len(d)),
                    d, s=18, c="tab:blue", alpha=0.7, zorder=3)
    ax1.axhline(lf_cr, color="crimson", lw=2,
                label=f"критериальный прогноз ({lf_cr:.2f})")
    ax1.axhline(1.0, color="gray", ls=":", label="случайный выбор (1.00)")
    ax1.set_ylabel("lift@10% на 19 несмещённых точках, безразм.")
    ax1.set_xlabel("набор внешних меток обучения")
    ax1.set_title(f"Разброс метрики по {config.SEED_N} семенам\n"
                  "при полностью фиксированных данных и гиперпараметрах")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    for t, d in zip(tags, data):
        ax2.plot(sorted(d), np.linspace(0, 1, len(d)), marker="o", ms=4,
                 label=f"{t} (СКО {np.std(d):.2f})")
    ax2.axvline(lf_cr, color="crimson", lw=2, label="критериальный прогноз")
    ax2.axvspan(lf_cr - step / 2, lf_cr + step / 2, color="crimson", alpha=0.12,
                label=f"шаг метрики = 1 точка = {step:.2f}")
    ax2.set_xlabel("lift@10%, безразм.")
    ax2.set_ylabel("эмпирическая функция распределения, доля семян")
    ax2.set_title("Накопленное распределение метрики по семенам")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig.suptitle("Разрешающая способность заверки: случайность обучения "
                 "против разницы между методами", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "seed_stability.png", dpi=130, bbox_inches="tight")

    # --- Сводка ---
    print("\nРазброс первичной метрики по семенам (одна и та же модель):")
    print(seeds_df.groupby("tag")["lift"]
          .agg(["min", "median", "max", "std"]).round(3).to_string())
    print(f"\nКритериальный прогноз (детерминирован): {lf_cr:.3f}")
    print(f"Шаг метрики: 1 точка = {step:.3f} lift")
    print("\nДоля семян, у которых модель обошла критериальный прогноз:")
    print(seeds_df.assign(win=seeds_df["lift"] > lf_cr)
          .groupby("tag")["win"].mean().round(3).to_string())
    print("\nВердикт по предзарегистрированному критерию (ансамбли по семенам):")
    print(res["verdict"][["method", "lift", "lift_criterial", "delta", "ci_lo",
                          "ci_hi", "p_shift", "superior"]].round(3)
          .to_string(index=False))
    if res["verdict"]["superior"].any():
        log("КРИТЕРИЙ ВЗЯТ: "
            + ", ".join(res["verdict"].loc[res["verdict"]["superior"], "method"]))
    else:
        log("критерий не взят")
    log("сохранено: outputs/seed_stability.png, "
        "outputs/metrics/seed_stability*.csv")


if __name__ == "__main__":
    main()
