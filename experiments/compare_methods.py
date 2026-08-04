"""Этап 7: сводное сравнение ВСЕХ методов фазы на одном эталоне.

ЗАЧЕМ ОТДЕЛЬНЫЙ ПРОГОН. Методы рождались этапами (лестница нетипичности,
TinyMAE, перенос с внешних меток, дистилляция, ансамбли по семенам, глубокие
модели без учителя), и каждый этап заверялся своим прогоном. Сравнивать их
между собой по этим таблицам нельзя: часть считалась по точечному lift, часть —
по критериальному эталону, а поправка на множественность у каждого этапа была
своя. Здесь все накопленные карты берутся ГОТОВЫМИ (ни одна модель не
переобучается) и прогоняются через один протокол с одной поправкой.

ЧТО ЗДЕСЬ ЕСТЬ И ЧЕГО НЕТ
-------------------------
Первичная метрика — согласие с критериальным прогнозом (AUC против его top-10%,
:mod:`src.crit_reference`). Столбца «превзошёл критериальный» нет и быть не
может: критериальный прогноз здесь эталон, его собственный AUC равен 1 по
построению. Вопрос этапа другой — какие методы и какие семейства методов
воспроизводят критериальные перспективные площади, ни разу их не видев.

Точечная метрика (lift@10% на 19 несмещённых точках) сохранена вторым
столбцом. Она НЕ первичная — этапы 5e и 6 показали, что её шаг (0.53 lift) и
разброс по семенам обучения перекрывают разницу между методами, — но убирать её
молча нельзя: читатель обязан видеть обе.

Выход: ``outputs/metrics/compare_methods{,_verdict,_family,_delta}.csv``,
``outputs/compare_methods.png``, ``outputs/compare_methods_map.png``.

Запуск из корня: ``python -m experiments.compare_methods``.
"""
import re
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
                 crit_reference, criterial_target)

NPZ = ["atypicality_v2_scores.npz", "atypicality_v3_scores.npz",
       "mae_scores.npz", "transfer_scores.npz", "transfer_cnn_scores.npz",
       "transfer_distil_scores.npz", "seed_stability_scores.npz",
       "crit_agreement_scores.npz"]
SKIP = {"valid", "emb", "centers", "criterial", "naive_hydro", "random"}

# Семейство метода определяется по префиксу имени карты — ровно тому, который
# присвоил её прогон-родитель. Порядок важен: первое совпадение выигрывает.
FAMILIES = [
    (r"^(v2|v3)_", "лестница нетипичности (эт. 4/4b)"),
    (r"^mae_", "TinyMAE, ошибка реконструкции (эт. 5)"),
    (r"^tr_", "перенос с меток MRDS (эт. 5b)"),
    (r"^cx_", "контекстный перенос, CNN (эт. 5c)"),
    (r"^(dm|ds)_", "дистилляция переноса (эт. 5d)"),
    (r"^se_", "ансамбли по семенам (эт. 5e)"),
    (r"^(vae|dagmm|dec|svdd|deep)_", "глубокие без учителя (эт. 7b)"),
]
BASELINES = {"naive_hydro": "планка: гидросеть",
             "random": "планка: случайная карта"}


def family_of(name: str) -> str:
    """Семейство карты по её имени; неизвестное имя — отдельная строка."""
    if name in BASELINES:
        return BASELINES[name]
    for pat, fam in FAMILIES:
        if re.match(pat, name):
            return fam
    return "прочее"


def load_all(log) -> dict[str, np.ndarray]:
    """Все сохранённые карты фазы. Ничего не обучается, только читается."""
    out: dict[str, np.ndarray] = {}
    for fname in NPZ:
        p = ROOT / "outputs" / "metrics" / fname
        if not p.exists():
            log(f"НЕТ {fname} — семейство в сравнение не войдёт")
            continue
        z = np.load(p)
        v = z["valid"].astype(bool)
        n = 0
        for k in z.files:
            if k in SKIP or z[k].ndim != 1:
                continue
            if k in out:
                log(f"дубль имени {k} в {fname} — пропущен")
                continue
            a = z[k].astype(float)
            out[k] = a if a.size == v.size else atypicality.expand_to_grid(a, v)
            n += 1
        log(f"{fname}: {n} карт")
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
    pool = np.flatnonzero(valid)
    crit = crit_reference.criterial_score(prognoz)
    ref = crit_reference.reference_mask(crit, pool)
    log(f"валидных ячеек {pool.size}; эталон = {int(ref[pool].sum())} ячеек "
        f"({ref[pool].mean():.1%} пула)")

    scores = load_all(log)
    own = list(scores)
    if not own:
        raise SystemExit("нет ни одной сохранённой карты — сначала прогоны этапов")
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    # Поправка на множественность — по числу претендентов ВСЕЙ фазы, а не
    # этапа: сравнение сводное, значит и цена перебора конфигураций сводная.
    alpha = 0.05 / (config.CREF_ORIENT_PENALTY * len(own))
    log(f"карт {len(scores)} (претендентов {len(own)}); "
        f"alpha = 0.05 / (2 x {len(own)}) = {alpha:.6f}")
    res = crit_reference.run_protocol(scores, crit, meta, valid, alpha, log=log)
    verdict = res["verdict"].copy()
    verdict["family"] = [family_of(m) for m in verdict["method"]]

    # --- Вторичная, точечная метрика: та же, что вела этапы 1-5e ---
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    verdict["lift_points"] = [
        assessment.capture_efficiency(
            crit_reference.orientation(scores[m], crit, pool) * scores[m],
            unb, pool=pool) for m in verdict["method"]]
    verdict["lift_points_criterial"] = assessment.capture_efficiency(
        crit, unb, pool=pool)
    log(f"точечная метрика на {unb.size} несмещённых точках добавлена")

    # --- Разность с лидером: блочный бутстрэп по лучшей карте каждого семейства ---
    lead = verdict.iloc[0]
    delta_rows = []
    for fam, grp in verdict[~verdict["method"].isin(BASELINES)].groupby("family"):
        best = grp.iloc[0]["method"]
        if best == lead["method"]:
            continue
        r = crit_reference.block_bootstrap(scores[lead["method"]], crit, meta,
                                           pool, score_b=scores[best])
        delta_rows.append({"family": fam, "leader": lead["method"],
                           "best_in_family": best, **r})
    delta_df = pd.DataFrame(delta_rows)
    log(f"разности лидера с {len(delta_df)} семействами посчитаны")

    fam_df = (verdict.groupby("family")
              .agg(n_maps=("method", "size"), auc_max=("auc", "max"),
                   auc_median=("auc", "median"),
                   best=("method", "first"),
                   n_reproduces=("reproduces", "sum"))
              .sort_values("auc_max", ascending=False).reset_index())

    out = ROOT / "outputs"
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out / "metrics" / "compare_methods"
    res["agree"].to_csv(f"{stem}.csv", index=False)
    verdict.to_csv(f"{stem}_verdict.csv", index=False)
    fam_df.to_csv(f"{stem}_family.csv", index=False)
    delta_df.to_csv(f"{stem}_delta.csv", index=False)

    # --- Рисунок 1: топ карт с интервалами, цвет = семейство ---
    top = verdict.head(20)
    fams = list(dict.fromkeys(verdict["family"]))
    cmap = plt.get_cmap("tab10")
    cidx = {f: cmap(i % 10) for i, f in enumerate(fams)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    yy = np.arange(len(top))
    ax1.barh(yy, top["auc"], color=[cidx[f] for f in top["family"]])
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
    ax1.set_title("Топ-20 карт фазы на едином эталоне\n"
                  f"усы — блочный бутстрэп, блок {config.CREF_BLOCK * 0.5:.0f} км")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3, axis="x")

    for f in fams:
        v = verdict[verdict["family"] == f]["auc"].to_numpy()
        ax2.scatter(v, np.full(v.size, fams.index(f))
                    + np.random.uniform(-.12, .12, v.size),
                    s=40, color=cidx[f], alpha=0.8)
    ax2.set_yticks(np.arange(len(fams)))
    ax2.set_yticklabels(fams, fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(0.5, color="gray", ls=":")
    ax2.axvline(config.CREF_AUC_MIN, color="crimson", lw=2)
    ax2.set_xlim(0.4, 1.0)
    ax2.set_xlabel("AUC против критериального эталона, безразм.")
    ax2.set_title("Все карты по семействам методов\n"
                  "разброс внутри семейства — цена выбора конфигурации")
    ax2.grid(alpha=0.3, axis="x")
    fig.suptitle("Этап 7: сводное сравнение методов фазы", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "compare_methods.png", dpi=130, bbox_inches="tight")

    # --- Рисунок 2: лидер, эталон, расхождения ---
    bname = lead["method"]
    bscore = lead["orient"] * scores[bname]
    agree_map = assessment.map_agreement(bscore, crit, config.CREF_AREA, pool)
    fig2, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    for ax, img, ttl in (
            (axes[0], sps.rankdata(bscore).reshape(meta.prf, meta.pic),
             f"лидер фазы: {bname} (AUC {lead['auc']:.3f})"),
            (axes[1], sps.rankdata(crit).reshape(meta.prf, meta.pic),
             "критериальный прогноз (эталон)")):
        ax.imshow(img, cmap="viridis")
        ax.set_title(ttl)
        ax.set_xlabel("столбец сетки (500 м)")
        ax.set_ylabel("строка сетки (500 м)")
    from matplotlib.colors import ListedColormap
    cmap3 = ListedColormap(["#eeeeee", "#1f77b4", "#d62728", "#2ca02c"])
    im = axes[2].imshow(agree_map.reshape(meta.prf, meta.pic), cmap=cmap3,
                        vmin=-0.5, vmax=3.5)
    axes[2].set_title(f"расхождения по top-{config.CREF_AREA:.0%}\n"
                      "синий — площади, которых нет у критериального")
    axes[2].set_xlabel("столбец сетки (500 м)")
    axes[2].set_ylabel("строка сетки (500 м)")
    cb = fig2.colorbar(im, ax=axes[2], fraction=0.046, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["ни один", "только лидер", "только эталон", "оба"])
    fig2.tight_layout()
    fig2.savefig(out / "compare_methods_map.png", dpi=130, bbox_inches="tight")

    cols = ["method", "family", "auc", "auc_ci_lo", "auc_ci_hi", "capture",
            "kappa", "p_shift", "reproduces", "lift_points"]
    print("\nСводная таблица (top-15):")
    print(verdict[cols].head(15).to_string(index=False,
                                           float_format=lambda x: f"{x:.3f}"))
    print("\nПо семействам:")
    print(fam_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    n_ok = int(verdict["reproduces"].sum())
    log(f"воспроизводят критериальный прогноз: {n_ok} карт из {len(own)}")
    log("готово: outputs/compare_methods.png, outputs/compare_methods_map.png, "
        "outputs/metrics/compare_methods*.csv")


if __name__ == "__main__":
    main()
