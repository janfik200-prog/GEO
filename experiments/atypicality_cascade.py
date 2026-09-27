"""Каскад по группам признаков (память + внимание) — предметный порядок стадий.

Продолжение обсуждения 20-27.08.2026: плоская лестница нетипичности
(`experiments/atypicality_v6.py`) считает все признаки одним пулом. Здесь —
последовательный каскад из 5 стадий, порядок задан предметной (геологической)
логикой "воронки разведки" (от глубинного контроля к поверхностному
отклику), не эмпирической переставляемостью (`feature_group_dependencies.py`,
которая та же логика подтвердила по краям цепочки, но не по порядку
внутри — см. отчёт пользователю 27.08.2026):

1. **геофизика** (gm, pf) — гравика/магнитка, глубинная структура фундамента;
2. **структура** (dist, lin, ter, relief2) — поверхностное выражение тех же
   разломных систем (гидросеть, линеаменты, рельеф);
3. **минералогия изменений** (ast, opt) — химический отпечаток
   рудоносного флюида (ASTER SWIR/TIR); по эмпирической диагностике —
   единственная группа, которую НИЧЕМ нельзя заменить (explained_by_rest=0.03);
4. **радар** (s1, psr) — текстура/влажность поверхности;
5. **оптика общего назначения** (l8, ls, s2, s2raw) — самый неспецифичный
   слой, сильнее всего зависит от растительности/землепользования.

Механизм памяти/внимания (минимальная реализация, без нейросети —
детекторы уже используют пул детекторов из `src/atypicality.py`, каскад —
это оркестрация вокруг них):

* **память** — после каждой стадии в неё добавляются первые ≤3 главных
  компоненты СЫРЫХ (не остаточных) признаков стадии; следующая стадия
  регрессирует (Ridge) свои сырые признаки на всю накопленную память и
  детектор нетипичности получает НЕВЯЗКУ — то, что предыдущие стадии ещё не
  объяснили;
* **внимание** — вес стадии в итоговом ансамбле = доля необъяснённой памятью
  дисперсии (1 - R² регрессии Ridge), т.е. чем новее сигнал стадии, тем
  сильнее её голос в объединённом скоре.

5 кандидатов (зафиксировано до прогона, alpha = 0.05/5 = 0.01):
``cascade_attention``, ``cascade_attention_dom`` (то же + доменная
нормировка каждой стадии), ``cascade_uniform``/``cascade_uniform_dom`` —
контроль без памяти/внимания (равные веса, детектор видит сырые признаки
стадии), ``ast_opt_only`` — работает ли одна самая информативная стадия
(минералогия изменений) сама по себе не хуже целого каскада.

Запуск из корня: ``python -m experiments.atypicality_cascade``.
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
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from src import assessment, atypicality, cell_mask, config, criterial_target, features_v2  # noqa: E402

SERVICE_COLS = ("row", "col", "x", "y", "mask_svita")

STAGES = [
    ("геофизика", ["gm", "pf"]),
    ("структура", ["dist", "lin", "ter", "relief2"]),
    ("минералогия_изменений", ["ast", "opt"]),
    ("радар", ["s1", "psr"]),
    ("оптика", ["l8", "ls", "s2", "s2raw"]),
]


def build_stage_cols(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        if c in SERVICE_COLS:
            continue
        g = features_v2.feature_group(c)
        groups.setdefault(g, []).append(c)
    out = []
    for name, gs in STAGES:
        cols = [c for g in gs for c in groups.get(g, [])]
        out.append((name, cols))
    return out


def weighted_rank_ensemble(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    n = next(iter(scores.values())).size
    acc = np.zeros(n)
    wsum = sum(weights.values())
    for k, s in scores.items():
        acc += (weights[k] / wsum) * (sps.rankdata(s) / n)
    return acc


def safe_mahalanobis(X: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Как `atypicality.robust_mahalanobis`, но с запасным вариантом.

    На маленьких доменных срезах (per_domain) робастная MinCovDet иногда
    падает с "covariance matrix of the support data is equal to 0" — часть
    признаков внутри домена вырождена (почти константа). Запасной вариант —
    обычная (не робастная) ковариация: менее устойчива к выбросам, но не
    падает; такое переключение — крайний случай, а не основной путь.
    """
    try:
        return atypicality.robust_mahalanobis(X, seed=seed)
    except ValueError:
        from sklearn.covariance import EmpiricalCovariance
        return EmpiricalCovariance().fit(X).mahalanobis(X)


def stage_detector(X: np.ndarray, domains: np.ndarray | None) -> np.ndarray:
    """Mahalanobis + IsolationForest, ранговый ансамбль; опционально по доменам."""
    if domains is None:
        s1 = safe_mahalanobis(X, seed=config.ANOM_SEED)
        s2 = atypicality.isoforest_score(X, seed=config.ANOM_SEED)
    else:
        s1 = atypicality.per_domain(safe_mahalanobis, X, domains, seed=config.ANOM_SEED)
        s2 = atypicality.per_domain(atypicality.isoforest_score, X, domains, seed=config.ANOM_SEED)
    return atypicality.rank_ensemble({"mahal": s1, "iso": s2})


def run_cascade(df: pd.DataFrame, valid: np.ndarray, stage_cols: list[tuple[str, list[str]]],
                 attention: bool, domains: np.ndarray | None, log) -> tuple[np.ndarray, dict, dict]:
    memory = None
    stage_scores: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for name, cols in stage_cols:
        Xs = atypicality.prepare_matrix(df[cols], valid)
        if memory is not None and attention:
            ridge = Ridge(alpha=1.0)
            ridge.fit(memory, Xs)
            resid = Xs - ridge.predict(memory)
            weight = max(1.0 - ridge.score(memory, Xs), 0.05)
        else:
            resid = Xs
            weight = 1.0
        stage_scores[name] = stage_detector(resid, domains)
        weights[name] = weight
        log(f"    стадия «{name}» ({len(cols)} признаков): вес={weight:.3f}")

        k = min(3, Xs.shape[1])
        pcs = PCA(n_components=k, random_state=config.ANOM_SEED).fit_transform(Xs)
        memory = pcs if memory is None else np.hstack([memory, pcs])

    combined = weighted_rank_ensemble(stage_scores, weights)
    return combined, weights, stage_scores


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:4.0f}s] {msg}", flush=True)

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    domains_full = cell_mask.build_domains(df)
    domains = domains_full[valid]
    log(f"dataset_v6: {df.shape[0]} ячеек, валидных {int(valid.sum())}")

    stage_cols = build_stage_cols(df)
    for name, cols in stage_cols:
        log(f"стадия «{name}»: {len(cols)} признаков -> {cols}")

    scores: dict[str, np.ndarray] = {}
    all_weights: dict[str, dict] = {}

    log("=== cascade_attention (память+внимание, глобально) ===")
    combined, w, _ = run_cascade(df, valid, stage_cols, attention=True, domains=None, log=log)
    scores["cascade_attention"] = atypicality.expand_to_grid(combined, valid)
    all_weights["cascade_attention"] = w

    log("=== cascade_attention_dom (память+внимание, по доменам) ===")
    combined, w, _ = run_cascade(df, valid, stage_cols, attention=True, domains=domains, log=log)
    scores["cascade_attention_dom"] = atypicality.expand_to_grid(combined, valid)
    all_weights["cascade_attention_dom"] = w

    log("=== cascade_uniform (контроль: без памяти/внимания, глобально) ===")
    combined, w, _ = run_cascade(df, valid, stage_cols, attention=False, domains=None, log=log)
    scores["cascade_uniform"] = atypicality.expand_to_grid(combined, valid)
    all_weights["cascade_uniform"] = w

    log("=== cascade_uniform_dom (контроль: без памяти/внимания, по доменам) ===")
    combined, w, _ = run_cascade(df, valid, stage_cols, attention=False, domains=domains, log=log)
    scores["cascade_uniform_dom"] = atypicality.expand_to_grid(combined, valid)
    all_weights["cascade_uniform_dom"] = w

    log("=== ast_opt_only (одна стадия «минералогия изменений» сама по себе) ===")
    ast_opt_cols = next(cols for name, cols in stage_cols if name == "минералогия_изменений")
    X_ao = atypicality.prepare_matrix(df[ast_opt_cols], valid)
    scores["ast_opt_only"] = atypicality.expand_to_grid(stage_detector(X_ao, None), valid)

    meta, prognoz = criterial_target.load_prognoz_grid()
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    own = ["cascade_attention", "cascade_attention_dom", "cascade_uniform",
           "cascade_uniform_dom", "ast_opt_only"]
    n_cfg = len(own)
    alpha = 0.05 / n_cfg
    log(f"{n_cfg} кандидатов, порог Бонферрони alpha={alpha:.4f}")

    result = assessment.run_protocol(scores, own, meta, valid, alpha,
                                      baseline="criterial", log=log)

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out_dir / "metrics" / "atypicality_cascade"
    result["pa"].to_csv(f"{stem}.csv", index=False)
    result["null"].to_csv(f"{stem}_null.csv", index=False)
    result["boot"].to_csv(f"{stem}_boot.csv", index=False)
    result["verdict"].to_csv(f"{stem}_verdict.csv", index=False)

    weights_rows = [{"config": cfg, "stage": st, "weight": w}
                     for cfg, wd in all_weights.items() for st, w in wd.items()]
    pd.DataFrame(weights_rows).to_csv(f"{stem}_weights.csv", index=False)

    # --- карта: ансамбли каскада против критериального и лучшего плоского результата ---
    plot_names = ["cascade_attention", "cascade_attention_dom", "cascade_uniform",
                  "ast_opt_only", "criterial"]
    unb = result["cells_unbiased"]
    pool_idx = np.flatnonzero(valid)

    def lift(name):
        return assessment.capture_efficiency(scores[name], unb, pool=pool_idx)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    prow, pcol = df["row"].to_numpy()[unb], df["col"].to_numpy()[unb]
    for ax, name in zip(axes.ravel(), plot_names):
        grid = np.full(len(df), np.nan)
        s = scores[name][valid]
        grid[valid] = sps.rankdata(s) / s.size
        im = ax.imshow(grid.reshape(meta.prf, meta.pic), cmap="magma", vmin=0, vmax=1)
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^", label="несмещённые точки заверки")
        ax.set_title(f"{name} (lift@10% = {lift(name):.2f})", fontsize=10)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot_names):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Каскад по группам признаков (5 стадий, предметный порядок) vs критериальная формула", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "atypicality_cascade_map.png", dpi=130, bbox_inches="tight")

    verdict = result["verdict"]
    print("\nВердикт (lift@10%, несмещённые точки; superior = CI>0 И p<alpha):")
    print(verdict[["method", "lift", "lift_criterial", "delta", "ci_lo", "ci_hi",
                   "p_shift", "superior"]].round(3).to_string(index=False))
    print("\nВеса стадий (внимание) по конфигурациям:")
    print(pd.DataFrame(weights_rows).pivot(index="stage", columns="config", values="weight").round(3))
    log("сохранено: outputs/atypicality_cascade*.png, outputs/metrics/atypicality_cascade*.csv")


if __name__ == "__main__":
    main()
