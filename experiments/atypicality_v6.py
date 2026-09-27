"""Лестница нетипичности (см. `src/atypicality.py`) на пуле `dataset_v6`.

Зачем: прежние прогоны (v1.1/v2/v3, `experiments/atypicality_v3.py`) считались
на `dataset_v2`/`v3` — до появления `ast`/`psr`/`relief2`/переработанного `s2`
и вообще до отбора v5->v6 (permutation importance убрала 33 шумовых признака).
Честная permutation importance по группе `ast` (20.08.2026) нашла реальный
сигнал, которого в старом пуле не было — открытый вопрос, меняется ли
результат лестницы (на v3: «методы неразличимы на 19 строгих точках», ни один
вердикт не был положительным, `outputs/metrics/atypicality_v3_boot.csv`) на
более богатом пуле.

Циркулярности здесь нет по конструкции (см. `src/atypicality.py` и
`experiments/atypicality_v3.py`): целевая переменная `prognoz` нигде не
участвует в обучении детекторов, поэтому геологические факторные слои
(geo/geo2 — единственные признаки, исключённые из ОБУЧЕНИЯ НА `prognoz` из-за
циркулярности, см. `criterial_target.V5_EXCLUDE_GROUPS`) можно вернуть в пул.

Два варианта пула считаются раздельно (оба — не служебные колонки
`dataset_v6.parquet`):

* ``ml`` (91 признак) — то же самое, что
  ``criterial_target.training_features(dataset="v6")``: дистанционка + рельеф
  + линеаменты + гидросеть, без geo/geo2;
* ``full`` (104 признака) — ``ml`` + geo (8, факторные слои формулы) + geo2
  (5, их раздельная детализация) — пул для случая «нет цели -> нет
  циркулярности».

Честная заверка — `src.assessment.run_protocol` (единый протокол для всех
фаз): P-A/lift@10%, сдвиговый null (500 сдвигов), бутстрэп разности с
критериальным по точечному и кластерному ресэмплингу, автоматический вердикт
(нижняя граница CI > 0 И p < Бонферрони-порог). Порог фиксируется ДО прогона
по числу опробованных конфигураций (36 = 2 пула x 18: 8 детекторов + ансамбль,
глобально и по доменам).

Запуск из корня: ``python -m experiments.atypicality_v6``.
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

from src import assessment, atypicality, cell_mask, config, criterial_target  # noqa: E402
from experiments.atypicality_v2 import LADDER, run_ladder  # noqa: E402

SERVICE_COLS = ("row", "col", "x", "y", "mask_svita")


def build_pools(df: pd.DataFrame) -> dict[str, list[str]]:
    """``ml`` (91, без geo/geo2) и ``full`` (104, с geo/geo2) списки колонок."""
    ml_cols = list(criterial_target.training_features(dataset="v6").columns)
    full_cols = [c for c in df.columns if c not in SERVICE_COLS]
    return {"ml": ml_cols, "full": full_cols}


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:4.0f}s] {msg}", flush=True)

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    domains = cell_mask.build_domains(df)
    log(f"dataset_v6: {df.shape[0]} ячеек, {df.shape[1]} колонок; "
        f"валидных {int(valid.sum())}")

    pools = build_pools(df)
    scores: dict[str, np.ndarray] = {}
    for pool_tag, cols in pools.items():
        X = atypicality.prepare_matrix(df[cols], valid)
        base = run_ladder(X, valid, tag=f"_{pool_tag}")
        scores.update(base)
        log(f"пул {pool_tag} ({len(cols)} признаков): лестница готова")

        dom_scores: dict[str, np.ndarray] = {}
        for name, fn in LADDER:
            s = atypicality.per_domain(fn, X, domains[valid])
            key = f"{name}_{pool_tag}_dom"
            scores[key] = atypicality.expand_to_grid(s, valid)
            dom_scores[key] = s
        scores[f"rank_ensemble_{pool_tag}_dom"] = atypicality.expand_to_grid(
            atypicality.rank_ensemble(dom_scores), valid)
        log(f"пул {pool_tag}: доменная нормировка готова "
            f"(доменов: {np.unique(domains[valid]).size})")

    meta, prognoz = criterial_target.load_prognoz_grid()
    scores["criterial"] = -prognoz.ravel()
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df))

    own = [k for k in scores if k not in ("criterial", "naive_hydro", "random")]
    n_cfg = len(own)
    alpha = 0.05 / n_cfg
    log(f"{n_cfg} конфигураций-кандидатов, порог Бонферрони alpha={alpha:.5f}")

    result = assessment.run_protocol(scores, own, meta, valid, alpha,
                                     baseline="criterial", log=log)

    out_dir = ROOT / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    stem = out_dir / "metrics" / "atypicality_v6"
    result["pa"].to_csv(f"{stem}.csv", index=False)
    result["null"].to_csv(f"{stem}_null.csv", index=False)
    result["boot"].to_csv(f"{stem}_boot.csv", index=False)
    result["verdict"].to_csv(f"{stem}_verdict.csv", index=False)
    np.savez_compressed(f"{stem}_scores.npz", valid=valid,
                        **{k.replace("|", "_"): v for k, v in scores.items()})

    # --- Карта: ансамбли обоих пулов (глобальный/доменный) против критериального ---
    plot_names = ["rank_ensemble_ml", "rank_ensemble_ml_dom",
                  "rank_ensemble_full", "rank_ensemble_full_dom", "criterial"]
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
        ax.scatter(pcol, prow, s=14, c="cyan", marker="^",
                  label="несмещённые точки заверки")
        ax.set_title(f"{name} (lift@10% = {lift(name):.2f})", fontsize=10)
        ax.set_xlabel("столбец (x500 м)")
        ax.set_ylabel("строка (x500 м)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="нормированный ранг")
    for ax in axes.ravel()[len(plot_names):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Лестница нетипичности на dataset_v6: ml (91, без geo/geo2) "
                 "vs full (104, с geo/geo2)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "atypicality_v6_map.png", dpi=130, bbox_inches="tight")

    verdict = result["verdict"]
    print("\nВердикт (lift@10%, несмещённые точки; superior = CI>0 И p<alpha):")
    print(verdict[["method", "lift", "lift_criterial", "delta", "ci_lo", "ci_hi",
                   "p_shift", "superior"]].round(3).to_string(index=False))
    log("сохранено: outputs/atypicality_v6*.png, outputs/metrics/atypicality_v6*.csv")


if __name__ == "__main__":
    main()
