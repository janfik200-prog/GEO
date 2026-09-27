"""Обучение и проверка модели на ОСНОВНОМ листе: угадывает ли она критериальный прогноз.

ВНИМАНИЕ (02.09.2026). Числа этого прогона — совпадение top-10% 91.9%,
ROC-AUC 0.997 — НЕ характеризуют работу признаков, и брать их в отчёты нельзя.
Причина установлена проверкой: столько же (ROC-AUC 0.995) даёт модель, которой
на вход подали ОДНИ КООРДИНАТЫ ячейки и ни одного признака. При шаге сетки
500 м случайно отложенная ячейка почти всегда окружена обучающими, поэтому
случайная перекрёстная проверка здесь измеряет пространственную интерполяцию.
Диагностика блоками 10 км, заложенная ниже, этот эффект показала лишь частично
(0.950) — буфера между блоками нет, и соседство работает через границу.
Честная оценка того же вопроса — `experiments/honest_forecast_main.py` (полосы
с разрывом 15 км, набор признаков с измеренной переносимостью, планка
«сколько даёт одна география»): совпадение top-10% 27.8% против 14.6% у модели
без признаков и 10% случайных. Прогон ниже сохранён как источник этих двух
диагностических чисел, а не как результат.

Главный вопрос прогона (постановка 01.09.2026): модель должна угадывать
КРИТЕРИАЛЬНЫЙ ПРОГНОЗ по сторонним признакам — именно это переносится на смежные
листы, где критериального расчёта нет. Точки заверки — не тот тип оруденения
(геохимическое опробование: литопробы, шлихи, проявления U/TR), они идут только
косвенной заверкой и в первичную оценку не входят.

Почему нельзя мерить угадывание на всём листе сразу: модель, обученная на всех
22 946 ячейках, воспроизводит мишень на 98.8% (AUC 0.99994) просто потому, что
видела все ответы. Такое число ничего не измеряет. Поэтому оценка идёт по
ячейкам, которых модель при обучении не видела.

Разбиение — по ЯЧЕЙКАМ, не по объектам и не по площадям с буфером:

* **основная оценка — случайная 5-кратная CV по ячейкам** (стратифицированная
  по мишени). Единица разбиения — ячейка-точка, ровно та же единица, в которой
  задана мишень. Каждая ячейка ровно один раз попадает в отложенную часть, из
  чего собирается OOF-прогноз на весь лист;
* **диагностика — та же CV, но блоками 10 км** (`CREF_BLOCK`, готовая
  `crit_reference.block_labels`). Это НЕ разметка объектами и не буфер: блок —
  технический приём, чтобы измерить, какая часть точности случайной CV взята
  не из признаков, а из соседства (соседние ячейки почти одинаковы, при
  случайном разбиении сосед ячейки из теста почти всегда лежит в обучении).
  Разрыв между двумя числами и есть мера этого эффекта. Если он мал —
  случайной CV можно верить как оценке переноса; если велик — перенос на
  смежные листы будет заметно хуже, чем обещает случайная CV.

Метрики угадывания (обе — на OOF, то есть вне обучения):

* ROC-AUC и PR-AUC по бинарной мишени top-`CREF_AREA` (10%) критериального
  прогноза. База PR-AUC при случайном прогнозе = доля позитивов = 0.10;
* `crit_reference.agreement` — AUC, capture (доля эталонной площади, накрытая
  top-10% модели, делённая на реализованную площадь), каппа Коэна на
  бинаризованных картах, ранговая корреляция с непрерывным `prognoz`.

Косвенная заверка точками — `assessment.run_protocol`, тот же код, которым
мерялись прошлые методы. Считается для OOF-карты. Читается как справка: 19
строго несмещённых ячеек дают дискретность 0.53 lift на точку, на таком числе
не проходит порог значимости даже сам критериальный эталон.

Модель — `HistBackgroundEnsemble`, признаки — тот же фильтр, что в
`forecast_wide.py` (импортируется оттуда): без группы `ls`, из `ast` только
`ast_kaolin` и `ast_b04`, пропуски передаются в модель как пропуски.

Выход: `outputs/metrics/main_sheet_train_test{,_folds,_null,_boot,_verdict}.csv`,
`data/processed/forecast_main.parquet`, `outputs/main_sheet_train_test_map.png`.

Запуск из корня репозитория: ``python -m experiments.train_test_main_sheet``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.forecast_wide import AST_KEEP, DROP_GROUPS, _keep  # noqa: E402

N_FOLDS = 5
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def cross_val_oof(X: np.ndarray, y: np.ndarray, splits, tag: str) -> tuple[np.ndarray, pd.DataFrame]:
    """OOF-прогноз и метрики по фолдам для готового разбиения ``splits``."""
    oof = np.full(y.size, np.nan)
    rows = []
    for k, (tr, te) in enumerate(splits):
        model = HistBackgroundEnsemble(random_state=config.CRIT_SEED).fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        rows.append({"scheme": tag, "fold": k,
                     "n_train": int(tr.size), "n_pos_train": int(y[tr].sum()),
                     "n_test": int(te.size), "n_pos_test": int(y[te].sum()),
                     "roc_auc": float(roc_auc_score(y[te], p)),
                     "pr_auc": float(average_precision_score(y[te], p))})
        log(f"  [{tag} {k}] ROC-AUC={rows[-1]['roc_auc']:.4f} "
            f"PR-AUC={rows[-1]['pr_auc']:.4f} "
            f"(тест {te.size}, поз. {int(y[te].sum())})")
    return oof, pd.DataFrame(rows)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()

    feat_all = criterial_target.training_features("v5_rebuilt")
    feat_df = feat_all[[c for c in feat_all.columns if _keep(c)]]
    feat_names = list(feat_df.columns)
    X = feat_df.to_numpy(dtype=float)
    if len(feat_df) != prognoz.size:
        raise ValueError(f"признаки {len(feat_df)} ячеек != сетка prognoz {prognoz.size}")

    y = criterial_target.training_positives(prognoz).astype(int)
    thr = float(np.quantile(prognoz.ravel(), config.CREF_AREA))
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    crit = crit_reference.criterial_score(prognoz)

    log(f"признаков: {len(feat_names)} (дроп {DROP_GROUPS}, из ast — {AST_KEEP}); "
        f"с пропусками: {int((~np.isfinite(X)).any(axis=0).sum())}")
    log(f"мишень: top-{config.CREF_AREA:.0%} критериального прогноза "
        f"(prognoz <= {thr:.4f}) — {int(y.sum())} из {y.size} ячеек")
    log(f"пул оценки: {pool.size} валидных ячеек; база PR-AUC = {y.mean():.4f}")

    # --- 1. Основная оценка: случайная CV по ячейкам ---
    log(f"=== Случайная {N_FOLDS}-кратная CV по ячейкам (единица — ячейка) ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.CRIT_SEED)
    oof_rand, folds_rand = cross_val_oof(X, y, list(skf.split(X, y)), "ячейки")

    # --- 2. Диагностика утечки через соседство: та же CV блоками 10 км ---
    blocks = crit_reference.block_labels(meta)
    log(f"=== Диагностика: та же CV блоками {config.CREF_BLOCK * meta.dx / 1000:.0f} км "
        f"({np.unique(blocks).size} блоков) — сколько точности даёт соседство ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_block, folds_block = cross_val_oof(X, y, list(gkf.split(X, y, groups=blocks)), "блоки")

    folds = pd.concat([folds_rand, folds_block], ignore_index=True)

    # --- 3. Финальная модель на всём листе (она уйдёт на смежные листы) ---
    log("=== Финальная модель на всём листе (для переноса) ===")
    full = HistBackgroundEnsemble(random_state=config.CRIT_SEED).fit(X, y)
    score_full = full.predict_proba(X)[:, 1]

    # --- Сводка угадывания критериального прогноза ---
    rows = []
    for name, sc in (("ячейки (основная)", oof_rand), ("блоки 10 км (диагностика)", oof_block),
                     ("весь лист (in-sample, не результат)", score_full)):
        agr = crit_reference.agreement(sc, crit, pool)
        sub = folds[folds["scheme"] == name.split()[0]] if "in-sample" not in name else None
        rows.append({
            "схема": name,
            "roc_auc_fold": sub["roc_auc"].mean() if sub is not None and len(sub) else np.nan,
            "roc_auc_fold_std": sub["roc_auc"].std() if sub is not None and len(sub) else np.nan,
            "pr_auc_fold": sub["pr_auc"].mean() if sub is not None and len(sub) else np.nan,
            "agr_auc": agr["auc"], "agr_capture": agr["capture"],
            "agr_kappa": agr["kappa"], "agr_rho": agr["rho"],
        })
    summary = pd.DataFrame(rows)
    print("\n=== УГАДЫВАНИЕ КРИТЕРИАЛЬНОГО ПРОГНОЗА (первичная оценка) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nБаза: PR-AUC случайного прогноза = {y.mean():.4f}; "
          f"capture=1 — случайно, {1/config.CREF_AREA:.0f} — идеально; каппа=1 — полное совпадение.")

    # Совпадение топ-зон — прямая, а не метрическая проверка «угадал ли».
    print("\n=== Совпадение top-N% модели и критериального прогноза ===")
    ov_rows = []
    for name, sc in (("ячейки", oof_rand), ("блоки 10 км", oof_block), ("весь лист", score_full)):
        for a in (0.05, 0.10, 0.20):
            tm = sc >= np.quantile(sc[pool], 1 - a)
            tc = crit >= np.quantile(crit[pool], 1 - a)
            ov_rows.append({"схема": name, "area": a, "n_крит": int(tc.sum()),
                            "пересечение": int((tm & tc).sum()),
                            "доля_крит": float((tm & tc).sum() / max(tc.sum(), 1)),
                            "IoU": float((tm & tc).sum() / max((tm | tc).sum(), 1))})
    overlap = pd.DataFrame(ov_rows)
    print(overlap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- Косвенная заверка точками (справочно) ---
    log("=== Косвенная заверка точками (не первичная метрика) ===")
    scores = {"hgb_oof": oof_rand, "criterial": crit,
              "naive_hydro": assessment.naive_hydro_score(feat_all),
              "random": assessment.random_score(len(feat_all), seed=config.VER_SEED)}
    result = assessment.run_protocol(scores, ["hgb_oof"], meta, valid, 0.05,
                                     baseline="criterial", log=log)
    print("\n=== Точки заверки — КОСВЕННАЯ заверка (иной тип оруденения) ===")
    print(result["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Сохранение ---
    metrics_dir = ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stem = metrics_dir / "main_sheet_train_test"
    summary.to_csv(f"{stem}.csv", index=False)
    folds.to_csv(f"{stem}_folds.csv", index=False)
    overlap.to_csv(f"{stem}_overlap.csv", index=False)
    result["null"].to_csv(f"{stem}_null.csv", index=False)
    result["boot"].to_csv(f"{stem}_boot.csv", index=False)
    result["verdict"].to_csv(f"{stem}_verdict.csv", index=False)

    cx, cy = meta.cell_centers()
    pd.DataFrame({
        "row": np.arange(y.size) // meta.pic, "col": np.arange(y.size) % meta.pic,
        "x": cx.ravel(), "y": cy.ravel(),
        "score_oof": oof_rand.astype(np.float32),
        "score_oof_block": oof_block.astype(np.float32),
        "score_full": score_full.astype(np.float32),
        "prognoz": prognoz.ravel().astype(np.float32),
        "target": y.astype(np.int8), "valid": valid,
    }).to_parquet(config.PROCESSED_DIR / "forecast_main.parquet", index=False)
    log(f"OK: {config.PROCESSED_DIR / 'forecast_main.parquet'}")

    # --- Карта ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sps

    def rank_grid(v: np.ndarray) -> np.ndarray:
        return (sps.rankdata(v) / v.size).reshape(meta.prf, meta.pic)

    g_crit = rank_grid(crit)
    g_rand, g_block = rank_grid(oof_rand), rank_grid(oof_block)
    a_rand = crit_reference.agreement(oof_rand, crit, pool)
    a_block = crit_reference.agreement(oof_block, crit, pool)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), sharex=True, sharey=True)
    panels = (
        (g_crit, "Критериальный прогноз (эталон)"),
        (g_rand, f"Модель вне обучения, ячейки\nAUC={a_rand['auc']:.3f} каппа={a_rand['kappa']:.2f}"),
        (g_block, f"Модель вне обучения, блоки 10 км\nAUC={a_block['auc']:.3f} каппа={a_block['kappa']:.2f}"),
    )
    for ax, (g, title) in zip(axes, panels):
        im = ax.imshow(g, cmap="magma", vmin=0, vmax=1)
        ax.contour(g_crit, levels=[1 - config.CREF_AREA], colors="#00e5ff", linewidths=1.0)
        ax.set_title(title)
        ax.set_xlabel("столбец (x500 м)")
    axes[0].set_ylabel("строка (x500 м)")
    fig.colorbar(im, ax=axes[2], shrink=0.7, label="нормированный ранг (выше = перспективнее)")

    im2 = axes[3].imshow(g_rand - g_crit, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    axes[3].contour(g_crit, levels=[1 - config.CREF_AREA], colors="#00e5ff", linewidths=1.0)
    axes[3].set_title("Расхождение: модель (ячейки) − критериальный\n"
                      "красное — модель считает перспективнее")
    axes[3].set_xlabel("столбец (x500 м)")
    fig.colorbar(im2, ax=axes[3], shrink=0.7, label="разность рангов")

    fig.suptitle(f"Основной лист: угадывание критериального прогноза по {len(feat_names)} "
                 f"сторонним признакам. Голубой контур — граница top-{config.CREF_AREA:.0%} эталона")
    fig.tight_layout()
    fig_path = ROOT / "outputs" / "main_sheet_train_test_map.png"
    fig.savefig(fig_path, dpi=140)
    log(f"OK: {fig_path}")


if __name__ == "__main__":
    main()
