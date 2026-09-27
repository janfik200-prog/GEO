"""Почему вложенный выбор радиуса проиграл фиксированному и что с этим делать.

ПОВОД. В `branch2_nested_tuning` радиус сглаживания, выбранный внутри обучающих
полос, дал карту ХУЖЕ, чем любой фиксированный радиус (совпадение 0.181 против
0.278 у фиксированных 10 км на 58 сырых признаках). При этом усреднённая по
восьми фолдам внутренняя кривая ведёт себя гладко и имеет ясный максимум на
2 км. Значит, дело не в самом критерии выбора, а в том, что максимум берётся
поштучно на каждом фолде: оценка на участке в 2-3 тысячи ячеек шумная, выбор
скачет от 0 до 10 км, и итоговая карта склеивается из полос, сглаженных
по-разному.

ЧТО ПРОВЕРЯЕТСЯ. Три правила выбора на одних и тех же внутренних прогнозах
(новых обучений нет, всё берётся из кэша):

* «максимум» — как в основном прогоне: argmax средней внутренней оценки;
* «правило одной ошибки» — берётся наименьший радиус, чья внутренняя оценка
  отличается от лучшей не более чем на одну стандартную ошибку среднего по
  внутренним фолдам (стандартный приём против выбора шума);
* «единый радиус» — один радиус на всю карту, максимум средней внутренней
  оценки по всем восьми фолдам сразу.

Первые два правила строго вложенные: выбор для полосы использует только её
собственную обучающую часть. Третье — не строго вложенное (усреднение по всем
фолдам задействует метки всех ячеек), это компромисс, и он отмечен как таковой.

Выход: `outputs/metrics/branch2_selection_rules.csv`.
Запуск из корня: ``python -m experiments.branch2_selection_rules``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from experiments.honest_feature_signal import N_STRIPS, poly_design  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes, smooth_field  # noqa: E402
from experiments.branch2_nested_tuning import (CONFIGS, SIGMA_KM, combine,  # noqa: E402
                                               detrend, log, score_cells)
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIXED_KM = (0.0, 2.0, 10.0)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)
    pr_flat = prognoz.ravel()
    top_c = pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)
    crit_r = detrend(crit, pool, D)
    top_c_r = (crit_r >= np.quantile(crit_r[pool], 1 - config.CREF_AREA)) & valid

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    strips = {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, _ = strip_ids(coord)
        strips[ax] = np.where(valid, sid, -1)

    inner = dict(np.load(config.PROCESSED_DIR / "branch2_inner_cache.npz",
                         allow_pickle=True))
    z = np.load(config.PROCESSED_DIR / "branch1_multiscale_cache.npz", allow_pickle=True)
    maps1 = z["maps"].item()
    log(f"внутренних прогнозов {len(inner)}, конфигураций {len(CONFIGS)}")

    # ------------------------------------------------------------------
    # Внутренние оценки по каждому внутреннему фолду (для правила одной ошибки)
    # ------------------------------------------------------------------
    rows = []
    for cfg, (inp, models, _) in CONFIGS.items():
        for ax in ("X", "Y"):
            for k in range(N_STRIPS):
                for j in range(N_STRIPS):
                    keys = [f"{ax}{k}|{j}|{inp}|{m}|по умолчанию" for m in models]
                    if not all(q in inner for q in keys):
                        continue
                    te = np.flatnonzero(np.isfinite(inner[keys[0]]))
                    raw = {m: inner[q][te].astype(float) for m, q in zip(models, keys)}
                    p0 = np.full(y.size, np.nan)
                    p0[te] = combine(raw, models, te.size)
                    mask = np.zeros(y.size, dtype=bool)
                    mask[te] = True
                    for km in SIGMA_KM:
                        p = p0 if km == 0 else smooth_field(p0, shape,
                                                            km * 1000.0 / meta.dx,
                                                            mask=mask)
                        sc = score_cells(p, te, y, crit, D)
                        rows.append({"конфигурация": cfg, "полоса": f"{ax}{k}",
                                     "внутр_фолд": j, "сигма_км": km,
                                     "оценка": sc["совпадение_без_тренда"]})
    inn = pd.DataFrame(rows)
    log(f"внутренних оценок {len(inn)}")

    # ------------------------------------------------------------------
    # Три правила выбора
    # ------------------------------------------------------------------
    def rule_max(g):
        m = g.groupby("сигма_км")["оценка"].mean()
        return float(m.idxmax())

    def rule_1se(g):
        m = g.groupby("сигма_км")["оценка"].agg(["mean", "std", "count"])
        if m["mean"].isna().all():
            return 0.0
        se = (m["std"] / np.sqrt(m["count"].clip(lower=1))).fillna(0.0)
        best = m["mean"].idxmax()
        ok = m.index[m["mean"] >= m["mean"].max() - se.loc[best]]
        return float(min(ok)) if len(ok) else float(best)

    choice = {}
    for cfg, g_cfg in inn.groupby("конфигурация"):
        glob = float(g_cfg.groupby("сигма_км")["оценка"].mean().idxmax())
        for fold, g in g_cfg.groupby("полоса"):
            choice[(cfg, fold)] = {"максимум": rule_max(g),
                                   "правило одной ошибки": rule_1se(g),
                                   "единый радиус": glob}
    print("\n=== Выбранный радиус, км (по полосам) ===")
    ch = pd.DataFrame([{"конфигурация": c, "полоса": f, **v}
                       for (c, f), v in choice.items()])
    print(ch.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # ------------------------------------------------------------------
    # Внешняя оценка каждого правила и фиксированных радиусов
    # ------------------------------------------------------------------
    out, per_fold, assembled = [], {}, {}
    for cfg, (inp, models, key1) in CONFIGS.items():
        variants = {f"фикс. {km:.0f} км": ("фикс", km) for km in FIXED_KM}
        for rule in ("максимум", "правило одной ошибки", "единый радиус"):
            variants[rule] = ("правило", rule)
        for vname, (kind, what) in variants.items():
            acc_ax, fold_sc = {}, []
            for ax in ("X", "Y"):
                acc = np.full(y.size, np.nan)
                for k in range(N_STRIPS):
                    mask = strips[ax] == k
                    te = np.flatnonzero(mask)
                    km = what if kind == "фикс" else choice[(cfg, f"{ax}{k}")][what]
                    p = np.array(maps1[key1][ax], dtype=float)
                    if km > 0:
                        p = smooth_field(p, shape, km * 1000.0 / meta.dx, mask=mask)
                    acc[te] = p[te]
                    sc = score_cells(acc, te, y, crit, D)
                    sc["полоса"] = f"{ax}{k}"
                    fold_sc.append(sc)
                acc_ax[ax] = acc
            per_fold[(cfg, vname)] = pd.DataFrame(fold_sc).set_index("полоса")
            s = mean_axes(acc_ax["X"], acc_ax["Y"])
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            assembled[(cfg, vname)] = s
            top = (s >= np.quantile(s[pool], 1 - config.CREF_AREA)) & valid
            sr = detrend(s, pool, D)
            tpr = (sr >= np.quantile(sr[pool], 1 - config.CREF_AREA)) & valid
            out.append({
                "конфигурация": cfg, "правило": vname,
                "радиус_км": (what if kind == "фикс"
                              else np.mean([choice[(cfg, f"{a}{k}")][what]
                                            for a in "XY" for k in range(N_STRIPS)])),
                "совпадение_верх_10": (top & top_c).sum() / max(top_c.sum(), 1),
                "совпадение_без_тренда": (tpr & top_c_r).sum() / max(top_c_r.sum(), 1),
                "согласие_каппа": crit_reference.agreement(
                    s, crit, pool, config.CREF_AREA)["kappa"],
                "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                "фолды_совпадение": np.nanmean([f["совпадение"] for f in fold_sc]),
                "фолды_без_тренда": np.nanmean([f["совпадение_без_тренда"]
                                                for f in fold_sc]),
                "фолды_roc_auc": np.nanmean([f["roc_auc"] for f in fold_sc]),
                "точек_19": int(top[unb].sum()), "точек_73": int(top[allp].sum())})
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "outputs" / "metrics" / "branch2_selection_rules.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Правила выбора радиуса против фиксированных значений ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}; случайный уровень совпадения 0.10")

    # парные тесты по 8 полосам: рабочая точка против остальных радиусов
    pr = []
    for cfg in CONFIGS:
        a = per_fold[(cfg, "единый радиус")]
        for vname in ("фикс. 0 км", "фикс. 2 км", "фикс. 10 км", "максимум",
                      "правило одной ошибки"):
            b = per_fold[(cfg, vname)]
            for metric in ("совпадение", "совпадение_без_тренда", "roc_auc"):
                d = (a[metric] - b[metric]).dropna()
                if d.size < 3 or np.allclose(d, 0):
                    continue
                pr.append({"конфигурация": cfg, "против": vname, "метрика": metric,
                           "разница": float(d.mean()), "лучше_в": int((d > 0).sum()),
                           "полос": int(d.size),
                           "p_уилкоксон": float(
                               sps.wilcoxon(d, alternative="greater").pvalue)})
    paired = pd.DataFrame(pr)
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch2_rules_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные тесты по 8 полосам: единый радиус против остальных ===")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if len(paired) else "нет сравнений")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size  # noqa: E731
    panels = [("Критериальный прогноз", crit, "эталон сравнения")]
    for cfg, vname in (("бустинг | 58 сырые", "единый радиус"),
                       ("бустинг | 58 сырые", "фикс. 10 км"),
                       ("бустинг + графовая сеть | 58 много масштабов", "фикс. 10 км")):
        r = res[(res["конфигурация"] == cfg) & (res["правило"] == vname)].iloc[0]
        panels.append((f"{cfg}\nсглаживание {r['радиус_км']:.0f} км",
                       assembled[(cfg, vname)],
                       f"совпадение {r['совпадение_верх_10']:.0%}, "
                       f"без тренда {r['совпадение_без_тренда']:.0%}"))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.8))
    for ax, (title, v, sub) in zip(axs, panels):
        ax.imshow(rank(v), cmap="magma")
        ax.set_title(f"{title}\n{sub}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ветка 2: рабочая точка — радиус выбран по внутренним данным, "
                 "а не по метрике отчёта", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch2_operating_point.png", dpi=130)
    log("OK: outputs/branch2_operating_point.png")
    log("OK: outputs/metrics/branch2_selection_rules.csv")


if __name__ == "__main__":
    main()
