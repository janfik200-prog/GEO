"""Обучение без учителя на данных организации: восстанавливается ли сама формула.

Вопрос другой, чем в `mineragenic_unsupervised.py`. Там набор докачки был
выбран именно потому, что в нём НЕТ входов критериальной формулы, и согласие с
её картой намеренно не было критерием отбора. Здесь наоборот: берётся
`dataset_mineragenic_org.parquet` — только данные организации, включая все 13
факторных слоёв генетической модели, — и мишенью служит результат самой
формулы. Спрашивается: проступает ли критериальный прогноз в структуре данных
сам по себе, без единой метки, или он держится исключительно на весах,
проставленных экспертом.

Почему это осмысленный вопрос, а не циркулярность. Циркулярность возникает,
когда мишень попадает в ОБУЧЕНИЕ. Здесь мишень не видит ни один детектор:
методы ищут структуру в 20 столбцах и ничего не знают ни о `prognoz`, ни о
точках. Мишень появляется только на этапе измерения. Высокое согласие означает
«экспертные веса согласованы с геометрией данных», низкое — «формула — это
именно решение эксперта, из данных она не выводится». Оба исхода
информативны, и оба — про формулу, а не про заверенное оруденение.

Три набора, чтобы разделить источники согласия:

* `факторные_13` — прямые входы формулы (`dist_tect*`, `dens_*`, `dist_magm`,
  `dist_struct`, `dist_facies`, `dist_paleo`, `geo2_*`). Потолок согласия;
* `поля_7` — гравика, магнитка, трансформанты и `dist_dnara`: данные
  организации, которые во ВХОДЫ формулы не входят. Согласие здесь — это уже
  независимое подтверждение, а не пересборка;
* `все_20` — вместе.

Ориентация карты. У детектора нетипичности нет знака: «редкая ячейка» может
отвечать и перспективной, и заведомо пустой зоне. Поэтому согласие считается
для карты и для её инверсии, и в таблице стоят оба числа. Выбор ориентации по
лучшему согласию — это использование мишени объёмом ровно в один бит, и он
отмечен в столбце `ориентация`; сравнивать по нему методы можно, объявлять
результат заверкой — нельзя.

Предметный граф здесь строится по ролям генетической модели (тектонический,
магматогенный, структурно-литологический, литолого-фациальный,
палеогеоморфологический, глубинные поля), а не по минерагеническому словарю:
факторные слои в словаре не разобраны, у них своя, более старая классификация
из постановки критериального анализа.

Выход: `outputs/metrics/org_unsup_run.csv`, `..._verdict.csv`, `..._pa.csv`,
`..._contrib.csv`, `outputs/org_unsup_map.png`,
`data/processed/org_unsup_scores.parquet`.

Запуск из корня: ``python -m experiments.org_unsupervised`` (второй раунд с
ранговым представлением — аргумент ``rank``).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src import (assessment, atypicality, cell_mask, config,  # noqa: E402
                 crit_reference, criterial_target)
from experiments.honest_forecast_final import trend_share  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402
from experiments.mineragenic_unsupervised import (  # noqa: E402
    K_CLUSTERS, STABLE, block_maps, build_maps, induced, kmeans_maps, ward_map)

MODE = sys.argv[1] if len(sys.argv) > 1 else "z"
TAG = "org_unsup" + ("_rank" if MODE == "rank" else "")
T0 = time.time()

#: Роли генетической модели для предметного графа. Источник — постановка
#: критериального анализа (`config.LAYER_FILES`, `GOLD_FEATURES`), а не
#: минерагенический словарь: там факторные слои не разобраны.
ROLES: dict[str, tuple[str, ...]] = {
    "тектонический": ("dist_tect1", "dist_tect2", "dens_tect", "geo2_fault_node",
                      "geo2_strike_sin", "geo2_strike_cos"),
    "магматогенный": ("dist_magm", "dens_magm"),
    "структурно_литологический": ("dist_struct", "geo2_contact"),
    "литолого_фациальный": ("dist_facies", "geo2_facies2"),
    "палеогеоморфологический": ("dist_paleo", "dist_dnara"),
    "глубинные_поля": ("gm_gr_ost_35", "gm_gr_1G_25", "gm_gr_2G_25",
                       "gm_mag_ost_35", "gm_mag_1G_25", "pf_mag_asig"),
}
FACTOR = tuple(c for role, cols in ROLES.items() if role != "глубинные_поля"
               for c in cols if c != "dist_dnara")
FIELDS = ROLES["глубинные_поля"] + ("dist_dnara",)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    meta, prognoz_grid = criterial_target.load_prognoz_grid()
    prognoz = prognoz_grid.ravel()
    y = criterial_target.training_positives(prognoz_grid).astype(int)
    shape = (meta.prf, meta.pic)
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_mineragenic_org.parquet")
    valid = cell_mask.build_valid_mask(criterial_target.training_features("v5_rebuilt"))
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    basis = np.column_stack([np.ones(prognoz.size), rbf_basis(cx, cy, RBF_GRID)])
    crit = crit_reference.criterial_score(prognoz)
    top_c = prognoz <= np.quantile(prognoz[pool], config.CREF_AREA)
    pts = assessment.load_verification_points(meta)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    all_cols = [c for c in df.columns if c not in ("row", "col", "x", "y")]
    sets = {"все_20": all_cols,
            "факторные_13": [c for c in all_cols if c in FACTOR],
            "поля_7": [c for c in all_cols if c in FIELDS]}
    west = valid & (cx < np.median(cx[valid]))
    log(f"раунд {MODE}: валидных ячеек {int(valid.sum())}, точек {unb.size}; наборы "
        + ", ".join(f"{k}({len(v)})" for k, v in sets.items()))

    rows, contrib = [], []
    scores = {"criterial": crit}
    wards = {}
    for tag, cols in sets.items():
        X = atypicality.prepare_matrix(df[cols], valid)
        if MODE == "rank":
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                    random_state=config.ANOM_SEED).fit_transform(X)
        blocks = {}
        for role, names in ROLES.items():
            idx = np.array([cols.index(n) for n in names if n in cols], dtype=int)
            if idx.size:
                blocks[role] = idx
        log(f"набор {tag}: {X.shape[1]} признаков, ролей {len(blocks)} "
            f"({', '.join(f'{k}:{v.size}' for k, v in blocks.items())})")

        maps = build_maps(X)
        d_km, rar_km, _ = kmeans_maps(X, config.ANOM_SEED)
        maps["kmeans_расстояние"], maps["kmeans_редкость"] = d_km, rar_km
        maps["ward_связный"], wards[tag] = ward_map(X, valid, shape)
        maps["блоки_ранги"], maps["блоки_pca_kmeans"] = block_maps(
            X, blocks, config.ANOM_SEED)
        log(f"    кандидатов {len(maps)}")

        Xw, east = X[west[valid]], valid & ~west
        for name, s_valid in maps.items():
            s_raw = atypicality.expand_to_grid(s_valid, valid)
            agr_p = crit_reference.agreement(s_raw, crit, pool, config.CREF_AREA)
            agr_m = crit_reference.agreement(-s_raw, crit, pool, config.CREF_AREA)
            top_p = s_raw >= np.quantile(s_raw[pool], 1 - config.CREF_AREA)
            top_m = (-s_raw) >= np.quantile(-s_raw[pool], 1 - config.CREF_AREA)
            sog_p = (top_c & top_p).sum() / max(top_c.sum(), 1)
            sog_m = (top_c & top_m).sum() / max(top_c.sum(), 1)
            flip = sog_m > sog_p
            s = -s_raw if flip else s_raw
            agr = agr_m if flip else agr_p
            cov, realized = assessment.coverage_and_area(
                s, unb, config.VER_AREA, pool=pool)
            rho_st = np.nan
            if name in STABLE:
                s_half = induced(name, Xw, X, config.ANOM_SEED)
                e = east[valid]
                rho_st = float(sps.spearmanr(s_half[e], s_valid[e]).statistic)
            rows.append({
                "набор": tag, "метод": name, "признаков": len(cols),
                "ориентация": "инверсия" if flip else "прямая",
                "совпадение_верх_10": max(sog_p, sog_m),
                "совпадение_другой_ориентации": min(sog_p, sog_m),
                "kappa": agr["kappa"], "rho_с_формулой": agr["rho"],
                "roc_auc_мишень": roc_auc_score(y[pool], s[pool]),
                "lift_10": cov / realized if realized > 0 else 0.0,
                "точек_из_19": int(round(cov * unb.size)),
                "доля_тренда_R2": trend_share(s, basis, valid),
                "устойчивость_ро": rho_st,
            })
            scores[f"{name}_{tag}"] = s
            contrib.append(atypicality.feature_contributions(
                X, -s_valid if flip else s_valid, cols).head(5)
                .assign(набор=tag, метод=name))
            log(f"  {tag:14s} {name:20s} согл={rows[-1]['совпадение_верх_10']:.1%} "
                f"({rows[-1]['ориентация']}) kappa={agr['kappa']:+.3f} "
                f"auc={rows[-1]['roc_auc_мишень']:.3f} "
                f"lift={rows[-1]['lift_10']:.2f} тренд={rows[-1]['доля_тренда_R2']:.3f}")

    rng = np.random.default_rng(config.VER_SEED)
    scores["случайный_контроль"] = np.where(valid, rng.random(valid.size), -np.inf)

    res = pd.DataFrame(rows).sort_values(
        ["набор", "совпадение_верх_10"], ascending=[True, False])
    (ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_run.csv",
               index=False, encoding="utf-8-sig")
    pd.concat(contrib).to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_contrib.csv",
                              index=False, encoding="utf-8-sig")
    print("\n=== Без учителя на данных организации: восстановление критериальной карты ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("справочно: С УЧИТЕЛЕМ на тех же 20 признаках согласие 51.2%, "
          "ROC-AUC 0.944, 6 точек из 19 (lift 3.16); "
          "случайная карта дала бы согласие 10% по построению")

    keys = [k for k in scores if k.endswith("_все_20")]
    sub = {k: v for k, v in scores.items()
           if k.endswith("_все_20") or k in ("criterial", "случайный_контроль")}
    prot = assessment.run_protocol(sub, keys + ["случайный_контроль"], meta, valid,
                                   0.05 / len(keys), baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_verdict.csv",
                           index=False, encoding="utf-8-sig")
    prot["pa"].to_csv(ROOT / "outputs" / "metrics" / f"{TAG}_pa.csv",
                      index=False, encoding="utf-8-sig")
    print("\n=== Заверка по точкам (справочно: отбор шёл по согласию, не по ней) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = pd.DataFrame({"row": df["row"], "col": df["col"], "x": cx, "y": cy,
                        "prognoz": prognoz, "valid": valid,
                        **{k: v for k, v in scores.items() if k != "criterial"}})
    out["кластер_ward_все_20"] = np.nan
    out.loc[valid, "кластер_ward_все_20"] = wards["все_20"]
    out.to_parquet(config.PROCESSED_DIR / f"{TAG}_scores.parquet", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = res[res["набор"] == "все_20"].iloc[0]["метод"]
    best_f = res[res["набор"] == "факторные_13"].iloc[0]["метод"]
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2))
    axes[0].imshow(np.where(valid, prognoz, np.nan).reshape(shape), origin="upper",
                   cmap="magma_r")
    axes[0].set_title("Критериальный метод (эталон)")
    for ax, key, ttl in ((axes[1], f"{best}_все_20",
                          f"Без учителя: {best}\nвсе 20 признаков организации"),
                         (axes[2], f"{best_f}_факторные_13",
                          f"Без учителя: {best_f}\nтолько 13 факторных слоёв")):
        disp = np.full(valid.size, np.nan)
        disp[valid] = sps.rankdata(scores[key][valid]) / int(valid.sum())
        ax.imshow(disp.reshape(shape), origin="upper", cmap="magma")
        ax.contour(np.nan_to_num(disp >= 0.9).reshape(shape).astype(float),
                   levels=[0.5], colors="cyan", linewidths=0.6)
        ax.set_title(ttl + "\nголубым — верхние 10%")
    axes[3].imshow(np.where(valid, atypicality.expand_to_grid(wards["все_20"], valid),
                            np.nan).reshape(shape), origin="upper", cmap="tab10")
    axes[3].contour(np.nan_to_num(top_c).reshape(shape).astype(float), levels=[0.5],
                    colors="black", linewidths=0.8)
    axes[3].set_title(f"Районирование: связный Уорд, {K_CLUSTERS} зон\n"
                      "чёрным — верхние 10% критериальной карты")
    pr, pc = np.divmod(unb, meta.pic)
    for ax in axes:
        ax.scatter(pc, pr, s=16, c="white", edgecolors="black", linewidths=0.5,
                   label="точки заверки (коренные, U/TR)")
        ax.set_xlabel("столбец сетки (шаг 500 м)")
    axes[0].set_ylabel("строка сетки")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"{TAG}_map.png", dpi=130)
    log(f"OK: outputs/{TAG}_map.png")


if __name__ == "__main__":
    main()
