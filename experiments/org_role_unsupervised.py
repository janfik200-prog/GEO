"""Факторные слои по ролям генетической модели: что вытаскивает каждая роль.

Продолжение `org_unsupervised.py`. Там на всех 13 факторных слоях сразу два
метода без учителя дали разные ответы и разошлись по драйверам: связный Уорд
пошёл за тектоникой (`dist_tect1/2`, `geo2_fault_node`), изолирующий лес — за
фациями и палеодолинами (`dist_facies`, `geo2_contact`, `geo2_facies2`,
`dist_paleo`). Отсюда гипотеза: роли генетической модели несут разную
структуру, и собранные по отдельности они дадут больше, чем общий прогон.

Здесь каждая роль прогоняется как самостоятельный набор, а затем из
получившихся карт собираются составные варианты. Составы зафиксированы ДО
прогона:

* `состав_лес_по_ролям` — изолирующий лес на каждой из пяти ролей, объединение
  по средним рангам (равный вес роли независимо от числа колонок);
* `состав_уорд_по_ролям` — то же связным Уордом;
* `состав_уорд_тект_плюс_лес_фп` — прямая проверка наблюдения из прошлого
  прогона: Уорд на тектонической роли плюс изолирующий лес на объединённом
  фациально-палеодолинном блоке;
* `состав_лучший_по_ролям` — на каждой роли берётся метод с лучшим согласием.
  Это ЯВНЫЙ отбор по мишени, верхняя граница, а не оценка; помечен в таблице.

Что здесь про мишень. Как и раньше, ни один детектор мишени в обучении не
видит. Но ориентация каждой карты (нетипичность = перспективность или
наоборот) выбирается по согласию — это один бит мишени на компоненту, то есть
до пяти бит на состав. Для честного сравнения рядом считается тот же состав
без выбора ориентации (`_без_выбора`): если разрыв велик, значит результат
держится на отборе, а не на структуре данных.

Представление рангового типа (нормальные оценки) — по итогу прошлого прогона:
на робастных z нуль-раздутые столбцы забирают карту себе.

Выход: `outputs/metrics/org_roles_run.csv`, `..._composite.csv`,
`..._verdict.csv`, `..._pa.csv`, `..._contrib.csv`,
`outputs/org_roles_map.png`, `data/processed/org_roles_scores.parquet`.

Запуск из корня: ``python -m experiments.org_role_unsupervised``
(аргумент ``z`` — контрольный раунд на робастных z).
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
    K_CLUSTERS, STABLE, build_maps, induced, kmeans_maps, ward_map)
from experiments.org_unsupervised import FACTOR, ROLES  # noqa: E402

MODE = "z" if len(sys.argv) > 1 and sys.argv[1] == "z" else "rank"
TAG = "org_roles" + ("_z" if MODE == "z" else "")
T0 = time.time()

#: Объединение трёх ролей, за которыми пошёл изолирующий лес в прошлом прогоне.
FACIES_PALEO = ("структурно_литологический", "литолого_фациальный",
                "палеогеоморфологический")


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def pct_rank(s: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Карта -> перцентили внутри валидных ячеек (вне маски -inf).

    Составы собираются именно по рангам: у махаланобиса и невязок хвосты
    несопоставимой тяжести, и среднее сырых значений отдало бы состав одной
    компоненте.
    """
    out = np.full(s.size, -np.inf)
    out[valid] = sps.rankdata(s[valid]) / int(valid.sum())
    return out


def combine(parts: list[np.ndarray], valid: np.ndarray) -> np.ndarray:
    """Состав: среднее перцентилей компонент, равный вес каждой."""
    acc = np.zeros(valid.size)
    for p in parts:
        acc += pct_rank(p, valid)
    acc /= len(parts)
    return np.where(valid, acc, -np.inf)


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
    west = valid & (cx < np.median(cx[valid]))
    east = valid & ~west

    all_cols = [c for c in df.columns if c not in ("row", "col", "x", "y")]
    sets = {role: [c for c in cols if c in all_cols] for role, cols in ROLES.items()}
    sets["фациально_палеодолинный"] = [c for r in FACIES_PALEO for c in ROLES[r]
                                       if c in all_cols]
    sets["факторные_13"] = [c for c in all_cols if c in FACTOR]
    log(f"раунд {MODE}: наборов {len(sets)}, "
        + ", ".join(f"{k}({len(v)})" for k, v in sets.items()))

    def measure(s_raw: np.ndarray, name: str, tag: str, rho_st=np.nan,
                pick: str = "по согласию") -> dict:
        """Одна строка таблицы: ориентация, согласие с формулой, точки, тренд."""
        top_p = s_raw >= np.quantile(s_raw[pool], 1 - config.CREF_AREA)
        top_m = (-s_raw) >= np.quantile(-s_raw[pool], 1 - config.CREF_AREA)
        sog_p = (top_c & top_p).sum() / max(top_c.sum(), 1)
        sog_m = (top_c & top_m).sum() / max(top_c.sum(), 1)
        flip = sog_m > sog_p and pick == "по согласию"
        s = -s_raw if flip else s_raw
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        cov, realized = assessment.coverage_and_area(s, unb, config.VER_AREA, pool=pool)
        return {"набор": tag, "метод": name, "признаков": len(sets.get(tag, [])),
                "ориентация": "инверсия" if flip else "прямая",
                "выбор_ориентации": pick,
                "совпадение_верх_10": max(sog_p, sog_m) if pick == "по согласию" else sog_p,
                "kappa": agr["kappa"], "rho_с_формулой": agr["rho"],
                "roc_auc_мишень": roc_auc_score(y[pool], s[pool]),
                "lift_10": cov / realized if realized > 0 else 0.0,
                "точек_из_19": int(round(cov * unb.size)),
                "доля_тренда_R2": trend_share(s, basis, valid),
                "устойчивость_ро": rho_st, "_s": s}

    rows, contrib, per_role, per_role_raw = [], [], {}, {}
    for tag, cols in sets.items():
        X = atypicality.prepare_matrix(df[cols], valid)
        if MODE == "rank":
            from sklearn.preprocessing import QuantileTransformer
            X = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                    random_state=config.ANOM_SEED).fit_transform(X)
        maps = build_maps(X)
        d_km, rar_km, _ = kmeans_maps(X, config.ANOM_SEED)
        maps["kmeans_расстояние"], maps["kmeans_редкость"] = d_km, rar_km
        maps["ward_связный"], _ = ward_map(X, valid, shape)
        log(f"набор {tag}: {X.shape[1]} признаков, кандидатов {len(maps)}")

        Xw = X[west[valid]]
        per_role[tag], per_role_raw[tag] = {}, {}
        for name, s_valid in maps.items():
            rho_st = np.nan
            if name in STABLE:
                s_half = induced(name, Xw, X, config.ANOM_SEED)
                e = east[valid]
                rho_st = float(sps.spearmanr(s_half[e], s_valid[e]).statistic)
            s_raw = atypicality.expand_to_grid(s_valid, valid)
            r = measure(s_raw, name, tag, rho_st)
            per_role[tag][name] = r["_s"]
            per_role_raw[tag][name] = np.where(valid, s_raw, -np.inf)
            rows.append(r)
            contrib.append(atypicality.feature_contributions(
                X, s_valid * (-1 if r["ориентация"] == "инверсия" else 1), cols)
                .head(5).assign(набор=tag, метод=name))
            log(f"  {tag:26s} {name:20s} согл={r['совпадение_верх_10']:.1%} "
                f"({r['ориентация']}) auc={r['roc_auc_мишень']:.3f} "
                f"lift={r['lift_10']:.2f} тренд={r['доля_тренда_R2']:.3f}")

    roles = list(ROLES)
    #: Состав -> (роль, метод) компонент. Из этого же списка собирается вариант
    #: `_без_выбора`: те же компоненты, но в исходной ориентации детектора.
    comp_spec = {
        "состав_лес_по_ролям": [(r, "изолирующий_лес") for r in roles],
        "состав_уорд_по_ролям": [(r, "ward_связный") for r in roles],
        "состав_уорд_тект_плюс_лес_фп": [("тектонический", "ward_связный"),
                                         ("фациально_палеодолинный",
                                          "изолирующий_лес")],
    }
    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "_s"} for r in rows])
    best_name = {r: tab[tab["набор"] == r].sort_values(
        "совпадение_верх_10", ascending=False).iloc[0]["метод"] for r in roles}
    comp_spec["состав_лучший_по_ролям"] = [(r, best_name[r]) for r in roles]
    log("лучший метод по ролям (отбор по мишени): "
        + ", ".join(f"{r}={m}" for r, m in best_name.items()))

    comp_rows, comp_scores = [], {}
    for name, spec in comp_spec.items():
        r = measure(combine([per_role[t][m] for t, m in spec], valid), name, "составы")
        r["признаков"] = sum(len(sets[t]) for t, _ in spec)
        comp_rows.append(r)
        comp_scores[name] = r["_s"]
        # Тот же состав, но компоненты в исходной ориентации детектора: цена
        # ровно тех бит мишени, что потрачены на выбор «нетипично = хорошо».
        r0 = measure(combine([per_role_raw[t][m] for t, m in spec], valid),
                     name + "_без_выбора", "составы", pick="прямая")
        r0["признаков"] = r["признаков"]
        comp_rows.append(r0)
        comp_scores[name + "_без_выбора"] = r0["_s"]
        flips = sum(1 for t, m in spec
                    if not np.array_equal(per_role[t][m], per_role_raw[t][m]))
        log(f"  состав {name:32s} согл={r['совпадение_верх_10']:.1%} "
            f"(без выбора {r0['совпадение_верх_10']:.1%}, инверсий {flips}/{len(spec)}) "
            f"lift={r['lift_10']:.2f} тренд={r['доля_тренда_R2']:.3f}")

    # Жадное наращивание ролей: сколько ролей реально нужно составу.
    greedy, used, rest = [], [], list(roles)
    while rest:
        cand = [(measure(combine([per_role[r][best_name[r]] for r in used + [r2]],
                                 valid), "жадный", "составы")["совпадение_верх_10"], r2)
                for r2 in rest]
        val, pick = max(cand)
        used.append(pick)
        rest.remove(pick)
        greedy.append({"шаг": len(used), "добавлена_роль": pick,
                       "ролей": len(used), "совпадение_верх_10": val})
        log(f"  жадно шаг {len(used)}: +{pick} -> согласие {val:.1%}")

    res = pd.concat([tab, pd.DataFrame([{k: v for k, v in r.items() if k != "_s"}
                                        for r in comp_rows])], ignore_index=True)
    res = res.sort_values(["набор", "совпадение_верх_10"], ascending=[True, False])
    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"{TAG}_run.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(greedy).to_csv(out_dir / f"{TAG}_greedy.csv", index=False,
                                encoding="utf-8-sig")
    pd.concat(contrib).to_csv(out_dir / f"{TAG}_contrib.csv", index=False,
                              encoding="utf-8-sig")
    print("\n=== Роли генетической модели по отдельности (без учителя) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nсправочно: на всех 13 факторных слоях сразу — Уорд 46.5%, лес 38.4%; "
          "с учителем на 20 признаках 51.2%; случайная карта 10%")
    print("\n=== Жадное наращивание ролей ===")
    print(pd.DataFrame(greedy).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rng = np.random.default_rng(config.VER_SEED)
    scores = {"criterial": crit, **comp_scores,
              **{f"{best_name[r]}_{r}": per_role[r][best_name[r]] for r in roles},
              "случайный_контроль": np.where(valid, rng.random(valid.size), -np.inf)}
    keys = [k for k in scores if k not in ("criterial",)]
    prot = assessment.run_protocol(scores, keys, meta, valid, 0.05 / len(keys),
                                   baseline="criterial", log=log)
    prot["verdict"].to_csv(out_dir / f"{TAG}_verdict.csv", index=False,
                           encoding="utf-8-sig")
    prot["pa"].to_csv(out_dir / f"{TAG}_pa.csv", index=False, encoding="utf-8-sig")
    print("\n=== Заверка по точкам (справочно: отбор шёл по согласию) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pd.DataFrame({"row": df["row"], "col": df["col"], "x": cx, "y": cy,
                  "prognoz": prognoz, "valid": valid,
                  **{k: v for k, v in scores.items() if k != "criterial"}}
                 ).to_parquet(config.PROCESSED_DIR / f"{TAG}_scores.parquet",
                              index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_comp = max(comp_rows, key=lambda r: r["совпадение_верх_10"])
    panels = [("Критериальный метод (эталон)", crit, "magma_r")]
    for r in roles[:4]:
        panels.append((f"роль: {r}\n{best_name[r]}", per_role[r][best_name[r]], "magma"))
    panels.append((f"состав: {best_comp['метод']}\n"
                   f"согласие {best_comp['совпадение_верх_10']:.0%}",
                   best_comp["_s"], "magma"))
    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5))
    pr, pc = np.divmod(unb, meta.pic)
    for ax, (ttl, s, cm) in zip(axes.ravel(), panels):
        disp = np.full(valid.size, np.nan)
        disp[valid] = sps.rankdata(s[valid]) / int(valid.sum())
        ax.imshow(disp.reshape(shape), origin="upper", cmap=cm)
        ax.contour(np.nan_to_num(top_c).reshape(shape).astype(float), levels=[0.5],
                   colors="cyan", linewidths=0.8)
        ax.scatter(pc, pr, s=14, c="white", edgecolors="black", linewidths=0.5)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("столбец сетки (шаг 500 м)")
    axes[0, 0].set_ylabel("строка сетки")
    fig.suptitle("Роли генетической модели без учителя, перцентили; "
                 "голубым — верхние 10% критериальной карты, точки — заверка",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"{TAG}_map.png", dpi=125)
    log(f"OK: outputs/{TAG}_map.png")


if __name__ == "__main__":
    main()
