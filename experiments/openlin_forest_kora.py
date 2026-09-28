"""Случайный лес, обрезанный «Корой»: правила из ветвей леса по критерию Бонгарда.

ИДЕЯ (предложение пользователя, 22.09.2026). У «Коры» пороги на признаки
грубые — терцили, одинаковые для всех правил. У леса пороги гибкие, но решение
размазано по сотням деревьев и не читается. Здесь одно подрезает другое: лес
строит деревья, а «Кора» идёт по каждой ветви от корня и ОБРУБАЕТ её в первом
узле, где путь стал характерным для одного класса — встречается минимум в
``S_MIN`` ячеек этого класса и минимум в ``R_MIN`` раз чаще, чем у другого.
Путь до этого узла становится правилом. Ветви, где обоих классов меньше
``S_MIN``, отбрасываются целиком. Обрубание в первом же подходящем узле и есть
требование минимальности из «Коры»: правило не содержит более короткого
характерного правила той же ветви.

Классы те же, что у «Коры»: настоящий снимок против многоканальной подделки с
теми же гистограммами, текстурой и связями каналов (:func:`openlin_kora.surrogate`).

ЧЕМ ПРОВЕРЯЕТСЯ. В прошлой проверке подвело не различение (AUC около 0.7), а
то, ЧТО различается: выделенные зоны избегали зон разломов. Поэтому решает не
AUC, а та же проверка по месту и та же устойчивость, что у «Коры»
(:mod:`openlin_kora_check`).

РЕЗУЛЬТАТ (22.09.2026) — не пройдено ничего, и хуже одной «Коры» по всем пунктам.

* AUC на проверке: 0.605 и 0.574 (чистый лес 0.622 и 0.565; одна «Кора» —
  0.706 и 0.679).
* Устойчивость: корреляция карт двух прогонов 0.02, простирание верхних 10 %
  3° и 88°. Два прогона дают по сути разные карты.
* По месту: 1.04x и 1.01x при нуле 0.99x, p = 0.32 и 0.43.
* Правил около 500 на прогон, и многие — одно и то же ядро с довеском условий
  из разных деревьев («COH12_cos1000 > 0.56 И ещё что-то»).

Запуск из корня репозитория:
    python -m experiments.openlin_forest_kora --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_kora import (N_TEST, N_TRAIN, R_MIN, S_MIN,
                                      features20, sample_rows, surrogate)

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

# ---- пре-регистрация (22.09.2026, до прогона) ------------------------------
N_TREES = 200
MAX_DEPTH = 4
MIN_LEAF = 50
AUC_MIN = 0.70                  # в ОБОИХ прогонах
SCORE_RHO_MIN = 0.6
STRIKE_MAX_DIFF = 10.0
FK_N_TESTS_PRIOR = 227          # 225 ветки + 2 проверки «Коры»
FK_N_TESTS = 2                  # 1а и 1б
ALPHA = 0.05 / (FK_N_TESTS_PRIOR + FK_N_TESTS)
N_SHIFTS = 10_000
assert 1.0 / (1 + N_SHIFTS) < ALPHA, "порог недостижим при таком числе сдвигов"
AREA = 0.10
#: Прогоны: (имя, поменять полосы, зерно подделки обучения, проверки).
RUNS = (("исходный", False, 100, 200), ("повтор", True, 300, 400))
# ---------------------------------------------------------------------------


def cut_rules(forest, x1: np.ndarray, x2: np.ndarray) -> list[dict]:
    """Ветви леса, обрубленные «Корой» в первом характерном узле."""
    rules, seen = [], set()
    for est in forest.estimators_:
        t = est.tree_
        stack = [(0, (), np.ones(len(x1), bool), np.ones(len(x2), bool))]
        while stack:
            node, conds, m1, m2 = stack.pop()
            if conds:
                s1, s2 = m1.mean(), m2.mean()
                cls = (1 if (s1 >= S_MIN and s1 >= R_MIN * s2) else
                       2 if (s2 >= S_MIN and s2 >= R_MIN * s1) else 0)
                if cls:
                    key = tuple((j, op, round(float(v), 3)) for j, op, v in conds)
                    if key not in seen:
                        seen.add(key)
                        own, other = (s1, s2) if cls == 1 else (s2, s1)
                        rules.append({"класс": cls, "условия": conds,
                                      "своих": own, "чужих": other})
                    continue                         # ветвь обрублена
                if s1 < S_MIN and s2 < S_MIN:
                    continue                         # ветвь слишком мала
            left = t.children_left[node]
            if left == -1:
                continue
            j, thr = int(t.feature[node]), float(t.threshold[node])
            l1, l2 = x1[:, j] <= thr, x2[:, j] <= thr
            stack.append((left, conds + ((j, "<=", thr),), m1 & l1, m2 & l2))
            stack.append((t.children_right[node], conds + ((j, ">", thr),),
                          m1 & ~l1, m2 & ~l2))
    return rules


def rule_hits(x: np.ndarray, conds: tuple) -> np.ndarray:
    m = np.ones(len(x), dtype=bool)
    for j, op, v in conds:
        m &= (x[:, j] <= v) if op == "<=" else (x[:, j] > v)
    return m


def score(x: np.ndarray, rules: list[dict]) -> np.ndarray:
    out = np.zeros(len(x))
    for cls, sign in ((1, 1.0), (2, -1.0)):
        rs = [r for r in rules if r["класс"] == cls]
        if rs:
            out += sign * np.mean([rule_hits(x, r["условия"]) for r in rs], 0)
    return out


def describe(conds: tuple, names: list[str]) -> str:
    parts = []
    for j, op, v in conds:
        n = names[j].replace("coh_summer_vv_", "лет.ког ").replace(
            "coh_winter_vv_", "зим.ког ")
        parts.append(f"{n} {'≤' if op == '<=' else '>'} {v:.2f}")
    return " И ".join(parts)


def neigh_all(x: np.ndarray, use: tuple, valid: np.ndarray, res_m: float,
              support: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Все 160 признаков шага 1; окрестность — только внутри ``support``."""
    from experiments.openlin_neigh import neigh_stack

    nx, nnames = neigh_stack(x, use, valid, res_m, base=use, support=support)
    return np.concatenate([x, nx]), list(use) + list(nnames)


def builder20(site: str):
    """Признаки по умолчанию — 20 отобранных (:mod:`openlin_feat_corr`)."""
    import pandas as pd

    keep = pd.read_csv(M / f"openlin_feat20_{site}.csv")["признак"].tolist()

    def build(stack, names):
        return stack[[names.index(n) for n in keep]], keep
    return build


def fit_runs(site: str, log, builder=None) -> tuple:
    """Оба прогона: лес, обрезанные правила, проверочные строки и карты.

    Вынесено из ``main``, чтобы объяснение (:mod:`openlin_forest_kora_shap`)
    работало ровно с теми же моделями и строками, что и проверка.
    ``builder(stack160, names160) -> (stack, names)`` задаёт набор признаков;
    по умолчанию — 20 отобранных. Возвращается и ``full160`` — все 160
    признаков по всему листу, из них проверка берёт направление.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from experiments.openlin_anfis import block_split
    from experiments.openlin_line_rank import auc
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    builder = builder or builder20(site)
    z = load_site(site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    tr0, te0 = block_split(valid.shape, res_m)
    tr0 &= valid
    te0 &= valid
    t0 = time.time()

    def feat(src, support):
        return builder(*neigh_all(src, use, valid, res_m, support))

    real = {"A": feat(x, tr0)[0], "B": feat(x, te0)[0]}
    full160 = neigh_all(x, use, valid, res_m, valid)
    full, names = builder(*full160)
    rows_full = full.reshape(full.shape[0], -1)[:, valid.ravel()].T
    log(f"площадка {site}: признаков {len(names)}, посчитаны "
        f"({time.time() - t0:.0f} с)")

    res = {}
    for run, swap, s_tr, s_te in RUNS:
        train, test = (te0, tr0) if swap else (tr0, te0)
        f_train = real["B"] if swap else real["A"]
        f_test = real["A"] if swap else real["B"]
        fk_tr = feat(surrogate(x, valid, np.random.default_rng(s_tr)), train)[0]
        fk_te = feat(surrogate(x, valid, np.random.default_rng(s_te)), test)[0]
        rng = np.random.default_rng(7)
        r_tr = sample_rows(f_train, train, N_TRAIN, rng)
        f_tr = sample_rows(fk_tr, train, N_TRAIN, rng)
        r_te = sample_rows(f_test, test, N_TEST, rng)
        f_te = sample_rows(fk_te, test, N_TEST, rng)

        rf = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                    min_samples_leaf=MIN_LEAF, n_jobs=-1,
                                    random_state=0)
        rf.fit(np.vstack([r_tr, f_tr]),
               np.r_[np.ones(len(r_tr)), np.zeros(len(f_tr))])
        rules = cut_rules(rf, r_tr, f_tr)
        y = np.r_[np.ones(len(r_te)), np.zeros(len(f_te))]
        xt = np.vstack([r_te, f_te])
        a_ens = auc(y, score(xt, rules))
        a_rf = auc(y, rf.predict_proba(xt)[:, 1])
        n1 = sum(r["класс"] == 1 for r in rules)
        smap = np.full(valid.shape, np.nan, dtype="float32")
        smap[valid] = score(rows_full, rules)
        res[run] = {"auc": a_ens, "auc_rf": a_rf, "rules": rules, "map": smap,
                    "r_te": r_te, "f_te": f_te, "rf": rf,
                    "r_tr": r_tr, "f_tr": f_tr}
        log(f"\n{run}: правил {len(rules)} (снимка {n1}, подделки "
            f"{len(rules) - n1}); AUC на проверке: лес+Кора {a_ens:.3f}, "
            f"чистый лес {a_rf:.3f} ({time.time() - t0:.0f} с)")
        # сильнейшие правила снимка, проверенные на отложенных полосах
        rows = []
        for r in rules:
            h1 = rule_hits(r_te, r["условия"]).mean()
            h2 = rule_hits(f_te, r["условия"]).mean()
            own, oth = (h1, h2) if r["класс"] == 1 else (h2, h1)
            rows.append({"прогон": run,
                         "класс": "снимок" if r["класс"] == 1 else "подделка",
                         "правило": describe(r["условия"], names),
                         "длина": len(r["условия"]),
                         "доля своих (пров)": own, "доля чужих (пров)": oth})
        df = pd.DataFrame(rows)
        df["перевес (пров)"] = df["доля своих (пров)"] / \
            df["доля чужих (пров)"].clip(lower=1.0 / N_TEST)
        res[run]["table"] = df
        top = df[df["класс"] == "снимок"].sort_values("перевес (пров)",
                                                      ascending=False).head(5)
        for _, r in top.iterrows():
            oth = r["доля чужих (пров)"] * N_TEST
            rat = "чужих 0" if oth < 1 else f"x{r['перевес (пров)']:.1f}"
            log(f"  {rat:>8s}  {100 * r['доля своих (пров)']:4.1f} %  "
                f"{r['правило']}")
    return res, names, z, valid, full160


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import pandas as pd

    res, names, z, valid, full160 = fit_runs(a.site, log)
    pd.concat([res[r]["table"] for r, *_ in RUNS]).to_csv(
        M / f"openlin_forest_kora_{a.site}.csv", index=False,
        encoding="utf-8-sig")
    checks(res, z, valid, full160, a.site, log, FK_N_TESTS_PRIOR, "")


def checks(res, z, valid, full160, site, log, n_prior: int, tag: str) -> None:
    """Три условия: AUC в обоих прогонах, устойчивость, совпадение по месту.

    ``n_prior`` — сколько тестов против карты разломов было в ветке ДО этого
    прогона; порог — 0.05 / (n_prior + 2).
    """
    from scipy.stats import spearmanr

    from experiments.openlin_kora_check import fault_masks, shift_p, strike

    alpha = 0.05 / (n_prior + FK_N_TESTS)
    assert 1.0 / (1 + N_SHIFTS) < alpha, "порог недостижим"
    stack160, names160 = full160
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    okA = all(res[r]["auc"] >= AUC_MIN for r, *_ in RUNS)
    log(f"\nAUC >= {AUC_MIN} в обоих прогонах: "
        + ", ".join(f"{r} {res[r]['auc']:.3f}" for r, *_ in RUNS)
        + f" -> {mark[okA]}")

    sA, sB = res["исходный"]["map"], res["повтор"]["map"]
    idx = np.flatnonzero(valid.ravel())
    pick = np.random.default_rng(3).choice(idx, min(50_000, len(idx)),
                                           replace=False)
    rho = spearmanr(sA.ravel()[pick], sB.ravel()[pick]).statistic
    f = {n: stack160[i] for i, n in enumerate(names160)}
    topA = valid & (sA >= np.quantile(sA[valid], 1 - AREA))
    topB = valid & (sB >= np.quantile(sB[valid], 1 - AREA))
    s_ = ("coh_summer_vv_COH12_sin1000", "coh_summer_vv_COH12_cos1000")
    stA, stB = strike(f, topA, *s_), strike(f, topB, *s_)
    d = abs(((stA - stB) + 90.0) % 180.0 - 90.0)
    okS = rho >= SCORE_RHO_MIN and d <= STRIKE_MAX_DIFF
    log(f"устойчивость: корреляция карт {rho:.2f} (порог {SCORE_RHO_MIN}), "
        f"простирание верхних 10 % {stA:.0f}° и {stB:.0f}°, разница {d:.0f}° "
        f"(порог {STRIKE_MAX_DIFF:.0f}°) -> {mark[okS]}")

    faults = fault_masks(z, valid.shape)
    rng = np.random.default_rng(11)
    r1a = shift_p(topA, faults["все"], valid, N_SHIFTS, rng)
    r1b = shift_p(topA, faults["СЗ"], valid, N_SHIFTS, rng)
    log(f"по месту (порог 0.05/{n_prior + FK_N_TESTS} = {alpha:.5f}):")
    log(f"  1а) верхние 10 % против всех зон разломов: {r1a[0]:.2f}x (сдвиг "
        f"{r1a[1]:.2f}x), p = {r1a[2]:.5f} -> {mark[r1a[2] < alpha]}")
    log(f"  1б) верхние 10 % против зон СЗ-разломов: {r1b[0]:.2f}x (сдвиг "
        f"{r1b[1]:.2f}x), p = {r1b[2]:.5f} -> {mark[r1b[2] < alpha]}")

    np.savez_compressed(M / f"openlin_forest_kora_{site}{tag}.npz", score_A=sA,
                        score_B=sB)
    plot(sA, sB, z, valid, site + tag, res, log,
         "«Кора»" if "кора" in tag else "лес + «Кора»")


def plot(sA, sB, z, valid, site, res, log, label: str = "лес + «Кора»") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec",
                                                     "#e34948"])
    fig, ax = plt.subplots(1, 2, figsize=(19, 8.6), facecolor="#fcfcfb")
    for a_, s, run in ((ax[0], sA, "исходный"), (ax[1], sB, "повтор")):
        lim = float(np.nanpercentile(np.abs(s), 99)) or 1.0
        im = a_.imshow(s, cmap=cmap, vmin=-lim, vmax=lim)
        a_.set_title(f"{label}, прогон «{run}»: AUC "
                     f"{res[run]['auc']:.2f}\nкрасное — правила снимка, "
                     f"синее — правила текстуры", fontsize=14, loc="left")
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).outline.set_visible(False)
    p = OUT / f"лес_кора_{site}.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
