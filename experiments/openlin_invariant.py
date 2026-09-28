"""«Лес + Кора» на признаках согласия: не «какое направление», а «согласны ли».

ОТКУДА ПОСТАНОВКА. SHAP по ансамблю (:mod:`openlin_forest_kora_shap`) показал:
в обоих прогонах модель опирается на одни и те же признаки (ранжирование
совпадает на 0.90, пятёрка главных — 5 из 5), и все они — направление, sin/cos
ориентации. Но какое направление считается «снимком», в двух прогонах разное:
у ``ndvi sin 0.5 км`` и ``лет.ког COH12 cos 1 км`` знак вклада перевернулся.
На настоящем снимке есть большие области с одинаковой ориентацией у всех
приборов, но сама ориентация в разных частях листа своя. Модель учит
«снимок — это направление X» по своей половине листа и на другой ломается:
корреляция карт 0.02, простирание 3° и 88°.

ЧТО МЕНЯЕТСЯ. Признаки направления заменяются признаками согласия, которые от
поворота не зависят. Для двух ориентаций в удвоенном угле согласие
cos(2θᵢ - 2θⱼ) = cosᵢ·cosⱼ + sinᵢ·sinⱼ: 1 — одно направление, -1 —
перпендикулярные. Всё прочее — модель, подделки, разбиение, пороги — как в
:mod:`openlin_forest_kora`.

Признаки (41):

* 15 ненаправленных из отобранных 20 (энергия, когерентность, гребень, сырая
  яркость);
* 18 — согласие между приборами: все пары из четырёх (яркость, NDVI, радар VH,
  летняя когерентность COH12) на каждом из трёх масштабов;
* 8 — согласие одного прибора между масштабами: 500 м с 1 км и 1 км с 2 км.

Запуск из корня репозитория:
    python -m experiments.openlin_invariant --site anabar
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from experiments.openlin_forest_kora import (RUNS, builder20, checks,
                                             fit_runs)

M = Path("outputs/metrics")

# ---- пре-регистрация (22.09.2026, до прогона) ------------------------------
INSTRUMENTS = ("gray", "ndvi", "vh", "coh_summer_vv_COH12")
SCALES = (500, 1000, 2000)
#: Сквозной счёт тестов против карты разломов: 225 ветки + 2 «Коры» + 2
#: «леса + Коры» = 229; этот прогон — 230-й и 231-й.
INV_N_TESTS_PRIOR = 229
#: РЕЗУЛЬТАТ «лес + Кора» (22.09.2026): не пройдено ничего. AUC 0.630 и 0.368,
#: чистый лес 0.501 и 0.289 — на второй половине листа ответ ПЕРЕВЁРНУТ;
#: корреляция карт -0.11; по месту 1.11x и 1.17x, p около 0.19. Планка —
#: «Кора» с терцилями на тех же признаках — дала 0.761 и 0.735, первое за ветку,
#: что держит 0.70 в обоих прогонах.
#: Вероятная причина переворота: подделка однородна по листу (текстура везде
#: как в среднем), а снимок нет. Гибкая модель учит «упорядоченнее среднего —
#: снимок» на своей половине, а на другой, где хаотичнее среднего, ответ
#: меняет знак. Грубые терцили «Коры» к этому нечувствительны.
# ---- «Кора» на признаках согласия — новый кандидат (22.09.2026, ДО прогона)
#: Те же условия и тот же код проверок (:func:`openlin_forest_kora.checks`);
#: этот прогон — 232-й и 233-й тесты против карты разломов.
KORA_INV_N_TESTS_PRIOR = 231
#: РЕЗУЛЬТАТ (22.09.2026): одно условие из трёх.
#:   AUC >= 0.70 в обоих прогонах: 0.761 и 0.735 — ПРОЙДЕНО, впервые в ветке.
#:   Устойчивость: корреляция карт 0.57 (порог 0.6), простирание верхних 10 %
#:   62° и 75°, разница 12° (порог 10°) — не пройдено, оба промаха близкие.
#:   По месту: 1.01x (p = 0.51) и 0.72x (p = 0.98) — не пройдено; выделенное
#:   к зонам разломов безразлично, а СЗ-зон даже избегает.
#: Правила «снимка» в обоих прогонах про одно: разные приборы (VH, летняя
#: когерентность, яркость, NDVI) согласны о направлении на 1-2 км, и прибор
#: согласен сам с собой между масштабами. Конкретные сочетания в прогонах
#: разные, тема одна.
# ---------------------------------------------------------------------------

SHORT = {"gray": "яркость", "ndvi": "NDVI", "vh": "VH",
         "coh_summer_vv_COH12": "лет.ког"}


def builder_invariant(site: str):
    """20 отобранных без sin/cos плюс 26 признаков согласия."""
    base = builder20(site)

    def build(stack, names):
        s20, n20 = base(stack, names)
        keep = [i for i, n in enumerate(n20)
                if not any(n.endswith(f"{k}{s}") for k in ("cos", "sin")
                           for s in SCALES)]
        out = [s20[i] for i in keep]
        out_n = [n20[i] for i in keep]
        idx = {n: i for i, n in enumerate(names)}

        def ang(ch, s):
            return stack[idx[f"{ch}_cos{s}"]], stack[idx[f"{ch}_sin{s}"]]

        for s in SCALES:
            for a, b in itertools.combinations(INSTRUMENTS, 2):
                ca, sa = ang(a, s)
                cb, sb = ang(b, s)
                out.append(ca * cb + sa * sb)
                out_n.append(f"согл {SHORT[a]}~{SHORT[b]} {s}")
        for ch in INSTRUMENTS:
            for s1, s2 in ((500, 1000), (1000, 2000)):
                c1, s1v = ang(ch, s1)
                c2, s2v = ang(ch, s2)
                out.append(c1 * c2 + s1v * s2v)
                out_n.append(f"согл {SHORT[ch]} {s1}~{s2}")
        return np.stack(out).astype("float32"), out_n
    return build


def kora_candidate(site: str, log) -> None:
    """«Кора» с терцилями на признаках согласия: полный набор проверок."""
    import pandas as pd

    from experiments.openlin_kora import (N_LEVELS, binarize, describe, kora,
                                          kora_score)
    from experiments.openlin_line_rank import auc

    build = builder_invariant(site)
    res, names, z, valid, full160 = fit_runs(site, log, build)
    full, _n = build(*full160)
    rows_full = full.reshape(full.shape[0], -1)[:, valid.ravel()].T

    out, tabs = {}, []
    for run, *_ in RUNS:
        r = res[run]
        pooled = np.vstack([r["r_tr"], r["f_tr"]])
        edges = np.stack([np.quantile(pooled[:, j],
                                      np.linspace(0, 1, N_LEVELS + 1)[1:-1])
                          for j in range(pooled.shape[1])])
        rules = kora(binarize(r["r_tr"], edges), binarize(r["f_tr"], edges),
                     len(names), lambda *_: None)
        y = np.r_[np.ones(len(r["r_te"])), np.zeros(len(r["f_te"]))]
        b_te = binarize(np.vstack([r["r_te"], r["f_te"]]), edges)
        a_k = auc(y, kora_score(b_te, rules))
        smap = np.full(valid.shape, np.nan, dtype="float32")
        smap[valid] = kora_score(binarize(rows_full, edges), rules)
        out[run] = {"auc": a_k, "map": smap}
        n1 = sum(q["класс"] == 1 for q in rules)
        log(f"\n{run}: правил «Коры» {len(rules)} (снимка {n1}); AUC на "
            f"проверке {a_k:.3f}")
        # сильнейшие правила снимка на проверке
        b_r = binarize(r["r_te"], edges)
        b_f = binarize(r["f_te"], edges)
        for q in rules:
            if q["класс"] != 1:
                continue
            c = list(q["условия"])
            h1, h2 = b_r[:, c].all(1).mean(), b_f[:, c].all(1).mean()
            tabs.append({"прогон": run, "правило": describe(q["условия"], names),
                         "доля снимка": h1, "доля подделки": h2})
        t = pd.DataFrame([x for x in tabs if x["прогон"] == run])
        t["перевес"] = t["доля снимка"] / t["доля подделки"].clip(lower=1e-4)
        for _, q in t.sort_values("перевес", ascending=False).head(6).iterrows():
            rat = ("подделки 0" if q["доля подделки"] * len(b_f) < 1
                   else f"x{q['перевес']:.1f}")
            log(f"  {rat:>11s}  {100 * q['доля снимка']:4.1f} %  {q['правило']}")
    pd.DataFrame(tabs).to_csv(M / f"openlin_kora_invariant_{site}.csv",
                              index=False, encoding="utf-8-sig")
    log("")
    checks(out, z, valid, full160, site, log, KORA_INV_N_TESTS_PRIOR,
           "_кора_инвар")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--kora", action="store_true",
                    help="кандидат — «Кора» с терцилями, а не «лес + Кора»")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    if a.kora:
        kora_candidate(a.site, log)
        return

    import pandas as pd

    from experiments.openlin_kora import N_LEVELS, binarize, kora, kora_score
    from experiments.openlin_line_rank import auc

    res, names, z, valid, full160 = fit_runs(a.site, log,
                                             builder_invariant(a.site))
    pd.concat([res[r]["table"] for r, *_ in RUNS]).to_csv(
        M / f"openlin_invariant_{a.site}.csv", index=False,
        encoding="utf-8-sig")

    # планка: «Кора» с терцилями на тех же признаках
    log("\nпланка — «Кора» на тех же признаках:")
    for run, *_ in RUNS:
        r = res[run]
        pooled = np.vstack([r["r_tr"], r["f_tr"]])
        edges = np.stack([np.quantile(pooled[:, j],
                                      np.linspace(0, 1, N_LEVELS + 1)[1:-1])
                          for j in range(pooled.shape[1])])
        rules = kora(binarize(r["r_tr"], edges), binarize(r["f_tr"], edges),
                     len(names), lambda *_: None)
        y = np.r_[np.ones(len(r["r_te"])), np.zeros(len(r["f_te"]))]
        s = kora_score(binarize(np.vstack([r["r_te"], r["f_te"]]), edges),
                       rules)
        log(f"  {run}: AUC «Кора» {auc(y, s):.3f}, лес+Кора {r['auc']:.3f}, "
            f"чистый лес {r['auc_rf']:.3f}")

    checks(res, z, valid, full160, a.site, log, INV_N_TESTS_PRIOR, "_инвар")


if __name__ == "__main__":
    main()
