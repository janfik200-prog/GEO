"""ANFIS и логрегрессия вместе: есть ли смысл их объединять.

ВОПРОС. По отдельности они дают почти одно и то же (2.07 против 2.09 на 10
признаках). Объединение имеет смысл ТОЛЬКО если модели ошибаются в разных
местах: усреднение двух почти одинаковых карт даёт третью почти такую же.
Поэтому первым делом считается не lift, а ранговая корреляция скоров — если она
близка к единице, дальше идти незачем, и это выясняется за минуту.

ЧТО СЧИТАЕТСЯ. На одних и тех же строках и том же блочном разбиении обучаются
обе модели, и сравниваются четыре карты: ANFIS, логрегрессия, их ранговое
среднее и ранговый максимум. Ранговое, а не арифметическое: у ANFIS выход после
сигмоиды, у логрегрессии вероятность, шкалы разные, и среднее сырых значений
меряло бы разницу шкал, а не согласие.

ТЕСТОВ ЗДЕСЬ НЕТ. Ни одного сдвигового нуля и ни одного p-value: прогон читается
только по lift, как и измерение цены утечки. Знаменатель Бонферрони не растёт.
Если гибрид окажется заметно выше обеих составляющих, он станет кандидатом и
получит свой addend и свою пре-регистрацию ОТДЕЛЬНЫМ прогоном.

Запуск из корня репозитория:
    python -m experiments.openlin_anfis_hybrid --site nevada --apply-to anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from experiments.openlin_anfis import (AREA, N_RULES, N_TRAIN_ROWS, SEED,
                                       TARGET_HALFWIDTH_M, TOL_M,
                                       anfis_predict, baseline_predict,
                                       block_split, fit_anfis, fit_baselines,
                                       rows, score_map, soft_target)
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, lift_at_top, load_site

M = Path("outputs/metrics")


def build(site: str, support: np.ndarray | None, neigh: bool):
    z = load_site(site)
    x, _y, valid, use = build_stack(z, CHANNELS)
    if neigh:
        nx, _n = neigh_stack(x, use, valid, float(z["res_m"]),
                             support=valid if support is None else support)
        x = np.concatenate([x, nx])
    return z, x, valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="nevada")
    ap.add_argument("--apply-to", default="anabar,anabar_dykes")
    ap.add_argument("--rules", type=int, default=N_RULES)
    ap.add_argument("--targets", default="hard,smooth")
    a = ap.parse_args()

    apply_sites = [s for s in a.apply_to.split(",") if s]
    out = []
    for neigh in (False, True):
        z0 = load_site(a.site)
        v0 = np.isfinite(z0["gray"])
        tr, va = block_split(v0.shape, float(z0["res_m"]))
        tr &= v0
        va &= v0
        _z, xt, _v = build(a.site, tr, neigh)
        _zv, xv, _vv = build(a.site, va, neigh)

        xtr = rows(xt, tr)
        dtr = z0["dist"][tr]
        rng = np.random.default_rng(SEED)
        if len(xtr) > N_TRAIN_ROWS:
            sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
            xtr, dtr = xtr[sel], dtr[sel]
        n_feat = xtr.shape[1]

        for tname in [t for t in a.targets.split(",") if t]:
            soft = tname == "smooth"
            ytr = (soft_target(dtr) if soft
                   else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))
            anfis = fit_anfis(xtr, ytr, a.rules, soft, lambda *_: None)
            logreg = fit_baselines(xtr, ytr, soft)["логрегрессия"]

            jobs = [(a.site, xv, va, "проверка", z0)]
            for s in apply_sites:
                z, x, valid = build(s, None, neigh)
                jobs.append((s, x, valid, "перенос", z))

            for site, x, area, mode, z in jobs:
                xr = rows(x, area)
                pa = anfis_predict(anfis, xr)
                pl = baseline_predict(logreg, xr)
                ra, rl = rankdata(pa) / len(pa), rankdata(pl) / len(pl)
                cards = {"ANFIS": pa, "логрегрессия": pl,
                         "ранговое среднее": 0.5 * (ra + rl),
                         "ранговый максимум": np.maximum(ra, rl)}
                rho = float(spearmanr(pa, pl).statistic)
                for name, s_rows in cards.items():
                    sc = score_map(s_rows, area)
                    got = lift_at_top(np.nan_to_num(sc, nan=-1e9), z["dist"],
                                      area, fraction=AREA, tol_m=TOL_M)
                    out.append({"признаков": n_feat, "таргет": tname,
                                "площадка": site, "режим": mode,
                                "карта": name, "lift": got["lift"],
                                "спирмен_ANFIS_логрег": rho})
                print(f"  {n_feat:>2} призн. · {tname:<6} · {site:<12} {mode:<9} "
                      f"спирмен {rho:+.3f}  " +
                      "  ".join(f"{k} {lift_at_top(np.nan_to_num(score_map(v, area), nan=-1e9), z['dist'], area, fraction=AREA, tol_m=TOL_M)['lift']:.3f}"
                                for k, v in cards.items()))

    df = pd.DataFrame(out)
    path = M / f"openlin_anfis_hybrid_{a.site}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print("\n=== выигрывает ли гибрид у лучшей из составляющих ===")
    piv = df.pivot_table(index=["признаков", "таргет", "площадка", "режим"],
                         columns="карта", values="lift")
    piv["лучшая одиночная"] = piv[["ANFIS", "логрегрессия"]].max(axis=1)
    piv["прирост среднего"] = piv["ранговое среднее"] - piv["лучшая одиночная"]
    print(piv.round(3).to_string())
    print(f"\nсредний прирост рангового среднего: "
          f"{piv['прирост среднего'].mean():+.3f}; "
          f"выиграл в {int((piv['прирост среднего'] > 0).sum())} случаях из "
          f"{len(piv)}")
    print(f"медианный спирмен ANFIS-логрегрессия: "
          f"{df['спирмен_ANFIS_логрег'].median():.3f}")
    print(f"\nсохранено: {path}")


if __name__ == "__main__":
    main()
