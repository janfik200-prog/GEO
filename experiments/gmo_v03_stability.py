"""Устойчивость карты на знаниях к настройкам, выбранным проектом без опоры.

Вопрос. Несколько чисел в карте v0.3c выбраны нами, а не взяты из знаний:
доля ячеек с полной мерой у критериев с радиусом (`Q_CORE` = 25 %), во
сколько раз дальше предел спада (`SPREAD` = 2) и сглаживание кровли
фундамента для складок (`SIGMA_M` = 2 км). Если верх карты заметно
меняется от соседних значений, карта держится на настройках, а не на
знаниях, и учить на ней рано.

Записано до расчёта. Карта устойчива к настройке, если при каждом соседнем
значении (а) ранговая корреляция итога с базовой картой не ниже 0.9 и
(б) верхние 20 % совпадают с базовыми не меньше чем на 70 %. Соседние
значения: доля 15 и 35 %; спад 1.5 и 3; сглаживание 1 и 4 км.

Порог видимости линеаментов (`MIN_SEEN` = 5 %: страта «свита × квинтиль
расчленённости» наблюдаема, если линии есть хотя бы в такой доле её ячеек)
проверяется значениями 2.5 и 10 %. Пересчитывать линии не нужно: порог
работает только при ранжировании уже посчитанных каналов способа Г
(`experiments/gmo_lineaments_background.py::rank_method`), здесь это
ранжирование повторено.

Карта ВНИГНИ и золото не используются: проверяется только сама карта.

Запуск::

    python -X utf8 -u -m experiments.gmo_v03_stability --вариант-б
    python -X utf8 -u -m experiments.gmo_v03_stability          # ветка А
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gmo_knowledge_map_v0 as km  # noqa: E402
from experiments import gmo_section_channels as sec  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TOP, RHO_MIN, TOP_MIN = 0.2, 0.9, 0.70


def visibility(df: pd.DataFrame, min_seen: float) -> pd.DataFrame:
    """Способ Г с другим порогом видимости: ранги в страте и маска наблюдаемости."""
    from experiments import gmo_lineaments_background as bgm
    bg = pd.read_parquet(bgm.OUT, columns=["row", "col", "bg_relief_q", "lincx_dens",
                                            "lincx_node_dens", "lincx_dist"])
    bg = bg.merge(pd.read_parquet(bgm.K23, columns=["row", "col", "svk_main"]),
                  on=["row", "col"], how="left")
    strata = bg.svk_main.astype(str) + "|" + bg.bg_relief_q.astype(str)
    obs = (bg.lincx_dens > 0).groupby(strata).transform("mean") >= min_seen
    new = bg[["row", "col"]].copy()
    for ch, col in (("dens", "lin_dens"), ("node_dens", "lin_node_dens"), ("dist", "lin_dist")):
        new[col] = bg[f"lincx_{ch}"].groupby(strata).rank(pct=True).where(obs)
    lc = pd.read_parquet(km.LINC, columns=["row", "col", "linc_dist"])
    new = new.merge(lc, on=["row", "col"], how="left")
    new["lin_dist_m"] = new.linc_dist.where(obs.to_numpy())
    d = df.drop(columns=["lin_dens", "lin_node_dens", "lin_dist", "lin_dist_m"])
    return d.merge(new.drop(columns="linc_dist"), on=["row", "col"], how="left")


def build(df: pd.DataFrame, argv: list[str], q=None, spread=None, sigma=None,
          min_seen=None) -> pd.Series:
    mod = importlib.reload(km)
    if q is not None:
        mod.Q_CORE = q
    if spread is not None:
        mod.SPREAD = spread
    mod.configure(argv)
    d = visibility(df, min_seen) if min_seen is not None else df.copy()
    d["unc_fault"] = mod.membership(d.unc_dist, mod.R_K7) * mod.membership(d.lin_dist_m, mod.R_K2)
    if sigma is not None:
        g = pd.read_parquet(sec.GRID)
        f = sec.fold_channels(g, sigma)
        d = d.drop(columns="fold_convex").merge(f[["row", "col", "fold_convex"]],
                                               on=["row", "col"], how="left")
    return mod.build(d).итог


def main() -> None:
    argv = ["--v0-3c"] + (["--вариант-б"] if "--вариант-б" in sys.argv else [])
    importlib.reload(km).configure(argv)
    df = km.load()
    base = build(df, argv)
    rb = base.rank(pct=True)
    top0 = rb >= 1 - TOP
    rows = []
    for name, kw in [("доля полной меры 15 %", {"q": 0.15}), ("доля полной меры 35 %", {"q": 0.35}),
                     ("спад до ×1.5", {"spread": 1.5}), ("спад до ×3", {"spread": 3.0}),
                     ("сглаживание складок 1 км", {"sigma": 1000.0}),
                     ("сглаживание складок 4 км", {"sigma": 4000.0}),
                     ("видимость линий 2.5 %", {"min_seen": 0.025}),
                     ("видимость линий 10 %", {"min_seen": 0.10})]:
        it = build(df, argv, **kw)
        r = it.rank(pct=True)
        rho = spearmanr(it, base, nan_policy="omit").statistic
        share = ((r >= 1 - TOP) & top0).sum() / top0.sum()
        rows.append({"настройка": name, "ρ с базовой": round(rho, 3),
                     "верх 20 % совпадает": f"{share:.0%}",
                     "устойчиво": "да" if rho >= RHO_MIN and share >= TOP_MIN else "нет"})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    print(f"\nВетка {'Б' if len(argv) > 1 else 'А'}, база v0.3c (доля 25 %, спад ×2, сглаживание 2 км, видимость 5 %); "
          f"порог: ρ ≥ {RHO_MIN}, верх ≥ {TOP_MIN:.0%}")
    print(t.to_string(index=False))
    out = ROOT / f"data/processed/v03_stability{'_b' if len(argv) > 1 else ''}.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Таблица: {out}")


if __name__ == "__main__":
    main()
