"""Сводка: что дали признаки окрестности поверх поячеечной таблицы.

Кладёт рядом два прогона одного и того же протокола — без окрестности
(``openlin_anfis_<site>.csv``) и с ней (``..._neigh.csv``) — по паре
(площадка обучения, площадка оценки), модели и таргету.

Запуск из корня репозитория:
    python -m experiments.openlin_neigh_report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.openlin_anfis import NB_ALPHA

M = Path("outputs/metrics")
SITES = ("nevada", "anabar", "anabar_dykes")
SHORT = {"nevada": "Невада", "anabar": "Анабар", "anabar_dykes": "дайки"}


def load(suffix: str) -> pd.DataFrame:
    out = []
    for s in SITES:
        p = M / f"openlin_anfis_{s}{suffix}.csv"
        if not p.exists():
            continue
        f = pd.read_csv(p)
        f.insert(0, "обучение", SHORT[s])
        out.append(f)
    if not out:
        raise SystemExit(f"нет ни одного файла с суффиксом '{suffix}'")
    df = pd.concat(out, ignore_index=True)
    df["площадка"] = df["площадка"].map(SHORT).fillna(df["площадка"])
    return df


def main() -> None:
    key = ["обучение", "площадка", "режим", "модель", "таргет"]
    base = load("").query("разбиение == 'block'")
    neigh = load("_neigh").query("разбиение == 'block'")

    m = base.merge(neigh, on=key, suffixes=("_без", "_с"))
    m["прирост"] = m["lift_с"] - m["lift_без"]
    cols = key + ["lift_без", "lift_с", "прирост"]
    if "p_с" in m:
        cols += ["p_без", "p_с"]
    m = m[cols].sort_values("прирост", ascending=False)

    print("=== блочная проверка: без окрестности -> с окрестностью " + "=" * 12)
    print(m.round(3).to_string(index=False))

    print("\n=== средний прирост lift по моделям " + "=" * 28)
    print(m.groupby(["модель", "режим"])["прирост"].agg(["mean", "max", "count"])
          .round(3).to_string())

    if "p_с" in m:
        win = m[m["p_с"] < NB_ALPHA]
        print(f"\nзначимо при alpha = {NB_ALPHA:.5f}: {len(win)} из {len(m)}")
        if len(win):
            print(win.round(4).to_string(index=False))

    out = M / "openlin_neigh_report.csv"
    m.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
