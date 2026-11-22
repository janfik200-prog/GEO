"""Откуда верх пробной карты v0.3a: разбор по объектам (снятие по одному).

Вопрос. Карта v0.3a поднимает восточную полосу у контакта рифея с
фундаментом и совпадает с картой ВНИГНИ на +0.24 (ранговая корреляция). Если
верх держится на одном объекте — несогласии, которое в v0.3a насыщено
литературным радиусом 1–2 км, — это повтор одного слоя, а не сходство
нескольких звеньев (та же проверка, что для линеаментов на уступах).

Записано до расчёта. Объект «держит» верх, если без него (а) верхние 20 %
карты совпадают с полной картой меньше чем на половину и (б) доля верхних
20 % в полосе у контакта (несогласие ближе 300 м: кровля фундамента не
глубже 300 м или выход фундамента ближе 300 м к рифею) падает больше чем
вдвое. Снимается по одному: несогласие (в путях и местах накопления),
складки, кромки полей в К4, вся разломная зона.

Оговорка: складки (выпуклость кровли), несогласие (глубина кровли) и
палеопонижения сохранности (К28г) построены по одной поверхности — кровле
фундамента из карты свит. Это три разных свойства поверхности (форма,
глубина, отклонение от наклона), но не три независимых наблюдения.

Карта ВНИГНИ — только заверка: ранговая корреляция показана для понимания,
откуда берётся сходство, ничего под неё не подбирается.

Запуск::

    python -X utf8 -m experiments.gmo_v03_diagnostics            # v0.3a
    python -X utf8 -m experiments.gmo_v03_diagnostics --v0-3b
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gmo_knowledge_map_v0 as km  # noqa: E402
from src import criterial_target  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TOP, NEAR_M = 0.2, 300.0


def without(model: dict, link: str | None, obj: str, crit: str | None = None) -> dict:
    m = copy.deepcopy(model)
    for l, objs in m.items():
        if link and l != link:
            continue
        if obj in objs:
            if crit:
                objs[obj].pop(crit, None)
                if not objs[obj]:
                    objs.pop(obj)
            else:
                objs.pop(obj)
    return m


def main() -> None:
    ver = "--v0-3b" if "--v0-3b" in sys.argv else "--v0-3a"
    km.configure([ver])
    df = km.load()
    meta, prog = criterial_target.load_prognoz_grid()
    vn = -np.asarray(prog, dtype=float).reshape(meta.prf, meta.pic)[df.row, df.col]
    near = (df.unc_dist <= NEAR_M).to_numpy()
    base = copy.deepcopy(km.MODEL)
    cases = [("полная карта", base),
             ("без несогласия (О2)", without(base, None, km.UNC)),
             ("без складок (О9)", without(base, None, "О9 складки")),
             ("без кромок полей (К4)", without(base, None, km.FAULT, "К4 разломные границы в полях")),
             ("без разломной зоны (О1)", without(base, None, km.FAULT))]
    rows, full = [], None
    for name, model in cases:
        km.MODEL.clear()
        km.MODEL.update(model)
        m = km.build(df)
        r = m.итог.rank(pct=True).to_numpy()
        top = r >= 1 - TOP
        if full is None:
            full = top
        ok = np.isfinite(vn) & np.isfinite(r)
        rows.append({"вариант": name,
                     "верх совпадает с полной, %": round(100 * (top & full).sum() / full.sum()),
                     "верх у контакта, %": round(100 * (top & near).sum() / top.sum()),
                     "ρ с ВНИГНИ": round(spearmanr(r[ok], vn[ok]).statistic, 2)})
    t = pd.DataFrame(rows)
    print(f"Версия {ver[2:]}; полоса у контакта — {100 * near.mean():.0f}% листа "
          f"(случайный верх попал бы туда на столько же)")
    print(t.to_string(index=False))
    out = ROOT / f"data/processed/v03_diagnostics_{ver[2:].replace('-', '_')}.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Таблица: {out}")


if __name__ == "__main__":
    main()
