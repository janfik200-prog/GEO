"""Снимок + рельеф: полнота по эталону GTK у трёх наборов (площадка настройки).

Набор снимка — рабочий тракт (UTS1, ``_настройка1``); набор рельефа — тот же
тракт по уклону GLO-30 с подтверждением уклоном ALOS (RELU1, ``_рельеф1``).
Сложение — правило из отчёта Артура (25.10.2026, раздел 5.3): линия рельефа
добавляется, если менее ``ADD_FRAC`` её длины лежит в ``ADD_TOL_M`` от линий
снимка. Линия рельефа здесь — связная компонента маски (8-связность): в npz
лежат маски, а не списки линий; у сшитых «цепочек» компонента и линия
совпадают, кроме пересечений, где две линии сливаются в одну компоненту.

Мера — полнота по слою 292 в ``TOL_M`` = 300 м и сдвиговый нуль эталона, тем
же кодом, что :mod:`openlin_transfer_check`. На площадке настройки — для
сведения, в счётчик не идёт (объявление RELU1).

    python -m experiments.openlin_union_recall --site utsjoki
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.openlin_transfer_check import (COMP, GTK_SHIFT_N, GTK_SHIFT_SEED,
                                                M, MAIN_LAYER, TOL_M, gtk_raster,
                                                recall, shift_null)

TUNE_SITES = ("utsjoki",)
ADD_FRAC = 0.30
ADD_TOL_M = 60.0


def add_new(relief: np.ndarray, optic: np.ndarray, res: float) -> np.ndarray:
    """Компоненты рельефа, у которых меньше ADD_FRAC ячеек в ADD_TOL_M от снимка."""
    from scipy.ndimage import distance_transform_edt, label

    d = distance_transform_edt(~optic, sampling=(res, res))
    lab, n = label(relief, structure=np.ones((3, 3), int))
    if not n:
        return np.zeros_like(relief)
    near = np.bincount(lab.ravel(), weights=(d <= ADD_TOL_M).ravel(), minlength=n + 1)
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = near / np.maximum(size, 1) < ADD_FRAC
    keep[0] = False
    return keep[lab]


def main() -> None:
    from scipy.ndimage import distance_transform_edt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--optic", default="_настройка1")
    ap.add_argument("--relief", default="_рельеф1")
    a = ap.parse_args()
    if a.site not in TUNE_SITES:
        raise SystemExit("сложение со сверкой по эталону — только на площадке настройки")

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    o = np.load(M / f"openlin_zone_lines_{a.site}_v3{a.optic}.npz")
    r = np.load(M / f"openlin_zone_lines_{a.site}_v3{a.relief}.npz")
    res = float(w["res_m"])
    shape = o["C_all"].shape
    valid = np.isfinite(w["shade"][:shape[0], :shape[1]])
    ref = gtk_raster(MAIN_LAYER, w, shape) & valid
    print(f"площадка {a.site}: эталон {MAIN_LAYER} ~{ref.sum() * res / 1000:.0f} км "
          f"ячеек; допуск {TOL_M:.0f} м, нуль — {GTK_SHIFT_N} сдвигов эталона")

    rows = []
    for name, key in (("в зонах", "C"), ("весь лист", "C_all")):
        opt_m = o[key][:shape[0], :shape[1]]
        rel_m = r[key][:shape[0], :shape[1]]
        new_m = add_new(rel_m, opt_m, res)
        sets = {"снимок": opt_m, "рельеф": rel_m,
                "добавлено рельефом": new_m, "снимок + рельеф": opt_m | new_m}
        print(f"\nнабор «{name}»:")
        for s, m in sets.items():
            dist = distance_transform_edt(~m, sampling=(res, res))
            obs = recall(ref, dist, TOL_M)
            null, mu = shift_null(ref, dist, valid, TOL_M, GTK_SHIFT_N, GTK_SHIFT_SEED)
            print(f"  {s:20s} ~{m.sum() * res / 1000:5.0f} км: полнота {100 * obs:5.1f} % "
                  f"(нуль {100 * mu:4.1f} %, сверх нуля {100 * (obs - mu):+5.1f})")
            rows.append(dict(набор=name, слой=s, км=m.sum() * res / 1000,
                             полнота=obs, нуль=mu))
    import pandas as pd
    out = M / f"openlin_union_recall_{a.site}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
