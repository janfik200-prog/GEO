"""Классы уверенности вместо бинарного отсева: отдаём всё, но с ярлыком.

ПОВОД. Правило «оба прибора видят линию не меньше чем на половине длины»
выбрасывает 460 веток из 613. Диагностика (``openlin_confirm_diag``) показала,
что это отсев не по истинности, а по доказуемости: среди отброшенных 246 веток
уверенно видит один прибор, и как класс они в разы превышают случайный уровень.
Выбрасывать их — значит терять настоящие линеаменты.

ЧТО ЗДЕСЬ ДЕЛАЕТСЯ. Ветка не отсеивается, а получает класс уверенности, и у
каждого класса измерена своя цена — сколько таких веток прошло бы случайно:

* **A** — оба прибора по отдельности видят линию на половине длины;
* **B** — один прибор видит уверенно, второй подтверждает частично;
* **C** — уверенно видит только один прибор;
* **D** — не подтверждена ни одним; отдаётся с рангом модели, без гарантии.

Классы вложены: A ⊂ B ⊂ C. Где резать — решает геофизик, как и с плотностью.

ВТОРАЯ ПРАВКА: азимут локальный. Раньше направление ветки бралось одно на всю
ветку, и у свидетеля тоже, поэтому на слегка изогнутой линии совпадающие по
положению линии не засчитывались по направлению. Локальный азимут по окну 1.1 км
даёт +4 ветки в классе A и +18 в классе B, причём случайный уровень не растёт.

Запуск из корня репозитория (после openlin_unsup и openlin_mae):
    python -m experiments.openlin_confidence --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import TOL_M
from experiments.openlin_confirm_diag import _az, azimuth_fields
from experiments.openlin_lines import _branches
from experiments.openlin_unsup import CONFIRM_ANGLE_DEG

M = Path("outputs/metrics")

#: Границы классов по долям подтверждения (доля длины ветки).
#: B выбран как «один ≥ 0.6, второй ≥ 0.1»: на Анабаре он даёт 242 ветки при
#: случайном уровне 2.2 (чистота ×108) против 157 и ×196 у класса A.
CLASSES = (
    ("A", lambda o, r: (o >= 0.5) & (r >= 0.5)),
    ("B", lambda o, r: (np.maximum(o, r) >= 0.6) & (np.minimum(o, r) >= 0.1)),
    ("C", lambda o, r: np.maximum(o, r) >= 0.5),
)

#: Сколько сдвигов на случайный уровень каждого класса.
N_SHIFTS = 25


def _fracs(lines: np.ndarray, others: dict[str, np.ndarray], res_m: float
           ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Доли подтверждения каждой ветки по каждому прибору (локальный азимут).

    Возвращает ``(координаты веток, доли оптики, доли радара)``.
    """
    from scipy.ndimage import distance_transform_edt

    _glob_self, loc_self = azimuth_fields(lines, res_m)
    prep = {}
    for tag, other in others.items():
        dt, idx = distance_transform_edt(~other, sampling=(res_m, res_m),
                                         return_indices=True)
        _g, loc_o = azimuth_fields(other, res_m)
        prep[tag] = (dt, idx, loc_o)

    coords, out = [], {t: [] for t in others}
    for pts in _branches(lines, res_m):
        if len(pts) < 3:
            continue
        coords.append(pts)
        r, c = pts[:, 0], pts[:, 1]
        my = loc_self[r, c]
        for tag, (dt, idx, loc_o) in prep.items():
            near = dt[r, c] <= TOL_M
            oth = loc_o[idx[0][r, c], idx[1][r, c]]
            d = np.abs(oth - my)
            same = np.minimum(d, 180.0 - d) <= CONFIRM_ANGLE_DEG
            out[tag].append(float((near & same & np.isfinite(oth)).mean()))
    return coords, np.asarray(out["оптика"]), np.asarray(out["радар"])


def classify(lines: np.ndarray, lo: np.ndarray, lr: np.ndarray, res_m: float,
             shifts: int = N_SHIFTS, log=print) -> tuple[pd.DataFrame, list]:
    """Класс уверенности каждой ветки и случайный уровень каждого класса."""
    coords, o, r = _fracs(lines, {"оптика": lo, "радар": lr}, res_m)
    km = np.array([len(p) * res_m / 1000.0 for p in coords])

    rng = np.random.default_rng(0)
    h, w = lines.shape
    null = []
    for _ in range(shifts):
        so = (int(rng.integers(h)), int(rng.integers(w)))
        sr = (int(rng.integers(h)), int(rng.integers(w)))
        _c, no, nr = _fracs(lines, {"оптика": np.roll(lo, so, (0, 1)),
                                    "радар": np.roll(lr, sr, (0, 1))}, res_m)
        null.append((no, nr))

    cls = np.full(len(coords), "D", dtype="<U1")
    rows = []
    for name, rule in reversed(CLASSES):        # от мягкого к строгому
        keep = rule(o, r)
        cls[keep] = name
    for name, rule in CLASSES:
        keep = rule(o, r)
        n0 = float(np.mean([rule(a, b).sum() for a, b in null]))
        rows.append({"класс": name, "линий": int(keep.sum()),
                     "длина_км": float(km[keep].sum()), "случайно": n0,
                     "чистота": float(keep.sum() / n0) if n0 else np.inf})
    rows.append({"класс": "D", "линий": int((cls == "D").sum()),
                 "длина_км": float(km[cls == "D"].sum()),
                 "случайно": np.nan, "чистота": np.nan})
    tab = pd.DataFrame(rows)
    log(tab.to_string(index=False, formatters={
        "длина_км": "{:.0f}".format, "случайно": "{:.1f}".format,
        "чистота": "×{:.0f}".format}))
    return pd.DataFrame({"класс": cls, "оптика": o, "радар": r,
                         "длина_км": km}), coords


def masks(cls: np.ndarray, coords: list, shape: tuple) -> dict:
    """Растровая маска на каждый класс (вложенные: A ⊂ B ⊂ C)."""
    out = {}
    order = ["A", "B", "C", "D"]
    for i, name in enumerate(order[:3]):
        m = np.zeros(shape, dtype=bool)
        for c, pts in zip(cls, coords):
            if c in order[:i + 1]:
                m[pts[:, 0], pts[:, 1]] = True
        out[name] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--shifts", type=int, default=N_SHIFTS)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    res_m = float(np.load(M / f"openlin_{a.site}.npz")["res_m"])
    out = {}
    for tag, fname in (("pca", f"openlin_unsup_{a.site}_{a.density}.npz"),
                       ("mae", f"openlin_mae_{a.site}_{a.density}.npz")):
        path = M / fname
        if not path.exists():
            log(f"нет {path.name}, пропускаю")
            continue
        z = np.load(path)
        lo, lr, la = z["lines_0"], z["lines_1"], z["lines_2"]
        log(f"\n=== {tag}: {len(_branches(la, res_m))} веток " + "=" * 28)
        df, coords = classify(la, lo, lr, res_m, a.shifts, log)
        out[tag] = masks(df["класс"].to_numpy(), coords, la.shape)
        df.to_csv(M / f"openlin_confidence_{tag}_{a.site}.csv", index=False,
                  encoding="utf-8-sig")

    if len(out) == 2:
        log("\n=== объединение двух способов сжатия " + "=" * 30)
        uni = {k: out["pca"][k] | out["mae"][k] for k in ("A", "B", "C")}
        for k in ("A", "B", "C"):
            log(f"  класс {k}: {uni[k].sum() * res_m / 1000:6.0f} км")
        np.savez_compressed(M / f"openlin_confidence_{a.site}.npz", **uni)
        log(f"\nсохранено: {M / f'openlin_confidence_{a.site}.npz'}")


if __name__ == "__main__":
    main()
