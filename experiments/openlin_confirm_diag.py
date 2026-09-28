"""Почему отсев по подтверждению выбрасывает так много линий.

ПОВОД. Геофизики смотрят на отсеянные ветки и видят среди них линеаменты.
Правило отбора («оба прибора, не меньше половины длины») — это фильтр на
ДОКАЗУЕМОСТЬ, а не на истинность: линия, которую хорошо видит только радар,
через него не проходит, хотя она реальна. Здесь разбирается, какая часть
отсева объясняется существом дела, а какая — техникой проверки.

ТРИ ПОДОЗРЕНИЯ, которые проверяются на данных:

1. ГЛОБАЛЬНЫЙ АЗИМУТ. Сейчас направление ветки берётся одно на всю ветку
   (главная ось), и у свидетеля тоже. На слегка изогнутой линии локальное
   направление отличается от главной оси, и совпадающие по положению линии не
   засчитываются по направлению. Здесь считается локальный азимут по
   скользящему окну вдоль ветки — и для нашей линии, и для свидетеля.

2. ВСЁ ИЛИ НИЧЕГО. Ветка проходит или не проходит целиком. Линия, подтверждённая
   на 40 % длины, выбрасывается вместе с подтверждённым куском.

3. ЖЁСТКОЕ «И». Требование обоих приборов режет линии, видимые одним. Вопрос не
   в том, ослаблять ли его, а в том, какой ценой: у каждого послабления
   считается свой случайный уровень тороидальным сдвигом.

Ничего не пересчитывается заново: берутся сохранённые карты линий прогона
``openlin_unsup``.

Запуск из корня репозитория:
    python -m experiments.openlin_confirm_diag --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import TOL_M
from experiments.openlin_lines import _axis, _branches
from experiments.openlin_unsup import CONFIRM_ANGLE_DEG, CONFIRM_FRAC

M = Path("outputs/metrics")

#: Окно скользящего азимута, ячеек. 11 = 1.1 км: короче — направление гуляет по
#: ступенькам растра, длиннее — окно снова спрямляет изгиб.
WIN = 11

#: Сколько сдвигов на случайный уровень. 25 — как в основном прогоне.
N_SHIFTS = 25


def _order(pts: np.ndarray) -> np.ndarray:
    """Порядок ячеек вдоль ветки: по проекции на её главную ось."""
    ax, _a, _b = _axis(pts)
    return np.argsort((pts - pts.mean(0)) @ ax)


def _az(pts: np.ndarray) -> float:
    """Азимут набора точек, градусы 0-180."""
    ax, _a, _b = _axis(pts)
    return float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180)


def azimuth_fields(lines: np.ndarray, res_m: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Два растра азимутов: глобальный по ветке и локальный по окну."""
    glob = np.full(lines.shape, np.nan, dtype="float32")
    loc = np.full(lines.shape, np.nan, dtype="float32")
    for pts in _branches(lines, res_m):
        if len(pts) < 3:
            continue
        a = _az(pts)
        glob[pts[:, 0], pts[:, 1]] = a
        if len(pts) < WIN:
            loc[pts[:, 0], pts[:, 1]] = a
            continue
        idx = _order(pts)
        half = WIN // 2
        for j, i in enumerate(idx):
            lo, hi = max(0, j - half), min(len(idx), j + half + 1)
            loc[pts[i, 0], pts[i, 1]] = _az(pts[idx[lo:hi]])
    return glob, loc


def _frac(pts: np.ndarray, my_az: np.ndarray, dt: np.ndarray,
          idx: tuple[np.ndarray, np.ndarray], az_other: np.ndarray,
          angle_deg: float) -> float:
    """Доля ячеек ветки, где рядом есть чужая линия того же направления."""
    r, c = pts[:, 0], pts[:, 1]
    near = dt[r, c] <= TOL_M
    oth = az_other[idx[0][r, c], idx[1][r, c]]
    d = np.abs(oth - my_az)
    same = np.minimum(d, 180.0 - d) <= angle_deg
    return float((near & same & np.isfinite(oth)).mean())


def table(lines: np.ndarray, others: dict[str, np.ndarray], res_m: float,
          log) -> pd.DataFrame:
    """Доли подтверждения каждой ветки: глобальный и локальный азимут."""
    from scipy.ndimage import distance_transform_edt

    _g_self, loc_self = azimuth_fields(lines, res_m)
    prep = {}
    for tag, other in others.items():
        dt, idx = distance_transform_edt(~other, sampling=(res_m, res_m),
                                         return_indices=True)
        glob_o, loc_o = azimuth_fields(other, res_m)
        prep[tag] = (dt, idx, glob_o, loc_o)

    rows = []
    for pts in _branches(lines, res_m):
        if len(pts) < 3:
            continue
        r, c = pts[:, 0], pts[:, 1]
        rec = {"ячеек": len(pts), "длина_км": len(pts) * res_m / 1000.0}
        for tag, (dt, idx, glob_o, loc_o) in prep.items():
            rec[f"{tag}_глоб"] = _frac(pts, _az(pts), dt, idx, glob_o,
                                       CONFIRM_ANGLE_DEG)
            rec[f"{tag}_лок"] = _frac(pts, loc_self[r, c], dt, idx, loc_o,
                                      CONFIRM_ANGLE_DEG)
        rows.append(rec)
    df = pd.DataFrame(rows)
    log(f"веток в рабочем списке: {len(df)}")
    return df


#: Правила отбора: имя и предикат по долям подтверждения. Первый аргумент —
#: доля подтверждения оптикой, второй — радаром.
RULES: dict[str, callable] = {
    "только оптика ≥ 0.5 (радар < 0.25)":
        lambda o, r: (o >= 0.5) & (r < 0.25),
    "только радар ≥ 0.5 (оптика < 0.25)":
        lambda o, r: (r >= 0.5) & (o < 0.25),
    "оба ≥ 0.5 (сейчас)":
        lambda o, r: (o >= 0.5) & (r >= 0.5),
    "оба ≥ 0.35":
        lambda o, r: (o >= 0.35) & (r >= 0.35),
    "оба ≥ 0.25":
        lambda o, r: (o >= 0.25) & (r >= 0.25),
    "один ≥ 0.5, второй ≥ 0.25":
        lambda o, r: (np.maximum(o, r) >= 0.5) & (np.minimum(o, r) >= 0.25),
    "один ≥ 0.6, второй ≥ 0.1":
        lambda o, r: (np.maximum(o, r) >= 0.6) & (np.minimum(o, r) >= 0.1),
    "сумма ≥ 0.8":
        lambda o, r: (o + r) >= 0.8,
    "хотя бы один ≥ 0.5":
        lambda o, r: (o >= 0.5) | (r >= 0.5),
}


def evaluate(df: pd.DataFrame, suffix: str, null: list[pd.DataFrame],
             log) -> pd.DataFrame:
    """Для каждого правила: сколько проходит, сколько случайно, чистота."""
    o, r = df[f"оптика{suffix}"].to_numpy(), df[f"радар{suffix}"].to_numpy()
    km = df["длина_км"].to_numpy()
    out = []
    for name, rule in RULES.items():
        keep = rule(o, r)
        n0 = float(np.mean([rule(d[f"оптика{suffix}"].to_numpy(),
                                 d[f"радар{suffix}"].to_numpy()).sum()
                            for d in null])) if null else float("nan")
        out.append({"правило": name, "линий": int(keep.sum()),
                    "длина_км": float(km[keep].sum()),
                    "случайно": n0,
                    "чистота": float(keep.sum() / n0) if n0 else float("inf")})
    res = pd.DataFrame(out)
    log(f"\n=== азимут: {'локальный' if 'лок' in suffix else 'глобальный'} "
        + "=" * 30)
    log(res.to_string(index=False,
                      formatters={"длина_км": "{:.0f}".format,
                                  "случайно": "{:.1f}".format,
                                  "чистота": "×{:.0f}".format}))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--shifts", type=int, default=N_SHIFTS)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    z = np.load(M / f"openlin_unsup_{a.site}_{a.density}.npz")
    res_m = float(np.load(M / f"openlin_{a.site}.npz")["res_m"])
    lo, lr, la = z["lines_0"], z["lines_1"], z["lines_2"]
    field = z["field"]
    log(f"площадка {a.site}, плотность '{a.density}', шаг {res_m:.0f} м")
    log(f"линий: оптика {lo.sum() * res_m / 1000:.0f} км, "
        f"радар {lr.sum() * res_m / 1000:.0f} км, "
        f"рабочий список {la.sum() * res_m / 1000:.0f} км")

    df = table(la, {"оптика": lo, "радар": lr}, res_m, log)

    # --- случайный уровень: те же ветки против сдвинутых свидетелей ---------
    rng = np.random.default_rng(0)
    h, w = la.shape
    null = []
    for i in range(a.shifts):
        sh_o = (int(rng.integers(h)), int(rng.integers(w)))
        sh_r = (int(rng.integers(h)), int(rng.integers(w)))
        null.append(table(la, {"оптика": np.roll(lo, sh_o, (0, 1)),
                               "радар": np.roll(lr, sh_r, (0, 1))},
                          res_m, lambda *_a: None))
        if i == 0:
            log(f"случайный уровень: {a.shifts} сдвигов, считается...")

    res_g = evaluate(df, "_глоб", null, log)
    res_l = evaluate(df, "_лок", null, log)

    # --- что именно теряется на текущем правиле ----------------------------
    o, r = df["оптика_глоб"].to_numpy(), df["радар_глоб"].to_numpy()
    keep = (o >= CONFIRM_FRAC) & (r >= CONFIRM_FRAC)
    drop = ~keep
    near = drop & (np.maximum(o, r) >= 0.5) & (np.minimum(o, r) >= 0.25)
    one = drop & (np.maximum(o, r) >= 0.5) & (np.minimum(o, r) < 0.25)
    none = drop & (np.maximum(o, r) < 0.5)
    log("\n=== из чего состоит отсев (текущее правило) " + "=" * 24)
    for name, m in (("прошли", keep), ("отсеяно всего", drop),
                    ("  подтверждены одним сильно, вторым частично", near),
                    ("  подтверждены только одним прибором", one),
                    ("  не подтверждены ни одним", none)):
        log(f"  {name:48s} {int(m.sum()):4d} веток, "
            f"{df['длина_км'].to_numpy()[m].sum():6.0f} км")

    # --- насколько отсеянные похожи на настоящие по силе границы -----------
    parts = _branches(la, res_m)
    strength = np.array([float(np.nanmean(field[p[:, 0], p[:, 1]]))
                         for p in parts if len(p) >= 3])
    log("\n=== средняя сила границы вдоль ветки, сигмы " + "=" * 24)
    for name, m in (("прошли", keep), ("одним сильно + вторым частично", near),
                    ("только одним прибором", one),
                    ("ни одним", none)):
        if m.sum():
            log(f"  {name:36s} {strength[m].mean():.2f} "
                f"(медиана {np.median(strength[m]):.2f})")

    df["сила"] = strength
    df.to_csv(M / f"openlin_confirm_diag_{a.site}.csv", index=False,
              encoding="utf-8-sig")
    pd.concat([res_g.assign(азимут="глобальный"),
               res_l.assign(азимут="локальный")]).to_csv(
        M / f"openlin_confirm_rules_{a.site}.csv", index=False,
        encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_confirm_diag_{a.site}.csv'}")


if __name__ == "__main__":
    main()
