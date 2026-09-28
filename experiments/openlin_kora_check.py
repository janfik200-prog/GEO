"""Проверка результата «Коры»: совпадает ли по МЕСТУ и устойчив ли.

ЧТО ПРОВЕРЯЕТСЯ. «Кора» нашла, что снимок отличается от текстуры согласованным
направлением: три независимых прибора (летняя когерентность радара, радар VH,
NDVI) видят одно и то же простирание на масштабе 1-2 км. Преобладающее
простирание таких зон ~127° (СЗ), вне их ~51-56° (СВ), и оба совпадают с
направлениями двух картированных систем глубинных разломов. Но совпадение
направлений — слабое свидетельство: систем всего две. Здесь проверяется место.

ПРОВЕРКА 1 (карта разломов — только для сверки, в построение не входит):

* 1а — верхние 10 % площади по оценке «Коры» против всех зон глубинных
  разломов: lift@10 %;
* 1б — зона, где все три прибора согласны на СЗ-направлении (первое правило
  «Коры»), против зон СЗ-разломов: во сколько раз доля разломных ячеек в зоне
  выше, чем по листу.

Значимость — тороидальным сдвигом карты (сохраняет её автокорреляцию и
форму). Счёт тестов СКВОЗНОЙ по линеаментной ветке, как требует дисциплина
проекта: до этого против карты разломов в ней проверено 225 гипотез
(``RANK_N_TESTS_PRIOR + RANK_N_TESTS`` в :mod:`openlin_line_rank`), эти две — 226
и 227-я.

ПРОВЕРКА 2 — устойчивость: «Кора» заново с поменянными местами полосами
обучения и проверки и с новыми подделками. Преобладающее простирание в верхних
10 % оценки не должно уйти больше чем на ``STRIKE_MAX_DIFF``, а ранговая
корреляция двух карт оценки — быть не ниже ``SCORE_RHO_MIN``.

РЕЗУЛЬТАТ (22.09.2026) — не пройдено ни одно из трёх условий.

* 1а: lift 0.45 при нуле 1.03, p = 0.976. Верхние 10 % оценки не просто не
  совпадают с зонами разломов, а избегают их. Зоны разломов при этом занимают
  55 % листа, так что потолок lift — 1.83.
* 1б: 1.39x при нуле 0.98x, p = 0.0155 — номинально меньше 0.05, но порог по
  сквозному счёту 0.00022, и до него два порядка.
* 2: карты оценки коррелируют на 0.64 (порог 0.6 взят), но преобладающее
  простирание верхних 10 % — 74° и 103°, разница 29° при пороге 10°. AUC
  повтора 0.679 (исходный 0.706); бустинг упал с 0.738 до 0.604, то есть
  «Кора» устойчивее бустинга, но СЗ-направление 127° — свойство одного правила,
  а не того, что «Кора» выделяет сильнее всего.

Запуск из корня репозитория (после двух прогонов openlin_kora):
    python -m experiments.openlin_kora_check --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")

# ---- пре-регистрация (22.09.2026, до прогона) ------------------------------
KORA_N_TESTS_PRIOR = 225        # 219 + 6 из openlin_line_rank
KORA_N_TESTS = 2                # 1а и 1б
ALPHA = 0.05 / (KORA_N_TESTS_PRIOR + KORA_N_TESTS)
N_SHIFTS = 10_000
#: Достижимость: минимальное p сдвигового нуля должно быть ниже порога, иначе
#: тест непроходим при любом результате.
assert 1.0 / (1 + N_SHIFTS) < ALPHA, "порог недостижим при таком числе сдвигов"
AREA = 0.10
STRIKE_MAX_DIFF = 10.0
SCORE_RHO_MIN = 0.6
#: Минимальный сдвиг, ячеек 100 м: ближе — сдвинутая карта почти совпадает с
#: исходной, и нуль вырождается.
MIN_SHIFT_PX = 50
# ---------------------------------------------------------------------------


def fault_masks(z: dict, shape) -> dict[str, np.ndarray]:
    """Зоны глубинных разломов на сетке площадки: СЗ, СВ и вместе."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    root = Path("data/Gis-integro/shp_dbf")
    tr = Affine(float(z["res_m"]), 0, float(z["x0"]), 0, -float(z["res_m"]),
                float(z["y_top"]))
    out = {}
    for key, name in (("СЗ", "glub_raz_nw"), ("СВ", "glub_r_nw")):
        p = root / f"{name}.shp"
        g = gpd.read_file(p)
        if g.crs is None:
            from experiments.build_dataset_v1 import read_sidecar_proj4

            pj = read_sidecar_proj4(p)
            if pj is None:
                raise SystemExit(f"у {p} нет системы координат")
            g = g.set_crs(pj, allow_override=True)
        g = g.to_crs(epsg=int(z["epsg"]))
        out[key] = rasterize(((s, 1) for s in g.geometry if s is not None),
                             out_shape=shape, transform=tr, fill=0,
                             dtype="uint8").astype(bool)
    out["все"] = out["СЗ"] | out["СВ"]
    return out


def shift_p(sel: np.ndarray, target: np.ndarray, valid: np.ndarray,
            n: int, rng) -> tuple[float, float, float]:
    """Отношение доли цели в выборке к доле по листу и его сдвиговый p."""
    base = target[valid].mean()

    def ratio(m):
        m = m & valid
        return target[m].mean() / base if m.any() else 0.0

    obs = ratio(sel)
    h, w = sel.shape
    null = np.empty(n)
    for i in range(n):
        dy = int(rng.integers(MIN_SHIFT_PX, h - MIN_SHIFT_PX))
        dx = int(rng.integers(MIN_SHIFT_PX, w - MIN_SHIFT_PX))
        null[i] = ratio(np.roll(np.roll(sel, dy, 0), dx, 1))
    p = (1 + int((null >= obs).sum())) / (1 + n)
    return float(obs), float(null.mean()), float(p)


def strike(f: dict, m: np.ndarray, s: str, c: str) -> float:
    """Преобладающее простирание по признакам направления, градусы от севера.

    Признаки — косинус и синус УДВОЕННОГО угла градиента в осях растра
    (x — восток, y — вниз). Простирание перпендикулярно градиенту.
    """
    th2 = np.arctan2(np.mean(f[s][m]), np.mean(f[c][m]))
    grad_az = (90.0 + np.degrees(th2 / 2.0)) % 180.0
    return (grad_az + 90.0) % 180.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import pandas as pd
    from scipy.stats import spearmanr

    from experiments.openlin_kora import features20
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(a.site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    sA = np.load(M / f"openlin_kora_{a.site}.npz")["score"]
    sB = np.load(M / f"openlin_kora_{a.site}_повтор.npz")["score"]
    valid &= np.isfinite(sA) & np.isfinite(sB)

    names20 = pd.read_csv(M / f"openlin_feat20_{a.site}.csv")["признак"].tolist()
    F = features20(x, use, valid, res_m, names20, valid)
    f = {n: F[i] for i, n in enumerate(names20)}
    q = {n: np.quantile(f[n][valid], 1 / 3) for n in names20}
    zone_nw = valid & (f["coh_summer_vv_COH12_sin1000"] < q["coh_summer_vv_COH12_sin1000"]) \
        & (f["vh_sin2000"] < q["vh_sin2000"]) & (f["ndvi_sin500"] < q["ndvi_sin500"])

    faults = fault_masks(z, valid.shape)
    for k, m in faults.items():
        log(f"зоны глубинных разломов {k}: {100 * m[valid].mean():.1f} % листа")
    log(f"порог значимости 0.05/{KORA_N_TESTS_PRIOR + KORA_N_TESTS} = {ALPHA:.5f}, "
        f"сдвигов {N_SHIFTS}")

    rng = np.random.default_rng(11)
    topA = valid & (sA >= np.quantile(sA[valid], 1 - AREA))
    r1a = shift_p(topA, faults["все"], valid, N_SHIFTS, rng)
    r1b = shift_p(zone_nw, faults["СЗ"], valid, N_SHIFTS, rng)
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    log("\nПРОВЕРКА 1 — по месту:")
    log(f"  1а) верхние 10 % оценки против всех зон разломов: lift "
        f"{r1a[0]:.2f} (при сдвиге {r1a[1]:.2f}), p = {r1a[2]:.5f} -> "
        f"{mark[r1a[2] < ALPHA]}")
    log(f"  1б) зона согласия на СЗ-направлении против зон СЗ-разломов: "
        f"{r1b[0]:.2f}x (при сдвиге {r1b[1]:.2f}x), p = {r1b[2]:.5f} -> "
        f"{mark[r1b[2] < ALPHA]}")

    topB = valid & (sB >= np.quantile(sB[valid], 1 - AREA))
    stA = strike(f, topA, "coh_summer_vv_COH12_sin1000", "coh_summer_vv_COH12_cos1000")
    stB = strike(f, topB, "coh_summer_vv_COH12_sin1000", "coh_summer_vv_COH12_cos1000")
    d = abs(((stA - stB) + 90.0) % 180.0 - 90.0)
    idx = np.flatnonzero(valid.ravel())
    pick = np.random.default_rng(3).choice(idx, min(50_000, len(idx)), replace=False)
    rho = spearmanr(sA.ravel()[pick], sB.ravel()[pick]).statistic
    okS = d <= STRIKE_MAX_DIFF and rho >= SCORE_RHO_MIN
    log("\nПРОВЕРКА 2 — устойчивость (полосы поменяны местами, новые подделки):")
    log(f"  AUC на проверке: исходный {float(np.load(M / f'openlin_kora_{a.site}.npz')['auc']) if 'auc' in np.load(M / f'openlin_kora_{a.site}.npz') else float('nan'):.3f}, "
        f"повтор {float(np.load(M / f'openlin_kora_{a.site}_повтор.npz')['auc']):.3f}")
    log(f"  простирание в верхних 10 %: {stA:.0f}° и {stB:.0f}°, разница "
        f"{d:.0f}° (порог {STRIKE_MAX_DIFF:.0f}°)")
    log(f"  корреляция карт оценки: {rho:.2f} (порог {SCORE_RHO_MIN})  -> "
        f"{mark[okS]}")

    pd.DataFrame([{"тест": "1а", "значение": r1a[0], "нуль": r1a[1],
                   "p": r1a[2], "порог": ALPHA, "пройдено": r1a[2] < ALPHA},
                  {"тест": "1б", "значение": r1b[0], "нуль": r1b[1],
                   "p": r1b[2], "порог": ALPHA, "пройдено": r1b[2] < ALPHA},
                  {"тест": "2 простирание", "значение": d, "нуль": np.nan,
                   "p": np.nan, "порог": STRIKE_MAX_DIFF, "пройдено": d <= STRIKE_MAX_DIFF},
                  {"тест": "2 корреляция", "значение": rho, "нуль": np.nan,
                   "p": np.nan, "порог": SCORE_RHO_MIN, "пройдено": rho >= SCORE_RHO_MIN}]
                 ).to_csv(M / f"openlin_kora_check_{a.site}.csv", index=False,
                          encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_kora_check_{a.site}.csv'}")


if __name__ == "__main__":
    main()
