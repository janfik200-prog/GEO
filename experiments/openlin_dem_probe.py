"""Проба: что подавать детектору из рельефа и подтверждает ли вторая модель.

ЗАЧЕМ. Перед прогоном метода по рельефу (:mod:`openlin_dem_lines`) нужно
выбрать вход детектора — замером, а не на глаз. Кандидаты:
  * «рельеф» — модель минус плавный фон (тот же ``detrend``, что у снимка):
    детектор по гессиану видит и гребни, и ложбины, направление света не
    нужно, слепых секторов нет;
  * «уклон» — крутизна склона: уступ даёт одну линию, а не две (бровка и
    подножие);
  * «отмывки» 45° и 135° — прежний слой (:mod:`openlin_relief_layer`), для
    сравнения.
Для каждого входа: линии детектора (без продления и сшивки) на GLO-30 и на
ALOS AW3D30; линия GLO подтверждена, если не меньше CONF_FRAC её длины лежит
в CONF_TOL_M от линий ALOS (допуски рабочего метода). Двойник — общая
многоканальная подделка обоих входов (сохраняет связь моделей, 0.87 на
масштабе 100 м), обрабатывается так же. Для сведения — доля подтверждённых
линий, идущих вдоль линий оптики (весь набор A_all рабочего метода).

Это замер, а не прогон: условий приёмки нет, в счётчик не идёт. Выбор входа
по его итогам объявляется в :mod:`openlin_dem_lines` до прогона.

Запуск из корня репозитория:
    python -m experiments.openlin_dem_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, M, mask_of, trace
from experiments.openlin_zone_lines_v3 import CONF_FRAC, CONF_TOL_M, work_npz


def inputs(dem: np.ndarray, valid: np.ndarray, res_m: float) -> dict:
    from experiments.openlin_relief_probe import hillshade
    from experiments.openlin_winter import detrend

    f = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    gy, gx = np.gradient(f, res_m)
    return {"рельеф": detrend(f, valid, res_m),
            "уклон": detrend(np.degrees(np.arctan(np.hypot(gx, gy))), valid, res_m),
            "отмывка 45": detrend(hillshade(f, res_m, 45.0), valid, res_m),
            "отмывка 135": detrend(hillshade(f, res_m, 135.0), valid, res_m)}


def confirmed(La: list, Lb: list, shape, res_m) -> list:
    from scipy.ndimage import distance_transform_edt

    d = distance_transform_edt(~mask_of(Lb, shape), sampling=(res_m, res_m))
    return [x for x in La
            if (d[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_dem_lines import load_alos
    from experiments.openlin_kora import surrogate
    from experiments.openlin_relief_probe import load_dem

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    g, al = load_dem(a.site, res_m), load_alos(a.site, res_m)
    valid = np.isfinite(g) & np.isfinite(al) & np.isfinite(w["shade"])
    I_g, I_a = inputs(g, valid, res_m), inputs(al, valid, res_m)
    work = np.load(work_npz(a.site))
    d_opt = distance_transform_edt(~work["A_all"], sampling=(res_m, res_m))

    def along_optics(L):
        return sum(1 for x in L
                   if (d_opt[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean() >= CONF_FRAC)

    def run(xg, xa):
        Lg = trace(np.where(valid, xg, np.nan), valid, res_m, local=True)
        La = trace(np.where(valid, xa, np.nan), valid, res_m, local=True)
        return Lg, La, confirmed(Lg, La, valid.shape, res_m)

    km = lambda L: sum(x["длина_км"] for x in L)
    rows = []
    for name in I_g:
        Lg, La, C = run(I_g[name], I_a[name])
        st = np.stack([I_g[name], I_a[name]]).astype("float32")
        tw = surrogate(np.nan_to_num(st), valid, np.random.default_rng(900))
        Tg, Ta, TC = run(tw[0], tw[1])
        opt = along_optics(C)
        rows.append((name, len(Lg), len(La), len(C), km(C), len(TC), opt))
        log(f"{name}: линий GLO {len(Lg)}, ALOS {len(La)}; подтверждено {len(C)} "
            f"({km(C):.0f} км, {100 * len(C) / max(len(Lg), 1):.0f} %); у двойника "
            f"{len(Tg)} / {len(Ta)}, подтверждено {len(TC)} -> мусор "
            f"{100 * len(TC) / max(len(C), 1):.1f} %; вдоль оптики {opt} "
            f"({100 * opt / max(len(C), 1):.0f} %) ({time.time() - t0:.0f} с)")
    log("\nитог (вход | GLO | ALOS | подтверждено | км | двойник | вдоль оптики):")
    for r in rows:
        log(" | ".join(str(v) if not isinstance(v, float) else f"{v:.0f}" for v in r))
    np.save(M / f"openlin_dem_probe_{a.site}.npy", np.array(rows, dtype=object),
            allow_pickle=True)


if __name__ == "__main__":
    main()
