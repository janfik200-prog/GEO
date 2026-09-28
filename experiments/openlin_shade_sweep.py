"""Подбор геометрии оператора уступов по измеримой цели, а не на глаз.

ЗАЧЕМ. Первая версия детектора (:mod:`openlin_shade`) прошла фазовый нуль, но
дала смешной урожай: 13 км линий на 6600 км² листа. Значения длины накопления,
ширины полосы и сглаживания я брал из общих соображений и мог ошибиться на
порядок — структуры на местности могут быть и в десять раз длиннее, и в пять
раз шире того, подо что настроено окно.

ЦЕЛЬ ПЕРЕБОРА. Доля площади, выживающая при ФИКСИРОВАННОЙ доле ложных. Порог
для каждой конфигурации берётся свой — тот, при котором подделка даёт ровно
``JUNK`` процентов от числа оставленных ячеек. Сравнивать конфигурации по
сырому числу ячеек нельзя: более шумный оператор всегда оставит больше.

ЧЕСТНОСТЬ ПОДБОРА. Настройка идёт на одном окне, а победитель проверяется на
другом, не участвовавшем в переборе. Без этого выбранные параметры описывали бы
конкретный кусок местности, а не свойство метода.

Запуск из корня репозитория:
    python -m experiments.openlin_shade_sweep --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_shade import (ACROSS_M, ALONG_M, OFFSET_M, SMOOTH_M,
                                       STEP_DEG, load_winter, strike_scan,
                                       sun_derivative)
from experiments.openlin_null_input import phase_randomize

M = Path("outputs/metrics")

#: Доля ложных, при которой сравниваются конфигурации, %.
JUNK = 10.0

#: Окна подбора и проверки: два непересекающихся квадрата листа.
WIN_TUNE = (900, 1100, 1000)
WIN_TEST = (2300, 2400, 1000)

#: Что перебирается. По одному параметру за раз вокруг текущей точки: полный
#: перебор впятером дал бы сотни конфигураций и часы счёта ради того же ответа.
GRID = {
    "along_m": (1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
    "across_m": (60.0, 100.0, 200.0, 400.0),
    "offset_m": (200.0, 400.0, 800.0, 1600.0),
    "smooth_m": (40.0, 60.0, 120.0, 240.0),
    "channel": ("shade", "B04", "ndvi"),
}
BASE = {"along_m": ALONG_M, "across_m": ACROSS_M, "offset_m": OFFSET_M,
        "smooth_m": SMOOTH_M, "channel": "shade"}


def window(z: dict, win: tuple, channel: str) -> np.ndarray:
    r, c, n = win
    return z[channel][r:r + n, c:c + n].astype("float32")


def yield_at_junk(real: np.ndarray, fake: np.ndarray, junk: float = JUNK
                  ) -> tuple[float, float]:
    """Доля ячеек, выживающая при заданной доле ложных, и сам порог."""
    a = real[np.isfinite(real)]
    b = fake[np.isfinite(fake)]
    if a.size == 0 or b.size == 0:
        return 0.0, float("nan")
    scale = a.size / b.size
    best = (0.0, float("nan"))
    for t in np.quantile(a, np.linspace(0.90, 0.99999, 60)):
        n = int((a >= t).sum())
        if n == 0:
            continue
        j = 100.0 * float((b >= t).sum()) * scale / n
        if j <= junk and n / a.size > best[0]:
            best = (n / a.size, float(t))
    return best


def one_config(z: dict, win: tuple, res_m: float, az: float, cfg: dict,
               log) -> tuple[float, float]:
    """Урожай одной конфигурации на одном окне."""
    img = window(z, win, cfg["channel"])
    valid = np.isfinite(img)
    rng = np.random.default_rng(100)
    fake_img = np.where(valid, phase_randomize(img, valid, rng), np.nan)

    out = []
    for src in (img, fake_img.astype("float32")):
        d = sun_derivative(src, valid, res_m, az, cfg["smooth_m"])
        r = strike_scan(d, valid, res_m, cfg["along_m"], cfg["across_m"],
                        STEP_DEG, cfg["smooth_m"], offset_m=cfg["offset_m"])
        out.append(r["оценка"])
    return yield_at_junk(out[0], out[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--res", type=float, default=20.0)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    z = load_winter(a.site, a.res)
    az = float(np.median(z["azimuth"]))
    log(f"площадка {a.site}, окно подбора {WIN_TUNE[2] * a.res / 1000:.0f} км, "
        f"цель — доля площади при {JUNK:.0f} % ложных")
    log(f"исходная точка: {BASE}")

    rows = []
    t_all = time.time()
    for key, values in GRID.items():
        log(f"\n{key}:")
        for v in values:
            cfg = dict(BASE)
            cfg[key] = v
            t0 = time.time()
            frac, thr = one_config(z, WIN_TUNE, a.res, az, cfg, log)
            rows.append({"параметр": key, "значение": v,
                         "доля площади, %": 100 * frac, "порог": thr,
                         "секунд": time.time() - t0})
            log(f"  {str(v):>8}: выживает {100 * frac:6.3f} % площади "
                f"(порог {thr:.2f}, {time.time() - t0:.0f} с)")
    log(f"\nвесь перебор: {(time.time() - t_all) / 60:.0f} мин")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(M / f"openlin_shade_sweep_{a.site}.csv", index=False,
              encoding="utf-8-sig")
    log(f"сохранено: {M / f'openlin_shade_sweep_{a.site}.csv'}")

    best = dict(BASE)
    for key in GRID:
        sub = df[df["параметр"] == key]
        v = sub.loc[sub["доля площади, %"].idxmax(), "значение"]
        best[key] = v
    log(f"\nлучший набор по окну подбора: {best}")

    log("\nпроверка на другом окне, не участвовавшем в переборе:")
    for name, cfg in (("исходная точка", BASE), ("лучший набор", best)):
        frac, thr = one_config(z, WIN_TEST, a.res, az, cfg, log)
        log(f"  {name:16s}: выживает {100 * frac:6.3f} % площади "
            f"(порог {thr:.2f})")


if __name__ == "__main__":
    main()
