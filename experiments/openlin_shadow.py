"""Тень как разность двух подсветок: низкое солнце минус высокое.

ОТКУДА ПОСТАНОВКА. Канал ``shade`` прежнего детектора — это снимок минус
сглаживание на 2 км. Такой канал в принципе не умеет отличить ТЕНЬ от тёмного
пятна, и проверка это поймала: найденные элементы садились на гидросеть втрое
чаще случайного, то есть мы выделяли русла и берега.

Отделить тень от собственной яркости поверхности можно наложением двух
подсветок: тень зависит от высоты солнца, яркость поверхности — нет. Тень от
уступа в 10 м тянется на 71 м при солнце в 8° и всего на 16 м при 32°, а тёмная
порода, вода и гарь темны одинаково при любой высоте. Значит разность двух
композитов — это почти чистая тень.

ЧТО ЗДЕСЬ ВАЖНО СОБЛЮСТИ. Обе подсветки обязаны застать поверхность в ОДНОМ
состоянии. Высокое солнце бывает летом, но летом нет снега, и разность «октябрь
минус июль» была бы разностью «снег минус тайга» — про растительность, а не про
тени. Поэтому высокая подсветка берётся в апреле: солнце уже 26-34°, а снег на
71-й широте лежит до конца мая. Азимут у обеих один и тот же (175-193° во все
месяцы: спутник солнечно-синхронный и снимает это место всегда в одно местное
время), так что тени различаются только длиной, а не направлением.

ЧЕГО ЭТО НЕ ЧИНИТ. Цензуру по простиранию. Азимут подсветки остаётся южным,
поэтому структура, идущая север-юг, тени не отбрасывает ни при какой высоте
солнца. Это лечится только вторым прибором (:mod:`openlin_radar`).

Запуск из корня репозитория:
    python -m experiments.openlin_shadow --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_shade import _stretch

COMP = Path("datacache/s2_lowsun/composite")

#: Канал, по которому считается разность. Красный: снег в нём ярок, тень
#: контрастна, растительность мешает меньше, чем в ближнем ИК.
BAND = "B04"


def standardize(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Робастная нормировка композита: медиана и межквартильный размах."""
    v = a[valid]
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    return ((a - med) / max(float(q3 - q1) / 1.349, 1e-6)).astype("float32")


def shadow_channel(low: np.ndarray, high: np.ndarray, valid: np.ndarray,
                   log) -> np.ndarray:
    """Разность подсветок с выровненным усилением.

    Просто вычитать нельзя: у композитов своя шкала. Масштаб и сдвиг
    подбираются по наименьшим квадратам так, чтобы высокая подсветка как можно
    полнее объясняла низкую; остаток и есть то, чего при высоком солнце нет.
    """
    lo = standardize(low, valid)
    hi = standardize(high, valid)
    x, y = hi[valid], lo[valid]
    k = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    log(f"  связь подсветок: наклон {k[0]:.3f}, сдвиг {k[1]:+.3f}, "
        f"корреляция {r:.3f}")
    log(f"  высокое солнце объясняет {100 * r ** 2:.0f} % картины низкого; "
        f"остаток — то, чего при высоком солнце нет")
    res = lo - (k[0] * hi + k[1])
    return np.where(valid, res, np.nan).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--res", type=float, default=20.0)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    base = f"winter_{a.site}_{int(a.res)}m_лист"
    zl = dict(np.load(COMP / f"{base}.npz"))
    zh = dict(np.load(COMP / f"{base}_высокое.npz"))
    low, high = zl[BAND].astype("float32"), zh[BAND].astype("float32")
    valid = np.isfinite(low) & np.isfinite(high)
    log(f"площадка {a.site}: низкая подсветка солнце "
        f"{np.median(zl['sun']):.1f}°, высокая {np.median(zh['sun']):.1f}°, "
        f"азимуты {np.median(zl['azimuth']):.0f}° и "
        f"{np.median(zh['azimuth']):.0f}°")
    log(f"  общих ячеек {100 * valid.mean():.1f} %")
    log(f"  тень от уступа 10 м: "
        f"{10 / np.tan(np.radians(float(np.median(zl['sun'])))):.0f} м против "
        f"{10 / np.tan(np.radians(float(np.median(zh['sun'])))):.0f} м")

    sh = shadow_channel(low, high, valid, log)
    p = COMP / f"{base}_тень.npz"
    np.savez_compressed(p, shadow=sh, low=standardize(low, valid),
                        high=standardize(high, valid),
                        res_m=a.res, x0=zl["x0"], y_top=zl["y_top"],
                        epsg=zl["epsg"], azimuth=zl["azimuth"],
                        sun=zl["sun"])
    log(f"сохранено: {p}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(low, f"низкое солнце {np.median(zl['sun']):.0f}°"),
              (high, f"высокое солнце {np.median(zh['sun']):.0f}°"),
              (sh, "разность: тень")]
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.6))
    for a_, (img, ttl) in zip(ax, panels):
        a_.imshow(_stretch(np.where(valid, img, np.nan)), cmap="gray",
                  vmin=0, vmax=1)
        a_.set_title(ttl, fontsize=14)
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    out = Path("outputs/steps") / f"тень_разность_{a.site}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    log(f"сохранено: {out}")


if __name__ == "__main__":
    main()
