"""Линеамент как ЦЕПОЧКА выстроенных элементов, а не сплошная линия.

ОТКУДА ПОСТАНОВКА. В классическом определении (O'Leary и др., 1976) линеамент —
«составной линейный элемент, ЧАСТИ КОТОРОГО ВЫСТРОЕНЫ в прямую или слабо
изогнутую линию». То есть непрерывного контраста вдоль него может не быть
вовсе: есть спрямлённый участок одной долины, уступ, изгиб другой долины — и
все они лежат на одной прямой. Мы же до сих пор искали сплошную линию и
естественным образом её не находили.

ЧТО ЗДЕСЬ. Голосуют не пиксели, а уже найденные короткие элементы (ветки
трассировки, их тысячи). Каждый элемент голосует за прямые, которые проходят
рядом с ним И идут примерно в его собственном направлении. Второе условие —
главное отличие от обычного преобразования Хафа: случайные точки легко
выстраиваются в прямую, а случайно ориентированные отрезки — нет.

Цепочка засчитывается, если элементы не сгрудились в одном месте, а растянуты
вдоль прямой: считается протяжённость и заполненность.

Значимость меряется фазовой подделкой: там элементов столько же, но их
направления не связаны с местностью, и выстроенных цепочек должно быть заметно
меньше.

Запуск из корня репозитория:
    python -m experiments.openlin_align --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_lines import _axis, _branches

M = Path("outputs/metrics")

#: Шаг перебора направлений цепочки, градусы.
ANGLE_STEP = 3.0

#: Насколько направление элемента может отличаться от направления цепочки.
#: Это и есть условие «выстроенности»: элемент голосует не за любую прямую,
#: проходящую рядом, а только за ту, вдоль которой он сам вытянут.
ALIGN_TOL_DEG = 25.0

#: Ширина коридора цепочки, м: насколько элемент может отстоять от прямой.
CORRIDOR_M = 700.0

#: Требования к цепочке: сколько элементов, какой протяжённости и насколько
#: равномерно они её заполняют.
MIN_ELEMS = 5
MIN_SPAN_M = 10000.0
MIN_FILL = 0.25

#: На сколько сигм число элементов в коридоре должно превышать случайное
#: ожидание. Это и есть главный порог: без него длинная полоса через весь лист
#: всегда набирает «цепочку» просто по плотности элементов.
MIN_Z = 4.0


def elements(mask: np.ndarray, res_m: float) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
    """Центры, азимуты и длины коротких элементов."""
    cx, cy, az, ln = [], [], [], []
    for pts in _branches(mask, res_m):
        if len(pts) < 3:
            continue
        ax, _a, _b = _axis(pts)
        proj = (pts - pts.mean(0)) @ ax
        cy.append(float(pts[:, 0].mean()))
        cx.append(float(pts[:, 1].mean()))
        az.append(float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180))
        ln.append(float(proj.max() - proj.min()) * res_m)
    return (np.array([cx, cy]).T, np.asarray(az), np.asarray(ln))


def chains(cent: np.ndarray, az: np.ndarray, ln: np.ndarray, res_m: float,
           area_km2: float, angle_step: float = ANGLE_STEP,
           tol: float = ALIGN_TOL_DEG, corridor_m: float = CORRIDOR_M,
           min_elems: int = MIN_ELEMS, min_span_m: float = MIN_SPAN_M,
           min_fill: float = MIN_FILL, min_z: float = MIN_Z) -> list[dict]:
    """Цепочки выстроенных элементов: голосование с учётом ориентации."""
    density_km2 = len(cent) / max(area_km2, 1.0)
    out = []
    used = np.zeros(len(cent), dtype=bool)
    corridor_px = corridor_m / res_m
    for a in np.arange(0.0, 180.0, angle_step):
        th = np.radians(a)
        # направление цепочки и нормаль к нему
        u = np.array([np.sin(th), -np.cos(th)])       # вдоль (x, y)
        n = np.array([np.cos(th), np.sin(th)])
        d = np.abs(az - a)
        fit = np.minimum(d, 180.0 - d) <= tol          # элемент смотрит туда же
        if fit.sum() < min_elems:
            continue
        idx = np.flatnonzero(fit)
        rho = cent[idx] @ n
        s = cent[idx] @ u
        # корзины по удалению от начала координат: шаг = ширина коридора
        bins = np.round(rho / corridor_px).astype(int)
        for b in np.unique(bins):
            sel = idx[bins == b]
            if len(sel) < min_elems:
                continue
            pos = np.sort(cent[sel] @ u) * res_m
            span = float(pos[-1] - pos[0])
            if span < min_span_m:
                continue
            # заполненность: какая доля протяжённости покрыта самими элементами
            fill = float(ln[sel].sum() / span)
            if fill < min_fill:
                continue
            # Сколько элементов оказалось бы в этой полосе само собой. Площадь
            # полосы умножается на плотность элементов, и ещё на долю тех, чьё
            # направление случайно попало в допуск (2*tol из 180 градусов).
            # Без этой поправки длинная полоса через весь лист всегда набирает
            # «цепочку»: на 60 км при нашей плотности случайно ожидается 14
            # элементов — ровно столько, сколько мы радостно засчитывали.
            area_km2 = (span / 1000.0) * (2 * corridor_m / 1000.0)
            n_exp = density_km2 * area_km2 * (2 * tol / 180.0)
            z = (len(sel) - n_exp) / np.sqrt(max(n_exp, 1.0))
            if z < min_z:
                continue
            out.append({"азимут": float(a), "элементов": int(len(sel)),
                        "ожидалось_случайно": float(n_exp),
                        "превышение_сигм": float(z),
                        "протяжённость_км": span / 1000.0,
                        "заполненность": fill,
                        "длина_элементов_км": float(ln[sel].sum()) / 1000.0,
                        "индексы": sel})
            used[sel] = True
    # убираем дубли: одна и та же цепочка найдётся на соседних углах
    out.sort(key=lambda c: -c["элементов"])
    keep, taken = [], np.zeros(len(cent), dtype=bool)
    for c in out:
        new = c["индексы"][~taken[c["индексы"]]]
        if len(new) < min_elems:
            continue
        taken[c["индексы"]] = True
        keep.append(c)
    return keep


def report(name: str, mask: np.ndarray, res_m: float, area_km2: float,
           log) -> list[dict]:
    cent, az, ln = elements(mask, res_m)
    t0 = time.time()
    ch = chains(cent, az, ln, res_m, area_km2)
    log(f"  {name}: элементов {len(cent)}, цепочек {len(ch)} "
        f"({time.time() - t0:.0f} с)")
    if ch:
        sp = np.array([c["протяжённость_км"] for c in ch])
        ne = np.array([c["элементов"] for c in ch])
        log(f"    протяжённость: медиана {np.median(sp):.0f} км, "
            f"максимум {sp.max():.0f} км; элементов в цепочке: медиана "
            f"{np.median(ne):.0f}")
    return ch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--operator", default="hessian")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_null_input import _mask
    from experiments.openlin_unet import load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(a.site)
    res_m = float(z["res_m"])
    area_km2 = float(np.isfinite(z["gray"]).sum()) * (res_m / 1000.0) ** 2
    log(f"площадка {a.site}: цепочки выстроенных элементов "
        f"(коридор {CORRIDOR_M:.0f} м, согласие направления "
        f"{ALIGN_TOL_DEG:.0f}°, от {MIN_ELEMS} элементов и {MIN_SPAN_M / 1000:.0f} км)")

    res = {}
    for fake in (False, True):
        tag = "подделка" if fake else "настоящий вход"
        m = _mask(a.site, OPTIC + RADAR, a.density, a.operator, fake,
                  np.random.default_rng(7 if not fake else 100))
        res[fake] = report(tag, m, res_m, area_km2, log)
        if not fake:
            np.save(M / f"_align_mask_{a.site}.npy", m)

    n_real, n_fake = len(res[False]), len(res[True])
    log(f"\nцепочек на настоящем входе {n_real}, на подделке {n_fake}, "
        f"мусора {100 * n_fake / max(n_real, 1):.0f} %")

    if res[False]:
        import pandas as pd
        df = pd.DataFrame([{k: v for k, v in c.items() if k != "индексы"}
                           for c in res[False]])
        df.to_csv(M / f"openlin_align_{a.site}.csv", index=False,
                  encoding="utf-8-sig")
        np.savez_compressed(M / f"openlin_align_{a.site}.npz",
                            **{f"c{i}": c["индексы"]
                               for i, c in enumerate(res[False])})
        log(f"сохранено: {M / f'openlin_align_{a.site}.csv'}")


if __name__ == "__main__":
    main()
