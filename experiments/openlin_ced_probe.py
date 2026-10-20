"""Проба на одном участке: чистка снимка ВДОЛЬ САМОЙ ЛИНИИ, с её изгибом.

ПОВОД. :mod:`openlin_clean_probe`: усреднение вдоль ПРЯМОГО окна 80 м удваивает
число длинных линий почти без шума, но обведённый в окне 7 уступ не возвращает
— он дуга, и прямое окно её размазывает. Здесь проверяется чистка, которая
поворачивает вместе с рисунком.

КАК ДЕЛАЕТСЯ (направленная диффузия, она же сглаживание, усиливающее
когерентность). В каждой ячейке по окрестности оценивается местное направление
рисунка (структурный тензор: градиент на масштабе ``SIGMA``, усреднение на
масштабе ``RHO``). Яркость «растекается» только вдоль этого направления и почти
не растекается поперёк; направление пересчитывается на каждом шаге, поэтому
сглаживание идёт по дуге. Время диффузии T задаёт длину усреднения вдоль:
примерно sqrt(2T) ячеек в одну сторону, то есть T = 8 соответствует окну 80 м,
T = 32 — 160 м, T = 72 — 240 м. Дальше действующий детектор без изменений.

ОПАСНОСТЬ та же, что у прямого окна: из шума получаются штрихи, только теперь
ещё и плавно изогнутые. Судит двойник на том же участке.

УЧАСТОК — 15 x 15 км вокруг окна 7 (на весь лист итеративный счёт на
процессоре долог; если проба положительна — считать лист). Сравниваются:
действующий детектор, прямое окно 80 м и направленная диффузия трёх
длительностей. Для каждого: место обведённого уступа в отклике участка, куски
линий в обведённом месте, число кусков по длинам на снимке и на двойнике.

Числа ``SIGMA``, ``RHO``, шаг и доля поперечной диффузии — обычные для метода
значения, не подобранные; это проба «работает ли вообще», не выбор параметров.

Это проба: рабочий набор не меняется, в счётчик не идёт.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_ced_probe.log``; подписи на
картинке дают полную ширину окна, вдвое больше: T = 8 там «~160 м» и
сопоставимо с прямым окном «80 м» из :mod:`openlin_clean_probe`).
Куски линий на участке, снимок / двойник:
                        от 1.0 км     от 1.5 км    уступ, процентиль (медиана)
  действующий            18 /  0        1 / 0       81.7
  прямое окно 80 м       36 /  3        7 / 0       80.6
  вдоль линии, T = 8     41 /  1       11 / 0       81.0
  вдоль линии, T = 32    61 / 25       18 / 9       80.5
  вдоль линии, T = 72    83 / 60       28 / 15      79.9
  1. ДЛЯ ОБВЕДЁННОГО УСТУПА — ОТРИЦАТЕЛЬНО. Его место в отклике не сдвинулось
     ни при какой длительности (80-82-й процентиль). Изгиб, значит, был не
     главной причиной: на очищенном снимке видно, что уступ — мягкая граница
     широкого тёмного участка, перепад тона шириной в сотню-другую метров, а
     не узкая тень. Детектор узких полос ставит такому объекту низкую оценку
     при любой чистке.
  2. Короткая диффузия (T = 8) на этом участке чуть лучше прямого окна (41 / 1
     против 36 / 3 от 1 км), но числа малы — один участок 15 x 15 км, и счёт
     в десятки раз дороже. Длинная диффузия рисует из шума плавные линии:
     при T = 72 шума уже 72 %.
  ВЫВОД: направленная диффузия на весь лист не считается; для прибавки
  длинных линий достаточно прямого окна 80 м.

Запуск из корня репозитория:
    python -m experiments.openlin_ced_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import COMP, OUT

CENTER = (813, 3534)                 # окно 7
HALF_CELLS = 375                     # участок 15 x 15 км
SIGMA = 1.0                          # масштаб градиента, ячеек
RHO = 4.0                            # масштаб оценки направления, ячеек (80 м)
TAU = 0.15                           # шаг явной схемы (устойчиво при <= 0.25)
ACROSS = 0.001                       # доля диффузии поперёк линии
TIMES = (8.0, 32.0, 72.0)            # время диффузии: окно вдоль ~ 80, 160, 240 м
LENGTHS_KM = (0.5, 1.0, 1.5, 2.0)
MARK = (695, 762, 3508, 3575)        # обведённое место: строки, столбцы листа
MARK_MIN_CELLS = 10
OUT_DIR = OUT.parent / "линеаменты_очистка"


def directed_diffusion(u0: np.ndarray, t_end: float, snapshots=()) -> dict:
    """Диффузия вдоль местного направления рисунка; снимки состояния по времени."""
    from scipy.ndimage import gaussian_filter

    u = u0.astype("float64").copy()
    out, t = {}, 0.0
    marks = sorted(set(snapshots) | {t_end})
    while t < t_end - 1e-9:
        us = gaussian_filter(u, SIGMA)
        gy, gx = np.gradient(us)
        j11 = gaussian_filter(gx * gx, RHO)
        j12 = gaussian_filter(gx * gy, RHO)
        j22 = gaussian_filter(gy * gy, RHO)
        th = 0.5 * np.arctan2(2 * j12, j11 - j22)      # направление ПОПЕРЁК линии (x, y)
        c, s = np.cos(th), np.sin(th)
        dxx = ACROSS * c * c + s * s                   # вдоль линии — 1, поперёк — ACROSS
        dxy = (ACROSS - 1.0) * s * c
        dyy = ACROSS * s * s + c * c
        uy, ux = np.gradient(u)
        f1 = dxx * ux + dxy * uy
        f2 = dxy * ux + dyy * uy
        u += TAU * (np.gradient(f1, axis=1) + np.gradient(f2, axis=0))
        t += TAU
        for m in marks:
            if m not in out and t >= m - 1e-9:
                out[m] = u.astype("float32").copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from experiments.openlin_clean_probe import oriented_response
    from experiments.openlin_iso import (Q_HIGH, Q_LOW, _describe, _pieces, hysteresis8,
                                         local_norm, ridge_response, to_lines_strict)
    from experiments.openlin_kora import surrogate
    from experiments.openlin_lines import _ridge_nms
    from experiments.openlin_shadow import standardize

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = v3.build_halves(a.site, lambda *_: None)
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])
    stack = np.stack([np.where(valid, x, np.nan) for x in (opt, halves["А"], halves["Б"])])
    twin = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900))[0]
    Hh, Ww = valid.shape
    r0 = min(max(CENTER[0] - HALF_CELLS, 0), Hh - 2 * HALF_CELLS)
    c0 = min(max(CENTER[1] - HALF_CELLS, 0), Ww - 2 * HALF_CELLS)
    sl = (slice(r0, r0 + 2 * HALF_CELLS), slice(c0, c0 + 2 * HALF_CELLS))
    vm = valid[sl]
    log(f"участок: строки {r0}-{r0 + 2 * HALF_CELLS}, столбцы {c0}-{c0 + 2 * HALF_CELLS}, "
        f"годных ячеек {100 * vm.mean():.1f} % ({time.time() - t0:.0f} с)")
    # стандартизация по всему листу, как в тракте; дальше работа в участке
    xr = np.where(valid, standardize(opt, valid), 0.0).astype("float32")[sl]
    xt = np.where(valid, standardize(twin, valid), 0.0).astype("float32")[sl]
    mark = (MARK[0] - r0, MARK[1] - r0, MARK[2] - c0, MARK[3] - c0)

    def detect(x, mode):
        r = oriented_response(x, 4) if mode == "прямое" else ridge_response(x)
        z = local_norm(np.where(vm, r, 0.0), vm, res_m)
        v = z[vm]
        _s, ridge = _ridge_nms(z, vm, 1.0)
        lo, hi = np.quantile(v, Q_LOW), np.quantile(v, Q_HIGH)
        mask = hysteresis8(np.where(ridge & vm, z, float(v.min())), lo, hi)
        _m, rows = to_lines_strict((mask, z >= lo), res_m)
        return z, rows, ridge & vm

    def in_mark(p):
        return int(((p[:, 0] >= mark[0]) & (p[:, 0] < mark[1])
                    & (p[:, 1] >= mark[2]) & (p[:, 1] < mark[3])).sum())

    z0, rows0, ridge0 = detect(xr, None)
    weak = [d for d in (_describe(p, res_m) for p in
                        _pieces(ridge0 & (z0 >= np.quantile(z0[vm], 0.80)), res_m))
            if d is not None]
    arc = np.concatenate([d["pts"] for d in weak
                          if d["длина_км"] < 0.8 and in_mark(d["pts"]) >= MARK_MIN_CELLS])
    log(f"ячеек обведённого уступа: {len(arc)}")

    def arc_rank(z):
        from scipy.ndimage import maximum_filter
        zm = maximum_filter(z, size=3)
        ref = np.sort(zm[vm])
        return np.percentile(100 * np.searchsorted(ref, zm[arc[:, 0], arc[:, 1]]) / len(ref),
                             [25, 50, 75])

    results = {}

    def report(name, img, zr, rr, rt):
        ln = np.array([d["длина_км"] for d in rr]); tl = np.array([d["длина_км"] for d in rt])
        hit = sorted((d["длина_км"] for d in rr if in_mark(d["pts"]) >= MARK_MIN_CELLS),
                     reverse=True)
        log(f"\n=== {name} ({time.time() - t0:.0f} с)")
        log(f"  уступ в отклике участка, процентиль (25/50/75 %): {np.round(arc_rank(zr), 1)}; "
            f"куски линий в обведённом месте, км: {np.round(hit, 2) if hit else 'нет'}")
        log("  куски линий, снимок / двойник (доля шума): " + "; ".join(
            f"от {L} км {int((ln >= L).sum())} / {int((tl >= L).sum())} "
            f"({100 * (tl >= L).sum() / max((ln >= L).sum(), 1):.0f} %)" for L in LENGTHS_KM))
        results[name] = (img, zr, rr)

    _z, rows0t, _r = detect(xt, None)
    report("действующий", xr, z0, rows0, rows0t)
    zs, rs, _r = detect(xr, "прямое"); _z, rst, _r = detect(xt, "прямое")
    report("прямое окно 80 м", xr, zs, rs, rst)
    dr = directed_diffusion(xr, max(TIMES), TIMES)
    dt = directed_diffusion(xt, max(TIMES), TIMES)
    log(f"диффузия посчитана ({time.time() - t0:.0f} с)")
    for T in TIMES:
        zd, rd, _r = detect(dr[T], None); _z, rdt, _r = detect(dt[T], None)
        report(f"вдоль линии, ~{np.sqrt(2 * T) * 2 * res_m:.0f} м", dr[T], zd, rd, rdt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = int(2500 / res_m)
    cr, cc = CENTER[0] - r0, CENTER[1] - c0
    ws = (slice(cr - half, cr + half), slice(cc - half, cc + half))
    names = list(results)
    fig, ax = plt.subplots(3, len(names), figsize=(6.4 * len(names), 19.6),
                           facecolor="#fcfcfb", squeeze=False)
    lo_, hi_ = np.percentile(xr[ws], [2, 98])
    for j, nm in enumerate(names):
        img, z, rows = results[nm]
        ax[0, j].imshow(img[ws], cmap="gray", vmin=lo_, vmax=hi_, interpolation="nearest")
        ax[0, j].set_title(f"{nm}\nснимок после чистки", loc="left", fontsize=13)
        ax[1, j].imshow(np.clip(z[ws], 0, None), cmap="magma", vmin=0,
                        vmax=float(np.quantile(z[vm], 0.99)), interpolation="nearest")
        ax[1, j].set_title("отклик детектора", loc="left", fontsize=13)
        ax[2, j].imshow(xr[ws], cmap="gray", vmin=lo_, vmax=hi_, interpolation="nearest")
        ov = np.zeros((2 * half, 2 * half, 4), "float32")
        for d in rows:
            p = d["pts"]
            k = (p[:, 0] >= ws[0].start) & (p[:, 0] < ws[0].stop) \
                & (p[:, 1] >= ws[1].start) & (p[:, 1] < ws[1].stop)
            ov[p[k, 0] - ws[0].start, p[k, 1] - ws[1].start] = (0.15, 0.75, 0.95, 1.0)
        ax[2, j].imshow(ov)
        ax[2, j].set_title("линии детектора на исходном снимке", loc="left", fontsize=13)
    for axx in ax.ravel():
        axx.set_xticks([]); axx.set_yticks([])
    fig.suptitle("Окно 7 (центр 813, 3534), 5 × 5 км, север сверху — обведённый уступ "
                 "вверху по центру", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "окно_7_вдоль_линии.png", dpi=70, facecolor=fig.get_facecolor())
    log(f"\nкартинка: {OUT_DIR / 'окно_7_вдоль_линии.png'} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
