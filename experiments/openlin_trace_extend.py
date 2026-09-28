"""Замер: что с откликом ЗА концом линии — можно ли продолжать трассировку.

ОТКУДА ПОСТАНОВКА. Набор линий, подтверждённых двумя съёмками, упирается в
условие длины: медиана 1.36 км при пороге 1.5. Сшивка обрывков своё уже
выбрала (разрыв доведён до 90-го процентиля замера, 1192 м), а замер концов
показал, что дело не в ней и не в карте зон: у края зоны обрывается 39 %
концов, остальные кончаются ВНУТРИ зоны на глубине 169 м (медиана). Значит
линия кончается там, где отклик гребня проседает ниже нижнего порога
гистерезиса, а не там, где кончается структура.

ЧТО МЕРЯЕТСЯ. Профиль нормированного отклика ``z`` вдоль продолжения линии за
её концом: от конца шагами по ячейке в СВОЁМ направлении (память направления),
до 3 км. Сравнивается с двумя нулями:
  1. тот же конец, но ход ПОПЕРЁК линии — отвечает на вопрос «а не везде ли тут
     такой отклик»;
  2. случайная точка листа и случайное направление — общий фон.
Профиль сравнивается и с действующими порогами гистерезиса (``Q_LOW`` = 0.90,
``Q_HIGH`` = 0.97 по всему кадру), чтобы видеть, НАСКОЛЬКО ниже порога живёт
продолжение.

ЧТО ИЗ ЭТОГО БЕРЁТСЯ. Порог продолжения и предельная длина продолжения не
выбираются, а считаются: порог — 95-й процентиль случайного нуля (доля ложных
продолжений 5 % на ячейку), предельная длина — расстояние, на котором медиана
настоящего профиля опускается до этого порога. Числа печатаются и идут в
пре-регистрацию следующего прогона.

Запуск из корня репозитория:
    python -m experiments.openlin_trace_extend --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, M, OUT, trace, zone_and_strike

STEP_MAX_M = 3000.0          # докуда шагать за конец линии
PERP_TOL = 1                 # ячеек вбок при выборке: берётся максимум из трёх
NULL_Q = 0.95                # квантиль случайного нуля = порог продолжения


def walk(z, valid, ends, dirs, res_m: float, t_max: int) -> np.ndarray:
    """Отклик вдоль хода из ``ends`` в направлениях ``dirs`` (N, T).

    В каждой точке берётся максимум из трёх ячеек: на оси хода и ``PERP_TOL``
    вбок в обе стороны — линия может слегка изгибаться, и жёсткая ось резала
    бы профиль там, где структура просто ушла на ячейку в сторону.
    """
    H, W = z.shape
    t = np.arange(1, t_max + 1, dtype="float32")
    perp = np.stack([-dirs[:, 1], dirs[:, 0]], 1)
    out = np.full((len(ends), t_max), np.nan, dtype="float32")
    for k in (-PERP_TOL, 0, PERP_TOL):
        pos = (ends[:, None, :] + dirs[:, None, :] * t[None, :, None]
               + perp[:, None, :] * k)
        r = np.round(pos[..., 0]).astype(int)
        c = np.round(pos[..., 1]).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c = np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)
        v = np.where(ok & valid[r, c], z[r, c], np.nan)
        out = np.fmax(out, v)
    # после первой же недоступной ячейки ход прекращается
    bad = ~np.isfinite(out)
    stop = np.where(bad.any(1), bad.argmax(1), t_max)
    mask = np.arange(t_max)[None, :] >= stop[:, None]
    out[mask] = np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_iso import Q_HIGH, Q_LOW
    from experiments.openlin_lines import _axis
    from experiments.openlin_shadow import standardize

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    valid = np.isfinite(img)
    x = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    z = local_norm(np.where(valid, ridge_response(x), 0.0), valid, res_m)
    z = np.where(valid, z, np.nan).astype("float32")
    v = z[valid]
    lo, hi = np.quantile(v, Q_LOW), np.quantile(v, Q_HIGH)
    log(f"отклик: нижний порог гистерезиса {lo:.3f} (квантиль {Q_LOW}), "
        f"верхний {hi:.3f} ({Q_HIGH}) ({time.time() - t0:.0f} с)")

    lines = trace(img, valid, res_m, local=True)
    log(f"линий >= 500 м: {len(lines)} ({time.time() - t0:.0f} с)")

    ends, dirs = [], []
    for r in lines:
        p = r["pts"]
        ax, e1, e2 = _axis(p)
        u = np.asarray(ax, dtype="float64")
        u = u / max(np.hypot(*u), 1e-9)
        mid = p.mean(0)
        for e in (e1, e2):
            s = np.sign((np.asarray(e, dtype="float64") - mid) @ u) or 1.0
            ends.append(np.asarray(e, dtype="float64"))
            dirs.append(u * s)
    ends, dirs = np.asarray(ends), np.asarray(dirs)
    t_max = int(STEP_MAX_M / res_m)
    log(f"концов линий: {len(ends)}, ход до {STEP_MAX_M:.0f} м "
        f"({t_max} ячеек)")

    real = walk(z, valid, ends, dirs, res_m, t_max)
    perp = walk(z, valid, ends, np.stack([-dirs[:, 1], dirs[:, 0]], 1),
                res_m, t_max)
    rng = np.random.default_rng(5)
    idx = np.flatnonzero(valid.ravel())
    pick = rng.choice(idx, len(ends), replace=False)
    r0, c0 = np.unravel_index(pick, valid.shape)
    ang = rng.uniform(0, 2 * np.pi, len(ends))
    rnd = walk(z, valid, np.stack([r0, c0], 1).astype("float64"),
               np.stack([np.cos(ang), np.sin(ang)], 1), res_m, t_max)
    log(f"ходы посчитаны ({time.time() - t0:.0f} с)")

    thr = float(np.nanquantile(rnd, NULL_Q))
    log(f"\nпорог продолжения = {NULL_Q:.0%}-й процентиль случайного нуля: "
        f"{thr:.3f} (нижний порог гистерезиса {lo:.3f}, "
        f"то есть на {100 * (1 - thr / lo):.0f} % ниже)")

    log("\nмедиана отклика за концом линии, по расстоянию:")
    log("   м     своё направление   поперёк   случайно   доля выше порога")
    for d_m in (100, 200, 400, 600, 900, 1200, 1600, 2000, 3000):
        k = int(d_m / res_m) - 1
        if k >= t_max:
            continue
        mr = np.nanmedian(real[:, k]); mp = np.nanmedian(perp[:, k])
        mn = np.nanmedian(rnd[:, k])
        share = float(np.nanmean(real[:, k] > thr))
        log(f"  {d_m:5d}   {mr:14.3f}   {mp:7.3f}   {mn:8.3f}   {100 * share:10.0f} %")

    med = np.array([np.nanmedian(real[:, k]) for k in range(t_max)])
    above = np.flatnonzero(med > thr)
    l_max = (above[-1] + 1) * res_m if len(above) else 0.0
    # непрерывность: докуда медиана НИ РАЗУ не опускалась ниже порога
    first_bad = np.flatnonzero(med <= thr)
    l_cont = (first_bad[0] * res_m) if len(first_bad) else t_max * res_m
    log(f"\nмедиана выше порога: последний раз на {l_max:.0f} м, "
        f"непрерывно — до {l_cont:.0f} м")

    zone, strike = zone_and_strike(a.site, int(round(100 / res_m)))
    zone = zone[:valid.shape[0], :valid.shape[1]]
    log(f"для сведения: доля листа под зоной согласия {100 * zone.mean():.1f} %")

    np.savez_compressed(M / f"openlin_trace_extend_{a.site}.npz",
                        real=real, perp=perp, rnd=rnd, thr=thr, lo=lo, hi=hi,
                        res_m=res_m)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = (np.arange(t_max) + 1) * res_m
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="#fcfcfb")
    for arr, name, col in ((real, "своё направление", "#b3261e"),
                           (perp, "поперёк линии", "#eda100"),
                           (rnd, "случайно", "#6b6b6b")):
        ax.plot(d, [np.nanmedian(arr[:, k]) for k in range(t_max)],
                color=col, lw=2, label=name)
        q1 = [np.nanquantile(arr[:, k], 0.25) for k in range(t_max)]
        q3 = [np.nanquantile(arr[:, k], 0.75) for k in range(t_max)]
        ax.fill_between(d, q1, q3, color=col, alpha=0.12, lw=0)
    ax.axhline(lo, color="#2a78d6", ls="--", lw=1.4,
               label=f"нижний порог гистерезиса ({lo:.2f})")
    ax.axhline(thr, color="#1d7a3c", ls=":", lw=1.6,
               label=f"порог продолжения = 95 % случайного нуля ({thr:.2f})")
    ax.set_xlabel("расстояние за концом линии, м")
    ax.set_ylabel("отклик гребня, сигм своей окрестности")
    ax.set_title("Есть ли чему продолжаться за концом линии\n"
                 f"{a.site}: {len(ends)} концов, полоса — квартили", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    p = OUT / f"отклик_за_концом_{a.site}.png"
    fig.savefig(p, dpi=110, facecolor=fig.get_facecolor())
    log(f"сохранено: {p} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
