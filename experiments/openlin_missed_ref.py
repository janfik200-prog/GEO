"""Чего мы не находим в эталоне и почему: форма рельефа под линиями GTK.

ЗАЧЕМ (Артур, 05.11.2026). Снимок + рельеф находят 40 % эталона GTK 292 с
учётом направления (:mod:`openlin_ref_match`). Прежде чем добавлять способы
поиска, посмотреть на пропущенные 60 %: что под ними на местности. Цель —
понять, какой объект детектор не видит, а не подогнать тракт под эталон.

ЧТО МЕРИТСЯ. Для каждой ячейки эталона — профиль поперёк линии (по её же
азимуту) длиной ±``HALF_M`` с шагом сетки: высота GLO-30 и зимний снимок
(shade). Ячейки делятся на «найдено» и «пропущено» по направленному
совпадению с набором «снимок + рельеф» (весь лист), то есть ровно тем, что
считает мера. Для сравнения — «нуль»: те же ячейки эталона, сдвинутые по тору
(общий список сдвигов меры), с теми же азимутами — профиль случайного места
под тем же углом.

Из профиля высоты (за вычетом наклона по краям) берутся два числа, м:
  * «впадина» = средняя высота краёв минус центр (> 0 — ложбина, < 0 — гребень);
  * «уступ» = |правый край − левый край|.
Порогов нет: печатаются распределения у найденного, пропущенного и нуля.
``HALF_M`` = 600 м — вдвое больше допуска сверки (300 м): профиль должен
выйти за полосу, в которой эталон знает положение линии. Центр — ±``CORE_M``
= 100 м, края — дальше ``EDGE_M`` = 300 м (тот же допуск).

Только замер на площадке настройки, в счётчик не идёт.
    python -m experiments.openlin_missed_ref --site utsjoki
"""
from __future__ import annotations

import argparse

import numpy as np

from cache_paths import ANABAR_DEM
from experiments.openlin_ref_match import (TOL_M, evaluate, positions,
                                           ref_directions, shifts)

HALF_M = 2 * TOL_M
CORE_M = 100.0
EDGE_M = TOL_M
N_NULL_SHIFTS = 20          # из общего списка сдвигов меры: для профилей нуля хватает
N_SAMPLE = 20000            # ячеек на группу (профили считаются по выборке)


def profiles(img, rc, az, res, rng, n):
    """Профили img поперёк азимута az в ячейках rc (выборка n), (n, 2*h+1)."""
    from scipy.ndimage import map_coordinates

    if len(rc) > n:
        idx = rng.choice(len(rc), n, replace=False)
        rc, az = rc[idx], az[idx]
    h = int(HALF_M / res)
    t = np.arange(-h, h + 1)
    a = np.radians(az + 90.0)                       # поперёк линии
    dr = -np.cos(a)[:, None] * t[None, :]           # ряд вниз = юг
    dc = np.sin(a)[:, None] * t[None, :]
    r = rc[:, 0, None] + dr
    c = rc[:, 1, None] + dc
    return map_coordinates(img, [r.ravel(), c.ravel()], order=1,
                           mode="nearest").reshape(r.shape), t * res


def shape_numbers(p, x):
    """«Впадина» и «уступ» по профилю высоты (наклон по краям вычтен)."""
    left, right = p[:, x <= -EDGE_M].mean(1), p[:, x >= EDGE_M].mean(1)
    core = p[:, np.abs(x) <= CORE_M].mean(1)
    trough = (left + right) / 2 - core
    step = np.abs(right - left)
    return trough, step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    a = ap.parse_args()
    if a.site != "utsjoki":
        raise SystemExit("только площадка настройки")

    from experiments.openlin_transfer_check import COMP, M, MAIN_LAYER
    from experiments.openlin_union_recall import add_new

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    o = np.load(M / f"openlin_zone_lines_{a.site}_v3_настройка1.npz")["C_all"]
    r = np.load(M / f"openlin_zone_lines_{a.site}_v3_рельеф1.npz")["C_all"]
    res = float(w["res_m"])
    valid = np.isfinite(w["shade"][:o.shape[0], :o.shape[1]])
    ang, fid = ref_directions(MAIN_LAYER, w, o.shape)
    ang = np.where(valid, ang, np.nan)
    sh = shifts(o.shape)[:N_NULL_SHIFTS]
    pos = positions(ang, fid, valid, sh)
    union = o | add_new(r, o, res)
    _lens, mins = evaluate(union, pos, res)
    big = np.iinfo(np.int64).max
    found = mins[0] < big
    rc0, az0 = pos[0][0], pos[0][1]
    print(f"эталон: ячеек {len(rc0)}, найдено (снимок + рельеф, направленно) "
          f"{100 * found.mean():.1f} %")

    dem = np.load(ANABAR_DEM / f"dem_glo30_{a.site}_лист_20m.npy")
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    shade = np.where(valid, w["shade"][:o.shape[0], :o.shape[1]], 0.0)
    rng = np.random.default_rng(5)
    groups = {"найдено": (rc0[found], az0[found]),
              "пропущено": (rc0[~found], az0[~found]),
              "нуль (случайное место)": (np.concatenate([p[0] for p in pos[1:]]),
                                         np.concatenate([p[1] for p in pos[1:]]))}
    print(f"\nпрофиль поперёк линии ±{HALF_M:.0f} м; центр ±{CORE_M:.0f} м, "
          f"края дальше {EDGE_M:.0f} м; высота GLO-30, м")
    print(f"  {'группа':24s} {'впадина: 10/50/90 %':>24s}  {'доля ложбин >0':>15s}  "
          f"{'уступ: 50/90 %':>16s}  {'контраст снимка (центр−края)':>28s}")
    out = {}
    for g, (rc, az) in groups.items():
        p, x = profiles(dem, rc, az, res, rng, N_SAMPLE)
        tr, st = shape_numbers(p, x)
        ps, _ = profiles(shade, rc, az, res, rng, N_SAMPLE)
        sc = ps[:, np.abs(x) <= CORE_M].mean(1) - (ps[:, x <= -EDGE_M].mean(1)
                                                   + ps[:, x >= EDGE_M].mean(1)) / 2
        q = lambda v, qs: " / ".join(f"{u:+.1f}" for u in np.percentile(v, qs))
        print(f"  {g:24s} {q(tr, [10, 50, 90]):>24s}  {100 * (tr > 0).mean():14.0f} %  "
              f"{q(st, [50, 90]):>16s}  {np.median(np.abs(sc)):28.3f}")
        out[g] = dict(trough=tr, step=st, prof=p.mean(0), x=x)
    np.savez_compressed(M / f"openlin_missed_ref_{a.site}.npz",
                        **{f"{k}_{g}": v[k] for g, v in out.items() for k in ("trough", "step", "prof")},
                        x=out["найдено"]["x"])

    # по линиям эталона: доля найденного у каждой длинной линии и её средняя «впадина»
    f0 = pos[0][2]
    per = []
    for f in np.unique(f0):
        m = f0 == f
        if m.sum() * res / 1000 < 10:
            continue
        p, x = profiles(dem, rc0[m], az0[m], res, rng, 2000)
        tr, st = shape_numbers(p, x)
        per.append((int(f), m.sum() * res / 1000, found[m].mean(), np.median(tr), np.median(st)))
    per.sort(key=lambda t: t[2])
    print("\nдлинные линии эталона (>= 10 км), от пропущенных к найденным:")
    print("  линия  длина_км  найдено  впадина_медиана_м  уступ_медиана_м")
    for f, L, fr, tr, st in per:
        print(f"  {f:5d}  {L:8.1f}  {100 * fr:6.0f} %  {tr:17.1f}  {st:15.1f}")
    np.save(M / f"openlin_missed_ref_lines_{a.site}.npy", np.array(per))
    plot_windows(w, dem, union, ang, fid, found, rc0, per, res, a.site)


def plot_windows(w, dem, union, ang, fid, found, rc0, per, res, site, n=6, win_km=4.0):
    """Окна по самым пропущенным длинным линиям: снимок, отмывка, наши линии, эталон."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_relief_probe import hillshade
    from experiments.openlin_shade import _stretch

    hs = hillshade(dem, res, 135.0)
    half = int(win_km * 1000 / res / 2)
    picks = [f for f, _L, fr, _t, _s in per if fr < 0.2][:n]
    fig, axes = plt.subplots(len(picks), 3, figsize=(15, 5 * len(picks)), facecolor="#fcfcfb")
    axes = np.atleast_2d(axes)
    H, W = dem.shape
    for row, f in zip(axes, picks):
        cells = np.argwhere(fid == f)
        rc = cells[len(cells) // 2]
        r0, r1 = max(rc[0] - half, 0), min(rc[0] + half, H)
        c0, c1 = max(rc[1] - half, 0), min(rc[1] + half, W)
        sl = (slice(r0, r1), slice(c0, c1))
        ref = (fid[sl] == f)
        for k, (img, ttl) in enumerate(((w["shade"][sl], "зимний снимок"),
                                        (hs[sl], "отмывка GLO-30 (свет 135°)"),
                                        (hs[sl], "наши линии (красн.) и эталон (синий)"))):
            ax = row[k]
            ax.imshow(_stretch(np.nan_to_num(img)), cmap="gray", vmin=0, vmax=1)
            if k == 2:
                ov = np.zeros(ref.shape + (4,), "float32")
                ov[union[sl]] = (0.85, 0.15, 0.1, 0.9)
                ov[ref] = (0.1, 0.45, 0.9, 0.9)
                ax.imshow(ov)
            ax.set_title(f"линия {f}: {ttl}", fontsize=10, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = f"outputs/steps/пропущенный_эталон_{site}.png"
    fig.savefig(p, dpi=90)
    print(f"сохранено: {p}")


if __name__ == "__main__":
    main()
