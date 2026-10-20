"""Проба: очистить снимок от мусора до детектора — усреднением ВДОЛЬ линии.

ПОВОД. Предложение пользователя 09.10.2026: «очистить снимок от мусора, а не
детектор». Три пробы с порогами детектора обведённый в окне 7 уступ не вернули:
он лежит около 90-го процентиля отклика, там, где шума столько же, сколько
сигнала (:mod:`openlin_weak_tract_probe`).

ЧТО ЗНАЧИТ «ОЧИСТИТЬ» ДЛЯ ЛИНИИ. Действующий детектор смотрит на снимок через
круглое окно в 1-2 ячейки: шум снега и съёмки усредняется только по этому
пятну. Но линия тянется на сотни метров, а шум вдоль неё независим. Если
усреднять ВДОЛЬ направления линии на длину L ячеек, шум падает примерно как
корень из L, а линия остаётся: окно 80 м вдоль даёт вдвое меньше шума, 160 м —
почти втрое. Уступу до уверенного уровня не хватает как раз множителя 1.5-2
(отклик 0.8-1.2 при затравке 1.86), а не порядка — этим проба отличается от
закрытой структурной ветки, где не хватало 8-20 раз.

КАК ДЕЛАЕТСЯ. Направление линии заранее неизвестно, поэтому снимок сглаживается
вытянутым окном во всех направлениях по очереди (шаг по углу — под вытянутость
окна), для каждого считается кривизна поперёк, берётся наибольшая по
направлениям. Поперёк окно прежнее (1 и 2 ячейки), вдоль — L. Дальше детектор
без изменений: нормировка по окрестности, утончение, те же два порога.
Считается в частотной области, ядра не строятся.

ОПАСНОСТЬ. Вытянутое окно и из чистого шума делает штрихи. Поэтому судит
двойник: тот же шум без линеаментов проходит ту же очистку и тот же детектор.
Выигрыш есть, только если длинных линий на снимке прибавилось, а на двойнике
— нет.

ЧТО СРАВНИВАЕТСЯ: действующий детектор; очистка с окном вдоль 80, 160 и 320 м;
отдельно — снимок из всех 55 сцен вместо 12 (усреднение по времени, а не вдоль
линии) с действующим детектором. Для каждого: место ячеек обведённого уступа в
отклике листа и число кусков линий по длинам на снимке и на двойнике.

Это проба: рабочий набор не меняется, в счётчик не идёт.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_clean_probe.log``).
Куски линий детектора (до продолжения и сшивки), снимок / двойник:
                     от 0.5 км          от 1.0 км        от 1.5 км     от 2.0 км
  действующий        2681 /  299        621 /    1       164 /   0      74 / 0
  вдоль 80 м         3876 / 1874       1057 /   90       371 /   2     150 / 0
  вдоль 160 м        4755 / 4128       1409 /  351       506 /  29     225 / 1
  вдоль 320 м        4931 / 5639       2070 / 1436       760 / 240     306 / 51
  55 сцен            2746 /  306        589 /    1       177 /   0      74 / 0
  1. ОКНО 80 М ВДОЛЬ ЛИНИИ УДВАИВАЕТ ЧИСЛО ДЛИННЫХ ЛИНИЙ ПОЧТИ БЕЗ ШУМА: от
     1.5 км — 371 против 164 при 2 у двойника; от 1 км — 1057 против 621 при
     90 у двойника (9 %). Цена — короткие линии: от 0.5 км шума становится
     48 % вместо 11 %, то есть нижнюю границу длины придётся поднять.
     Окна 160 и 320 м шум обгоняет: вытянутое окно делает штрихи из шума.
  2. ОБВЕДЁННЫЙ УСТУП НЕ ВЕРНУЛСЯ: его место в отклике не выросло (82-й
     процентиль -> 80 -> 74 -> 66). Он дуга, а прямое окно размазывает дугу
     тем сильнее, чем оно длиннее.
  3. 55 СЦЕН НЕ МЕНЯЮТ НИЧЕГО — ни место уступа (81), ни числа линий. Значит,
     «мусор» на снимке — не шум съёмки, который усредняется по времени, а
     настоящая мелкая фактура поверхности, одинаковая на всех сценах. Чистка
     набором сцен закрыта: чистить нечего.
  Что именно прибавилось при окне 80 м — новые линии или прежние, ставшие
  цельнее, — не разобрано; нужен полный тракт и просмотр.

Запуск из корня репозитория:
    python -m experiments.openlin_clean_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import COMP, OUT

ALONG_CELLS = (4, 8, 16)             # окно вдоль линии, ячеек (80, 160, 320 м)
LENGTHS_KM = (0.5, 1.0, 1.5, 2.0)
MARK = (695, 762, 3508, 3575)        # обведённое место в окне 7: строки, столбцы листа
MARK_MIN_CELLS = 10
WORKERS = 2                          # на машине идёт другой прогон
OUT_DIR = OUT.parent / "линеаменты_очистка"


def oriented_response(x: np.ndarray, along: float, sigmas=(1.0, 2.0)) -> np.ndarray:
    """Кривизна поперёк линии после сглаживания вдоль неё, максимум по углам.

    В частотной области: гауссово окно (поперёк s, вдоль ``along``) и вторая
    производная поперёк — произведение, зависящее только от угла.
    """
    from scipy import fft

    H, W = x.shape
    F = fft.rfft2(x, workers=WORKERS)
    fy = fft.fftfreq(H)[:, None].astype("float32")
    fx = fft.rfftfreq(W)[None, :].astype("float32")
    best = np.zeros((H, W), dtype="float32")
    for s in sigmas:
        la = max(float(along), s)
        n_th = int(np.ceil(np.pi * la / s))           # шаг по углу ~ s / la
        for k in range(n_th):
            th = np.pi * k / n_th
            fa = fy * np.cos(th) + fx * np.sin(th)    # частота вдоль линии
            fn = -fy * np.sin(th) + fx * np.cos(th)   # частота поперёк
            h = (2 * np.pi * fn) ** 2 * np.exp(-2 * np.pi ** 2 * (la ** 2 * fa ** 2
                                                                  + s ** 2 * fn ** 2))
            r = fft.irfft2(F * h, s=(H, W), workers=WORKERS)
            np.maximum(best, (s ** 2) * np.abs(r).astype("float32"), out=best)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from experiments.openlin_iso import (Q_HIGH, Q_LOW, hysteresis8, local_norm,
                                         ridge_response, to_lines_strict)
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
    twin_img = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900))[0]
    log(f"двойник построен ({time.time() - t0:.0f} с)")

    def detect(img, vmask, along):
        """Отклик, маска линий и куски; along=None — действующий детектор."""
        x = np.where(vmask, standardize(img, vmask), 0.0).astype("float32")
        r = ridge_response(x) if along is None else oriented_response(x, along)
        z = local_norm(np.where(vmask, r, 0.0), vmask, res_m)
        v = z[vmask]
        _s, ridge = _ridge_nms(z, vmask, 1.0)
        lo, hi = np.quantile(v, Q_LOW), np.quantile(v, Q_HIGH)
        mask = hysteresis8(np.where(ridge & vmask, z, float(v.min())), lo, hi)
        _m, rows = to_lines_strict((mask, z >= lo), res_m)
        return z, rows

    # ячейки обведённого уступа: его куски на уровне 80 % действующего детектора
    from experiments.openlin_iso import _describe, _pieces
    z0, rows0 = detect(opt, valid, None)
    _s, ridge0 = _ridge_nms(z0, valid, 1.0)
    weak = [d for d in (_describe(p, res_m) for p in
                        _pieces(ridge0 & valid & (z0 >= np.quantile(z0[valid], 0.80)), res_m))
            if d is not None]

    def in_mark(p):
        return int(((p[:, 0] >= MARK[0]) & (p[:, 0] < MARK[1])
                    & (p[:, 1] >= MARK[2]) & (p[:, 1] < MARK[3])).sum())

    arc = np.concatenate([d["pts"] for d in weak
                          if d["длина_км"] < 0.8 and in_mark(d["pts"]) >= MARK_MIN_CELLS])
    log(f"ячеек обведённого уступа: {len(arc)}")

    def arc_rank(z, vmask):
        """Место уступа в отклике листа: лучшая из ячеек в 20 м от каждой его ячейки."""
        from scipy.ndimage import maximum_filter
        zm = maximum_filter(z, size=3)
        ref = np.sort(zm[vmask])
        return np.percentile(100 * np.searchsorted(ref, zm[arc[:, 0], arc[:, 1]]) / len(ref),
                             [25, 50, 75])

    def report(name, zr, rr, rt):
        ln = np.array([d["длина_км"] for d in rr]); tl = np.array([d["длина_км"] for d in rt])
        hit = sorted((d["длина_км"] for d in rr if in_mark(d["pts"]) >= MARK_MIN_CELLS),
                     reverse=True)
        log(f"\n=== {name} ({time.time() - t0:.0f} с)")
        log(f"  уступ в отклике листа, процентиль (25/50/75 %): "
            f"{np.round(arc_rank(zr, valid), 1)}; куски линий в обведённом месте, км: "
            f"{np.round(hit, 2) if hit else 'нет'}")
        log("  куски линий детектора, снимок / двойник (доля шума): " + "; ".join(
            f"от {L} км {int((ln >= L).sum())} / {int((tl >= L).sum())} "
            f"({100 * (tl >= L).sum() / max((ln >= L).sum(), 1):.0f} %)" for L in LENGTHS_KM))

    results = {}
    _zt, rows0t = detect(twin_img, valid, None)
    report("действующий детектор", z0, rows0, rows0t)
    results["действующий"] = (z0, rows0)
    for L in ALONG_CELLS:
        zr, rr = detect(opt, valid, L)
        _z, rt = detect(twin_img, valid, L)
        report(f"очистка вдоль линии, окно {L * res_m:.0f} м", zr, rr, rt)
        results[f"вдоль {L * res_m:.0f} м"] = (zr, rr)

    # усреднение по времени: все 55 сцен вместо 12 (двойник — из этого же снимка)
    p_all = COMP / f"winter_{a.site}_20m_лист_все.npz"
    if p_all.exists():
        full = np.load(p_all)["shade"].astype("float32")[:valid.shape[0], :valid.shape[1]]
        vf = valid & np.isfinite(full)
        zf, rf = detect(np.where(vf, full, 0.0), valid, None)
        tw = surrogate(np.nan_to_num(np.where(vf, full, np.nan))[None], valid,
                       np.random.default_rng(900))[0]
        _z, rft = detect(tw, valid, None)
        report("все 55 сцен вместо 12, действующий детектор", zf, rf, rft)
        results["55 сцен"] = (zf, rf)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_shade import _stretch

    base = _stretch(np.where(valid, opt, np.nan))
    n_cell = int(5000 / res_m); half = n_cell // 2
    r, c = 813, 3534
    sl = (slice(r - half, r + half), slice(c - half, c + half))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results)
    fig, ax = plt.subplots(2, len(names), figsize=(7.2 * len(names), 15),
                           facecolor="#fcfcfb", squeeze=False)
    for j, nm in enumerate(names):
        z, rows = results[nm]
        ax[0, j].imshow(np.clip(z[sl], 0, None), cmap="magma", vmin=0,
                        vmax=float(np.quantile(z[valid], 0.99)), interpolation="nearest")
        ax[0, j].set_title(f"{nm}: отклик", loc="left", fontsize=14)
        ax[1, j].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        for d in rows:
            p = d["pts"]
            k = (p[:, 0] >= sl[0].start) & (p[:, 0] < sl[0].stop) \
                & (p[:, 1] >= sl[1].start) & (p[:, 1] < sl[1].stop)
            ov[p[k, 0] - sl[0].start, p[k, 1] - sl[1].start] = (0.15, 0.75, 0.95, 1.0)
        ax[1, j].imshow(ov)
        ax[1, j].set_title(f"{nm}: линии детектора", loc="left", fontsize=14)
    for axx in ax.ravel():
        axx.set_xticks([]); axx.set_yticks([])
    fig.suptitle("Окно 7 (центр 813, 3534), 5 × 5 км, север сверху — место с обведённым "
                 "уступом (вверху по центру)", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "окно_7_сравнение.png", dpi=75, facecolor=fig.get_facecolor())
    log(f"\nкартинка: {OUT_DIR / 'окно_7_сравнение.png'} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
