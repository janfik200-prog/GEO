"""Проба: слабые гребни без затравки, подтверждённые непрерывным откликом половин.

ПОВОД. Просмотр 09.10.2026: в окне 7 обведён дугообразный уступ — «детектор
видит, но как линия это не выводится». Разбор (:mod:`openlin_ridge_classes`):
дуга лежит на гребне отклика между 80 и 90 % — ниже нижнего порога
гистерезиса, то есть отброшена порогом, хотя гребень цельный.

ПОЧЕМУ НЕЛЬЗЯ ПРОСТО ПОНИЗИТЬ ПОРОГ. На двойнике слабых гребней столько же,
сколько на снимке; связные сети слабых гребней у шума такие же длинные
(выше 80 % и длиннее 1 км по числу ячеек — 3299 против 3340).

ЧТО ПРОВЕРЯЕТСЯ. Шум в двух половинах съёмки разный, а уступ — один. Поэтому:
  кандидаты — куски гребней полного снимка выше уровня q БЕЗ требования
              затравки (разрезанные в узлах, длина — по прямой);
  подтверждение — НЕПРЕРЫВНОЕ: средний отклик КАЖДОЙ половины вдоль куска
              выше ``V7_BRIDGE_Q``-го процентиля нуля, нуль — те же куски,
              перенесённые по тору. Маски линий половин не строятся: слабый
              гребень и в половинах ниже порога, в маске его нет (известная
              цена нынешнего подтверждения).
Судит двойник: то же на карте-двойнике, отношение подтверждённых
двойник : снимок — мусор; приемлемо, как везде в ветке, не больше ``JUNK_MAX``.
Отдельно — подтверждается ли обведённый уступ и сколько подтверждённого нет в
рабочем наборе.

ПЕРВЫЙ ЗАПУСК (допуск положения 60 м, длины от 0.5 км) уступ НЕ вернул, и
разбор показал две причины: (1) дуга режется на куски 0.21-0.44 км, короче
нижней границы; (2) при допуске 60 м нуль поднимается до 1.94 и обгоняет
сигнал, тогда как половины лежат на одной сетке и при допуске 0 куски уступа
дают 1.3-1.6 при пороге нуля 0.73. Поэтому сетка расширена: допуск 0 и 20 м,
длины от 0.2 км.

Это проба: рабочий набор не меняется, в счётчик не идёт. Уровни, длины и
допуски — сетка для замера, не выбор; числа для прогона берутся из результата.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_weak_confirm.log``).
  1. СВЕРКА ПО НЕПРЕРЫВНОМУ ОТКЛИКУ ПОЛОВИН НИЧЕГО НЕ ОТСЕИВАЕТ: при допуске 0
     её проходят 90 % кусков снимка и 86 % кусков двойника (уровень 80 %, от
     0.5 км: 4058 из 4509 и 714 из 827). Половины — части того же снимка, его
     гребень в них есть по построению. Кандидат «подтверждение по непрерывному
     отклику», записанный в ``openlin_zone_lines_v3`` как следующий, в этом
     виде ЗАКРЫТ.
  2. Различает длина куска ПО ПРЯМОЙ (кусок разрезан в узлах): уровень 80 %,
     от 0.2 км — мусор 67 %, от 0.5 км — 18 %, от 1 км — 0.7 % (946 против 7).
     Связные сети слабых гребней по числу ячеек шум не отличают, прямые
     отрезки — отличают.
  3. Обведённый уступ настоящий (опора его кусков 1.3-1.6 при пороге нуля
     0.93), но рассыпан на куски 0.2-0.45 км и по одному куску не проходит.

Запуск из корня репозитория:
    python -m experiments.openlin_weak_confirm_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import COMP, M, OUT

LEVELS = (0.90, 0.85, 0.80)
LENGTHS_KM = (0.2, 0.5, 1.0)
TOLS = (0, 1)                # допуск положения при сверке с половинами, ячеек
#: обведённое пользователем место в окне 7 (центр 813, 3534): строки и столбцы листа
MARK = (695, 762, 3508, 3575)
MARK_MIN_CELLS = 10          # кусок «задевает» место, если внутри не меньше стольких ячеек
OUT_DIR = OUT.parent / "линеаменты_слабые"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from scipy.ndimage import distance_transform_edt, maximum_filter

    from experiments.openlin_iso import _describe, _pieces, local_norm, ridge_response
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
    Hh, Ww = valid.shape
    q0 = v3.V7_BRIDGE_Q

    def resp(o):
        x = np.where(valid, standardize(o, valid), 0.0).astype("float32")
        return local_norm(np.where(valid, ridge_response(x), 0.0), valid, res_m)

    def prepare(o, ha, hb):
        """Куски гребней по уровням и отклики половин."""
        z = resp(o)
        _s, ridge = _ridge_nms(z, valid, 1.0)
        ridge &= valid
        rows = {}
        for q in LEVELS:
            thr = float(np.quantile(z[valid], q))
            rows[q] = [d for d in (_describe(p, res_m) for p in
                                   _pieces(ridge & (z >= thr), res_m)) if d is not None]
        return rows, resp(ha), resp(hb)

    def support(rows, za, zb, tol, rng):
        """Опора куска — меньший из средних откликов двух половин — и порог нуля."""
        fa, fb = (maximum_filter(x, size=2 * tol + 1) if tol else x for x in (za, zb))
        stat = np.array([min(fa[d["pts"][:, 0], d["pts"][:, 1]].mean(),
                             fb[d["pts"][:, 0], d["pts"][:, 1]].mean()) for d in rows])
        null = []
        for _ in range(v3.V7_BRIDGE_SHIFTS):
            dr = int(rng.integers(200, Hh - 200)); dc = int(rng.integers(200, Ww - 200))
            for d in rows:
                r_, c_ = (d["pts"][:, 0] + dr) % Hh, (d["pts"][:, 1] + dc) % Ww
                if valid[r_, c_].all():
                    null.append(min(fa[r_, c_].mean(), fb[r_, c_].mean()))
        return stat, float(np.quantile(np.array(null), q0))

    real = prepare(opt, halves["А"], halves["Б"])
    log(f"снимок готов ({time.time() - t0:.0f} с)")
    stack = np.stack([np.where(valid, x, np.nan) for x in (opt, halves["А"], halves["Б"])])
    sg = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900))
    twin = prepare(sg[0], sg[1], sg[2])
    log(f"двойник готов ({time.time() - t0:.0f} с)")

    work = np.load(v3.work_npz(a.site))
    d_any = distance_transform_edt(~work["A_all"], sampling=(res_m, res_m))
    d_conf = distance_transform_edt(~work["C_all"], sampling=(res_m, res_m))

    def far(d, dist):
        return (dist[d["pts"][:, 0], d["pts"][:, 1]] <= v3.CONF_TOL_M).mean() < v3.CONF_FRAC

    def in_mark(d):
        p = d["pts"]
        return int(((p[:, 0] >= MARK[0]) & (p[:, 0] < MARK[1])
                    & (p[:, 1] >= MARK[2]) & (p[:, 1] < MARK[3])).sum())

    best = None
    for tol in TOLS:
        log(f"\n===== допуск положения {tol * res_m:.0f} м: подтверждено снимок / двойник, "
            f"мусор = двойник : снимок (приемлемо <= {100 * v3.JUNK_MAX:.0f} %)")
        for q in LEVELS:
            rows, trows = real[0][q], twin[0][q]
            stat, thr = support(rows, real[1], real[2], tol, np.random.default_rng(31))
            tstat, tthr = support(trows, twin[1], twin[2], tol, np.random.default_rng(31))
            ln = np.array([d["длина_км"] for d in rows])
            tln = np.array([d["длина_км"] for d in trows])
            for L in LENGTHS_KM:
                c, tc = (ln >= L) & (stat > thr), (tln >= L) & (tstat > tthr)
                n, tn = int(c.sum()), int(tc.sum())
                new = [d for d, k in zip(rows, c) if k and far(d, d_conf)]
                newer = [d for d in new if far(d, d_any)]
                junk = tn / max(n, 1)
                log(f"  уровень {100 * q:.0f} %, от {L:.1f} км: кандидатов "
                    f"{int((ln >= L).sum())} / {int((tln >= L).sum())}; подтверждено {n} "
                    f"({ln[c].sum():.0f} км) / {tn} ({tln[tc].sum():.0f} км); мусор "
                    f"{100 * junk:.1f} % | нет среди подтверждённых рабочего набора "
                    f"{len(new)} ({sum(d['длина_км'] for d in new):.0f} км), из них нет "
                    f"вообще {len(newer)} ({sum(d['длина_км'] for d in newer):.0f} км)")
                if junk <= v3.JUNK_MAX and (best is None or len(new) > best[0]):
                    best = (len(new), q, L, tol,
                            [d for d, k in zip(rows, c) if k])
            hit = [(d, s) for d, s in zip(rows, stat) if in_mark(d) >= MARK_MIN_CELLS]
            log(f"    обведённый уступ (окно 7), порог нуля {thr:.2f}: "
                + ("кусков нет" if not hit else "; ".join(
                    f"{d['длина_км']:.2f} км, опора {s:.2f} "
                    f"{'ДА' if s > thr else 'нет'}" for d, s in hit)))

    if best is None:
        log("\nни одна ячейка сетки не уложилась в порог мусора")
        return
    _n, q, L, tol, keep = best
    log(f"\nячейка с наибольшим числом новых при мусоре в пределах порога: уровень "
        f"{100 * q:.0f} %, от {L:.1f} км, допуск {tol * res_m:.0f} м — её и рисуем")
    mask_new = np.zeros(valid.shape, bool)
    mask_old = np.zeros(valid.shape, bool)
    for d in keep:
        (mask_new if far(d, d_conf) else mask_old)[d["pts"][:, 0], d["pts"][:, 1]] = True
    np.savez_compressed(M / f"openlin_weak_confirm_{a.site}.npz", new=mask_new,
                        old=mask_old, level=q, length_km=L, tol_cells=tol)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_review_tiles import pick_windows
    from experiments.openlin_shade import _stretch

    base = _stretch(np.where(valid, opt, np.nan))
    n_cell = int(5000 / res_m)
    half = n_cell // 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    wins = [(813, 3534)] + [(r, c) for r, c, _k in pick_windows(mask_new, n_cell, 11, 1.0)]
    px = 1000.0 / res_m
    for i, (r, c) in enumerate(wins, 1):
        r = min(max(r, half), Hh - half)
        c = min(max(c, half), Ww - half)
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        fig, ax = plt.subplots(1, 2, figsize=(19, 10), facecolor="#fcfcfb")
        for k in (0, 1):
            ax[k].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[k].plot([10, 10 + px], [n_cell - 10] * 2, color="#111", lw=3)
            ax[k].text(10, n_cell - 16, "1 км", color="#111", fontsize=12)
            ax[k].set_xticks([]); ax[k].set_yticks([])
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[work["C_all"][sl]] = (0.85, 0.05, 0.05, 0.9)
        ov[mask_new[sl]] = (0.10, 0.75, 0.25, 1.0)
        ax[1].imshow(ov)
        ax[0].set_title("без разметки — смотрите сами", loc="left", fontsize=14)
        ax[1].set_title("красные — рабочий набор (подтверждённые); зелёные — слабые "
                        "гребни,\nподтверждённые откликом обеих половин, которых в "
                        "рабочем наборе нет", loc="left", fontsize=13)
        fig.suptitle(f"Окно {i} из {len(wins)} · 5 × 5 км · центр ({r}, {c}) · север "
                     f"сверху" + (" · место, обведённое при просмотре" if i == 1 else ""),
                     x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT_DIR / f"{i:02d}_окно_{r}_{c}.png", dpi=105,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"окна: {OUT_DIR} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
