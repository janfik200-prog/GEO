"""Проба: слабые гребни без затравки, пропущенные через продолжение и сшивку.

ПОВОД. В окне 7 обведён уступ, который детектор видит, но в линию не
превращает. :mod:`openlin_weak_confirm_probe` показал: уступ лежит на гребне
между 80 и 90 % и режется на куски 0.2-0.45 км. Одиночные куски такой длины от
шума не отличить (мусор 63-67 %), а длинные прямые куски отличаются резко: от
1 км на уровне 80 % — 985 на снимке против 7 на двойнике. Значит, слабые куски
надо сначала СОБРАТЬ в линию, а уже потом судить по длине.

ПОПУТНЫЙ ВЫВОД той пробы: сверка с половинами по непрерывному отклику ничего
не отсеивает (проходит 90 % кусков снимка и 86 % кусков двойника). Половины —
части того же снимка, и его гребень в них есть по построению; независимой
проверкой это не является. Поэтому здесь подтверждения нет вовсе, судит
только двойник.

ЧТО ДЕЛАЕТСЯ. Кандидаты — гребни выше уровня q без требования затравки; дальше
рабочий тракт без изменений: продолжение концов и сшивка с пределом бокового
сдвига по близнецам. Для сравнения — действующий детектор (затравка 97 %,
продолжение от 90 %) через тот же тракт. Считается на снимке и на двойнике:
сколько линий длиннее L и каково отношение двойник : снимок. Число L для
слабых линий берётся там, где это отношение опускается до ``JUNK_MAX``.
Отдельно — попал ли в линии обведённый уступ.

Это проба: рабочий набор не меняется, в счётчик не идёт.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_weak_tract.log``).
Линии после сшивки, снимок / двойник (мусор):
                         от 1.0 км            от 1.5 км           от 2.0 км
  действующий детектор   1497 / 149 (10 %)    564 / 13 (2.3 %)    233 / 1
  уровень 85 %           2060 / 294 (14 %)    784 / 24 (3.1 %)    311 / 0
  уровень 80 %           2369 / 439 (18.5 %)  919 / 50 (5.4 %)    363 / 8
  1. ОБВЕДЁННЫЙ УСТУП НЕ СОБРАЛСЯ: его куски (0.71, 0.34, 0.26, 0.21 км) так и
     остались кусками. Причина — остальной тракт тоже настроен на сильные
     линии: продолжение концов идёт, пока отклик выше 90 %, опора мостика
     требует ~1.0 сигмы, а уступ весь лежит на 0.8-1.2. К тому же он дуга.
  2. ПРИБАВКА ЕСТЬ, НО У НЕЁ СВОЙ МУСОР. Считать надо ПРИРОСТ: на уровне 80 %
     от 1 км добавляется 872 линии снимка и 290 линий двойника — мусор
     прироста 33 %, не годится; от 1.5 км добавляется 355 и 37 — мусор 10 %.
     Совсем новых (нет в рабочем наборе ни в каком виде) от 1.5 км — 70
     линий, 129 км.
  3. ПОПУТНО: уступ проверен как граница тона и как более широкая полоса —
     сильнее он от этого не становится (на своём месте в отклике: узкая
     полоса 87-93-й процентиль, граница тона 79-91-й, полоса 40-80 м
     75-84-й). В любом из этих детекторов он лежит около 90-го процентиля —
     там, где шума столько же, сколько сигнала.

Запуск из корня репозитория:
    python -m experiments.openlin_weak_tract_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import COMP, M, OUT

LEVELS = (0.85, 0.80)
LENGTHS_KM = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
MARK = (695, 762, 3508, 3575)        # обведённое место в окне 7: строки, столбцы листа
MARK_MIN_CELLS = 10
OUT_DIR = OUT.parent / "линеаменты_слабые_тракт"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_iso import (Q_HIGH, Q_LOW, hysteresis8, local_norm,
                                         ridge_response, to_lines_strict)
    from experiments.openlin_kora import surrogate
    from experiments.openlin_lines import _axis, _ridge_nms
    from experiments.openlin_shadow import standardize

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = v3.build_halves(a.site, lambda *_: None)
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])
    zone = np.zeros(valid.shape, bool)           # сшивка зону не использует

    def tract(o, level):
        """Линии после продолжения и сшивки; level=None — действующий детектор."""
        x = np.where(valid, standardize(o, valid), 0.0).astype("float32")
        z = local_norm(np.where(valid, ridge_response(x), 0.0), valid, res_m)
        v = z[valid]
        _s, ridge = _ridge_nms(z, valid, 1.0)
        zr = np.where(ridge & valid, z, float(v.min()))
        if level is None:
            mask = hysteresis8(zr, np.quantile(v, Q_LOW), np.quantile(v, Q_HIGH))
        else:
            mask = zr >= np.quantile(v, level)
        _m, rows = to_lines_strict((mask, z >= np.quantile(v, Q_LOW)), res_m)
        for r in rows:
            ax, _a, _b = _axis(r["pts"])
            r["азимут"] = float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180.0)
        rows = v3.extend_ends(rows, z, valid, res_m)
        L = v3.link_same_side(rows, zone, zone, res_m, None, z=z,
                              offset_m=v3.TWIN_OFFSET_M)
        for x_ in L:                                # «цепочки»: мостики через кулисы
            kul = x_.get("кулисы")
            if kul is not None and len(kul):
                W_ = valid.shape[1]
                x_["pts"] = x_["pts"][~np.isin(x_["pts"][:, 0] * W_ + x_["pts"][:, 1],
                                               kul[:, 0] * W_ + kul[:, 1])]
        return L

    stack = np.stack([np.where(valid, x, np.nan) for x in (opt, halves["А"], halves["Б"])])
    twin_img = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900))[0]
    log(f"двойник построен ({time.time() - t0:.0f} с)")

    work = np.load(v3.work_npz(a.site))
    d_any = distance_transform_edt(~work["A_all"], sampling=(res_m, res_m))

    def far(d):
        return (d_any[d["pts"][:, 0], d["pts"][:, 1]] <= v3.CONF_TOL_M).mean() < v3.CONF_FRAC

    def in_mark(d):
        p = d["pts"]
        return int(((p[:, 0] >= MARK[0]) & (p[:, 0] < MARK[1])
                    & (p[:, 1] >= MARK[2]) & (p[:, 1] < MARK[3])).sum())

    keep_for_draw = None
    for level in (None,) + LEVELS:
        name = "действующий детектор" if level is None else f"уровень {100 * level:.0f} %, без затравки"
        R = tract(opt, level)
        T = tract(twin_img, level)
        ln = np.array([d["длина_км"] for d in R]); tln = np.array([d["длина_км"] for d in T])
        log(f"\n=== {name}: линий после сшивки снимок {len(R)} / двойник {len(T)} "
            f"({time.time() - t0:.0f} с)")
        ok_L = None
        for L in LENGTHS_KM:
            n, tn = int((ln >= L).sum()), int((tln >= L).sum())
            new = [d for d in R if d["длина_км"] >= L and far(d)]
            junk = tn / max(n, 1)
            log(f"  от {L:.2f} км: {n} ({ln[ln >= L].sum():.0f} км) / {tn} "
                f"({tln[tln >= L].sum():.0f} км); мусор {100 * junk:.1f} % | нет в "
                f"рабочем наборе вообще: {len(new)} ({sum(d['длина_км'] for d in new):.0f} км)")
            if ok_L is None and junk <= v3.JUNK_MAX:
                ok_L = L
        hit = sorted((d["длина_км"] for d in R if in_mark(d) >= MARK_MIN_CELLS), reverse=True)
        log(f"  обведённый уступ (окно 7): линии, задевающие место, км: "
            f"{np.round(hit, 2) if hit else 'нет'}; мусор укладывается в "
            f"{100 * v3.JUNK_MAX:.0f} % начиная с длины {ok_L} км")
        if level == LEVELS[-1] and ok_L is not None:
            keep_for_draw = ([d for d in R if d["длина_км"] >= ok_L], ok_L, level)

    if keep_for_draw is None:
        return
    keep, ok_L, level = keep_for_draw
    mask_new = np.zeros(valid.shape, bool); mask_old = np.zeros(valid.shape, bool)
    for d in keep:
        (mask_new if far(d) else mask_old)[d["pts"][:, 0], d["pts"][:, 1]] = True
    np.savez_compressed(M / f"openlin_weak_tract_{a.site}.npz", new=mask_new, old=mask_old,
                        level=level, length_km=ok_L)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_review_tiles import pick_windows
    from experiments.openlin_shade import _stretch

    base = _stretch(np.where(valid, opt, np.nan))
    Hh, Ww = valid.shape
    n_cell = int(5000 / res_m); half = n_cell // 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    wins = [(813, 3534)] + [(r, c) for r, c, _k in pick_windows(mask_new, n_cell, 11, 1.0)]
    px = 1000.0 / res_m
    for i, (r, c) in enumerate(wins, 1):
        r = min(max(r, half), Hh - half); c = min(max(c, half), Ww - half)
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        fig, ax = plt.subplots(1, 2, figsize=(19, 10), facecolor="#fcfcfb")
        for k in (0, 1):
            ax[k].imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[k].plot([10, 10 + px], [n_cell - 10] * 2, color="#111", lw=3)
            ax[k].text(10, n_cell - 16, "1 км", color="#111", fontsize=12)
            ax[k].set_xticks([]); ax[k].set_yticks([])
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[work["A_all"][sl]] = (0.95, 0.60, 0.0, 0.9)
        ov[work["C_all"][sl]] = (0.85, 0.05, 0.05, 0.9)
        ov[mask_new[sl]] = (0.10, 0.75, 0.25, 1.0)
        ax[1].imshow(ov)
        ax[0].set_title("без разметки — смотрите сами", loc="left", fontsize=14)
        ax[1].set_title(f"красные и жёлтые — рабочий набор; зелёные — линии из слабых "
                        f"гребней (уровень {100 * level:.0f} %,\nот {ok_L} км после "
                        f"сшивки), которых в рабочем наборе нет вообще",
                        loc="left", fontsize=13)
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
