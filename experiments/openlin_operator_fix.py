"""Оператор границы: почему текстура забивает линию и что с этим делать.

ИЗМЕРЕНИЕ, С КОТОРОГО ВСЁ НАЧАЛОСЬ. На синтетике нынешний оператор даёт на
ровном фоне с одной линией отклик 0.20, а на чистой текстуре без единой линии —
2.16. То есть присутствие текстуры даёт сигнал в десять раз больший, чем сама
линия: оператор меряет «здесь пёстро», а не «здесь граница».

ПРИЧИНА. Сила границы — это длина разности средних по двум окнам. У
коррелированной текстуры средние двух окон и так гуляют: масштаб корреляции
500 м, а окна разнесены на 600 м, поэтому разность средних имеет заметную
дисперсию сама по себе. Максимум по двенадцати направлениям довершает дело —
он систематически выбирает наибольшее отклонение, и на чистом шуме получается
вытянутое пятно вдоль какого-нибудь из направлений. Это и есть «перья».

ЧТО ПРЕДЛАГАЕТСЯ. Нормировать скачок на разброс ВНУТРИ окон: граница значима,
когда разность средних велика по сравнению с тем, как величина гуляет внутри
самих участков. Это обычная t-статистика вместо голой разности. Считается теми
же свёртками, дополнительно нужен только второй момент.

Запуск из корня репозитория:
    python -m experiments.openlin_operator_fix
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.openlin_domains import (N_ANGLES, OFFSET_M, WIN_LONG_M,
                                         WIN_SHORT_M, _kernel, _smooth,
                                         edges_from_components)
from experiments.openlin_operator_test import N, RES_M, cases
from experiments.openlin_slides2 import EDGE, INK, INK2, RAMP, SLATE, SURF

OUT = Path("outputs/steps")


def edges_t(comp: np.ndarray, valid: np.ndarray, res_m: float,
            n_angles: int = N_ANGLES) -> tuple[np.ndarray, np.ndarray]:
    """Сила границы, нормированная на разброс внутри окон (t-статистика).

    Для каждого направления считаются средние и дисперсии величины в двух окнах,
    и скачок делится на ожидаемый разброс этого скачка. Отклик текстуры при этом
    подавляется: у пёстрого места дисперсия внутри окон большая, и знаменатель
    растёт вместе с числителем.
    """
    best = np.full(valid.shape, -np.inf, dtype="float32")
    ang = np.zeros(valid.shape, dtype="float32")
    sq = (comp ** 2).astype("float32")
    for i in range(n_angles):
        th = np.pi * i / n_angles
        s_par, s_perp = WIN_LONG_M / res_m, WIN_SHORT_M / res_m
        kp = _kernel(s_par, s_perp, th, OFFSET_M / res_m)
        km = _kernel(s_par, s_perp, th, -OFFSET_M / res_m)
        # эффективное число независимых отсчётов в окне
        n_eff = float(kp.sum() ** 2 / (kp ** 2).sum())
        mp, mm = _smooth(comp, valid, kp), _smooth(comp, valid, km)
        vp = np.maximum(_smooth(sq, valid, kp) - mp ** 2, 0.0)
        vm = np.maximum(_smooth(sq, valid, km) - mm ** 2, 0.0)
        num = ((mp - mm) ** 2).sum(0)
        den = ((vp + vm).sum(0) / n_eff) + 1e-9
        strength = np.sqrt(num / den).astype("float32")
        upd = strength > best
        best = np.where(upd, strength, best)
        ang = np.where(upd, np.degrees(th), ang)
    return np.where(valid, best, np.nan), ang


#: Ширина центрального окна линейного оператора, м. 100 м = одна ячейка: узкий
#: линеамент на снимке имеет именно такую ширину, шире брать — размажем его
#: вместе с фоном.
CENTER_M = 100.0


def edges_ridge(comp: np.ndarray, valid: np.ndarray, res_m: float,
                n_angles: int = N_ANGLES) -> tuple[np.ndarray, np.ndarray]:
    """Отклик на УЗКУЮ линию: центр против двух боков, с нормировкой.

    Оператор границы сравнивает два участка по разные стороны и по построению
    слеп к узкому линеаменту: у борозды слева и справа одна и та же текстура,
    разность средних около нуля, а сама борозда попадает в зазор между окнами.
    Здесь сравнивается центральная полоса с полусуммой боковых — то есть
    ищется именно линия, а не ступень.
    """
    best = np.full(valid.shape, -np.inf, dtype="float32")
    ang = np.zeros(valid.shape, dtype="float32")
    sq = (comp ** 2).astype("float32")
    for i in range(n_angles):
        th = np.pi * i / n_angles
        s_par = WIN_LONG_M / res_m
        kc = _kernel(s_par, CENTER_M / res_m, th, 0.0)
        kp = _kernel(s_par, WIN_SHORT_M / res_m, th, OFFSET_M / res_m)
        km = _kernel(s_par, WIN_SHORT_M / res_m, th, -OFFSET_M / res_m)
        n_eff = float(kc.sum() ** 2 / (kc ** 2).sum())

        mc, mp, mm = (_smooth(comp, valid, k) for k in (kc, kp, km))
        vc = np.maximum(_smooth(sq, valid, kc) - mc ** 2, 0.0)
        vp = np.maximum(_smooth(sq, valid, kp) - mp ** 2, 0.0)
        vm = np.maximum(_smooth(sq, valid, km) - mm ** 2, 0.0)

        num = ((mc - 0.5 * (mp + mm)) ** 2).sum(0)
        den = ((vc + 0.25 * (vp + vm)).sum(0) / n_eff) + 1e-9
        strength = np.sqrt(num / den).astype("float32")
        upd = strength > best
        best = np.where(upd, strength, best)
        ang = np.where(upd, np.degrees(th), ang)
    return np.where(valid, best, np.nan), ang


def measure(img: np.ndarray, s: np.ndarray, line: np.ndarray | None) -> dict:
    """Отклик на самой линии против отклика вдали от неё."""
    out = {"макс": float(np.nanmax(s)), "фон": float(np.nanmedian(s))}
    if line is not None:
        from scipy.ndimage import binary_dilation

        on = binary_dilation(line > 0, np.ones((3, 3), dtype=bool))
        off = ~binary_dilation(line > 0, np.ones((15, 15), dtype=bool))
        out["на линии"] = float(np.nanmean(s[on]))
        out["вне линии"] = float(np.nanmean(s[off]))
        out["контраст"] = out["на линии"] / max(out["вне линии"], 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = cases()
    line = data["линия"] > 0
    valid = np.ones((N, N), dtype=bool)
    ops = {"как сейчас": edges_from_components,
           "с нормировкой": edges_t,
           "на линию": edges_ridge}

    fig = plt.figure(figsize=(4.6 * len(data), 17.0), facecolor=SURF)
    gs = fig.add_gridspec(4, len(data), hspace=0.14, wspace=0.06, left=0.015,
                          right=0.985, top=0.94, bottom=0.05)
    rows = []
    for i, (name, img) in enumerate(data.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=17, color=INK, fontweight="bold", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(EDGE)
        if i == 0:
            ax.set_xlabel("вход", fontsize=13, color=INK2)

        for j, (op_name, op) in enumerate(ops.items(), start=1):
            s = np.nan_to_num(op(img[None].astype("float32"), valid,
                                 RES_M)[0], nan=0.0)
            m = measure(img, s, line if "линия" in name else None)
            rows.append({"вход": name, "оператор": op_name, **m})
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(s, cmap=RAMP)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(SLATE)
            if i == 0:
                ax.set_xlabel(f"оператор: {op_name}", fontsize=13, color=INK2)

    import pandas as pd

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

    # ключевое число: во сколько раз линия видна на фоне собственных
    # артефактов оператора — сравниваем «шум + линия» с «шум»
    print()
    for op_name in ops:
        q = tab[(tab["оператор"] == op_name)].set_index("вход")
        if "контраст" in q:
            c1 = q.loc["линия", "контраст"]
            c2 = q.loc["шум + линия", "контраст"]
            print(f"{op_name:14s} контраст линии: на чистом фоне {c1:.2f}, "
                  f"в текстуре {c2:.2f}")
    fig.savefig(OUT / "operator_fix.png", dpi=150, facecolor=SURF)
    plt.close(fig)
    print(f"\n{OUT / 'operator_fix.png'}")


if __name__ == "__main__":
    main()
