"""Замер: откуда брать условие на длину линий вместо необоснованных 1.5 км.

ПОВОД. Условие приёмки «медиана длины набора >= 1.5 км» (``V3_MEDIAN_KM_MIN``,
перенесено из ``V2_MEDIAN_KM_MIN``) введено без замера — круглое число, до
правила «числа только из данных». Оно дважды закрыло набор, который был чище
рабочего, но короче: симметричная обработка половин (1.45 км) и опора мостика
по пути (1.42 км). Вопрос пользователя 09.10.2026: «откуда взялся порог»,
поручение — «разберись с порогом длины».

ЧТО НАДО ПОНЯТЬ. Условие на длину защищает от набора, собранного из коротких
обрывков, которые ничем не отличаются от шума. Значит, мерить длину надо не
против круглого числа, а против того, что даёт ШУМ: карта-двойник проходит тот
же тракт, и её линии — это длины, которые получаются без единого линеамента.

ЧТО МЕРИТСЯ. Рабочий тракт («цепочки») на снимке и на трёх двойниках, длины
линий по стадиям:
  до сшивки (обрывки от 500 м) -> после сшивки -> подтверждённые половинами.
По ним: квантили длин снимка и двойника; на каком месте в распределении
двойника стоят 1.5 км; какая доля линий и длины снимка лежит выше верхних
квантилей двойника. Длины сохраняются в
``outputs/metrics/openlin_length_probe_{site}.npz``.

Это замер: рабочий набор не меняется, в счётчик не идёт, порог не трогается.
Замена условию объявляется отдельно, до следующего прогона.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_length_probe.log``).
Длины, км (25 / 50 / 75 / 90 / 95 / 99 %):
  после сшивки:    снимок 3191 линия,  0.76 / 1.06 / 1.51 / 2.24 / 2.76 / 4.29;
                   двойник (один) ~1193, 0.66 / 0.84 / 1.08 / 1.34 / 1.51 / 1.98;
  подтверждённые:  снимок 1724,        0.89 / 1.27 / 1.84 / 2.57 / 3.23 / 4.91;
                   двойник ~27,        0.66 / 0.90 / 1.11 / 1.39 / 1.57 / 1.76;
  в зонах, подтв.: снимок 182,         1.10 / 1.65 / 2.20 / 3.26 / 3.82 / 6.37.
ЧТО ИЗ ЭТОГО СЛЕДУЕТ.
  1. 1.5 км — НЕ ПРОИЗВОЛЬНЫЙ МАСШТАБ: это 95-й процентиль длины линий двойника
     после сшивки (1.51 км), то есть длина, до которой шум дотягивает в одном
     случае из двадцати. Совпадение случайное — число назначалось без замера.
  2. ПРОИЗВОЛЬНА ФОРМА УСЛОВИЯ. «Медиана >= 1.5 км» значит «половина набора
     длиннее того, что шум даёт в 5 % случаев». Откуда «половина» — ниоткуда.
     Сам двойник короче 1.5 км в 95 % линий, подтверждённый набор — в 63 %.
  3. Набор длиной от шума отличается уверенно: выше 95-го процентиля двойника
     37 % подтверждённых линий листа (58 % длины) и 55 % в зонах (75 % длины)
     при 5 % у двойника.
  4. ПОПУТНО: сшивка плодит шумовые линии. У двойника линий от 500 м до сшивки
     ~297, после ~1193 (в 4 раза); у снимка 2681 -> 3191 (в 1.2 раза). После
     сшивки шумовых линий треть от настоящих; убирает их только подтверждение
     половинами (остаётся ~27). Это тот же вывод, что дали близнецы: правило
     сшивки слишком свободное.

Запуск из корня репозитория:
    python -m experiments.openlin_length_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import (COMP, M, MIN_LEN_M, N_FAKES, mask_of,
                                            trace, zone_and_strike)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_kora import surrogate
    from experiments.openlin_shadow import standardize

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = v3.build_halves(a.site, log)
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])
    k = int(round(100 / res_m))
    zone, strike = zone_and_strike(a.site, k)
    zone = zone[:valid.shape[0], :valid.shape[1]] & valid
    strike = strike[:valid.shape[0], :valid.shape[1]]
    up = lambda x: np.repeat(np.repeat(x, k, 0), k, 1)[:valid.shape[0], :valid.shape[1]]
    zone = up(v3.close_along_strike(zone[::k, ::k], strike[::k, ::k], 100.0)) & valid
    cnt = w["count"][:valid.shape[0], :valid.shape[1]]
    seam = np.zeros(valid.shape, bool)
    seam[:, :-1] |= cnt[:, :-1] != cnt[:, 1:]
    seam[:-1, :] |= cnt[:-1, :] != cnt[1:, :]
    seam |= ~valid

    def tract(o, ho, hf):
        """Рабочий тракт «цепочки»; длины линий по стадиям, км."""
        xx = np.where(valid, standardize(o, valid), 0.0).astype("float32")
        zz = local_norm(np.where(valid, ridge_response(xx), 0.0), valid, res_m)
        raw = v3.extend_ends(trace(o, valid, res_m, local=True, min_len_m=0),
                             zz, valid, res_m)
        before = [x["длина_км"] for x in raw if x["длина_км"] * 1000 >= MIN_LEN_M]
        L = v3.link_same_side(raw, zone, strike, res_m, None, z=zz)
        L = [x for x in L if x["длина_км"] * 1000 >= MIN_LEN_M]
        L = v3.drop_seam_lines(L, seam, res_m)
        W_ = valid.shape[1]
        for x in L:                                  # «цепочки»: мостики через кулисы
            kul = x.get("кулисы")
            if kul is not None and len(kul):
                x["pts"] = x["pts"][~np.isin(x["pts"][:, 0] * W_ + x["pts"][:, 1],
                                             kul[:, 0] * W_ + kul[:, 1])]
        dts = [distance_transform_edt(~mask_of(trace(h, valid, res_m, local=True),
                                               valid.shape), sampling=(res_m, res_m))
               for h in (ho, hf)]
        Az, Cz = v3.select2(L, zone, strike, dts)
        Aw, Cw = v3.select2(L, zone, strike, dts, gate=False)
        ln = lambda S: np.array([x["длина_км"] for x in S], dtype="float32")
        return dict(до_сшивки=np.array(before, "float32"), после_сшивки=ln(Aw),
                    подтверждённые=ln(Cw), в_зонах=ln(Az), в_зонах_подтв=ln(Cz))

    real = tract(opt, halves["А"], halves["Б"])
    log(f"снимок готов ({time.time() - t0:.0f} с)")
    stack = np.stack([np.where(valid, x, np.nan) for x in (opt, halves["А"], halves["Б"])])
    fakes = []
    for i in range(N_FAKES):
        sg = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900 + i))
        fakes.append(tract(sg[0], sg[1], sg[2]))
        log(f"двойник {i + 1} готов ({time.time() - t0:.0f} с)")
    fake = {key: np.concatenate([f[key] for f in fakes]) for key in real}
    np.savez_compressed(M / f"openlin_length_probe_{a.site}.npz",
                        **{f"снимок_{k_}": v for k_, v in real.items()},
                        **{f"двойник_{k_}": v for k_, v in fake.items()})

    qs = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    log("\nдлины линий, км (квантили 25 / 50 / 75 / 90 / 95 / 99 %); у двойника — "
        f"сумма {N_FAKES} двойников")
    for key in real:
        for nm, d in (("снимок ", real), ("двойник", fake)):
            x = d[key]
            if len(x) < 5:
                log(f"  {key:16s} {nm}: линий {len(x)} — мало для квантилей"
                    + (f", длины {np.round(np.sort(x), 2)}" if len(x) else ""))
                continue
            log(f"  {key:16s} {nm}: линий {len(x):5d}, сумма {x.sum():6.0f} км | "
                + " / ".join(f"{v:.2f}" for v in np.quantile(x, qs)))

    old = v3.V3_MEDIAN_KM_MIN
    log(f"\nместо прежнего порога {old} км в распределениях (доля линий короче):")
    for key in ("до_сшивки", "после_сшивки", "подтверждённые"):
        log(f"  {key:16s}: снимок {100 * (real[key] < old).mean():.1f} %, "
            f"двойник {100 * (fake[key] < old).mean():.1f} %")

    log("\nснимок против шума: какая часть подтверждённого набора длиннее, чем "
        "бывает у двойника после сшивки")
    ref = fake["после_сшивки"]
    for name, x in (("весь лист", real["подтверждённые"]),
                    ("в зонах", real["в_зонах_подтв"])):
        if not len(x):
            continue
        for qv in (0.5, 0.9, 0.95, 0.99):
            t = float(np.quantile(ref, qv))
            m = x > t
            log(f"  {name:9s}: длиннее {100 * qv:.0f}-го процентиля двойника ({t:.2f} км) — "
                f"{100 * m.mean():.1f} % линий, {100 * x[m].sum() / x.sum():.1f} % длины "
                f"(у самого двойника {100 * (1 - qv):.0f} % линий)")
    log(f"\nготово ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
