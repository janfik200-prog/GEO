"""Устойчивая «Кора»: отбор правил по повторяемости и новая пара прогонов.

ОТКУДА ПОСТАНОВКА. «Кора» на признаках согласия (:mod:`openlin_invariant`)
первой в ветке удержала AUC >= 0.70 на обеих половинах листа (0.761 и 0.735),
но устойчивость карты не дотянула: корреляция двух прогонов 0.57 при пороге
0.6, простирание верхних 10 % расходится на 12° при пороге 10°. На каждую
половину листа приходилась одна подделка и одна выборка строк, и часть правил
— случайность именно этой выборки.

ЧТО МЕНЯЕТСЯ (зафиксировано до прогона, 22.09.2026):

* в каждой половине «Кора» запускается ``K`` раз, каждый со своей подделкой
  и своей выборкой строк; остаются правила, найденные минимум в ``KEEP`` из
  ``K`` запусков (отбор по устойчивости, Meinshausen & Bühlmann); оценка ячейки
  — по этим правилам, границы терцилей — по объединённой выборке всех запусков;
* проверяется на НОВОЙ паре прогонов, а не на тех двух, где было «почти»:
  полосы поперечные (вертикальные вместо горизонтальных), зёрна подделок новые;
* условия прежние: AUC >= 0.70 в обоих прогонах, корреляция карт >= 0.6,
  простирание верхних 10 % расходится не больше чем на 10°.

Чего здесь сознательно НЕТ: сглаживания карт перед сравнением и смягчения
порогов — и то и другое улучшает число, а не метод. Проверки по месту против
разломов тоже нет: цель ветки — линеаменты, разломы — сверка после.

РЕЗУЛЬТАТ (22.09.2026) — все три условия пройдены, с запасом.

* AUC: 0.845 и 0.828 (было 0.761 и 0.735 без отбора по повторяемости).
* Корреляция карт двух прогонов: 0.84 (было 0.57).
* Простирание верхних 10 %: 68° и 68°, разница 1° (было 12°).
* Устойчивых правил 3178 и 2588 из 36-39 тысяч найденных хотя бы раз;
  правила «снимка», найденные во всех 7 запусках, — согласие направления
  между приборами (яркость, NDVI, VH, летняя когерентность) на 0.5-2 км и
  согласие прибора с самим собой между масштабами.
* Проверка на гидросеть (для сведения, не условие): верхние 10 % НЕ тяготеют к
  воде — 0.88-0.97x при нуле 1.00x, p = 0.65-0.85; средняя оценка у воды и
  вдали от неё одинакова. Первый результат ветки, не сводящийся к руслам.

Оговорка: разбиение и зёрна новые, но лист тот же, и метод доводился на нём
за несколько шагов. Внешняя проверка — другой участок или эталон.

Запуск из корня репозитория:
    python -m experiments.openlin_kora_stable --site anabar
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

# ---- пре-регистрация (22.09.2026, до прогона) ------------------------------
K = 7
KEEP = 5
SEED_BASE = 1000
AUC_MIN = 0.70
SCORE_RHO_MIN = 0.6
STRIKE_MAX_DIFF = 10.0
AREA = 0.10
# ---------------------------------------------------------------------------


def block_split_vertical(shape, res_m):
    """Те же полосы с буфером, что в :func:`openlin_anfis.block_split`, но
    поперечные: разбиение делается на транспонированной сетке."""
    from scipy.ndimage import binary_erosion

    from experiments.openlin_anfis import BUFFER_M
    from experiments.openlin_unet import stripe_split

    val = stripe_split((shape[1], shape[0])).T
    k = int(round(BUFFER_M / res_m))
    st = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
    return (binary_erosion(~val, st, border_value=1),
            binary_erosion(val, st, border_value=1))


#: Для итоговой картинки: какие ячейки считаются зоной согласия (доля площади
#: с наибольшей оценкой, средней по двум прогонам), шаг сетки штрихов и их
#: длина, км. Это только оформление карты, на проверки не влияет.
ZONE_SHARE = 0.20
TICK_STEP_KM = 2.0
TICK_LEN_KM = 1.6
TICK_SCALE = 1000


def plot_map(site: str, log) -> None:
    """Итоговая карта: зоны согласия поверх композита и штрихи простирания.

    Простирание в ячейке — среднее по четырём приборам (яркость, NDVI, VH,
    летняя когерентность) на масштабе 1 км: векторы удвоенного угла
    складываются, поэтому противоположные направления не гасят друг друга.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from experiments.openlin_forest_kora import neigh_all
    from experiments.openlin_invariant import INSTRUMENTS
    from experiments.openlin_shade import _stretch
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    d = np.load(M / f"openlin_kora_stable_{site}.npz")
    s = 0.5 * (d["score_1"] + d["score_2"])
    valid &= np.isfinite(s)
    zone = valid & (s >= np.quantile(s[valid], 1 - ZONE_SHARE))

    stack, names = neigh_all(x, use, valid, res_m, valid)
    idx = {n: i for i, n in enumerate(names)}
    c2 = sum(stack[idx[f"{ch}_cos{TICK_SCALE}"]] for ch in INSTRUMENTS)
    s2 = sum(stack[idx[f"{ch}_sin{TICK_SCALE}"]] for ch in INSTRUMENTS)
    grad_az = (90.0 + np.degrees(np.arctan2(s2, c2) / 2.0)) % 180.0
    strike = (grad_az + 90.0) % 180.0

    step = int(round(TICK_STEP_KM * 1000 / res_m))
    half = TICK_LEN_KM * 1000 / res_m / 2.0
    segs = []
    for r in range(step // 2, valid.shape[0], step):
        for c in range(step // 2, valid.shape[1], step):
            win = zone[max(r - step // 2, 0):r + step // 2,
                       max(c - step // 2, 0):c + step // 2]
            if win.mean() < 0.5:
                continue
            a = np.radians(strike[r, c])
            dx, dy = np.sin(a) * half, -np.cos(a) * half
            segs.append([(c - dx, r - dy), (c + dx, r + dy)])

    base = _stretch(np.where(valid, z["gray"], np.nan))
    fig, ax = plt.subplots(1, 2, figsize=(20, 9.2), facecolor="#fcfcfb")
    ax[0].imshow(base, cmap="gray", vmin=0, vmax=1)
    ov = np.zeros(zone.shape + (4,), dtype="float32")
    ov[zone] = (0.89, 0.29, 0.28, 0.45)
    ax[0].imshow(ov)
    ax[0].set_title(f"зоны согласия приборов ({int(ZONE_SHARE * 100)} % листа "
                    f"с наибольшей оценкой)\nповерх летнего композита",
                    fontsize=14, loc="left")
    ax[1].imshow(base, cmap="gray", vmin=0, vmax=1, alpha=0.55)
    ax[1].add_collection(LineCollection(segs, colors="#b3261e", linewidths=1.6))
    ax[1].set_xlim(0, valid.shape[1]); ax[1].set_ylim(valid.shape[0], 0)
    ax[1].set_title(f"преобладающее простирание в зонах согласия "
                    f"(штрихи через {TICK_STEP_KM:g} км,\nсреднее по яркости, "
                    f"NDVI, VH и радарной когерентности на 1 км)",
                    fontsize=14, loc="left")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
    fig.suptitle("Устойчивая «Кора» на признаках согласия: AUC 0.85 и 0.83 на "
                 "двух половинах листа, карты совпадают на 0.84, с гидросетью "
                 "не связано", fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / f"зоны_согласия_{site}.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p} (штрихов {len(segs)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--plot", action="store_true",
                    help="только итоговая карта по уже посчитанному")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    if a.plot:
        plot_map(a.site, log)
        return

    from scipy.stats import spearmanr

    from experiments.openlin_forest_kora import neigh_all
    from experiments.openlin_invariant import builder_invariant
    from experiments.openlin_kora import (N_LEVELS, N_TEST, N_TRAIN, binarize,
                                          describe, kora, kora_score,
                                          sample_rows, surrogate)
    from experiments.openlin_kora_check import strike
    from experiments.openlin_line_rank import auc
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(a.site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    build = builder_invariant(a.site)
    tr0, te0 = block_split_vertical(valid.shape, res_m)
    tr0 &= valid
    te0 &= valid
    log(f"площадка {a.site}: полосы поперечные, ячеек {int(tr0.sum())} и "
        f"{int(te0.sum())}; K = {K}, правило остаётся при {KEEP} из {K}")

    t0 = time.time()

    def feat(src, support):
        return build(*neigh_all(src, use, valid, res_m, support))

    real = {"A": feat(x, tr0), "B": feat(x, te0)}
    names = real["A"][1]
    full160 = neigh_all(x, use, valid, res_m, valid)
    full, _n = build(*full160)
    rows_full = full.reshape(full.shape[0], -1)[:, valid.ravel()].T
    log(f"признаки снимка посчитаны ({time.time() - t0:.0f} с)")

    maps, aucs = {}, {}
    for run, swap, off in (("прогон 1", False, 0), ("прогон 2", True, 100)):
        train, test = (te0, tr0) if swap else (tr0, te0)
        f_train = real["B" if swap else "A"][0]
        f_test = real["A" if swap else "B"][0]
        found, samples = [], []
        for k in range(K):
            seed = SEED_BASE + off + k
            fk = feat(surrogate(x, valid, np.random.default_rng(seed)), train)[0]
            rng = np.random.default_rng(seed + 50)
            r_tr = sample_rows(f_train, train, N_TRAIN, rng)
            f_tr = sample_rows(fk, train, N_TRAIN, rng)
            samples.append((r_tr, f_tr))
        pooled = np.vstack([np.vstack(s) for s in samples])
        edges = np.stack([np.quantile(pooled[:, j],
                                      np.linspace(0, 1, N_LEVELS + 1)[1:-1])
                          for j in range(pooled.shape[1])])
        for k, (r_tr, f_tr) in enumerate(samples):
            rules = kora(binarize(r_tr, edges), binarize(f_tr, edges),
                         len(names), lambda *_: None)
            found.append({(q["класс"], q["условия"]) for q in rules})
            log(f"  {run}, запуск {k + 1}/{K}: правил {len(rules)} "
                f"({time.time() - t0:.0f} с)")
        cnt = Counter(r for s in found for r in s)
        stable = [{"класс": c, "условия": u} for (c, u), n in cnt.items()
                  if n >= KEEP]
        n1 = sum(r["класс"] == 1 for r in stable)
        log(f"  {run}: устойчивых правил {len(stable)} (снимка {n1}) из "
            f"{len(cnt)} найденных хотя бы раз")

        fk_te = feat(surrogate(x, valid,
                               np.random.default_rng(SEED_BASE + off + 99)),
                     test)[0]
        rng = np.random.default_rng(SEED_BASE + off + 98)
        r_te = sample_rows(f_test, test, N_TEST, rng)
        f_te = sample_rows(fk_te, test, N_TEST, rng)
        y = np.r_[np.ones(len(r_te)), np.zeros(len(f_te))]
        aucs[run] = auc(y, kora_score(binarize(np.vstack([r_te, f_te]), edges),
                                      stable))
        m = np.full(valid.shape, np.nan, dtype="float32")
        m[valid] = kora_score(binarize(rows_full, edges), stable)
        maps[run] = m
        log(f"  {run}: AUC на проверке {aucs[run]:.3f}")
        top = sorted((r for r in stable if r["класс"] == 1),
                     key=lambda r: -cnt[(1, r["условия"])])[:5]
        for r in top:
            log(f"    {cnt[(1, r['условия'])]}/{K}  "
                f"{describe(r['условия'], names)}")

    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    ok_auc = all(v >= AUC_MIN for v in aucs.values())
    sA, sB = maps["прогон 1"], maps["прогон 2"]
    idx = np.flatnonzero(valid.ravel())
    pick = np.random.default_rng(3).choice(idx, min(50_000, len(idx)),
                                           replace=False)
    rho = spearmanr(sA.ravel()[pick], sB.ravel()[pick]).statistic
    stack160, names160 = full160
    f = {n: stack160[i] for i, n in enumerate(names160)}
    topA = valid & (sA >= np.quantile(sA[valid], 1 - AREA))
    topB = valid & (sB >= np.quantile(sB[valid], 1 - AREA))
    s_ = ("coh_summer_vv_COH12_sin1000", "coh_summer_vv_COH12_cos1000")
    stA, stB = strike(f, topA, *s_), strike(f, topB, *s_)
    d = abs(((stA - stB) + 90.0) % 180.0 - 90.0)
    log(f"\nAUC >= {AUC_MIN} в обоих: " + ", ".join(
        f"{k} {v:.3f}" for k, v in aucs.items()) + f" -> {mark[ok_auc]}")
    log(f"корреляция карт: {rho:.2f} (порог {SCORE_RHO_MIN}) -> "
        f"{mark[rho >= SCORE_RHO_MIN]}")
    log(f"простирание верхних 10 %: {stA:.0f}° и {stB:.0f}°, разница {d:.0f}° "
        f"(порог {STRIKE_MAX_DIFF:.0f}°) -> {mark[d <= STRIKE_MAX_DIFF]}")

    np.savez_compressed(M / f"openlin_kora_stable_{a.site}.npz", score_1=sA,
                        score_2=sB)
    from experiments.openlin_forest_kora import plot
    plot(sA, sB, z, valid, f"{a.site}_кора_устойч",
         {"исходный": {"auc": aucs["прогон 1"]},
          "повтор": {"auc": aucs["прогон 2"]}}, log,
         "устойчивая «Кора»")


if __name__ == "__main__":
    main()
