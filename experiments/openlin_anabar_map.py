"""Карта Анабара: что программа выделила как линеаменты.

Одна площадка, крупно. Модель — та же табличная ANFIS, что и в
``openlin_anfis``: обучается на обучающих полосах блочного разбиения, скор
считается по всей площади. Верхние 10% площади — это и есть выделенное
программой; они чистятся от одиночных ячеек и показываются поверх композита,
рядом — разломы по карте, чтобы было видно, совпадает или нет.

Запуск из корня репозитория:
    python -m experiments.openlin_anabar_map [--site anabar] [--area 0.10]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from experiments.openlin_anfis import (AREA, N_TRAIN_ROWS, SEED, TARGET_HALFWIDTH_M,
                                       TOL_M,
                                       anfis_predict, baseline_predict,
                                       block_split, fit_anfis, fit_baselines,
                                       rows, score_map, soft_target)
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, lift_at_top, load_site

#: Мельче этого куска пятно считается одиночным срабатыванием, а не объектом.
#: 20 ячеек по 100 м — примерно 2 км длины при ширине в одну ячейку.
MIN_BLOB = 20

TITLES = {"anabar": "Анабар, разломы", "anabar_dykes": "Анабар, дайки"}


def robust(a: np.ndarray) -> np.ndarray:
    """Растяжка по 2-98 перцентилю: у композита есть остаточные выбросы."""
    v = a[np.isfinite(a)]
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((a - lo) / ((hi - lo) or 1.0), 0, 1)


def clean(mask: np.ndarray) -> np.ndarray:
    """Замкнуть разрывы в один шаг и убрать куски мельче ``MIN_BLOB``."""
    from scipy.ndimage import binary_closing, label

    m = binary_closing(mask, np.ones((3, 3), dtype=bool))
    lab, n = label(m, structure=np.ones((3, 3), dtype=int))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = np.bincount(lab.ravel())[1:] >= MIN_BLOB
    return keep[lab]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--area", type=float, default=AREA)
    ap.add_argument("--rules", type=int, default=16)
    ap.add_argument("--neigh", action="store_true",
                    help="добавить признаки окрестности")
    ap.add_argument("--model", default="ANFIS",
                    choices=("ANFIS", "логрегрессия", "бустинг"))
    ap.add_argument("--target", default="hard", choices=("hard", "smooth"))
    ap.add_argument("--train-on", default="",
                    help="обучать на другой площадке (перенос); по умолчанию на своей")
    a = ap.parse_args()

    def stack(site: str, support: np.ndarray | None = None):
        """Каналы площадки; окрестность собирается только из ``support``."""
        zz = load_site(site)
        xx, _, vv, uu = build_stack(zz, CHANNELS)
        if a.neigh:
            nx, _n = neigh_stack(xx, uu, vv, float(zz["res_m"]),
                                 support=vv if support is None else support)
            xx = np.concatenate([xx, nx])
        return zz, xx, vv

    z, _x0, valid = stack(a.site)
    dist = z["dist"]
    tr, va = block_split(valid.shape, float(z["res_m"]))
    tr &= valid
    va &= valid

    # Обучение: либо на обучающих полосах этой же площадки, либо на другой
    # площадке целиком (перенос — тогда полосы здесь нужны только для честной
    # оценки, обучающая часть в неё не входит).
    if a.train_on:
        # Ровно то же обучение, что в прогоне метрик: обучающие полосы исходной
        # площадки, окрестность собрана только из них.
        z0 = load_site(a.train_on)
        v0 = np.isfinite(z0["gray"])
        t0, _ = block_split(v0.shape, float(z0["res_m"]))
        t0 &= v0
        zt, xt, _vt = stack(a.train_on, support=t0)
        tr_mask, dist_tr = t0, zt["dist"]
    else:
        _zt, xt, _vt = stack(a.site, support=tr)
        tr_mask, dist_tr = tr, dist

    # Оценка на своей площадке: окрестность собирается по всему кадру при
    # переносе (обучающих ячеек здесь нет вовсе) и только по проверочным
    # полосам, когда обучались на этой же площадке.
    _z2, x, _v2 = stack(a.site, support=None if a.train_on else va)

    soft = a.target == "smooth"
    xtr = rows(xt, tr_mask)
    dtr = dist_tr[tr_mask]
    if len(xtr) > N_TRAIN_ROWS:      # та же подвыборка, что в прогоне метрик
        sel = np.random.default_rng(SEED).choice(len(xtr), N_TRAIN_ROWS,
                                                 replace=False)
        xtr, dtr = xtr[sel], dtr[sel]
    ytr = (soft_target(dtr) if soft
           else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))
    if a.model == "ANFIS":
        model = fit_anfis(xtr, ytr, a.rules, soft=soft, log=lambda *_: None)
        pred = anfis_predict(model, rows(x, valid))
    else:
        est = fit_baselines(xtr, ytr, soft)[a.model]
        pred = baseline_predict(est, rows(x, valid))
    score = score_map(pred, valid)

    thr = float(np.quantile(score[valid], 1 - a.area))
    top = clean((score >= thr) & valid)
    near = (dist <= TOL_M)
    hit = float(near[top].mean()) if top.sum() else float("nan")

    # Считаем качество только на проверочных полосах: на обучающих модель уже
    # видела разметку, и общий по площади lift был бы завышен (1.65 против 0.99).
    area_mask = valid if a.train_on else va
    got = lift_at_top(np.nan_to_num(score, nan=-1e9), dist, area_mask,
                      fraction=a.area, tol_m=TOL_M)
    hit_va = (float(near[top & area_mask].mean())
              if (top & area_mask).sum() else float("nan"))

    gray = robust(z["gray"])
    km = (valid.shape[1] * float(z["res_m"]) / 1000.0,
          valid.shape[0] * float(z["res_m"]) / 1000.0)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.6))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("композит (оптика + радар, 100 м)", fontsize=11)

    axes[1].imshow(gray, cmap="gray")
    axes[1].imshow(np.ma.masked_where(~top, np.zeros_like(gray)),
                   cmap=ListedColormap(["#ff8c1a"]), alpha=0.95)
    axes[1].set_title("выделено программой: верхние {:.0f}% площади\n"
                      "после чистки от одиночных ячеек".format(100 * a.area),
                      fontsize=11)

    axes[2].imshow(gray, cmap="gray", alpha=0.55)
    axes[2].imshow(np.ma.masked_where(~top, np.zeros_like(gray)),
                   cmap=ListedColormap(["#ff8c1a"]), alpha=0.9)
    fault = (dist <= TARGET_HALFWIDTH_M) & valid   # за границей снимка не рисуем
    axes[2].imshow(np.ma.masked_where(~fault, np.zeros_like(gray)),
                   cmap=ListedColormap(["#e8402a"]), alpha=0.95)
    where = "по всей площади" if a.train_on else "на проверочных полосах"
    axes[2].set_title("то же и разломы по карте\n"
                      "{} lift@{:.0f}% = {:.2f}, совпало {:.0f}% выделенного"
                      .format(where, 100 * a.area, got["lift"], 100 * hit_va),
                      fontsize=11)
    axes[2].legend(handles=[Patch(color="#ff8c1a", label="выделено программой"),
                            Line2D([], [], color="#e8402a", lw=3,
                                   label="разломы по карте")],
                   loc="lower left", fontsize=9.5, framealpha=0.92)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(*km), fontsize=9.5)

    fig.suptitle("{}: линеаменты, выделенные табличной моделью ({}{}, таргет {}{})"
                 .format(TITLES.get(a.site, a.site), a.model,
                         " + окрестность" if a.neigh else "", a.target,
                         f", обучение на {a.train_on}" if a.train_on else ""),
                 fontsize=14.5, y=0.97)
    fig.text(0.5, 0.035,
             ("модель обучена на другой площадке и на этой не видела ничего; "
              if a.train_on else
              "модель обучена на обучающих полосах блочного разбиения (буфер 2 км); ")
             + "lift = 1.00 — уровень случайной карты",
             ha="center", fontsize=9.5, color="#444")

    out = Path("outputs/openlin_{}_lineaments{}{}.png".format(
        a.site, "_neigh" if a.neigh else "",
        f"_from_{a.train_on}" if a.train_on else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("сохранено: {}  lift(проверка)={:.2f}  совпало {:.0f}% "
          "(по всей площади {:.0f}%)".format(out, got["lift"], 100 * hit_va,
                                             100 * hit))


if __name__ == "__main__":
    main()
