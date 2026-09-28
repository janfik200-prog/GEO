"""Диагностика: а линеаменты ли это вообще?

Претензия по делу: выделенные отрезки короткие, кривые и рассыпаны по площади —
похоже не на разломную сеть, а на обводку рельефной текстуры. Здесь это
проверяется, а не обсуждается. Три вопроса, три измерения:

1. С ЧЕМ ОНИ СОВПАДАЮТ. Тем же трассировщиком, но БЕЗ всякой модели, линии
   снимаются прямо с композита — со светлых гребней и с тёмных ложбин. Если
   линии модели совпадают с этими «линиями рельефа» сильнее, чем с картой
   разломов, значит модель обводит текстуру, а не структуру.
2. КАК ОНИ ОРИЕНТИРОВАНЫ. Роза азимутов линий модели против розы азимутов
   разломов по карте. У разломной сети направления выражены; если у модели роза
   плоская или смотрит в другую сторону — это не та же сеть.
3. ЧТО ЗНАЧИТ УВЕРЕННОСТЬ. Точность считается по корзинам частоты. Если у
   «жёлтых» линий (прошли почти во всех обучениях) попадание не выше, чем у
   «фиолетовых», то частота меряет устойчивость модели, а не близость к разлому,
   и как мера уверенности бесполезна.

Запуск из корня репозитория (после openlin_lines_ens):
    python -m experiments.openlin_lines_diag [--site anabar] [--tag _густо]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.openlin_anfis import TOL_M
from experiments.openlin_lines import (DENSITY_DEFAULT, _axis, _branches,
                                       line_metrics, trace_lines)
from experiments.openlin_lines_ens import CONSENSUS, thin
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")


def azimuths(mask: np.ndarray, res_m: float) -> np.ndarray:
    """Азимуты веток маски, градусы 0-180 (0 — север)."""
    out = []
    for pts in _branches(mask, res_m):
        if len(pts) < 3:
            continue
        ax, _a, _b = _axis(pts)
        out.append(np.degrees(np.arctan2(ax[1], -ax[0])) % 180)
    return np.asarray(out)


def near_fraction(a: np.ndarray, b: np.ndarray, res_m: float) -> float:
    """Доля длины ``a``, лежащая в допуске от ``b``."""
    from scipy.ndimage import distance_transform_edt

    if not a.any() or not b.any():
        return float("nan")
    return float((distance_transform_edt(~b, sampling=(res_m, res_m))[a]
                  <= TOL_M).mean())


def rose(ax, az: np.ndarray, color: str, title: str) -> None:
    if not len(az):
        ax.set_title(title + "\n(нет линий)", fontsize=9.5)
        return
    bins = np.arange(0, 181, 15)
    h, _ = np.histogram(az, bins=bins)
    h = h / h.sum()
    theta = np.radians(bins[:-1] + 7.5)
    # роза симметрична: направление линии определено с точностью до разворота
    ax.bar(np.concatenate([theta, theta + np.pi]), np.concatenate([h, h]),
           width=np.radians(15), color=color, edgecolor="#333", linewidth=0.4)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=9.5, pad=14)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--train-on", default="nevada")
    ap.add_argument("--model", default="логрегрессия")
    ap.add_argument("--target", default="smooth")
    ap.add_argument("--tag", default="_густо")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    a = ap.parse_args()

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    dist = z["dist"]
    fault_axis = (dist <= res_m * 0.7) & valid

    p = M / (f"openlin_lines_ens_{a.train_on}{a.tag}_{a.model}_{a.target}"
             f"_{a.site}.npz")
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала прогон openlin_lines_ens")
    freq = np.load(p)["freq"]
    lines = thin((freq >= CONSENSUS) & valid) & valid

    # --- 1. линии рельефа: тот же трассировщик, никакой модели ---------------
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    ridge_lines, _ = trace_lines(gray, valid, valid, res_m, a.density)
    valley_lines, _ = trace_lines(-gray, valid, valid, res_m, a.density)
    terrain = ridge_lines | valley_lines

    rows = [
        {"с чем сравниваем": "разломы по карте",
         "доля длины линий модели рядом": near_fraction(lines, fault_axis, res_m)},
        {"с чем сравниваем": "линии рельефа (гребни+ложбины, без модели)",
         "доля длины линий модели рядом": near_fraction(lines, terrain, res_m)},
        {"с чем сравниваем": "линии рельефа: только гребни",
         "доля длины линий модели рядом": near_fraction(lines, ridge_lines, res_m)},
        {"с чем сравниваем": "линии рельефа: только ложбины",
         "доля длины линий модели рядом": near_fraction(lines, valley_lines, res_m)},
    ]
    base_fault = float(((dist <= TOL_M) & valid).sum() / valid.sum())
    from scipy.ndimage import distance_transform_edt
    base_terrain = float((distance_transform_edt(~terrain, sampling=(res_m, res_m))
                          <= TOL_M)[valid].mean())
    print("=== 1. с чем совпадают линии модели " + "=" * 30)
    for r in rows:
        b = base_fault if "разломы" in r["с чем сравниваем"] else base_terrain
        r["случайная линия"] = b
        r["во сколько раз"] = r["доля длины линий модели рядом"] / b
        print(f"  {r['с чем сравниваем']:<46} "
              f"{r['доля длины линий модели рядом']:.3f}  "
              f"(случайная {b:.3f}, в {r['во сколько раз']:.2f} раза)")

    # --- 2. азимуты ---------------------------------------------------------
    az_model = azimuths(lines, res_m)
    az_fault = azimuths(fault_axis, res_m)
    az_terrain = azimuths(terrain, res_m)
    print("\n=== 2. азимуты " + "=" * 51)
    for nm, az in (("линии модели", az_model), ("разломы по карте", az_fault),
                   ("линии рельефа", az_terrain)):
        if len(az):
            h, _ = np.histogram(az, bins=np.arange(0, 181, 15))
            h = h / h.sum()
            print(f"  {nm:<20} веток {len(az):>4}  "
                  f"максимум розы {h.max():.2f} (плоская роза = 0.083), "
                  f"главное направление {int(np.argmax(h) * 15 + 7)}°")

    # --- 3. уверенность против попадания ------------------------------------
    print("\n=== 3. точность по корзинам частоты " + "=" * 30)
    conf = []
    edges = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in edges:
        band = thin((freq >= lo) & (freq < hi) & valid) & valid
        r = line_metrics(band, dist, valid, res_m)
        conf.append({"частота": f"{lo:.2f}-{hi if hi <= 1 else 1:.2f}",
                     "точность": r["точность"], "длина_км": r["длина_км"]})
        print(f"  частота {lo:.2f}-{min(hi, 1):.2f}: точность "
              f"{r['точность']:.3f}, длина {r['длина_км']:.0f} км")
    print(f"  случайная линия: {base_fault:.3f}")

    # --- картинка -----------------------------------------------------------
    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.35,
                          wspace=0.28)

    ax = fig.add_subplot(gs[0, 0], projection="polar")
    rose(ax, az_model, "#12a150", f"линии модели (n={len(az_model)})")
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    rose(ax, az_fault, "#e8402a", f"разломы по карте (n={len(az_fault)})")
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    rose(ax, az_terrain, "#8a8a8a", f"линии рельефа без модели (n={len(az_terrain)})")

    ax = fig.add_subplot(gs[1, :2])
    d = pd.DataFrame(rows)
    pos = np.arange(len(d))
    ax.barh(pos, d["доля длины линий модели рядом"], 0.6,
            color=["#e8402a", "#8a8a8a", "#b0b0b0", "#6f6f6f"],
            edgecolor="#333", linewidth=0.4)
    ax.plot(d["случайная линия"], pos, "k|", ms=15, mew=2,
            label="случайная линия")
    ax.set_yticks(pos)
    ax.set_yticklabels(d["с чем сравниваем"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("доля длины линий модели в допуске 250 м")
    ax.set_title("С чем на самом деле совпадают выделенные линии", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    ax = fig.add_subplot(gs[1, 2])
    c = pd.DataFrame(conf)
    ax.bar(c["частота"], c["точность"], 0.6, color="#12a150",
           edgecolor="#333", linewidth=0.4)
    ax.axhline(base_fault, color="#333", ls="--", lw=1.2)
    ax.set_ylabel("точность")
    ax.set_xlabel("частота в ансамбле")
    ax.set_title("Значит ли что-нибудь уверенность\n"
                 "(пунктир — случайная линия)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Проверка: линеаменты или обводка рельефной текстуры?",
                 fontsize=14, y=0.97)
    out = Path(f"outputs/openlin_{a.site}_lines_diag.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(M / f"openlin_lines_diag_{a.site}.csv", index=False,
                              encoding="utf-8-sig")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
