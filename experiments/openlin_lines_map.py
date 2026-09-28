"""Картинка итога трассировки: линии на карте и их воспроизводимость.

Четыре панели. Слева вверху — Невада, где модель работает: трассированные линии
на фоне композита и разломов по карте. Справа вверху — Анабар той же моделью,
обученной на Неваде. Слева внизу — главный вопрос трассировки: два списка линий,
полученные на двух подвыборках одного и того же обучения, разными цветами.
Справа внизу — точность против доли площади, которую даст линия, брошенная
наугад.

Запуск из корня репозитория (после openlin_lines):
    python -m experiments.openlin_lines_map
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from experiments.openlin_anfis import TARGET_HALFWIDTH_M, TOL_M, block_split
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")
OUT = Path("outputs/openlin_lines_map.png")

#: Что показываем в верхнем ряду: (обучение, модель, таргет, площадка, подпись).
TOP = (("nevada", "бустинг", "hard", "nevada",
        "Невада: обучение и проверка на одной площадке,\nоценка на проверочных полосах"),
       ("nevada", "логрегрессия", "smooth", "anabar",
        "Анабар: та же модель, обученная на Неваде,\nздесь не видела ничего"))

#: Пара для панели устойчивости — медианный по совпадению случай, а не лучший.
STAB = ("nevada", "ANFIS", "hard", "anabar")

SHORT = {"nevada": "Невада", "anabar": "Анабар", "anabar_dykes": "дайки"}


def robust(a: np.ndarray) -> np.ndarray:
    v = a[np.isfinite(a)]
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((a - lo) / ((hi - lo) or 1.0), 0, 1)


def fat(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Утолщение линии до видимой на экране толщины."""
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, np.ones((k, k), dtype=bool))


def load_pred(train: str, model: str, target: str, site: str, sub: int = 1):
    p = M / f"openlin_lines_{train}_{model}_{target}_{sub}_{site}.npz"
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала прогон openlin_lines")
    return np.load(p)["pred"]


def base_rate(z: dict, area: np.ndarray) -> float:
    return float(((z["dist"] <= TOL_M) & area).sum() / area.sum())


def draw_map(ax, site: str, pred: np.ndarray, area: np.ndarray, title: str) -> None:
    z = load_site(site)
    gray = robust(z["gray"])
    valid = np.isfinite(z["gray"])
    ax.imshow(gray, cmap="gray", alpha=0.75)
    fault = (z["dist"] <= float(z["res_m"]) * 0.7) & valid   # осевая линия карты
    ax.imshow(np.ma.masked_where(~fault, np.zeros_like(gray)),
              cmap=ListedColormap(["#e8402a"]), alpha=0.75)
    line = fat(pred & area)
    ax.imshow(np.ma.masked_where(~line, np.zeros_like(gray)),
              cmap=ListedColormap(["#12a150"]), alpha=1.0)
    if area.sum() < valid.sum():                 # показать, где вообще считали
        ax.imshow(np.ma.masked_where(area | ~valid, np.zeros_like(gray)),
                  cmap=ListedColormap(["#ffffff"]), alpha=0.55)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    fig = plt.figure(figsize=(16.5, 13.0))
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.16,
                          height_ratios=[1.0, 0.92])

    frames = []
    for t in ("nevada", "anabar", "anabar_dykes"):
        p = M / f"openlin_lines_{t}.csv"
        if p.exists():
            f = pd.read_csv(p)
            f.insert(0, "обучение", SHORT[t])
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["подвыборка"] == 1]

    # --- верхний ряд: карты ------------------------------------------------
    for col, (train, model, target, site, cap) in enumerate(TOP):
        z = load_site(site)
        valid = np.isfinite(z["gray"])
        if site == train:
            _tr, area = block_split(valid.shape, float(z["res_m"]))
            area &= valid
        else:
            area = valid
        pred = load_pred(train, model, target, site)
        row = df[(df["обучение"] == SHORT[train]) & (df["модель"] == model)
                 & (df["таргет"] == target) & (df["площадка"] == site)]
        acc = float(row["точность"].iloc[0])
        pv = float(row["p"].iloc[0]) if "p" in row and len(row) else float("nan")
        n = int(row["веток"].iloc[0])
        draw_map(fig.add_subplot(gs[0, col]), site, pred, area,
                 f"{cap}\n{model}, таргет {target}: веток {n}\n"
                 f"точность {acc:.2f} против {base_rate(z, area):.2f} "
                 f"у случайной линии (p = {pv:.3f})")

    # --- устойчивость ------------------------------------------------------
    train, model, target, site = STAB
    z = load_site(site)
    valid = np.isfinite(z["gray"])
    a1, a2 = (load_pred(train, model, target, site, s) for s in (1, 2))
    ax = fig.add_subplot(gs[1, 0])
    gray = robust(z["gray"])
    ax.imshow(gray, cmap="gray", alpha=0.7)
    ax.imshow(np.ma.masked_where(~fat(a1), np.zeros_like(gray)),
              cmap=ListedColormap(["#1f66c1"]), alpha=0.95)
    ax.imshow(np.ma.masked_where(~fat(a2), np.zeros_like(gray)),
              cmap=ListedColormap(["#e8a020"]), alpha=0.95)
    st = pd.read_csv(M / f"openlin_lines_{train}_stability.csv")
    val = st[(st["модель"] == model) & (st["таргет"] == target)
             & (st["площадка"] == site)]["совпадение"]
    ax.set_title("Воспроизводимость: два списка линий с двух подвыборок\n"
                 "ОДНОГО И ТОГО ЖЕ обучения; совпадение {:.2f}"
                 .format(float(val.iloc[0]) if len(val) else float("nan")),
                 fontsize=10)
    ax.legend(handles=[Line2D([], [], color="#1f66c1", lw=3, label="подвыборка 1"),
                       Line2D([], [], color="#e8a020", lw=3, label="подвыборка 2")],
              loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- точность против случайной линии -----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    base = {}
    for s in ("nevada", "anabar", "anabar_dykes"):
        zz = load_site(s)
        vv = np.isfinite(zz["gray"])
        _t, va = block_split(vv.shape, float(zz["res_m"]))
        base[SHORT[s]] = {"проверка": base_rate(zz, va & vv),
                          "перенос": base_rate(zz, vv)}
    d = df.dropna(subset=["точность"]).copy()
    d["площадка"] = d["площадка"].map(SHORT)
    d["случайная"] = [base[r["площадка"]][r["режим"]] for _, r in d.iterrows()]
    d = d.sort_values("точность", ascending=False)
    pos = np.arange(len(d))
    ax.barh(pos, d["точность"], 0.72, color=np.where(d["режим"] == "проверка",
                                                     "#12a150", "#8fb8de"),
            edgecolor="#333", linewidth=0.4)
    ax.plot(d["случайная"], pos, "k|", ms=13, mew=2, label="случайная линия")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{r['площадка']} · {r['модель']} · {r['таргет']}"
                        for _, r in d.iterrows()], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel("точность: доля длины линий в допуске 250 м от карты")
    ax.set_title("Все прогоны: точность против случайной линии\n"
                 "зелёное — своя проверочная часть, синее — перенос", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Трассировка линеаментов: из площади в список линий",
                 fontsize=15, y=0.955)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"сохранено: {OUT}")


if __name__ == "__main__":
    main()
