"""Картинка итога ансамблевой трассировки: частота линии вместо списка линий.

Верхний ряд — Анабар: поле частоты (как часто линия прошла через ячейку по
подвыборкам), консенсусные линии поверх карты разломов и, для сравнения,
одиночная трассировка из ``openlin_lines``. Нижний ряд — то, ради чего ансамбль
и делался: воспроизводимость (одиночная против ансамблевой) и точность
консенсусных линий против случайной линии.

Запуск из корня репозитория (после openlin_lines_ens):
    python -m experiments.openlin_lines_ens_map
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

from experiments.openlin_anfis import TOL_M, block_split
from experiments.openlin_lines_ens import CONSENSUS
from experiments.openlin_lines_map import SHORT, fat, robust
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")
OUT = Path("outputs/openlin_lines_ens_map.png")

#: Что показываем крупно: (обучение, модель, таргет, площадка).
SHOW = ("nevada", "логрегрессия", "smooth", "anabar")


def load_freq(train: str, model: str, target: str, site: str) -> np.ndarray:
    p = M / f"openlin_lines_ens_{train}_{model}_{target}_{site}.npz"
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала прогон openlin_lines_ens")
    return np.load(p)["freq"]


def load_single(train: str, model: str, target: str, site: str) -> np.ndarray:
    p = M / f"openlin_lines_{train}_{model}_{target}_1_{site}.npz"
    return np.load(p)["pred"] if p.exists() else np.zeros((1, 1), dtype=bool)


def table(prefix: str) -> pd.DataFrame:
    frames = []
    for t in ("nevada", "anabar", "anabar_dykes"):
        p = M / f"{prefix}_{t}.csv"
        if p.exists():
            f = pd.read_csv(p)
            f.insert(0, "обучение", SHORT[t])
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    train, model, target, site = SHOW
    z = load_site(site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    gray = robust(z["gray"])
    fault = (z["dist"] <= res_m * 0.7) & valid

    freq = load_freq(train, model, target, site)
    cons = (freq >= CONSENSUS) & valid
    single = load_single(train, model, target, site)

    ens = table("openlin_lines_ens")
    row = ens[(ens["обучение"] == SHORT[train]) & (ens["модель"] == model)
              & (ens["таргет"] == target) & (ens["площадка"] == site)].iloc[0]

    fig = plt.figure(figsize=(16.5, 12.6))
    gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.30,
                          height_ratios=[1.0, 0.86])

    # 1. поле частоты
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gray, cmap="gray", alpha=0.5)
    im = ax.imshow(np.where(valid & (freq > 0), freq, np.nan), cmap="magma",
                   vmin=0, vmax=1)
    ax.set_title("частота линии по {} подвыборкам\n"
                 "жёлтое — линия была почти всегда".format(int(row["подвыборок"])),
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02)

    # 2. консенсус на карте
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(gray, cmap="gray", alpha=0.75)
    ax.imshow(np.ma.masked_where(~fault, np.zeros_like(gray)),
              cmap=ListedColormap(["#e8402a"]), alpha=0.75)
    ax.imshow(np.ma.masked_where(~fat(cons), np.zeros_like(gray)),
              cmap=ListedColormap(["#12a150"]), alpha=1.0)
    ax.set_title("консенсус (частота >= {:.0%}) и разломы по карте\n"
                 "точность {:.2f}, полнота {:.3f}"
                 .format(CONSENSUS, row["точность_50"], row["полнота_50"]),
                 fontsize=10)
    ax.legend(handles=[Line2D([], [], color="#12a150", lw=3, label="консенсус"),
                       Line2D([], [], color="#e8402a", lw=3, label="карта")],
              loc="lower left", fontsize=9, framealpha=0.9)

    # 3. одиночная трассировка для сравнения
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(gray, cmap="gray", alpha=0.75)
    ax.imshow(np.ma.masked_where(~fault, np.zeros_like(gray)),
              cmap=ListedColormap(["#e8402a"]), alpha=0.75)
    if single.shape == gray.shape:
        ax.imshow(np.ma.masked_where(~fat(single), np.zeros_like(gray)),
                  cmap=ListedColormap(["#1f66c1"]), alpha=1.0)
    ax.set_title("для сравнения: одиночная трассировка\n"
                 "(одна подвыборка, тот же вход)", fontsize=10)

    for a in fig.axes[:4]:
        a.set_xticks([])
        a.set_yticks([])

    # 4. воспроизводимость: одиночная против ансамблевой
    ax = fig.add_subplot(gs[1, 0])
    st = table("openlin_lines")
    sing = pd.read_csv(M / "openlin_lines_nevada_stability.csv")["совпадение"].dropna()
    pair = ens["совпадение_половин"].dropna()
    # Сравниваются одинаковые величины: взаимное покрытие двух списков линий.
    # Колонка «спирмен_половин» из этого прогона сюда не идёт — она считалась по
    # объединению ненулевых ячеек и на разреженном поле уходит в минус по
    # построению (в модуле это уже исправлено, но прогон был до правки).
    bp = ax.boxplot([sing.values, pair.values],
                    tick_labels=["одиночная трассировка\n(два списка линий)",
                                 "ансамбль\n(две половины, консенсус)"],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ("#e0a3a3", "#12a150")):
        patch.set_facecolor(c)
    ax.axhline(float(np.median(sing)), color="#b05555", ls="--", lw=1.2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("взаимное покрытие двух списков линий")
    ax.set_title("Воспроизводимость: помогло ли усреднение\n"
                 "(пунктир — медиана одиночной трассировки)", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # 5. точность консенсуса против случайной линии
    ax = fig.add_subplot(gs[1, 1:])
    base = {}
    for s in ("nevada", "anabar", "anabar_dykes"):
        zz = load_site(s)
        vv = np.isfinite(zz["gray"])
        _t, vaa = block_split(vv.shape, float(zz["res_m"]))
        base[SHORT[s]] = {
            "проверка": float(((zz["dist"] <= TOL_M) & vaa & vv).sum()
                              / (vaa & vv).sum()),
            "перенос": float(((zz["dist"] <= TOL_M) & vv).sum() / vv.sum())}
    d = ens.dropna(subset=["точность_50"]).copy()
    d["площадка"] = d["площадка"].map(SHORT)
    d["случайная"] = [base[r["площадка"]][r["режим"]] for _, r in d.iterrows()]
    d = d.sort_values("точность_50", ascending=False)
    pos = np.arange(len(d))
    ax.barh(pos, d["точность_50"], 0.72,
            color=np.where(d["режим"] == "проверка", "#12a150", "#8fb8de"),
            edgecolor="#333", linewidth=0.4)
    ax.plot(d["случайная"], pos, "k|", ms=13, mew=2, label="случайная линия")
    ax.set_yticks(pos)
    short_m = {"логрегрессия": "логрег", "бустинг": "буст", "ANFIS": "ANFIS"}
    ax.set_yticklabels([f"{r['площадка']}·{short_m[r['модель']]}·{r['таргет']}"
                        for _, r in d.iterrows()], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlabel("точность консенсусных линий (допуск 250 м)")
    ax.set_title("Все прогоны ансамбля: зелёное — своя проверочная часть, "
                 "синее — перенос", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Ансамбль трассировок: частота линии вместо одного списка",
                 fontsize=15, y=0.95)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"сохранено: {OUT}")


if __name__ == "__main__":
    main()
