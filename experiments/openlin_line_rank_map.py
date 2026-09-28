"""Картинка ранжирования линий: уверенность, перенесённая с другой площадки.

Слева — линии Анабара, раскрашенные предсказанной уверенностью: модель обучена
на Неваде и Анабар не видела. В середине — ROC переноса против планок. Справа —
правила ANFIS: чем настоящая линия отличается от случайной, словами.

Запуск из корня репозитория (после openlin_line_rank):
    python -m experiments.openlin_line_rank_map
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize

from experiments.openlin_anfis import SEED, anfis_predict, fit_anfis
from experiments.openlin_line_rank import FEATURES, auc, line_table
from experiments.openlin_lines_map import robust
from experiments.openlin_unet import load_site

M = Path("outputs/metrics")

#: Последовательная шкала для уверенности: одна плавная ветвь от тёмного к
#: светлому, без смены цвета — величина читается светлотой.
VIRIDIS = ["#3b2a70", "#2a788e", "#22a884", "#7ad151", "#fde725"]
C_ANFIS, C_LOGREG, C_BOOST, C_LEN = "#2a78d6", "#eb6834", "#c98b3a", "#8a8a8a"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SHORT = {"anabar": "Анабар", "nevada": "Невада"}
WHERE = {"anabar": "Анабаре", "nevada": "Неваде"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="nevada")
    ap.add_argument("--test", default="anabar")
    ap.add_argument("--density", default="предельно")
    a = ap.parse_args()

    tr = line_table(a.train, a.density)
    te, coords, valid, res_m = line_table(a.test, a.density, with_coords=True)
    xtr = tr[list(FEATURES)].to_numpy("float32")
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr = (xtr - mu) / sd
    xte = (te[list(FEATURES)].to_numpy("float32") - mu) / sd
    ytr = tr["подтверждена"].to_numpy("float32")
    yte = te["подтверждена"].to_numpy()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    anfis = fit_anfis(xtr, ytr, 8, soft=False, log=lambda *_: None)
    lr = LogisticRegression(max_iter=1000,
                            class_weight="balanced").fit(xtr, ytr > 0.5)
    gb = HistGradientBoostingClassifier(random_state=SEED, max_iter=200,
                                        class_weight="balanced").fit(xtr, ytr > 0.5)
    scores = {"ANFIS": anfis_predict(anfis, xte),
              "логрегрессия": lr.predict_proba(xte)[:, 1],
              "бустинг": gb.predict_proba(xte)[:, 1],
              "только длина": te["длина_км"].to_numpy()}

    z = load_site(a.test)
    gray = robust(z["gray"])
    conf = np.full(valid.shape, -1.0, dtype="float32")
    s_anfis = scores["ANFIS"]
    rank = s_anfis.argsort().argsort() / max(len(s_anfis) - 1, 1)
    for pts, v in zip(coords, rank):
        conf[pts[:, 0], pts[:, 1]] = v

    # Утолщение с сохранением значения: обычная дилатация маски потеряла бы
    # раскраску, поэтому берётся максимум по окну.
    from scipy.ndimage import grey_dilation

    thick = grey_dilation(conf, size=(3, 3))
    thick = np.where(thick >= 0, thick, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 0.8, 0.95]})
    axes[0].imshow(gray, cmap="gray", alpha=0.8)
    cmap = LinearSegmentedColormap.from_list("v", VIRIDIS)
    im = axes[0].imshow(np.ma.masked_invalid(thick), cmap=cmap,
                        norm=Normalize(0, 1))
    axes[0].set_title("линии {}, раскрашенные уверенностью\n"
                      "модель обучена на площадке {} и эту не видела"
                      .format(SHORT.get(a.test, a.test),
                              SHORT.get(a.train, a.train)),
                      fontsize=10.5, color=INK)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.040, pad=0.02)
    cb.set_label("ранг уверенности", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("{:.0f} x {:.0f} км".format(
        valid.shape[1] * res_m / 1000, valid.shape[0] * res_m / 1000),
        fontsize=9, color=INK2)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    for name, color in (("ANFIS", C_ANFIS), ("логрегрессия", C_LOGREG),
                        ("бустинг", C_BOOST), ("только длина", C_LEN)):
        s = scores[name]
        o = np.argsort(-s)
        tp = np.cumsum(yte[o]) / max(yte.sum(), 1)
        fp = np.cumsum(1 - yte[o]) / max((1 - yte).sum(), 1)
        ax.plot(np.r_[0, fp], np.r_[0, tp], color=color, lw=2,
                label=f"{name} — AUC {auc(yte, s):.3f}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("доля ложных срабатываний", fontsize=9, color=INK2)
    ax.set_ylabel("доля найденных подтверждённых", fontsize=9, color=INK2)
    ax.set_title("Перенос {} → {}\nпунктир — случайный порядок"
                 .format(SHORT.get(a.train, a.train), SHORT.get(a.test, a.test)),
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    ax.grid(alpha=0.2, color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)

    ax = axes[2]
    ax.set_facecolor(SURFACE)
    with torch.no_grad():
        sg = torch.exp(anfis.log_s).clamp_min(1e-3)
        d = (torch.from_numpy(xtr)[:, None, :] - anfis.c[None]) / sg[None]
        w = torch.softmax(-0.5 * (d * d).sum(-1), dim=1).numpy()
    win = w.argmax(1)
    base = float((ytr > 0.5).mean())
    c = anfis.c.detach().numpy()
    rec = []
    for k in range(c.shape[0]):
        m = win == k
        if m.sum() < 10:
            continue
        o = np.argsort(-np.abs(c[k]))
        cond = " И ".join(
            f"{FEATURES[i].replace('_', ' ')} "
            f"{'выс.' if c[k][i] > 0 else 'низ.'}"
            for i in o[:2] if abs(c[k][i]) >= 0.5)
        rec.append((float((ytr[m] > 0.5).mean()) / base, cond or "(нет условий)"))
    rec.sort(key=lambda t: t[0])
    y = np.arange(len(rec))
    ax.barh(y, [r[0] for r in rec], 0.7,
            color=[C_ANFIS if r[0] >= 1 else "#d8d7d3" for r in rec],
            edgecolor="#333", linewidth=0.4)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rec], fontsize=7.6, color=INK2)
    ax.set_xlabel("во сколько раз чаще подтверждается", fontsize=9, color=INK2)
    ax.set_title("Правила ANFIS: чем настоящая линия\nотличается от случайной "
                 "(обучение, {})".format(SHORT.get(a.train, a.train)),
                 fontsize=10.5, color=INK)
    ax.grid(axis="x", alpha=0.2, color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)

    fig.suptitle("Ранжирование линий по геометрии: единственное, что "
                 "перенеслось между площадками", fontsize=13.5, color=INK, y=0.99)
    out = Path(f"outputs/openlin_line_rank_{a.train}_to_{a.test}.png")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
