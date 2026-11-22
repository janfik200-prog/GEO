"""Пробная карта на знаниях рядом с критериальной картой ВНИГНИ — только для просмотра.

Карта ВНИГНИ здесь — заверка, а не мишень (решение 21.10.2026): ничего не
подгоняется, обе карты показаны в одной шкале — процентиль ранга по листу
(1 — самые перспективные ячейки). Шкала критериального прогноза обратная:
перспективно меньшее значение (`src/criterial_target.training_positives`),
поэтому её знак меняется. Третья панель — где совпадают верхние 20 %
площади.

Входные данные: `data/processed/knowledge_map_<версия>.parquet`
(`experiments/gmo_knowledge_map_v0.py`), критериальный прогноз
(`src/criterial_target.load_prognoz_grid`).

Запуск::

    python -X utf8 -m experiments.gmo_map_vs_vnigni v0_3a
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import criterial_target, mapstyle, outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TOP = 0.2


def main() -> None:
    ver = sys.argv[1] if len(sys.argv) > 1 else "v0_3a"
    m = pd.read_parquet(ROOT / f"data/processed/knowledge_map_{ver}.parquet")
    meta, prog = criterial_target.load_prognoz_grid()
    p = -np.asarray(prog, dtype=float).reshape(meta.prf, meta.pic)
    ours = np.full((meta.prf, meta.pic), np.nan)
    ours[m.row, m.col] = m.итог.rank(pct=True)
    vn = np.where(np.isfinite(p), p, np.nan)
    both = np.isfinite(ours) & np.isfinite(vn)
    vn_r = np.full_like(vn, np.nan)
    vn_r[both] = pd.Series(vn[both]).rank(pct=True).to_numpy()
    ours_r = np.full_like(ours, np.nan)
    ours_r[both] = pd.Series(ours[both]).rank(pct=True).to_numpy()
    rho = spearmanr(ours_r[both], vn_r[both]).statistic
    to, tv = ours_r >= 1 - TOP, vn_r >= 1 - TOP
    share = (to & tv)[both].sum() / to[both].sum()
    agree = np.full(ours.shape, np.nan)
    agree[both] = 0
    agree[both & to & ~tv] = 1
    agree[both & tv & ~to] = 2
    agree[both & to & tv] = 3
    print(f"ячеек на обеих картах {both.sum()}; ранговая корреляция {rho:+.2f}; "
          f"верхние {TOP:.0%} совпадают на {share:.0%} (случайно — {TOP:.0%})")

    fig, ax = plt.subplots(1, 3, figsize=(20, 7.2))
    for a, img, title in ((ax[0], ours_r, f"Наша карта на знаниях {ver.replace('_', '.')}\n"
                                          "(итог, процентиль по листу)"),
                          (ax[1], vn_r, "Критериальная карта ВНИГНИ\n(процентиль по листу)")):
        im = a.imshow(img, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        fig.colorbar(im, ax=a, shrink=0.8, label="процентиль: 1 — самые перспективные")
        a.set_title(title, fontsize=12)
    cm = ListedColormap(["#eeeeee", "#1f77b4", "#d62728", "#2ca02c"])
    ax[2].imshow(agree, cmap=cm, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax[2].set_title(f"Верхние {TOP:.0%} площади: где совпадают", fontsize=12)
    ax[2].legend(handles=[Patch(color="#2ca02c", label="в верхних у обеих"),
                          Patch(color="#1f77b4", label="только у нашей"),
                          Patch(color="#d62728", label="только у ВНИГНИ"),
                          Patch(color="#eeeeee", label="ни у одной")],
                 loc="lower left", fontsize=10, framealpha=0.95)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Лист R-48-XI,XII, сетка 500 м, север сверху. Ранговая корреляция {rho:+.2f}; "
                 f"верхние {TOP:.0%} совпадают на {share:.0%} (случайно — {TOP:.0%}). "
                 "Карта ВНИГНИ — заверка, не мишень", fontsize=12)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cm), ax=ax[2], shrink=0.8).remove()   # та же ширина панели
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = outputs.png(f"gmo_map_vs_vnigni_{ver}.png")
    mapstyle.save_fig(fig, out, dpi=120)
    print(f"Рисунок: {out}")


if __name__ == "__main__":
    main()
