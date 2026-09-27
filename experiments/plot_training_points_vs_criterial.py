"""Точки обучения (позитивы критериального анализа) рядом с критериальным прогнозом.

Разметка — :func:`src.criterial_target.training_positives`: top-``CREF_AREA``
(10%) критериального прогноза, без группировки в связные компоненты и без
пространственного буфера (LOO-протокол закрыт 20.08.2026). Решение 01.09.2026:
обучающая разметка выровнена с эталоном согласия, по которому меряются все
методы; прежний абсолютный порог 0.15 давал в 5.2 раза меньше позитивов и
оставлял три перспективные зоны вообще без обучающих меток.

Для сравнения на карте показан и прежний порог 0.15 — видно, какие площади
раньше выпадали из обучения.

Точки заверки (73) здесь не участвуют — они независимая оценка карт
(``src/assessment.py``), не обучающая разметка.

Выход: ``outputs/training_points_vs_criterial.png``.
Запуск из корня репозитория: ``python -m experiments.plot_training_points_vs_criterial``.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target  # noqa: E402


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    flat = prognoz.ravel()

    pos_mask = criterial_target.training_positives(prognoz)
    pos_idx = np.flatnonzero(pos_mask)
    bg_candidates = np.flatnonzero(~pos_mask)
    thr_train = float(np.quantile(flat, config.CREF_AREA))

    # Прежняя разметка — только для сравнения на карте, в обучении не участвует.
    old_idx = np.flatnonzero(flat <= config.CRIT_TARGET_THRESHOLD)

    rng = np.random.default_rng(config.CRIT_SEED)
    n_bg = min(config.CRIT_N_BACKGROUND, len(bg_candidates))
    bg_idx = rng.choice(bg_candidates, size=n_bg, replace=False)

    print(f"позитивы обучения (top-{config.CREF_AREA:.0%}): {len(pos_idx)} ячеек "
          f"= {100 * len(pos_idx) / flat.size:.2f}% листа, prognoz <= {thr_train:.4f}")
    print(f"прежняя разметка (prognoz <= {config.CRIT_TARGET_THRESHOLD}): {len(old_idx)} ячеек")
    print(f"фон: {n_bg} случайных из {len(bg_candidates)} (seed={config.CRIT_SEED})")
    print(f"баланс классов: {len(pos_idx)} поз / {n_bg} фон = "
          f"{100 * len(pos_idx) / (len(pos_idx) + n_bg):.1f}% позитивов")

    def rc(idx):
        return idx // meta.pic, idx % meta.pic

    pos_r, pos_c = rc(pos_idx)
    bg_r, bg_c = rc(bg_idx)
    old_r, old_c = rc(old_idx)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, (ax_tr, ax_crit) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    ax_tr.set_facecolor("#f0ede5")
    ax_tr.set_xlim(0, meta.pic)
    ax_tr.set_ylim(meta.prf, 0)
    ax_tr.scatter(bg_c + 0.5, bg_r + 0.5, s=2, c="#9a9a9a", alpha=0.45,
                  label=f"фон обучения (n={n_bg}, случайная выборка)", zorder=2)
    ax_tr.scatter(pos_c + 0.5, pos_r + 0.5, s=6, c="#d7263d", marker="s", linewidths=0,
                  label=f"позитивы обучения: top-{config.CREF_AREA:.0%} "
                        f"(prognoz ≤ {thr_train:.3f}, n={len(pos_idx)})", zorder=3)
    ax_tr.scatter(old_c + 0.5, old_r + 0.5, s=4, c="#2a1a5e", marker="s", linewidths=0,
                  label=f"прежнее ядро: prognoz ≤ {config.CRIT_TARGET_THRESHOLD} "
                        f"(n={len(old_idx)})", zorder=4)
    ax_tr.set_title(f"Обучающая выборка: top-{config.CREF_AREA:.0%} критериального "
                    f"прогноза, без объектов и буфера")
    ax_tr.set_xlabel("столбец (x500 м)")
    ax_tr.set_ylabel("строка (x500 м)")
    ax_tr.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_tr.set_aspect("equal")

    # Сырые значения prognoz, НЕ ранг: ранговая раскраска (как в compare_methods
    # и atypicality_*) растягивает шкалу и делает «перспективной» на вид верхнюю
    # пятую часть листа, из-за чего порог визуально теряется.
    im = ax_crit.imshow(prognoz, cmap="magma_r")
    ax_crit.contour(prognoz, levels=[thr_train], colors="#00e5ff", linewidths=1.6)
    ax_crit.contour(prognoz, levels=[config.CRIT_TARGET_THRESHOLD],
                    colors="#39ff14", linewidths=1.0, linestyles="--")
    ax_crit.legend(handles=[
        Line2D([], [], color="#00e5ff", lw=1.6,
               label=f"граница обучения: top-{config.CREF_AREA:.0%} = {thr_train:.3f}"),
        Line2D([], [], color="#39ff14", lw=1.0, ls="--",
               label=f"прежний порог {config.CRIT_TARGET_THRESHOLD}"),
    ], loc="lower left", fontsize=8, framealpha=0.9)
    ax_crit.set_title("Критериальный прогноз, сырые значения prognoz\n"
                      "(меньше = перспективнее)")
    ax_crit.set_xlabel("столбец (x500 м)")
    fig.colorbar(im, ax=ax_crit, shrink=0.8, label="prognoz (нативная шкала)")

    fig.suptitle(f"Обучающая разметка = эталон согласия: {len(pos_idx)} позитивов "
                 f"(top-{config.CREF_AREA:.0%}), было {len(old_idx)} при пороге "
                 f"{config.CRIT_TARGET_THRESHOLD}")
    fig.tight_layout()
    out_path = ROOT / "outputs" / "training_points_vs_criterial.png"
    fig.savefig(out_path, dpi=140)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
