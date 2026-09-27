"""Портрет лучшего метода проекта: голосование трёх моделей по рангам.

ЗАЧЕМ. В сводном прогоне `supervised_sweep` голосование трёх сильнейших моделей
на признаках с окрестностью оказалось лучшей картой проекта (ROC-AUC 0.769 на
разрыве 15 км, совпадение с критериальной картой по верхним 10% — 33.4%), но на
общей картинке оно занимает одну панель среди прочих. Здесь метод разобран
отдельно: показаны карты всех трёх участников, их расхождение между собой,
итог голосования, эталон и то, где итог с эталоном сходится, а где расходится.

Ничего не переобучается: все карты берутся из
`data/processed/supervised_sweep_scores.parquet`, метрики — из
`outputs/metrics/supervised_sweep.csv` (тот же прогон, тот же протокол полос с
разрывом 15 км).

Выход: `outputs/vote_ensemble.png`.
Запуск из корня: ``python -m experiments.plot_vote_ensemble``.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats as sps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, config, criterial_target  # noqa: E402

CTX = "с окрестностью 111"
PARTS = [("случайный лес", "случайный лес"),
         ("гистограммный бустинг", "гистограммный бустинг"),
         ("перцептрон (ANN литературы)", "перцептрон (сеть 64-32)")]
VOTE = "голосование трёх сильнейших"
BASE = "ТОЛЬКО география (планка) | сырые 37"
OUT = ROOT / "outputs" / "vote_ensemble.png"


def rank_map(v: np.ndarray, valid: np.ndarray, shape) -> np.ndarray:
    """Скор -> процентиль внутри валидного пула, невалидные ячейки -> NaN."""
    out = np.full(v.size, np.nan)
    out[valid] = sps.rankdata(v[valid]) / valid.sum()
    return out.reshape(shape)


def panel(ax, img, title, sub, cmap="magma"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    df = pd.read_parquet(config.PROCESSED_DIR / "supervised_sweep_scores.parquet")
    res = pd.read_csv(ROOT / "outputs" / "metrics" / "supervised_sweep.csv")
    shape = (int(df["row"].max()) + 1, int(df["col"].max()) + 1)
    valid = df["valid"].to_numpy(dtype=bool)
    prognoz = df["prognoz"].to_numpy(dtype=float)

    met = {}
    for _, r in res.iterrows():
        met[f"{r['метод']} | {r['вход']}"] = r

    def sub(key):
        r = met[key]
        return (f"ROC-AUC {r['фолды_roc_auc']:.3f}"
                f"   совпадение верх-10% {r['совпадение_верх_10']:.0%}")

    # Ранги участников: голосование усредняет именно их, а не вероятности.
    parts_rank = {}
    for col, label in PARTS:
        parts_rank[label] = rank_map(df[f"{col} | {CTX}"].to_numpy(), valid, shape)
    vote = rank_map(df[f"{VOTE} | {CTX}"].to_numpy(), valid, shape)
    crit = rank_map(-prognoz, valid, shape)

    # Расхождение участников: разброс рангов в ячейке, ради чего и голосуют.
    stack = np.vstack([m.ravel()[valid] for m in parts_rank.values()])
    spread = np.full(valid.size, np.nan)
    spread[valid] = stack.std(axis=0)
    spread_n = (spread / np.nanmax(spread)).reshape(shape)

    # Совпадение верхних 10% площади с критериальной картой.
    top_c = prognoz <= np.quantile(prognoz[valid], config.CREF_AREA)
    sv = df[f"{VOTE} | {CTX}"].to_numpy()
    top_m = sv >= np.quantile(sv[valid], 1 - config.CREF_AREA)
    cat = np.where(top_c & top_m, 3, np.where(top_m, 2, np.where(top_c, 1, 0))).astype(float)
    cat[~valid] = np.nan
    cat = cat.reshape(shape)

    meta, _ = criterial_target.load_prognoz_grid()
    pts = assessment.load_verification_points(meta)
    unb = assessment.unbiased_cells(pts)
    all_c = pts["cell"].to_numpy()
    bias = np.setdiff1d(all_c, unb)

    fig, axes = plt.subplots(2, 4, figsize=(20.5, 11.4))
    for ax, (label, img) in zip(axes[0, :3], parts_rank.items()):
        key = f"{dict((b, a) for a, b in PARTS)[label]} | {CTX}"
        panel(ax, img, label, sub(key))

    ax = axes[0, 3]
    ax.imshow(spread_n, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Расхождение трёх моделей", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.045, "разброс рангов в ячейке: жёлтое — модели спорят",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1, 0]
    panel(ax, vote, "ГОЛОСОВАНИЕ ПО РАНГАМ", sub(f"{VOTE} | {CTX}"))
    for cells, color, mark, size in ((bias, "#39d6ff", "o", 26),
                                     (unb, "#ffffff", "^", 46)):
        if len(cells):
            ax.scatter(cells % shape[1], cells // shape[1], s=size, marker=mark,
                       facecolors="none", edgecolors=color, linewidths=1.3)
    ax.legend(handles=[
        Line2D([], [], marker="^", color="none", markeredgecolor="#ffffff",
               markersize=9, label=f"непредвзятые точки заверки ({len(unb)})"),
        Line2D([], [], marker="o", color="none", markeredgecolor="#39d6ff",
               markersize=7, label=f"прочие точки ({len(bias)})")],
        loc="lower left", fontsize=8, framealpha=0.65)

    panel(axes[1, 1], crit, "Критериальный прогноз", "эталон геологов, ГИС Интегро")

    ax = axes[1, 2]
    cmap = ListedColormap(["#e9edf2", "#4f7fd4", "#e0a32b", "#c0392b"])
    ax.imshow(cat, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("Верхние 10% площади: где сходятся", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.045,
            f"совпадение {met[f'{VOTE} | {CTX}']['совпадение_верх_10']:.0%} "
            f"при случайном уровне {config.CREF_AREA:.0%}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor="#c0392b", label="оба метода"),
                       Patch(facecolor="#e0a32b", label="только голосование"),
                       Patch(facecolor="#4f7fd4", label="только критериальный")],
              loc="lower left", fontsize=8, framealpha=0.8)

    ax = axes[1, 3]
    bars = [(dict((b, a) for a, b in PARTS)[lbl] + f" | {CTX}", lbl.split(" (")[0])
            for lbl in parts_rank]
    bars += [(f"{VOTE} | {CTX}", "ГОЛОСОВАНИЕ"),
             (f"{VOTE} | сырые 37", "голосование, без окрестности"),
             (BASE, "только география (планка)")]
    names = [n for _, n in bars]
    roc = [met[k]["фолды_roc_auc"] for k, _ in bars]
    agr = [met[k]["совпадение_верх_10"] for k, _ in bars]
    ypos = np.arange(len(bars))[::-1]
    colors = ["#7f8c9b"] * 3 + ["#c0392b", "#e0a32b", "#4f7fd4"]
    ax.barh(ypos + 0.19, roc, height=0.36, color=colors, label="ROC-AUC на полосах")
    ax.barh(ypos - 0.19, agr, height=0.36, color=colors, alpha=0.5,
            label="совпадение верх-10%")
    for y, v in zip(ypos, roc):
        ax.text(v + 0.008, y + 0.19, f"{v:.3f}", va="center", fontsize=8)
    for y, v in zip(ypos, agr):
        ax.text(v + 0.008, y - 0.19, f"{v:.0%}", va="center", fontsize=8)
    ax.axvline(0.5, color="#555", ls=":", lw=1)
    ax.text(0.5, -0.6, "случайный\nROC-AUC", fontsize=7.5, ha="center", va="center",
            color="#555", bbox=dict(fc="white", ec="none", pad=1))
    ax.axvline(config.CREF_AREA, color="#555", ls="--", lw=1)
    ax.text(config.CREF_AREA, -0.6, "случайное\nсовпадение", fontsize=7.5,
            ha="center", va="center", color="#555",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-1.0, len(bars) - 0.4)
    ax.set_xlabel("доля")
    ax.set_title("Целое против частей", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Голосование трёх моделей по рангам — лучшая карта проекта\n"
                 "лес + бустинг + перцептрон на 37 признаках и их скользящих средних "
                 "в окнах 2.5 и 7.5 км; обучение не ближе 15 км от проверяемой полосы",
                 fontsize=14)
    fig.text(0.5, 0.012,
             "Усредняются РАНГИ, а не вероятности: у трёх семейств шкалы "
             "несопоставимы, и среднее вероятностей отдало бы вес самой уверенной "
             "модели, а не самой точной. "
             "Цвет карт — процентиль перспективности внутри листа (светлое выше).",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"OK: {OUT}")
    print(f"голосование: ROC-AUC {met[f'{VOTE} | {CTX}']['фолды_roc_auc']:.4f}, "
          f"совпадение {met[f'{VOTE} | {CTX}']['совпадение_верх_10']:.4f}; "
          f"участники: " + ", ".join(
              f"{lbl.split(' (')[0]} {met[dict((b, a) for a, b in PARTS)[lbl] + ' | ' + CTX]['фолды_roc_auc']:.4f}"
              for lbl in parts_rank))


if __name__ == "__main__":
    main()
