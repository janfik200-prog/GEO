"""Наше воссоздание критериальной формулы — на всю область расчёта.

ЗАЧЕМ. Воссоздание метода ГИС Интегро на Python (`src/features.py`, мера Плюты)
до сих пор считалось только на основном листе: шесть факторных слоёв формулы
были оцифрованы в границах листа и за рамкой их просто не было. С присланной
24.09.2026 сборкой `ПРОГНОЗ_ВОСПРОИЗВ` они появились на всю область расчёта
(7557 кв.км, ~150 км южнее листа), поэтому формулу можно посчитать самим и
сравнить с тем, что посчитала организация (`by_crt`). Это проверка точности
воссоздания вне листа — то, чего раньше сделать было нельзя.

СООТВЕТСТВИЕ СЛОЁВ (установлено корреляцией с факторными столбцами листа
`dataset_v5_rebuilt.parquet`, все значения выше 0.99):
  тектонический СЗ  tect1  <- NW_razl        (0.997)
  тектонический СВ  tect2  <- NE_razl        (0.996)
  литолого-фациальный facies <- fazii_dist   (0.999)
  палеогеоморфологический paleo <- paleo_dist (0.999)
  структурно-литологический struct <- kory_dist (0.996)
  магматогенный     magm   <- dayki_all_dist (0.993; `dayki_dist` даёт 0.939,
                                              то есть в формулу идут все дайки)
Слой `gran_dist` (гранитоиды) в шестёрку формулы не входит — он проверяется
отдельно как возможный седьмой критерий.

ЧТО СЧИТАЕТСЯ.
1. Формула как она есть: степенная трансформация каждого расстояния ->
   min-max в [0,1] -> взвешенное L1-расстояние до эталона-минимума (веса
   `config.TAXONOMY_WEIGHTS`). Меньше = перспективнее, как в ГИС Интегро.
2. Две области нормировки, потому что min-max зависит от того, по какой
   площади он взят:
   * «по всей области» — так считал бы сам ГИС Интегро на многолистовой сетке;
   * «по листу» — пороги берутся с листа и применяются наружу без пересчёта,
     то есть перенос без подстройки под новую территорию.
3. Сверка: наше воссоздание против нативного `prognoz` листа (контроль, что
   слои сопоставлены верно) и против `by_crt` организации на трёх площадях —
   вся область расчёта, лист, смежная часть.
4. Проверка состава формулы: веса критериев восстанавливаются регрессией
   `by_crt` на нормированные критерии (тем же способом, каким они были
   восстановлены на листе). Если веса совпали с `config.TAXONOMY_WEIGHTS` —
   организация считала ту же формулу; отдельно проверяется, добавляет ли
   седьмой критерий `gran_dist` объяснение.

ВХОД: data/SBORKA_NESKOLKO_LISTOV/ПРОГНОЗ_ВОСПРОИЗВ/PGRID/
ВЫХОД: outputs/02_критериальный_метод/criterial_reconstruct_multisheet.png,
       outputs/metrics/criterial_reconstruct_multisheet.log,
       data/processed/criterial_reconstruct_multisheet.parquet
Запуск из корня: ``python -m experiments.criterial_reconstruct_multisheet``
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, features, mapstyle, outputs  # noqa: E402
from src.integro_grid import read_pgrid, read_property  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402
from experiments.criterial_multisheet_compare import PGRID_DIR, SOUTH, zone_of  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
ROLE_LAYER = {
    "tect1": "NW_razl",         # тектонический северо-западный
    "tect2": "NE_razl",         # тектонический северо-восточный
    "facies": "fazii_dist",     # литолого-фациальный
    "paleo": "paleo_dist",      # палеогеоморфологический
    "struct": "kory_dist",      # структурно-литологический (коры выветривания)
    "magm": "dayki_all_dist",   # магматогенный (дайки)
}
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def load_layers() -> tuple:
    """Факторные слои, маска расчёта, балл организации и центры ячеек."""
    pgrid = next(PGRID_DIR.glob("*.pgrid"))
    meta = read_pgrid(pgrid)

    def prop(name: str) -> np.ndarray:
        path = PGRID_DIR / f"{pgrid.stem}.{name}.property"
        return read_property(path, meta, meta.property_by_name(name)).astype(float).ravel()

    layers = {role: prop(name) for role, name in ROLE_LAYER.items()}
    layers["gran"] = prop("gran_dist")
    msk = prop("maska_progn").astype(bool)
    by_crt = prop("by_crt")
    x, y = meta.cell_centers()
    return meta, layers, msk, by_crt, x.ravel(), y.ravel()


def criteria(layers: dict, norm: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Критерии в [0,1]: трансформация и min-max по ячейкам `norm`.

    Эталон = минимум (0), как в ГИС Интегро: расстояние 0 — перспективно.
    """
    cols, names = [], []
    for role, kind in config.TAXONOMY_TRANSFORMS.items():
        t = features._criterion_transform(layers[role], kind)
        lo, hi = float(np.nanmin(t[norm])), float(np.nanmax(t[norm]))
        cols.append((t - lo) / (hi - lo) if hi > lo else np.zeros_like(t))
        names.append(role)
    return np.column_stack(cols), names


def formula(layers: dict, norm: np.ndarray) -> np.ndarray:
    """Взвешенное L1-расстояние до эталона (меньше = перспективнее)."""
    z, names = criteria(layers, norm)
    w = np.array([config.TAXONOMY_WEIGHTS[n] for n in names], dtype=float)
    return (np.abs(z) * w).sum(axis=1)


def agree(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray,
          cells: np.ndarray) -> dict:
    """Согласие двух карт перспективности (больше = лучше) на площади."""
    u, v = a[cells], b[cells]
    D = poly_design(x[cells], y[cells])
    ru = u - D @ np.linalg.lstsq(D, u, rcond=None)[0]
    rv = v - D @ np.linalg.lstsq(D, v, rcond=None)[0]
    tu, tv = u >= np.quantile(u, 1 - AREA), v >= np.quantile(v, 1 - AREA)
    tru, trv = ru >= np.quantile(ru, 1 - AREA), rv >= np.quantile(rv, 1 - AREA)
    return {"ячеек": int(cells.sum()),
            "совпадение": float((tu & tv).sum() / tv.sum()),
            "без_наклона": float((tru & trv).sum() / trv.sum()),
            "ранговая": float(sps.spearmanr(u, v).statistic)}


def panel(ax, img, ext, title, rect) -> None:
    cmap = ListedColormap(mapstyle.CLASS_COLORS)
    cmap.set_bad("#ffffff")
    ax.imshow(np.ma.masked_invalid(img), origin="upper", extent=ext, cmap=cmap,
              norm=BoundaryNorm(np.arange(-0.5, 5.5), 5), interpolation="nearest")
    x0, y0, w, h = rect
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#101010", lw=1.2,
                           ls="--"))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
    ax.set_ylabel("Y, км")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.set_aspect("equal")


def main() -> None:
    meta, layers, msk, by_crt, x, y = load_layers()
    zone = zone_of(x, y)
    in_sheet = zone == "лист"
    say(f"Область расчёта: {int(msk.sum())} ячеек ({msk.sum() * 0.25:.0f} кв.км); "
        f"лист {int((msk & in_sheet).sum())}, смежная "
        f"{int((msk & ~in_sheet).sum())}, из них юг {int((msk & (zone == SOUTH)).sum())}")
    for role, name in ROLE_LAYER.items():
        v = layers[role][msk]
        say(f"  {role:<7} <- {name:<15} среднее {np.nanmean(v):8.0f} м, "
            f"пропусков {int((~np.isfinite(v)).sum())}")

    # --- две области нормировки
    ours_all = formula(layers, msk)                 # min-max по всей области
    ours_sheet = formula(layers, msk & in_sheet)    # пороги с листа, без пересчёта
    theirs = by_crt

    # --- контроль: наше воссоздание против нативного prognoz листа
    meta_s, prognoz = criterial_target.load_prognoz_grid()
    sx, sy = meta_s.cell_centers()
    tree = cKDTree(np.c_[x[msk], y[msk]])
    d, i = tree.query(np.c_[sx.ravel(), sy.ravel()])
    ok = d <= 375.0
    nat = prognoz.ravel()[ok]
    say("\nКонтроль сопоставления слоёв (лист, %d ячеек):" % int(ok.sum()))
    for tag, v in (("нормировка по всей области", ours_all),
                   ("нормировка по листу", ours_sheet)):
        r = sps.spearmanr(v[msk][i][ok], nat).statistic
        say(f"  наше воссоздание vs нативный prognoz: ранговая {r:+.4f}  ({tag})")
    say("  балл организации by_crt vs нативный prognoz: ранговая %+.4f"
        % sps.spearmanr(theirs[msk][i][ok], nat).statistic)

    # --- согласие с расчётом организации
    areas = {"вся область расчёта": msk,
             "лист": msk & in_sheet,
             "вся смежная часть": msk & ~in_sheet,
             "  юго-запад": msk & (zone == "юго-запад"),
             "  юг": msk & (zone == SOUTH)}
    say("\nНаше воссоздание против расчёта организации (случайный уровень 0.10):")
    rows = []
    for tag, v in (("нормировка по всей области", ours_all),
                   ("нормировка по листу", ours_sheet)):
        say(f"  --- {tag} ---")
        for name, cells in areas.items():
            if cells.sum() < 200:
                continue
            m = agree(-v, -theirs, x, y, cells)   # минус: больше = перспективнее
            rows.append({"нормировка": tag, "площадь": name.strip(), **m})
            say(f"    {name:<22} ячеек {m['ячеек']:6d}, совпадение {m['совпадение']:.3f}, "
                f"без наклона {m['без_наклона']:.3f}, ранговая {m['ранговая']:+.4f}")

    # --- состав формулы: какие веса стоят за by_crt
    z, names = criteria(layers, msk)
    g = features._criterion_transform(layers["gran"], "sqrt")
    lo, hi = float(np.nanmin(g[msk])), float(np.nanmax(g[msk]))
    g = (g - lo) / (hi - lo)
    say("\nВеса критериев, восстановленные регрессией by_crt (лист -> вся область):")
    for tag, cells in (("лист", msk & in_sheet), ("вся область", msk)):
        A = z[cells]
        b = theirs[cells]
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        r2 = 1 - ((b - A @ beta) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        scale = beta.max() if beta.max() > 0 else 1.0
        say(f"  {tag}: R2 = {r2:.4f}; " + ", ".join(
            f"{n} {v / scale:.3f} (в формуле {config.TAXONOMY_WEIGHTS[n]:.3f})"
            for n, v in zip(names, beta)))
        A7 = np.column_stack([A, g[cells]])
        beta7, *_ = np.linalg.lstsq(A7, b, rcond=None)
        r27 = 1 - ((b - A7 @ beta7) ** 2).sum() / ((b - b.mean()) ** 2).sum()
        say(f"     с седьмым критерием (гранитоиды): R2 = {r27:.4f}, "
            f"его вес {beta7[-1] / (beta7[:-1].max() or 1):.3f}")

    # --- карта
    shape = (meta.prf, meta.pic)
    yy = y.reshape(shape)
    flip = yy[0, 0] < yy[-1, 0]

    def grid(v: np.ndarray) -> np.ndarray:
        out = np.where(msk, v, np.nan).reshape(shape)
        return out[::-1] if flip else out

    ext = (x.min() - 250, x.max() + 250, y.min() - 250, y.max() + 250)
    sxx = x[msk & in_sheet]
    syy = y[msk & in_sheet]
    rect = (sxx.min() - 250, syy.min() - 250,
            sxx.max() - sxx.min() + 500, syy.max() - syy.min() + 500)

    def classes(v: np.ndarray) -> np.ndarray:
        ref = v[msk]
        cuts = np.percentile(ref, mapstyle.CLASS_EDGES[1:-1])
        out = np.digitize(v, cuts).astype(float)
        return grid(out)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 9.8), constrained_layout=True)
    panel(axes[0], classes(-ours_all), ext,
          "Наше воссоздание формулы (Python, мера Плюты)", rect)
    panel(axes[1], classes(-theirs), ext,
          "Расчёт организации (ГИС Интегро, by_crt)", rect)

    top_o = np.zeros(x.size)
    top_t = np.zeros(x.size)
    top_o[msk] = (-ours_all)[msk] >= np.quantile((-ours_all)[msk], 1 - AREA)
    top_t[msk] = (-theirs)[msk] >= np.quantile((-theirs)[msk], 1 - AREA)
    both = np.where(msk, top_o + 2 * top_t, np.nan)
    colors = ("#f3f3f3", "#4a6fb5", "#f2b544", "#a3122b")
    labels = ("ни один расчёт", "только наше воссоздание",
              "только расчёт организации", "оба расчёта")
    cmap2 = ListedColormap(colors)
    cmap2.set_bad("#ffffff")
    axes[2].imshow(np.ma.masked_invalid(grid(both)), origin="upper", extent=ext,
                   cmap=cmap2, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
                   interpolation="nearest")
    x0, y0, w, h = rect
    axes[2].add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#101010",
                                lw=1.2, ls="--"))
    axes[2].set_title("Верхние 10% площади: согласие двух расчётов", fontsize=12)
    axes[2].set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
    axes[2].set_ylabel("Y, км")
    axes[2].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    axes[2].yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    axes[2].set_aspect("equal")
    axes[2].legend(handles=[Patch(facecolor=c, edgecolor="#777777", label=l)
                            for c, l in zip(colors, labels)],
                   loc="lower left", fontsize=9, framealpha=0.95)

    handles = [Patch(facecolor=c, edgecolor="#777777", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    handles.append(plt.Line2D([], [], color="#101010", ls="--",
                              label="габариты основного листа"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False,
               fontsize=10, title="Классы левой и средней панели")
    fig.suptitle("Критериальная формула на всей области расчёта: наше "
                 "воссоздание и расчёт организации", fontsize=13)
    path = outputs.png("criterial_reconstruct_multisheet.png")
    mapstyle.save_fig(fig, path, dpi=180)
    say(f"\nКарта: {path}")

    out = pd.DataFrame({"x": x, "y": y, "в_маске": msk,
                        "в_габаритах_листа": in_sheet,
                        "воссоздание_вся_область": ours_all,
                        "воссоздание_пороги_листа": ours_sheet,
                        "by_crt": theirs})
    out.to_parquet(config.PROCESSED_DIR / "criterial_reconstruct_multisheet.parquet",
                   index=False)
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics"
                              / "criterial_reconstruct_multisheet.csv",
                              index=False, encoding="utf-8-sig")
    log_path = ROOT / "outputs" / "metrics" / "criterial_reconstruct_multisheet.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(LOG), encoding="utf-8")
    say(f"Числа: {log_path}")


if __name__ == "__main__":
    main()
