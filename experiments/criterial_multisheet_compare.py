"""Критериальный прогноз на несколько листов против нашего прогноза переноса.

ЗАЧЕМ. До 24.09.2026 критериальный расчёт существовал только на основном листе
R-48-XI,XII (154x149, 22.9 тыс. ячеек). Сравнивать наш прогноз переноса со
старым методом за рамкой листа было не с чем: там была только наша карта.
Организация пересчитала критериальную формулу на сетку из нескольких листов
(`прогноз_воспроизв.pgrid`, свойство `by_crt`, 230x452, шаг 500 м), и площадь
расчёта уходит примерно на 150 км южнее рамки основного листа. Здесь обе карты
впервые кладутся рядом на одну сетку и считается согласие — отдельно на всей
общей площади и отдельно на той её части, что лежит ВНЕ габаритов основного
листа, то есть там, где наша модель работает переносом, а не по памяти.

ЧТО СЧИТАЕТСЯ. Две метрики протокола: доля совпадения верхних 10% как есть и
после снятия координатного тренда (полином 2-й степени от X и Y снимается с
каждой карты отдельно — иначе обе карты «совпадают» просто потому, что обе
растут в одну сторону). Плюс ранговая корреляция непрерывных баллов.

ПЛОЩАДИ СРАВНЕНИЯ. Сравнение идёт по ВСЕЙ области критериального расчёта
(30 229 ячеек, 7557 кв.км): критериальный выход есть и южнее листа, и
территории должны быть сопоставимы. Чтобы накрыть её целиком, наша карта
берётся на архивной сетке (`forecast_arch_ядро5км.parquet`, 231 x 452,
X 499750-614750, Y 7694750-7920250) — текущая широкая сетка обрывается на
Y 7769250 и не достаёт 897 кв.км на юге. Цена — сборка признаков там прежней
версии, согласие на листе выходит ниже примерно на 0.05, это учтено в логе.
Смежная часть дополнительно делится на квадранты по углу основного листа
(X 536750, Y 7843750): юго-запад, юг и запад. Южный квадрант показывается
отдельной строкой — по заключению геолога-эксперта организации (24.09.2026)
там фундамент выходит из-под чехла (89.5% площади против 3-7% на листе и
юго-западе), обстановка другая. Западный квадрант критериальным расчётом
практически не покрыт (90 ячеек, 22 кв.км), поэтому отдельно не разбирается.

ПОЛЯРНОСТЬ. У критериального балла шкала обратная — как и у `prognoz` на листе,
МЕНЬШЕ значит перспективнее (проверено 24.09.2026: на критериальных объектах
листа `by_crt` медиана 0.110, на фоне 0.189; ранговая корреляция `by_crt` со
старым `prognoz` на перекрытии +0.70). Поэтому перспективность считается как
минус `by_crt`, иначе карты выходят зеркальными.

ВХОД:
  data/SBORKA_NESKOLKO_LISTOV/ПРОГНОЗ_ВОСПРОИЗВ/PGRID/прогноз_воспроизв.pgrid
  data/processed/forecast_arch_ядро5км.parquet (см. `critmulti_detrend_target`)
ВЫХОД: outputs/07_перенос_на_смежные/transfer_map_против_критериального.png,
       outputs/metrics/critmulti_compare.log
Запуск из корня: ``python -m experiments.criterial_multisheet_compare``
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mapstyle, outputs  # noqa: E402
from src.integro_grid import read_pgrid, read_property  # noqa: E402
from experiments.honest_feature_signal import poly_design  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PGRID_DIR = ROOT / "data" / "SBORKA_NESKOLKO_LISTOV" / "ПРОГНОЗ_ВОСПРОИЗВ" / "PGRID"
WIDE_PARQUET = ROOT / "data" / "processed" / "forecast_arch_ядро5км.parquet"
SCORE_COL = "порог верх 10%, сам балл (опора)"   # рабочая форма цели
AREA = 0.10          # верхние 10% — рабочая доля площади протокола
STEP = 500.0
SHEET_X0 = 536750.0  # юго-западный угол основного листа, центр ячейки
SHEET_Y0 = 7843750.0
SOUTH = "юг"          # фундамент выходит из-под чехла, обстановка другая
EXCLUDED = SOUTH      # имя сохранено для скриптов, которые его импортируют


def zone_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Квадрант относительно основного листа: лист / запад / юго-запад / юг."""
    west = np.asarray(x, dtype=float) < SHEET_X0
    south = np.asarray(y, dtype=float) < SHEET_Y0
    return np.where(west & ~south, "запад",
                    np.where(west & south, "юго-запад",
                             np.where(south, SOUTH, "лист")))


def load_criterial() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Критериальный балл (сырой, меньше = перспективнее), маска и центры ячеек."""
    pgrid = next(PGRID_DIR.glob("*.pgrid"))
    meta = read_pgrid(pgrid)

    def prop(name: str) -> np.ndarray:
        path = PGRID_DIR / f"{pgrid.stem}.{name}.property"
        return read_property(path, meta, meta.property_by_name(name))

    crt = prop("by_crt").astype(float)
    msk = prop("maska_progn").astype(bool)
    x, y = meta.cell_centers()
    return crt, msk, x, y


def detrend(v: np.ndarray, cells: np.ndarray, D: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(D[cells], v[cells], rcond=None)
    return v - D @ beta


def compare(ours: np.ndarray, crit: np.ndarray, cells: np.ndarray,
            D: np.ndarray) -> dict[str, float]:
    """Согласие двух карт на наборе ячеек: как есть и после снятия наклона."""
    a, b = ours[cells], crit[cells]
    ta = a >= np.quantile(a, 1 - AREA)
    tb = b >= np.quantile(b, 1 - AREA)
    da = detrend(ours, cells, D)[cells]
    db = detrend(crit, cells, D)[cells]
    tda = da >= np.quantile(da, 1 - AREA)
    tdb = db >= np.quantile(db, 1 - AREA)
    rho, _ = sps.spearmanr(a, b)
    return {"ячеек": float(cells.sum()),
            "совпадение": float((ta & tb).sum() / max(tb.sum(), 1)),
            "совпадение_без_наклона": float((tda & tdb).sum() / max(tdb.sum(), 1)),
            "ранговая_корреляция": float(rho)}


def crosscheck_old(crt: np.ndarray, msk: np.ndarray, cx: np.ndarray,
                   cy: np.ndarray) -> dict[str, float]:
    """Новая критериальная карта против старой на листе — потолок сравнения.

    Формула та же, но пересчитана на другой сетке и на геологии нескольких
    листов, поэтому сама с собой она совпадает не полностью. Без этого числа
    согласие нашей карты не с чем соотнести.
    """
    from scipy.spatial import cKDTree

    from src import config
    from src.integro_grid import read_pgrid as _read_pgrid

    meta = _read_pgrid(config.GOLD_TARGET_PGRID)
    prop = config.CRIT_TARGET_PROPERTY
    path = config.GOLD_TARGET_PGRID.parent / f"{config.GOLD_TARGET_PGRID.stem}.{prop}.property"
    old = read_property(path, meta, meta.property_by_name(prop)).astype(float)
    ox, oy = meta.cell_centers()
    tree = cKDTree(np.c_[cx[msk], cy[msk]])
    dist, idx = tree.query(np.c_[ox.ravel(), oy.ravel()])
    new = np.where(dist <= STEP * 0.75, crt[msk][idx], np.nan)
    cells = np.isfinite(new)
    a = -old.ravel()                                   # перспективность
    b = np.where(cells, -new, np.nan)
    b = np.where(np.isfinite(b), b, np.nanmedian(b[cells]))
    return compare(a, b, cells, poly_design(ox.ravel(), oy.ravel()))


def classes(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Номер класса 0..4 по границам процентилей, взятым с общей площади."""
    cuts = np.percentile(ref, mapstyle.CLASS_EDGES[1:-1])
    out = np.digitize(v, cuts).astype(float)
    out[~np.isfinite(v)] = np.nan
    return out


def panel(ax, img, ext, title, sheet_rect) -> None:
    cmap = ListedColormap(mapstyle.CLASS_COLORS)
    cmap.set_bad("#ffffff")
    ax.imshow(np.ma.masked_invalid(img), origin="upper", extent=ext,
              cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 5.5), 5),
              interpolation="nearest")
    x0, y0, w, h = sheet_rect
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#101010",
                           lw=1.2, ls="--"))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
    ax.set_ylabel("Y, км")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.set_aspect("equal")


def main() -> None:
    crt, msk, cx, cy = load_criterial()
    fw = pd.read_parquet(WIDE_PARQUET)
    log_lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    # --- общий холст: объединение двух сеток, шаг 500 м, решётка нашей сетки
    x_min, x_max = float(fw.x.min()), float(fw.x.max())
    y_max = float(fw.y.max())
    y_low = min(float(cy[msk].min()), float(fw.y.min()))
    y_min = np.floor((y_low - y_max) / STEP) * STEP + y_max
    n_col = int(round((x_max - x_min) / STEP)) + 1
    n_row = int(round((y_max - y_min) / STEP)) + 1
    shape = (n_row, n_col)
    ext = (x_min - STEP / 2, x_max + STEP / 2, y_min - STEP / 2, y_max + STEP / 2)
    say(f"Холст: {n_row}x{n_col} ячеек, X {x_min:.0f}-{x_max:.0f}, "
        f"Y {y_min:.0f}-{y_max:.0f}")

    def place(values, xs, ys):
        out = np.full(shape, np.nan)
        r = np.round((y_max - np.asarray(ys, dtype=float)) / STEP).astype(int)
        c = np.round((np.asarray(xs, dtype=float) - x_min) / STEP).astype(int)
        ok = (r >= 0) & (r < n_row) & (c >= 0) & (c < n_col)
        out[r[ok], c[ok]] = np.asarray(values, dtype=float)[ok]
        return out

    ours = place(fw[SCORE_COL].to_numpy(), fw.x.to_numpy(), fw.y.to_numpy())
    sheet = place(fw["в_габаритах_листа"].to_numpy().astype(float),
                  fw.x.to_numpy(), fw.y.to_numpy())
    # минус — приведение к нашей полярности: больше значит перспективнее
    crit = place(-crt[msk], cx[msk], cy[msk])

    common = np.isfinite(ours) & np.isfinite(crit)
    # обе карты показываются и считаются на одной территории — области расчёта
    ours = np.where(common, ours, np.nan)
    crit = np.where(common, crit, np.nan)
    in_sheet = np.nan_to_num(sheet) > 0.5
    xx0, yy0 = np.meshgrid(np.linspace(x_min, x_max, n_col),
                           np.linspace(y_max, y_min, n_row))
    zone = zone_of(xx0, yy0)
    work = common                          # вся область критериального расчёта
    say(f"Критериальный расчёт: {int(msk.sum())} ячеек "
        f"({msk.sum() * 0.25:.0f} кв.км)")
    say(f"Наш прогноз переноса: {len(fw)} ячеек ({len(fw) * 0.25:.0f} кв.км)")
    say(f"Общая площадь двух карт: {int(common.sum())} ячеек "
        f"({common.sum() * 0.25:.0f} кв.км)")
    for z in ("лист", "юго-запад", SOUTH, "запад"):
        m = common & (zone == z)
        say(f"  {z:<16} {int(m.sum()):6d} ячеек ({m.sum() * 0.25:6.0f} кв.км)")
    say(f"Не накрыто нашей сеткой: {int(msk.sum()) - int(common.sum())} ячеек")

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_col),
                         np.linspace(y_max, y_min, n_row))
    D = poly_design(xx.ravel(), yy.ravel())
    o1 = np.where(np.isfinite(ours), ours, np.nanmedian(ours[work])).ravel()
    c1 = np.where(np.isfinite(crit), crit, np.nanmedian(crit[work])).ravel()

    for name, cells in (("вся область критериального расчёта", work.ravel()),
                        ("внутри габаритов листа", (common & in_sheet).ravel()),
                        ("вся смежная часть", (common & ~in_sheet).ravel()),
                        ("  юго-запад", (common & (zone == "юго-запад")).ravel()),
                        ("  юг (фундамент из-под чехла)",
                         (common & (zone == SOUTH)).ravel())):
        m = compare(o1, c1, cells, D)
        say(f"{name}: ячеек {m['ячеек']:.0f}, совпадение верхних 10% "
            f"{m['совпадение']:.3f}, без наклона {m['совпадение_без_наклона']:.3f}, "
            f"ранговая корреляция {m['ранговая_корреляция']:+.3f}")

    m = crosscheck_old(crt, msk, cx, cy)
    say(f"потолок: новая критериальная карта против старой на листе — ячеек "
        f"{m['ячеек']:.0f}, совпадение {m['совпадение']:.3f}, без наклона "
        f"{m['совпадение_без_наклона']:.3f}, ранговая корреляция "
        f"{m['ранговая_корреляция']:+.3f}")

    # --- две панели рядом
    cls_o = classes(ours, ours[work])
    cls_c = classes(crit, crit[work])
    sx = fw.loc[fw["в_габаритах_листа"], ["x", "y"]]
    rect = (sx.x.min() - 250, sx.y.min() - 250,
            sx.x.max() - sx.x.min() + 500, sx.y.max() - sx.y.min() + 500)

    # третья панель: где верхние 10% двух карт сходятся, а где расходятся
    top_o = np.full(shape, np.nan)
    top_c = np.full(shape, np.nan)
    top_o[work] = (ours[work] >= np.quantile(ours[work], 1 - AREA))
    top_c[work] = (crit[work] >= np.quantile(crit[work], 1 - AREA))
    both = np.full(shape, np.nan)
    both[work] = (np.nan_to_num(top_o)[work] * 1
                  + np.nan_to_num(top_c)[work] * 2)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 9.8), constrained_layout=True)
    panel(axes[0], cls_o, ext, "Наш прогноз (перенос, ядро 5 км)", rect)
    panel(axes[1], cls_c, ext, "Критериальный прогноз (ГИС Интегро)", rect)
    agree_colors = ("#f3f3f3", "#4a6fb5", "#f2b544", "#a3122b")
    agree_labels = ("ни один метод", "только наш прогноз",
                    "только критериальный", "оба метода")
    cmap2 = ListedColormap(agree_colors)
    cmap2.set_bad("#ffffff")
    axes[2].imshow(np.ma.masked_invalid(both), origin="upper", extent=ext,
                   cmap=cmap2, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
                   interpolation="nearest")
    x0, y0, w, h = rect
    axes[2].add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#101010",
                                lw=1.2, ls="--"))
    axes[2].set_title("Верхние 10% площади расчёта: согласие методов", fontsize=12)
    axes[2].set_xlabel("X, км (Гаусса-Крюгера, зона 18)")
    axes[2].set_ylabel("Y, км")
    axes[2].xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    axes[2].yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    axes[2].set_aspect("equal")
    axes[2].legend(handles=[Patch(facecolor=c, edgecolor="#777777", label=l)
                            for c, l in zip(agree_colors, agree_labels)],
                   loc="lower left", fontsize=9, framealpha=0.95)

    handles = [Patch(facecolor=c, edgecolor="#777777", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    handles.append(plt.Line2D([], [], color="#101010", ls="--",
                              label="габариты основного листа"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6,
               frameon=False, fontsize=10,
               title="Классы левой и средней панели")
    fig.suptitle("Перспективность по двум методам на общей сетке 500 м, "
                 "вся область критериального расчёта "
                 "(классы — процентили этой площади)", fontsize=13)
    path = outputs.png("transfer_map_против_критериального.png")
    mapstyle.save_fig(fig, path, dpi=180)
    say(f"Карта: {path}")

    log_path = ROOT / "outputs" / "metrics" / "critmulti_compare.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
