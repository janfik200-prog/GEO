"""Сопоставление входов и выхода: какие значения признаков дают высокий балл.

ЗАЧЕМ. Разложение по Шепли показало, НА ЧТО модель опирается (рельеф 47%,
линеаменты 27%, спектры 11%) и в какую сторону. Теперь нужно увидеть это на
карте и в числах: как выглядят сами входные слои там, где выход высокий, и
совпадает ли рисунок входа с рисунком выхода. Это и есть паспорт перспективной
ячейки — он нужен геологу, чтобы судить о смысле прогноза, а не о метрике.

ЧТО СЧИТАЕТСЯ.
1. Карты: пять ведущих входных признаков и выходной балл в одних границах и
   одной сетке 500 м — визуальное сопоставление входа и выхода.
2. Числа: медиана каждого ведущего признака в верхних 10% площади и в фоне,
   отдельно по листу и юго-западу, в своих единицах, плюс отрыв в сигмах фона.
3. Согласованность рисунка: ранговая связь каждого входного слоя с выходом.

ВХОД: data/processed/wide_v1_archive/dataset_wide.parquet,
      terrain_drift_maps.parquet (выход, вариант C)
ВЫХОД: outputs/03_признаковое_пространство/input_output_match.png,
       outputs/metrics/input_output_match.log и .csv
Запуск из корня: python -X utf8 -m experiments.input_output_match
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, mapstyle, outputs  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
# ведущие признаки по разложению Шепли, с единицами и толкованием
LEAD = [
    ("dem_elev", "Высота, м", "м"),
    ("lin_dist", "Расстояние до линеамента, м", "м"),
    ("relief2_curv_5km", "Кривизна поверхности, окно 5 км", "усл."),
    ("dem_incision", "Врезание, м", "м"),
    ("s2_iron_ox", "Индекс окислов железа", "усл."),
]
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def main() -> None:
    shape = (ARCH_META.prf, ARCH_META.pic)
    wide = pd.read_parquet(config.PROCESSED_DIR / "wide_v1_archive"
                           / "dataset_wide.parquet")
    out = pd.read_parquet(config.PROCESSED_DIR / "terrain_drift_maps.parquet")
    x, y = out["x"].to_numpy(float), out["y"].to_numpy(float)
    common = out["в_области"].to_numpy(bool)
    score = out["C: рельеф по зонам"].to_numpy(float)
    zone = zone_of(x, y)
    in_sheet = common & (zone == "лист")
    sw = common & ~in_sheet

    rows = []
    say("СОПОСТАВЛЕНИЕ ВХОДОВ И ВЫХОДА")
    say("медианы в верхних 10%% площади и в фоне; отрыв — в сигмах фона")
    for col, title, unit in LEAD:
        v = wide[col].to_numpy(float)
        say("")
        say("%s (%s)" % (title, col))
        for tag, cells in (("лист", in_sheet), ("юго-запад", sw)):
            top = cells & (score >= np.nanquantile(score[cells], 1 - AREA))
            bg = cells & ~top
            mt, mb = np.nanmedian(v[top]), np.nanmedian(v[bg])
            sd = np.nanstd(v[bg]) or np.nan
            rho = sps.spearmanr(v[cells], score[cells],
                                nan_policy="omit").statistic
            say("   %-10s верхние 10%%: %9.1f %s, фон: %9.1f %s, отрыв %+.2f "
                "сигмы, связь с баллом %+.2f" % (tag, mt, unit, mb, unit,
                                                 (mt - mb) / sd, rho))
            rows.append({"признак": col, "площадь": tag, "верхние_10": mt,
                         "фон": mb, "отрыв_сигм": (mt - mb) / sd,
                         "связь_с_баллом": rho})
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "metrics"
                              / "input_output_match.csv", index=False,
                              encoding="utf-8-sig")

    # --- карты: пять входов и выход
    ext = (x.min() - 250, x.max() + 250, y.min() - 250, y.max() + 250)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.4), constrained_layout=True)
    panels = [(c, t, u) for c, t, u in LEAD] + [("__выход__", "Балл прогноза "
                                                 "(вариант с нормировкой "
                                                 "рельефа)", "доля площади")]
    for axi, (col, title, unit) in zip(axes.ravel(), panels):
        if col == "__выход__":
            v = np.where(common, score, np.nan)
            cuts = np.nanpercentile(v[common], mapstyle.CLASS_EDGES[1:-1])
            img = np.where(common, np.digitize(v, cuts).astype(float), np.nan)
            axi.imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                       origin="upper", interpolation="nearest",
                       vmin=-0.5, vmax=4.5,
                       cmap=matplotlib.colors.ListedColormap(
                           mapstyle.CLASS_COLORS))
        else:
            v = np.where(common, wide[col].to_numpy(float), np.nan)
            lo, hi = np.nanpercentile(v[common], [2, 98])
            im = axi.imshow(np.ma.masked_invalid(v.reshape(shape)), extent=ext,
                            origin="upper", interpolation="nearest",
                            cmap="cividis", vmin=lo, vmax=hi)
            fig.colorbar(im, ax=axi, shrink=0.62, label=unit)
        axi.set_title(title, fontsize=11)
        axi.set_xlabel("X, км")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda v_, _: "%.0f" % (v_ / 1000))
        axi.yaxis.set_major_formatter(lambda v_, _: "%.0f" % (v_ / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Входные слои и выходной балл на одной площади", fontsize=13)
    path = outputs.png("input_output_match.png")
    mapstyle.save_fig(fig, path, dpi=150)
    say("")
    say("карта: %s" % path)
    (ROOT / "outputs" / "metrics" / "input_output_match.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
