"""Две карты рядом на расширенной территории: наш прогноз и критериальный метод.

ЗАЧЕМ. Показать текущее состояние обеих линий на одной площади: слева —
последняя версия нашего прогноза (вариант C из `terrain_drift_fix.py`, рельеф
нормирован по зонам, из-за чего снят западный перекос), справа — последняя
версия критериального метода (наше воссоздание формулы на всей области
расчёта, эталон с 04.10.2026). Обе карты на архивной сетке 500 м, в одних
границах и в одной шкале классов по доле площади.

ОГОВОРКА. У прогнозной карты рельефные признаки нормированы внутри зоны (лист
и смежная площадь отдельно), поэтому балл сравним внутри своей зоны лучше, чем
между зонами: после нормировки смежная площадь перестала проигрывать листу
просто за более расчленённый рельеф и занимает примерно половину верхних 10%.

ВХОД: data/processed/terrain_drift_maps.parquet,
      data/processed/criterial_reconstruct_multisheet.parquet
ВЫХОД: outputs/07_перенос_на_смежные/forecast_vs_criterial_wide.png,
       outputs/metrics/forecast_vs_criterial_wide.log
Запуск из корня: python -m experiments.forecast_vs_criterial_wide
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, config, criterial_target, mapstyle, outputs  # noqa: E402
from experiments.criterial_multisheet_compare import zone_of  # noqa: E402
from experiments.criterial_reconstruct_multisheet import panel  # noqa: E402
from experiments.critmulti_detrend_target import ARCH_META  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OURS_COL = "C: рельеф по зонам"
AREA = 0.10
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def main() -> None:
    shape = (ARCH_META.prf, ARCH_META.pic)
    d = pd.read_parquet(config.PROCESSED_DIR / "terrain_drift_maps.parquet")
    x = d["x"].to_numpy(float)
    y = d["y"].to_numpy(float)
    common = d["в_области"].to_numpy(bool)
    in_sheet = d["в_габаритах_листа"].to_numpy(bool)
    ours = d[OURS_COL].to_numpy(float)

    ref = pd.read_parquet(config.PROCESSED_DIR
                          / "criterial_reconstruct_multisheet.parquet")
    rmsk = ref["в_маске"].to_numpy(bool)
    dist, idx = cKDTree(np.column_stack([ref["x"].to_numpy()[rmsk],
                                         ref["y"].to_numpy()[rmsk]])).query(
        np.column_stack([x, y]))
    crit = np.full(x.size, np.nan)
    crit[common] = -ref["воссоздание_вся_область"].to_numpy()[rmsk][idx[common]]

    zone = zone_of(x, y)
    say("площадь %d ячеек (%.0f кв.км): лист %d, юго-запад %d, юг %d, запад %d"
        % (common.sum(), 0.25 * common.sum(), (common & in_sheet).sum(),
           (common & (zone == "юго-запад")).sum(), (common & (zone == "юг")).sum(),
           (common & (zone == "запад")).sum()))
    say("ранговая корреляция двух карт: вся область %+.3f, лист %+.3f, "
        "смежная %+.3f"
        % (sps.spearmanr(ours[common], crit[common]).statistic,
           sps.spearmanr(ours[common & in_sheet], crit[common & in_sheet]).statistic,
           sps.spearmanr(ours[common & ~in_sheet],
                         crit[common & ~in_sheet]).statistic))

    # распределение верхних 10% по зонам
    for tag, v in (("наш прогноз", ours), ("критериальный", crit)):
        top = np.zeros(x.size, bool)
        top[common] = v[common] >= np.quantile(v[common], 1 - AREA)
        say("%-16s верхние 10%%: лист %.0f кв.км, юго-запад %.0f, юг %.0f"
            % (tag, 0.25 * (top & in_sheet).sum(),
               0.25 * (top & (zone == "юго-запад")).sum(),
               0.25 * (top & (zone == "юг")).sum()))

    # точки заверки
    meta, _ = criterial_target.load_prognoz_grid()
    pts = assessment.load_verification_points(meta)
    cx, cy = meta.cell_centers()
    pt_x = cx.ravel()[pts["cell"].to_numpy()]
    pt_y = cy.ravel()[pts["cell"].to_numpy()]

    def grid(v: np.ndarray) -> np.ndarray:
        ref_vals = v[common]
        cuts = np.percentile(ref_vals, mapstyle.CLASS_EDGES[1:-1])
        out = np.where(common, np.digitize(v, cuts).astype(float), np.nan)
        return out.reshape(shape)

    ext = (x.min() - 250, x.max() + 250, y.min() - 250, y.max() + 250)
    sx, sy = x[common & in_sheet], y[common & in_sheet]
    rect = (sx.min() - 250, sy.min() - 250,
            sx.max() - sx.min() + 500, sy.max() - sy.min() + 500)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 10.2), constrained_layout=True)
    panel(axes[0], grid(ours), ext,
          "Наш прогноз, последняя версия\n(обучение на листе, рельеф нормирован "
          "по зонам)", rect)
    panel(axes[1], grid(crit), ext,
          "Критериальный метод, последняя версия\n(наше воссоздание формулы на "
          "всей области)", rect)
    for axi in axes:
        axi.scatter(pt_x, pt_y, s=9, c="#101010", marker="^", zorder=3,
                    label="точки заверки")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)

    handles = [matplotlib.patches.Patch(facecolor=c, edgecolor="#777777", label=l)
               for c, l in zip(mapstyle.CLASS_COLORS, mapstyle.CLASS_LABELS)]
    handles.append(plt.Line2D([], [], color="#101010", ls="--",
                              label="габариты основного листа"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False,
               fontsize=10, title="Классы перспективности (доли площади всей "
                                  "области расчёта)")
    fig.suptitle("Расширенная территория, 7664 кв.км: прогноз и критериальный "
                 "метод", fontsize=13)
    path = outputs.png("forecast_vs_criterial_wide.png")
    mapstyle.save_fig(fig, path, dpi=180)
    say("карта: %s" % path)
    (ROOT / "outputs" / "metrics" / "forecast_vs_criterial_wide.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
