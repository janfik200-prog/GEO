"""WP20 (ТЗ v3.0): четыре семьи нулевых проверок одной таблицей и одним рисунком.

Новых расчётов нет: всё читается из уже сохранённых файлов Пакета 05 и
прогонов XFER / XFER2.

* семья 1 — внутренний счётчик ветки (124 проверки), строки XFER и XFER2 из
  логов ``_zone_lines_*_v3_перенос.log``;
* семья 2 — GTK, 159 сдвигов, протокол 25.10, восстановленный 26.10 (``openlin_transfer_check_k1``);
* семья 3 — GTK, 5000 сдвигов (``outputs/k1/results/**/gtk_shifts.csv``);
* семья 4 — условные нули: 159 ближайших по воде и уклону; страты у воды /
  вдали от воды (только Лапландия, post hoc).

Запуск из корня:
    python -m experiments.openlin_release_nulls
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("outputs/k1/results")
FIG = Path("outputs/k1/figures")
M = Path("outputs/metrics")
SITES = {"lapland": RES, "finland65": RES / "finland65"}
SET_NAMES = {"C": "в зонах", "C_all": "весь лист"}


def transfer_check_rows(site):
    """Семья 2: две объявленные проверки (слой 292, 300 м, оба набора) исходного протокола."""
    p = M / f"openlin_transfer_check_{site}.csv"
    d = pd.read_csv(p)
    return d[(d["слой"] == 292) & (d["допуск_м"] == 300.0) & d["p"].notna()].to_dict("records")


def main() -> None:
    rows = []
    # семья 1 — по объявлениям в коде (порог) и логам (наблюдение)
    for site, run, n, alpha in (("lapland", "XFER", 118, 0.05 / 118), ("finland65", "XFER2", 124, 0.05 / 124)):
        rows.append({"family": "1 внутренний счётчик", "site": site, "test": f"{run}: 4 безэталонных условия",
                     "B": 5000, "seed": "см. openlin_zone_lines_v3", "statistic": "мусор / сдвиговый нуль / "
                     "устойчивость / длина", "alpha": round(alpha, 6), "correction": f"0.05/{n}",
                     "pre_post": "объявлено до прогона", "observed": "см. PACKET09_RESULTS / START_REPORT",
                     "p": 0.0002, "lift": None, "source": f"outputs/metrics/_zone_lines_{site}_v3_перенос.log"})
    # семьи 2–4 — GTK
    for site, base in SITES.items():
        js = json.loads((base / "gtk_spatial_stratified_exploratory.json").read_text(encoding="utf-8"))
        for r in transfer_check_rows(site):
            rows.append({"family": "2 GTK-159 (протокол 25.10)", "site": site,
                         "test": f"{r['набор']}, слой 292, 300 м", "B": 159, "seed": 2510,
                         "statistic": "полнота GTK у наших линий", "alpha": 0.025,
                         "correction": "своя: 2 проверки (0.05/2)", "pre_post": "объявлено до прогона "
                         "(повтор правила; исходные 159 сдвигов 25.10 — MISSING)",
                         "observed": round(r["полнота"], 4), "null_mean": round(r["нуль_ср"], 4),
                         "null_max": round(r["нуль_макс"], 4), "p": round(r["p"], 5),
                         "lift": round(r["подъём"], 3),
                         "source": f"outputs/metrics/openlin_transfer_check_{site}.csv"})
        for key, name in SET_NAMES.items():
            v = js["наборы"][key]["292_300"]
            rows.append({"family": "3 GTK-5000", "site": site, "test": f"{name}, слой 292, 300 м", "B": 5000,
                         "seed": 2511, "statistic": "полнота GTK у наших линий", "alpha": None,
                         "correction": "нет (разведочная)", "pre_post": "после первого просмотра GTK",
                         "observed": round(v["полнота"], 4), "null_mean": round(v["пул5000"]["ср"], 4),
                         "null_max": round(v["пул5000"]["макс"], 4), "k": v["пул5000"]["превышений"],
                         "p": round(v["пул5000"]["p"], 5), "lift": round(v["пул5000"]["подъём"], 3),
                         "source": f"{base.as_posix()}/gtk_shifts.csv"})
            t = v["по_местности159"]
            rows.append({"family": "4 условный (по местности)", "site": site, "test": f"{name}, слой 292, 300 м",
                         "B": 159, "seed": 2511, "statistic": "полнота; 159 сдвигов из 5000, ближайших по воде "
                         "и уклону", "alpha": None, "correction": "нет", "pre_post": "post hoc",
                         "observed": round(v["полнота"], 4), "null_mean": round(t["ср"], 4),
                         "null_max": round(t["макс"], 4), "k": t["превышений"], "p": round(t["p"], 5),
                         "lift": round(t["подъём"], 3), "source": f"{base.as_posix()}/gtk_spatial_stratified_exploratory.json"})
        loc = js["местность"]
        rows.append({"family": "4 условный (баланс местности)", "site": site, "test": "доля ячеек GTK у воды "
                     "(≤ 300 м от берега/болота OSM)", "B": 5000, "seed": 2511, "statistic": "у воды",
                     "observed": round(loc["наблюдение"][0], 4), "null_mean": round(loc["пул_ср"][0], 4),
                     "null_sd": round(loc["пул_sd"][0], 4), "matched_mean": round(loc["отобранные_ср"][0], 4),
                     "pre_post": "диагностика", "source": f"{base.as_posix()}/gtk_spatial_stratified_exploratory.json"})
    wp = RES / "gtk_water_stratified_posthoc.json"
    if wp.exists():
        for k, v in json.loads(wp.read_text(encoding="utf-8")).items():
            rows.append({"family": "4 условный (страты воды)", "site": "lapland", "test": k, "B": 5000,
                         "seed": 2511, "pre_post": "post hoc", "observed": json.dumps(v, ensure_ascii=False)[:300],
                         "source": wp.as_posix()})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "NULL_FAMILIES.csv", index=False, encoding="utf-8-sig")
    print(f"сохранено: {RES / 'NULL_FAMILIES.csv'} ({len(out)} строк)")
    plot()


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6), facecolor="white")
    for ri, (site, base) in enumerate(SITES.items()):
        sh = pd.read_csv(base / "gtk_shifts.csv")
        js = json.loads((base / "gtk_spatial_stratified_exploratory.json").read_text(encoding="utf-8"))
        for ci, (key, name) in enumerate(SET_NAMES.items()):
            ax = axs[ri, ci]
            s = sh[sh["набор"] == key]
            obs = js["наборы"][key]["292_300"]["полнота"]
            v = js["наборы"][key]["292_300"]
            bins = np.linspace(min(s["полнота"].min(), obs) * 0.95, max(s["полнота"].max(), obs) * 1.05, 50)
            ax.hist(s["полнота"], bins=bins, color="#9aa7b8", label="5000 сдвигов (семья 3, разведочная)")
            ax.hist(s.loc[s["по_местности"], "полнота"], bins=bins, color="#e08a2c", alpha=0.85,
                    label="159 по местности (семья 4, post hoc)")
            ax.axvline(obs, color="#c0392b", lw=2.2, label=f"наблюдение {100 * obs:.1f} %")
            ax.set_title(f"{site}, {name}: подъём {v['пул5000']['подъём']:.2f}×, p = {v['пул5000']['p']:.4f} "
                         f"(5000); {v['по_местности159']['подъём']:.2f}×, p = {v['по_местности159']['p']:.4f} (159)",
                         loc="left", fontsize=9)
            ax.set_xlabel("полнота GTK 292 у наших линий, τ = 300 м")
            ax.legend(fontsize=7.5)
        ax = axs[ri, 2]
        s = sh[sh["набор"] == "C_all"]
        loc = js["местность"]
        ax.hist(s["у_воды"], bins=50, color="#9aa7b8", label="5000 сдвигов эталона")
        ax.hist(s.loc[s["по_местности"], "у_воды"], bins=50, color="#e08a2c", alpha=0.85, label="159 по местности")
        ax.axvline(loc["наблюдение"][0], color="#1f5fd6", lw=2.2,
                   label=f"GTK на месте: {100 * loc['наблюдение'][0]:.0f} %")
        ax.set_title(f"{site}: доля ячеек GTK у воды — несоответствие нуля", loc="left", fontsize=9)
        ax.set_xlabel("доля ячеек эталона ≤ 300 м от берега/болота (OSM)")
        ax.legend(fontsize=7.5)
    fig.suptitle("Нулевые распределения сверки с GTK 292. Семья 2 (159 сдвигов, протокол 25.10, α = 0.025) — "
                 "в NULL_FAMILIES.csv; здесь семьи 3 и 4. Поправка 0.05/N только у внутреннего счётчика (семья 1).",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_nulls.png", dpi=150)
    plt.close(fig)
    print(f"сохранено: {FIG / 'fig_nulls.png'}")


if __name__ == "__main__":
    main()
