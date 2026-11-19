"""WP18 / WP25 (ТЗ v3.0): сводные таблицы, опись с хэшами, сверка чисел и архив передачи.

Ничего не пересчитывает: читает уже сохранённые результаты и раскладывает их
по каталогам передачи (раздел 11 ТЗ v3.0).

Шаги:
1. ``tables`` — ABLATION_SYNTHESIS.csv, BASELINE_MAIN_COMPARISON.csv,
   site_covariates.csv (``outputs/k1/results``);
2. ``check`` — каждое число из ``release/CLAIMS_EVIDENCE.csv`` сверяется со
   строкой первичного файла; итог в ``release/CLAIMS_CHECK.csv``;
3. ``pack`` — копирует файлы в ``release/LINEAMENTS_HANDOFF/01_…09_``, пишет
   MASTER_MANIFEST.csv и CHECKSUMS.csv (sha256 каждого файла), снимок кода,
   и упаковывает ``LINEAMENTS_HANDOFF_28OCT2026.zip`` в корень проекта.

Данные ВНИГНИ (``data/``), кэш снимков (``datacache/``) и пароли в архив не
попадают: копируется только перечисленное в ``LAYOUT``.

Запуск из корня:
    python -m experiments.openlin_release_pack tables
    python -m experiments.openlin_release_pack check
    python -m experiments.openlin_release_pack pack
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RES = Path("outputs/k1/results")
REL = Path("release")
OUTDIR = REL / "LINEAMENTS_HANDOFF"
ZIP = Path("LINEAMENTS_HANDOFF_28OCT2026.zip")


# ---------------------------------------------------------------- 1. таблицы
def tables() -> None:
    rows = []
    for site in ("anabar", "lapland"):
        d = json.loads(Path(f"outputs/k1/ablations/{site}_ablation.json").read_text(encoding="utf-8"))
        var = {v["вариант"]: v for v in d["варианты"]}
        for p in d["пары"]:
            a, b = var.get(p["от"], {}), var.get(p["к"], {})
            rows.append({"site": site, "step_from": p["от"], "step_to": p["к"],
                         "lines_from": a.get("линий"), "lines_to": b.get("линий"),
                         "km_from": a.get("км"), "km_to": b.get("км"),
                         "equal_budget_km": p["км"], "v2_F1_from": p["v2 F1 от"], "v2_F1_to": p["v2 F1 к"],
                         "paired_delta_v2": round(p["разность_F1_v2"], 4),
                         "CI95_lo": round(p["ДИ95"][0], 4), "CI95_hi": round(p["ДИ95"][1], 4),
                         "verdict_v2": p["вывод"], "GTK_F1_from": p.get("GTK F1 от"), "GTK_F1_to": p.get("GTK F1 к"),
                         "GTK_paired_posthoc": (p.get("GTK_по_участкам_после_прогона") or "").strip("; "),
                         "reference": "v2 (ArcticDEM + LSD), τ = 60 м; GTK τ = 300 м",
                         "source": f"outputs/k1/ablations/{site}_ablation.json"})
    pd.DataFrame(rows).to_csv(RES / "ABLATION_SYNTHESIS.csv", index=False, encoding="utf-8-sig")

    allp = []
    for site in ("anabar", "lapland"):
        p = pd.read_csv(RES / f"baselines_paired_ci_{site}.csv")
        g = pd.read_csv(RES / f"baselines_global_{site}.csv")
        for _, r in p.iterrows():
            other = r["comparison"].split("− ")[1]
            go = g[(g.budget == r["budget"]) & (g.reference == r["reference"])]
            ours = go[go.method == "наша"].iloc[0]
            oth = go[go.method == other].iloc[0]
            allp.append({"site": site, "comparison": r["comparison"], "budget": r["budget"],
                         "reference": r["reference"], "tolerance_m": ours["tolerance_m"],
                         "length_ours_km": ours["auto_length_km"], "length_other_km": oth["auto_length_km"],
                         "P_ours": ours["P"], "R_ours": ours["R"], "P_other": oth["P"], "R_other": oth["R"],
                         "micro_F1_ours": ours["micro_F1"], "micro_F1_other": oth["micro_F1"],
                         "macro_F1_ours": ours["macro_F1"], "macro_F1_other": oth["macro_F1"],
                         "n_tiles": r["n_tiles"], "paired_mean_delta": r["paired_mean_delta"],
                         "CI95_lo": r["CI95_lo"], "CI95_hi": r["CI95_hi"],
                         # PACKET04_DECLARATION: бутстрэп объявлен только при полной длине против v2;
                         # micro-F1 при бюджете нашей карты объявлен, его парный интервал — нет
                         "declared": ("объявлено (Пакет 04)" if r["reference"] == "v2" and r["budget"] == "полная длина"
                                      else "micro-F1 объявлен, парный ДИ — после прогона"
                                      if r["reference"] == "v2" and r["budget"] == "бюджет наш"
                                      else "после прогона Пакета 04 (разведочно)"),
                         "source": f"outputs/k1/results/D09_F1_AUDIT_{site}.csv"})
    pd.DataFrame(allp).to_csv(RES / "BASELINE_MAIN_COMPARISON.csv", index=False, encoding="utf-8-sig")

    sc = json.loads(Path("outputs/k1/transfer/site_characteristics.json").read_text(encoding="utf-8"))
    cov = []
    for s, v in sc.items():
        c = v["характеристики"]
        cov.append({"site": s, "area_km2": c["площадь_км2"], "water_share": round(c["вода_доля"], 4),
                    "forest_NDVI_gt_0.35": round(c["лес_NDVI>0.35"], 4),
                    "slope_median_deg_ArcticDEM": round(c["уклон_медиана_ArcticDEM"], 2),
                    "winter_scenes": c["зима_сцен"], "winter_dates": c["зима_даты"],
                    "sun_elev_min": round(c["солнце_высота"][0], 1), "sun_elev_max": round(c["солнце_высота"][1], 1),
                    "sun_azimuth_median": round(c["солнце_азимут_медиана"], 1),
                    "summer_scenes_ok": c["лето_сцен_годно"], "S1_scenes": c["S1_сцен"],
                    "halfB_single_scene_share": round(c["половина_Б_одна_сцена_доля"], 3),
                    "status": "exploratory, не причинный вывод",
                    "water_source": "OSM" if s != "anabar" else "гидросеть заказчика (в архив не входит)"})
    pd.DataFrame(cov).to_csv(RES / "site_covariates.csv", index=False, encoding="utf-8-sig")
    chance_levels()
    print("сохранено: ABLATION_SYNTHESIS.csv, BASELINE_MAIN_COMPARISON.csv, site_covariates.csv")


def chance_levels() -> None:
    """Случайный уровень P против эталона v2: доля области оценки в полосе τ = 60 м вокруг эталона.

    Это знаменатель «P / случайный» из AUTO_REFERENCE_RESULTS (7.6 % и 26.1 %);
    в Пакете 02 он считался без сохранения в файл.
    """
    import geopandas as gpd
    from scipy.ndimage import distance_transform_edt as edt
    from experiments.openlin_auto_reference import REF, TOL_M, grid, rast
    from experiments.openlin_baselines import RES_M, eval_area
    out = {}
    for site in ("anabar", "lapland", "finland65"):
        g = grid(site)
        ok = eval_area(site, g)
        ref = rast(gpd.read_file(REF / f"{site}_arcticdem_mlsd.gpkg").geometry, g) & ok
        band = edt(~ref, sampling=RES_M) <= TOL_M
        out[site] = {"доля_полосы_60м": float((band & ok).sum() / ok.sum()), "ячеек_области": int(ok.sum())}
    (RES / "chance_levels_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("случайный уровень P (v2, 60 м):", {k: round(v["доля_полосы_60м"], 4) for k, v in out.items()})


# ---------------------------------------------------------------- 2. сверка чисел
def _lookup(path: str, locator: str):
    """Значение по указателю: CSV «col=val;col=val|колонка» или JSON «a::b::c».

    В JSON ключи сами содержат «/» (``anabar/optical_full``), поэтому разделитель — «::».
    """
    p = Path(path)
    if p.suffix == ".csv":
        cond, col = locator.split("|")
        d = pd.read_csv(p)
        for kv in cond.split(";"):
            k, v = kv.split("=", 1)
            d = d[d[k].astype(str) == v]
        if len(d) != 1:
            raise ValueError(f"строк {len(d)} по «{cond}»")
        return float(d.iloc[0][col])
    obj = json.loads(p.read_text(encoding="utf-8"))
    for k in locator.split("::"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    return float(obj)


def check() -> None:
    rows = list(csv.DictReader(open(REL / "CLAIMS_EVIDENCE.csv", encoding="utf-8-sig")))
    out, bad = [], 0
    for r in rows:
        if not r.get("raw_path") or not r.get("locator"):
            out.append(r | {"check": r.get("status") or "—"})
            continue
        try:
            v = _lookup(r["raw_path"], r["locator"]) * float(r.get("scale") or 1)
            ok = abs(v - float(r["result_value"])) <= float(r.get("tol") or 0.0005)
            out.append(r | {"recomputed": round(v, 4), "check": "VERIFIED" if ok else "CONFLICT"})
            bad += not ok
        except Exception as e:
            out.append(r | {"check": f"MISSING ({e})"})
            bad += 1
    keys = list(out[0].keys()) + [k for k in ("recomputed", "check") if k not in out[0]]
    with open(REL / "CLAIMS_CHECK.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    n = sum(1 for r in out if r["check"] == "VERIFIED")
    print(f"сверка: {n} VERIFIED, проблем {bad} из {len(out)} -> {REL / 'CLAIMS_CHECK.csv'}")
    for r in out:
        if r["check"] not in ("VERIFIED",) and r.get("locator"):
            print("  ", r["claim_id"], r["check"], r.get("recomputed"), "!=", r["result_value"])


# ---------------------------------------------------------------- 3. архив
M = "outputs/metrics/"
F = "outputs/k1/figures/"
FR = F + "release/"
LAYOUT = {
    "01_MANUSCRIPT": ["release/manuscript/ARTICLE_IPUZ_FINAL.docx", "release/manuscript/ARTICLE_IPUZ_FINAL.md",
                      "release/manuscript/ABSTRACT_EN.md", "release/manuscript/REFERENCES_CHECKED.md",
                      "release/manuscript/CLAIMS_VS_EVIDENCE.md", "release/manuscript/FIGURE_REFERENCES.md",
                      "release/manuscript/ИЗМЕНЕНИЯ_v19_v20.md", "release/manuscript/ARTICLE_IPUZ_FINAL.pdf",
                      "release/manuscript/build_v20_from_v19.py", "release/manuscript/build_v21_figures.py", "release/manuscript/build_v22_review.py", "release/manuscript/build_v25_review.py", "release/manuscript/build_v26_review.py", "release/manuscript/ИСХОДНАЯ_v24_для_подачи.docx",
                      "experiments/openlin_article_figures.py",
                      F + "article/art_window.png", F + "article/art_sheets.png", F + "article/art_f1.png",
                      {"from": "Статья_ИПУЗ_2026_линеаменты_REVISED_v19.docx", "to": "ИСХОДНАЯ_v19.docx"}],
    "02_REPORT": ["release/report/FULL_RESEARCH_REPORT.docx", "release/report/FULL_RESEARCH_REPORT.md",
                  "release/report/RELATED_WORK_MATRIX.docx", "release/report/RELATED_WORK_MATRIX.md",
                  "release/report/METHODS_EXPLAINED.md"],
    "03_COMPARISON": [RES.as_posix() + f"/{n}" for n in (
        "BASELINE_MAIN_COMPARISON.csv", "D09_F1_AUDIT_anabar.csv", "D09_F1_AUDIT_lapland.csv",
        "baselines_global_anabar.csv", "baselines_global_lapland.csv", "baselines_paired_ci_anabar.csv",
        "baselines_paired_ci_lapland.csv", "length_budget_curves_anabar.csv", "length_budget_curves_lapland.csv",
        "ABLATION_SYNTHESIS.csv", "chance_levels_v2.json")] + [F + "fig_budget_vs_f1_anabar.png", F + "fig_budget_vs_f1_lapland.png",
                                      "release/D09_INTEGRATED.md", M + "_d09_audit_anabar.log",
                                      M + "_d09_audit_lapland.log"],
    "04_NULLS": ["release/NULL_FAMILIES.md", RES.as_posix() + "/NULL_FAMILIES.csv", F + "fig_nulls.png",
                 {"from": RES.as_posix() + "/gtk_shifts.csv", "to": "gtk_shifts_lapland.csv"},
                 {"from": RES.as_posix() + "/finland65/gtk_shifts.csv", "to": "gtk_shifts_finland65.csv"},
                 RES.as_posix() + "/gtk_spatial_stratified_exploratory.json",
                 {"from": RES.as_posix() + "/finland65/gtk_spatial_stratified_exploratory.json",
                  "to": "gtk_spatial_stratified_exploratory_finland65.json"},
                 RES.as_posix() + "/gtk_water_stratified_posthoc.json",
                 M + "openlin_transfer_check_lapland.csv", M + "openlin_transfer_check_finland65.csv",
                 M + "openlin_transfer_check_lapland_shifts.csv", M + "openlin_transfer_check_finland65_shifts.csv",
                 "outputs/k1/null_validation/anabar_null_validation.json",
                 "outputs/k1/null_validation/lapland_null_validation.json",
                 F + "surrogate_nonstationarity_anabar.png", F + "surrogate_nonstationarity_lapland.png"],
    "05_VECTORS": ["outputs/k1/vectors/anabar_lines.gpkg", "outputs/k1/vectors/lapland_lines.gpkg",
                   "outputs/k1/vectors/finland65_lines.gpkg", "outputs/k1/vectors/export_check.json",
                   "outputs/k1/baselines/anabar_baselines.gpkg", "outputs/k1/baselines/lapland_baselines.gpkg",
                   "outputs/k1/reference/anabar_arcticdem_mlsd.gpkg", "outputs/k1/reference/lapland_arcticdem_mlsd.gpkg",
                   "outputs/k1/reference/finland65_arcticdem_mlsd.gpkg"],
    "06_FIGURES": [FR + n for n in ("real_anabar.png", "real_anabar.pdf", "real_lapland_gtk.png",
                                    "real_lapland_gtk.pdf", "real_finland65.png", "real_finland65.pdf",
                                    "side_by_side_examples.png", "side_by_side_examples.pdf", "problem_cases.png",
                                    "problem_cases.pdf", "transfer_3sites.png", "transfer_3sites.pdf",
                                    "fig_scheme.png", "fig_scheme.pdf", "figure_manifest.csv")]
                  + [F + n for n in ("injection_azimuth.png", "ablation_tradeoff_anabar.png",
                                     "ablation_tradeoff_lapland.png", "shore_cases.png", "xai_groups.png",
                                     "fig_data_sources.png", "gtk_null_distribution.png",
                                     "gtk_null_distribution_finland65.png", "auto_reference_v2_windows.png",
                                     "surrogate_windows_anabar.png", "selection_manifest.csv")]
                  + ["release/figure_manifest.md", "audit/RELEASE_FIGURE_RULE.md", "audit/SELECTION_RULE.md"],
    "07_PRACTICE": ["release/practice/DEMO_README.md", FR + "line_traces.json", FR + "line_trace_642.png",
                    FR + "line_trace_25.png", "release/practice/command_log.txt"],
    "08_REPRODUCIBILITY": ["release/MASTER_MANIFEST.csv", "release/REPRODUCE.md", "release/RELEASE_NOTES.md",
                           "release/CLAIMS_EVIDENCE.csv", "release/CLAIMS_CHECK.csv", "release/environment.txt",
                           "release/git_state.txt", "release/code_snapshot.zip", "release/pytest_28oct.txt", "audit/DISCREPANCIES.md",
                           "release/DISCREPANCIES_FINAL.md", "release/REPRO_STATUS.md", "audit/data_manifest.csv",
                           "experiments/OPENLIN_RULES.md", RES.as_posix() + "/site_covariates.csv",
                           "audit/third_site_selection.json", "audit/PACKET09_DECLARATION.md"]
                          + [f"audit/PACKET{n:02d}_{k}.md" for n in range(4, 11) for k in ("DECLARATION", "RESULTS")
                             if Path(f"audit/PACKET{n:02d}_{k}.md").exists() and n != 9]
                          + ["audit/PACKET09_RESULTS.md", "audit/START_REPORT.md", "audit/PACKET01_REPORT.md",
                             "audit/AUTO_REFERENCE_DECLARATION.md", "audit/AUTO_REFERENCE_V2_DECLARATION.md",
                             "audit/AUTO_REFERENCE_RESULTS.md", "audit/ОТВЕТЫ_НА_ДЕЛЬТА_ТЗ_27.10.md"],
    "09_EXPERT_PREP": ["release/ANNOTATION_STATUS.md", "outputs/k1/artifact_audit/sample_manifest.csv",
                       "outputs/k1/artifact_audit/expert_labels.csv",
                       "outputs/k1/artifact_audit/expert_sheet_anabar_1.png",
                       "outputs/k1/artifact_audit/expert_sheet_anabar_2.png",
                       "outputs/k1/artifact_audit/expert_sheet_lapland_1.png",
                       "outputs/k1/artifact_audit/expert_sheet_lapland_2.png",
                       "audit/Протокол_слепой_разметки_ЧЕРНОВИК_v0.docx"],
    ".": ["release/README.md", "release/EXECUTIVE_SUMMARY.docx", "release/EXECUTIVE_SUMMARY.md"],
}

#: строки MASTER_MANIFEST: (site, stage, method, run_id, crs, файл, код, эталон, бюджет, допуск, рисунок, статус)
RUNS = [
    ("anabar", "C_all", "тракт TWIN (снимок)", "TWIN, перегон 26.10 (D08)", "EPSG:32647",
     M + "openlin_zone_lines_anabar_v3_бл_линии.npz", "experiments/openlin_zone_lines_v3.py", "—", "полная", "—",
     "real_anabar", "VERIFIED"),
    ("anabar", "C", "тракт TWIN, в зонах", "TWIN", "EPSG:32647", M + "openlin_zone_lines_anabar_v3_бл.npz",
     "experiments/openlin_zone_lines_v3.py", "—", "полная", "—", "—", "VERIFIED"),
    ("anabar", "DEM1 add", "рельеф GLO-30 + ALOS, одна линия", "DEM1", "EPSG:32647",
     M + "openlin_dem_lines_anabar_одна_линии.npz", "experiments/openlin_dem_lines.py", "—", "полная", "—",
     "real_anabar", "VERIFIED"),
    ("lapland", "C_all", "тракт XFER (перенос)", "XFER, восстановлен 26.10", "EPSG:32635",
     M + "openlin_zone_lines_lapland_v3_перенос_линии.npz", "experiments/openlin_zone_lines_v3.py", "—", "полная",
     "—", "real_lapland_gtk", "VERIFIED"),
    ("finland65", "C_all", "тракт XFER2 (перенос)", "XFER2, 26.10", "EPSG:32635",
     M + "openlin_zone_lines_finland65_v3_перенос_линии.npz", "experiments/openlin_zone_lines_v3.py", "—",
     "полная", "—", "real_finland65", "VERIFIED"),
    ("all", "GPKG", "векторный экспорт", "Пакет 01 + 28.10 (finland65)", "по площадке",
     "outputs/k1/vectors/export_check.json", "experiments/openlin_export_vectors.py", "—", "—", "—", "—", "VERIFIED"),
    ("anabar", "eval", "наша / LSD / Хаф", "D09 audit 27.10", "EPSG:32647", RES.as_posix() + "/D09_F1_AUDIT_anabar.csv",
     "experiments/openlin_d09_audit.py", "v2", "полная / наш / LSD", "60", "side_by_side_examples", "VERIFIED"),
    ("lapland", "eval", "наша / LSD / Хаф", "D09 audit 27.10", "EPSG:32635", RES.as_posix() + "/D09_F1_AUDIT_lapland.csv",
     "experiments/openlin_d09_audit.py", "v2; GTK292", "полная / наш / LSD", "60; 300", "side_by_side_examples",
     "VERIFIED"),
    ("anabar", "eval", "кривые бюджета", "release 28.10", "EPSG:32647", RES.as_posix() + "/length_budget_curves_anabar.csv",
     "experiments/openlin_release_baselines.py", "v2", "сетка 20 км…полная", "60", "fig_budget_vs_f1_anabar",
     "VERIFIED"),
    ("lapland", "eval", "кривые бюджета", "release 28.10", "EPSG:32635", RES.as_posix() + "/length_budget_curves_lapland.csv",
     "experiments/openlin_release_baselines.py", "v2; GTK292", "сетка 20 км…полная", "60; 300",
     "fig_budget_vs_f1_lapland", "VERIFIED (GTK-кривая — после просмотра)"),
    ("lapland", "null", "GTK-159 (семья 2)", "openlin_transfer_check 26.10", "EPSG:32635",
     M + "openlin_transfer_check_lapland.csv", "experiments/openlin_transfer_check_k1.py", "GTK292", "—", "300", "—",
     "VERIFIED; исходные 159 сдвигов 25.10 MISSING"),
    ("lapland", "null", "GTK-5000 (семья 3) + условный (4)", "Пакет 05", "EPSG:32635", RES.as_posix() + "/gtk_shifts.csv",
     "experiments/openlin_gtk_null.py", "GTK292", "—", "150–1000", "fig_nulls", "VERIFIED, разведочно"),
    ("finland65", "null", "GTK-159 / 5000 / условный", "Пакет 09", "EPSG:32635",
     RES.as_posix() + "/finland65/gtk_shifts.csv", "experiments/openlin_gtk_null.py", "GTK292", "—", "300",
     "fig_nulls", "VERIFIED, разведочно"),
    ("anabar", "reference", "авто-эталон v2", "AUTO_REF v2", "EPSG:32647", "outputs/k1/reference/anabar_eval_v2.json",
     "experiments/openlin_auto_reference.py", "v2", "полная", "60", "auto_reference_v2_windows", "VERIFIED"),
    ("lapland", "reference", "авто-эталон v2", "AUTO_REF v2", "EPSG:32635", "outputs/k1/reference/lapland_eval_v2.json",
     "experiments/openlin_auto_reference.py", "v2; GTK292", "полная", "60; 300", "—", "VERIFIED"),
    ("finland65", "reference", "авто-эталон v2", "AUTO_REF v2", "EPSG:32635",
     "outputs/k1/reference/finland65_eval_v2.json", "experiments/openlin_auto_reference.py", "v2", "полная", "60",
     "—", "VERIFIED"),
    ("anabar", "ablation", "V0–V5", "Пакет 07", "EPSG:32647", "outputs/k1/ablations/anabar_ablation.json",
     "experiments/openlin_ablation.py", "v2", "равная длина пары", "60", "ablation_tradeoff_anabar", "VERIFIED"),
    ("lapland", "ablation", "V0–V4", "Пакет 07", "EPSG:32635", "outputs/k1/ablations/lapland_ablation.json",
     "experiments/openlin_ablation.py", "v2; GTK292 (post hoc)", "равная длина пары", "60; 300",
     "ablation_tradeoff_lapland", "VERIFIED"),
    ("anabar; lapland", "null", "IAAFT по листу / по участкам", "Пакет 08", "по площадке",
     "outputs/k1/null_validation/lapland_null_validation.json", "experiments/openlin_null_diag.py", "—", "—", "—",
     "surrogate_nonstationarity_*", "VERIFIED, разведочно"),
    ("anabar", "injection", "96 врезок", "AZRESP после правки NMS", "EPSG:32647", "audit/injections_raw.csv",
     "experiments/openlin_azimuth_response.py", "—", "—", "—", "injection_azimuth",
     "VERIFIED (после правки); до правки — только агрегат лога"),
    ("all", "xai", "групповая важность «Коры»", "Пакет 10", "—", "outputs/k1/xai/rank_transfer.json",
     "experiments/openlin_kora_xai.py", "—", "—", "—", "xai_groups", "VERIFIED"),
    ("lapland", "GTK length", "длина GTK 292", "D03", "EPSG:32635", "audit/DISCREPANCIES.md", "—", "GTK292", "—",
     "—", "—", "2690 и 1894 км VERIFIED; 2529 км EXCLUDED"),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot() -> None:
    """Состояние git, окружение и копия незакоммиченного кода ветки (без данных)."""
    run = lambda *a: subprocess.run(a, capture_output=True, text=True, encoding="utf-8").stdout
    (REL / "git_state.txt").write_text(
        f"снято {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n$ git rev-parse --abbrev-ref HEAD\n"
        + run("git", "rev-parse", "--abbrev-ref", "HEAD") + "\n$ git log -5 --oneline\n" + run("git", "log", "-5", "--oneline")
        + "\n$ git status --short\n" + run("git", "status", "--short") + "\n$ git diff --stat\n" + run("git", "diff", "--stat"),
        encoding="utf-8")
    import importlib.metadata as md
    pk = ["numpy", "scipy", "scikit-image", "opencv-python-headless", "geopandas", "shapely", "rasterio", "pandas",
          "matplotlib", "networkx", "scikit-learn", "python-docx", "pystac-client", "planetary-computer"]
    env = [f"python {sys.version.split()[0]}", f"platform {sys.platform}"]
    for n in pk:
        try:
            env.append(f"{n} {md.version(n)}")
        except md.PackageNotFoundError:
            env.append(f"{n} — не установлен")
    env.append("GPU не использовался (torch только CPU)")
    (REL / "environment.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    with zipfile.ZipFile(REL / "code_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        files = (sorted(Path("experiments").glob("openlin_*.py")) + [Path("experiments/OPENLIN_RULES.md"),
                 Path("experiments/md_to_docx.py"), Path("experiments/common.py")]
                 + sorted(Path("src").glob("openlin*.py")) + sorted(Path("tests").glob("test_*openlin*.py"))
                 + [Path("tests/test_nms_orientation.py"), Path("cache_paths.py"), Path("requirements.txt")])
        for f in files:
            if f.exists():
                z.write(f, f.as_posix())
        z.writestr("git_diff.patch", run("git", "diff"))


def pack() -> None:
    snapshot()
    rows = []
    for r in RUNS:
        p = Path(r[5])
        rows.append(dict(zip(("site", "stage", "method", "run_id", "crs", "result_file", "code", "reference",
                              "budget", "tolerance_m", "figure", "evidence_status"), r))
                    | {"sha256": sha(p) if p.exists() else "MISSING"})
    pd.DataFrame(rows).to_csv(REL / "MASTER_MANIFEST.csv", index=False, encoding="utf-8-sig")

    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    missing, sums = [], []
    for folder, items in LAYOUT.items():
        dst = OUTDIR / folder
        dst.mkdir(parents=True, exist_ok=True)
        for it in items:
            src, name = (Path(it["from"]), it["to"]) if isinstance(it, dict) else (Path(it), Path(it).name)
            if not src.exists():
                missing.append(str(src))
                continue
            shutil.copy2(src, dst / name)
            sums.append({"path": (Path(folder) / name).as_posix(), "source": src.as_posix(),
                         "bytes": src.stat().st_size, "sha256": sha(src)})
    pd.DataFrame(sums).to_csv(OUTDIR / "08_REPRODUCIBILITY" / "CHECKSUMS.csv", index=False, encoding="utf-8-sig")
    (OUTDIR / "08_REPRODUCIBILITY" / "MISSING_FILES.txt").write_text(
        "Файлы из списка сборки, которых нет на диске:\n" + ("\n".join(missing) or "нет") + "\n", encoding="utf-8")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUTDIR.rglob("*")):
            if f.is_file():
                z.write(f, (Path("LINEAMENTS_HANDOFF") / f.relative_to(OUTDIR)).as_posix())
    print(f"файлов {len(sums)}, отсутствует {len(missing)}: {missing}")
    print(f"архив: {ZIP} ({ZIP.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    {"tables": tables, "check": check, "pack": pack}[sys.argv[1]]()
