"""Задача №8 реестра: линеаменты напрямую по космоснимку против рельефной ветки
и разломов геологической карты.

Три пункта из реестра («Что доделать»), в порядке:

1. Линеаменты по оптике (Sentinel-2, Landsat 8/9) — измерить, а не предполагать
   по аналогии с ``src/s2_composite.py`` (98.3% пикселей — растительность).
2. Линеаменты по радару (Sentinel-1 C-диапазон, ALOS PALSAR-2 L-диапазон) — не
   в постановке, но обе съёмки уже скачаны для датасета и радар не глушится
   покровом.
3. Сравнение до вида, который просят: розы азимутов (карта разломов СЗ/СВ
   раздельно против автоматики), доля разломов карты, «пойманных»
   автоматикой, доля линий без соответствия на карте.

Для каждого источника: конвейер Кэнни -> Хаф из :mod:`src.lineaments`
(переиспользуется без изменений, порог тот же, что у рельефа — рабочее
разрешение 100 м совпадает), стоп-правило воспроизводимости — чётные/нечётные
по дате половины выборки сцен (не азимуты, как у рельефа: см.
:mod:`src.lineaments_sat`).

Выход: ``outputs/metrics/lineaments_sat_summary.csv``,
``outputs/lineaments_sat_roses.png``, ``outputs/lineaments_sat_overlay.png``.
Запуск из корня: ``python -m experiments.lineaments_sat_run``.

РЕЗУЛЬТАТ (07.08.2026, сцены полностью из локального кэша, без докачки).

Стоп-правило воспроизводимости (:data:`config.LIN_MIN_SPEARMAN` = 0.5) не
проходит НИ ОДИН из четырёх источников — включая оба радара, для которых
априори ожидалась более чистая структурная сверка (не глушится травяным
покровом):

| Источник | Сцен | Spearman половин | Отрезков | Длина | Corr. с dist_tect1/2 |
|---|---|---|---|---|---|
| Sentinel-2 | 180 | +0.225 | 420 | 575 км | +0.07 / +0.11 |
| Landsat 8/9 | 90 | +0.325 | 449 | 619 км | +0.18 / +0.27 |
| Sentinel-1 | 24 | −0.202 | 436 | 616 км | +0.11 / +0.16 |
| ALOS PALSAR-2 | 56 | −0.613 | 409 | 575 км | +0.12 / +0.16 |

Для сравнения — рельефная ветка (:mod:`src.lineaments`) на той же паре
половин-как-проверке даёт Spearman = 0.69 (см. её докстринг). То есть
разбиение выборки радикально хуже держит структуру, чем разбиение по
азимутам подсветки: у рельефа объект между половинами не меняется (один и
тот же DEM), у снимков половина сцен — это другая физическая выборка (другая
дата, другой угол обзора, остаточные шумы композита), и на 100-метровом шаге
эта изменчивость сопоставима по амплитуде с самими линиями.

Превью (``outputs/lineaments_sat_overlay.png``) объясняет, почему: линии всех
ЧЕТЫРЁХ источников визуально ложатся на те же гребни и распадки, что и
рельефная отмывка — у оптики это освещённость склона при низком солнце
(71° с.ш.), у радара то же самое даёт геометрия съёмки (layover/shadow вдоль
уступов). То есть автоматика на снимках измеряет РЕЛЬЕФ ЧЕРЕЗ ДОПОЛНИТЕЛЬНЫЙ,
более шумный канал (атмосфера, сезонная влажность, спекл), а не независимую
от рельефа структуру — и слабее рельефа собственной воспроизводимостью.

Сверка с картой разломов (``поймано``/``без_карты`` при 500/1000/2000 м):
доля пойманной карты растёт с порогом ожидаемо (13.7-16.4% -> 54.3-59.4%), но
одинаково у всех источников и без явного лидера; доля линий без соответствия
на карте (16.7-30.9% в зависимости от порога) — не отличима от того, что даёт
случайный узор со сравнимой плотностью, самостоятельным подтверждением
«неоткартированных нарушений» служить не может.

Розы азимутов (``outputs/lineaments_sat_roses.png``): карта разломов
двумодальна и чиста (tect1/СЗ пик 145°, tect2/СВ пик 45° — ожидаемо для
щита). Автоматика по снимкам многолепестковая и размытая; ближе прочих к
карте — Landsat 8/9 и радары (заметное усиление в СВ-секторе, ~ту же
четверть, что и tect2), но ни один источник не даёт чистой бимодальности.

ВЫВОД: постановка «по космоснимку» измерена и закрыта отрицательно для ВСЕХ
четырёх источников, включая не предусмотренные постановкой радарные — что
само по себе полезный результат (гипотеза «радар видит структуру чище
оптики под покровом» здесь НЕ подтвердилась: PALSAR-2 хуже всех проходит
стоп-правило). Рабочей веткой линеаментных признаков остаётся только
рельефная (:mod:`src.lineaments`, префикс ``lin_``) — уже в датасете.
Признаки этого модуля (``lins2_``/``linl8_``/``lins1_``/``linpsr_``) в
датасет не добавлены.
"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import (cell_mask, config, data_loader, integro_grid,  # noqa: E402
                 lineaments_sat, s2_composite, sat_sources, stac_grid)

SOURCES = ("s2", "l8", "s1", "psr")
NAMES = {"s2": "Sentinel-2 (оптика)", "l8": "Landsat 8/9 (оптика)",
        "s1": "Sentinel-1 (радар C)", "psr": "ALOS PALSAR-2 (радар L)"}
MIN_ITEMS = 4          # меньше сцен -> источник пропускается, а не считается на шуме


def _date_key(kind):
    return (lambda s: s["datetime"]) if kind == "s2" else (lambda it: it.datetime)


def _fetch_items(kind, meta):
    if kind == "s2":
        return s2_composite.search_scenes(meta)
    if kind == "l8":
        return stac_grid.search(
            config.L8_COLLECTION, meta, config.L8_YEARS, config.L8_MONTHS,
            query={"eo:cloud_cover": {"lt": config.L8_MAX_CLOUD},
                  "platform": {"in": list(config.L8_PLATFORMS)}},
            group_key="landsat:wrs_path", max_per_group=config.L8_MAX_SCENES_PER_PATH)
    if kind == "s1":
        return stac_grid.search(config.S1_COLLECTION, meta, config.S1_YEARS, config.S1_MONTHS,
                                group_key="sat:relative_orbit",
                                max_per_group=config.S1_MAX_SCENES_PER_ORBIT)
    return stac_grid.search(config.PSR_COLLECTION, meta, config.PSR_YEARS)


def _grayscale(kind, meta, items, min_obs=None):
    if kind == "s2":
        return lineaments_sat.s2_grayscale(meta, items)
    if kind == "l8":
        return lineaments_sat.l8_grayscale(meta, items, min_obs=min_obs)
    if kind == "s1":
        return lineaments_sat.s1_grayscale(meta, items, min_obs=min_obs)
    return lineaments_sat.psr_grayscale(meta, items, min_obs=min_obs)


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5.parquet")
    valid = cell_mask.build_valid_mask(df)
    log(f"сетка {meta.prf}x{meta.pic}, валидных ячеек {int(valid.sum())}")

    base = data_loader.find_base_dir()
    with tempfile.TemporaryDirectory() as alias_dir:
        layers, _ = data_loader.load_all_layers(base / config.SHP_SUBDIR, Path(alias_dir))
    tect1_az, tect1_len = lineaments_sat.geometry_azimuths(layers["tect1"])
    tect2_az, tect2_len = lineaments_sat.geometry_azimuths(layers["tect2"])
    log(f"карта разломов: СЗ (tect1) {len(layers['tect1'])} объектов, "
        f"СВ (tect2) {len(layers['tect2'])} объектов")
    map_dist = np.minimum(df["dist_tect1"].to_numpy(), df["dist_tect2"].to_numpy())[valid]

    rows = []
    rose_data = {"tect1_СЗ": lineaments_sat.azimuth_rose(tect1_az, tect1_len),
                "tect2_СВ": lineaments_sat.azimuth_rose(tect2_az, tect2_len)}
    overlays = {}

    for kind in SOURCES:
        log(f"--- {NAMES[kind]} ---")
        items = _fetch_items(kind, meta)
        log(f"  найдено сцен: {len(items)}")
        if len(items) < MIN_ITEMS:
            log(f"  меньше {MIN_ITEMS} сцен, источник пропущен")
            continue

        even, odd = lineaments_sat.split_even_odd(items, key=_date_key(kind))
        half_min_obs = 1 if kind != "s2" else None   # у s2 порог фиксирован в модуле
        img_full, res_m, n_obs = _grayscale(kind, meta, items)
        img_a, _, _ = _grayscale(kind, meta, even, min_obs=half_min_obs)
        img_b, _, _ = _grayscale(kind, meta, odd, min_obs=half_min_obs)
        rep = lineaments_sat.reproducibility(img_a, img_b)
        verdict = ("ВЕТКА ОТКРЫТА" if np.isfinite(rep["rho"]) and rep["rho"] >= config.LIN_MIN_SPEARMAN
                  else f"ВЕТКА ЗАКРЫВАЕТСЯ (порог {config.LIN_MIN_SPEARMAN})")
        log(f"  {len(items)} сцен ({len(even)}/{len(odd)} половины), шаг {res_m:.0f} м, "
            f"валидных пикселей {np.isfinite(img_full).mean():.1%}")
        log(f"  воспроизводимость: Spearman={rep['rho']:.3f} "
            f"(площадь с линиями {rep['covered_frac']:.1%}, общей {rep['overlap_frac']:.1%}) "
            f"-> {verdict}")

        prefix = f"lin{kind}_"
        feats, segments = lineaments_sat.source_features(meta, img_full, res_m, prefix)
        assert len(feats) == len(df), f"длина {len(feats)} != датасета {len(df)}"
        lens_px = [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in segments]
        lens_km = np.asarray(lens_px) * res_m / 1000.0
        log(f"  отрезков: {len(segments)}, суммарная длина {lens_km.sum():.0f} км")

        row = {"источник": NAMES[kind], "сцен": len(items), "шаг_м": res_m,
              "валидных_пикс": float(np.isfinite(img_full).mean()),
              "воспроизводимость_rho": rep["rho"], "площадь_с_линиями": rep["covered_frac"],
              "вердикт": verdict, "отрезков": len(segments), "длина_км": float(lens_km.sum())}

        a = feats[f"{prefix}dist"].to_numpy()[valid]
        for col in ("dist_tect1", "dist_tect2", "dens_tect", "dist_magm"):
            if col not in df.columns:
                continue
            b = df[col].to_numpy()[valid]
            ok = np.isfinite(a) & np.isfinite(b)
            rho_c = float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 10 else float("nan")
            row[f"rho_{col}"] = rho_c
            log(f"  {prefix}dist ~ {col:<11} Spearman = {rho_c:+.3f}")

        ms = lineaments_sat.match_stats(a, map_dist)
        log("  сверка \"поймано/не поймано\" с объединённой картой разломов (tect1+tect2):")
        for _, r in ms.iterrows():
            row[f"поймано_{r['порог_м']:.0f}м"] = r["доля_карты_поймана"]
            row[f"без_карты_{r['порог_м']:.0f}м"] = r["доля_авто_без_карты"]
            log(f"    порог {r['порог_м']:.0f} м: карта поймана {r['доля_карты_поймана']:.1%}, "
                f"линий без соответствия на карте {r['доля_авто_без_карты']:.1%}")

        rows.append(row)
        az, length = lineaments_sat.segment_azimuths(segments)
        rose_data[kind] = lineaments_sat.azimuth_rose(az, length)
        overlays[kind] = (img_full, segments, res_m)

    summary = pd.DataFrame(rows)
    out_csv = config.PROJECT_ROOT / "outputs" / "metrics" / "lineaments_sat_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    log(f"сводка сохранена: {out_csv}")
    print("\n" + summary.to_string(index=False))

    # --- Превью: розы азимутов (карта СЗ/СВ + автоматика по источникам) ---
    n_roses = len(rose_data)
    fig, axes = plt.subplots(1, n_roses, subplot_kw={"projection": "polar"},
                             figsize=(4 * n_roses, 4.5))
    axes = np.atleast_1d(axes)
    for ax, (name, (edges, hist)) in zip(axes, rose_data.items()):
        centers = np.radians((edges[:-1] + edges[1:]) / 2.0)
        width = np.radians(edges[1] - edges[0])
        # 0..180 -> симметрично на круг (у линии нет направления)
        ax.bar(centers, hist, width=width, bottom=0.0, color="steelblue")
        ax.bar(centers + np.pi, hist, width=width, bottom=0.0, color="steelblue")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(name, fontsize=10)
        ax.set_yticklabels([])
    fig.suptitle("Розы азимутов: карта разломов (СЗ/СВ) против автоматики по источникам",
                fontsize=13)
    fig.tight_layout()
    out_roses = ROOT / "outputs" / "lineaments_sat_roses.png"
    fig.savefig(out_roses, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Превью: отрезки поверх композита, по источникам ---
    if overlays:
        fig, axes = plt.subplots(1, len(overlays), figsize=(6 * len(overlays), 6))
        axes = np.atleast_1d(axes)
        for ax, (kind, (img, segments, res_m)) in zip(axes, overlays.items()):
            lo, hi = np.nanpercentile(img, [2, 98])
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.add_collection(LineCollection(
                [[(x0, y0), (x1, y1)] for (x0, y0), (x1, y1) in segments],
                colors="red", linewidths=0.5))
            ax.set_title(f"{NAMES[kind]}: {len(segments)} отрезков", fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        out_overlay = ROOT / "outputs" / "lineaments_sat_overlay.png"
        fig.savefig(out_overlay, dpi=110, bbox_inches="tight")
        plt.close(fig)
        log(f"сохранено: {out_roses}, {out_overlay}")
    else:
        log(f"сохранено: {out_roses} (превью отрезков пропущено — нет источников)")


if __name__ == "__main__":
    main()
