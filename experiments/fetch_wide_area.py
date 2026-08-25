"""Докачка данных на расширенную территорию — 4 полных листа ГГК-200.

v2 (16.08.2026): сетка переопределена под явный запрос — покрыть ЦЕЛИКОМ
4 листа: основной (R-48-XI,XII), западный (R-48-IX,X), юго-западный
(R-48-XV,XVI) и южный (R-48-XVII,XVIII). Прежняя версия (см. `wide_v1_archive/`)
была снята с рамки одного архива заказчика и покрывала эти листы лишь частично
(не хватало ~40 км по западному краю, зато на юг уходила на лишние ~74 км за
пределы южного листа, в зону R-49-XIII,XIV). Новая рамка снята напрямую с
георазмеченных сканов листов (`datacache/ggk200/<SHEET>/*.tif`, EPSG:4326),
углы репроецированы в СК проекта через `pyproj` (тот же proj4, что у
`config.GRAVMAG_PGRID`).

ВАЖНО — чего эта докачка НЕ даёт: критериальный прогноз (`prognoz`, цель
обучения) нельзя пересчитать за пределами основного листа — все 6 входных
факторных слоёв формулы (`fasii`/facies, `glub_raz_nw`+`glub_r_nw`/tect,
`gr_dol_vp_poly`/paleo, `kory`/struct, `dayki_buf`/magm в `data/Gis-integro/shp_dbf/`)
физически ограничены контуром основного листа. Значит, широкая сетка расширяет
только ПРИЗНАКИ (спутник/рельеф/грав-маг), а не обучающую цель — использовать
3 новых листа для train/val в текущем виде нельзя без новых факторных слоёв от
заказчика на эти листы.

Что НЕ входит в этот скрипт (не про докачку, отдельная задача):
* `pf`/`gm` (33) — нативная сетка грав_маг.pgrid заказчика, без сети; часть
  новой рамки может быть вне её покрытия — обрабатывается отдельно (NaN vs
  глобальная компиляция);
* `geo2` (6) и `geo` (7) — исключены из-за циркулярности с `prognoz`;
* `mingeo` (4), `other` (1), `dist` (2), `relief_v1` (1), `ls` (7) — либо
  считаются с уже скачанного архива заказчика, либо не требуют новых данных.

Сетка (шаг 500 м, CRS та же, что и у `config.GOLD_TARGET_PGRID` — Пулково-1942,
зона Гаусса-Крюгера):

* X0=461500, Y0=7769000, шаг 500 м, 299 столбцов x 303 строки = 90 597 ячеек
  (против 22 946 на исходном листе, то есть ~3.95х; против 104 412 у прежней
  v1-сетки — новая уже прежней по Y, но шире по X).

Идемпотентно: кэш сцен (``datacache/*_anabar``) ключуется по ID сцены, не по
области — уже скачанные сцены переиспользуются, докачиваются только новые.
Старые ``*_wide.parquet`` (под v1-сеткой) заранее перенесены в
``data/processed/wide_v1_archive/`` — этот скрипт создаёт файлы заново под
новую сетку и новую форму, коллизий с v1 нет. DEM кэшируется отдельным файлом
(ключ включает форму растра), поэтому появится новый файл (~сотни МБ).

Выход: ``data/processed/{s2,s1,l8,psr,ast,terrain_v2,lineament,relief_extra}_wide.parquet``.
Запуск из корня: ``python -m experiments.fetch_wide_area``.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, integro_grid, lineaments, relief_v2_extra  # noqa: E402
from src import s2_composite, sat_sources, terrain_v2  # noqa: E402

#: Объединение 4 листов ГГК-200 (основной+запад+юго-запад+юг), снято с
#: георазмеченных сканов `datacache/ggk200/<SHEET>/*.tif` (EPSG:4326),
#: репроецировано в СК проекта и округлено наружу до шага 500 м:
#: X 461803-610876 -> [461500, 611000), Y 7769318-7920370 -> [7769000, 7920500).
WIDE_META = integro_grid.GridMeta(
    obj_count=299 * 303, prop_count=0,
    pic=299, prf=303, dx=500.0, dy=500.0, x0=461500.0, y0=7769000.0,
    properties=[],
)

OUT_DIR = config.PROCESSED_DIR
STEPS: list[str] = []   # для итогового отчёта по времени


def log(t0, msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def _save(df: pd.DataFrame, name: str, meta) -> None:
    n_cells = meta.prf * meta.pic
    assert len(df) == n_cells, f"{name}: {len(df)} строк вместо {n_cells}"
    path = OUT_DIR / f"{name}_wide.parquet"
    df.to_parquet(path, index=False)
    print(f"  -> {path} ({df.shape[0]} x {df.shape[1]})")


def _done(*names: str) -> bool:
    return all((OUT_DIR / f"{n}_wide.parquet").exists() for n in names)


def step_s2(t0):
    if _done("s2"):
        log(t0, "--- Sentinel-2: уже посчитано, пропуск ---")
        return
    log(t0, "--- Sentinel-2: поиск сцен ---")
    scenes = s2_composite.search_scenes(WIDE_META)
    log(t0, f"отобрано сцен: {len(scenes)}")
    comp, n_obs = s2_composite.median_composite(
        scenes, WIDE_META,
        progress=lambda i, n, sc: log(t0, f"  s2 {i}/{n}: {sc['id']}"))
    ratios = s2_composite.band_ratios(comp)
    feats = s2_composite.to_grid_features(comp, ratios, n_obs, WIDE_META)
    log(t0, f"s2: {len(feats.columns)} признаков")
    _save(feats, "s2", WIDE_META)


def step_sensor(t0, kind: str):
    if _done(kind):
        log(t0, f"--- {kind}: уже посчитано, пропуск ---")
        return
    fn, name = sat_sources.SENSORS[kind]
    log(t0, f"--- {name} ---")
    feats, items = fn(WIDE_META, log=lambda msg: log(t0, f"  {msg}"))
    log(t0, f"{kind}: {len(items)} сцен/тайлов, {len(feats.columns)} признаков")
    _save(feats, kind, WIDE_META)


def step_terrain(t0):
    if _done("terrain_v2", "lineament", "relief_extra"):
        log(t0, "--- Рельеф: уже посчитано, пропуск ---")
        return
    log(t0, "--- Рельеф (Copernicus DEM GLO-30): ter/lin/relief2 ---")
    full = terrain_v2.terrain_features(WIDE_META, keep_dropped=True)
    ter = full[list(terrain_v2.TERRAIN_COLS)]
    log(t0, f"ter: {len(ter.columns)} признаков")
    _save(ter, "terrain_v2", WIDE_META)

    elev = terrain_v2.projected_dem(WIDE_META)
    lin = lineaments.lineament_features(WIDE_META, elev)
    log(t0, f"lin: {len(lin.columns)} признаков")
    _save(lin, "lineament", WIDE_META)

    dem_elev_500 = full["dem_elev"].to_numpy(float)
    relief2_full = relief_v2_extra.relief_extra_features(WIDE_META, dem_elev_500, keep_dropped=True)
    relief2 = relief2_full[list(relief_v2_extra.RELIEF2_COLS)]
    log(t0, f"relief2: {len(relief2.columns)} признаков")
    _save(relief2, "relief_extra", WIDE_META)


def _heartbeat(t0, stop: "threading.Event") -> None:
    """Печатает метку каждые 20с, пока идёт долгое блокирующее чтение.

    Без неё долгий одиночный сетевой запрос (десятки-сотни секунд без единой
    строки в stdout) совпадает с обрывом фонового процесса — похоже на
    таймаут по бездействию вывода на уровне среды выполнения, не связанный ни
    с кодом, ни с сетью.
    """
    while not stop.wait(20):
        log(t0, "  ...")


def main() -> None:
    t0 = time.time()
    stop_heartbeat = threading.Event()
    threading.Thread(target=_heartbeat, args=(t0, stop_heartbeat), daemon=True).start()
    n_cells = WIDE_META.prf * WIDE_META.pic
    print(f"Расширенная сетка: {WIDE_META.prf}x{WIDE_META.pic} = {n_cells} ячеек, "
          f"шаг {WIDE_META.dx:.0f} м (исходный лист: 22 946 ячеек, x{n_cells / 22946:.2f})")

    # Порядок — от менее объёмных источников к более объёмным (по квотам сцен:
    # ast<=12 всего, psr — немного немерцающих мозаичных тайлов без квоты,
    # s1<=12/орбита, l8<=15/путь, s2 — 480 сцен, самый тяжёлый и долгий шаг,
    # поэтому идёт последним).
    jobs = [
        ("terrain (ter/lin/relief2)", lambda: step_terrain(t0)),
        ("ast", lambda: step_sensor(t0, "ast")),
        ("psr", lambda: step_sensor(t0, "psr")),
        ("s1", lambda: step_sensor(t0, "s1")),
        ("l8", lambda: step_sensor(t0, "l8")),
        ("s2", lambda: step_s2(t0)),
    ]
    failed = []
    for name, job in jobs:
        try:
            job()
        except Exception as exc:
            log(t0, f"ОШИБКА в блоке {name}: {exc!r}")
            failed.append(name)
        print()

    stop_heartbeat.set()
    log(t0, f"Готово за {(time.time() - t0) / 60:.1f} мин.")
    if failed:
        print(f"НЕ УДАЛОСЬ: {', '.join(failed)} — см. лог выше, перезапуск скрипта "
              f"докачает только недостающее (кэш идемпотентен).")


if __name__ == "__main__":
    main()
