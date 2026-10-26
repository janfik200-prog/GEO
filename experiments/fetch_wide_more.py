"""Докачка после переезда: восстановление кэшей съёмок и добор покрытия широкой сетки.

Две задачи в одном прогоне.

1. ВОССТАНОВЛЕНИЕ. Каталог ``datacache`` на новый компьютер не переносился —
   он качается заново. Здесь он наполняется теми же рецептами и квотами, что
   и раньше, но результат композитов выбрасывается: готовые
   ``data/processed/*_features.parquet`` не перезаписываются, кэш нужен только
   для будущих пересчётов. Сюда же — Copernicus DEM на обе сетки и листы
   ГГК-200 (из них снята рамка ``WIDE_META``).

2. ДОБОР ПОКРЫТИЯ НА ШИРОКОЙ СЕТКЕ. Зонд по каталогам 19.10.2026 показал, что
   дыры на ``WIDE_META`` (ASTER 44%, Landsat 8/9 67%, Sentinel-1 79%) — это не
   отсутствие съёмки, а квоты, подобранные под один лист:

   * ASTER — общий потолок 12 сцен на территорию вчетверо больше листа; все 22
     летние сцены с облачностью < 60% накрывают контурами 71% против 39%;
   * Landsat 8/9 — квота 15 на path, а на широкой сетке у path по 2-3 ряда:
     лучшие сцены собирались в одном ряду. Квота переведена на кадр path/row;
   * Sentinel-1 — 24 сцены из 124 доступных, квота на орбиту поднята.

   Снять квоты целиком нельзя: ``stac_grid._composite`` держит все сцены в
   памяти на полной сетке, 381 сцена Landsat — десятки гигабайт.

   Плюс два источника, которых на широкой сетке не было:

   * Landsat 7 ETM+ 1999-2002 (до отказа SLC в мае 2003) — открытая замена
     фрагмента ``landsat_fragm.pgrid`` от организации, который покрывает 28%
     широкой сетки; 49 летних сцен накрывают её целиком. Это не те же числа,
     что ``ls_*`` (там DN, здесь отражение Collection 2), сшивать их можно
     только после сверки на перекрытии;
   * ASTER TIR — на листе посчитан (``astir_features``), на широкой сетке нет.

   Выход — НОВЫЕ файлы ``data/processed/<сенсор>_wide_v3.parquet``; прежние
   ``*_wide.parquet``, ``dataset_wide.parquet`` и база не трогаются: решение,
   заменять ли признаки, принимает сверка, а не факт скачивания.

Идемпотентно: кэш сцен ключуется по ID сцены и форме сетки, готовые шаги
пропускаются, прерванный прогон просто перезапускается.

Запуск из корня: ``python -u -m experiments.fetch_wide_more``.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cache_paths import CACHE  # noqa: E402
from src import aster_tir, config, integro_grid, s2_composite, sat_sources, stac_grid, terrain_v2  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

MAIN_META = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
OUT = config.PROCESSED_DIR
DONE_DIR = CACHE / "_restore_done"   # отметки восстановленных кэшей: композит не сохраняется

# Квоты добора на широкой сетке.
L8_PER_TILE = 8          # сцен на кадр path/row (было 15 на path из 2-3 рядов)
L7_PER_TILE = 8
S1_PER_ORBIT = 30        # было 12; 124 сцены на 2 орбиты
AST_MAX = 30             # было 12 на всю территорию; летних сцен < 60% всего 22
L7_YEARS = (1999, 2000, 2001, 2002)   # SLC отказал 31.05.2003 — дальше полосы

T0 = time.time()
_search_orig = stac_grid.search


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def search_by_tile(collection, meta, years=None, months=None, query=None,
                   group_key=None, max_per_group=None, max_items=2000):
    """``stac_grid.search``, но Landsat квотируется по кадру path/row, а не по path."""
    if group_key != "landsat:wrs_path" or max_per_group is None:
        return _search_orig(collection, meta, years, months, query, group_key,
                            max_per_group, max_items)
    items = _search_orig(collection, meta, years, months, query, max_items=max_items)
    tiles: dict[str, list] = {}
    for it in items:
        p = it.properties
        tiles.setdefault(f"{p.get('landsat:wrs_path')}/{p.get('landsat:wrs_row')}", []).append(it)
    out = []
    for k in sorted(tiles):
        out += stac_grid._pick(tiles[k], max_per_group)
    return out


def coverage(df: pd.DataFrame, prefix: str) -> float:
    cols = [c for c in df.columns if c.startswith(prefix)
            and not c.endswith(("n_obs", "valid_frac"))]
    return float(df[cols].notna().any(axis=1).mean()) if cols else float("nan")


def compare(new: pd.DataFrame, name: str, prefix: str, old_name: str | None) -> None:
    msg = f"  покрытие {name}: {coverage(new, prefix) * 100:.1f}% ячеек"
    old_path = OUT / f"{old_name}_wide.parquet" if old_name else None
    if old_path and old_path.exists():
        old = pd.read_parquet(old_path)
        old_prefix = "ls_" if old_name == "ls" else prefix
        msg += f" (было {coverage(old, old_prefix) * 100:.1f}%)"
    log(msg)


def save_v3(df: pd.DataFrame, name: str) -> None:
    n = WIDE_META.prf * WIDE_META.pic
    assert len(df) == n, f"{name}: {len(df)} строк вместо {n}"
    path = OUT / f"{name}_wide_v3.parquet"
    df.to_parquet(path, index=False)
    log(f"  -> {path.name} ({df.shape[0]} x {df.shape[1]})")


def done_v3(name: str) -> bool:
    return (OUT / f"{name}_wide_v3.parquet").exists()


# ------------------------------------------------------------ восстановление
def restore(tag: str, fn) -> None:
    mark = DONE_DIR / tag
    if mark.exists():
        log(f"--- {tag}: кэш уже восстановлен, пропуск ---")
        return
    log(f"--- восстановление кэша: {tag} ---")
    fn()
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    mark.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")


def restore_dem() -> None:
    for name, meta in (("лист", MAIN_META), ("широкая", WIDE_META)):
        elev = terrain_v2.projected_dem(meta)
        log(f"  DEM {name}: {elev.shape}, валидно {np.isfinite(elev).mean() * 100:.1f}%")


def restore_ggk200() -> None:
    from experiments import fetch_ggk200
    fetch_ggk200.main()


def restore_sensor(kind: str):
    def run():
        fn, title = sat_sources.SENSORS[kind]
        feats, items = fn(MAIN_META, log=lambda m: log(f"  {m}"))
        log(f"  {title}: {len(items)} сцен в кэше, композит не сохраняется")
    return run


def restore_s2() -> None:
    scenes = s2_composite.search_scenes(MAIN_META)
    log(f"  Sentinel-2: {len(scenes)} сцен")
    s2_composite.median_composite(scenes, MAIN_META,
                                  progress=lambda i, n, sc: log(f"  s2 {i}/{n}: {sc['id']}"))


# ------------------------------------------------------------ добор широкой сетки
def wide_sensor(kind: str, out_name: str, prefix: str, old_name: str | None,
                patch: dict, rename: str | None = None, bands: dict | None = None):
    def run():
        if done_v3(out_name):
            log(f"--- {out_name}: уже посчитано, пропуск ---")
            return
        saved_cfg = {k: getattr(config, k) for k in patch}
        saved_bands = dict(sat_sources.L8_BANDS)
        stac_grid.search = search_by_tile
        try:
            for k, v in patch.items():
                setattr(config, k, v)
            if bands:
                sat_sources.L8_BANDS.update(bands)
            fn, title = sat_sources.SENSORS[kind]
            log(f"--- широкая сетка: {out_name} ({title}), {patch} ---")
            feats, items = fn(WIDE_META, log=lambda m: log(f"  {m}"))
        finally:
            stac_grid.search = _search_orig
            for k, v in saved_cfg.items():
                setattr(config, k, v)
            sat_sources.L8_BANDS.clear()
            sat_sources.L8_BANDS.update(saved_bands)
        if rename:
            feats = feats.rename(columns=lambda c: rename + c[len(prefix):] if c.startswith(prefix) else c)
        log(f"  {len(items)} сцен, {feats.shape[1]} признаков")
        compare(feats, out_name, rename or prefix, old_name)
        save_v3(feats, out_name)
    return run


def l7_lwir_asset() -> dict:
    """Имя теплового ассета у Landsat 7 в каталоге — не ``lwir11``, как у 8/9."""
    items = _search_orig(config.L8_COLLECTION, WIDE_META, L7_YEARS, (7, 8),
                         query={"platform": {"in": ["landsat-7"]}}, max_items=5)
    keys = [k for k in items[0].assets if k.startswith("lwir")] if items else []
    if not keys:
        raise RuntimeError("у сцен Landsat 7 нет теплового ассета")
    log(f"  тепловой ассет Landsat 7: {keys[0]}")
    return {"lwir11": (keys[0], 1)}


def wide_l7() -> None:
    if done_v3("l7"):
        log("--- l7: уже посчитано, пропуск ---")
        return
    wide_sensor("l8", "l7", "l8_", "ls",
                {"L8_PLATFORMS": ("landsat-7",), "L8_YEARS": L7_YEARS,
                 "L8_MAX_SCENES_PER_PATH": L7_PER_TILE},
                rename="l7_", bands=l7_lwir_asset())()


def wide_astir() -> None:
    if done_v3("astir"):
        log("--- astir: уже посчитано, пропуск ---")
        return
    log("--- широкая сетка: astir (ASTER TIR, LP DAAC) ---")
    feats, granules = aster_tir.fetch(WIDE_META, log=lambda m: log(f"  {m}"))
    log(f"  {len(granules)} гранул, {feats.shape[1]} признаков")
    compare(feats, "astir", "astir_", None)
    save_v3(feats, "astir")


def heartbeat(stop: threading.Event) -> None:
    while not stop.wait(30):
        log("  ...")


def main() -> None:
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    jobs = [
        ("DEM", lambda: restore("dem", restore_dem)),
        ("ГГК-200", lambda: restore("ggk200", restore_ggk200)),
        ("ast широкая", wide_sensor("ast", "ast", "ast_", "ast",
                                    {"AST_MAX_SCENES": AST_MAX})),
        ("s1 широкая", wide_sensor("s1", "s1", "s1_", "s1",
                                   {"S1_MAX_SCENES_PER_ORBIT": S1_PER_ORBIT})),
        ("l8 широкая", wide_sensor("l8", "l8", "l8_", "l8",
                                   {"L8_MAX_SCENES_PER_PATH": L8_PER_TILE})),
        ("l7 широкая", wide_l7),
        ("astir широкая", wide_astir),
        ("кэш s1 лист", lambda: restore("s1_main", restore_sensor("s1"))),
        ("кэш l8 лист", lambda: restore("l8_main", restore_sensor("l8"))),
        ("кэш psr лист", lambda: restore("psr_main", restore_sensor("psr"))),
        ("кэш ast лист", lambda: restore("ast_main", restore_sensor("ast"))),
        ("кэш astir лист", lambda: restore("astir_main", restore_sensor("astir"))),
        ("кэш s2 лист", lambda: restore("s2_main", restore_s2)),
    ]
    failed = []
    for name, job in jobs:
        try:
            job()
        except Exception as exc:  # noqa: BLE001 — один сенсор не должен ронять остальные
            log(f"ОШИБКА в блоке {name}: {exc!r}")
            failed.append(name)
    stop.set()
    log(f"Готово за {(time.time() - T0) / 60:.1f} мин.")
    if failed:
        log(f"НЕ УДАЛОСЬ: {', '.join(failed)} — перезапуск докачает только недостающее.")


if __name__ == "__main__":
    main()
