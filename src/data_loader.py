"""Загрузка геоданных: поиск каталога, чтение шейп-файлов, CRS.

Имена части шейп-файлов записаны кириллицей, что ломает чтение в некоторых
сборках GDAL. :func:`prepare_ascii_aliases` создаёт временные ASCII-копии, и
дальше все слои читаются уже по алиасам.
"""

import os
import re
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from . import config


def find_base_dir() -> Path:
    """Найти каталог с данными (содержащий ``shp_dbf/svita_new.shp``).

    Перебирает :data:`config.BASE_DIR_CANDIDATES`; если не нашёл, пытается
    распаковать один из :data:`config.ZIP_CANDIDATES` рядом и ищет в нём.

    :raises FileNotFoundError: если данные не найдены ни одним способом.
    """
    sentinel = config.SHP_SUBDIR + "/" + config.MASK_SENTINEL

    for base in config.BASE_DIR_CANDIDATES:
        if (base / config.SHP_SUBDIR / config.MASK_SENTINEL).exists():
            return base

    for zip_path in config.ZIP_CANDIDATES:
        if not zip_path.exists():
            continue
        unzip_dir = zip_path.parent / "prog_zip"
        unzip_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(unzip_dir)
        for base in [unzip_dir, *unzip_dir.rglob("*")]:
            if (base / config.SHP_SUBDIR / config.MASK_SENTINEL).exists():
                return base

    raise FileNotFoundError(
        f"Не найден каталог с {config.SHP_SUBDIR} (ожидался {sentinel}). "
        "Проверь BASE_DIR_CANDIDATES в src/config.py или положи рядом Прогноз.zip."
    )


def read_sidecar_proj4(shp_path: Path) -> str | None:
    """Прочитать proj4-проекцию из sidecar-файла ``*_shp.pj4`` (формат ГИС Интегро)."""
    sidecar = shp_path.with_name(shp_path.stem + "_shp.pj4")
    if sidecar.exists():
        txt = sidecar.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"pj4=(.+)", txt)
        if m:
            return m.group(1).strip()
    return None


def prepare_ascii_aliases(shp_dir: Path, alias_dir: Path) -> dict[str, Path]:
    """Создать ASCII-алиасы для шейп-файлов с кириллическими именами.

    Возвращает отображение ``имя_слоя -> путь к .shp``. Файлы с безопасными
    именами остаются на месте; небезопасные копируются в ``alias_dir`` как
    ``layer_NN.*`` (вместе с sidecar ``_shp.pj4``).
    """
    stems: dict[bytes, set[bytes]] = {}
    for name_b in os.listdir(os.fsencode(shp_dir)):
        if not name_b.endswith((b".shp", b".shx", b".dbf", b".prj", b".pj4")) or name_b.endswith(b"_shp.pj4"):
            continue
        base_b, ext_b = os.path.splitext(name_b)
        stems.setdefault(base_b, set()).add(ext_b)

    aliases: dict[str, Path] = {}
    alias_idx = 0
    for base_b, exts in sorted(stems.items()):
        try:
            base_s = os.fsdecode(base_b)
            safe = all(ord(ch) < 128 and (ch.isalnum() or ch in "_-. ") for ch in base_s)
        except Exception:
            safe, base_s = False, None

        if safe:
            aliases[base_s] = shp_dir / f"{base_s}.shp"
            continue

        alias = f"layer_{alias_idx:02d}"
        alias_idx += 1
        for ext_b in exts:
            src = os.path.join(os.fsencode(shp_dir), base_b + ext_b)
            dst = alias_dir / f"{alias}{os.fsdecode(ext_b)}"
            shutil.copyfile(src, dst)
        pj4_src = os.path.join(os.fsencode(shp_dir), base_b + b"_shp.pj4")
        if os.path.exists(pj4_src):
            shutil.copyfile(pj4_src, alias_dir / f"{alias}_shp.pj4")
        aliases[alias] = alias_dir / f"{alias}.shp"
    return aliases


def load_layer(path: Path) -> gpd.GeoDataFrame:
    """Прочитать шейп-файл, убрать пустые геометрии и восстановить CRS из ``.pj4``."""
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        proj4 = read_sidecar_proj4(path)
        if proj4:
            gdf = gdf.set_crs(CRS.from_proj4(proj4), allow_override=True)
    return gdf


def to_crs_safe(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    """Привести слой к ``target_crs``, аккуратно обрабатывая отсутствующий CRS."""
    if gdf.crs is None and target_crs is not None:
        return gdf.set_crs(target_crs, allow_override=True)
    if target_crs is None or gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)


def collect_points(mask_crs, aliases: dict[str, Path]) -> gpd.GeoDataFrame | None:
    """Собрать все точечные слои (рудопроявления), исключая базовые слои-факторы.

    Возвращает объединённый GeoDataFrame в ``mask_crs`` или ``None``, если точек нет.
    """
    layers = []
    for name, shp_path in aliases.items():
        if name in config.BASE_LAYER_NAMES:
            continue
        gdf = to_crs_safe(load_layer(shp_path), mask_crs)
        types = {str(x) for x in gdf.geom_type.unique()}
        if "Point" in types or "MultiPoint" in types:
            gdf = gdf.copy()
            gdf["source_layer"] = name
            layers.append(gdf)
    if not layers:
        return None
    pts = pd.concat(layers, ignore_index=True)
    return gpd.GeoDataFrame(pts, geometry="geometry", crs=mask_crs)


def load_all_layers(
    shp_dir: Path, alias_dir: Path
) -> tuple[dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame | None]:
    """Загрузить все слои-факторы и точки.

    Возвращает ``(layers, points)``, где ``layers`` — словарь по ролям из
    :data:`config.LAYER_FILES` (``mask``, ``facies``, ``tect1`` …), приведённый к
    CRS маски, а ``points`` — точечные рудопроявления (или ``None``).
    """
    aliases = prepare_ascii_aliases(shp_dir, alias_dir)

    mask = load_layer(aliases[config.LAYER_FILES["mask"]])
    layers: dict[str, gpd.GeoDataFrame] = {"mask": mask}
    for role, stem in config.LAYER_FILES.items():
        if role == "mask":
            continue
        layers[role] = to_crs_safe(load_layer(aliases[stem]), mask.crs)

    points = collect_points(mask.crs, aliases)
    return layers, points
