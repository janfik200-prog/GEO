"""Зонд гиперспектральных архивов EnMAP и PRISMA над листом R-48-XI,XII.

Гиперспектр (0.4-2.5 мкм, ~230 каналов) — единственный дистанционный способ
различить серицит, каолинит и карбонаты по ФОРМЕ полос поглощения; ASTER на шести
широких каналах SWIR этого не может. Но обе миссии снимают ПО ЗАЯВКАМ, а не
сплошным покрытием, поэтому вопрос «есть ли данные на лист» решается зондом, а не
предположением — как это уже было с ASTER (``experiments/probe_aster.py``).

Считается не число сцен, а доля площади листа: полоса EnMAP 30 км при размере
листа ~75x77 км физически не может покрыть его одной сценой. Пересечение
считается в проекции сетки (метры), а не в градусах.

Результат зонда 05.08.2026:

- EnMAP: над листом 5 сцен L2A, объединение 40.9 % площади. Летних малооблачных —
  НОЛЬ: всё пригодное снято 22.05.2025, а маска QUALITY_SNOW самой миссии даёт
  на этих сценах 49-54 % снега; единственные летние (27.07.2022) закрыты облаками
  на 91-98 %. Малооблачная бесснежная сцена 04.09.2022 не дотянула до края листа
  0.9 км. Вывод: архива для признаков нет, но район снимается — вопрос решается
  заявкой на съёмку (EnMAP принимает заявки бесплатно).
- PRISMA: каталог ASI анонимно не опрашивается (302 на логин), нужен аккаунт.

Запуск из корня: ``python -m experiments.probe_hyperspectral``.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, integro_grid  # noqa: E402
from src.s2_composite import bbox_wgs84  # noqa: E402

DLR_STAC = "https://geoservice.dlr.de/eoc/ogc/stac/v1/search"
ENMAP_COLLECTIONS = ("ENMAP_HSI_L2A", "ENMAP_HSI_L0_QL")
SUMMER_MONTHS = (6, 7, 8, 9)
MAX_CLOUD = 30.0
PRISMA_PORTAL = "https://prisma.asi.it/"


def sheet_polygon():
    """Контур листа в проекции сетки + трансформер из WGS84 в неё."""
    from pyproj import CRS, Transformer
    from shapely.geometry import box

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    crs = CRS.from_proj4(proj4)
    x0, x1 = meta.x0, meta.x0 + meta.pic * meta.dx
    y1, y0 = meta.y_top, meta.y_top - meta.prf * meta.dy
    tr = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    return box(x0, y0, x1, y1), tr, meta


def search_enmap(collection: str, bbox: list[float]) -> list[dict]:
    """Сцены коллекции DLR STAC над охватом листа (анонимно, без регистрации)."""
    params = {"collections": collection, "limit": "500",
              "bbox": ",".join(f"{v:.4f}" for v in bbox)}
    with urllib.request.urlopen(f"{DLR_STAC}?{urllib.parse.urlencode(params)}",
                                timeout=90) as resp:
        return json.load(resp)["features"]


def _cloud(props: dict) -> float | None:
    """Облачность STAC: у DLR приходит строкой, отсутствие — это не 0, а None."""
    v = props.get("eo:cloud_cover")
    return None if v is None else float(v)


def coverage(features: list[dict], sheet, tr) -> tuple[float, list[tuple]]:
    """Доля листа, покрытая объединением сцен, и вклад каждой сцены."""
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union

    rows, geoms = [], []
    for f in features:
        g = transform(lambda x, y: tr.transform(x, y), shape(f["geometry"]))
        inter = g.intersection(sheet)
        if inter.is_empty:
            continue
        geoms.append(inter)
        p = f["properties"]
        rows.append((p.get("datetime", "")[:10], _cloud(p),
                     100.0 * inter.area / sheet.area, f["id"]))
    union = unary_union(geoms).area / sheet.area * 100.0 if geoms else 0.0
    rows.sort(key=lambda r: -r[2])
    return union, rows


def near_misses(features: list[dict], sheet, tr) -> list[tuple]:
    """Сцены, прошедшие МИМО листа, с расстоянием до его края.

    Важны для вердикта: если малооблачная летняя сцена не дотянула считанные
    километры, значит район снимается, и вопрос решается заявкой на съёмку,
    а не отсутствием интереса миссии к широте.
    """
    from shapely.geometry import shape
    from shapely.ops import transform

    out = []
    for f in features:
        g = transform(lambda x, y: tr.transform(x, y), shape(f["geometry"]))
        if g.intersects(sheet):
            continue
        p = f["properties"]
        out.append((p.get("datetime", "")[:10], _cloud(p),
                    g.distance(sheet) / 1000.0))
    out.sort(key=lambda r: r[2])
    return out


def snow_fraction(feature: dict) -> float | None:
    """Доля пикселей со снегом по маске QUALITY_SNOW самой миссии.

    На 71 гр. с.ш. дата съёмки сама по себе ничего не доказывает — снег меряется
    по маске продукта, а не предполагается по календарю.
    """
    import numpy as np
    import rasterio

    href = feature.get("assets", {}).get("QUALITY_SNOW", {}).get("href")
    if not href:
        return None
    try:
        with rasterio.open(href) as src:
            a = src.read(1)
    except Exception:
        return None
    return float(np.mean(a == 1) * 100.0)


def probe_prisma() -> str:
    """Доступен ли каталог PRISMA анонимно (ожидаемо — нет, нужен аккаунт ASI)."""
    req = urllib.request.Request(PRISMA_PORTAL, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            return f"портал отвечает {resp.status}, но поиск за логином ASI"
    except urllib.error.HTTPError as e:
        return f"портал отвечает {e.code}, но поиск за логином ASI"
    except Exception as e:                        # сеть/TLS — фиксируем как есть
        return f"каталог недоступен анонимно ({type(e).__name__}: {e})"


def main() -> None:
    sheet, tr, meta = sheet_polygon()
    bbox = bbox_wgs84(meta, pad_m=0.0)
    print(f"Лист: {meta.pic}x{meta.prf} ячеек по {meta.dx:.0f} м = "
          f"{meta.pic * meta.dx / 1000:.1f}x{meta.prf * meta.dy / 1000:.1f} км, "
          f"площадь {sheet.area / 1e6:.0f} км²")
    print(f"Охват WGS84: {bbox[0]:.2f}..{bbox[2]:.2f} в.д., "
          f"{bbox[1]:.2f}..{bbox[3]:.2f} с.ш.\n")

    for col in ENMAP_COLLECTIONS:
        feats = search_enmap(col, bbox)
        union, rows = coverage(feats, sheet, tr)
        print(f"=== {col}: сцен над листом {len(rows)} из {len(feats)} в охвате ===")
        print(f"{'дата':>10} {'облачн.':>8} {'покрытие листа':>15}")
        for date, cloud, frac, _id in rows:
            c = "н/д" if cloud is None else f"{cloud:.0f}%"
            print(f"{date:>10} {c:>8} {frac:>14.1f}%")
        print(f"объединение всех сцен: {union:.1f}% площади листа")

        good = [f for f in feats
                if int(f["properties"]["datetime"][5:7]) in SUMMER_MONTHS
                and (_cloud(f["properties"]) if _cloud(f["properties"]) is not None
                     else 100.0) <= MAX_CLOUD]
        gunion, grows = coverage(good, sheet, tr)
        print(f"из них летних (VI-IX) с облачностью <= {MAX_CLOUD:.0f}%: "
              f"{len(grows)} сцен, объединение {gunion:.1f}%")

        miss = near_misses(feats, sheet, tr)
        if miss:
            print("прошли мимо листа:")
            for date, cloud, km in miss:
                c = "н/д" if cloud is None else f"{cloud:.0f}%"
                print(f"  {date}  облачность {c:>5}  не дотянула {km:.1f} км")
        print()

    print("=== снег по маске QUALITY_SNOW (доля пикселей сцены) ===")
    for f in search_enmap("ENMAP_HSI_L0_QL", bbox):
        snow = snow_fraction(f)
        if snow is not None:
            print(f"  {f['properties']['datetime'][:10]}  снег {snow:5.1f}%  "
                  f"облачность {_cloud(f['properties'])}%")

    print(f"\n=== PRISMA (ASI) ===\n{probe_prisma()}")


if __name__ == "__main__":
    main()
