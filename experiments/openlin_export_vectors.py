"""Векторный экспорт наборов линий в GeoPackage (ТЗ К1, Пакет 01).

ЗАЧЕМ. Тракт хранит итог двумя способами: маской 20 м (``openlin_zone_lines_*.npz``,
``openlin_dem_lines_*.npz``) и поштучно — ячейками каждой линии
(``*_линии.npz``, :func:`openlin_zone_lines.save_line_sets`). Маска для
вектора не годится: касающиеся линии в ней сливаются. Экспорт идёт с
поштучного файла.

КАК ЯЧЕЙКИ ЛИНИИ СТАНОВЯТСЯ ПОЛИЛИНИЕЙ. Ячейки линии — неупорядоченный набор
(после сшивки и продолжения концов в нём бывают утолщения и отростки, после
«цепочек» — разрывы на месте убранных мостиков). Порядок восстанавливается
графом, без сглаживания и без подбора:
* вершины — ячейки, рёбра — 8-соседство, вес — расстояние между центрами
  (1 или sqrt 2 ячейки);
* в каждой связной части берётся минимальное остовное дерево (убирает
  треугольники на утолщениях);
* дерево режется на пути: сначала самый длинный (диаметр), потом самый
  длинный из оставшихся отростков и так далее, пока не кончатся рёбра;
* каждый путь — одна часть MultiLineString через центры ячеек.
Ни одна ячейка не теряется и не добавляется; это проверяется обратной
растеризацией (см. ``check``). Длина вектора — длина этих путей; длина по
трассе — из тракта (``len_trace_km``); обе пишутся в атрибуты.

Запуск из корня:
    python -m experiments.openlin_export_vectors                  # все площадки
    python -m experiments.openlin_export_vectors --site finland65  # одна, прочие не трогаются
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
COMP = Path("datacache/s2_lowsun/composite")
OUT = Path("outputs/k1/vectors")

#: (площадка, файл маски, файл поштучно, ключ набора, слой GeoPackage, описание)
SETS = (
    ("anabar", "openlin_zone_lines_anabar_v3_бл", "C_all", "optical_full",
     "снимок, весь лист (рабочий набор TWIN)"),
    ("anabar", "openlin_zone_lines_anabar_v3_бл", "C", "optical_zones",
     "снимок, в зонах согласия (TWIN)"),
    ("anabar", "openlin_dem_lines_anabar_одна", "C", "dem1",
     "рельеф DEM1, все подтверждённые"),
    ("anabar", "openlin_dem_lines_anabar_одна", "add", "dem1_added",
     "рельеф DEM1, добавленные к снимку"),
    ("lapland", "openlin_zone_lines_lapland_v3_перенос", "C_all", "auto_full",
     "перенос XFER, весь лист"),
    ("lapland", "openlin_zone_lines_lapland_v3_перенос", "C", "auto_zones",
     "перенос XFER, в зонах согласия"),
    ("finland65", "openlin_zone_lines_finland65_v3_перенос", "C_all", "auto_full",
     "перенос XFER2, весь лист"),
    ("finland65", "openlin_zone_lines_finland65_v3_перенос", "C", "auto_zones",
     "перенос XFER2, в зонах согласия"),
)


def cells_to_paths(cells: np.ndarray) -> list[np.ndarray]:
    """Пути (строка, столбец) через все ячейки линии — см. шапку."""
    import networkx as nx

    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    G = nx.Graph()
    G.add_nodes_from(range(len(cells)))
    for (r, c), i in idx.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = idx.get((r + dr, c + dc))
                if j is not None and j > i:
                    G.add_edge(i, j, weight=float(np.hypot(dr, dc)))
    paths = []
    for comp in nx.connected_components(G):
        T = nx.minimum_spanning_tree(G.subgraph(comp))
        if T.number_of_nodes() == 1:
            n = next(iter(T.nodes))
            paths.append(cells[[n, n]])          # одиночная ячейка — точка-отрезок
            continue
        while T.number_of_edges():
            # самый длинный путь в лесу T: две волны от произвольной вершины
            part = max(nx.connected_components(T), key=len)
            sub = T.subgraph(part)
            a = next(iter(part))
            d = nx.single_source_dijkstra_path_length(sub, a)
            a = max(d, key=d.get)
            d, pth = nx.single_source_dijkstra(sub, a)
            b = max(d, key=d.get)
            p = pth[b]
            paths.append(cells[p])
            T = nx.Graph(T)
            T.remove_edges_from(zip(p[:-1], p[1:]))
            T.remove_nodes_from([n for n in list(T.nodes) if T.degree(n) == 0])
    return paths


def export(site, stem, key, layer, desc, log):
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString

    w = np.load(COMP / f"winter_{site}_20m_лист.npz")
    res, x0, ytop, epsg = float(w["res_m"]), float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    d = np.load(M / f"{stem}_линии.npz")
    pts, ids = d[f"{key}_pts"], d[f"{key}_id"]
    lens, azs = d[f"{key}_len_km"], d[f"{key}_az"]
    rows = []
    for i in range(len(lens)):
        cells = np.unique(pts[ids == i], axis=0)
        parts = []
        for p in cells_to_paths(cells):
            xy = np.c_[x0 + (p[:, 1] + 0.5) * res, ytop - (p[:, 0] + 0.5) * res]
            parts.append(LineString(xy))
        geom = parts[0] if len(parts) == 1 else MultiLineString(parts)
        rows.append({"line_id": i, "site": site, "set": layer, "stage": "final",
                     "source": stem, "n_cells": len(cells), "n_parts": len(parts),
                     "len_trace_km": float(lens[i]),
                     "len_vector_km": geom.length / 1000, "azimuth_deg": float(azs[i]),
                     "geometry": geom})
    g = gpd.GeoDataFrame(rows, crs=f"EPSG:{epsg}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{site}_lines.gpkg"
    g.to_file(path, layer=layer, driver="GPKG")
    log(f"{site}/{layer} ({desc}): линий {len(g)}, по трассе {g.len_trace_km.sum():.0f} км, "
        f"по вектору {g.len_vector_km.sum():.0f} км; частей больше одной у "
        f"{int((g.n_parts > 1).sum())} линий -> {path}")
    return g, (res, x0, ytop, epsg), np.load(M / f"{stem}.npz")[key]


def check(g, grid, mask, log) -> dict:
    """Обратная растеризация: вектор должен дать ровно ячейки маски."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    res, x0, ytop, _e = grid
    H, W = mask.shape
    # центры ячеек лежат на вершинах путей: растеризуем вершины, а не линию,
    # чтобы диагональные шаги не задевали соседние ячейки
    r_all = []
    for geom in g.geometry:
        parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for part in parts:
            xy = np.asarray(part.coords)
            r_all.append(np.c_[np.floor((ytop - xy[:, 1]) / res),
                               np.floor((xy[:, 0] - x0) / res)].astype(int))
    rc = np.unique(np.concatenate(r_all), axis=0)
    back = np.zeros((H, W), bool)
    back[rc[:, 0], rc[:, 1]] = True
    same = bool((back == mask[:H, :W]).all())
    out = {"ячеек_маски": int(mask.sum()), "ячеек_вектора": int(back.sum()),
           "лишних": int((back & ~mask).sum()), "потерянных": int((mask & ~back).sum()),
           "совпадает": same}
    log(f"  обратная растеризация: маска {out['ячеек_маски']}, вектор "
        f"{out['ячеек_вектора']}, лишних {out['лишних']}, потерянных "
        f"{out['потерянных']} -> {'СОВПАДАЕТ' if same else 'НЕ совпадает'}")
    return out


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="только эта площадка; остальные файлы не трогаются")
    only = ap.parse_args().site

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    rep_path = OUT / "export_check.json"
    report = json.loads(rep_path.read_text(encoding="utf-8")) if only and rep_path.exists() else {}
    sets = [s_ for s_ in SETS if only in (None, s_[0])]
    for site in {s_[0] for s_ in sets}:
        p = OUT / f"{site}_lines.gpkg"
        if p.exists():
            p.unlink()
    for site, stem, key, layer, desc in sets:
        g, grid, mask = export(site, stem, key, layer, desc, log)
        report[f"{site}/{layer}"] = check(g, grid, mask, log) | {
            "линий": len(g), "км_по_трассе": round(float(g.len_trace_km.sum()), 1),
            "км_по_вектору": round(float(g.len_vector_km.sum()), 1), "epsg": grid[3]}
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    log(f"\nсохранено: {OUT / 'export_check.json'} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
