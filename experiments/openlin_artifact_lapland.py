"""Проверочный (не тракт): приборные и ложные прямые в наборе линий Лапландии (XFER).

Только замер: пороги тракта не трогаются, в тракт ничего не идёт, эталон GTK не
используется. Слои (все вне входа тракта, только для проверки):
  * швы плиток снимка — смена поля count композита;
  * границы плиток ЦМР — целые градусы (батарея openlin_artifact_battery);
  * дороги, ЛЭП, просеки, железная дорога — OSM через Overpass
    (openlin_artifact_lapland_osm.py, кэш datacache/osm);
  * вода, болото, граница леса — WorldCover 2021, 10 м -> сетка листа 20 м
    (openlin_artifact_lapland_wc.py, кэш datacache/worldcover/wc_lapland_20m.npz).

Мера везде одна: доля ячеек линий (или длины линий, признанных объяснёнными) в
слое против доли ПЛОЩАДИ в слое и сдвигового нуля (линии едут по тору, слой
стоит). «Объясняемая доля» = наблюдаемая минус среднее нуля.

Запуск:  python -m experiments.openlin_artifact_lapland
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           gaussian_filter, label)

from experiments.openlin_artifact_battery import components, dem_tile_edges

NPZ = Path("outputs/metrics/openlin_zone_lines_lapland_v3_перенос.npz")
COMP = Path("datacache/s2_lowsun/composite/winter_lapland_20m_лист.npz")
WC = Path("datacache/worldcover/wc_lapland_20m.npz")
OSM = Path("datacache/osm")
RES = 20.0
NEAR_C = 3                  # 60 м = CONF_TOL_M тракта
ALIGN_DEG = 20.0            # DIR_TOL_DEG тракта
FRAC = 0.5                  # доля ячеек линии у слоя, чтобы линия считалась объяснённой
N_SHIFT = 2000
SEED = 23


def angdiff(a, b):
    return np.abs(((a - b) + 90.0) % 180.0 - 90.0)


def az_of(dr, dc):
    return np.degrees(np.arctan2(dc, -dr)) % 180.0


def osm_layer(name, shape, x0, yt, epsg, tr, split=None):
    """Растр линий OSM: маска и угол (0..179, -1 нет) ближайшего сегмента."""
    from PIL import Image, ImageDraw

    f = OSM / f"lapland_{name}.json"
    if not f.exists():
        return None
    els = json.loads(f.read_text(encoding="utf-8"))["elements"]
    H, W = shape
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    n = 0
    for e in els:
        if split and not split(e.get("tags", {})):
            continue
        g = e.get("geometry")
        if not g or len(g) < 2:
            continue
        lon = [p["lon"] for p in g]
        lat = [p["lat"] for p in g]
        x, y = tr.transform(lon, lat)
        c = (np.array(x) - x0) / RES
        r = (yt - np.array(y)) / RES
        for i in range(len(c) - 1):
            a = int(round(az_of(r[i + 1] - r[i], c[i + 1] - c[i]))) % 180 + 1
            dr.line([(c[i], r[i]), (c[i + 1], r[i + 1])], fill=a, width=1)
        n += 1
    a = np.array(img, dtype=np.int16)
    mask = a > 0
    ang = np.where(mask, a - 1, -1).astype(np.int16)
    return {"mask": mask, "ang": ang, "n_ways": n}


def make_near(mask, ang=None, k=NEAR_C):
    """near: ячейка ближе k ячеек к слою; nang: угол ближайшей ячейки слоя."""
    if ang is None:
        d = distance_transform_edt(~mask)
        return d <= k, None
    d, ind = distance_transform_edt(~mask, return_indices=True)
    nang = ang[ind[0], ind[1]]
    return d <= k, nang


class Cells:
    """Ячейки набора линий: координаты, номер куска, азимут куска."""

    def __init__(self, mask):
        lab, n = label(mask, np.ones((3, 3), bool))
        r, c = np.nonzero(mask)
        self.r, self.c = r, c
        self.comp = lab[r, c] - 1
        self.n = n
        cnt = np.bincount(self.comp, minlength=n).astype(float)
        self.cnt = cnt
        az = np.zeros(n)
        # азимут по SVD куска (как components) — считаем векторно
        order = np.argsort(self.comp, kind="stable")
        bounds = np.searchsorted(self.comp[order], np.arange(n + 1))
        for i in range(n):
            idx = order[bounds[i]:bounds[i + 1]]
            q = np.stack([r[idx], c[idx]], 1).astype(float)
            q -= q.mean(0)
            if len(q) < 2:
                continue
            vt = np.linalg.svd(q, full_matrices=False)[2]
            az[i] = az_of(vt[0][0], vt[0][1])
        self.az = az
        self.N = len(r)


def stats(cells, shape, layers, dr=0, dc=0):
    """Доли по слоям при сдвиге (dr, dc). Возвращает словарь и объединения флагов."""
    H, W = shape
    rr = (cells.r + dr) % H
    cc = (cells.c + dc) % W
    out = {}
    flags = {}
    for name, L in layers.items():
        near = L["near"][rr, cc]
        out[name + "|ячейки"] = near.mean()
        frac = np.bincount(cells.comp, near, minlength=cells.n) / cells.cnt
        f = frac >= FRAC
        out[name + "|длина_у"] = cells.cnt[f].sum() / cells.N
        if L.get("nang") is not None:
            ok = near & (angdiff(L["nang"][rr, cc], cells.az[cells.comp]) <= ALIGN_DEG)
            fa = np.bincount(cells.comp, ok, minlength=cells.n) / cells.cnt >= FRAC
            out[name + "|длина_вдоль"] = cells.cnt[fa].sum() / cells.N
            flags[name] = fa
        else:
            flags[name] = f
    return out, flags


def main():
    t0 = time.time()

    def log(m=""):
        print(m, flush=True)

    from pyproj import Transformer

    z = np.load(NPZ)
    w = np.load(COMP)
    x0, yt, epsg = float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    count = w["count"].astype("int16")
    shape = z["C"].shape
    H, W = shape
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    wc = np.load(WC)["wc"]

    layers: dict[str, dict] = {}
    # --- швы снимка
    seam = np.zeros(shape, bool)
    seam[:, :-1] |= count[:, :-1] != count[:, 1:]
    seam[:-1, :] |= count[:-1, :] != count[1:, :]
    layers["шов снимка"] = {"near": make_near(seam)[0], "area_mask": seam}
    # --- границы плиток ЦМР
    edges, box = dem_tile_edges(shape, RES, x0, yt, epsg)
    layers["граница ЦМР"] = {"near": make_near(edges)[0]} if edges.any() else None
    if layers["граница ЦМР"] is None:
        del layers["граница ЦМР"]
    log(f"швы count: {100 * seam.mean():.2f} % ячеек; границы градусов "
        f"lon {box[0]:.0f}..{box[1]:.0f} lat {box[2]:.0f}..{box[3]:.0f}: "
        f"{int(edges.sum())} ячеек")
    # --- OSM
    anth_mask = np.zeros(shape, bool)
    anth_ang = np.full(shape, -1, np.int16)
    for name, split in (("roads", None), ("power", None), ("cutline", None),
                        ("railway", None)):
        L = osm_layer(name, shape, x0, yt, epsg, tr, split)
        if L is None:
            log(f"OSM {name}: файла нет — слой не проверен")
            continue
        log(f"OSM {name}: {L['n_ways']} путей, {100 * L['mask'].mean():.2f} % ячеек")
        near, nang = make_near(L["mask"], L["ang"])
        layers["OSM " + name] = {"near": near, "nang": nang}
        anth_mask |= L["mask"]
        anth_ang = np.where(L["mask"], L["ang"], anth_ang)
    if anth_mask.any():
        near, nang = make_near(anth_mask, anth_ang)
        layers["OSM всё линейное"] = {"near": near, "nang": nang}
    # --- вода, болото, граница леса (WorldCover 20 м)
    water = wc == 80
    shore = water & ~binary_erosion(water, np.ones((3, 3), bool))
    g = gaussian_filter(water.astype("float32"), 2.0)
    gr, gc = np.gradient(g)
    sh_ang = np.where(shore, np.round(az_of(-gc, gr)).astype(np.int16) % 180, -1).astype(np.int16)
    near, nang = make_near(shore, sh_ang)
    layers["берег (WC 80)"] = {"near": near, "nang": nang}
    inside = water & (distance_transform_edt(water) > NEAR_C)
    layers["внутри воды"] = {"near": inside}
    wet = wc == 90
    layers["болото (WC 90)"] = {"near": wet & (distance_transform_edt(wet) > NEAR_C)}
    land = ~water
    tree = wc == 10
    edge = (tree & binary_dilation(land & ~tree, np.ones((3, 3), bool))) | \
           (land & ~tree & binary_dilation(tree, np.ones((3, 3), bool)))
    layers["граница леса/открытого"] = {"near": make_near(edge)[0]}
    for k in (3, 6, 12):
        layers[f"count={k}"] = {"near": count == k}
    log(f"вода {100 * water.mean():.1f} %, берег {100 * shore.mean():.2f} %, "
        f"лес {100 * tree.mean():.1f} %; ({time.time() - t0:.0f} с)")
    area = {k: float(L["near"].mean()) for k, L in layers.items()}

    rng = np.random.default_rng(SEED)
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    shifts = [(int(rng.integers(lo_r, H - lo_r)), int(rng.integers(lo_c, W - lo_c)))
              for _ in range(N_SHIFT)]

    res_all = {}
    for set_name in ("C_all", "C", "A_all"):
        cells = Cells(z[set_name])
        obs, fl = stats(cells, shape, layers)
        nulls = {k: [] for k in obs}
        un_null = []
        explain = [k for k in fl if k in ("шов снимка", "OSM всё линейное",
                                          "берег (WC 80)", "внутри воды")]
        un_obs = np.zeros(cells.n, bool)
        for k in explain:
            un_obs |= fl[k]
        for dr, dc in shifts:
            o, f = stats(cells, shape, layers, dr, dc)
            for k, v in o.items():
                nulls[k].append(v)
            u = np.zeros(cells.n, bool)
            for k in explain:
                u |= f[k]
            un_null.append(cells.cnt[u].sum() / cells.N)
        log(f"\n=== набор {set_name}: {cells.n} кусков, {cells.N} ячеек "
            f"(~{cells.N * RES / 1000:.0f} км), ({time.time() - t0:.0f} с)")
        log(f"{'слой':26s} {'площадь':>8s} {'ячейки':>8s} {'нуль':>7s} {'x':>5s} {'p':>7s} | "
            f"{'длина у слоя':>12s} {'нуль':>6s} | {'длина вдоль':>11s} {'нуль':>6s}")
        for k in layers:
            a = area[k]
            v = obs[k + "|ячейки"]
            nu = np.array(nulls[k + "|ячейки"])
            p = (1 + (nu >= v).sum()) / (1 + N_SHIFT)
            s = (f"{k:26s} {100 * a:7.2f}% {100 * v:7.2f}% {100 * nu.mean():6.2f}% "
                 f"{v / max(a, 1e-9):5.2f} {p:7.4f} | "
                 f"{100 * obs[k + '|длина_у']:11.1f}% {100 * np.mean(nulls[k + '|длина_у']):5.1f}%")
            if k + "|длина_вдоль" in obs:
                s += (f" | {100 * obs[k + '|длина_вдоль']:10.1f}% "
                      f"{100 * np.mean(nulls[k + '|длина_вдоль']):5.1f}%")
            log(s)
        uo = cells.cnt[un_obs].sum() / cells.N
        un = np.array(un_null)
        log(f"объединение (шов, OSM вдоль, берег вдоль, внутри воды): {100 * uo:.1f} % длины, "
            f"нуль {100 * un.mean():.1f} % (95-й {100 * np.quantile(un, 0.95):.1f} %), "
            f"избыток {100 * (uo - un.mean()):.1f} п.п., p = {(1 + (un >= uo).sum()) / (1 + N_SHIFT):.4f}")
        res_all[set_name] = (cells, fl, un_obs)

    # --- кривизна и азимуты (набор C_all, C)
    for set_name in ("C_all", "C"):
        L = components(z[set_name], RES)
        k = np.array([r["кривизна"] for r in L])
        ln = np.array([r["длина_м"] for r in L]) / 1000
        azs = np.array([r["азимут"] for r in L])
        q = np.percentile(k, [10, 50, 90])
        ultra = (k < 0.005) & (ln >= 3.0)
        log(f"\nкривизна {set_name}: 10/50/90 % = {100 * q[0]:.2f}/{100 * q[1]:.2f}/{100 * q[2]:.2f} %, "
            f"длина куска макс {ln.max():.1f} км, ультрапрямых >3 км: {int(ultra.sum())}; "
            f">=3 км всего: {int((ln >= 3).sum())}")
        wts = ln
        t = np.radians(2 * azs)
        R = np.hypot((np.sin(t) * wts).sum(), (np.cos(t) * wts).sum()) / wts.sum()
        h = np.histogram(azs, bins=12, range=(0, 180), weights=wts)[0]
        log(f"азимуты {set_name}: R = {R:.2f}; доли длины по секторам 15° (0..180): "
            + " ".join(f"{100 * v / h.sum():.0f}" for v in h))
    np.savez_compressed("outputs/metrics/openlin_artifact_lapland.npz",
                        flags_road=res_all["C_all"][1].get("OSM всё линейное", np.zeros(1)),
                        flags_shore=res_all["C_all"][1]["берег (WC 80)"],
                        flags_water=res_all["C_all"][1]["внутри воды"],
                        flags_seam=res_all["C_all"][1]["шов снимка"])
    log(f"\nготово ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
