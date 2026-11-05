"""Проверочный (не тракт): приборные и ложные прямые в наборе линий любой площадки.

Обобщение openlin_artifact_lapland / _excl (те не тронуты, работают как прежде).
Только замер: пороги тракта не трогаются, в тракт ничего не идёт. Эталон GTK 292
используется ТОЛЬКО в отдельном блоке «для сведения» (площадка настройки) и в
отборе не участвует. Слои (все вне входа тракта):
  * шов снимка (смена count внутри данных) и КРАЙ ДАННЫХ (граница дыры композита);
  * границы плиток ЦМР (целые градусы);
  * OSM: дороги, ЛЭП, просеки, железная дорога; русла (waterway) — datacache/osm;
  * WorldCover 10 м -> 20 м: берег воды (80), внутри воды, болото (90), край леса (10);
  * дыры композита: сетка обрезается по рамке валидных ячеек, сдвиговый нуль идёт
    по тору ВНУТРИ рамки (иначе линии уезжали бы в дыру, где линий быть не может).

Мера та же, что у Лапландии: доля ячеек линий (и длины кусков, признанных
объяснёнными: >= 50 % ячеек куска в 60 м от слоя и, где слой линейный, угол
<= 20 °) против сдвигового нуля. Допуски 60 м / 20 ° — CONF_TOL_M / DIR_TOL_DEG тракта.
Береговая дуга отличается от прямого участка русла кривизной: порог дуги — 90-й
процентиль кривизны кусков вдали от воды (замер, не круглое число).

Запуск:  python -m experiments.openlin_artifact_site --site utsjoki
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, distance_transform_edt,
                           gaussian_filter, label)

from experiments.openlin_artifact_battery import dem_tile_edges

RES = 20.0
NEAR_C = 3                  # 60 м = CONF_TOL_M тракта
ALIGN_DEG = 20.0            # DIR_TOL_DEG тракта
FRAC = 0.5
N_SHIFT = 2000
SEED = 23
OSM = Path("datacache/osm")
NPZ_BY_SITE = {
    "utsjoki": "outputs/metrics/openlin_zone_lines_utsjoki_v3_настройка1.npz",
    "lapland": "outputs/metrics/openlin_zone_lines_lapland_v3_перенос.npz",
}


def angdiff(a, b):
    return np.abs(((a - b) + 90.0) % 180.0 - 90.0)


def az_of(dr, dc):
    return np.degrees(np.arctan2(dc, -dr)) % 180.0


def comp_path(site):
    return Path(f"datacache/s2_lowsun/composite/winter_{site}_20m_лист.npz")


def wc_path(site):
    return Path(f"datacache/worldcover/wc_{site}_20m.npz")


def osm_layer(site, name, shape, x0, yt, tr, keep=None):
    """Растр линий OSM: маска и угол (0..179, -1 нет)."""
    from PIL import Image, ImageDraw
    f = OSM / f"{site}_{name}.json"
    if not f.exists():
        return None
    els = json.loads(f.read_text(encoding="utf-8"))["elements"]
    H, W = shape
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    n = 0
    for e in els:
        if keep and not keep(e.get("tags", {})):
            continue
        g = e.get("geometry")
        if not g or len(g) < 2:
            continue
        x, y = tr.transform([p["lon"] for p in g], [p["lat"] for p in g])
        c = (np.array(x) - x0) / RES
        r = (yt - np.array(y)) / RES
        for i in range(len(c) - 1):
            a = int(round(az_of(r[i + 1] - r[i], c[i + 1] - c[i]))) % 180 + 1
            dr.line([(c[i], r[i]), (c[i + 1], r[i + 1])], fill=a, width=1)
        n += 1
    a = np.array(img, dtype=np.int16)
    mask = a > 0
    return {"mask": mask, "ang": np.where(mask, a - 1, -1).astype(np.int16), "n_ways": n}


def make_near(mask, ang=None, k=NEAR_C):
    if ang is None:
        return distance_transform_edt(~mask) <= k, None
    d, ind = distance_transform_edt(~mask, return_indices=True)
    return d <= k, ang[ind[0], ind[1]]


def grad_ang(field, mask):
    """Угол границы по градиенту сглаженного поля, в ячейках mask."""
    g = gaussian_filter(field.astype("float32"), 2.0)
    gr, gc = np.gradient(g)
    return np.where(mask, np.round(az_of(-gc, gr)).astype(np.int16) % 180, -1).astype(np.int16)


class Cells:
    """Куски набора: ячейки, номер куска, азимут, кривизна, длина (км), центр."""

    def __init__(self, mask):
        lab, n = label(mask, np.ones((3, 3), bool))
        r, c = np.nonzero(mask)
        self.r, self.c = r, c
        self.comp = lab[r, c] - 1
        self.n = n
        self.cnt = np.bincount(self.comp, minlength=n).astype(float)
        self.cr = np.bincount(self.comp, r, minlength=n) / np.maximum(self.cnt, 1)
        self.cc = np.bincount(self.comp, c, minlength=n) / np.maximum(self.cnt, 1)
        az = np.zeros(n); curv = np.full(n, np.nan); ln = np.zeros(n)
        order = np.argsort(self.comp, kind="stable")
        b = np.searchsorted(self.comp[order], np.arange(n + 1))
        for i in range(n):
            idx = order[b[i]:b[i + 1]]
            if len(idx) < 2:
                continue
            q = np.stack([r[idx], c[idx]], 1).astype(float)
            q -= q.mean(0)
            vt = np.linalg.svd(q, full_matrices=False)[2]
            t = q @ vt[0]
            L = (t.max() - t.min()) * RES
            az[i] = az_of(vt[0][0], vt[0][1])
            if L > 0:
                ln[i] = L / 1000
                curv[i] = float(np.abs(q @ vt[1]).mean()) * RES / L
        self.az, self.curv, self.len_km = az, curv, ln
        self.N = len(r)


def stats(cells, layers, shape, dr=0, dc=0):
    H, W = shape
    rr = (cells.r + dr) % H
    cc = (cells.c + dc) % W
    out, flags = {}, {}
    for name, L in layers.items():
        near = L["near"][rr, cc]
        out[name + "|ячейки"] = near.mean()
        f = np.bincount(cells.comp, near, minlength=cells.n) / cells.cnt >= FRAC
        out[name + "|длина_у"] = cells.cnt[f].sum() / cells.N
        if L.get("nang") is not None:
            ok = near & (angdiff(L["nang"][rr, cc], cells.az[cells.comp]) <= ALIGN_DEG)
            fa = np.bincount(cells.comp, ok, minlength=cells.n) / cells.cnt >= FRAC
            out[name + "|длина_вдоль"] = cells.cnt[fa].sum() / cells.N
            flags[name] = fa
        else:
            flags[name] = f
    return out, flags


def load_site(site, npz=None, log=print):
    """Всё нужное для проверки: наборы, слои (обрезаны по рамке валидных ячеек)."""
    from pyproj import Transformer
    z = dict(np.load(npz or NPZ_BY_SITE[site]))
    w = np.load(comp_path(site))
    x0, yt, epsg = float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    shade = w["shade"]; count = w["count"].astype("int16")
    valid = np.isfinite(shade)
    rows = np.where(valid.any(1))[0]; cols = np.where(valid.any(0))[0]
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    Hf, Wf = valid.shape
    log(f"{site}: сетка {Hf}x{Wf}, валидных {100 * valid.mean():.1f} %, рамка валидных ряды {r0}..{r1} "
        f"столбцы {c0}..{c1} ({100 * (r1 - r0) * (c1 - c0) / valid.size:.1f} % сетки); "
        f"внутри рамки пропусков {100 * (~valid[r0:r1, c0:c1]).mean():.2f} %")
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    wc = np.load(wc_path(site))["wc"]
    crop = (slice(r0, r1), slice(c0, c1))
    shape = (r1 - r0, c1 - c0)
    d = {"site": site, "z": {k: v[crop] for k, v in z.items() if v.shape == valid.shape},
         "x0": x0 + c0 * RES, "yt": yt - r0 * RES, "epsg": epsg, "shape": shape,
         "crop": (r0, c0), "shade": shade[crop], "count": count[crop], "valid": valid[crop],
         "wc": wc[crop], "sun": np.asarray(w["azimuth"]), "tr": tr}
    layers: dict[str, dict] = {}
    # --- шов (внутри данных) и край данных
    cnt, val = d["count"], d["valid"]
    seam = np.zeros(shape, bool)
    dx = (cnt[:, :-1] != cnt[:, 1:]) & val[:, :-1] & val[:, 1:]
    dy = (cnt[:-1, :] != cnt[1:, :]) & val[:-1, :] & val[1:, :]
    seam[:, :-1] |= dx; seam[:-1, :] |= dy
    edge_data = val & ~binary_erosion(val, np.ones((3, 3), bool), border_value=1)
    edge_data |= binary_dilation(~val, np.ones((3, 3), bool)) & val
    layers["шов снимка (внутри данных)"] = {"near": make_near(seam)[0]}
    layers["край данных"] = {"near": make_near(edge_data)[0]} if edge_data.any() else None
    if layers["край данных"] is None:
        del layers["край данных"]
    # --- границы ЦМР
    edges, box = dem_tile_edges(shape, RES, d["x0"], d["yt"], epsg)
    if edges.any():
        layers["граница ЦМР"] = {"near": make_near(edges)[0]}
    log(f"шов {100 * seam.mean():.2f} % ячеек, край данных {100 * edge_data.mean():.2f} %, границы "
        f"градусов {int(edges.sum())} ячеек")
    # --- OSM (растрируем на обрезанной сетке)
    anth_m = np.zeros(shape, bool); anth_a = np.full(shape, -1, np.int16)
    for name in ("roads", "power", "cutline", "railway"):
        L = osm_layer(site, name, shape, d["x0"], d["yt"], tr)
        if L is None:
            log(f"OSM {name}: файла нет"); continue
        log(f"OSM {name}: {L['n_ways']} путей, {100 * L['mask'].mean():.3f} % ячеек")
        near, nang = make_near(L["mask"], L["ang"])
        layers["OSM " + name] = {"near": near, "nang": nang}
        anth_m |= L["mask"]; anth_a = np.where(L["mask"], L["ang"], anth_a)
    if anth_m.any():
        near, nang = make_near(anth_m, anth_a)
        layers["OSM техногенное (всё)"] = {"near": near, "nang": nang}
    d["anth"] = (anth_m, anth_a)
    # --- русла OSM
    Lw = osm_layer(site, "waterway", shape, d["x0"], d["yt"], tr)
    chan_m = Lw["mask"] if Lw else np.zeros(shape, bool)
    chan_a = Lw["ang"] if Lw else np.full(shape, -1, np.int16)
    if Lw:
        log(f"OSM waterway: {Lw['n_ways']} путей, {100 * chan_m.mean():.3f} % ячеек")
        near, nang = make_near(chan_m, chan_a)
        layers["русло OSM"] = {"near": near, "nang": nang}
    # --- WorldCover
    wcc = d["wc"]
    water = wcc == 80
    shore = water & ~binary_erosion(water, np.ones((3, 3), bool))
    sh_a = grad_ang(water, shore)
    near, nang = make_near(shore, sh_a)
    layers["берег (WC 80)"] = {"near": near, "nang": nang}
    layers["внутри воды"] = {"near": water & (distance_transform_edt(water) > NEAR_C)}
    wm = np.where(shore, sh_a, np.where(chan_m, chan_a, -1)).astype(np.int16)
    near, nang = make_near(shore | chan_m, wm)
    layers["вода: берег или русло"] = {"near": near, "nang": nang}
    d["shore"] = shore; d["chan"] = (chan_m, chan_a); d["water"] = water
    wet = wcc == 90
    layers["болото (WC 90)"] = {"near": wet & (distance_transform_edt(wet) > NEAR_C)}
    land = ~water; tree = wcc == 10
    edge = (tree & binary_dilation(land & ~tree, np.ones((3, 3), bool))) | \
           (land & ~tree & binary_dilation(tree, np.ones((3, 3), bool)))
    e_a = grad_ang(np.where(land, tree, 0.5), edge)
    near, nang = make_near(edge, e_a)
    layers["край леса"] = {"near": near, "nang": nang}
    d["forest_edge"] = (edge, e_a)
    d["layers"] = layers
    log("WorldCover, доля площади: " + ", ".join(
        f"{k} {100 * (wcc == k).mean():.1f}" for k in (10, 20, 30, 60, 80, 90, 100)))
    return d


def shifts_for(shape, n=N_SHIFT):
    H, W = shape
    rng = np.random.default_rng(SEED)
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    return [(int(rng.integers(lo_r, H - lo_r)), int(rng.integers(lo_c, W - lo_c))) for _ in range(n)]


EXPLAIN = ("шов снимка (внутри данных)", "край данных", "OSM техногенное (всё)",
           "вода: берег или русло", "внутри воды")


GROUPS_OPTICAL = {"техногенное": ["OSM техногенное (всё)"], "вода": ["вода: берег или русло"],
                  "край леса": ["край леса"], "шов/край данных": ["шов снимка (внутри данных)", "край данных"]}


def analyse(d, explain=EXPLAIN, groups=GROUPS_OPTICAL, tag="", log=print, t0=None, n_shift=N_SHIFT):
    """Основной замер: слои против сдвигового нуля, дуги/прямые у воды, Венн по длине, кривизна, азимуты."""
    global EXPLAIN
    EXPLAIN = explain
    t0 = t0 or time.time()
    layers, z, shape = d["layers"], d["z"], d["shape"]
    area = {k: float(L["near"].mean()) for k, L in layers.items()}
    shifts = shifts_for(shape, n_shift)
    log(f"слои готовы ({time.time() - t0:.0f} с)")
    # порог дуги: 90-й процентиль кривизны кусков вдали от воды (замер)
    water_near = layers["вода: берег или русло"]["near"]
    cells_ref = Cells(z["C_all"] | z["A_all"])
    fw = np.bincount(cells_ref.comp, water_near[cells_ref.r, cells_ref.c], minlength=cells_ref.n) / cells_ref.cnt
    far = (fw == 0) & (cells_ref.len_km >= 0.5) & np.isfinite(cells_ref.curv)
    k_arc = float(np.percentile(cells_ref.curv[far], 90))
    log(f"порог береговой дуги: 90-й процентиль кривизны кусков >= 0.5 км вдали от воды "
        f"(n = {int(far.sum())}) = {100 * k_arc:.2f} % (медиана {100 * np.median(cells_ref.curv[far]):.2f} %)")
    res = {}
    for sn in ("C_all", "C", "A_all", "A"):
        cells = Cells(z[sn])
        if cells.n == 0:
            continue
        obs, fl = stats(cells, layers, shape)
        nulls = {k: [] for k in obs}
        un_null, arc_null, arcsh_null = [], [], []
        for dr, dc in shifts:
            o, f = stats(cells, layers, shape, dr, dc)
            for k, v in o.items():
                nulls[k].append(v)
            u = np.zeros(cells.n, bool)
            for k in EXPLAIN:
                if k in f:
                    u |= f[k]
            un_null.append(cells.cnt[u].sum() / cells.N)
            wf = f["вода: берег или русло"]
            arc = wf & (cells.curv > k_arc)
            arc_null.append(cells.cnt[arc].sum() / cells.N)
            arcsh_null.append(cells.cnt[arc].sum() / max(cells.cnt[wf].sum(), 1))
        fo = fl["вода: берег или русло"]
        arc_o = fo & (cells.curv > k_arc)
        st_o = fo & (cells.curv <= k_arc)
        log(f"\n=== набор {sn}: {cells.n} кусков, {cells.N} ячеек (~{cells.N * RES / 1000:.0f} км) "
            f"({time.time() - t0:.0f} с)")
        log(f"{'слой':28s} {'площадь':>8s} {'ячейки':>8s} {'нуль':>7s} {'x':>5s} {'p':>7s} | "
            f"{'длина у слоя':>12s} {'нуль':>6s} | {'длина вдоль':>11s} {'нуль':>6s} {'95-й':>6s} {'p':>7s}")
        for k in layers:
            v = obs[k + "|ячейки"]; nu = np.array(nulls[k + "|ячейки"])
            p = (1 + (nu >= v).sum()) / (1 + n_shift)
            s = (f"{k:28s} {100 * area[k]:7.2f}% {100 * v:7.2f}% {100 * nu.mean():6.2f}% "
                 f"{v / max(area[k], 1e-9):5.2f} {p:7.4f} | "
                 f"{100 * obs[k + '|длина_у']:11.1f}% {100 * np.mean(nulls[k + '|длина_у']):5.1f}%")
            if k + "|длина_вдоль" in obs:
                nl = np.array(nulls[k + "|длина_вдоль"]); vo = obs[k + "|длина_вдоль"]
                s += (f" | {100 * vo:10.1f}% {100 * nl.mean():5.1f}% {100 * np.quantile(nl, .95):5.1f}% "
                      f"{(1 + (nl >= vo).sum()) / (1 + n_shift):7.4f}")
            log(s)
        uo = np.zeros(cells.n, bool)
        for k in EXPLAIN:
            if k in fl:
                uo |= fl[k]
        un = np.array(un_null); uv = cells.cnt[uo].sum() / cells.N
        log(f"объединение ({', '.join(EXPLAIN)}): {100 * uv:.1f} % длины, нуль {100 * un.mean():.1f} % "
            f"(95-й {100 * np.quantile(un, .95):.1f}), избыток {100 * (uv - un.mean()):.1f} п.п., "
            f"p = {(1 + (un >= uv).sum()) / (1 + n_shift):.4f}")
        # --- береговая дуга против прямого участка русла
        an, asn = np.array(arc_null), np.array(arcsh_null)
        log(f"вода вдоль: {100 * cells.cnt[fo].sum() / cells.N:.1f} % длины; из неё дуг (кривизна > "
            f"{100 * k_arc:.2f} %) {100 * cells.cnt[arc_o].sum() / max(cells.cnt[fo].sum(), 1):.1f} % "
            f"(нуль {100 * asn.mean():.1f} %, 95-й {100 * np.quantile(asn, .95):.1f}), прямых "
            f"{100 * cells.cnt[st_o].sum() / cells.N:.1f} % длины набора; дуг у воды {100 * cells.cnt[arc_o].sum() / cells.N:.1f} % "
            f"длины (нуль {100 * an.mean():.1f} %, избыток {100 * (cells.cnt[arc_o].sum() / cells.N - an.mean()):.1f} п.п.)")
        # --- раздельные доли (Венн по длине)
        grp = {}
        for gk, names_ in groups.items():
            gm = np.zeros(cells.n, bool)
            for nm_ in names_:
                if nm_ in fl:
                    gm |= fl[nm_]
            grp[gk] = gm
        ln = cells.cnt * RES / 1000; tot = ln.sum()
        nm = list(grp)
        ssum = sum(grp[k].astype(int) for k in nm)
        only = {k: grp[k] & (ssum == 1) for k in nm}
        log("  раздельно (только один): " + "; ".join(f"{k} {100 * ln[only[k]].sum() / tot:.1f}%" for k in nm)
            + f"; пересечения >=2: {100 * ln[ssum >= 2].sum() / tot:.1f}%; не объяснено ни одним: "
            f"{100 * ln[ssum == 0].sum() / tot:.1f}%")
        res[sn] = (cells, fl, uo)
    # --- кривизна, ультрапрямые, азимуты, открытая тундра против леса
    wcc = d["wc"]
    openm = np.isin(wcc, (20, 30, 60, 100)) & ~d["layers"]["край леса"]["near"] & \
        ~layers["вода: берег или русло"]["near"]
    forest = (wcc == 10) & ~d["layers"]["край леса"]["near"]
    sun = d["sun"]
    log(f"\nсолнце (азимут сцен): {sun.min():.0f}..{sun.max():.0f}°, среднее {sun.mean():.0f}°")
    for sn in ("C_all", "C"):
        if sn not in res:
            continue
        cells = res[sn][0]
        k = cells.curv; ln = cells.len_km
        ok = np.isfinite(k)
        q = np.percentile(k[ok], [10, 50, 90])
        ult = ok & (k < 0.005) & (ln >= 3.0)
        log(f"\nкривизна {sn}: 10/50/90 % = {100 * q[0]:.2f}/{100 * q[1]:.2f}/{100 * q[2]:.2f} %, "
            f"длина куска макс {ln.max():.1f} км, >=3 км: {int((ln >= 3).sum())}, ультрапрямых (<0.5 % и >=3 км): {int(ult.sum())}")
        for nmx, msk in (("вся", np.ones(cells.n, bool)), ("открытая", None), ("лес", None)):
            if nmx == "вся":
                sel = msk
            else:
                m2 = openm if nmx == "открытая" else forest
                sel = np.bincount(cells.comp, m2[cells.r, cells.c], minlength=cells.n) / cells.cnt >= 0.8
            if sel.sum() < 3:
                log(f"азимуты {sn} [{nmx}]: кусков {int(sel.sum())} — мало"); continue
            wts = ln[sel]; azs = cells.az[sel]
            t = np.radians(2 * azs)
            R = np.hypot((np.sin(t) * wts).sum(), (np.cos(t) * wts).sum()) / wts.sum()
            h = np.histogram(azs, bins=12, range=(0, 180), weights=wts)[0]
            log(f"азимуты {sn} [{nmx}] кусков {int(sel.sum())} ({wts.sum():.0f} км): R = {R:.2f}; секторы 15° (0..180): "
                + " ".join(f"{100 * v / h.sum():.0f}" for v in h))
    log(f"\nготово ({time.time() - t0:.0f} с)")
    return {"res": res, "k_arc": k_arc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--npz", default=None)
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m="": print(m, flush=True)
    d = load_site(a.site, a.npz, log)
    analyse(d, log=log, t0=t0)



if __name__ == "__main__":
    main()
