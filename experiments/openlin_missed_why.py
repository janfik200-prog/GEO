"""Почему не найдена часть эталона: разбор пропущенных ячеек по причинам.

ЗАЧЕМ (Артур, 05.11.2026). Снимок + рельеф находят 40 % эталона GTK 292
(:mod:`openlin_ref_match`, с учётом направления). Здесь каждая ПРОПУЩЕННАЯ
ячейка эталона относится к первой подходящей причине:

  1. «под другим углом»  — в TOL_M есть наша итоговая линия, но разница
     направлений больше DIR_TOL_DEG (эталон провёл прямую через изогнутую форму);
  2. «сдвиг 300–600 м»   — параллельная итоговая линия есть, но дальше TOL_M и
     ближе 2*TOL_M (положение эталона известно с точностью масштаба);
  3. «видели, но отсеяли» — в TOL_M есть параллельный СЫРОЙ обрывок трассировки
     (снимок или рельеф, без сшивки, подтверждения и порога длины);
  4. «есть форма, детектор не видит» — сырых обрывков нет, но под линией
     заметная ложбина, уступ или контраст снимка: сильнее 90-го процентиля тех
     же величин на случайных местах (нуль — сдвинутый эталон);
  5. «на наших данных не видно» — ничего из этого.

Каждая доля печатается рядом с той же долей у сдвинутого эталона (случайные
места под теми же углами): осмысленна разница. Только замер на площадке
настройки, тракт не меняется, в счётчик не идёт.
    python -m experiments.openlin_missed_why --site utsjoki
"""
from __future__ import annotations

import argparse

import numpy as np

from cache_paths import ANABAR_DEM
from experiments.openlin_missed_ref import CORE_M, EDGE_M, profiles, shape_numbers
from experiments.openlin_ref_match import (DIR_TOL_DEG, TOL_M, _axial_diff, our_directions,
                                           positions, ref_directions, shifts)

N_NULL = 5               # сдвигов для нуля причин (из общего списка меры)
VIS_Q = 90               # «заметно» — сильнее 90-го процентиля случайных мест


def near_any(rc, tree, res, radius):
    """Есть ли наша ячейка ближе radius (без направления)."""
    d, _ = tree.query(rc * res, k=1, distance_upper_bound=radius)
    return np.isfinite(d)


def near_aligned(rc, ang, tree, our_ang, res, radius):
    """Есть ли наша ячейка ближе radius с разницей направлений <= DIR_TOL_DEG."""
    from scipy.spatial import cKDTree
    out = np.zeros(len(rc), bool)
    t = cKDTree(rc * res)
    pairs = t.sparse_distance_matrix(tree, radius, output_type="ndarray")
    if len(pairs):
        ok = _axial_diff(ang[pairs["i"]], our_ang[pairs["j"]]) <= DIR_TOL_DEG
        out[np.unique(pairs["i"][ok])] = True
    return out


def tree_of(mask, ang_map, res):
    from scipy.spatial import cKDTree
    rr, cc = np.nonzero(mask & np.isfinite(ang_map))
    return cKDTree(np.stack([rr, cc], 1) * res), ang_map[rr, cc]


def raw_fragments(img, valid, res, resp_fn):
    """Сырые обрывки трассировки (без порога длины): маска и угол ячеек."""
    from experiments.openlin_zone_lines import trace
    L = trace(img, valid, res, local=True, min_len_m=0, response_fn=resp_fn)
    m = np.zeros(valid.shape, bool)
    ang = np.full(valid.shape, np.nan, "float32")
    for x in L:
        m[x["pts"][:, 0], x["pts"][:, 1]] = True
        ang[x["pts"][:, 0], x["pts"][:, 1]] = x["азимут"]
    return m, ang


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    a = ap.parse_args()
    if a.site != "utsjoki":
        raise SystemExit("только площадка настройки")

    from experiments.openlin_relief_lines import load_slopes, scarp_response
    from experiments.openlin_transfer_check import COMP, M, MAIN_LAYER
    from experiments.openlin_union_recall import add_new

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    o = np.load(M / f"openlin_zone_lines_{a.site}_v3_настройка1.npz")["C_all"]
    r = np.load(M / f"openlin_zone_lines_{a.site}_v3_рельеф1.npz")["C_all"]
    res = float(w["res_m"])
    shape = o.shape
    shade = w["shade"][:shape[0], :shape[1]].astype("float32")
    valid = np.isfinite(shade)
    union = o | add_new(r, o, res)
    ang_ref, fid = ref_directions(MAIN_LAYER, w, shape)
    ang_ref = np.where(valid, ang_ref, np.nan)
    pos = positions(ang_ref, fid, valid, shifts(shape)[:N_NULL])

    print("наши итоговые линии: направления...", flush=True)
    t_fin, a_fin = tree_of(union, our_directions(union, res), res)
    print("сырые обрывки: снимок и рельеф...", flush=True)
    m1, g1 = raw_fragments(shade, valid, res, None)
    slope, _ = load_slopes(a.site, res)
    m2, g2 = raw_fragments(np.where(valid, slope, np.nan), valid & np.isfinite(slope), res,
                           scarp_response)
    raw = m1 | m2
    g = np.where(m1, g1, g2)
    t_raw, a_raw = tree_of(raw, g, res)

    dem = np.load(ANABAR_DEM / f"dem_glo30_{a.site}_лист_20m.npy")
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    sh0 = np.where(valid, shade, 0.0)
    rng = np.random.default_rng(5)

    def features(rc, ang):
        p, x = profiles(dem, rc, ang, res, rng, len(rc))
        tr, st = shape_numbers(p, x)
        ps, _ = profiles(sh0, rc, ang, res, rng, len(rc))
        con = np.abs(ps[:, np.abs(x) <= CORE_M].mean(1)
                     - (ps[:, x <= -EDGE_M].mean(1) + ps[:, x >= EDGE_M].mean(1)) / 2)
        return tr, st, con

    # пороги «заметно» — по случайным местам (сдвинутый эталон)
    rcn = np.concatenate([p[0] for p in pos[1:]])
    agn = np.concatenate([p[1] for p in pos[1:]])
    trn, stn, conn = features(rcn, agn)
    thr = dict(tr=np.percentile(trn, VIS_Q), st=np.percentile(stn, VIS_Q),
               con=np.percentile(conn, VIS_Q))
    print(f"«заметно» (90-й процентиль случайных мест): ложбина > {thr['tr']:.1f} м, "
          f"уступ > {thr['st']:.1f} м, контраст снимка > {thr['con']:.3f}")

    names = ["найдено", "под другим углом", "сдвиг 300–600 м", "видели, но отсеяли",
             "есть форма, детектор не видит", "на наших данных не видно"]

    def classify(rc, ang):
        found = near_aligned(rc, ang, t_fin, a_fin, res, TOL_M)
        anyfin = near_any(rc, t_fin, res, TOL_M)
        off = near_aligned(rc, ang, t_fin, a_fin, res, 2 * TOL_M)
        rawn = near_aligned(rc, ang, t_raw, a_raw, res, TOL_M)
        tr, st, con = features(rc, ang)
        vis = (tr > thr["tr"]) | (st > thr["st"]) | (con > thr["con"])
        cat = np.full(len(rc), 5)
        for k, m in reversed(list(enumerate([found, anyfin, off, rawn, vis]))):
            cat[m] = k
        return cat

    c0 = classify(pos[0][0], pos[0][1])
    cn = np.concatenate([classify(p[0], p[1]) for p in pos[1:]])
    print(f"\nячейки эталона по причинам (весь лист, снимок + рельеф; нуль — "
          f"{N_NULL} сдвигов эталона):")
    rows = []
    for k, n in enumerate(names):
        ob, nu = (c0 == k).mean(), (cn == k).mean()
        print(f"  {n:32s} {100 * ob:5.1f} %   случайно {100 * nu:5.1f} %   "
              f"разница {100 * (ob - nu):+5.1f}")
        rows.append(dict(причина=n, доля=ob, нуль=nu))
    miss = c0 > 0
    print("\nсреди пропущенных (100 %):")
    for k in range(1, 6):
        print(f"  {names[k]:32s} {100 * (c0[miss] == k).mean():5.1f} %")
    import pandas as pd
    out = M / f"openlin_missed_why_{a.site}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    np.save(M / f"openlin_missed_why_cat_{a.site}.npy", c0)
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
