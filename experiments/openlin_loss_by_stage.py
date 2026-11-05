"""Где теряются линии эталона: разбор по этапам тракта (площадка настройки).

ЗАЧЕМ. На встрече 28.10.2026 сшивка названа «самым слабым звеном». Сверка с
GTK меряет только итог (полноту); чинить сшивку есть смысл, только если
эталонные линии теряются именно на ней. Здесь каждая ячейка эталона (GTK 292
на сетке 20 м) относится к первому этапу, на котором рядом с ней кончается
наш след:

  * «итог»       — рядом подтверждённая линия набора (C_all);
  * «подтверждение» — рядом линия после сшивки (A_all), но она не подтвердилась;
  * «сшивка»     — рядом есть сырой обрывок трассировки того же простирания,
                    но в линию набора он не вырос (сшивка, отсев по длине,
                    отсев швов);
  * «детектор»   — рядом нет даже сырого обрывка того же простирания.

«Рядом» — ``TOL_M`` = 300 м, тот же допуск, что у сверки
(:mod:`openlin_transfer_check`, из масштаба эталона). «Того же простирания» —
разница азимутов не больше ``DIR_TOL_DEG`` = 20° (тот же допуск, что у зон
согласия, :mod:`openlin_zone_lines`). Сырые обрывки плотные, поэтому каждая
доля печатается рядом с той же долей для эталона, сдвинутого по тору
(``N_SHIFTS`` сдвигов, поля как у ``shift_null_conf``): осмысленна только
разница с нулём.

Только замер на площадке НАСТРОЙКИ (``openlin_probe.SITES["utsjoki"]``): в
счётчик не идёт, пороги не меняет. На контрольной Лапландии не запускать —
разбор потерь по её эталону и есть подстройка под контроль.

    python -m experiments.openlin_loss_by_stage --site utsjoki --tag _настройка0
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.openlin_transfer_check import GTK, TOL_M
from experiments.openlin_zone_lines import COMP, DIR_TOL_DEG, M, trace

TUNE_SITES = ("utsjoki",)
N_SHIFTS = 50
SHIFT_SEED = 23
STEP_M = 20.0           # шаг точек вдоль линии эталона — шаг сетки


def gtk_points(w: dict, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Точки линий GTK 292 с шагом сетки: (ряд, столбец), азимут, номер линии."""
    import geopandas as gpd
    from shapely.geometry import box

    res, x0, y_top = float(w["res_m"]), float(w["x0"]), float(w["y_top"])
    H, W = shape
    g = gpd.read_file(GTK / "lin500k_292.gpkg").to_crs(epsg=int(w["epsg"]))
    g = g.clip(box(x0, y_top - H * res, x0 + W * res, y_top))
    rc, az, fid = [], [], []
    for i, geom in enumerate(g.geometry):
        parts = getattr(geom, "geoms", [geom])
        for p in parts:
            xy = np.asarray(p.coords)[:, :2]
            for (xa, ya), (xb, yb) in zip(xy[:-1], xy[1:]):
                L = np.hypot(xb - xa, yb - ya)
                n = max(int(L // STEP_M), 1)
                t = (np.arange(n) + 0.5) / n
                xs, ys = xa + t * (xb - xa), ya + t * (yb - ya)
                r = ((y_top - ys) / res).astype(int)
                c = ((xs - x0) / res).astype(int)
                ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                rc.append(np.stack([r[ok], c[ok]], 1))
                az.append(np.full(ok.sum(), np.degrees(np.arctan2(xb - xa, yb - ya)) % 180))
                fid.append(np.full(ok.sum(), i))
    return np.concatenate(rc), np.concatenate(az), np.concatenate(fid)


def near_mask(pts, mask, res, tol):
    """Есть ли ячейка маски ближе tol (по расстоянию, без направления)."""
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(~mask, sampling=(res, res))
    return d[pts[:, 0], pts[:, 1]] <= tol


def near_aligned(pts, az, tree, raw_az, res, tol, dir_tol):
    """Есть ли сырой обрывок ближе tol с разницей азимутов не больше dir_tol."""
    out = np.zeros(len(pts), bool)
    for i, nb in enumerate(tree.query_ball_point(pts * res, tol)):
        if nb:
            d = np.abs(raw_az[nb] - az[i]) % 180
            out[i] = (np.minimum(d, 180 - d) <= dir_tol).any()
    return out


def tiers(pts, az, d, tree, raw_az, res):
    c = near_mask(pts, d["C_all"], res, TOL_M)
    a_ = near_mask(pts, d["A_all"], res, TOL_M)
    r = near_aligned(pts, az, tree, raw_az, res, TOL_M, DIR_TOL_DEG)
    return {"итог": c, "подтверждение": a_ & ~c, "сшивка": r & ~a_ & ~c,
            "детектор": ~r & ~a_ & ~c}


def main() -> None:
    from scipy.spatial import cKDTree

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--tag", default="_настройка0")
    a = ap.parse_args()
    if a.site not in TUNE_SITES:
        raise SystemExit("разбор потерь — только на площадке настройки")

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    d = dict(np.load(M / f"openlin_zone_lines_{a.site}_v3{a.tag}.npz"))
    res = float(w["res_m"])
    shape = d["C_all"].shape
    opt = w["shade"][:shape[0], :shape[1]].astype("float32")
    valid = np.isfinite(opt)

    raw = trace(opt, valid, res, local=True, min_len_m=0)
    raw_pts = np.concatenate([r["pts"] for r in raw])
    raw_az = np.concatenate([np.full(len(r["pts"]), r["азимут"]) for r in raw])
    tree = cKDTree(raw_pts * res)
    print(f"площадка {a.site}: сырых обрывков {len(raw)}, ячеек {len(raw_pts)} "
          f"(~{len(raw_pts) * res / 1000:.0f} км)", flush=True)

    pts, az, fid = gtk_points(w, shape)
    ok = valid[pts[:, 0], pts[:, 1]]
    pts, az, fid = pts[ok], az[ok], fid[ok]
    print(f"эталон GTK 292: линий {len(np.unique(fid))}, точек {len(pts)} "
          f"(~{len(pts) * STEP_M / 1000:.0f} км)", flush=True)

    obs = {k: v.mean() for k, v in tiers(pts, az, d, tree, raw_az, res).items()}
    H, W = shape
    rng = np.random.default_rng(SHIFT_SEED)
    null = {k: [] for k in obs}
    for _ in range(N_SHIFTS):
        lo_r, lo_c = min(200, H // 8), min(200, W // 8)
        sh = np.stack([(pts[:, 0] + rng.integers(lo_r, H - lo_r)) % H,
                       (pts[:, 1] + rng.integers(lo_c, W - lo_c)) % W], 1)
        keep = valid[sh[:, 0], sh[:, 1]]
        for k, v in tiers(sh[keep], az[keep], d, tree, raw_az, res).items():
            null[k].append(v.mean())

    print(f"\nгде кончается след у ячеек эталона (допуск {TOL_M:.0f} м, "
          f"простирание {DIR_TOL_DEG:.0f}°; нуль — {N_SHIFTS} сдвигов эталона):")
    for k in obs:
        n = np.array(null[k])
        print(f"  {k:14s} {100 * obs[k]:5.1f} %   нуль {100 * n.mean():5.1f} % "
              f"(5–95 %: {100 * np.percentile(n, 5):.1f}–{100 * np.percentile(n, 95):.1f})   "
              f"разница {100 * (obs[k] - n.mean()):+5.1f}")

    t = tiers(pts, az, d, tree, raw_az, res)
    rows = []
    for f in np.unique(fid):
        m = fid == f
        rows.append({"линия": int(f), "длина_км": m.sum() * STEP_M / 1000,
                     **{k: float(v[m].mean()) for k, v in t.items()}})
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("длина_км", ascending=False)
    out = M / f"openlin_loss_by_stage_{a.site}{a.tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nпо линиям эталона (длиннее 10 км, доли длины):")
    print(df[df["длина_км"] >= 10].round(2).head(25).to_string(index=False))
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
