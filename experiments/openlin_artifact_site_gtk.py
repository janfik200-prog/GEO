"""Проверочный (не тракт), ТОЛЬКО ДЛЯ СВЕДЕНИЯ, площадка настройки: какая доля длины эталона
GTK 292 лежит вдоль берегов и русел.

Зачем: будущий отсев береговых дуг не должен выбросить настоящие линеаменты. Эталон
в отбор линий не идёт, здесь он только измеряется теми же слоями и теми же допусками
(60 м, 20 °), что и наш набор. GTK режется на куски длиной с медиану кусков нашего
набора C_all (замер, не круглое число); кривизна куска и порог дуги — как в
openlin_artifact_site (90-й процентиль кривизны кусков вдали от воды). Нуль — сдвиг
точек GTK по тору внутри рамки валидных ячеек (слои стоят).
Запуск:  python -m experiments.openlin_artifact_site_gtk --site utsjoki
"""
from __future__ import annotations
import argparse
import numpy as np
import experiments.openlin_artifact_site as m

GTK = "datacache/gtk_lineaments/lin500k_292.gpkg"
N_SH = 1000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--site", default="utsjoki")
    a = ap.parse_args()
    import geopandas as gpd
    from shapely.geometry import box
    d = m.load_site(a.site, log=lambda s="": print(s, flush=True))
    layers, shape, z = d["layers"], d["shape"], d["z"]
    H, W = shape
    cref = m.Cells(z["C_all"])
    piece_km = float(np.median(cref.len_km[cref.len_km > 0]))
    wn = layers["вода: берег или русло"]["near"]
    cr = m.Cells(z["C_all"] | z["A_all"])
    fw = np.bincount(cr.comp, wn[cr.r, cr.c], minlength=cr.n) / cr.cnt
    far = (fw == 0) & (cr.len_km >= 0.5) & np.isfinite(cr.curv)
    k_arc = float(np.percentile(cr.curv[far], 90))
    print(f"кусок GTK = {piece_km:.2f} км (медиана C_all), порог дуги {100 * k_arc:.2f} %", flush=True)
    g = gpd.read_file(GTK).to_crs(epsg=d["epsg"])
    x0, yt = d["x0"], d["yt"]
    bx = box(x0, yt - H * m.RES, x0 + W * m.RES, yt)
    g = g[g.intersects(bx)]
    pts_r, pts_c, pid, paz, pcurv, plen = [], [], [], [], [], []
    n = 0
    for geom in g.geometry:
        geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for ls in geoms:
            L = ls.length
            if L < 3 * m.RES:
                continue
            s = np.arange(0, L, m.RES)
            xy = np.array([ls.interpolate(t).coords[0] for t in s])
            c = (xy[:, 0] - x0) / m.RES; r = (yt - xy[:, 1]) / m.RES
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            step = max(int(round(piece_km * 1000 / m.RES)), 8)
            for i0 in range(0, len(s), step):
                sl = slice(i0, min(i0 + step, len(s)))
                if len(s[sl]) < max(step // 2, 6) or not ok[sl].all():
                    continue
                q = np.stack([r[sl], c[sl]], 1)
                qc = q - q.mean(0)
                vt = np.linalg.svd(qc, full_matrices=False)[2]
                t = qc @ vt[0]
                Lm = (t.max() - t.min()) * m.RES
                if Lm <= 0:
                    continue
                pts_r.append(np.floor(q[:, 0]).astype(int)); pts_c.append(np.floor(q[:, 1]).astype(int))
                pid.append(np.full(len(q), n)); paz.append(m.az_of(vt[0][0], vt[0][1]))
                pcurv.append(float(np.abs(qc @ vt[1]).mean()) * m.RES / Lm); plen.append(Lm / 1000)
                n += 1
    R = np.concatenate(pts_r); C = np.concatenate(pts_c); P = np.concatenate(pid)
    paz = np.array(paz); pcurv = np.array(pcurv); plen = np.array(plen)
    cnt = np.bincount(P, minlength=n).astype(float)
    print(f"GTK 292: линий в листе {len(g)}, кусков {n}, ~{cnt.sum() * m.RES / 1000:.0f} км по кускам", flush=True)

    def flags(dr, dc, name):
        L = layers[name]
        rr = (R + dr) % H; cc = (C + dc) % W
        near = L["near"][rr, cc]
        fu = np.bincount(P, near, minlength=n) / cnt >= m.FRAC
        ok = near & (m.angdiff(L["nang"][rr, cc], paz[P]) <= m.ALIGN_DEG)
        fa = np.bincount(P, ok, minlength=n) / cnt >= m.FRAC
        return fu, fa

    names = ["берег (WC 80)", "русло OSM", "вода: берег или русло", "OSM техногенное (всё)", "край леса"]
    names = [k for k in names if k in layers]
    rng = np.random.default_rng(m.SEED)
    sh = [(int(rng.integers(200, H - 200)), int(rng.integers(200, W - 200))) for _ in range(N_SH)]
    tot = cnt.sum()
    print(f"\n{'слой':26s} {'площадь':>8s} | {'длина у слоя':>12s} {'нуль':>6s} | {'длина вдоль':>11s} {'нуль':>6s} {'95-й':>6s}")
    obs_fa = {}
    for k in names:
        fu, fa = flags(0, 0, k)
        obs_fa[k] = fa
        nu = [], []
        for dr, dc in sh:
            u, v = flags(dr, dc, k)
            nu[0].append(cnt[u].sum() / tot); nu[1].append(cnt[v].sum() / tot)
        print(f"{k:26s} {100 * layers[k]['near'].mean():7.2f}% | {100 * cnt[fu].sum() / tot:11.1f}% "
              f"{100 * np.mean(nu[0]):5.1f}% | {100 * cnt[fa].sum() / tot:10.1f}% {100 * np.mean(nu[1]):5.1f}% "
              f"{100 * np.quantile(nu[1], .95):5.1f}%", flush=True)
    fa = obs_fa["вода: берег или русло"]
    arc = fa & (pcurv > k_arc); st = fa & (pcurv <= k_arc)
    print(f"\nGTK вдоль воды (берег или русло): {100 * cnt[fa].sum() / tot:.1f} % длины; из неё дуг "
          f"(кривизна куска > {100 * k_arc:.2f} %) {100 * cnt[arc].sum() / max(cnt[fa].sum(), 1):.0f} %, "
          f"прямых {100 * cnt[st].sum() / max(cnt[fa].sum(), 1):.0f} %")
    print(f"отсев «вода вдоль и дуга» выбросил бы {100 * cnt[arc].sum() / tot:.1f} % длины GTK; "
          f"отсев «вода вдоль» целиком — {100 * cnt[fa].sum() / tot:.1f} %")
    print(f"кривизна кусков GTK 10/50/90 %: " + "/".join(f"{100 * np.percentile(pcurv, q):.2f}" for q in (10, 50, 90))
          + f" %; доля кусков с кривизной > порога дуги вообще: {100 * cnt[pcurv > k_arc].sum() / tot:.1f} % длины "
          f"(у воды {100 * cnt[arc].sum() / max(cnt[fa].sum(), 1):.0f} %, вдали {100 * cnt[(~fa) & (pcurv > k_arc)].sum() / max(cnt[~fa].sum(), 1):.0f} %)")
    print(f"доля нашего набора C_all вдоль воды — см. основной прогон.")


if __name__ == "__main__":
    main()
