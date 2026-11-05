"""Проверочный (не тракт): раздельные и пересекающиеся доли длины по артефактам, Лапландия.

Дополнение к openlin_artifact_lapland: диаграмма Венна по длине (дороги+ЛЭП
вдоль, берег вдоль, край леса вдоль), нуль объединения сдвигами, доля зоны у
берега (почему набор В в зонах так береговой), длины объяснённых линий.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter
from pyproj import Transformer
import experiments.openlin_artifact_lapland as m

def main():
    z = np.load(m.NPZ); w = np.load(m.COMP)
    x0, yt, epsg = float(w["x0"]), float(w["y_top"]), int(w["epsg"])
    shape = z["C"].shape; H, W = shape
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    wc = np.load(m.WC)["wc"]
    am = np.zeros(shape, bool); aa = np.full(shape, -1, np.int16)
    for n in ("roads", "power"):
        L = m.osm_layer(n, shape, x0, yt, epsg, tr)
        am |= L["mask"]; aa = np.where(L["mask"], L["ang"], aa)
    layers = {}
    nr, na = m.make_near(am, aa); layers["дороги+ЛЭП"] = {"near": nr, "nang": na}
    water = wc == 80
    shore = water & ~binary_erosion(water, np.ones((3, 3), bool))
    g = gaussian_filter(water.astype("float32"), 2.0); gr, gc = np.gradient(g)
    ang = np.where(shore, np.round(m.az_of(-gc, gr)).astype(np.int16) % 180, -1).astype(np.int16)
    nr, na = m.make_near(shore, ang); layers["берег"] = {"near": nr, "nang": na}
    tree = (wc == 10).astype("float32"); land = ~water
    edge = ((wc == 10) & binary_dilation(land & (wc != 10), np.ones((3, 3), bool))) | \
           (land & (wc != 10) & binary_dilation(wc == 10, np.ones((3, 3), bool)))
    g = gaussian_filter(np.where(land, tree, 0.5).astype("float32"), 2.0); gr, gc = np.gradient(g)
    ang = np.where(edge, np.round(m.az_of(-gc, gr)).astype(np.int16) % 180, -1).astype(np.int16)
    nr, na = m.make_near(edge, ang); layers["край леса"] = {"near": nr, "nang": na}
    zone = z["zone"]
    print(f"доля зоны: {100 * zone.mean():.1f} % листа; у берега (60 м) в зоне "
          f"{100 * layers['берег']['near'][zone].mean():.1f} % против {100 * layers['берег']['near'].mean():.1f} % по листу; "
          f"у края леса в зоне {100 * layers['край леса']['near'][zone].mean():.1f} % против {100 * layers['край леса']['near'].mean():.1f} %")
    rng = np.random.default_rng(m.SEED)
    sh = [(int(rng.integers(200, H - 200)), int(rng.integers(200, W - 200))) for _ in range(m.N_SHIFT)]
    names = list(layers)
    for sn in ("C_all", "C"):
        cells = m.Cells(z[sn])
        obs, fl = m.stats(cells, shape, layers)
        ln = cells.cnt * m.RES / 1000
        F = {k: fl[k] for k in names}
        tot = ln.sum()
        print(f"\n=== {sn}: {cells.n} кусков, {tot:.0f} км")
        for k in names:
            print(f"  {k:12s} вдоль: {100 * ln[F[k]].sum() / tot:5.1f} % длины; медиана длины куска {np.median(ln[F[k]]) if F[k].any() else 0:.2f} км "
                  f"(остальные {np.median(ln[~F[k]]):.2f} км)")
        U = F["дороги+ЛЭП"] | F["берег"]
        U3 = U | F["край леса"]
        only = {k: F[k] & ~np.logical_or.reduce([F[j] for j in names if j != k]) for k in names}
        print("  только один: " + "; ".join(f"{k} {100 * ln[only[k]].sum() / tot:.1f}" for k in names)
              + f"; пересечения (>=2): {100 * ln[(sum(F[k].astype(int) for k in names) >= 2)].sum() / tot:.1f} %")
        print(f"  объединение дороги+ЛЭП/берег: {100 * ln[U].sum() / tot:.1f} %; +край леса: {100 * ln[U3].sum() / tot:.1f} %; "
              f"не объяснено: {100 * ln[~U3].sum() / tot:.1f} %")
        nu2, nu3 = [], []
        nk = {k: [] for k in names}
        for dr, dc in sh:
            _, f = m.stats(cells, shape, layers, dr, dc)
            u = f["дороги+ЛЭП"] | f["берег"]
            nu2.append(ln[u].sum() / tot); nu3.append(ln[u | f["край леса"]].sum() / tot)
            for k in names: nk[k].append(ln[f[k]].sum() / tot)
        for k in names:
            print(f"  нуль {k}: {100 * np.mean(nk[k]):.1f} % (95-й {100 * np.quantile(nk[k], .95):.1f})")
        print(f"  нуль объединения: {100 * np.mean(nu2):.1f} % / с краем леса {100 * np.mean(nu3):.1f} % "
              f"(95-й {100 * np.quantile(nu3, .95):.1f}); избыток {100 * (ln[U].sum() / tot - np.mean(nu2)):.1f} / "
              f"{100 * (ln[U3].sum() / tot - np.mean(nu3)):.1f} п.п.")
if __name__ == "__main__":
    main()
