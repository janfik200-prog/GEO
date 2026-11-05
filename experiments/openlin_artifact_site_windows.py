"""Проверочный (не тракт): парные окна для глазной проверки артефактов площадки (снимок и рельеф).

Слева фон без разметки (снимок: зимний композит shade; рельеф: отмывка GLO-30), справа он же с линиями
C_all (красные — вне зон, синие — в зонах, набор C) и слоем-причиной (голубой — вода, зелёный — край леса,
жёлтый — граница плитки ЦМР, фиолетовый — край расхождения GLO-ALOS, оранжевый — дороги/ЛЭП, серый — плоско).
Жёлтый ореол — линия из подписи. Эталон GTK не накладывается. По 3 окна на каждый найденный тип,
окна не ближе 6 км друг к другу. Папка outputs/линеаменты_проверка_utsjoki_артефакты/.
Запуск:  python -m experiments.openlin_artifact_site_windows --site utsjoki --set снимок|рельеф
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.openlin_artifact_site as m
import experiments.openlin_artifact_relief as rel
from experiments.openlin_shade import _stretch

WIN_KM = 3.5
MIN_SEP_KM = 6.0
PER_TYPE = 3
COLORS = {"вода": (0.0, 0.8, 0.8, 0.45), "край леса": (0.1, 0.75, 0.2, 0.45),
          "граница плитки ЦМР": (1.0, 0.85, 0.0, 0.55), "края расхождения": (0.7, 0.2, 0.9, 0.35),
          "техногенное": (1.0, 0.5, 0.0, 0.6), "плоско": (0.5, 0.5, 0.5, 0.30),
          "шов/край данных": (1.0, 0.0, 1.0, 0.5)}


def hillshade(g, az=315.0, alt=45.0, res=20.0, zf=3.0):
    gy, gx = np.gradient(g, res)
    slope = np.arctan(zf * np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    a, h = np.radians(360 - az + 90), np.radians(alt)
    return np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - asp - np.pi / 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--set", dest="kind", choices=("снимок", "рельеф"), required=True)
    a = ap.parse_args()
    out = Path(f"outputs/линеаменты_проверка_{a.site}_артефакты")
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob(f"{a.kind}_*.png"):
        f.unlink()
    log = lambda s="": None
    npz = m.NPZ_BY_SITE[a.site] if a.kind == "снимок" else rel.NPZ.format(site=a.site)
    d = m.load_site(a.site, npz, log)
    layers, z, shape = d["layers"], d["z"], d["shape"]
    H, W = shape
    if a.kind == "рельеф":
        rel.add_relief_layers(d, log)
        g = d["dems"][0]
        img = hillshade(g)
        img = (img - np.nanpercentile(img, 1)) / (np.nanpercentile(img, 99) - np.nanpercentile(img, 1))
        img = np.clip(img, 0, 1)
    else:
        img = _stretch(np.where(np.isfinite(d["shade"]), d["shade"], np.nan))
    C, Cz = z["C_all"], z["C"]
    cells = m.Cells(C)
    _, fl = m.stats(cells, layers, shape)
    inz = np.bincount(cells.comp, Cz[cells.r, cells.c], minlength=cells.n) / cells.cnt >= 0.5
    wn = layers["вода: берег или русло"]["near"]
    cr_ = m.Cells(z["C_all"] | z["A_all"])
    fw = np.bincount(cr_.comp, wn[cr_.r, cr_.c], minlength=cr_.n) / cr_.cnt
    far = (fw == 0) & (cr_.len_km >= 0.5) & np.isfinite(cr_.curv)
    k_arc = float(np.percentile(cr_.curv[far], 90))
    water_f = fl["вода: берег или русло"]
    anth = fl["OSM техногенное (всё)"]
    forest = fl["край леса"]
    ult = np.isfinite(cells.curv) & (cells.curv < 0.005) & (cells.len_km >= 3.0)
    known = water_f | anth | forest
    # (тип: кандидаты, слой для показа)
    T = {}
    T["берег или русло: дуга"] = (water_f & (cells.curv > k_arc), "вода")
    T["берег или русло: прямой участок"] = (water_f & (cells.curv <= k_arc), "вода")
    if "OSM roads" in fl:
        T["дороги"] = (fl["OSM roads"] & ~water_f, "техногенное")
    if "OSM power" in fl:
        T["ЛЭП"] = (fl["OSM power"] & ~water_f, "техногенное")
    T["край леса"] = (forest & ~water_f & ~anth, "край леса")
    T["ультрапрямые"] = (ult, "вода")
    if a.kind == "снимок":
        T["шов/край данных"] = (fl["шов снимка (внутри данных)"] | fl.get("край данных", np.zeros(cells.n, bool)),
                                "шов/край данных")
    else:
        T["граница плитки ЦМР"] = (fl["граница ЦМР"], "граница плитки ЦМР")
        T["края расхождения GLO-ALOS (полосы)"] = (fl["края расхождения GLO-ALOS"] & ~known, "края расхождения")
        T["плоско (квантование ALOS)"] = (fl["плоско (квантование ALOS)"] & ~known, "плоско")
        sect = (m.angdiff(cells.az, 21.8) <= m.ALIGN_DEG) | (m.angdiff(cells.az, 180 - 21.8) <= m.ALIGN_DEG)
        T["азимут следа TanDEM-X (22 или 158), вне слоёв"] = (
            sect & ~known & ~fl["граница ЦМР"] & ~fl["края расхождения GLO-ALOS"], "края расхождения")
    T["не объяснено (контроль)"] = (~(known | fl.get("шов снимка (внутри данных)", known)
                                      | (fl["граница ЦМР"] if "граница ЦМР" in fl else known)
                                      | (fl["края расхождения GLO-ALOS"] if "края расхождения GLO-ALOS" in fl else known)
                                      | (fl["плоско (квантование ALOS)"] if "плоско (квантование ALOS)" in fl else known)),
                                    "вода")
    n = int(WIN_KM * 1000 / m.RES); half = n // 2
    sep = MIN_SEP_KM * 1000 / m.RES
    items = []
    for kind, (cand, lay) in T.items():
        idx = np.where(cand)[0]
        idx = idx[np.argsort(-cells.cnt[idx])]
        taken = []
        for i in idx:
            if not (half <= cells.cr[i] < H - half and half <= cells.cc[i] < W - half):
                continue
            if any(np.hypot(cells.cr[i] - r_, cells.cc[i] - c_) < sep for r_, c_ in taken):
                continue
            taken.append((cells.cr[i], cells.cc[i])); items.append((kind, int(i), lay))
            if len(taken) == PER_TYPE:
                break
        print(f"{kind}: кандидатов {len(idx)}, окон {len(taken)}")
    layer_mask = {"вода": layers["вода: берег или русло"]["near"] & False}
    # показываем сами линии слоя (не расширенную окрестность)
    show = {"вода": d["shore"] | d["chan"][0], "край леса": d["forest_edge"][0],
            "техногенное": d["anth"][0]}
    edges_only = layers.get("граница ЦМР")
    if edges_only is not None:
        show["граница плитки ЦМР"] = edges_only["near"]
    for k_ in ("края расхождения GLO-ALOS", "плоско (квантование ALOS)", "шов снимка (внутри данных)"):
        if k_ in layers:
            show[{"края расхождения GLO-ALOS": "края расхождения", "плоско (квантование ALOS)": "плоско",
                  "шов снимка (внутри данных)": "шов/край данных"}[k_]] = layers[k_]["near"]
    thumbs = []
    for j, (kind, i, lay) in enumerate(items, 1):
        r, c = int(cells.cr[i]), int(cells.cc[i])
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        base = img[sl]
        ov = np.zeros(base.shape + (4,), "float32")
        if lay in show:
            ov[show[lay][sl]] = COLORS[lay]
        ov[C[sl]] = (0.85, 0.05, 0.05, 0.95)
        ov[Cz[sl]] = (0.10, 0.30, 0.95, 0.95)
        mine = cells.comp == i
        E, N = d["x0"] + c * m.RES, d["yt"] - r * m.RES
        title = (f"{kind}: линия {cells.cnt[i] * m.RES / 1000:.1f} км, азимут {cells.az[i]:.0f}°, "
                 f"кривизна {100 * cells.curv[i]:.2f} %, центр E {E:.0f} N {N:.0f}, "
                 f"{'в зоне' if inz[i] else 'вне зоны'}")
        fig, ax = plt.subplots(1, 2, figsize=(17, 8.6), facecolor="#fcfcfb")
        for k in (0, 1):
            ax[k].imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[k].set_xticks([]); ax[k].set_yticks([])
            px = 1000.0 / m.RES
            ax[k].plot([10, 10 + px], [base.shape[0] - 10] * 2, color="#111", lw=3)
            ax[k].text(10, base.shape[0] - 16, "1 км", color="#111", fontsize=12)
        ax[1].imshow(ov)
        rr = cells.r[mine] - (r - half); cc2 = cells.c[mine] - (c - half)
        ok = (rr >= 0) & (rr < n) & (cc2 >= 0) & (cc2 < n)
        ax[1].scatter(cc2[ok], rr[ok], s=2, facecolors="none", edgecolors="#ffd400", linewidths=0.3)
        ax[0].set_title("без разметки — смотрите сами", loc="left", fontsize=13)
        ax[1].set_title(f"красные — C_all вне зон, синие — в зонах (C); цветная заливка — слой «{lay}»; "
                        "жёлтый ореол — линия из подписи", loc="left", fontsize=10)
        fig.suptitle(f"{a.kind} {j:02d}. {title}", x=0.01, ha="left", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        safe = kind.replace(" ", "_").replace(":", "").replace("/", "-").replace("(", "").replace(")", "")
        fig.savefig(out / f"{a.kind}_{j:02d}_{safe}_E{E:.0f}_N{N:.0f}.png", dpi=90, facecolor=fig.get_facecolor())
        plt.close(fig)
        thumbs.append((j, kind, title, base, ov))
    cols = 4; rows = int(np.ceil(len(thumbs) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 5.4 * rows), facecolor="#fcfcfb")
    ax = np.atleast_2d(ax)
    for x in ax.ravel():
        x.axis("off")
    for (j, kind, title, base, ov), x in zip(thumbs, ax.ravel()):
        x.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); x.imshow(ov)
        x.set_title(f"{j:02d} {kind}\n" + title.split(": ", 1)[1].split(", центр")[0], fontsize=7.5, loc="left")
    fig.suptitle(f"{a.site}, набор «{a.kind}»: окна по типам артефактов (красные — C_all, синие — в зонах)",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"00_сводный_лист_{a.kind}.png", dpi=90, facecolor=fig.get_facecolor())
    print(len(thumbs), "окон + сводный лист")
    for j, kind, title, *_ in thumbs:
        print(j, title)


if __name__ == "__main__":
    main()
