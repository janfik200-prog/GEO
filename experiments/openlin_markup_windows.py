"""Окна Анабара для разметки геологом (заверка, не обучение).

Окна выбираются только по входам (снимок S2 зима, рельеф GLO-30, NDVI) — карта Анфисы и её линии
не используются, чтобы разметка оставалась независимой проверкой. Старые 9 окон (по ним уже смотрели
карты и ставили отметки) исключены. Размер окна 8 км — как окна ANZ*.

Правило выбора:
1. кандидаты — окна 8×8 км с шагом 4 км, годных клеток ≥ 99 %, без пересечения со старыми окнами;
2. признаки окна: перепад высот (p95−p5), средний уклон, мелкая текстура снимка, плотность тонких
   полос (гессиан σ = 1 клетка выше 90 % по листу), средний NDVI, средняя яркость;
3. «сложность» = среднее процентильных рангов уклона, перепада, текстуры, плотности полос;
4. берём верхнюю половину по сложности, делим на 8 групп k-средних (seed 0) по признакам,
   из каждой группы — самое сложное окно, не пересекающееся с уже взятыми.

    python -m experiments.openlin_markup_windows
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from experiments import openlin_cand1 as C1
from experiments.openlin_transfer_check import COMP

SITE = "anabar"
WIN_KM = 8
STEP_KM = 4
OUT = Path("outputs/разметка_анабар_8окон")


def hillshade(dem, res, az_deg, alt_deg=30.0):
    gy, gx = np.gradient(dem, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(360 - az_deg + 90), np.radians(alt_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def thin_ridges(x):
    hrr = ndi.gaussian_filter(x, 1, order=(2, 0))
    hcc = ndi.gaussian_filter(x, 1, order=(0, 2))
    hrc = ndi.gaussian_filter(x, 1, order=(1, 1))
    disc = np.sqrt(np.maximum((hrr - hcc) ** 2 / 4 + hrc ** 2, 0))
    return np.maximum(np.abs((hrr + hcc) / 2) + disc, 0)


def main():
    img, dem, valid, meta = C1.load_inputs(SITE)
    z = np.load(COMP / f"winter_{SITE}_20m_лист.npz")
    H, W = valid.shape
    ndvi, b04, b08 = (z[k][:H, :W].astype("float64") for k in ("ndvi", "B04", "B08"))
    res = meta["res_m"]
    wp, st = int(WIN_KM * 1000 / res), int(STEP_KM * 1000 / res)

    old = [(int((i + 0.5) * H / 3 - wp / 2), int((j + 0.5) * W / 3 - wp / 2)) for i in range(3) for j in range(3)]
    inter = lambda a, b: abs(a[0] - b[0]) < wp and abs(a[1] - b[1]) < wp

    gy, gx = np.gradient(dem, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    tex = np.abs(img - ndi.gaussian_filter(img, 5))
    rid = thin_ridges(img)
    rid_hi = rid > np.quantile(rid[valid], 0.9)

    cand, F = [], []
    for r0 in range(0, H - wp + 1, st):
        for c0 in range(0, W - wp + 1, st):
            sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
            v = valid[sl]
            if v.mean() < 0.99 or any(inter((r0, c0), o) for o in old):
                continue
            d = dem[sl][v]
            cand.append((r0, c0))
            F.append([np.percentile(d, 95) - np.percentile(d, 5), slope[sl][v].mean(), tex[sl][v].mean(),
                      rid_hi[sl][v].mean(), np.nanmean(ndvi[sl][v]), img[sl][v].mean()])
    F = np.array(F)
    names = ["перепад_м", "уклон_град", "текстура", "доля_тонких_полос", "ndvi", "яркость"]
    rank = lambda a: (np.argsort(np.argsort(a)) + 0.5) / len(a)
    cx = np.mean([rank(F[:, i]) for i in (0, 1, 2, 3)], axis=0)
    top = np.nonzero(cx >= np.median(cx))[0]
    from sklearn.cluster import KMeans
    Fs = (F - F.mean(0)) / F.std(0)
    lab = KMeans(8, n_init=20, random_state=0).fit_predict(Fs[top])
    chosen = []
    for g in range(8):
        for i in top[lab == g][np.argsort(-cx[top[lab == g]])]:
            if not any(inter(cand[i], cand[j]) for j in chosen):
                chosen.append(i)
                break
    chosen.sort(key=lambda i: cand[i])
    print(f"кандидатов {len(cand)}, в верхней половине {len(top)}, выбрано {len(chosen)}")

    import rasterio
    from rasterio.transform import from_origin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    crs = f"EPSG:{int(meta['epsg'])}"
    hs315, hs45 = hillshade(dem, res, 315), hillshade(dem, res, 45)
    info = []
    for n, i in enumerate(chosen, 1):
        r0, c0 = cand[i]
        sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
        x0, yt = meta["x0"] + c0 * res, meta["y_top"] - r0 * res
        tr = from_origin(x0, yt, res, res)
        d = OUT / f"окно_{n}"
        d.mkdir(exist_ok=True)
        for fname, arr in (("снимок_S2_зима.tif", img[sl]), ("рельеф_GLO30.tif", dem[sl]),
                           ("рельеф_тень_СЗ.tif", hs315[sl]), ("рельеф_тень_СВ.tif", hs45[sl]),
                           ("S2_B08_B04_NDVI.tif", np.stack([b08[sl], b04[sl], ndvi[sl]]))):
            a = arr if arr.ndim == 3 else arr[None]
            with rasterio.open(d / fname, "w", driver="GTiff", height=wp, width=wp, count=a.shape[0],
                               dtype="float32", crs=crs, transform=tr, compress="deflate") as f:
                f.write(a.astype("float32"))
        fig, axs = plt.subplots(1, 2, figsize=(24, 12.6))
        ext = (x0 / 1000, (x0 + wp * res) / 1000, (yt - wp * res) / 1000, yt / 1000)
        b = img[sl]
        lo, hi = np.percentile(b, [2, 98])
        axs[0].imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=ext)
        axs[0].set_title("снимок Sentinel-2, зима, низкое солнце (20 м)", fontsize=15)
        axs[1].imshow(hs315[sl], cmap="gray", vmin=0, vmax=1, extent=ext)
        axs[1].set_title("рельеф GLO-30, теневая отмывка с СЗ", fontsize=15)
        for ax in axs:
            ax.set_xticks(np.arange(np.ceil(ext[0]), ext[1] + 1e-6, 1))
            ax.set_yticks(np.arange(np.ceil(ext[2]), ext[3] + 1e-6, 1))
            ax.grid(color="yellow", alpha=0.35, lw=0.6)
            ax.tick_params(labelsize=9, labelrotation=0)
            ax.set_xlabel(f"x, км ({crs}, UTM 47N)"); ax.set_ylabel("y, км")
        fig.suptitle(f"Анабар, окно для разметки {n} (8×8 км). Сетка 1 км", fontsize=16)
        fig.tight_layout()
        fig.savefig(d / f"окно_{n}.png", dpi=110)
        plt.close(fig)
        info.append({"окно": n, "x_min": x0, "x_max": x0 + wp * res, "y_min": yt - wp * res, "y_max": yt,
                     "epsg": int(meta["epsg"]), "строка0": r0, "столбец0": c0, "сложность": float(cx[i]),
                     **{k: float(v) for k, v in zip(names, F[i])}})

    # обзор листа
    fig, ax = plt.subplots(figsize=(13, 12))
    lo, hi = np.percentile(img[valid], [2, 98])
    ax.imshow(np.where(valid, img, np.nan), cmap="gray", vmin=lo, vmax=hi)
    for r0, c0 in old:
        ax.add_patch(plt.Rectangle((c0, r0), wp, wp, fill=False, ec="grey", ls="--", lw=1.2))
    for n, i in enumerate(chosen, 1):
        r0, c0 = cand[i]
        ax.add_patch(plt.Rectangle((c0, r0), wp, wp, fill=False, ec="red", lw=2))
        ax.text(c0 + wp / 2, r0 + wp / 2, str(n), color="red", fontsize=20, ha="center", va="center", weight="bold")
    ax.set_title("Анабар: 8 окон для разметки (красные); серые пунктирные — прежние окна", fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / "обзор_листа.png", dpi=100)
    plt.close(fig)

    import geopandas as gpd
    from shapely.geometry import box
    gpd.GeoDataFrame(info, geometry=[box(o["x_min"], o["y_min"], o["x_max"], o["y_max"]) for o in info],
                     crs=crs).to_file(OUT / "окна.gpkg", layer="окна", driver="GPKG")
    json.dump({"правило": __doc__, "окна": info}, open(OUT / "окна.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for o in info:
        print(o["окно"], {k: round(o[k], 3) for k in ["сложность"] + names})


if __name__ == "__main__":
    main()
