"""Батарея проверок на артефакты: не прибор ли и не человек ли это нарисовал.

ОТКУДА ПОСТАНОВКА. После исправления подавления немаксимумов
(:func:`openlin_lines._ridge_nms`, 23.09.2026) набор подтверждённых линий вырос
втрое, и главным стало ШИРОТНОЕ семейство (75-105°), которого прибор раньше не
мог увидеть. Прежде чем что-то утверждать про это семейство, оно обязано
пройти проверки на четыре источника ложных прямых, ни одна из которых в ветке
ещё не делалась.

ЧТО ПРОВЕРЯЕТСЯ.
  1. ШВЫ ПЛИТОК СНИМКА. Композит склеен из четырёх плиток Sentinel-2; на стыке
     меняется число сцен в ячейке (поле ``count``), и вдоль стыка возможен
     скачок яркости. Шов берётся как граница смены ``count``.
  2. ГРАНИЦЫ ПЛИТОК ЦМР. Рельеф у нас не входной слой, но ЦМР протекает внутрь
     дважды: радиометрическое выравнивание Sentinel-1 RTC (каналы VH и
     когерентность строят карту зон) и ортотрансформирование L2A. Плитки
     Copernicus DEM нарезаны по целым градусам, поэтому проверяется близость
     линий к сетке целых градусов широты и долготы.
  3. ТЕХНОГЕННЫЕ ПРЯМЫЕ (сейсмопрофили, зимники, просеки). Отличаются
     геометрией: они прямее всего природного и держат один азимут на многие
     километры. Меряется распределение кривизны (разброс поперёк оси, делённый
     на длину) и выделяется хвост «ультрапрямых».
  4. ВЕТРОВЫЕ ПОЛОСЫ (заструги, надувы). Отличаются тем, что азимут им задаёт
     ветер — один на весь лист, — и тем, что от съёмки к съёмке они меняются.
     Меряется концентрация азимутов и доля подтверждения по секторам.

КАК СЧИТАЕТСЯ БЛИЗОСТЬ. Для швов и границ ЦМР — доля ячеек линий ближе 100 м к
шву против доли площади и сдвиговый нуль на 2000 сдвигов, как в основном
прогоне: сдвигаются линии, шов остаётся на месте.

ЭТО ЗАМЕР, А НЕ ГИПОТЕЗА: пороги не объявляются, p печатается как величина.

Запуск из корня репозитория:
    python -m experiments.openlin_artifact_battery --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, M, OUT

NEAR_M = 100.0               # та же полоса, что у проверки гидросети
SHIFT_N = 2000
SHIFT_SEED = 23
STRAIGHT_Q = 0.005           # «ультрапрямые»: разброс поперёк < 0.5 % длины
LONG_KM = 3.0                # техногенные прямые держат азимут километрами


def components(mask, res_m: float) -> list[dict]:
    """Связные куски маски с азимутом, длиной и кривизной."""
    from scipy.ndimage import label

    lab, n = label(mask, np.ones((3, 3), bool))
    out = []
    for i in range(1, n + 1):
        r, c = np.where(lab == i)
        p = np.stack([r, c], 1).astype(float)
        q = p - p.mean(0)
        vt = np.linalg.svd(q, full_matrices=False)[2]
        t = q @ vt[0]
        ln = float(t.max() - t.min()) * res_m
        if ln <= 0:
            continue
        out.append({"pts": np.stack([r, c], 1), "длина_м": ln,
                    "кривизна": float(np.abs(q @ vt[1]).mean()) * res_m / ln,
                    "азимут": float(np.degrees(
                        np.arctan2(vt[0][1], -vt[0][0])) % 180.0)})
    return out


def near_share(lines, dist, res_m: float) -> float:
    """Доля ячеек линий ближе NEAR_M к объекту."""
    if not lines:
        return 0.0
    p = np.concatenate([r["pts"] for r in lines])
    return float((dist[p[:, 0], p[:, 1]] <= NEAR_M).mean())


def shift_p(lines, dist, shape, res_m: float) -> tuple[float, float]:
    """Сдвиговый нуль: линии едут по тору, объект стоит на месте."""
    rng = np.random.default_rng(SHIFT_SEED)
    H, W = shape
    obs = near_share(lines, dist, res_m)
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    out = []
    for _ in range(SHIFT_N):
        dr = int(rng.integers(lo_r, H - lo_r))
        dc = int(rng.integers(lo_c, W - lo_c))
        moved = [{"pts": np.stack([(r["pts"][:, 0] + dr) % H,
                                   (r["pts"][:, 1] + dc) % W], 1)}
                 for r in lines]
        out.append(near_share(moved, dist, res_m))
    out = np.array(out)
    return float(out.mean()), float((1 + (out >= obs).sum()) / (1 + SHIFT_N))


def dem_tile_edges(shape, res_m: float, x0: float, y_top: float, epsg: int):
    """Границы целых градусов широты и долготы — нарезка плиток Copernicus DEM."""
    from pyproj import Transformer

    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    step = 10                                   # считаем на сетке 200 м, потом растягиваем
    rr = np.arange(0, shape[0], step)
    cc = np.arange(0, shape[1], step)
    xx = x0 + (cc + 0.5) * res_m
    yy = y_top - (rr + 0.5) * res_m
    X, Y = np.meshgrid(xx, yy)
    lon, lat = tr.transform(X, Y)
    edge = np.zeros(lon.shape, bool)
    for f in (lon, lat):
        k = np.floor(f)
        edge[:, :-1] |= k[:, :-1] != k[:, 1:]
        edge[:-1, :] |= k[:-1, :] != k[1:, :]
    big = np.repeat(np.repeat(edge, step, 0), step, 1)[:shape[0], :shape[1]]
    return big, (float(np.floor(lon.min())), float(np.ceil(lon.max())),
                 float(np.floor(lat.min())), float(np.ceil(lat.max())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from scipy.ndimage import binary_dilation, distance_transform_edt

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    count = w["count"].astype("int16")
    from experiments.openlin_zone_lines_v3 import work_npz
    d = np.load(work_npz(a.site))        # рабочий набор — «цепочки»
    A, C = d["A"], d["C"]
    shape = C.shape
    count = count[:shape[0], :shape[1]]
    linesA, linesC = components(A, res_m), components(C, res_m)
    log(f"площадка {a.site}: набор А {len(linesA)} кусков, набор В "
        f"{len(linesC)} ({time.time() - t0:.0f} с)")

    # ---- 1. швы плиток снимка ---------------------------------------------
    seam = np.zeros(shape, bool)
    seam[:, :-1] |= count[:, :-1] != count[:, 1:]
    seam[:-1, :] |= count[:-1, :] != count[1:, :]
    log(f"\n1. швы плиток снимка: ячеек шва {int(seam.sum())} "
        f"({100 * seam.mean():.2f} % листа)")
    if seam.any():
        ds = distance_transform_edt(~seam, sampling=(res_m, res_m))
        base = float((ds <= NEAR_M).mean())
        for name, L in (("А", linesA), ("В", linesC)):
            obs = near_share(L, ds, res_m)
            null, p = shift_p(L, ds, shape, res_m)
            log(f"   набор {name}: у шва {100 * obs:.1f} % при {100 * base:.1f} % "
                f"площади ({obs / max(base, 1e-9):.2f}x), сдвиговый нуль "
                f"{100 * null:.1f} %, p = {p:.4f}")

    # ---- 2. границы плиток ЦМР --------------------------------------------
    edges, box = dem_tile_edges(shape, res_m, float(w["x0"]), float(w["y_top"]),
                                int(w["epsg"]))
    log(f"\n2. границы целых градусов (плитки Copernicus DEM): долгота "
        f"{box[0]:.0f}..{box[1]:.0f}, широта {box[2]:.0f}..{box[3]:.0f}; "
        f"ячеек границы {int(edges.sum())}")
    if edges.any():
        de = distance_transform_edt(~edges, sampling=(res_m, res_m))
        base = float((de <= NEAR_M).mean())
        for name, L in (("А", linesA), ("В", linesC)):
            obs = near_share(L, de, res_m)
            null, p = shift_p(L, de, shape, res_m)
            log(f"   набор {name}: у границы {100 * obs:.1f} % при "
                f"{100 * base:.1f} % площади ({obs / max(base, 1e-9):.2f}x), "
                f"нуль {100 * null:.1f} %, p = {p:.4f}")
    else:
        log("   границ целых градусов в пределах листа нет — проверка не нужна")

    # ---- 3. техногенные прямые --------------------------------------------
    log("\n3. кривизна линий (разброс поперёк оси / длина):")
    for name, L in (("А", linesA), ("В", linesC)):
        k = np.array([r["кривизна"] for r in L])
        ln = np.array([r["длина_м"] for r in L]) / 1000
        q = np.percentile(k, [10, 50, 90])
        ultra = (k < STRAIGHT_Q) & (ln >= LONG_KM)
        log(f"   набор {name}: {100 * q[0]:.1f} / {100 * q[1]:.1f} / "
            f"{100 * q[2]:.1f} % (10/50/90 %); ультрапрямых длиннее "
            f"{LONG_KM:.0f} км: {int(ultra.sum())}")
        if ultra.any():
            azs = np.array([r["азимут"] for r in L])[ultra]
            log("      их азимуты: "
                + ", ".join(f"{v:.0f}°" for v in np.sort(azs)))

    # ---- 4. ветровые полосы: концентрация азимутов -------------------------
    log("\n4. азимуты: концентрация и вклад главного сектора")
    for name, L in (("А", linesA), ("В", linesC)):
        azs = np.array([r["азимут"] for r in L])
        wts = np.array([r["длина_м"] for r in L])
        t = np.radians(2 * azs)
        R = float(np.hypot((np.sin(t) * wts).sum() / wts.sum(),
                           (np.cos(t) * wts).sum() / wts.sum()))
        h = np.histogram(azs, bins=12, range=(0, 180), weights=wts)[0]
        top = int(np.argmax(h))
        log(f"   набор {name}: R = {R:.2f} (1 = все линии в одном направлении), "
            f"главный сектор {15 * top}-{15 * top + 15}° несёт "
            f"{100 * h[top] / h.sum():.0f} % длины")

    np.savez_compressed(M / f"openlin_artifact_battery_{a.site}.npz",
                        seam=seam, dem_edges=edges)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_shade import _stretch

    base_img = _stretch(np.where(np.isfinite(w["shade"]), w["shade"], np.nan))
    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#fcfcfb")
    ax.imshow(base_img[:shape[0], :shape[1]], cmap="gray", vmin=0, vmax=1)
    ov = np.zeros(shape + (4,), "float32")
    ov[binary_dilation(seam, np.ones((3, 3), bool))] = (0.16, 0.47, 0.84, 0.9)
    ov[binary_dilation(edges, np.ones((3, 3), bool))] = (0.1, 0.6, 0.3, 0.9)
    ov[binary_dilation(C, np.ones((3, 3), bool))] = (0.70, 0.10, 0.10, 1.0)
    ax.imshow(ov)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Линии набора В (бордовые) против швов плиток снимка (синие)\n"
                 "и границ плиток ЦМР (зелёные)", loc="left")
    fig.tight_layout()
    p = OUT / f"артефакты_швы_{a.site}.png"
    fig.savefig(p, dpi=100, facecolor=fig.get_facecolor())
    log(f"\nсохранено: {p} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
