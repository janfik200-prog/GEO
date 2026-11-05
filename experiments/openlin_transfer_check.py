"""Сверка перенесённого набора линий с чужой разметкой линеаментов.

ОТКУДА ПОСТАНОВКА. На Анабаре эталона нет, и все условия ветки внутренние:
мусор против подделки, подтверждение половинами, устойчивость, длина против
шума. Они говорят, что линии не случайны, но не говорят, совпадают ли они с
тем, что геолог называет линеаментом. Открытая задача 5 (OPENLIN_RULES.md) —
перенос на площадку с замороженными порогами; здесь к ней добавлена разметка.

ПЛОЩАДКА И ЭТАЛОН. Лапландия (``openlin_probe.SITES["lapland"]``), лист того же
размера, что R-48-XI,XII. Эталон — линеаменты GTK 1:500 000 (Engström et al.
2025, Norwegian Journal of Geology 105, doi 10.17850/njg105-1-3, CC BY 4.0),
служба ``gtkdata.gtk.fi/.../kokoavaWMS/MapServer``:

* основной — слой 292 «LiDAR»: линии, проведённые вручную по теневой отмывке
  рельефа LiDAR 2 м. Это ближайшее к нашему входу: зимний снимок при низком
  солнце — та же тень от рельефа, только снятая из космоса;
* для сведения — 291 (объединённый), 293 (магнитка), 294 (электроразведка).
  Магнитные и электроразведочные линии по снимку видны быть не обязаны.

ЧЕГО ЭТО НЕ ЯВЛЯЕТСЯ. Эталон — не полная разметка: масштаб 1:500 000, медиана
длины линии 24 км, 10-й процентиль 10 км. Наших линий на листе — тысячи, по
1-3 км. Поэтому «точность» (доля наших линий у эталонных) здесь не меряет
ошибку метода — большинство мелких линеаментов в эталон не вошло по масштабу;
она печатается только для сведения. Проверяется ПОЛНОТА: какая доля длины
эталонных линий лежит рядом с нашими.

ОБЪЯВЛЕНИЕ (24.10.2026, до прогона сверки и до прогона переноса):

* величина — доля ячеек эталона (слой 292, растр на сетке 20 м, только
  внутри общих ячеек снимка), лежащих не дальше ``TOL_M`` от линий набора;
* ``TOL_M`` = 300 м — из самого эталона: авторы пишут, что масштаб
  интерпретации 1:500 000 «примерно соответствует растру 300 м»; точнее
  положения линии эталон не знает;
* нуль — тот же сдвиг по тору, что в ветке (``shift_null_conf``: поля
  min(200, H/8) ячеек): сдвигается эталон, карта расстояний до наших линий
  остаётся на месте; рушатся только места, вся структура обеих карт цела.
  Длинные эталонные линии при сдвиге вдоль своего простирания частично
  совпадают сами с собой — нуль от этого только выше, то есть строже;
* наборы — два, как в ветке: «в зонах» (C) и «весь лист» (C_all);
* счётчик против этого эталона — новый, свой: ``GTK_N_TESTS`` = 2 (прежних
  проверок против линеаментов GTK не было); alpha = 0.05 / 2;
* достижимость: n >= 2/alpha - 1 = 79; берётся n = 4/alpha - 1 = 159 — запас
  4x, как у красного канала (3.9x) и NDVI (4.4x);
* условие: p <= alpha для набора. Подъём над нулём (доля / средняя доля нуля)
  печатается как размер эффекта, не как условие — порога ему нет из замера.

Запуск из корня репозитория (после ``openlin_zone_lines_v3 --site lapland
--перенос``):
    python -m experiments.openlin_transfer_check --site lapland
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
OUT = Path("outputs/steps")
COMP = Path("datacache/s2_lowsun/composite")
GTK = Path("datacache/gtk_lineaments")

# ---- объявление (24.10.2026, до прогона) -----------------------------------
TOL_M = 300.0
GTK_N_TESTS = 2
GTK_ALPHA = 0.05 / GTK_N_TESTS
GTK_SHIFT_N = int(round(4 / GTK_ALPHA)) - 1          # 159
GTK_SHIFT_SEED = 11                                  # то же зерно, что SHIFT_SEED
#: Площадка НАСТРОЙКИ (05.11.2026, объявлено до прогона): разметка GTK 292 там
#: используется для настройки тракта, поэтому p против неё — НЕ доказательство,
#: а ориентир. Отдельной семьи в счётчик не заводится (``GTK_N_TESTS`` остаётся
#: 2 и относится только к контролю — Лапландии); для площадок из этого набора
#: печатается «ДЛЯ СВЕДЕНИЯ» вместо «ПРОЙДЕНО». Числа (допуск, сдвиги, alpha)
#: те же, их происхождение выше. Сама разметка того же происхождения, что на
#: Лапландии (те же авторы), — независимость контроля неполная.
TUNE_SITES = ("utsjoki",)
MAIN_LAYER = "292"
INFO_LAYERS = {"291": "объединённый", "293": "магнитка", "294": "электроразведка"}
#: только для сведения: кривая полноты по допуску
INFO_TOLS_M = (150.0, 300.0, 600.0, 1000.0)
# ---------------------------------------------------------------------------


def gtk_raster(layer: str, w: dict, shape) -> np.ndarray:
    """Линии слоя GTK на сетке композита (касание ячейки — ячейка линии)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    g = gpd.read_file(GTK / f"lin500k_{layer}.gpkg").to_crs(epsg=int(w["epsg"]))
    res = float(w["res_m"])
    tr = Affine(res, 0, float(w["x0"]), 0, -res, float(w["y_top"]))
    return rasterize(((s, 1) for s in g.geometry if s is not None),
                     out_shape=shape, transform=tr, fill=0, all_touched=True,
                     dtype="uint8").astype(bool)


def recall(ref: np.ndarray, dist: np.ndarray, tol: float) -> float:
    return float((dist[ref] <= tol).mean()) if ref.any() else float("nan")


def shift_null(ref: np.ndarray, dist: np.ndarray, valid: np.ndarray, tol: float,
               n: int, seed: int) -> tuple[np.ndarray, float]:
    """Сдвиг эталона по тору; ячейки, упавшие вне снимка, не считаются."""
    H, W = ref.shape
    rr, cc = np.nonzero(ref)
    near = dist <= tol
    rng = np.random.default_rng(seed)
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    out = np.empty(n)
    for i in range(n):
        dr = int(rng.integers(lo_r, H - lo_r))
        dc = int(rng.integers(lo_c, W - lo_c))
        r2, c2 = (rr + dr) % H, (cc + dc) % W
        ok = valid[r2, c2]
        out[i] = near[r2[ok], c2[ok]].mean() if ok.any() else np.nan
    return out, float(np.nanmean(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="lapland")
    ap.add_argument("--tag", default="_перенос",
                    help="суффикс файла набора линий (по умолчанию — перенос)")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from scipy.ndimage import distance_transform_edt

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    d = np.load(M / f"openlin_zone_lines_{a.site}_v3{a.tag}.npz")
    res = float(w["res_m"])
    shape = d["C_all"].shape
    valid = np.isfinite(w["shade"][:shape[0], :shape[1]])
    log(f"площадка {a.site}: сетка {shape}, шаг {res:.0f} м, общих ячеек "
        f"{100 * valid.mean():.1f} %")
    log(f"объявлено: допуск {TOL_M:.0f} м, проверок {GTK_N_TESTS}, alpha "
        f"{GTK_ALPHA:.4f}, сдвигов {GTK_SHIFT_N} (пол p {1 / (1 + GTK_SHIFT_N):.4f}, "
        f"запас {GTK_ALPHA * (1 + GTK_SHIFT_N):.1f}x)")

    refs = {L: gtk_raster(L, w, shape) & valid
            for L in (MAIN_LAYER, *INFO_LAYERS)}
    for L, r in refs.items():
        log(f"  слой {L}: ячеек линий {int(r.sum())} (~{r.sum() * res / 1000:.0f} км)")

    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    rows = []
    for name, key in (("в зонах", "C"), ("весь лист", "C_all")):
        ours = d[key][:shape[0], :shape[1]]
        dist = distance_transform_edt(~ours, sampling=(res, res))
        dref = distance_transform_edt(~refs[MAIN_LAYER], sampling=(res, res))
        obs = recall(refs[MAIN_LAYER], dist, TOL_M)
        null, mu = shift_null(refs[MAIN_LAYER], dist, valid, TOL_M,
                              GTK_SHIFT_N, GTK_SHIFT_SEED)
        p = float((1 + (null >= obs).sum()) / (1 + GTK_SHIFT_N))
        log(f"\nнабор «{name}»: линий-ячеек {int(ours.sum())} "
            f"(~{ours.sum() * res / 1000:.0f} км)")
        log(f"  ПОЛНОТА по слою LiDAR в {TOL_M:.0f} м: {100 * obs:.1f} % "
            f"(нуль {100 * mu:.1f} %, макс {100 * np.nanmax(null):.1f} %), "
            f"подъём {obs / max(mu, 1e-9):.2f}x, p = {p:.4f} "
            f"(порог {GTK_ALPHA:.4f}) -> "
            + ("ДЛЯ СВЕДЕНИЯ (площадка настройки)" if a.site in TUNE_SITES
               else mark[p <= GTK_ALPHA]))
        prec = float((dref[ours] <= TOL_M).mean()) if ours.any() else float("nan")
        base = float((dref[valid] <= TOL_M).mean())
        log(f"  для сведения, доля наших линий у эталона: {100 * prec:.1f} % "
            f"при {100 * base:.1f} % площади ({prec / max(base, 1e-9):.2f}x)")
        log("  для сведения, полнота по допуску: " + "; ".join(
            f"{t:.0f} м {100 * recall(refs[MAIN_LAYER], dist, t):.1f} %"
            for t in INFO_TOLS_M))
        for L, ttl in INFO_LAYERS.items():
            o = recall(refs[L], dist, TOL_M)
            _n, m_ = shift_null(refs[L], dist, valid, TOL_M, GTK_SHIFT_N,
                                GTK_SHIFT_SEED)
            log(f"  для сведения, слой {L} ({ttl}): {100 * o:.1f} % при нуле "
                f"{100 * m_:.1f} % ({o / max(m_, 1e-9):.2f}x)")
        rows.append(dict(набор=name, полнота=obs, нуль=mu, p=p,
                         подъём=obs / max(mu, 1e-9), доля_у_эталона=prec,
                         площадь_у_эталона=base))

    import pandas as pd
    p_csv = M / f"openlin_transfer_check_{a.site}.csv"
    pd.DataFrame(rows).to_csv(p_csv, index=False, encoding="utf-8-sig")
    log(f"\nсохранено: {p_csv}")
    plot(w, d, refs[MAIN_LAYER], valid, a.site, log)


def plot(w: dict, d, ref: np.ndarray, valid: np.ndarray, site: str, log) -> None:
    """Снимок, наши линии (весь лист) и эталон LiDAR поверх."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    shape = ref.shape
    img = _stretch(np.where(valid, w["shade"][:shape[0], :shape[1]], np.nan))
    ours = binary_dilation(d["C_all"][:shape[0], :shape[1]], iterations=2)
    refd = binary_dilation(ref, iterations=3)
    fig, ax = plt.subplots(1, 2, figsize=(22, 10), facecolor="#fcfcfb")
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("зимний композит при низком солнце (shade)", fontsize=14,
                    loc="left")
    ax[1].imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.45)
    ov = np.zeros(shape + (4,), dtype="float32")
    ov[ours] = (0.70, 0.15, 0.12, 0.9)
    ov[refd] = (0.10, 0.45, 0.85, 0.75)
    ax[1].imshow(ov)
    ax[1].set_title("красное — наши линии (весь лист), синее — линеаменты GTK "
                    "по LiDAR 1:500 000", fontsize=14, loc="left")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"перенос_{site}_сверка_GTK.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
