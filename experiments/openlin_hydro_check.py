"""Проверка на гидрографию: не берега ли мы выделяем.

ЗАЧЕМ. Фазовая подделка разрушает ВСЮ связную структуру снимка, поэтому её
проходит что угодно настоящее: береговая линия озера, бровка речной террасы,
край наледи. Вывод «отклик не шумовой» она даёт, вывод «это уступ» — нет.

Эта проверка отвечает на тот вопрос, который нуль не может. Линии гидросети
(``dnl``) и её ареалы (``dnara``) из комплекта заказчика растеризуются на ту же
сетку, и считается, какая доля найденных элементов лежит рядом с водой. Планка
«минус расстояние до гидросети» в этом проекте обязательна (см.
``src/assessment.py``): на прошлой фазе она давала lift@10 % = 4.15, выше обоих
содержательных методов, и любой метод обязан с ней сопоставляться.

Случайное ожидание берётся не из головы, а из самой карты: какая доля ПЛОЩАДИ
листа лежит в том же допуске от воды. Плюс сдвиговый нуль — та же маска
элементов, сдвинутая тором, чтобы сохранить её геометрию и плотность.

Запуск из корня репозитория:
    python -m experiments.openlin_hydro_check --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_shade import lines_from_field
from experiments.openlin_both import THR, MIN_LEN_M

M = Path("outputs/metrics")

#: Допуски расстояния до воды, м.
TOLS = (100.0, 250.0, 500.0, 1000.0)

#: Сколько случайных сдвигов для нуля.
N_SHIFT = 50


def hydro_mask(grid_shape, res_m: float, x0: float, y_top: float, epsg: int,
               log) -> np.ndarray:
    """Растр гидросети на сетке поля: линии и ареалы вместе."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    from src import config

    tr = Affine(res_m, 0, x0, 0, -res_m, y_top)
    shapes = []
    for name in ("dnl", "dnara"):
        p = Path(config.TOPO_SHP_DIR) / f"{name}.shp"
        g = gpd.read_file(p)
        if g.crs is None:
            # у слоёв заказчика CRS лежит не в .prj, а в отдельном sidecar
            # .pj4 — без него geopandas считает геометрию «наивной»
            from experiments.build_dataset_v1 import read_sidecar_proj4

            g = g.set_crs(read_sidecar_proj4(p), allow_override=True)
        g = g.to_crs(epsg=epsg)
        log(f"  {name}: объектов {len(g)}")
        shapes.extend(g.geometry.values)
    return rasterize(((s, 1) for s in shapes if s is not None),
                     out_shape=grid_shape, transform=tr,
                     fill=0, dtype="uint8").astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--runs", default="оптика:,радар:_радар",
                    help="что проверять: пары «имя:суффикс» через запятую")
    ap.add_argument("--thr", type=float, default=None,
                    help="порог оценки; по умолчанию из openlin_both.THR")
    a = ap.parse_args()

    from scipy.ndimage import distance_transform_edt

    def log(m=""):
        print(m, flush=True)

    runs = [tuple(r.split(":")) for r in a.runs.split(",")]
    z = dict(np.load(M / f"openlin_shade_{a.site}{runs[0][1]}.npz"))
    res_m = float(z["res_m"])
    log(f"растеризация гидросети на сетку {z['оценка'].shape}, шаг "
        f"{res_m:.0f} м:")
    water = hydro_mask(z["оценка"].shape, res_m, float(z["x0"]),
                       float(z["y_top"]), int(z["epsg"]), log)
    dt = distance_transform_edt(~water, sampling=(res_m, res_m))
    log(f"  ячеек воды {100 * water.mean():.2f} % площади")

    rng = np.random.default_rng(7)
    for name, tag in runs:
        zz = dict(np.load(M / f"openlin_shade_{a.site}{tag}.npz"))
        thr = a.thr if a.thr is not None else THR.get(name, 3.5)
        mask, df = lines_from_field(zz["оценка"], res_m, thr, MIN_LEN_M)
        ok = np.isfinite(zz["оценка"])
        log(f"\n{name}: элементов {len(df)}")
        log(f"{'допуск':>8} {'элементов':>10} {'площади':>9} {'превышение':>11}"
            f" {'при сдвиге':>11} {'p':>6}")
        for tol in TOLS:
            near = dt[mask & ok] <= tol
            base = float((dt[ok] <= tol).mean())
            obs = float(near.mean())
            sh = []
            for _ in range(N_SHIFT):
                dy = int(rng.integers(200, mask.shape[0] - 200))
                dx = int(rng.integers(200, mask.shape[1] - 200))
                ms = np.roll(np.roll(mask, dy, 0), dx, 1) & ok
                if ms.sum() > 10:
                    sh.append(float((dt[ms] <= tol).mean()))
            sh = np.asarray(sh)
            p = (1 + int((sh >= obs).sum())) / (1 + len(sh))
            log(f"{tol:8.0f} {100 * obs:9.1f}% {100 * base:8.1f}% "
                f"{obs / max(base, 1e-9):11.2f}x {100 * sh.mean():10.1f}% "
                f"{p:6.3f}")


if __name__ == "__main__":
    main()
