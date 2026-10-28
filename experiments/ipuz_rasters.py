"""WP03 ТЗ ИПУЗ: экспорт трёх растров и независимая проверка геометрии.

Пишет в каталог заморозки три GeoTIFF в одной сетке, одной системе координат
и с одной валидной маской:

  * ``crit.tif``          — критериальный балл в шкале перспективности
                            (возрастает в сторону перспективной площади);
  * ``oof_remote.tif``    — слепая карта по дистанционным признакам
                            (та же, что даёт O₁₀ = 0,198603);
  * ``overlap_class.tif`` — классы пересечения верхних децилей:
                            0 — вне валидной маски,
                            1 — ни одна карта не относит ячейку к верхним 10 %,
                            2 — только критериальная карта,
                            3 — только дистанционная карта,
                            4 — общие ячейки.

После записи растры читаются заново и O₁₀ считается прямо из ``overlap_class``
без обращения к исходным таблицам: это независимый пересчёт, а не повтор
той же арифметики. Результат сверяется с ``outputs/metrics`` до 1e-9.

Запуск::

    python -m experiments.ipuz_rasters --freeze <каталог_заморозки>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (чинит PROJ_LIB до первого импорта rasterio)
import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

SHAPE = (154, 149)
CELL_M = 500.0
PROJ4 = ("+proj=tmerc +lon_0=105 +lat_0=0 +x_0=500000 +y_0=0 +k_0=1 "
         "+ellps=krass +units=m +no_defs")

# Классы overlap_class.tif и ожидаемые размеры по ТЗ.
CLASS_NAMES = {
    0: "вне валидной маски",
    1: "ни одна карта не в верхних 10 %",
    2: "только критериальная карта",
    3: "только дистанционная карта",
    4: "общие ячейки",
}
EXPECTED_CLASSES = {0: 41, 1: 18778, 2: 1836, 3: 1836, 4: 455}
EXPECTED_O10 = 455 / 2291


class MissingInput(SystemExit):
    """Понятный отказ вместо подмены отсутствующего слоя нулями."""


def require(paths: dict[str, Path]) -> None:
    missing = {k: p for k, p in paths.items() if not p.exists()}
    if missing:
        lines = ["Не найдены входные файлы — расчёт остановлен:"]
        lines += ["  %-22s %s" % (k, p) for k, p in missing.items()]
        lines.append("Подставлять нулевые или случайные данные вместо них нельзя.")
        raise MissingInput("\n".join(lines))


def top_decile(v: np.ndarray, pool: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Верхняя дециль внутри общей валидной маски.

    Маска выбирается до сортировки: квантиль считается только по ``pool``,
    иначе в порог попадают ячейки, которых нет на другой карте.
    """
    m = pool & np.isfinite(v)
    thr = float(np.quantile(v[m], 1.0 - frac))
    out = np.zeros(v.size, dtype=bool)
    out[m] = v[m] >= thr
    return out


def count_ties(v: np.ndarray, pool: np.ndarray, frac: float = 0.10) -> int:
    """Сколько ячеек стоит ровно на пороге — источник неустойчивости границы."""
    m = pool & np.isfinite(v)
    thr = float(np.quantile(v[m], 1.0 - frac))
    return int((v[m] == thr).sum())


def write_tif(path: Path, data: np.ndarray, transform, dtype: str,
              nodata, описание: str) -> None:
    with rasterio.open(
            path, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1],
            count=1, dtype=dtype, crs=CRS.from_proj4(PROJ4),
            transform=transform, nodata=nodata,
            compress="deflate", predictor=2 if dtype.startswith("float") else 1,
            tiled=False) as dst:
        dst.write(data.astype(dtype), 1)
        dst.update_tags(описание=описание, лист="R-48-XI,XII",
                        шаг_м="%.0f" % CELL_M)
        dst.set_band_description(1, описание)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path,
                    help="каталог заморозки (в нём 01_inputs/target_mask.csv)")
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="корень проекта с data/processed")
    ap.add_argument("--out", type=Path, default=None,
                    help="куда писать растры (по умолчанию <freeze>/05_maps)")
    ap.add_argument("--metrics", type=Path, default=None,
                    help="CSV рабочей точки для сверки O₁₀")
    a = ap.parse_args()

    freeze = a.freeze.resolve()
    out = (a.out or freeze / "05_maps").resolve()
    mask_csv = freeze / "01_inputs" / "target_mask.csv"
    oof_pq = a.root / "data" / "processed" / "operating_point_masked.parquet"
    metrics_csv = a.metrics or (a.root / "outputs" / "metrics" /
                                "operating_point_masked.csv")
    require({"target_mask.csv": mask_csv, "operating_point_masked.parquet": oof_pq,
             "operating_point_masked.csv": metrics_csv})
    out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(mask_csv)
    if len(t) != SHAPE[0] * SHAPE[1]:
        raise SystemExit("В маске %d строк, ожидалось %d (%d×%d)"
                         % (len(t), SHAPE[0] * SHAPE[1], *SHAPE))
    # Порядок строк должен быть строчно-мажорным от северного края.
    if not (t["row"].to_numpy() == np.repeat(np.arange(SHAPE[0]), SHAPE[1])).all() \
            or not (t["col"].to_numpy() == np.tile(np.arange(SHAPE[1]), SHAPE[0])).all():
        raise SystemExit("Порядок ячеек в target_mask.csv не строчно-мажорный")

    x = t["x"].to_numpy(float)
    y = t["y"].to_numpy(float)
    valid = t["valid"].to_numpy(bool)
    # В формуле меньший балл = перспективнее, растр пишем в прямой шкале.
    crit = -t["crit_score"].to_numpy(float)

    op = pd.read_parquet(oof_pq)
    if len(op) != len(t):
        raise SystemExit("Длина parquet рабочей точки %d не равна сетке %d"
                         % (len(op), len(t)))
    remote = op["score"].to_numpy(float)

    # Геопривязка: верхний левый угол, шаг 500 м, ось Y вниз.
    transform = from_origin(x.min() - CELL_M / 2, y.max() + CELL_M / 2,
                            CELL_M, CELL_M)

    # Общая валидная маска — одна на обе карты.
    pool = valid & np.isfinite(crit) & np.isfinite(remote)
    a10 = top_decile(crit, pool)
    b10 = top_decile(remote, pool)
    both = a10 & b10

    code = np.zeros(pool.size, dtype=np.uint8)
    code[pool] = 1
    code[a10 & ~b10] = 2
    code[b10 & ~a10] = 3
    code[both] = 4

    crit_g = np.where(pool, crit, np.nan).reshape(SHAPE)
    remote_g = np.where(pool, remote, np.nan).reshape(SHAPE)
    code_g = code.reshape(SHAPE)

    write_tif(out / "crit.tif", crit_g, transform, "float32", np.nan,
              "критериальный балл, шкала перспективности (инвертирован)")
    write_tif(out / "oof_remote.tif", remote_g, transform, "float32", np.nan,
              "слепая карта по дистанционным признакам, полосы 15 км")
    write_tif(out / "overlap_class.tif", code_g, transform, "uint8", 255,
              "0 вне маски; 1 ни одна; 2 только критерий; 3 только модель; 4 общие")

    # ---------------------------------------------------- независимая проверка
    geom: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for name in ("crit.tif", "oof_remote.tif", "overlap_class.tif"):
        with rasterio.open(out / name) as ds:
            arrays[name] = ds.read(1)
            geom.append({
                "файл": name, "форма": [ds.height, ds.width],
                "crs": ds.crs.to_proj4(),
                "transform": [round(v, 6) for v in list(ds.transform)[:6]],
                "nodata": None if ds.nodata is None or np.isnan(ds.nodata)
                else float(ds.nodata),
                "dtype": str(ds.dtypes[0]),
            })
    one, two, three = geom
    совпали = (one["форма"] == two["форма"] == three["форма"]
               and one["crs"] == two["crs"] == three["crs"]
               and one["transform"] == two["transform"] == three["transform"])
    if not совпали:
        raise SystemExit("CRS/transform/shape растров различаются — это FAIL, "
                         "а не допустимая картографическая погрешность")

    cls = arrays["overlap_class.tif"].ravel()
    counts = {int(k): int((cls == k).sum()) for k in range(5)}
    o10_raster = counts[4] / (counts[2] + counts[4])

    ref = pd.read_csv(metrics_csv)
    o10_csv = float(ref.loc[ref["признаков"] == 38, "совпадение_верх_10"].iloc[0])

    проверки = {
        "классы_совпали_с_ТЗ": counts == EXPECTED_CLASSES,
        "сумма_классов_равна_сетке": sum(counts.values()) == SHAPE[0] * SHAPE[1],
        "ядра_равны_по_площади": (counts[2] + counts[4]) == (counts[3] + counts[4]),
        "ядро_2291": (counts[2] + counts[4]) == 2291,
        "обучающая_метка_2295": int(t["pos_train_2295"].sum()) == 2295,
        "эталонное_ядро_2291": int(t["pos_reference_a10_2291"].sum()) == 2291,
        "ячейка_в_одном_классе": bool(((cls >= 0) & (cls <= 4)).all()),
        "вне_маски_не_в_ядрах": int(((~pool) & (a10 | b10)).sum()) == 0,
        "O10_из_растра_равен_CSV": abs(o10_raster - o10_csv) < 1e-9,
        "O10_из_растра_равен_455_2291": abs(o10_raster - EXPECTED_O10) < 1e-12,
    }

    отчёт = {
        "лист": "R-48-XI,XII",
        "сетка": {"форма": list(SHAPE), "ячеек": SHAPE[0] * SHAPE[1],
                  "шаг_м": CELL_M, "валидных": int(pool.sum())},
        "геопривязка": geom,
        "классы_overlap_class": {
            str(k): {"имя": CLASS_NAMES[k], "ячеек": counts[k],
                     "ожидалось": EXPECTED_CLASSES[k]} for k in range(5)},
        "O10": {"из_растра": o10_raster, "из_CSV": o10_csv,
                "разность": abs(o10_raster - o10_csv),
                "запись": "%d/%d" % (counts[4], counts[2] + counts[4])},
        "равные_порогу_ячейки": {
            "критериальная_карта": count_ties(crit, pool),
            "дистанционная_карта": count_ties(remote, pool)},
        "nan_внутри_валидной_маски": {
            "критериальная_карта": int((valid & ~np.isfinite(crit)).sum()),
            "дистанционная_карта": int((valid & ~np.isfinite(remote)).sum())},
        "проверки": проверки,
        "итог": "PASS" if all(проверки.values()) else "FAIL",
    }
    (out / "raster_geometry_check.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8")

    for k in range(5):
        print("класс %d  %-38s %6d  (ТЗ: %d)"
              % (k, CLASS_NAMES[k], counts[k], EXPECTED_CLASSES[k]))
    print("O10 из растра = %d/%d = %.17g" % (counts[4], counts[2] + counts[4],
                                             o10_raster))
    print("O10 из CSV    =            %.17g" % o10_csv)
    print("разность      = %.3g" % abs(o10_raster - o10_csv))
    for k, v in проверки.items():
        print("  %-32s %s" % (k, "OK" if v else "ОШИБКА"))
    print("итог:", отчёт["итог"])
    if отчёт["итог"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
