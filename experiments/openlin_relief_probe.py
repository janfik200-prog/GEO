"""Проба: отмывка рельефа с нескольких азимутов — закрывает ли она слепую полосу.

ОТКУДА ПОСТАНОВКА. Оптика при южном солнце (184°) не видит уступы, вытянутые
вдоль освещения: слепая полоса ±15° у меридиана измерена врезкой
(:mod:`openlin_azimuth_response`). Спектральные каналы её не закрывают —
геометрия солнца у них та же. Заказчик 28.09.2026 разрешил модель рельефа
(«можно, но лучше без неё»), и в литературе отмывка с нескольких азимутов —
стандартный приём против отбора по направлению. Основной результат остаётся
без рельефа; это проба отдельного, помеченного варианта.

ЧТО МЕРИТСЯ. Copernicus GLO-30 на сетке листа -> отмывки с азимутов 0, 45, 90,
135° при высоте солнца 9° (как у зимних сцен) -> тот же детектор. Вопросы:
  1. сколько линий на отмывках у снимка и у двойника (не шум ли это);
  2. в секторах, которые оптика ВИДИТ, какая доля линий рельефа совпадает с
     рабочим набором оптики — это мера того, что линии рельефа — те же
     объекты;
  3. сколько линий рельефа лежит в слепом секторе оптики и нет ли их в наборе.
Подтверждения двумя съёмками у рельефа нет (модель одна), поэтому это замер,
а не прогон: в счётчик не идёт.

Запуск из корня репозитория:
    python -m experiments.openlin_relief_probe --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from cache_paths import ANABAR_DEM
from experiments.openlin_zone_lines import COMP, M, _adiff, mask_of, trace

AZIMUTHS = (0.0, 45.0, 90.0, 135.0)   # противоположные дают тот же рисунок с обратным знаком
SUN_ELEV = 9.0                        # как у зимних сцен (7.7-9.6°)
SUN_AZ_OPT = 184.0
BLIND_DEG = 15.0                      # измеренная полуширина слепой полосы оптики


def load_dem(site: str, res_m: float) -> np.ndarray:
    """GLO-30 на сетке листа (кэш); плитки читаются окном прямо с AWS."""
    from rasterio.enums import Resampling

    from experiments.openlin_probe import _read_on_grid
    from experiments.openlin_winter import site_grid
    from src.dem_features import _tile_url

    path = ANABAR_DEM / f"dem_glo30_{site}_лист_{int(res_m)}m.npy"
    if path.exists():
        return np.load(path)
    grid = site_grid(site, res_m)
    lon0, lat0, lon1, lat1 = grid.bbox_wgs84()
    dem = np.zeros((grid.height, grid.width), "float32")
    for lat in range(int(np.floor(lat0)), int(np.floor(lat1)) + 1):
        for lon in range(int(np.floor(lon0)), int(np.floor(lon1)) + 1):
            try:
                part = _read_on_grid(_tile_url(lat, lon), grid, Resampling.bilinear)
            except Exception as exc:                 # плитки над морем может не быть
                print(f"  плитка {lat},{lon}: нет ({exc!r:.80})", flush=True)
                continue
            dem = np.where(part != 0, part, dem)
            print(f"  плитка {lat},{lon}: покрыто {100 * (part != 0).mean():.0f} % листа",
                  flush=True)
    dem = np.where(dem != 0, dem, np.nan).astype("float32")
    ANABAR_DEM.mkdir(parents=True, exist_ok=True)
    np.save(path, dem)
    return dem


def hillshade(dem: np.ndarray, res_m: float, az_deg: float) -> np.ndarray:
    """Обычная отмывка: освещённость склона при солнце с азимута az_deg."""
    gy, gx = np.gradient(dem, res_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)                  # север сверху: строки растут к югу
    zen = np.radians(90.0 - SUN_ELEV)
    az = np.radians(az_deg)
    return (np.cos(zen) * np.cos(slope)
            + np.sin(zen) * np.sin(slope) * np.cos(az - aspect)).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_kora import surrogate
    from experiments.openlin_winter import detrend
    from experiments.openlin_zone_lines_v3 import CONF_FRAC, CONF_TOL_M, work_npz

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    dem = load_dem(a.site, res_m)
    valid = np.isfinite(dem) & np.isfinite(w["shade"])
    print(f"рельеф: высоты {np.nanmin(dem):.0f}-{np.nanmax(dem):.0f} м, покрытие "
          f"{100 * np.isfinite(dem).mean():.1f} % ({time.time() - t0:.0f} с)", flush=True)
    hs = [detrend(hillshade(np.nan_to_num(dem, nan=float(np.nanmedian(dem))), res_m, az),
                  valid, res_m) for az in AZIMUTHS]
    stack = np.stack(hs).astype("float32")
    twin = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900))

    work = np.load(work_npz(a.site))
    d_conf = distance_transform_edt(~work["C_all"], sampling=(res_m, res_m))
    d_any = distance_transform_edt(~work["A_all"], sampling=(res_m, res_m))

    def near(L, d):
        return np.array([(d[x["pts"][:, 0], x["pts"][:, 1]] <= CONF_TOL_M).mean()
                         >= CONF_FRAC for x in L]) if L else np.zeros(0, bool)

    print("\nазимут отмывки | линий снимок / двойник | в слепом секторе оптики "
          "снимок / двойник | совпадают с рабочим набором: видимые сектора / слепой",
          flush=True)
    union_blind = np.zeros(valid.shape, bool)
    for k, az in enumerate(AZIMUTHS):
        Lr = trace(np.where(valid, stack[k], np.nan), valid, res_m, local=True)
        Lt = trace(np.where(valid, twin[k], np.nan), valid, res_m, local=True)
        br = np.array([_adiff(x["азимут"], SUN_AZ_OPT) <= BLIND_DEG for x in Lr])
        bt = np.array([_adiff(x["азимут"], SUN_AZ_OPT) <= BLIND_DEG for x in Lt])
        m = near(Lr, d_conf)
        seen = 100 * m[~br].mean() if (~br).any() else 0
        blind = 100 * m[br].mean() if br.any() else 0
        km_b = sum(x["длина_км"] for x, b in zip(Lr, br) if b)
        union_blind |= mask_of([x for x, b in zip(Lr, br) if b], valid.shape)
        print(f"  {az:5.0f}°       | {len(Lr):5d} / {len(Lt):5d}          | "
              f"{int(br.sum()):4d} ({km_b:.0f} км) / {int(bt.sum()):4d}            | "
              f"{seen:4.0f} % / {blind:4.0f} %   ({time.time() - t0:.0f} с)", flush=True)

    p = np.argwhere(union_blind)
    new = (d_any[p[:, 0], p[:, 1]] > CONF_TOL_M).mean() if len(p) else 0
    print(f"\nячейки линий рельефа в слепом секторе (все четыре отмывки вместе): "
          f"{len(p) * res_m / 1000:.0f} км; из них вне любых линий оптики "
          f"(дальше {CONF_TOL_M:.0f} м): {100 * new:.0f} %", flush=True)
    np.savez_compressed(M / f"openlin_relief_probe_{a.site}.npz", blind=union_blind)
    print(f"готово ({time.time() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
