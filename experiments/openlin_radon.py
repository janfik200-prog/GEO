"""Линеаменты как ДЛИННЫЕ и УЗКИЕ объекты: интегрирование вдоль прямых.

ПОЧЕМУ ПРЕЖНИЕ ПОПЫТКИ НЕ ВИДЕЛИ РЕГИОНАЛЬНЫХ СТРУКТУР. Весь конвейер считал
признаки в окнах 0.5-2 км и собирал отрезки длиной в километры. Структура в
десятки километров в таком масштабе не даёт контраста ни в одном окне: она
размазана по всем сразу. Попытка просто сгладить вход до километра делает хуже —
изотропное сглаживание убивает поперечную узость, а именно она отличает линию от
пятна.

ЧТО ЗДЕСЬ. Растр поворачивается на каждый из нескольких десятков углов, и в
повёрнутой системе координат линия становится горизонтальной строкой. Тогда:

* усреднение ВДОЛЬ строки на десятки километров накапливает слабый сигнал линии
  и подавляет шум (в корень из числа ячеек);
* сравнение строки с соседними (в сотнях метров) сохраняет поперечную узость;
* нормировка на разброс внутри полос отсекает места, где сильный отклик даёт
  просто пёстрая текстура.

Это преобразование Радона с локальной нормировкой: классический способ искать
протяжённые прямые, и он работает на тех же данных, без смены источника.

Запуск из корня репозитория:
    python -m experiments.openlin_radon --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")

#: Длина накопления вдоль линии, м. Это и есть масштаб искомого объекта:
#: 20 км против прежних 2 км у оператора границы.
ALONG_M = 20000.0

#: Полуширина полосы поперёк линии и расстояние до сравниваемых соседних полос.
#: Узость сохраняется: 200 м — две ячейки растра.
ACROSS_M = 200.0
OFFSET_M = 600.0

#: Шаг перебора направлений, градусы. На длине 20 км ошибка в 2 градуса уводит
#: конец линии на 700 м — мельче дробить смысла нет.
ANGLE_STEP = 2.0


def _rotate(a: np.ndarray, angle: float, order: int = 1) -> np.ndarray:
    from scipy.ndimage import rotate

    return rotate(a, angle, reshape=True, order=order, mode="constant",
                  cval=0.0, prefilter=False)


def radon_lines(img: np.ndarray, valid: np.ndarray, res_m: float,
                along_m: float = ALONG_M, across_m: float = ACROSS_M,
                offset_m: float = OFFSET_M, step: float = ANGLE_STEP
                ) -> tuple[np.ndarray, np.ndarray]:
    """Сила длинной прямой и её азимут для каждой ячейки."""
    from scipy.ndimage import uniform_filter1d

    n_along = max(int(round(along_m / res_m)), 3)
    n_across = max(int(round(across_m / res_m)), 1)
    n_off = max(int(round(offset_m / res_m)), 2)

    x = np.where(valid, np.nan_to_num(img, nan=0.0), 0.0).astype("float32")
    w = valid.astype("float32")
    best = np.full(img.shape, -np.inf, dtype="float32")
    ang = np.zeros(img.shape, dtype="float32")
    h0, w0 = img.shape

    for a in np.arange(0.0, 180.0, step):
        xr, wr = _rotate(x, a), _rotate(w, a)
        # усреднение вдоль линии (ось 1 после поворота) с учётом маски
        num = uniform_filter1d(xr, n_along, axis=1, mode="nearest")
        den = uniform_filter1d(wr, n_along, axis=1, mode="nearest")
        mean = num / np.maximum(den, 1e-3)
        sq = uniform_filter1d(xr ** 2, n_along, axis=1, mode="nearest")
        var = np.maximum(sq / np.maximum(den, 1e-3) - mean ** 2, 1e-9)

        # полоса по центру против двух соседних, всё уже усреднено вдоль
        c = uniform_filter1d(mean, n_across, axis=0, mode="nearest")
        up = np.roll(c, +n_off, axis=0)
        dn = np.roll(c, -n_off, axis=0)
        v = uniform_filter1d(var, n_across, axis=0, mode="nearest")
        # эффективное число независимых отсчётов вдоль линии
        s = (c - 0.5 * (up + dn)) / np.sqrt(v / n_along + 1e-9)
        s = np.abs(s).astype("float32")
        s = np.where(wr > 0.5, s, 0.0)

        back = _rotate(s, -a)
        # поворот меняет размер: вырезаем центр исходного размера
        dy = (back.shape[0] - h0) // 2
        dx = (back.shape[1] - w0) // 2
        back = back[dy:dy + h0, dx:dx + w0]
        upd = back > best
        best = np.where(upd, back, best)
        ang = np.where(upd, a, ang)
    return np.where(valid, best, np.nan), ang


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--along", type=float, default=ALONG_M)
    ap.add_argument("--channel", default="gray")
    a = ap.parse_args()

    from experiments.openlin_unet import load_site

    def log(m=""):
        print(m, flush=True)

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    img = np.nan_to_num(z[a.channel],
                        nan=float(np.nanmedian(z[a.channel][valid])))
    log(f"площадка {a.site}, канал {a.channel}, накопление вдоль "
        f"{a.along / 1000:.0f} км, полоса {ACROSS_M:.0f} м")

    t0 = time.time()
    s, ang = radon_lines(img, valid, res_m, along_m=a.along)
    log(f"поле длинных прямых посчитано за {time.time() - t0:.0f} с")
    np.savez_compressed(M / f"openlin_radon_{a.site}.npz",
                        field=s.astype("float32"), azimuth=ang)
    log(f"сохранено: {M / f'openlin_radon_{a.site}.npz'}")


if __name__ == "__main__":
    main()
