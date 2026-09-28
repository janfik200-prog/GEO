"""Границы доменов: разлом как скачок признакового вектора, а не гребень скора.

ЗАЧЕМ ДРУГОЙ ОПЕРАТОР. Трассировка гребней вероятностного поля дала линии,
которые совпадают с обводкой рельефной текстуры (в 1.8 раза) лучше, чем с картой
разломов (в 1.36 раза), а частота в ансамбле не предсказывает попадание вовсе
(``openlin_lines_diag``). Причина в самом операторе: гребень поля «здесь вероятен
разлом» — это место, где поверхность поля выглядит как хребет, а не место, где
меняется геологический блок. Разлом — ГРАНИЦА между областями с разной
сигнатурой. Здесь и ищется именно она.

КАК СЧИТАЕТСЯ. Для каждой ячейки и каждого из ``N_ANGLES`` направлений берутся
два вытянутых вдоль этого направления окна, сдвинутых поперёк него на
``OFFSET_M`` в обе стороны, и считается расстояние между средними признаковыми
векторами этих окон. Сила границы — максимум по направлениям, азимут — на каком
направлении максимум достигнут. Признаки предварительно сжимаются PCA до
``N_PC`` компонент и отбеливаются, поэтому евклидово расстояние здесь — это
расстояние Махаланобиса в исходном признаковом пространстве, а сила границы
измеряется В СИГМАХ, то есть в тех же единицах, что и предел обнаружения из
``src/detection_limit.py`` (там порог 1.48-1.81 sigma, а реальные коридоры
0.086-0.194 sigma).

ЧЕМ ЭТО ЛУЧШЕ ПРЕДЫДУЩЕГО. Обучения здесь нет вообще: оператор считается прямо
из снимка, переносить нечего и переобучаться нечему. Значит и вопрос
воспроизводимости снимается: результат — детерминированная функция входа.

ПЛАНКИ. Кроме случайной линии обязательна вторая, введённая после диагностики:
линии рельефа, снятые тем же трассировщиком с голого композита без всякой
модели. Случайная линия оказалась слишком слабым эталоном.

Запуск из корня репозитория:
    python -m experiments.openlin_domains --site anabar
    python -m experiments.openlin_domains --site nevada,anabar,anabar_dykes
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import TOL_M, block_split
from experiments.openlin_lines import (DENSITY, DENSITY_DEFAULT, line_metrics,
                                       shift_null, trace_lines)
from experiments.openlin_lines_ens import thin
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, load_site

M = Path("outputs/metrics")

# --------------------------------------------------------------------------
# Параметры оператора
# --------------------------------------------------------------------------

#: Сколько направлений перебирается. 12 — шаг 15 градусов; на разломной сети с
#: двумя-тремя выраженными системами этого хватает, а стоимость линейна по числу
#: направлений.
N_ANGLES = 12

#: Сколько компонент PCA оставляем. 8 объясняют основную часть дисперсии 55
#: признаков, а расстояние в 8 измерениях считается на порядок дешевле, чем в 55.
N_PC = 8

#: Геометрия окна. Вдоль предполагаемой линии окно длинное (2 км — масштаб, на
#: котором разлом ещё прямой), поперёк узкое (200 м — вдвое шире полуширины
#: разметки). Сдвиг 300 м в каждую сторону: меньше — окна перекрываются и скачок
#: размазывается, больше — сравниваются уже не соседние блоки.
WIN_LONG_M = 2000.0
WIN_SHORT_M = 200.0
OFFSET_M = 300.0

N_SHIFTS = 8000

#: Пре-регистрация. Свой addend в своей секции. Обучения здесь нет, поэтому нет
#: и осей «модель» и «таргет»: кандидатов ровно столько, сколько площадок — 3.
#: ПЕРВИЧНАЯ МЕТРИКА: точность линий границ доменов при плотности
#: ``DENSITY_DEFAULT``. Остальные плотности считаются и печатаются как кривая,
#: но тестов по ним нет.
DOM_N_TESTS_PRIOR = 168        # 42 + 42 + 42 + 42 предыдущих фаз
DOM_N_TESTS = 3
DOM_ALPHA = 0.05 / (DOM_N_TESTS_PRIOR + DOM_N_TESTS)

#: Достижимость: 1/(1+8000) = 0.000125 против alpha = 0.00029, запас 2.3x.
assert 1.0 / (1 + N_SHIFTS) < DOM_ALPHA, "порог недостижим при таком числе сдвигов"


# --------------------------------------------------------------------------

def _kernel(sigma_long: float, sigma_short: float, theta: float,
            offset: float) -> np.ndarray:
    """Вытянутое вдоль ``theta`` гауссово окно, сдвинутое поперёк на ``offset``.

    ``theta`` — азимут от севера (ось строк направлена на юг, поэтому вдоль
    линии вектор ``(-cos, sin)``).
    """
    r = int(3 * max(sigma_long, sigma_short) + abs(offset) + 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype("float64")
    uy, ux = -np.cos(theta), np.sin(theta)      # вдоль линии
    vy, vx = np.sin(theta), np.cos(theta)       # поперёк линии
    py, px = yy - offset * vy, xx - offset * vx
    a = py * uy + px * ux
    b = py * vy + px * vx
    k = np.exp(-0.5 * ((a / sigma_long) ** 2 + (b / sigma_short) ** 2))
    return k / k.sum()


def _smooth(stack: np.ndarray, mask: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Нормированная на маску свёртка стека (C,H,W) одним ядром."""
    from scipy.signal import fftconvolve

    m = mask.astype("float32")
    den = fftconvolve(m, k, mode="same")
    num = fftconvolve(stack * m[None], k[None], mode="same")
    return num / np.maximum(den, 1e-3)[None]


def domain_edges(x: np.ndarray, valid: np.ndarray, res_m: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Сила границы доменов (в сигмах) и её азимут (градусы 0-180)."""
    from sklearn.decomposition import PCA

    rows = x[:, valid].T.astype("float32")
    sel = np.random.default_rng(42).choice(
        len(rows), size=min(50_000, len(rows)), replace=False)
    pca = PCA(n_components=min(N_PC, x.shape[0]), whiten=True,
              random_state=42).fit(rows[sel])
    comp = np.full((pca.n_components_,) + valid.shape, 0.0, dtype="float32")
    comp[:, valid] = pca.transform(rows).T
    return edges_from_components(comp, valid, res_m)


def edges_from_components(comp: np.ndarray, valid: np.ndarray, res_m: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Оператор границы поверх УЖЕ сжатого представления.

    Вынесено из :func:`domain_edges`, чтобы в одно и то же место подставлялись
    и компоненты PCA, и латент автоэнкодера: сравнение представлений имеет смысл
    только если всё остальное в точности одинаково.
    """
    best = np.full(valid.shape, -np.inf, dtype="float32")
    ang = np.zeros(valid.shape, dtype="float32")
    for i in range(N_ANGLES):
        th = np.pi * i / N_ANGLES
        kp = _kernel(WIN_LONG_M / res_m, WIN_SHORT_M / res_m, th,
                     OFFSET_M / res_m)
        km = _kernel(WIN_LONG_M / res_m, WIN_SHORT_M / res_m, th,
                     -OFFSET_M / res_m)
        d = _smooth(comp, valid, kp) - _smooth(comp, valid, km)
        strength = np.sqrt((d * d).sum(0)).astype("float32")
        upd = strength > best
        best = np.where(upd, strength, best)
        ang = np.where(upd, np.degrees(th), ang)
    return np.where(valid, best, np.nan), ang


def terrain_lines(z: dict, valid: np.ndarray, res_m: float,
                  density: str) -> np.ndarray:
    """Планка: линии рельефа тем же трассировщиком, без всякой модели."""
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    a, _ = trace_lines(gray, valid, valid, res_m, density)
    b, _ = trace_lines(-gray, valid, valid, res_m, density)
    return a | b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="anabar", help="через запятую")
    ap.add_argument("--density", default=DENSITY_DEFAULT, choices=tuple(DENSITY))
    ap.add_argument("--all-density", action="store_true",
                    help="посчитать кривую по всем плотностям")
    ap.add_argument("--no-null", action="store_true")
    args = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"направлений {N_ANGLES}, компонент PCA {N_PC}, окно "
        f"{WIN_LONG_M:.0f}x{WIN_SHORT_M:.0f} м, сдвиг {OFFSET_M:.0f} м")
    log(f"пре-регистрация: первичная метрика — точность при плотности "
        f"'{DENSITY_DEFAULT}'; тестов в семействе "
        f"{DOM_N_TESTS_PRIOR + DOM_N_TESTS}, alpha = {DOM_ALPHA:.5f}, "
        f"минимально достижимое p = {1 / (1 + N_SHIFTS):.6f}\n")

    out = []
    for site in [s for s in args.site.split(",") if s]:
        z = load_site(site)
        valid = np.isfinite(z["gray"])
        res_m = float(z["res_m"])
        dist = z["dist"]
        x, _y, _v, use = build_stack(z, CHANNELS)
        nx, _n = neigh_stack(x, use, valid, res_m)
        x = np.concatenate([x, nx])

        t0 = time.time()
        strength, ang = domain_edges(x, valid, res_m)
        log(f"{site}: поле границ посчитано за {time.time() - t0:.0f} с; "
            f"признаков {x.shape[0]}")

        # Контраст на самих разломах против фона — в тех же сигмах, что и предел
        # обнаружения в src/detection_limit.py.
        on = (dist <= TOL_M) & valid
        log(f"  сила границы: на разломах {np.nanmean(strength[on]):.3f} sigma, "
            f"вне {np.nanmean(strength[valid & ~on]):.3f} sigma, "
            f"отношение {np.nanmean(strength[on]) / np.nanmean(strength[valid & ~on]):.3f}")

        terr = terrain_lines(z, valid, res_m, args.density)
        terr_axis = thin(terr) & valid
        base = float(((dist <= TOL_M) & valid).sum() / valid.sum())
        r_terr = line_metrics(terr_axis, dist, valid, res_m)

        densities = tuple(DENSITY) if args.all_density else (args.density,)
        for dens in densities:
            lines, br = trace_lines(np.nan_to_num(strength, nan=0.0), valid,
                                    valid, res_m, dens)
            r = line_metrics(lines, dist, valid, res_m)
            row = {"площадка": site, "плотность": dens, "линий": len(br),
                   "длина_км": r["длина_км"], "точность": r["точность"],
                   "полнота": r["полнота"], "случайная": base,
                   "линии_рельефа": r_terr["точность"]}
            if not args.no_null and dens == DENSITY_DEFAULT and lines.any():
                row.update(shift_null(lines, dist, valid, r["точность"],
                                      n_shifts=N_SHIFTS))
                row["значим"] = row["p"] < DOM_ALPHA
            out.append(row)
            pp = f"  p = {row['p']:.5f}" if "p" in row else ""
            log(f"  плотность {dens:<7} линий {len(br):>4}  {r['длина_км']:>5.0f} км"
                f"  точность {r['точность']:.3f}  полнота {r['полнота']:.3f}"
                f"  (случайная {base:.3f}, линии рельефа "
                f"{r_terr['точность']:.3f}){pp}")

        np.savez_compressed(M / f"openlin_domains_{site}.npz",
                            strength=np.nan_to_num(strength, nan=0.0), angle=ang)
        log()

    df = pd.DataFrame(out)
    path = M / "openlin_domains.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"сохранено: {path}")


if __name__ == "__main__":
    main()
