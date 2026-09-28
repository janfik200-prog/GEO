"""Уступ по тени: детектор, устроенный по физике низкого солнца.

ЗАЧЕМ ОТДЕЛЬНЫЙ ДЕТЕКТОР. Прежняя цепочка (признаки окрестности 0.5-2 км,
сжатие в компоненты, оператор по компонентам) на зимнем композите даёт то же,
что и раньше: на фазовой подделке линий не меньше, чем на настоящем входе.
И не может дать другого — измеренные нагрузки компонент показывают, что 90 %
веса там приходится на усреднение по окрестности, то есть тонкий протяжённый
контраст уничтожается ДО оператора. Менять вход, оставляя этот тракт, смысла
нет.

На зимнем композите копать линию из многомерного описания и не нужно: она
видна прямо в одном канале. Зато появляется признак, которого раньше не было
ни в каком виде.

ФИЗИКА. Все сцены композита сняты при солнце с юга (азимут 178-188°) и высоте
8-10°. Уступ высотой в первые метры при такой высоте солнца даёт тень длиной в
десятки метров. Поэтому уступ — это не «граница двух областей вообще», а пара
«освещённый склон / тень за ним», ориентированная ВСЕГДА одинаково
относительно солнца. Отсюда три шага:

1. производная яркости ВДОЛЬ направления, в котором ложится тень. Уступ даёт
   в ней резкий отклик; пятно яркости (другая порода, другой снег) не даёт
   ничего, потому что не создаёт перепада именно в этом направлении;
2. накопление отклика ВДОЛЬ простирания на километры при узости поперёк.
   Слабый, но протяжённый сигнал растёт как корень из числа отсчётов, а шум
   гасится; так отличается линия от края склона;
3. ПОСТОЯНСТВО ЗНАКА вдоль линии. Настоящий уступ по всей длине повёрнут к
   солнцу одной и той же стороной. Случайная текстура, набравшая ту же
   амплитуду, знак по длине перекидывает. Этот признак не требует ни эталона,
   ни второго прибора и стоит ровно ничего.

ЧЕГО ЭТОТ ДЕТЕКТОР НЕ УВИДИТ. Структуру, идущую строго на солнце (север-юг):
она не даёт перепада освещённости вовсе. Это ограничение самой съёмки, а не
обработки, и его надо помнить при разборе роз азимутов.

Запуск из корня репозитория:
    python -m experiments.openlin_shade --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")

#: Масштаб сглаживания перед производной, м. Меньше — лезет шум отдельных
#: ячеек, больше — размазывается сама тень. Перебор (:mod:`openlin_shade_sweep`)
#: показал, что на исход это не влияет вовсе: 40, 60, 120 и 240 м дали
#: одинаковый урожай. Значение оставлено от лучшего набора.
SMOOTH_M = 40.0

#: Длина накопления вдоль простирания, м. Первая версия требовала 4 км, и это
#: была главная ошибка геометрии: урожай при 10 % ложных вырос в 340 раз при
#: замене на 1 км (0.001 % площади против 0.340 %). То, что видно на снимке,
#: живёт на масштабе километра, и при четырёхкилометровом окне не выживало в
#: требовании непрерывности по всем отрезкам. 8 и 16 км не дают вообще ничего.
ALONG_M = 1000.0

#: Ширина полосы поперёк, м. На урожай тоже не влияет (60, 100, 200 и 400 м
#: дали одно и то же), меняется только численное значение порога.
ACROSS_M = 60.0

#: Отступ соседних полос сравнения поперёк простирания, м. Без этого сравнения
#: полоса меряется сама с собой, и плавно наклонённый участок плато набирает
#: идеальную непрерывность и огромную силу, будучи обычным склоном. Величина
#: отступа при этом важна: 200, 400 и 800 м дают 0.001 % площади, 1600 м — уже
#: 0.170 %. Ближняя полоса сравнения попадает на ту же структуру и вычитает её
#: саму из себя.
OFFSET_M = 1600.0

#: Шаг перебора простираний, градусы. На километре ошибка в 3° уводит конец
#: линии на 50 м — это меньше ширины полосы, мельче дробить нечего.
STEP_DEG = 3.0


def load_winter(site: str, res_m: float, full: bool = True,
                path: str | None = None) -> dict:
    """Композит-вход. По умолчанию зимний оптический, но тем же детектором
    обрабатывается и радарный (:mod:`openlin_radar`): у него та же физика
    подсветки, только азимут другой."""
    if path:
        return dict(np.load(path))
    p = COMP / (f"winter_{site}_{int(res_m)}m" + ("_лист" if full else "")
                + ".npz")
    if not p.exists():
        raise SystemExit(f"нет {p} — сначала: python -m experiments."
                         f"openlin_winter --site {site} --full")
    return dict(np.load(p))


def sun_derivative(img: np.ndarray, valid: np.ndarray, res_m: float,
                   az_deg: float, smooth_m: float = SMOOTH_M) -> np.ndarray:
    """Производная яркости вдоль направления, в котором ложится тень."""
    from scipy.ndimage import gaussian_filter

    # азимут солнца — откуда светит; тень ложится в противоположную сторону
    a = np.radians(az_deg + 180.0)
    d_col, d_row = np.sin(a), -np.cos(a)     # строка растёт на юг
    x = np.where(valid, np.nan_to_num(img), 0.0).astype("float32")
    s = max(smooth_m / res_m, 0.8)
    g_row = gaussian_filter(x, s, order=(1, 0), mode="nearest")
    g_col = gaussian_filter(x, s, order=(0, 1), mode="nearest")
    d = g_row * d_row + g_col * d_col
    return np.where(valid, d, np.nan).astype("float32")


def _rot(a: np.ndarray, ang: float, order: int = 1) -> np.ndarray:
    from scipy.ndimage import rotate

    return rotate(a, ang, reshape=True, order=order, mode="constant", cval=0.0,
                  prefilter=False)


def _crop(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    dy = (a.shape[0] - shape[0]) // 2
    dx = (a.shape[1] - shape[1]) // 2
    return a[dy:dy + shape[0], dx:dx + shape[1]]


#: На сколько отрезков делится окно накопления. Это и есть требование
#: непрерывности: берётся МИНИМУМ по отрезкам, а не среднее по всему окну.
N_PARTS = 4


def strike_scan(d: np.ndarray, valid: np.ndarray, res_m: float,
                along_m: float = ALONG_M, across_m: float = ACROSS_M,
                step: float = STEP_DEG, smooth_m: float = SMOOTH_M,
                n_parts: int = N_PARTS, offset_m: float = OFFSET_M,
                log=None) -> dict:
    """Перебор простираний: сила, непрерывность, знак и азимут в каждой ячейке.

    Для каждого угла растр поворачивается так, чтобы простирание легло вдоль
    строки. Тогда накопление — это усреднение по строке, а узость поперёк
    сохраняется узким окном по столбцу.

    ПОЧЕМУ МИНИМУМ, А НЕ СРЕДНЕЕ. Первая версия усредняла отклик по всему окну
    накопления и давала поле из «комет»: окно сообщало о линии всюду, где оно
    хоть краем задевало сильный элемент, и каждый обрывок долины размазывался в
    хвост во всю длину окна. Требования, чтобы линия существовала по всей длине
    окна, в среднем попросту нет.

    Поэтому окно делится на ``n_parts`` отрезков, и берётся минимум модуля
    отрезочных средних при условии, что знак у всех отрезков один. Элемент,
    попавший в край окна, даёт ноль: в дальних отрезках отклика нет. Выживает
    только то, что тянется всю длину окна и всюду повёрнуто к солнцу одной
    стороной.
    """
    from scipy.ndimage import uniform_filter1d

    n_sub = max(int(round(along_m / res_m / n_parts)), 3)
    n_across = max(int(round(across_m / res_m)), 1)
    n_off = max(int(round(offset_m / res_m)), 2)
    # независимых отсчётов на отрезке меньше, чем ячеек: соседние ячейки
    # связаны сглаживанием, сделанным перед производной
    n_eff = max(along_m / n_parts / (2.0 * smooth_m), 2.0)
    # центры отрезков относительно ячейки: ..., -1.5, -0.5, +0.5, +1.5 ...
    offs = [int(round((k - (n_parts - 1) / 2.0) * n_sub))
            for k in range(n_parts)]

    x = np.where(valid, np.nan_to_num(d), 0.0).astype("float32")
    w = valid.astype("float32")
    shape = d.shape
    best = {k: np.zeros(shape, dtype="float32")
            for k in ("сила", "непрерывность", "знак", "азимут", "оценка")}
    t0 = time.time()
    angles = np.arange(0.0, 180.0, step)

    def along(v):
        return uniform_filter1d(v, n_sub, axis=1, mode="nearest")

    def across(v):
        return uniform_filter1d(v, n_across, axis=0, mode="nearest")

    for i, ang in enumerate(angles):
        xr, wr = _rot(x, ang), _rot(w, ang)
        den = np.maximum(across(along(wr)), 1e-3)
        c = across(along(xr)) / den                        # среднее отрезка
        amean = across(along(np.abs(xr))) / den
        sq = across(along(xr ** 2)) / den
        sd = np.sqrt(np.maximum(sq - c ** 2, 1e-12))
        # центральная полоса против двух соседних: убирает региональный наклон
        mean = c - 0.5 * (np.roll(c, n_off, axis=0)
                          + np.roll(c, -n_off, axis=0))
        # разность трёх полос шумит примерно в полтора раза сильнее одной
        sd = sd * np.float32(np.sqrt(1.5))

        lo = None                    # минимум модуля по отрезкам
        hi = None                    # максимум модуля по отрезкам
        sgn = None                   # знак совпадает у всех отрезков
        sd_sum = np.zeros_like(sd)
        am_sum = np.zeros_like(sd)
        for o in offs:
            m_k = np.roll(mean, o, axis=1)
            a_k = np.abs(m_k)
            sd_sum += np.roll(sd, o, axis=1)
            am_sum += np.roll(amean, o, axis=1)
            s_k = m_k >= 0
            if lo is None:
                lo, hi, sgn = a_k, a_k.copy(), s_k
            else:
                lo = np.minimum(lo, a_k)
                hi = np.maximum(hi, a_k)
                sgn &= (s_k == (mean >= 0))
        sd_typ = sd_sum / len(offs)
        am_typ = am_sum / len(offs)

        # сила: во сколько сигм отличается от нуля САМЫЙ СЛАБЫЙ отрезок
        t = lo / (sd_typ / np.sqrt(n_eff) + 1e-9)
        # непрерывность: 1 — отрезки равны, 0 — один несёт всё
        cont = lo / (hi + 1e-9)
        # постоянство знака внутри отрезков
        r = lo / (am_typ + 1e-9)
        ok = (wr > 0.5) & sgn
        t = np.where(ok, t, 0.0).astype("float32")
        cont = np.where(ok, cont, 0.0).astype("float32")
        r = np.where(ok, r, 0.0).astype("float32")

        s_back = _crop(_rot(t * cont, -ang), shape)
        upd = s_back > best["оценка"]
        best["оценка"] = np.where(upd, s_back, best["оценка"])
        best["сила"] = np.where(upd, _crop(_rot(t, -ang), shape), best["сила"])
        best["непрерывность"] = np.where(
            upd, _crop(_rot(cont, -ang), shape), best["непрерывность"])
        best["знак"] = np.where(upd, _crop(_rot(r, -ang), shape), best["знак"])
        best["азимут"] = np.where(upd, ang, best["азимут"])
        if log and (i + 1) % 10 == 0:
            log(f"    простираний {i + 1}/{len(angles)} "
                f"({time.time() - t0:.0f} с)")
    return {k: np.where(valid, v, np.nan) for k, v in best.items()}


def fdr_curve(real: np.ndarray, fake: np.ndarray) -> "object":
    """Сколько ячеек остаётся при каждом пороге и какая доля из них — мусор.

    Доля мусора не предполагается, а меряется: подделка не содержит ни одного
    протяжённого объекта, поэтому КАЖДАЯ её ячейка выше порога — ложная. Их
    число, приведённое к одинаковой площади, и есть ожидаемое число ложных на
    настоящем входе.
    """
    import pandas as pd

    a = real[np.isfinite(real)]
    b = fake[np.isfinite(fake)]
    scale = len(a) / max(len(b), 1)
    rows = []
    for t in np.arange(2.0, 9.01, 0.25):
        n = int((a >= t).sum())
        if n == 0:
            break
        nf = float((b >= t).sum()) * scale
        rows.append({"порог": float(t), "ячеек": n,
                     "доля мусора, %": 100 * nf / n})
    return pd.DataFrame(rows)


def lines_from_field(score: np.ndarray, res_m: float, thr: float,
                     min_len_m: float = 1000.0
                     ) -> tuple[np.ndarray, "object"]:
    """Линии выше калиброванного порога.

    Трассировка по квантилям площади здесь не годится: сигнал занимает доли
    процента листа, и квантильный порог забирает вместе с ним весь шумовой вал.
    Порог берётся из кривой доли ложных, а дальше остаётся только сделать из
    пятен осевые линии.
    """
    import pandas as pd
    from skimage.morphology import remove_small_objects, skeletonize

    from experiments.openlin_lines import _axis, _branches, _path_length

    m = np.nan_to_num(score, nan=0.0) >= thr
    # пятно меньше полосы усреднения — это одна ячейка шума, а не линия
    m = remove_small_objects(m, int(round(4 * ACROSS_M / res_m)))
    sk = skeletonize(m)
    rows, keep = [], np.zeros_like(sk)
    for pts in _branches(sk, res_m):
        if len(pts) < 3:
            continue
        ax, _a, _b = _axis(pts)
        proj = (pts - pts.mean(0)) @ ax
        length = float(proj.max() - proj.min()) * res_m
        if length < min_len_m:
            continue
        path = _path_length(pts, res_m)
        keep[pts[:, 0], pts[:, 1]] = True
        rows.append({"длина_км": length / 1000.0,
                     "прямолинейность": min(length / path, 1.0) if path else 0,
                     "азимут": float(np.degrees(np.arctan2(ax[1], -ax[0]))
                                     % 180),
                     "оценка": float(np.nanmean(score[pts[:, 0], pts[:, 1]]))})
    return keep, pd.DataFrame(rows)


def _stretch(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    v = a[np.isfinite(a)]
    p_lo, p_hi = np.percentile(v, [lo, hi])
    return np.clip((a - p_lo) / max(p_hi - p_lo, 1e-9), 0, 1)


def plot(site: str, density: str, log, win: tuple | None = None,
         lines: bool = True, img: np.ndarray | None = None) -> None:
    """Композит, производная вдоль тени, поле уступов и линии поверх снимка."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_lines import trace_lines

    z = dict(np.load(M / f"openlin_shade_{site}.npz"))
    res_m = float(z["res_m"])
    if img is None:
        img = load_winter(site, res_m)["shade"].astype("float32")
    sc = z["оценка"]
    valid = np.isfinite(sc)

    if lines:
        t0 = time.time()
        mask, df = trace_lines(np.nan_to_num(sc, nan=0.0), valid, valid,
                               res_m, density)
        log(f"линий прослежено {len(df)}, суммарно "
            f"{mask.sum() * res_m / 1000:.0f} км ({time.time() - t0:.0f} с)")
    else:
        mask, df = np.zeros(sc.shape, dtype=bool), []

    if win:
        r, c, n = win
        sl = (slice(r, r + n), slice(c, c + n))
        img, sc, mask = img[sl], sc[sl], mask[sl]
        z = {k: (v[sl] if np.ndim(v) == 2 else v) for k, v in z.items()}
        valid = np.isfinite(sc)

    fig, ax = plt.subplots(1, 4, figsize=(23, 6.2))
    base = _stretch(img)
    ax[0].imshow(base, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("зимний композит", fontsize=13)
    ax[1].imshow(_stretch(z["dsun"], 1, 99), cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("производная вдоль тени", fontsize=13)
    ax[2].imshow(_stretch(sc, 50, 99.5), cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("поле уступов: сила x непрерывность", fontsize=13)
    ax[3].imshow(base, cmap="gray", vmin=0, vmax=1)
    ov = np.zeros(mask.shape + (4,), dtype="float32")
    ov[mask] = (1.0, 0.15, 0.1, 1.0)
    ax[3].imshow(ov)
    ax[3].set_title(f"линии поверх снимка ({len(df)} шт.)"
                    if lines else "трассировка не считалась", fontsize=13)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    p = Path("outputs/steps") / (f"уступы_{site}"
                                 + ("_крупно" if win else "") + ".png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=105)
    log(f"сохранено: {p}")
    np.save(M / f"_shade_mask_{site}.npy", mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--res", type=float, default=20.0)
    ap.add_argument("--along", type=float, default=ALONG_M)
    ap.add_argument("--across", type=float, default=ACROSS_M)
    ap.add_argument("--step", type=float, default=STEP_DEG)
    ap.add_argument("--channel", default="shade")
    ap.add_argument("--density", default="густо")
    ap.add_argument("--plot", action="store_true",
                    help="только картинки по уже посчитанному полю")
    ap.add_argument("--no-lines", action="store_true",
                    help="не тратить время на трассировку: только поле")
    ap.add_argument("--input", default=None,
                    help="другой композит вместо зимнего оптического")
    ap.add_argument("--tag", default="",
                    help="суффикс имён результатов, чтобы не затирать прогоны")
    ap.add_argument("--fake", action="store_true",
                    help="фазовая подделка входа: спектр и текстура те же, "
                         "протяжённых объектов нет")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    if a.plot:
        plot(a.site, a.density, log, lines=not a.no_lines)
        plot(a.site, a.density, log, win=(1400, 1500, 900),
             lines=not a.no_lines)
        return

    z = load_winter(a.site, a.res, path=a.input)
    img = z[a.channel].astype("float32")
    valid = np.isfinite(img)
    if a.fake:
        from experiments.openlin_null_input import phase_randomize

        rng = np.random.default_rng(100)
        img = np.where(valid, phase_randomize(img, valid, rng), np.nan)
        img = img.astype("float32")
    az = float(np.median(z["azimuth"]))
    sun = float(np.nanmedian(z["sun"]))
    log(f"площадка {a.site}: зимний композит {img.shape[0]} x {img.shape[1]}, "
        f"шаг {a.res:.0f} м, канал {a.channel}")
    log(f"подсветка с азимута {az:.0f}° (высота "
        + (f"{sun:.1f}°" if np.isfinite(sun) else "радар, сбоку")
        + f") — тень ложится на {(az + 180) % 360:.0f}°")
    log(f"накопление {a.along / 1000:.1f} км вдоль, полоса {a.across:.0f} м "
        f"поперёк, шаг простирания {a.step:.0f}°")

    t0 = time.time()
    d = sun_derivative(img, valid, a.res, az)
    log(f"\nпроизводная вдоль тени посчитана за {time.time() - t0:.0f} с")
    res = strike_scan(d, valid, a.res, a.along, a.across, a.step, log=log)
    log(f"поле уступов посчитано за {time.time() - t0:.0f} с")

    s = res["оценка"][valid]
    hi = np.percentile(s, 99)
    log(f"\nоценка: медиана {np.median(s):.2f}, 99-й процентиль {hi:.2f}, "
        f"максимум {s.max():.2f}")
    # проверка неподтасованная: постоянство знака смотрится в верхнем проценте
    # СИЛЫ, в которую оно не входит. Смотреть его в верхнем проценте оценки
    # бессмысленно — оценка сама на него множится
    f = res["сила"][valid]
    top = f >= np.percentile(f, 99)
    log(f"непрерывность: по листу "
        f"{np.median(res['непрерывность'][valid]):.2f}, в верхнем проценте "
        f"силы {np.median(res['непрерывность'][valid][top]):.2f}")

    p = M / (f"openlin_shade_{a.site}{a.tag}"
             + ("_подделка" if a.fake else "") + ".npz")
    np.savez_compressed(p, dsun=d.astype("float32"),
                        **{k: v.astype("float32") for k, v in res.items()},
                        res_m=a.res, azimuth=az, sun=sun,
                        x0=z["x0"], y_top=z["y_top"], epsg=z["epsg"])
    log(f"\nсохранено: {p}")
    if a.fake:
        return                       # картинки рисуются только по настоящему
    plot(a.site + a.tag, a.density, log, lines=not a.no_lines,
         img=img)


if __name__ == "__main__":
    main()
