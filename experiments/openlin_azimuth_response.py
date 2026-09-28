"""Передаточная функция трассировки по азимуту: какие направления мы видим.

ОТКУДА ПОСТАНОВКА. Популяция простираний найденных линий (сектор 30-60°
преобладает) может быть свойством не листа, а прибора: при низком солнце уступ,
вытянутый вдоль азимута подсветки, не даёт перепада яркости и гаснет, а уступ
поперёк — усиливается. Фазовая подделка этого не ловит по построению: IAAFT
сохраняет амплитудный спектр, то есть воспроизводит ту же анизотропию, и
отвечает только на вопрос «не шум ли», а не «не цензура ли».

ЧТО МЕРИТСЯ. В РЕАЛЬНЫЙ зимний композит врезается синтетический уступ
известного контраста, длины и ширины (:func:`src.detection_limit.insert_fault`,
тот же инструмент, что в «шаге 0» основной задачи), после чего тот же самый
тракт трассировки спрашивается, нашёл ли он его. Перебираются азимуты (12
секторов по 15°) и амплитуды, и первая амплитуда — НОЛЬ: это встроенный нуль,
доля ложных тревог того же прибора на той же площадке. «Поймано» без него
ничего не значит.

ДВА РЕЖИМА, потому что вопросов два:
  * ``одинаковый контраст`` — амплитуда одна для всех азимутов. Меряет
    смещение САМОГО ДЕТЕКТОРА: нормировку по окрестности, подавление немаксимумов,
    гистерезис и анизотропию фона;
  * ``одинаковая высота`` — амплитуда умножается на |sin(азимут - солнце)|,
    ламбертово приближение для уступа одной высоты при солнце на 184°. Меряет
    цензуру ОСВЕЩЕНИЕМ вместе с детектором, то есть то, что видит геолог.

ЭТО ЗАМЕР, А НЕ ГИПОТЕЗА. Ни одного p-value; накопительный знаменатель
проверок ветки не растёт.

Запуск из корня репозитория:
    python -m experiments.openlin_azimuth_response --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import (COMP, DIR_TOL_DEG, M, MIN_LEN_M,
                                            OUT, trace)
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, extend_ends

# ---- пре-регистрация (23.09.2026, до прогона) ------------------------------
AZ_BINS = 12                 # шаг 15°, как ROSE_BIN ветки
#: Лестница амплитуд ИЗМЕРЕНА, а не взята из «шага 0»: там ладдер 0..4 sigma
#: рассчитан на детектор края, а наш ищет узкий гребень при локальной
#: нормировке отклика, и порог у него выше. Пробный свип на вырезке 20 x 20 км
#: (23.09.2026): 2 и 4 sigma — 0 из 4 врезок, 8 sigma — 1 из 4, 16 sigma — 4 из
#: 4. Поэтому лестница покрывает найденный диапазон, а ноль остаётся первым.
AMPS = (0.0, 4.0, 8.0, 12.0, 16.0)
#: Длина врезки — близко к медиане длины нашего подтверждённого набора (1.68 км),
#: чтобы мерить обнаружение тех объектов, о которых мы говорим, а не любых.
INS_LEN_M = 2000.0
#: Ширина и профиль — по тому, что детектор физически ищет. Отклик гребня
#: считается на ядрах SIGMAS = (1, 2) ячейки, то есть 20 и 40 м, и ловит УЗКУЮ
#: ТЕНЬ от уступа, а не саму ступень: при солнце в 5° уступ высотой 3.5 м даёт
#: тень 40 м. Первая версия замера врезала ступень шириной 100 м и получила
#: нули на всех амплитудах вплоть до 2 sigma — не потому, что детектор слеп, а
#: потому, что врезался объект, на который он не реагирует по построению.
INS_WIDTH_M = 40.0
PROFILE = "ridge"
N_PER_BIN = 8                # врезок на азимут в одном прогоне
#: Разнос врезок: окно детренда у каждой 3 sigma = 3 км, поэтому 8 км развода
#: гарантируют, что остаточные поля соседних врезок не складываются.
SITE_MIN_KM = 8.0
SUN_AZ = 184.0
MATCH_FRAC = 0.5             # доля следа врезки, покрытая линией
MATCH_TOL_M = CONF_TOL_M     # 60 м, как допуск подтверждения
MATCH_TOL_DEG = DIR_TOL_DEG  # 20°, как допуск направления
SEED = 17
# ---------------------------------------------------------------------------


def sites(shape, res_m: float, valid, rng) -> np.ndarray:
    """Центры врезок: внутри кадра, не ближе SITE_MIN_KM друг к другу."""
    step = int(SITE_MIN_KM * 1000 / res_m)
    pad = int(INS_LEN_M / res_m)
    rr = np.arange(pad, shape[0] - pad, step)
    cc = np.arange(pad, shape[1] - pad, step)
    grid = np.array([(r, c) for r in rr for c in cc])
    ok = np.array([valid[r, c] for r, c in grid])
    grid = grid[ok]
    rng.shuffle(grid)
    return grid


def insert_window(x, center, az_deg: float, amp: float, sigma: float,
                  res_m: float) -> None:
    """Врезка в окне вокруг центра, на месте.

    Две вещи, без которых врезка была бы неверной.

    ПЕРВАЯ — единицы угла. ``insert_fault`` принимает угол ОСИ В РАСТРЕ, а у нас
    везде географический азимут; перевод — ``axis_from_geographic``.

    ВТОРАЯ — дальнее поле. Профиль ``scarp`` это ступень: одна сторона светлее
    на всю ширину растра. В окне такая ступень дала бы перепад по границе окна,
    то есть паразитный прямоугольник вокруг каждой врезки. И дело не только в
    окне: наш композит сам детрендирован на ``DETREND_M`` (2 км), то есть
    настоящий уступ в этих данных выглядит как ступень МИНУС её же сглаженная
    копия. Поэтому к врезке применяется тот же детренд — и дальнее поле
    исчезает само, без единого нового параметра.
    """
    from scipy.ndimage import gaussian_filter

    from experiments.openlin_winter import DETREND_M
    from src.detection_limit import axis_from_geographic, insert_fault

    s = DETREND_M / res_m / 2.0
    half = int(INS_LEN_M / 2 / res_m + 3 * s) + 2
    r0, c0 = max(int(center[0]) - half, 0), max(int(center[1]) - half, 0)
    r1 = min(int(center[0]) + half + 1, x.shape[0])
    c1 = min(int(center[1]) + half + 1, x.shape[1])
    win = np.zeros((r1 - r0, c1 - c0), dtype="float64")
    step = insert_fault(
        win, (float(center[0] - r0), float(center[1] - c0)),
        axis_from_geographic(az_deg), length_m=INS_LEN_M,
        width_m=INS_WIDTH_M, amplitude=amp, sigma=sigma, res_m=res_m,
        profile=PROFILE)
    x[r0:r1, c0:c1] += step - gaussian_filter(step, s, mode="nearest")


def track(shape, center, az_deg: float, res_m: float) -> np.ndarray:
    """Точки следа врезки: отрезок INS_LEN_M через центр под азимутом."""
    a = np.radians(az_deg)
    u = np.array([-np.cos(a), np.sin(a)])
    t = np.arange(-INS_LEN_M / 2, INS_LEN_M / 2 + res_m, res_m) / res_m
    p = np.round(center + u * t[:, None]).astype(int)
    ok = ((p[:, 0] >= 0) & (p[:, 0] < shape[0])
          & (p[:, 1] >= 0) & (p[:, 1] < shape[1]))
    return p[ok]


def found(lines, tracks, res_m: float) -> np.ndarray:
    """Для каждой врезки: нашлась ли линия, покрывающая её след."""
    from scipy.spatial import cKDTree

    if not lines:
        return np.zeros(len(tracks), dtype=bool)
    pts = np.concatenate([r["pts"] for r in lines])
    own = np.concatenate([np.full(len(r["pts"]), i)
                          for i, r in enumerate(lines)])
    tree = cKDTree(pts.astype(float))
    out = np.zeros(len(tracks), dtype=bool)
    for k, (tr, az) in enumerate(tracks):
        hits = tree.query_ball_point(tr.astype(float), r=MATCH_TOL_M / res_m)
        cnt: dict[int, int] = {}
        for h in hits:
            if not len(h):
                continue
            # ячейка следа засчитывается линии один раз, даже если её точек рядом много
            for j in np.unique(own[np.asarray(h, dtype=int)]):
                cnt[int(j)] = cnt.get(int(j), 0) + 1
        if not cnt:
            continue
        best = max(cnt, key=cnt.get)
        if cnt[best] / len(tr) < MATCH_FRAC:
            continue
        d = abs(((lines[best]["азимут"] - az) + 90.0) % 180.0 - 90.0)
        out[k] = d <= MATCH_TOL_DEG
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize
    from src.detection_limit import insert_fault, robust_sigma

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    valid = np.isfinite(img)
    sigma = robust_sigma(img[valid])
    rng = np.random.default_rng(SEED)
    grid = sites(img.shape, res_m, valid, rng)
    need = AZ_BINS * N_PER_BIN
    if len(grid) < need:
        raise SystemExit(f"мест под врезки {len(grid)}, нужно {need}")
    log(f"площадка {a.site}: sigma композита {sigma:.4f}, мест под врезки "
        f"{len(grid)}, берётся {need}")

    az = np.array([(b + 0.5) * 180.0 / AZ_BINS for b in range(AZ_BINS)])
    plan = [(az[i % AZ_BINS], grid[i]) for i in range(need)]

    def run_amp(amp: float, mode: str) -> np.ndarray:
        x = img.astype("float64").copy()
        tracks = []
        for a_deg, ctr in plan:
            k = (abs(np.sin(np.radians(a_deg - SUN_AZ)))
                 if mode == "высота" else 1.0)
            if amp > 0 and k > 0:
                insert_window(x, ctr, a_deg, amp * k, sigma, res_m)
            tracks.append((track(img.shape, ctr, a_deg, res_m), a_deg))
        xx = np.where(valid, x, np.nan).astype("float32")
        lines = trace(xx, valid, res_m, local=True, min_len_m=0)
        zz = local_norm(np.where(valid, ridge_response(np.where(
            valid, standardize(xx, valid), 0.0).astype("float32")), 0.0),
            valid, res_m)
        lines = extend_ends(lines, zz, valid, res_m)
        lines = [r for r in lines if r["длина_км"] * 1000 >= MIN_LEN_M]
        return found(lines, tracks, res_m)

    res = {}
    for mode in ("контраст", "высота"):
        for amp in AMPS:
            if amp == 0.0 and mode == "высота":
                res[(mode, amp)] = res[("контраст", 0.0)]
                continue
            hit = run_amp(amp, mode)
            res[(mode, amp)] = hit
            log(f"  режим «{mode}», амплитуда {amp:.2f} sigma: поймано "
                f"{hit.sum()} из {len(hit)} ({time.time() - t0:.0f} с)")

    bins = np.array([i % AZ_BINS for i in range(need)])
    log("\nдоля обнаружения по азимуту (строки — амплитуда в sigma):")
    for mode in ("контраст", "высота"):
        log(f"\n  режим «{mode}»"
            + ("  (одинаковый контраст у всех азимутов)" if mode == "контраст"
               else "  (одинаковая высота уступа, солнце 184°)"))
        head = "   амп  " + " ".join(f"{a_:>5.0f}" for a_ in az)
        log(head)
        for amp in AMPS:
            hit = res[(mode, amp)]
            row = [100 * hit[bins == b].mean() for b in range(AZ_BINS)]
            log(f"  {amp:4.2f}  " + " ".join(f"{v:5.0f}" for v in row))
    null = res[("контраст", 0.0)]
    log(f"\nнуль (амплитуда 0): ложных обнаружений "
        f"{100 * null.mean():.0f} % в среднем по азимутам")

    np.savez_compressed(M / f"openlin_azimuth_response_{a.site}.npz",
                        **{f"{m}_{amp}": res[(m, amp)] for m, amp in res},
                        az=az, bins=bins, sigma=sigma)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), facecolor="#fcfcfb",
                           sharey=True)
    for k, mode in enumerate(("контраст", "высота")):
        for amp in AMPS:
            hit = res[(mode, amp)]
            y = [100 * hit[bins == b].mean() for b in range(AZ_BINS)]
            ax[k].plot(az, y, marker="o", lw=1.8,
                       label=f"{amp:.2f} sigma" + (" (нуль)" if amp == 0 else ""))
        ax[k].axvline(SUN_AZ % 180.0, color="#b3261e", ls="--", lw=1.2)
        ax[k].text((SUN_AZ % 180.0) + 2, 95, "азимут солнца", color="#b3261e",
                   fontsize=9)
        ax[k].set_xlabel("азимут врезанного уступа, градусы")
        ax[k].set_title("одинаковый контраст" if mode == "контраст"
                        else "одинаковая высота уступа", loc="left")
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("доля обнаружения, %")
    ax[0].legend(fontsize=9)
    fig.suptitle("Какие направления видит трассировка: врезка известного "
                 f"уступа в реальный композит ({a.site})", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / f"передаточная_по_азимуту_{a.site}.png"
    fig.savefig(p, dpi=110, facecolor=fig.get_facecolor())
    log(f"сохранено: {p} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
