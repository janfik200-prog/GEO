"""Замер: что детектор видит, но в линию не превращает.

ПОВОД. Просмотр 09.10.2026 наложения всех линий детектора на лист: «это не все
линеаменты». При этом карта отклика, по оценке 28.09.2026, «видит всё». Значит,
потеря — на шаге «отклик -> линия»: утончение до гребня и два порога
(``Q_HIGH`` = 97 % — отсюда линия начинается, ``Q_LOW`` = 90 % — до сюда она
продолжается). Оба порога относительные и заданы без замера против шума.

ЧТО МЕРИТСЯ. Все гребни отклика (после утончения) делятся на классы:
  «линия»      — выше 90 % и связан с ячейкой выше 97 %: это и есть выход
                 детектора;
  «без затравки» — выше 90 %, но ни одной ячейки выше 97 % в цепочке нет:
                 отброшен верхним порогом;
  «слабый»     — между 80 и 90 %;
  «очень слабый» — между 70 и 80 %.
Для каждого класса — длина на снимке и на карте-двойнике (том же шуме без
линеаментов, со своими порогами). Отношение двойник / снимок показывает, чего
в классе больше: если шума столько же, сколько на снимке, класс брать нельзя;
если шума мало, а класс отброшен — там лежат потерянные линеаменты.

Картинки — те же окна, что в :mod:`openlin_detector_view`, и весь лист: по ним
видно, в какой класс попадает линеамент, который глаз видит, а линии нет.

Это замер: пороги не меняются, рабочий набор не трогается, в счётчик не идёт.

РЕЗУЛЬТАТ (09.10.2026). Длина гребней, км, снимок / двойник:
  линия 9296 / 10416;  без затравки 5557 / 5221;  слабый 12984 / 13544;
  очень слабый 11814 / 12487.
  1. ПО ДЛИНЕ ГРЕБНЕЙ СНИМОК ОТ ШУМА НЕ ОТЛИЧАЕТСЯ НИ В ОДНОМ КЛАССЕ, включая
     сам выход детектора. Это следствие относительных порогов: 3 и 10 % самых
     сильных ячеек найдутся на любой картинке с таким спектром. Снимок от шума
     отличает не сила отклика, а СВЯЗНОСТЬ в длинные цепочки: линий от 500 м
     у снимка втрое больше; цепочек «без затравки» от 200 м — 3685 (1044 км)
     против 1760 (416 км).
  2. ПОНИЖАТЬ ПОРОГИ БЕССМЫСЛЕННО: каждый следующий класс добавляет столько же
     шума, сколько сигнала. На окнах жёлтые, оранжевые и красные гребни лежат
     равномерной пылью без рисунка (окно 1).
  3. В классе «без затравки» есть избыток над шумом — около 630 км цепочек от
     200 м, — но какие из них настоящие, по одной цепочке не определить.
  Просмотрено одно окно из восьми. Какие именно линеаменты пользователь видит
  пропущенными и в какой класс они попадают — не установлено, нужны примеры.

Запуск из корня репозитория:
    python -m experiments.openlin_ridge_classes --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, OUT

OUT_DIR = OUT.parent / "линеаменты_классы_гребней"
WIN_KM = 5.0
#: уровни ниже действующего нижнего порога — шагами по 10 % отклика
Q_WEAK = (0.80, 0.70)
COLORS = {"линия": (0.15, 0.75, 0.95), "без затравки": (0.98, 0.90, 0.10),
          "слабый": (1.00, 0.55, 0.05), "очень слабый": (0.90, 0.15, 0.15)}


def classes(img, valid, res_m: float) -> tuple[dict, np.ndarray]:
    """Гребни отклика по классам и сам нормированный отклик."""
    from scipy.ndimage import label

    from experiments.openlin_iso import (Q_HIGH, Q_LOW, hysteresis8, local_norm,
                                         ridge_response)
    from experiments.openlin_lines import _ridge_nms
    from experiments.openlin_shadow import standardize

    x = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    z = local_norm(np.where(valid, ridge_response(x), 0.0), valid, res_m)
    v = z[valid]
    hi, lo = np.quantile(v, Q_HIGH), np.quantile(v, Q_LOW)
    w1, w2 = (np.quantile(v, q) for q in Q_WEAK)
    _s, ridge = _ridge_nms(z, valid, 1.0)
    ridge &= valid
    kept = hysteresis8(np.where(ridge, z, float(v.min())), lo, hi)
    out = {"линия": kept,
           "без затравки": ridge & (z >= lo) & ~kept,
           "слабый": ridge & (z >= w1) & (z < lo),
           "очень слабый": ridge & (z >= w2) & (z < w1)}
    # длина цепочек «без затравки»: одиночные ячейки и длинные гребни — разное
    lab, n = label(out["без затравки"], structure=np.ones((3, 3), int))
    out["_orphan_sizes"] = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    return out, z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments import openlin_zone_lines_v3 as v3
    from experiments.openlin_detector_view import DEFAULT_WINDOWS
    from experiments.openlin_iso import MIN_LEN_M as PIECE_MIN_M
    from experiments.openlin_kora import surrogate
    from experiments.openlin_shade import _stretch

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = v3.build_halves(a.site, lambda *_: None)
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])
    real, z = classes(opt, valid, res_m)
    log(f"снимок готов ({time.time() - t0:.0f} с)")
    stack = np.stack([np.where(valid, x, np.nan) for x in (opt, halves["А"], halves["Б"])])
    twin, _zt = classes(surrogate(np.nan_to_num(stack), valid,
                                  np.random.default_rng(900))[0], valid, res_m)
    log(f"двойник готов ({time.time() - t0:.0f} с)")

    km = lambda m: m.sum() * res_m / 1000.0
    log("\nгребни отклика по классам: длина на снимке / на двойнике, км, и "
        "отношение двойник : снимок (доля шума)")
    for name in COLORS:
        r, f = km(real[name]), km(twin[name])
        log(f"  {name:13s}: {r:7.0f} / {f:7.0f}   {100 * f / max(r, 1e-9):5.1f} %")
    k_min = int(PIECE_MIN_M / res_m)
    for nm, d in (("снимок ", real), ("двойник", twin)):
        s = d["_orphan_sizes"]
        long_ = s[s >= k_min]
        log(f"  «без затравки», {nm}: цепочек {len(s)}, из них от {PIECE_MIN_M:.0f} м "
            f"{len(long_)} ({long_.sum() * res_m / 1000:.0f} км); квантили длины, м "
            f"(50/90/99 %): " + " / ".join(f"{q * res_m:.0f}"
                                           for q in np.quantile(s, [.5, .9, .99])))

    base = np.nan_to_num(_stretch(np.where(valid, opt, np.nan)), nan=1.0)
    gray = np.repeat(base[..., None], 3, axis=2).astype("float32")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from scipy.ndimage import binary_dilation

    sheet = gray.copy()
    for name in ("очень слабый", "слабый", "без затравки", "линия"):
        sheet[binary_dilation(real[name], np.ones((3, 3), bool))] = COLORS[name]
    try:
        plt.imsave(OUT_DIR / "весь_лист_классы.png", np.clip(sheet, 0, 1))
        log("  весь_лист_классы.png")
    except OSError as exc:
        log(f"  весь_лист_классы.png: НЕ записан ({exc.strerror})")

    n_cell = int(WIN_KM * 1000 / res_m)
    half = n_cell // 2
    zmax = float(np.quantile(z[valid], 0.99))
    px = 1000.0 / res_m
    legend = "голубой — линия детектора; жёлтый — выше 90 %, но без затравки 97 %;\n" \
             "оранжевый — 80-90 %; красный — 70-80 %"
    for i, (r, c) in enumerate(DEFAULT_WINDOWS, 1):
        r = min(max(r, half), valid.shape[0] - half)
        c = min(max(c, half), valid.shape[1] - half)
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        fig, ax = plt.subplots(2, 2, figsize=(19, 19.8), facecolor="#fcfcfb")
        for axx in ax.ravel():
            axx.imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[0, 0].set_title("1. снимок — смотрите сами", loc="left", fontsize=15)
        ax[0, 1].imshow(np.clip(z[sl], 0, None), cmap="magma", vmin=0, vmax=zmax,
                        interpolation="nearest")
        ax[0, 1].set_title("2. отклик детектора (светлее — сильнее)", loc="left",
                           fontsize=15)
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[real["линия"][sl]] = (*COLORS["линия"], 1.0)
        ax[1, 0].imshow(ov)
        ax[1, 0].set_title("3. что детектор отдаёт как линии", loc="left", fontsize=15)
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        for name in ("очень слабый", "слабый", "без затравки", "линия"):
            ov[real[name][sl]] = (*COLORS[name], 1.0)
        ax[1, 1].imshow(ov)
        ax[1, 1].set_title("4. все гребни отклика по классам\n" + legend, loc="left",
                           fontsize=13)
        for axx in ax.ravel():
            axx.plot([10, 10 + px], [n_cell - 10] * 2, color="#eee", lw=3)
            axx.text(10, n_cell - 16, "1 км", color="#eee", fontsize=12)
            axx.set_xticks([]); axx.set_yticks([])
        fig.suptitle(f"Окно {i} · {WIN_KM:.0f} × {WIN_KM:.0f} км · центр ({r}, {c}) · "
                     f"север сверху", x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUT_DIR / f"{i:02d}_окно_{r}_{c}.png", dpi=80,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"готово: {OUT_DIR} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
