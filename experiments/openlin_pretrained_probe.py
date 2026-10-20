"""Проба: что находят на наших окнах ГОТОВЫЕ обученные детекторы линий и границ.

ОТКУДА ПОСТАНОВКА. Вопрос пользователя 09.10.2026: «может, снимать линии с
детектора каким-нибудь ИИ-методом». Размеченных линеаментов нет, поэтому самый
дешёвый вариант — сети, обученные на чужих данных, без дообучения:
  DexiNed — сеть границ, обучена на фотографиях с ручной разметкой контуров
            (набор BIPED);
  SOLD2   — сеть отрезков, обучена без разметки на синтетике и фотографиях
            зданий (каркасные сцены).
Обе учились на фотографиях предметов и построек, не на снежных тенях; вопрос
пробы — дают ли они на наших окнах осмысленный рисунок или мусор.

ЧТО ДЕЛАЕТСЯ. Несколько окон 5 x 5 км (те же, что в картинках по стадиям
детектора), каждое увеличивается втрое: тени у нас шириной в 1-2 ячейки, а
сети рассчитаны на объекты в несколько пикселей. Вход — сам снимок; для
сравнения рядом рабочий набор линий. Весь лист не считается.

Это не прогон и не замер: чисел нет, только картинки для глаза; в счётчик
не идёт, рабочий набор не трогается.

ЗАВИСИМОСТЬ. Нужна библиотека kornia (в requirements.txt её нет намеренно —
ставится в отдельную папку, чтобы не трогать рабочее окружение):
    pip install --no-deps --target <папка> kornia kornia_rs
    PYTHONPATH=<папка> python -m experiments.openlin_pretrained_probe
Веса обеих сетей скачиваются при первом запуске в кэш torch.

РЕЗУЛЬТАТ (09.10.2026, восемь окон, ``outputs/линеаменты_готовые_сети/``).
  SOLD2 НЕПРИГОДНА: 0 отрезков в шести окнах из восьми, 5 и 2 в остальных, и
    те — на резких границах наледи, а не на уступах. Сеть ищет прямые рёбра
    построек, на естественном рельефе молчит.
  DexiNed РАБОТАЕТ КАК ДЕТЕКТОР КОНТРАСТНЫХ ГРАНИЦ: на контрастном окне (7)
    обводит наледи, уступы и борта долины чисто и непрерывно, без дробления на
    обрывки; на слабоконтрастном окне (1, там пропущенный линеамент из мелких
    кусков) выделяет только ту же самую яркую цепочку и ничего больше — слабые
    параллельные тени, которые наш детектор видит, для неё не существуют.
  ВЫВОД: готовые сети не находят того, чего не находит наш детектор, и теряют
    слабые линии. Заменой не годятся. Единственное полезное свойство — у
    DexiNed граница непрерывная там, где наша линия рвётся; это довод искать
    непрерывность (сшивка, трассировка), а не довод брать сеть.
  Просмотрены глазом модели два окна из восьми (1 и 7); оценка по остальным —
  за пользователем.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import COMP, OUT

OUT_DIR = OUT.parent / "линеаменты_готовые_сети"
WIN_KM = 5.0
UPSCALE = 3                  # тень в 1-2 ячейки -> 3-6 пикселей на входе сети
THREADS = 2                  # на машине идёт другой прогон — не занимать процессор


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from kornia.feature import SOLD2
        from kornia.models.dexined import DexiNed
    except ImportError as exc:
        raise SystemExit(f"нужны torch и kornia (см. шапку файла): {exc}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.openlin_detector_view import DEFAULT_WINDOWS
    from experiments.openlin_shade import _stretch
    from experiments.openlin_zone_lines_v3 import work_npz

    torch.set_num_threads(THREADS)
    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    d = np.load(work_npz(a.site))
    A, C = d["A_all"], d["C_all"]
    shape = C.shape
    img = img[:shape[0], :shape[1]]
    base = np.nan_to_num(_stretch(np.where(np.isfinite(img), img, np.nan)), nan=0.5)

    dex = DexiNed(pretrained=True).eval()
    sold = SOLD2(pretrained=True).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    print(f"сети загружены ({time.time() - t0:.0f} с)", flush=True)

    n_cell = int(WIN_KM * 1000 / res_m)
    half = n_cell // 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    px = 1000.0 / res_m
    for i, (r, c) in enumerate(DEFAULT_WINDOWS, 1):
        r = min(max(r, half), shape[0] - half)
        c = min(max(c, half), shape[1] - half)
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        g = torch.from_numpy(base[sl].astype("float32"))[None, None]
        g = F.interpolate(g, scale_factor=UPSCALE, mode="bilinear", align_corners=False)
        with torch.no_grad():
            # вход DexiNed: три канала, вычитание среднего и масштаб 0-255
            edges = torch.sigmoid(dex((g.repeat(1, 3, 1, 1) - mean) * 255.0))
            segs = sold(g)["line_segments"][0].numpy() / UPSCALE     # (N, 2, 2): строка, столбец
        edges = F.interpolate(edges, size=(n_cell, n_cell), mode="area")[0, 0].numpy()

        fig, ax = plt.subplots(2, 2, figsize=(19, 19.6), facecolor="#fcfcfb")
        for axx in ax.ravel():
            axx.imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[0, 0].set_title("1. снимок — смотрите сами", loc="left", fontsize=15)
        ov = np.zeros((n_cell, n_cell, 4), "float32")
        ov[A[sl] & ~C[sl]] = (0.95, 0.60, 0.0, 0.95)
        ov[C[sl]] = (0.85, 0.05, 0.05, 0.95)
        ax[0, 1].imshow(ov)
        ax[0, 1].set_title("2. наш рабочий набор: красные подтверждены, жёлтые — нет",
                           loc="left", fontsize=15)
        ax[1, 0].imshow(edges, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax[1, 0].set_title("3. готовая сеть границ (DexiNed): светлее — увереннее",
                           loc="left", fontsize=15)
        for s in segs:
            ax[1, 1].plot(s[:, 1], s[:, 0], color="#19c3f0", lw=1.6)
        ax[1, 1].set_title(f"4. готовая сеть отрезков (SOLD2): {len(segs)} отрезков",
                           loc="left", fontsize=15)
        for axx in ax.ravel():
            axx.plot([10, 10 + px], [n_cell - 10] * 2, color="#eee", lw=3)
            axx.text(10, n_cell - 16, "1 км", color="#eee", fontsize=12)
            axx.set_xlim(0, n_cell - 1); axx.set_ylim(n_cell - 1, 0)
            axx.set_xticks([]); axx.set_yticks([])
        fig.suptitle(f"Окно {i} · {WIN_KM:.0f} × {WIN_KM:.0f} км · центр ({r}, {c}) · "
                     f"север сверху", x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = OUT_DIR / f"{i:02d}_окно_{r}_{c}.png"
        fig.savefig(p, dpi=80, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  {p.name}: отрезков {len(segs)} ({time.time() - t0:.0f} с)", flush=True)
    print(f"готово: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
