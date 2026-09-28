"""Гибрид CNN + SVM: свёрточные признаки Невады, классификатор — по Анабару.

ИДЕЯ. Чистый перенос U-Net Невада -> Анабар дал lift 1.36 при потолке 12.5:
голова, обученная под невадские яркости, на другой территории решает неверно.
Но СВЁРТОЧНЫЕ ПРИЗНАКИ переносятся лучше голов — они описывают локальный рисунок
(перепад, вытянутость, текстуру), а не абсолютный уровень сигнала. Поэтому здесь
свёртки берутся с Невады как готовый экстрактор и замораживаются, а решающее
правило учится заново на самой целевой территории.

ПОЧЕМУ SVM, А НЕ ЕЩЁ ОДНА СЕТЬ. Обучающих пикселей на целевой площадке мало и
они сильно скоррелированы пространственно; SVM с ядром — классический выбор для
режима «много признаков, мало независимых примеров», у него один параметр
регуляризации и нет риска, что результат окажется розыгрышем инициализации.
Дообучается он за секунды, поэтому такую адаптацию можно повторять под каждую
новую территорию, не переобучая свёртки.

ЧЕСТНОСТЬ. Разбиение целевой площадки — те же пространственные полосы с эрозией
на радиус патча, что в :mod:`experiments.openlin_unet`: SVM учится на одних
полосах, меряется на других, и признаки для валидационных полос считаются той же
замороженной сетью. Планки те же: Кэнни -> Хаф, случайная маска и чистый перенос
U-Net без SVM — последний показывает, что именно добавил гибрид.

Запуск из корня репозитория (сначала нужен обученный U-Net Невады):
    python -m experiments.openlin_unet --site nevada --epochs 120
    python -m experiments.openlin_cnnsvm --features nevada --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_unet import (PATCH, STRIDE, TOLERANCES_M, SEED,
                                      build_stack, canny_hough_mask, lift_at_top,
                                      lift_of_mask, load_site, make_unet,
                                      stripe_split, patch_origins)

#: Сколько пикселей идёт в обучение SVM. Не «все»: соседние пиксели почти
#: одинаковы, и миллион строк даёт ту же информацию, что десятки тысяч, но
#: обучается на порядки дольше (ядровой SVM растёт как квадрат выборки).
N_TRAIN_PIXELS = 30_000
#: Отношение фон/цель в обучающей выборке. Держим умеренным: при естественных
#: 1:12 SVM выгоднее всегда отвечать «фон».
BG_RATIO = 3.0
#: Число компонент приближения ядра. Nystroem + линейный SVM вместо честного
#: SVC(RBF): на десятках тысяч строк это тот же результат за минуты вместо часов.
N_KERNEL_COMPONENTS = 800


def cnn_features(model, x: np.ndarray, shape: tuple[int, int], log) -> np.ndarray:
    """Попиксельные признаки замороженной сети: выход последнего блока декодера.

    Берётся именно декодер, а не бутылочное горлышко: у горлышка разрешение в
    четыре раза грубее, и линия шириной в полторы ячейки в нём уже неразличима.
    """
    import torch

    feats: list[np.ndarray] = []

    def hook(_m, _inp, out):
        feats.append(out.detach().numpy())

    handle = model.d1.register_forward_hook(hook)
    n_ch = None
    acc = cnt = None
    model.eval()
    with torch.no_grad():
        for r in range(0, shape[0] - PATCH + 1, STRIDE):
            for c in range(0, shape[1] - PATCH + 1, STRIDE):
                feats.clear()
                model(torch.from_numpy(x[None, :, r:r + PATCH, c:c + PATCH]))
                f = feats[0][0]                       # (C, PATCH, PATCH)
                if acc is None:
                    n_ch = f.shape[0]
                    acc = np.zeros((n_ch, *shape), dtype="float32")
                    cnt = np.zeros(shape, dtype="float32")
                acc[:, r:r + PATCH, c:c + PATCH] += f
                cnt[r:r + PATCH, c:c + PATCH] += 1
    handle.remove()
    out = acc / np.maximum(cnt, 1)[None]
    log(f"  признаков CNN: {n_ch} каналов, покрыто "
        f"{100 * np.mean(cnt > 0):.1f}% площадки")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="nevada",
                    help="чей обученный U-Net берётся как экстрактор признаков")
    ap.add_argument("--site", default="anabar", help="целевая площадка")
    args = ap.parse_args()

    import torch
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    np.random.seed(SEED)
    t0 = time.time()

    def log(msg=""):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    weights = Path("outputs/metrics") / f"openlin_unet_{args.features}.pt"
    if not weights.exists():
        raise SystemExit(f"нет {weights} — сначала обучите U-Net на «{args.features}»")

    z = load_site(args.site)
    x, y, valid, channels = build_stack(z)
    dist = z["dist"]
    res_m = float(z["res_m"])
    log(f"целевая площадка «{args.site}»: {y.shape[0]}x{y.shape[1]}, "
        f"каналы {channels}; признаки CNN с «{args.features}»")

    model = make_unet(x.shape[0])
    model.load_state_dict(torch.load(weights))
    feat = cnn_features(model, x, y.shape, log)
    # Сырые каналы добавляются к свёрточным: они несут абсолютный уровень, а
    # свёртки после нормировки его частично теряют.
    feat = np.concatenate([feat, x], axis=0)
    log(f"  итого признаков на пиксель: {feat.shape[0]}")

    val_mask = stripe_split(y.shape)
    tr_area = (~val_mask) & valid
    va_area = val_mask & valid
    log(f"полосы: обучение {int(tr_area.sum())} ячеек, "
        f"сверка {int(va_area.sum())} ячеек")

    # Обучающая выборка пикселей: цель = ячейки у размеченной линии.
    target = (dist <= 250.0)
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero((target & tr_area).ravel())
    neg = np.flatnonzero((~target & tr_area).ravel())
    n_pos = min(len(pos), int(N_TRAIN_PIXELS / (1 + BG_RATIO)))
    n_neg = min(len(neg), int(n_pos * BG_RATIO))
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    X = feat.reshape(feat.shape[0], -1).T[idx]
    Y = target.ravel()[idx].astype(int)
    log(f"обучающих пикселей: {len(idx)} (цель {n_pos}, фон {n_neg})")

    svm = make_pipeline(
        StandardScaler(),
        Nystroem(n_components=N_KERNEL_COMPONENTS, random_state=SEED),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                  random_state=SEED))
    svm.fit(X, Y)
    log("SVM обучен")

    flat = feat.reshape(feat.shape[0], -1).T
    score = np.full(y.shape, -np.inf, dtype="float32")
    va_idx = np.flatnonzero(va_area.ravel())
    # Считаем блоками: матрица признаков всей площадки в память ядра не влезает.
    for k in range(0, len(va_idx), 200_000):
        part = va_idx[k:k + 200_000]
        score.ravel()[part] = svm.decision_function(flat[part])
    log("скор посчитан")

    # Планка: чистый перенос U-Net без SVM, с теми же весами.
    unet_pred = Path("outputs/metrics") / f"openlin_unet_{args.features}_to_{args.site}_pred.npz"
    prob_unet = np.load(unet_pred)["prob"] if unet_pred.exists() else None

    canny_full = canny_hough_mask(z["gray"], valid)
    canny = canny_full & va_area
    frac_canny = float(canny.sum() / va_area.sum())

    rows = []
    for tol in TOLERANCES_M:
        base_rate = float(((dist <= tol) & va_area).sum() / va_area.sum())
        for frac in (0.02, 0.05, 0.10):
            rows.append({"метод": f"CNN+SVM top{int(100 * frac)}%", "допуск_м": tol,
                         "доля_цели": base_rate,
                         **lift_at_top(score, dist, va_area, frac, tol)})
        rows.append({"метод": "CNN+SVM при площади Кэнни", "допуск_м": tol,
                     "доля_цели": base_rate,
                     **lift_at_top(score, dist, va_area, frac_canny, tol)})
        if prob_unet is not None:
            rows.append({"метод": "U-Net перенос, площадь Кэнни", "допуск_м": tol,
                         "доля_цели": base_rate,
                         **lift_at_top(prob_unet, dist, va_area, frac_canny, tol)})
        rows.append({"метод": "Кэнни→Хаф", "допуск_м": tol, "доля_цели": base_rate,
                     **lift_of_mask(canny, dist, va_area, tol)})
        rnd = np.zeros_like(va_area)
        pick = rng.choice(va_idx, size=min(int(frac_canny * va_area.sum()),
                                           len(va_idx)), replace=False)
        rnd.ravel()[pick] = True
        rows.append({"метод": "случайная маска", "допуск_м": tol,
                     "доля_цели": base_rate,
                     **lift_of_mask(rnd, dist, va_area, tol)})

    df = pd.DataFrame(rows)
    log("")
    log(f"LIFT НА «{args.site}» (полосы сверки, обучения на них не было):")
    log(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = Path("outputs/metrics")
    df.insert(0, "площадка", args.site)
    df.insert(1, "признаки", args.features)
    df.to_csv(out / f"openlin_cnnsvm_{args.features}_to_{args.site}.csv",
              index=False, encoding="utf-8-sig")
    np.savez_compressed(
        out / f"openlin_cnnsvm_{args.features}_to_{args.site}_pred.npz",
        score=score, va_area=va_area)
    log(f"сохранено: outputs/metrics/openlin_cnnsvm_{args.features}_to_{args.site}.csv")


if __name__ == "__main__":
    main()
