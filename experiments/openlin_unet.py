"""Обучение U-Net воспроизводить разметку разломов по снимку.

ЗАЧЕМ ИМЕННО НЕВАДА. Шаг 0 на открытых площадках
(:mod:`experiments.openlin_probe`) измерил контраст разломов в композите: Невада
+0.96 sigma при пороге обнаружения 0.60, Abitibi +0.09 (и -0.06 после исключения
воды, :mod:`experiments.openlin_water`), Анабар 0.086-0.194. Обучать сеть имеет
смысл там, где сигнал заведомо выше порога: если U-Net не воспроизводит разметку
даже на Неваде, дело в модели или в постановке, а не во входных данных. Это
отладочная площадка, а не аналог Анабара.

ЧТО ИМЕННО ПРЕДСКАЗЫВАЕТСЯ. Не «разлом» как геологический объект, а ЛИНИЯ,
которую в этом месте провёл составитель карты. Цель — автоматизация ручной
операции, поэтому эталон здесь равен разметке, а метрика — согласию с ней.
Утверждать по результату что-либо о реальной тектонике нельзя.

ЧЕСТНОСТЬ ВАЛИДАЦИИ. Патчи соседних ячеек перекрываются, поэтому случайный
holdout лежит внутри обучающих патчей и меряет память, а не обобщение — тот же
урок, что зафиксирован в :mod:`src.mae_model`. Здесь площадка делится на
БЛОКИ-полосы, валидационные блоки эродированы на радиус патча, и ни один
обучающий патч не пересекает валидационный блок.

ПЛАНКИ ОБЯЗАТЕЛЬНЫ. Сеть сравнивается не с нулём, а с (1) детектором Кэнни ->
Хаф — тем самым конвейером, которым ветка линеаментов работала до сих пор, и
(2) случайной маской той же площади. Выигрыш у случайной маски ничего не
доказывает; доказывает выигрыш у Кэнни.

Запуск из корня репозитория:
    python -m experiments.openlin_unet --site nevada
    python -m experiments.openlin_unet --site nevada --epochs 60
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Параметры (все — с обоснованием, как принято в src/config.py)
# --------------------------------------------------------------------------

#: Полуширина растеризованного таргета. 150 м при шаге 100 м — полторы ячейки:
#: тоньше делать нельзя (линия исчезает при пулинге), толще — таргет перестаёт
#: быть линией и превращается в площадь.
TARGET_HALFWIDTH_M = 150.0

#: Сторона патча в ячейках. 128 x 100 м = 12.8 км: заметно длиннее типичного
#: отрезка разметки (медиана 0.64 км), то есть сеть видит линию целиком вместе с
#: её продолжением, а не фрагмент.
PATCH = 128
#: Шаг нарезки патчей. Половина патча — стандартное перекрытие для сегментации.
STRIDE = 64

#: Доля площадки под валидацию и число полос. Полосы, а не случайные блоки:
#: разломы Basin & Range вытянуты, и случайные квадраты рвали бы одни и те же
#: структуры между train и val.
VAL_FRACTION = 0.3
N_STRIPES = 5

#: Веса Tversky. beta > alpha штрафует пропуски сильнее ложных срабатываний:
#: таргет занимает единицы процентов площади, и при симметричном лоссе сеть
#: выигрывает, предсказывая пустоту.
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7

EPOCHS = 40
BATCH = 8
LR = 3e-4
SEED = 42
#: Эпох без улучшения val до остановки. Веса берутся лучшие по val в любом
#: случае, patience экономит время, а не улучшает результат.
PATIENCE = 15

#: Допуски сверки с разметкой, м. Те же величины, что в задаче №8
#: (``config.LIN_SAT_MATCH_M``), чтобы числа читались в одной шкале.
TOLERANCES_M = (250.0, 500.0, 1000.0)


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------

def load_site(site: str) -> dict:
    path = Path("outputs/metrics") / f"openlin_{site}.npz"
    if not path.exists():
        raise SystemExit(f"нет {path} — сначала: python -m experiments.openlin_probe "
                         f"--site {site}")
    return dict(np.load(path))


#: Порядок каналов входа. Радарные (``vv``/``vh``) подключаются, только если они
#: есть в файле площадки — иначе модель молча меняла бы размерность входа между
#: обучением и переносом.
CHANNELS = ("gray", "ndvi", "vv", "vh",
            "coh_winter_vv_COH12", "coh_summer_vv_COH12",
            "coh_summer_vv_COH24", "coh_summer_vv_COH48",
            "coh_summer_vv_rho", "coh_summer_vv_tau")


def build_stack(z: dict, channels: tuple[str, ...] | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Каналы входа, таргет, маска валидных ячеек и фактический список каналов.

    Вход нормируется робастно (медиана и IQR), а не min-max: у композита есть
    остаточные облачные выбросы, и min-max отдал бы им всю шкалу. Радар уже в
    децибелах, то есть в логарифмической шкале, и та же нормировка ему подходит.
    """
    gray, dist = z["gray"], z["dist"]
    valid = np.isfinite(gray)

    def norm(a: np.ndarray) -> np.ndarray:
        v = a[np.isfinite(a)]
        med = np.median(v)
        q1, q3 = np.percentile(v, [25, 75])
        scale = (q3 - q1) or 1.0
        out = (a - med) / scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    use = tuple(c for c in (channels or CHANNELS) if c in z)
    x = np.stack([norm(z[c]) for c in use])
    y = (dist <= TARGET_HALFWIDTH_M).astype("float32")
    return x, y, valid, use


def stripe_split(shape: tuple[int, int]) -> np.ndarray:
    """Маска валидационных полос: ``True`` — валидация."""
    h, _ = shape
    edges = np.linspace(0, h, N_STRIPES + 1).astype(int)
    rng = np.random.default_rng(SEED)
    n_val = max(1, int(round(VAL_FRACTION * N_STRIPES)))
    val_idx = set(rng.choice(N_STRIPES, size=n_val, replace=False).tolist())
    mask = np.zeros(shape, dtype=bool)
    for i in range(N_STRIPES):
        if i in val_idx:
            mask[edges[i]:edges[i + 1]] = True
    return mask


def patch_origins(shape: tuple[int, int], val_mask: np.ndarray,
                  want_val: bool) -> list[tuple[int, int]]:
    """Левые верхние углы патчей, целиком лежащих в своей части разбиения.

    Патч берётся ТОЛЬКО если он не пересекает чужую часть — это и есть эрозия
    границы на радиус патча, без которой обучающий патч заглядывал бы в
    валидационную полосу.
    """
    h, w = shape
    out = []
    for r in range(0, h - PATCH + 1, STRIDE):
        for c in range(0, w - PATCH + 1, STRIDE):
            block = val_mask[r:r + PATCH, c:c + PATCH]
            if want_val and block.all():
                out.append((r, c))
            elif not want_val and not block.any():
                out.append((r, c))
    return out


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------

def make_unet(in_ch: int):
    """Маленький U-Net на 3 уровня: обучение идёт на CPU, глубина не бесплатна."""
    import torch
    from torch import nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(in_ch, 16), block(16, 32), block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.d2 = block(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.d1 = block(32, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)

    return UNet()


def tversky_loss(logit, target):
    """Tversky: асимметричный Dice, штрафует пропуски сильнее ложных тревог."""
    import torch

    p = torch.sigmoid(logit)
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    return 1 - (tp + 1.0) / (tp + TVERSKY_ALPHA * fp + TVERSKY_BETA * fn + 1.0)


# --------------------------------------------------------------------------
# Метрики: согласие с разметкой в буфере
# --------------------------------------------------------------------------

def lift_of_mask(mask: np.ndarray, dist: np.ndarray, area_mask: np.ndarray,
                 tol_m: float) -> dict[str, float]:
    """``lift`` маски относительно разметки: во сколько раз чаще под ней разлом.

    ПОЧЕМУ НЕ completeness/correctness. Первый прогон показал, что при плотности
    разметки Невады буфер 250 м покрывает 25% площадки, и случайная маска равной
    площади набирает completeness 0.98 — метрика упирается в потолок независимо
    от метода. ``lift@area`` этим не страдает: он делит долю пойманного на долю
    занятой площади, поэтому случайная маска даёт 1.0 по построению при любой
    плотности. Это ровно та метрика, которой в проекте меряется всё остальное
    (см. ``src/assessment.py`` и ``config.VER_PRIMARY``).
    """
    pred = mask & area_mask
    target = (dist <= tol_m) & area_mask
    if pred.sum() == 0 or target.sum() == 0:
        return {"lift": float("nan"), "площадь": 0.0, "точность": float("nan")}
    precision = float(target[pred].mean())
    base = float(target[area_mask].mean())
    return {"lift": precision / base if base else float("nan"),
            "площадь": float(pred.sum() / area_mask.sum()),
            "точность": precision}


def lift_at_top(prob: np.ndarray, dist: np.ndarray, area_mask: np.ndarray,
                fraction: float, tol_m: float) -> dict[str, float]:
    """То же для непрерывного скора: верхние ``fraction`` площади по вероятности."""
    thr = float(np.quantile(prob[area_mask], 1 - fraction))
    return lift_of_mask((prob >= thr) & area_mask, dist, area_mask, tol_m)


def canny_hough_mask(gray: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Планка: тот же конвейер Кэнни -> Хаф, что в ветке линеаментов проекта.

    Масштабируется до той же площади, что и предсказание сети, иначе сравнение
    мерило бы порог, а не метод.
    """
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    g = np.nan_to_num(gray, nan=float(np.nanmedian(gray)))
    g = (g - np.nanmin(g)) / (np.ptp(g[np.isfinite(g)]) or 1.0)
    edges = canny(g, sigma=2.0) & valid
    lines = probabilistic_hough_line(edges, threshold=10, line_length=10,
                                     line_gap=3, rng=SEED)
    mask = np.zeros_like(valid, dtype=bool)
    from skimage.draw import line as draw_line
    for (c0, r0), (c1, r1) in lines:
        rr, cc = draw_line(r0, c0, r1, c1)
        ok = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
        mask[rr[ok], cc[ok]] = True
    return mask


# --------------------------------------------------------------------------
# Прогон
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="nevada",
                    help="площадка обучения; через запятую — обучение сразу на "
                         "нескольких (признаки перестают быть заточены под одну "
                         "местность). Каналы берутся общие для всех.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--apply-to", default=None,
                    help="применить обученную модель к другой площадке (перенос). "
                         "Обучение при этом идёт на --site, а метрики считаются "
                         "на указанной площадке ЦЕЛИКОМ: она в обучении не "
                         "участвовала, поэтому делить её на полосы незачем.")
    args = ap.parse_args()

    import torch
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    def log(msg=""):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    names = [s.strip() for s in args.site.split(",") if s.strip()]
    raw = [load_site(n) for n in names]
    # Общие каналы: если одна площадка собрана без радара, обучаться на четырёх
    # каналах нельзя — модель должна видеть одинаковый вход везде.
    common = tuple(c for c in CHANNELS if all(c in z for z in raw))
    stacks = [build_stack(z, channels=common) for z in raw]
    channels = stacks[0][3]
    log(f"площадок обучения: {len(names)} ({', '.join(names)}); каналы {channels}")

    tr_org: list[tuple[int, int, int]] = []
    va_org: list[tuple[int, int, int]] = []
    for i, (name, (xs, ys, vs, _)) in enumerate(zip(names, stacks)):
        vm = stripe_split(ys.shape)
        tr = [(i, r, c) for r, c in patch_origins(ys.shape, vm, want_val=False)]
        va = [(i, r, c) for r, c in patch_origins(ys.shape, vm, want_val=True)]
        tr_org += tr
        va_org += va
        log(f"  «{name}»: {ys.shape[0]}x{ys.shape[1]}, таргет "
            f"{100 * ys[vs].mean():.2f}% площади; патчей {len(tr)}/{len(va)}")

    # Для одиночной площадки дальше нужны её же массивы (оценка идёт на ней).
    x, y, valid, _ = stacks[0]
    z = raw[0]
    res_m = float(z["res_m"])
    dist = z["dist"]
    val_mask = stripe_split(y.shape)
    log(f"итого патчей: обучение {len(tr_org)}, валидация {len(va_org)} "
        f"(полосы, эрозия на радиус патча)")
    if not tr_org or not va_org:
        raise SystemExit("не хватило патчей — уменьшите PATCH или N_STRIPES")

    def batch_of(origins, idx, augment=False):
        """Батч патчей; при ``augment`` — случайный элемент группы симметрий квадрата.

        Аугментация именно диэдральная (8 вариантов: отражения и повороты на 90°),
        а не произвольный поворот: произвольный требует интерполяции, которая
        размывает линию толщиной в полторы ячейки — то есть портит ровно тот
        объект, который сеть учится находить. Обучающих патчей всего несколько
        десятков, и без аугментации сеть переобучается за 40 эпох (измерено:
        train 0.37 при val 0.76).
        """
        picks = [origins[i] for i in idx]
        xs = np.stack([stacks[s][0][:, r:r + PATCH, c:c + PATCH] for s, r, c in picks])
        ys = np.stack([stacks[s][1][None, r:r + PATCH, c:c + PATCH] for s, r, c in picks])
        if augment:
            k = int(rng.integers(4))
            xs, ys = np.rot90(xs, k, axes=(2, 3)), np.rot90(ys, k, axes=(2, 3))
            if rng.random() < 0.5:
                xs, ys = xs[:, :, ::-1], ys[:, :, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(xs)),
                torch.from_numpy(np.ascontiguousarray(ys)))

    model = make_unet(x.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)

    best, best_state, best_ep, since = float("inf"), None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(tr_org))
        tot = 0.0
        for k in range(0, len(order), BATCH):
            xb, yb = batch_of(tr_org, order[k:k + BATCH], augment=True)
            opt.zero_grad()
            loss = tversky_loss(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(order[k:k + BATCH])
        model.eval()
        with torch.no_grad():
            vtot = 0.0
            for k in range(0, len(va_org), BATCH):
                xb, yb = batch_of(va_org, list(range(k, min(k + BATCH, len(va_org)))))
                vtot += float(tversky_loss(model(xb), yb)) * xb.shape[0]
        vloss = vtot / len(va_org)
        if vloss < best:
            best, best_state, best_ep, since = vloss, {
                k: v.clone() for k, v in model.state_dict().items()}, ep, 0
        else:
            since += 1
        if ep % 5 == 0 or ep == 1:
            log(f"  эпоха {ep:3d}: train {tot / len(tr_org):.4f}  val {vloss:.4f}")
        # Ранняя остановка: веса всё равно берутся лучшие по val, но продолжать
        # после длинного плато — только жечь время.
        if since >= PATIENCE:
            log(f"  ранняя остановка на эпохе {ep}: {PATIENCE} эпох без улучшения")
            break

    model.load_state_dict(best_state)
    log(f"лучший val loss {best:.4f} (эпоха {best_ep})")
    out = Path("outputs/metrics")
    out.mkdir(parents=True, exist_ok=True)
    # Запятая в имени файла читается плохо и ломает пути в скриптах — «+».
    site_tag = args.site.replace(",", "+")
    torch.save(best_state, out / f"openlin_unet_{site_tag}.pt")

    def predict(stack: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Вероятность по всей площадке; перекрытия патчей усредняются."""
        model.eval()
        acc = np.zeros(shape, dtype="float32")
        cnt = np.zeros(shape, dtype="float32")
        with torch.no_grad():
            for r in range(0, shape[0] - PATCH + 1, STRIDE):
                for c in range(0, shape[1] - PATCH + 1, STRIDE):
                    xb = torch.from_numpy(stack[None, :, r:r + PATCH, c:c + PATCH])
                    p = torch.sigmoid(model(xb))[0, 0].numpy()
                    acc[r:r + PATCH, c:c + PATCH] += p
                    cnt[r:r + PATCH, c:c + PATCH] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

    if args.apply_to:
        # Перенос: модель обучена на --site, меряется на чужой площадке целиком.
        zt = load_site(args.apply_to)
        # Каналы фиксируются по площадке обучения: у модели уже есть первый слой
        # известной ширины, и «лишний» канал цели её бы просто не запустил.
        x, y, valid, ch_t = build_stack(zt, channels=channels)
        if ch_t != channels:
            raise SystemExit(f"на «{args.apply_to}» нет каналов "
                             f"{set(channels) - set(ch_t)} — пересоберите площадку")
        dist = zt["dist"]
        res_m = float(zt["res_m"])
        z = zt
        prob = predict(x, y.shape)
        area = valid
        log(f"ПЕРЕНОС на «{args.apply_to}»: {y.shape[0]}x{y.shape[1]}, "
            f"валидных ячеек {int(valid.sum())}, обучения на ней НЕ было")
    else:
        prob = predict(x, y.shape)
        # Сравнение — только на валидационных полосах и только по валидным ячейкам
        area = val_mask & valid

    # Планка 1 считается один раз: детектор от допуска не зависит.
    canny_full = canny_hough_mask(z["gray"], valid)
    log(f"Кэнни→Хаф: {int((canny_full & area).sum())} ячеек "
        f"({100 * (canny_full & area).sum() / area.sum():.2f}% площади сверки)")

    rows = []
    for tol in TOLERANCES_M:
        base_rate = float(((dist <= tol) & area).sum() / area.sum())
        # Сеть на нескольких долях площади: lift обязан расти к верхушке скора,
        # иначе модель ранжирует не по делу.
        for frac in (0.02, 0.05, 0.10):
            got = lift_at_top(prob, dist, area, frac, tol)
            rows.append({"метод": f"U-Net top{int(100 * frac)}%", "допуск_м": tol,
                         "доля_цели": base_rate, **got})

        canny = canny_full & area
        rows.append({"метод": "Кэнни→Хаф", "допуск_м": tol,
                     "доля_цели": base_rate, **lift_of_mask(canny, dist, area, tol)})
        # Сеть при ТОЙ ЖЕ площади, что у Кэнни: иначе сравнение мерило бы порог.
        frac_canny = float(canny.sum() / area.sum())
        rows.append({"метод": "U-Net при площади Кэнни", "допуск_м": tol,
                     "доля_цели": base_rate,
                     **lift_at_top(prob, dist, area, frac_canny, tol)})

        # Планка 2: случайная маска той же площади — по построению даёт lift 1.0,
        # и служит проверкой самой метрики, а не конкурентом.
        rnd = np.zeros_like(area)
        flat = np.flatnonzero(area.ravel())
        pick = np.random.default_rng(SEED).choice(
            flat, size=min(int(frac_canny * area.sum()), flat.size), replace=False)
        rnd.ravel()[pick] = True
        rows.append({"метод": "случайная маска", "допуск_м": tol,
                     "доля_цели": base_rate, **lift_of_mask(rnd, dist, area, tol)})

    df = pd.DataFrame(rows)
    log("")
    log("LIFT ОТНОСИТЕЛЬНО РАЗМЕТКИ (валидационные полосы):")
    log(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    tag = f"{site_tag}_to_{args.apply_to}" if args.apply_to else site_tag
    df.insert(0, "площадка", args.apply_to or args.site)
    df.insert(1, "обучена_на", args.site)
    df.to_csv(out / f"openlin_unet_{tag}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out / f"openlin_unet_{tag}_pred.npz",
                        prob=prob, canny=canny_full, area=area)
    log(f"сохранено: outputs/metrics/openlin_unet_{tag}.csv и _pred.npz")


if __name__ == "__main__":
    main()
