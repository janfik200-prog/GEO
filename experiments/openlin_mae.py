"""Представление вместо PCA: маскированный автоэнкодер и та же проверка.

ЗАЧЕМ. Скачок сигнатуры сейчас меряется в пространстве PCA — линейном и общем
на всю площадку. Линией в нём становится перепад СРЕДНЕГО. Смена типа текстуры
при том же среднем линией не станет, а для линеамента это ровно такой же
признак. Автоэнкодер даёт нелинейное представление, обученное на самом снимке,
и вопрос ставится измеримо: даёт ли латент линии, которых PCA не видит.

ЧЕМ СРАВНИВАЕМ, РАЗ РАЗМЕТКИ НЕТ. Тем же критерием, что и всё в этой ветке:
доля линий, подтверждённых ОБОИМИ приборами вдоль длины и по направлению,
против числа линий, которые прошли бы тот же фильтр случайно. У PCA это 153
линии при 0.9 случайных (чистота x174). Латент побеждает, если даёт больше
подтверждённых при той же чистоте, — и проигрывает, если добирает линий за счёт
чистоты.

ЧТО ЗА МОДЕЛЬ. Полностью свёрточный автоэнкодер без понижения разрешения:
расширяющиеся свёртки (1, 2, 4) дают рецептивное поле около 1.5 км при
сохранении шага 100 м, то есть латент остаётся ПОЯЧЕЕЧНЫМ и подставляется в
оператор границы вместо компонент PCA один в один. Обучение маскированное:
часть ячеек обнуляется, и сеть обязана восстановить их по окрестности — это
заставляет латент кодировать контекст, а не значение самой ячейки.

Обучения с учителем здесь нет: разметка не используется ни на одном шаге.

Запуск из корня репозитория:
    python -m experiments.openlin_mae --site anabar --density "предельно"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.openlin_domains import edges_from_components
from experiments.openlin_lines import DENSITY_DEFAULT, _branches, trace_lines
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, load_site
from experiments.openlin_unsup import OPTIC, RADAR, confirmed_branches

M = Path("outputs/metrics")

#: Размер латента. Восемь — столько же, сколько компонент у PCA: сравниваются
#: представления, а не их размерности.
N_LATENT = 8

#: Расширения свёрток. 1-2-4 при ядре 3x3 дают рецептивное поле 15 ячеек = 1.5 км
#: — тот же масштаб, на котором считаются признаки окрестности.
DILATIONS = (1, 2, 4)
N_HIDDEN = 32

#: Доля обнуляемых ячеек при обучении. 0.5 — стандарт маскированных
#: автоэнкодеров: меньше — задача решается копированием, больше — контекста не
#: остаётся.
MASK_RATIO = 0.5

CROP = 128           # размер обучающего окна, ячеек
BATCH = 8
STEPS = 400
LR = 2e-3
SEED = 42


class ConvAE(nn.Module):
    """Свёрточный автоэнкодер без понижения разрешения."""

    def __init__(self, n_in: int, n_latent: int = N_LATENT):
        super().__init__()
        layers, prev = [], n_in
        for d in DILATIONS:
            layers += [nn.Conv2d(prev, N_HIDDEN, 3, padding=d, dilation=d),
                       nn.ReLU()]
            prev = N_HIDDEN
        layers += [nn.Conv2d(prev, n_latent, 3, padding=1)]
        self.enc = nn.Sequential(*layers)
        self.dec = nn.Sequential(
            nn.Conv2d(n_latent, N_HIDDEN, 3, padding=1), nn.ReLU(),
            nn.Conv2d(N_HIDDEN, n_in, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def train_ae(x: np.ndarray, valid: np.ndarray, log,
             steps: int = STEPS) -> ConvAE:
    """Обучение на случайных окнах растра; разметка не используется."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = ConvAE(x.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    h, w = valid.shape
    xt = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype("float32"))
    vt = torch.from_numpy(valid.astype("float32"))

    t0 = time.time()
    hist = []
    for step in range(steps):
        rr = rng.integers(0, h - CROP, size=BATCH)
        cc = rng.integers(0, w - CROP, size=BATCH)
        crop = torch.stack([xt[:, r:r + CROP, c:c + CROP]
                            for r, c in zip(rr, cc)])
        vm = torch.stack([vt[r:r + CROP, c:c + CROP] for r, c in zip(rr, cc)])
        keep = (torch.rand(crop.shape[0], 1, CROP, CROP) > MASK_RATIO).float()
        out = model(crop * keep)
        # Потеря считается ТОЛЬКО на скрытых и валидных ячейках: на видимых сеть
        # решала бы задачу копированием, а вне кадра восстанавливать нечего.
        m = (1.0 - keep) * vm[:, None]
        loss = ((out - crop) ** 2 * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
        if step % 200 == 199:
            # Средняя по последним 100 шагам, а не мгновенная: у маскированной
            # задачи потеря скачет от того, какие именно ячейки скрыты.
            log(f"    шаг {step + 1}/{steps}, потеря "
                f"{np.mean(hist[-100:]):.4f}")
    first, last = np.mean(hist[:100]), np.mean(hist[-100:])
    log(f"    обучено за {time.time() - t0:.0f} с, "
        f"параметров {sum(p.numel() for p in model.parameters()):,}; "
        f"потеря {first:.3f} -> {last:.3f} (падение {100 * (1 - last / first):.0f}%)")
    return model


def latent_map(model: ConvAE, x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Латент по всему растру, отбелённый как компоненты PCA."""
    model.eval()
    with torch.no_grad():
        z = model.enc(torch.from_numpy(
            np.nan_to_num(x, nan=0.0).astype("float32"))[None])[0].numpy()
    out = np.zeros_like(z)
    for i in range(z.shape[0]):
        v = z[i][valid]
        out[i] = (z[i] - v.mean()) / (v.std() or 1.0)
    return np.where(valid[None], out, 0.0).astype("float32")


def build(site: str, chans: tuple[str, ...]):
    z = load_site(site)
    use = tuple(c for c in chans if c in z)
    x, _y, valid, _u = build_stack(z, use)
    res_m = float(z["res_m"])
    nx, _n = neigh_stack(x, use, valid, res_m, base=use)
    return z, valid, res_m, np.concatenate([x, nx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"площадка {a.site}, плотность '{a.density}'; латент {N_LATENT}, "
        f"маскирование {MASK_RATIO}, шагов {a.steps}")

    lines = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        log(f"\n--- {name} ---")
        z, valid, res_m, x = build(a.site, chans)
        model = train_ae(x, valid, log, steps=a.steps)
        comp = latent_map(model, x, valid)
        strength, _ang = edges_from_components(comp, valid, res_m)
        ln, br = trace_lines(np.nan_to_num(strength, nan=0.0), valid, valid,
                             res_m, a.density)
        lines[name] = ln
        log(f"    линий {len(br)}, длина {ln.sum() * res_m / 1000:.0f} км")

    l_all, lo, lr = (lines["оптика+радар"], lines["оптика"], lines["радар"])
    keep_r, n_r, n_all = confirmed_branches(l_all, lr, res_m)
    keep_o, n_o, _ = confirmed_branches(l_all, lo, res_m)
    keep = keep_r & keep_o
    n_keep = sum(1 for p in _branches(keep, res_m) if len(p) >= 3)

    rng = np.random.default_rng(1)
    h, w = l_all.shape
    fake = []
    for _ in range(25):
        s1 = (int(rng.integers(h)), int(rng.integers(w)))
        s2 = (int(rng.integers(h)), int(rng.integers(w)))
        k1, _p, _n = confirmed_branches(l_all, np.roll(lr, s1, (0, 1)) & valid,
                                        res_m)
        k2, _p, _n = confirmed_branches(l_all, np.roll(lo, s2, (0, 1)) & valid,
                                        res_m)
        fake.append(sum(1 for p in _branches(k1 & k2, res_m) if len(p) >= 3))

    log("\n=== латент автоэнкодера против PCA " + "=" * 30)
    log(f"  прослежено {n_all} веток, подтверждено обоими {n_keep} "
        f"({keep.sum() * res_m / 1000:.0f} км)")
    log(f"  прошло бы случайно {np.mean(fake):.1f}, "
        f"чистота x{n_keep / max(np.mean(fake), 1e-9):.0f}")
    log("  для сравнения PCA на той же плотности: см. "
        f"openlin_unsup_{a.site}_{a.density}.csv")

    pd.DataFrame([{"представление": "автоэнкодер", "плотность": a.density,
                   "прослежено": n_all, "подтверждено": n_keep,
                   "длина_км": keep.sum() * res_m / 1000,
                   "случайно": float(np.mean(fake)),
                   "чистота": n_keep / max(np.mean(fake), 1e-9)}]).to_csv(
        M / f"openlin_mae_{a.site}_{a.density}{a.tag}.csv", index=False,
        encoding="utf-8-sig")
    np.savez_compressed(M / f"openlin_mae_{a.site}_{a.density}{a.tag}.npz",
                        lines_0=lo, lines_1=lr, lines_2=l_all, confirmed=keep)
    log(f"\nсохранено: {M / f'openlin_mae_{a.site}_{a.density}.csv'}")


if __name__ == "__main__":
    main()
