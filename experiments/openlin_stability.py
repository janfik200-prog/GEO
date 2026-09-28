"""Отбор по воспроизводимости: линия остаётся, если переживает возмущение.

ЗАЧЕМ. Эталона нет и не будет, а когда появится — только для сверки, не для
обучения. Значит отличать объект от артефакта нужно по свойству, которое есть у
объекта и отсутствует у ошибки обработки.

Такое свойство — устойчивость. Артефакт оператора порождён конкретным
положением окна относительно конкретной реализации текстуры: сдвиньте растр на
ячейку, выбросьте один канал, добавьте слабый шум — и артефакт уедет или
исчезнет. Настоящая граница на местности останется там же.

КАК СЧИТАЕТСЯ. Прогон повторяется N раз, каждый раз вход слегка возмущается:

* сдвиг растра на 1-2 ячейки (положение окна меняется, местность — нет);
* выбрасывается один случайный канал (bootstrap по каналам);
* добавляется гауссов шум ~0.15 разброса канала.

Для каждой ячейки считается доля прогонов, в которых она попала в линию. Линия
проходит, если её медианная воспроизводимость выше порога. Порог не берётся с
потолка: та же процедура гоняется по фазовой подделке, и порог ставится там,
где воспроизводимость артефактов уже не дотягивает.

Запуск из корня репозитория:
    python -m experiments.openlin_stability --site anabar --runs 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_lines import _branches, trace_lines
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_null_input import phase_randomize
from experiments.openlin_unet import build_stack, load_site
from experiments.openlin_unsup import OPTIC, RADAR

M = Path("outputs/metrics")

#: Насколько возмущается вход. Сдвиг в ячейках, доля выбрасываемых каналов и
#: уровень добавляемого шума в долях разброса канала. Возмущение должно быть
#: достаточным, чтобы сбить артефакт, и слишком слабым, чтобы сдвинуть объект.
SHIFT_PX = 2
DROP_FRAC = 0.2
NOISE_SD = 0.15


def perturb(z: dict, use: tuple, valid: np.ndarray, rng) -> dict:
    """Возмущённая копия композита: сдвиг, выброс канала, шум."""
    out = dict(z)
    dy = int(rng.integers(-SHIFT_PX, SHIFT_PX + 1))
    dx = int(rng.integers(-SHIFT_PX, SHIFT_PX + 1))
    drop = set(rng.choice(len(use), max(1, int(len(use) * DROP_FRAC)),
                          replace=False).tolist())
    for i, c in enumerate(use):
        a = np.roll(z[c], (dy, dx), (0, 1)).astype("float32")
        if i in drop:
            a = np.full_like(a, float(np.nanmedian(a[valid])))
        else:
            sd = float(np.nanstd(a[valid])) * NOISE_SD
            a = a + rng.normal(0.0, sd, a.shape).astype("float32")
        out[c] = np.where(valid, a, np.nan)
    return out


def one_run(z: dict, use: tuple, valid: np.ndarray, res_m: float,
            density: str, operator: str) -> np.ndarray:
    """Маска линий одного прогона."""
    from sklearn.decomposition import PCA

    from experiments.openlin_domains import edges_from_components
    from experiments.openlin_edges3 import edges_hessian

    x, _y, valid2, _u = build_stack(z, use)
    nx, _n = neigh_stack(x, use, valid2, res_m, base=use)
    stack = np.concatenate([x, nx]).astype("float32")
    rows = stack[:, valid2].T
    sel = np.random.default_rng(42).choice(len(rows), min(50_000, len(rows)),
                                           replace=False)
    pca = PCA(n_components=8, whiten=True, random_state=42).fit(rows[sel])
    comp = np.zeros((8,) + valid2.shape, dtype="float32")
    comp[:, valid2] = pca.transform(rows).T
    op = edges_hessian if operator == "hessian" else edges_from_components
    s, _a = op(comp, valid2, res_m)
    mask, _df = trace_lines(np.nan_to_num(s, nan=0.0), valid2, valid2, res_m,
                            density)
    return mask


def frequency(z: dict, use: tuple, valid: np.ndarray, res_m: float,
              density: str, operator: str, runs: int, seed: int, log
              ) -> tuple[np.ndarray, np.ndarray]:
    """Карта частоты: как часто ячейка попадала в линию, и базовый прогон."""
    from scipy.ndimage import binary_dilation

    rng = np.random.default_rng(seed)
    base = one_run(z, use, valid, res_m, density, operator)
    acc = np.zeros(valid.shape, dtype="float32")
    for i in range(runs):
        t0 = time.time()
        m = one_run(perturb(z, use, valid, rng), use, valid, res_m, density,
                    operator)
        # допуск положения 250 м: линия считается воспроизведённой, если
        # оказалась рядом, а не ячейка в ячейку
        acc += binary_dilation(m, np.ones((5, 5), dtype=bool)).astype("float32")
        log(f"    прогон {i + 1}/{runs}: {m.sum() * res_m / 1000:.0f} км "
            f"({time.time() - t0:.0f} с)")
    return acc / runs, base


def line_stability(base: np.ndarray, freq: np.ndarray, res_m: float
                   ) -> tuple[list, np.ndarray]:
    """Медианная воспроизводимость каждой ветки базового прогона."""
    parts = [p for p in _branches(base, res_m) if len(p) >= 3]
    vals = np.array([float(np.median(freq[p[:, 0], p[:, 1]])) for p in parts])
    return parts, vals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--operator", default="hessian")
    ap.add_argument("--runs", type=int, default=8)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    z = load_site(a.site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    use = tuple(c for c in OPTIC + RADAR if c in z)
    log(f"площадка {a.site}, {len(use)} каналов, оператор {a.operator}, "
        f"{a.runs} возмущённых прогонов")

    log("\n  настоящий вход:")
    freq, base = frequency(z, use, valid, res_m, a.density, a.operator,
                           a.runs, 7, log)
    parts, vals = line_stability(base, freq, res_m)
    log(f"  веток {len(parts)}, медианная воспроизводимость "
        f"{np.median(vals):.2f}")

    log("\n  фазовая подделка:")
    rng = np.random.default_rng(100)
    zf = dict(z)
    for c in use:
        zf[c] = np.where(valid, phase_randomize(z[c], valid, rng), np.nan)
    freq_f, base_f = frequency(zf, use, valid, res_m, a.density, a.operator,
                               a.runs, 11, log)
    parts_f, vals_f = line_stability(base_f, freq_f, res_m)
    log(f"  веток {len(parts_f)}, медианная воспроизводимость "
        f"{np.median(vals_f):.2f}")

    log("\nсколько остаётся при отборе по воспроизводимости:")
    log(f"{'порог':>7s} {'линий':>7s} {'км':>7s} {'на подделке':>12s} "
        f"{'мусора, %':>10s}")
    km = np.array([len(p) for p in parts]) * res_m / 1000
    km_f = np.array([len(p) for p in parts_f]) * res_m / 1000
    scale = len(parts) / max(len(parts_f), 1)
    for t in np.arange(0.3, 1.01, 0.1):
        n, nf = int((vals >= t).sum()), int((vals_f >= t).sum())
        if n == 0:
            continue
        log(f"{t:7.1f} {n:7d} {km[vals >= t].sum():7.0f} {nf * scale:12.1f} "
            f"{100 * nf * scale / n:10.1f}")

    np.savez_compressed(M / f"openlin_stability_{a.site}.npz",
                        freq=freq, base=base, vals=vals)
    log(f"\nсохранено: {M / f'openlin_stability_{a.site}.npz'}")


if __name__ == "__main__":
    main()
