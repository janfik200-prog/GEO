"""Главная проверка: а находит ли метод хоть что-нибудь, чего нет в шуме.

ИДЕЯ. Все прежние проверки сравнивали линии с линиями (оптика против радара,
сдвиг карты по тору). Они отвечают на вопрос «совпадают ли два прогона», но не
на вопрос «есть ли вообще объект». Здесь вопрос ставится иначе.

Композит подвергается ФАЗОВОЙ РАНДОМИЗАЦИИ: у каждого канала берётся спектр,
амплитуды сохраняются, а фазы заменяются случайными. На выходе картинка с той
же автокорреляцией, той же пёстростью и тем же масштабом текстуры, но без
единой настоящей линии: все протяжённые объекты разрушены, потому что именно
фаза их держит.

Дальше по этой подделке прогоняется весь метод. Если она даёт столько же линий
той же длины, что и настоящий снимок, значит метод рисует собственные
артефакты, а не находит объекты. Разница между «настоящее» и «подделка» — и
есть честная мера того, что метод видит.

Запуск из корня репозитория:
    python -m experiments.openlin_null_input --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_lines import DENSITY_DEFAULT, _branches, trace_lines
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import build_stack, load_site
from experiments.openlin_unsup import OPTIC, RADAR

M = Path("outputs/metrics")

#: переключатель: вернуть маску вместо чисел (внутренний)
_RETURN_MASK = [False]


def phase_randomize(img: np.ndarray, valid: np.ndarray, rng) -> np.ndarray:
    """Та же текстура, те же масштабы, но все протяжённые объекты разрушены.

    Спектр амплитуд сохраняется, фазы заменяются случайными с сохранением
    эрмитовой симметрии, поэтому результат остаётся вещественным и имеет ту же
    автокорреляционную функцию, что исходная картинка.
    """
    x = np.where(valid, img, np.nanmedian(img[valid])).astype("float64")
    x = np.nan_to_num(x, nan=float(np.nanmedian(img[valid])))
    f = np.fft.rfft2(x)
    ph = rng.uniform(-np.pi, np.pi, f.shape)
    ph[0, 0] = 0.0
    out = np.fft.irfft2(np.abs(f) * np.exp(1j * ph), s=x.shape)
    # вернуть исходные среднее и разброс: оператор нормирует вход, но пусть
    # подделка совпадает с оригиналом и по этим двум числам тоже
    out = (out - out.mean()) / (out.std() or 1.0)
    return (out * x[valid].std() + x[valid].mean()).astype("float32")


def _mask(site: str, chans, density: str, operator: str, fake: bool, rng,
          compress: str = "pca"):
    """Маска линий одного прогона."""
    _RETURN_MASK[0] = True
    try:
        return run(site, chans, density, operator, fake, rng, None, compress)
    finally:
        _RETURN_MASK[0] = False


def run(site: str, chans, density: str, operator: str, fake: bool, rng,
        log, compress: str = "pca") -> tuple[int, float, float]:
    """Прогон цепочки: возвращает (веток, км, средняя длина в м)."""
    from sklearn.decomposition import PCA

    from experiments.openlin_edges2 import edges_curve
    from experiments.openlin_edges3 import edges_hessian
    from experiments.openlin_domains import edges_from_components

    z = load_site(site)
    use = tuple(c for c in chans if c in z)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    if fake:
        z = dict(z)
        for c in use:
            z[c] = np.where(valid, phase_randomize(z[c], valid, rng), np.nan)

    x, _y, valid2, _u = build_stack(z, use)
    if compress == "raw":
        stack = x.astype("float32")
    else:
        nx, _n = neigh_stack(x, use, valid2, res_m, base=use)
        stack = np.concatenate([x, nx]).astype("float32")
    if compress == "raw":
        # без признаков окрестности и без сжатия: оператор идёт прямо по
        # каналам. Нужно потому, что окрестность 0.5-2 км растворяет тонкую
        # протяжённую линию ещё до шага 3
        comp = x.astype("float32")
    elif compress == "ae":
        # сеть обучается заново на том входе, который ей дали: на подделке
        # своя, иначе сравнение было бы нечестным
        from experiments.openlin_mae import latent_map, train_ae

        model = train_ae(stack, valid2, lambda *_a, **_k: None)
        comp = latent_map(model, stack, valid2)
    else:
        rows = stack[:, valid2].T
        sel = rng.choice(len(rows), min(50_000, len(rows)), replace=False)
        pca = PCA(n_components=8, whiten=True, random_state=42).fit(rows[sel])
        comp = np.zeros((8,) + valid2.shape, dtype="float32")
        comp[:, valid2] = pca.transform(rows).T

    op = {"hessian": edges_hessian, "curve": edges_curve,
          "old": edges_from_components}[operator]
    s, _ang = op(comp, valid2, res_m)
    mask, _df = trace_lines(np.nan_to_num(s, nan=0.0), valid2, valid2, res_m,
                            density)
    if _RETURN_MASK[0]:
        return mask
    parts = [p for p in _branches(mask, res_m) if len(p) >= 3]
    km = float(mask.sum()) * res_m / 1000.0
    return len(parts), km, km / max(len(parts), 1) * 1000


def run_confirm(site: str, density: str, operator: str, fake: bool, rng,
                log) -> tuple[int, int, float]:
    """Полный цикл с подтверждением: (прослежено, подтверждено, км)."""
    from experiments.openlin_confidence import _fracs
    from experiments.openlin_unsup import CONFIRM_FRAC

    masks = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        masks[name] = _mask(site, chans, density, operator, fake, rng)
    lo, lr, la = masks["оптика"], masks["радар"], masks["оптика+радар"]
    res_m = float(load_site(site)["res_m"])
    coords, f_o, f_r = _fracs(la, {"оптика": lo, "радар": lr}, res_m)
    ok = (f_o >= CONFIRM_FRAC) & (f_r >= CONFIRM_FRAC)
    km = sum(len(p) for p, good in zip(coords, ok) if good) * res_m / 1000.0
    return len(coords), int(ok.sum()), km


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    ap.add_argument("--operators", default="old,curve,hessian")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"площадка {a.site}, плотность '{a.density}'")
    log("сравниваются настоящий композит и его фазовая подделка "
        "(тот же спектр, никаких линий)\n")
    log(f"{'оператор':10s} {'вход':12s} {'веток':>7s} {'км':>7s} "
        f"{'средняя':>9s}")
    for operator in [o for o in a.operators.split(",") if o]:
        res = {}
        for fake in (False, True):
            rng = np.random.default_rng(7)
            t0 = time.time()
            n, km, mean = run(a.site, OPTIC + RADAR, a.density, operator,
                              fake, rng, log)
            res[fake] = (n, km, mean)
            log(f"{operator:10s} {'подделка' if fake else 'настоящий':12s} "
                f"{n:7d} {km:7.0f} {mean:8.0f}м  ({time.time() - t0:.0f} с)")
        n0, km0, m0 = res[False]
        n1, km1, m1 = res[True]
        log(f"{'':10s} {'отношение':12s} {n0 / max(n1, 1):7.2f} "
            f"{km0 / max(km1, 1e-9):7.2f} {m0 / max(m1, 1e-9):8.2f}\n")


if __name__ == "__main__":
    main()
