"""Прогон площадки с исправленным третьим шагом.

То же, что :mod:`experiments.openlin_unsup`, но поле линейности считается
оператором из :mod:`experiments.openlin_edges2`: отклик на линию (центральная
полоса против боковых) и на ступень, оба нормированы на разброс внутри окон и
приведены к локальному фону.

Зачем: прежний оператор в текстуре не видел узкую линию вовсе (на синтетике она
попадала в верхние 20 % поля лишь на 14 % длины — ниже случайных 20 %), зато сам
рисовал ковёр из вытянутых артефактов. Новый берёт ту же линию на 62 % длины, а
границу блоков — на 98 %.

Печатает числа рядом с прежними и сохраняет карты линий в том же формате, что
``openlin_unsup``, чтобы дальше работали все существующие разборы.

Запуск из корня репозитория:
    python -m experiments.openlin_unsup2 --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_edges2 import edges_both, edges_curve
from experiments.openlin_edges3 import edges_hessian
from experiments.openlin_gate import components
from experiments.openlin_lines import DENSITY_DEFAULT, _branches, trace_lines
from experiments.openlin_unsup import OPTIC, RADAR, confirmed_branches

M = Path("outputs/metrics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--operator", default="hessian",
                    choices=("hessian", "curve", "both"),
                    help="hessian — детектор по кривизне (шаг 3 заново); "
                         "curve — короткое окно с накоплением; both — окно 2 км")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"площадка {a.site}, плотность '{a.density}', оператор: {a.operator}")

    fields, lines = {}, {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        t0 = time.time()
        z, valid, res_m, comp = components(a.site, chans)
        op = {"hessian": edges_hessian, "curve": edges_curve,
              "both": edges_both}[a.operator]
        s, ang = op(comp, valid, res_m)
        s = np.nan_to_num(s, nan=0.0)
        mask, df = trace_lines(s, valid, valid, res_m, a.density)
        n = len([p for p in _branches(mask, res_m) if len(p) >= 3])
        fields[name] = (s, ang)
        lines[name] = mask
        log(f"  {name:13s} {len(comp)} компонент -> {n:4d} веток, "
            f"{mask.sum() * res_m / 1000:6.0f} км  ({time.time() - t0:.0f} с)")

    lo, lr, la = lines["оптика"], lines["радар"], lines["оптика+радар"]
    keep, n_pass, n_all = confirmed_branches(la, lo, res_m)
    keep_r, n_pass_r, _ = confirmed_branches(la, lr, res_m)
    both = np.zeros_like(la)
    # ветка проходит, если её подтверждают оба прибора по отдельности
    from experiments.openlin_unsup import CONFIRM_FRAC
    from experiments.openlin_confidence import _fracs

    coords, f_opt, f_rad = _fracs(la, {"оптика": lo, "радар": lr}, res_m)
    ok = (f_opt >= CONFIRM_FRAC) & (f_rad >= CONFIRM_FRAC)
    for good, pts in zip(ok, coords):
        if good:
            both[pts[:, 0], pts[:, 1]] = True
    log(f"\n  подтверждено обоими: {int(ok.sum())} веток, "
        f"{both.sum() * res_m / 1000:.0f} км (из {len(coords)})")

    # --- случайный уровень тем же тороидальным сдвигом ---------------------
    rng = np.random.default_rng(0)
    h, w = la.shape
    n0 = []
    for _ in range(25):
        so = (int(rng.integers(h)), int(rng.integers(w)))
        sr = (int(rng.integers(h)), int(rng.integers(w)))
        _c, a_o, a_r = _fracs(la, {"оптика": np.roll(lo, so, (0, 1)),
                                   "радар": np.roll(lr, sr, (0, 1))}, res_m)
        n0.append(int(((a_o >= CONFIRM_FRAC) & (a_r >= CONFIRM_FRAC)).sum()))
    null = float(np.mean(n0))
    log(f"  случайный уровень: {null:.1f} ветки, "
        f"чистота ×{ok.sum() / max(null, 1e-9):.0f}")

    out = M / f"openlin_unsup_{a.site}_{a.density}_{a.tag}.npz"
    np.savez_compressed(out, lines_0=lo, lines_1=lr, lines_2=la,
                        confirmed=both, field=fields["оптика+радар"][0],
                        azimuth=fields["оптика+радар"][1])
    log(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
