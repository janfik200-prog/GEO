"""Дописать каналы интерферометрической когерентности в готовый файл площадки.

Оптика и радарные композиты собираются десятками минут, а когерентность берётся
из готового глобального продукта за секунды — перекачивать ради неё всё остальное
незачем. Скрипт читает ``openlin_<площадка>.npz``, добавляет каналы ``coh_*`` и
сохраняет обратно.

Запуск из корня репозитория:
    python -m experiments.openlin_addcoh --site anabar
    python -m experiments.openlin_addcoh --site nevada --site abitibi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_probe import SITES, build_grid, coherence

M = Path("outputs/metrics")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", action="append", required=True,
                    help="можно указать несколько раз")
    args = ap.parse_args()

    for name in args.site:
        # Цель могла быть переопределена (anabar_dykes) — сетка та же, что у базы.
        base = name.split("_")[0] if name.split("_")[0] in SITES else name
        path = M / f"openlin_{name}.npz"
        if not path.exists():
            print(f"!! нет {path} — пропущено")
            continue
        z = dict(np.load(path))
        grid = build_grid(SITES[base], float(z["res_m"]))
        print(f"\n=== {name} (сетка {grid.height}x{grid.width}) ===")

        def log(msg=""):
            print(msg, flush=True)

        z.update(coherence(grid, log))
        np.savez_compressed(path, **z)
        chans = [k for k in z if k.startswith("coh_")]
        print(f"сохранено: {path}; каналов когерентности {len(chans)}")


if __name__ == "__main__":
    main()
