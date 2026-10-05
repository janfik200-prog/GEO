"""Композит и две половины на ПОЛНОМ наборе зимних сцен (55 вместо 12).

Обработка та же, что у композита на 12 сценах (:mod:`openlin_winter`):
выравнивание яркости каждой сцены, медиана по ячейке, детренд на 2 км; NDVI
сцены — по сырым яркостям. Половины — те же: внутри плитки сцены по дате
раздаются через одну. Меняется только число сцен: по плиткам 10 / 9 / 17 / 19
вместо 3, в половинах 5-10 сцен на плитку вместо 2 и 1.

Файлы — под новыми именами (``…_лист_все…``), прежние не трогаются. Медиана
считается полосами строк через файл на диске: 55 сцен листа разом в памяти —
несколько гигабайт на канал.

Запуск из корня репозитория:
    python -m experiments.openlin_all_scenes --site anabar
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments.openlin_fetch_all import DEST
from experiments.openlin_winter import MIN_SCENES, _ndvi, detrend, normalize
from experiments.openlin_zone_lines import COMP

TAG = "_все"


def scene_groups() -> tuple[list, dict]:
    """Все сцены по дате и их раздача половинам — как в build_halves."""
    files = sorted(DEST.glob("*_лист_20m.npz"), key=lambda f: f.name.split("_")[2][:8])
    by_tile: dict[str, list] = {}
    for f in files:
        tile = next(p for p in f.name.split("_") if p.startswith("T"))
        by_tile.setdefault(tile, []).append(f)
    halves = {"А": [], "Б": []}
    for _t, group in sorted(by_tile.items()):
        halves["А"] += group[0::2]
        halves["Б"] += group[1::2]
    return files, halves


def median_of(files: list, layer, min_scenes: int) -> tuple[np.ndarray, np.ndarray]:
    """Медиана слоя по сценам и число сцен в ячейке, полосами строк."""
    first = layer(dict(np.load(files[0])))
    H, W = first.shape
    # Windows не даёт удалить файл, пока он отображён в память: отображение
    # закрывается явно, а сбой уборки временной папки не роняет прогон
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        cube = np.lib.format.open_memmap(Path(td) / "cube.npy", mode="w+",
                                         dtype="float32", shape=(len(files), H, W))
        cube[0] = first
        for k, f in enumerate(files[1:], 1):
            cube[k] = layer(dict(np.load(f)))
        med = np.full((H, W), np.nan, "float32")
        cnt = np.zeros((H, W), "int16")
        for r0 in range(0, H, 128):
            part = np.asarray(cube[:, r0:r0 + 128])
            n = np.isfinite(part).sum(0)
            with np.errstate(all="ignore"):
                m = np.nanmedian(part, axis=0)
            med[r0:r0 + 128] = np.where(n >= min_scenes, m, np.nan)
            cnt[r0:r0 + 128] = n
        mm = cube._mmap
        del cube, part
        mm.close()
    return med, cnt


def band(name: str):
    def layer(z):
        a = z[name].astype("float32")
        return normalize(a, np.isfinite(a) & (a > 0))
    return layer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    t0 = time.time()
    old = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(old["res_m"])
    files, halves = scene_groups()
    print(f"сцен {len(files)}; в половинах {len(halves['А'])} и {len(halves['Б'])}",
          flush=True)

    out = {}
    for b in ("B04", "B08"):
        out[b], cnt = median_of(files, band(b), MIN_SCENES)
        out.setdefault("count", cnt)
        out["count"] = np.minimum(out["count"], cnt)
        print(f"  {b}: ячеек с медианой {100 * np.isfinite(out[b]).mean():.1f} % "
              f"({time.time() - t0:.0f} с)", flush=True)
    out["ndvi"], _c = median_of(files, _ndvi, 1)
    valid = np.isfinite(out["B04"])
    out["shade"] = detrend(out["B04"], valid, res_m)
    meta = [dict(np.load(f)) for f in files[:0]]
    sun = np.array([float(np.load(f)["sun"]) for f in files])
    az = np.array([float(np.load(f)["az"]) for f in files])
    np.savez_compressed(COMP / f"winter_{a.site}_20m_лист{TAG}.npz", **out,
                        res_m=old["res_m"], x0=old["x0"], y_top=old["y_top"],
                        epsg=old["epsg"], sun=sun, azimuth=az,
                        scenes=np.array([f.name for f in files]))
    v, k = np.unique(out["count"], return_counts=True)
    top = sorted(zip(k, v), reverse=True)[:6]
    print("композит: солнце %.1f–%.1f°, азимут %.0f–%.0f°; сцен в ячейке (доля листа): %s"
          % (sun.min(), sun.max(), az.min(), az.max(),
             ", ".join(f"{int(n)}: {100 * c / out['count'].size:.0f} %" for c, n in top)),
          flush=True)

    for name, group in halves.items():
        med, cnt = median_of(group, band("B04"), 1)
        sh = detrend(med, np.isfinite(med), res_m)
        np.savez_compressed(COMP / f"winter_{a.site}_20m_лист{TAG}_{name}.npz",
                            shade=sh, count=cnt)
        print(f"половина «{name}»: сцен {len(group)}, покрытие "
              f"{100 * np.isfinite(sh).mean():.1f} %, сцен в ячейке: медиана "
              f"{int(np.median(cnt))}, минимум {int(cnt[np.isfinite(sh)].min())}, "
              f"доля ячеек с одной сценой {100 * (cnt == 1).mean():.1f} % "
              f"({time.time() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
