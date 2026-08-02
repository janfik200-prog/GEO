"""Этап 2a: медианный летний композит Sentinel-2 на лист R-48-XI,XII.

Идемпотентно: каждая сцена, посаженная на сетку, кладётся в кэш
``datacache/s2_anabar``; повторный запуск ничего не качает заново. Прерванный
запуск можно просто перезапустить.

Стоп-правило ветки (из плана фазы): если после масок SCL и вегетации валидных
ячеек меньше 30% — композит в датасет v2 не идёт, остаёмся на Landsat 7 и
текстурных признаках. Проверка печатается в конце как вердикт.

Контроль качества:

* сколько сцен найдено и отобрано, по тайлам и годам;
* карта числа наблюдений на пиксель (где медиана вообще осмысленна);
* доля валидных ячеек после всех масок — против стоп-правила;
* корреляция каналов S2 с каналами Landsat 7 из v1 (санитарная проверка
  посадки на сетку: если снимки о разном, корреляция будет около нуля);
* превью каналов и отношений.

Выход: ``data/processed/s2_features.parquet``, ``outputs/s2_composite_preview.png``.
Запуск из корня: ``python -m experiments.fetch_s2``.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import cell_mask, config, integro_grid, s2_composite  # noqa: E402


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    bbox = s2_composite.bbox_wgs84(meta)
    log(f"охват листа: {bbox[0]:.2f}..{bbox[2]:.2f} в.д., {bbox[1]:.2f}..{bbox[3]:.2f} с.ш.")

    scenes = s2_composite.search_scenes(meta)
    by_tile = Counter(s["tile"] for s in scenes)
    by_year = Counter(s["datetime"][:4] for s in scenes)
    log(f"отобрано сцен: {len(scenes)}; по тайлам {dict(sorted(by_tile.items()))}")
    log(f"по годам {dict(sorted(by_year.items()))}; "
        f"облачность {min(s['cloud'] for s in scenes):.1f}..{max(s['cloud'] for s in scenes):.1f}%")

    comp, n_obs = s2_composite.median_composite(
        scenes, meta,
        progress=lambda i, n, sc: log(f"сцена {i}/{n}: {sc['id']} "
                                      f"(облачность {sc['cloud']:.0f}%)"))
    ratios = s2_composite.band_ratios(comp)
    feats = s2_composite.to_grid_features(comp, ratios, n_obs, meta)
    log(f"композит собран: {len(feats.columns)} признаков на {len(feats)} ячеек")

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    valid = cell_mask.build_valid_mask(df)
    assert len(feats) == len(df), f"длина {len(feats)} != датасета {len(df)}"

    # --- Покрытие: стоп-правило ветки ---
    has_data = np.isfinite(feats["s2_b04"].to_numpy())
    frac = float((has_data & valid).sum() / valid.sum())
    print(f"\nЧисло наблюдений на пиксель: медиана {np.median(n_obs):.0f}, "
          f"минимум {n_obs.min():.0f}, максимум {n_obs.max():.0f}")
    print(f"Доля валидных ячеек с композитом: {frac:.1%} "
          f"({int((has_data & valid).sum())} из {int(valid.sum())})")
    verdict = "ВЕТКА ОТКРЫТА" if frac >= 0.30 else "ВЕТКА ЗАКРЫВАЕТСЯ (стоп-правило <30%)"
    print(f"Вердикт стоп-правила: {verdict}")

    print(f"\n{'признак':<18}{'NaN, %':>9}{'min':>12}{'медиана':>12}{'max':>12}")
    for c in feats.columns:
        v = feats[c].to_numpy()[valid]
        nan_pct = 100.0 * np.isnan(v).mean()
        if np.isnan(v).all():
            print(f"{c:<18}{nan_pct:>9.1f}{'-':>12}{'-':>12}{'-':>12}")
            continue
        print(f"{c:<18}{nan_pct:>9.1f}{np.nanmin(v):>12.4g}"
              f"{np.nanmedian(v):>12.4g}{np.nanmax(v):>12.4g}")

    # --- Санитарная проверка посадки: S2 против Landsat 7 из v1 ---
    pairs = [("s2_b02", "ls_ch1"), ("s2_b04", "ls_ch3"), ("s2_b8a", "ls_ch4"),
             ("s2_b11", "ls_ch5"), ("s2_b12", "ls_ch7")]
    print("\nСверка с Landsat 7 (v1) — проверка, что снимки посажены на ту же сетку:")
    for s2c, lsc in pairs:
        if lsc not in df.columns:
            continue
        a, b = feats[s2c].to_numpy()[valid], df[lsc].to_numpy()[valid]
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 else np.nan
        flag = "" if r > 0.3 else "   <-- ВНИМАНИЕ: связь слабая"
        print(f"  {s2c:<10} ~ {lsc:<8} r = {r:+.2f} по {int(ok.sum())} ячейкам{flag}")

    out_path = config.PROCESSED_DIR / "s2_features.parquet"
    feats.to_parquet(out_path, index=False)

    # --- Превью ---
    shape = (meta.prf, meta.pic)
    mask2d = np.asarray(valid).reshape(shape)
    show = ["s2_b04", "s2_b8a", "s2_b11", "s2_b12",
            "s2_iron_ox", "s2_ferrous", "s2_clay", "s2_ndvi",
            "s2_b04_std", "s2_clay_std", "s2_valid_frac", "s2_n_obs"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for ax, c in zip(axes.ravel(), show):
        img = np.where(mask2d, feats[c].to_numpy().reshape(shape), np.nan)
        if np.isfinite(img).sum() < 10:
            ax.set_title(f"{c}: нет данных", fontsize=10)
            ax.axis("off")
            continue
        lo, hi = np.nanpercentile(img, [2, 98])
        im = ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi)
        ax.set_title(c, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Медианный композит Sentinel-2 L2A, {'-'.join(map(str, config.S2_MONTHS))} "
                 f"месяцы {min(config.S2_YEARS)}-{max(config.S2_YEARS)}, "
                 f"{len(scenes)} сцен, шаг {config.S2_RES_M:.0f} м -> ячейки {meta.dx:.0f} м",
                 fontsize=14)
    fig.tight_layout()
    out_png = ROOT / "outputs" / "s2_composite_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    log(f"сохранено: {out_path}, {out_png}")


if __name__ == "__main__":
    main()
