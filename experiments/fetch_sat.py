"""Этап 8: сбор остальных бесплатных съёмок на лист R-48-XI,XII.

Один прогон — один сенсор: ``python -m experiments.fetch_sat s1`` (или ``l8``,
``psr``, ``ast``, либо ``all`` подряд). Идемпотентно: каждая сцена, посаженная
на сетку, кладётся в свой кэш, повторный запуск ничего не качает заново, а
прерванный запуск просто перезапускается.

Контроль качества у всех сенсоров одинаковый и печатается в конце:

* сколько сцен найдено, по группам (орбиты / path / даты) и годам;
* число наблюдений на пиксель — где медиана вообще осмысленна;
* доля валидных ячеек после масок (стоп-правило ветки — 30%, как у S2);
* корреляция с уже имеющимися слоями: радар обязан быть связан с рельефом, но
  НЕ обязан его повторять; если |r| с dem_elev выше 0.9 — сенсор меряет рельеф,
  и признак дублирует группу ``dem_*``;
* превью каналов и индексов.

Выход: ``data/processed/<sensor>_features.parquet`` и
``outputs/<sensor>_preview.png``.
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

from src import cell_mask, config, integro_grid, sat_sources  # noqa: E402

STOP_RULE_FRAC = 0.30


def run(sensor: str, log) -> None:
    fn, title = sat_sources.SENSORS[sensor]
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    log(f"=== {title} ===")

    t = time.time()
    feats, items = fn(meta, log)
    log(f"композит собран за {time.time() - t:.0f} c: "
        f"{len(feats.columns)} признаков на {len(feats)} ячеек")

    by_year = Counter(i.datetime.year for i in items)
    log(f"сцен по годам: {dict(sorted(by_year.items()))}")

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v2.parquet")
    valid = cell_mask.build_valid_mask(df)
    assert len(feats) == len(df), f"длина {len(feats)} != датасета {len(df)}"

    # --- Покрытие и стоп-правило ---
    n_obs_col = [c for c in feats.columns if c.endswith("n_obs")]
    base = feats.columns[0]
    has = np.isfinite(feats[base].to_numpy())
    frac = float((has & valid).sum() / valid.sum())
    if n_obs_col:
        no = feats[n_obs_col[0]].to_numpy()
        print(f"\nНаблюдений на ячейку: медиана {np.nanmedian(no):.1f}, "
              f"минимум {np.nanmin(no):.1f}, максимум {np.nanmax(no):.1f}")
    print(f"Доля валидных ячеек с композитом: {frac:.1%} "
          f"({int((has & valid).sum())} из {int(valid.sum())})")
    print("Вердикт стоп-правила: "
          + ("ВЕТКА ОТКРЫТА" if frac >= STOP_RULE_FRAC
             else f"ВЕТКА ЗАКРЫВАЕТСЯ (валидных <{STOP_RULE_FRAC:.0%})"))

    print(f"\n{'признак':<20}{'NaN, %':>9}{'min':>12}{'медиана':>12}{'max':>12}")
    for c in feats.columns:
        v = feats[c].to_numpy()[valid]
        nan_pct = 100.0 * np.isnan(v).mean()
        if np.isnan(v).all():
            print(f"{c:<20}{nan_pct:>9.1f}{'-':>12}{'-':>12}{'-':>12}")
            continue
        print(f"{c:<20}{nan_pct:>9.1f}{np.nanmin(v):>12.4g}"
              f"{np.nanmedian(v):>12.4g}{np.nanmax(v):>12.4g}")

    # --- Проверка на дубль уже имеющихся групп ---
    print("\nСвязь с готовыми слоями (|r| > 0.9 = сенсор дублирует, а не дополняет):")
    ref = [c for c in ("dem_elev", "dem_slope", "s2_ndvi", "s2_b12", "lin_dens")
           if c in df.columns]
    worst = []
    for c in feats.columns:
        if c.endswith(("_std", "valid_frac", "n_obs")):
            continue
        a = feats[c].to_numpy()[valid]
        for r in ref:
            b = df[r].to_numpy()[valid]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 100:
                continue
            rr = float(np.corrcoef(a[ok], b[ok])[0, 1])
            worst.append((abs(rr), c, r, rr))
    for _, c, r, rr in sorted(worst, reverse=True)[:8]:
        flag = "   <-- ДУБЛЬ" if abs(rr) > 0.9 else ""
        print(f"  {c:<18} ~ {r:<10} r = {rr:+.2f}{flag}")

    out_path = config.PROCESSED_DIR / f"{sensor}_features.parquet"
    feats.to_parquet(out_path, index=False)

    # --- Превью ---
    shape = (meta.prf, meta.pic)
    mask2d = np.asarray(valid).reshape(shape)
    show = [c for c in feats.columns if not c.endswith("_std")][:12]
    n = len(show)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(20, 3.5 * ((n + 3) // 4)))
    for ax, c in zip(np.ravel(axes), show):
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
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}: {len(items)} сцен, шаг {config.SAT_RES_M:.0f} м "
                 f"-> ячейки {meta.dx:.0f} м", fontsize=14)
    fig.tight_layout()
    out_png = ROOT / "outputs" / f"{sensor}_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"сохранено: {out_path}, {out_png}")


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    arg = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    names = list(sat_sources.SENSORS) if arg == "all" else [arg]
    unknown = [n for n in names if n not in sat_sources.SENSORS]
    if unknown:
        raise SystemExit(f"неизвестный сенсор {unknown}; "
                         f"доступны: {list(sat_sources.SENSORS)} или all")
    for name in names:
        try:
            run(name, log)
        except Exception as exc:                     # один сенсор не роняет прогон
            log(f"СЕНСОР {name} УПАЛ: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
