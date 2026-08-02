"""Этап 2d: производные рельефа v2 на ячейки сетки + контроль качества.

Зачем отдельный прогон, а не сразу в сборщик v2: рельеф — единственный источник
признаков, который в v1 считался на ГЕОГРАФИЧЕСКОМ растре (градусы). На 71° с.ш.
пиксель по долготе втрое короче, чем по широте, поэтому уклон и кривизна в v1
частично измеряли проекцию, а не рельеф. Здесь DEM перепроецирован в метрику
сетки, и надо убедиться, что новые признаки (а) не пустые, (б) не дублируют
старые, (в) не являются шумом перепроецирования.

Контроль:

* доля NaN и базовая статистика каждого признака (в физических единицах);
* корреляция ``dem_elev`` со старым ``relief_m`` (должна быть высокой — это
  проверка того, что мы посадили DEM на ту же сетку, а не сдвинули её);
* корреляционная матрица новых признаков между собой и число обусловленности
  (не тащим ли мы в пул алгебраически зависимые величины, как было с полем
  = регионалка + остаток);
* превью-панель 9 карт.

Выход: ``data/processed/terrain_v2.parquet``, ``outputs/terrain_v2_preview.png``.
Запуск из корня: ``python -m experiments.terrain_v2_run``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import cell_mask, config, integro_grid, terrain_v2  # noqa: E402

UNITS = {
    "dem_elev": "м", "dem_elev_std": "м", "dem_slope": "м/м", "dem_slope_std": "м/м",
    "dem_curv": "1/м", "dem_tpi_500": "м", "dem_tpi_2km": "м",
    "dem_relief_1km": "м", "dem_incision": "м",
}


def main() -> None:
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    print(f"Сетка: {meta.prf}x{meta.pic}, шаг {meta.dx} м; "
          f"DEM перепроецируется в {config.TER_RES_M:.0f} м, поле {config.TER_PAD_PX} px")

    full = terrain_v2.terrain_features(meta, keep_dropped=True)
    ter = full[list(terrain_v2.TERRAIN_COLS)]
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    valid = cell_mask.build_valid_mask(df)
    assert len(ter) == len(df), f"длина {len(ter)} != датасета {len(df)}"

    print(f"\nСтатистика по {int(valid.sum())} валидным ячейкам:")
    print(f"{'признак':<16}{'ед.':<6}{'NaN':>6}{'min':>10}{'медиана':>10}{'max':>10}")
    for c in full.columns:
        v = full[c].to_numpy()[valid]
        print(f"{c:<16}{UNITS[c]:<6}{np.isnan(v).sum():>6}"
              f"{np.nanmin(v):>10.3g}{np.nanmedian(v):>10.3g}{np.nanmax(v):>10.3g}")

    # --- Сверка посадки на сетку: новая высота против старого relief_m ---
    if "relief_m" in df.columns:
        old = df["relief_m"].to_numpy()[valid]
        new = ter["dem_elev"].to_numpy()[valid]
        ok = np.isfinite(old) & np.isfinite(new)
        r = np.corrcoef(old[ok], new[ok])[0, 1]
        bias = float(np.median(new[ok] - old[ok]))
        print(f"\nСверка с v1: corr(dem_elev, relief_m) = {r:.4f} по {ok.sum()} ячейкам, "
              f"медианное смещение {bias:+.1f} м")
        if r < 0.95:
            print("  ВНИМАНИЕ: корреляция низкая — возможен сдвиг растра относительно сетки")

    # --- Не дублируют ли новые признаки друг друга ---
    def corr_report(tbl: pd.DataFrame, title: str) -> float:
        M = tbl.to_numpy()[valid]
        M = M[np.isfinite(M).all(axis=1)]
        C = np.corrcoef((M - M.mean(0)) / M.std(0), rowvar=False)
        cond = float(np.linalg.cond(C))
        names = list(tbl.columns)
        print(f"\n{title}: число обусловленности {cond:.3g}")
        iu = np.triu_indices_from(C, k=1)
        for w in np.argsort(-np.abs(C[iu]))[:5]:
            i, j = iu[0][w], iu[1][w]
            print(f"  {names[i]:<16} ~ {names[j]:<16} r = {C[i, j]:+.2f}")
        if cond > 1e3:
            print("  ВНИМАНИЕ: обусловленность высокая — проверить алгебраические зависимости")
        return cond

    corr_report(full, "Все посчитанные производные")
    corr_report(ter, f"Пул после отбраковки {', '.join(terrain_v2.TERRAIN_DROPPED)}")

    out_path = config.PROCESSED_DIR / "terrain_v2.parquet"
    ter.to_parquet(out_path, index=False)

    # --- Превью ---
    shape = (meta.prf, meta.pic)
    mask2d = valid.to_numpy().reshape(shape) if hasattr(valid, "to_numpy") else valid.reshape(shape)
    ncol = 4
    nrow = int(np.ceil(len(terrain_v2.TERRAIN_COLS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.7 * nrow))
    for ax, c in zip(axes.ravel(), terrain_v2.TERRAIN_COLS):
        img = np.where(mask2d, ter[c].to_numpy().reshape(shape), np.nan)
        lo, hi = np.nanpercentile(img, [2, 98])
        im = ax.imshow(img, cmap="terrain" if c == "dem_elev" else "viridis",
                       vmin=lo, vmax=hi)
        ax.set_title(f"{c}, {UNITS[c]}", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.ravel()[len(terrain_v2.TERRAIN_COLS):]:
        ax.axis("off")
    fig.suptitle("Производные рельефа v2 (Copernicus DEM GLO-30 в проекции сетки, "
                 f"{config.TER_RES_M:.0f} м -> ячейки {meta.dx:.0f} м)", fontsize=13)
    fig.tight_layout()
    out_png = ROOT / "outputs" / "terrain_v2_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\nСохранено: {out_path}, {out_png}")


if __name__ == "__main__":
    main()
