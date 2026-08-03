"""Этап 2c: линеаменты по рельефу — проверка воспроизводимости и признаки.

Порядок и логика проверок:

1. **Стоп-правило ветки** — карты плотности линий по чётным и нечётным
   азимутам отмывки сравниваются по Spearman. Низкое согласие означает, что
   «линеаменты» рисует освещение, а не структура; тогда ветка закрывается и в
   датасет v2 идёт только плотность краёв.
2. **Внешняя сверка** — расстояние до линеамента против расстояния до разломов,
   вынесенных геологом на карту (``dist_tect1/tect2`` из v1). Совпадение не
   обязано быть полным (автоматика видит и то, чего нет на карте), но полная
   независимость означала бы, что линии — шум.
3. Статистика признаков и превью: отрезки поверх отмывки + карты признаков.

Выход: ``data/processed/lineament_features.parquet``,
``outputs/lineaments_preview.png``.
Запуск из корня: ``python -m experiments.lineaments_run``.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src import cell_mask, config, integro_grid, lineaments, terrain_v2  # noqa: E402

UNITS = {"lin_dens": "км/км²", "lin_node_dens": "усл. ед.", "lin_dist": "м",
         "lin_aniso_r": "0..1", "lin_dir_sin": "-1..1", "lin_dir_cos": "-1..1"}


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    elev = terrain_v2.projected_dem(meta)
    elev = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    log(f"DEM в проекции сетки: {elev.shape}, шаг {config.TER_RES_M:.0f} м")

    # --- 1. Стоп-правило: воспроизводимость между половинами азимутов ---
    rep = lineaments.azimuth_reproducibility(elev)
    rho = rep["rho"]
    verdict = ("ВЕТКА ОТКРЫТА" if rho >= config.LIN_MIN_SPEARMAN
               else f"ВЕТКА ЗАКРЫВАЕТСЯ (порог {config.LIN_MIN_SPEARMAN})")
    log(f"воспроизводимость по азимутам: Spearman = {rho:.3f} "
        f"(площадь с линиями {rep['covered_frac']:.1%}, из неё общей "
        f"{rep['overlap_frac']:.1%}) -> {verdict}")

    segments = lineaments.extract_lines(lineaments.edge_map(elev))
    lens_km = [np.hypot(x1 - x0, y1 - y0) * config.TER_RES_M / 1000.0
               for (x0, y0), (x1, y1) in segments]
    log(f"отрезков: {len(segments)}, длина {np.min(lens_km):.1f}..{np.max(lens_km):.1f} км, "
        f"медиана {np.median(lens_km):.1f} км, суммарно {np.sum(lens_km):.0f} км")

    feats = lineaments.lineament_features(meta, elev=elev)
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    valid = cell_mask.build_valid_mask(df)
    assert len(feats) == len(df), f"длина {len(feats)} != датасета {len(df)}"

    print(f"\nСтатистика по {int(valid.sum())} валидным ячейкам:")
    print(f"{'признак':<16}{'ед.':<10}{'min':>10}{'медиана':>10}{'max':>10}{'нулей, %':>10}")
    for c in lineaments.LINEAMENT_COLS:
        v = feats[c].to_numpy()[valid]
        print(f"{c:<16}{UNITS[c]:<10}{np.nanmin(v):>10.3g}{np.nanmedian(v):>10.3g}"
              f"{np.nanmax(v):>10.3g}{100.0 * (v == 0).mean():>10.1f}")

    # --- 2. Внешняя сверка с разломами геолога ---
    print("\nСверка с тектоникой из v1 (линии автоматики против карты геолога):")
    a = feats["lin_dist"].to_numpy()[valid]
    for col in ("dist_tect1", "dist_tect2", "dens_tect", "dist_magm"):
        if col not in df.columns:
            continue
        b = df[col].to_numpy()[valid]
        ok = np.isfinite(a) & np.isfinite(b)
        rho_c = spearmanr(a[ok], b[ok]).statistic
        print(f"  lin_dist ~ {col:<11} Spearman = {rho_c:+.3f}")
    dens = feats["lin_dens"].to_numpy()[valid]
    for col in ("dens_tect", "dens_magm"):
        if col in df.columns:
            b = df[col].to_numpy()[valid]
            ok = np.isfinite(dens) & np.isfinite(b)
            print(f"  lin_dens ~ {col:<11} Spearman = "
                  f"{spearmanr(dens[ok], b[ok]).statistic:+.3f}")

    # --- 3. Не пересказывают ли линеаменты просто расчленённость рельефа ---
    ter_path = config.PROCESSED_DIR / "terrain_v2.parquet"
    if ter_path.exists():
        from src import terrain_v2 as tv2

        ter = pd.read_parquet(ter_path)
        print("\nСвязь с рельефом (края отмывки — это в первую очередь бровки долин):")
        for lc in lineaments.LINEAMENT_COLS:
            x = feats[lc].to_numpy()[valid]
            best = max(
                ((abs(spearmanr(x[np.isfinite(x) & np.isfinite(ter[tc].to_numpy()[valid])],
                                ter[tc].to_numpy()[valid][np.isfinite(x) & np.isfinite(
                                    ter[tc].to_numpy()[valid])]).statistic), tc)
                 for tc in tv2.TERRAIN_COLS), key=lambda t: t[0])
            flag = "   <-- дублирует рельеф" if best[0] > 0.7 else ""
            print(f"  {lc:<14} сильнейшая связь: {best[1]:<15} |rho| = {best[0]:.2f}{flag}")

    out_path = config.PROCESSED_DIR / "lineament_features.parquet"
    feats.to_parquet(out_path, index=False)

    # --- 3. Превью ---
    shape = (meta.prf, meta.pic)
    mask2d = np.asarray(valid).reshape(shape)
    fig = plt.figure(figsize=(20, 11))
    ax0 = fig.add_subplot(2, 4, 1)
    ax0.imshow(lineaments.hillshade(elev, 315.0), cmap="gray")
    ax0.add_collection(LineCollection([[(x0, y0), (x1, y1)]
                                       for (x0, y0), (x1, y1) in segments],
                                      colors="red", linewidths=0.6))
    ax0.set_title(f"отрезки ({len(segments)}) поверх отмывки 315°", fontsize=10)
    ax0.set_xticks([]), ax0.set_yticks([])
    for i, c in enumerate(lineaments.LINEAMENT_COLS, start=2):
        ax = fig.add_subplot(2, 4, i)
        img = np.where(mask2d, feats[c].to_numpy().reshape(shape), np.nan)
        lo, hi = np.nanpercentile(img, [2, 98])
        im = ax.imshow(img, cmap="magma", vmin=lo, vmax=hi)
        ax.set_title(f"{c}, {UNITS[c]}", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Линеаменты по рельефу: {len(config.LIN_AZIMUTHS)} азимутов отмывки, "
                 f"воспроизводимость по половинам азимутов Spearman = {rho:.2f}",
                 fontsize=14)
    fig.tight_layout()
    out_png = ROOT / "outputs" / "lineaments_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    log(f"сохранено: {out_path}, {out_png}")


if __name__ == "__main__":
    main()
