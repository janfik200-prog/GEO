"""Сборка признаков dataset_v5_rebuilt на широкой сетке смежной территории.

Цель — не переобучение, а применение уже обученного ансамбля за пределами
листа `prognoz.pgrid`: нужен ровно тот же набор из 138 признаков (135 без
циркулярных `dist_dnl`/`dist_dnara` не входят в стоп-лист, поэтому остаются
в X), что и в `dataset_v5_rebuilt.parquet` (см.
`src/criterial_target.py::training_features(dataset="v5_rebuilt")`).

Источники по группам:

- `ter`, `lin`, `relief2`, `s2` (+`s2raw`), `s1`, `psr`, `ast`, `l8` — уже
  скачаны и посчитаны на `WIDE_META` (`experiments/fetch_wide_area.py`,
  файлы `data/processed/*_wide.parquet`), здесь только читаются.
- `gm` (грав/магнитка, 15 из 17 слоёв) — билинейный ресемплинг
  `config.GRAVMAG_PGRID` на `WIDE_META`; сетка-источник (305x455) почти
  полностью покрывает `WIDE_META` (231x452) — щель ~946 м только на
  восточном крае.
- `pf` (трансформанты потенциальных полей, 12 из 16) — `src/potential_fields.py`,
  БПФ на нативной сетке `грав_маг.pgrid`, затем ресемплинг на `WIDE_META`;
  функция уже принимает произвольный `target_meta`, без изменений.
- `dist` (`dist_dnl`, `dist_dnara`) — дистанционные растры от рек (ТОПО
  `dnl.shp`/`dnara.shp`) на `WIDE_META`. Сами шейп-файлы охватывают только
  исходный лист, но это НЕ ограничение: `distance_raster` считает расстояние
  до ближайшей геометрии из любой точки сетки, включая точки далеко за
  пределами bbox исходного слоя — значения корректны на всей `WIDE_META`.
- `ls` (Landsat 7 ETM+ legacy, 6 из 7 каналов) — билинейный ресемплинг
  `config.LANDSAT_PGRID`. В отличие от грав/магнитки и рек, это РАСТР:
  `landsat_fragm.pgrid` (2778x2683, 30 м) покрывает только северную полосу
  `WIDE_META` (y от 7 841 050 до 7 921 540 из диапазона листа
  7 694 500-7 920 500) — южные ~2/3 широкой сетки получают NaN в `ls_*`.
  Это реальный пробел покрытия исходных данных, не ошибка сборки.
- `opt` (`ast_mgoh`, `s2_ireci`) — считаются напрямую из уже прочитанных
  `s2_wide`/`ast_wide` (`src/optical_derived.py`), без новых вычислений на
  сетке.

`geo`/`geo2` и остальные циркулярные признаки (`dist_tect1/2`, `dens_tect`,
`dist_magm`, `dens_magm`, `dist_struct`, `dist_facies`, `dist_paleo`) в выборку
не входят — они уже исключены из обучающего X (`config.V5_EXCLUDE_GROUPS`,
`config.CRIT_EXCLUDE_FACTOR_FEATURES`), поэтому покрытие соответствующих
шейп-файлов (`tect1/2`, `magm`, `struct`, `facies`, `paleo`, `svita_new`) на
широкой сетке не проверялось и не требуется для применения модели.

Выход: `data/processed/dataset_wide.parquet` — те же 135 столбцов-признаков,
что и `training_features(dataset="v5_rebuilt")`, плюс `row`, `col`, `x`, `y`.

Запуск из корня репозитория: ``python -m experiments.build_dataset_wide``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, criterial_target, integro_grid, optical_derived, potential_fields  # noqa: E402
from src.vector_features import distance_raster  # noqa: E402
from src.data_loader import load_layer  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

GM_COLS = ["gm_gr_1GFI_25", "gm_gr_1GX_25", "gm_gr_1GY_25", "gm_gr_1G_25", "gm_gr_2G_25",
           "gm_gr_all", "gm_gr_ost_35", "gm_mag_1GFI_25", "gm_mag_1GX_25", "gm_mag_1GY_25",
           "gm_mag_1G_25", "gm_mag_2G_25", "gm_mag_flt_35", "gm_mag_ost_35", "gm_mg_all"]
LS_COLS = ["ls_ch1", "ls_ch3", "ls_ch4", "ls_ch5", "ls_ch6", "ls_ch7"]
PF_COLS = ["pf_gm_corr", "pf_gr_asig", "pf_gr_tdr", "pf_gr_up1", "pf_gr_vz",
           "pf_mag_asig", "pf_mag_tdr", "pf_mag_up1", "pf_mag_up10", "pf_mag_up2",
           "pf_mag_up5", "pf_mag_vz"]
OPT_COLS = ["ast_mgoh", "s2_ireci"]
DIST_COLS = ["dist_dnl", "dist_dnara"]

WIDE_PARQUETS = ["terrain_v2", "lineament", "relief_extra", "s2", "s1", "psr", "ast", "l8"]


def main() -> None:
    target_meta = WIDE_META
    target_proj4 = integro_grid.read_grid_proj4(config.GRAVMAG_PGRID)
    print(f"Целевая сетка: {target_meta.prf}x{target_meta.pic} = "
          f"{target_meta.prf * target_meta.pic} ячеек, шаг {target_meta.dx:.0f} м")

    parts: dict[str, pd.DataFrame] = {}
    for name in WIDE_PARQUETS:
        parts[name] = pd.read_parquet(config.PROCESSED_DIR / f"{name}_wide.parquet")
        print(f"{name}_wide.parquet: {parts[name].shape}")

    print("грав/магнитка (gm)...", flush=True)
    gm_layers = integro_grid.to_common_grid(
        config.GRAVMAG_PGRID, target_meta, method="bilinear", prefix="gm_",
        angle_props=config.ANGLE_PROPS, proj4=target_proj4,
    )
    gm_df = pd.DataFrame({c: gm_layers[c].ravel() for c in GM_COLS})

    print("Landsat 7 legacy (ls)...", flush=True)
    ls_layers = integro_grid.to_common_grid(
        config.LANDSAT_PGRID, target_meta, method="average", prefix="ls_",
        nodata=config.LANDSAT_NODATA, proj4=target_proj4,
    )
    ls_df = pd.DataFrame({c: ls_layers[c].ravel() for c in LS_COLS})
    ls_cov = 100.0 * np.isfinite(ls_df["ls_ch1"]).mean()
    print(f"  покрытие ls_*: {ls_cov:.1f}% ячеек (легаси-фрагмент уже сеток шире)")

    print("трансформанты потенциальных полей (pf)...", flush=True)
    pf_df = potential_fields.potential_field_features(target_meta, proj4=target_proj4)[PF_COLS]

    print("дистанции до рек (dist_dnl, dist_dnara)...", flush=True)
    dist_layers = {}
    for role in ("dnl", "dnara"):
        gdf = load_layer(config.TOPO_SHP_DIR / f"{role}.shp")
        dist_layers[f"dist_{role}"] = distance_raster(target_meta, gdf.geometry.values).ravel()
    dist_df = pd.DataFrame(dist_layers)[DIST_COLS]

    print("оптические индексы (opt)...", flush=True)
    opt_src = pd.concat([
        parts["s2"][["s2_b04", "s2_b05", "s2_b06", "s2_b07", "s2_b8a"]],
        parts["ast"][["ast_b08", "ast_b09"]],
    ], axis=1)
    opt_df = optical_derived.optical_derived_features(opt_src)[OPT_COLS]

    x, y = target_meta.cell_centers()
    rows, cols = np.indices(target_meta.shape)
    base = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(),
                          "x": x.ravel(), "y": y.ravel()})

    needed = list(criterial_target.training_features(dataset="v5_rebuilt").columns)
    all_parts = pd.concat(
        [base, gm_df, ls_df, pf_df, dist_df, opt_df, *parts.values()], axis=1)
    missing = [c for c in needed if c not in all_parts.columns]
    if missing:
        raise RuntimeError(f"нет в собранных слоях: {missing}")
    out = all_parts[["row", "col", "x", "y"] + needed]

    out_path = config.PROCESSED_DIR / "dataset_wide.parquet"
    out.to_parquet(out_path, index=False)
    print(f"OK: {out_path} {out.shape}")

    nan_pct = out[needed].isna().mean().sort_values(ascending=False)
    print("Топ-10 признаков по доле NaN:")
    print(nan_pct.head(10).to_string())


if __name__ == "__main__":
    main()
