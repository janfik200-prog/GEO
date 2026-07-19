"""Сборка датасета v1 на общей сетке (золото, лист R-48-XI,XII).

Целевая сетка — сетка критериального расчёта ``prognoz.pgrid`` (500 м, 154x149).
Слои:

- гравика/магнитка (17 гридов, 500 м) — билинейно; поля направлений
  (``config.ANGLE_PROPS``) — через sin/cos;
- Landsat 7 ETM+ (7 каналов, 30 м) — агрегация ``average``, DN=0 -> NaN;
- рельеф ``topo5_new`` (100 м) — ``average``, пересчёт в метры (x0.2);
- векторные слои: дистанционные растры (реки dnl/dnara из ТОПО; разломы, дайки,
  коры, фации, палеодолины из shp_dbf) и плотностные (разломы — длина,
  дайки — площадь, радиус ``config.DENSITY_RADIUS``);
- маска территории по свитам (``svita_new``).

Прямые признаки минерагенической карты (геохимические ореолы/опробование,
привнос урана, точки рудопроявлений) в датасет НЕ включаются — стоп-лист
постановки, используются только для заверки результата.

Выход (в ``data/processed/``): ``dataset_v1.npz`` (стек слоёв + геометрия сетки),
``dataset_v1.parquet`` (пиксель x признаки), ``dataset_v1_preview.png``,
``dataset_v1_sources.md`` (документация источников и контроль качества).

Запуск из корня репозитория: ``python experiments/build_dataset_v1.py``.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_loader import load_layer, read_sidecar_proj4  # noqa: E402
from src.integro_grid import read_grid_proj4, read_pgrid, to_common_grid  # noqa: E402
from src.vector_features import density_raster, distance_raster  # noqa: E402


def check_crs(name: str, proj4: str | None, target_proj4: str, report: list[str]) -> None:
    """Сверить CRS слоя с целевой; расхождение — в отчёт (не фатально, но видно)."""
    from pyproj import CRS

    if proj4 is None:
        report.append(f"- ВНИМАНИЕ: {name}: sidecar .pj4 не найден, CRS принята на веру")
    elif not CRS.from_proj4(proj4).equals(CRS.from_proj4(target_proj4)):
        report.append(f"- ВНИМАНИЕ: {name}: CRS отличается от целевой: `{proj4}`")
    else:
        report.append(f"- {name}: CRS совпадает с целевой")


def main() -> None:
    report: list[str] = []
    target_meta = read_pgrid(config.GOLD_TARGET_PGRID)
    target_proj4 = read_grid_proj4(config.GOLD_TARGET_PGRID)
    print(f"Целевая сетка: {target_meta.prf}x{target_meta.pic}, шаг {target_meta.dx} м")
    report.append(f"Целевая сетка: `{config.GOLD_TARGET_PGRID.name}` "
                  f"{target_meta.prf}x{target_meta.pic}, шаг {target_meta.dx} м, CRS `{target_proj4}`")
    report.append("")
    report.append("## Контроль CRS")

    layers: dict[str, np.ndarray] = {}

    # --- Растровые источники ---
    check_crs("грав_маг.pgrid", read_grid_proj4(config.GRAVMAG_PGRID), target_proj4, report)
    layers.update(to_common_grid(
        config.GRAVMAG_PGRID, target_meta, method="bilinear", prefix="gm_",
        angle_props=config.ANGLE_PROPS, proj4=target_proj4,
    ))
    print(f"грав/маг: {sum(k.startswith('gm_') for k in layers)} слоёв")

    check_crs("landsat_fragm.pgrid", read_grid_proj4(config.LANDSAT_PGRID), target_proj4, report)
    layers.update(to_common_grid(
        config.LANDSAT_PGRID, target_meta, method="average", prefix="ls_",
        nodata=config.LANDSAT_NODATA, proj4=target_proj4,
    ))
    print(f"landsat: {sum(k.startswith('ls_') for k in layers)} слоёв")

    check_crs("topo5_new.pgrid", read_grid_proj4(config.TOPO5_PGRID), target_proj4, report)
    topo = to_common_grid(
        config.TOPO5_PGRID, target_meta, method="average",
        scale=config.TOPO5_TO_METERS, proj4=target_proj4,
    )
    layers["relief_m"] = topo["prop0"]
    print("рельеф: relief_m")

    # --- Векторные источники ---
    vector_specs = [
        # (имя слоя, путь, дистанционный, плотностной measure|None)
        ("dnl", config.TOPO_SHP_DIR / "dnl.shp", "dist_dnl", None),
        ("dnara", config.TOPO_SHP_DIR / "dnara.shp", "dist_dnara", None),
        ("tect1", None, "dist_tect1", "length"),
        ("tect2", None, "dist_tect2", "length"),
        ("magm", None, "dist_magm", "area"),
        ("struct", None, "dist_struct", None),
        ("facies", None, "dist_facies", None),
        ("paleo", None, "dist_paleo", None),
    ]
    shp_dir = config.GOLD_TARGET_PGRID.parents[1] / config.SHP_SUBDIR
    density_geoms: dict[str, list] = {"length": [], "area": []}
    for role, path, dist_name, dens_measure in vector_specs:
        if path is None:
            path = shp_dir / f"{config.LAYER_FILES[role]}.shp"
        gdf = load_layer(path)
        check_crs(path.name, read_sidecar_proj4(path), target_proj4, report)
        layers[dist_name] = distance_raster(target_meta, gdf.geometry.values)
        if dens_measure:
            density_geoms[dens_measure].extend(gdf.geometry.values)
        print(f"{dist_name}: {len(gdf)} геометрий ({', '.join(sorted(set(gdf.geom_type)))})")

    # Плотности: разломы (обе системы, длина) и дайки (площадь).
    layers["dens_tect"] = density_raster(
        target_meta, density_geoms["length"], config.DENSITY_RADIUS, "length")
    layers["dens_magm"] = density_raster(
        target_meta, density_geoms["area"], config.DENSITY_RADIUS, "area")
    print("плотности: dens_tect, dens_magm")

    # Маска территории по свитам (1 = ячейка внутри полигонов svita_new).
    mask_gdf = load_layer(shp_dir / f"{config.LAYER_FILES['mask']}.shp")
    layers["mask_svita"] = (distance_raster(target_meta, mask_gdf.geometry.values) == 0
                            ).astype(np.uint8)
    print(f"mask_svita: {int(layers['mask_svita'].sum())} ячеек из {target_meta.obj_count}")

    # --- Контроль качества ---
    report.append("")
    report.append("## Слои и качество")
    report.append("| Слой | min | max | NaN, % |")
    report.append("|---|---|---|---|")
    for name, arr in layers.items():
        assert arr.shape == target_meta.shape, f"{name}: форма {arr.shape} != {target_meta.shape}"
        nan_pct = 100.0 * np.isnan(arr.astype(np.float64)).mean()
        report.append(f"| {name} | {np.nanmin(arr):.3g} | {np.nanmax(arr):.3g} | {nan_pct:.1f} |")

    # --- Сохранение ---
    out_dir = config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset_v1.npz",
        **layers,
        _x0=target_meta.x0, _y0=target_meta.y0, _dx=target_meta.dx, _dy=target_meta.dy,
        _pic=target_meta.pic, _prf=target_meta.prf, _proj4=np.str_(target_proj4 or ""),
    )
    print(f"OK: {out_dir / 'dataset_v1.npz'} ({len(layers)} слоёв)")

    # Таблица пиксель x признаки (parquet, если доступен pyarrow).
    import pandas as pd
    x, y = target_meta.cell_centers()
    rows, cols = np.indices(target_meta.shape)
    table = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(),
                          "x": x.ravel(), "y": y.ravel()})
    for name, arr in layers.items():
        table[name] = arr.ravel()
    try:
        table.to_parquet(out_dir / "dataset_v1.parquet", index=False)
        print(f"OK: {out_dir / 'dataset_v1.parquet'} {table.shape}")
    except ImportError:
        table.to_csv(out_dir / "dataset_v1.csv.gz", index=False)
        print("pyarrow не найден — таблица сохранена как dataset_v1.csv.gz")

    # Превью всех слоёв.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(layers)
    ncols = 6
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        im = ax.imshow(layers[name], origin="upper", cmap="viridis")
        ax.set_title(name, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Датасет v1: все слои на общей сетке 500 м (строка 0 — север)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "dataset_v1_preview.png", dpi=110)
    print(f"OK: {out_dir / 'dataset_v1_preview.png'}")

    # Документация источников.
    doc = [
        "# Датасет v1 — источники и контроль (сборка: experiments/build_dataset_v1.py)",
        "",
        "Стоп-лист (в датасет не входят, только заверка): геохимические ореолы,",
        "геохимическое опробование, привнос урана, точки рудопроявлений.",
        "",
        "Landsat: DN=0 трактуется как NoData (фон повёрнутой сцены); в редких",
        "случаях DN=0 может быть валидным тёмным пикселем — принято осознанно.",
        "Рельеф relief_m = topo5_new * 0.2 (единицы источника — метры x5).",
        "",
        *report,
    ]
    (out_dir / "dataset_v1_sources.md").write_text("\n".join(doc), encoding="utf-8")
    print(f"OK: {out_dir / 'dataset_v1_sources.md'}")


if __name__ == "__main__":
    main()
