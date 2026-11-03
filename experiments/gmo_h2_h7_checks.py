"""Гипотезы Г2 и Г7 предметной модели — без точек с золотом.

Г2 (куда отнести критерий К18). Каолинит `ast_kaolin` — один из сильнейших
признаков проекта, но неясно, что он измеряет: коры выветривания (объект
О10, место накопления этапа II) или изменение пород при реактивации разломных
зон (О7, мобилизация этапа II).
  Записано до расчёта. «Коры»: на обнажённых ячейках каолинит выше у контура
  кор и на плоских водоразделах, а внутри разломных зон — не выше.
  «Изменение пород»: наоборот. Оба или ни одного — «не различимо».
  Учитываются свита (полевошпатовые песчаники сами дают каолинит при
  выветривании), обнажённость и высота.

Г7 (канал звена «сохранность», К28). Прогноз проекта наполовину держится на
высоте. Если высота отражает эрозионный срез, по ней предсказывается
стратиграфический уровень коренной породы на поверхности.
  Записано до расчёта. «Срез»: (а) локально — в понижениях рельефа выходят
  более древние породы, чем на окружающих водоразделах (связь положения на
  склоне `dem_tpi_2km` с уровнем положительна); (б) регионально — после
  вычета линейного наклона моноклинали по X и Y связь высоты с уровнем
  сохраняется. Нет ни (а), ни (б) — «не работает».

Входные данные: `data/processed/dataset_v5_rebuilt.parquet` (каналы и
обнажённость), `data/processed/svita_labeled.gpkg` (свиты),
`data/Gis-integro/shp_dbf/kory.shp` — только как заверка, в признаки не
входит (слой формулы ⚠).

Неопределённость — бутстрэп блоков 10 км. Результат — в консоли и
`outputs/13_генетическая_модель/gmo_h2_h7_checks.png`.

Запуск::

    python -X utf8 -m experiments.gmo_h2_h7_checks
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, mapstyle, outputs, spectral_mask  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
SVITY = ROOT / "data/processed/svita_labeled.gpkg"
CELL, SUB, BLOCK, N_BOOT = 500.0, 10, 10_000.0, 1000
rng = np.random.default_rng(2)

#: Стратиграфический уровень снизу вверх: чем больше, тем моложе.
LEVEL = {"архей и нижний протерозой": 0, "ильинская": 1, "бурдурская": 2,
         "лабастахская": 3, "усть-ильинская": 4, "котуйканская": 5,
         "юсмастахская": 6, "кембрий (немакит-далдынская, медвежинская)": 7}


def main_unit(df: pd.DataFrame) -> pd.Series:
    """Преобладающее коренное подразделение ячейки (без четвертичных и интрузий)."""
    from rasterio import features
    from rasterio.transform import from_origin
    sv = gpd.read_file(SVITY).to_crs(anabar_inset.SHEET_CRS)
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    x0 = (df.x - (df.col + 0.5) * CELL).iloc[0]
    ytop = (df.y + (df.row + 0.5) * CELL).iloc[0]
    T = from_origin(x0, ytop, CELL / SUB, CELL / SUB)
    fr = {}
    for name in LEVEL:
        g = sv.geometry[sv.svita == name]
        if len(g) == 0:
            continue
        r = features.rasterize(((p, 1) for p in g), out_shape=(nrow * SUB, ncol * SUB),
                               transform=T, fill=0, dtype="uint8")
        fr[name] = r.reshape(nrow, SUB, ncol, SUB).mean(axis=(1, 3))[df.row, df.col]
    f = pd.DataFrame(fr, index=df.index)
    main = f.idxmax(axis=1).where(f.max(axis=1) >= 0.5)
    return main


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]


def boot(fn, df: pd.DataFrame) -> np.ndarray:
    bid = (np.floor(df.x / BLOCK) * 1000 + np.floor(df.y / BLOCK)).to_numpy()
    ub = np.unique(bid)
    idx = [np.flatnonzero(bid == b) for b in ub]
    out = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[i] for i in rng.integers(0, len(ub), len(ub))])
        out.append(fn(df.iloc[take]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def h2(df: pd.DataFrame) -> pd.DataFrame:
    d = df[spectral_mask.exposure_mask(df) & df.ast_kaolin.notna() & df.unit.notna()].copy()
    d["kory"] = (d.dist_struct <= 250).astype(float)
    d["fault"] = ((d.dist_tect1 <= 0) | (d.dist_tect2 <= 0)).astype(float)
    flat = d.dem_slope <= d.dem_slope.median()
    d["flat_ws"] = (flat & (d.dem_tpi_2km > 0)).astype(float)
    units = sorted(d.unit.unique())
    names = ["kory", "flat_ws", "fault"]

    def fit(s):
        X = np.column_stack([s[names].to_numpy(), z(s[spectral_mask.BARE_COL]).fillna(0),
                             z(s.dem_elev).fillna(0)]
                            + [(s.unit == u).astype(float).to_numpy() for u in units[1:]])
        return ols(X, z(s.ast_kaolin).to_numpy())[:3]

    est = fit(d)
    lo, hi = boot(fit, d)
    print(f"\nГ2: обнажённых ячеек с ASTER {len(d)}; у кор {int(d.kory.sum())}, "
          f"плоский водораздел {int(d.flat_ws.sum())}, в разломной зоне {int(d.fault.sum())}")
    print("   без поправок (разность средних каолинита, в стандартных отклонениях):")
    for n in names:
        k = z(d.ast_kaolin)
        print(f"     {n}: {k[d[n] == 1].mean() - k[d[n] == 0].mean():+.2f}")
    print("   по свитам, средний каолинит (ст. откл.):",
          z(d.ast_kaolin).groupby(d.unit).mean().round(2).to_dict())
    res = pd.DataFrame({"фактор": ["у контура кор", "плоский водораздел", "разломная зона"],
                        "эффект": est.round(3), "95% от": lo.round(3), "95% до": hi.round(3)})
    print("   с поправкой на свиту, обнажённость и высоту:")
    print(res.to_string(index=False))
    return res


def h7(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.unit.notna() & df.dem_elev.notna() & df.dem_tpi_2km.notna()].copy()
    d["level"] = d.unit.map(LEVEL).astype(float)
    xs, ys = z(d.x), z(d.y)

    def fit(s):
        X = np.column_stack([z(s.x), z(s.y), z(s.dem_elev), z(s.dem_tpi_2km)])
        return ols(X, s.level.to_numpy())[2:]

    def raw(s):
        return np.array([np.corrcoef(s.dem_elev, s.level)[0, 1],
                         np.corrcoef(s.dem_tpi_2km, s.level)[0, 1]])

    est, r = fit(d), raw(d)
    lo, hi = boot(fit, d)
    rlo, rhi = boot(raw, d)
    print(f"\nГ7: коренных ячеек {len(d)}; уровни: "
          f"{d.unit.value_counts().reindex(list(LEVEL)).dropna().astype(int).to_dict()}")
    print(f"   корреляция без поправок: высота {r[0]:+.2f} [{rlo[0]:+.2f}; {rhi[0]:+.2f}], "
          f"положение на склоне {r[1]:+.2f} [{rlo[1]:+.2f}; {rhi[1]:+.2f}]")
    res = pd.DataFrame({"фактор": ["высота", "положение на склоне (2 км)"],
                        "уровней на 1 ст. откл.": est.round(3),
                        "95% от": lo.round(3), "95% до": hi.round(3)})
    print("   после вычета наклона по X и Y:")
    print(res.to_string(index=False))
    print("   средняя высота по подразделениям, м:",
          d.groupby("unit").dem_elev.mean().reindex(list(LEVEL)).dropna().round(0).to_dict())
    return d


def draw(d7: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    order = [u for u in LEVEL if u in set(d7.unit)]
    data = [d7.dem_elev[d7.unit == u] for u in order]
    ax[0].boxplot(data, showfliers=False)
    ax[0].set_xticks(range(1, len(order) + 1), [u.split(" (")[0] for u in order],
                     rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("высота рельефа, м")
    ax[0].set_title("Высота по подразделениям (снизу вверх по разрезу)", fontsize=10)
    bins = np.quantile(d7.dem_tpi_2km, np.linspace(0, 1, 11))
    mid = d7.groupby(pd.cut(d7.dem_tpi_2km, bins, include_lowest=True), observed=True)
    ax[1].plot(mid["dem_tpi_2km"].mean(), mid["level"].mean(), "o-", color="#7b241c")
    ax[1].set_xlabel("положение на склоне в окне 2 км, м (меньше нуля — понижение)")
    ax[1].set_ylabel("стратиграфический уровень,\n0 — архей, 7 — кембрий")
    ax[1].set_title("Г7: уровень коренной породы и положение на рельефе", fontsize=10)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м: высота и эрозионный срез", fontsize=12)
    fig.tight_layout()
    out = outputs.png("gmo_h2_h7_checks.png")
    mapstyle.save_fig(fig, out, dpi=140)
    print(f"\nКарта: {out}")


def main() -> None:
    cols = ["row", "col", "x", "y", "ast_kaolin", "dist_struct", "dist_tect1", "dist_tect2",
            "dem_elev", "dem_slope", "dem_tpi_2km", spectral_mask.BARE_COL]
    df = pd.read_parquet(DATA, columns=cols)
    df["unit"] = main_unit(df)
    h2(df)
    draw(h7(df))


if __name__ == "__main__":
    main()
