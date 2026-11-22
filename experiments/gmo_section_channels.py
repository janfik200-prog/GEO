"""Каналы предметной модели из карты распределения свит: складки, базальный горизонт, несогласное залегание.

Цель. В пробной карте v0.3 место накопления видно лишь на 36 % листа: из
мест ответа ВНИГНИ 1 своим каналом наблюдаются только узлы линеаментов,
согласованность линеаментов (К21) снята. Карта распределения свит
(`experiments/gmo_svity_section.py`) даёт то, чего у линеаментов нет, —
форму залегания. Здесь из неё считаются три кандидата, каждый — гипотеза,
записанная до расчёта:

* **О9 складки, К21 по форме кровли фундамента.** При согласном залегании
  изгиб у всех границ один, поэтому срез 500 м к складкам не применяется;
  краевая полоса 4 км от невосстановленных ячеек отброшена. Антиклинальный перегиб —
  место, где поверхность кровли фундамента выпукла вверх: отрицательный
  лапласиан высоты кровли (сглаженной по окну `SIGMA_M`). Канал
  `fold_convex` — выпуклость, м на км² (больше — антиформа). Направление
  «замок или крыло» не решено (вопрос к ВНИГНИ и к литературе), поэтому
  отдельно — `fold_slope`, крутизна крыла, градусы.
* **О13 грубообломочные породы, базальный горизонт ильинской свиты.** По
  литературе (Оленёкское поднятие, записка Г9, 7.4) золото — в базальных
  конгломератах нижнего рифея. Базальный горизонт лежит на кровле
  фундамента, поэтому канал — глубина подошвы ильинской там, где она есть
  под ячейкой и не глубже 500 м (`il_base_depth`), на выходах
  ильинской — 0. **Это та же поверхность, что у О2 (несогласие, К7)**:
  ожидание, записанное до расчёта, — каналы почти совпадают (ρ > 0.8) на
  общей площади, и тогда О13 и О2 нельзя вести как два независимых
  объекта (правило по образцу Ог1: одна поверхность — одно звено).
* **О3 локальные несогласия, кандидат.** Там, где выходы свит не
  укладываются в согласное пологое залегание (`misfit_m`, область
  «разрез не восстанавливается»), возможны три причины: разлом со
  смещением, местное несогласие, изменение мощности. Канал
  `discord_m` — среднее несогласие рельефа с моделью, м; он не отличает
  причины и годится только как кандидат с пометкой.

Проверка независимости (как для линеаментов): связь каждого канала с
расчленённостью рельефа (`relief2_tri_std`) и с высотой; доля дисперсии,
объяснённая свитой на поверхности (η²). Карта ВНИГНИ, пробы и ореолы
золота не используются.

Оговорка, общая для всех трёх: поверхность кровли подобрана сплайном с 10
узлами по контактам геологической карты, а контакты пологих толщ проведены
по рельефу. Складки здесь — перегибы масштаба 10–20 км, мелкие складки и
кливаж не видны.

Выход: `data/processed/section_channels.parquet` (row, col, fold_convex,
fold_slope, il_base_depth, discord_m), рисунок
`outputs/13_генетическая_модель/gmo_section_channels.png`.

Запуск::

    python -X utf8 -m experiments.gmo_section_channels
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mapstyle, outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GRID = ROOT / "data/processed/svity_section_grid.parquet"
DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
OUT = ROOT / "data/processed/section_channels.parquet"
CELL, SIGMA_M, MAX_DEPTH_M = 500.0, 2000.0, 500.0


def to_img(g, col, shape):
    a = np.full(shape, np.nan)
    a[g.row, g.col] = g[col].astype(float)
    return a


def eta2(v: pd.Series, grp: pd.Series) -> float:
    ok = v.notna() & grp.notna()
    r = v[ok].rank()
    return float(r.groupby(grp[ok]).transform("mean").var() / r.var())


def main() -> None:
    g = pd.read_parquet(GRID)
    shape = (int(g.row.max()) + 1, int(g.col.max()) + 1)
    ok = g.reliab == 1
    z = to_img(g.assign(z=g.basement_elev.where(ok)), "z", shape)
    known = np.isfinite(z)
    _, (ir, ic) = ndimage.distance_transform_edt(~known, return_indices=True)
    zf = ndimage.gaussian_filter(z[ir, ic], SIGMA_M / CELL)
    lap = ndimage.laplace(zf) / (CELL / 1000.0) ** 2          # м на км²
    gy, gx = np.gradient(zf, CELL)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # Свиты лежат согласно и мощности постоянны: все границы — кровля плюс
    # постоянная, изгиб у них один. Поэтому складки видны на любой глубине,
    # и срез 500 м к ним не применяется. Краевая полоса 2·SIGMA от
    # невосстановленных ячеек и рамки отброшена — там изгиб создаёт
    # заполнение пропусков, а не данные.
    pad = np.pad(known, 1, constant_values=False)
    edge = ndimage.distance_transform_edt(pad)[1:-1, 1:-1] * CELL
    inside = known & (edge > 2 * SIGMA_M)
    out = g[["row", "col"]].copy()
    out["fold_convex"] = np.where(inside, -lap, np.nan)[g.row, g.col]
    out["fold_slope"] = np.where(inside, slope, np.nan)[g.row, g.col]
    il = g.il_present & (g.il_bot_depth <= MAX_DEPTH_M)
    out["il_base_depth"] = g.il_bot_depth.where(il)
    out["discord_m"] = g.misfit_m.where(g.reliab > 0)
    out.to_parquet(OUT, index=False)

    d = g[["row", "col", "surf_map", "basement_depth"]].merge(out, on=["row", "col"])
    d = d.merge(pd.read_parquet(DATA, columns=["row", "col", "relief2_tri_std", "dem_elev"]),
                on=["row", "col"], how="left")
    # К7 своим каналом — как в карте на знаниях: глубина кровли под рифеем
    print("Каналы (наблюдается, % ячеек | ρ с расчленённостью | ρ с высотой | η² свиты):")
    for c in ("fold_convex", "fold_slope", "il_base_depth", "discord_m"):
        s = d[c]
        print(f"   {c:14s} {100 * s.notna().mean():5.1f}% | "
              f"{spearmanr(s, d.relief2_tri_std, nan_policy='omit').statistic:+.2f} | "
              f"{spearmanr(s, d.dem_elev, nan_policy='omit').statistic:+.2f} | "
              f"{eta2(s, d.surf_map):.3f}")
    both = d.il_base_depth.notna() & d.basement_depth.notna()
    rho = spearmanr(d.il_base_depth[both], d.basement_depth[both]).statistic
    print(f"\nО13 против О2 (подошва ильинской против кровли фундамента) на общей площади "
          f"{both.sum()} ячеек: ρ = {rho:+.3f}; совпадают значения "
          f"{100 * np.mean(np.isclose(d.il_base_depth[both], d.basement_depth[both], atol=1)):.0f}%")
    fc = d.fold_convex
    print(f"\nВыпуклость кровли: 5–95% {fc.quantile(.05):+.2f}…{fc.quantile(.95):+.2f} м/км²; "
          f"антиформы (выпуклость > 0) {100 * (fc > 0).mean() / fc.notna().mean():.0f}% наблюдаемой площади; "
          f"крутизна крыла 5–95% {d.fold_slope.quantile(.05):.2f}…{d.fold_slope.quantile(.95):.2f}°")

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    panels = [("fold_convex", "О9: выпуклость кровли фундамента, м/км²\n(больше нуля — антиформа)", "RdBu_r"),
              ("il_base_depth", "О13: глубина подошвы ильинской свиты, м\n(базальный горизонт; до 500 м)", "viridis_r"),
              ("discord_m", "О3 (кандидат): несогласие выходов свит\nс пологим залеганием, м", "magma_r")]
    for a, (c, title, cmap) in zip(ax, panels):
        img = to_img(out, c, shape)
        if c == "fold_convex":
            v = np.nanquantile(np.abs(img), 0.95)
            im = a.imshow(img, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
        else:
            im = a.imshow(img, cmap=cmap, interpolation="nearest")
        a.set_title(title, fontsize=11)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м: каналы предметной модели из карты распределения свит "
                 "(север сверху; масштаб складок 10–20 км)", fontsize=12)
    fig.tight_layout()
    png = outputs.png("gmo_section_channels.png")
    mapstyle.save_fig(fig, png, dpi=120)
    print(f"\nРисунок: {png}")


if __name__ == "__main__":
    main()
