"""Линеаменты: поправка на «свой фон» (рельеф и свиты) и исправление полосчатости.

Цель. После очистки от долин (`experiments/gmo_lineaments_clean.py`, каналы
``linc_*``) оставшиеся линии собраны на уступах бурдурской свиты, а их
плотность по-прежнему следует за расчленённостью рельефа. Значит, О1
«разломная зона» (К1 узлы, К2 удалённость, К3 плотность) и О9 «складки»
(К21 согласованность) частично показывают выходы свиты и уступы, а не
разломы. Здесь проверяется, можно ли вычесть из каналов ожидаемый фон —
сколько линий дала бы ячейка просто из-за своей расчленённости и своей свиты.

Способы (все поверх ``linc_*``, то есть уже без долинных отрезков):

* **А «остаток»** — внутри каждой свиты (преобладающая свита ячейки,
  ``svk_main``) пуассоновская регрессия счёта пикселей линий (для узлов —
  их плотности) на логарифм расчленённости и его квадрат; канал — остаток
  Пирсона (наблюдено − ожидаемо) / √ожидаемо. Удалённость — остаток
  линейной регрессии log(удалённость + 100 м) на те же переменные.
* **Б «ранг в страте»** — перцентильный ранг внутри страты «свита × квинтиль
  расчленённости» (до 35 страт). Если в страте линии есть меньше чем в
  ``MIN_SEEN`` ячеек, канал там не определён (NaN): рельеф не позволяет
  увидеть линии, и ноль там — не «нет разлома», а «не наблюдается».
* **В «без контактов»** — по аналогии с долинами из линий выносятся отрезки
  вдоль контактов свит: больше половины длины ближе ``W_M`` к контакту и под
  углом к нему не больше ``ANGLE_DEG``. Контакты — границы полигонов
  `svita_labeled.gpkg` (все выделы, включая четвертичные и долериты; рамка
  карты не контакт), направление контакта — по структурному тензору границ.
* **Г «В + Б»** — отрезки вдоль контактов вынесены, остаток нормирован рангом
  в страте.

Записано до расчёта (критерии успеха, для плотности и узлов):

1. связь поправленного канала с расчленённостью (``relief2_tri_std``, ρ
   Спирмена по ячейкам листа) |ρ| < 0.3; контроль — то же с уклоном
   ``dem_slope`` (поправка не должна сработать только на одну меру рельефа);
2. бурдурская свита: отношение «доля бурдурской среди верхних 20 % ячеек
   канала / её доля площади листа» в пределах 0.8–1.25 (до поправки длина
   линий — 53 % на 28 % площади, отношение около 1.9); то же для полосы 1 км
   вдоль контактов бурдурской;
3. канал не повторяет контуры свит: доля дисперсии рангов канала,
   объяснённая преобладающей свитой (η²), < 0.05; отношение для полосы 1 км
   вдоль контактов любых свит 0.8–1.25; карта с контурами свит;
4. охранное условие: поправленный канал, посчитанный отдельно по чётным и
   нечётным азимутам отмывки, воспроизводится (ρ Спирмена ≥ 0.5, как
   стоп-правило ветки ``config.LIN_MIN_SPEARMAN``). Без него критерии 1–3
   выполнил бы и чистый шум.

Удалённость (К2) считается теми же способами, но критерии 1–3 к ней
прикладываются справочно. Карта ВНИГНИ, точки заверки, пробы и ореолы
золота в этой задаче не используются вовсе — поправка не подгоняется под
совпадение с ними.

Полосчатость (вторая задача). Горизонтальные полосы на западе листа в звене
«место накопления» пробной карты дают не ``lin_aniso_r`` (К21), а
``lin_node_dens`` (К1): ``scipy.ndimage.uniform_filter`` считает скользящую
сумму вычитанием и оставляет вдоль строки после каждой линии хвост
остатка округления ±1e-17 (52 % ячеек листа отрицательны, 3 % — крошечные
положительные вместо нуля). Свёртка по рангу превращает знак этого хвоста в
полосы. Исправление — `src.lineaments.line_rasters(..., exact_nodes=True)`.
У ``aniso_r`` свои дефекты (ячейка с одной линией получает 1, ячейка без
линий — 0 «хаос»); исправление — согласованность в окне 2.5 км, NaN там, где
линий в окне меньше 2 км (`src.lineaments.coherence_cells`). Прежние выходы
(`dataset_v5_rebuilt.parquet`, `lineament_clean.parquet`) не переписываются.

Результат (13.11.2026; числа печатает прогон, таблица —
`data/processed/lineament_bg.criteria.csv`):

* все критерии проходит только плотность способа Г (ρ с расчленённостью
  0.02, с уклоном 0.08; бурдурская 1.12, полоса у её контактов 1.09, у всех
  контактов 0.88; η² 0.026; половины 0.55). Узлы Г не проходят по η²
  (0.055 при пороге 0.05), у Б плотность упирается в полосу у контактов
  бурдурской (1.2505 при пороге 1.25) — оба промаха пограничные;
* цена Г: канал определён лишь на 50 % листа — в ровных стратах (нижние три
  дециля расчленённости) линий почти нет, и там О1 по линеаментам не
  наблюдается вовсе;
* сам по себе вынос отрезков вдоль контактов (В) почти ничего не даёт:
  правило ловит 11 % отрезков при 9 % у случайно повёрнутых, доля длины на
  бурдурской даже растёт (56 → 59 %) — линии сидят на уступах внутри свиты, а
  не на её границах;
* остаток регрессии (А) переворачивает зависимость (ρ −0.42): ровные места
  с нулём линий выходят выше расчленённых;
* К21: согласованность в окне свободна от фона (ρ −0.17, η² 0.04), но не
  воспроизводится по половинам азимутов ни при каком окне (0.32–0.37; в
  ячейке, как в v0.2, — 0.30) — ниже стоп-правила 0.5. Оговорка: проверка
  половинами для направленных каналов особенно строгая (каждая половина
  лучше видит края поперёк своих азимутов), но ``dir_sin`` в той же
  проверке даёт 0.48.

Выход: `data/processed/lineament_bg.parquet` — row, col и каналы:

* ``linf_node_dens``, ``lincf_node_dens`` — узлы без хвостов округления (все
  линии / без долин); ``linf_aniso_r``, ``lincf_aniso_r`` — согласованность в
  окне 2.5 км (NaN — линий мало);
* ``lincres_*`` (А), ``lincrank_*`` (Б), ``lincx_*`` (В), ``lincxrank_*`` (Г) —
  ``dens``, ``node_dens``, ``dist`` и ``aniso_r`` каждого способа;
* ``lincb_*`` — копия способа, прошедшего критерии (если такой есть);
* ``bg_relief_q`` — квинтиль расчленённости; ``bg_observed`` /
  ``bg_observed_x`` — страта способа Б / Г наблюдаема (вне её канал NaN).

Рисунки в `outputs/13_генетическая_модель/`: ``gmo_lineaments_bg_stripes.png``
(до/после полосчатости), ``gmo_lineaments_bg_methods.png`` (каналы способов
на карте с контурами свит), ``gmo_lineaments_bg_relief.png`` (связь с
расчленённостью и обогащение бурдурской).

Запуск::

    python -X utf8 -u -m experiments.gmo_lineaments_background
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gmo_lineaments_clean as glc  # noqa: E402
from src import anabar_inset, config, integro_grid, lineaments, mapstyle, outputs, terrain_v2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
LINC = ROOT / "data/processed/lineament_clean.parquet"
K23 = ROOT / "data/processed/k23_svity_grid.parquet"
SVITY = ROOT / "data/processed/svita_labeled.gpkg"
OUT = ROOT / "data/processed/lineament_bg.parquet"
RES = config.TER_RES_M
PAD = config.TER_PAD_PX
K = 5                                # пикселей 100 м в ячейке 500 м
W_M, ANGLE_DEG = 500.0, 30.0         # отрезок «вдоль контакта» — как «вдоль долины»
SENS_W = (300.0, 800.0)
BAND_M = 1000.0                      # полоса вдоль контактов для критериев 2–3
MIN_SEEN = 0.05                      # Б: страта наблюдаема, если линии есть в ≥ 5 % ячеек
TOP = 0.20
RELIEF, RELIEF_CTRL = "relief2_tri_std", "dem_slope"
BURDUR = "бурдурская"
METHODS = {"linc_": "без поправки (v0.2)", "lincres_": "А остаток регрессии",
           "lincrank_": "Б ранг в страте", "lincx_": "В без контактов",
           "lincxrank_": "Г без контактов + ранг"}
CHANNELS = (("dens", "↑"), ("node_dens", "↑"), ("dist", "↓"))
rng = np.random.default_rng(13)
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:5.0f} с] {msg}", flush=True)


# ---------- свиты на рабочем растре 100 м ----------
class Contacts:
    """Контакты свит в кадре рабочего растра: расстояние и ось ближайшего контакта.

    Интерфейс как у ``gmo_lineaments_clean.Valleys`` (``dist``, ``near_axis``),
    чтобы отрезки классифицировались той же функцией ``classify``.
    """

    def __init__(self, labels: np.ndarray, only: int | None = None):
        mapped = labels > 0
        edge = np.zeros_like(mapped)
        for axis in (0, 1):
            for shift in (1, -1):
                nb = np.roll(labels, shift, axis=axis)
                diff = (nb != labels) & mapped & (nb > 0)
                if only is not None:
                    diff &= (labels == only) | (nb == only)
                edge |= diff
        # roll заворачивает край массива — у рамки рабочего растра контактов нет
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = False
        self.mask = edge
        self.dist, idx = ndimage.distance_transform_edt(~edge, return_indices=True)
        self.dist = self.dist * RES
        # Ось контакта: структурный тензор индикаторов выделов. Нормаль к
        # контакту — главное направление градиента; ось — перпендикуляр.
        jxx = np.zeros(labels.shape)
        jyy = np.zeros(labels.shape)
        jxy = np.zeros(labels.shape)
        for c in np.unique(labels[mapped]):
            ind = ndimage.gaussian_filter((labels == c).astype(float), 1.5)
            gy, gx = np.gradient(ind)
            jxx += gx * gx
            jyy += gy * gy
            jxy += gx * gy
        jxx, jyy, jxy = (ndimage.gaussian_filter(a, 2.0) for a in (jxx, jyy, jxy))
        normal = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        axis = normal + np.pi / 2
        self.near_axis = axis[idx[0], idx[1]]


def svita_raster(meta, svity: gpd.GeoDataFrame) -> tuple[np.ndarray, list[str]]:
    from rasterio import features
    from rasterio.transform import from_origin

    names = sorted(svity.svita.dropna().unique())
    T = from_origin(meta.x0 - PAD * RES, meta.y_top + PAD * RES, RES, RES)
    shape = (meta.prf * K + 2 * PAD, meta.pic * K + 2 * PAD)
    lab = features.rasterize(((g, names.index(s) + 1) for g, s in zip(svity.geometry, svity.svita)
                              if g is not None and s in names),
                             out_shape=shape, transform=T, fill=0, dtype="int32")
    return lab, names


def cell_mean(a: np.ndarray, meta) -> np.ndarray:
    core = a[PAD:PAD + meta.prf * K, PAD:PAD + meta.pic * K]
    return core.reshape(meta.prf, K, meta.pic, K).mean(axis=(1, 3)).ravel()


# ---------- каналы по набору отрезков ----------
def channels(segs, shape, meta, prefix: str) -> pd.DataFrame:
    r = lineaments.line_rasters(segs, shape, RES, exact_nodes=True)
    f = lineaments.rasters_to_cell_features(meta, r, prefix, pad=PAD)
    f[f"{prefix}aniso_r"] = lineaments.coherence_cells(meta, r, pad=PAD)
    f[f"{prefix}npx"] = f[f"{prefix}dens"] * (meta.dx / 1000) ** 2 / (RES / 1000)
    return f[[f"{prefix}{c}" for c in ("dens", "node_dens", "dist", "aniso_r", "dir_sin", "npx")]]


# ---------- способы поправки ----------
def residual_method(df: pd.DataFrame, src: str, dst: str) -> None:
    """А: остаток регрессии внутри свиты на log расчленённости (+ квадрат)."""
    from sklearn.linear_model import LinearRegression, PoissonRegressor

    z = np.log(df[RELIEF].to_numpy())
    X = np.c_[z, z ** 2]
    for ch in ("npx", "node_dens", "dist"):
        out = np.full(len(df), np.nan)
        y = df[f"{src}{ch}"].to_numpy(float)
        for _, idx in df.groupby("svk_main").indices.items():
            if ch == "dist":
                yy = np.log(y[idx] + RES)
                m = LinearRegression().fit(X[idx], yy)
                out[idx] = yy - m.predict(X[idx])
            else:
                scale = 1.0 if ch == "npx" else 25.0     # узлы кратны 1/25 -> счёт
                yy = y[idx] * scale
                if yy.sum() == 0:
                    continue
                m = PoissonRegressor(alpha=1e-6, max_iter=1000).fit(X[idx], yy)
                mu = m.predict(X[idx])
                out[idx] = (yy - mu) / np.sqrt(mu)
        df[f"{dst}{'dens' if ch == 'npx' else ch}"] = out
    df[f"{dst}aniso_r"] = df[f"{src}aniso_r"]


def rank_method(df: pd.DataFrame, src: str, dst: str) -> None:
    """Б: перцентильный ранг внутри страты «свита × квинтиль расчленённости»."""
    strata = df.svk_main.astype(str) + "|" + df.bg_relief_q.astype(str)
    seen = (df[f"{src}dens"] > 0).groupby(strata).transform("mean")
    obs = seen >= MIN_SEEN
    df[f"{dst}observed"] = obs
    for ch in ("dens", "node_dens", "dist", "aniso_r"):
        r = df[f"{src}{ch}"].groupby(strata).rank(pct=True)
        df[f"{dst}{ch}"] = r.where(obs)


# ---------- критерии ----------
def top_mask(v: pd.Series, how: str) -> np.ndarray:
    """Верхние 20 % определённых ячеек; связки разбиваются случайно, не порядком строк."""
    s = (-v if how == "↓" else v).to_numpy(float)
    ok = np.isfinite(s)
    key = np.where(ok, s, -np.inf) + rng.uniform(0, 1e-9, len(s)) * ok
    order = np.argsort(-key)
    m = np.zeros(len(s), bool)
    m[order[:int(round(TOP * ok.sum()))]] = True
    return m


def eta2(v: pd.Series, groups: pd.Series) -> float:
    ok = v.notna()
    r = v[ok].rank(pct=True)
    g = groups[ok]
    between = (r.groupby(g).transform("mean") - r.mean()) ** 2
    return float(between.sum() / ((r - r.mean()) ** 2).sum())


def criteria(df: pd.DataFrame, col: str, how: str) -> dict:
    v = df[col]
    ok = v.notna()
    top = top_mask(v, how)
    sgn = -1 if how == "↓" else 1
    res = {"покрытие, %": 100 * ok.mean(),
           "ρ расчл.": sgn * spearmanr(v[ok], df.loc[ok, RELIEF]).statistic,
           "ρ уклон": sgn * spearmanr(v[ok], df.loc[ok, RELIEF_CTRL]).statistic}
    for key, frac in (("бурд.", "svk_br"), ("конт. бурд.", "band_br"), ("конт. всех", "band_all")):
        f = df[frac].fillna(0)
        res[f"{key} отн."] = float(f[top].mean() / f[ok].mean())
    res["η² свиты"] = eta2(sgn * v, df.svk_main)
    return res


def passes(r: dict, rep: float | None) -> bool:
    return (abs(r["ρ расчл."]) < 0.3 and abs(r["ρ уклон"]) < 0.3
            and 0.8 <= r["бурд. отн."] <= 1.25 and 0.8 <= r["конт. бурд. отн."] <= 1.25
            and 0.8 <= r["конт. всех отн."] <= 1.25 and r["η² свиты"] < 0.05
            and rep is not None and rep >= config.LIN_MIN_SPEARMAN)


# ---------- конвейер на одном наборе отрезков ----------
def build(segs, shape, meta, base: pd.DataFrame, valleys, contacts, w_m: float = W_M):
    """Все способы на одном наборе отрезков (весь набор или половина азимутов)."""
    is_valley = glc.classify(segs, valleys, glc.W_M)
    clean = [s for s, q in zip(segs, is_valley) if not q]
    # classify берёт угол из модуля долин; для контактов порог тот же (30°)
    is_contact = glc.classify(clean, contacts, w_m)
    nocont = [s for s, q in zip(clean, is_contact) if not q]
    df = base.copy()
    for pre, ss in (("lin_", segs), ("linc_", clean), ("lincx_", nocont)):
        f = channels(ss, shape, meta, pre)
        df = pd.concat([df, f.iloc[df.idx].reset_index(drop=True)], axis=1)
    residual_method(df, "linc_", "lincres_")
    rank_method(df, "linc_", "lincrank_")
    rank_method(df, "lincx_", "lincxrank_")
    return df, clean, nocont, is_contact


def coherence_halves(elev, valleys, meta) -> None:
    """К21: воспроизводимость согласованности по половинам азимутов при разных окнах."""
    az = list(config.LIN_AZIMUTHS)
    res = []
    for half in (az[::2], az[1::2]):
        s = lineaments.extract_lines(lineaments.edge_map(elev, azimuths=half, res_m=RES))
        s = [x for x, q in zip(s, glc.classify(s, valleys, glc.W_M)) if not q]
        r = lineaments.line_rasters(s, elev.shape, RES, exact_nodes=True)
        f = lineaments.rasters_to_cell_features(meta, r, "c_", pad=PAD)
        d = {"в ячейке (как v0.2, где есть линии)": f.c_aniso_r.where(f.c_dens > 0).to_numpy(),
             "направление dir_sin (для сравнения)": f.c_dir_sin.where(f.c_dens > 0).to_numpy()}
        for w, mp in ((3, 10), (5, 20), (9, 40)):
            d[f"окно {w * 0.5:g} км, ≥ {mp * RES / 1000:g} км линий"] = lineaments.coherence_cells(
                meta, r, pad=PAD, win_cells=w, min_px=mp)
        res.append(d)
    for k in res[0]:
        a, b = res[0][k], res[1][k]
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"   {k}: ячеек {ok.sum()}, ρ половин {spearmanr(a[ok], b[ok]).statistic:.2f}")


def length_shares(segs, shape, meta, br_px, band_px, mapped_px) -> tuple[float, float]:
    ln = lineaments.density_raster(segs, shape)
    core = (slice(PAD, PAD + meta.prf * K), slice(PAD, PAD + meta.pic * K))
    ln, br, band, mp = ln[core], br_px[core], band_px[core], mapped_px[core]
    tot = ln[mp].sum()
    return float(ln[br & mp].sum() / tot), float(ln[band & mp].sum() / tot)


# ---------- рисунки ----------
def to_grid(df, col, meta):
    a = np.full((meta.prf, meta.pic), np.nan)
    a[df.row, df.col] = df[col]
    return a


def extent(meta):
    return [meta.x0, meta.x0 + meta.pic * meta.dx, meta.y_top - meta.prf * meta.dy, meta.y_top]


def svita_outlines(ax, svity, lw_br=1.0):
    svity.boundary.plot(ax=ax, color="#555555", linewidth=0.25)
    br = svity[svity.svita == BURDUR].dissolve()
    br.boundary.plot(ax=ax, color="black", linewidth=lw_br)


def frame(ax, meta, title):
    e = extent(meta)
    ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X, км (Гаусс–Крюгер, осевой 105°)", fontsize=8)
    ax.set_ylabel("Y, км", fontsize=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.tick_params(labelsize=7)


def rank_map(ax, fig, df, col, how, meta, title, svity=None):
    v = df[col] if how != "↓" else -df[col]
    a = to_grid(df.assign(_r=v.rank(pct=True)), "_r", meta)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(a, extent=extent(meta), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    if svity is not None:
        svita_outlines(ax, svity)
    frame(ax, meta, title)
    fig.colorbar(im, ax=ax, shrink=0.75, label="доля 0–1 по рангу")


def draw_stripes(df, meta) -> Path:
    fig, axs = plt.subplots(2, 2, figsize=(13, 12.5))
    panels = (("linc_node_dens_old", "↑", "К1 узлы, канал v0.2 (linc_node_dens): до"),
              ("linc_node_dens", "↑", "К1 узлы без хвостов округления (lincf_node_dens): после"),
              ("linc_aniso_r_old", "↑", "К21 согласованность v0.2 (linc_aniso_r): до"),
              ("linc_aniso_r", "↑", "К21 согласованность в окне 2.5 км (lincf_aniso_r): после"))
    for ax, (col, how, ttl) in zip(axs.flat, panels):
        rank_map(ax, fig, df, col, how, meta, ttl)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#e6e6e6")],
               labels=["канал не определён (в окне меньше 2 км линий)"],
               loc="lower center", fontsize=9, frameon=False)
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м: каналы К1 и К21 так, как их видит пробная "
                 "карта (доля по рангу).\nПолосы на западе — хвост остатка округления "
                 "±1e-17 в узлах; у согласованности — 1 в ячейке с одной линией, 0 без линий",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = outputs.png("gmo_lineaments_bg_stripes.png")
    mapstyle.save_fig(fig, out, dpi=120)
    return out


def draw_methods(df, meta, svity) -> Path:
    cols = list(METHODS)
    fig, axs = plt.subplots(3, len(cols), figsize=(26, 17))
    names = {"dens": "К3 плотность", "node_dens": "К1 узлы", "dist": "К2 близость"}
    for i, (ch, how) in enumerate(CHANNELS):
        for j, m in enumerate(cols):
            rank_map(axs[i, j], fig, df, f"{m}{ch}", how, meta,
                     f"{names[ch]}: {METHODS[m]}", svity)
    fig.legend(handles=[Line2D([], [], color="black", lw=1.2),
                        Line2D([], [], color="#555555", lw=0.6),
                        plt.Rectangle((0, 0), 1, 1, color="#e6e6e6")],
               labels=["контур бурдурской свиты", "контуры остальных выделов карты",
                       "канал не определён (Б: в страте линий почти нет)"],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м: каналы разломной зоны О1 по линеаментам без "
                 "долин и способы поправки на свой фон (рельеф и свиты); близость — "
                 "ранг по убыванию расстояния", fontsize=14)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    out = outputs.png("gmo_lineaments_bg_methods.png")
    mapstyle.save_fig(fig, out, dpi=90)
    return out


def draw_relief(df, table: pd.DataFrame) -> Path:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    q = pd.qcut(df[RELIEF], 10, labels=False) + 1
    colors = dict(zip(METHODS, ("#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd")))
    for ax, ch in zip(axs[:2], ("dens", "node_dens")):
        for m, lab in METHODS.items():
            v = df[f"{m}{ch}"].rank(pct=True)
            ax.plot(range(1, 11), v.groupby(q).mean(), "o-", color=colors[m], label=lab)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlabel("дециль расчленённости рельефа (relief2_tri_std), 1 — самый ровный\n"
                      "(у Б и Г децили 1–3 не наблюдаемы: линий почти нет)")
        ax.set_ylabel("средний ранг канала в дециле, доля 0–1")
        ax.set_title({"dens": "К3 плотность линий", "node_dens": "К1 узлы"}[ch], fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
    ax = axs[2]
    t = table[table["канал"] == "dens"].set_index("способ")
    x = np.arange(len(t))
    for k, (col, lab) in enumerate((("бурд. отн.", "бурдурская свита"),
                                     ("конт. бурд. отн.", "полоса 1 км у контактов бурдурской"),
                                     ("конт. всех отн.", "полоса 1 км у контактов всех выделов"))):
        ax.bar(x + (k - 1) * 0.27, t[col], width=0.27, label=lab)
    ax.axhspan(0.8, 1.25, color="#2ca02c", alpha=0.12, label="критерий 0.8–1.25")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xticks(x, ["без\nпоправки", "А\nостаток", "Б\nранг", "В без\nконтактов",
                      "Г без конт.\n+ ранг"], fontsize=8)
    ax.set_ylim(0, 2.6)
    ax.set_ylabel("доля среди верхних 20 % ячеек / доля площади")
    ax.set_title("К3 плотность: обогащение верхних 20 % ячеек", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м: связь каналов разломной зоны с рельефом и "
                 "свитами до и после поправки", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = outputs.png("gmo_lineaments_bg_relief.png")
    mapstyle.save_fig(fig, out, dpi=120)
    return out


# ---------- главный ход ----------
def main() -> None:
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    raw = terrain_v2.projected_dem(meta)
    valid = np.isfinite(raw)
    elev = np.where(valid, raw, np.nanmedian(raw))
    shape = elev.shape
    log(f"рельеф {shape}, шаг {RES:.0f} м")

    acc, ang = glc.thalwegs(raw, valid)
    valleys = glc.Valleys(acc, ang, glc.A0_KM2)
    log("тальвеги посчитаны")

    svity = gpd.read_file(SVITY).to_crs(anabar_inset.SHEET_CRS)
    lab, names = svita_raster(meta, svity)
    contacts = Contacts(lab)
    br_id = names.index(BURDUR) + 1
    br_px = lab == br_id
    band_br = Contacts(lab, only=br_id).dist <= BAND_M
    band_all = contacts.dist <= BAND_M
    log(f"свиты на растре: выделов {len(names)}, контактов {contacts.mask.mean() * 100:.1f} % пикселей")

    base = pd.read_parquet(DATA, columns=["row", "col", RELIEF, RELIEF_CTRL, "lin_node_dens",
                                          "lin_aniso_r"])
    base = base.rename(columns={"lin_node_dens": "lin_node_dens_ds", "lin_aniso_r": "lin_aniso_r_ds"})
    k23 = pd.read_parquet(K23, columns=["row", "col", "svk_cover", "svk_br", "svk_main"])
    base = base.merge(k23, on=["row", "col"], how="left")
    old = pd.read_parquet(LINC, columns=["row", "col", "linc_dens", "linc_node_dens", "linc_aniso_r"])
    base = base.merge(old.rename(columns=lambda c: c + "_old" if c.startswith("linc_") else c),
                      on=["row", "col"], how="left")
    base["idx"] = base.row * meta.pic + base.col
    base["band_br"] = cell_mean(band_br.astype(float), meta)[base.idx]
    base["band_all"] = cell_mean(band_all.astype(float), meta)[base.idx]
    base["bg_relief_q"] = pd.qcut(base[RELIEF], 5, labels=False) + 1
    # Домен — как у пробной карты: ячейки, покрытые геологической картой хотя бы наполовину
    base = base[base.svk_cover.fillna(0) >= 0.5].reset_index(drop=True)

    segs = lineaments.extract_lines(lineaments.edge_map(elev, res_m=RES))
    log(f"отрезков {len(segs)}")
    df, clean, nocont, is_contact = build(segs, shape, meta, base, valleys, contacts)
    log(f"способы посчитаны: без долин {len(clean)}, из них вдоль контактов {is_contact.sum()} "
        f"({100 * is_contact.mean():.0f} %)")

    # Сверки пересчёта с прежними выходами
    rho = spearmanr(df.linc_dens, df.linc_dens_old).statistic
    same = np.allclose(df.linc_dens, df.linc_dens_old)
    log(f"linc_dens совпадает с lineament_clean.parquet: {same} (ρ = {rho:.3f})")
    lnd = df.linc_node_dens_old
    print(f"   хвосты округления в linc_node_dens (до): отрицательных {100 * (lnd < 0).mean():.1f} %, "
          f"0 < x < 1e-9: {100 * ((lnd > 0) & (lnd < 1e-9)).mean():.1f} %; после: "
          f"{100 * (df.linc_node_dens < 0).mean():.1f} % и "
          f"{100 * ((df.linc_node_dens > 0) & (df.linc_node_dens < 1e-9)).mean():.1f} %")
    print(f"   lin_node_dens: ρ(после, датасет) = "
          f"{spearmanr(df.lin_node_dens, df.lin_node_dens_ds).statistic:.3f}, "
          f"max |разница| = {np.abs(df.lin_node_dens - df.lin_node_dens_ds).max():.2e}")
    a_old = df.linc_aniso_r_old
    print(f"   linc_aniso_r (до): 0 у {100 * (a_old == 0).mean():.0f} % ячеек, 1 у "
          f"{100 * (a_old > 0.999).mean():.0f} %; в окне: NaN у "
          f"{100 * df.linc_aniso_r.isna().mean():.0f} %, 1 у {100 * (df.linc_aniso_r > 0.999).mean():.1f} %")

    # Нуль-модель для правила «вдоль контакта»: те же отрезки, повёрнутые случайно
    null = np.mean([glc.classify(glc.rotated(clean), contacts, W_M).mean() for _ in range(10)])
    print(f"   отрезков вдоль контактов {100 * is_contact.mean():.1f} %, при случайном повороте "
          f"{100 * null:.1f} %")
    mapped = lab > 0
    for nm, ss in (("все линии", segs), ("без долин", clean), ("без долин и контактов", nocont)):
        s_br, s_band = length_shares(ss, shape, meta, br_px, band_br, mapped)
        print(f"   {nm}: длина на бурдурской {100 * s_br:.0f} %, ближе 1 км к её контактам "
              f"{100 * s_band:.0f} %")
    core = (slice(PAD, PAD + meta.prf * K), slice(PAD, PAD + meta.pic * K))
    print(f"   площадь: бурдурская {100 * br_px[core][mapped[core]].mean():.0f} %, полоса 1 км у её "
          f"контактов {100 * band_br[core][mapped[core]].mean():.0f} %, у контактов всех выделов "
          f"{100 * band_all[core][mapped[core]].mean():.0f} %")

    # Охранное условие: воспроизводимость по половинам азимутов
    log("половины азимутов...")
    az = list(config.LIN_AZIMUTHS)
    halves = []
    for half in (az[::2], az[1::2]):
        hs = lineaments.extract_lines(lineaments.edge_map(elev, azimuths=half, res_m=RES))
        halves.append(build(hs, shape, meta, base, valleys, contacts)[0])
        log(f"   половина {half}: отрезков {len(hs)}")

    rows = []
    for m, lab_m in METHODS.items():
        for ch, how in CHANNELS:
            col = f"{m}{ch}"
            a, b = halves[0][col], halves[1][col]
            ok = a.notna() & b.notna()
            rep = float(spearmanr(a[ok], b[ok]).statistic)
            r = {"способ": lab_m, "канал": ch} | criteria(df, col, how) | {"воспр. половин": rep}
            r["проходит"] = passes(r, rep) if ch != "dist" else "справочно"
            rows.append(r)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nКритерии (домен — ячейки, покрытые картой ≥ 50 %; ρ для близости — со знаком "
          "«ближе = больше»):")
    print(table.round(2).to_string(index=False))

    # Согласованность (К21) — справочно: связь с рельефом и свитами до и после
    print("\nК21 согласованность (справочно):")
    for col in ("linc_aniso_r_old", "linc_aniso_r", "lincx_aniso_r"):
        src = col.replace("_old", "")
        a, b = halves[0][src], halves[1][src]
        ok = a.notna() & b.notna()
        rep = spearmanr(a[ok], b[ok]).statistic
        print(f"   {col}: " + ", ".join(f"{k} {v:.2f}" for k, v in criteria(df, col, "↑").items())
              + (f", воспр. половин {rep:.2f}" if col == src else ""))

    print("\nК21: воспроизводимость по половинам азимутов (линии без долин):")
    coherence_halves(elev, valleys, meta)

    # Чувствительность способа В к ширине полосы контакта
    print("\nЧувствительность «вдоль контакта» к ширине полосы:")
    for w in SENS_W:
        d, _, nc, isc = build(segs, shape, meta, base, valleys, contacts, w_m=w)
        r = criteria(d, "lincx_dens", "↑")
        r2 = criteria(d, "lincxrank_dens", "↑")
        print(f"   W = {w:.0f} м: вынесено {100 * isc.mean():.0f} %; В: ρ расчл. {r['ρ расчл.']:.2f}, "
              f"бурд. {r['бурд. отн.']:.2f}, η² {r['η² свиты']:.3f}; Г: ρ расчл. {r2['ρ расчл.']:.2f}, "
              f"бурд. {r2['бурд. отн.']:.2f}, η² {r2['η² свиты']:.3f}")

    # Выход. lincb_*: первый способ, у которого плотность прошла все критерии;
    # узлы того же способа пишутся, только если прошли и они. Близость — того
    # же способа (критерии к ней справочные), согласованность — канал в окне
    # по тому же набору отрезков (сам по себе проходит проверки фона).
    def ok(m, ch):
        sel = (table["способ"] == METHODS[m]) & (table["канал"] == ch)
        return bool(table.loc[sel, "проходит"].eq(True).all())

    good = [m for m in METHODS if m != "linc_" and ok(m, "dens")]
    out = df[["row", "col", "bg_relief_q"]].copy()
    out["bg_observed"] = df["lincrank_observed"]
    out["bg_observed_x"] = df["lincxrank_observed"]
    out["linf_node_dens"] = df.lin_node_dens
    out["linf_aniso_r"] = df.lin_aniso_r
    out["lincf_node_dens"] = df.linc_node_dens
    out["lincf_aniso_r"] = df.linc_aniso_r
    for m in METHODS:
        if m == "linc_":
            continue
        for ch in ("dens", "node_dens", "dist", "aniso_r"):
            out[f"{m}{ch}"] = df[f"{m}{ch}"]
    out["lincx_dir_sin"] = df.lincx_dir_sin
    if good:
        best = good[0]
        lines = "lincx_" if best.startswith("lincx") else "linc_"
        out["lincb_dens"] = df[f"{best}dens"]
        out["lincb_dist"] = df[f"{best}dist"]
        out["lincb_aniso_r"] = df[f"{lines}aniso_r"]
        out["lincb_dir_sin"] = df[f"{lines}dir_sin"]
        msg = "узлы тоже прошли" if ok(best, "node_dens") else "узлы НЕ прошли — lincb_node_dens не пишется"
        if ok(best, "node_dens"):
            out["lincb_node_dens"] = df[f"{best}node_dens"]
        print(f"\nПлотность прошла все критерии: {[METHODS[g] for g in good]}; "
              f"lincb_* = {METHODS[best]} ({msg})")
    else:
        print("\nНи один способ не прошёл все критерии по плотности; lincb_* не пишутся")
    out.to_parquet(OUT, index=False)
    table.to_csv(OUT.with_suffix(".criteria.csv"), index=False, encoding="utf-8-sig")
    log(f"сохранено: {OUT} ({len(out)} ячеек, {out.shape[1]} колонок)")

    for f in (draw_stripes(df, meta), draw_methods(df, meta, svity), draw_relief(df, table)):
        log(f"рисунок: {f}")


if __name__ == "__main__":
    main()
