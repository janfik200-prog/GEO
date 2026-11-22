"""Карта распределения свит рифея по разрезу: что лежит под каждой ячейкой листа.

Цель. Запрос ВНИГНИ от 13.11.2026: «Надо сделать карту распределения свит. То
есть если у нас горизонтальное залегание то это достаточно несложно». Карта
выходов свит на поверхность уже есть (`experiments/gmo_svita_labels.py`,
канал `k23_svity_grid`). Здесь — распределение по разрезу: для каждой ячейки
500 м — какие свиты рифея есть под ней (в том числе под более молодыми
толщами, кембрием и четвертичным чехлом), на какой глубине их кровля и
подошва и сколько рифея сохранилось. С золотом (ореолы, пробы) и с картой
ВНИГНИ результат не сверяется — это не заверка.

Входные данные:
* `data/processed/svita_labeled.gpkg` — геологическая карта листа
  R-48-XI,XII с названиями подразделений (свиты и подсвиты, фундамент,
  кембрий; четвертичные отложения и долериты границ не дают и в подгонку не
  входят);
* рельеф 100 м — `src/terrain_v2.projected_dem` (тот же, что в
  `experiments/gmo_lineaments_clean.py`);
* мощности подразделений — стратиграфическая колонка листа ГГК-200
  (источник [1] записки `Г9-литология-свит-черновик.md`), словарь `COLUMN`.

Метод. Рифей листа — пологая моноклиналь, слои согласны. Поэтому весь разрез
описывается одной опорной поверхностью `f(x, y)` — высотой подошвы ильинской
свиты (кровли фундамента) — и восемью мощностями подразделений: высота k-й
границы = f + сумма мощностей ниже неё. Подгоняются одновременно:
1. **точки контактов** — пиксели, где на карте соседствуют два смежных по
   разрезу подразделения: высота рельефа там равна высоте их границы;
2. **площади выходов** — в каждом пикселе карты рельеф должен лежать между
   подошвой и кровлей того подразделения, что показано на карте (штраф
   только за выход из этого интервала).
Потери — по Хьюберу (порог 50 м), чтобы блоки со смещением по разломам не
тянули поверхность. Сначала плоскость, затем плавная поверхность —
кубические B-сплайны со штрафом за кривизну; число узлов выбирается проверкой
с исключением блоков 10 км. Основной вариант — мощности в пределах колонки
листа; для сравнения — мощности свободные (только по данным).

Кембрий лежит на рифее с размывом, параллельным слоям его считать нельзя:
его подошва — отдельная плоскость по контактам кембрия с котуйканской и
юсмастахской свитами; под кембрием рифей срезан этой плоскостью.

Проверки (числа — в консоли и в `data/processed/svity_section_checks.json`):
(а) свита на поверхности по модели против карты — доля совпадения по ячейкам
    и матрица ошибок; то же при подгонке без блоков 10 км (5 частей), на
    исключённых блоках;
(б) остатки высот контактов от поверхностей; где рельеф систематически
    выходит из интервала своего подразделения (среднее в окне 2.5 км больше
    `MISFIT_M`) — область помечается «не восстанавливается», значения в ней
    не выдаются;
(в) угол и азимут падения: по общей поверхности, по отдельной плоскости для
    каждой границы и по парам близких точек — против литературных 3–7°;
(г) мощности: колонка листа, свободная подгонка и оценка по парам близких
    точек соседних границ; рядом — мощности по статьям для запада щита;
(д) геометрия таблицы: нет кровель над рельефом, отрицательных мощностей,
    нарушений порядка свит и свит ниже кровли фундамента.
Оговорка: контакты пологих толщ на карте сами проведены по рельефу, поэтому
согласие с картой — проверка внутренней согласованности карты и модели, а не
независимое подтверждение глубин. Скважин и разрезов в проекте нет.

Выход:
* `data/processed/svity_section_grid.parquet` — row, col, x, y; для каждой из
  пяти свит списка ВНИГНИ (il, br, lb, ul, kt): `*_share` (доля ячейки, под
  которой свита есть), `*_present`, `*_top_depth`, `*_bot_depth` (глубина
  кровли и подошвы от поверхности, м), `*_thick` (сохранившаяся мощность, м;
  глубины и мощности — средние по 25 пикселям 100 м ячейки, где свита в части
  ячейки размыта, там её мощность 0, поэтому порядок свит сохраняется);
  `rif_thick` (мощность сохранившегося рифея, м), `basement_depth` (глубина
  до фундамента, м), `under_cambrian`, `surf_model`, `surf_map`, `misfit_m`,
  `reliab` (0 — вне карты, 1 — восстановлено, 2 — не восстанавливается);
  срез 500 м (`MAX_DEPTH_M`, ответ ВНИГНИ 19.11): `*_in500` (1 есть не глубже
  среза, 0 нет, NaN глубже или не восстановлено), `basement_elev` (высота
  кровли фундамента, м), `basement_resid_m` (её отклонение от региональной
  плоскости, м; меньше нуля — палеопонижение; глубже среза NaN);
* рисунки `outputs/13_генетическая_модель/gmo_svity_section_*.png`: карта
  распределения свит, разрезы, мощность рифея, остатки.

Запуск::

    python -X utf8 -u -m experiments.gmo_svity_section
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, config, ggk200_base, integro_grid, mapstyle, outputs, terrain_v2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SVITY = ROOT / "data/processed/svita_labeled.gpkg"
DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
OUT = ROOT / "data/processed/svity_section_grid.parquet"
CHECKS = ROOT / "data/processed/svity_section_checks.json"
MAIN = (105.98, 70.65, 108.02, 71.35)
RES = config.TER_RES_M          # 100 м
PAD = config.TER_PAD_PX
CELL, SUB = 500.0, 5
BLOCK_M, N_FOLD = 10_000.0, 5
HUBER_M = 50.0
MISFIT_M, MISFIT_WIN = 30.0, 25  # порог и окно (пикс. 100 м) для «не восстанавливается»
PAIR_R = 4000.0                  # радиус пар близких точек, м
#: Срез по глубине (ответ ВНИГНИ 19.11.2026, вопрос 10: «глубоко лезть не
#: стоит. Максимум метров 500»). Глубже среза свита и кровля фундамента
#: «не наблюдаются» (NaN в колонках *_in500, basement_resid_m), а не «нет»;
#: полные глубины остаются в таблице как оценка.
MAX_DEPTH_M = 500.0
t0 = time.time()
rng = np.random.default_rng(13)

#: Подразделения карты снизу вверх: код -> (нижняя граница, верхняя граница).
#: Границы 0..8: 0 — подошва ильинской (кровля фундамента), 8 — кровля
#: котуйканской. -1 и 99 — границы нет (фундамент снизу, юсмастахская сверху).
UNIT_CODE = {"AR": 1, "PR2il1": 2, "PR2il2": 3, "PR2il": 4, "PR2br1": 5, "PR2br2": 6,
             "PR2br3": 7, "PR2br": 8, "PR2lb": 9, "PR3ul": 10, "PR3kt": 11,
             "PR3jus": 12, "Cm1": 13}
INTERVAL = {1: (-1, 0), 2: (0, 1), 3: (1, 2), 4: (0, 2), 5: (2, 3), 6: (3, 4), 7: (4, 5),
            8: (2, 5), 9: (5, 6), 10: (6, 7), 11: (7, 8), 12: (8, 99)}
COVER_CODE = {"QIV": 20, "QIIIsr": 20, "Q": 20, "intr": 21, "vbPR3": 21, "bP2-T1": 21}
BOUNDARY = ["фундамент / ильинская", "il₁ / il₂", "ильинская / бурдурская", "br₁ / br₂",
            "br₂ / br₃", "бурдурская / лабастахская", "лабастахская / усть-ильинская",
            "усть-ильинская / котуйканская", "котуйканская / юсмастахская"]
#: Мощности по стратиграфической колонке листа ГГК-200 ([1] записки Г9), м:
#: il₁, il₂, br₁, br₂, br₃, лабастахская, усть-ильинская, котуйканская.
SUBUNIT = ["il₁", "il₂", "br₁", "br₂", "br₃", "lb", "ul", "kt"]
COLUMN = np.array([(70, 98), (60, 110), (140, 193), (120, 130), (150, 160),
                   (90, 160), (65, 80), (310, 320)], dtype=float)
FREE = np.array([(5.0, 600.0)] * 8)
#: Свиты списка ВНИГНИ: ключ -> (название, нижняя граница, верхняя граница).
SV = {"il": ("ильинская", 0, 2), "br": ("бурдурская", 2, 5), "lb": ("лабастахская", 5, 6),
      "ul": ("усть-ильинская", 6, 7), "kt": ("котуйканская", 7, 8)}
#: Класс «свита на поверхности»: 0 фундамент … 6 юсмастахская и выше.
CLASS = ["фундамент", "ильинская", "бурдурская", "лабастахская", "усть-ильинская",
         "котуйканская", "юсмастахская"]
CODE2CLASS = np.array([-1, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6, -1])
CLASS_TOP = [0, 2, 5, 6, 7, 8]   # верхняя граница классов 0..5
COLORS = {"фундамент": "#9e9e9e", "ильинская": "#8e44ad", "бурдурская": "#f1c40f",
          "лабастахская": "#e67e22", "усть-ильинская": "#c0392b", "котуйканская": "#7b241c",
          "юсмастахская": "#5d6d7e", "кембрий": "#27ae60"}
FS = 15   # базовый размер шрифта: рисунки идут в презентацию


def log(msg: str) -> None:
    print(f"[{time.time() - t0:5.0f} с] {msg}", flush=True)


# ---------- данные ----------
def load():
    """Рельеф 100 м, растр подразделений карты и привязка пикселей."""
    from rasterio import features
    from rasterio.transform import from_origin

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    dem = terrain_v2.projected_dem(meta)
    ds = pd.read_parquet(DATA, columns=["row", "col", "x", "y"])
    c0 = ds.iloc[0]
    left, top = c0.x - (c0.col + 0.5) * CELL, c0.y + (c0.row + 0.5) * CELL
    T = from_origin(left - PAD * RES, top + PAD * RES, RES, RES)
    g = gpd.read_file(SVITY).to_crs(anabar_inset.SHEET_CRS)
    code = UNIT_CODE | COVER_CODE
    order = np.argsort(-g.area.to_numpy())      # мелкие полигоны поверх крупных
    lab = features.rasterize(((g.geometry.iloc[i], code[g.unit.iloc[i]]) for i in order),
                             out_shape=dem.shape, transform=T, fill=0, dtype="uint8")
    return meta, dem, lab, ds, (left - PAD * RES, top + PAD * RES)


def contact_points(lab, dem) -> np.ndarray:
    """Точки контактов смежных по разрезу подразделений: граница, строка, столбец, высота."""
    H, W = lab.shape
    pair = {}
    for a, (_, ha) in INTERVAL.items():
        for b, (lb, _) in INTERVAL.items():
            if ha == lb and 0 <= ha <= 8:
                pair[(a, b)] = ha
    out = []
    for dr, dc in ((0, 1), (1, 0)):
        a, b = lab[:H - dr, :W - dc], lab[dr:, dc:]
        for (p, q), k in pair.items():
            r, c = np.nonzero(((a == p) & (b == q)) | ((a == q) & (b == p)))
            out.append(np.c_[np.full(len(r), k), r + dr / 2, c + dc / 2,
                             (dem[r, c] + dem[r + dr, c + dc]) / 2])
    return np.vstack(out)


def cell_samples(lab):
    """Центры ячеек 500 м с коренным подразделением рифея или фундаментом."""
    H, W = lab.shape
    r, c = np.mgrid[SUB // 2:H:SUB, SUB // 2:W:SUB]
    r, c = r.ravel(), c.ravel()
    m = (lab[r, c] >= 1) & (lab[r, c] <= 12)
    return r[m], c[m], lab[r[m], c[m]]


# ---------- модель ----------
class Model:
    """Опорная поверхность (плоскость или B-сплайн) и мощности подразделений."""

    def __init__(self, shape, nk: int, lam: float, bounds: np.ndarray):
        self.H, self.W = shape
        self.nk, self.lam, self.bounds = nk, lam, bounds
        self.knots = np.r_[[0.0] * 3, np.linspace(0, 1, nk + 1), [1.0] * 3] if nk else None
        self.n = nk + 3 if nk else 0
        self.nb = self.n ** 2 if nk else 3
        self.theta = None

    def basis(self, r, c) -> np.ndarray:
        u = np.clip(np.asarray(c, float) / self.W, 0, 1 - 1e-9)
        v = np.clip(np.asarray(r, float) / self.H, 0, 1 - 1e-9)
        if not self.nk:
            return np.c_[np.ones_like(u), u, v]
        bu = BSpline.design_matrix(u, self.knots, 3).toarray()
        bv = BSpline.design_matrix(v, self.knots, 3).toarray()
        return (bu[:, :, None] * bv[:, None, :]).reshape(len(u), self.nb)

    def penalty(self) -> np.ndarray:
        if not self.nk:
            return np.zeros((0, 3))
        d, eye = np.diff(np.eye(self.n), 2, axis=0), np.eye(self.n)
        return np.vstack([np.kron(d, eye), np.kron(eye, d)])

    @staticmethod
    def _cum(k) -> np.ndarray:
        return (np.asarray(k)[:, None] > np.arange(8)[None, :]).astype(float)

    def fit(self, dem, pr, pc, pl, con, start: "Model | None" = None) -> "Model":
        z = dem[pr, pc]
        lo = np.array([INTERVAL[v][0] for v in pl])
        hi = np.array([INTERVAL[v][1] for v in pl])
        B = self.basis(pr, pc)
        a_lo, a_hi = np.c_[B, self._cum(lo)], np.c_[B, self._cum(hi)]
        has_lo, has_hi = lo >= 0, hi < 99
        a_c = np.c_[self.basis(con[:, 1], con[:, 2]), self._cum(con[:, 0])]
        z_c = con[:, 3]
        P, nb = self.penalty(), self.nb

        def obj(th):
            r_lo = np.where(has_lo, np.maximum(a_lo @ th - z, 0), 0)   # рельеф ниже подошвы
            r_hi = np.where(has_hi, np.maximum(z - a_hi @ th, 0), 0)   # рельеф выше кровли
            r_c = a_c @ th - z_c
            val, grad = 0.0, np.zeros_like(th)
            for res, A, sign in ((r_lo, a_lo, 1), (r_hi, a_hi, -1), (r_c, a_c, 1)):
                a = np.abs(res)
                val += np.where(a < HUBER_M, 0.5 * res ** 2, HUBER_M * (a - 0.5 * HUBER_M)).sum()
                grad += sign * (A.T @ np.clip(res, -HUBER_M, HUBER_M))
            if len(P):
                p = P @ th[:nb]
                val += 0.5 * self.lam * (p @ p)
                grad[:nb] += self.lam * (P.T @ p)
            return val, grad

        if start is not None and start.theta is not None:
            # начальное приближение — поверхность более простой модели
            rr, cc = np.mgrid[0:self.H:10, 0:self.W:10]
            coef = np.linalg.lstsq(self.basis(rr.ravel(), cc.ravel()),
                                   start.surface(rr.ravel(), cc.ravel()), rcond=None)[0]
            x0 = np.r_[coef, np.clip(start.theta[-8:], self.bounds[:, 0], self.bounds[:, 1])]
        else:
            x0 = np.r_[np.zeros(nb), self.bounds.mean(1)]
        bd = [(None, None)] * nb + [tuple(b) for b in self.bounds]
        self.theta = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bd,
                              options=dict(maxiter=5000, maxfun=8000, ftol=1e-11, gtol=1e-7)).x
        return self

    @property
    def thick(self) -> np.ndarray:
        return self.theta[-8:]

    @property
    def offsets(self) -> np.ndarray:
        """Высота границ 0..8 над опорной поверхностью."""
        return np.r_[0.0, np.cumsum(self.thick)]

    def surface(self, r, c) -> np.ndarray:
        return self.basis(r, c) @ self.theta[:self.nb]

    def grid(self) -> np.ndarray:
        """Опорная поверхность во всех пикселях рельефа."""
        out = np.empty((self.H, self.W))
        cc = np.arange(self.W)
        for i in range(0, self.H, 50):
            rr = np.arange(i, min(i + 50, self.H))
            R, C = np.meshgrid(rr, cc, indexing="ij")
            out[i:i + 50] = self.surface(R.ravel(), C.ravel()).reshape(R.shape)
        return out


def surf_class(h, off) -> np.ndarray:
    """Класс свиты на поверхности по высоте рельефа над опорной поверхностью."""
    return np.searchsorted(off[CLASS_TOP], h)


def violation(h, codes, off) -> np.ndarray:
    """На сколько метров рельеф выходит из интервала своего подразделения."""
    lo = np.array([INTERVAL[v][0] for v in codes])
    hi = np.array([INTERVAL[v][1] for v in codes])
    z_lo = np.where(lo >= 0, off[np.clip(lo, 0, 8)], -1e9)
    z_hi = np.where(hi < 99, off[np.clip(hi, 0, 8)], 1e9)
    return np.maximum(z_lo - h, 0) + np.maximum(h - z_hi, 0)


def score(m: Model, dem, pr, pc, pl, con) -> dict:
    h = dem[pr, pc] - m.surface(pr, pc)
    pred, act = surf_class(h, m.offsets), CODE2CLASS[pl]
    v = violation(h, pl, m.offsets)
    rc = con[:, 3] - (m.surface(con[:, 1], con[:, 2]) + m.offsets[con[:, 0].astype(int)])
    return dict(acc=float((pred == act).mean()), acc1=float((np.abs(pred - act) <= 1).mean()),
                unit_ok=float((v == 0).mean()), viol=float(v.mean()),
                c_rms=float(np.sqrt((rc ** 2).mean())), c_mad=float(np.median(np.abs(rc))),
                pred=pred, act=act, rc=rc)


def folds(r, c) -> np.ndarray:
    """Номер части (0..N_FOLD-1) по блокам 10 км; блоки общие для точек и контактов."""
    k = int(BLOCK_M / RES)
    bid = (np.floor(np.asarray(r) / k) * 1000 + np.floor(np.asarray(c) / k)).astype(int)
    perm = np.random.default_rng(13).permutation(1_000_000)
    return perm[bid % 1_000_000] % N_FOLD


def cross_validate(make, dem, pr, pc, pl, con, start) -> dict:
    fp, fc = folds(pr, pc), folds(con[:, 1], con[:, 2])
    pred, act, rc = [], [], []
    ok = []
    for i in range(N_FOLD):
        m = make().fit(dem, pr[fp != i], pc[fp != i], pl[fp != i], con[fc != i], start=start)
        s = score(m, dem, pr[fp == i], pc[fp == i], pl[fp == i], con[fc == i])
        pred.append(s["pred"]); act.append(s["act"]); rc.append(s["rc"])
        ok.append(s["unit_ok"] * (fp == i).sum())
    pred, act, rc = map(np.concatenate, (pred, act, rc))
    return dict(acc=float((pred == act).mean()), acc1=float((np.abs(pred - act) <= 1).mean()),
                unit_ok=float(sum(ok) / len(pr)),
                c_rms=float(np.sqrt((rc ** 2).mean())), c_mad=float(np.median(np.abs(rc))),
                pred=pred, act=act)


def confusion(pred, act) -> pd.DataFrame:
    t = pd.crosstab(pd.Categorical.from_codes(act, CLASS), pd.Categorical.from_codes(pred, CLASS),
                    dropna=False)
    t.index.name, t.columns.name = "на карте", "по модели"
    return t


# ---------- проверки падения и мощностей ----------
def dip_az(gx: float, gy: float) -> tuple[float, float]:
    """Угол (°) и азимут (°) падения по уклону высоты на восток и на север."""
    return (float(np.degrees(np.arctan(np.hypot(gx, gy)))),
            float(np.degrees(np.arctan2(-gx, -gy)) % 360))


def boundary_planes(con) -> pd.DataFrame:
    """Отдельная плоскость для каждой границы: падение и остатки."""
    rows = []
    for k in range(9):
        p = con[con[:, 0] == k]
        if len(p) < 30:
            continue
        x, y, z = p[:, 2] * RES, -p[:, 1] * RES, p[:, 3]
        A = np.c_[np.ones(len(x)), x - x.mean(), y - y.mean()]
        co = np.linalg.lstsq(A, z, rcond=None)[0]
        d, a = dip_az(co[1], co[2])
        rows.append({"граница": BOUNDARY[k], "точек": len(p),
                     "размах, км": round(max(np.ptp(x), np.ptp(y)) / 1e3),
                     "падение, °": round(d, 2), "азимут, °": round(a),
                     "остаток от плоскости, м": round(float((z - A @ co).std()))})
    return pd.DataFrame(rows)


def local_pairs(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Падение и мощность по парам точек ближе `PAIR_R`: без общей поверхности.

    Для пар точек одной границы разность высот = уклон x смещение; для пар
    точек двух соседних границ — то же плюс мощность подразделения между ними.
    """
    xy = np.c_[con[:, 2] * RES, -con[:, 1] * RES]
    z, kk = con[:, 3], con[:, 0].astype(int)
    dips, ths = [], []
    for a in range(9):
        for b in (a, a + 1):
            ia, ib = np.flatnonzero(kk == a), np.flatnonzero(kk == b)
            if b > 8 or len(ia) < 30 or len(ib) < 30:
                continue
            near = cKDTree(xy[ia]).query_ball_tree(cKDTree(xy[ib]), PAIR_R)
            i = np.repeat(np.arange(len(ia)), [len(p) for p in near])
            if len(i) < 200:
                continue
            j = np.concatenate([np.asarray(p, int) for p in near])
            d, dz = xy[ib][j] - xy[ia][i], z[ib][j] - z[ia][i]
            if a == b:
                far = np.hypot(*d.T) > 500
                co = np.linalg.lstsq(d[far], dz[far], rcond=None)[0]
                dd, az = dip_az(*co)
                dips.append({"граница": BOUNDARY[a], "пар": int(far.sum()),
                             "падение, °": round(dd, 2), "азимут, °": round(az)})
            else:
                co = np.linalg.lstsq(np.c_[d, np.ones(len(d))], dz, rcond=None)[0]
                ths.append({"подразделение": SUBUNIT[a], "пар": len(dz),
                            "мощность по парам, м": round(float(co[2]))})
    return pd.DataFrame(dips), pd.DataFrame(ths)


# ---------- разрез под ячейками ----------
def cambrian_base(lab, dem):
    """Подошва кембрия — плоскость по его контактам с котуйканской и юсмастахской."""
    H, W = lab.shape
    pts = []
    for dr, dc in ((0, 1), (1, 0)):
        a, b = lab[:H - dr, :W - dc], lab[dr:, dc:]
        r, c = np.nonzero(((a == 13) & np.isin(b, (11, 12))) | ((b == 13) & np.isin(a, (11, 12))))
        pts.append(np.c_[r + dr / 2, c + dc / 2, (dem[r, c] + dem[r + dr, c + dc]) / 2])
    p = np.vstack(pts)
    A = np.c_[np.ones(len(p)), p[:, 1], p[:, 0]]
    co = np.linalg.lstsq(A, p[:, 2], rcond=None)[0]
    res = p[:, 2] - A @ co
    rr, cc = np.mgrid[:H, :W]
    d, az = dip_az(co[1] / RES, -co[2] / RES)
    info = {"точек": len(p), "остаток, м": round(float(res.std()), 1),
            "падение, °": round(d, 2), "азимут, °": round(az)}
    return co[0] + co[1] * cc + co[2] * rr, info


def misfit_field(dem, lab, f, off) -> np.ndarray:
    """Средний в окне выход рельефа из интервала своего подразделения, м."""
    rock = (lab >= 1) & (lab <= 12)
    v = np.zeros(dem.shape)
    v[rock] = violation((dem - f)[rock], lab[rock], off)
    w = ndimage.uniform_filter(rock.astype(float), MISFIT_WIN)
    s = ndimage.uniform_filter(v, MISFIT_WIN)
    out = np.where(w > 0.1, s / np.maximum(w, 1e-9), np.nan)
    # под четвертичными отложениями и долеритами — по ближайшему коренному окну
    idx = ndimage.distance_transform_edt(np.isnan(out), return_distances=False,
                                         return_indices=True)
    return out[idx[0], idx[1]]


def cell_mean(a, meta, where=None) -> np.ndarray:
    """Среднее по ячейке 500 м (5 x 5 пикселей); `where` — по каким пикселям."""
    a = a[PAD:PAD + meta.prf * SUB, PAD:PAD + meta.pic * SUB]
    a = a.reshape(meta.prf, SUB, meta.pic, SUB)
    if where is None:
        return a.mean(axis=(1, 3))
    w = where[PAD:PAD + meta.prf * SUB, PAD:PAD + meta.pic * SUB].reshape(a.shape)
    n = w.sum(axis=(1, 3))
    return np.where(n > 0, (a * w).sum(axis=(1, 3)) / np.maximum(n, 1), np.nan)


def cell_mode(a, meta, n_class: int) -> np.ndarray:
    a = a[PAD:PAD + meta.prf * SUB, PAD:PAD + meta.pic * SUB]
    a = a.reshape(meta.prf, SUB, meta.pic, SUB)
    cnt = np.stack([(a == k).sum(axis=(1, 3)) for k in range(n_class)], -1)
    return np.where(cnt.sum(-1) > 0, cnt.argmax(-1), -1)


def build_grid(meta, dem, lab, ds, f, off, cm_plane, misfit) -> pd.DataFrame:
    """Таблица по ячейкам: свиты под ячейкой, глубины, мощности, надёжность."""
    is_cm = lab == 13
    top = np.where(is_cm, np.minimum(dem, cm_plane), dem)   # кровля рифея
    cols = {}
    # Глубины считаются по всем пикселям ячейки: где свита размыта, её кровля и
    # подошва совпадают (мощность 0). Тогда подошва свиты в каждом пикселе равна
    # кровле нижележащей, средние по ячейке сохраняют порядок слоёв, а мощности
    # свит в сумме дают мощность рифея (при усреднении только по пикселям, где
    # свита есть, порядок соседних свит нарушался на 2–25 м в ~800 ячейках).
    for key, (_, lo, hi) in SV.items():
        zt = np.minimum(f + off[hi], top)
        zb = np.minimum(f + off[lo], zt)
        here = zt > zb
        cols[f"{key}_share"] = cell_mean(here.astype(float), meta)
        cols[f"{key}_top_depth"] = cell_mean(dem - zt, meta)
        cols[f"{key}_bot_depth"] = cell_mean(dem - zb, meta)
        cols[f"{key}_thick"] = cell_mean(zt - zb, meta)
    cols["rif_thick"] = cell_mean(np.clip(top - f, 0, None), meta)
    cols["basement_depth"] = cell_mean(np.clip(dem - f, 0, None), meta)
    cols["basement_elev"] = cell_mean(f, meta)
    cols["under_cambrian"] = cell_mean(is_cm.astype(float), meta) >= 0.5
    cols["misfit_m"] = cell_mean(misfit, meta)
    pred = surf_class(dem - f, off)
    cls_map = np.where(lab <= 13, CODE2CLASS[np.minimum(lab, 13)], -1)
    names = np.array(CLASS + ["нет"])
    cols["surf_model"] = names[cell_mode(pred, meta, len(CLASS))]
    sm = names[cell_mode(cls_map, meta, len(CLASS))]
    cov = cell_mode(np.select([lab == 0, lab == 13, lab == 20, lab == 21], [0, 1, 2, 3], 4),
                    meta, 5)
    sm = np.select([cov == 0, cov == 1, cov == 2, cov == 3],
                   ["нет карты", "кембрий", "четвертичные", "долериты"], sm)
    cols["surf_map"] = sm
    out_map = cell_mean((lab == 0).astype(float), meta) >= 0.5
    cols["reliab"] = np.select([out_map, cols["misfit_m"] > MISFIT_M], [0, 2], 1).astype("int8")

    g = pd.DataFrame({k: v.ravel() for k, v in cols.items()})
    g["row"], g["col"] = np.divmod(np.arange(len(g)), meta.pic)
    g = ds[["row", "col", "x", "y"]].merge(g, on=["row", "col"], how="left")
    bad = g.reliab != 1
    for key in SV:
        g[f"{key}_present"] = (g[f"{key}_share"] >= 0.5) & ~bad
        absent = ~(g[f"{key}_share"] >= 0.5)
        g.loc[bad | absent, [f"{key}_top_depth", f"{key}_bot_depth", f"{key}_thick"]] = np.nan
        g.loc[bad, f"{key}_share"] = np.nan
    g.loc[bad, ["rif_thick", "basement_depth", "basement_elev"]] = np.nan
    g.loc[bad, "surf_model"] = "нет"
    # Срез 500 м: 1 — свита есть и её кровля не глубже среза, 0 — свиты нет
    # (размыта или не отлагалась), NaN — глубже среза или разрез не восстановлен.
    for key in SV:
        v = g[f"{key}_present"].astype(float)
        v[g[f"{key}_present"] & (g[f"{key}_top_depth"] > MAX_DEPTH_M)] = np.nan
        v[bad] = np.nan
        g[f"{key}_in500"] = v
    # Палеопонижения кровли фундамента (ответ ВНИГНИ 19.11, вопрос 2: золото
    # сохраннее «там, где были понижения и лакуны, в которые легли рифейские
    # породы»): высота кровли минус региональная плоскость по надёжным ячейкам,
    # м; меньше нуля — понижение. Масштаб — десятки километров (гибкость
    # поверхности), понижения мельче ~10 км не видны. Глубже среза — NaN.
    ok = ~bad & g.basement_elev.notna()
    A = np.c_[np.ones(ok.sum()), g.x[ok], g.y[ok]]
    co = np.linalg.lstsq(A, g.basement_elev[ok], rcond=None)[0]
    g["basement_resid_m"] = g.basement_elev - (co[0] + co[1] * g.x + co[2] * g.y)
    g.loc[bad | (g.basement_depth > MAX_DEPTH_M), "basement_resid_m"] = np.nan
    order = ["row", "col", "x", "y"]
    for key in SV:
        order += [f"{key}_{s}" for s in ("present", "share", "top_depth", "bot_depth", "thick")]
    order += [f"{key}_in500" for key in SV]
    order += ["rif_thick", "basement_depth", "basement_elev", "basement_resid_m",
              "under_cambrian", "surf_model", "surf_map", "misfit_m", "reliab"]
    return g[order]


# ---------- рисунки ----------
def px_lonlat(r, c, origin):
    from pyproj import Transformer
    tr = Transformer.from_crs(anabar_inset.SHEET_CRS, 4326, always_xy=True)
    return tr.transform(origin[0] + np.asarray(c, float) * RES,
                        origin[1] - np.asarray(r, float) * RES)


def cell_mesh(meta, origin):
    """Долгота и широта углов ячеек 500 м для pcolormesh."""
    r, c = np.mgrid[0:meta.prf + 1, 0:meta.pic + 1]
    return px_lonlat(PAD + r * SUB, PAD + c * SUB, origin)


def to_img(g: pd.DataFrame, col: str, meta) -> np.ndarray:
    a = np.full((meta.prf, meta.pic), np.nan)
    a[g.row, g.col] = g[col].astype(float)
    return a


def base(ax, title: str | None = None, ylabel: bool = True) -> None:
    ggk200_base.draw(ax, bounds=MAIN, decimate=2, alpha=0.35, sheets=("R-48-XI,XII",))
    ax.set_xlabel("долгота, ° в.д.", fontsize=FS)
    if ylabel:
        ax.set_ylabel("широта, ° с.ш.", fontsize=FS)
    ax.tick_params(labelsize=FS - 2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    if title:
        ax.set_title(title, fontsize=FS + 2)


def grey(ax, lon, lat, g, meta) -> None:
    bad = np.where(to_img(g, "reliab", meta) == 2, 1.0, np.nan)
    ax.pcolormesh(lon, lat, bad, cmap=ListedColormap(["#4d4d4d"]), vmin=0, vmax=1, zorder=3)


DEPTH_EDGES = [0, 50, 150, 300, 500, 2000]
DEPTH_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#d9d9d9"]
DEPTH_LABELS = ["до 50 м", "50–150 м", "150–300 м", "300–500 м",
                "глубже 500 м —\nне оценивается"]


def draw_distribution(g, meta, origin) -> Path:
    lon, lat = cell_mesh(meta, origin)
    fig, axes = plt.subplots(2, 3, figsize=(22, 15.5))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.06, wspace=0.14, hspace=0.2)
    for ax, (key, (name, _, _)) in zip(axes.ravel(), SV.items()):
        base(ax, f"{name.capitalize()} свита")
        here = to_img(g, f"{key}_present", meta) == 1
        d = to_img(g, f"{key}_top_depth", meta)
        buried = np.where(here & (d > 5), d, np.nan)
        ax.pcolormesh(lon, lat, buried, cmap=ListedColormap(DEPTH_COLORS),
                      norm=BoundaryNorm(DEPTH_EDGES, len(DEPTH_COLORS)), zorder=2)
        atsurf = np.where(here & (d <= 5), 1.0, np.nan)
        ax.pcolormesh(lon, lat, atsurf, cmap=ListedColormap([COLORS[name]]), vmin=0, vmax=1,
                      zorder=2)
        grey(ax, lon, lat, g, meta)
        ok = g[f"{key}_present"]
        ax.text(0.02, 0.02, f"есть под {100 * ok.sum() / (g.reliab == 1).sum():.0f}% площади,\n"
                f"из них на поверхности {100 * (ok & (g[f'{key}_top_depth'] <= 5)).sum() / max(ok.sum(), 1):.0f}%",
                transform=ax.transAxes, fontsize=FS - 1, va="bottom",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.5"))
    ax = axes.ravel()[5]
    ax.axis("off")
    handles = [Patch(facecolor=COLORS[v[0]], edgecolor="k", label=f"{v[0]} — на поверхности")
               for v in SV.values()]
    handles += [Patch(facecolor="none", edgecolor="none", label=""),
                Patch(facecolor="none", edgecolor="none", label="свита под более молодыми")]
    handles[-1].set_label("под более молодыми толщами,\nглубина кровли:")
    handles += [Patch(facecolor=c, edgecolor="k", label=l)
                for c, l in zip(DEPTH_COLORS, DEPTH_LABELS)]
    handles += [Patch(facecolor="none", edgecolor="none", label=""),
                Patch(facecolor="#4d4d4d", edgecolor="k", label="разрез не\nвосстанавливается"),
                Patch(facecolor="white", edgecolor="k",
                      label="свиты под ячейкой нет\n(размыта или\nне отлагалась)")]
    ax.legend(handles=handles, loc="upper left", fontsize=FS, frameon=False,
              bbox_to_anchor=(0.0, 1.0))
    anabar_inset.draw(ax.inset_axes([0.6, 0.0, 0.4, 0.44]), fontsize=FS - 4, title=False,
                      legend=False)
    fig.suptitle("Лист R-48-XI,XII: распределение свит рифея по разрезу — где свита есть "
                 "под ячейкой и на какой глубине её кровля", fontsize=FS + 5)
    out = outputs.png("gmo_svity_section_map.png")
    mapstyle.save_fig(fig, out, dpi=110)
    return out


def profile_lines(lab, grad_az: float, offsets_km=(-20.0, 0.0, 20.0)):
    """Линии вкрест простирания (по падению) через лист: списки (строка, столбец)."""
    H, W = lab.shape
    rr, cc = np.nonzero(lab > 0)
    r0, c0 = rr.mean(), cc.mean()
    a = np.radians(grad_az)
    d = np.array([-np.cos(a), np.sin(a)])       # шаг по (строка, столбец) в сторону падения
    n = np.array([d[1], -d[0]])
    lines = []
    for off in offsets_km:
        s = np.arange(-900, 901)
        r = r0 + off * 10 * n[0] + s * d[0]
        c = c0 + off * 10 * n[1] + s * d[1]
        ok = (r >= 0) & (r < H - 1) & (c >= 0) & (c < W - 1)
        r, c = r[ok], c[ok]
        inside = lab[r.round().astype(int), c.round().astype(int)] > 0
        idx = np.flatnonzero(inside)
        lines.append((r[idx[0]:idx[-1] + 1], c[idx[0]:idx[-1] + 1]))
    return lines


def draw_profiles(lines, dem, lab, f, off, cm_plane, misfit, grad_az) -> Path:
    fig, axes = plt.subplots(len(lines), 1, figsize=(22, 6.0 * len(lines)))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.93, bottom=0.06, hspace=0.3)
    bands = [("фундамент", -99, 0), ("ильинская", 0, 2), ("бурдурская", 2, 5),
             ("лабастахская", 5, 6), ("усть-ильинская", 6, 7), ("котуйканская", 7, 8),
             ("юсмастахская", 8, 99)]
    zmin = 100 * np.floor(min(np.nanmin(dem[lab > 0]) - 150, 0) / 100)
    for ax, (r, c), name in zip(axes, lines, "АБВ"):
        ri, ci = r.round().astype(int), c.round().astype(int)
        s = np.arange(len(r)) * RES / 1e3
        z, ff, lb = dem[ri, ci], f[ri, ci], lab[ri, ci]
        top = np.where(lb == 13, np.minimum(z, cm_plane[ri, ci]), z)
        for nm, lo, hi in bands:
            zb = np.full_like(z, zmin) if lo < 0 else ff + off[lo]
            zt = top if hi > 8 else np.minimum(ff + off[hi], top)
            ax.fill_between(s, zb, zt, where=zt > zb, color=COLORS[nm], alpha=0.85, linewidth=0)
        ax.fill_between(s, top, z, where=z > top + 0.5, color=COLORS["кембрий"], linewidth=0)
        for k in range(9):
            zk = ff + off[k]
            ax.plot(s, np.where(zk <= top, zk, np.nan), color="k", lw=0.9 if k in CLASS_TOP else 0.4)
        ax.plot(s, z, color="k", lw=2.0, zorder=4)
        bad = misfit[ri, ci] > MISFIT_M
        ax.fill_between(s, zmin, z, where=bad, color="#4d4d4d", linewidth=0, zorder=3)
        ax.set_xlim(0, s[-1]); ax.set_ylim(zmin, 100 * np.ceil((z.max() + 60) / 100))
        ax.set_ylabel("высота, м", fontsize=FS)
        ax.set_xlabel("расстояние вдоль разреза, км", fontsize=FS)
        ax.tick_params(labelsize=FS - 1)
        ax.grid(axis="y", color="0.6", lw=0.4)
        ax.set_title(f"Разрез {name}–{name}′", fontsize=FS + 2, loc="left")
        ax.text(0.005, 0.95, f"{name}  юго-восток", transform=ax.transAxes, fontsize=FS,
                va="top", fontweight="bold")
        ax.text(0.995, 0.95, f"северо-запад  {name}′", transform=ax.transAxes, fontsize=FS,
                va="top", ha="right", fontweight="bold")
    fig.canvas.draw()
    bb = axes[0].get_window_extent()
    xl, yl = axes[0].get_xlim(), axes[0].get_ylim()
    exag = (bb.height / (yl[1] - yl[0])) / (bb.width / ((xl[1] - xl[0]) * 1e3))
    handles = [Patch(facecolor=COLORS[n], label=n if n == "фундамент" else f"{n} свита")
               for n, _, _ in bands[::-1]]
    handles.insert(0, Patch(facecolor=COLORS["кембрий"], label="кембрий"))
    handles += [Line2D([0], [0], color="k", lw=2, label="рельеф"),
                Line2D([0], [0], color="k", lw=0.9, label="границы свит"),
                Line2D([0], [0], color="k", lw=0.4, label="границы подсвит"),
                Patch(facecolor="#4d4d4d", label="разрез не\nвосстанавливается")]
    fig.legend(handles=handles, loc="center right", fontsize=FS, framealpha=0.95)
    fig.suptitle(f"Разрезы вкрест простирания (по азимуту падения {grad_az:.0f}°): рельеф и "
                 f"границы свит; вертикальный масштаб крупнее горизонтального в {exag:.0f} раз",
                 fontsize=FS + 4)
    out = outputs.png("gmo_svity_section_profiles.png")
    mapstyle.save_fig(fig, out, dpi=110)
    return out


def side_panel(fig, rect, handles=None, title=None):
    """Врезка положения листа и легенда справа от карты."""
    anabar_inset.draw(fig.add_axes(rect), fontsize=FS - 4, title=False, legend=False)
    if handles:
        fig.legend(handles=handles, loc="upper left", fontsize=FS - 1, framealpha=0.95,
                   bbox_to_anchor=(rect[0] - 0.01, 0.9), title=title, title_fontsize=FS - 1)


def draw_thickness(g, meta, origin, lines) -> Path:
    lon, lat = cell_mesh(meta, origin)
    fig, ax = plt.subplots(figsize=(18, 11.5))
    fig.subplots_adjust(left=0.06, right=0.66, top=0.92, bottom=0.08)
    base(ax, "Мощность сохранившегося рифея под ячейкой")
    edges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]
    cmap = plt.get_cmap("YlOrBr", len(edges) - 1)
    a = to_img(g, "rif_thick", meta)
    im = ax.pcolormesh(lon, lat, np.where(a > 1, a, np.nan), cmap=cmap,
                       norm=BoundaryNorm(edges, cmap.N), zorder=2, alpha=0.9)
    zero = np.where(a <= 1, 1.0, np.nan)
    ax.pcolormesh(lon, lat, zero, cmap=ListedColormap(["#9e9e9e"]), vmin=0, vmax=1, zorder=2)
    grey(ax, lon, lat, g, meta)
    for (r, c), name in zip(lines, "АБВ"):
        x, y = px_lonlat(r, c, origin)
        ax.plot(x, y, color="#0b3d91", lw=2.5, zorder=4)
        lab_kw = dict(fontsize=FS + 2, color="#0b3d91", fontweight="bold", zorder=5,
                      textcoords="offset points",
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        ax.annotate(name, (x[0], y[0]), xytext=(-8, 8), ha="right", va="bottom", **lab_kw)
        ax.annotate(f"{name}′", (x[-1], y[-1]), xytext=(8, -8), ha="left", va="top", **lab_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=edges)
    cb.set_label("мощность рифея от кровли фундамента\nдо поверхности (под кембрием — до его подошвы), м",
                 fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 2)
    side_panel(fig, [0.74, 0.08, 0.24, 0.36], [
        Patch(facecolor="#9e9e9e", edgecolor="k", label="рифея нет: фундамент\nна поверхности"),
        Patch(facecolor="#4d4d4d", edgecolor="k", label="разрез не\nвосстанавливается"),
        Line2D([0], [0], color="#0b3d91", lw=2.5, label="линии разрезов")])
    out = outputs.png("gmo_svity_section_thickness.png")
    mapstyle.save_fig(fig, out, dpi=110)
    return out


def draw_residuals(g, meta, origin, con, rc) -> Path:
    lon, lat = cell_mesh(meta, origin)
    fig, axes = plt.subplots(1, 2, figsize=(19, 11))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.2, wspace=0.1)
    ax = axes[0]
    base(ax, "Остатки высот контактов от поверхностей")
    x, y = px_lonlat(con[:, 1], con[:, 2], origin)
    o = np.argsort(np.abs(rc))
    sc = ax.scatter(x[o], y[o], c=rc[o], s=7, cmap="RdBu_r", vmin=-120, vmax=120, zorder=3,
                    linewidths=0)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09)
    cb.set_label("высота контакта на карте минус высота границы по модели, м\n"
                 "(красное — контакт выше, чем ждёт модель; синее — ниже)", fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 2)
    ax = axes[1]
    base(ax, "Несогласие рельефа и карты с моделью, среднее в окне 2.5 км", ylabel=False)
    a = to_img(g, "misfit_m", meta)
    a[to_img(g, "reliab", meta) == 0] = np.nan
    edges = [0, 5, 15, 30, 60, 100, 200]
    cmap = ListedColormap(["#f7f7f7", "#fee8c8", "#fdbb84", "#ef6548", "#b30000", "#4d0000"])
    im = ax.pcolormesh(lon, lat, a, cmap=cmap, norm=BoundaryNorm(edges, cmap.N), zorder=2,
                       alpha=0.9)
    bad = (to_img(g, "reliab", meta) == 2).astype(float)
    lonc = 0.25 * (lon[:-1, :-1] + lon[1:, :-1] + lon[:-1, 1:] + lon[1:, 1:])
    latc = 0.25 * (lat[:-1, :-1] + lat[1:, :-1] + lat[:-1, 1:] + lat[1:, 1:])
    ax.contour(lonc, latc, bad, levels=[0.5], colors="k", linewidths=2.0, zorder=4)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, ticks=edges)
    cb.set_label("на сколько метров рельеф выходит из интервала\nподразделения, показанного "
                 f"на карте, м (чёрный контур — больше {MISFIT_M:.0f} м:\nразрез не "
                 "восстанавливается; возможны смещения по разломам)", fontsize=FS)
    cb.ax.tick_params(labelsize=FS - 2)
    fig.suptitle("Лист R-48-XI,XII: где модель пологого согласного залегания не сходится с картой",
                 fontsize=FS + 5)
    out = outputs.png("gmo_svity_section_residuals.png")
    mapstyle.save_fig(fig, out, dpi=110)
    return out


# ---------- запуск ----------
def main() -> None:
    meta, dem, lab, ds, origin = load()
    con = contact_points(lab, dem)
    pr, pc, pl = cell_samples(lab)
    log(f"рельеф {dem.shape}; ячеек с коренным подразделением {len(pr)}; точек контактов {len(con)}")
    checks: dict = {"ячеек": int(len(pr)), "точек контактов": int(len(con))}

    # --- плоскость, затем сплайны; выбор по проверке с исключением блоков ---
    plane = Model(dem.shape, 0, 0.0, COLUMN).fit(dem, pr, pc, pl, con)
    rows, best = [], None
    for label, nk, lam in [("плоскость", 0, 0.0), ("сплайн 2 узла", 2, 10.0),
                           ("сплайн 4 узла", 4, 10.0), ("сплайн 6 узлов", 6, 10.0),
                           ("сплайн 8 узлов", 8, 10.0), ("сплайн 8 узлов, жёстче", 8, 100.0),
                           ("сплайн 10 узлов", 10, 10.0)]:
        make = lambda nk=nk, lam=lam: Model(dem.shape, nk, lam, COLUMN)  # noqa: E731
        m = make().fit(dem, pr, pc, pl, con, start=plane if nk else None)
        s = score(m, dem, pr, pc, pl, con)
        cv = cross_validate(make, dem, pr, pc, pl, con, plane if nk else None)
        rows.append({"поверхность": label, "совпадение свиты": s["acc"], "с исключением": cv["acc"],
                     "±1 свита, искл.": cv["acc1"], "подразделение, искл.": cv["unit_ok"],
                     "остаток контактов, м": s["c_rms"], "с искл., м": cv["c_rms"],
                     "медиана |остатка| искл., м": cv["c_mad"]})
        log(f"{label}: совпадение {s['acc']:.3f}, с исключением {cv['acc']:.3f}, "
            f"остаток контактов {s['c_rms']:.0f} / {cv['c_rms']:.0f} м")
        if best is None or cv["acc"] > best[2]["acc"] + 0.005:
            best = (label, m, cv, s, make)
    sweep = pd.DataFrame(rows)
    print("\nПодбор поверхности (мощности в пределах колонки листа):")
    print(sweep.round(3).to_string(index=False))
    label, model, cv, s, make = best
    print(f"\nВыбрано: {label}")
    checks["подбор поверхности"] = sweep.round(3).to_dict("records")
    checks["выбрано"] = label

    # --- (а) свита на поверхности: модель против карты ---
    print("\n(а) Свита на поверхности, центры ячеек 500 м — подгонка по всем данным:")
    cm_all = confusion(s["pred"], s["act"])
    print(cm_all.to_string())
    print(f"   совпадение {s['acc']:.3f}; с точностью до соседней свиты {s['acc1']:.3f}; "
          f"подразделение (подсвита) {s['unit_ok']:.3f}")
    print(f"\n(а) То же с исключением блоков {BLOCK_M / 1e3:.0f} км ({N_FOLD} частей):")
    cm_cv = confusion(cv["pred"], cv["act"])
    print(cm_cv.to_string())
    print(f"   совпадение {cv['acc']:.3f}; с точностью до соседней свиты {cv['acc1']:.3f}; "
          f"подразделение {cv['unit_ok']:.3f}")
    recall = (np.diag(cm_cv) / cm_cv.sum(1).clip(lower=1)).round(3)
    base_rate = float(cm_all.sum(1).max() / cm_all.to_numpy().sum())
    print(f"   по свитам (доля ячеек свиты, названных верно): {recall.to_dict()}")
    print(f"   для сравнения: самая частая свита на карте занимает {base_rate:.3f}")
    checks["а"] = {"совпадение": s["acc"], "±1": s["acc1"], "подразделение": s["unit_ok"],
                   "с исключением": cv["acc"], "с исключением ±1": cv["acc1"],
                   "с исключением подразделение": cv["unit_ok"],
                   "матрица (все данные)": cm_all.to_dict(), "матрица (исключение)": cm_cv.to_dict(),
                   "по свитам (исключение)": recall.to_dict(), "самая частая свита": base_rate}

    # --- (б) остатки контактов ---
    f = model.grid()
    off = model.offsets
    rc = s["rc"]
    kk = con[:, 0].astype(int)
    res_t = pd.DataFrame([{"граница": BOUNDARY[k], "точек": int((kk == k).sum()),
                           "средний, м": round(float(rc[kk == k].mean()), 1),
                           "разброс, м": round(float(rc[kk == k].std()), 1),
                           "медиана |остатка|, м": round(float(np.median(np.abs(rc[kk == k]))), 1)}
                          for k in range(9) if (kk == k).sum()])
    print("\n(б) Остатки высот контактов от поверхностей модели:")
    print(res_t.to_string(index=False))
    print(f"   все контакты: разброс {rc.std():.0f} м, медиана |остатка| {np.median(np.abs(rc)):.0f} м; "
          f"с исключением блоков {cv['c_rms']:.0f} и {cv['c_mad']:.0f} м")
    checks["б"] = {"по границам": res_t.to_dict("records"), "разброс": float(rc.std()),
                   "медиана": float(np.median(np.abs(rc))), "искл. разброс": cv["c_rms"],
                   "искл. медиана": cv["c_mad"]}

    # --- (в) падение ---
    gy, gx = np.gradient(f, RES)                 # по строкам (на юг) и по столбцам (на восток)
    inside = lab > 0
    dip = np.degrees(np.arctan(np.hypot(gx, gy)))[inside]
    d_mean, az_mean = dip_az(float(gx[inside].mean()), float(-gy[inside].mean()))
    az_px = np.degrees(np.arctan2(-gx, gy))[inside] % 360
    planes = boundary_planes(con)
    pair_dip, pair_th = local_pairs(con)
    relief_span = float(off[8])
    print(f"\n(в) Падение по общей поверхности: среднее {d_mean:.2f}° по азимуту {az_mean:.0f}°; "
          f"по пикселям медиана {np.median(dip):.2f}°, 5–95% {np.percentile(dip, 5):.2f}–"
          f"{np.percentile(dip, 95):.2f}°, наибольшее {dip.max():.2f}°; азимут 25–75% "
          f"{np.percentile(az_px, 25):.0f}–{np.percentile(az_px, 75):.0f}°")
    print("   отдельная плоскость для каждой границы:")
    print(planes.to_string(index=False))
    print(f"   по парам точек одной границы ближе {PAIR_R / 1e3:.0f} км:")
    print(pair_dip.to_string(index=False))
    print(f"   литература: 3–7° на запад. Доля площади с падением ≥ 3°: {100 * (dip >= 3).mean():.1f}%")
    checks["в"] = {"среднее падение": d_mean, "азимут": az_mean,
                   "медиана": float(np.median(dip)), "5%": float(np.percentile(dip, 5)),
                   "95%": float(np.percentile(dip, 95)), "макс": float(dip.max()),
                   "доля ≥3°": float((dip >= 3).mean()),
                   "плоскости по границам": planes.to_dict("records"),
                   "по парам": pair_dip.to_dict("records")}

    # --- (г) мощности ---
    free_plane = Model(dem.shape, 0, 0.0, FREE).fit(dem, pr, pc, pl, con)
    free = Model(dem.shape, model.nk, model.lam, FREE).fit(dem, pr, pc, pl, con, start=free_plane)
    s_free = score(free, dem, pr, pc, pl, con)
    cv_free = cross_validate(lambda: Model(dem.shape, model.nk, model.lam, FREE),
                             dem, pr, pc, pl, con, free_plane)
    th = pd.DataFrame({"подразделение": SUBUNIT,
                       "колонка листа, м": [f"{a:.0f}–{b:.0f}" for a, b in COLUMN],
                       "модель (в пределах колонки), м": model.thick.round(0),
                       "свободная подгонка, м": free.thick.round(0)})
    th = th.merge(pair_th[["подразделение", "мощность по парам, м"]], on="подразделение", how="left")
    print("\n(г) Мощности подразделений:")
    print(th.to_string(index=False))
    grp = {"ильинская": (0, 2), "бурдурская": (2, 5), "лабастахская": (5, 6),
           "усть-ильинская": (6, 7), "котуйканская": (7, 8), "мукунская серия": (0, 6),
           "весь разрез до кровли котуйканской": (0, 8)}
    th_sv = pd.DataFrame([{"толща": n, "колонка листа, м": f"{COLUMN[a:b, 0].sum():.0f}–{COLUMN[a:b, 1].sum():.0f}",
                           "модель, м": round(float(model.thick[a:b].sum())),
                           "свободная подгонка, м": round(float(free.thick[a:b].sum())),
                           "по парам, м": (round(float(pair_th.set_index("подразделение")
                                                       ["мощность по парам, м"].reindex(SUBUNIT[a:b]).sum(min_count=b - a)))
                                           if pair_th.set_index("подразделение")["мощность по парам, м"]
                                           .reindex(SUBUNIT[a:b]).notna().all() else None)}
                          for n, (a, b) in grp.items()])
    # по статьям для запада и северо-запада щита (черновик этапа IV, раздел 7)
    lit = {"усть-ильинская": "55–70", "котуйканская": "250–320", "мукунская серия": "600–800"}
    th_sv.insert(2, "по статьям, м", th_sv["толща"].map(lit).fillna("—"))
    print(th_sv.to_string(index=False))
    print(f"   свободная подгонка: совпадение свиты {s_free['acc']:.3f} (с исключением {cv_free['acc']:.3f}), "
          f"остаток контактов {s_free['c_rms']:.0f} м (с исключением {cv_free['c_rms']:.0f} м)")
    gyf, gxf = np.gradient(free.grid(), RES)
    d_free, az_free = dip_az(float(gxf[inside].mean()), float(-gyf[inside].mean()))
    print(f"   падение при свободных мощностях: {d_free:.2f}° по азимуту {az_free:.0f}°")
    checks["г"] = {"подразделения": th.to_dict("records"), "свиты": th_sv.to_dict("records"),
                   "свободная: совпадение": s_free["acc"], "свободная: с исключением": cv_free["acc"],
                   "свободная: остаток": s_free["c_rms"], "свободная: остаток искл.": cv_free["c_rms"],
                   "свободная: падение": d_free, "свободная: азимут": az_free}

    # --- кембрий, надёжность, таблица по ячейкам ---
    cm_plane, cm_info = cambrian_base(lab, dem)
    print(f"\nПодошва кембрия (отдельная плоскость): {cm_info}")
    misfit = misfit_field(dem, lab, f, off)
    g = build_grid(meta, dem, lab, ds, f, off, cm_plane, misfit)
    g.to_parquet(OUT, index=False)
    n_map = int((g.reliab > 0).sum())
    ok = g.reliab == 1
    print(f"\nЯчеек {len(g)}: вне карты {int((g.reliab == 0).sum())}, восстановлено {int(ok.sum())} "
          f"({100 * ok.sum() / n_map:.1f}% карты), не восстанавливается {int((g.reliab == 2).sum())} "
          f"({100 * (g.reliab == 2).sum() / n_map:.1f}%)")
    summ = []
    for key, (name, _, _) in SV.items():
        p = g[f"{key}_present"]
        surf = p & (g[f"{key}_top_depth"] <= 5)
        summ.append({"свита": name, "есть под ячейкой, % площади": round(100 * p.sum() / ok.sum(), 1),
                     "на поверхности, %": round(100 * surf.sum() / ok.sum(), 1),
                     "под более молодыми, %": round(100 * (p & ~surf).sum() / ok.sum(), 1),
                     "глубина кровли там, м (медиана)": round(float(g.loc[p & ~surf, f"{key}_top_depth"].median()), 0),
                     "наибольшая, м": round(float(g.loc[p, f"{key}_top_depth"].max()), 0),
                     "мощность, м (медиана)": round(float(g.loc[p, f"{key}_thick"].median()), 0)})
    summ = pd.DataFrame(summ)
    print(summ.to_string(index=False))
    in500 = {SV[k][0]: round(100 * float((g[f"{k}_in500"] == 1).sum() / ok.sum()), 1) for k in SV}
    deep = {SV[k][0]: round(100 * float((g[f"{k}_present"] & g[f"{k}_in500"].isna()).sum()
                                        / ok.sum()), 1) for k in SV}
    print(f"   срез {MAX_DEPTH_M:.0f} м — свита есть не глубже среза, % площади: {in500}; "
          f"глубже среза (не оценивается): {deep}")
    print(f"   кровля фундамента не глубже среза: {100 * (g.basement_depth[ok] <= MAX_DEPTH_M).mean():.1f}% "
          f"площади; отклонение от плоскости там: 5–95% "
          f"{g.basement_resid_m.quantile(0.05):.0f}…{g.basement_resid_m.quantile(0.95):.0f} м")
    checks["срез 500 м"] = {"свита не глубже среза": in500, "глубже среза": deep,
                           "фундамент не глубже среза": float((g.basement_depth[ok] <= MAX_DEPTH_M).mean())}
    print(f"   мощность сохранившегося рифея: медиана {g.rif_thick[ok].median():.0f} м, "
          f"наибольшая {g.rif_thick[ok].max():.0f} м; рифея нет (фундамент на поверхности) — "
          f"{100 * (g.rif_thick[ok] <= 1).mean():.1f}% площади; глубина до фундамента: медиана "
          f"{g.basement_depth[ok].median():.0f} м, наибольшая {g.basement_depth[ok].max():.0f} м")
    # ячейки под чехлом (четвертичные, долериты, кембрий): что под ними по модели
    cov = g[ok & g.surf_map.isin(["четвертичные", "долериты", "кембрий"])]
    print("   под чехлом (карта не показывает рифей), свита у поверхности по модели:")
    print(cov.groupby(["surf_map", "surf_model"]).size().unstack(fill_value=0).to_string())
    checks["ячейки"] = {"всего": len(g), "вне карты": int((g.reliab == 0).sum()),
                        "восстановлено": int(ok.sum()), "не восстанавливается": int((g.reliab == 2).sum()),
                        "свиты": summ.to_dict("records"),
                        "рифей медиана": float(g.rif_thick[ok].median()),
                        "рифей макс": float(g.rif_thick[ok].max()),
                        "рифея нет, доля": float((g.rif_thick[ok] <= 1).mean()),
                        "фундамент макс": float(g.basement_depth[ok].max()),
                        "под чехлом": cov.groupby(["surf_map", "surf_model"]).size().unstack(fill_value=0).to_dict()}
    checks["кембрий"] = cm_info

    # --- (д) геометрическая согласованность таблицы ---
    r = g[ok]
    keys = list(SV)[::-1]                      # сверху вниз: kt, ul, lb, br, il
    geo = {"кровля выше поверхности": 0, "мощность < 0": 0, "подошва выше кровли": 0}
    for key in keys:
        geo["кровля выше поверхности"] += int((r[f"{key}_top_depth"] < -0.5).sum())
        geo["мощность < 0"] += int((r[f"{key}_thick"] < -0.5).sum())
        geo["подошва выше кровли"] += int((r[f"{key}_bot_depth"] < r[f"{key}_top_depth"] - 0.5).sum())
    geo["порядок соседних свит нарушен"] = int(sum(
        (r[f"{a}_bot_depth"] > r[f"{b}_top_depth"] + 0.5).sum() for a, b in zip(keys[:-1], keys[1:])))
    geo["ильинская ниже кровли фундамента"] = int((r.il_bot_depth > r.basement_depth + 0.5).sum())
    tsum = sum(r[f"{k}_thick"].fillna(0) for k in keys)
    # юсмастахская в пять свит списка не входит, но в мощность рифея входит
    full = ((r[[f"{k}_present" for k in keys]].all(axis=1) | (r.rif_thick <= 1))
            & (r.surf_model != "юсмастахская"))
    geo["сумма мощностей ≠ мощность рифея (>5 м), ячеек без юсмастахской"] = int(
        ((tsum - r.rif_thick).abs() > 5)[full].sum())
    print(f"\n(д) Геометрическая согласованность (ячеек {len(r)}): {geo}")
    checks["д"] = geo
    CHECKS.write_text(json.dumps(checks, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"таблица: {OUT}")

    lines = profile_lines(lab, az_mean)
    for p in (draw_distribution(g, meta, origin),
              draw_profiles(lines, dem, lab, f, off, cm_plane, misfit, az_mean),
              draw_thickness(g, meta, origin, lines),
              draw_residuals(g, meta, origin, con, rc)):
        log(f"рисунок: {p}")


if __name__ == "__main__":
    main()
