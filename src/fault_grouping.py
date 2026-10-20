"""Группировка отдельных линейных объектов в разломные зоны — задача №3.

Задача реестра: «использовать нейросеть для группирования разломов с
геологической карты и формирования разломных зон, обобщающих группы
однонаправленных разломов».

ГДЕ БЛОК И ПОЧЕМУ ЭТОТ МОДУЛЬ НЕ О НЕЙРОСЕТИ. Слоя отдельных разломов в
переданных данных нет: ``glub_raz_nw`` — 4 полигона, ``glub_r_nw`` — 2, и это
уже результат искомого обобщения (обоснование см. в блоке ``FG_*``
:mod:`src.config`). Нейросеть по существу нужна на ДРУГОМ шаге — векторизации
разломов с георастра ГГК-200, где она учит рисунок условного знака; сама же
группировка при 150-1000 отрезков — это кластеризация, и сеть на ней была бы
k-means с лишними ручками и без способа себя проверить: меток нет.

Поэтому здесь сначала строится и ИЗМЕРЯЕТСЯ группировка на популяции, которая
в данных уже есть — на дайках (заполненные трещины, 151 объект, роза по длине
повторяет СВ-систему разломов). Если группировка не воспроизводит переданные
зоны там, где ответ известен заранее, вкладываться в векторизацию растра
незачем. Модуль умышленно принимает любую геометрию (полигон, линию), чтобы тот
же метод без правок принял вектор разломов, когда он появится.

МЕТОД — ДВА ЭТАПА ПО ТЕКСТУ ЗАДАЧИ.

1. ОДНОНАПРАВЛЕННЫЕ ГРУППЫ. Азимут оси — величина ОСЕВАЯ (0° и 180° — одно и то
   же направление), поэтому распределение считается на удвоенном угле; тот же
   приём и по той же причине, что в :mod:`src.features_v11`. Моды взвешенного
   длиной кругового KDE и есть направления.
2. ЗОНЫ ВНУТРИ ГРУППЫ. Объекты одного направления разносит по площади
   перпендикулярный снос ``rho``; сгущение по нему — это и есть зона.
   Кластеризация одномерная (DBSCAN по ``rho``), поэтому «обменного курса»
   между километрами и градусами не возникает. Вдоль простирания зона не
   режется ничем: коридор карты протянут через весь лист, и обрезать его
   означало бы навязать масштаб.

ЧЕГО МЕТОД НАМЕРЕННО НЕ ДЕЛАЕТ. Ширина зоны не задаётся: она получается из
разброса самих объектов поперёк простирания и снизу ограничена лишь
:data:`config.FG_MIN_ZONE_WIDTH_M`. Ширина коридоров карты (8-21 км) — то, с чем
результат сверяется на выходе, и подставлять её на входе значило бы измерить
собственную подсказку.

ПРОТОКОЛ СВЕРКИ — тот же, что у задачи №2, и это не экономия, а условие
сравнимости чисел: lift@площадь ячеек коридора под картой зон, тороидальный
сдвиговый null (:func:`src.fracture_zones.fault_agreement`), контроль рельефа
вычитанием (:func:`src.fracture_zones.screen_candidate`) и накопительный
знаменатель Бонферрони. Добавлена одна метрика, которой у задачи №2 не было:
IoU бинарной маски зон с коридором (:func:`mask_iou`). Она отвечает на вопрос
задачи прямее, чем lift: «совпали ли ПЛОЩАДИ», а не «попали ли верхние 10%
карты в цель». Значимость у неё меряется тем же сдвигом
(:func:`mask_iou_pvalue`).

Порядок вызовов::

    axial_segments -> direction_modes -> group_zones -> zone_field / zone_mask
                                                     -> mask_iou / mask_iou_pvalue
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

#: Колонки таблицы осей, которую возвращает :func:`axial_segments`.
SEGMENT_COLS: tuple[str, ...] = ("x", "y", "az_deg", "length_m", "width_m")


# --------------------------------------------------------------- геометрия оси

def axial_segments(geoms) -> pd.DataFrame:
    """Ось каждой геометрии: центр, азимут (0-180° от севера), длина, ширина.

    Ось снимается описанным прямоугольником минимальной площади. Для дайки-
    полигона (буферный коридор, удлинение медианно 6.7) это устойчивее, чем
    главная компонента вершин: вершины сгущены на изгибах, и PCA тянет ось к
    ним. Ширина считается как ``площадь / длина``, а не как короткая сторона
    прямоугольника: у изогнутого объекта прямоугольник заметно шире самого
    объекта, а отношение площадей от изгиба почти не зависит.

    Азимут ОСЕВОЙ: 0-180°, отсчёт от севера по часовой стрелке. Геометрия может
    быть любой (полигон, линия) — метод не знает, откуда взялся объект, и тот же
    вызов примет вектор разломов, когда он появится.
    """
    rows = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        rect = g.minimum_rotated_rectangle
        # У прямого отрезка описанный прямоугольник вырождается в линию, и
        # ``exterior`` у него нет. Это не экзотика, а ровно тот случай, ради
        # которого модуль принимает линии: прямой разлом на карте рисуется
        # именно так, и молча терять его нельзя.
        ring = getattr(rect, "exterior", rect)
        pts = np.asarray(ring.coords, dtype=float)[:, :2]
        if pts.shape[0] > 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]                       # замыкающая вершина кольца
        if pts.shape[0] < 2:
            continue
        edges = np.roll(pts, -1, axis=0) - pts
        if pts.shape[0] == 2:                    # отрезок: ребро одно
            edges = edges[:1]
        lens = np.hypot(edges[:, 0], edges[:, 1])
        k = int(np.argmax(lens))
        length = float(lens[k])
        if length <= 0:
            continue
        area = float(g.area) if g.area > 0 else float(rect.area)
        dx, dy = edges[k]
        rows.append({
            "x": float(g.centroid.x), "y": float(g.centroid.y),
            "az_deg": float(np.degrees(np.arctan2(dx, dy)) % 180.0),
            "length_m": length,
            "width_m": area / length if length else np.nan,
        })
    return pd.DataFrame(rows, columns=list(SEGMENT_COLS))


def along_rho(x, y, az_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Координаты в осях зоны: ``along`` вдоль простирания, ``rho`` поперёк.

    Азимут отсчитывается от севера, оси сетки — восток/север, поэтому единичный
    вектор простирания ``u = (sin a, cos a)``, а нормаль ``n = (cos a, -sin a)``.
    """
    a = np.radians(az_deg)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * np.sin(a) + y * np.cos(a), x * np.cos(a) - y * np.sin(a)


def axial_diff_deg(a, b):
    """Осевая разность азимутов в градусах, 0-90 (180° — то же направление)."""
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 90.0) % 180.0 - 90.0
    return np.abs(d)


# --------------------------------------------------------- этап 1: направления

def direction_modes(az_deg, weights=None, bandwidth_deg: float | None = None,
                    grid_deg: float | None = None) -> dict:
    """Моды осевого распределения азимутов (взвешенный круговой KDE).

    Считается на УДВОЕННОМ угле: это стандартный способ работать с осевой
    величиной, при котором 179° и 1° оказываются соседями, а не краями шкалы
    (в проекте он уже принят для направления поля, см. :mod:`src.features_v11`).
    Ядро — обёрнутое гауссово с сигмой ``2 x bandwidth`` в удвоенном
    пространстве, что соответствует ``bandwidth`` по самому азимуту.

    Вес по умолчанию — длина объекта: группа из десятка коротких даек и группа
    из одной длинной не равнозначны, а роза по числу объектов их уравнивает.

    Возвращает сетку, плотность, азимуты мод и их веса. Порог «мода или шум»
    здесь не применяется намеренно: моды отбраковываются позже, по числу членов
    зоны (:data:`config.FG_MIN_ZONE_MEMBERS`), то есть по факту, а не по высоте
    сглаженной кривой.
    """
    bw = config.FG_AZ_BANDWIDTH_DEG if bandwidth_deg is None else bandwidth_deg
    step = config.FG_AZ_GRID_DEG if grid_deg is None else grid_deg
    az = np.asarray(az_deg, dtype=float)
    w = np.ones_like(az) if weights is None else np.asarray(weights, dtype=float)
    if az.size == 0:
        return {"grid_deg": np.empty(0), "density": np.empty(0),
                "modes_deg": np.empty(0), "modes_weight": np.empty(0)}

    grid = np.arange(0.0, 180.0, step)
    # Разность на удвоенном угле = 2 x осевая разность, поэтому и сигма берётся
    # в тех же удвоенных единицах.
    d2 = 2.0 * axial_diff_deg(az[None, :], grid[:, None])
    dens = (w[None, :] * np.exp(-0.5 * (d2 / (2.0 * bw)) ** 2)).sum(axis=1)

    left, right = np.roll(dens, 1), np.roll(dens, -1)     # круговые соседи
    peak = np.where((dens > left) & (dens >= right))[0]
    modes = grid[peak]

    lab = assign_modes(az, modes) if modes.size else np.empty(0, dtype=int)
    mw = (np.array([w[lab == i].sum() for i in range(modes.size)])
          if modes.size else np.empty(0))
    order = np.argsort(-mw)
    return {"grid_deg": grid, "density": dens,
            "modes_deg": modes[order], "modes_weight": mw[order]}


def assign_modes(az_deg, modes_deg) -> np.ndarray:
    """Номер ближайшей по осевой разности моды для каждого объекта."""
    az = np.asarray(az_deg, dtype=float)
    modes = np.asarray(modes_deg, dtype=float)
    if modes.size == 0:
        return np.full(az.shape, -1, dtype=int)
    return np.argmin(axial_diff_deg(az[:, None], modes[None, :]), axis=1)


# --------------------------------------------------------------- этап 2: зоны

@dataclass
class Zone:
    """Полоса вдоль простирания: направление, положение поперёк, охват вдоль."""

    az_deg: float
    rho_min: float
    rho_max: float
    along_min: float
    along_max: float
    weight: float                      # суммарная длина объектов, м
    members: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def n(self) -> int:
        return int(self.members.size)

    @property
    def rho_c(self) -> float:
        return 0.5 * (self.rho_min + self.rho_max)

    @property
    def along_c(self) -> float:
        return 0.5 * (self.along_min + self.along_max)

    @property
    def half_width(self) -> float:
        """Половина ширины зоны, не меньше пола :data:`config.FG_MIN_ZONE_WIDTH_M`.

        Пол нужен потому, что у идеально коллинеарной группы разброс поперёк
        нулевой, и зона выродилась бы в линию нулевой площади — то есть в
        линеамент, а задача просит именно площадной объект.
        """
        return max(0.5 * (self.rho_max - self.rho_min),
                   0.5 * config.FG_MIN_ZONE_WIDTH_M)

    @property
    def half_length(self) -> float:
        return 0.5 * (self.along_max - self.along_min)


def zones_of_mode(seg: pd.DataFrame, az_mode: float, idx=None,
                  eps_m: float | None = None,
                  min_members: int | None = None) -> list[Zone]:
    """Зоны одной однонаправленной группы: сгущения по перпендикулярному сносу.

    Кластеризация одномерная (DBSCAN по ``rho``) — сознательно, а не за
    неимением лучшего: положение объекта ВДОЛЬ простирания к принадлежности
    зоне отношения не имеет, и добавление его в метрику разорвало бы длинный
    коридор на куски там, где в нём просто разрежены дайки.

    Охват вдоль простирания берётся с учётом собственной длины объектов
    (центр ± L/2), иначе зона из двух длинных даек оказалась бы короче их самих.
    """
    from sklearn.cluster import DBSCAN

    eps = config.FG_RHO_EPS_M if eps_m is None else eps_m
    mmin = config.FG_MIN_ZONE_MEMBERS if min_members is None else min_members
    idx = np.arange(len(seg)) if idx is None else np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []

    sub = seg.iloc[idx]
    along, rho = along_rho(sub["x"].to_numpy(), sub["y"].to_numpy(), az_mode)
    half = 0.5 * sub["length_m"].to_numpy()
    labels = DBSCAN(eps=eps, min_samples=mmin).fit(rho.reshape(-1, 1)).labels_

    zones = []
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        if m.sum() < mmin:
            continue
        zones.append(Zone(
            az_deg=float(az_mode),
            rho_min=float(rho[m].min()), rho_max=float(rho[m].max()),
            along_min=float((along[m] - half[m]).min()),
            along_max=float((along[m] + half[m]).max()),
            weight=float(sub["length_m"].to_numpy()[m].sum()),
            members=idx[m],
        ))
    return zones


def zones_of_mode_kde(seg: pd.DataFrame, az_mode: float, idx=None,
                      bw_m: float | None = None, step_m: float | None = None,
                      min_members: int | None = None) -> list[Zone]:
    """То же, но зона — СГУЩЕНИЕ по ``rho``, а не связная компонента.

    Зачем нужен второй оператор, видно из измерения самих популяций:
    :func:`zones_of_mode` делит моду по РАЗРЫВАМ в ``rho``, а разрывы есть только
    у разреженного входа. У 3087 рельефных линеаментов медианный зазор — десятки
    метров, зазоров шире порога слияния нет вовсе, и связная компонента ровно
    одна на всю моду. Здесь граница зоны проходит не по пустому месту, а по
    ДОЛИНЕ взвешенной длиной плотности ``rho``: пик плотности — тело зоны,
    локальный минимум — её край. Такой оператор определён на любом входе, а на
    разреженном сводится к первому, потому что там долины и есть разрывы.

    Полоса ширины ядра равна порогу слияния первого оператора
    (:data:`config.FG_RHO_KDE_BW_M`), то есть физический масштаб у операторов
    один и тот же и различаются они только правилом разреза.
    """
    bw = config.FG_RHO_KDE_BW_M if bw_m is None else bw_m
    step = config.FG_RHO_KDE_STEP_M if step_m is None else step_m
    mmin = config.FG_MIN_ZONE_MEMBERS if min_members is None else min_members
    idx = np.arange(len(seg)) if idx is None else np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []

    sub = seg.iloc[idx]
    along, rho = along_rho(sub["x"].to_numpy(), sub["y"].to_numpy(), az_mode)
    w = sub["length_m"].to_numpy()
    half = 0.5 * w

    grid = np.arange(rho.min() - bw, rho.max() + bw + step, step)
    dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - rho[None, :]) / bw) ** 2)
            ).sum(axis=1)
    # Долины между пиками: границы зон. Края сетки — тоже границы, иначе крайние
    # объекты остались бы без зоны.
    lo, hi = dens[:-2], dens[2:]
    valley = np.where((dens[1:-1] <= lo) & (dens[1:-1] < hi))[0] + 1
    edges = np.concatenate([[grid[0]], grid[valley], [grid[-1]]])

    zones = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rho >= a) & (rho < b) if b < edges[-1] else (rho >= a) & (rho <= b)
        if m.sum() < mmin:
            continue
        zones.append(Zone(
            az_deg=float(az_mode),
            rho_min=float(rho[m].min()), rho_max=float(rho[m].max()),
            along_min=float((along[m] - half[m]).min()),
            along_max=float((along[m] + half[m]).max()),
            weight=float(w[m].sum()),
            members=idx[m],
        ))
    return zones


#: Операторы этапа 2, оба пре-регистрированы: разрез по разрывам и по долинам
#: плотности. Имена — ключи в таблицах прогона.
ZONE_OPERATORS = {"dbscan": zones_of_mode, "kde": zones_of_mode_kde}


def group_zones(seg: pd.DataFrame, weights=None, operator: str = "dbscan",
                **kw) -> tuple[list[Zone], dict]:
    """Полный проход: направления -> зоны. Возвращает ``(зоны, сводка по модам)``."""
    w = seg["length_m"].to_numpy() if weights is None else np.asarray(weights)
    zones_of = ZONE_OPERATORS[operator]
    modes = direction_modes(seg["az_deg"].to_numpy(), weights=w, **kw)
    lab = assign_modes(seg["az_deg"].to_numpy(), modes["modes_deg"])
    zones: list[Zone] = []
    for i, az in enumerate(modes["modes_deg"]):
        zones.extend(zones_of(seg, float(az), idx=np.where(lab == i)[0]))
    modes = dict(modes)
    modes["labels"] = lab
    return zones, modes


# -------------------------------------------------------------------- растры

def _zone_distances(zone: Zone, meta) -> tuple[np.ndarray, np.ndarray]:
    """Расстояния ячеек до тела зоны: поперёк и вдоль (внутри — нули)."""
    X, Y = meta.cell_centers()
    along, rho = along_rho(X, Y, zone.az_deg)
    d_perp = np.maximum(np.abs(rho - zone.rho_c) - zone.half_width, 0.0)
    d_along = np.maximum(np.abs(along - zone.along_c) - zone.half_length, 0.0)
    return d_perp, d_along


def zone_mask(zones: list[Zone], meta) -> np.ndarray:
    """Бинарная маска «ячейка внутри какой-нибудь зоны», формы ``meta.shape``."""
    out = np.zeros(meta.shape, dtype=bool)
    for z in zones:
        d_perp, d_along = _zone_distances(z, meta)
        out |= (d_perp <= 0) & (d_along <= 0)
    return out


def zone_field(zones: list[Zone], meta) -> np.ndarray:
    """Непрерывная карта зон: вклад каждой зоны с гауссовым спадом от её тела.

    Нужна затем, что бинарная маска — карта с одним значением внутри, и lift на
    ней меряет только реализованную площадь. Спад делает шкалу непрерывной,
    сохраняя порядок «зона > окрестность > далеко», а вес зоны (суммарная длина
    объектов) отличает мощную группу от случайной тройки. Сигма спада — половина
    собственной ширины зоны, то есть масштаб берётся из данных, а не назначается.

    Метрики протокола ранговые, поэтому конкретная форма спада на результат не
    влияет — важен только порядок ячеек.

    ОГОВОРКА, ИЗМЕРЕННАЯ НА ПИЛОТЕ ЗАДАЧИ №3. Порядка ВНУТРИ тела зоны это поле
    не задаёт: спад начинается за границей, внутри значение постоянно. Поэтому
    верхние проценты карты определяются исключительно хвостами соседних зон, и
    lift@площадь на такой карте меряет не зону, а её ближайшее соседство (у
    СВ-кандидата пилота — lift 0.000 при непересекающихся зонах). Для карт зон
    читаемая метрика — :func:`mask_iou` по маске, а не lift по полю.
    """
    out = np.zeros(meta.shape, dtype=float)
    for z in zones:
        d_perp, d_along = _zone_distances(z, meta)
        sigma = max(z.half_width, 0.5 * config.FG_MIN_ZONE_WIDTH_M)
        out += z.weight * np.exp(
            -0.5 * ((d_perp / sigma) ** 2 + (d_along / sigma) ** 2))
    return out


# ------------------------------------------------------------------- метрики

def mask_iou(mask, target_idx, pool=None) -> float:
    """IoU маски зон с ячейками цели.

    Прямой ответ на вопрос задачи — «совпали ли площади» — в отличие от lift,
    который меряет попадание верхних процентов карты. У IoU нет свободы выбора
    площади: карта отвечает за свой размер сама, и зона размером в лист будет
    наказана знаменателем.
    """
    flat = np.asarray(mask).ravel().astype(bool)
    tgt = np.zeros(flat.shape, dtype=bool)
    tgt[np.asarray(target_idx, dtype=int)] = True
    if pool is not None:
        keep = np.zeros(flat.shape, dtype=bool)
        keep[np.asarray(pool, dtype=int)] = True
        flat, tgt = flat & keep, tgt & keep
    union = (flat | tgt).sum()
    return float((flat & tgt).sum() / union) if union else 0.0


def mask_iou_pvalue(mask, target_idx, meta, pool=None,
                    n_shifts: int | None = None,
                    seed: int | None = None) -> dict[str, float]:
    """Значимость IoU тороидальным сдвигом маски.

    Схема сдвига дословно та же, что у :func:`src.assessment.spatial_null_pvalue`
    (отражения плюс циклический сдвиг по обеим осям), и по той же причине:
    двигается КАНДИДАТ, цель и пул стоят, поэтому null сохраняет и размер, и
    форму, и связность зон. Перестановка ячеек дала бы null, у которого зон нет
    вовсе, и любой связный кандидат оказался бы «значим».
    """
    n_shifts = config.FG_N_SHIFTS if n_shifts is None else n_shifts
    rng = np.random.default_rng(config.VER_SEED if seed is None else seed)
    img = np.asarray(mask, dtype=bool).reshape(meta.shape)

    observed = mask_iou(img, target_idx, pool)
    null = np.empty(n_shifts)
    for i in range(n_shifts):
        a = img
        if rng.integers(2):
            a = np.flipud(a)
        if rng.integers(2):
            a = np.fliplr(a)
        a = np.roll(a, (int(rng.integers(meta.prf)), int(rng.integers(meta.pic))),
                    axis=(0, 1))
        null[i] = mask_iou(a, target_idx, pool)
    return {"observed": observed, "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)),
            "p": float((1 + (null >= observed).sum()) / (1 + n_shifts))}


def zone_table(zones: list[Zone]) -> pd.DataFrame:
    """Сводка по зонам: направление, размеры, вес — для печати и сверки с картой."""
    return pd.DataFrame([{
        "азимут": z.az_deg,
        "объектов": z.n,
        "длина_км": 2 * z.half_length / 1000.0,
        "ширина_км": 2 * z.half_width / 1000.0,
        "суммарная_длина_км": z.weight / 1000.0,
    } for z in zones])
