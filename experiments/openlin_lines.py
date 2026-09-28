"""Трассировка линеаментов по скору: из площади в список линий.

ЗАЧЕМ. До сих пор ответом модели была площадь — верхние 10% ячеек, — и
сравнивалась она с разметкой тоже по площади. Геологу нужен не процент площади,
а линии: где проходит нарушение, какой длины и какого направления. Здесь скор
превращается в список веток, а качество меряется между линиями:

* ``точность`` — какая доля длины выделенных линий легла в допуск от карты;
* ``полнота``  — какая доля длины карты покрыта выделенными линиями.

ЧТО ВАЖНО ПОМНИТЬ ПРО БИНАРИЗАЦИЮ. В этой же ветке уже показано, что список
линий Canny->Hough не переживает пересчёт на другой выборке (Спирмен
+0.23...-0.61 по четырём сенсорам), тогда как та же картинка как непрерывное
поле когерентности переживает (0.61-0.99): ломался именно шаг бинаризации, а не
данные. Трассировка — снова бинаризация, поэтому её устойчивость проверяется
здесь же (``--stability``): модель переобучается на другой подвыборке строк, и
два списка линий сравниваются между собой той же мерой, что и с картой.

КАК ТРАССИРУЕТСЯ. Скор сглаживается, по гессиану ищется направление поперёк
линии, немаксимумы в этом направлении подавляются (остаётся тонкий гребень),
дальше гистерезисный порог по двум квантилям площади, скелетизация, разрез в
узлах — и каждая ветка проверяется на длину и прямолинейность.

Запуск из корня репозитория:
    python -m experiments.openlin_lines --site nevada --apply-to anabar,anabar_dykes
    python -m experiments.openlin_lines --site anabar --stability
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import (N_TRAIN_ROWS, SEED, TARGET_HALFWIDTH_M,
                                       TOL_M, anfis_predict, baseline_predict,
                                       block_split, fit_anfis, fit_baselines,
                                       rows, score_map, soft_target)
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, load_site

M = Path("outputs/metrics")

# --------------------------------------------------------------------------
# Параметры трассировки
# --------------------------------------------------------------------------

#: Сглаживание скора перед поиском гребня, м. 300 м — вдвое шире полуширины
#: разметки (150 м): уже — и гребень рвётся на шуме модели, шире — и две
#: соседние линии сливаются в одну.
LINE_SIGMA_M = 300.0

#: Гистерезис в долях площади: ядро линии берётся из верхних HI, продолжение
#: дотягивается по верхним LO, если связано с ядром. Один порог вместо двух рвал
#: бы линию там, где модель просела на одной ячейке.
HI_AREA = 0.05
LO_AREA = 0.15

#: Отсев веток. Короче 3 км — не линеамент, а пятно текстуры (на 100-метровой
#: сетке это 30 ячеек). Прямолинейность — отношение размаха вдоль главной оси к
#: длине пути; 0.8 отсекает извилистые обводки контрастных пятен, оставляя дуги
#: разломов, которые тоже не идеально прямые.
MIN_LEN_M = 3000.0
MIN_STRAIGHT = 0.8

#: Длина отростка, который состригается со скелета перед разрезом в узлах.
#: 1 км — заметно меньше минимальной длины ветки, то есть отросток такой длины
#: сам по себе линеаментом стать не может, но узел от него рубит настоящий.
PRUNE_M = 1000.0

#: Сшивка разрывов. Гребень скора рвётся везде, где модель просела на сотне
#: метров, и без сшивки одна линия распадается на обрубки, каждый из которых по
#: отдельности короче ``MIN_LEN_M``. Сшиваются только концы, которые смотрят
#: друг на друга: разрыв не длиннее 2 км И направление разрыва отличается от
#: направления обеих веток не больше чем на 20 градусов. Это не смягчение
#: порога — порог длины остаётся прежним, восстанавливается связность.
LINK_GAP_M = 2000.0
LINK_ANGLE_DEG = 20.0

#: Плотность трассировки — это НЕ свойство метода, а запрос пользователя, и
#: поэтому вынесена в параметр. Пороги выше отвечают на вопрос «дай только то, в
#: чём уверены» и дают полтора десятка отрезков на лист — как карта линеаментов
#: это бесполезно, геологу нужны сотни, а отбор он делает сам по длине и
#: уверенности. Пресеты ниже задают шкалу от «редко» до «густо»; для каждого
#: считается своя точность, и кривая «точность против числа линий» показывается
#: целиком, чтобы выбор плотности не выглядел подгонкой под результат.
#: Поля: (sigma сглаживания, HI, LO, минимальная длина, прямолинейность).
#: Два самых плотных пресета добавлены под задачу выделения линеаментов: там
#: мелкие короткие линии — часть ответа, а не мусор. Докуда уплотнять, решает не
#: вкус, а стоп-правило: взаимное покрытие независимых приборов (оптика против
#: радара) должно оставаться выше случайного уровня, см. openlin_unsup --sweep.
DENSITY = {
    "редко":       (300.0, 0.05, 0.15, 3000.0, 0.80),
    "средне":      (200.0, 0.15, 0.35, 1500.0, 0.70),
    "густо":       (150.0, 0.25, 0.50, 1000.0, 0.60),
    "очень густо": (100.0, 0.35, 0.65,  600.0, 0.50),
    "предельно":   ( 75.0, 0.45, 0.80,  400.0, 0.45),
    # Ступень под задачу «выделить всё, что есть»: пороги ослаблены до предела,
    # на котором линии ещё остаются линиями. На Анабаре даёт 1747 веток против
    # 1217 у «предельно», прирост идёт за счёт слабых и коротких.
    "максимум":    ( 75.0, 0.60, 0.90,  300.0, 0.40),
}
DENSITY_DEFAULT = "густо"

N_SHIFTS = 4000

#: Пре-регистрация. Свой addend в своей секции, как принято в этой ветке: на
#: числа предыдущих прогонов уже ссылаются, и складывать их в одну константу
#: нельзя. Список кандидатов тот же самый — 7 пар (площадка обучения, площадка
#: оценки) x 3 модели x 2 таргета = 42, — меняется только форма ответа.
#: ПЕРВИЧНАЯ МЕТРИКА объявлена до просмотра результатов: ``точность`` (доля
#: длины выделенных линий в допуске от карты). Полнота считается и печатается,
#: но по ней тест не проводится — иначе на каждую клетку приходилось бы по две
#: попытки.
LN_N_TESTS_PRIOR = 84          # 42 поячеечных + 42 с окрестностью
LN_N_TESTS = 42
LN_ALPHA = 0.05 / (LN_N_TESTS_PRIOR + LN_N_TESTS)

#: Достижимость: минимальное p сдвигового нуля = 1/(1+4000) = 0.00025 против
#: alpha = 0.00040, запас 1.6x. Меньше 2500 сдвигов — и тест стал бы
#: непроходимым при любом результате.
assert 1.0 / (1 + N_SHIFTS) < LN_ALPHA, "порог недостижим при таком числе сдвигов"


# --------------------------------------------------------------------------
# Трассировка
# --------------------------------------------------------------------------

def _ridge_nms(score: np.ndarray, valid: np.ndarray, sigma_px: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Тонкий гребень скора: подавление немаксимумов поперёк линии.

    Возвращает ``(сглаженный скор, маска гребня)``. Направление поперёк линии —
    собственный вектор гессиана с самым отрицательным числом: вдоль линии
    кривизна почти нулевая, поперёк — максимальная по модулю и отрицательная.
    Сглаживание нормированное на маску, иначе пустые углы повёрнутого кадра
    дали бы у границы ложный гребень.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    num = gaussian_filter(np.where(valid, score, 0.0).astype("float32"), sigma_px)
    den = gaussian_filter(valid.astype("float32"), sigma_px)
    s = num / np.maximum(den, 1e-3)

    hxx = gaussian_filter(s, sigma_px, order=(0, 2))
    hyy = gaussian_filter(s, sigma_px, order=(2, 0))
    hxy = gaussian_filter(s, sigma_px, order=(1, 1))
    half = 0.5 * (hxx + hyy)
    rad = np.sqrt((0.5 * (hxx - hyy)) ** 2 + hxy ** 2)
    lam_min = half - rad                       # самое отрицательное число

    # Собственный вектор для lam_min у матрицы [[hxx, hxy], [hxy, hyy]] — это
    # (hxy, lam_min - hxx) в порядке (x, y). ДО 23.09.2026 компоненты стояли
    # переставленными (vx = lam_min - hxx, vy = hxy), то есть направление
    # подавления было отражено относительно диагонали: у линий под 45° оно
    # почти совпадает с верным, а у линий ВДОЛЬ ОСЕЙ РАСТРА поворачивалось на
    # 90°, и немаксимумы подавлялись вдоль гребня вместо поперёк. На чистой
    # синтетике это незаметно (вдоль гребня отклик постоянен, сравнение даёт
    # равенство), а на шумном фоне половина ячеек гребня выбрасывалась.
    # ИЗМЕРЕНО на изотропном фоне, покрытие следа врезки маской гребня:
    # азимут   7° 39 % -> 96 %, 82° 48 % -> 93 %, 97° 43 % -> 87 %,
    # 172° 47 % -> 87 %, при этом 52° 94 % -> 96 % (верные азимуты не портятся).
    # Следствие для ветки: до этой правки трассировка была слепа к северо-южным
    # и широтным линеаментам, и все прежние популяции простираний цензурированы.
    vx = hxy
    vy = lam_min - hxx
    nrm = np.hypot(vx, vy)
    # вырождение: hxy = 0 и lam_min = hxx, то есть гессиан диагонален и самая
    # отрицательная кривизна вдоль x — поперёк линии тогда именно x
    flat = nrm < 1e-12
    vx = np.where(flat, 1.0, vx / np.maximum(nrm, 1e-12))
    vy = np.where(flat, 0.0, vy / np.maximum(nrm, 1e-12))

    yy, xx = np.indices(s.shape).astype("float32")
    fwd = map_coordinates(s, [yy + vy, xx + vx], order=1, mode="nearest")
    bwd = map_coordinates(s, [yy - vy, xx - vx], order=1, mode="nearest")
    ridge = valid & (lam_min < 0) & (s >= fwd) & (s >= bwd)
    return s, ridge


def _components(seg: np.ndarray) -> list[np.ndarray]:
    from scipy.ndimage import label

    lab, n = label(seg, structure=np.ones((3, 3), dtype=int))
    if not n:
        return []
    order = np.argsort(lab.ravel(), kind="stable")
    vals = lab.ravel()[order]
    starts = np.searchsorted(vals, np.arange(1, n + 1), side="left")
    ends = np.searchsorted(vals, np.arange(1, n + 1), side="right")
    return [np.stack(np.unravel_index(order[a:b], seg.shape), axis=1)
            for a, b in zip(starts, ends)]


def _degree(sk: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    k = np.ones((3, 3), dtype=int)
    k[1, 1] = 0
    return convolve(sk.astype(int), k, mode="constant")


#: На сколько градусов направления двух сегментов могут расходиться, чтобы
#: считать их одной линией, проходящей через узел. 35 градусов: у пересечения
#: двух разных линий расхождение близко к прямому углу, у продолжения одной и
#: той же — в пределах точности скелета.
JOIN_ANGLE_DEG = 35.0


def _end_dirs(pts: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Два конца сегмента и направления выхода из них (наружу от сегмента)."""
    if len(pts) == 1:                       # сегмент в одну ячейку
        d = np.array([1.0, 0.0])
        return np.stack([pts[0], pts[0]]), np.stack([d, -d])
    c = (pts - pts.mean(0)).astype("float64")
    _u, _sv, vt = np.linalg.svd(c, full_matrices=False)
    order = np.argsort(c @ vt[0])
    a, b = pts[order[0]], pts[order[-1]]
    k = max(min(k, len(pts) - 1), 1)
    da = (pts[order[0]] - pts[order[k]]).astype("float64")
    db = (pts[order[-1]] - pts[order[-1 - k]]).astype("float64")
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    da = da / na if na else np.array([1.0, 0.0])
    db = db / nb if nb else np.array([1.0, 0.0])
    return np.stack([a, b]), np.stack([da, db])


def _join_through_nodes(sk: np.ndarray, segs: list[np.ndarray],
                        max_angle: float = JOIN_ANGLE_DEG
                        ) -> list[list[int]]:
    """Сцепить сегменты, которые продолжают друг друга через узел.

    Разрез скелета в каждом узле рубит длинную линию всюду, где её пересекает
    другая: на густой сети средняя ветка получается вдвое короче, чем сама
    линия. Здесь в каждом узле сегменты соединяются попарно по направлению —
    продолжение проходит насквозь, а пересечение остаётся пересечением.
    """
    from scipy.ndimage import label

    deg = _degree(sk)
    nodes, n_nodes = label(sk & (deg > 2), structure=np.ones((3, 3), dtype=int))
    if not n_nodes:
        return [[i] for i in range(len(segs))]

    # какие концы сегментов примыкают к каким узлам
    at_node: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    h, w = sk.shape
    for i, pts in enumerate(segs):
        ends, dirs = _end_dirs(pts)
        for e in (0, 1):
            r, c = int(ends[e][0]), int(ends[e][1])
            lab = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and nodes[rr, cc]:
                        lab = int(nodes[rr, cc])
            if lab:
                at_node.setdefault(lab, []).append((i, e, dirs[e]))

    parent = list(range(len(segs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used: set[tuple[int, int]] = set()
    for lab, items in at_node.items():
        pairs = []
        for p in range(len(items)):
            for q in range(p + 1, len(items)):
                ip, ep, dp = items[p]
                iq, eq, dq = items[q]
                if ip == iq:
                    continue
                # продолжение: выход одного противоположен выходу другого
                cosang = float(np.clip(-np.dot(dp, dq), -1.0, 1.0))
                angle = np.degrees(np.arccos(cosang))
                if angle <= max_angle:
                    pairs.append((angle, p, q))
        for _angle, p, q in sorted(pairs):
            ip, ep, _d = items[p]
            iq, eq, _d2 = items[q]
            if (ip, ep) in used or (iq, eq) in used:
                continue
            if find(ip) == find(iq):
                continue
            used.add((ip, ep))
            used.add((iq, eq))
            parent[find(ip)] = find(iq)

    groups: dict[int, list[int]] = {}
    for i in range(len(segs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _branches(mask: np.ndarray, res_m: float,
              prune_m: float = PRUNE_M,
              join: bool = False) -> list[np.ndarray]:
    """Скелет маски, разрезанный в узлах; каждая ветка — массив координат.

    Перед разрезом короткие отростки состригаются. Без этого один шип длиной в
    три ячейки создаёт узел и разрубает длинную линию пополам — трассировка
    рассыпается на обрубки, которые потом не проходят отсев по длине.

    ОТРОСТОК — это сегмент, у которого один конец свободный, а другой упирается
    в узел. Сегмент с ДВУМЯ свободными концами отростком не является: это
    самостоятельная короткая линия, и решать её судьбу должен отсев по длине, а
    не состригание. Раньше условие было «есть хотя бы один свободный конец», и
    на Анабаре это стирало 823 км целых линий длиной 400–1000 м — то есть тех,
    которые по порогу длины (400 м) обязаны были пройти.
    """
    from skimage.morphology import skeletonize

    sk = skeletonize(mask)
    n_prune = max(1, int(round(prune_m / res_m)))
    for _ in range(3):
        deg = _degree(sk)
        spur = np.zeros_like(sk)
        for pts in _components(sk & (deg <= 2)):
            ends = int((deg[pts[:, 0], pts[:, 1]] == 1).sum())
            if len(pts) <= n_prune and ends == 1:
                spur[pts[:, 0], pts[:, 1]] = True
        if not spur.any():
            break
        sk &= ~spur

    deg = _degree(sk)
    seg = sk & (deg <= 2)                       # узлы (степень >2) выкусываем
    from scipy.ndimage import label

    lab, n = label(seg, structure=np.ones((3, 3), dtype=int))
    if not n:
        return []
    order = np.argsort(lab.ravel(), kind="stable")
    vals = lab.ravel()[order]
    starts = np.searchsorted(vals, np.arange(1, n + 1), side="left")
    ends = np.searchsorted(vals, np.arange(1, n + 1), side="right")
    segs = [np.stack(np.unravel_index(order[a:b], mask.shape), axis=1)
            for a, b in zip(starts, ends)]
    if not join:
        return segs
    return [np.concatenate([segs[i] for i in grp])
            for grp in _join_through_nodes(sk, segs)]


def _path_length(pts: np.ndarray, res_m: float) -> float:
    """Длина пути по ветке, м: диагональный шаг длиннее ортогонального в 1.41.

    Считать длину числом ячеек нельзя — у диагональной линии это занижает путь в
    1.41 раза, и прямолинейность (размах / путь) вылезает выше единицы.
    """
    have = {(int(y), int(x)) for y, x in pts}
    tot = 0.0
    for y, x in have:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (y + dy, x + dx) in have:
                tot += 1.4142135 if dy and dx else 1.0
    return tot * res_m


def _axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Главная ось ветки и два её крайних пикселя вдоль этой оси."""
    c = (pts - pts.mean(0)).astype("float64")
    _u, _sv, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt[0]
    return vt[0], pts[int(proj.argmin())], pts[int(proj.argmax())]


def _link_gaps(branches: list[np.ndarray], shape: tuple[int, int],
               res_m: float) -> np.ndarray:
    """Маска отрезков, сшивающих коллинеарные концы соседних веток."""
    from skimage.draw import line as draw_line

    gap_px = LINK_GAP_M / res_m
    cos_lim = np.cos(np.radians(LINK_ANGLE_DEG))
    ends = []
    for i, pts in enumerate(branches):
        if len(pts) < 3:
            continue
        ax, a, b = _axis(pts)
        ends.append((i, ax, a))
        ends.append((i, ax, b))

    out = np.zeros(shape, dtype=bool)
    for k, (i, ax_i, p) in enumerate(ends):
        for j, ax_j, q in ends[k + 1:]:
            if j == i:
                continue
            d = (q - p).astype("float64")
            n = np.hypot(*d)
            if n < 1e-6 or n > gap_px:
                continue
            u = d / n
            if abs(u @ ax_i) < cos_lim or abs(u @ ax_j) < cos_lim:
                continue
            rr, cc = draw_line(int(p[0]), int(p[1]), int(q[0]), int(q[1]))
            out[rr, cc] = True
    return out


def trace_lines(score: np.ndarray, valid: np.ndarray, area_mask: np.ndarray,
                res_m: float, density: str | tuple | None = None
                ) -> tuple[np.ndarray, pd.DataFrame]:
    """Скор -> маска линий и таблица веток (длина, прямолинейность, азимут).

    ``density`` — ключ из ``DENSITY`` или готовая пятёрка параметров (её берёт
    развёртка кривой «точность против плотности»); по умолчанию
    ``DENSITY_DEFAULT``.
    """
    from scipy.ndimage import label

    d = density or DENSITY_DEFAULT
    sigma_m, hi_area, lo_area, min_len, min_straight = (
        DENSITY[d] if isinstance(d, str) else d)
    sigma_px = sigma_m / res_m
    # Гребень ищется ТОЛЬКО по области оценки: вне её скор не посчитан (NaN), и
    # сглаживание по всему кадру разнесло бы NaN по растру целиком.
    s, ridge = _ridge_nms(score, area_mask, sigma_px)

    v = s[area_mask]
    hi = float(np.quantile(v, 1 - hi_area))
    lo = float(np.quantile(v, 1 - lo_area))
    cand = ridge & (s >= lo) & area_mask
    lab, n = label(cand, structure=np.ones((3, 3), dtype=int))
    if n:
        seeded = np.zeros(n + 1, dtype=bool)
        seeded[np.unique(lab[cand & (s >= hi)])] = True
        seeded[0] = False
        cand = seeded[lab]

    # Состригать больше, чем допускает порог длины, нельзя: иначе правило,
    # задуманное против шипов, съедает самостоятельные короткие линии.
    prune_m = min(PRUNE_M, min_len)
    parts = _branches(cand, res_m, prune_m)
    links = _link_gaps(parts, cand.shape, res_m) & area_mask
    if links.any():
        parts = _branches(cand | links, res_m, prune_m)

    keep = np.zeros(valid.shape, dtype=bool)
    recs, dropped = [], {"коротких": 0, "кривых": 0}
    for pts in parts:
        if len(pts) < 3:
            continue
        c = (pts - pts.mean(0)).astype("float64")
        _u, _sv, vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ vt[0]
        length = float(proj.max() - proj.min()) * res_m
        path = _path_length(pts, res_m)
        straight = min(length / path, 1.0) if path else 0.0
        if length < min_len:
            dropped["коротких"] += 1
            continue
        if straight < min_straight:
            dropped["кривых"] += 1
            continue
        keep[pts[:, 0], pts[:, 1]] = True
        recs.append({"длина_км": length / 1000.0,
                     "прямолинейность": straight,
                     "азимут": float(np.degrees(np.arctan2(vt[0][1], -vt[0][0])) % 180),
                     "ячеек": len(pts)})
    df = pd.DataFrame(recs)
    df.attrs["отсеяно"] = dropped
    return keep, df


# --------------------------------------------------------------------------
# Метрика линия-против-линии
# --------------------------------------------------------------------------

def line_metrics(pred: np.ndarray, dist: np.ndarray, area_mask: np.ndarray,
                 res_m: float) -> dict:
    """Точность и полнота по длине линий, обе в пределах ``TOL_M``."""
    from scipy.ndimage import distance_transform_edt

    p = pred & area_mask
    ref = (dist <= res_m * 0.7) & area_mask          # осевая линия карты
    if not p.any() or not ref.any():
        return {"точность": float("nan"), "полнота": float("nan"),
                "длина_км": 0.0, "ячеек": 0}
    dt_pred = distance_transform_edt(~p, sampling=(res_m, res_m))
    return {"точность": float((dist[p] <= TOL_M).mean()),
            "полнота": float((dt_pred[ref] <= TOL_M).mean()),
            "длина_км": float(p.sum()) * res_m / 1000.0,
            "ячеек": int(p.sum())}


def shift_null(pred: np.ndarray, dist: np.ndarray, area_mask: np.ndarray,
               observed: float, n_shifts: int = N_SHIFTS) -> dict:
    """Сдвиговый нуль ПО ЛИНИЯМ: та же геометрия, положенная в другое место.

    Нуль сохраняет и число линий, и их длины, и их направления — меняется только
    положение. Вопрос ставится ровно так, как надо: лучше ли эти линии легли на
    карту, чем те же самые линии в случайном месте площадки.
    """
    rng = np.random.default_rng(SEED)
    h, w = pred.shape
    base = int((pred & area_mask).sum())
    vals = []
    for _ in range(n_shifts):
        dy, dx = int(rng.integers(h)), int(rng.integers(w))
        m = np.roll(pred, (dy, dx), (0, 1)) & area_mask
        if m.sum() < 0.3 * base:
            continue
        vals.append(float((dist[m] <= TOL_M).mean()))
    v = np.asarray(vals)
    return {"нуль": float(v.mean()), "q95": float(np.quantile(v, 0.95)),
            "p": float((1 + int((v >= observed).sum())) / (1 + len(v))),
            "сдвигов": len(v)}


def agreement(a: np.ndarray, b: np.ndarray, res_m: float) -> float:
    """Взаимное покрытие двух списков линий — мера устойчивости трассировки.

    Симметрично: доля длины каждого списка, лежащая в допуске от другого,
    усреднённая. 1 — списки совпали, 0 — не пересеклись нигде.
    """
    from scipy.ndimage import distance_transform_edt

    if not a.any() or not b.any():
        return float("nan")
    da = distance_transform_edt(~a, sampling=(res_m, res_m))
    db = distance_transform_edt(~b, sampling=(res_m, res_m))
    return 0.5 * (float((db[a] <= TOL_M).mean()) + float((da[b] <= TOL_M).mean()))


# --------------------------------------------------------------------------

def build(site: str, support: np.ndarray | None, neigh: bool) -> tuple:
    """Каналы площадки; окрестность собирается только из ``support``."""
    z = load_site(site)
    x, _, valid, use = build_stack(z, CHANNELS)
    if neigh:
        nx, _ = neigh_stack(x, use, valid, float(z["res_m"]),
                            support=valid if support is None else support)
        x = np.concatenate([x, nx])
    return z, x, valid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="nevada")
    ap.add_argument("--apply-to", default="")
    ap.add_argument("--no-neigh", action="store_true")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--stability", action="store_true",
                    help="переобучить на другой подвыборке и сравнить списки линий")
    ap.add_argument("--rules", type=int, default=16)
    ap.add_argument("--density", default=DENSITY_DEFAULT, choices=tuple(DENSITY))
    args = ap.parse_args()
    neigh = not args.no_neigh

    def log(m=""):
        print(m, flush=True)

    apply_sites = [s for s in args.apply_to.split(",") if s]
    log(f"обучение на {args.site}, перенос на {', '.join(apply_sites) or '-'}")
    dp = DENSITY[args.density]
    log(f"признаки окрестности: {'да' if neigh else 'нет'}; плотность "
        f"'{args.density}': отсев веток длина >= {dp[3] / 1000:.1f} км, "
        f"прямолинейность >= {dp[4]}")
    log(f"пре-регистрация: первичная метрика — точность; тестов в семействе "
        f"{LN_N_TESTS_PRIOR + LN_N_TESTS}, alpha = {LN_ALPHA:.5f}, "
        f"минимально достижимое p = {1 / (1 + N_SHIFTS):.5f}")

    z0 = load_site(args.site)
    v0 = np.isfinite(z0["gray"])
    tr, va = block_split(v0.shape, float(z0["res_m"]))
    tr &= v0
    va &= v0

    _z, xt, _v = build(args.site, tr, neigh)
    xtr_full = rows(xt, tr)
    dtr_full = z0["dist"][tr]
    log(f"обучающих ячеек {len(xtr_full):,}, признаков {xtr_full.shape[1]}\n")

    out, traces = [], {}
    for tname, soft in (("hard", False), ("smooth", True)):
        for sub in ((0, 1) if args.stability else (0,)):
            rng = np.random.default_rng(SEED + sub)
            xtr, dtr = xtr_full, dtr_full
            if len(xtr) > N_TRAIN_ROWS:
                sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
                xtr, dtr = xtr[sel], dtr[sel]
            ytr = (soft_target(dtr) if soft
                   else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))

            t0 = time.time()
            models = {"ANFIS": fit_anfis(xtr, ytr, args.rules, soft, lambda *_: None)}
            models.update(fit_baselines(xtr, ytr, soft))
            log(f"-- таргет {tname}{' (подвыборка №2)' if sub else ''}: "
                f"обучено за {time.time() - t0:.0f} с")

            for mname, est in models.items():
                def predict(xr, _e=est, _m=mname):
                    return (anfis_predict(_e, xr) if _m == "ANFIS"
                            else baseline_predict(_e, xr))

                jobs = [(args.site, va, "проверка")]
                jobs += [(s, None, "перенос") for s in apply_sites]
                for site, amask, mode in jobs:
                    z, x, valid = build(site, amask, neigh)
                    area = valid if amask is None else amask
                    res_m = float(z["res_m"])
                    sc = score_map(predict(rows(x, area)), area)
                    pred, br = trace_lines(sc, valid, area, res_m, args.density)
                    r = line_metrics(pred, z["dist"], area, res_m)
                    row = {"таргет": tname, "подвыборка": sub + 1, "модель": mname,
                           "площадка": site, "режим": mode, "веток": len(br),
                           "медиана_длины_км": (float(br["длина_км"].median())
                                                if len(br) else float("nan")),
                           **r}
                    # Нуль считается только по первой подвыборке: вторая — это
                    # диагностика устойчивости, а не ещё одна попытка в семействе.
                    if not args.no_null and sub == 0 and np.isfinite(r["точность"]):
                        row.update(shift_null(pred, z["dist"], area, r["точность"]))
                        row["значим"] = row["p"] < LN_ALPHA
                    out.append(row)
                    traces[(tname, mname, site, sub)] = (pred, res_m)
                    np.savez_compressed(
                        M / f"openlin_lines_{args.site}_{mname}_{tname}_{sub + 1}"
                            f"_{site}.npz", pred=pred, score=sc)
                    pp = f"  p = {row['p']:.4f}" if "p" in row else ""
                    dr = br.attrs.get("отсеяно", {})
                    log(f"   {mname:<14} {site:<14} {mode:<9} веток {len(br):>3}  "
                        f"(отсеяно коротких {dr.get('коротких', 0)}, "
                        f"кривых {dr.get('кривых', 0)})  "
                        f"точность {r['точность']:.3f}  "
                        f"полнота {r['полнота']:.3f}{pp}")
    log()

    df = pd.DataFrame(out)
    path = M / f"openlin_lines_{args.site}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"сохранено: {path}")

    if args.stability:
        log("\n=== устойчивость трассировки: две подвыборки обучения " + "=" * 12)
        st = []
        for (tname, mname, site, sub), (pred, res_m) in traces.items():
            if sub:
                continue
            other = traces.get((tname, mname, site, 1))
            if other is None:
                continue
            st.append({"таргет": tname, "модель": mname, "площадка": site,
                       "совпадение": agreement(pred, other[0], res_m)})
        sdf = pd.DataFrame(st)
        log(sdf.round(3).to_string(index=False))
        log(f"\nмедиана совпадения списков линий: {sdf['совпадение'].median():.3f}")
        sdf.to_csv(M / f"openlin_lines_{args.site}_stability.csv", index=False,
                   encoding="utf-8-sig")


if __name__ == "__main__":
    main()
