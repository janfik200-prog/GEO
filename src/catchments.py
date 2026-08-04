"""Водосборы точек заверки: заверка по площади сноса, а не по одной ячейке.

ЗАЧЕМ. Точки заверки на листе — шлиховые и литохимические ореолы в руслах.
Обломок из коренного источника уезжает вниз по течению, поэтому ячейка точки —
это не то место, где находится источник, а место, где обломок остановился.
Заверка «попала ли ячейка точки в top-10% карты» штрафует метод, который
правильно указал коренной источник в трёх километрах выше по склону, и
вознаграждает метод, который просто подсветил долины.

Здесь единицей заверки становится ВОДОСБОР точки — множество ячеек, сток с
которых проходит через точку. Метод засчитывается тем выше, чем большая доля
водосбора попала в его перспективную площадь.

ВТОРОЙ ЭФФЕКТ, РАДИ КОТОРОГО ЭТО И ДЕЛАЕТСЯ. Точечная метрика на 19 точках
квантована: одна точка = 0.53 lift, промежуточных значений не существует, и
этап 5e показал, что случайность обучения гуляет по этой шкале на 1-4 точки.
Водосборная метрика непрерывна: доля площади водосбора в top-a% меняется
плавно, поэтому различает методы там, где точечная не может.

НОРМИРОВКА. Скор водосбора = доля его ячеек в top-``area`` пула, делённая на
``area``. У случайной карты эта величина равна единице при ЛЮБОМ размере
водосбора, поэтому крупные водосборы не получают преимущества автоматически —
известная ловушка водосборных метрик на плоском рельефе.

НАПРАВЛЕНИЯ СТОКА — В ДВА ШАГА, и это принципиально. Ямы и плоские участки на
тундровом рельефе не исключение, а правило, поэтому сначала рельеф заполняется
приоритетным затоплением с добавкой epsilon (поверхность становится строго
убывающей к выходу, плоскости получают уклон), и только потом по заполненной
поверхности считается обычный D8 — приёмник по максимальному уклону.

Более короткий вариант «приёмник = ячейка, из которой её открыл фронт
затопления» был опробован и отвергнут по измерению: он даёт формально корректное
дерево, но поток в нём не концентрируется — максимальная накопленная площадь по
листу вышла 601 ячейка (150 км2) вместо тысяч, то есть речной сети такое дерево
не воспроизводит и водосборы получаются вырожденно мелкими.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from src import config

# 8 соседей: смещения (drow, dcol) и их длина в ячейках
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_sinks(dem: np.ndarray, valid: np.ndarray | None = None,
               eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Заполнение ям приоритетным затоплением; возвращает (высоты, край).

    Обход идёт от выходов (край листа и границы с невалидной областью) внутрь
    по возрастанию высоты. Каждая открытая ячейка поднимается до уровня
    ``max(своя высота, уровень открывшей + eps)``, поэтому заполненная
    поверхность строго убывает вдоль пути к выходу: замкнутых понижений не
    остаётся, а на плоских участках появляется искусственный уклон нужного
    направления. Без добавки ``eps`` заполненная яма была бы идеально плоской и
    D8 не смог бы выбрать направление.
    """
    dem = np.asarray(dem, dtype=float)
    prf, pic = dem.shape
    if valid is None:
        valid = np.isfinite(dem)
    valid = np.asarray(valid, dtype=bool).reshape(prf, pic)

    z = np.where(valid & np.isfinite(dem), dem, np.nan)
    filled = np.full((prf, pic), np.inf)
    done = ~valid
    is_edge = np.zeros((prf, pic), dtype=bool)
    heap: list[tuple[float, int, int]] = []

    for r in range(prf):
        for c in range(pic):
            if not valid[r, c]:
                continue
            edge = r == 0 or c == 0 or r == prf - 1 or c == pic - 1
            if not edge:
                for dr, dc in _NB:
                    if not valid[r + dr, c + dc]:
                        edge = True
                        break
            if edge:
                lvl = float(z[r, c]) if np.isfinite(z[r, c]) else 0.0
                filled[r, c] = lvl
                is_edge[r, c] = True
                done[r, c] = True
                heapq.heappush(heap, (lvl, r, c))

    while heap:
        lvl, r, c = heapq.heappop(heap)
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < prf and 0 <= cc < pic) or done[rr, cc]:
                continue
            done[rr, cc] = True
            zn = z[rr, cc]
            new = max(float(zn) if np.isfinite(zn) else lvl, lvl + eps)
            filled[rr, cc] = new
            heapq.heappush(heap, (new, rr, cc))
    return filled, is_edge


def flow_dirs_d8(dem: np.ndarray, valid: np.ndarray | None = None,
                 eps: float = 1e-3) -> np.ndarray:
    """D8 по заполненному рельефу: приёмник = сосед с максимальным уклоном.

    ``dem`` — высоты (prf, pic), ``valid`` — маска валидных ячеек той же формы.
    Возвращает массив плоских индексов приёмников; у выходов (край листа,
    граница с невалидной областью) и у невалидных ячеек приёмник равен -1.

    Уклон считается с учётом длины хода: у диагональных соседей перепад делится
    на sqrt(2), иначе сток систематически уходил бы по диагоналям.
    """
    dem = np.asarray(dem, dtype=float)
    prf, pic = dem.shape
    if valid is None:
        valid = np.isfinite(dem)
    valid = np.asarray(valid, dtype=bool).reshape(prf, pic)
    filled, is_edge = fill_sinks(dem, valid, eps)

    rec = np.full(prf * pic, -1, dtype=np.int64)
    for r in range(prf):
        for c in range(pic):
            if not valid[r, c] or is_edge[r, c]:
                continue
            here = filled[r, c]
            best, best_idx = 0.0, -1
            for dr, dc in _NB:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < prf and 0 <= cc < pic) or not valid[rr, cc]:
                    continue
                drop = here - filled[rr, cc]
                if drop <= 0:
                    continue
                slope = drop / (1.41421356 if dr and dc else 1.0)
                if slope > best:
                    best, best_idx = slope, rr * pic + cc
            rec[r * pic + c] = best_idx
    return rec


def _children(rec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR-представление дерева стока: (начала, отсортированные потомки)."""
    n = rec.size
    has = rec >= 0
    order = np.argsort(rec[has], kind="stable")
    kids = np.flatnonzero(has)[order]
    counts = np.bincount(rec[has], minlength=n)
    starts = np.r_[0, np.cumsum(counts)]
    return starts, kids


def upstream_catchment(rec: np.ndarray, seed: int, max_cells: int,
                       starts: np.ndarray | None = None,
                       kids: np.ndarray | None = None) -> np.ndarray:
    """Ячейки, сток с которых проходит через ``seed`` (включая саму ячейку).

    Обход в ширину вверх по дереву стока с ограничением ``max_cells``: на
    выположенном рельефе водосбор может занять пол-листа, а такой «водосбор»
    ничего не заверяет — он просто накрывает всю карту. Ограничение по числу
    ячеек, а не по расстоянию, потому что заверяет именно площадь сноса.
    """
    if starts is None or kids is None:
        starts, kids = _children(rec)
    out = [int(seed)]
    frontier = [int(seed)]
    while frontier and len(out) < max_cells:
        nxt = []
        for c in frontier:
            for k in kids[starts[c]:starts[c + 1]]:
                nxt.append(int(k))
                out.append(int(k))
                if len(out) >= max_cells:
                    break
            if len(out) >= max_cells:
                break
        frontier = nxt
    return np.array(out[:max_cells], dtype=np.int64)


def build_catchments(df: pd.DataFrame, meta, point_idx: np.ndarray,
                     valid: np.ndarray, max_cells: int | None = None,
                     dem_col: str = "dem_elev") -> list[np.ndarray]:
    """Водосборы точек заверки, ограниченные валидными ячейками."""
    max_cells = max_cells or config.CATCH_MAX_CELLS
    dem = df[dem_col].to_numpy(float).reshape(meta.prf, meta.pic)
    rec = flow_dirs_d8(dem, valid.reshape(meta.prf, meta.pic))
    starts, kids = _children(rec)
    out = []
    for p in np.asarray(point_idx, dtype=int):
        cells = upstream_catchment(rec, int(p), max_cells, starts, kids)
        out.append(cells[valid[cells]])
    return out


def catchment_capture(score: np.ndarray, catchments: list[np.ndarray],
                      area: float | None = None,
                      pool: np.ndarray | None = None) -> float:
    """Средняя по водосборам доля площади в top-``area``, делённая на ``area``.

    Непрерывный аналог lift@area: у случайной карты равен 1 при любом размере
    водосбора, у одноячеечного водосбора совпадает с точечным lift.
    """
    area = area or config.VER_AREA
    if not catchments:
        return 0.0
    ref = score if pool is None else score[pool]
    thr = np.quantile(ref, 1.0 - area)
    realized = float((ref >= thr).mean())
    if realized <= 0:
        return 0.0
    shares = [float((score[c] >= thr).mean()) if c.size else 0.0
              for c in catchments]
    return float(np.mean(shares)) / realized
