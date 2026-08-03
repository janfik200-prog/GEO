"""Тесты src/lineaments.py: синтетический уступ и долина — без сети."""
import numpy as np

from src import lineaments


RES = 100.0


def _fault_scarp(n=200, row=100):
    """Прямолинейный уступ поперёк листа: ступень высоты 60 м вдоль строки."""
    e = np.zeros((n, n))
    e[row:] = 60.0
    return e + np.random.default_rng(0).normal(scale=0.5, size=(n, n))


def test_hillshade_depends_on_azimuth_and_stays_in_range():
    e = _fault_scarp()
    hs_n = lineaments.hillshade(e, 0.0, res_m=RES)
    hs_e = lineaments.hillshade(e, 90.0, res_m=RES)
    assert hs_n.min() >= 0.0 and hs_n.max() <= 1.0
    # уступ вытянут по широте: подсветка с севера даёт на нём больший контраст,
    # чем подсветка вдоль него — это и есть азимутальная предвзятость отмывки
    assert hs_n[95:105].std() > hs_e[95:105].std()


def test_scarp_is_found_as_a_line_of_correct_orientation():
    e = _fault_scarp()
    segs = lineaments.extract_lines(lineaments.edge_map(e, res_m=RES))
    assert segs, "прямолинейный уступ не найден вовсе"
    rows = [(y0 + y1) / 2 for (_, y0), (_, y1) in segs]
    assert min(abs(r - 100) for r in rows) < 5      # линия там, где уступ
    # преобладающее направление — субширотное (|dy| мало при большом |dx|)
    long_seg = max(segs, key=lambda s: (s[1][0] - s[0][0]) ** 2 + (s[1][1] - s[0][1]) ** 2)
    (x0, y0), (x1, y1) = long_seg
    assert abs(y1 - y0) < 0.3 * abs(x1 - x0)


def test_rasters_give_distance_and_anisotropy():
    shape = (60, 60)
    segs = [((5, 30), (55, 30))]                    # одна горизонтальная линия
    r = lineaments.line_rasters(segs, shape, res_m=RES)
    assert r["length"][30, 20] > 0
    assert r["dist"][30, 20] == 0.0                 # на самой линии
    assert r["dist"][10, 20] == 20 * RES            # 20 пикселей выше линии
    assert r["nodes"].max() == 0.0                  # одна система -> узлов нет

    crossed = segs + [((30, 5), (30, 55))]          # вторая система, поперёк
    r2 = lineaments.line_rasters(crossed, shape, res_m=RES)
    assert r2["nodes"][30, 30] > 0                  # пересечение -> узел


def test_azimuth_classes_separate_orthogonal_systems():
    segs = [((0, 0), (10, 0)), ((0, 0), (0, 10))]
    cls = lineaments._azimuth_class(segs)
    assert cls[0] != cls[1]
