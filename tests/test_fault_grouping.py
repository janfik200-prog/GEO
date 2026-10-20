"""Тесты src/fault_grouping.py на синтетике (без датасета и шейп-файлов).

Смысл файла — проверить метод там, где правильный ответ известен точно: рои
задаются с заданным азимутом и заданным перпендикулярным сносом, и метод обязан
вернуть именно их. Низкие числа на реальных данных после этого читаются как
свойство данных, а не как ошибка реализации (тот же довод, что в
tests/test_fracture_zones.py).
"""
import numpy as np
import pandas as pd
import pytest

from src import config, fault_grouping as fg

shapely = pytest.importorskip("shapely")
from shapely import affinity                      # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402


class _Meta:
    """Заглушка GridMeta: нужны шаг, размер и центры ячеек."""

    def __init__(self, prf=100, pic=100, dx=500.0, dy=500.0, x0=0.0, y0=0.0):
        self.prf, self.pic, self.dx, self.dy = prf, pic, dx, dy
        self.x0, self.y0 = x0, y0

    @property
    def shape(self):
        return self.prf, self.pic

    @property
    def y_top(self):
        return self.y0 + self.prf * self.dy

    def cell_centers(self):
        cols, rows = np.meshgrid(np.arange(self.pic), np.arange(self.prf))
        return (self.x0 + (cols + 0.5) * self.dx,
                self.y_top - (rows + 0.5) * self.dy)


def _bar(x, y, az_deg, length=8000.0, width=800.0):
    """Прямоугольник-«дайка»: центр (x, y), ось под азимутом ``az_deg`` от севера."""
    box = Polygon([(-width / 2, -length / 2), (width / 2, -length / 2),
                   (width / 2, length / 2), (-width / 2, length / 2)])
    # Азимут от севера по часовой стрелке = поворот вертикального бруска на -az.
    return affinity.translate(affinity.rotate(box, -az_deg, origin=(0, 0)), x, y)


def _swarm(az_deg, rho, n=6, x0=50000.0, y0=50000.0, span=40000.0, seed=0):
    """Рой из ``n`` параллельных брусков вдоль одной полосы со сносом ``rho``."""
    rng = np.random.default_rng(seed)
    a = np.radians(az_deg)
    u = np.array([np.sin(a), np.cos(a)])      # вдоль простирания
    nvec = np.array([np.cos(a), -np.sin(a)])  # поперёк
    base = np.array([x0, y0]) + rho * nvec
    out = []
    for t in np.linspace(-span / 2, span / 2, n):
        jitter = rng.normal(scale=300.0)      # разброс поперёк, много меньше eps
        p = base + t * u + jitter * nvec
        out.append(_bar(p[0], p[1], az_deg))
    return out


# --- Ось объекта ---

@pytest.mark.parametrize("az", [0.0, 25.0, 45.0, 90.0, 135.0, 170.0])
def test_axial_segments_recovers_azimuth(az):
    """Азимут оси восстанавливается с точностью много лучше ширины ядра."""
    seg = fg.axial_segments([_bar(0.0, 0.0, az)])
    assert len(seg) == 1
    assert fg.axial_diff_deg(seg["az_deg"].iloc[0], az) < 1.0
    assert seg["length_m"].iloc[0] == pytest.approx(8000.0, rel=1e-3)
    assert seg["width_m"].iloc[0] == pytest.approx(800.0, rel=1e-3)


def test_axial_segments_accepts_lines():
    """Линия принимается наравне с полигоном — вход для вектора разломов."""
    seg = fg.axial_segments([LineString([(0, 0), (0, 10000)])])
    assert fg.axial_diff_deg(seg["az_deg"].iloc[0], 0.0) < 1.0
    assert seg["length_m"].iloc[0] == pytest.approx(10000.0, rel=1e-3)


def test_axial_segments_skips_degenerate():
    """Пустая и вырожденная геометрия не роняет разбор и не даёт строк."""
    seg = fg.axial_segments([Polygon(), LineString([(0, 0), (0, 0)])])
    assert len(seg) == 0


def test_axial_diff_is_axial():
    """179° и 1° — одна ось, разность 2°, а не 178°."""
    assert fg.axial_diff_deg(179.0, 1.0) == pytest.approx(2.0)
    assert fg.axial_diff_deg(0.0, 90.0) == pytest.approx(90.0)


def test_along_rho_orthogonal():
    """Смещение вдоль простирания не меняет rho, поперёк — не меняет along."""
    az = 40.0
    a = np.radians(az)
    u, n = (np.sin(a), np.cos(a)), (np.cos(a), -np.sin(a))
    al0, r0 = fg.along_rho([0.0], [0.0], az)
    al1, r1 = fg.along_rho([1000 * u[0]], [1000 * u[1]], az)
    al2, r2 = fg.along_rho([1000 * n[0]], [1000 * n[1]], az)
    assert r1[0] - r0[0] == pytest.approx(0.0, abs=1e-6)
    assert al1[0] - al0[0] == pytest.approx(1000.0)
    assert al2[0] - al0[0] == pytest.approx(0.0, abs=1e-6)
    assert r2[0] - r0[0] == pytest.approx(1000.0)


# --- Этап 1: направления ---

def test_direction_modes_finds_two_swarms():
    """Два роя 40° и 140° дают две моды на своих местах."""
    geoms = _swarm(40.0, 0.0) + _swarm(140.0, 0.0, seed=1)
    seg = fg.axial_segments(geoms)
    modes = fg.direction_modes(seg["az_deg"], weights=seg["length_m"])
    assert modes["modes_deg"].size == 2
    got = np.sort(modes["modes_deg"])
    assert fg.axial_diff_deg(got[0], 40.0) <= config.FG_AZ_BANDWIDTH_DEG
    assert fg.axial_diff_deg(got[1], 140.0) <= config.FG_AZ_BANDWIDTH_DEG


def test_direction_modes_weight_is_length():
    """Длинный одиночный объект весит больше пачки коротких — вес это длина."""
    seg = pd.DataFrame({"x": [0.0, 0, 0, 0], "y": [0.0, 0, 0, 0],
                        "az_deg": [10.0, 100.0, 100.0, 100.0],
                        "length_m": [30000.0, 1000.0, 1000.0, 1000.0],
                        "width_m": [100.0] * 4})
    modes = fg.direction_modes(seg["az_deg"], weights=seg["length_m"])
    # Моды отсортированы по весу: первой обязана идти мода одиночного длинного.
    assert fg.axial_diff_deg(modes["modes_deg"][0], 10.0) <= config.FG_AZ_BANDWIDTH_DEG


def test_direction_modes_wraps_around_zero():
    """Рой около 0° не разваливается на две моды по краям шкалы."""
    geoms = _swarm(178.0, 0.0) + _swarm(2.0, 0.0, seed=1)
    seg = fg.axial_segments(geoms)
    modes = fg.direction_modes(seg["az_deg"], weights=seg["length_m"])
    assert modes["modes_deg"].size == 1


def test_direction_modes_empty_input():
    modes = fg.direction_modes([], weights=[])
    assert modes["modes_deg"].size == 0


# --- Этап 2: зоны ---

def test_zones_split_when_far_and_merge_when_close():
    """Порог слияния работает по перпендикулярному сносу, а не по расстоянию."""
    az = 40.0
    far = fg.axial_segments(_swarm(az, 0.0) + _swarm(az, 3 * config.FG_RHO_EPS_M, seed=1))
    near = fg.axial_segments(_swarm(az, 0.0) + _swarm(az, 0.4 * config.FG_RHO_EPS_M, seed=1))
    assert len(fg.zones_of_mode(far, az)) == 2
    assert len(fg.zones_of_mode(near, az)) == 1


def test_zone_ignores_gap_along_strike():
    """Разрыв вдоль простирания зону не режет: коридор тянется через лист."""
    az = 40.0
    a = np.radians(az)
    u = np.array([np.sin(a), np.cos(a)])
    far = 60000.0 * u
    geoms = (_swarm(az, 0.0, n=3, span=5000.0)
             + _swarm(az, 0.0, n=3, span=5000.0, x0=50000 + far[0],
                      y0=50000 + far[1], seed=2))
    zones = fg.zones_of_mode(fg.axial_segments(geoms), az)
    assert len(zones) == 1
    assert 2 * zones[0].half_length > 60000.0


def test_zone_needs_min_members():
    """Двух объектов на зону мало — это отрезок, а не зона."""
    seg = fg.axial_segments(_swarm(40.0, 0.0, n=2))
    assert fg.zones_of_mode(seg, 40.0) == []


def test_zone_width_floor_applies():
    """У идеально коллинеарной группы ширина не нулевая, а равна полу."""
    az = 40.0
    a = np.radians(az)
    u = np.array([np.sin(a), np.cos(a)])
    geoms = [_bar(50000 + t * u[0], 50000 + t * u[1], az)
             for t in (-10000.0, 0.0, 10000.0)]
    zones = fg.zones_of_mode(fg.axial_segments(geoms), az)
    assert len(zones) == 1
    assert 2 * zones[0].half_width == pytest.approx(config.FG_MIN_ZONE_WIDTH_M)


def test_group_zones_end_to_end():
    """Два роя разных направлений -> по зоне на каждое, с верными азимутами."""
    geoms = _swarm(40.0, 0.0) + _swarm(140.0, 0.0, seed=1)
    zones, modes = fg.group_zones(fg.axial_segments(geoms))
    assert len(zones) == 2
    az = sorted(z.az_deg for z in zones)
    assert fg.axial_diff_deg(az[0], 40.0) <= config.FG_AZ_BANDWIDTH_DEG
    assert fg.axial_diff_deg(az[1], 140.0) <= config.FG_AZ_BANDWIDTH_DEG
    assert modes["labels"].size == len(geoms)
    assert fg.zone_table(zones).shape[0] == 2


# --- Второй оператор: сгущения вместо связных компонент ---

def _dense_two_bands(az=40.0, sep=None, n=40, seed=0):
    """Две параллельные полосы БЕЗ пустого промежутка между ними.

    Объекты рассыпаны по всему интервалу, но сгущены у двух центров: разрывов в
    rho нет, а два сгущения есть. Ровно тот случай, в котором разрез по разрывам
    обязан слить всё в одну зону, а разрез по долинам плотности — разделить.
    """
    sep = 6.0 * config.FG_RHO_EPS_M if sep is None else sep
    rng = np.random.default_rng(seed)
    a = np.radians(az)
    u = np.array([np.sin(a), np.cos(a)])
    nvec = np.array([np.cos(a), -np.sin(a)])
    # Фон кладётся регулярно с шагом в половину порога слияния: так гарантируется,
    # что разрывов шире порога нет вовсе, и первый оператор обязан слить всё.
    back = np.arange(-sep / 2, sep / 2 + 1.0, config.FG_RHO_EPS_M / 2)
    rho = np.concatenate([
        rng.normal(-sep / 2, config.FG_RHO_EPS_M / 3, n),
        rng.normal(+sep / 2, config.FG_RHO_EPS_M / 3, n),
        back,
    ])
    along = rng.uniform(-20000.0, 20000.0, rho.size)
    base = np.array([50000.0, 50000.0])
    return [_bar(*(base + t * u + r * nvec), az_deg=az)
            for t, r in zip(along, rho)]


def test_kde_splits_dense_bands_where_dbscan_chains():
    """На плотном входе разрез по разрывам вырождается, разрез по долинам — нет."""
    seg = fg.axial_segments(_dense_two_bands())
    rho = np.sort(fg.along_rho(seg["x"].to_numpy(), seg["y"].to_numpy(), 40.0)[1])
    assert np.diff(rho).max() < config.FG_RHO_EPS_M      # разрывов нет по построению

    assert len(fg.zones_of_mode(seg, 40.0)) == 1
    zones = fg.zones_of_mode_kde(seg, 40.0)
    assert len(zones) == 2
    # Проверяется не положение центров (они смещены фоном, попавшим в зону), а
    # то, что разрез прошёл МЕЖДУ сгущениями: каждое целиком в своей зоне.
    lower, upper = sorted(zones, key=lambda z: z.rho_c)
    n = 40
    assert len(set(range(n)) & set(lower.members.tolist())) >= 0.9 * n
    assert len(set(range(n, 2 * n)) & set(upper.members.tolist())) >= 0.9 * n


def test_kde_matches_dbscan_on_sparse_input():
    """На разреженном входе операторы дают одно и то же: долины и есть разрывы."""
    az = 40.0
    seg = fg.axial_segments(_swarm(az, 0.0) + _swarm(az, 4 * config.FG_RHO_EPS_M, seed=1))
    a = fg.zones_of_mode(seg, az)
    b = fg.zones_of_mode_kde(seg, az)
    assert len(a) == len(b) == 2
    assert sorted(z.n for z in a) == sorted(z.n for z in b)


def test_kde_single_concentration_stays_one_zone():
    """Одно сгущение — одна зона: оператор не режет там, где долин нет."""
    seg = fg.axial_segments(_swarm(40.0, 0.0, n=12))
    assert len(fg.zones_of_mode_kde(seg, 40.0)) == 1


def test_kde_keeps_every_object():
    """Разрез по долинам не теряет объекты: сумма членов зон = числу объектов."""
    seg = fg.axial_segments(_dense_two_bands())
    zones = fg.zones_of_mode_kde(seg, 40.0, min_members=1)
    assert sum(z.n for z in zones) == len(seg)


def test_group_zones_operator_switch():
    """Оператор выбирается по имени и меняет результат, а не молча игнорируется."""
    seg = fg.axial_segments(_dense_two_bands())
    a, _ = fg.group_zones(seg, operator="dbscan")
    b, _ = fg.group_zones(seg, operator="kde")
    assert len(b) > len(a)
    assert set(fg.ZONE_OPERATORS) == {"dbscan", "kde"}


# --- Растры и метрики ---

def test_zone_mask_covers_members_and_not_everything():
    """Маска накрывает сами объекты, но не весь лист."""
    meta = _Meta()
    seg = fg.axial_segments(_swarm(40.0, 0.0))
    zones = fg.zones_of_mode(seg, 40.0)
    mask = fg.zone_mask(zones, meta)
    X, Y = meta.cell_centers()
    for _, r in seg.iterrows():
        j = int(np.argmin((X - r["x"]) ** 2 + (Y - r["y"]) ** 2))
        assert mask.ravel()[j]
    assert 0.0 < mask.mean() < 0.5


def test_zone_field_peaks_inside_and_decays_outside():
    """Поле максимально в теле зоны и убывает с удалением поперёк."""
    meta = _Meta()
    zones = fg.zones_of_mode(fg.axial_segments(_swarm(40.0, 0.0)), 40.0)
    field = fg.zone_field(zones, meta)
    mask = fg.zone_mask(zones, meta)
    assert field[mask].min() >= field[~mask].max() - 1e-9
    assert field.min() >= 0.0


def test_zone_field_empty_is_flat():
    assert fg.zone_field([], _Meta()).max() == 0.0


def test_mask_iou_bounds():
    """IoU: совпадение -> 1, непересечение -> 0, вложение -> отношение площадей."""
    meta = _Meta(10, 10)
    m = np.zeros(meta.shape, dtype=bool)
    m[:5, :] = True
    idx_same = np.where(m.ravel())[0]
    idx_other = np.where(~m.ravel())[0]
    assert fg.mask_iou(m, idx_same) == pytest.approx(1.0)
    assert fg.mask_iou(m, idx_other) == pytest.approx(0.0)
    half = idx_same[: idx_same.size // 2]
    assert fg.mask_iou(m, half) == pytest.approx(0.5)


def test_mask_iou_respects_pool():
    """Ячейки вне пула не считаются ни в пересечение, ни в объединение."""
    meta = _Meta(10, 10)
    m = np.zeros(meta.shape, dtype=bool)
    m[:5, :] = True
    target = np.where(m.ravel())[0]
    pool = np.arange(50)             # ровно верхняя половина
    assert fg.mask_iou(m, target, pool=pool) == pytest.approx(1.0)
    assert fg.mask_iou(m, target, pool=np.arange(50, 100)) == pytest.approx(0.0)


def test_mask_iou_pvalue_significant_when_aligned():
    """Совпадающая с целью полоса значима, а поперечная — нет.

    Обе маски одинаковой площади и одинаковой связности, различие только в
    положении, поэтому тест меряет именно то, ради чего сдвиговый null и введён.
    """
    meta = _Meta(60, 60)
    target_img = np.zeros(meta.shape, dtype=bool)
    target_img[20:30, :] = True
    target = np.where(target_img.ravel())[0]

    aligned = np.zeros(meta.shape, dtype=bool)
    aligned[20:30, :] = True
    crossing = np.zeros(meta.shape, dtype=bool)
    crossing[:, 20:30] = True

    p_aligned = fg.mask_iou_pvalue(aligned, target, meta, n_shifts=200)
    p_cross = fg.mask_iou_pvalue(crossing, target, meta, n_shifts=200)
    assert p_aligned["observed"] == pytest.approx(1.0)
    assert p_aligned["p"] < 0.05
    assert p_cross["p"] > 0.05
    assert p_cross["observed"] < p_aligned["observed"]
