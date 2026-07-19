"""Тесты чтения нативных сеток ГИС Интегро (`.pgrid`/`.property`).

Пропускаются, если эталонных файлов нет на диске (не всегда закоммичены).
Запуск из корня репозитория: ``python -m pytest -q``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integro_grid import load_pgrid_dataset, read_pgrid, read_property  # noqa: E402

PGRID_PATH = Path(__file__).resolve().parents[1] / "data" / "Gis-integro" / "Расчет" / "prognoz.pgrid"

pytestmark = pytest.mark.skipif(not PGRID_PATH.exists(), reason="эталонный prognoz.pgrid не найден")


def test_read_pgrid_header():
    meta = read_pgrid(PGRID_PATH)
    assert meta.obj_count == 22946
    assert meta.pic == 149
    assert meta.prf == 154
    assert meta.pic * meta.prf == meta.obj_count
    assert meta.dx == 500 and meta.dy == 500
    assert meta.prop_count == 15
    assert len(meta.properties) == 15
    names = {p.name for p in meta.properties}
    assert {"ID", "sed", "lyth", "prognoz"} <= names


@pytest.mark.parametrize("name", ["lyth", "tect_nw", "magm", "paleo", "struct", "tect_ne"])
def test_property_matches_header_stats(name):
    meta = read_pgrid(PGRID_PATH)
    prop = meta.property_by_name(name)
    assert prop.stat_exists
    arr = read_property(PGRID_PATH.with_name(f"prognoz.{name}.property"), meta, prop)
    assert arr.shape == (meta.prf, meta.pic)
    assert np.isclose(arr.min(), prop.vmin, atol=1e-2)
    assert np.isclose(arr.max(), prop.vmax, atol=1e-2)
    assert np.isclose(arr.mean(), prop.average, atol=1e-2)
    assert np.isclose(arr.std(ddof=1), prop.stdev, atol=1e-1)


def test_bitmask_property_shape():
    meta = read_pgrid(PGRID_PATH)
    prop = meta.property_by_name("ID")
    arr = read_property(PGRID_PATH.with_name("prognoz.ID.property"), meta, prop)
    assert arr.shape == (meta.prf, meta.pic)
    assert set(np.unique(arr)) <= {0, 1}


def test_load_pgrid_dataset_reads_all_available_properties():
    meta, arrays = load_pgrid_dataset(PGRID_PATH)
    assert set(arrays) == {p.name for p in meta.properties}
    for arr in arrays.values():
        assert arr.shape == meta.shape


def test_transform_and_cell_centers_north_up():
    meta = read_pgrid(PGRID_PATH)
    x0, dx, _, y_top, _, neg_dy = meta.transform
    assert (x0, dx, neg_dy) == (meta.x0, meta.dx, -meta.dy)
    assert y_top == meta.y0 + meta.prf * meta.dy
    x, y = meta.cell_centers()
    assert x.shape == y.shape == meta.shape
    assert y[0, 0] > y[-1, 0]          # строка 0 — север
    assert x[0, 0] < x[0, -1]          # столбец 0 — запад
    assert np.isclose(y[0, 0], meta.y_top - meta.dy / 2)


def _toy_meta(pic, prf, dx, x0=0.0, y0=0.0):
    from src.integro_grid import GridMeta
    return GridMeta(obj_count=pic * prf, prop_count=0, pic=pic, prf=prf,
                    dx=dx, dy=dx, x0=x0, y0=y0)


def test_resample_average_downscales_blocks():
    pytest.importorskip("rasterio")
    from src.integro_grid import resample_to_grid
    # шахматка 0/1 с блоками 1x1 при агрегации 2->1 даёт среднее 0.5
    src_meta = _toy_meta(8, 8, dx=1.0)
    dst_meta = _toy_meta(4, 4, dx=2.0)
    checker = np.indices((8, 8)).sum(axis=0) % 2
    out = resample_to_grid(checker.astype(np.float32), src_meta, dst_meta, "average")
    assert out.shape == (4, 4)
    assert np.allclose(out, 0.5)


def test_resample_bilinear_keeps_constant():
    pytest.importorskip("rasterio")
    from src.integro_grid import resample_to_grid
    src_meta = _toy_meta(10, 10, dx=1.0)
    dst_meta = _toy_meta(5, 5, dx=1.0, x0=2.6, y0=2.1)
    out = resample_to_grid(np.full((10, 10), 7.0, np.float32), src_meta, dst_meta)
    assert np.allclose(out, 7.0)


def test_resample_angle_no_wraparound_artifact():
    pytest.importorskip("rasterio")
    from src.integro_grid import resample_angle_to_grid
    # постоянное поле углов у разрыва: +179 и -179 усредняются в 180, а не в 0
    src_meta = _toy_meta(8, 8, dx=1.0)
    dst_meta = _toy_meta(4, 4, dx=2.0)
    angles = np.where(np.indices((8, 8)).sum(axis=0) % 2, 179.0, -179.0)
    out = resample_angle_to_grid(angles.astype(np.float32), src_meta, dst_meta)
    assert np.all(np.abs(np.abs(out) - 180.0) < 1.5)


def test_to_common_grid_resamples_and_scales(tmp_path):
    pytest.importorskip("rasterio")
    from src.integro_grid import to_common_grid
    # мини-.pgrid 4x4 (шаг 1) с одним float32-свойством из констант 10.0
    pgrid_xml = (
        "<Grid><ObjCount>16</ObjCount><PropCount>1</PropCount>"
        "<Pic>4</Pic><Prf>4</Prf><DX>1</DX><DY>1</DY><X0>0</X0><Y0>0</Y0>"
        "<Properties><TGrDocProp><PropName>val</PropName><Caption>val</Caption>"
        "<PropKind>pkData</PropKind><DataType>dtSingle</DataType>"
        "<StatExists>False</StatExists></TGrDocProp></Properties></Grid>"
    )
    (tmp_path / "toy.pgrid").write_text(pgrid_xml, encoding="cp1251")
    np.full(16, 10.0, np.float32).tofile(tmp_path / "toy.val.property")

    dst = _toy_meta(2, 2, dx=2.0)
    out = to_common_grid(tmp_path / "toy.pgrid", dst, method="average",
                         prefix="t_", scale=0.5)
    assert set(out) == {"t_val"}
    assert out["t_val"].shape == (2, 2)
    assert np.allclose(out["t_val"], 5.0)   # среднее 10.0 * scale 0.5

    # nodata: значение 10.0 объявляем фоном -> весь результат NaN
    out2 = to_common_grid(tmp_path / "toy.pgrid", dst, method="average", nodata=10.0)
    assert np.all(np.isnan(out2["val"]))


SBORKA_LANDSAT = PGRID_PATH.parents[2] / "SBORKA_DOP" / "КОСМОСНИМОК" / "landsat_fragm.pgrid"


@pytest.mark.skipif(not SBORKA_LANDSAT.exists(), reason="СБОРКА_ДОП не распакована")
def test_read_byte_property_landsat():
    meta = read_pgrid(SBORKA_LANDSAT)
    prop = meta.property_by_name("ch1")
    assert prop.dtype == "dtByte"
    arr = read_property(SBORKA_LANDSAT.with_name("landsat_fragm.ch1.property"), meta, prop)
    assert arr.shape == (meta.prf, meta.pic)
    assert arr.dtype == np.uint8
