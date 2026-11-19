"""ANFU шаг 6: сдвиг точек ломаной к оси вогнутости — на синтетике (эталон не читается)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("scipy")


def _valley(col0=252, H=400, W=500):
    c = np.arange(W)[None, :].repeat(H, 0).astype(float)
    return 50 - 30 * np.exp(-0.5 * ((c - col0) / 4.0) ** 2)      # ложбина вдоль столбца col0


def test_snap_moves_point_to_valley_axis():
    from experiments.openlin_anfu_snap import apply, snap_shifts
    from experiments.openlin_fam import multi_ridge
    dem = _valley()
    conc, _ = multi_ridge(dem, (60,))
    valid = np.ones_like(dem, bool)
    # ломаная вдоль столбца 100-м ячейки 49 → пиксель 20 м 49*5+2 = 247, ось ложбины 252 (+100 м)
    pl = np.stack([np.arange(20, 60, dtype=float), np.full(40, 49.0)], 1)
    q, nv, sh = snap_shifts(pl, conc, valid, 5)
    new = apply(q, nv, sh, 5)
    col20 = new[:, 1] * 5 + 2
    assert np.allclose(col20, 252, atol=1)


def test_flat_dem_no_shift_and_edge_no_shift():
    from experiments.openlin_anfu_snap import snap_shifts
    from experiments.openlin_fam import multi_ridge
    dem = np.zeros((400, 500))
    conc, _ = multi_ridge(dem, (60,))
    pl = np.stack([np.arange(20, 60, dtype=float), np.full(40, 49.0)], 1)
    _, _, sh = snap_shifts(pl, conc, np.ones_like(dem, bool), 5)
    assert np.all(sh == 0)                                       # плато: ближайшее к нулю
    conc = multi_ridge(_valley(col0=3), (60,))[0]
    pl = np.stack([np.arange(20, 60, dtype=float), np.full(40, 0.0)], 1)   # у края листа
    _, _, sh = snap_shifts(pl, conc, np.ones_like(conc, bool), 5)
    assert np.all(sh == 0)


def test_shift_bounded_140m():
    from experiments.openlin_anfu_snap import SNAP_HALF_PX, _OFFS
    assert len(_OFFS) == 15 and SNAP_HALF_PX * 20 == 140
    assert list(_OFFS[:3]) == [0, -1, 1]
