"""Тесты реестра признаков золотой задачи (стоп-лист, углы, сборка матрицы)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.gold_features import build_feature_matrix, load_feature_matrix, validate_features  # noqa: E402

DATASET = config.PROCESSED_DIR / "dataset_v1.npz"


def test_stop_layer_rejected():
    with pytest.raises(ValueError, match="стоп-листа"):
        validate_features(["gm_gr_all", "prognoz"])


def test_unknown_layer_rejected():
    with pytest.raises(ValueError, match="вне реестра"):
        validate_features(["gm_gr_all", "totally_new_layer"])


def test_registry_has_no_overlap():
    train = set(config.GOLD_FEATURES_TRAIN)
    assert not train & set(config.GOLD_FEATURES_STOP)
    assert not train & set(config.GOLD_FEATURES_ANGLE)
    assert not train & set(config.GOLD_FEATURES_AUX)


def test_angles_become_sin_cos():
    layers = {
        "gm_gr_all": np.zeros((2, 2), np.float32),
        "gm_gr_1GFI_25": np.full((2, 2), 90.0, np.float32),
    }
    m = build_feature_matrix(layers)
    assert "gm_gr_1GFI_25" not in m.columns
    assert np.allclose(m["gm_gr_1GFI_25_sin"], 1.0)
    assert np.allclose(m["gm_gr_1GFI_25_cos"], 0.0, atol=1e-6)


@pytest.mark.skipif(not DATASET.exists(), reason="dataset_v1.npz не собран")
def test_dataset_v1_passes_registry():
    m = load_feature_matrix()
    # все слои датасета покрыты реестром, стоп-лист не просочился
    assert len(m) == 154 * 149
    assert m.shape[1] == len(config.GOLD_FEATURES_TRAIN) + 2 * len(config.GOLD_FEATURES_ANGLE)
