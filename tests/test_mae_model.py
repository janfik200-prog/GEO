"""Тесты src/mae_model.py: механика патчей и способность MAE ловить
контекстную аномалию (синтетика, без сети и без реальных данных)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src import config, mae_model  # noqa: E402


PRF, PIC, PATCH = 40, 36, 7


@pytest.fixture(scope="module")
def toy():
    """Гладкое поле-«фон» + локальная аномалия, ломающая рисунок.

    Фон — плавная синусоида: он предсказуем по окрестности, и MAE обязан
    научиться его достраивать. Аномалия занимает 3x3 ячейки и по амплитуде
    равна размаху фона: по ЗНАЧЕНИЮ такая ячейка не рекордна (край синусоиды
    даёт столько же), поэтому тест проверяет именно контекст, а не выброс.
    """
    r, c = np.mgrid[0:PRF, 0:PIC]
    f0 = np.sin(r / 4.0) * np.cos(c / 5.0)
    f1 = np.cos((r + c) / 6.0)
    stack = np.stack([f0, f1]).astype(np.float32)
    ar, ac = PRF // 2, PIC // 2
    stack[:, ar - 1:ar + 2, ac - 1:ac + 2] *= -1.0     # переворот знака = разрыв рисунка
    vld = np.ones((PRF, PIC), dtype=np.float32)
    rows, cols = np.mgrid[0:PRF, 0:PIC]
    centers = np.stack([rows.ravel(), cols.ravel()], axis=1)
    return stack, vld, centers, (ar, ac)


def test_build_stack_places_cells_and_marks_invalid():
    n = PRF * PIC
    rows, cols = np.divmod(np.arange(n), PIC)
    feat = pd.DataFrame({"a": np.arange(n, dtype=float),
                         "b": np.arange(n, dtype=float) ** 0.5})
    valid = np.ones(n, dtype=bool)
    valid[:PIC] = False                                # первая строка невалидна
    stack, vld = mae_model.build_stack(feat, valid, rows, cols, (PRF, PIC))

    assert stack.shape == (2, PRF, PIC) and vld.shape == (PRF, PIC)
    assert vld[0].sum() == 0 and vld[1:].min() == 1.0
    # невалидные ячейки — ноль (= медиана после робастной стандартизации)
    assert np.allclose(stack[:, 0, :], 0.0)
    # монотонный признак должен остаться монотонным по валидным ячейкам
    assert stack[0, 1, 0] < stack[0, -1, -1]


def test_gather_patches_returns_the_right_window():
    stack = np.arange(PRF * PIC, dtype=np.float32).reshape(1, PRF, PIC)
    vld = np.ones((PRF, PIC), dtype=np.float32)
    sp, _ = mae_model.pad_stack(stack, vld, PATCH)
    rc = np.array([[10, 12], [20, 5]])
    got = mae_model.gather_patches(sp, torch.from_numpy(rc[:, 0]),
                                   torch.from_numpy(rc[:, 1]), PATCH)
    assert got.shape == (2, 1, PATCH, PATCH)
    p = PATCH // 2
    for i, (r, c) in enumerate(rc):
        exp = stack[0, r - p:r + p + 1, c - p:c + p + 1]   # окно вокруг центра
        assert np.allclose(got[i, 0].numpy(), exp)
        assert got[i, 0, p, p].item() == stack[0, r, c]


def test_sample_mask_ratio_is_exact_and_center_forced():
    gen = torch.Generator().manual_seed(0)
    m = mae_model.sample_mask(16, PATCH, 0.5, gen)
    k = round(0.5 * PATCH * PATCH)
    assert (m.sum(dim=(1, 2, 3)) == k).all()
    m2 = mae_model.sample_mask(16, PATCH, 0.5, gen, force_center=True)
    assert (m2[:, 0, PATCH // 2, PATCH // 2] == 1).all()


def test_block_split_is_spatial_and_eroded():
    rows, cols = np.mgrid[0:PRF, 0:PIC]
    rows, cols = rows.ravel(), cols.ravel()
    is_val = mae_model.block_split(rows, cols, PATCH, block=10, val_frac=0.25,
                                   seed=0)
    assert 0 < is_val.sum() < is_val.size
    pad = PATCH // 2
    # эрозия: валидационный центр не ближе pad к границе своего блока
    assert ((rows[is_val] % 10 >= pad) & (rows[is_val] % 10 < 10 - pad)).all()
    # блочность: валидация не рассыпана по одиночным ячейкам
    assert np.unique(rows[is_val] // 10 * 100 + cols[is_val] // 10).size < 20


def test_model_is_tiny_and_shapes_match(toy):
    stack, vld, centers, _ = toy
    model = mae_model.TinyMAE(stack.shape[0], patch=PATCH)
    assert mae_model.n_params(model) < 100_000
    sp, vp = mae_model.pad_stack(stack, vld, PATCH)
    rc = torch.from_numpy(centers[:4])
    x = mae_model.gather_patches(sp, rc[:, 0], rc[:, 1], PATCH)
    v = mae_model.gather_patches(vp, rc[:, 0], rc[:, 1], PATCH)
    m = mae_model.sample_mask(4, PATCH, 0.5, torch.Generator().manual_seed(0))
    z = model.encode(x, v, m)
    assert z.shape == (4, config.MAE_EMB_DIM)
    assert model(x, v, m).shape == x.shape


@pytest.fixture(scope="module")
def trained(toy):
    stack, vld, centers, _ = toy
    is_val = mae_model.block_split(centers[:, 0], centers[:, 1], PATCH,
                                   block=10, val_frac=0.25, seed=0)
    model, hist = mae_model.train_mae(stack, vld, centers, is_val, patch=PATCH,
                                      epochs=12, batch=128, seed=0)
    return model, hist, stack, vld, centers


def test_training_reduces_holdout_loss(trained):
    _, hist, *_ = trained
    # блочный holdout: падение loss'а здесь — обобщение, а не память
    assert hist["val_loss"].iloc[-3:].min() < 0.6 * hist["val_loss"].iloc[0]


def test_embedding_shape_and_variability(trained):
    model, _, stack, vld, centers = trained
    emb = mae_model.embed(model, stack, vld, centers, patch=PATCH)
    assert emb.shape == (len(centers), config.MAE_EMB_DIM)
    assert np.isfinite(emb).all() and emb.std(axis=0).max() > 0


def test_reconstruction_error_finds_context_anomaly(trained):
    model, _, stack, vld, centers = trained
    (ar, ac) = (PRF // 2, PIC // 2)
    err = mae_model.reconstruction_error(model, stack, vld, centers,
                                         patch=PATCH, repeats=4)
    anom = np.flatnonzero((np.abs(centers[:, 0] - ar) <= 1)
                          & (np.abs(centers[:, 1] - ac) <= 1))
    top = np.argsort(err)[-int(0.05 * err.size):]
    assert np.isin(anom, top).mean() >= 0.5
