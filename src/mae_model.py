"""Этап 5: крошечный self-supervised свёрточный MAE на патчах.

ЗАЧЕМ ЭТО ПОСЛЕ ЭТАПА 4b
------------------------
Этапы 4 и 4b показали измерением: лестница нетипичности не превосходит
критериальный прогноз, и дело не в наборе признаков — выдача алгоритму тех же
восьми геологических слоёв, из которых собран критериальный прогноз, изменила
результат ровно на 0.00. Причина в том, ЧТО детектор видит: ячейка для него —
вектор чисел, соседняя ячейка — независимое наблюдение. Геолог же читает не
значение в точке, а РИСУНОК поля вокруг неё: узел разломов, форму аномалии,
смену текстуры. Этой информации в векторе признаков одной ячейки нет вообще.

Патчевый masked-автоэнкодер (линия MAE, He et al. 2022; в геологии — GFM4MPM,
уменьшенный здесь на два порядка) — самый дешёвый способ дать модели контекст.
Сеть учится восстанавливать закрытые куски патча по открытым и обязана ради
этого выучить локальный рисунок поля. Меток не используется ни на одном шаге,
поэтому все точки заверки и критериальный прогноз остаются независимыми.

Модель даёт две вещи, и это РАЗНЫЕ величины:

* :func:`reconstruction_error` — насколько ячейка НЕ предсказуема по своему
  окружению (центр патча закрывается принудительно). Это не «нетипичность
  вектора» лестницы, а контекстная неожиданность: ячейка может иметь совершенно
  рядовые значения и всё равно ломать локальный рисунок;
* :func:`embed` — эмбеддинг ячейки (16 чисел), сжатое описание её окружения.
  Поверх него запускается та же лестница из :mod:`src.atypicality`: вопрос
  «есть ли нетипичные ОКРЕСТНОСТИ» вместо «есть ли нетипичные ячейки».

ЧЕСТНОЕ ОГРАНИЧЕНИЕ, зафиксированное до прогона: обе величины по-прежнему
ненаправленные — они говорят «здесь необычно», а не «здесь благоприятно».
Контекст это ортогональное улучшение (модель видит больше), а не устранение
причины разрыва, найденной в 4b. Если и контекст не помогает, вывод будет тот
же и уже без гипотез про недостаток данных.

ПЕРЕОБУЧЕНИЕ И ЧЕСТНАЯ ВАЛИДАЦИЯ. Патчи соседних ячеек перекрываются на 10 из
11 столбцов: случайный holdout лежит целиком внутри обучающих патчей и меряет
память, а не обобщение. Поэтому валидация — блоками 20x20 ячеек (10 км), и
блоки holdout ЭРОДИРОВАНЫ на радиус патча, чтобы валидационный патч не
захватывал обучающую территорию. Число эпох выбирается по этому loss'у.

Запуск обучения — :mod:`experiments.mae_train`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from . import atypicality, config


def build_stack(feat: pd.DataFrame, valid: np.ndarray, rows: np.ndarray,
                cols: np.ndarray, shape: tuple[int, int],
                clip_z: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Растровый стек признаков ``(C, prf, pic)`` и канал валидности ``(prf, pic)``.

    Стандартизация — та же робастная (медиана/IQR), что у лестницы
    (:func:`src.atypicality.prepare_matrix`), чтобы сравнение методов не
    упиралось в разную предобработку. Невалидные ячейки заполняются нулём =
    медианой: сеть не должна принимать дыру за аномалию, а канал валидности
    сообщает ей, где данных нет.

    Обрезка по ``clip_z`` обязательна: в пуле есть колонки с выбросами в
    десятки IQR, и без обрезки MSE-обучение занималось бы только ими, тогда как
    искомый контраст цели — 0.2-1.0 IQR (замер этапа 4).
    """
    clip_z = config.ANOM_CLIP_Z if clip_z is None else clip_z
    X = atypicality.prepare_matrix(feat, valid)      # (n_valid, C), робастные z
    n_cells, n_ch = valid.size, X.shape[1]
    full = np.zeros((n_cells, n_ch), dtype=np.float32)
    full[valid] = np.clip(X, -clip_z, clip_z)

    prf, pic = shape
    stack = np.zeros((n_ch, prf, pic), dtype=np.float32)
    stack[:, rows, cols] = full.T
    vld = np.zeros((prf, pic), dtype=np.float32)
    vld[rows[valid], cols[valid]] = 1.0
    return stack, vld


def pad_stack(stack: np.ndarray, vld: np.ndarray, patch: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Дополнение краёв на радиус патча режимом ``edge`` -> тензоры torch.

    ``edge``, а не нулями: нулевое поле за краем листа — это «медианные
    значения при нулевой валидности», сеть читала бы его как обрыв данных и
    штрафовала бы приграничные ячейки за близость к краю, а не за геологию.
    """
    pad = patch // 2
    s = np.pad(stack, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    v = np.pad(vld, pad, mode="edge")
    return torch.from_numpy(s), torch.from_numpy(v)[None]


def gather_patches(stack_p: torch.Tensor, rows: torch.Tensor,
                   cols: torch.Tensor, patch: int) -> torch.Tensor:
    """Патчи ``(B, C, patch, patch)`` вокруг центров из дополненного стека.

    Индексируется на лету, а не материализуется таблицей: 22 905 патчей по
    52 канала это 576 МБ, тогда как сам стек — 5 МБ.
    """
    ar = torch.arange(patch)
    ridx = (rows[:, None, None] + ar[None, :, None])
    cidx = (cols[:, None, None] + ar[None, None, :])
    return stack_p[:, ridx, cidx].permute(1, 0, 2, 3)


class TinyMAE(nn.Module):
    """Свёрточный masked-автоэнкодер на ~50 тыс. параметров.

    Кодировщик получает патч с ЗАНУЛЁННЫМИ закрытыми позициями плюс два
    служебных канала (валидность данных и сама маска — сеть обязана знать, где
    закрыто, иначе ноль неотличим от медианного значения), сжимает его в вектор
    длины ``emb`` и восстанавливает патч обратно. Декодер собран на
    ``interpolate`` + свёртка, а не на ``ConvTranspose2d``: шахматных артефактов
    на выходе не возникает, а размеры задаются явно.
    """

    def __init__(self, n_ch: int, patch: int | None = None,
                 emb: int | None = None, hid: int | None = None):
        super().__init__()
        patch = config.MAE_PATCH if patch is None else patch
        emb = config.MAE_EMB_DIM if emb is None else emb
        hid = config.MAE_HIDDEN if hid is None else hid
        s1, s2 = (patch + 1) // 2, ((patch + 1) // 2 + 1) // 2
        self.sizes, self.hid, self.n_ch = (patch, s1, s2), hid, n_ch

        self.enc = nn.Sequential(
            nn.Conv2d(n_ch + 2, hid, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid, hid, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hid, hid, 3, stride=2, padding=1), nn.GELU())
        self.to_emb = nn.Linear(hid * s2 * s2, emb)
        self.from_emb = nn.Linear(emb, hid * s2 * s2)
        self.dec1 = nn.Conv2d(hid, hid, 3, padding=1)
        self.dec2 = nn.Conv2d(hid, hid, 3, padding=1)
        self.out = nn.Conv2d(hid, n_ch, 3, padding=1)

    def encode(self, x: torch.Tensor, vld: torch.Tensor,
               m: torch.Tensor) -> torch.Tensor:
        h = self.enc(torch.cat([x * (1.0 - m), vld, m], dim=1))
        return self.to_emb(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        p, s1, s2 = self.sizes
        h = self.from_emb(z).view(-1, self.hid, s2, s2)
        h = F.gelu(self.dec1(F.interpolate(h, size=(s1, s1), mode="nearest")))
        h = F.gelu(self.dec2(F.interpolate(h, size=(p, p), mode="nearest")))
        return self.out(h)

    def forward(self, x: torch.Tensor, vld: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x, vld, m))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_mask(batch: int, patch: int, ratio: float, gen: torch.Generator,
                force_center: bool = False) -> torch.Tensor:
    """Маска ``(B, 1, patch, patch)``: 1 = позиция закрыта.

    Доля закрытого фиксирована по построению (берётся ровно ``ratio`` позиций
    с наименьшим шумом), а не биномиальна: иначе сложность задачи гуляет от
    патча к патчу и loss между эпохами несопоставим.
    """
    k = max(1, int(round(ratio * patch * patch)))
    noise = torch.rand(batch, patch * patch, generator=gen)
    idx = noise.argsort(dim=1)[:, :k]
    m = torch.zeros(batch, patch * patch)
    m.scatter_(1, idx, 1.0)
    m = m.view(batch, 1, patch, patch)
    if force_center:
        m[:, :, patch // 2, patch // 2] = 1.0
    return m


def masked_loss(rec: torch.Tensor, x: torch.Tensor, vld: torch.Tensor,
                m: torch.Tensor) -> torch.Tensor:
    """MSE только по закрытым И валидным позициям (как в исходном MAE)."""
    w = m * vld
    denom = w.sum() * rec.shape[1]
    return (((rec - x) ** 2) * w).sum() / denom.clamp(min=1.0)


def block_split(rows: np.ndarray, cols: np.ndarray, patch: int,
                block: int | None = None, val_frac: float | None = None,
                seed: int | None = None) -> np.ndarray:
    """Пространственный holdout блоками: ``True`` = ячейка в валидации.

    Блоки, а не случайные ячейки: патчи соседних ячеек перекрываются на 10 из
    11 столбцов. Валидационные центры дополнительно ЭРОДИРОВАНЫ на радиус
    патча внутрь своего блока — иначе валидационный патч заглядывает в
    обучающий блок и loss занижен.
    """
    block = config.MAE_BLOCK if block is None else block
    val_frac = config.MAE_VAL_FRAC if val_frac is None else val_frac
    seed = config.MAE_SEED if seed is None else seed

    br, bc = rows // block, cols // block
    bid = br * (bc.max() + 1) + bc
    uniq = np.unique(bid)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(val_frac * uniq.size)))
    val_blocks = set(rng.choice(uniq, size=n_val, replace=False).tolist())

    pad = patch // 2
    inner = ((rows % block >= pad) & (rows % block < block - pad)
             & (cols % block >= pad) & (cols % block < block - pad))
    return np.array([b in val_blocks for b in bid]) & inner


def train_mae(stack: np.ndarray, vld: np.ndarray, centers_rc: np.ndarray,
              is_val: np.ndarray, patch: int | None = None,
              epochs: int | None = None, batch: int | None = None,
              lr: float | None = None, mask_ratio: float | None = None,
              seed: int | None = None, log=None
              ) -> tuple[TinyMAE, pd.DataFrame]:
    """Обучение MAE. Возвращает модель с ЛУЧШИМ блочным val-loss и историю.

    Берётся лучшая эпоха, а не последняя: число эпох — гиперпараметр, и
    выбирать его по обучающему loss'у на перекрывающихся патчах бессмысленно.
    """
    patch = config.MAE_PATCH if patch is None else patch
    epochs = config.MAE_EPOCHS if epochs is None else epochs
    batch = config.MAE_BATCH if batch is None else batch
    lr = config.MAE_LR if lr is None else lr
    mask_ratio = config.MAE_MASK_RATIO if mask_ratio is None else mask_ratio
    seed = config.MAE_SEED if seed is None else seed

    torch.manual_seed(seed)
    stack_p, vld_p = pad_stack(stack, vld, patch)
    rr = torch.from_numpy(centers_rc[:, 0].astype(np.int64))
    cc = torch.from_numpy(centers_rc[:, 1].astype(np.int64))
    tr = np.flatnonzero(~is_val)
    va = np.flatnonzero(is_val)

    model = TinyMAE(stack.shape[0], patch=patch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    def run_batch(idx, train: bool):
        x = gather_patches(stack_p, rr[idx], cc[idx], patch)
        v = gather_patches(vld_p, rr[idx], cc[idx], patch)
        m = sample_mask(len(idx), patch, mask_ratio, gen)
        rec = model(x, v, m)
        loss = masked_loss(rec, x, v, m)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach()) * len(idx)

    hist, best = [], (np.inf, None)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(tr)
        tr_loss = sum(run_batch(torch.from_numpy(order[i:i + batch]), True)
                      for i in range(0, order.size, batch)) / max(order.size, 1)
        model.eval()
        # Валидация с ФИКСИРОВАННЫМ зерном маски: иначе разброс между эпохами
        # это разброс масок, а не качества модели.
        gen_va = torch.Generator().manual_seed(seed + 1000)
        with torch.no_grad():
            va_loss = 0.0
            for i in range(0, va.size, batch):
                idx = torch.from_numpy(va[i:i + batch])
                x = gather_patches(stack_p, rr[idx], cc[idx], patch)
                v = gather_patches(vld_p, rr[idx], cc[idx], patch)
                m = sample_mask(len(idx), patch, mask_ratio, gen_va)
                va_loss += float(masked_loss(model(x, v, m), x, v, m)) * len(idx)
            va_loss /= max(va.size, 1)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        if va_loss < best[0]:
            best = (va_loss, {k: t.clone() for k, t in model.state_dict().items()})
        if log is not None:
            log(f"эпоха {ep:3d}: train {tr_loss:.4f}  val(блочный) {va_loss:.4f}"
                + ("  <- лучшая" if va_loss == best[0] else ""))
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return model, pd.DataFrame(hist)


@torch.no_grad()
def embed(model: TinyMAE, stack: np.ndarray, vld: np.ndarray,
          centers_rc: np.ndarray, patch: int | None = None,
          batch: int | None = None) -> np.ndarray:
    """Эмбеддинги ячеек ``(N, emb)``: патч подаётся БЕЗ маскирования."""
    patch = config.MAE_PATCH if patch is None else patch
    batch = config.MAE_BATCH if batch is None else batch
    stack_p, vld_p = pad_stack(stack, vld, patch)
    rr = torch.from_numpy(centers_rc[:, 0].astype(np.int64))
    cc = torch.from_numpy(centers_rc[:, 1].astype(np.int64))
    out = []
    for i in range(0, len(centers_rc), batch):
        sl = slice(i, i + batch)
        x = gather_patches(stack_p, rr[sl], cc[sl], patch)
        v = gather_patches(vld_p, rr[sl], cc[sl], patch)
        m = torch.zeros(x.shape[0], 1, patch, patch)
        out.append(model.encode(x, v, m).numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def reconstruction_error(model: TinyMAE, stack: np.ndarray, vld: np.ndarray,
                         centers_rc: np.ndarray, patch: int | None = None,
                         batch: int | None = None, repeats: int | None = None,
                         mask_ratio: float | None = None,
                         seed: int | None = None) -> np.ndarray:
    """Контекстная неожиданность ячейки ``(N,)``: MSE В ЦЕНТРЕ патча.

    Центр закрывается ПРИНУДИТЕЛЬНО в каждой реализации маски, поэтому величина
    отвечает ровно на вопрос «насколько ячейка предсказуема по своему
    окружению». Ошибка берётся только в центре, а не по всему патчу: иначе
    высокий скор получала бы ячейка, чьи СОСЕДИ необычны, и аномалия
    размазывалась бы на радиус патча.
    """
    patch = config.MAE_PATCH if patch is None else patch
    batch = config.MAE_BATCH if batch is None else batch
    repeats = config.MAE_ERR_REPEATS if repeats is None else repeats
    mask_ratio = config.MAE_MASK_RATIO if mask_ratio is None else mask_ratio
    seed = config.MAE_SEED if seed is None else seed

    stack_p, vld_p = pad_stack(stack, vld, patch)
    rr = torch.from_numpy(centers_rc[:, 0].astype(np.int64))
    cc = torch.from_numpy(centers_rc[:, 1].astype(np.int64))
    c = patch // 2
    acc = np.zeros(len(centers_rc), dtype=np.float64)
    for r in range(repeats):
        gen = torch.Generator().manual_seed(seed + r)
        for i in range(0, len(centers_rc), batch):
            sl = slice(i, i + batch)
            x = gather_patches(stack_p, rr[sl], cc[sl], patch)
            v = gather_patches(vld_p, rr[sl], cc[sl], patch)
            m = sample_mask(x.shape[0], patch, mask_ratio, gen, force_center=True)
            rec = model(x, v, m)
            err = ((rec[:, :, c, c] - x[:, :, c, c]) ** 2).mean(dim=1)
            acc[sl] += err.numpy()
    return acc / repeats
