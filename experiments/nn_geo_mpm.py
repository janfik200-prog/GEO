"""Блок 6B: нейросети из литературы по прогнозной оценке — на наших данных.

ЗАЧЕМ ЭТО И ПОЧЕМУ ИМЕННО ЭТИ ДВЕ АРХИТЕКТУРЫ
---------------------------------------------
В проекте уже прогнаны: неглубокий автоэнкодер (`src/atypicality.py`),
свёрточный masked-автоэнкодер на патчах (`src/mae_model.py`), глубокие модели
без учителя (`src/deep_unsup.py`: DAGMM, DEC, VAE, Deep SVDD) и перенос
направления с чужих меток (`src/transfer_nn.py`). Общее у них одно: НИ ОДНА
не обучалась с учителем на нашей собственной мишени. Между тем в предметной
литературе (mineral prospectivity mapping) с 2019 года доминируют ровно два
семейства, и оба — с учителем:

* **свёрточная сеть на многоканальных патчах.** Слои-признаки складываются в
  каналы «изображения», сеть смотрит не на ячейку, а на КАРТИНКУ вокруг неё.
  Работы: лёгкая 2D-FCN по золоту Сюнэршаня (Minerals, 2026, 16(5):450),
  расширение выборки поворотами/отражениями по Фэнсяню (Computers &
  Geosciences, 2022 — LeNet, точность 91.4%, каппа 0.812, ROC-AUC 0.958),
  ансамбль шести классических остовов по Баванггоу (Sci Rep, 2024). Обзор
  (Minerals, 2024, 14(10):1021) объясняет причину моды прямо: разнородные
  геологические слои легко приводятся к формату изображения.
* **графовая свёрточная сеть.** Ячейки — вершины графа, рёбра — соседство
  (пространственное или по сходству признаков), свёртка усредняет признаки
  соседей. Работы: самоансамблирующаяся GCN (Mathematical Geosciences, 2025,
  57:629-656 — обыгрывает GCN, CNN, случайный лес и SVM), GCN с вниманием
  (Ore Geology Reviews, 2025, 180:106554 — 83.6% против 81.9% у обычной GCN),
  гетерогенные графы (Nat Resour Res, 2024, 33:1365-1376), открытая реализация
  GAT/GCN на k-ближайших соседях (github.com/Delavari-Alireza).

Ни того, ни другого у нас не было. Это и есть содержание блока 6B
технологической схемы: под конкретную статью — свой мини-датасет и
воспроизведение метода на наших данных.

ДАТАСЕТ СОБРАН ПОД МЕТОД, А НЕ ВЗЯТ ГОТОВЫМ
-------------------------------------------
Свёрточной сети нужен не таблица «ячейка-строка», а сплошной растровый стек
(каналы x строки x столбцы) без дыр: пропуск внутри патча сеть не умеет
обходить так, как гистограммный бустинг. Поэтому здесь строится отдельный
вход — `stack_image()`: 37 признаков набора переноса раскладываются в
изображение 37 x 149 x 154, пропуски заполняются медианой канала ПО ОБУЧАЮЩИМ
ячейкам фолда, нормировка робастная (медиана/МКР) тоже по обучающим.

Набор именно 37, а не основной 58 — сознательно: группы `l8`, `s1` и `dist`
за контуром листа значат не то же самое (`project-transfer-feature-set-37`), а
одна из целей прогона — расширение территории. Модель, обученная на 58, на
широкую сетку не переносится, и тогда последний пункт задачи невыполним.

ЧТО МЕРЯЕТСЯ И ПРОТИВ ЧЕГО
--------------------------
Протокол — тот же, что для всех прогнозов с 02.09.2026: полосы (4 по X + 4 по
Y) с разрывом 15 км между обучением и тестом. Рядом ОБЯЗАТЕЛЬНО две планки:
случайный уровень и «сколько даёт одна география» (модель на 8x8 гауссовых
базисов, знающая только «где»). Для свёрточной сети вторая планка критична:
патч 7.5 км — это и есть географический контекст, и без планки нельзя отличить
«сеть выучила рисунок поля» от «сеть выучила, в каком углу листа мы находимся».

Отдельно проверено, что патч не заглядывает в обучение: радиус патча 3.5 км
против разрыва 15 км, то есть окно тестовой ячейки целиком лежит в тестовой
полосе. Для графа то же самое считается через радиус приёма: 2 слоя свёртки по
сетке = 2 ячейки = 1 км, тоже сильно меньше разрыва. А вот граф k-ближайших
соседей ПО ПРИЗНАКАМ (как в открытой реализации) связывает ячейки через весь
лист, и его радиус приёма разрывом не ограничен — он считается здесь как
вариант, верный литературе, но с явной пометкой, что его число несопоставимо с
остальными.

Мишень — top-10% критериального прогноза как отдельные ячейки (решение
01.09.2026), то же определение, по которому меряется согласие методов.

ЧТО ПОЛУЧИЛОСЬ (прогон 04.09.2026)
----------------------------------
По ROC-AUC на разрыве 15 км (среднее по 8 полосам): графовая сеть по соседству
0.799, бустинг на тех же 37 признаках 0.749, свёрточная 15x15 — 0.719, планка
«только география» 0.675. Графовая сеть — единственная, чьё превосходство над
географической планкой держится на парном тесте по полосам (+0.124, критерий
Уилкоксона p=0.039); у бустинга то же превосходство (+0.074) статистически не
подтверждается (p=0.25). Разброс по четырём сидам
(`nn_geo_mpm_seeds.py`): у графовой сети 0.7966 +- 0.0030 — вывод от сида не
зависит; у свёрточной 0.7568 +- 0.0292 — зависит сильно, и прогон на сиде 42
оказался её худшим.

Но контроль (`nn_geo_mpm_control.py`) показывает, за счёт чего выигрыш. Обе
архитектуры отличаются от бустинга ровно тем, что видят ОКРЕСТНОСТЬ ячейки.
Если ту же окрестность дать бустингу простым скользящим средним признаков, он
догоняет графовую сеть по ROC-AUC (0.779 против 0.799, парный тест p=0.055) и
ОБГОНЯЕТ обе сети по согласию с критериальной картой: усреднение в окне 15x15
даёт совпадение верх-10% 34.2% и каппа 0.269 против 26.1%/0.178 у свёрточной и
24.4%/0.160 у графовой. То есть вклад нейросетей здесь — это контекст, а не
архитектура, и контекст воспроизводится одной строкой `uniform_filter`.

Отдельный отрицательный результат: граф k-ближайших соседей ПО ПРИЗНАКАМ (то,
что используется в открытой реализации GAT/GCN) оказался ХУЖЕ всех — совпадение
17.9% против 20.4% у голой географии. Связывание похожих по признакам ячеек
через весь лист не помогает, а размывает.

Ни одна карта не проходит порог превосходства на 19 несмещённых точках заверки
(alpha 0.0083 = 0.05 на 6 карт): у всех сетей lift 2.10, ровно как у
критериальной карты, у бустинга 3.16 при p=0.022 — выше порога.

Выход: `outputs/metrics/nn_geo_mpm_folds.csv` (по фолдам),
       `outputs/metrics/nn_geo_mpm.csv` (карта целиком),
       `outputs/metrics/nn_geo_mpm_verdict.csv` (точки заверки),
       `outputs/metrics/nn_geo_mpm_paired.csv` (парные тесты по полосам),
       `data/processed/nn_geo_mpm_scores.parquet`,
       `outputs/nn_geo_mpm_map.png`, `outputs/nn_geo_mpm_wide.png`.
Запуск из корня: ``python -m experiments.nn_geo_mpm``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import N_STRIPS, STRIP_BUFFER_M  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402

T0 = time.time()
SEED = config.CRIT_SEED
CNN_EPOCHS = 25
CNN_BATCH = 256
CNN_BG_RATIO = 2.0        # фона на один позитив в эпохе
GCN_EPOCHS = 300
KNN_K = 8

torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - 1))


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Датасет под свёрточную сеть: растровый стек каналов
# --------------------------------------------------------------------------
def stack_image(X: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Таблица (ячейки x признаки) -> изображение (каналы x строки x столбцы)."""
    return np.ascontiguousarray(X.T.reshape((X.shape[1],) + shape).astype(np.float32))


def robust_stats(X: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Медиана и МКР по ОБУЧАЮЩИМ ячейкам; МКР=0 заменяется на СКО, потом на 1."""
    sub = X[rows]
    med = np.nanmedian(sub, axis=0)
    q1, q3 = np.nanpercentile(sub, [25, 75], axis=0)
    iqr = (q3 - q1) / 1.349
    bad = ~np.isfinite(iqr) | (iqr <= 1e-9)
    if bad.any():
        sd = np.nanstd(sub[:, bad], axis=0)
        iqr[bad] = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return np.nan_to_num(med, nan=0.0), iqr


def normalize(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    """Робастная нормировка + заполнение пропусков нулём (= медианой обучения)."""
    Z = (X - med) / iqr
    Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
    return Z.astype(np.float32)


class PatchCNN(nn.Module):
    """Лёгкая свёрточная сеть в духе 2D-FCN Сюнэршаня: три блока и усреднение.

    Сознательно маленькая (~40 тыс. параметров): позитивов 2295, и сеть уровня
    ResNet-50 из статьи по Баванггоу здесь выучит выборку наизусть. Глобальное
    усреднение вместо полносвязного слоя по всему патчу — чтобы сеть не могла
    запомнить абсолютное положение объекта внутри окна.
    """

    def __init__(self, c_in: int, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(2 * width, 1))

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def make_padded(img: np.ndarray, r: int) -> torch.Tensor:
    """Изображение с полями шириной r (край продлевается) — для краевых патчей."""
    return torch.from_numpy(np.pad(img, ((0, 0), (r, r), (r, r)), mode="edge"))


def take_patches(padded: torch.Tensor, idx: np.ndarray, r: int, ncol: int) -> torch.Tensor:
    """Патчи (B, C, 2r+1, 2r+1) вокруг плоских индексов ячеек."""
    rr = (idx // ncol)[:, None] + np.arange(2 * r + 1)
    cc = (idx % ncol)[:, None] + np.arange(2 * r + 1)
    p = padded[:, torch.from_numpy(rr)[:, :, None], torch.from_numpy(cc)[:, None, :]]
    return p.permute(1, 0, 2, 3).contiguous()


def augment(batch: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Расширение выборки поворотами и отражениями (диэдральная группа, 8 вариантов).

    Приём из работы по Фэнсяню: месторождение — редкое событие, позитивов мало,
    а геологическая обстановка не имеет выделенного «верха», поэтому повороты
    патча законны и увеличивают выборку восьмикратно без новых данных.
    """
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    if k:
        batch = torch.rot90(batch, k, dims=(2, 3))
    if torch.randint(0, 2, (1,), generator=gen).item():
        batch = torch.flip(batch, dims=(3,))
    return batch


def train_cnn(img: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, r: int,
              ncol: int, seed: int = SEED) -> nn.Module:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    padded = make_padded(img, r)
    model = PatchCNN(img.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    pos = tr_idx[y[tr_idx] == 1]
    neg = tr_idx[y[tr_idx] == 0]
    n_neg = min(len(neg), int(CNN_BG_RATIO * len(pos)))
    model.train()
    for ep in range(CNN_EPOCHS):
        sel = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
        rng.shuffle(sel)
        tot = 0.0
        for b in range(0, len(sel), CNN_BATCH):
            bi = sel[b:b + CNN_BATCH]
            xb = augment(take_patches(padded, bi, r, ncol), gen)
            yb = torch.from_numpy(y[bi].astype(np.float32))
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(bi)
        if ep in (0, CNN_EPOCHS // 2, CNN_EPOCHS - 1):
            log(f"    эпоха {ep + 1:2d}/{CNN_EPOCHS}: BCE={tot / len(sel):.4f}")
    return model


@torch.no_grad()
def predict_cnn(model: nn.Module, img: np.ndarray, idx: np.ndarray, r: int,
                ncol: int) -> np.ndarray:
    model.eval()
    padded = make_padded(img, r)
    out = np.empty(len(idx), dtype=np.float64)
    for b in range(0, len(idx), 2048):
        bi = idx[b:b + 2048]
        out[b:b + 2048] = torch.sigmoid(model(take_patches(padded, bi, r, ncol))).numpy()
    return out


# --------------------------------------------------------------------------
# Датасет под графовую сеть: вершины-ячейки и два способа провести рёбра
# --------------------------------------------------------------------------
def grid_edges(shape: tuple[int, int], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Рёбра по соседству на сетке (8 направлений), только между валидными."""
    H, W = shape
    idx = np.arange(H * W).reshape(shape)
    v = valid.reshape(shape)
    ra, rb = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys = slice(max(0, -dy), H - max(0, dy))
        xs = slice(max(0, -dx), W - max(0, dx))
        ys2 = slice(max(0, dy), H - max(0, -dy))
        xs2 = slice(max(0, dx), W - max(0, -dx))
        m = (v[ys, xs] & v[ys2, xs2]).ravel()
        ra.append(idx[ys, xs].ravel()[m])
        rb.append(idx[ys2, xs2].ravel()[m])
    a, b = np.concatenate(ra), np.concatenate(rb)
    return np.concatenate([a, b]), np.concatenate([b, a])


def knn_edges(Z: np.ndarray, pool: np.ndarray, k: int = KNN_K):
    """Рёбра к k ближайшим соседям В ПРОСТРАНСТВЕ ПРИЗНАКОВ (как в открытой реализации).

    Внимание: такие рёбра соединяют ячейки через весь лист, поэтому радиус
    приёма графа не ограничен пространственным разрывом. Результат этого
    варианта нельзя ставить в один ряд с остальными — см. докстринг модуля.
    """
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(Z[pool])
    _, ind = nn_.kneighbors(Z[pool])
    a = np.repeat(pool, k)
    b = pool[ind[:, 1:].ravel()]
    return np.concatenate([a, b]), np.concatenate([b, a])


def norm_adj(src: np.ndarray, dst: np.ndarray, n: int) -> torch.Tensor:
    """Симметрично нормированная матрица смежности с петлями (D^-1/2 (A+I) D^-1/2)."""
    loop = np.arange(n)
    r = np.concatenate([src, loop])
    c = np.concatenate([dst, loop])
    deg = np.bincount(r, minlength=n).astype(np.float64)
    val = 1.0 / np.sqrt(deg[r] * deg[c])
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([r, c])), torch.from_numpy(val).float(), (n, n)
    ).coalesce()


class GCN(nn.Module):
    """Двухслойная графовая свёртка (Kipf & Welling) — базовая форма из статей MPM."""

    def __init__(self, c_in: int, hid: int = 32, p: float = 0.4):
        super().__init__()
        self.l1 = nn.Linear(c_in, hid)
        self.l2 = nn.Linear(hid, hid // 2)
        self.out = nn.Linear(hid // 2, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x, A):
        h = torch.relu(torch.sparse.mm(A, self.l1(self.drop(x))))
        h = torch.relu(torch.sparse.mm(A, self.l2(self.drop(h))))
        return self.out(h).squeeze(-1)


def train_gcn(Z: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, A: torch.Tensor,
              seed: int = SEED) -> np.ndarray:
    torch.manual_seed(seed)
    x = torch.from_numpy(Z)
    yt = torch.from_numpy(y.astype(np.float32))
    tr = torch.from_numpy(tr_idx)
    w = float((y[tr_idx] == 0).sum()) / max(float((y[tr_idx] == 1).sum()), 1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))
    model = GCN(Z.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    model.train()
    for ep in range(GCN_EPOCHS):
        opt.zero_grad()
        loss = lossf(model(x, A)[tr], yt[tr])
        loss.backward()
        opt.step()
        if ep in (0, GCN_EPOCHS - 1):
            log(f"    эпоха {ep + 1:3d}/{GCN_EPOCHS}: BCE={float(loss):.4f}")
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, A)).numpy().astype(np.float64)


# --------------------------------------------------------------------------
# Протокол полос: сплошной прогноз, каждая ячейка предсказана «вслепую»
# --------------------------------------------------------------------------
def strip_ids(coord: np.ndarray):
    edges = np.linspace(coord.min(), coord.max(), N_STRIPS + 1)
    return np.clip(np.digitize(coord, edges[1:-1]), 0, N_STRIPS - 1), edges


def blind(fit_predict, coord: np.ndarray, y: np.ndarray, pool: np.ndarray,
          key: str, ax: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Сплошная карта по полосам одной оси + калибровка квантилями обучения.

    `fit_predict(tr_idx, all_idx) -> вероятности для all_idx`. Калибровка —
    как в `honest_forecast_main`: значение переводится в квантиль относительно
    распределения ЭТОЙ ЖЕ модели на её ОБУЧАЮЩИХ ячейках, иначе полосы
    несопоставимы между собой и общий top-10% собирается из одной полосы.
    """
    sid, edges = strip_ids(coord)
    raw = np.full(y.size, np.nan)
    cal = np.full(y.size, np.nan)
    rows = []
    for k in range(N_STRIPS):
        te = np.flatnonzero((sid == k) & np.isin(np.arange(y.size), pool))
        tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                             (coord > edges[k + 1] + STRIP_BUFFER_M))
                            & np.isin(np.arange(y.size), pool))
        p = fit_predict(tr, np.concatenate([tr, te]))
        p_tr, p_te = p[:len(tr)], p[len(tr):]
        raw[te] = p_te
        srt = np.sort(p_tr)
        cal[te] = np.searchsorted(srt, p_te, side="left") / max(srt.size, 1)
        au = roc_auc_score(y[te], p_te) if len(np.unique(y[te])) > 1 else np.nan
        pr = average_precision_score(y[te], p_te) if len(np.unique(y[te])) > 1 else np.nan
        rows.append({"ключ": key, "полоса": f"{ax}{k}", "n_обучение": len(tr),
                     "поз_обучение": int(y[tr].sum()), "n_тест": len(te),
                     "поз_тест": int(y[te].sum()), "roc_auc": au, "pr_auc": pr})
        log(f"  [{key} {ax}{k}] обучение {len(tr)} ({int(y[tr].sum())} поз.), "
            f"тест {len(te)} ({int(y[te].sum())} поз.), ROC-AUC={au:.4f}")
    return raw, cal, rows


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    cols = [c for c in spec["признаки_перенос"] if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)

    log(f"признаков {len(cols)} (набор переноса), сетка {shape}, пул {pool.size}, "
        f"позитивов {int(y.sum())} ({100 * y.mean():.1f}%)")
    log(f"протокол: полосы {N_STRIPS} на ось, разрыв {STRIP_BUFFER_M / 1000:.0f} км")

    cache = config.PROCESSED_DIR / "nn_geo_mpm_cache.npz"
    fold_rows: list[dict] = []
    scores: dict[str, np.ndarray] = {}

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        scores = {k: z[k] for k in z.files if k != "folds"}
        fold_rows = list(z["folds"])
        log(f"прогнозы по полосам взяты из кэша {cache.name}")
    else:
        # --- свёрточные сети на патчах разного размера --------------------
        for r in (7, 3):
            tag_base = f"свёрточная, патч {2 * r + 1}x{2 * r + 1} ({(2 * r + 1) * 0.5:.1f} км)"
            for ax, coord in (("X", cx), ("Y", cy)):
                def fp(tr, allx, r=r):
                    med, iqr = robust_stats(X, tr)
                    img = stack_image(normalize(X, med, iqr), shape)
                    m = train_cnn(img, y, tr, r, meta.pic)
                    return predict_cnn(m, img, allx, r, meta.pic)
                log(f"{tag_base}, ось {ax}")
                _, cal, rows = blind(fp, coord, y, pool, f"cnn{r}", ax)
                scores[f"cnn{r}_{ax}"] = cal
                fold_rows += rows

        # --- графовые сети -------------------------------------------------
        med, iqr = robust_stats(X, pool)
        Zg = normalize(X, med, iqr)
        graphs = {"grid8": norm_adj(*grid_edges(shape, valid), y.size),
                  "knn8": norm_adj(*knn_edges(Zg, pool, KNN_K), y.size)}
        for gname, A in graphs.items():
            for ax, coord in (("X", cx), ("Y", cy)):
                def fp(tr, allx, A=A):
                    m2, i2 = robust_stats(X, tr)
                    p = train_gcn(normalize(X, m2, i2), y, tr, A)
                    return p[allx]
                log(f"граф {gname}, ось {ax}")
                _, cal, rows = blind(fp, coord, y, pool, f"gcn_{gname}", ax)
                scores[f"gcn_{gname}_{ax}"] = cal
                fold_rows += rows

        # --- планки: бустинг на тех же 37 признаках и «только география» ---
        def fp_boost(tr, allx):
            m = HistBackgroundEnsemble(random_state=SEED).fit(X[tr], y[tr])
            return m.predict_proba(X[allx])[:, 1]

        RB = rbf_basis(cx, cy, RBF_GRID)

        def fp_geo(tr, allx):
            m = HistBackgroundEnsemble(random_state=SEED).fit(RB[tr], y[tr])
            return m.predict_proba(RB[allx])[:, 1]

        for tag, fp_ in (("boost", fp_boost), ("geo", fp_geo)):
            for ax, coord in (("X", cx), ("Y", cy)):
                log(f"{tag}, ось {ax}")
                _, cal, rows = blind(fp_, coord, y, pool, tag, ax)
                scores[f"{tag}_{ax}"] = cal
                fold_rows += rows

        np.savez(cache, folds=np.array(fold_rows, dtype=object), **scores)
        log(f"кэш сохранён: {cache.name}")

    folds = pd.DataFrame(list(fold_rows))
    folds.to_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm_folds.csv", index=False)

    # --- сводка по картам ---------------------------------------------------
    METHODS = {
        "свёрточная, патч 15x15 (7.5 км)": "cnn7",
        "свёрточная, патч 7x7 (3.5 км)": "cnn3",
        "граф GCN, соседство на сетке": "gcn_grid8",
        "граф GCN, kNN по признакам*": "gcn_knn8",
        "бустинг на тех же 37 признаках": "boost",
        "ТОЛЬКО география (планка)": "geo",
    }
    top_c = prognoz.ravel() <= np.quantile(prognoz.ravel()[pool], config.CREF_AREA)
    rows = []
    maps = {}
    for name, key in METHODS.items():
        s = np.nanmean(np.vstack([scores[f"{key}_X"], scores[f"{key}_Y"]]), axis=0)
        s = np.where(np.isfinite(s), s, np.nanmin(s))
        maps[name] = s
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        f_ = folds[folds["ключ"] == key]
        rows.append({
            "метод": name,
            "фолды_roc_auc": float(f_["roc_auc"].mean()) if len(f_) else np.nan,
            "фолды_roc_std": float(f_["roc_auc"].std()) if len(f_) else np.nan,
            "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
            "карта_pr_auc": average_precision_score(y[pool], s[pool]),
            "согласие_каппа": agr["kappa"], "согласие_ро": agr["rho"],
            "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
            "IoU": (top_c & top_m).sum() / max((top_c | top_m).sum(), 1),
        })
    res = pd.DataFrame(rows).sort_values("совпадение_верх_10", ascending=False)
    res.to_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm.csv", index=False)
    print("\n=== Нейросети из литературы против планок (разрыв 15 км) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nСлучайный уровень: ROC-AUC 0.5000, совпадение верх-10% "
          f"{config.CREF_AREA:.0%}, PR-AUC {y[pool].mean():.4f}")
    print("* граф kNN по признакам связывает ячейки через весь лист — его радиус "
          "приёма не ограничен разрывом 15 км, число несопоставимо с остальными")

    # --- косвенная заверка точками -----------------------------------------
    alpha = 0.05 / len(METHODS)
    log(f"заверка точками, порог значимости {alpha:.4f} (0.05 на {len(METHODS)} карт)")
    prot = assessment.run_protocol({**maps, "criterial": crit},
                                   list(maps), meta, valid, alpha,
                                   baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm_verdict.csv",
                           index=False)
    prot["pa"].to_csv(ROOT / "outputs" / "metrics" / "nn_geo_mpm_pa.csv", index=False)
    print("\n=== Точки заверки (косвенная проверка) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = pd.DataFrame({"row": np.repeat(np.arange(meta.prf), meta.pic),
                        "col": np.tile(np.arange(meta.pic), meta.prf),
                        "x": cx, "y": cy, "prognoz": prognoz.ravel(),
                        "target": y, "valid": valid,
                        **{k: v.astype(np.float32) for k, v in maps.items()}})
    out.to_parquet(config.PROCESSED_DIR / "nn_geo_mpm_scores.parquet", index=False)
    log("OK: data/processed/nn_geo_mpm_scores.parquet")

    # --- карты --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size  # noqa: E731
    order = list(res["метод"])
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(rank(crit), cmap="magma")
    axes[0, 0].set_title("Критериальный прогноз\n(эталон геологов)")
    for ax, name in zip(axes.ravel()[1:], order):
        ax.imshow(rank(maps[name]), cmap="magma")
        r_ = res.loc[res["метод"] == name].iloc[0]
        ax.set_title(f"{name}\nсовпадение верх-10%: {r_['совпадение_верх_10']:.0%}, "
                     f"ROC-AUC {r_['карта_roc_auc']:.3f}", fontsize=9)
    axes[1, 3].axis("off")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Нейросети из литературы по прогнозной оценке: прогноз вслепую "
                 f"(обучение не ближе {STRIP_BUFFER_M / 1000:.0f} км), {len(cols)} признаков",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "nn_geo_mpm_map.png", dpi=130)
    log("OK: outputs/nn_geo_mpm_map.png")

    # --- расширение территории ---------------------------------------------
    best_nn = next(n for n in order if n.startswith(("свёрточная", "граф GCN, сосед")))
    log(f"расширение территории лучшей нейросетью: {best_nn}")
    wide_extend(best_nn, X, y, cols, shape, meta, pool)


def wide_extend(best_nn: str, X: np.ndarray, y: np.ndarray, cols: list[str],
                shape: tuple[int, int], meta, pool: np.ndarray) -> None:
    """Перенос на широкую сетку ГГК-200 (303 x 299) — только для свёрточной сети.

    Обучение здесь идёт на всём листе сразу: это не валидация, а получение
    прод-модели для площади, где критериального расчёта нет и проверить
    обобщение честно нечем (та же оговорка, что в `forecast_wide.py`).
    """
    from experiments.fetch_wide_area import WIDE_META

    if not best_nn.startswith("свёрточная"):
        log(f"лучшая сеть — {best_nn}; граф на широкой сетке требует пересборки "
            "смежности, перенос выполняется свёрточной как ближайшей по качеству")
    r = 7 if "15x15" in best_nn else 3

    med, iqr = robust_stats(X, pool)
    img = stack_image(normalize(X, med, iqr), shape)
    model = train_cnn(img, y, pool, r, meta.pic)

    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    Xw = wide[cols].to_numpy(dtype=float)
    # Нормировка ТЕМИ ЖЕ статистиками листа: иначе перенос сравнивал бы ячейки
    # смежной территории между собой, а не с обучающей территорией.
    img_w = stack_image(normalize(Xw, med, iqr), WIDE_META.shape)
    idx_w = np.arange(len(wide))
    score = predict_cnn(model, img_w, idx_w, r, WIDE_META.pic)
    out = wide[["row", "col", "x", "y"]].copy()
    out["score"] = score.astype(np.float32)
    out.to_parquet(config.PROCESSED_DIR / "forecast_wide_cnn.parquet", index=False)
    log(f"OK: data/processed/forecast_wide_cnn.parquet {out.shape}")

    import matplotlib.pyplot as plt
    grid = score.reshape(WIDE_META.shape)
    fig, ax = plt.subplots(figsize=(9, 9.5))
    im = ax.imshow(grid, origin="upper", cmap="viridis",
                   extent=[WIDE_META.x0, WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx,
                           WIDE_META.y0, WIDE_META.y_top])
    ax.plot([meta.x0, meta.x0 + meta.pic * meta.dx, meta.x0 + meta.pic * meta.dx,
             meta.x0, meta.x0],
            [meta.y0, meta.y0, meta.y_top, meta.y_top, meta.y0],
            color="white", lw=1.6, label="контур основного листа (обучение)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Перенос свёрточной сети (патч {2 * r + 1}x{2 * r + 1}) на смежную "
                 f"территорию\nобучение — весь основной лист, {len(cols)} признаков "
                 "набора переноса\nза контуром листа критериального расчёта нет: "
                 "это экстраполяция, а не проверенный прогноз")
    ax.set_xlabel("X, м")
    ax.set_ylabel("Y, м")
    fig.colorbar(im, ax=ax, label="оценка сети (вероятность класса «перспективно»)",
                 fraction=0.03)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "nn_geo_mpm_wide.png", dpi=130)
    log("OK: outputs/nn_geo_mpm_wide.png")


if __name__ == "__main__":
    main()
