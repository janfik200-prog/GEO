"""Глубокие модели без учителя для прогнозной оценки: DAGMM, DEC, Deep SVDD, VAE.

ОТКУДА ВЗЯТ НАБОР
-----------------
Архитектуры отобраны по обзорам прогнозной оценки (MPM) 2016-2025, а не по
вкусу. Линия развития в предметной области такая:

* Xiong & Zuo (2016), Computers & Geosciences — глубокий автоэнкодер, ошибка
  реконструкции как мера аномальности. Исходная работа всей линии; у нас уже
  реализована в :mod:`src.atypicality` (неглубокий AE) и :mod:`src.mae_model`
  (свёрточный masked-AE на патчах);
* Luo et al. (2020), Applied Geochemistry — вариационный автоэнкодер; скор
  складывается из ошибки реконструкции и KL-члена, то есть учитывает не только
  «плохо восстановилось», но и «попало в маловероятную область латента»;
* deep AE + смесь гауссиан (DAGMM, Zong et al. 2018; в геохимии — deep
  autoencoder Gaussian mixture model, 2025) — латент и признаки ошибки
  реконструкции подаются в оценочную сеть, которая параметризует смесь
  гауссиан; скор = энергия выборки. Ключевое отличие от AE: плотность
  оценивается СОВМЕСТНО с обучением латента, а не поверх готового;
* Deep Embedded Clustering (DEC, Xie et al. 2016; в MPM — Sci Rep 2025, где
  DEC-варианты дали prediction rate 69-72% против 66-68% у k-means и GMM) —
  латент и разбиение на кластеры оптимизируются одновременно;
* Deep SVDD (Ruff et al. 2018) — одноклассовая гиперсфера: контроль семейства,
  самый жёсткий вариант «нормальность = компактность».

ЧТО ЗДЕСЬ ПРИНЦИПИАЛЬНО
-----------------------
Ни одна модель не видит ни точек заверки, ни критериального прогноза. Все они
отвечают на вопрос «насколько эта ячейка не похожа на фон листа», причём
разными способами: невязкой (VAE), плотностью смеси (DAGMM), принадлежностью
кластеру (DEC), расстоянием до центра гиперсферы (SVDD).

НАПРАВЛЕНИЕ ШКАЛЫ по-прежнему не выводится из данных без меток — все скоры
ненаправленные, и знак согласия с эталоном выбирается на заверке
(:mod:`src.crit_reference`, там же плата за этот выбор).

СЕМЕНА. Этап 5e показал, что одиночное обучение — розыгрыш лотереи: одна и та
же конфигурация при разных семенах давала метрику в диапазоне трёх точек
заверки. Поэтому боевой интерфейс модуля — :func:`ensemble_score`: ранговое
усреднение по нескольким семенам. Одиночные прогоны остаются доступны для
диагностики разброса, но претендентом идёт ансамбль.

Все сети маленькие (тысячи параметров) и считаются на CPU: 22 905 ячеек и
50-60 признаков не требуют и не оправдывают ничего большего.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats
from torch import nn

from . import config


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def _mlp(dims: list[int], bias: bool = True, out_act: bool = False) -> nn.Sequential:
    """Полносвязный стек ``dims`` с GELU между слоями."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2 or out_act:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _batches(n: int, batch: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch):
        yield perm[i:i + batch]


class _AEBase(nn.Module):
    """Общий автоэнкодер: ``dims`` = вход -> скрытые -> латент."""

    def __init__(self, n_in: int, hidden=None, latent: int | None = None,
                 bias: bool = True):
        super().__init__()
        hidden = list(hidden or config.DU_HIDDEN)
        latent = latent or config.DU_LATENT
        self.enc = _mlp([n_in] + hidden + [latent], bias=bias)
        self.dec = _mlp([latent] + hidden[::-1] + [n_in], bias=bias)

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


def _pretrain_ae(X: torch.Tensor, model: _AEBase, epochs: int, batch: int,
                 lr: float, gen: torch.Generator) -> None:
    """Предобучение автоэнкодера на MSE (общий старт для DEC и Deep SVDD)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for idx in _batches(X.shape[0], batch, gen):
            _, rec = model(X[idx])
            loss = ((rec - X[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


# --------------------------------------------------------------------------
# 1. Вариационный автоэнкодер (Luo et al. 2020)
# --------------------------------------------------------------------------

class VAE(nn.Module):
    """VAE; скор аномальности = ошибка реконструкции + beta * KL.

    Смысл двух слагаемых разный, и складывать их осмысленно: большая невязка —
    «фон не умеет такое восстанавливать», большой KL — «код ячейки лежит
    далеко от того, чем фон вообще пользуется». Аномалия может проявиться
    любым из двух способов.
    """

    def __init__(self, n_in: int, hidden=None, latent: int | None = None):
        super().__init__()
        hidden = list(hidden or config.DU_HIDDEN)
        latent = latent or config.DU_LATENT
        self.enc = _mlp([n_in] + hidden, out_act=True)
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.dec = _mlp([latent] + hidden[::-1] + [n_in])

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.dec(z), mu, logvar


def vae_score(X: np.ndarray, seed: int | None = None, epochs: int | None = None,
              beta: float | None = None) -> np.ndarray:
    """Скор VAE (выше = аномальнее)."""
    seed = config.DU_SEED if seed is None else seed
    epochs = epochs or config.DU_EPOCHS
    beta = config.DU_VAE_BETA if beta is None else beta
    _seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    model = VAE(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=config.DU_LR)
    model.train()
    for _ in range(epochs):
        for idx in _batches(Xt.shape[0], config.DU_BATCH, gen):
            rec, mu, logvar = model(Xt[idx])
            recon = ((rec - Xt[idx]) ** 2).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1).mean()
            loss = recon + beta * kl / Xt.shape[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        rec, mu, logvar = model(Xt)
        err = ((rec - Xt) ** 2).mean(1)
        kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1)
    return (err + beta * kl / Xt.shape[1]).numpy().astype(float)


# --------------------------------------------------------------------------
# 2. DAGMM (Zong et al. 2018; в геохимии — deep AE-GMM)
# --------------------------------------------------------------------------

class DAGMM(nn.Module):
    """Автоэнкодер + оценочная сеть смеси гауссиан на расширенном латенте.

    В смесь подаётся не сам латент, а ``[z, cos(x, x_rec), ||x - x_rec|| / ||x||]``:
    два признака ошибки реконструкции добавляют смеси ровно ту информацию,
    ради которой в линии Xiong & Zuo и считалась невязка, но теперь плотность
    оценивается совместно с латентом, а не поверх готового.
    """

    def __init__(self, n_in: int, hidden=None, latent: int | None = None,
                 k: int | None = None):
        super().__init__()
        hidden = list(hidden or config.DU_HIDDEN)
        latent = latent or config.DU_LATENT
        self.k = k or config.DU_DAGMM_K
        self.ae = _AEBase(n_in, hidden, latent)
        self.est = nn.Sequential(nn.Linear(latent + 2, 16), nn.GELU(),
                                 nn.Dropout(0.25), nn.Linear(16, self.k),
                                 nn.Softmax(dim=1))

    def latent(self, x):
        z_c, rec = self.ae(x)
        eps = 1e-8
        cos = (x * rec).sum(1) / (x.norm(dim=1) * rec.norm(dim=1) + eps)
        rel = (x - rec).norm(dim=1) / (x.norm(dim=1) + eps)
        return torch.cat([z_c, cos.unsqueeze(1), rel.unsqueeze(1)], dim=1), rec

    def gmm_params(self, z, gamma):
        n = gamma.shape[0]
        sum_g = gamma.sum(0) + 1e-8                                 # (k,)
        phi = sum_g / n
        mu = (gamma.T @ z) / sum_g.unsqueeze(1)                     # (k, d)
        d = z.unsqueeze(1) - mu.unsqueeze(0)                        # (n, k, d)
        cov = torch.einsum("nk,nkd,nke->kde", gamma, d, d) / sum_g[:, None, None]
        return phi, mu, cov

    def energy(self, z, phi, mu, cov):
        """-log плотности смеси; численно — через разложение Холецкого."""
        k, d, _ = cov.shape
        eye = torch.eye(d, dtype=cov.dtype).unsqueeze(0)
        # Регуляризация берётся ОТНОСИТЕЛЬНОЙ (доля от среднего диагонального
        # элемента) и при неудаче наращивается. Фиксированные 1e-4 разложение не
        # спасали: на пуле эмбеддинга MAE латент садится в подпространство, и
        # Холецкий падал на «не положительно определена». Эскалация нужна ещё и
        # потому, что PSD ковариации гарантирована только в точной арифметике —
        # в float32 собственные значения уходят в минус.
        scale = torch.diagonal(cov, dim1=1, dim2=2).mean().detach().clamp(min=1e-6)
        chol = None
        for mult in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            c, info = torch.linalg.cholesky_ex(cov + eye * (scale * mult + 1e-6))
            if int(info.max()) == 0:
                chol = c
                break
        if chol is None:                     # смесь выродилась полностью
            chol = torch.linalg.cholesky(eye.expand(k, d, d) * scale)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)                     # (n, k, d)
        sol = torch.linalg.solve_triangular(
            chol.unsqueeze(0).expand(z.shape[0], k, d, d),
            diff.unsqueeze(-1), upper=False).squeeze(-1)
        maha = (sol ** 2).sum(-1)                                   # (n, k)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(1)
        log_p = (torch.log(phi + 1e-8).unsqueeze(0) - 0.5 * maha
                 - 0.5 * logdet.unsqueeze(0) - 0.5 * d * np.log(2 * np.pi))
        return -torch.logsumexp(log_p, dim=1)


def dagmm_score(X: np.ndarray, seed: int | None = None,
                epochs: int | None = None) -> np.ndarray:
    """Энергия выборки DAGMM (выше = аномальнее)."""
    seed = config.DU_SEED if seed is None else seed
    epochs = epochs or config.DU_EPOCHS
    _seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    model = DAGMM(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=config.DU_LR)
    model.train()
    for _ in range(epochs):
        for idx in _batches(Xt.shape[0], config.DU_BATCH, gen):
            xb = Xt[idx]
            z, rec = model.latent(xb)
            gamma = model.est(z)
            phi, mu, cov = model.gmm_params(z, gamma)
            energy = model.energy(z, phi, mu, cov).mean()
            # Штраф вырождения ковариаций: без него компонента схлопывается на
            # компактную группу выбросов и объявляет её нормой — та же ловушка,
            # что у смеси гауссиан в лестнице (ANOM_GMM_REG).
            pen = (1.0 / torch.diagonal(cov, dim1=1, dim2=2).clamp(min=1e-6)).sum()
            loss = (((rec - xb) ** 2).mean()
                    + config.DU_DAGMM_L_ENERGY * energy
                    + config.DU_DAGMM_L_COV * pen)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        z, _ = model.latent(Xt)
        gamma = model.est(z)
        phi, mu, cov = model.gmm_params(z, gamma)
        out = model.energy(z, phi, mu, cov)
    return np.nan_to_num(out.numpy().astype(float), nan=0.0,
                         posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# 3. Deep Embedded Clustering (Xie et al. 2016; в MPM — Sci Rep 2025)
# --------------------------------------------------------------------------

class DEC(nn.Module):
    """Автоэнкодер + обучаемые центры кластеров в латенте (мягкое разбиение)."""

    def __init__(self, n_in: int, hidden=None, latent: int | None = None,
                 k: int | None = None):
        super().__init__()
        self.ae = _AEBase(n_in, hidden, latent)
        self.k = k or config.DU_DEC_K
        self.centers = nn.Parameter(
            torch.zeros(self.k, latent or config.DU_LATENT))

    def soft_assign(self, z):
        """Мера принадлежности по t-распределению Стьюдента (alpha = 1)."""
        d2 = torch.cdist(z, self.centers) ** 2
        q = 1.0 / (1.0 + d2)
        return q / q.sum(1, keepdim=True)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Целевое распределение P: заострение Q с нормировкой на размер кластера."""
    w = q ** 2 / q.sum(0)
    return w / w.sum(1, keepdim=True)


def dec_scores(X: np.ndarray, seed: int | None = None,
               epochs: int | None = None) -> dict[str, np.ndarray]:
    """DEC; возвращает ДВА скора — они отвечают на разные вопросы.

    * ``dec_dist`` — расстояние до ближайшего центра в латенте: «ячейка не
      принадлежит толком ни одному кластеру фона»;
    * ``dec_rarity`` — минус логарифм доли кластера, к которому ячейка
      отнесена: «ячейка принадлежит редкому типу площади».

    Разделять обязательно: в литературе по MPM перспективным объявляют именно
    редкий кластер (а какой — решает эксперт по метке или по геологии), но
    выбирать кластер ПО ЗАВЕРКЕ было бы подгонкой. Правило рарности задано
    заранее и от эталона не зависит.
    """
    seed = config.DU_SEED if seed is None else seed
    epochs = epochs or config.DU_EPOCHS
    _seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    model = DEC(Xt.shape[1])
    _pretrain_ae(Xt, model.ae, epochs, config.DU_BATCH, config.DU_LR, gen)

    from sklearn.cluster import KMeans

    model.eval()
    with torch.no_grad():
        Z = model.ae.enc(Xt).numpy()
    km = KMeans(n_clusters=model.k, n_init=10, random_state=seed).fit(Z)
    model.centers.data = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=config.DU_LR)
    model.train()
    target = None
    for ep in range(config.DU_DEC_EPOCHS):
        if ep % config.DU_DEC_UPDATE == 0:
            with torch.no_grad():
                target = _target_distribution(model.soft_assign(model.ae.enc(Xt)))
        for idx in _batches(Xt.shape[0], config.DU_BATCH, gen):
            z = model.ae.enc(Xt[idx])
            q = model.soft_assign(z)
            kl = (target[idx] * (torch.log(target[idx] + 1e-8)
                                 - torch.log(q + 1e-8))).sum(1).mean()
            rec = model.ae.dec(z)
            # Член реконструкции сохраняется (вариант IDEC): без него латент
            # свободно схлопывается в центры и перестаёт описывать данные.
            loss = kl + ((rec - Xt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        z = model.ae.enc(Xt)
        d = torch.cdist(z, model.centers)
        assign = d.argmin(1).numpy()
        dist = d.min(1).values.numpy().astype(float)
    share = np.bincount(assign, minlength=model.k) / assign.size
    rarity = -np.log(np.clip(share[assign], 1e-6, None))
    return {"dec_dist": dist, "dec_rarity": rarity.astype(float)}


# --------------------------------------------------------------------------
# 4. Deep SVDD (Ruff et al. 2018)
# --------------------------------------------------------------------------

def deep_svdd_score(X: np.ndarray, seed: int | None = None,
                    epochs: int | None = None) -> np.ndarray:
    """Квадрат расстояния до центра гиперсферы в латенте (выше = аномальнее).

    Сеть намеренно БЕЗ СМЕЩЕНИЙ, а центр фиксируется после предобучения и не
    обучается: иначе решение схлопывается в тривиальное (сеть выдаёт константу,
    равную центру, и радиус равен нулю для всех) — известное вырождение
    Deep SVDD.
    """
    seed = config.DU_SEED if seed is None else seed
    epochs = epochs or config.DU_EPOCHS
    _seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    ae = _AEBase(Xt.shape[1], bias=False)
    _pretrain_ae(Xt, ae, epochs // 2, config.DU_BATCH, config.DU_LR, gen)

    enc = ae.enc
    enc.eval()
    with torch.no_grad():
        c = enc(Xt).mean(0)
        # Компоненты центра, случайно оказавшиеся около нуля, тоже ведут к
        # схлопыванию: отодвигаем их от нуля (рецепт авторов метода).
        c[c.abs() < 0.1] = 0.1 * torch.sign(c[c.abs() < 0.1] + 1e-12)

    opt = torch.optim.Adam(enc.parameters(), lr=config.DU_LR)
    enc.train()
    for _ in range(epochs // 2):
        for idx in _batches(Xt.shape[0], config.DU_BATCH, gen):
            loss = ((enc(Xt[idx]) - c) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    with torch.no_grad():
        return ((enc(Xt) - c) ** 2).sum(1).numpy().astype(float)


# --------------------------------------------------------------------------
# Реестр и ансамбль по семенам
# --------------------------------------------------------------------------

#: Имя -> функция, возвращающая один скор либо словарь скоров.
METHODS = {
    "vae": vae_score,
    "dagmm": dagmm_score,
    "dec": dec_scores,
    "svdd": deep_svdd_score,
}


def ensemble_score(name: str, X: np.ndarray, seeds=None, log=None,
                   **kwargs) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Ранговый ансамбль по семенам — боевой интерфейс модуля.

    Возвращает ``(скоры ансамбля, скоры по семенам)``. Усреднение по рангам, а
    не по значениям: шкалы энергии, расстояния и невязки несопоставимы, а
    ранговая шкала одинакова у всех семян одной модели.
    """
    seeds = list(seeds or config.DU_SEEDS)
    per_seed: dict[str, list] = {}
    for s in seeds:
        out = METHODS[name](X, seed=s, **kwargs)
        out = out if isinstance(out, dict) else {name: out}
        for k, v in out.items():
            per_seed.setdefault(k, []).append(v)
        if log:
            log(f"  {name}: семя {s} готово")
    ens = {k: np.mean([stats.rankdata(v) / len(v) for v in vs], axis=0)
           for k, vs in per_seed.items()}
    return ens, per_seed
