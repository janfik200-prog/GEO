"""Этап 5b: нейросетевой ПЕРЕНОС НАПРАВЛЕНИЯ с внешних размеченных территорий.

ЗАЧЕМ ЭТОТ МОДУЛЬ ПОСЛЕ ЭТАПА 5. Этап 5 (TinyMAE) измерил структурный предел
любого метода без учителя: он спрашивает «насколько ячейка НЕОБЫЧНА», тогда как
критериальный анализ спрашивает «насколько ячейка БЛАГОПРИЯТНА». Знание о том,
какой конец шкалы признака благоприятен (ближе к разлому — лучше, а не хуже),
в данных без меток отсутствует принципиально: ни архитектура, ни число эпох,
ни размер патча его не создают.

Направление обязано прийти извне. Источников ровно три, и только один честен:

1. экспертный приор — метод превращается в критериальный, победить самого себя
   нельзя;
2. наши 19 несмещённых точек — это тестовая выборка, обучение на ней уничтожает
   заверку;
3. ЧУЖИЕ метки: тысячи рудопроявлений на территориях, где разведка проведена
   (MRDS, США). Сеть учится «как выглядит фертильная геофизическая обстановка»
   там, где меток много, и ЗАМОРОЖЕННОЙ применяется к Анабару.

Этот модуль реализует (3). Наш объект не участвует в обучении ни на одном шаге,
поэтому 19 точек остаются нетронутой независимой заверкой, а сам метод при этом
направленный — то, чего не мог дать MAE.

ОБЩЕЕ ПРИЗНАКОВОЕ ПРОСТРАНСТВО. Переносить можно только то, что измерено и там,
и здесь. Это глобальные гриды GMT 2' (``experiments.common.DS``): магнитное поле
``mag4km``, гравика ``faa``, вертикальный градиент гравики ``vgg``, геоид
``geoid``, рельеф ``relief``. Радиометрия (единственный прямой ключ к урану)
есть только по США — над Сибирью её нет, поэтому она в перенос НЕ входит.

ДВА РЕЖИМА ПРИМЕНЕНИЯ К АНАБАРУ:

* ``global`` — те же глобальные гриды, сэмплированные в центрах наших ячеек.
  Честно, но грубо: 2' на широте 71° — это ~1.2 x 3.7 км, то есть ~1300
  различимых пикселей на 22 905 ячеек;
* ``local`` — наши собственные поля 500 м, приведённые к каналам обучения по
  физическому смыслу (``gm_mg_all``->mag4km, ``gm_gr_all``->faa,
  ``gm_gr_2V_25``->vgg, ``dem_elev``->relief) квантильным сопоставлением.
  Направление берётся из чужих меток, разрешение — из своих данных.

ЕДИНИЦЫ ИЗМЕРЕНИЯ РАЗНЫЕ (нТл на 4 км против условных единиц pgrid), поэтому
сравнивать абсолютные уровни нельзя — сопоставляются РАНГИ (quantile matching).
Это стандартная доменная адаптация по маргиналам; она сохраняет порядок ячеек
внутри листа, а именно порядок и оценивается метрикой lift@10%.

ЧЕСТНЫЕ ОГРАНИЧЕНИЯ, которые нельзя замолчать:

* MRDS — база РАЗВЕДАННОСТИ, а не рудоносности: точки сгущаются вдоль дорог и
  в известных рудных районах. Сеть частично учит географию разведки США;
* США-Кордильера и Анабарский щит — разные геодинамические обстановки. Перенос
  проверяется эмпирически (по нашим 19 точкам), а не постулируется;
* геоид над листом 2 x 0.7 градуса почти постоянен и работает как константа.

Проверка обучения — на ВНЕШНЕЙ территории, пространственной блочной CV по
ячейкам 2 градуса: если сеть не бьёт линейный критериальный индекс там, где
меток тысячи, переносить нечего (стоп-правило ``MIN_CV_AUC``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sps

from cache_paths import MRDS_CSV
from experiments.common import DS, SCALE, fetch, sample_global

MIN_CV_AUC = 0.55          # стоп-правило: ниже — переносить нечего
BLOCK_DEG = 2.0            # сторона блока пространственной CV на внешней территории

# Соответствие «канал обучения -> наш локальный слой 500 м» по физическому
# смыслу. mag4km — магнитное поле, приведённое к высоте 4 км; ближайший наш
# аналог — полное магнитное поле. vgg — вертикальный градиент гравики; ближайший
# аналог — вторая вертикальная производная. geoid локального аналога не имеет
# (длинноволновое поле), берётся из глобального грида как есть.
LOCAL_ANALOGUE: dict[str, str] = {
    "mag4km": "gm_mg_all",
    "faa": "gm_gr_all",
    "vgg": "gm_gr_2V_25",
    "relief": "dem_elev",
    "geoid": "",           # пусто = взять глобальный грид в центрах ячеек
}


# ------------------------------------------------------------------ метки
def load_occurrences(commodities: tuple[str, ...],
                     country: str = "United States",
                     bbox: tuple[float, float, float, float] | None = None,
                     ) -> pd.DataFrame:
    """Рудопроявления MRDS по списку полезных ископаемых.

    ``commodities`` — подстроки в нижнем регистре (``("gold", "uranium")``),
    ищутся в трёх полях commod1..3. ``bbox`` = (lon0, lon1, lat0, lat1).
    """
    cols = ["latitude", "longitude", "commod1", "commod2", "commod3", "country"]
    m = pd.read_csv(MRDS_CSV, low_memory=False, usecols=cols)
    m = m[m["country"] == country].dropna(subset=["latitude", "longitude"])
    txt = (m["commod1"].fillna("") + "," + m["commod2"].fillna("") + ","
           + m["commod3"].fillna("")).str.lower()
    hit = np.zeros(len(m), dtype=bool)
    for c in commodities:
        hit |= txt.str.contains(c, regex=False).to_numpy()
    m = m.loc[hit, ["longitude", "latitude"]].rename(
        columns={"longitude": "lon", "latitude": "lat"})
    if bbox is not None:
        m = m[m["lon"].between(bbox[0], bbox[1]) & m["lat"].between(bbox[2], bbox[3])]
    return m.reset_index(drop=True)


def feature_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Матрица глобальных ковариат (N, len(DS)) в точках (lon, lat)."""
    return np.column_stack([sample_global(d, lon, lat) for d in DS])


# ------------------------------------------------------------------ окрестность (этап 5c)
def sample_global_vec(ds: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Векторный сэмпл глобального грида — как ``sample_global``, но без цикла.

    ``experiments.common.sample_global`` определяет тайл питоновским списковым
    выражением по точкам; для миллионов точек патчевой выборки это часы. Здесь
    номер тайла считается арифметикой над массивами. Тайлы берутся из того же
    кэша тем же ``fetch``, значения — те же (проверяется тестом).
    """
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    out = np.full(lon.size, np.nan)
    lon_sw = (np.floor(lon / 60.0) * 60).astype(np.int64)
    lat_sw = (np.floor((lat + 90) / 60.0) * 60 - 90).astype(np.int64)
    key = lon_sw * 1000 + lat_sw
    for k in np.unique(key[np.isfinite(lon) & np.isfinite(lat)]):
        m = key == k
        lsw, bsw = int(lon_sw[m][0]), int(lat_sw[m][0])
        ln = f"E{lsw:03d}" if lsw >= 0 else f"W{-lsw:03d}"
        lt = f"N{bsw:02d}" if bsw >= 0 else f"S{-bsw:02d}"
        cache_key = (ds, f"{lt}{ln}")
        if cache_key not in _TILES:
            p = fetch(ds, f"{lt}{ln}")
            if p is None:
                _TILES[cache_key] = None
            else:
                a = np.array(Image.open(p)).astype(np.float32)
                a[a == 0] = np.nan
                _TILES[cache_key] = a * SCALE[ds]
        a = _TILES[cache_key]
        if a is None:
            continue
        n = a.shape[0]
        dx = 60.0 / n
        c = ((lon[m] - lsw) / dx).astype(np.int64)
        r = (((bsw + 60) - lat[m]) / dx).astype(np.int64)
        ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
        vals = np.full(m.sum(), np.nan, dtype=np.float32)
        vals[ok] = a[r[ok], c[ok]]
        out[np.flatnonzero(m)] = vals
    return out


_TILES: dict = {}


def patch_lonlat(lon: np.ndarray, lat: np.ndarray, patch: int,
                 step_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Координаты узлов патча вокруг каждого центра — В КИЛОМЕТРАХ, не в пикселях.

    Гриды GMT лежат в географических координатах, поэтому пиксель 2' по долготе
    на широте 40 градусов это 2.8 км, а на широте 71 — уже 1.2 км. Патч,
    отсчитанный в пикселях, охватывал бы над Анабаром вдвое меньшую территорию,
    чем над США, и сеть сравнивала бы разные масштабы. Поэтому шаг задаётся в
    километрах и переводится в градусы по широте каждого центра.
    """
    k = (patch - 1) // 2
    off = np.arange(-k, k + 1) * step_km
    dy, dx = np.meshgrid(off, off, indexing="ij")
    dlat = dy.ravel()[None, :] / 111.32
    dlon = dx.ravel()[None, :] / (111.32 * np.cos(np.radians(lat))[:, None] + 1e-9)
    return lon[:, None] + dlon, lat[:, None] + dlat


def extract_patches(lon: np.ndarray, lat: np.ndarray, patch: int,
                    step_km: float) -> np.ndarray:
    """Патчи глобальных гридов (N, C, patch, patch) вокруг точек."""
    plon, plat = patch_lonlat(np.asarray(lon, float), np.asarray(lat, float),
                              patch, step_km)
    n = len(lon)
    out = np.empty((n, len(DS), patch, patch), dtype=np.float32)
    for j, d in enumerate(DS):
        v = sample_global_vec(d, plon.ravel(), plat.ravel())
        out[:, j] = v.reshape(n, patch, patch)
    return out


def context_features(lon: np.ndarray, lat: np.ndarray, patch: int,
                     step_km: float) -> np.ndarray:
    """Признаки окрестности: значение в центре + среднее и разброс по трём радиусам.

    Дешёвая альтернатива свёрточной сети с тем же смыслом: рудный район — это
    не точка с «правильным» полем, а участок с характерным РИСУНКОМ поля.
    Радиусы вложенные (треть, две трети, весь патч), поэтому признаки описывают
    поле на трёх масштабах сразу.
    """
    return context_from_patches(extract_patches(lon, lat, patch, step_km))


def context_from_patches(P: np.ndarray) -> np.ndarray:
    """То же, но по уже извлечённым патчам (N, C, p, p) — без повторной выборки."""
    k = (P.shape[-1] - 1) // 2
    cols = [P[:, :, k, k]]
    for rad in (max(1, k // 3), max(2, 2 * k // 3), k):
        sub = P[:, :, k - rad:k + rad + 1, k - rad:k + rad + 1]
        cols.append(np.nanmean(sub, axis=(2, 3)))
        cols.append(np.nanstd(sub, axis=(2, 3)))
    return np.concatenate(cols, axis=1).astype(np.float64)


def build_training_set(pos: pd.DataFrame, bbox: tuple[float, float, float, float],
                       n_bg: int, seed: int) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """presence-background выборка на внешней территории.

    Фон — равномерные случайные точки в том же bbox: presence-background учит
    контраст «обстановка рудопроявления против обстановки территории вообще».
    Требуем в 3 раза больше кандидатов фона, чем нужно, потому что часть точек
    попадёт на пропуски гридов (океан, дыры покрытия).
    """
    rng = np.random.default_rng(seed)
    bl = rng.uniform(bbox[0], bbox[1], n_bg * 3)
    ba = rng.uniform(bbox[2], bbox[3], n_bg * 3)
    Xp = feature_matrix(pos["lon"].to_numpy(), pos["lat"].to_numpy())
    Xb = feature_matrix(bl, ba)
    okp, okb = np.isfinite(Xp).all(1), np.isfinite(Xb).all(1)
    Xp, plon, plat = Xp[okp], pos["lon"].to_numpy()[okp], pos["lat"].to_numpy()[okp]
    Xb, blon, blat = Xb[okb][:n_bg], bl[okb][:n_bg], ba[okb][:n_bg]
    X = np.vstack([Xp, Xb])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xb))].astype(np.int64)
    return X, y, np.r_[plon, blon], np.r_[plat, blat]


def spatial_groups(lon: np.ndarray, lat: np.ndarray,
                   deg: float = BLOCK_DEG) -> np.ndarray:
    """Идентификатор блока ``deg`` x ``deg`` градусов для групповой CV.

    Рудопроявления кучкуются в рудных районах; случайная CV измеряла бы память
    о районе, а не обобщение на новую территорию.
    """
    return (np.floor(lon / deg).astype(np.int64) * 100000
            + np.floor(lat / deg).astype(np.int64))


# ------------------------------------------------------------------ нормировка и адаптация
def robust_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Медиана и IQR-масштаб обучающей матрицы (устойчивы к выбросам гридов)."""
    med = np.nanmedian(X, axis=0)
    q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
    scale = (q3 - q1) / 1.349
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return med, scale


def apply_stats(X: np.ndarray, med: np.ndarray, scale: np.ndarray,
                clip: float = 5.0) -> np.ndarray:
    return np.clip((X - med) / scale, -clip, clip).astype(np.float32)


def quantile_match(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Перевести ``x`` в маргинальное распределение ``ref`` по рангам.

    Нужно, когда величина измерена в разных единицах (наш pgrid против нТл GMT)
    или на разном носителе: сопоставляются не значения, а места в ряду. Порядок
    ячеек внутри листа сохраняется — а метрика lift@10% зависит только от него.
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    ref = np.asarray(ref, float)
    ref = np.sort(ref[np.isfinite(ref)])
    out = np.full(x.shape, np.nan)
    if ok.sum() == 0 or ref.size == 0:
        return out
    r = (sps.rankdata(x[ok]) - 0.5) / ok.sum()
    out[ok] = np.quantile(ref, r)
    return out


def harmonize_local(df: pd.DataFrame, Xtr: np.ndarray, lon: np.ndarray,
                    lat: np.ndarray) -> np.ndarray:
    """Локальные поля 500 м в каналах обучения (квантильное сопоставление).

    Для каждого канала обучения берётся наш слой-аналог (``LOCAL_ANALOGUE``) и
    переводится в маргинальное распределение соответствующего столбца обучающей
    матрицы. Канал без локального аналога (geoid) берётся из глобального грида.
    """
    cols = []
    for j, d in enumerate(DS):
        src = LOCAL_ANALOGUE.get(d, "")
        if src and src in df.columns:
            cols.append(quantile_match(df[src].to_numpy(float), Xtr[:, j]))
        else:
            cols.append(quantile_match(sample_global(d, lon, lat), Xtr[:, j]))
    return np.column_stack(cols)


# ------------------------------------------------------------------ нейросеть
def _torch():
    import torch
    return torch


class _BinaryNet:
    """Общая механика обучения бинарного классификатора presence-background.

    Вынесена из ``FertilityNet``, чтобы точечная (MLP) и патчевая (CNN) модели
    учились ОДНИМ И ТЕМ ЖЕ кодом: иначе разница в метриках между ними могла бы
    объясняться разницей в оптимизации, а не в архитектуре.
    """

    net = None

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray | None = None,
            yv: np.ndarray | None = None, epochs: int = 200, batch: int = 512,
            lr: float = 3e-3, seed: int = 42,
            w: np.ndarray | None = None) -> pd.DataFrame:
        """``w`` — веса объектов (доменное перевзвешивание, этап 5d).

        Веса применяются только к обучению; валидационный loss считается без
        них, иначе фолды с разным весовым составом стали бы несравнимыми.
        """
        torch = _torch()
        g = torch.Generator().manual_seed(seed)
        Xt = torch.from_numpy(np.asarray(X, np.float32))
        yt = torch.from_numpy(np.asarray(y, np.float32))
        wt = (None if w is None
              else torch.from_numpy(np.asarray(w, np.float32)))
        # Позитивов на порядок меньше фона — веса классов выравнивают вклад,
        # иначе сеть выучит «всё пусто» и будет формально права.
        w_pos = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(w_pos, dtype=torch.float32))
        loss_el = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(w_pos, dtype=torch.float32),
            reduction="none")
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        hist, best, best_state = [], np.inf, None
        for ep in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                out = self.net(Xt[idx]).squeeze(1)
                if wt is None:
                    loss = loss_fn(out, yt[idx])
                else:
                    ww = wt[idx]
                    loss = (loss_el(out, yt[idx]) * ww).sum() / ww.sum()
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            rec = {"epoch": ep, "train_loss": tot / len(perm)}
            if Xv is not None:
                self.net.eval()
                with torch.no_grad():
                    lv = loss_fn(self.net(torch.from_numpy(
                        np.asarray(Xv, np.float32))).squeeze(1),
                        torch.from_numpy(np.asarray(yv, np.float32)))
                rec["val_loss"] = float(lv)
                if rec["val_loss"] < best:
                    best = rec["val_loss"]
                    best_state = {k: v.clone()
                                  for k, v in self.net.state_dict().items()}
            hist.append(rec)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return pd.DataFrame(hist)

    def predict(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        """Вероятность фертильности; строки с пропусками возвращаются как NaN."""
        torch = _torch()
        X = np.asarray(X, float)
        ok = np.isfinite(X).all(axis=tuple(range(1, X.ndim)))
        out = np.full(len(X), np.nan)
        if ok.sum() == 0:
            return out
        self.net.eval()
        parts = []
        with torch.no_grad():
            Xt = torch.from_numpy(X[ok].astype(np.float32))
            for i in range(0, len(Xt), batch):
                parts.append(torch.sigmoid(
                    self.net(Xt[i:i + batch]).squeeze(1)).numpy())
        out[ok] = np.concatenate(parts)
        return out


class FertilityNet(_BinaryNet):
    """MLP-классификатор «фертильности» обстановки: P(рудопроявление | геофизика).

    Сеть намеренно крошечная (два скрытых слоя): сложность ограничена не числом
    меток (их тысячи), а тем, что модель обязана ПЕРЕНОСИТЬСЯ на другой
    континент — переученная на деталях США она перенос завалит. Dropout и выбор
    состояния по блочной CV — против этого же.
    """

    def __init__(self, n_in: int, hidden: tuple[int, int] = (64, 32),
                 dropout: float = 0.2, seed: int = 42):
        torch = _torch()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden[0]), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[0], hidden[1]), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[1], 1),
        )


class FertilityCNN(_BinaryNet):
    """Свёрточный классификатор фертильности по ПАТЧУ полей вокруг точки.

    Соединяет два результата предыдущих этапов: контекст (этап 5 показал, что
    окрестность несёт сигнал, которого нет в векторе одной ячейки) и направление
    (этап 5b — направление приходит из чужих меток). Сеть видит не пять чисел,
    а рисунок пяти полей на площадке в десятки километров — масштабе рудного
    района, и учится отличать рисунок рудоносной обстановки от фоновой.

    Глобальный average pooling в конце делает ответ независимым от того, где
    именно внутри патча лежит особенность: рудопроявление на карте MRDS
    привязано с точностью до сотен метров, а патч — десятки километров.
    """

    def __init__(self, n_ch: int, patch: int, hid: int = 16, dropout: float = 0.2,
                 seed: int = 42):
        torch = _torch()
        torch.manual_seed(seed)
        self.patch = patch
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_ch, hid, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(hid, hid, 3, stride=2, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(hid, hid * 2, 3, stride=2, padding=1), torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hid * 2, 1),
        )


def prep_patches(P: np.ndarray, med: np.ndarray, scale: np.ndarray,
                 clip: float = 5.0) -> np.ndarray:
    """Патчи -> вход CNN: робастные z + канал валидности на каждый признак.

    Пропуски (океан, дыры покрытия гридов) заменяются нулём, то есть медианой
    после нормировки, и отмечаются отдельным каналом. Выбрасывать патч целиком
    нельзя: у побережья пропуск есть почти всегда, и выборка перекосилась бы
    вглубь континента.
    """
    Z = (P - med[None, :, None, None]) / scale[None, :, None, None]
    v = np.isfinite(Z).astype(np.float32)
    Z = np.clip(np.nan_to_num(Z, nan=0.0), -clip, clip).astype(np.float32)
    return np.concatenate([Z, v], axis=1)


class DistilNet:
    """Сеть-ученик: переносит НАПРАВЛЕНИЕ учителя в локальный пул признаков.

    Учитель (перенос с чужих меток) знает, какая обстановка благоприятна, но
    видит только пять глобальных полей с шагом 2' — на нашем листе это грубее
    самих данных листа. Ученик учится воспроизводить логит учителя по 52
    локальным признакам (магнито-гравитационные трансформанты 500 м,
    линеаменты, Sentinel-2, рельеф) и тем самым выражает то же направление на
    том разрешении, на котором эти признаки измерены.

    Меток объекта в обучении нет ни у учителя, ни у ученика — 19 несмещённых
    точек остаются независимой заверкой. Регрессия, а не классификация: цель —
    непрерывный логит, и порядок ячеек (единственное, от чего зависит lift)
    сохраняется монотонно.
    """

    def __init__(self, n_in: int, hidden: tuple[int, int] = (64, 32),
                 dropout: float = 0.2, seed: int = 42):
        torch = _torch()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden[0]), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[0], hidden[1]), torch.nn.GELU(),
            torch.nn.Linear(hidden[1], 1),
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def fit(self, X: np.ndarray, t: np.ndarray, Xv=None, tv=None,
            epochs: int = 300, batch: int = 512, lr: float = 3e-3,
            seed: int = 42) -> pd.DataFrame:
        torch = _torch()
        g = torch.Generator().manual_seed(seed)
        Xt = torch.from_numpy(np.asarray(X, np.float32))
        tt = torch.from_numpy(np.asarray(t, np.float32))
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        hist, best, best_state = [], np.inf, None
        for ep in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]).squeeze(1), tt[idx])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            rec = {"epoch": ep, "train_loss": tot / len(perm)}
            if Xv is not None:
                self.net.eval()
                with torch.no_grad():
                    lv = loss_fn(self.net(torch.from_numpy(
                        np.asarray(Xv, np.float32))).squeeze(1),
                        torch.from_numpy(np.asarray(tv, np.float32)))
                rec["val_loss"] = float(lv)
                if rec["val_loss"] < best:
                    best = rec["val_loss"]
                    best_state = {k: v.clone()
                                  for k, v in self.net.state_dict().items()}
            hist.append(rec)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return pd.DataFrame(hist)

    def predict(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        torch = _torch()
        self.net.eval()
        parts = []
        with torch.no_grad():
            Xt = torch.from_numpy(np.asarray(X, np.float32))
            for i in range(0, len(Xt), batch):
                parts.append(self.net(Xt[i:i + batch]).squeeze(1).numpy())
        return np.concatenate(parts)


def domain_weights(X_src: np.ndarray, X_tgt: np.ndarray,
                   clip: float = 10.0, seed: int = 42) -> np.ndarray:
    """Веса объектов-источников по сходству с целевым доменом (covariate shift).

    Сеть учится на США, а применяется на Анабаре: часть обучающей выборки
    (кордильерские обстановки) в нашем листе не встречается вовсе, и правило,
    выученное на ней, переносить не на что. Логистический дискриминатор
    «источник или цель» даёт отношение плотностей p/(1-p); им перевзвешивается
    обучение, и сеть уделяет внимание тем участкам признакового пространства,
    которые на листе реально есть.

    Обрезка сверху обязательна: без неё несколько экстремальных объектов
    забирают почти весь вес и выборка эффективно схлопывается.
    """
    from sklearn.linear_model import LogisticRegression
    Xs = np.nan_to_num(np.asarray(X_src, float))
    Xt = np.nan_to_num(np.asarray(X_tgt, float))
    X = np.vstack([Xs, Xt])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))].astype(int)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-4, 1 - 1e-4)
    w = np.clip(p / (1.0 - p), 0.0, clip)
    m = w.mean()
    return w / m if m > 0 else np.ones(len(Xs))


def patch_stats(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Медиана и IQR-масштаб по каналам патчевой выборки (N, C, p, p)."""
    flat = np.moveaxis(P, 1, 0).reshape(P.shape[1], -1)
    med = np.nanmedian(flat, axis=1)
    q1, q3 = np.nanpercentile(flat, [25, 75], axis=1)
    scale = (q3 - q1) / 1.349
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return med, scale
