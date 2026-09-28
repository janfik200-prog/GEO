"""Поячеечная (табличная) постановка задачи: ANFIS и его планки.

ИДЕЯ, КОТОРУЮ ПРОВЕРЯЕМ. Развернуть растр площадки в таблицу: строка — ячейка,
X — все каналы композита в этой ячейке, y — есть ли в ней разметка. Обучить
нечёткую систему вывода Такаги — Сугено (ANFIS): она неглубокая, учится за
минуты и выдаёт правила вида «если яркость низкая И когерентность низкая, то
здесь разлом». Обученное перенести на соседний участок.

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ ОТДЕЛЬНО ОТ САМОЙ МОДЕЛИ. Табличная постановка снимает с
ячейки её окрестность: модель видит спектр точки, но не видит, что точка лежит
на линии. Поэтому у прогона две независимые оси, и их нельзя мешать:

1. *Разбиение.* «Три четверти строк в обучение, четверть в проверку» при
   случайном перемешивании строк — это проверка на соседях обучающих ячеек.
   Композит с шагом 100 м автокоррелирован на километры, поэтому такая проверка
   меряет интерполяцию внутри одного и того же участка, а не перенос. Прогон
   считает ОБА разбиения — ``random`` (буквально по рецепту) и ``block``
   (полосы с эрозией на ``BUFFER_M``, как в остальном проекте) — и печатает их
   рядом. Разница между ними и есть цена утечки, выраженная в lift.
2. *Модель.* ANFIS сравнивается с логистической регрессией и градиентным
   бустингом НА ТЕХ ЖЕ строках и том же разбиении. Без этих планок нельзя
   отличить «нечёткий вывод нашёл закономерность» от «в поячеечных признаках
   вообще что-то есть, и это возьмёт любая модель».

Второй вариант таргета из той же идеи — не 0/1, а плавная принадлежность от 1 в
центре разметки до 0 на ``TARGET_SOFT_M`` — считается как ``--target smooth``.

Итоговая метрика — тот же ``lift@area`` и тот же сдвиговый нуль, что во всех
остальных прогонах ветки, чтобы числа читались в одной шкале.

Запуск из корня репозитория:
    python -m experiments.openlin_anfis --site nevada --apply-to anabar,anabar_dykes
    python -m experiments.openlin_anfis --site nevada,abitibi --apply-to anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.openlin_neigh import neigh_stack, support_radius_m
from experiments.openlin_unet import (CHANNELS, TARGET_HALFWIDTH_M, build_stack,
                                      lift_at_top, load_site, stripe_split)
from src import assessment

M = Path("outputs/metrics")

# --------------------------------------------------------------------------
# Параметры
# --------------------------------------------------------------------------

#: Число правил. Сеточное разбиение (по 2-3 функции принадлежности на вход)
#: при 10 входах дало бы 2^10..3^10 правил — неподъёмно и бессмысленно.
#: Поэтому разбиение рассеянное: правила инициализируются центрами k-means по
#: обучающей выборке, каждое правило — гауссова принадлежность сразу по всем
#: входам. 16 правил — на порядок больше числа каналов и на два порядка меньше
#: числа обучающих строк, то есть модель ещё далека от запоминания выборки.
N_RULES = 16

#: Ширина буфера между обучением и проверкой при блочном разбиении, м.
#: 2 км — с запасом больше и полуширины таргета (150 м), и типичного радиуса
#: автокорреляции композита; без буфера «блочное» разбиение — это то же
#: случайное, только с ровной границей.
BUFFER_M = 2000.0

#: Плавный таргет: принадлежность падает от 1 на разметке до 0 на этом
#: расстоянии. 500 м — удвоенный допуск сверки, дальше связь заведомо не
#: физическая, а натянутая.
TARGET_SOFT_M = 500.0

#: Сколько строк берётся в обучение. Полный лист — 662 тыс. ячеек; на двух
#: площадках это 1.3 млн строк, что для ANFIS избыточно и медленно.
#: 200 тыс. — предел, за которым метрики на проверке перестают меняться.
N_TRAIN_ROWS = 200_000

EPOCHS = 40
BATCH = 4096
LR = 5e-3
SEED = 42

#: Доля площади, на которой считается lift, и допуск сверки. Те же, что в
#: ``openlin_null`` и ``openlin_channels``: 10% площади уводит сдвиговый нуль от
#: вырождения, которое даёт узкая полоса (см. историю прогонов по дайкам).
AREA = 0.10
TOL_M = 250.0

N_SHIFTS = 2000

#: Пре-регистрация. Семейство — ВСЕ прогоны табличной постановки против одной и
#: той же разметки, а не один вызов скрипта: иначе «обучимся ещё на одной
#: площадке» бесплатно ослабляет порог. Полный список кандидатов объявлен до
#: первого просмотра результатов и состоит из трёх запусков:
#:   обучение nevada       -> проверка nevada + перенос anabar, anabar_dykes = 3
#:   обучение anabar       -> проверка anabar + перенос anabar_dykes        = 2
#:   обучение anabar_dykes -> проверка anabar_dykes + перенос anabar        = 2
#: итого 7 пар (площадка обучения, площадка оценки) x 3 модели x 2 таргета = 42.
#: Знаменатель фиксирован по ПОЛНОМУ списку, включая пары, которые могли бы не
#: посчитаться.
#: Случайное разбиение в семейство НЕ входит: оно здесь не кандидат в метод, а
#: измерение утечки, и читается только по lift рядом с блочным.
#: Достижимость порога: минимальное p сдвигового нуля = 1/(1+N_SHIFTS) =
#: 0.0005 < alpha = 0.0012, то есть порог в принципе берётся — но запас всего
#: вдвое, и при N_SHIFTS меньше 1000 тест стал бы непроходимым при любом
#: результате.
ANFIS_N_TESTS = 42
ALPHA = 0.05 / ANFIS_N_TESTS

#: Пре-регистрация прогона с признаками окрестности (``--neigh``). Отдельное
#: слагаемое в своей секции, а не правка ANFIS_N_TESTS: на константу выше уже
#: ссылаются опубликованные числа, и добавление члена туда молча сдвинуло бы
#: порог уже посчитанных прогонов. Список кандидатов тот же самый — 7 пар
#: (площадка обучения, площадка оценки) x 3 модели x 2 таргета = 42, — меняется
#: только вход, поэтому знаменатель складывается.
NB_N_TESTS_PRIOR = ANFIS_N_TESTS
NB_N_TESTS = 42
NB_ALPHA = 0.05 / (NB_N_TESTS_PRIOR + NB_N_TESTS)

#: Достижимость порога. При N_SHIFTS = 2000 минимальное p сдвигового нуля равно
#: 0.0005 при alpha = 0.0006, то есть запас 1.2x: тест взялся бы только идеальным
#: результатом. Поэтому здесь сдвигов вдвое больше — 1/(1+4000) = 0.00025,
#: запас 2.4x, как и у прогона без окрестности. Цена — около 45 с на один тест.
NB_N_SHIFTS = 4000

#: Пре-регистрация обучения СРАЗУ НА НЕСКОЛЬКИХ площадках (``--site a,b``).
#: Своё слагаемое в своей секции. Кандидаты: обучение nevada+abitibi ->
#: проверка на отложенных частях обеих площадок (2) + перенос на anabar и
#: anabar_dykes (2) = 4 пары x 3 модели x 2 таргета = 24.
#: Знаменатель накопительный: 168 предыдущих фаз (42 поячеечных + 42 с
#: окрестностью + 42 линий + 42 ансамбля) + 3 границ доменов = 171.
MS_N_TESTS_PRIOR = 171
MS_N_TESTS = 24
MS_ALPHA = 0.05 / (MS_N_TESTS_PRIOR + MS_N_TESTS)

#: Достижимость: при 4000 сдвигах минимальное p = 0.00025 против alpha =
#: 0.000256 — запас 1.02x, то есть тест берётся только идеальным результатом.
#: Поэтому здесь сдвигов вдвое больше: 1/(1+8000) = 0.000125, запас 2.05x.
MS_N_SHIFTS = 8000


# --------------------------------------------------------------------------
# Табличный датасет
# --------------------------------------------------------------------------

def soft_target(dist: np.ndarray) -> np.ndarray:
    """Плавная принадлежность «здесь разлом»: 1 на разметке, 0 за TARGET_SOFT_M."""
    return np.clip(1.0 - dist / TARGET_SOFT_M, 0.0, 1.0).astype("float32")


def block_split(shape: tuple[int, int], res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Обучающая и проверочная маски полос с выеденным буфером между ними.

    Возвращает ``(train, val)``; ячейки ближе ``BUFFER_M`` к границе не попадают
    никуда. Именно этот выброс отличает блочную проверку от случайной: без него
    проверочная ячейка стоит вплотную к обучающей и меряется не перенос, а
    интерполяция.
    """
    from scipy.ndimage import binary_erosion

    val = stripe_split(shape)
    k = int(round(BUFFER_M / res_m))
    st = np.ones((2 * k + 1, 2 * k + 1), dtype=bool)
    val_core = binary_erosion(val, st, border_value=1)
    train_core = binary_erosion(~val, st, border_value=1)
    return train_core, val_core


def random_split(shape: tuple[int, int], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Разбиение строк 3/4 - 1/4 случайным перемешиванием, как в исходной идее."""
    rng = np.random.default_rng(SEED)
    pick = rng.random(shape) < 0.75
    return pick & valid, (~pick) & valid


def rows(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Каналы (C,H,W) -> строки (N,C) по маске."""
    return x[:, mask].T.astype("float32")


# --------------------------------------------------------------------------
# ANFIS (Такаги — Сугено первого порядка, рассеянное разбиение)
# --------------------------------------------------------------------------

class Anfis(nn.Module):
    """Слои классического ANFIS, свёрнутые в четыре строки forward.

    Слой 1 — гауссовы функции принадлежности; слой 2 — произведение как t-норма
    (в логарифме это сумма, поэтому считается как ``logmu``); слой 3 —
    нормировка сил правил (``softmax`` от ``logmu`` тождественно равен
    ``mu / sum(mu)``, но не переполняется); слои 4-5 — линейные заключения
    Сугено и их взвешенная сумма.
    """

    def __init__(self, n_in: int, n_rules: int, centers: np.ndarray | None = None):
        super().__init__()
        c = (torch.randn(n_rules, n_in) if centers is None
             else torch.tensor(centers, dtype=torch.float32))
        self.c = nn.Parameter(c)
        self.log_s = nn.Parameter(torch.zeros(n_rules, n_in))
        self.a = nn.Parameter(torch.zeros(n_rules, n_in))
        self.b = nn.Parameter(torch.zeros(n_rules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.exp(self.log_s).clamp_min(1e-3)
        d = (x[:, None, :] - self.c[None]) / s[None]
        logmu = -0.5 * (d * d).sum(-1)
        w = torch.softmax(logmu, dim=1)
        f = x @ self.a.T + self.b[None]
        return (w * f).sum(1)

    def rule_report(self, names: tuple[str, ...]) -> pd.DataFrame:
        """Центры правил в единицах нормированного входа — что модель считает
        «низким» и «высоким» по каждому каналу."""
        c = self.c.detach().numpy()
        df = pd.DataFrame(c, columns=list(names))
        df.insert(0, "правило", range(len(df)))
        df["смещение"] = self.b.detach().numpy()
        return df


def fit_anfis(xtr: np.ndarray, ytr: np.ndarray, n_rules: int,
              soft: bool, log) -> Anfis:
    from sklearn.cluster import KMeans

    torch.manual_seed(SEED)
    km = KMeans(n_clusters=n_rules, n_init=4, random_state=SEED).fit(
        xtr[np.random.default_rng(SEED).choice(len(xtr), size=min(20000, len(xtr)),
                                               replace=False)])
    model = Anfis(xtr.shape[1], n_rules, km.cluster_centers_)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    xt = torch.from_numpy(xtr)
    yt = torch.from_numpy(ytr)
    if soft:
        loss_fn = nn.MSELoss()
    else:
        pos = float(ytr.mean()) or 1e-6
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32))

    n = len(xt)
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            out = model(xt[idx])
            loss = loss_fn(torch.sigmoid(out) if soft else out, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 10 == 9 or ep == 0:
            log(f"    эпоха {ep + 1:>2}/{EPOCHS}  loss {tot / n:.4f}")
    return model


def anfis_predict(model: Anfis, x: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 65536):
            out.append(torch.sigmoid(model(torch.from_numpy(x[i:i + 65536]))).numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Планки на тех же строках
# --------------------------------------------------------------------------

def fit_baselines(xtr: np.ndarray, ytr: np.ndarray, soft: bool) -> dict:
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.linear_model import LogisticRegression, Ridge

    if soft:
        return {"логрегрессия": Ridge().fit(xtr, ytr),
                "бустинг": HistGradientBoostingRegressor(
                    random_state=SEED, max_iter=200).fit(xtr, ytr)}
    yb = (ytr > 0.5).astype(int)
    return {"логрегрессия": LogisticRegression(max_iter=1000,
                                               class_weight="balanced").fit(xtr, yb),
            "бустинг": HistGradientBoostingClassifier(
                random_state=SEED, max_iter=200,
                class_weight="balanced").fit(xtr, yb)}


def baseline_predict(est, x: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return est.predict_proba(x)[:, 1]
    return est.predict(x)


# --------------------------------------------------------------------------
# Оценка
# --------------------------------------------------------------------------

def score_map(pred_rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, np.nan, dtype="float32")
    out[mask] = pred_rows
    return out


def evaluate(score: np.ndarray, dist: np.ndarray, area_mask: np.ndarray,
             with_null: bool, alpha: float = ALPHA,
             n_shifts: int = N_SHIFTS) -> dict:
    got = lift_at_top(np.nan_to_num(score, nan=-1e9), dist, area_mask,
                      fraction=AREA, tol_m=TOL_M)
    row = {"lift": got["lift"], "точность": got["точность"]}
    if with_null:
        meta = SimpleNamespace(prf=score.shape[0], pic=score.shape[1])
        pool = np.flatnonzero(area_mask.ravel())
        target = np.flatnonzero(((dist <= TOL_M) & area_mask).ravel())
        null = assessment.spatial_null_pvalue(
            np.where(area_mask, score.astype(float), np.nan), target, meta,
            area=AREA, pool=pool, n_shifts=n_shifts, seed=SEED)
        row.update({"нуль": null["null_mean"], "q95": null["null_q95"],
                    "p": null["p"], "значим": null["p"] < alpha})
    return row


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="nevada",
                    help="площадка обучения; через запятую — несколько")
    ap.add_argument("--apply-to", default="",
                    help="площадки переноса через запятую")
    ap.add_argument("--rules", type=int, default=N_RULES)
    ap.add_argument("--target", default="both", choices=("hard", "smooth", "both"))
    ap.add_argument("--split", default="both", choices=("random", "block", "both"))
    ap.add_argument("--neigh", action="store_true",
                    help="добавить признаки окрестности (openlin_neigh)")
    ap.add_argument("--no-null", action="store_true")
    args = ap.parse_args()

    def log(msg=""):
        print(msg, flush=True)

    train_sites = [s for s in args.site.split(",") if s]
    apply_sites = [s for s in args.apply_to.split(",") if s]
    tag = "+".join(train_sites) + ("_neigh" if args.neigh else "")
    if len(train_sites) > 1:            # обучение на нескольких площадках
        alpha, n_shifts = MS_ALPHA, MS_N_SHIFTS
        n_tests = MS_N_TESTS_PRIOR + MS_N_TESTS
    elif args.neigh:
        alpha, n_shifts = NB_ALPHA, NB_N_SHIFTS
        n_tests = NB_N_TESTS_PRIOR + NB_N_TESTS
    else:
        alpha, n_shifts, n_tests = ALPHA, N_SHIFTS, ANFIS_N_TESTS

    log(f"обучение на: {', '.join(train_sites)}")
    log(f"правил {args.rules}, строк в обучение до {N_TRAIN_ROWS}, "
        f"буфер блочного разбиения {BUFFER_M:.0f} м")
    log(f"пре-регистрация: тестов в семействе {n_tests}, alpha = {alpha:.5f}, "
        f"минимально достижимое p = {1 / (1 + n_shifts):.5f}\n")

    # --- данные площадок обучения -----------------------------------------
    data = {}
    for s in train_sites + [a for a in apply_sites if a not in train_sites]:
        z = load_site(s)
        x, y_hard, valid, use = build_stack(z, CHANNELS)
        data[s] = {"x": x, "dist": z["dist"], "valid": valid, "use": use,
                   "res_m": float(z["res_m"])}
        log(f"  {s}: каналов {len(use)}, валидных ячеек "
            f"{int(valid.sum()):,}, доля разметки в буфере {TOL_M:.0f} м "
            f"{100 * float(((z['dist'] <= TOL_M) & valid).sum() / valid.sum()):.1f}%")
    use = data[train_sites[0]]["use"]
    for s in data:
        if data[s]["use"] != use:
            raise SystemExit(f"разный набор каналов: {s}")
    log()

    # Признаки окрестности зависят от того, ИЗ ЧЕГО их разрешено собирать:
    # носитель окна (6 км при sigma 2 км) шире буфера разбиения (2 км), поэтому
    # при блочной проверке обучение и проверка получают каждый свою маску, и
    # окно физически не заглядывает за границу. См. шапку openlin_neigh.
    feat_cache: dict[tuple[str, int], np.ndarray] = {}

    def feats(s: str, support: np.ndarray) -> np.ndarray:
        d = data[s]
        if not args.neigh:
            return d["x"]
        key = (s, hash(support.tobytes()))
        if key not in feat_cache:
            nx, _ = neigh_stack(d["x"], d["use"], d["valid"], d["res_m"],
                                support=support)
            feat_cache[key] = np.concatenate([d["x"], nx])
        return feat_cache[key]

    if args.neigh:
        log(f"признаки окрестности: носитель окна {support_radius_m():.0f} м, "
            f"считаются отдельно для каждой части разбиения")

    splits = ("random", "block") if args.split == "both" else (args.split,)
    targets = (("hard", False), ("smooth", True)) if args.target == "both" else (
        (args.target, args.target == "smooth"),)

    rows_out = []
    rng = np.random.default_rng(SEED)

    for split in splits:
        log(f"=== разбиение: {split} " + "=" * 40)
        parts = {}
        for s in train_sites:
            d = data[s]
            if split == "random":
                tr, va = random_split(d["valid"].shape, d["valid"])
            else:
                tr, va = block_split(d["valid"].shape, d["res_m"])
                tr &= d["valid"]
                va &= d["valid"]
            parts[s] = (tr, va)
            log(f"  {s}: обучение {int(tr.sum()):,} яч., проверка {int(va.sum()):,} яч., "
                f"выброшено в буфер {int(d['valid'].sum() - tr.sum() - va.sum()):,}")

        xtr_all = np.concatenate([rows(feats(s, parts[s][0]), parts[s][0])
                                  for s in train_sites])
        dtr_all = np.concatenate([data[s]["dist"][parts[s][0]] for s in train_sites])
        if len(xtr_all) > N_TRAIN_ROWS:
            sel = rng.choice(len(xtr_all), size=N_TRAIN_ROWS, replace=False)
            xtr_all, dtr_all = xtr_all[sel], dtr_all[sel]

        for tname, soft in targets:
            ytr = (soft_target(dtr_all) if soft
                   else (dtr_all <= TARGET_HALFWIDTH_M).astype("float32"))
            log(f"\n  -- таргет {tname} (доля положительных "
                f"{100 * float((ytr > 0.5).mean()):.1f}%)")

            t0 = time.time()
            models = {"ANFIS": fit_anfis(xtr_all, ytr, args.rules, soft, log)}
            log(f"    ANFIS обучен за {time.time() - t0:.0f} с")
            models.update(fit_baselines(xtr_all, ytr, soft))

            for mname, est in models.items():
                def predict(xr):
                    return (anfis_predict(est, xr) if mname == "ANFIS"
                            else baseline_predict(est, xr))

                # проверка на отложенной части площадок обучения
                for s in train_sites:
                    va = parts[s][1]
                    sc = score_map(predict(rows(feats(s, va), va)), va)
                    r = evaluate(sc, data[s]["dist"], va,
                                 with_null=(not args.no_null) and split == "block",
                                 alpha=alpha, n_shifts=n_shifts)
                    rows_out.append({"разбиение": split, "таргет": tname,
                                     "модель": mname, "площадка": s,
                                     "режим": "проверка", **r})

                # перенос: вся площадка целиком
                for s in apply_sites:
                    d = data[s]
                    sc = score_map(predict(rows(feats(s, d["valid"]), d["valid"])),
                                   d["valid"])
                    r = evaluate(sc, d["dist"], d["valid"],
                                 with_null=(not args.no_null) and split == "block",
                                 alpha=alpha, n_shifts=n_shifts)
                    rows_out.append({"разбиение": split, "таргет": tname,
                                     "модель": mname, "площадка": s,
                                     "режим": "перенос", **r})
                    if split == "block" and tname == "hard":
                        np.savez_compressed(
                            M / f"openlin_anfis_{tag}_to_{s}_pred.npz",
                            score=sc, dist=d["dist"], model=mname)

                last = [r for r in rows_out if r["модель"] == mname
                        and r["разбиение"] == split and r["таргет"] == tname]
                for r in last:
                    p = f"  p = {r['p']:.4f}{'  ЗНАЧИМ' if r.get('значим') else ''}" \
                        if "p" in r else ""
                    log(f"    {mname:<14} {r['площадка']:<14} {r['режим']:<9} "
                        f"lift {r['lift']:.3f}{p}")

            if split == "block" and tname == "hard":
                names = use
                if args.neigh:
                    names = tuple(use) + neigh_stack(
                        data[train_sites[0]]["x"], use,
                        data[train_sites[0]]["valid"],
                        data[train_sites[0]]["res_m"])[1]
                models["ANFIS"].rule_report(names).to_csv(
                    M / f"openlin_anfis_{tag}_rules.csv", index=False,
                    encoding="utf-8-sig")
        log()

    df = pd.DataFrame(rows_out)
    out = M / f"openlin_anfis_{tag}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # цена утечки: тот же прогон при случайном и при блочном разбиении
    if {"random", "block"} <= set(df["разбиение"]):
        piv = df[df["режим"] == "проверка"].pivot_table(
            index=["модель", "таргет", "площадка"], columns="разбиение",
            values="lift")
        log("=== цена утечки (lift на проверке) " + "=" * 25)
        log(piv.assign(**{"во сколько раз": piv["random"] / piv["block"]}
                       ).round(3).to_string())
    log(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
