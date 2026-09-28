"""«Кора» (алгоритм Бонгарда) на 20 признаках: чем снимок отличается от текстуры.

ПОСТАНОВКА. Два класса ячеек: настоящий снимок и его подделка, у которой та же
текстура, но нет ни одного протяжённого объекта. «Кора» ищет короткие логические
сочетания условий на признаки — «энергия высокая И направление такое-то И гребень
низкий», — которые часто встречаются в одном классе и редко в другом. Ответ
получается в виде правил, читаемых словами. Эталон не нужен: разметки нет ни
в каком виде, правило «эталон только для сверки» соблюдается.

ПОДДЕЛКА ИМЕННО ДЛЯ ЭТОЙ ЗАДАЧИ. Прежняя фазовая подделка
(:func:`openlin_null_input.phase_randomize`) сохраняла у канала только среднее
и разброс, гистограмма становилась нормальной, а каналы подделывались
независимо. «Кора» нашла бы именно это — правила про хвосты гистограмм («очень
яркая ячейка бывает только на снимке») и про сочетание каналов («яркое И низкий
NDVI»), то есть про устройство спектра, а не про расположение на местности.
Здесь подделка многоканальная: одни и те же случайные фазы для всех каналов
(сохраняются спектры и взаимные спектры, Prichard & Theiler 1994) и подгонка
гистограммы каждого канала к исходной в несколько итераций (IAAFT, Schreiber &
Schmitz). У подделки те же гистограммы, та же текстура и те же связи между
каналами — разрушено только расположение.

ПРОВЕРКА. Полосы с буфером 2 км, окрестность признаков считается только внутри
своей части; на проверке — другая реализация подделки. Рядом планка — бустинг
на тех же 20 признаках: без неё не понять, нашла ли «Кора» что-то сверх того,
что возьмёт любая модель.

Запуск из корня репозитория:
    python -m experiments.openlin_kora --site anabar
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

# ---- пре-регистрация (22.09.2026, до первого прогона) ----------------------
#: Уровней на признак: низкий / средний / высокий по терцилям обучающей выборки.
N_LEVELS = 3
#: Наибольшая длина сочетания — классическая «Кора-3».
MAX_ORDER = 3
#: Сочетание характерно для класса, если встречается минимум в ``S_MIN`` его
#: ячеек и минимум в ``R_MIN`` раз чаще, чем в другом классе.
S_MIN = 0.02
R_MIN = 3.0
#: Порог успеха на отложенных полосах.
TEST_AUC_MIN = 0.70
#: Ячеек на класс в обучении и проверке.
N_TRAIN = 20_000
N_TEST = 20_000
#: Итераций подгонки гистограммы в подделке.
IAAFT_ITER = 8
SEED_TRAIN, SEED_TEST = 100, 200
# ---------------------------------------------------------------------------

LEVEL_WORD = ("низк.", "сред.", "высок.")


def surrogate(x: np.ndarray, valid: np.ndarray, rng,
              n_iter: int = IAAFT_ITER) -> np.ndarray:
    """Многоканальная подделка: общие фазы + подгонка гистограмм (C,H,W)."""
    c = x.shape[0]
    med = np.array([np.median(x[k][valid]) for k in range(c)])
    X = np.where(valid[None], x, med[:, None, None]).astype("float64")
    F = np.fft.rfft2(X, axes=(1, 2))
    amp = np.abs(F)
    theta = rng.uniform(-np.pi, np.pi, F.shape[1:])
    theta[0, 0] = 0.0
    Y = np.fft.irfft2(F * np.exp(1j * theta)[None], s=X.shape[1:],
                      axes=(1, 2))
    target = [np.sort(X[k][valid]) for k in range(c)]

    def remap(Y):
        out = Y.copy()
        for k in range(c):
            v = Y[k][valid]
            ranks = np.argsort(np.argsort(v))
            out[k][valid] = target[k][ranks]
        return out

    for _ in range(n_iter):
        Y = remap(Y)
        G = np.fft.rfft2(Y, axes=(1, 2))
        Y = np.fft.irfft2(amp * np.exp(1j * np.angle(G)), s=X.shape[1:],
                          axes=(1, 2))
    Y = remap(Y)
    return np.where(valid[None], Y, 0.0).astype("float32")


def features20(x: np.ndarray, use: tuple, valid: np.ndarray, res_m: float,
               names20: list[str], support: np.ndarray) -> np.ndarray:
    """Те же 20 признаков, окрестность — только внутри ``support``."""
    from experiments.openlin_neigh import neigh_stack

    nx, nnames = neigh_stack(x, use, valid, res_m, base=use, support=support)
    allx = np.concatenate([x, nx])
    alln = list(use) + list(nnames)
    return allx[[alln.index(n) for n in names20]]


def sample_rows(f: np.ndarray, mask: np.ndarray, n: int, rng) -> np.ndarray:
    idx = np.flatnonzero(mask.ravel())
    pick = rng.choice(idx, min(n, len(idx)), replace=False)
    return f.reshape(f.shape[0], -1)[:, pick].T


def binarize(rows: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Строки -> матрица предикатов (N, F*L): «признак j на уровне l»."""
    n, f = rows.shape
    lev = np.stack([np.digitize(rows[:, j], edges[j]) for j in range(f)], 1)
    out = np.zeros((n, f * N_LEVELS), dtype=bool)
    for j in range(f):
        for l in range(N_LEVELS):
            out[:, j * N_LEVELS + l] = lev[:, j] == l
    return out


def kora(b1: np.ndarray, b2: np.ndarray, n_feat: int, log) -> list[dict]:
    """Характерные сочетания обоих классов по Бонгарду.

    Предикаты хранятся упакованными битами, пересечение — побитовое И,
    поддержка — число единиц. Сочетание пропускается, если в нём уже есть
    характерное подсочетание: длинное правило, повторяющее короткое, ничего
    не добавляет.
    """
    p1 = np.packbits(b1, axis=0)
    p2 = np.packbits(b2, axis=0)
    n1, n2 = len(b1), len(b2)
    found: list[dict] = []
    char = {1: set(), 2: set()}

    def support(cols):
        a1 = np.bitwise_and.reduce(p1[:, cols], axis=1)
        a2 = np.bitwise_and.reduce(p2[:, cols], axis=1)
        return (np.bitwise_count(a1).sum() / n1,
                np.bitwise_count(a2).sum() / n2)

    t0 = time.time()
    for order in range(1, MAX_ORDER + 1):
        cnt = 0
        for feats in itertools.combinations(range(n_feat), order):
            for levels in itertools.product(range(N_LEVELS), repeat=order):
                cols = tuple(f * N_LEVELS + l for f, l in zip(feats, levels))
                subs = [s for r in range(1, order)
                        for s in itertools.combinations(cols, r)]
                s1, s2 = support(list(cols))
                for cls, own, other in ((1, s1, s2), (2, s2, s1)):
                    if own < S_MIN or own < R_MIN * max(other, 1e-12):
                        continue
                    if any(s in char[cls] for s in subs):
                        continue
                    char[cls].add(cols)
                    found.append({"класс": cls, "условия": cols,
                                  "длина": order, "своих": own,
                                  "чужих": other,
                                  "перевес": own / max(other, 1e-6)})
                    cnt += 1
        log(f"  сочетаний длины {order}: характерных {cnt} "
            f"({time.time() - t0:.0f} с)")
    return found


def kora_score(b: np.ndarray, rules: list[dict]) -> np.ndarray:
    k1 = [r["условия"] for r in rules if r["класс"] == 1]
    k2 = [r["условия"] for r in rules if r["класс"] == 2]

    def hits(ks):
        if not ks:
            return np.zeros(len(b))
        return np.mean([b[:, list(c)].all(1) for c in ks], axis=0)

    return hits(k1) - hits(k2)


def describe(cols: tuple, names20: list[str]) -> str:
    parts = []
    for c in cols:
        j, l = divmod(c, N_LEVELS)
        n = names20[j].replace("coh_summer_vv_", "лет.ког ").replace(
            "coh_winter_vv_", "зим.ког ")
        parts.append(f"{n} {LEVEL_WORD[l]}")
    return " И ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--swap", action="store_true",
                    help="поменять местами полосы обучения и проверки")
    ap.add_argument("--seed-train", type=int, default=SEED_TRAIN)
    ap.add_argument("--seed-test", type=int, default=SEED_TEST)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    from experiments.openlin_anfis import block_split
    from experiments.openlin_line_rank import auc
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    names20 = pd.read_csv(M / f"openlin_feat20_{a.site}.csv")["признак"].tolist()
    z = load_site(a.site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    train, test = block_split(valid.shape, res_m)
    if a.swap:
        train, test = test, train
    train &= valid
    test &= valid
    log(f"площадка {a.site}: признаков {len(names20)}, ячеек обучения "
        f"{int(train.sum())}, проверки {int(test.sum())}")

    t0 = time.time()
    fk_tr = surrogate(x, valid, np.random.default_rng(a.seed_train))
    fk_te = surrogate(x, valid, np.random.default_rng(a.seed_test))
    log(f"подделки готовы ({time.time() - t0:.0f} с); проверка гистограмм: "
        f"наибольшее расхождение квантилей "
        f"{max(np.abs(np.quantile(x[k][valid], [.05, .5, .95]) - np.quantile(fk_tr[k][valid], [.05, .5, .95])).max() for k in range(len(use))):.3f}")

    F = {}
    for name, src, sup in (("снимок/обуч", x, train), ("снимок/пров", x, test),
                           ("подделка/обуч", fk_tr, train),
                           ("подделка/пров", fk_te, test)):
        F[name] = features20(src, use, valid, res_m, names20, sup)
    log(f"признаки посчитаны ({time.time() - t0:.0f} с)")

    rng = np.random.default_rng(7)
    r_tr = sample_rows(F["снимок/обуч"], train, N_TRAIN, rng)
    f_tr = sample_rows(F["подделка/обуч"], train, N_TRAIN, rng)
    r_te = sample_rows(F["снимок/пров"], test, N_TEST, rng)
    f_te = sample_rows(F["подделка/пров"], test, N_TEST, rng)

    pooled = np.vstack([r_tr, f_tr])
    edges = np.stack([np.quantile(pooled[:, j], np.linspace(0, 1, N_LEVELS + 1)[1:-1])
                      for j in range(pooled.shape[1])])
    b_rtr, b_ftr = binarize(r_tr, edges), binarize(f_tr, edges)
    b_rte, b_fte = binarize(r_te, edges), binarize(f_te, edges)

    log("\nперебор сочетаний:")
    rules = kora(b_rtr, b_ftr, len(names20), log)
    n1 = sum(r["класс"] == 1 for r in rules)
    n2 = sum(r["класс"] == 2 for r in rules)
    log(f"характерных для снимка {n1}, для подделки {n2}")

    y_te = np.r_[np.ones(len(b_rte)), np.zeros(len(b_fte))]
    s_te = kora_score(np.vstack([b_rte, b_fte]), rules)
    auc_kora = auc(y_te, s_te)
    gb = HistGradientBoostingClassifier(max_iter=300, random_state=0)
    gb.fit(np.vstack([r_tr, f_tr]), np.r_[np.ones(len(r_tr)), np.zeros(len(f_tr))])
    auc_gb = auc(y_te, gb.predict_proba(np.vstack([r_te, f_te]))[:, 1])
    ok = auc_kora >= TEST_AUC_MIN
    log(f"\nAUC «снимок против подделки» на отложенных полосах:")
    log(f"  Кора    {auc_kora:.3f}  (порог {TEST_AUC_MIN:.2f}) -> "
        f"{'ПРОЙДЕНО' if ok else 'НЕ пройдено'}")
    log(f"  бустинг {auc_gb:.3f}  (планка)")

    # правила, которые сами по себе держатся на проверке
    rows = []
    for r in rules:
        c = list(r["условия"])
        t1 = b_rte[:, c].all(1).mean()
        t2 = b_fte[:, c].all(1).mean()
        rows.append({"класс": "снимок" if r["класс"] == 1 else "подделка",
                     "правило": describe(r["условия"], names20),
                     "длина": r["длина"],
                     "доля своих (обуч)": r["своих"],
                     "перевес (обуч)": r["перевес"],
                     "доля своих (пров)": t1 if r["класс"] == 1 else t2,
                     "перевес (пров)": (t1 / max(t2, 1e-6)) if r["класс"] == 1
                     else (t2 / max(t1, 1e-6))})
    df = pd.DataFrame(rows).sort_values(["класс", "перевес (пров)"],
                                        ascending=[False, False])
    df.to_csv(M / f"openlin_kora_{a.site}{a.tag}.csv", index=False,
              encoding="utf-8-sig")
    for cls in ("снимок", "подделка"):
        d = df[df["класс"] == cls]
        held = d[d["перевес (пров)"] >= R_MIN]
        log(f"\nправила «{cls}»: {len(d)}, держатся на проверке (перевес >= "
            f"{R_MIN:.0f}) — {len(held)}. Сильнейшие на проверке:")
        for _, r in d.head(8).iterrows():
            # перевес над нулём чужих — это деление на ноль, а не «в 90 тысяч
            # раз»: такие правила печатаются как «у чужих не встретилось»
            other = r["доля своих (пров)"] / max(r["перевес (пров)"], 1e-9)
            rat = (f"x{r['перевес (пров)']:5.1f}" if other * N_TEST >= 1
                   else "  чужих 0")
            log(f"  {rat}  {100 * r['доля своих (пров)']:4.1f} %  "
                f"{r['правило']}")
    log(f"\nсохранено: {M / f'openlin_kora_{a.site}.csv'}")

    # карта оценки по всему листу — на настоящем снимке
    full = features20(x, use, valid, res_m, names20, valid)
    rows_all = full.reshape(full.shape[0], -1)[:, valid.ravel()].T
    score = np.full(valid.shape, np.nan, dtype="float32")
    score[valid] = kora_score(binarize(rows_all, edges), rules)
    np.savez_compressed(M / f"openlin_kora_{a.site}{a.tag}.npz", score=score,
                        auc=auc_kora, auc_gb=auc_gb)
    plot(score, z, valid, a.site + a.tag, auc_kora, auc_gb, log)


def plot(score, z, valid, site, auc_kora, auc_gb, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from experiments.openlin_shade import _stretch

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec",
                                                     "#e34948"])
    fig, ax = plt.subplots(1, 2, figsize=(19, 8.6), facecolor="#fcfcfb")
    ax[0].imshow(_stretch(np.where(valid, z["gray"], np.nan)), cmap="gray",
                 vmin=0, vmax=1)
    ax[0].set_title("летний композит, 100 м", fontsize=14, loc="left")
    lim = float(np.nanpercentile(np.abs(score), 99)) or 1.0
    im = ax[1].imshow(score, cmap=cmap, vmin=-lim, vmax=lim)
    ax[1].set_title(f"оценка «Коры»: красное — сочетания, характерные для "
                    f"снимка,\nсинее — для текстуры. AUC на проверке "
                    f"{auc_kora:.2f} (бустинг {auc_gb:.2f})", fontsize=14,
                    loc="left")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
        for s in a_.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    p = OUT / f"кора_{site}.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
