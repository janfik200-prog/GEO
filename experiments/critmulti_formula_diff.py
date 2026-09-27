"""Почему расчёт организации ведёт себя иначе, чем сама формула.

ЗАЧЕМ. Наше воссоздание формулы воспроизводит исходный балл листа
(`prognoz`) с ранговой корреляцией +0.984, а многолистовой расчёт
организации (`by_crt`) — только +0.703, и карты заметно разные. Значит
отличается не оцифровка слоёв (они одни и те же), а сам расчёт. Здесь
разбирается, чем именно.

ЧТО СЧИТАЕТСЯ.
1. Свойства балла: сколько различных значений, не квантован ли он.
2. Форма отклика. Для каждого фактора функция вклада восстанавливается из
   данных без предположений о виде (аддитивная модель, обратная подгонка по
   квантильным корзинам): отдельно из `by_crt` и, для контроля, из нативного
   `prognoz` на листе. Рядом рисуется то, что предписывает формула —
   степенная трансформация, min-max и вес. Видно и форму, и радиус влияния.
3. Зональность: веса восстанавливаются отдельно на листе, юго-западе и юге.
   Разные веса по зонам означали бы расчёт по листам с последующей сшивкой.
4. Сглаживание: наше воссоздание сглаживается гауссовым ядром разной ширины,
   ищется ширина, при которой согласие с `by_crt` максимально.
5. Перебор трансформаций: формула строится заново, но степень для каждого
   фактора выбирается перебором (расстояние как есть, корень, кубический
   корень, квадрат) по наилучшему объяснению цели. Контроль — нативный
   `prognoz`, для которого перебор обязан вернуть трансформации из
   `config.TAXONOMY_TRANSFORMS`.
6. Доля балла, объяснённая координатным полиномом второй степени: показывает,
   насколько каждый расчёт превратился в общий наклон по территории.

ВХОД: data/SBORKA_NESKOLKO_LISTOV/ПРОГНОЗ_ВОСПРОИЗВ/PGRID/
ВЫХОД: outputs/02_критериальный_метод/critmulti_formula_diff.png,
       outputs/metrics/critmulti_formula_diff.log
Запуск из корня: python -m experiments.critmulti_formula_diff
"""

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, criterial_target, mapstyle, outputs  # noqa: E402
from experiments.criterial_multisheet_compare import SOUTH, zone_of  # noqa: E402
from experiments.criterial_reconstruct_multisheet import (ROLE_LAYER,  # noqa: E402
                                                          criteria, formula,
                                                          load_layers)
from experiments.honest_feature_signal import poly_design  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BINS = 40          # квантильных корзин на фактор
ROUNDS = 12        # проходов обратной подгонки
RUS = {"tect1": "тектонический СЗ (NW_razl)", "tect2": "тектонический СВ (NE_razl)",
       "facies": "литолого-фациальный", "paleo": "палеогеоморфологический",
       "struct": "структурно-литологический (коры)", "magm": "магматогенный (дайки)"}
TRANSFORMS = {"как есть": lambda d: d, "корень": np.sqrt, "куб.корень": np.cbrt,
              "квадрат": lambda d: d ** 2}
LOG: list[str] = []


def say(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def additive(layers: dict, target: np.ndarray, cells: np.ndarray) -> tuple:
    """Аддитивная модель target ~ sum f_j(слой_j), обратная подгонка.

    Возвращает узлы и значения функций (в сигмах цели) и долю объяснённой
    дисперсии. Форма функций ничем не ограничена — так видно, что на самом
    деле делает расчёт с каждым расстоянием.
    """
    roles = list(ROLE_LAYER)
    y = target[cells]
    y = (y - y.mean()) / y.std()
    edges, idx = {}, {}
    for r in roles:
        v = layers[r][cells]
        e = np.unique(np.quantile(v, np.linspace(0, 1, BINS + 1)))
        edges[r] = e
        idx[r] = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
    f = {r: np.zeros(len(edges[r]) - 1) for r in roles}
    for _ in range(ROUNDS):
        for r in roles:
            resid = y - sum(f[q][idx[q]] for q in roles if q != r)
            m = np.bincount(idx[r], resid, len(f[r]))
            n = np.bincount(idx[r], None, len(f[r]))
            f[r] = np.where(n > 0, m / np.maximum(n, 1), 0.0)
            f[r] -= np.average(f[r], weights=np.maximum(n, 1))
    fit = sum(f[r][idx[r]] for r in roles)
    r2 = 1 - ((y - fit) ** 2).mean() / y.var()
    mids = {r: 0.5 * (edges[r][:-1] + edges[r][1:]) for r in roles}
    return mids, f, float(r2)


def radius95(mid: np.ndarray, fv: np.ndarray) -> float:
    """Расстояние, на котором вклад фактора выбирает 95% своего размаха."""
    lo, hi = fv.min(), fv.max()
    if hi - lo < 1e-9:
        return float("nan")
    lvl = lo + 0.95 * (hi - lo)
    over = np.flatnonzero(fv >= lvl)
    return float(mid[over[0]]) if len(over) else float("nan")


def weights(layers: dict, target: np.ndarray, cells: np.ndarray) -> tuple:
    z, names = criteria(layers, cells)
    A, b = z[cells], target[cells]
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = 1 - ((b - A @ beta) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    return dict(zip(names, beta / beta.max())), float(r2)


def fit_kinds(layers: dict, target: np.ndarray, cells: np.ndarray,
              kinds: tuple) -> tuple:
    """Формула с заданными степенями: min-max по площади и веса методом OLS."""
    cols = []
    for r, k in zip(ROLE_LAYER, kinds):
        v = TRANSFORMS[k](np.clip(layers[r], 0, None))
        lo, hi = float(v[cells].min()), float(v[cells].max())
        cols.append((v - lo) / (hi - lo))
    A = np.column_stack(cols)[cells]
    b = target[cells]
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = 1 - ((b - A @ beta) ** 2).sum() / ((b - b.mean()) ** 2).sum()
    return float(r2), beta / beta.max()


def best_kinds(layers: dict, target: np.ndarray, cells: np.ndarray) -> tuple:
    """Перебор степеней по всем факторам: какая форма объясняет цель лучше."""
    best = None
    for kinds in itertools.product(TRANSFORMS, repeat=len(ROLE_LAYER)):
        r2, w = fit_kinds(layers, target, cells, kinds)
        if best is None or r2 > best[0]:
            best = (r2, w, kinds)
    return best


def main() -> None:
    meta, layers, msk, by_crt, x, y = load_layers()
    zone = zone_of(x, y)
    in_sheet = zone == "лист"
    shape = (meta.prf, meta.pic)

    say("1. СВОЙСТВА БАЛЛА ОРГАНИЗАЦИИ")
    v = by_crt[msk]
    say("   ячеек %d, различных значений %d (%.1f%% от числа ячеек), "
        "от %.4f до %.4f" % (v.size, np.unique(v).size,
                             100 * np.unique(v).size / v.size, v.min(), v.max()))

    # нативный балл листа — контроль формы
    meta_s, prognoz = criterial_target.load_prognoz_grid()
    sx, sy = meta_s.cell_centers()
    tree = cKDTree(np.column_stack([x[msk], y[msk]]))
    d, i = tree.query(np.column_stack([sx.ravel(), sy.ravel()]))
    ok = d <= 375.0
    native = np.full(x.size, np.nan)
    native[np.flatnonzero(msk)[i[ok]]] = prognoz.ravel()[ok]
    nat_cells = msk & np.isfinite(native)

    say("\n2. ФОРМА ОТКЛИКА (аддитивная модель без предположений о виде)")
    mid_c, f_c, r2_c = additive(layers, by_crt, msk)
    say("   by_crt, вся область:      объяснено %.3f дисперсии" % r2_c)
    mid_cs, f_cs, r2_cs = additive(layers, by_crt, msk & in_sheet)
    say("   by_crt, лист:             объяснено %.3f" % r2_cs)
    mid_n, f_n, r2_n = additive(layers, np.nan_to_num(native), nat_cells)
    say("   нативный prognoz, лист:   объяснено %.3f (контроль)" % r2_n)

    # то же в терминах формулы: вклад w_j * z_j, приведённый к сигмам балла
    z_all, names = criteria(layers, msk)
    S = formula(layers, msk)
    sd = S[msk].std()
    say("\n   радиус, на котором фактор выбирает 95%% своего влияния, км:")
    say("   %-34s %8s %8s %8s" % ("фактор", "формула", "by_crt", "prognoz"))
    col = {n: j for j, n in enumerate(names)}   # порядок столбцов criteria() свой
    fam = {}
    for r in ROLE_LAYER:
        w = config.TAXONOMY_WEIGHTS[r]
        dd = layers[r][msk]
        o = np.argsort(dd)
        fam[r] = (dd[o], w * z_all[msk, col[r]][o] / sd)
        say("   %-34s %8.1f %8.1f %8.1f"
            % (RUS[r], radius95(*fam[r]) / 1000,
               radius95(mid_c[r], f_c[r]) / 1000,
               radius95(mid_n[r], f_n[r]) / 1000))

    say("\n   размах вклада фактора, сигм балла:")
    say("   %-34s %8s %8s %8s" % ("фактор", "формула", "by_crt", "prognoz"))
    for r in ROLE_LAYER:
        say("   %-34s %8.2f %8.2f %8.2f"
            % (RUS[r], np.ptp(fam[r][1]), np.ptp(f_c[r]), np.ptp(f_n[r])))

    say("\n3. ВЕСА ПО ЗОНАМ (доля от наибольшего веса)")
    zones = {"вся область": msk, "лист": msk & in_sheet,
             "юго-запад": msk & (zone == "юго-запад"), "юг": msk & (zone == SOUTH)}
    say("   %-14s %6s " % ("зона", "R2") + " ".join("%7s" % r for r in ROLE_LAYER))
    for tag, cells in zones.items():
        if cells.sum() < 500:
            continue
        w, r2 = weights(layers, by_crt, cells)
        say("   %-14s %6.3f " % (tag, r2) + " ".join("%7.3f" % w[r] for r in ROLE_LAYER))
    w0, r0 = weights(layers, np.nan_to_num(native), nat_cells)
    say("   %-14s %6.3f " % ("prognoz/лист", r0)
        + " ".join("%7.3f" % w0[r] for r in ROLE_LAYER))
    say("   %-14s %6s " % ("сама формула", "-")
        + " ".join("%7.3f" % (config.TAXONOMY_WEIGHTS[r]
                              / max(config.TAXONOMY_WEIGHTS.values())) for r in ROLE_LAYER))

    say("\n4. СГЛАЖИВАНИЕ: ширина ядра, при которой наш балл ближе всего к by_crt")
    g_ours = np.where(msk, S, np.nan).reshape(shape)
    good = msk.reshape(shape)
    base = np.nan_to_num(g_ours)
    for km in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        sig = km * 1000 / meta.dx
        if sig == 0:
            sm = base
        else:
            num = gaussian_filter(base * good, sig, mode="nearest")
            den = gaussian_filter(good.astype(float), sig, mode="nearest")
            sm = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
        sv = sm.ravel()[msk]
        say("   ядро %.1f км: ранговая корреляция с by_crt %+.4f"
            % (km, sps.spearmanr(sv, by_crt[msk]).statistic))

    say("\n5. ПЕРЕБОР ТРАНСФОРМАЦИЙ: какая степень расстояния объясняет цель")
    own = tuple(config.TAXONOMY_TRANSFORMS[r] for r in ROLE_LAYER)
    own = tuple({"sqrt": "корень", "cbrt": "куб.корень"}[k] for k in own)
    for tag, target, cells in (("by_crt, вся область", by_crt, msk),
                               ("by_crt, лист", by_crt, msk & in_sheet),
                               ("prognoz, лист (контроль)", np.nan_to_num(native),
                                nat_cells)):
        r2_own, w_own = fit_kinds(layers, target, cells, own)
        r2_best, w_best, kinds = best_kinds(layers, target, cells)
        say("   %s" % tag)
        say("     степени формулы: R2=%.4f, веса " % r2_own
            + " ".join("%s %.2f" % (r, v) for r, v in zip(ROLE_LAYER, w_own)))
        say("     лучшие степени:  R2=%.4f, " % r2_best
            + ", ".join("%s=%s" % (r, k) for r, k in zip(ROLE_LAYER, kinds)))
        say("     их веса:         "
            + " ".join("%s %.2f" % (r, v) for r, v in zip(ROLE_LAYER, w_best)))

    say("\n6. ДОЛЯ БАЛЛА, ОБЪЯСНЁННАЯ КООРДИНАТНЫМ ПОЛИНОМОМ 2-Й СТЕПЕНИ")
    for tag, cells in (("вся область", msk), ("лист", msk & in_sheet),
                       ("вне листа", msk & ~in_sheet)):
        D = poly_design(x[cells], y[cells])
        vals = []
        for v in (S, by_crt):
            b = v[cells]
            vals.append(1 - ((b - D @ np.linalg.lstsq(D, b, rcond=None)[0]) ** 2).sum()
                        / ((b - b.mean()) ** 2).sum())
        say("   %-12s наше воссоздание %.3f, балл организации %.3f"
            % (tag, vals[0], vals[1]))

    # ---- рисунок: функции вклада
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), constrained_layout=True)
    for ax, r in zip(axes.ravel(), ROLE_LAYER):
        dd, ff = fam[r]
        ax.plot(dd / 1000, ff - ff.mean(), color="#2b6cb0", lw=2, label="формула")
        ax.plot(mid_c[r] / 1000, f_c[r], color="#a3122b", lw=2,
                label="расчёт организации (by_crt)")
        ax.plot(mid_n[r] / 1000, f_n[r], color="#f2b544", lw=2, ls="--",
                label="исходный балл листа (prognoz)")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title(RUS[r], fontsize=11)
        ax.set_xlabel("расстояние до объекта фактора, км")
        ax.set_ylabel("вклад в балл, сигм")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Что каждый фактор на самом деле делает с баллом: формула "
                 "против расчёта организации\n(вверх = хуже для "
                 "перспективности; шкала балла обратная)", fontsize=13)
    path = outputs.png("critmulti_formula_diff.png")
    mapstyle.save_fig(fig, path, dpi=180)
    say("\nРисунок: %s" % path)

    out = ROOT / "outputs" / "metrics" / "critmulti_formula_diff.log"
    out.write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
