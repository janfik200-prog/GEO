"""Гейт ANFIS для оператора границы: сила скачка в сигмах СВОЕГО домена.

ЗАЧЕМ. Оператор ``openlin_domains`` меряет, насколько скачком меняется
признаковый вектор при пересечении линии, и выражает это в сигмах, посчитанных
по ВСЕЙ площадке. Но одна сигма значит разное в разных местах: на однородном
плато скачок в сигму — событие, в пёстрой залесённой пойме — рутина. Из-за
этого карта смещена: в спокойных областях линий недобор, в пёстрых перебор
случайных.

ЧТО ДЕЛАЕТ ГЕЙТ. Слой посылок ANFIS делит площадку на ``N_RULES`` нечётких
доменов по спектру: центры ставит k-means, принадлежность гауссова, веса
нормированы (softmax) — ячейка принадлежит нескольким доменам сразу. Сила
границы пересчитывается в сигмы СВОЕГО домена: из неё вычитается взвешенное
среднее по домену и делится на взвешенный разброс, а вклады доменов
складываются с теми же весами.

БЕЗ ТАРГЕТА, И ЭТО ПРИНЦИПИАЛЬНО. Берётся только слой посылок, обучаемые
заключения Сугено не используются. Обученный на линеаментах ANFIS в роли гейта
дал бы утечку: подтверждённые линии — это то, на чём мы потом проверяем
результат. Здесь разметка не участвует нигде.

ЧЕМ ПРОВЕРЯЕТСЯ. Тем же критерием, что и вся линеаментная ветка: сколько линий
подтверждено независимо оптикой и радаром вдоль длины и по направлению, против
числа линий, прошедших бы фильтр случайно. Планка — тот же прогон без гейта:
153 линии, 0.9 случайных, чистота x170.

Запуск из корня репозитория:
    python -m experiments.openlin_gate --site anabar --density "предельно"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import N_RULES, SEED
from experiments.openlin_domains import N_PC, edges_from_components
from experiments.openlin_lines import DENSITY_DEFAULT, _branches, trace_lines
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, load_site
from experiments.openlin_unsup import OPTIC, RADAR, confirmed_branches

M = Path("outputs/metrics")

#: Мягкость принадлежности домену. Ширина гауссианы задаётся как доля среднего
#: расстояния между центрами k-means: 0.7 — принадлежность заметно размыта
#: (ячейка обычно относится к двум-трём доменам), но не вырождается в равномерную.
GATE_SPREAD = 0.7

#: Минимальный вес домена, при котором его статистика вообще считается. Домен,
#: набравший меньше ячейки эффективного веса, дал бы разброс из шума.
MIN_WEIGHT = 100.0


def components(site: str, chans: tuple[str, ...]):
    """Сжатое представление площадки: то же, что у оператора без гейта."""
    from sklearn.decomposition import PCA

    z = load_site(site)
    use = tuple(c for c in chans if c in z)
    x, _y, valid, _u = build_stack(z, use)
    res_m = float(z["res_m"])
    nx, _n = neigh_stack(x, use, valid, res_m, base=use)
    stack = np.concatenate([x, nx])

    rows = stack[:, valid].T.astype("float32")
    sel = np.random.default_rng(SEED).choice(
        len(rows), size=min(50_000, len(rows)), replace=False)
    pca = PCA(n_components=min(N_PC, stack.shape[0]), whiten=True,
              random_state=SEED).fit(rows[sel])
    comp = np.zeros((pca.n_components_,) + valid.shape, dtype="float32")
    comp[:, valid] = pca.transform(rows).T
    return z, valid, res_m, comp


def rule_weights(comp: np.ndarray, valid: np.ndarray,
                 n_rules: int = N_RULES,
                 spread: float = GATE_SPREAD) -> np.ndarray:
    """Слой посылок ANFIS: нормированные силы правил, (n_rules, H, W).

    Центры — k-means по представлению, принадлежность гауссова, нормировка
    softmax по правилам. Обучаемых параметров нет: это чистое разбиение
    признакового пространства, а не предсказание.
    """
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans

    rows = comp[:, valid].T.astype("float32")
    sel = np.random.default_rng(SEED).choice(
        len(rows), size=min(50_000, len(rows)), replace=False)
    km = KMeans(n_clusters=n_rules, n_init=4, random_state=SEED).fit(rows[sel])
    c = km.cluster_centers_.astype("float32")

    # Ширина — доля среднего расстояния между центрами: масштаб задаётся самими
    # данными, а не подобранным числом.
    dd = cdist(c, c)
    spread = spread * float(dd[dd > 0].mean())

    logmu = -0.5 * (cdist(rows, c) / spread) ** 2
    logmu -= logmu.max(axis=1, keepdims=True)
    mu = np.exp(logmu)
    mu /= mu.sum(axis=1, keepdims=True)

    out = np.zeros((n_rules,) + valid.shape, dtype="float32")
    out[:, valid] = mu.T
    return out


def gated(strength: np.ndarray, w: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Сила границы в сигмах своего домена, мягко по весам правил."""
    s = np.where(valid, strength, 0.0).astype("float64")
    out = np.zeros(valid.shape, dtype="float64")
    used = 0
    for k in range(w.shape[0]):
        wk = np.where(valid, w[k], 0.0).astype("float64")
        tot = wk.sum()
        if tot < MIN_WEIGHT:
            continue
        mu = (wk * s).sum() / tot
        var = (wk * (s - mu) ** 2).sum() / tot
        sd = np.sqrt(max(var, 1e-12))
        out += wk * (s - mu) / sd
        used += 1
    return np.where(valid, out, np.nan).astype("float32"), used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    ap.add_argument("--rules", type=int, default=N_RULES)
    ap.add_argument("--spread", type=float, default=GATE_SPREAD,
                    help="ширина принадлежности в долях расстояния между центрами")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"площадка {a.site}, плотность '{a.density}', доменов {a.rules}, "
        f"размытость {a.spread}; гейт без таргета")

    lines_g, lines_p = {}, {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        z, valid, res_m, comp = components(a.site, chans)
        strength, _ang = edges_from_components(comp, valid, res_m)
        strength = np.nan_to_num(strength, nan=0.0)
        w = rule_weights(comp, valid, a.rules, a.spread)
        g, used = gated(strength, w, valid)
        g = np.nan_to_num(g, nan=0.0)

        lines_p[name], bp = trace_lines(strength, valid, valid, res_m, a.density)
        lines_g[name], bg = trace_lines(g, valid, valid, res_m, a.density)
        log(f"  {name:<13} доменов с весом {used}/{a.rules}, медианный "
            f"макс. вес {np.median(w[:, valid].max(axis=0)):.2f}; "
            f"веток без гейта {len(bp)}, с гейтом {len(bg)}")

    rows_out = []
    for tag, ln in (("без гейта", lines_p), ("с гейтом", lines_g)):
        l_all, lo, lr = ln["оптика+радар"], ln["оптика"], ln["радар"]
        kr, _nr, n_all = confirmed_branches(l_all, lr, res_m)
        ko, _no, _ = confirmed_branches(l_all, lo, res_m)
        keep = kr & ko
        n_keep = sum(1 for p in _branches(keep, res_m) if len(p) >= 3)

        rng = np.random.default_rng(1)
        h, w_ = l_all.shape
        fake = []
        for _ in range(25):
            s1 = (int(rng.integers(h)), int(rng.integers(w_)))
            s2 = (int(rng.integers(h)), int(rng.integers(w_)))
            k1, _p, _n = confirmed_branches(l_all, np.roll(lr, s1, (0, 1)) & valid,
                                            res_m)
            k2, _p, _n = confirmed_branches(l_all, np.roll(lo, s2, (0, 1)) & valid,
                                            res_m)
            fake.append(sum(1 for p in _branches(k1 & k2, res_m) if len(p) >= 3))
        rows_out.append({"вариант": tag, "прослежено": n_all,
                         "подтверждено": n_keep,
                         "длина_км": keep.sum() * res_m / 1000,
                         "случайно": float(np.mean(fake)),
                         "чистота": n_keep / max(np.mean(fake), 1e-9)})
        np.savez_compressed(
            M / f"openlin_gate_{a.site}_{a.density}_{a.spread}"
            f"_{tag.split()[0]}.npz",
            confirmed=keep, lines=l_all)

    df = pd.DataFrame(rows_out)
    log("\n=== гейт против его отсутствия " + "=" * 34)
    log(df.round(2).to_string(index=False))
    df.to_csv(M / f"openlin_gate_{a.site}_{a.density}_{a.spread}.csv",
              index=False,
              encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_gate_{a.site}_{a.density}.csv'}")


if __name__ == "__main__":
    main()
