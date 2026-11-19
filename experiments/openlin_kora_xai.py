"""Пакет 10 ТЗ К1: групповая пермутационная важность и знаки правил устойчивой «Коры».

Объявление — ``audit/PACKET10_DECLARATION.md`` (26.10.2026, до расчёта).
Модель — openlin_kora_stable без изменений (пересобирается: правила не
сохранялись); проверка — AUC обоих прогонов совпадает с логом. Объясняется
различие «реальный стек против суррогата», не геология.

Запуск из корня:
    python -m experiments.openlin_kora_xai --site anabar
    python -m experiments.openlin_kora_xai --сравнить anabar lapland finland65
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path("outputs/k1/xai")
FIG = Path("outputs/k1/figures")
N_REP = 5
BLOCK = 200                      # участок 20 км на сетке «Коры» 100 м
BLOCK_ROWS, BLOCK_MIN = 2000, 500
GROUPS = ("оптика", "радар", "когерентность", "согласие приборов", "согласие масштабов")


def group_of(name: str) -> str:
    if name.startswith("согл"):
        return "согласие масштабов" if re.search(r"\d~\d", name) else "согласие приборов"
    if name.startswith(("gray", "ndvi")):
        return "оптика"
    if name.startswith(("vh", "vv")):
        return "радар"
    if name.startswith("coh"):
        return "когерентность"
    return "прочее"


def fit(site, log):
    """Повтор openlin_kora_stable.main: правила, границы уровней, проверочные строки."""
    from scipy.stats import spearmanr  # noqa: F401
    from experiments import openlin_kora_stable as KS
    from experiments.openlin_forest_kora import neigh_all
    from experiments.openlin_invariant import builder_invariant
    from experiments.openlin_kora import (N_LEVELS, N_TEST, N_TRAIN, binarize, kora, kora_score,
                                          sample_rows, surrogate)
    from experiments.openlin_line_rank import auc
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    build = builder_invariant(site)
    tr0, te0 = KS.block_split_vertical(valid.shape, res_m)
    tr0 &= valid
    te0 &= valid

    def feat(src, support):
        return build(*neigh_all(src, use, valid, res_m, support))

    real = {"A": feat(x, tr0), "B": feat(x, te0)}
    names = real["A"][1]
    runs = {}
    for run, swap, off in (("прогон 1", False, 0), ("прогон 2", True, 100)):
        train, test = (te0, tr0) if swap else (tr0, te0)
        f_train = real["B" if swap else "A"][0]
        f_test = real["A" if swap else "B"][0]
        found, samples = [], []
        for k in range(KS.K):
            seed = KS.SEED_BASE + off + k
            fk = feat(surrogate(x, valid, np.random.default_rng(seed)), train)[0]
            rng = np.random.default_rng(seed + 50)
            samples.append((sample_rows(f_train, train, N_TRAIN, rng), sample_rows(fk, train, N_TRAIN, rng)))
        pooled = np.vstack([np.vstack(s) for s in samples])
        edges = np.stack([np.quantile(pooled[:, j], np.linspace(0, 1, N_LEVELS + 1)[1:-1])
                          for j in range(pooled.shape[1])])
        for r_tr, f_tr in samples:
            rules = kora(binarize(r_tr, edges), binarize(f_tr, edges), len(names), lambda *_: None)
            found.append({(q["класс"], q["условия"]) for q in rules})
        cnt = Counter(r for s in found for r in s)
        stable = [{"класс": c, "условия": u} for (c, u), n in cnt.items() if n >= KS.KEEP]
        fk_te = feat(surrogate(x, valid, np.random.default_rng(KS.SEED_BASE + off + 99)), test)[0]
        rng = np.random.default_rng(KS.SEED_BASE + off + 98)
        r_te = sample_rows(f_test, test, N_TEST, rng)
        f_te = sample_rows(fk_te, test, N_TEST, rng)
        y = np.r_[np.ones(len(r_te)), np.zeros(len(f_te))]
        a0 = auc(y, kora_score(binarize(np.vstack([r_te, f_te]), edges), stable))
        log(f"  {site}, {run}: устойчивых правил {len(stable)}, AUC на проверке {a0:.3f} (сверка с логом)")
        runs[run] = dict(stable=stable, edges=edges, test=test, f_test=f_test, fk_te=fk_te,
                         r_te=r_te, f_te=f_te, auc=a0)
    return names, runs, N_LEVELS


def importance(names, run, n_levels, log):
    from experiments.openlin_kora import binarize, kora_score, sample_rows
    from experiments.openlin_line_rank import auc
    edges, stable = run["edges"], run["stable"]

    def score(R, F):
        y = np.r_[np.ones(len(R)), np.zeros(len(F))]
        return auc(y, kora_score(binarize(np.vstack([R, F]), edges), stable))

    def drop(R, F, cols, seed):
        X = np.vstack([R, F]).copy()
        p = np.random.default_rng(seed).permutation(len(X))
        X[:, cols] = X[p][:, cols]
        y = np.r_[np.ones(len(R)), np.zeros(len(F))]
        return auc(y, kora_score(binarize(X, edges), stable))

    R, F = run["r_te"], run["f_te"]
    base = score(R, F)
    groups = {g: [j for j, n in enumerate(names) if group_of(n) == g] for g in GROUPS}
    gimp = {g: float(base - np.mean([drop(R, F, c, s) for s in range(N_REP)])) for g, c in groups.items() if c}
    fimp = {n: float(base - np.mean([drop(R, F, [j], s) for s in range(N_REP)])) for j, n in enumerate(names)}
    # по участкам 20 км в проверочной полосе
    test = run["test"]
    H, W = test.shape
    blocks = []
    for i in range(0, H, BLOCK):
        for j in range(0, W, BLOCK):
            m = np.zeros_like(test)
            m[i:i + BLOCK, j:j + BLOCK] = True
            m &= test
            if m.sum() < BLOCK_MIN:
                continue
            rng = np.random.default_rng(i * 1000 + j)
            Rb = sample_rows(run["f_test"], m, BLOCK_ROWS, rng)
            Fb = sample_rows(run["fk_te"], m, BLOCK_ROWS, rng)
            b0 = score(Rb, Fb)
            blocks.append({"участок": f"{i // BLOCK}_{j // BLOCK}", "AUC": float(b0),
                           **{g: float(b0 - drop(Rb, Fb, c, 0)) for g, c in groups.items() if c}})
    # знак: высокий минус низкий уровень в правилах «снимок» минус то же у «подделки»
    sign = np.zeros(len(names))
    for r in stable:
        for p in r["условия"]:
            j, lev = divmod(int(p), n_levels)
            d = (lev == n_levels - 1) - (lev == 0)
            sign[j] += d if r["класс"] == 1 else -d
    return {"AUC": base, "группы": gimp, "признаки": fimp, "участки": blocks,
            "знак": dict(zip(names, sign.tolist())),
            "группа_признака": {n: group_of(n) for n in names}}


def run_site(site, log):
    t0 = time.time()
    names, runs, nl = fit(site, log)
    out = {}
    for run, r in runs.items():
        imp = importance(names, r, nl, log)
        out[run] = imp
        log(f"  {run}: групповая важность (падение AUC): " + ", ".join(
            f"{g} {v:+.3f}" for g, v in sorted(imp["группы"].items(), key=lambda kv: -kv[1])))
        if imp["участки"]:
            for g in GROUPS:
                vals = [b[g] for b in imp["участки"] if g in b]
                if vals:
                    log(f"    {g}: по {len(vals)} участкам медиана {np.median(vals):+.3f}, "
                        f"размах {min(vals):+.3f}..{max(vals):+.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{site}_xai.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"сохранено: {OUT / f'{site}_xai.json'} ({time.time() - t0:.0f} с)")


def compare(sites, log):
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = {s: json.loads((OUT / f"{s}_xai.json").read_text(encoding="utf-8")) for s in sites}
    res = {}
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            for run in ("прогон 1", "прогон 2"):
                A, B = d[a][run], d[b][run]
                gs = [g for g in GROUPS if g in A["группы"] and g in B["группы"]]
                fs = [f for f in A["признаки"] if f in B["признаки"]]
                rg = spearmanr([A["группы"][g] for g in gs], [B["группы"][g] for g in gs]).statistic
                rf = spearmanr([A["признаки"][f] for f in fs], [B["признаки"][f] for f in fs]).statistic
                sa, sb = np.sign([A["знак"][f] for f in fs]), np.sign([B["знак"][f] for f in fs])
                both = (sa != 0) & (sb != 0)
                same = float((sa[both] == sb[both]).mean()) if both.any() else np.nan
                flips = [f for f, x, y in zip(fs, sa, sb) if x * y < 0]
                res[f"{a} ~ {b}, {run}"] = {"Спирмен_групп": float(rg), "Спирмен_признаков": float(rf),
                                           "знак_совпадает": same, "переворот_знака": flips}
                log(f"  {a} ~ {b}, {run}: Спирмен групп {rg:+.2f}, признаков {rf:+.2f}; знак совпадает "
                    f"у {100 * same:.0f} % признаков; перевёрнут: {', '.join(flips) if flips else 'нет'}")
    (OUT / "rank_transfer.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    w = 0.8 / len(sites)
    for k, s in enumerate(sites):
        vals = [np.mean([d[s][r]["группы"].get(g, np.nan) for r in ("прогон 1", "прогон 2")]) for g in GROUPS]
        ax.bar(np.arange(len(GROUPS)) + k * w, vals, w, label=s)
    ax.set_xticks(np.arange(len(GROUPS)) + w * (len(sites) - 1) / 2)
    ax.set_xticklabels(GROUPS, fontsize=9)
    ax.set_ylabel("падение AUC при перемешивании группы")
    ax.set_title("«Кора»: что отличает реальный стек от суррогата (среднее двух прогонов)", loc="left")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "xai_groups.png", dpi=150)
    plt.close(fig)
    log(f"сохранено: {FIG / 'xai_groups.png'}, {OUT / 'rank_transfer.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--сравнить", dest="cmp", nargs="+", default=None)
    a = ap.parse_args()
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    if a.site:
        run_site(a.site, log)
    if a.cmp:
        compare(a.cmp, log)


if __name__ == "__main__":
    main()
