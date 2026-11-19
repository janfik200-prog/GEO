"""ANFU: улучшение Анфисы по шагам (audit/ANFU_DECLARATION.md, 11.11.2026).

Замороженный ``openlin_fam.py`` (FAM2/FAM3/FAMLAP) не меняется: здесь —
расширения поверх его функций. Мера шага — F1 при равном бюджете на 16 окнах
Утсйоки (окна FAM1), среднее по 3 зёрнам; правило «оставить, если > +0,003».

Конфигурация шага — набор ключей:
  big    — шаг 1: вогнутость/выпуклость σ 480/960 м, прямолинейность на 3,3 и 9,9 км;
  pairs  — шаг 3: парные правила «И» для 4 признаков с наибольшим размахом вкладов;
  tol    — шаг 4: фон обучения без ячеек ближе 300 м к эталону;
  radar  — шаг 2: признаки радара S1 VV;
  sheets8 — шаг 5: листы обучения fam_r1…r8.

Запуск: python -m experiments.openlin_anfu --cfg big,pairs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from experiments import openlin_fam as fam
from experiments.openlin_fam import (CELL_M, COMP, DEM, FAM3_TRAIN, LR, EPOCHS, N_DIR, N_TERMS,
                                     NEG_PER_POS, OUT, POS_CAP, RES_IN, SEEDS, TOL_M, WIN_SEED,
                                     all_windows, block, budget_f1, gtk_label, hessian_ridge,
                                     log_to, quantile_norm, win_mask, windows)

BIG_SIG_M = (480, 960)            # продолжение ряда 60/120/240 удвоением
BIG_LEN_M = (3300, 9900)          # 1,1 км × 3 и × 9; ≤ 10 км (10-й проц. длины GTK)
N_PAIR_FEAT = 4                   # 4 признака → 6 пар
KEEP_DELTA = 0.003                # наибольший разброс зёрен нечёткой модели (fam2/fam3.csv)
BASE_F1 = 0.2765                  # FAM3, Утсйоки 16 окон
ANFU_N_TESTS = 0
#: ИТОГИ ШАГОВ (Утсйоки 16 окон, база 0,27648; audit/ANFU_RESULTS.md, аудит 11.11.2026):
#: шаг 1 big — F1 0,1885, ΔF1 −0,0880, порог 0,0758 (плацебо −0,0062) — ОТКАТ: модель
#:   опирается на крупную выпуклость (размах 4,32), выбор сгущается, полнота 0,275 → 0,145;
#: шаг 3 pairs [6,3,8,4] — F1 0,2720, ΔF1 −0,0045, порог 0,0300 — ОТКАТ;
#: шаг 4 tol — F1 0,2782, ΔF1 +0,0017, порог 0,0055 — ОТКАТ.
#: шаг 2 radar — F1 0,1961, ΔF1 −0,0804, порог 0,1001 — ОТКАТ: радар отличает метку на листах
#:   обучения (прямолинейность +0,3…+0,4 z), на Утсйоки нет (+0,09).
#: шаг 6 snap (openlin_anfu_snap.py) — S 0,1328 → 0,1193, ранг 6 из 20 плацебо — ОТКАТ.
#: Итог ANFU: ни один шаг не принят; модель FAM3, линии LINB2.
#: Бустинг/логистическая для сведения (поправка 3 п. 7): база 0,2930/0,2848, big 0,2132/0,1889, tol 0,2864/0,2854.
SHEETS8 = tuple(f"fam_r{i}" for i in range(1, 9))
PLACEBO_SHIFT = 70                # 7 км на сетке 100 м
BOOT_N, BOOT_SEED = 1000, 2511


def window_tallies(score, m16, ref, wins, n):
    """Выбор K лучших ячеек по 16 окнам (как budget_f1) и счёт по каждому окну."""
    K = int((ref & m16).sum())
    sel = fam.select_top(score, m16, K)
    rad = TOL_M / CELL_M
    d_ref = ndi.distance_transform_edt(~ref)
    d_sel = ndi.distance_transform_edt(~sel)
    out = []
    for r, c in wins:
        sl = (slice(r, r + n), slice(c, c + n))
        s_, rf = sel[sl], ref[sl] & m16[sl]
        out.append((float((d_ref[sl][s_] <= rad).sum()), float(s_.sum()),
                    float((d_sel[sl][rf] <= rad).sum()), float(rf.sum())))
    return np.array(out)                                # (окна, 4)


def f1_from(t):
    P = t[..., 0] / np.maximum(t[..., 1], 1)
    R = t[..., 2] / np.maximum(t[..., 3], 1)
    return np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)


def boot_delta(A, B):
    """Парный бутстрэп окон: A, B — (зёрна, окна, 4); ΔF1 = A − B, среднее по зёрнам."""
    rng = np.random.default_rng(BOOT_SEED)
    nw = A.shape[1]
    d = []
    for _ in range(BOOT_N):
        i = rng.integers(0, nw, nw)
        d.append(f1_from(A[:, i].sum(1)).mean() - f1_from(B[:, i].sum(1)).mean())
    return np.array(d)


def line_mean_stack(r, len_m):
    """Среднее r вдоль отрезка len_m по N_DIR направлениям (те же ядра, что у FAM), FFT."""
    from scipy.signal import fftconvolve
    L = int(round(len_m / CELL_M))
    L += (L + 1) % 2
    half = L // 2
    pad = np.pad(r, half, mode="edge")
    out = []
    for i in range(N_DIR):
        th = np.pi * i / N_DIR
        k = np.zeros((L, L))
        for t in np.linspace(-half, half, 4 * L):
            k[int(round(half - t * np.cos(th))), int(round(half + t * np.sin(th)))] = 1
        out.append(fftconvolve(pad, k / k.sum(), mode="valid"))
    return np.stack(out)


def straight_len(r, len_m):
    s = line_mean_stack(r, len_m)
    return s.max(0) - s.mean(0)


def big_features(site, F, log):
    """Шаг 1: +6 признаков на сетке 100 м (кэш)."""
    c = OUT / f"anfu_big_{site}.npz"
    if c.exists():
        return np.load(c)["G"]
    dem = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy").astype("float64")
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    k = int(CELL_M / RES_IN)
    d100 = block(dem, k, "mean")[:F.shape[0], :F.shape[1]]
    pos, neg = None, None
    for s in BIG_SIG_M:                                # на сетке 100 м: σ 4,8 и 9,6 ячейки
        lmax, lmin = hessian_ridge(d100, s / CELL_M)
        a, b = np.maximum(lmax, 0), np.maximum(-lmin, 0)
        pos = a if pos is None else np.maximum(pos, a)
        neg = b if neg is None else np.maximum(neg, b)
    G = [pos, neg]
    for L in BIG_LEN_M:
        G.append(straight_len(F[..., 5], L))           # вогнутость
        G.append(straight_len(F[..., 0], L))           # тёмная полоса
    G = np.stack(G, -1).astype("float32")
    np.savez_compressed(c, G=G)
    log(f"  {site}: признаки шага 1 посчитаны")
    return G


def radar_features(site, F, log):
    """Шаг 2: тёмная/светлая полоса радара VV (σ 40/80/160 м) и её прямолинейность 1,1 км."""
    c = OUT / f"anfu_radar_{site}.npz"
    if c.exists():
        return np.load(c)["G"]
    z = np.load(Path("datacache/s1_look") / f"s1_vv_{site}_лист_20m_половины.npz")
    vv = np.nanmean(np.stack([z["А"], z["Б"]]), 0).astype("float64")
    vv = np.where(np.isfinite(vv), vv, np.nanmedian(vv))
    dark, bright = fam.multi_ridge(vv, fam.IMG_SIG_M)
    k = int(CELL_M / RES_IN)
    d, b = block(dark, k, "max"), block(bright, k, "max")
    G = np.stack([d, b, fam.straightness(d)], -1)[:F.shape[0], :F.shape[1]].astype("float32")
    np.savez_compressed(c, G=G)
    log(f"  {site}: признаки радара посчитаны")
    return G


def site_table(site, cfg, log):
    F, valid, meta = fam.features(site, log)
    parts = [F]
    if "big" in cfg:
        parts.append(big_features(site, F, log))
    if "bigplacebo" in cfg:                            # поправка 1.2: сдвиг по циклу на 7 км
        parts.append(np.roll(big_features(site, F, log), PLACEBO_SHIFT, axis=0))
    if "radar" in cfg:
        parts.append(radar_features(site, F, log))
    F = np.concatenate(parts, -1)
    valid = valid & np.all(np.isfinite(F), -1)
    ref, lab = gtk_label(meta, valid.shape, site)
    wins, n = all_windows(valid)
    m = win_mask(valid.shape, wins, n) & valid
    Z = quantile_norm(F, m)
    return dict(F=F, Z=Z, ref=ref, lab=lab, m=m, valid=valid)


def build_train(train, cfg, log):
    """Как openlin_fam.build_train; при tol фон без ячеек ближе TOL_M к эталону."""
    rng = np.random.default_rng(WIN_SEED)
    Xs, ys = [], []
    for s in train:
        t = site_table(s, cfg, log)
        Z, lab, m = t["Z"], t["lab"], t["m"]
        bg = ~lab & m
        if "tol" in cfg:
            near = ndi.distance_transform_edt(~t["ref"]) * CELL_M <= TOL_M
            bg &= ~near
        pos, neg = np.flatnonzero((lab & m).ravel()), np.flatnonzero(bg.ravel())
        pos = rng.choice(pos, min(POS_CAP, len(pos)), replace=False)
        neg = rng.choice(neg, min(NEG_PER_POS * len(pos), len(neg)), replace=False)
        idx = np.concatenate([pos, neg])
        Xs.append(Z.reshape(-1, Z.shape[-1])[idx]); ys.append(lab.ravel()[idx].astype(int))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    log(f"обучение: строк {len(y)}, эталон {int(y.sum())}, признаков {X.shape[1]}")
    return X, y


def fit_fam_pairs(X, y, seed, pairs=()):
    """Нечёткая аддитивная модель (как openlin_fam.fit_fam) + парные правила «И»."""
    import torch
    torch.manual_seed(seed)
    d = X.shape[1]
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    c = torch.nn.Parameter(torch.tensor(np.tile([-1.0, 0.0, 1.0], (d, 1)), dtype=torch.float32))
    ls = torch.nn.Parameter(torch.zeros(d, N_TERMS))
    b = torch.nn.Parameter(torch.zeros(d, N_TERMS))
    b0 = torch.nn.Parameter(torch.zeros(()))
    pw_ = torch.nn.Parameter(torch.zeros(max(len(pairs), 1), N_TERMS, N_TERMS))
    pi = torch.tensor([p[0] for p in pairs] or [0]); pj = torch.tensor([p[1] for p in pairs] or [0])
    opt = torch.optim.Adam([c, ls, b, b0] + ([pw_] if pairs else []), lr=LR)
    pw = torch.tensor(float((y == 0).sum() / max((y == 1).sum(), 1)))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    g = torch.Generator().manual_seed(seed)

    def forward(x):
        mu = torch.exp(-0.5 * ((x[:, :, None] - c[None]) / torch.exp(ls)[None]) ** 2)
        mu = mu / (mu.sum(-1, keepdim=True) + 1e-9)
        out = (mu * b[None]).sum((1, 2)) + b0
        if pairs:
            mi, mj = mu[:, pi], mu[:, pj]                       # (n, P, 3)
            out = out + torch.einsum("npa,npb,pab->n", mi, mj, pw_)
        return out
    n = len(Xt)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            lossf(forward(Xt[idx]), yt[idx]).backward()
            opt.step()
    params = {"c": c.detach().numpy(), "s": np.exp(ls.detach().numpy()),
              "b": b.detach().numpy(), "b0": float(b0.detach()),
              "pairs": list(pairs), "pw": pw_.detach().numpy() if pairs else None}

    def predict(Xn):
        with torch.no_grad():
            return forward(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return predict, params


def term_range(prm):
    order = np.argsort(prm["c"], 1)
    bb = np.take_along_axis(prm["b"], order, 1)
    return bb.max(1) - bb.min(1)


def run(cfg, log, pair_src=None):
    train = SHEETS8 if "sheets8" in cfg else FAM3_TRAIN
    X, y = build_train(train, cfg, log)
    u = site_table("utsjoki", cfg, log)
    w_, e_, n_ = windows(u["valid"])
    m16 = win_mask(u["valid"].shape, w_ + e_, n_) & u["valid"]
    pairs = ()
    if "pairs" in cfg:
        rng_ = term_range(pair_src)
        top = [int(i) for i in np.argsort(-rng_)[:N_PAIR_FEAT]]
        pairs = tuple((top[a], top[b]) for a in range(len(top)) for b in range(a + 1, len(top)))
        log(f"пары по размаху вкладов (зерно 1 принятой модели): признаки {top}")
    out = {}
    prm1 = None
    for name, m in (("16 окон", m16), ("52 окна", u["m"])):
        out[name] = []
    preds = []
    for seed in SEEDS:
        pr, prm = fit_fam_pairs(X, y, seed, pairs)
        preds.append(pr)
        prm1 = prm1 or prm
    wins16 = w_ + e_
    tall = []
    for pr in preds:
        sc = np.full(m16.shape, -np.inf)
        sc[m16] = pr(u["Z"][m16])
        tall.append(window_tallies(sc, m16, u["ref"], wins16, n_))
    out["tallies"] = np.stack(tall)
    for name, m in (("16 окон", m16), ("52 окна", u["m"])):
        K = int((u["ref"] & m).sum())
        v = []
        for pr in preds:
            sc = np.full(m.shape, -np.inf)
            sc[m] = pr(u["Z"][m])
            v.append(budget_f1(sc, m, u["ref"], K)[2])
        out[name] = (float(np.mean(v)), float(np.std(v)))
    return out, prm1, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="")
    ap.add_argument("--accepted", default="", help="принятая до шага конфигурация")
    ap.add_argument("--ref-f1", type=float, default=BASE_F1)
    ap.add_argument("--placebo-delta", type=float, default=None)
    a = ap.parse_args()
    cfg = tuple(x for x in a.cfg.split(",") if x)
    acc = tuple(x for x in a.accepted.split(",") if x)
    tag = "+".join(cfg) or "база"
    log = log_to(OUT / f"_anfu_{tag}.log")
    log(f"ANFU; объявление audit/ANFU_DECLARATION.md; конфигурация {cfg}; принятая {acc}; "
        f"F1 принятой {a.ref_f1:.4f}; правило > +{KEEP_DELTA}")
    pair_src = None
    if "pairs" in cfg:
        p = OUT / f"anfu_params_{'+'.join(acc) or 'база'}.json"
        pair_src = {k: np.array(v) for k, v in json.load(open(p))["seed1"].items()
                    if k in ("c", "b")}
    res, prm1, _ = run(cfg, log, pair_src)
    f16, s16 = res["16 окон"]
    log(f"Утсйоки 16 окон: F1 {f16:.4f} ± {s16:.4f}; 52 окна: {res['52 окна'][0]:.4f}")
    np.save(OUT / f"anfu_tallies_{tag}.npy", res["tallies"])
    d = f16 - a.ref_f1
    acc_t = OUT / f"anfu_tallies_{'+'.join(acc) or 'база'}.npy"
    thr = KEEP_DELTA
    if acc_t.exists() and cfg != acc:
        bd = boot_delta(res["tallies"], np.load(acc_t))
        log(f"бутстрэп окон ({BOOT_N}, зерно {BOOT_SEED}): ΔF1 среднее {bd.mean():+.4f}, SD {bd.std():.4f}, "
            f"доля Δ ≤ 0: {(bd <= 0).mean():.3f}")
        thr = max(thr, 2 * bd.std())
    if a.placebo_delta is not None:
        thr = max(thr, a.placebo_delta)
        log(f"ΔF1 плацебо {a.placebo_delta:+.4f} учтён в пороге")
    log(f"ШАГ: ΔF1 = {d:+.4f}; порог {thr:.4f} -> " + ("ОСТАВИТЬ" if d > thr else "ОТКАТИТЬ"))
    json.dump({"cfg": cfg, "f16": f16, "s16": s16, "f52": res["52 окна"][0],
               "seed1": {k: (np.asarray(v).tolist() if v is not None and not isinstance(v, (int, float))
                             else v) for k, v in prm1.items()}},
              open(OUT / f"anfu_params_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False)


def run_baselines(cfg, log):
    """Поправка 3 п. 7: бустинг и логистическая для сведения (решений не меняет)."""
    train = SHEETS8 if "sheets8" in cfg else FAM3_TRAIN
    X, y = build_train(train, cfg, log)
    u = site_table("utsjoki", cfg, log)
    w_, e_, n_ = windows(u["valid"])
    m16 = win_mask(u["valid"].shape, w_ + e_, n_) & u["valid"]
    keep = ("градиентный бустинг", "логистическая регрессия")
    res = {k: {"16 окон": [], "52 окна": []} for k in keep}
    for seed in SEEDS:
        bl = fam.baselines(X, y, seed)
        for k in keep:
            for name, m in (("16 окон", m16), ("52 окна", u["m"])):
                K = int((u["ref"] & m).sum())
                sc = np.full(m.shape, -np.inf)
                sc[m] = bl[k](u["Z"][m])
                res[k][name].append(budget_f1(sc, m, u["ref"], K)[2])
    for k in keep:
        log(f"  для сведения {'+'.join(cfg) or 'база':12s} {k:24s} 16 окон "
            f"{np.mean(res[k]['16 окон']):.4f} ± {np.std(res[k]['16 окон']):.4f}; "
            f"52 окна {np.mean(res[k]['52 окна']):.4f}")
    return res


def main_baselines(cfgs):
    log = log_to(OUT / "_anfu_baselines.log")
    log("ANFU поправка 3 п. 7: бустинг/логистическая для сведения, Утсйоки")
    for c in cfgs:
        run_baselines(tuple(x for x in c.split(",") if x), log)


if __name__ == "__main__":
    import sys
    if "--baselines" in sys.argv:
        main_baselines(sys.argv[sys.argv.index("--baselines") + 1:] or [""])
    else:
        main()
