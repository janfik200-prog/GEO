"""MEAS1: замер шума главной меры S (``openlin_ref_match.curve_score``) — audit/MEAS1_DECLARATION.md.

Утсйоки (настройка), описательно, ``MEAS1_N_TESTS`` = 0. Функции меры не меняются:
они вызываются на возмущённых наборах (П1, П2) и на половинах эталона (П3).

Запуск: python -m experiments.openlin_meas1
"""
from __future__ import annotations

import numpy as np

from experiments import openlin_ref_match as rm
from experiments.openlin_fam import OUT, log_to

SITE = "utsjoki"
SETS = {"снимок UTS1": "_настройка1", "рельеф RELU1": "_рельеф1", "Анфиса LINB2": "_анфиса2"}
PAIRS = (("Анфиса LINB2", "рельеф RELU1"), ("рельеф RELU1", "снимок UTS1"),
         ("Анфиса LINB2", "снимок UTS1"))
K_FIX = 103                         # K семейства на невозмущённых входах (UTS1 1039 км)
JIT_M = (20.0, 40.0, 60.0)          # 1, 2 пикселя (сведения) и 60 м — медиана |сдвига| шага 6 ANFU (порог)
Z95 = 1.96                          # двусторонние 95 % (поправка 1 п. 1)
BASE_S = {"снимок UTS1": 0.071, "рельеф RELU1": 0.110, "Анфиса LINB2": 0.133}  # LINB2_RESULTS
N_SEEDS = 20                        # ошибка SD ≈ 1/√(2·19) = 0,16
MEAS1_N_TESTS = 0
DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load_ids(tag, shape):
    from pathlib import Path
    p = Path("outputs/metrics") / f"openlin_zone_lines_{SITE}_v3{tag}"
    if tag == "_анфиса2":
        return np.load(f"{p}.npz")["ids"].astype(np.int32)
    L = np.load(f"{p}_линии.npz")
    ids = np.zeros(shape, np.int32)
    pts, lid = L["C_all_pts"], L["C_all_id"]
    ids[pts[:, 0], pts[:, 1]] = lid + 1
    return ids


def jitter(ids, d_px, rng, valid):
    """Каждая линия — параллельный перенос на d_px по нормали к её главной оси, знак на линию."""
    rr, cc = np.nonzero(ids)
    lab = ids[rr, cc]
    n = int(ids.max())
    cnt = np.bincount(lab, minlength=n + 1).astype(float)
    mr = np.bincount(lab, rr, n + 1) / np.maximum(cnt, 1)
    mc = np.bincount(lab, cc, n + 1) / np.maximum(cnt, 1)
    dr, dc = rr - mr[lab], cc - mc[lab]
    crr = np.bincount(lab, dr * dr, n + 1)
    ccc = np.bincount(lab, dc * dc, n + 1)
    crc = np.bincount(lab, dr * dc, n + 1)
    th = 0.5 * np.arctan2(2 * crc, crr - ccc)             # главная ось (ряд, столбец)
    nr, nc = -np.sin(th), np.cos(th)                       # нормаль к оси
    sg = rng.choice([-1.0, 1.0], n + 1)
    r2 = rr + np.round(sg[lab] * d_px * nr[lab]).astype(int)
    c2 = cc + np.round(sg[lab] * d_px * nc[lab]).astype(int)
    H, W = ids.shape
    ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
    out = np.zeros(ids.shape, bool)
    out[r2[ok], c2[ok]] = True
    return out & valid


def translate(mask, dr, dc, valid):
    out = np.zeros_like(mask)
    H, W = mask.shape
    src = mask[max(-dr, 0):H - max(dr, 0), max(-dc, 0):W - max(dc, 0)]
    out[max(dr, 0):H - max(-dr, 0), max(dc, 0):W - max(-dc, 0)] = src
    return out & valid


def score(mask, pos, res):
    lens, mins = rm.evaluate(mask, pos, res)
    return rm.curve_score(lens, mins, K_FIX)["S"], lens.sum()


def main():
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    log = log_to(OUT / "_meas1.log")
    log("MEAS1; объявление audit/MEAS1_DECLARATION.md; Утсйоки, описательно")
    w = dict(np.load(COMP / f"winter_{SITE}_20m_лист.npz"))
    res = float(w["res_m"])
    from pathlib import Path
    masks = {s: np.load(Path("outputs/metrics") / f"openlin_zone_lines_{SITE}_v3{t}.npz")["C_all"]
             for s, t in SETS.items()}
    shape20 = masks["снимок UTS1"].shape
    valid20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    masks = {s: m[:shape20[0], :shape20[1]] for s, m in masks.items()}   # поправка 3: без обрезки
    allv = np.ones(shape20, bool)                                         # как в опубликованных S
    ids = {s: load_ids(t, shape20)[:shape20[0], :shape20[1]]
           for s, t in SETS.items()}
    for s in SETS:
        same = bool(np.array_equal(ids[s] > 0, masks[s]))
        log(f"  {s}: линий {len(np.unique(ids[s])) - 1}, метки = маска: {same}")
    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(valid20, ang, np.nan)
    sh = rm.shifts(shape20)
    pos = rm.positions(ang, fid, valid20, sh)
    rows = []

    def rec(kind, rep, vals):
        for s, (S, km) in vals.items():
            rows.append(dict(возмущение=kind, повтор=rep, набор=s, S=S, км=km))

    base = {s: score(m, pos, res) for s, m in masks.items()}
    rec("исходное", 0, base)
    for s, (S, km) in base.items():
        log(f"  исходное {s:14s} {km:6.0f} км  S = {S:.4f}")
        if abs(S - BASE_S[s]) > 0.001:
            raise SystemExit(f"контроль входа: S {s} {S:.4f} != {BASE_S[s]} — остановка")
        sj = score(ids[s] > 0, pos, res)[0]
        if abs(sj - S) > 1e-9:
            raise SystemExit(f"контроль: S по меткам линий {sj:.4f} != S по маске {S:.4f}")
    log("П1: перенос на 1 пиксель, 8 направлений")
    for i, (dr, dc) in enumerate(DIRS8):
        rec("П1", i, {s: score(translate(m, dr, dc, allv), pos, res) for s, m in masks.items()})
    for d in JIT_M:
        log(f"П2: дрожание линий {d:.0f} м, {N_SEEDS} зёрен")
        for k in range(N_SEEDS):
            vals = {}
            for j, s in enumerate(SETS):
                rng = np.random.default_rng(1000 * (k + 1) + j)
                vals[s] = score(jitter(ids[s], d / res, rng, allv), pos, res)
            rec(f"П2 {d:.0f} м", k, vals)
    log(f"П3: половины эталона, {N_SEEDS} зёрен")
    lines = np.unique(fid[fid > 0])
    for k in range(N_SEEDS):
        rng = np.random.default_rng(5000 + k)
        half = rng.permutation(lines)[:len(lines) // 2]
        for h, name in ((0, "А"), (1, "Б")):
            pick = np.isin(lines, half) if h == 0 else ~np.isin(lines, half)
            keep_ids = lines[pick]
            ph = [(rc[np.isin(f, keep_ids)], a[np.isin(f, keep_ids)], f[np.isin(f, keep_ids)])
                  for rc, a, f in pos]
            rec("П3", f"{k}{name}", {s: score(m, ph, res) for s, m in masks.items()})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "meas1.csv", index=False, encoding="utf-8-sig")
    kinds = ["П1", "П2 20 м", "П2 40 м", "П2 60 м"]
    km_min = df[df.набор == "снимок UTS1"].км.min()
    log(f"наименьшая длина возмущённого UTS1 {km_min:.0f} км" +
        (" — КОРОЧЕ 1030 км (хвост кривой сплющен)" if km_min < 1030 else ""))

    def p3(a_, b_=None):
        sub = df[df.возмущение == "П3"].copy()
        sub["разб"], sub["пол"] = sub.повтор.str[:-1], sub.повтор.str[-1]
        t = sub.pivot_table(index="разб", columns=["пол", "набор"], values="S")
        v = (lambda h: t[(h, a_)] - (t[(h, b_)] if b_ else 0))
        return ((v("А") - v("Б")) / 2).std(ddof=1)
    log("=== шум S по наборам: среднее ± SD; П3 — SD по 20 разбиениям, (А − Б)/2")
    for s in SETS:
        line = " | ".join(f"{k} {df[(df.возмущение == k) & (df.набор == s)].S.mean():.4f} ± "
                          f"{df[(df.возмущение == k) & (df.набор == s)].S.std(ddof=1):.4f}" for k in kinds)
        log(f"  {s:14s} исходное {base[s][0]:.4f} | {line} | П3 SD {p3(s):.4f}")
    log("=== ΔS по парам: смещение (среднее − исходное) и SD при одном возмущении обоих")
    for a_, b_ in PAIRS:
        d0 = base[a_][0] - base[b_][0]
        parts = []
        for k in kinds:
            sub = df[df.возмущение == k].pivot(index="повтор", columns="набор", values="S")
            d = sub[a_] - sub[b_]
            parts.append(f"{k} смещ. {d.mean() - d0:+.4f}, SD {d.std(ddof=1):.4f}")
        sub = df[df.возмущение == "П2 60 м"].pivot(index="повтор", columns="набор", values="S")
        d60 = sub[a_] - sub[b_]
        sd2, sd3 = d60.std(ddof=1), p3(a_, b_)
        thr = Z95 * np.sqrt(sd2 ** 2 + sd3 ** 2)
        bias = d60.mean() - d0
        log(f"  {a_} − {b_}: исходное {d0:+.4f} | " + " | ".join(parts) + f" | П3 SD {sd3:.4f}")
        log(f"    ПОРОГ пары = 1,96 × √({sd2:.4f}² + {sd3:.4f}²) = {thr:.4f} -> "
            + ("больше порога" if abs(d0) > thr else "в пределах шума меры")
            + (f"; ОГРАНИЧЕНИЕ: |смещение П2(60)| {abs(bias):.4f} > SD {sd2:.4f}" if abs(bias) > sd2 else ""))

if __name__ == "__main__":
    main()
