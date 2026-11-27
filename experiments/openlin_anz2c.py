"""ANZ2c: линии по гребню сырого выхода сети ANZ2 (logit) — без сшивки и сглаживания.

Объявление audit/ANZ2C_DECLARATION.md. Сеть ANZ2 не переобучается; порог 0,9930 — из ANZ2.
    python -m experiments.openlin_anz2c --run
    python -m experiments.openlin_anz2c --windows
ANZ2C_N_TESTS = 0.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy import ndimage as ndi

from experiments import openlin_anz1 as A1
from experiments import openlin_anz2 as A2
from experiments import openlin_anz2b as B2
from experiments import openlin_syn1 as S1
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_iso import Q_LOW
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, LINK_MIN_LEN_M

ANZ2C_N_TESTS = 0
TEACHER = "Анфиса-карта ANZ2, линии ANZ2c по гребню сырого выхода сети; учитель — искусственные линии"
CREST_SIG = 1.0                  # σ гессиана гребня — 1 клетка (20 м, RES_IN), объявление п. 2
OUT_DIR = OUT / "anz2c"
_EIGHT = np.ones((3, 3), bool)


def logit_of(p):
    return float(np.log(p / (1.0 - p)))


def predict_logit(net, X, tile=512, pad=16):
    """Как A1.predict, но без сигмоиды (объявление п. 1)."""
    import torch
    net.eval()
    C, H, W = X.shape
    out = np.zeros((H, W), "float32")
    dev = next(net.parameters()).device
    with torch.no_grad():
        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                a0, b0 = max(r0 - pad, 0), max(c0 - pad, 0)
                a1, b1 = min(r0 + tile + pad, H), min(c0 + tile + pad, W)
                x = torch.from_numpy(X[None, :, a0:a1, b0:b1].copy()).to(dev)
                p = net(x)[0, 0].cpu().numpy()
                out[r0:min(r0 + tile, H), c0:min(c0 + tile, W)] = p[r0 - a0:r0 - a0 + min(tile, H - r0),
                                                                       c0 - b0:c0 - b0 + min(tile, W - c0)]
    return out


def crest(lg, valid):
    """Гребень: λmin гессиана < 0 и сглаженный logit ≥ соседей вдоль нормали (±1 клетка)."""
    s = CREST_SIG
    g = ndi.gaussian_filter(lg, s)
    hrr = ndi.gaussian_filter(lg, s, order=(2, 0))
    hcc = ndi.gaussian_filter(lg, s, order=(0, 2))
    hrc = ndi.gaussian_filter(lg, s, order=(1, 1))
    tr = hrr + hcc
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - (hrr * hcc - hrc ** 2), 0))
    lmin = tr / 2 - disc
    nr, nc = hrc, lmin - hrr                          # собственный вектор при λmin — нормаль к гребню
    n = np.hypot(nr, nc)
    bad = n < 1e-9
    nr = np.where(bad, 1.0, nr / np.where(bad, 1, n)); nc = np.where(bad, 0.0, nc / np.where(bad, 1, n))
    rr, cc = np.mgrid[0:lg.shape[0], 0:lg.shape[1]].astype("float32")
    a = ndi.map_coordinates(g, [rr + nr, cc + nc], order=1, mode="nearest")
    b = ndi.map_coordinates(g, [rr - nr, cc - nc], order=1, mode="nearest")
    return (g >= a) & (g >= b) & (lmin < 0) & valid


def lines_crest(lg, valid, hi, lo, log, lbl):
    from skimage.morphology import skeletonize
    lg = np.where(valid, lg, -50.0).astype("float32")
    sel = crest(lg, valid) & (lg >= lo)
    c = skeletonize(sel)
    lab, n_all = ndi.label(c, structure=_EIGHT)
    seeds = np.unique(lab[c & (lg >= hi)])
    seeds = seeds[seeds > 0]
    c = np.isin(lab, seeds)
    raw = B2.long_paths(c)               # long_paths уже отсекает куски < 200 м по клеточной длине
    polys = [p for p in raw if B2.plen_m(p) >= LINK_MIN_LEN_M]
    L = B2.rasterize(polys, lg.shape)
    tort = np.array([B2.plen_m(p) / max(np.hypot(*(p[-1] - p[0])) * RES_IN, 1e-9) for p in polys]) if polys else np.ones(1)
    holes_lab, nh = ndi.label(~L)
    path_cells = sum(len(p) for p in polys)
    st = {"площадь отбора км²": float(sel.sum()) * RES_IN ** 2 / 1e6, "частей гребня": int(n_all),
          "частей с клеткой ≥ верх": int(len(seeds)), "доля клеток гребня в таких частях": float(c.sum() / max(int((lab > 0).sum()), 1)),
          "км гребня": float(c.sum()) * RES_IN / 1000, "клеток потеряно (части → пути)": int(c.sum() - path_cells),
          "путей до отсева 200 м": len(raw), "км путей до отсева": sum(map(B2.plen_m, raw)) / 1000,
          "линий": len(polys), "км": sum(map(B2.plen_m, polys)) / 1000,
          "км по клеткам": path_cells * RES_IN / 1000,
          "извилистость (медиана, 90 %)": [float(np.median(tort)), float(np.quantile(tort, 0.9))],
          "замкнутых ячеек": int(nh - 1), "клеток вне годных": int((L & ~valid).sum())}
    log(f"    {lbl}: отбор {st['площадь отбора км²']:.1f} км², частей {n_all} → с клеткой ≥ верх {len(seeds)} "
        f"({st['доля клеток гребня в таких частях']:.2f} клеток); гребня {st['км гребня']:.0f} км, потеряно на путях "
        f"{st['клеток потеряно (части → пути)']} клеток; путей до отсева {len(raw)} ({st['км путей до отсева']:.0f} км); "
        f"линий {st['линий']} ({st['км']:.0f} км по центрам клеток, {st['км по клеткам']:.0f} км по клеткам); извилистость медиана "
        f"{st['извилистость (медиана, 90 %)'][0]:.2f}, 90 % {st['извилистость (медиана, 90 %)'][1]:.2f}; замкнутых ячеек "
        f"{st['замкнутых ячеек']}; клеток вне годных {st['клеток вне годных']}")
    if st["клеток вне годных"]:
        raise SystemExit(f"СТОП: {lbl}: клеток линий вне годных {st['клеток вне годных']} (поправка 1 п. 4)")
    return L, polys, st


def run(log):
    import torch
    from src.detection_limit import robust_sigma
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a2 = json.loads((A2.OUT_DIR / "anz2_итог.json").read_text(encoding="utf-8"))
    thr = float(a2["порог"]["значение"])
    hi = logit_of(thr)
    res = {"объявление": "audit/ANZ2C_DECLARATION.md", "учитель": TEACHER, "ANZ2C_N_TESTS": ANZ2C_N_TESTS,
           "порог карты ANZ2": thr, "верх logit": hi, "σ гребня клеток": CREST_SIG}
    dev = A1.device()
    net = A1.make_net(12).to(dev)
    net.load_state_dict(torch.load(A2.OUT_DIR / "anz2_net.pt", map_location=dev))
    log(f"  сеть ANZ2 ({dev}); порог {thr:.4f} → logit {hi:.3f}; низ = logit({Q_LOW:.0%} карты подделки Б листа); "
        f"σ гребня {CREST_SIG:g} клетка")

    def chain(site, lg, lgB, v):
        bad = int((~np.isfinite(lgB[v])).sum() + (~np.isfinite(lg[v])).sum())
        lo = float(np.quantile(lgB[v], Q_LOW))           # 90 %-квантиль самого logit подделки Б (поправка 1 п. 1)
        low_p = 1 / (1 + np.exp(-lo))
        log(f"  {site}: клеток с нечисловым logit {bad}")
        L, polys, st = lines_crest(lg, v, hi, lo, log, f"{site} снимок")
        LB, _p, stB = lines_crest(lgB, v, hi, lo, log, f"{site} подделка Б")
        r = {"низ p": low_p, "низ logit": lo, "снимок": st, "подделка Б": stB,
             "доля шума": stB["км"] / max(st["км"], 1e-9), "способ длины": "евклидово по центрам клеток пути",
             "лист обучения": site != "utsjoki"}
        log(f"  {site}: низ {low_p:.4f} (logit {lo:.2f}); линий {st['км']:.1f} км (Б {stB['км']:.1f}); доля шума "
            f"{r['доля шума']:.3f}")
        return L, polys, r
    train_sites = S1.SITES[:4]
    out = {}
    for i, site in enumerate(train_sites):
        img, dem, raw, valid, _m = A1._sheet(site)
        lg = predict_logit(net, A2.feature_stack2(img, dem, raw, valid))
        lgB = predict_logit(net, A2.feature_stack2(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i), valid))
        L, polys, res[site] = chain(site, lg, lgB, valid)
        if site == "anabar":
            out["anabar"] = (1 / (1 + np.exp(-lg)), L, polys)
    iu = S1.SITES.index("utsjoki")
    img, dem, raw, valid, _m = A1._sheet("utsjoki")
    lu = predict_logit(net, A2.feature_stack2(img, dem, raw, valid))
    lBu = predict_logit(net, A2.feature_stack2(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + iu), valid))
    Lu, polys_u, res["utsjoki"] = chain("utsjoki", lu, lBu, valid)
    old_sig = S.IMG_SIG_M
    S.IMG_SIG_M = A2.IMG_SIG_ANZ2
    try:
        plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(S1.SEED_PLAN_UTS), S1.N_PROPOSALS)
        i2, d2, a2_, truth = S.apply_plan(img, dem, raw, plan, robust_sigma(img[valid].astype("float64")),
                                          S.relief_sigma(dem, valid), RES_IN, valid=valid)
    finally:
        S.IMG_SIG_M = old_sig
    a2_[~np.isfinite(raw)] = np.nan
    li = predict_logit(net, A2.feature_stack2(i2, d2, a2_, valid))
    Li, _p, _s = lines_crest(li, valid, hi, res["utsjoki"]["низ logit"], log, "utsjoki + врезки")
    cells, dirs = A1.line_dirs(Li)
    cov = S1.recall_of_insertions(cells, dirs, truth, plan, CONF_TOL_M, TAU)
    amps = np.array([p["amp"] for p in plan])
    thin = np.array([A2._is_thin(p) for p in plan])
    rng_s = np.random.default_rng(A1.SEED_RANDOM)
    shifts = [(int(rng_s.integers(valid.shape[0])), int(rng_s.integers(valid.shape[1]))) for _ in range(A1.N_SHIFTS)]
    covs_rnd = [S1.recall_of_insertions((cells + np.array(sh)) % np.array(valid.shape), dirs, truth, plan, CONF_TOL_M, TAU)
                for sh in shifts]
    rows = []
    for grp, gm in (("узкие 20 м", thin), ("прежние ширины", ~thin)):
        for lo_, hi_ in zip(S1.AMP_BINS[:-1], S1.AMP_BINS[1:]):
            sel = gm & (amps >= lo_) & (amps < hi_) & np.isfinite(cov)
            rate = float(np.mean(cov[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
            rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in covs_rnd])) if sel.any() else float("nan")
            rows.append({"группа": grp, "корзина": f"{lo_:g}–{hi_:g}σ", "врезок": int(sel.sum()), "найдено": rate,
                         "случайный сдвиг": rnd})
            log(f"  (а) {grp} {lo_:g}–{hi_:g}σ: врезок {int(sel.sum())}, найдено {rate:.3f}; сдвиги {rnd:.3f}")
    res["(а)"] = rows
    fb_ = sum(res[k]["подделка Б"]["км"] for k in train_sites)
    fa_ = sum(res[k]["снимок"]["км"] for k in train_sites)
    log(f"=== сводно по листам обучения (оптимистично, не контроль): линии {fa_:.0f} км, Б {fb_:.0f} км, доля шума "
        f"{fb_ / max(fa_, 1e-9):.3f}; описательно, оценка снизу, не точность")
    for name, path in (("ANZ2", A2.OUT_DIR / "anz2_итог.json"), ("ANZ2b", B2.OUT_DIR / "anz2b_итог.json")):
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
            for k in ("utsjoki", "anabar"):
                km = j[k].get("км линий", j[k].get("снимок", {}).get("км"))
                log(f"  рядом {name} {k}: линий {km:.1f} км, доля шума {j[k]['доля шума после сшивки']:.3f}")
        except (OSError, KeyError, TypeError) as e:
            log(f"  {name} рядом не прочитан: {e}")
    A1.OUT_DIR, A1.TEACHER, A1.NAME = OUT_DIR, TEACHER, "ANZ2c"
    A1._save("utsjoki", 1 / (1 + np.exp(-lu)), Lu, polys_u, res["utsjoki"]["доля шума"], S1.load_site("utsjoki")[4], log)
    sa, La, pa = out["anabar"]
    A1._save("anabar", sa, La, pa, res["anabar"]["доля шума"], S1.load_site("anabar")[4], log)
    res["код"] = {"anz2c": __import__("experiments.openlin_len1", fromlist=["sha"]).sha(__file__)}
    (OUT_DIR / "anz2c_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz2c_итог.json'}; время {time.time() - t0:.0f} с")


def main():
    from experiments.openlin_fam import log_to
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--run" in sys.argv:
        log = log_to(OUT_DIR / "_anz2c.log")
        log("ANZ2c; audit/ANZ2C_DECLARATION.md")
        run(log)
    elif "--windows" in sys.argv:
        A1.OUT_DIR, A1.NAME = OUT_DIR, "ANZ2c"
        A1.windows(log_to(OUT_DIR / "_anz2c_windows.log"))


if __name__ == "__main__":
    main()
