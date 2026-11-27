"""ANZ3: Анфиса со слабыми длинными линиями и зонами мелких линий — направленные интегралы, грубая ветвь (обзор ≥ 1 км),
два выхода «линия»/«зона», учитель — врезки ANZ2 + зоны штриховки. Линии — цепочка ANZ2c (гребень logit).

Объявление audit/ANZ3_DECLARATION.md.
    python -m experiments.openlin_anz3 --run
    python -m experiments.openlin_anz3 --windows
    python -m experiments.openlin_anz3 --validate      # заверка отметками Артура — один раз, после аудита
    python -m experiments.openlin_anz3 --run --crop 600 --epochs 1   # проба
ANZ3_N_TESTS = 0.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time

import numpy as np
from scipy import ndimage as ndi

from experiments import openlin_anz1 as A1
from experiments import openlin_anz2 as A2
from experiments import openlin_anz2c as A2C
from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments import openlin_syn1 as S1
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU
from experiments.openlin_fam import OUT, RES_IN
from experiments.openlin_iso import LOCAL_WIN_M, Q_LOW
from experiments.openlin_zone_lines import DIR_TOL_DEG
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, LINK_MIN_LEN_M, TWIN_OFFSET_M, V3_GAP_M

ANZ3_N_TESTS = 0
TEACHER = "Анфиса ANZ3 (направленные интегралы, обзор ≥ 1 км, линия/зона), линии по гребню; учитель — искусственные линии и зоны"
SEP_M = S.SEP_M                                   # 1000 м = LOCAL_WIN_M / 2
N_DIR = 32                                        # шаг π/32 ≈ atan(20/200) (п. 1)
LINE_L_M = (LINK_MIN_LEN_M, SEP_M)                # 200 и 1000 м (п. 1)
POOL = 8                                          # 7·k·20 ≥ SEP_M → k = 8 (п. 2)
WIDTHS_FINE, WIDTHS_COARSE, WIDTH_FUSE = (32, 64, 128), (64, 64, 64), 64
ZONE_D_M = (V3_GAP_M, LOCAL_WIN_M)                # диаметр зоны 900–2000 м (п. 3)
ZONE_AX = (0.5, 1.0)                              # отношение осей
STROKE_L_M = (TWIN_OFFSET_M, LINK_MIN_LEN_M)      # длина штриха 127–200 м
STROKE_STEP_M = TWIN_OFFSET_M                     # шаг 127 м
STROKE_AMP = (S.AMP_SIG[0], 2.0)                  # 0,25–2σ (нижние 3 октавы)
ZONE_BINS = (0.25, 0.5, 1.0, 2.0)
ZONE_MIN_CELLS = int(math.pi * (V3_GAP_M / 2) ** 2 / RES_IN ** 2)   # круг диаметром 900 м
MARKS = "audit/ANZ3_marks/отметки_Артура_24.11.npz"
MARKS_SHA = "8382ac10831bedcd86da2d37cf8719e22d0518a220aaaa9ec8f36e4dfb776977"
OUT_DIR = OUT / "anz3"
_EIGHT = np.ones((3, 3), bool)


# ---------------------------------------------------------------- вход: направленные интегралы
def _line_kernels(L_m):
    n = int(round(L_m / RES_IN)) | 1
    c = n // 2
    ks = np.zeros((N_DIR, 1, n, n), "float32")
    t = np.linspace(-c, c, 4 * n)
    for d in range(N_DIR):
        a = math.pi * d / N_DIR
        rr, cc = c - t * math.cos(a), c + t * math.sin(a)       # соглашение ядер ветки: (−cos a, sin a)
        r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
        fr, fc = rr - r0, cc - c0
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (1, 0, fr * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 1, fr * fc)):
            rr2, cc2 = np.clip(r0 + dr, 0, n - 1), np.clip(c0 + dc, 0, n - 1)
            np.add.at(ks[d, 0], (rr2, cc2), w)
        ks[d, 0] /= ks[d, 0].sum()
    return ks


def oriented_maps(img, valid):
    """4 карты: для тёмной/светлой полосы σ 20 м и L = 200/1000 м — max по 32 направлениям − медиана по направлениям."""
    import torch
    import torch.nn.functional as F
    dev = A1.device()
    out = []
    for s in (+1, -1):
        r = np.asarray(C2.local_norm_nz(C1.ridge_dir(img, (A2.THIN_M,), s)[0], valid), "float32")
        r = np.where(valid, r, 0.0).astype("float32")
        x = torch.from_numpy(r)[None, None].to(dev)
        for L in LINE_L_M:
            k = torch.from_numpy(_line_kernels(L)).to(dev)
            p = k.shape[-1] // 2
            xp = F.pad(x, [p] * 4, mode="reflect")
            H = r.shape[0]
            parts = []
            for a in range(0, H, 512):                    # полосами: 32 канала на весь лист ~2 ГБ + медиана
                y = F.conv2d(xp[:, :, a:min(a + 512, H) + 2 * p], k)[0]
                parts.append((y.max(0).values - y.median(0).values).cpu().numpy())
            out.append(np.concatenate(parts, 0))
    return out


def feature_stack3(img, dem, alos_raw, valid):
    X = A2.feature_stack2(img, dem, alos_raw, valid)
    extra = [A1._rank_normal(np.asarray(m, "float64"), valid) for m in oriented_maps(img, valid)]
    return np.concatenate([X, np.stack(extra)]).astype("float32")


# ---------------------------------------------------------------- сеть
def make_net3(in_ch):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from experiments.ext_anza.azconv import AZConv2d

    def blocks(c, widths):
        layers = []
        for w in widths:
            layers += [AZConv2d(c, w, kernel_size=3, num_rules=A1.N_RULES), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        return nn.Sequential(*layers)

    class Net3(nn.Module):
        def __init__(self):
            super().__init__()
            self.fine = blocks(in_ch, WIDTHS_FINE)
            self.coarse = blocks(in_ch, WIDTHS_COARSE)
            self.fuse = blocks(WIDTHS_FINE[-1] + WIDTHS_COARSE[-1], (WIDTH_FUSE,))
            self.head = nn.Conv2d(WIDTH_FUSE, 2, 1)

        def forward(self, x):
            f = self.fine(x)
            g = self.coarse(F.avg_pool2d(x, POOL))
            g = F.interpolate(g, size=f.shape[-2:], mode="bilinear", align_corners=False)
            return self.head(self.fuse(torch.cat([f, g], 1)))
    return Net3()


def predict3(net, X, tile=512, pad=64):
    """Сырые logit (2, H, W) кусками; pad 64 клетки ≥ половины поля зрения грубой ветви; края — до кратного POOL."""
    import torch
    import torch.nn.functional as F
    dev = next(net.parameters()).device
    net.eval()
    C, H, W = X.shape
    out = np.zeros((2, H, W), "float32")
    with torch.no_grad():
        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                a0, b0 = max(r0 - pad, 0), max(c0 - pad, 0)
                a1, b1 = min(r0 + tile + pad, H), min(c0 + tile + pad, W)
                x = torch.from_numpy(X[None, :, a0:a1, b0:b1].copy()).to(dev)
                h, w = x.shape[-2:]
                ph, pw = (-h) % POOL, (-w) % POOL
                if ph or pw:
                    x = F.pad(x, [0, pw, 0, ph], mode="reflect")
                p = net(x)[0, :, :h, :w].cpu().numpy()
                hh, ww = min(tile, H - r0), min(tile, W - c0)
                out[:, r0:r0 + hh, c0:c0 + ww] = p[:, r0 - a0:r0 - a0 + hh, c0 - b0:c0 - b0 + ww]
    return out


# ---------------------------------------------------------------- зоны штриховки
def _logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def make_zones(plan, shape, valid, rng, n):
    """n зон-эллипсов с разносом SEP_M от следов плана и друг от друга; штрихи — врезки S.insertion шириной 20 м."""
    zones, tries = [], 0
    while len(zones) < n and tries < 50 * n:
        tries += 1
        c = np.array([rng.uniform(0, shape[0]), rng.uniform(0, shape[1])])
        if not valid[int(c[0]), int(c[1])]:
            continue
        D = _logu(rng, *ZONE_D_M)
        a_px, b_px = D / 2 / RES_IN, D / 2 / RES_IN * rng.uniform(*ZONE_AX)
        reach = a_px + SEP_M / RES_IN
        if any(np.hypot(*(c - z["center"])) < reach + z["a_px"] for z in zones):
            continue
        rot = rng.uniform(0, math.pi)
        az = rng.uniform(0.0, 360.0)
        kind = str(rng.choice(("тёмная", "светлая")))
        amp = _logu(rng, *STROKE_AMP)                 # один контраст на зону (поправка 2 п. 1)
        th = math.radians(az)
        u = np.array([-math.cos(th), math.sin(th)])   # вдоль штриха (строка, столбец)
        v = np.array([u[1], -u[0]])
        step = STROKE_STEP_M / RES_IN
        strokes = []
        R = int(a_px / step) + 2
        for i in range(-R, R + 1):
            t_along = -a_px
            while t_along < a_px:
                Ls = _logu(rng, *STROKE_L_M)
                pc = c + v * (i * step + rng.uniform(-0.5, 0.5) * step) + u * (t_along + Ls / 2 / RES_IN)
                d = pc - c
                x1 = d[0] * math.cos(rot) + d[1] * math.sin(rot)
                y1 = -d[0] * math.sin(rot) + d[1] * math.cos(rot)
                if (x1 / a_px) ** 2 + (y1 / b_px) ** 2 <= 1 and 0 <= pc[0] < shape[0] and 0 <= pc[1] < shape[1] \
                        and valid[int(pc[0]), int(pc[1])]:
                    strokes.append(S.insertion(kind, (float(pc[0]), float(pc[1])),
                                               az + rng.uniform(-DIR_TOL_DEG / 2, DIR_TOL_DEG / 2), Ls, amp, rng,
                                               bend=1.0, width_m=A2.THIN_M))
                t_along += Ls / RES_IN + step * (1 + rng.uniform(-0.5, 0.5))
        zones.append({"center": c, "a_px": a_px, "b_px": b_px, "rot": rot, "amp": amp, "kind": kind,
                      "strokes": strokes})
    return zones


def plan_with_zones(shape, valid, rng):
    """Поправка 2: сначала зоны (1/8 числа принятых линий плана), потом из плана убираются линии, след которых ближе
    SEP_M к краю зоны (порядок «линии, потом зоны» давал 0 зон — план с разносом 1 км занимает весь лист)."""
    plan = S.make_plan(shape, valid, RES_IN, rng, S1.N_PROPOSALS)
    zones = make_zones(plan, shape, valid, rng, max(1, len(plan) // 8))
    keep = []
    for p in plan:
        t = S.trace_points(p, RES_IN)
        if all(np.hypot(*(t - z["center"]).T).min() >= z["a_px"] + SEP_M / RES_IN for z in zones):
            keep.append(p)
    return keep, zones, len(plan)


def zone_mask(zones, shape, grow_px=0.0):
    m = np.zeros(shape, bool)
    for z in zones:
        a, b = z["a_px"] + grow_px, z["b_px"] + grow_px
        r0, r1 = int(max(z["center"][0] - a - 1, 0)), int(min(z["center"][0] + a + 2, shape[0]))
        c0, c1 = int(max(z["center"][1] - a - 1, 0)), int(min(z["center"][1] + a + 2, shape[1]))
        rr, cc = np.mgrid[r0:r1, c0:c1]
        d0, d1 = rr - z["center"][0], cc - z["center"][1]
        x1 = d0 * math.cos(z["rot"]) + d1 * math.sin(z["rot"])
        y1 = -d0 * math.sin(z["rot"]) + d1 * math.cos(z["rot"])
        m[r0:r1, c0:c1] |= (x1 / a) ** 2 + (y1 / b) ** 2 <= 1
    return m


def insert_all(img, dem, alos, plan, zones, valid, sig_i, sig_d):
    strokes = [s for z in zones for s in z["strokes"]]
    i2, d2, a2, truth = S.apply_plan(img, dem, alos, plan + strokes, sig_i, sig_d, RES_IN, valid=valid)
    return i2, d2, a2, truth[:len(plan)]


def labels(truth, zones, valid):
    lab_l = A1.label_map(truth, valid)
    zin = zone_mask(zones, valid.shape)
    zring = zone_mask(zones, valid.shape, TWIN_OFFSET_M / RES_IN)
    lab_l[zring] = -1                                    # штрихи зоны в канале «линия» не учатся
    lab_z = np.where(zin, 1, np.where(zring, -1, 0)).astype("int8")
    lab_z[~valid] = -1
    return np.stack([lab_l, lab_z])


# ---------------------------------------------------------------- обучение
def train_net3(sheets, log, epochs=A1.EPOCHS):
    import torch
    torch.manual_seed(A1.SEED_TORCH)
    rng = np.random.default_rng(A1.SEED_TORCH)
    dev = A1.device()
    net = make_net3(sheets[0][0].shape[0]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=A1.LR, weight_decay=A1.WD)
    P = A1.PATCH
    per_epoch = sum(max(1, (X.shape[1] // P) * (X.shape[2] // P)) for X, _ in sheets)
    pos = [np.argwhere((lab[0] == 1) | (lab[1] == 1)) for _, lab in sheets]
    log(f"  сеть ANZ3: тонкая {WIDTHS_FINE}, грубая (÷{POOL}) {WIDTHS_COARSE}, слияние {WIDTH_FUSE}, 2 выхода; устройство "
        f"{dev}; участков на эпоху {per_epoch}, эпох {epochs}")
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _it in range(0, per_epoch, A1.BATCH):
            xb, yb = [], []
            for _ in range(A1.BATCH):
                k = rng.integers(len(sheets))
                X, lab = sheets[k]
                H, W = lab.shape[-2:]
                ps = min(P, H, W) // POOL * POOL
                if rng.random() < A1.FG_BIAS and len(pos[k]):
                    r, c = pos[k][rng.integers(len(pos[k]))]
                    r0 = int(np.clip(r - ps // 2, 0, H - ps)); c0 = int(np.clip(c - ps // 2, 0, W - ps))
                else:
                    r0 = int(rng.integers(0, H - ps + 1)); c0 = int(rng.integers(0, W - ps + 1))
                xb.append(X[:, r0:r0 + ps, c0:c0 + ps]); yb.append(lab[:, r0:r0 + ps, c0:c0 + ps])
            x = torch.from_numpy(np.stack(xb)).to(dev)
            y = torch.from_numpy(np.stack(yb).astype("int64")).to(dev)
            lg = net(x)
            loss = A1._loss(lg[:, :1], y[:, :1]) + A1._loss(lg[:, 1:], y[:, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        log(f"    эпоха {ep + 1}/{epochs}: потеря {tot / max(1, per_epoch // A1.BATCH):.4f}; {time.time() - t0:.0f} с")
    return net


# ---------------------------------------------------------------- зоны: карта → области
def zones_from_map(lz, valid, thr_logit):
    m = (lz >= thr_logit) & valid
    lab, n = ndi.label(m, structure=_EIGHT)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= ZONE_MIN_CELLS)[0] + 1
    return np.isin(lab, keep)


def _p(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- прогон
def run(log, epochs=A1.EPOCHS):
    import torch
    from src.detection_limit import robust_sigma
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"объявление": "audit/ANZ3_DECLARATION.md", "учитель": TEACHER, "ANZ3_N_TESTS": ANZ3_N_TESTS}
    train_sites = S1.SITES[:4]
    assert "utsjoki" not in train_sites
    old_sig = S.IMG_SIG_M
    S.IMG_SIG_M = A2.IMG_SIG_ANZ2
    try:
        log(f"  torch {torch.__version__}; устройство {A1.device()}; ширины врезок снимка {S.IMG_SIG_M}; направлений {N_DIR}, "
            f"L {LINE_L_M}; кратность грубой ветви {POOL}; зоны: диаметр {ZONE_D_M}, оси {ZONE_AX}, штрих {STROKE_L_M} м, шаг "
            f"{STROKE_STEP_M} м, контраст {STROKE_AMP}σ; листы обучения {', '.join(train_sites)}")
        tr, real, fakeB = [], {}, {}
        for i, site in enumerate(train_sites):
            img, dem, raw, valid, _m = A1._sheet(site)
            f_img, f_dem, f_alos = S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_A + i)
            rng = np.random.default_rng(S1.SEED_FAKE_A + i + S1.SEED_PLAN_SHIFT)
            plan, zones, n0 = plan_with_zones(valid.shape, valid, rng)
            sig_i, sig_d = robust_sigma(f_img[valid].astype("float64")), S.relief_sigma(f_dem, valid)
            i2, d2, a2, truth = insert_all(f_img, f_dem, f_alos, plan, zones, valid, sig_i, sig_d)
            a2[~np.isfinite(raw)] = np.nan
            lab = labels(truth, zones, valid)
            tr.append((feature_stack3(i2, d2, a2, valid), lab))
            log(f"  {site}: врезок {len(plan)} (из {n0}), зон {len(zones)} (штрихов {sum(len(z['strokes']) for z in zones)}); клеток "
                f"«линия» {int((lab[0] == 1).sum())}, «зона» {int((lab[1] == 1).sum())}")
            real[site] = (feature_stack3(img, dem, raw, valid), valid)
            fakeB[site] = (feature_stack3(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i), valid), valid)
        net = train_net3(tr, log, epochs)
        del tr
        torch.save(net.state_dict(), OUT_DIR / "anz3_net.pt")
        maps = {k: predict3(net, X) for k, (X, _v) in real.items()}
        mapsB = {k: predict3(net, X) for k, (X, _v) in fakeB.items()}
        thr = {}
        for ch, nm in ((0, "линия"), (1, "зона")):
            allr = np.concatenate([_p(maps[k][ch][real[k][1]]) for k in maps])
            thr[nm] = float(np.quantile(allr, A1.Q_TOP))
            log(f"=== порог «{nm}»: 98 % оценок настоящих клеток листов обучения = {thr[nm]:.4f}")
        res["порог"] = thr
        hi_l = A2C.logit_of(min(thr["линия"], 1 - 1e-7)); hi_z = A2C.logit_of(min(thr["зона"], 1 - 1e-7))

        def chain(site, mp, mpB, v):
            lo = float(np.quantile(mpB[0][v], Q_LOW))
            L, polys, st = A2C.lines_crest(mp[0], v, hi_l, lo, log, f"{site} снимок")
            LB, _p_, stB = A2C.lines_crest(mpB[0], v, hi_l, lo, log, f"{site} подделка Б")
            Z, ZB = zones_from_map(mp[1], v, hi_z), zones_from_map(mpB[1], v, hi_z)
            km2 = lambda m: float(m.sum()) * RES_IN ** 2 / 1e6
            r = {"низ logit": lo, "снимок": st, "подделка Б": stB, "доля шума линий": stB["км"] / max(st["км"], 1e-9),
                 "зоны км² снимок": km2(Z), "зоны км² Б": km2(ZB), "доля шума зон": km2(ZB) / max(km2(Z), 1e-9),
                 "лист обучения": site != "utsjoki"}
            log(f"  {site}: линий {st['км']:.1f} км (Б {stB['км']:.1f}), доля шума {r['доля шума линий']:.3f}; зоны "
                f"{km2(Z):.1f} км² (Б {km2(ZB):.1f}), доля {r['доля шума зон']:.3f}")
            return L, polys, Z, r
        out = {}
        for k in train_sites:
            L, polys, Z, res[k] = chain(k, maps[k], mapsB[k], real[k][1])
            if k == "anabar":
                out["anabar"] = (maps[k], L, polys, Z)
        iu = S1.SITES.index("utsjoki")
        img, dem, raw, valid, _m = A1._sheet("utsjoki")
        mu = predict3(net, feature_stack3(img, dem, raw, valid))
        mBu = predict3(net, feature_stack3(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + iu), valid))
        Lu, polys_u, Zu, res["utsjoki"] = chain("utsjoki", mu, mBu, valid)
        # (а) и (з): план 3400 + зоны
        rng = np.random.default_rng(S1.SEED_PLAN_UTS)
        plan, zones, n0 = plan_with_zones(valid.shape, valid, rng)
        log(f"  utsjoki (а)/(з): врезок {len(plan)} (из {n0}), зон {len(zones)}")
        i2, d2, a2_, truth = insert_all(img, dem, raw, plan, zones, valid, robust_sigma(img[valid].astype("float64")),
                                        S.relief_sigma(dem, valid))
    finally:
        S.IMG_SIG_M = old_sig
    a2_[~np.isfinite(raw)] = np.nan
    mi = predict3(net, feature_stack3(i2, d2, a2_, valid))
    Li, _p2, _s = A2C.lines_crest(mi[0], valid, hi_l, res["utsjoki"]["низ logit"], log, "utsjoki + врезки")
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
    Zi = zones_from_map(mi[1], valid, hi_z)
    zrows = []
    zm = [zone_mask([z], valid.shape) & valid for z in zones]
    zf = np.array([float(Zi[m].mean()) if m.any() else np.nan for m in zm])
    zf_rnd = [np.array([float(np.roll(Zi, sh, (0, 1))[m].mean()) if m.any() else np.nan for m in zm]) for sh in shifts]
    za = np.array([z["amp"] for z in zones])
    for lo_, hi_ in zip(ZONE_BINS[:-1], ZONE_BINS[1:]):
        sel = (za >= lo_) & (za < hi_) & np.isfinite(zf)
        rate = float(np.mean(zf[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
        rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in zf_rnd])) if sel.any() else float("nan")
        zrows.append({"корзина": f"{lo_:g}–{hi_:g}σ", "зон": int(sel.sum()), "найдено": rate, "случайный сдвиг": rnd})
        log(f"  (з) зоны {lo_:g}–{hi_:g}σ: зон {int(sel.sum())}, найдено {rate:.3f}; сдвиги {rnd:.3f}")
    res["(з)"] = zrows
    fb_ = sum(res[k]["подделка Б"]["км"] for k in train_sites)
    fa_ = sum(res[k]["снимок"]["км"] for k in train_sites)
    log(f"=== сводно по листам обучения (оптимистично, не контроль): линии {fa_:.0f} км, Б {fb_:.0f} км, доля шума "
        f"{fb_ / max(fa_, 1e-9):.3f}; описательно, оценка снизу, не точность")
    try:
        j = json.loads((A2C.OUT_DIR / "anz2c_итог.json").read_text(encoding="utf-8"))
        for k in ("utsjoki", "anabar"):
            log(f"  рядом ANZ2c {k}: линий {j[k]['снимок']['км']:.1f} км, доля шума {j[k]['доля шума']:.3f}")
    except (OSError, KeyError) as e:
        log(f"  ANZ2c рядом не прочитан: {e}")
    res["код"] = {"anz3": __import__("experiments.openlin_len1", fromlist=["sha"]).sha(__file__)}
    _save("utsjoki", mu, Lu, polys_u, Zu, res["utsjoki"], S1.load_site("utsjoki")[4], log)
    ma, La, pa, Za = out["anabar"]
    _save("anabar", ma, La, pa, Za, res["anabar"], S1.load_site("anabar")[4], log)
    (OUT_DIR / "anz3_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz3_итог.json'}; время {time.time() - t0:.0f} с")


def _save(site, mp, L, polys, Z, r, meta, log):
    A1.OUT_DIR, A1.TEACHER, A1.NAME = OUT_DIR, TEACHER, "ANZ3"
    A1._save(site, _p(mp[0]), L, polys, r["доля шума линий"], meta, log)
    np.savez_compressed(OUT_DIR / f"anz3_{site}_зоны.npz", zone_score=_p(mp[1]).astype("float16"), zones=Z,
                        line_logit=mp[0].astype("float16"))


# ---------------------------------------------------------------- окна (с зонами)
def windows(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for site, win_km in (("utsjoki", 6), ("anabar", 8)):
        img, _d, _r, valid, _m = A1._sheet(site)
        z = np.load(OUT_DIR / f"anz1_{site}.npz")
        zz = np.load(OUT_DIR / f"anz3_{site}_зоны.npz")
        sc, L, junk, Z = z["score"].astype("float32"), z["lines"], float(z["junk"]), zz["zones"]
        zs = zz["zone_score"].astype("float32")
        dst = OUT_DIR / f"anz3_{site}"
        dst.mkdir(exist_ok=True)
        H, Wd = img.shape
        wp = int(win_km * 1000 / RES_IN)
        k = 0
        for i in range(3):
            for j in range(3):
                k += 1
                r0, c0 = int((i + 0.5) * H / 3 - wp / 2), int((j + 0.5) * Wd / 3 - wp / 2)
                sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
                fig, axs = plt.subplots(1, 4, figsize=(36, 9.4))
                b = img[sl]
                lo, hi = np.nanpercentile(b, [2, 98])
                ext = (0, win_km, win_km, 0)
                for ax in axs:
                    ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=ext)
                axs[0].set_title("зимний S2", fontsize=14)
                axs[1].imshow(sc[sl], cmap="magma", alpha=0.55, extent=ext, vmin=0, vmax=1)
                axs[1].set_title("карта «линия»", fontsize=14)
                axs[2].imshow(zs[sl], cmap="viridis", alpha=0.5, extent=ext, vmin=0, vmax=1)
                axs[2].contour(Z[sl], levels=[0.5], colors="cyan", extent=(0, win_km, 0, win_km), origin="upper")
                axs[2].set_title("карта «зона» (контур — зоны выше порога)", fontsize=14)
                rr, cc = np.nonzero(L[sl])
                axs[3].plot(cc * RES_IN / 1000, rr * RES_IN / 1000, ".", ms=1.6, color="red")
                axs[3].set_title("линии (гребень)", fontsize=14)
                for ax in axs:
                    ax.set_xlim(0, win_km); ax.set_ylim(win_km, 0)
                fig.suptitle(f"ANZ3 {site}, окно {k}; описательная карта, доля линий подделки ≈ {junk:.0%} (оценка снизу, "
                             f"не точность)" + ("; лист обучения" if site != "utsjoki" else ""), fontsize=12)
                fig.tight_layout()
                p = dst / f"окно_{k}.png"
                fig.savefig(p, dpi=70)
                plt.close(fig)
                log(f"  {p}")


# ---------------------------------------------------------------- заверка (один раз)
def validate(log):
    """Заверка на листе обучения, один раз (поправка 1): нижняя граница; сдвиги — внутри тех же окон; два подъёма зон."""
    raw = open(MARKS, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    assert h == MARKS_SHA, f"хэш отметок изменился: {h}"
    mk = np.load(MARKS)
    res = {"хэш": h, "подпись": "заверка на листе обучения; нижняя граница; не точность"}
    img, _d, _r, valid, _m = A1._sheet("anabar")
    H, W = valid.shape
    wp = int(8 * 1000 / RES_IN)
    box = {k: (int(((k - 1) // 3 + 0.5) * H / 3 - wp / 2), int(((k - 1) % 3 + 0.5) * W / 3 - wp / 2)) for k in range(1, 7)}
    lc, ll, lw = mk["line_cells"], mk["line_len"], mk["line_win"]
    ids = np.repeat(np.arange(len(ll)), ll)
    rng_s = np.random.default_rng(A1.SEED_RANDOM)
    shifts = [(int(rng_s.integers(wp)), int(rng_s.integers(wp))) for _ in range(A1.N_SHIFTS)]

    def share(L, sh=None):
        f = np.zeros(len(ll))
        for k in np.unique(lw):
            r0, c0 = box[int(k)]
            Lw = L[r0:r0 + wp, c0:c0 + wp]
            if sh is not None:
                Lw = np.roll(Lw, sh, (0, 1))                 # сдвиг по тору внутри окна
            d = ndi.distance_transform_edt(~Lw) * RES_IN
            for i in np.nonzero(lw == k)[0]:
                c = lc[ids == i] - np.array([r0, c0])
                ok = (c[:, 0] >= 0) & (c[:, 0] < wp) & (c[:, 1] >= 0) & (c[:, 1] < wp)
                f[i] = float((d[c[ok, 0], c[ok, 1]] <= CONF_TOL_M).mean()) if ok.any() else np.nan
        return float(np.nanmean(f >= S1.FOUND_FRAC)), f
    for name, path in (("ANZ3", OUT_DIR / "anz1_anabar.npz"), ("ANZ2c", A2C.OUT_DIR / "anz1_anabar.npz")):
        L = np.load(path)["lines"]
        s, f = share(L)
        rnd = [share(L, sh)[0] for sh in shifts]
        res[name] = {"доля отмеченных линий найдена (≥ 50 % в 60 м)": s, "сдвиги в окнах мин–макс": [min(rnd), max(rnd)],
                     "сдвиги среднее": float(np.mean(rnd)), "медиана доли клеток": float(np.nanmedian(f))}
        log(f"  заверка линий {name}: найдено {s:.3f} из {len(ll)}; медиана покрытия {np.nanmedian(f):.2f}; сдвиги в окнах "
            f"{min(rnd):.3f}–{max(rnd):.3f} (среднее {np.mean(rnd):.3f})")
    Z = np.load(OUT_DIR / "anz3_anabar_зоны.npz")["zones"]
    zc = mk["zone_cells"]
    inside = float(Z[zc[:, 0], zc[:, 1]].mean())
    overall = float(Z[valid].mean())
    w34 = np.zeros_like(valid)
    for k in (3, 4):
        r0, c0 = box[k]
        w34[r0:r0 + wp, c0:c0 + wp] = True
    w34 &= valid
    in_w = float(Z[w34].mean())
    res["зоны"] = {"доля отмеченных зон в зонах ANZ3": inside, "доля листа": overall, "доля окон 3–4": in_w,
                   "подъём к листу": inside / max(overall, 1e-9), "подъём к окнам 3–4": inside / max(in_w, 1e-9)}
    log(f"  заверка зон: в отмеченных зонах {inside:.3f} площади; по листу {overall:.3f} (подъём "
        f"{inside / max(overall, 1e-9):.1f}); по окнам 3–4 {in_w:.3f} (подъём {inside / max(in_w, 1e-9):.1f})")
    (OUT_DIR / "anz3_заверка.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def main():
    global OUT_DIR
    from experiments.openlin_fam import log_to
    epochs = A1.EPOCHS
    if "--crop" in sys.argv:
        A1.CROP = int(sys.argv[sys.argv.index("--crop") + 1])
        OUT_DIR = OUT / f"anz3_crop{A1.CROP}"
    if "--epochs" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("--epochs") + 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--run" in sys.argv:
        log = log_to(OUT_DIR / "_anz3.log")
        log(f"ANZ3; audit/ANZ3_DECLARATION.md; кусок {A1.CROP or 'нет'}; эпох {epochs}")
        run(log, epochs)
    elif "--windows" in sys.argv:
        windows(log_to(OUT_DIR / "_anz3_windows.log"))
    elif "--validate" in sys.argv:
        validate(log_to(OUT_DIR / "_anz3_заверка.log"))


if __name__ == "__main__":
    main()
