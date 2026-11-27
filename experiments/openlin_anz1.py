"""ANZ1: карта Анфисы по клеткам на слоях AZConv2d (ANZA, anza_lira, MIT) + линии max-min путями (LIRA).

Объявление audit/ANZ1_DECLARATION.md (поправки 1–2). Учитель — искусственные линии SYN1 (врезки в подделки).
    python -m experiments.openlin_anz1 --run        # обучение, карты, порог, линии, проверки, выход
    python -m experiments.openlin_anz1 --windows    # окна (после охотника)
    python -m experiments.openlin_anz1 --run --crop 600 --epochs 1   # отладка на кусках
ANZ1_N_TESTS = 0. Видеокарта — если свободна (разрешение Артура 24.11), иначе процессор.
"""
from __future__ import annotations

import heapq
import json
import math
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.special import ndtri

from experiments import openlin_anfl as A
from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments import openlin_syn1 as S1
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU
from experiments.openlin_fam import DEM_SIG_M, OUT, RES_IN
from experiments.openlin_iso import LOCAL_WIN_M, Q_LOW
from experiments.openlin_zone_lines import DIR_TOL_DEG
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, LINK_MIN_LEN_M, TWIN_OFFSET_M, V3_GAP_M

ANZ1_N_TESTS = 0
TEACHER = "Анфиса-карта (ANZA), линии LIRA; учитель — искусственные линии (ANZ1)"
# сеть и обучение — configs/article3_spacenet_recover_azcat.yaml (anza_lira; поправки 1–2)
N_RULES, WIDTHS = 6, (32, 64, 128)
LR, WD, EPOCHS, PATCH, FG_BIAS, BATCH = 3e-4, 1e-4, 18, 256, 0.7, 2
POS_WEIGHT, TV_A, TV_B = 6.0, 0.35, 0.6
Q_TOP = 0.98                       # верхний узел сетки N_THR (SYN1b)
SEED_TORCH = 42                    # seed конфигурации источника
SEED_RANDOM, N_SHIFTS = 3500, 20   # случайный выбор в (а): 20 тороидальных сдвигов линий; зерно 3500 (описательно)
CROP = None
OUT_DIR = OUT / "anz1"
NAME = "ANZ1"                       # подпись окон (ANZ1b подменяет)


# ---------------------------------------------------------------- вход по клеткам
def _rank_normal(a, valid):
    """Ранги по годным клеткам листа → нормальное распределение (как признаки ANFL0b)."""
    out = np.zeros(a.shape, "float32")
    v = a[valid]
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), "float64")
    r[order] = (np.arange(len(v)) + 0.5) / len(v)
    out[valid] = ndtri(r).astype("float32")
    return out


def rel_window(a, valid):
    k = max(int(round(LOCAL_WIN_M / RES_IN)), 3)
    w = valid.astype("float64")
    x = np.where(valid, a, 0.0)
    den = np.maximum(uniform_filter(w, k, mode="nearest"), 1e-6)
    m = uniform_filter(x, k, mode="nearest") / den
    sq = uniform_filter(x * x, k, mode="nearest") / den
    return (a - m) / np.sqrt(np.maximum(sq - m * m, 1e-12))


def feature_stack(img, dem, alos_raw, valid) -> np.ndarray:
    """10 карт (объявление п. 1), каждая — ранги по листу → нормальное."""
    maps = [C2.local_norm_nz(C1.response(ch, img, dem)[0], valid) for ch in range(5)]
    gy, gx = np.gradient(dem, RES_IN)
    maps.append(np.degrees(np.arctan(np.hypot(gx, gy))))
    maps.append(rel_window(dem, valid))
    maps.append(rel_window(img, valid))
    alos = np.where(np.isfinite(alos_raw), alos_raw, np.nanmedian(alos_raw))
    for s in (+1, -1):
        maps.append(C2.local_norm_nz(C1.ridge_dir(alos, DEM_SIG_M, s)[0], valid))
    return np.stack([_rank_normal(np.asarray(m, "float64"), valid) for m in maps])


def label_map(truth, valid):
    """1 — ≤ 60 м от следа врезки; −1 — кольцо 60–127 м (не учится); 0 — дальше; −1 вне годных."""
    d = S1._trace_dist_m(truth, valid.shape)
    lab = np.where(d <= CONF_TOL_M, 1, np.where(d <= TWIN_OFFSET_M, -1, 0)).astype("int8")
    lab[~valid] = -1
    return lab


# ---------------------------------------------------------------- сеть
def make_net(in_ch):
    import torch.nn as nn

    from experiments.ext_anza.azconv import AZConv2d
    layers, c = [], in_ch
    for w in WIDTHS:
        layers += [AZConv2d(c, w, kernel_size=3, num_rules=N_RULES), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
        c = w
    layers.append(nn.Conv2d(c, 1, 1))
    return nn.Sequential(*layers)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss(logit, lab):
    import torch
    import torch.nn.functional as F
    m = (lab >= 0).float()
    y = lab.clamp(min=0).float()
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(POS_WEIGHT, device=logit.device),
                                             reduction="none")
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logit) * m
    tp, fp, fn = (p * y).sum(), (p * (1 - y)).sum(), ((1 - p) * y * m).sum()
    tv = 1 - (tp + 1) / (tp + TV_A * fp + TV_B * fn + 1)
    return bce + tv


def train_net(sheets, log, epochs=EPOCHS):
    """sheets: список (X (10,H,W), lab (H,W)). Эпоха — столько участков, сколько помещается в листы без перекрытия."""
    import torch
    torch.manual_seed(SEED_TORCH)
    rng = np.random.default_rng(SEED_TORCH)
    dev = device()
    net = make_net(sheets[0][0].shape[0]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    per_epoch = sum(max(1, (X.shape[1] // PATCH) * (X.shape[2] // PATCH)) for X, _ in sheets)
    pos = [np.argwhere(lab == 1) for _, lab in sheets]
    log(f"  сеть: AZConv2d×{len(WIDTHS)} ширины {WIDTHS}, правил {N_RULES}; устройство {dev}; участков на эпоху {per_epoch}, "
        f"эпох {epochs}, участок {PATCH}, доля с врезкой {FG_BIAS}, lr {LR}, wd {WD}, BCE(pos_weight {POS_WEIGHT}) + Tversky "
        f"({TV_A}, {TV_B})")
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for it in range(0, per_epoch, BATCH):
            xb, yb = [], []
            for _ in range(BATCH):
                k = rng.integers(len(sheets))
                X, lab = sheets[k]
                H, W = lab.shape
                ps = min(PATCH, H, W)
                if rng.random() < FG_BIAS and len(pos[k]):
                    r, c = pos[k][rng.integers(len(pos[k]))]
                    r0 = int(np.clip(r - ps // 2, 0, H - ps)); c0 = int(np.clip(c - ps // 2, 0, W - ps))
                else:
                    r0 = int(rng.integers(0, H - ps + 1)); c0 = int(rng.integers(0, W - ps + 1))
                xb.append(X[:, r0:r0 + ps, c0:c0 + ps]); yb.append(lab[r0:r0 + ps, c0:c0 + ps])
            x = torch.from_numpy(np.stack(xb)).to(dev)
            y = torch.from_numpy(np.stack(yb)[:, None].astype("int64")).to(dev)
            loss = _loss(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        log(f"    эпоха {ep + 1}/{epochs}: потеря {tot / max(1, per_epoch // BATCH):.4f}; {time.time() - t0:.0f} с")
    return net


def predict(net, X, tile=512, pad=16):
    """Карта оценки 0…1 по листу кусками с перекрытием pad."""
    import torch
    dev = next(net.parameters()).device
    net.eval()
    C, H, W = X.shape
    out = np.zeros((H, W), "float32")
    with torch.no_grad():
        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                a0, b0 = max(r0 - pad, 0), max(c0 - pad, 0)
                a1, b1 = min(r0 + tile + pad, H), min(c0 + tile + pad, W)
                x = torch.from_numpy(X[None, :, a0:a1, b0:b1].copy()).to(dev)
                p = torch.sigmoid(net(x))[0, 0].cpu().numpy()
                out[r0:min(r0 + tile, H), c0:min(c0 + tile, W)] = p[r0 - a0:r0 - a0 + min(tile, H - r0),
                                                                       c0 - b0:c0 - b0 + min(tile, W - c0)]
    return out


# ---------------------------------------------------------------- линии (LIRA)
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def widest_path(score, start, goal, box):
    """Путь с наибольшим слабым звеном (anza_lira path_completion/widest_path.py, MIT; ребро — меньшая из оценок
    двух клеток), в окне box = (r0, r1, c0, c1). Возвращает (слабое звено, путь)."""
    r0, r1, c0, c1 = box
    best = {start: (1.0, 0)}
    parent = {}
    q = [(-1.0, 0, start)]
    while q:
        ns, ln, p = heapq.heappop(q)
        s = -ns
        if best.get(p) != (s, ln):
            continue
        if p == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return s, path[::-1]
        for dr, dc in _NB8:
            n = (p[0] + dr, p[1] + dc)
            if not (r0 <= n[0] < r1 and c0 <= n[1] < c1):
                continue
            e = min(float(score[p]), float(score[n]))
            if e <= 0:
                continue
            cand = (min(s, e), ln + 1)
            prev = best.get(n, (-1.0, 2 ** 31 - 1))
            if cand[0] > prev[0] or (cand[0] == prev[0] and cand[1] < prev[1]):
                best[n] = cand
                parent[n] = p
                heapq.heappush(q, (-cand[0], cand[1], n))
    return 0.0, []


def lines_from_map(score, valid, thr, bridge_thr, log, label):
    """Порог → скелет → куски ≥ 200 м → мостики LIRA между встречными концами (геометрия сшивки ветки: разрыв ≤ 900 м,
    сдвиг ≤ 127 м, угол ≤ 20°), принят, если слабое звено ≥ bridge_thr; каждый конец — один мостик (жадно).
    Возвращает (маска линий, маска до сшивки, число мостиков, км мостиков, полилинии)."""
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches
    from experiments.openlin_linb5 import _end, length_px_m
    score = np.where(valid, score, 0.0).astype("float32")   # вне годных путь не идёт (widest_path: ребро ≤ 0 закрыто)
    sk = skeletonize((score >= thr) & valid)
    pieces = [pl for pl in branches(sk) if length_px_m(pl) >= LINK_MIN_LEN_M]
    base = np.zeros(valid.shape, bool)
    for pl in pieces:
        q = np.round(pl).astype(int)
        base[q[:, 0], q[:, 1]] = True
    ends = []
    for i, pl in enumerate(pieces):
        if len(pl) < 2:
            continue
        for head in (True, False):
            p, d = _end(pl, head)
            ends.append((i, p, d))
    gap, off = V3_GAP_M / RES_IN, TWIN_OFFSET_M / RES_IN
    cos_tol = math.cos(math.radians(DIR_TOL_DEG))
    cand = []
    if ends:
        tree = cKDTree(np.array([e[1] for e in ends]))
        for a, b in tree.query_pairs(gap):
            ia, pa, da = ends[a]
            ib, pb, db = ends[b]
            if ia == ib or np.dot(da, -db) < cos_tol:
                continue
            v = pb - pa
            if np.hypot(*v) > 0:
                if np.dot(v, da) <= 0 or np.dot(-v, db) <= 0:
                    continue
                if max(abs(da[0] * v[1] - da[1] * v[0]), abs(db[0] * v[1] - db[1] * v[0])) > off:
                    continue
            cand.append((a, b))
    H, W = valid.shape
    m = int(off) + 2
    found = []
    for a, b in cand:
        pa = tuple(np.round(ends[a][1]).astype(int)); pb = tuple(np.round(ends[b][1]).astype(int))
        box = (max(min(pa[0], pb[0]) - m, 0), min(max(pa[0], pb[0]) + m + 1, H),
               max(min(pa[1], pb[1]) - m, 0), min(max(pa[1], pb[1]) + m + 1, W))
        s, path = widest_path(score, pa, pb, box)
        if path and s >= bridge_thr:
            found.append((s, a, b, path))
    found.sort(key=lambda t: -t[0])
    used, mask, n_br, br_cells = set(), base.copy(), 0, 0
    for s, a, b, path in found:
        if a in used or b in used:
            continue
        used |= {a, b}
        pp = np.array(path)
        mask[pp[:, 0], pp[:, 1]] = True
        n_br += 1
        br_cells += len(pp)
    polys = [pl for pl in branches(skeletonize(mask)) if length_px_m(pl) >= LINK_MIN_LEN_M]
    final = np.zeros(valid.shape, bool)
    for pl in polys:
        q = np.round(pl).astype(int)
        final[q[:, 0], q[:, 1]] = True
    log(f"    {label}: кусков ≥ 200 м {len(pieces)} ({base.sum() * RES_IN / 1000:.0f} км), пар-кандидатов {len(cand)}, "
        f"принято мостиков {n_br} ({br_cells * RES_IN / 1000:.1f} км), линий {len(polys)} ({final.sum() * RES_IN / 1000:.0f} км)")
    return final, base, n_br, br_cells * RES_IN / 1000, polys


def line_dirs(mask):
    """Направление клеток линий: главная ось клеток маски в окне 5×5 (соглашение ядер ветки)."""
    m = mask.astype("float64")
    rr, cc = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype("float64")
    k = 5
    n = uniform_filter(m, k)
    mr, mc = uniform_filter(m * rr, k) / np.maximum(n, 1e-9), uniform_filter(m * cc, k) / np.maximum(n, 1e-9)
    srr = uniform_filter(m * rr * rr, k) / np.maximum(n, 1e-9) - mr ** 2
    scc = uniform_filter(m * cc * cc, k) / np.maximum(n, 1e-9) - mc ** 2
    src = uniform_filter(m * rr * cc, k) / np.maximum(n, 1e-9) - mr * mc
    ang = 0.5 * np.arctan2(2 * src, srr - scc)              # угол главной оси от оси строк
    ur, uc = np.cos(ang), np.sin(ang)
    cells = np.argwhere(mask)
    return cells, (np.arctan2(uc, -ur) % np.pi)[mask]


# ---------------------------------------------------------------- прогон
def _sheet(site):
    img, dem, raw, valid, meta = S1.load_site(site)
    if CROP:
        sl = (slice(1500, 1500 + CROP), slice(1700, 1700 + CROP))
        img, dem, raw, valid = img[sl], dem[sl], raw[sl], valid[sl]
    return img, dem, raw, valid, meta


def run(log, epochs=EPOCHS):
    from src.detection_limit import robust_sigma
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"объявление": "audit/ANZ1_DECLARATION.md", "учитель": TEACHER, "ANZ1_N_TESTS": ANZ1_N_TESTS}
    train_sites = S1.SITES[:4]
    assert "utsjoki" not in train_sites
    import torch
    import scipy
    from experiments.ext_anza.azconv import AZConvConfig
    log(f"  версии: torch {torch.__version__}, cuda {torch.version.cuda}, numpy {np.__version__}, scipy {scipy.__version__}; "
        f"устройство {device()}; seed torch {SEED_TORCH}; потолок пула врезок на лист {S1.N_PROPOSALS}; "
        f"зёрна подделки А {S1.SEED_FAKE_A}+i, плана +{S1.SEED_PLAN_SHIFT}, подделки Б {S1.SEED_FAKE_B}+i; az_*-конфигурация {AZConvConfig()}")
    log("  Утсйоки в обучение не входит: листы обучения " + ", ".join(train_sites))
    tr, real, fakeB, fakeA0 = [], {}, {}, {}
    for i, site in enumerate(train_sites):
        img, dem, raw, valid, _m = _sheet(site)
        f_img, f_dem, f_alos = S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_A + i)
        plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(S1.SEED_FAKE_A + i + S1.SEED_PLAN_SHIFT),
                           S1.N_PROPOSALS)
        sig_i, sig_d = robust_sigma(f_img[valid].astype("float64")), S.relief_sigma(f_dem, valid)
        i2, d2, a2, truth = S.apply_plan(f_img, f_dem, f_alos, plan, sig_i, sig_d, RES_IN, valid=valid)
        a2[~np.isfinite(raw)] = np.nan
        X = feature_stack(i2, d2, a2, valid)
        lab = label_map(truth, valid)
        tr.append((X, lab))
        log(f"  {site}: подделка А + врезки: врезок {len(plan)}, клеток «да» {int((lab == 1).sum())}, «нет» {int((lab == 0).sum())}")
        f_alos2 = f_alos.copy(); f_alos2[~np.isfinite(raw)] = np.nan
        fakeA0[site] = (feature_stack(f_img, f_dem, f_alos2, valid), valid)
        real[site] = (feature_stack(img, dem, raw, valid), valid)
        fb = S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i)
        fakeB[site] = (feature_stack(*fb, valid), valid)
    net = train_net(tr, log, epochs)
    del tr
    import torch
    torch.save(net.state_dict(), OUT_DIR / "anz1_net.pt")
    maps = {k: predict(net, X) for k, (X, _v) in real.items()}
    mapsB = {k: predict(net, X) for k, (X, _v) in fakeB.items()}
    mapsA0 = {k: predict(net, X) for k, (X, _v) in fakeA0.items()}
    allr = np.concatenate([maps[k][real[k][1]] for k in maps])
    thr = float(np.quantile(allr, Q_TOP))
    log(f"=== порог карты: квантиль {Q_TOP:.0%} оценок настоящих клеток листов обучения = {thr:.4f} (выбран после "
        f"диагностики SYN1, не проверка; оптимистичен — листы обучения)")
    ar = af = 0
    area = {}
    for k in maps:
        v = real[k][1]
        a_r = float((maps[k][v] >= thr).sum()) * RES_IN ** 2 / 1e6
        a_f = float((mapsB[k][v] >= thr).sum()) * RES_IN ** 2 / 1e6
        a_0 = float((mapsA0[k][v] >= thr).sum()) * RES_IN ** 2 / 1e6
        ar += a_r; af += a_f
        area[k] = {"снимок км²": a_r, "подделка Б км²": a_f, "подделка А без врезок км²": a_0}
        log(f"  {k}: площадь выше порога — снимок {a_r:.2f} км², подделка Б {a_f:.2f} км² ({a_f / max(a_r, 1e-9):.3f}), "
            f"подделка А без врезок {a_0:.2f} км² ({a_0 / max(a_r, 1e-9):.3f})")
    log(f"  сводно: подделка Б / снимок по площади {af / max(ar, 1e-9):.3f} (оценка снизу: тот же генератор, что «нет»)")
    res["порог"] = {"значение": thr, "квантиль": Q_TOP, "площадь": area}

    def chain(site, sc, v, scB, log_):
        bthr = float(np.quantile(scB[v], Q_LOW))
        L, base, nb, kmb, polys = lines_from_map(sc, v, thr, bthr, log_, f"{site} снимок")
        LB, baseB, nbB, kmbB, _p = lines_from_map(scB, v, thr, bthr, log_, f"{site} подделка Б")
        km = lambda m: m.sum() * RES_IN / 1000
        j_before = km(baseB) / max(km(base), 1e-9)
        j_after = km(LB) / max(km(L), 1e-9)
        log_(f"  {site}: порог мостика (90 % карты подделки Б) {bthr:.4f}; линий снимка {km(L):.1f} км; доля шума линий до "
             f"сшивки {j_before:.3f}, после {j_after:.3f}")
        return L, polys, {"порог мостика": bthr, "км линий": km(L), "км линий подделки Б": km(LB), "мостиков": nb,
                          "км мостиков": kmb, "мостиков на подделке Б": nbB, "км до сшивки": km(base),
                          "км до сшивки подделки Б": km(baseB), "доля шума до сшивки": j_before,
                          "доля шума после сшивки": j_after, "лист обучения": site != "utsjoki"}
    # ---------------- листы обучения fam_r1–3: линии и доля шума по листам (п. 5 поправки 1; оптимистично)
    for k in train_sites[:3]:
        _L, _p, res[k] = chain(k + " (лист обучения)", maps[k], real[k][1], mapsB[k], log)
    # ---------------- Утсйоки (проверка)
    iu = S1.SITES.index("utsjoki")
    img, dem, raw, valid, meta_u = _sheet("utsjoki")
    Xu = feature_stack(img, dem, raw, valid)
    su = predict(net, Xu)
    sBu = predict(net, feature_stack(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + iu), valid))
    Lu, polys_u, res["utsjoki"] = chain("utsjoki", su, valid, sBu, log)
    # (а) врезки в настоящий фон
    plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(S1.SEED_PLAN_UTS), S1.N_PROPOSALS)
    i2, d2, a2, truth = S.apply_plan(img, dem, raw, plan, robust_sigma(img[valid].astype("float64")),
                                     S.relief_sigma(dem, valid), RES_IN, valid=valid)
    a2[~np.isfinite(raw)] = np.nan
    si = predict(net, feature_stack(i2, d2, a2, valid))
    Li, _b, _n, _k, _p = lines_from_map(si, valid, thr, res["utsjoki"]["порог мостика"], log, "utsjoki + врезки")
    cells, dirs = line_dirs(Li)
    cov = S1.recall_of_insertions(cells, dirs, truth, plan, CONF_TOL_M, TAU)
    amps = np.array([p["amp"] for p in plan])
    f_len = Li.sum() * RES_IN / 1000
    # случайный выбор той же длины и формы: линии снимка со врезками сдвинуты по тору на случайный вектор
    rng_s = np.random.default_rng(SEED_RANDOM)
    shifts = [(int(rng_s.integers(valid.shape[0])), int(rng_s.integers(valid.shape[1]))) for _ in range(N_SHIFTS)]
    covs_rnd = [S1.recall_of_insertions((cells + np.array(sh)) % np.array(valid.shape), dirs, truth, plan, CONF_TOL_M, TAU)
                for sh in shifts]
    rows = []
    for lo, hi in zip(S1.AMP_BINS[:-1], S1.AMP_BINS[1:]):
        sel = (amps >= lo) & (amps < hi) & np.isfinite(cov)
        rate = float(np.mean(cov[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
        rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in covs_rnd])) if sel.any() else float("nan")
        rows.append({"корзина": f"{lo:g}–{hi:g}σ", "врезок": int(sel.sum()), "найдено": rate, "случайный выбор": rnd})
        log(f"  (а) {lo:g}–{hi:g}σ: врезок {int(sel.sum())}, найдено {rate:.3f}; случайный сдвиг тех же линий ({N_SHIFTS}) {rnd:.3f}")
    total_km = valid.sum() * RES_IN ** 2 / 1e6
    log(f"  (а) линий на снимке с врезками {f_len:.1f} км (площадь листа {total_km:.0f} км²); σ врезок — по подделке "
        f"(контраст относительно шума подделки, не настоящего фона)")
    res["(а)"] = rows
    _save("utsjoki", su, Lu, polys_u, res["utsjoki"]["доля шума после сшивки"], meta_u, log)
    # ---------------- Анабар (цель; лист обучения — оценки оптимистичны)
    v = real["anabar"][1]
    La, polys_a, res["anabar"] = chain("anabar", maps["anabar"], v, mapsB["anabar"], log)
    tk = [k for k in train_sites if k in res]
    sb = sum(res[k]["км до сшивки подделки Б"] for k in tk); sa = sum(res[k]["км до сшивки"] for k in tk)
    fb_, fa_ = sum(res[k]["км линий подделки Б"] for k in tk), sum(res[k]["км линий"] for k in tk)
    log(f"=== сводно по листам обучения ({', '.join(tk)}; оптимистично, не контроль): линии снимка {fa_:.0f} км, "
        f"подделка Б {fb_:.0f} км; доля шума до сшивки {sb / max(sa, 1e-9):.3f}, после {fb_ / max(fa_, 1e-9):.3f}; "
        f"описательная карта, отношение — оценка снизу, не точность")
    _, _, _, _, meta_a = S1.load_site("anabar")
    _save("anabar", maps["anabar"], La, polys_a, res["anabar"]["доля шума после сшивки"], meta_a, log)
    res["код"] = {"anz1": __import__("experiments.openlin_len1", fromlist=["sha"]).sha(__file__)}
    (OUT_DIR / "anz1_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz1_итог.json'}")


def _save(site, score, mask, polys, junk, meta, log):
    import geopandas as gpd
    from shapely.geometry import LineString
    np.savez_compressed(OUT_DIR / f"anz1_{site}.npz", score=score.astype("float16"), lines=mask, junk=junk,
                        teacher=TEACHER)
    rows = []
    for pl in polys:
        if len(pl) < 2:
            continue
        x = meta["x0"] + (pl[:, 1] + 0.5) * RES_IN
        y = meta["y_top"] - (pl[:, 0] + 0.5) * RES_IN
        rows.append({"учитель": TEACHER, "доля_шума_листа": float(junk),
                     "лист_обучения": site != "utsjoki", "geometry": LineString(np.c_[x, y])})
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{int(meta['epsg'])}") if rows else \
        gpd.GeoDataFrame({"учитель": [], "доля_шума_листа": [], "лист_обучения": []}, geometry=[],
                         crs=f"EPSG:{int(meta['epsg'])}")
    g.to_file(OUT_DIR / f"anz1_{site}.gpkg", driver="GPKG")
    log(f"  {site}: {OUT_DIR / f'anz1_{site}.npz'}, .gpkg ({len(rows)} линий; доля шума листа {junk:.3f})")


def windows(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for site, win_km in (("utsjoki", 6), ("anabar", 8)):
        img, dem, _raw, valid, _m = _sheet(site)
        z = np.load(OUT_DIR / f"anz1_{site}.npz")
        sc, L, junk = z["score"].astype("float32"), z["lines"], float(z["junk"])
        dst = OUT_DIR / f"anz1_{site}"
        dst.mkdir(exist_ok=True)
        H, Wd = img.shape
        wp = int(win_km * 1000 / RES_IN)
        k = 0
        for i in range(3):
            for j in range(3):
                k += 1
                r0, c0 = int((i + 0.5) * H / 3 - wp / 2), int((j + 0.5) * Wd / 3 - wp / 2)
                sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
                fig, axs = plt.subplots(1, 3, figsize=(27, 9.4))
                b = img[sl]
                lo, hi = np.nanpercentile(b, [2, 98])
                for ax in axs:
                    ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=(0, win_km, win_km, 0))
                axs[0].set_title("зимний S2, без линий", fontsize=14)
                axs[1].imshow(sc[sl], cmap="magma", alpha=0.55, extent=(0, win_km, win_km, 0), vmin=0, vmax=1)
                axs[1].set_title("карта Анфисы (ANZA): оценка «похоже на линию»", fontsize=14)
                rr, cc = np.nonzero(L[sl])
                axs[2].plot(cc * RES_IN / 1000, rr * RES_IN / 1000, ".", ms=1.6, color="red")
                axs[2].set_title("линии LIRA (красные)", fontsize=14)
                for ax in axs:
                    ax.set_xlim(0, win_km); ax.set_ylim(win_km, 0)
                fig.suptitle(f"{NAME} {site}, окно {k}; учитель — искусственные линии. Описательная карта: отношение км "
                             f"линий подделки к км линий снимка ≈ {junk:.0%} — доля линий, неотличимых от шума метода "
                             f"(не точность); линии слабее ~2σ не видны" + ("; лист обучения" if site != "utsjoki" else ""),
                             fontsize=12)
                fig.tight_layout()
                p = dst / f"окно_{k}.png"
                fig.savefig(p, dpi=70)
                plt.close(fig)
                log(f"  {p}")


def main():
    global CROP, OUT_DIR
    from experiments.openlin_fam import log_to
    epochs = EPOCHS
    if "--crop" in sys.argv:
        CROP = int(sys.argv[sys.argv.index("--crop") + 1])
        OUT_DIR = OUT / f"anz1_crop{CROP}"
    if "--epochs" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("--epochs") + 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--run" in sys.argv:
        log = log_to(OUT_DIR / "_anz1.log")
        log(f"ANZ1; audit/ANZ1_DECLARATION.md; кусок {CROP or 'нет'}; эпох {epochs}")
        run(log, epochs)
    elif "--windows" in sys.argv:
        log = log_to(OUT_DIR / "_anz1_windows.log")
        windows(log)


if __name__ == "__main__":
    main()
