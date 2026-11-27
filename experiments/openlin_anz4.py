"""ANZ4: ANZ3 на родном шаге снимка 10 м (B04 Sentinel-2). Объявление audit/ANZ4_DECLARATION.md (+ поправки).

Шаг задаётся ДО первого импорта цепочки (поправка 1 registrar): openlin_fam.RES_IN = 10, затем импортируются модули —
их `from ... import RES_IN`, значения по умолчанию `res_m=RES_IN` и производные константы берут 10 м. Код ANZ3 не правится.
Отличия по коду: загрузка листа 10 м (рельеф 20 м ×2 билинейно), POOL 16, N_DIR 64, PATCH 512; признаки во float16,
карты настоящих листов и подделки Б — после обучения по одному листу (память).

    python -m experiments.openlin_anz4 --check                 # проверки шага/единиц + float16 + проба линии
    python -m experiments.openlin_anz4 --run --crop 600 --epochs 1   # проба
    python -m experiments.openlin_anz4 --run
    python -m experiments.openlin_anz4 --windows
    python -m experiments.openlin_anz4 --validate             # повторная заверка отметками Артура — после аудита
ANZ4_N_TESTS = 0.
"""
from __future__ import annotations

import experiments.openlin_fam as FAM

RES10 = 10.0
FAM.RES_IN = RES10                                 # до импорта цепочки (поправка 1)

import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from experiments import openlin_anz1 as A1  # noqa: E402
from experiments import openlin_anz2 as A2  # noqa: E402
from experiments import openlin_anz2c as A2C  # noqa: E402
from experiments import openlin_anz3 as A3  # noqa: E402
from experiments import openlin_syn1 as S1  # noqa: E402
from experiments import openlin_syn_insert as S  # noqa: E402
from experiments.openlin_acon1 import TAU  # noqa: E402
from experiments.openlin_fam import COMP, DEM, OUT  # noqa: E402
from experiments.openlin_iso import Q_LOW  # noqa: E402
from experiments.openlin_zone_lines_v3 import CONF_TOL_M, V3_GAP_M  # noqa: E402

ANZ4_N_TESTS = 0
ANZ4_N_TESTS_PRIOR = 160
TEACHER = "Анфиса ANZ4 (ANZ3 на снимке 10 м), линии по гребню; учитель — искусственные линии и зоны"
POOL = 16                                          # 7·k·10 ≥ 1000 → k ≥ 14,3 → 16 (п. 5)
N_DIR = 64                                         # π/64 ≈ atan(10/200) (п. 4)
PATCH = 512                                        # 5,12 км (п. 6)
ZONE_MIN_CELLS_EXP = 6361                          # int(π·450²/10²) (поправка 1)
A1.PATCH, A3.POOL, A3.N_DIR = PATCH, POOL, N_DIR
OUT_DIR = OUT / "anz4"
A3.OUT_DIR = OUT_DIR
MARKUP = Path("outputs/разметка_анабар_8окон")
RESUME = "--resume" in sys.argv                 # поправка 6


# ---------------------------------------------------------------- лист 10 м
def load_site10(site):
    """Снимок 10 м (медиана B04 зимних сцен, shade), рельеф GLO-30 и ALOS 20 м ×2 билинейно (новой детали нет).
    Проверки: те же сцены и угол сетки, что у листа 20 м; форма 10 м ≥ 2× формы 20 м."""
    z = np.load(COMP / f"winter_{site}_10m_лист_родн.npz")
    z2 = np.load(COMP / f"winter_{site}_20m_лист.npz")
    assert set(z["scenes"]) == set(z2["scenes"]), f"{site}: сцены 10 м и 20 м различаются"
    assert abs(float(z["x0"]) - float(z2["x0"])) < 1e-3 and abs(float(z["y_top"]) - float(z2["y_top"])) < 1e-3, \
        f"{site}: угол сетки 10 м не совпадает с 20 м"
    img20 = z2["shade"].astype("float64")
    dem20 = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy").astype("float64")
    H, W = min(img20.shape[0], dem20.shape[0]), min(img20.shape[1], dem20.shape[1])
    v20 = ((z2["count"] > 0) & np.isfinite(img20))[:H, :W] & np.isfinite(dem20[:H, :W])
    alos20 = np.load(DEM / f"dem_alos30_{site}_лист_20m.npy").astype("float64")[:H, :W]
    assert z["shade"].shape[0] >= 2 * H and z["shade"].shape[1] >= 2 * W, f"{site}: лист 10 м меньше 2× листа 20 м"
    img = z["shade"][:2 * H, :2 * W].astype("float64")
    v10 = ((z["count"] > 0) & np.isfinite(z["shade"]))[:2 * H, :2 * W]
    rep = lambda m: np.repeat(np.repeat(m, 2, 0), 2, 1)
    up = lambda a: ndi.zoom(a, 2, order=1, grid_mode=True, mode="nearest")
    valid = v10 & rep(v20)
    fill = lambda a: np.where(np.isfinite(a), a, np.nanmedian(a))
    dem = up(fill(dem20[:H, :W]))
    raw = up(fill(alos20))
    raw[rep(~np.isfinite(alos20))] = np.nan
    img = fill(img)
    meta = {"res_m": RES10, "x0": float(z["x0"]), "y_top": float(z["y_top"]), "epsg": float(z["epsg"])}
    if S1.CROP:
        Hh, Ww = valid.shape
        r0, c0 = max(Hh // 2 - S1.CROP // 2, 0), max(Ww // 2 - S1.CROP // 2, 0)
        sl = (slice(r0, r0 + S1.CROP), slice(c0, c0 + S1.CROP))
        img, dem, raw, valid = img[sl].copy(), dem[sl].copy(), raw[sl].copy(), valid[sl].copy()
        meta = dict(meta, x0=meta["x0"] + c0 * RES10, y_top=meta["y_top"] - r0 * RES10)
    return img, dem, raw, valid, meta


S1.load_site = load_site10


def check_units(log):
    """Поправка 1: во всех модулях цепочки RES_IN = 10, значения по умолчанию res_m = 10, производные константы."""
    bad = []
    for name, m in list(sys.modules.items()):
        if not name.startswith("experiments.openlin_") or m is None:
            continue
        if getattr(m, "RES_IN", RES10) != RES10:
            bad.append(f"{name}.RES_IN = {m.RES_IN}")
        for fn_name, fn in inspect.getmembers(m, inspect.isfunction):
            if fn.__module__ != name:
                continue
            for p in inspect.signature(fn).parameters.values():
                if p.name in ("res_m", "res") and isinstance(p.default, (int, float)) and p.default == 20.0:
                    bad.append(f"{name}.{fn_name}({p.name}=20)")
    k200 = A3._line_kernels(200).shape[-1]
    checks = {"ZONE_MIN_CELLS": (A3.ZONE_MIN_CELLS, ZONE_MIN_CELLS_EXP), "ядро 200 м": (k200, 21),
              "ядро 1000 м": (A3._line_kernels(1000).shape[-1], 101), "A1.PATCH": (A1.PATCH, PATCH),
              "A3.POOL": (A3.POOL, POOL), "A3.N_DIR": (A3.N_DIR, N_DIR), "A2.THIN_M": (A2.THIN_M, 10.0),
              "ширины врезок снимка": (tuple(A2.IMG_SIG_ANZ2), (10.0, 40, 80, 160)),
              "S1.load_site": (S1.load_site is load_site10, True)}
    for k, (got, exp) in checks.items():
        if got != exp:
            bad.append(f"{k}: {got} вместо {exp}")
    # проба: прямая линия 1000 м под 30° на сетке 10 м — длина пути и угол
    from experiments import openlin_anz2b as B2
    n = 101
    t = np.arange(n) * 1.0
    a = math.radians(30)
    pl = np.c_[200 - t * math.cos(a), 50 + t * math.sin(a)]
    L = B2.plen_m(pl)
    if abs(L - 1000) > 1:
        bad.append(f"plen_m прямой 1000 м = {L:.1f}")
    log(f"  проверка единиц: {len(bad)} расхождений; ZONE_MIN_CELLS {A3.ZONE_MIN_CELLS}, ядро 200 м {k200} клеток, "
        f"PATCH {A1.PATCH}, POOL {A3.POOL}, N_DIR {A3.N_DIR}, THIN_M {A2.THIN_M}, ширины {A2.IMG_SIG_ANZ2}; plen_m 1000 м → {L:.1f}")
    for b in bad:
        log("   ! " + b)
    if bad:
        raise SystemExit("единицы не сходятся — остановка (поправка 1)")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def markup_hashes(log):
    for f in ("окна.json", "ПАМЯТКА_для_разметки.txt"):
        log(f"  разметка геолога (не открывается до прихода): {f} sha256 {_sha(MARKUP / f)}")


def feats(img, dem, raw, valid):
    return A3.feature_stack3(img, dem, raw, valid).astype(np.float16)


# ---------------------------------------------------------------- обучение / прогноз (float16 → float32 на устройстве)
def _ckpt_wrap(net, on):
    """Поправка 4: слои AZConv на время обучения — с пересчётом промежуточных на обратном проходе (память), float32.
    BatchNorm вне пересчёта (бегущие статистики не обновляются дважды). Снятие обёрток возвращает прежние имена весов."""
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint

    from experiments.ext_anza.azconv import AZConv2d

    class Ckpt(nn.Module):
        _anz4_ckpt = True                             # метка: снимается по ней, а не по классу (класс создаётся заново)

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return checkpoint(self.m, x, use_reentrant=False)
    for seq in (net.fine, net.coarse, net.fuse):
        for i, layer in enumerate(seq):
            if on and isinstance(layer, AZConv2d):
                seq[i] = Ckpt(layer)
            elif not on and getattr(layer, "_anz4_ckpt", False):
                seq[i] = layer.m
    return net


def load_net4(path):
    """Сеть ANZ4 с диска; веса попытки 3 сохранены с обёртками (имена «….m.…») — приставка снимается."""
    import torch
    net = A3.make_net3(16).to(A1.device())
    sd = torch.load(path, map_location=A1.device())
    net.load_state_dict({k.replace(".m.", "."): v for k, v in sd.items()})
    return net


def train_net4(sheets, log, epochs):
    import torch
    torch.manual_seed(A1.SEED_TORCH)
    rng = np.random.default_rng(A1.SEED_TORCH)
    dev = A1.device()
    net = _ckpt_wrap(A3.make_net3(sheets[0][0].shape[0]).to(dev), True)
    opt = torch.optim.AdamW(net.parameters(), lr=A1.LR, weight_decay=A1.WD)
    P = A1.PATCH
    per_epoch = sum(max(1, (X.shape[1] // P) * (X.shape[2] // P)) for X, _ in sheets)
    pos = [np.argwhere((lab[0] == 1) | (lab[1] == 1)) for _, lab in sheets]
    log(f"  сеть ANZ4: тонкая {A3.WIDTHS_FINE}, грубая (÷{POOL}) {A3.WIDTHS_COARSE}, слияние {A3.WIDTH_FUSE}, 2 выхода; "
        f"устройство {dev}; участок {P} клеток = {P * RES10 / 1000:.2f} км; участков на эпоху {per_epoch}, эпох {epochs}")
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        tot, n_ok, n_skip = 0.0, 0, 0
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
            # поправка 3: участки батча — по одному, градиент накапливается, шаг после BATCH участков;
            # поправка 4: float32 (bf16 дал NaN в первой эпохе), память — пересчётом слоёв AZConv
            opt.zero_grad()
            ok, part = True, 0.0
            for xq, yq in zip(xb, yb):
                x = torch.from_numpy(xq[None]).to(dev).float()
                y = torch.from_numpy(yq[None].astype("int64")).to(dev)
                lg = net(x)
                loss = (A1._loss(lg[:, :1], y[:, :1]) + A1._loss(lg[:, 1:], y[:, 1:])) / len(xb)
                if not torch.isfinite(loss):
                    ok = False
                    break
                loss.backward()
                part += float(loss.detach())
            if ok:
                ok = all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
            if ok:                                   # поправка 5: шаг с нечисловой потерей/градиентом пропускается
                opt.step()
                tot += part
                n_ok += 1
            else:
                n_skip += 1
        log(f"    эпоха {ep + 1}/{epochs}: потеря {tot / max(1, n_ok):.4f}; пропущено шагов {n_skip}; {time.time() - t0:.0f} с")
        if n_skip > 0.01 * (n_ok + n_skip) or not all(torch.isfinite(p).all() for p in net.parameters()):
            raise SystemExit(f"нечисловых шагов {n_skip} из {n_ok + n_skip} (> 1 %) или веса не числа — остановка")
    return _ckpt_wrap(net, False)


def predict4(net, X, tile=512, pad=64):
    """Как ANZ3.predict3; pad 64 ≥ 7·16/2 = 56; края — до кратного POOL; float16 → float32."""
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
                x = torch.from_numpy(X[None, :, a0:a1, b0:b1].copy()).to(dev).float()
                h, w = x.shape[-2:]
                ph, pw = (-h) % POOL, (-w) % POOL
                if ph or pw:
                    x = F.pad(x, [0, pw, 0, ph], mode="reflect")
                p = net(x)[0, :, :h, :w].cpu().numpy()
                hh, ww = min(tile, H - r0), min(tile, W - c0)
                out[:, r0:r0 + hh, c0:c0 + ww] = p[:, r0 - a0:r0 - a0 + hh, c0 - b0:c0 - b0 + ww]
    return out


# ---------------------------------------------------------------- признаки в фоновых процессах (поправка 3; результат тот же)
def _job(args):
    """Один набор признаков → файлы в tmp. kind: train (подделка А + врезки и зоны), real, fakeB, uts_ins (план 3400)."""
    import pickle
    from src.detection_limit import robust_sigma
    site, i, kind, tmp, crop = args
    A1.CROP = S1.CROP = crop
    t0 = time.time()
    img, dem, raw, valid, _m = S1.load_site(site)
    old_sig = S.IMG_SIG_M
    S.IMG_SIG_M = A2.IMG_SIG_ANZ2
    try:
        extra, msg = None, ""
        if kind == "train":
            f_img, f_dem, f_alos = S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_A + i)
            rng = np.random.default_rng(S1.SEED_FAKE_A + i + S1.SEED_PLAN_SHIFT)
            plan, zones, n0 = A3.plan_with_zones(valid.shape, valid, rng)
            sig_i, sig_d = robust_sigma(f_img[valid].astype("float64")), S.relief_sigma(f_dem, valid)
            i2, d2, a2, truth = A3.insert_all(f_img, f_dem, f_alos, plan, zones, valid, sig_i, sig_d)
            a2[~np.isfinite(raw)] = np.nan
            lab = A3.labels(truth, zones, valid)
            np.save(tmp / f"{site}_{kind}_lab.npy", lab)
            X = feats(i2, d2, a2, valid)
            msg = (f"{site}: врезок {len(plan)} (из {n0}), зон {len(zones)} (штрихов "
                   f"{sum(len(z['strokes']) for z in zones)}); клеток «линия» {int((lab[0] == 1).sum())}, «зона» "
                   f"{int((lab[1] == 1).sum())}")
        elif kind == "real":
            X = feats(img, dem, raw, valid)
        elif kind == "fakeB":
            X = feats(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i), valid)
        elif kind == "uts_ins":
            rng = np.random.default_rng(S1.SEED_PLAN_UTS)
            plan, zones, n0 = A3.plan_with_zones(valid.shape, valid, rng)
            i2, d2, a2_, truth = A3.insert_all(img, dem, raw, plan, zones, valid,
                                               robust_sigma(img[valid].astype("float64")), S.relief_sigma(dem, valid))
            a2_[~np.isfinite(raw)] = np.nan
            X = feats(i2, d2, a2_, valid)
            extra = {"plan": plan, "zones": zones, "truth": truth, "n0": n0}
            msg = f"utsjoki (а)/(з): врезок {len(plan)} (из {n0}), зон {len(zones)}"
    finally:
        S.IMG_SIG_M = old_sig
    np.save(tmp / f"{site}_{kind}_X.npy", X)
    if extra is not None:
        (tmp / f"{site}_{kind}_extra.pkl").write_bytes(pickle.dumps(extra))
    return site, kind, msg, time.time() - t0


def _pool(n):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(n, maxtasksperchild=1)


# ---------------------------------------------------------------- прогон
def run(log, epochs):
    import torch
    from src.detection_limit import robust_sigma
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check_units(log)
    markup_hashes(log)
    res = {"объявление": "audit/ANZ4_DECLARATION.md", "учитель": TEACHER, "ANZ4_N_TESTS": ANZ4_N_TESTS,
           "ANZ4_N_TESTS_PRIOR": ANZ4_N_TESTS_PRIOR, "шаг м": RES10}
    train_sites = S1.SITES[:4]
    assert "utsjoki" not in train_sites
    old_sig = S.IMG_SIG_M
    S.IMG_SIG_M = A2.IMG_SIG_ANZ2
    try:
        log(f"  torch {torch.__version__}; устройство {A1.device()}; шаг {RES10} м; ширины врезок снимка {S.IMG_SIG_M}; "
            f"направлений {N_DIR}, L {A3.LINE_L_M}; кратность грубой ветви {POOL}; участок {PATCH}; листы обучения "
            f"{', '.join(train_sites)}; признаки float16")
        import pickle
        tmp = OUT_DIR / "_tmp"
        tmp.mkdir(exist_ok=True)
        crop = S1.CROP
        iu = S1.SITES.index("utsjoki")
        # 1) признаки обучения — 2 процесса; 2) остальные 11 наборов — 1 процесс в фоне, пока идёт обучение (память)
        todo = [(s, i, "train", tmp, crop) for i, s in enumerate(train_sites)
                if not ((tmp / f"{s}_train_X.npy").exists() and (tmp / f"{s}_train_lab.npy").exists())]
        if len(todo) < len(train_sites):
            log(f"  признаки обучения с диска (посчитаны прежним запуском теми же зёрнами): "
                f"{', '.join(s for s in train_sites if s not in [t[0] for t in todo])}")
        with _pool(2) as p:
            for site, kind, msg, dt in p.imap_unordered(_job, todo):
                log(f"  {msg}; признаки {dt:.0f} с; {time.time() - t0:.0f} с")
        rest = [(s, i, k, tmp, crop) for i, s in enumerate(train_sites) for k in ("real", "fakeB")] + \
               [("utsjoki", iu, "real", tmp, crop), ("utsjoki", iu, "fakeB", tmp, crop), ("utsjoki", iu, "uts_ins", tmp, crop)]
        # поправка 5: обучение — в одиночку на видеокарте; остальные наборы — после него, 2 процесса
        if RESUME and (OUT_DIR / "anz4_net.pt").exists():      # поправка 6: сеть попытки 3 (обучена полностью)
            net = load_net4(OUT_DIR / "anz4_net.pt")
            log(f"  продолжение: сеть с диска (попытка 3, 18 эпох, пропусков 0), sha1 "
                f"{hashlib.sha1((OUT_DIR / 'anz4_net.pt').read_bytes()).hexdigest()[:12]}")
        else:
            tr = [(np.load(tmp / f"{s}_train_X.npy"), np.load(tmp / f"{s}_train_lab.npy")) for s in train_sites]
            log(f"  признаки обучения загружены ({sum(X.nbytes for X, _ in tr) / 2**30:.1f} ГБ float16)")
            net = train_net4(tr, log, epochs)
            del tr
            torch.save(net.state_dict(), OUT_DIR / "anz4_net.pt")
        rest = [r for r in rest if not (tmp / f"{r[0]}_{r[2]}_X.npy").exists()]
        log(f"  наборов признаков досчитать: {len(rest)} (по одному процессу — память)")
        with _pool(1) as p:
            for site, kind, msg, dt in p.imap_unordered(_job, rest):
                log(f"  признаки {site} {kind}: {dt:.0f} с; {time.time() - t0:.0f} с" + (f"; {msg}" if msg else ""))
        maps, mapsB, valids = {}, {}, {}
        for i, site in enumerate(train_sites):
            valids[site] = S1.load_site(site)[3]
            maps[site] = predict4(net, np.load(tmp / f"{site}_real_X.npy", mmap_mode="r"))
            mapsB[site] = predict4(net, np.load(tmp / f"{site}_fakeB_X.npy", mmap_mode="r"))
            log(f"  карты {site} (снимок, Б): {time.time() - t0:.0f} с")
        thr = {}
        for ch, nm in ((0, "линия"), (1, "зона")):
            allr = np.concatenate([A3._p(maps[k][ch][valids[k]]) for k in maps])
            thr[nm] = float(np.quantile(allr, A1.Q_TOP))
            log(f"=== порог «{nm}»: 98 % оценок настоящих клеток листов обучения = {thr[nm]:.4f}")
        res["порог"] = thr
        hi_l = A2C.logit_of(min(thr["линия"], 1 - 1e-7)); hi_z = A2C.logit_of(min(thr["зона"], 1 - 1e-7))

        def chain(site, mp, mpB, v):
            lo = float(np.quantile(mpB[0][v], Q_LOW))
            L, polys, st = A2C.lines_crest(mp[0], v, hi_l, lo, log, f"{site} снимок")
            LB, _p_, stB = A2C.lines_crest(mpB[0], v, hi_l, lo, log, f"{site} подделка Б")
            Z, ZB = A3.zones_from_map(mp[1], v, hi_z), A3.zones_from_map(mpB[1], v, hi_z)
            km2 = lambda m: float(m.sum()) * RES10 ** 2 / 1e6
            r = {"низ logit": lo, "снимок": st, "подделка Б": stB, "доля шума линий": stB["км"] / max(st["км"], 1e-9),
                 "зоны км² снимок": km2(Z), "зоны км² Б": km2(ZB), "доля шума зон": km2(ZB) / max(km2(Z), 1e-9),
                 "лист обучения": site != "utsjoki"}
            log(f"  {site}: линий {st['км']:.1f} км (Б {stB['км']:.1f}), доля шума {r['доля шума линий']:.3f}; зоны "
                f"{km2(Z):.1f} км² (Б {km2(ZB):.1f}), доля {r['доля шума зон']:.3f}")
            return L, polys, Z, r
        out = {}
        for k in train_sites:
            L, polys, Z, res[k] = chain(k, maps[k], mapsB[k], valids[k])
            if k == "anabar":
                out["anabar"] = (maps[k], L, polys, Z)
        del mapsB
        maps = {"anabar": maps["anabar"]}
        valid = S1.load_site("utsjoki")[3]
        mu = predict4(net, np.load(tmp / "utsjoki_real_X.npy", mmap_mode="r"))
        mBu = predict4(net, np.load(tmp / "utsjoki_fakeB_X.npy", mmap_mode="r"))
        Lu, polys_u, Zu, res["utsjoki"] = chain("utsjoki", mu, mBu, valid)
        del mBu
        ex = pickle.loads((tmp / "utsjoki_uts_ins_extra.pkl").read_bytes())
        plan, zones, truth = ex["plan"], ex["zones"], ex["truth"]
        log(f"  utsjoki (а)/(з): врезок {len(plan)} (из {ex['n0']}), зон {len(zones)}")
    finally:
        S.IMG_SIG_M = old_sig
    mi = predict4(net, np.load(tmp / "utsjoki_uts_ins_X.npy", mmap_mode="r"))
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
    for grp, gm in (("узкие 10 м", thin), ("прежние ширины 40–160 м", ~thin)):
        for lo_, hi_ in zip(S1.AMP_BINS[:-1], S1.AMP_BINS[1:]):
            sel = gm & (amps >= lo_) & (amps < hi_) & np.isfinite(cov)
            rate = float(np.mean(cov[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
            rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in covs_rnd])) if sel.any() else float("nan")
            rows.append({"группа": grp, "корзина": f"{lo_:g}–{hi_:g}σ", "врезок": int(sel.sum()), "найдено": rate,
                         "случайный сдвиг": rnd})
            log(f"  (а) {grp} {lo_:g}–{hi_:g}σ: врезок {int(sel.sum())}, найдено {rate:.3f}; сдвиги {rnd:.3f}")
    res["(а)"] = rows
    Zi = A3.zones_from_map(mi[1], valid, hi_z)
    zrows = []
    zm = [A3.zone_mask([z], valid.shape) & valid for z in zones]
    zf = np.array([float(Zi[m].mean()) if m.any() else np.nan for m in zm])
    zf_rnd = [np.array([float(np.roll(Zi, sh, (0, 1))[m].mean()) if m.any() else np.nan for m in zm]) for sh in shifts]
    za = np.array([z["amp"] for z in zones])
    for lo_, hi_ in zip(A3.ZONE_BINS[:-1], A3.ZONE_BINS[1:]):
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
        j = json.loads((OUT / "anz3" / "anz3_итог.json").read_text(encoding="utf-8"))
        for k in ("utsjoki", "anabar"):
            log(f"  рядом ANZ3 (20 м) {k}: линий {j[k]['снимок']['км']:.1f} км, доля шума {j[k]['доля шума линий']:.3f} "
                f"(корзины и позиции врезок другие — сравнение не парное)")
    except (OSError, KeyError) as e:
        log(f"  ANZ3 рядом не прочитан: {e}")
    res["код"] = {"anz4": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12],
                  "anz3": hashlib.sha1(Path(A3.__file__).read_bytes()).hexdigest()[:12]}
    _save("utsjoki", mu, Lu, polys_u, Zu, res["utsjoki"], S1.load_site("utsjoki")[4], log)
    ma, La, pa, Za = out["anabar"]
    _save("anabar", ma, La, pa, Za, res["anabar"], S1.load_site("anabar")[4], log)
    (OUT_DIR / "anz4_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz4_итог.json'}; время {time.time() - t0:.0f} с")
    __import__("shutil").rmtree(OUT_DIR / "_tmp", ignore_errors=True)


def _save(site, mp, L, polys, Z, r, meta, log):
    A1.OUT_DIR, A1.TEACHER, A1.NAME = OUT_DIR, TEACHER, "ANZ4"
    A1._save(site, A3._p(mp[0]), L, polys, r["доля шума линий"], meta, log)
    np.savez_compressed(OUT_DIR / f"anz4_{site}_зоны.npz", zone_score=A3._p(mp[1]).astype("float16"), zones=Z,
                        line_logit=mp[0].astype("float16"))


# ---------------------------------------------------------------- окна для глаз (прежние 3×3; не пересекают 8 окон разметки)
def _markup_boxes_cells(meta):
    info = json.loads((MARKUP / "окна.json").read_text(encoding="utf-8"))["окна"]
    return [((meta["y_top"] - o["y_max"]) / RES10, (o["x_min"] - meta["x0"]) / RES10,
             (meta["y_top"] - o["y_min"]) / RES10, (o["x_max"] - meta["x0"]) / RES10) for o in info]


def windows(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for site, win_km in (("utsjoki", 6), ("anabar", 8)):
        img, _d, _r, valid, meta = S1.load_site(site)
        z = np.load(OUT_DIR / f"anz1_{site}.npz")
        zz = np.load(OUT_DIR / f"anz4_{site}_зоны.npz")
        sc, L, junk, Z = z["score"].astype("float32"), z["lines"], float(z["junk"]), zz["zones"]
        zs = zz["zone_score"].astype("float32")
        dst = OUT_DIR / f"anz4_{site}"
        dst.mkdir(exist_ok=True)
        H, Wd = img.shape
        wp = int(win_km * 1000 / RES10)
        boxes = _markup_boxes_cells(meta) if site == "anabar" else []
        k = 0
        for i in range(3):
            for j in range(3):
                k += 1
                r0, c0 = int((i + 0.5) * H / 3 - wp / 2), int((j + 0.5) * Wd / 3 - wp / 2)
                for (a0, b0, a1, b1) in boxes:
                    assert r0 + wp <= a0 or a1 <= r0 or c0 + wp <= b0 or b1 <= c0, \
                        f"окно {k} пересекает окно разметки геолога — не рисую (поправка 1)"
                sl = (slice(r0, r0 + wp), slice(c0, c0 + wp))
                fig, axs = plt.subplots(1, 4, figsize=(36, 9.4))
                b = img[sl]
                lo, hi = np.nanpercentile(b, [2, 98])
                ext = (0, win_km, win_km, 0)
                for ax in axs:
                    ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=ext)
                axs[0].set_title("зимний S2, 10 м", fontsize=14)
                axs[1].imshow(sc[sl], cmap="magma", alpha=0.55, extent=ext, vmin=0, vmax=1)
                axs[1].set_title("карта «линия»", fontsize=14)
                axs[2].imshow(zs[sl], cmap="viridis", alpha=0.5, extent=ext, vmin=0, vmax=1)
                axs[2].contour(Z[sl], levels=[0.5], colors="cyan", extent=(0, win_km, 0, win_km), origin="upper")
                axs[2].set_title("карта «зона» (контур — зоны выше порога)", fontsize=14)
                rr, cc = np.nonzero(L[sl])
                axs[3].plot(cc * RES10 / 1000, rr * RES10 / 1000, ".", ms=1.0, color="red")
                axs[3].set_title("линии (гребень)", fontsize=14)
                for ax in axs:
                    ax.set_xlim(0, win_km); ax.set_ylim(win_km, 0)
                fig.suptitle(f"ANZ4 (10 м) {site}, окно {k}; описательная карта, доля линий подделки ≈ {junk:.0%} (оценка "
                             f"снизу, не точность)" + ("; лист обучения" if site != "utsjoki" else ""), fontsize=12)
                fig.tight_layout()
                p = dst / f"окно_{k}.png"
                fig.savefig(p, dpi=70)
                plt.close(fig)
                log(f"  {p}")


# ---------------------------------------------------------------- повторная заверка отметками Артура (обращение 2)
def validate(log):
    raw_ = open(A3.MARKS, "rb").read()
    h = hashlib.sha256(raw_).hexdigest()
    assert h == A3.MARKS_SHA, f"хэш отметок изменился: {h}"
    mk = np.load(A3.MARKS)
    res = {"хэш": h, "обращение": 2,
           "подпись": "повторная заверка (отметки открыты для ANZ3), лист обучения; нижняя граница; не точность; "
                      "не сравнение с ANZ3 (допуск 60 м = 6 клеток против 3)"}
    _img, _d, _r, valid, _m = S1.load_site("anabar")
    H, W = valid.shape
    H2, W2 = H // 2, W // 2                               # сетка 20 м, в которой отмечено (угол общий)
    wp2 = int(8 * 1000 / 20)
    box2 = {k: (int(((k - 1) // 3 + 0.5) * H2 / 3 - wp2 / 2), int(((k - 1) % 3 + 0.5) * W2 / 3 - wp2 / 2)) for k in range(1, 7)}
    box = {k: (2 * r, 2 * c) for k, (r, c) in box2.items()}
    wp = 2 * wp2
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
                Lw = np.roll(Lw, sh, (0, 1))
            d = ndi.distance_transform_edt(~Lw) * RES10
            for i in np.nonzero(lw == k)[0]:
                c = (2 * lc[ids == i] + 0.5) - np.array([r0, c0])     # центр клетки 20 м в индексах 10 м
                ok = (c[:, 0] >= 0) & (c[:, 0] <= wp - 1) & (c[:, 1] >= 0) & (c[:, 1] <= wp - 1)
                if not ok.any():
                    f[i] = np.nan
                    continue
                dv = ndi.map_coordinates(d, c[ok].T, order=1, mode="nearest")
                f[i] = float((dv <= CONF_TOL_M).mean())
        return float(np.nanmean(f >= S1.FOUND_FRAC)), f
    L = np.load(OUT_DIR / "anz1_anabar.npz")["lines"]
    s, f = share(L)
    rnd = [share(L, sh)[0] for sh in shifts]
    res["ANZ4"] = {"доля отмеченных линий найдена (≥ 50 % в 60 м)": s, "сдвиги в окнах мин–макс": [min(rnd), max(rnd)],
                   "сдвиги среднее": float(np.mean(rnd)), "медиана доли клеток": float(np.nanmedian(f))}
    log(f"  заверка линий ANZ4: найдено {s:.3f} из {len(ll)}; медиана покрытия {np.nanmedian(f):.2f}; сдвиги в окнах "
        f"{min(rnd):.3f}–{max(rnd):.3f} (среднее {np.mean(rnd):.3f})")
    try:
        j = json.loads((OUT / "anz3" / "anz3_заверка.json").read_text(encoding="utf-8"))
        log(f"  рядом ANZ3 (20 м, обращение 1): найдено {j['ANZ3']['доля отмеченных линий найдена (≥ 50 % в 60 м)']:.3f}; "
            f"сдвиги {j['ANZ3']['сдвиги в окнах мин–макс']}")
    except (OSError, KeyError) as e:
        log(f"  ANZ3 рядом не прочитан: {e}")
    Z = np.load(OUT_DIR / "anz4_anabar_зоны.npz")["zones"]
    zc = mk["zone_cells"]
    zc10 = np.concatenate([2 * zc + np.array(d) for d in ((0, 0), (0, 1), (1, 0), (1, 1))])
    inside = float(Z[zc10[:, 0], zc10[:, 1]].mean())
    overall = float(Z[valid].mean())
    w34 = np.zeros_like(valid)
    for k in (3, 4):
        r0, c0 = box[k]
        w34[r0:r0 + wp, c0:c0 + wp] = True
    w34 &= valid
    in_w = float(Z[w34].mean())
    res["зоны"] = {"доля отмеченных зон в зонах ANZ4": inside, "доля листа": overall, "доля окон 3–4": in_w,
                   "подъём к листу": inside / max(overall, 1e-9), "подъём к окнам 3–4": inside / max(in_w, 1e-9)}
    log(f"  заверка зон: в отмеченных зонах {inside:.3f} площади; по листу {overall:.3f} (подъём "
        f"{inside / max(overall, 1e-9):.1f}); по окнам 3–4 {in_w:.3f} (подъём {inside / max(in_w, 1e-9):.1f})")
    (OUT_DIR / "anz4_заверка.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------- проверка до прогона
def check(log):
    check_units(log)
    markup_hashes(log)
    A1.CROP = S1.CROP = 600
    img, dem, raw, valid, meta = S1.load_site("anabar")
    X32 = A3.feature_stack3(img, dem, raw, valid).astype("float32")
    X16 = X32.astype(np.float16).astype("float32")
    d = np.abs(X32 - X16)
    log(f"  float16 против float32 (Анабар, кусок 600×600, {X32.shape[0]} карт): макс. расхождение {d.max():.4f}, "
        f"медиана {np.median(d):.5f}; диапазон значений {X32.min():.2f}…{X32.max():.2f}")
    # совпадение 10 м с 20 м на куске: снимок 10 м, усреднённый блоком 2×2, против листа 20 м
    z2 = np.load(COMP / "winter_anabar_20m_лист.npz")["shade"].astype("float64")
    A1.CROP = S1.CROP = None
    img10, *_r = S1.load_site("anabar")
    Hh, Ww = img10.shape
    m10 = img10.reshape(Hh // 2, 2, Ww // 2, 2).mean((1, 3))
    m20 = z2[:Hh // 2, :Ww // 2]
    ok = np.isfinite(m20)
    log(f"  снимок 10 м (блок 2×2) против листа 20 м по всему Анабару: r {np.corrcoef(m10[ok], m20[ok])[0, 1]:.4f} "
        f"(shade считается на своей сетке — r < 1 ожидаем)")


def main():
    global OUT_DIR
    from experiments.openlin_fam import log_to
    epochs = A1.EPOCHS
    if "--crop" in sys.argv:
        A1.CROP = S1.CROP = int(sys.argv[sys.argv.index("--crop") + 1])
        OUT_DIR = OUT / f"anz4_crop{A1.CROP}"
        A3.OUT_DIR = OUT_DIR
    if "--epochs" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("--epochs") + 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--check" in sys.argv:
        check(log_to(OUT_DIR / "_anz4_проверка.log"))
    elif "--run" in sys.argv:
        log = log_to(OUT_DIR / ("_anz4_продолжение.log" if RESUME else "_anz4.log"))
        log(f"ANZ4; audit/ANZ4_DECLARATION.md; шаг {RES10} м; кусок {A1.CROP or 'нет'}; эпох {epochs}")
        run(log, epochs)
    elif "--windows" in sys.argv:
        windows(log_to(OUT_DIR / "_anz4_windows.log"))
    elif "--validate" in sys.argv:
        validate(log_to(OUT_DIR / "_anz4_заверка.log"))


if __name__ == "__main__":
    main()
