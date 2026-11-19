"""FAM1: нечёткая аддитивная модель (3 терма на признак, как в статье о малых
водотоках) учится повторять линеаменты GTK 292 по зимнему снимку и GLO-30.

Объявление — audit/FAM1_DECLARATION.md (10.11.2026, до расчёта). Площадка
настройки Утсйоки: обучение на западных окнах, проверка на восточных; p-проверок
нет, в счётчик ничего не идёт.

Запуск из корня: python -m experiments.openlin_fam --site utsjoki
Только процессор (CUDA_VISIBLE_DEVICES="").
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

COMP = Path("datacache/s2_lowsun/composite")
DEM = Path("datacache/anabar_dem")
GTK = Path("datacache/gtk_lineaments")
OUT = Path("outputs/fam")

# ---- объявление FAM1 (10.11.2026, до расчёта) ------------------------------
RES_IN = 20.0                 # шаг входа
CELL_M = 100.0                # сетка анализа: эталон 1:500 000, допуск 300 м
TOL_M = 300.0                 # допуск сверки ветки (openlin_transfer_check.TOL_M)
LAYER = "292"
IMG_SIG_M = (40, 80, 160)     # тёмная/светлая полоса снимка
DEM_SIG_M = (60, 120, 240)    # вогнутость/выпуклость рельефа (GLO-30 — 30 м)
GRAD_SIG_M = 40
STD_WIN_M = 100
LINE_LEN_M = 1100             # прямолинейность: отрезок 11 ячеек 100 м
N_DIR = 12
WIN_KM = 10
WEST_MAX_KM, EAST_MIN_KM = 38, 48   # зазор 10 км
N_WIN = 8
MIN_VALID = 0.99
WIN_SEED = 2510
N_TERMS = 3
LR, EPOCHS = 5e-3, 100
SEEDS = (1, 2, 3)
FAM1_N_TESTS = 0
FEATURES = ["тёмная полоса", "светлая полоса", "градиент снимка", "разброс яркости",
            "уклон", "вогнутость", "выпуклость", "прямолинейность снимка",
            "прямолинейность рельефа"]
# ---------------------------------------------------------------------------


def log_to(path):
    t0 = time.time()
    f = open(path, "w", encoding="utf-8")

    def log(*a):
        s = f"[{time.time() - t0:7.0f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()
    return log


def hessian_ridge(x, sig_px):
    """Наибольшее собственное число гессиана * σ² (положительно у тёмной полосы)."""
    hrr = ndi.gaussian_filter(x, sig_px, order=(2, 0))
    hcc = ndi.gaussian_filter(x, sig_px, order=(0, 2))
    hrc = ndi.gaussian_filter(x, sig_px, order=(1, 1))
    tr, det = hrr + hcc, hrr * hcc - hrc ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    return (tr / 2 + disc) * sig_px ** 2, (tr / 2 - disc) * sig_px ** 2


def multi_ridge(x, sigs_m):
    pos, neg = None, None
    for s in sigs_m:
        lmax, lmin = hessian_ridge(x, s / RES_IN)
        a, b = np.maximum(lmax, 0), np.maximum(-lmin, 0)
        pos = a if pos is None else np.maximum(pos, a)
        neg = b if neg is None else np.maximum(neg, b)
    return pos, neg


def block(x, k, how):
    H, W = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    v = x[:H, :W].reshape(H // k, k, W // k, k)
    return np.nanmax(v, axis=(1, 3)) if how == "max" else np.nanmean(v, axis=(1, 3))


def straightness(r):
    """Максимум по направлениям среднего вдоль отрезка минус среднее по направлениям."""
    L = int(round(LINE_LEN_M / CELL_M))
    half = L // 2
    resp = []
    for i in range(N_DIR):
        th = np.pi * i / N_DIR
        k = np.zeros((L, L))
        for t in np.linspace(-half, half, 4 * L):
            k[int(round(half - t * np.cos(th))), int(round(half + t * np.sin(th)))] = 1
        resp.append(ndi.convolve(r, k / k.sum(), mode="nearest"))
    resp = np.stack(resp)
    return resp.max(0) - resp.mean(0)


def features(site, log):
    """Признаки листа; кэш в outputs/fam/feat_<лист>.npz (вход не меняется)."""
    c = OUT / f"feat_{site}.npz"
    if c.exists():
        q = np.load(c)
        log(f"признаки {site} из кэша")
        return q["F"], q["valid"], {k: float(q[k]) for k in ("res_m", "x0", "y_top", "epsg")}
    F, valid, meta = _features(site, log)
    F = F.astype("float32")          # одинаково при первом расчёте и из кэша
    np.savez_compressed(c, F=F, valid=valid, **meta)
    return F, valid, meta


def _features(site, log):
    z = np.load(COMP / f"winter_{site}_20m_лист.npz")
    img = z["shade"].astype("float64")
    valid20 = (z["count"] > 0) & np.isfinite(img)
    dem = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy").astype("float64")
    valid20 &= np.isfinite(dem)
    fill = lambda a: np.where(np.isfinite(a), a, np.nanmedian(a))
    img, dem = fill(img), fill(dem)
    meta = {k: float(z[k]) for k in ("res_m", "x0", "y_top", "epsg")}
    k = int(CELL_M / RES_IN)
    dark, bright = multi_ridge(img, IMG_SIG_M)
    grad = ndi.gaussian_gradient_magnitude(img, GRAD_SIG_M / RES_IN)
    w = int(STD_WIN_M / RES_IN)
    m1 = ndi.uniform_filter(img, w)
    std = np.sqrt(np.maximum(ndi.uniform_filter(img ** 2, w) - m1 ** 2, 0))
    gy, gx = np.gradient(dem, RES_IN)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    conc, conv = multi_ridge(dem, DEM_SIG_M)
    log("признаки 20 м посчитаны")
    F = [block(dark, k, "max"), block(bright, k, "max"), block(grad, k, "mean"),
         block(std, k, "mean"), block(slope, k, "mean"), block(conc, k, "max"),
         block(conv, k, "max")]
    F.append(straightness(F[0]))
    F.append(straightness(F[5]))
    valid = block(valid20.astype(float), k, "mean") >= 1.0
    log("прямолинейность посчитана")
    return np.stack(F, -1), valid, meta


def gtk_file(site):
    """Файл эталона: север Финляндии (FAM2), Остроботния — свой, иначе прежний."""
    if site == "finland65":
        return GTK / f"finland65_lin500k_{LAYER}.gpkg"
    if site.startswith("fam_"):
        return GTK / f"north_lin500k_{LAYER}.gpkg"
    return GTK / f"lin500k_{LAYER}.gpkg"


def gtk_label(meta, shape, site="utsjoki"):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    g = gpd.read_file(gtk_file(site)).to_crs(epsg=int(meta["epsg"]))
    tr = Affine(CELL_M, 0, meta["x0"], 0, -CELL_M, meta["y_top"])
    ref = rasterize([(geom, 1) for geom in g.geometry if geom is not None],
                    out_shape=shape, transform=tr, fill=0, all_touched=True,
                    dtype="uint8").astype(bool)
    lab = ndi.binary_dilation(ref, np.ones((3, 3), bool))   # + одна ячейка
    return ref, lab


def all_windows(valid):
    """FAM2: все неперекрывающиеся окна 10 км с >= 99 % полных ячеек."""
    n = int(WIN_KM * 1000 / CELL_M)
    H, W = valid.shape
    return [(r, c) for r in range(0, H - n + 1, n) for c in range(0, W - n + 1, n)
            if valid[r:r + n, c:c + n].mean() >= MIN_VALID], n


def windows(valid):
    """Неперекрывающиеся окна 10 км с >= 99 % полных ячеек; 8 на запад, 8 на восток."""
    n = int(WIN_KM * 1000 / CELL_M)
    H, W = valid.shape
    west, east = [], []
    for r in range(0, H - n + 1, n):
        for c in range(0, W - n + 1, n):
            x0, x1 = c * CELL_M / 1000, (c + n) * CELL_M / 1000
            if valid[r:r + n, c:c + n].mean() < MIN_VALID:
                continue
            if x1 <= WEST_MAX_KM:
                west.append((r, c))
            elif x0 >= EAST_MIN_KM:
                east.append((r, c))
    rng = np.random.default_rng(WIN_SEED)
    pick = lambda L: [L[i] for i in sorted(rng.choice(len(L), min(N_WIN, len(L)), replace=False))]
    return pick(west), pick(east), n


def win_mask(shape, wins, n):
    m = np.zeros(shape, bool)
    for r, c in wins:
        m[r:r + n, c:c + n] = True
    return m


# ---------------------------------------------------------------- модели
def fit_fam(X, y, seed, log=None):
    import torch
    torch.manual_seed(seed)
    d = X.shape[1]
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    c = torch.nn.Parameter(torch.tensor(np.tile([-1.0, 0.0, 1.0], (d, 1)), dtype=torch.float32))
    ls = torch.nn.Parameter(torch.zeros(d, N_TERMS))
    b = torch.nn.Parameter(torch.zeros(d, N_TERMS))
    b0 = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([c, ls, b, b0], lr=LR)
    pw = torch.tensor(float((y == 0).sum() / max((y == 1).sum(), 1)))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    g = torch.Generator().manual_seed(seed)

    def forward(x):
        mu = torch.exp(-0.5 * ((x[:, :, None] - c[None]) / torch.exp(ls)[None]) ** 2)
        mu = mu / (mu.sum(-1, keepdim=True) + 1e-9)
        return (mu * b[None]).sum((1, 2)) + b0

    n = len(Xt)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(forward(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    params = {"c": c.detach().numpy(), "s": np.exp(ls.detach().numpy()),
              "b": b.detach().numpy(), "b0": float(b0.detach())}

    def predict(Xn):
        with torch.no_grad():
            return forward(torch.tensor(Xn, dtype=torch.float32)).numpy()
    return predict, params


def baselines(X, y, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer
    from sklearn.tree import DecisionTreeClassifier
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    out = {}
    m = make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(max_iter=2000))
    m.fit(X, y, logisticregression__sample_weight=w)
    out["аддитивная на сплайнах"] = m
    m = LogisticRegression(max_iter=2000).fit(X, y, sample_weight=w)
    out["логистическая регрессия"] = m
    out["дерево, глубина 4"] = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X, y, sample_weight=w)
    out["градиентный бустинг"] = HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, random_state=seed).fit(X, y, sample_weight=w)
    # перцептрон без весов: балансируем повтором положительных строк
    pos = np.flatnonzero(y == 1)
    rep = np.concatenate([np.arange(len(y)), np.repeat(pos, int(max(w.max() - 1, 0)))])
    out["перцептрон 32-16"] = MLPClassifier((32, 16), max_iter=200, early_stopping=True,
                                            random_state=seed).fit(X[rep], y[rep])
    return {k: (lambda mm: (lambda Z: mm.predict_proba(Z)[:, 1]))(v) for k, v in out.items()}


# ---------------------------------------------------------------- мера
def budget_f1(score, test, ref, K):
    """Равный бюджет: K лучших ячеек; P, R, F1 с допуском TOL_M."""
    s = np.where(test, score, -np.inf)
    flat = np.flatnonzero(np.isfinite(s))
    k = min(K, len(flat))
    top = flat[np.argsort(-s.ravel()[flat], kind="stable")[:k]]
    sel = np.zeros(score.shape, bool)
    sel.ravel()[top] = True
    rad = TOL_M / CELL_M
    d_ref = ndi.distance_transform_edt(~ref)
    d_sel = ndi.distance_transform_edt(~sel)
    P = float((d_ref[sel] <= rad).mean()) if sel.any() else 0.0
    rt = ref & test
    R = float((d_sel[rt] <= rad).mean())
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, int(sel.sum())


def quantile_norm(F, fitmask):
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                             subsample=200000, random_state=0)
    qt.fit(F[fitmask])
    out = np.zeros_like(F)
    out[...] = np.nan
    flat = F.reshape(-1, F.shape[-1])
    ok = np.all(np.isfinite(flat), 1)
    o = out.reshape(-1, F.shape[-1])
    o[ok] = qt.transform(flat[ok])
    return out


def old_tract(site, shape, log):
    """Маски UTS1 / RELU1 (C_all), сведённые к 100 м."""
    out = {}
    k = int(CELL_M / RES_IN)
    for name, tag in (("старый тракт: снимок (UTS1)", "настройка1"),
                      ("старый тракт: рельеф (RELU1)", "рельеф1")):
        p = Path(f"outputs/metrics/openlin_zone_lines_{site}_v3_{tag}.npz")
        if not p.exists():
            log(f"  нет {p}")
            continue
        z = np.load(p)
        key = "C_all" if "C_all" in z.files else z.files[0]
        m = block(z[key].astype(float), k, "max")[:shape[0], :shape[1]] > 0
        out[name] = m
    return out


# ---- FAM2 (audit/FAM2_DECLARATION.md, 10.11.2026) -------------------------
FAM2_TRAIN = ("fam_n1", "fam_n2", "fam_n3")
FAM2_TEST = ("utsjoki", "finland65")
POS_CAP, NEG_PER_POS = 30000, 3     # как в статье о водотоках: до 30 000 и втрое больше фона
FAM2_N_TESTS = 0
# FAM3 (audit/FAM3_DECLARATION.md): листы по расчленённости рельефа (fam_r3 = fam_n1)
FAM3_TRAIN = ("fam_r1", "fam_r2", "fam_r3")
FAM3_N_TESTS = 0


def site_table(site, log):
    """Признаки, выровненные по неразмеченным ячейкам окон этого же листа, метка, окна."""
    F, valid, meta = features(site, log)
    ref, lab = gtk_label(meta, valid.shape, site)
    wins, n = all_windows(valid)
    m = win_mask(valid.shape, wins, n) & valid
    Z = quantile_norm(F, m)
    log(f"  {site}: окон {len(wins)}, ячеек эталона в окнах {int((ref & m).sum())}")
    return dict(F=F, Z=Z, ref=ref, lab=lab, m=m, valid=valid)


def build_train(train, log):
    """Выборка обучения FAM2/FAM3: ячейки эталона окон (до POS_CAP) + втрое фона."""
    rng = np.random.default_rng(WIN_SEED)
    Xs, ys = [], []
    for s in train:
        t = site_table(s, log)
        Z, lab, m = t["Z"], t["lab"], t["m"]
        pos, neg = np.flatnonzero((lab & m).ravel()), np.flatnonzero((~lab & m).ravel())
        pos = rng.choice(pos, min(POS_CAP, len(pos)), replace=False)
        neg = rng.choice(neg, min(NEG_PER_POS * len(pos), len(neg)), replace=False)
        idx = np.concatenate([pos, neg])
        Xs.append(Z.reshape(-1, Z.shape[-1])[idx]); ys.append(lab.ravel()[idx].astype(int))
        del t
    X, y = np.concatenate(Xs), np.concatenate(ys)
    log(f"обучение: строк {len(y)}, из них эталон {int(y.sum())}")
    return X, y


def main_fam2(tag="fam2", train=FAM2_TRAIN):
    """FAM2 и FAM3: одно и то же, разные листы обучения (FAM3 — отбор по рельефу)."""
    log = log_to(OUT / f"_{tag}.log")
    log(f"{tag.upper()}; объявление audit/{tag.upper()}_DECLARATION.md; обучение {train}")
    X, y = build_train(train, log)
    tests = {s: site_table(s, log) for s in FAM2_TEST}
    # Объявлено: Утсйоки — 16 окон FAM1 (8 запад + 8 восток). Первый прогон
    # проверял на всех 52 окнах (расхождение с объявлением, запись в
    # audit/FAM2_DECLARATION.md); 52 окна оставлены сверх объявленного.
    u = tests["utsjoki"]
    w_, e_, n_ = windows(u["valid"])
    tests = {"utsjoki (16 окон FAM1, объявлено)":
             dict(u, m=win_mask(u["valid"].shape, w_ + e_, n_) & u["valid"]),
             "utsjoki (все 52 окна)": u, "finland65": tests["finland65"]}
    models = {}
    params_all = []
    for seed in SEEDS:
        pred, prm = fit_fam(X, y, seed)
        params_all.append(prm)
        models.setdefault("нечёткая аддитивная (3 терма)", []).append(pred)
        for k_, p_ in baselines(X, y, seed).items():
            models.setdefault(k_, []).append(p_)
        log(f"  зерно {seed} обучено")
    import csv
    rows = []
    for s, t in tests.items():
        te, ref, F, Z = t["m"], t["ref"], t["F"], t["Z"]
        K = int((ref & te).sum())
        log(f"=== проверка {s}: K = {K}")
        r0 = np.random.default_rng(0)
        rnd = [budget_f1(np.where(te, r0.random(te.shape), -np.inf), te, ref, K)[2] for _ in range(5)]
        rows.append((s, "случайный выбор", np.mean(rnd), np.std(rnd), np.nan, np.nan))
        log(f"  случайный: F1 {np.mean(rnd):.3f}")
        for j in (0, 5, 7, 8):
            P, R, F1, _ = budget_f1(np.where(te, F[..., j], -np.inf), te, ref, K)
            rows.append((s, FEATURES[j] + " (без обучения)", F1, 0, P, R))
            log(f"  {FEATURES[j]} (без обучения): P {P:.3f} R {R:.3f} F1 {F1:.3f}")
        if s.startswith("utsjoki"):
            for name, mm in old_tract("utsjoki", te.shape, log).items():
                P, R, F1, _ = budget_f1(np.where(mm, 1.0, 0.0) + 1e-6 * F[..., 0], te, ref, K)
                rows.append((s, name, F1, 0, P, R))
                log(f"  {name}: ячеек в маске {int((mm & te).sum())}; P {P:.3f} R {R:.3f} F1 {F1:.3f}")
        for k_, preds in models.items():
            v = []
            for pr in preds:
                sc = np.full(te.shape, -np.inf)
                sc[te] = pr(Z[te])
                v.append(budget_f1(sc, te, ref, K)[:3])
            v = np.array(v)
            rows.append((s, k_, v[:, 2].mean(), v[:, 2].std(), v[:, 0].mean(), v[:, 1].mean()))
            log(f"  {k_}: F1 {v[:, 2].mean():.3f} ± {v[:, 2].std():.3f} (P {v[:, 0].mean():.3f}, R {v[:, 1].mean():.3f})")
    prm = params_all[0]
    log("вклады термов (зерно 1), низкий / средний / высокий:")
    order = np.argsort(prm["c"], 1)
    for j, fname in enumerate(FEATURES):
        bb = prm["b"][j][order[j]]
        log(f"  {fname}: " + " / ".join(f"{x:+.2f}" for x in bb) + f"  (размах {bb.max() - bb.min():.2f})")
    with open(OUT / f"{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["проверка", "модель", "F1", "F1_std", "P", "R"])
        wr.writerows(rows)
    json.dump({"params": [{k: np.asarray(v).tolist() for k, v in p.items()} for p in params_all]},
              open(OUT / f"{tag}_params.json", "w", encoding="utf-8"))
    log("сохранено:", OUT / f"{tag}.csv")


# ---- FAMLAP (audit/FAMLAP_DECLARATION.md + поправки 1, 2; 11.11.2026) ----------
# Разовый контроль на Лапландии. Семья «сверка с GTK на контроле»: prior 6
# (XFER 2 + Пакет 05 4), свои 2 (Т1 fam3, Т2 fam2).
FAMLAP_PRIOR, FAMLAP_OWN = 6, 2
FAMLAP_ALPHA = 0.05 / (FAMLAP_PRIOR + FAMLAP_OWN)          # 0.00625
FAMLAP_SHIFT_N = int(round(4 / FAMLAP_ALPHA)) - 1          # 639, пол 1/640
FAMLAP_SHIFT_SEED = 11                                     # как openlin_transfer_check; общие для Т1, Т2
FREEZE_TOL = 0.002
FAMLAP_MIN_WIN = 16                                        # окон FAM1; меньше — неустойчиво
FAM_NAME = "нечёткая аддитивная (3 терма)"
U16 = "utsjoki (16 окон FAM1, объявлено)"


def select_top(score, test, K):
    """Те же K ячеек, что выбирает budget_f1."""
    s = np.where(test, score, -np.inf)
    flat = np.flatnonzero(np.isfinite(s))
    top = flat[np.argsort(-s.ravel()[flat], kind="stable")[:min(K, len(flat))]]
    sel = np.zeros(score.shape, bool)
    sel.ravel()[top] = True
    return sel


def f1_shifted(sel, ref, test, n, seed):
    """F1 настоящий (сдвиг 0) и при n сдвигах эталона по тору.

    Поля сдвига min(200, H/8) ячеек, как shift_null. Выбранные ячейки стоят;
    R — по ячейкам сдвинутого эталона внутри окон; P — по карте расстояний до
    эталона, сдвинутой тем же сдвигом (np.roll; на краях приближение)."""
    rad = TOL_M / CELL_M
    d_ref = ndi.distance_transform_edt(~ref)
    d_sel = ndi.distance_transform_edt(~sel)
    H, W = ref.shape
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)

    def f1(dr, dc):
        rs = np.roll(ref, (dr, dc), (0, 1)) & test
        P = float((np.roll(d_ref, (dr, dc), (0, 1))[sel] <= rad).mean()) if sel.any() else 0.0
        R = float((d_sel[rs] <= rad).mean()) if rs.any() else 0.0
        return 2 * P * R / (P + R) if P + R else 0.0
    rng = np.random.default_rng(seed)
    shifts = [(int(rng.integers(lo_r, H - lo_r)), int(rng.integers(lo_c, W - lo_c)))
              for _ in range(n)]
    return f1(0, 0), np.array([f1(dr, dc) for dr, dc in shifts])


def main_lapland():
    import csv
    import hashlib
    import shutil

    import cloudpickle
    log = log_to(OUT / "_famlap.log")
    src = Path(__file__)
    h = hashlib.sha256(src.read_bytes()).hexdigest()
    bk = Path("audit/backup_famlap")
    bk.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, bk / "openlin_fam.py")
    log(f"FAMLAP; объявление audit/FAMLAP_DECLARATION.md (+ поправки 1, 2); "
        f"openlin_fam.py sha256 {h}; копия {bk / 'openlin_fam.py'}")
    log(f"семья: prior {FAMLAP_PRIOR} + свои {FAMLAP_OWN}; alpha {FAMLAP_ALPHA:.5f}; сдвигов "
        f"{FAMLAP_SHIFT_N} (зерно {FAMLAP_SHIFT_SEED}); пол {1 / (1 + FAMLAP_SHIFT_N):.5f}; "
        f"запас {FAMLAP_ALPHA * (1 + FAMLAP_SHIFT_N):.1f}x")
    dp = DEM / "dem_glo30_lapland_лист_20m.npy"
    if not dp.exists():
        log(f"ОСТАНОВ: нет {dp} (GLO-30 20 м Лапландии не скачан)")
        return
    dem = np.load(dp)
    if np.isnan(dem).any():
        log(f"ОСТАНОВ: GLO-30 20 м Лапландии недокачан ({np.isnan(dem).mean():.4f} пустых)")
        return
    # ---- проверка заморозки на 16 окнах Утсйоки, до данных Лапландии
    u = site_table("utsjoki", log)
    w_, e_, n_ = windows(u["valid"])
    um = win_mask(u["valid"].shape, w_ + e_, n_) & u["valid"]
    uK = int((u["ref"] & um).sum())
    frozen = {}
    for tag, train in (("fam3", FAM3_TRAIN), ("fam2", FAM2_TRAIN)):
        log(f"=== обучение {tag}: {train}")
        X, y = build_train(train, log)
        models, params = {}, []
        for seed in SEEDS:
            pred, prm = fit_fam(X, y, seed)
            params.append(prm)
            models.setdefault(FAM_NAME, []).append(pred)
            for k_, p_ in baselines(X, y, seed).items():
                models.setdefault(k_, []).append(p_)
        v = []
        for pr in models[FAM_NAME]:
            sc = np.full(um.shape, -np.inf)
            sc[um] = pr(u["Z"][um])
            v.append(budget_f1(sc, um, u["ref"], uK)[2])
        got, want = float(np.mean(v)), None
        with open(OUT / f"{tag}.csv", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["проверка"] == U16 and row["модель"] == FAM_NAME:
                    want = float(row["F1"])
        ok = want is not None and abs(got - want) <= FREEZE_TOL
        log(f"  заморозка {tag}: Утсйоки 16 окон, среднее F1 по зёрнам {got:.6f}; в {tag}.csv "
            f"{want}; {'СОВПАЛО' if ok else 'НЕ СОВПАЛО'} (допуск {FREEZE_TOL})")
        if not ok:
            log("ОСТАНОВ: заморозка не подтверждена, Лапландия не считается")
            return
        with open(OUT / f"frozen_{tag}.pkl", "wb") as f:
            cloudpickle.dump({"models": models, "fam_params": params}, f)
        frozen[tag] = models
    del u
    log("=== заморозка подтверждена (fam3, fam2); Лапландия — один раз")
    # ---- Лапландия
    t = site_table("lapland", log)
    te, ref, F, Z = t["m"], t["ref"], t["F"], t["Z"]
    K = int((ref & te).sum())
    n_win = int(round(te.sum() / (WIN_KM * 1000 / CELL_M) ** 2))
    stable = n_win >= FAMLAP_MIN_WIN
    log(f"Лапландия: окон {n_win}{'' if stable else ' (< 16: НЕУСТОЙЧИВО, в решение не идёт)'}; "
        f"K = {K}; ячеек эталона на листе {int(ref.sum())}")
    rows = []

    def score_of(preds):
        sc = np.full(te.shape, -np.inf)
        sc[te] = np.mean([pr(Z[te]) for pr in preds], 0)          # карта = среднее трёх зёрен
        return sc

    def per_seed(preds):
        out = []
        for pr in preds:
            sc = np.full(te.shape, -np.inf)
            sc[te] = pr(Z[te])
            out.append(budget_f1(sc, te, ref, K)[2])
        return float(np.mean(out))
    res = {}
    for i, tag in enumerate(("fam3", "fam2")):
        sc = score_of(frozen[tag][FAM_NAME])
        sel = select_top(sc, te, K)
        f1, null = f1_shifted(sel, ref, te, FAMLAP_SHIFT_N, FAMLAP_SHIFT_SEED)
        assert abs(f1 - budget_f1(sc, te, ref, K)[2]) < 1e-12
        p = (1 + int((null >= f1).sum())) / (1 + FAMLAP_SHIFT_N)
        verdict = "ПРОЙДЕНО" if p <= FAMLAP_ALPHA else "НЕ ПРОЙДЕНО"
        res[tag] = (f1, p <= FAMLAP_ALPHA)
        log(f"Т{i + 1} {tag} {FAM_NAME}: F1 карты (среднее зёрен) {f1:.4f}; среднее F1 по зёрнам "
            f"{per_seed(frozen[tag][FAM_NAME]):.4f}; нуль {null.mean():.4f} (95-й проц. "
            f"{np.percentile(null, 95):.4f}, макс {null.max():.4f}); p {p:.5f} -> {verdict}")
        rows.append(("lapland", f"Т{i + 1} {tag} {FAM_NAME}", f1, null.mean(), p, verdict))
    # ---- решение для Анабара (как объявлено)
    passed = [k for k, (_, ok) in res.items() if ok]
    if not stable:
        dec = "окон < 16 — результат неустойчив, решение для Анабара не принимается"
    elif not passed:
        dec = "ни одна не прошла — модели FAM на Анабар не идут (отрицательный итог)"
    elif len(passed) == 1:
        dec = f"прошла одна — на Анабар идёт {passed[0]}"
    else:
        d = res["fam3"][0] - res["fam2"][0]
        dec = (f"прошли обе, ΔF1 (fam3 − fam2) = {d:+.4f}; "
               + ("ничья (|ΔF1| < 0,003) — FAM3" if abs(d) < 0.003
                  else f"на Анабар идёт {'fam3' if d > 0 else 'fam2'}"))
    log("РЕШЕНИЕ: " + dec)
    # ---- описательно
    log("--- описательно (без p):")
    r0 = np.random.default_rng(0)
    rnd = [budget_f1(np.where(te, r0.random(te.shape), -np.inf), te, ref, K)[2] for _ in range(5)]
    log(f"  случайный: F1 {np.mean(rnd):.4f} ± {np.std(rnd):.4f}")
    rows.append(("lapland", "случайный", np.mean(rnd), np.nan, np.nan, ""))
    for j, fname in enumerate(FEATURES):
        P, R, F1, _ = budget_f1(np.where(te, F[..., j], -np.inf), te, ref, K)
        log(f"  {fname} (без обучения): P {P:.3f} R {R:.3f} F1 {F1:.4f}")
        rows.append(("lapland", fname + " (без обучения)", F1, np.nan, np.nan, ""))
    p_old = Path("outputs/metrics/openlin_zone_lines_lapland_v3_перенос.npz")
    if p_old.exists():
        z = np.load(p_old)
        c = z["C_all"].astype(float)
        k = int(round(c.shape[0] / te.shape[0]))
        mm = block(c, k, "max")[:te.shape[0], :te.shape[1]] > 0
        P, R, F1, _ = budget_f1(np.where(mm, 1.0, 0.0) + 1e-6 * F[..., 0], te, ref, K)
        r1 = np.random.default_rng(0)
        rr = [budget_f1(np.where(mm, 1.0, 0.0) + 1e-6 * r1.random(te.shape), te, ref, K)[2]
              for _ in range(5)]
        log(f"  старый тракт (перенос TWIN): маска {c.shape} -> блок {k}; ячеек в окнах "
            f"{int((mm & te).sum())}; по тёмной полосе F1 {F1:.4f}; случайно внутри маски "
            f"F1 {np.mean(rr):.4f} ({min(rr):.4f}–{max(rr):.4f})")
        rows.append(("lapland", "старый тракт, по тёмной полосе", F1, np.nan, np.nan, ""))
        rows.append(("lapland", "старый тракт, случайно в маске", np.mean(rr), np.nan, np.nan, ""))
    else:
        log(f"  нет {p_old}")
    for tag in ("fam3", "fam2"):
        for k_, preds in frozen[tag].items():
            P, R, F1, _ = budget_f1(score_of(preds), te, ref, K)
            log(f"  {tag} {k_}: P {P:.3f} R {R:.3f} F1 карты {F1:.4f}; среднее по зёрнам "
                f"{per_seed(preds):.4f}")
            rows.append(("lapland", f"{tag} {k_}", F1, np.nan, np.nan, ""))
    with open(OUT / "famlap.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["лист", "модель", "F1", "F1_нуль_среднее", "p", "вердикт"])
        wr.writerows(rows)
    log("сохранено:", OUT / "famlap.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="utsjoki")
    ap.add_argument("--fam2", action="store_true")
    ap.add_argument("--fam3", action="store_true")
    ap.add_argument("--lapland", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.lapland:
        return main_lapland()
    if a.fam3:
        return main_fam2("fam3", FAM3_TRAIN)
    if a.fam2:
        return main_fam2()
    OUT.mkdir(parents=True, exist_ok=True)
    log = log_to(OUT / f"_fam1_{a.site}.log")
    log("FAM1; объявление audit/FAM1_DECLARATION.md")
    F, valid, meta = features(a.site, log)
    ref, lab = gtk_label(meta, valid.shape)
    west, east, n = windows(valid)
    log(f"окна: запад {len(west)}, восток {len(east)}")
    tr, te = win_mask(valid.shape, west, n) & valid, win_mask(valid.shape, east, n) & valid
    Z = quantile_norm(F, tr)
    K = int((ref & te).sum())
    log(f"ячеек эталона: обучение {int((ref & tr).sum())}, проверка K = {K}; "
        f"метка (с расширением) в обучении {lab[tr].mean() * 100:.1f} %")
    Xtr, ytr = Z[tr], lab[tr].astype(int)
    rows = []

    def score_map(pred):
        s = np.full(valid.shape, -np.inf)
        s[te] = pred(Z[te])
        return s

    rng = np.random.default_rng(0)
    rnd = [budget_f1(np.where(te, rng.random(valid.shape), -np.inf), te, ref, K)[2] for _ in range(5)]
    rows.append(("случайный выбор", "нет", float(np.mean(rnd)), float(np.std(rnd)), np.nan, np.nan))
    log(f"  случайный: F1 {np.mean(rnd):.3f}")
    for j, name in ((0, "тёмная полоса (без обучения)"), (5, "вогнутость (без обучения)"),
                    (7, "прямолинейность снимка (без обучения)"), (8, "прямолинейность рельефа (без обучения)")):
        s = np.where(te, F[..., j], -np.inf)
        P, R, F1, _ = budget_f1(s, te, ref, K)
        rows.append((name, "нет", F1, 0.0, P, R))
        log(f"  {name}: P {P:.3f} R {R:.3f} F1 {F1:.3f}")
    for name, m in old_tract(a.site, valid.shape, log).items():
        P, R, F1, k_used = budget_f1(np.where(m, 1.0, 0.0) + 1e-6 * F[..., 0], te, ref, K)
        rows.append((name, "нет", F1, 0.0, P, R))
        log(f"  {name}: ячеек в маске {int((m & te).sum())} (бюджет {K}); P {P:.3f} R {R:.3f} F1 {F1:.3f}")
    res = {}
    params_all = []
    for seed in SEEDS:
        pred, prm = fit_fam(Xtr, ytr, seed)
        params_all.append(prm)
        res.setdefault("нечёткая аддитивная (3 терма)", []).append(budget_f1(score_map(pred), te, ref, K))
        for k_, p_ in baselines(Xtr, ytr, seed).items():
            res.setdefault(k_, []).append(budget_f1(score_map(p_), te, ref, K))
        log(f"  зерно {seed} готово")
    for k_, v in res.items():
        v = np.array([x[:3] for x in v])
        rows.append((k_, "", float(v[:, 2].mean()), float(v[:, 2].std()), float(v[:, 0].mean()), float(v[:, 1].mean())))
        log(f"  {k_}: F1 {v[:, 2].mean():.3f} ± {v[:, 2].std():.3f} (P {v[:, 0].mean():.3f}, R {v[:, 1].mean():.3f})")
    # вклады термов (зерно 1): кривые вклада по признаку
    prm = params_all[0]
    log("вклады термов (зерно 1), низкий / средний / высокий:")
    order = np.argsort(prm["c"], 1)
    for j, fname in enumerate(FEATURES):
        bb = prm["b"][j][order[j]]
        log(f"  {fname}: " + " / ".join(f"{x:+.2f}" for x in bb) + f"  (размах {bb.max() - bb.min():.2f})")
    import csv
    with open(OUT / f"fam1_{a.site}.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["модель", "обучение", "F1", "F1_std", "P", "R"])
        wr.writerows(rows)
    json.dump({"west": west, "east": east, "K": K,
               "params": [{k: np.asarray(v).tolist() for k, v in p.items()} for p in params_all]},
              open(OUT / f"fam1_{a.site}.json", "w", encoding="utf-8"), ensure_ascii=False)
    log("сохранено:", OUT / f"fam1_{a.site}.csv")


if __name__ == "__main__":
    main()
