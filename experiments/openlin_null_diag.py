"""Пакет 08 ТЗ К1: что сохраняет двойник тракта, нуль строже (IAAFT по участкам), врезки.

Объявление — ``audit/PACKET08_DECLARATION.md`` (26.10.2026, до расчёта).

* диагностика: снимок против двойника тракта (``openlin_kora.surrogate`` —
  IAAFT, 8 итераций, зерно 900): гистограмма, радиальный спектр, спектр
  направлений, неоднородность (локальное ст. откл. в окне 2 км), окна рядом;
* IAAFT по участкам 20 x 20 км (зёрна 1900-1902) против глобального (900-902):
  мусор этапа подтверждения рабочего тракта, без линий у границ участков;
* тепловая карта врезок «азимут x амплитуда» (после правки — сырые исходы,
  до правки — таблица лога, отчётный агрегат).

Запуск из корня:
    python -m experiments.openlin_null_diag --site anabar
    python -m experiments.openlin_null_diag --site lapland
    python -m experiments.openlin_null_diag --врезки
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
OUT = Path("outputs/k1/null_validation")
FIG = Path("outputs/k1/figures")

# ---- объявление (audit/PACKET08_DECLARATION.md) ----------------------------
SEEDS_GLOBAL = (900, 901, 902)       # как в рабочем прогоне
SEEDS_BLOCK = (1900, 1901, 1902)
BLOCK = 1000                         # участок 20 км на сетке 20 м
EDGE_TOL_M, EDGE_FRAC = 60.0, 0.30   # «лежит вдоль» границы участка
LOCAL_WIN = 100                      # 2 км = DETREND_M тракта
N_RINGS, N_SECT = 20, 12
JUNK_MAX = 0.20
# ----------------------------------------------------------------------------


def context(site, log):
    """Всё, что тракт берёт с настоящего листа и не меняет для подделок."""
    from experiments import openlin_zone_lines_v3 as Z
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res = float(w["res_m"])
    opt = w["shade"].astype("float32")
    halves = Z.build_halves(site, log, "B04")
    valid = np.isfinite(opt) & np.isfinite(halves["А"]) & np.isfinite(halves["Б"])
    k = int(round(100 / res))
    zone, strike = Z.zone_and_strike(site, k)
    zone = zone[:valid.shape[0], :valid.shape[1]] & valid
    strike = strike[:valid.shape[0], :valid.shape[1]]
    up = lambda x: np.repeat(np.repeat(x, k, 0), k, 1)[:valid.shape[0], :valid.shape[1]]
    zone = up(Z.close_along_strike(zone[::k, ::k], strike[::k, ::k], 100.0, log)) & valid
    cnt = w["count"][:valid.shape[0], :valid.shape[1]]
    seam = np.zeros(valid.shape, bool)
    seam[:, :-1] |= cnt[:, :-1] != cnt[:, 1:]
    seam[:-1, :] |= cnt[:-1, :] != cnt[1:, :]
    seam |= ~valid
    return dict(Z=Z, res=res, opt=opt, hA=halves["А"], hB=halves["Б"], valid=valid,
                zone=zone, strike=strike, seam=seam)


def tract(o, ha, hb, c):
    """Рабочий тракт (TWIN, цепочки) — те же вызовы, что в openlin_zone_lines_v3.main.
    Возвращает наборы А и В в зонах (gate) и по листу."""
    from scipy.ndimage import distance_transform_edt as edt
    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize
    Z, res, valid = c["Z"], c["res"], c["valid"]
    xx = np.where(valid, standardize(o, valid), 0.0).astype("float32")
    zz = local_norm(np.where(valid, ridge_response(xx), 0.0), valid, res)
    raw = Z.trace(o, valid, res, local=True, min_len_m=0)
    raw = Z.extend_ends(raw, zz, valid, res, None)
    L = Z.link_same_side(raw, c["zone"], c["strike"], res, None, z=zz, support="mean",
                         offset_m=Z.TWIN_OFFSET_M)
    L = [x for x in L if x["длина_км"] * 1000 >= Z.MIN_LEN_M]
    L = Z.drop_seam_lines(L, c["seam"], res)
    W_ = valid.shape[1]
    for x in L:
        kul = x.get("кулисы")
        if kul is not None and len(kul):
            x["pts"] = x["pts"][~np.isin(x["pts"][:, 0] * W_ + x["pts"][:, 1], kul[:, 0] * W_ + kul[:, 1])]
    dts = [edt(~Z.mask_of(Z.trace(h, valid, res, local=True), valid.shape), sampling=(res, res))
           for h in (ha, hb)]
    A, C = Z.select2(L, c["zone"], c["strike"], dts, gate=True)
    Aw, Cw = Z.select2(L, c["zone"], c["strike"], dts, gate=False)
    return {"A": A, "C": C, "A_all": Aw, "C_all": Cw}


def blockwise(stack, valid, rng):
    """IAAFT отдельно в каждом участке BLOCK x BLOCK (неполные у края — тоже)."""
    from experiments.openlin_kora import surrogate
    out = np.zeros_like(stack)
    H, W = valid.shape
    for r0 in range(0, H, BLOCK):
        for c0 in range(0, W, BLOCK):
            sl = (slice(r0, min(r0 + BLOCK, H)), slice(c0, min(c0 + BLOCK, W)))
            v = valid[sl]
            if v.sum() < 100:
                continue
            out[:, sl[0], sl[1]] = surrogate(stack[:, sl[0], sl[1]], v, rng)
    return out


def edge_mask(shape, res):
    from scipy.ndimage import distance_transform_edt as edt
    H, W = shape
    b = np.zeros(shape, bool)
    b[np.arange(BLOCK, H, BLOCK), :] = True
    b[:, np.arange(BLOCK, W, BLOCK)] = True
    return edt(~b, sampling=res) <= EDGE_TOL_M


def keep(L, near):
    return [x for x in L if near[x["pts"][:, 0], x["pts"][:, 1]].mean() < EDGE_FRAC]


def diagnostics(site, c, fake, log):
    from scipy.ndimage import uniform_filter
    from scipy.stats import ks_2samp
    v = c["valid"]
    x, y = c["opt"], fake
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(v.ravel()), 200_000, replace=False)
    ks = ks_2samp(x.ravel()[idx], y.ravel()[idx]).statistic

    def spec(a):
        a = np.where(v, a, np.median(a[v])) - np.median(a[v])
        F = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
        H, W = a.shape
        fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(H)), np.fft.fftshift(np.fft.fftfreq(W)), indexing="ij")
        fr = np.hypot(fx, fy)
        ang = (np.degrees(np.arctan2(fy, fx)) % 180)
        edges = np.geomspace(2 / max(H, W), 0.5, N_RINGS + 1)
        rad = np.array([F[(fr >= edges[i]) & (fr < edges[i + 1])].mean() for i in range(N_RINGS)])
        m = (fr >= edges[0]) & (fr < 0.5)
        sect = np.array([F[m & (ang >= s * 180 / N_SECT) & (ang < (s + 1) * 180 / N_SECT)].sum()
                         for s in range(N_SECT)])
        return rad, sect / sect.sum(), edges

    rx, sx, edges = spec(x)
    ry, sy, _ = spec(y)

    def localsd(a):
        a = np.where(v, a, 0.0).astype("float64")
        n = uniform_filter(v.astype("float64"), LOCAL_WIN)
        m1 = uniform_filter(a, LOCAL_WIN) / np.maximum(n, 1e-9)
        m2 = uniform_filter(a * a, LOCAL_WIN) / np.maximum(n, 1e-9)
        return np.sqrt(np.maximum(m2 - m1 ** 2, 0))

    lx, ly = localsd(x), localsd(y)
    cvx, cvy = float(lx[v].std() / lx[v].mean()), float(ly[v].std() / ly[v].mean())
    d = {"KS_гистограмма": float(ks), "спектр_отношение_по_кольцам": (ry / rx).round(3).tolist(),
         "кольца_частоты": edges.round(5).tolist(),
         "направления_снимок": sx.round(4).tolist(), "направления_двойник": sy.round(4).tolist(),
         "направления_макс_разность": float(np.abs(sx - sy).max()),
         "неоднородность_CV_лок_ст_откл": {"снимок": cvx, "двойник": cvy}}
    log(f"  диагностика: KS гистограмм {ks:.4f}; спектр двойник/снимок по кольцам "
        f"{np.min(ry / rx):.3f}-{np.max(ry / rx):.3f}; спектр направлений, наибольшая разность доли "
        f"сектора {np.abs(sx - sy).max():.4f}; неоднородность (CV лок. ст. откл. 2 км): снимок "
        f"{cvx:.3f}, двойник {cvy:.3f}")
    return d, lx, ly


def site_run(site, log):
    from experiments.openlin_kora import surrogate
    t0 = time.time()
    c = context(site, log)
    v = c["valid"]
    stack = np.nan_to_num(np.stack([np.where(v, a, np.nan) for a in (c["opt"], c["hA"], c["hB"])]))
    g0 = surrogate(stack, v, np.random.default_rng(SEEDS_GLOBAL[0]))
    diag, lx, ly = diagnostics(site, c, g0[0], log)
    rb = blockwise(stack, v, np.random.default_rng(SEEDS_BLOCK[0]))
    _d, _lx, lb = diagnostics(site, c, rb[0], lambda *_: None)
    diag["неоднородность_CV_лок_ст_откл"]["двойник_по_участкам"] = float(lb[v].std() / lb[v].mean())
    log(f"  неоднородность у двойника по участкам: {diag['неоднородность_CV_лок_ст_откл']['двойник_по_участкам']:.3f}")
    plot_windows(site, c, g0[0], rb[0], log)
    plot_localsd(site, lx, ly, lb, v, log)

    near = edge_mask(v.shape, c["res"])
    real = {k: keep(L, near) for k, L in tract(c["opt"], c["hA"], c["hB"], c).items()}
    log(f"  снимок (без линий у границ участков): зоны А {len(real['A'])} / В {len(real['C'])}, "
        f"лист А {len(real['A_all'])} / В {len(real['C_all'])} ({time.time() - t0:.0f} с)")
    res = {"диагностика": diag, "снимок": {k: len(L) for k, L in real.items()}}
    for kind, seeds in (("глобальный", SEEDS_GLOBAL), ("по участкам", SEEDS_BLOCK)):
        sums = {"A": 0, "C": 0, "A_all": 0, "C_all": 0}
        for s in seeds:
            sg = (surrogate(stack, v, np.random.default_rng(s)) if kind == "глобальный"
                  else blockwise(stack, v, np.random.default_rng(s)))
            f = {k: keep(L, near) for k, L in tract(sg[0], sg[1], sg[2], c).items()}
            for k in sums:
                sums[k] += len(f[k])
            log(f"    {kind}, зерно {s}: зоны А {len(f['A'])} / В {len(f['C'])}, лист А {len(f['A_all'])} / "
                f"В {len(f['C_all'])} ({time.time() - t0:.0f} с)")
        jz = (sums["C"] / max(sums["A"], 1)) / max(len(real["C"]) / max(len(real["A"]), 1), 1e-9)
        jw = (sums["C_all"] / max(sums["A_all"], 1)) / max(len(real["C_all"]) / max(len(real["A_all"]), 1), 1e-9)
        res[kind] = {"суммы": sums, "мусор_зоны": jz, "мусор_лист": jw,
                     "линий_до_подтверждения_лист_на_подделку": sums["A_all"] / len(seeds)}
        log(f"  {kind}: мусор этапа подтверждения — в зонах {100 * jz:.1f} %, весь лист {100 * jw:.1f} % "
            f"(порог {100 * JUNK_MAX:.0f} %: {'пройдено' if max(jz, jw) <= JUNK_MAX else 'НЕ пройдено'}); "
            f"линий подделки до подтверждения {sums['A_all'] / len(seeds):.0f} против {len(real['A_all'])} у снимка")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{site}_null_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    log(f"сохранено: {OUT / f'{site}_null_validation.json'}")


def _img(a, v):
    lo, hi = np.percentile(a[v], [2, 98])
    return np.clip((np.where(v, a, lo) - lo) / (hi - lo), 0, 1)


def plot_windows(site, c, g, b, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader(open("outputs/k1/figures/selection_manifest.csv", encoding="utf-8-sig"))
            if r["site"] == site]
    v = c["valid"]
    ims = [_img(c["opt"], v), _img(g, v), _img(b, v)]
    fig, axs = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows)), facecolor="white")
    axs = np.atleast_2d(axs)
    for i, r in enumerate(rows):
        rr, cc = int(r["row"]), int(r["col"])
        sl = (slice(rr - 125, rr + 125), slice(cc - 125, cc + 125))
        for j, (im, t) in enumerate(zip(ims, ("снимок (shade)", "двойник тракта: IAAFT по листу",
                                                "IAAFT по участкам 20 км"))):
            axs[i, j].imshow(im[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            axs[i, j].set_title(f"окно {r['window']}: {t}", loc="left", fontsize=10)
    fig.suptitle(f"{site}: снимок и подделки рядом, 5 x 5 км, без результатов алгоритма", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"surrogate_windows_{site}.png", dpi=90)
    plt.close(fig)
    log(f"сохранено: {FIG / f'surrogate_windows_{site}.png'}")


def plot_localsd(site, lx, ly, lb, v, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="white")
    hi = np.percentile(lx[v], 99)
    for a_, m, t in zip(axs, (lx, ly, lb), ("снимок", "IAAFT по листу", "IAAFT по участкам")):
        im = a_.imshow(np.where(v, m, np.nan)[::5, ::5], vmin=0, vmax=hi, cmap="magma")
        a_.set_title(f"{t}: CV {m[v].std() / m[v].mean():.3f}", loc="left")
        a_.set_xticks([]); a_.set_yticks([])
    fig.colorbar(im, ax=axs, fraction=0.02)
    fig.suptitle(f"{site}: локальное ст. откл. в окне 2 км — неоднородность снимка по площади", x=0.01, ha="left")
    fig.savefig(FIG / f"surrogate_nonstationarity_{site}.png", dpi=110)
    plt.close(fig)
    log(f"сохранено: {FIG / f'surrogate_nonstationarity_{site}.png'}")


def injections(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open("audit/injections_raw.csv", encoding="utf-8-sig")))
    az = sorted({float(r["azimuth_deg"]) for r in rows})
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        amps = sorted({float(r["amplitude_sigma"]) for r in rows if r["mode"] == mode})
        Mx = np.array([[np.mean([int(r["hit"]) for r in rows if r["mode"] == mode and
                                 float(r["amplitude_sigma"]) == a and float(r["azimuth_deg"]) == z])
                        for z in az] for a in amps])
        out[f"после правки, {mode}"] = (amps, Mx)
    # до правки — таблица лога (отчётный агрегат, сырых исходов нет)
    txt = Path("outputs/metrics/_azimuth_response2.log").read_text(encoding="utf-8")
    for mode in ("контраст", "высота"):
        blk = txt.split(f"режим «{mode}»  (")[1].split("\n\n")[0].splitlines()[2:]
        amps, rows_ = [], []
        for line in blk:
            p = line.split()
            if len(p) == len(az) + 1 and re.match(r"^\d", p[0]):
                amps.append(float(p[0])); rows_.append([float(q) / 100 for q in p[1:]])
        out[f"до правки (агрегат лога), {mode}"] = (amps, np.array(rows_))
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), facecolor="white")
    for a_, (name, (amps, Mx)) in zip(axs.ravel(), sorted(out.items(), key=lambda kv: kv[0][::-1])):
        im = a_.imshow(Mx, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
        a_.set_xticks(range(len(az))); a_.set_xticklabels([f"{z:.0f}" for z in az], fontsize=8)
        a_.set_yticks(range(len(amps))); a_.set_yticklabels([f"{x:g}" for x in amps])
        a_.set_xlabel("азимут врезки, °"); a_.set_ylabel("амплитуда, σ")
        a_.set_title(name, loc="left", fontsize=10)
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                a_.text(j, i, f"{100 * Mx[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if Mx[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=axs, fraction=0.02, label="доля найденных (8 врезок на клетку)")
    fig.suptitle("Анабар: врезки 2 км, 40 м; найдено при ≥ 50 % следа в 60 м и 20°; солнце 184°",
                 x=0.01, ha="left")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "injection_azimuth.png", dpi=130)
    plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "injection_heatmaps.json").write_text(json.dumps(
        {k: {"амплитуды": a, "азимуты": az, "доля": m.round(3).tolist()} for k, (a, m) in out.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"сохранено: {FIG / 'injection_azimuth.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=None)
    ap.add_argument("--врезки", dest="inj", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    if a.inj:
        injections(log)
    if a.site:
        site_run(a.site, log)


if __name__ == "__main__":
    main()
