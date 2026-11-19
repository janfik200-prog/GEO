"""ANFL0b: цепочка ANFL0 с признаками самой линии на 20 м (audit/ANFL0B_DECLARATION.md). Учитель — GTK 292 (временно).

    python -m experiments.openlin_anfl0b --run        # обучение, выбор, npz/gpkg (без окон)
    python -m experiments.openlin_anfl0b --windows    # окна (после охотника)
ANFL0B_N_TESTS = 0. Лидар не используется. Только процессор (CUDA_VISIBLE_DEVICES="").
Общие части (кандидаты, метка, мера, Анфиса, вывод) — experiments/openlin_anfl.py без изменений.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from experiments import openlin_anfl as A
from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments.openlin_fam import DEM, DEM_SIG_M, FAM3_TRAIN, OUT, RES_IN, TOL_M
from experiments.openlin_iso import LOCAL_WIN_M

ANFL0B_N_TESTS = 0
TAG = "anfl0b"
FEATS = (A.FEATS[:8]
         + [f"z на линии: {x}" for x in ("тёмная полоса", "светлая полоса", "перепад снимка", "ложбина", "гребень")]
         + ["уклон GLO-30, °", "положение в рельефе (окно 2 км)", "яркость снимка (окно 2 км)", "доля без ALOS",
            "ALOS: ложбина (z)", "ALOS: гребень (z)", "до линии другого канала, м"])
QNORM = (5, 6, 7, 13, 14, 15, 17, 18, 19)       # ранговая нормировка по листу (9–13 — уже в сигмах окрестности)
FROZEN: dict = {}                                 # поправка 1 п. 7: что заморожено для ANFL1 (заполняется в main)


def zmaps(img, dem, valid):
    return [C2.local_norm_nz(C1.response(ch, img, dem)[0], valid) for ch in range(5)]


def rel_window(a, valid):
    """(значение − среднее в окне LOCAL_WIN_M) / разброс в окне — по годным ячейкам."""
    k = max(int(round(LOCAL_WIN_M / RES_IN)), 3)
    w = valid.astype("float64")
    x = np.where(valid, a, 0.0)
    den = np.maximum(uniform_filter(w, k, mode="nearest"), 1e-6)
    m = uniform_filter(x, k, mode="nearest") / den
    sq = uniform_filter(x * x, k, mode="nearest") / den
    sd = np.sqrt(np.maximum(sq - m * m, 1e-12))
    return ((a - m) / sd).astype("float32")


def line_features(site, pool, img, dem, valid, log):
    from scipy.stats import norm, rankdata
    Z = zmaps(img, dem, valid)
    gy, gx = np.gradient(dem, RES_IN)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    rel_e, rel_b = rel_window(dem, valid), rel_window(img, valid)
    raw = np.load(DEM / f"dem_alos30_{site}_лист_20m.npy").astype("float64")[:valid.shape[0], :valid.shape[1]]
    miss = ~np.isfinite(raw)
    alos = np.where(miss, np.nanmedian(raw), raw)
    za = [C2.local_norm_nz(C1.ridge_dir(alos, DEM_SIG_M, s)[0], valid) for s in (+1, -1)]
    X = np.zeros((len(pool), len(FEATS)))
    chs = np.array([p["ch"] for p in pool])
    # поправка 1 п. 5: ячейки вне valid (мосты сшивки через пропуски) не входят в средние по линии;
    # линий без годных ячеек нет — они выводятся из пула в site_data до признаков
    part = sum(not valid[p["cells"][:, 0], p["cells"][:, 1]].all() for p in pool)
    log(f"  линий с частью ячеек вне годной маски: {part} (их средние — только по годным ячейкам)")
    for ch in range(5):                                       # до ближайшей линии другого канала (как ANFL0)
        other = np.zeros(valid.shape, bool)
        for p in pool:
            if p["ch"] != ch:
                other[p["cells"][:, 0], p["cells"][:, 1]] = True
        dist = (distance_transform_edt(~other) * RES_IN).astype("float32")
        del other
        for i in np.flatnonzero(chs == ch):
            c = pool[i]["cells"]
            X[i, 19] = dist[c[:, 0], c[:, 1]].mean()
        del dist
    for i, p in enumerate(pool):
        c = p["cells"]
        c = c[valid[c[:, 0], c[:, 1]]]
        r_, c_ = c[:, 0], c[:, 1]
        X[i, :5] = np.where(np.arange(5) == p["ch"], 1.0, -1.0)
        X[i, 5] = p["mz"]
        pl = p["poly"]
        seg = np.hypot(*np.diff(pl, axis=0).T).sum() * RES_IN
        chord = float(np.hypot(*(pl[-1] - pl[0]))) * RES_IN
        X[i, 6] = np.log(max(seg, RES_IN))
        X[i, 7] = min(chord / max(seg, 1e-9), 1.0)
        for k in range(5):                      # поправка 1 п. 3: собственный канал исключён (0 — «нет сведений»)
            X[i, 8 + k] = 0.0 if k == p["ch"] else Z[k][r_, c_].mean()
        X[i, 13] = slope[r_, c_].mean()
        X[i, 14] = rel_e[r_, c_].mean()
        X[i, 15] = rel_b[r_, c_].mean()
        X[i, 16] = miss[r_, c_].mean()
        X[i, 17] = za[0][r_, c_].mean()
        X[i, 18] = za[1][r_, c_].mean()
    assert np.isfinite(X).all(), "в признаках есть NaN/inf"
    log(f"  признаки {site}: {X.shape[0]} линий × {X.shape[1]}; различных значений «z на линии: ложбина» "
        f"{len(np.unique(np.round(X[:, 11], 6)))}, «уклон» {len(np.unique(np.round(X[:, 13], 6)))} (у FAM 100 м на Утсйоки было 13 804 / 17 301)")
    for k in range(len(FEATS)):
        q = np.quantile(X[:, k], [0.1, 0.5, 0.9])
        log(f"    «{FEATS[k]}» 10/50/90 %: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    for k in QNORM:
        X[:, k] = norm.ppf((rankdata(X[:, k], method="average") - 0.5) / len(X))
    return X


def site_data(site, log, teacher=True, keep_img=False):
    pool, img, dem, valid, meta = A.candidates(site, log)
    n0 = len(pool)
    pool = [p for p in pool if valid[p["cells"][:, 0], p["cells"][:, 1]].any()]
    log(f"  линий без годных ячеек выведено из пула: {n0 - len(pool)}")
    import hashlib
    h = hashlib.sha256()                                      # sha пула как в A.candidates (поправка 1 п. 7)
    for p in pool:
        h.update(np.array([p["ch"]], "int32").tobytes())
        h.update(np.ascontiguousarray(p["cells"], "int32").tobytes())
    FROZEN.setdefault("пул CAND3", {})[site] = h.hexdigest()[:16]
    X = line_features(site, pool, img, dem, valid, log)
    d = {"pool": pool, "X": X, "valid": valid, "meta": meta,
         "img": img if keep_img else None, "dem": dem if keep_img else None}
    del img, dem
    if teacher:
        y, inside, frac, ref = A.gtk_teacher(site, pool, valid, meta)
        d.update(y=y, inside=inside, frac=frac, ref=ref,
                 dref=(distance_transform_edt(~ref) * RES_IN).astype("float32"))
        log(f"  {site}: учитель GTK 292, t = {TOL_M:.0f} м: «да» {int(y.sum())} из {len(y)} ({100 * y.mean():.1f} %)")
    return d


def counts(d, sel):
    """Для склеенного F1: (выбранных в t, выбранных, эталона в t от выбранных, эталона), по ячейкам."""
    m = A.sel_mask(d["pool"], sel, d["valid"].shape)
    if not m.any():
        return 0, 0, 0, int(d["ref"].sum())
    ds = distance_transform_edt(~m) * RES_IN
    return (int((d["dref"][m] <= TOL_M).sum()), int(m.sum()), int((ds[d["ref"]] <= TOL_M).sum()), int(d["ref"].sum()))


def pooled_f1(cs):
    a, b, c, e = (sum(x[i] for x in cs) for i in range(4))
    P, R = (a / b if b else 0.0), (c / e if e else 0.0)
    return (2 * P * R / (P + R) if P + R else 0.0), P, R


def km_of(d, sel):
    return A.sel_mask(d["pool"], sel, d["valid"].shape).sum() * RES_IN / 1000


def run(log):
    from scipy.stats import spearmanr
    log("=== обучение: " + ", ".join(FAM3_TRAIN))
    tr = {s: site_data(s, log) for s in FAM3_TRAIN}
    for s in FAM3_TRAIN:
        tr[s]["ok"] = tr[s]["inside"] & (tr[s]["y"] >= 0)
    model, params = A.train(np.concatenate([tr[s]["X"][tr[s]["ok"]] for s in FAM3_TRAIN]),
                            np.concatenate([tr[s]["y"][tr[s]["ok"]] for s in FAM3_TRAIN]), log)
    # поправка 1 п. 4: описательная модель без «уклона», «положения в рельефе» и «z на линии: ложбина»
    ABL = (11, 13, 14)
    def drop(X):
        Y = X.copy()
        Y[:, ABL] = 0.0
        return Y
    model_abl, _pa = A.train(drop(np.concatenate([tr[s]["X"][tr[s]["ok"]] for s in FAM3_TRAIN])),
                             np.concatenate([tr[s]["y"][tr[s]["ok"]] for s in FAM3_TRAIN]), lambda *a: None)
    log(f"  описательно: та же модель без признаков {[FEATS[k] for k in ABL]} (обнулены)")
    sc_tr = {}
    for s in FAM3_TRAIN:
        rest = [q for q in FAM3_TRAIN if q != s]
        m_s, _p = A.train(np.concatenate([tr[q]["X"][tr[q]["ok"]] for q in rest]),
                          np.concatenate([tr[q]["y"][tr[q]["ok"]] for q in rest]), lambda *a: None)
        sc_tr[s] = m_s(tr[s]["X"])
    grid = np.quantile(np.concatenate(list(sc_tr.values())), np.linspace(0.02, 0.98, A.N_THR))
    best = (-1.0, None, 0.0, 0.0)
    for thr in grid:
        f, P, R = pooled_f1([counts(tr[s], sc_tr[s] >= thr) for s in FAM3_TRAIN])
        if f > best[0]:
            best = (f, thr, P, R)
    f_tr, thr, P_tr, R_tr = best
    km_sel = sum(km_of(tr[s], sc_tr[s] >= thr) for s in FAM3_TRAIN)
    km_all = sum(km_of(tr[s], np.ones(len(tr[s]["pool"]), bool)) for s in FAM3_TRAIN)
    share = km_sel / km_all
    log(f"  порог по СКЛЕЕННЫМ отложенным оценкам («лист за листом»): {thr:.3f}; F1 отложенных {f_tr:.3f} "
        f"(P {P_tr:.3f}, R {R_tr:.3f}); доля длины выбора {100 * share:.1f} % — переносится на новые листы")
    best_f = (-1.0, None, None)
    for k in range(len(FEATS)):
        for sg in (+1, -1):
            cs = []
            for s in FAM3_TRAIN:
                kmx = km_of(tr[s], np.ones(len(tr[s]["pool"]), bool)) * share
                cs.append(counts(tr[s], A.top_by_km(sg * tr[s]["X"][:, k], tr[s]["pool"], kmx)))
            f = pooled_f1(cs)[0]
            if f > best_f[0]:
                best_f = (f, k, sg)
    log(f"  лучший одиночный признак на обучении (та же доля длины): «{FEATS[best_f[1]]}» "
        f"({'+' if best_f[2] > 0 else '−'}), F1 {best_f[0]:.3f}")
    A_FEATS = A.FEATS
    A.FEATS = FEATS
    try:
        log("=== правила Анфисы\n" + A.rules_text(params))
    finally:
        A.FEATS = A_FEATS
    del tr
    res = {"учитель": A.TEACHER, "порог": float(thr), "доля длины": float(share)}
    for site in ("utsjoki", "anabar"):
        log(f"=== {site}")
        d = site_data(site, log, teacher=(site == "utsjoki"))
        sc = model(d["X"])
        km_pool = km_of(d, np.ones(len(sc), bool))
        sel = A.top_by_km(sc, d["pool"], share * km_pool)
        km = km_of(d, sel)
        area = d["valid"].sum() * RES_IN ** 2 / 1e6
        chs = np.array([p["ch"] for p in d["pool"]])
        rho_e = spearmanr(sc, d["X"][:, 14])[0]
        rho_v = spearmanr(sc, d["X"][:, 11])[0]
        log(f"  выбрано {int(sel.sum())} из {len(sel)} линий, {km:.0f} км ({100 * km / km_pool:.1f} % пула), "
            f"{km / area:.2f} км/км²; по каналам: " + ", ".join(f"{C1.CHANNELS[ch]} {int((sel & (chs == ch)).sum())}"
                                                              for ch in range(5)))
        log(f"  тяга к долинам: корреляция оценки (Спирмен) с положением в рельефе {rho_e:+.2f}, с ложбиной на линии "
            f"{rho_v:+.2f}")
        np.savez_compressed(OUT / f"{TAG}_{site}.npz", score=sc, sel=sel, share=share, thr=thr, teacher=A.TEACHER,
                            ch=chs, X=d["X"])
        A.export_gpkg(OUT / f"{TAG}_{site}.gpkg", d["pool"], sel, sc, d["meta"])
        log(f"  {OUT / f'{TAG}_{site}.gpkg'}")
        r = {"км": km, "корр. с рельефом": rho_e, "корр. с ложбиной": rho_v}
        if site == "utsjoki":
            cs = counts(d, sel)
            F, P, R = pooled_f1([cs])
            log(f"  Анфиса: F1 {F:.3f} (P {P:.3f}, R {R:.3f}) при {km:.0f} км")
            rows = [pooled_f1([counts(d, A.top_by_km(np.random.default_rng(A.RAND_SEED + rs).random(len(sel)),
                                                     d["pool"], km))])[0] for rs in range(A.N_RAND)]
            log(f"  случайные кандидаты при той же длине: F1 {np.mean(rows):.3f} [мин {np.min(rows):.3f}, макс {np.max(rows):.3f}]")
            s1 = A.top_by_km(best_f[2] * d["X"][:, best_f[1]], d["pool"], km)
            F1_, P1, R1 = pooled_f1([counts(d, s1)])
            log(f"  лучший одиночный признак при той же длине: F1 {F1_:.3f} (P {P1:.3f}, R {R1:.3f}) при {km_of(d, s1):.0f} км")
            Fa, Pa, Ra = pooled_f1([counts(d, np.ones(len(sel), bool))])
            log(f"  все кандидаты: F1 {Fa:.3f} (P {Pa:.3f}, R {Ra:.3f}) при {km_pool:.0f} км")
            ref = d["ref"]
            l6 = np.load(OUT / "linb6_utsjoki.npz")["ids"][:ref.shape[0], :ref.shape[1]] > 0
            km6 = l6.sum() * RES_IN / 1000
            ds6 = distance_transform_edt(~l6) * RES_IN
            P6, R6 = float(np.mean(d["dref"][l6] <= TOL_M)), float(np.mean(ds6[ref] <= TOL_M))
            Fq, Pq, Rq = pooled_f1([counts(d, A.top_by_km(sc, d["pool"], km6))])
            log(f"  LINB6: F1 {2 * P6 * R6 / (P6 + R6):.3f} (P {P6:.3f}, R {R6:.3f}) при {km6:.0f} км; "
                f"Анфиса при той же длине: F1 {Fq:.3f} (P {Pq:.3f}, R {Rq:.3f})")
            r.update(F1=F, P=P, R=R, случайные=float(np.mean(rows)), одиночный=F1_, все=Fa,
                     LINB6=2 * P6 * R6 / (P6 + R6), анфиса_при_LINB6=Fq)
        sc_a = model_abl(drop(d["X"]))
        sel_a = A.top_by_km(sc_a, d["pool"], share * km_pool)
        txt = ""
        if site == "utsjoki":
            Fb, Pb, Rb = pooled_f1([counts(d, sel_a)])
            txt = f"F1 {Fb:.3f} (P {Pb:.3f}, R {Rb:.3f}); "
            r["без_рельефных"] = Fb
        log(f"  описательно, без рельефных признаков: {txt}корреляция с положением в рельефе "
            f"{spearmanr(sc_a, d['X'][:, 14])[0]:+.2f}, с ложбиной {spearmanr(sc_a, d['X'][:, 11])[0]:+.2f}; "
            f"общих линий с основным выбором {100 * (sel & sel_a).sum() / max(sel.sum(), 1):.0f} %")
        log(f"  для сравнения ANFL0: корреляция с вогнутостью 0,54 (Утсйоки) / 0,64 (Анабар)")
        res[site] = r
        del d
    res["frozen"] = FROZEN
    (OUT / f"{TAG}_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                          encoding="utf-8")


def windows_step(log):
    """Окна после охотника: кандидаты пересобираются (детерминированно), выбор — из npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for site, win_km in (("utsjoki", 6), ("anabar", 8)):
        pool, img, dem, valid, _meta = A.candidates(site, log)
        pool = [p for p in pool if valid[p["cells"][:, 0], p["cells"][:, 1]].any()]   # как site_data
        sel = np.load(OUT / f"{TAG}_{site}.npz")["sel"]
        assert len(sel) == len(pool), "пул не совпал с прогоном"
        dst = OUT / f"{TAG}_{site}"
        dst.mkdir(exist_ok=True)
        gy, gx = np.gradient(dem, RES_IN)
        az = np.radians(315)
        shade = np.clip(0.6 + 0.4 * (-(gx * np.sin(az) - gy * np.cos(az)) / np.maximum(np.hypot(gx, gy), 1e-3)
                                     * np.tanh(np.hypot(gx, gy) * 10)), 0, 1)
        H, Wd = img.shape
        wp = int(win_km * 1000 / RES_IN)
        k = 0
        for i in range(3):
            for j in range(3):
                k += 1
                r0, c0 = int((i + 0.5) * H / 3 - wp / 2), int((j + 0.5) * Wd / 3 - wp / 2)
                fig, axs = plt.subplots(1, 3, figsize=(27, 9.4))
                for ax, (t, a, show) in zip(axs, (("зимний S2, без линий", img, False), ("S2 + линии", img, True),
                                                  ("рельеф GLO-30 + линии", shade, True))):
                    b = a[r0:r0 + wp, c0:c0 + wp]
                    lo, hi = np.nanpercentile(b, [2, 98])
                    ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=(0, win_km, win_km, 0))
                    ax.set_title(t, fontsize=14)
                    if show:
                        for flag, col, ms in ((False, "#9bd", 0.8), (True, "red", 1.8)):
                            for idx in np.flatnonzero(sel == flag):
                                y = pool[idx]["cells"] - [r0, c0]
                                m = (y[:, 0] >= 0) & (y[:, 0] < wp) & (y[:, 1] >= 0) & (y[:, 1] < wp)
                                if m.any():
                                    y = y[m] * RES_IN / 1000
                                    ax.plot(y[:, 1], y[:, 0], ".", ms=ms, color=col)
                    ax.set_xlim(0, win_km)
                    ax.set_ylim(win_km, 0)
                fig.suptitle(f"ANFL0b: Анфиса, учитель GTK 292 (временно); {site}, окно {k} ({win_km}×{win_km} км), "
                             f"строка {r0}, столбец {c0}. Красные — выбрала Анфиса, голубые — отброшенные кандидаты",
                             fontsize=14)
                fig.tight_layout()
                p = dst / f"окно_{k}.png"
                fig.savefig(p, dpi=70)
                plt.close(fig)
                log(f"  {p}")


def main():
    from experiments.openlin_fam import __file__ as fam_file
    from experiments.openlin_fam import log_to
    from experiments.openlin_len1 import sha
    head = (f"sha256 кода {sha(__file__)}, ANFL {sha(A.__file__)}, FAM {sha(fam_file)}, CAND3 "
            f"{sha(A.C3.__file__)}, CAND2 {sha(C2.__file__)}, CAND1 {sha(C1.__file__)}")
    if "--run" in sys.argv:
        log = log_to(OUT / f"_{TAG}.log")
        log(f"ANFL0b; audit/ANFL0B_DECLARATION.md; {head}")
        from experiments import openlin_iso
        from experiments.openlin_transfer_check import COMP
        FROZEN.update({"код": {n: sha(f) for n, f in (("anfl0b", __file__), ("anfl", A.__file__), ("fam", fam_file),
                                                      ("cand1", C1.__file__), ("cand2", C2.__file__),
                                                      ("cand3", A.C3.__file__), ("iso", openlin_iso.__file__))},
                       "входы": {s: {"S2": sha(COMP / f"winter_{s}_20m_лист.npz"),
                                     "GLO-30": sha(DEM / f"dem_glo30_{s}_лист_20m.npy"),
                                     "ALOS": sha(DEM / f"dem_alos30_{s}_лист_20m.npy"),
                                     "FAM feat (не используется в ANFL0b)": sha(OUT / f"feat_{s}.npz")}
                                 for s in FAM3_TRAIN + ("utsjoki", "anabar")},
                       "признаки": FEATS, "ранговая нормировка": [k + 1 for k in QNORM],
                       "метка": f">= {A.LABEL_FRAC:.0%} ячеек в t от линий учителя, lo=None; GTK t={TOL_M:.0f} м",
                       "модель": f"FAM {A.N_TERMS} терма, LR {A.LR}, эпох {A.EPOCHS}, зёрна {A.SEEDS}",
                       "порог": f"лист за листом, склеенный F1, сетка {A.N_THR}; перенос по доле длины",
                       "участок": "весь годный лист"})
        log("  frozen: " + json.dumps(FROZEN, ensure_ascii=False))
        assert len(FEATS) == 20
        A.self_test()
        log("  инварианты: 20 признаков, метка на синтетике — ок")
        log("  признаки (порядок): " + "; ".join(f"{i + 1} {f}" for i, f in enumerate(FEATS)))
        run(log)
    elif "--windows" in sys.argv:
        log = log_to(OUT / f"_{TAG}_windows.log")
        log(f"ANFL0b окна; {head}")
        windows_step(log)


if __name__ == "__main__":
    main()
