"""ANFL: Анфиса (нечёткая аддитивная модель FAM) выбирает линии из кандидатов CAND3.

ANFL0 (audit/ANFL0_DECLARATION.md) — вся цепочка с временным учителем GTK 292:
    python -m experiments.openlin_anfl --anfl0
Когда придёт разметка геологов, меняется только источник меток (labels_from_lines) — остальное то же.
ANFL0_N_TESTS = 0. Лидар не используется. Только процессор (CUDA_VISIBLE_DEVICES="").
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments import openlin_cand3 as C3
from experiments.openlin_fam import (DEM, DEM_SIG_M, EPOCHS, FAM3_TRAIN, LR, N_TERMS, OUT, RES_IN, SEEDS,
                                     TOL_M, all_windows, fit_fam, win_mask)

ANFL0_N_TESTS = 0
LABEL_FRAC = 0.5                      # «да» — >= 50 % ячеек в t (правило LINB6/CAND3)
N_RAND, RAND_SEED = 20, 1900          # случайные кандидаты при равной длине (как CAND1–3)
N_THR = 50                            # сетка порогов для F1 на обучении — разрешение поиска, не порог
TEACHER = "GTK 292 (временно)"        # подпись каждой карты и векторного файла ANFL0
FEATS = (["канал: тёмная полоса", "канал: светлая полоса", "канал: перепад снимка", "канал: ложбина",
          "канал: гребень", "сила (z)", "log длины", "прямизна"]
         + [f"FAM: {x}" for x in ("тёмная полоса", "светлая полоса", "градиент снимка", "разброс яркости",
                                  "уклон", "вогнутость", "выпуклость", "прямолинейность снимка",
                                  "прямолинейность рельефа")]
         + ["ALOS: ложбина (z)", "ALOS: гребень (z)", "до линии другого канала, м"])
QNORM = (5, 6, 7, 17, 18, 19)         # признаки с квантильной нормировкой по листу (FAM — уже нормированы)


# ---------------------------------------------------------------- кандидаты (CAND3, с ломаными)
def candidates(site, log):
    """CAND3 как принят: трассировка CAND2 + отбор без дублей. Возвращает линии пула с ломаными."""
    from experiments.openlin_anfisa_lines import rasterize
    img, dem, valid, meta = C1.load_inputs(site)
    chans, polys = {}, {}
    for ch in range(len(C1.CHANNELS)):
        r, th = C1.response(ch, img, dem)
        z = C2.local_norm_nz(r, valid)
        lines, km = C2.trace_z(z, th, valid, lambda *a: None, "снимок")
        if site == "utsjoki":
            assert (len(lines), round(km)) == C3.CAND2_UTS[ch], "трассировка не совпала с CAND2/CAND3"
        ids = rasterize(lines, valid.shape, 1)
        n = int(ids.max())
        cells = np.bincount(ids.ravel(), minlength=n + 1)
        zsum = np.bincount(ids.ravel(), weights=z.ravel().astype("float64"), minlength=n + 1)
        chans[ch] = {"ids": ids, "cells": cells, "mz": zsum / np.maximum(cells, 1), "n": n, "km": km}
        polys[ch] = lines
    taken, keep, cl = C3.dedupe(chans, valid.shape, log)
    pool = []
    for ch, j in taken:
        rr, cc, start = cl[ch]
        a, b = start[j - 1], start[j]
        pool.append({"ch": ch, "cells": np.stack([rr[a:b], cc[a:b]], 1), "poly": polys[ch][j - 1],
                     "mz": float(chans[ch]["mz"][j])})
    km = sum(len(p["cells"]) for p in pool) * RES_IN / 1000
    area = valid.sum() * RES_IN ** 2 / 1e6
    import hashlib
    h = hashlib.sha256()
    for p in pool:
        h.update(np.array([p["ch"]], "int32").tobytes())
        h.update(np.ascontiguousarray(p["cells"], "int32").tobytes())
    log(f"  {site}: кандидатов CAND3 {len(pool)}, {km:.0f} км (растр, сумма), {km / area:.2f} км/км²; по каналам: "
        + ", ".join(f"{C1.CHANNELS[ch]} {sum(p['ch'] == ch for p in pool)}" for ch in range(5))
        + f"; sha пула {h.hexdigest()[:16]}")
    return pool, img, dem, valid, meta


# ---------------------------------------------------------------- признаки линии
def line_features(site, pool, valid, log):
    from experiments.openlin_fam import features, quantile_norm
    F, v100, _meta = features(site, log)
    wins, n = all_windows(v100)
    Z = quantile_norm(F, win_mask(v100.shape, wins, n) & v100)
    alos = np.load(DEM / f"dem_alos30_{site}_лист_20m.npy").astype("float64")[:valid.shape[0], :valid.shape[1]]
    alos = np.where(np.isfinite(alos), alos, np.nanmedian(alos))
    za = [C2.local_norm_nz(C1.ridge_dir(alos, DEM_SIG_M, s)[0], valid) for s in (+1, -1)]
    X = np.zeros((len(pool), len(FEATS)))
    chs = np.array([p["ch"] for p in pool])
    for ch in range(5):                                       # расстояние до ближайшей линии ДРУГОГО канала
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
        X[i, :5] = np.where(np.arange(5) == p["ch"], 1.0, -1.0)
        X[i, 5] = p["mz"]
        pl = p["poly"]
        seg = np.hypot(*np.diff(pl, axis=0).T).sum() * RES_IN       # длина ломаной, м
        chord = float(np.hypot(*(pl[-1] - pl[0]))) * RES_IN
        X[i, 6] = np.log(max(seg, RES_IN))
        X[i, 7] = min(chord / max(seg, 1e-9), 1.0)                  # прямизна = хорда / длина ломаной
        r100, c100 = np.minimum(c[:, 0] // 5, Z.shape[0] - 1), np.minimum(c[:, 1] // 5, Z.shape[1] - 1)
        zz = Z[r100, c100]
        X[i, 8:17] = np.nanmean(zz, 0) if np.isfinite(zz).any() else 0.0
        X[i, 17] = za[0][c[:, 0], c[:, 1]].mean()
        X[i, 18] = za[1][c[:, 0], c[:, 1]].mean()
    X[:, 8:17] = np.nan_to_num(X[:, 8:17])
    from scipy.stats import norm, rankdata
    for k in QNORM:                                           # квантили по листу -> нормальное распределение
        q = np.quantile(X[:, k], [0.1, 0.5, 0.9])
        log(f"    до нормировки «{FEATS[k]}»: 10/50/90 % {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
        X[:, k] = norm.ppf((rankdata(X[:, k], method="average") - 0.5) / len(X))
    log(f"  признаки: {X.shape[0]} линий × {X.shape[1]}")
    return X


# ---------------------------------------------------------------- метки (одна функция для GTK и разметки)
def labels_from_lines(gdf, area, pool, meta, shape, t_m, lo=None):
    """Метка кандидата: 1, если >= 50 % его ячеек в t_m от линий учителя; 0 — иначе (lo=None, как ANFL0) или
    только при доле < lo, а между lo и 50 % — -1 «не учится» (для ANFL1, lo задаётся его поправкой).
    Только кандидаты внутри участка (area — маска 20 м). Возвращает (y, inside, frac, ref)."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    tr = Affine(RES_IN, 0, meta["x0"], 0, -RES_IN, meta["y_top"])
    ref = rasterize([(g, 1) for g in gdf.geometry if g is not None], out_shape=shape, transform=tr, fill=0,
                    all_touched=True, dtype="uint8").astype(bool)
    d = distance_transform_edt(~ref) * RES_IN
    frac = np.array([np.mean(d[p["cells"][:, 0], p["cells"][:, 1]] <= t_m) for p in pool])
    inside = np.array([area[p["cells"][:, 0], p["cells"][:, 1]].mean() >= 0.5 for p in pool])
    y = (frac >= LABEL_FRAC).astype(int)
    if lo is not None:
        y[(frac >= lo) & (frac < LABEL_FRAC)] = -1
    return y, inside, frac, ref & area


def self_test():
    """Инварианты: 20 признаков; метка на синтетике; lo=None — двоичная метка как ANFL0."""
    import geopandas as gpd
    from shapely.geometry import LineString
    assert len(FEATS) == 20
    meta = {"x0": 0.0, "y_top": 2000.0, "epsg": 32635}
    shape = (100, 100)
    g = gpd.GeoDataFrame({"geometry": [LineString([(0, 1010), (2000, 1010)])]}, crs="EPSG:32635")   # строка 49
    near = {"cells": np.array([[50, c] for c in range(10, 60)])}                      # 20 м от учителя
    far = {"cells": np.array([[90, c] for c in range(10, 60)])}                       # 800 м
    half = {"cells": np.array([[50 if c < 30 else 70, c] for c in range(10, 50)])}    # половина рядом
    area = np.ones(shape, bool)
    y, ins, fr, _ = labels_from_lines(g, area, [near, far, half], meta, shape, 300.0)
    assert list(y) == [1, 0, 1] and ins.all(), (y, fr)
    y2, *_ = labels_from_lines(g, area, [near, far, half], meta, shape, 127.0, lo=0.1)
    assert list(y2) == [1, 0, 1], y2
    third = {"cells": np.array([[50 if c < 20 else 70, c] for c in range(10, 50)])}  # четверть рядом
    y3, *_ = labels_from_lines(g, area, [third], meta, shape, 127.0, lo=0.1)
    assert list(y3) == [-1], y3


def gtk_teacher(site, pool, valid, meta):
    import geopandas as gpd

    from experiments.openlin_fam import gtk_file
    g = gpd.read_file(gtk_file(site)).to_crs(epsg=int(meta["epsg"]))
    return labels_from_lines(g, valid, pool, meta, valid.shape, TOL_M)


# ---------------------------------------------------------------- мера
def sel_mask(pool, sel, shape):
    m = np.zeros(shape, bool)
    for i in np.flatnonzero(sel):
        c = pool[i]["cells"]
        m[c[:, 0], c[:, 1]] = True
    return m


def f1_len(pool, sel, ref, t_m=TOL_M, d_ref=None):
    """F1 по длине (ячейкам растра): P — доля ячеек выбранных в t от эталона, R — доля ячеек эталона в t
    от выбранных. d_ref — готовое расстояние до эталона, м (кэш; результат тот же)."""
    m = sel_mask(pool, sel, ref.shape)
    if not m.any() or not ref.any():
        return 0.0, 0.0, 0.0, 0.0
    if d_ref is None:
        d_ref = distance_transform_edt(~ref) * RES_IN
    d_sel = distance_transform_edt(~m) * RES_IN
    P = float(np.mean(d_ref[m] <= t_m))
    R = float(np.mean(d_sel[ref] <= t_m))
    return P, R, (2 * P * R / (P + R) if P + R else 0.0), m.sum() * RES_IN / 1000


def top_by_km(score, pool, km):
    """Лучшие по оценке линии, пока сумма ячеек <= km."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum([len(pool[i]["cells"]) for i in order]) * RES_IN / 1000
    n = int(np.searchsorted(cum, km, side="right"))
    sel = np.zeros(len(pool), bool)
    sel[order[:n]] = True
    return sel


# ---------------------------------------------------------------- Анфиса
def train(Xtr, ytr, log):
    preds, params = [], []
    for s in SEEDS:
        p, prm = fit_fam(Xtr, ytr, s)
        preds.append(p)
        params.append(prm)
    log(f"  Анфиса: {N_TERMS} терма, LR {LR}, эпох {EPOCHS}, зёрна {SEEDS}; строк {len(ytr)}, «да» {int(ytr.sum())}")
    return (lambda X: np.mean([p(X) for p in preds], 0)), params


def rules_text(params):
    """Правила Анфисы: для каждого признака — центры трёх термов и их вклад (среднее по зёрнам)."""
    c = np.mean([p["c"] for p in params], 0)
    b = np.mean([p["b"] for p in params], 0)
    imp = b.max(1) - b.min(1)
    lines = ["| признак | центры термов (низк. / средн. / выс.) | вклад термов | размах вклада |", "|---|---|---|---|"]
    for k in np.argsort(-imp):
        o = np.argsort(c[k])
        lines.append(f"| {FEATS[k]} | {' / '.join(f'{x:+.2f}' for x in c[k][o])} | "
                     f"{' / '.join(f'{x:+.2f}' for x in b[k][o])} | {imp[k]:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- вывод
def export_gpkg(path, pool, sel, score, meta):
    import geopandas as gpd
    from shapely.geometry import LineString
    rows = []
    for i in np.flatnonzero(sel):
        pl = pool[i]["poly"]
        x = meta["x0"] + (pl[:, 1] + 0.5) * RES_IN
        y = meta["y_top"] - (pl[:, 0] + 0.5) * RES_IN
        if len(pl) < 2:
            continue
        rows.append({"канал": C1.CHANNELS[pool[i]["ch"]], "оценка": float(score[i]), "учитель": TEACHER,
                     "длина_м": float(len(pool[i]["cells"]) * RES_IN), "geometry": LineString(np.c_[x, y])})
    gpd.GeoDataFrame(rows, crs=f"EPSG:{int(meta['epsg'])}").to_file(path, driver="GPKG")


def windows(site, pool, sel, img, dem, log, win_km, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dst = OUT / f"anfl0_{site}"
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
            r0 = int((i + 0.5) * H / 3 - wp / 2)
            c0 = int((j + 0.5) * Wd / 3 - wp / 2)
            fig, axs = plt.subplots(1, 3, figsize=(27, 9.4))
            for ax, (t, a, show) in zip(axs, (("зимний S2, без линий", img, False), ("S2 + линии", img, True),
                                              ("рельеф GLO-30 + линии", shade, True))):
                b = a[r0:r0 + wp, c0:c0 + wp]
                lo, hi = np.nanpercentile(b, [2, 98])
                ax.imshow(b, cmap="gray", vmin=lo, vmax=hi, extent=(0, win_km, win_km, 0))
                ax.set_title(t, fontsize=14)
                if not show:
                    continue
                for flag, col, ms in ((False, "#9bd", 0.8), (True, "red", 1.8)):
                    for idx in np.flatnonzero(sel == flag):
                        y = pool[idx]["cells"] - [r0, c0]
                        m = (y[:, 0] >= 0) & (y[:, 0] < wp) & (y[:, 1] >= 0) & (y[:, 1] < wp)
                        if m.any():
                            y = y[m] * RES_IN / 1000
                            ax.plot(y[:, 1], y[:, 0], ".", ms=ms, color=col)
                ax.set_xlim(0, win_km)
                ax.set_ylim(win_km, 0)
            fig.suptitle(f"{title}; {site}, окно {k} ({win_km}×{win_km} км), строка {r0}, столбец {c0}. "
                         f"Красные — выбрала Анфиса, голубые — отброшенные кандидаты", fontsize=14)
            fig.tight_layout()
            p = dst / f"окно_{k}.png"
            fig.savefig(p, dpi=70)
            plt.close(fig)
            log(f"  {p}")


# ---------------------------------------------------------------- ANFL0
def site_data(site, log, teacher=True):
    pool, img, dem, valid, meta = candidates(site, log)
    X = line_features(site, pool, valid, log)
    if not teacher or site not in FAM3_TRAIN:       # обучающие листы: снимок и рельеф не держим (память)
        d = {"pool": pool, "X": X, "img": img, "dem": dem, "valid": valid, "meta": meta}
    else:
        d = {"pool": pool, "X": X, "img": None, "dem": None, "valid": valid, "meta": meta}
    del img, dem
    if teacher:
        y, inside, frac, ref = gtk_teacher(site, pool, valid, meta)
        d.update(y=y, inside=inside, frac=frac, ref=ref)
        log(f"  {site}: учитель GTK 292, t = {TOL_M:.0f} м: «да» {int(y.sum())} из {len(y)} ({100 * y.mean():.1f} %)")
    return d


def anfl0(log):
    log("=== обучение: " + ", ".join(FAM3_TRAIN))
    tr = {s: site_data(s, log) for s in FAM3_TRAIN}
    for s in FAM3_TRAIN:                       # учатся только кандидаты внутри участка и с меткой 0/1
        tr[s]["ok"] = tr[s]["inside"] & (tr[s]["y"] >= 0)
    Xtr = np.concatenate([tr[s]["X"][tr[s]["ok"]] for s in FAM3_TRAIN])
    ytr = np.concatenate([tr[s]["y"][tr[s]["ok"]] for s in FAM3_TRAIN])
    model, params = train(Xtr, ytr, log)
    # порог выбора (поправка 1 registrar): «лист за листом» — модель на двух листах оценивает третий;
    # порог — наибольший F1 по длине на отложенных оценках; итоговая модель — на всех трёх
    sc_tr = {}
    for s in FAM3_TRAIN:
        rest = [q for q in FAM3_TRAIN if q != s]
        m_s, _p = train(np.concatenate([tr[q]["X"][tr[q]["ok"]] for q in rest]),
                        np.concatenate([tr[q]["y"][tr[q]["ok"]] for q in rest]), lambda *a: None)
        sc_tr[s] = m_s(tr[s]["X"])
    log("  порог — по отложенным оценкам «лист за листом» (модель на двух листах, оценка третьего)")
    allsc = np.concatenate(list(sc_tr.values()))
    grid = np.quantile(allsc, np.linspace(0.02, 0.98, N_THR))
    best = (-1, None)
    for s in FAM3_TRAIN:
        tr[s]["dref"] = (distance_transform_edt(~tr[s]["ref"]) * RES_IN).astype("float32")
    for thr in grid:
        tp = rp = 0.0
        for s in FAM3_TRAIN:
            P, R, _f, _km = f1_len(tr[s]["pool"], sc_tr[s] >= thr, tr[s]["ref"], d_ref=tr[s]["dref"])
            tp, rp = tp + P, rp + R
        P, R = tp / len(FAM3_TRAIN), rp / len(FAM3_TRAIN)      # средние по трём отложенным листам
        f = 2 * P * R / (P + R) if P + R else 0.0
        if f > best[0]:
            best = (f, thr, P, R)
    f_tr, thr, P_tr, R_tr = best
    log(f"  порог выбора (по обучению): {thr:.3f}; F1 обучения {f_tr:.3f} (P {P_tr:.3f}, R {R_tr:.3f})")
    share_tr = np.mean([sel_mask(tr[s]["pool"], sc_tr[s] >= thr, tr[s]["ref"].shape).sum() /
                        max(sel_mask(tr[s]["pool"], np.ones(len(tr[s]["pool"]), bool), tr[s]["ref"].shape).sum(), 1)
                        for s in FAM3_TRAIN])
    # лучший одиночный признак (на обучении, при той же доле длины)
    best_f = (-1, None, None)
    for k in range(len(FEATS)):
        for sg in (+1, -1):
            fs = []
            for s in FAM3_TRAIN:
                pool = tr[s]["pool"]
                km_all = sum(len(p["cells"]) for p in pool) * RES_IN / 1000
                fs.append(f1_len(pool, top_by_km(sg * tr[s]["X"][:, k], pool, share_tr * km_all), tr[s]["ref"],
                                 d_ref=tr[s]["dref"])[2])
            if np.mean(fs) > best_f[0]:
                best_f = (float(np.mean(fs)), k, sg)
    log(f"  лучший одиночный признак на обучении: «{FEATS[best_f[1]]}» ({'+' if best_f[2] > 0 else '−'}), "
        f"F1 {best_f[0]:.3f}; доля длины выбора на обучении {100 * share_tr:.1f} %")
    log("=== правила Анфисы\n" + rules_text(params))
    del tr
    res = {}
    for site in ("utsjoki", "anabar"):
        log(f"=== {site}")
        d = site_data(site, log, teacher=(site == "utsjoki"))
        sc = model(d["X"])
        sel = sc >= thr
        km_sel = sel_mask(d["pool"], sel, d["valid"].shape).sum() * RES_IN / 1000
        area = d["valid"].sum() * RES_IN ** 2 / 1e6
        log(f"  выбрано {int(sel.sum())} из {len(sel)} линий, {km_sel:.0f} км (объединение), {km_sel / area:.2f} км/км²; "
            f"по каналам: " + ", ".join(f"{C1.CHANNELS[ch]} {int((sel & (np.array([p['ch'] for p in d['pool']]) == ch)).sum())}"
                                        for ch in range(5)))
        np.savez_compressed(OUT / f"anfl0_{site}.npz", score=sc, sel=sel, thr=thr, teacher=TEACHER,
                            ch=np.array([p["ch"] for p in d["pool"]]), X=d["X"])
        export_gpkg(OUT / f"anfl0_{site}.gpkg", d["pool"], sel, sc, d["meta"])
        log(f"  {OUT / f'anfl0_{site}.gpkg'}")
        if site == "utsjoki":
            ref = d["ref"]
            d6 = (distance_transform_edt(~ref) * RES_IN).astype("float32")      # кэш расстояния до GTK
            P, R, F, km = f1_len(d["pool"], sel, ref, d_ref=d6)
            log(f"  Анфиса: F1 {F:.3f} (P {P:.3f}, R {R:.3f}) при {km:.0f} км")
            rows = []
            for rs in range(N_RAND):
                rnd = np.random.default_rng(RAND_SEED + rs).random(len(sel))
                rows.append(f1_len(d["pool"], top_by_km(rnd, d["pool"], km), ref, d_ref=d6)[2])
            log(f"  случайные кандидаты при той же длине: F1 {np.mean(rows):.3f} [мин {np.min(rows):.3f}, макс {np.max(rows):.3f}]")
            P1, R1, F1_, km1 = f1_len(d["pool"], top_by_km(best_f[2] * d["X"][:, best_f[1]], d["pool"], km), ref, d_ref=d6)
            log(f"  лучший одиночный признак при той же длине: F1 {F1_:.3f} (P {P1:.3f}, R {R1:.3f}) при {km1:.0f} км")
            Pa, Ra, Fa, kma = f1_len(d["pool"], np.ones(len(sel), bool), ref, d_ref=d6)
            log(f"  все кандидаты: F1 {Fa:.3f} (P {Pa:.3f}, R {Ra:.3f}) при {kma:.0f} км")
            l6 = np.load(OUT / "linb6_utsjoki.npz")["ids"][:ref.shape[0], :ref.shape[1]] > 0
            km6 = l6.sum() * RES_IN / 1000
            ds6 = distance_transform_edt(~l6) * RES_IN
            P6, R6 = float(np.mean(d6[l6] <= TOL_M)), float(np.mean(ds6[ref] <= TOL_M))
            Pq, Rq, Fq, kmq = f1_len(d["pool"], top_by_km(sc, d["pool"], km6), ref, d_ref=d6)
            log(f"  LINB6: F1 {2 * P6 * R6 / (P6 + R6):.3f} (P {P6:.3f}, R {R6:.3f}) при {km6:.0f} км; "
                f"Анфиса при той же длине: F1 {Fq:.3f} (P {Pq:.3f}, R {Rq:.3f}) при {kmq:.0f} км")
            res[site] = {"F1": F, "P": P, "R": R, "km": km, "rand": float(np.mean(rows)), "single": F1_,
                         "all": Fa, "linb6": 2 * P6 * R6 / (P6 + R6), "anfl_at_linb6": Fq}
        windows(site, d["pool"], sel, d["img"], d["dem"], log, 6 if site == "utsjoki" else 8,
                "ANFL0: Анфиса, учитель GTK 292 (временно)")
        del d
    (OUT / "anfl0_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    from experiments.openlin_fam import log_to
    from experiments.openlin_len1 import sha
    if "--anfl0" in sys.argv:
        log = log_to(OUT / "_anfl0.log")
        log(f"ANFL0; audit/ANFL0_DECLARATION.md; sha256 кода {sha(__file__)}, CAND3 {sha(C3.__file__)}, "
            f"CAND2 {sha(C2.__file__)}, CAND1 {sha(C1.__file__)}")
        self_test()
        log("  инварианты: 20 признаков, метка на синтетике, lo=None — двоичная — ок")
        from experiments.openlin_transfer_check import COMP
        for s in FAM3_TRAIN + ("utsjoki", "anabar"):
            log(f"  входы {s}: S2 {sha(COMP / f'winter_{s}_20m_лист.npz')}, GLO-30 "
                f"{sha(DEM / f'dem_glo30_{s}_лист_20m.npy')}, ALOS {sha(DEM / f'dem_alos30_{s}_лист_20m.npy')}, "
                f"FAM {sha(OUT / f'feat_{s}.npz')}")
        log(f"  заморожено: термов {N_TERMS}, LR {LR}, эпох {EPOCHS}, зёрна {SEEDS}; метка: >= {LABEL_FRAC:.0%} ячеек "
            f"в {TOL_M:.0f} м от линий учителя, lo=None; нормировка рангами по листу у признаков "
            f"{[i + 1 for i in QNORM]}, FAM-признаки — квантильно по окнам листа (как FAM); порог: «лист за листом», "
            f"F1 по длине, сетка {N_THR} квантилей оценок 2-98 %; кандидаты — весь годный лист, обучение — внутри "
            f"участка (годные ячейки); зерно случайных {RAND_SEED} (описание), p-проверок нет")
        log("  признаки (порядок): " + "; ".join(f"{i + 1} {f}" for i, f in enumerate(FEATS)))
        anfl0(log)


if __name__ == "__main__":
    main()
