"""FAM4: Анфиса, обученная на согласии снимка и рельефа (старый тракт) вместо GTK.

Объявление audit/FAM4_DECLARATION.md. Всё как FAM3, кроме метки: эталон — ячейки 100 м,
где есть линия снимка и в пределах одной ячейки линия рельефа (и наоборот), метка = эталон
+ одна ячейка (как gtk_label). Описательно, FAM4_N_TESTS = 0.
    python -m experiments.openlin_fam4 --tract     # тракт на fam_r1–3 (линии для метки)
    python -m experiments.openlin_fam4 --train     # обучение и заморозка frozen_fam4.pkl
    python -m experiments.openlin_fam4 --eval      # Утсйоки, участки TS2, TS1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

FAM4_N_TESTS = 0
M = Path("outputs/metrics")


def tract_file(site):
    return M / f"openlin_zone_lines_{site}_v3_fam4_тракт.npz"


def run_tracts(log):
    from experiments import openlin_lowsun
    from experiments.openlin_relief_lines import load_slopes
    from experiments.openlin_ts1 import run_tract
    from experiments.openlin_zone_lines import COMP
    from experiments.openlin_zone_lines_v3 import build_halves
    from experiments.openlin_fam import FAM3_TRAIN
    orig = openlin_lowsun.scene_cache
    for site in FAM3_TRAIN:
        if tract_file(site).exists():
            log(f"{site}: линии тракта уже есть")
            continue
        # fam_r3 — тот же лист, что fam_n1; сцены (те же 12) лежат в кэше fam_n1
        openlin_lowsun.scene_cache = (lambda s, _o=orig: _o("fam_n1") if s == "fam_r3" else _o(s))
        try:
            w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
            h = build_halves(site, log)
        finally:
            openlin_lowsun.scene_cache = orig
        tI = run_tract(w["shade"].astype("float32"), h["А"], h["Б"], 20.0, "снимок", count=w["count"], log=log)
        g, a = load_slopes(site, 20.0)
        tR = run_tract(g, a, a, 20.0, "рельеф", log=log)
        np.savez_compressed(tract_file(site), снимок=tI["mask"], рельеф=tR["mask"])
        log(f"{site}: снимок {tI['mask'].sum() * 0.02:.0f} км, рельеф {tR['mask'].sum() * 0.02:.0f} км")


def consensus_label(meta, shape, site):
    """Замена gtk_label: (эталон, метка) на сетке 100 м по согласию снимка и рельефа."""
    from experiments.openlin_fam import CELL_M, RES_IN, block
    z = np.load(tract_file(site))
    k = int(CELL_M / RES_IN)
    i100 = (block(z["снимок"].astype(float), k, "max") > 0)[:shape[0], :shape[1]]
    r100 = (block(z["рельеф"].astype(float), k, "max") > 0)[:shape[0], :shape[1]]
    pad = lambda m: np.pad(m, ((0, shape[0] - m.shape[0]), (0, shape[1] - m.shape[1])))
    i100, r100 = pad(i100), pad(r100)
    st = np.ones((3, 3), bool)
    ref = (i100 & ndi.binary_dilation(r100, st)) | (r100 & ndi.binary_dilation(i100, st))
    return ref, ndi.binary_dilation(ref, st)


def checks_before(log):
    """Поправка 1: расстояние листов обучения до участков; хэш коммита; контроль run_tract."""
    import subprocess

    from pyproj import Geod
    from shapely.geometry import box

    from experiments.openlin_fam import FAM3_TRAIN
    from experiments.openlin_typesites_fetch import load_sites, make_grid
    from experiments.openlin_winter import site_grid
    log("коммит: " + subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())
    geod = Geod(ellps="WGS84")
    S = load_sites()
    best = (np.inf, None, None)
    for t in FAM3_TRAIN:
        bt = box(*site_grid(t, 100.0).bbox_wgs84(pad_m=0))
        for n, st in S.items():
            if st["lat"] is None:
                continue
            bs = box(*make_grid(st, 20.0).bbox_wgs84(pad_m=0))
            if bt.intersects(bs):
                raise SystemExit(f"ОСТАНОВ: лист {t} пересекается с участком {n}")
            p1, p2 = bt.exterior.interpolate(bt.exterior.project(bs.centroid)), bs.centroid
            dkm = geod.inv(p1.x, p1.y, p2.x, p2.y)[2] / 1000
            if dkm < best[0]:
                best = (dkm, t, n)
    log(f"ближайший участок к листам обучения: {best[2]} от {best[1]} ~{best[0]:.0f} км (пересечений нет)")


def freeze_check_a(log):
    """(а) run_tract на старых входах Утсйоки побитово = сохранённым UTS1/RELU1."""
    from experiments.openlin_relief_lines import load_slopes
    from experiments.openlin_ts1 import run_tract
    from experiments.openlin_zone_lines import COMP
    w = dict(np.load(COMP / "winter_utsjoki_20m_лист.npz"))
    hA = np.load(COMP / "winter_utsjoki_20m_лист_А.npz")["shade"]
    hB = np.load(COMP / "winter_utsjoki_20m_лист_Б.npz")["shade"]
    t = run_tract(w["shade"].astype("float32"), hA, hB, 20.0, "снимок", count=w["count"], log=log)
    g, a = load_slopes("utsjoki", 20.0)
    t2 = run_tract(g, a, a, 20.0, "рельеф", log=log)
    ok = (np.array_equal(t["mask"], np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"])
          and np.array_equal(t2["mask"], np.load(M / "openlin_zone_lines_utsjoki_v3_рельеф1.npz")["C_all"]))
    log(f"заморозка (а): run_tract на Утсйоки совпал с UTS1/RELU1 побитово: {ok}")
    return ok


def fit_all(fam, log):
    orig = fam.gtk_label
    fam.gtk_label = consensus_label
    try:
        X, y = fam.build_train(fam.FAM3_TRAIN, log)
    finally:
        fam.gtk_label = orig
    models, params = {}, []
    for seed in fam.SEEDS:
        pred, prm = fam.fit_fam(X, y, seed)
        models.setdefault(fam.FAM_NAME, []).append(pred)
        params.append(prm)
    return X, models, params


def f1_utsjoki16(fam, models, log):
    """F1 на 16 окнах FAM1 Утсйоки (GTK) — только для сверки двух обучений и рядом со случайным."""
    u = fam.site_table("utsjoki", log)
    w_, e_, n_ = fam.windows(u["valid"])
    te = fam.win_mask(u["valid"].shape, w_ + e_, n_) & u["valid"]
    K = int((u["ref"] & te).sum())
    v = []
    for pr in models[fam.FAM_NAME]:
        sc = np.full(te.shape, -np.inf)
        sc[te] = pr(u["Z"][te])
        v.append(fam.budget_f1(sc, te, u["ref"], K)[2])
    r0 = np.random.default_rng(0)
    rnd = np.mean([fam.budget_f1(np.where(te, r0.random(te.shape), -np.inf), te, u["ref"], K)[2] for _ in range(5)])
    return float(np.mean(v)), float(rnd)


def train(log):
    import cloudpickle

    from experiments import openlin_fam as fam
    checks_before(log)
    if not freeze_check_a(log):
        raise SystemExit("ОСТАНОВ: заморозка (а) не прошла")
    for site in fam.FAM3_TRAIN:
        F, valid, meta = fam.features(site, log)
        ref, _lab = consensus_label(meta, valid.shape, site)
        log(f"{site}: эталон согласия {int(ref.sum())} ячеек 100 м ({100 * ref[valid].mean():.2f} % листа)")
    X, models, params = fit_all(fam, log)
    _X2, models2, _p2 = fit_all(fam, log)
    f1a, rnd = f1_utsjoki16(fam, models, log)
    f1b, _ = f1_utsjoki16(fam, models2, log)
    ok = abs(f1a - f1b) <= fam.FREEZE_TOL
    log(f"заморозка (б): F1 Утсйоки 16 окон, обучение 1 {f1a:.6f}, обучение 2 {f1b:.6f}, "
        f"разница {abs(f1a - f1b):.6f} (допуск {fam.FREEZE_TOL}) -> {'СОВПАЛО' if ok else 'НЕ СОВПАЛО'}; "
        f"для сведения: случайный отбор {rnd:.3f}, FAM3 на тех же окнах — fam3.csv")
    if not ok:
        raise SystemExit("ОСТАНОВ: заморозка (б) не прошла")
    _old_train(fam, log, X, models, params, cloudpickle)


def _old_train(fam, log, X, models, params, cloudpickle):
    # самопроверка заморозки: модель, загруженная из файла, даёт тот же ответ на выборке
    p0 = np.mean([p(X[:5000]) for p in models[fam.FAM_NAME]], 0)
    with open(fam.OUT / "frozen_fam4.pkl", "wb") as f:
        cloudpickle.dump({"models": models, "fam_params": params}, f)
    back = cloudpickle.load(open(fam.OUT / "frozen_fam4.pkl", "rb"))["models"][fam.FAM_NAME]
    p1 = np.mean([p(X[:5000]) for p in back], 0)
    log(f"заморозка: расхождение после загрузки {np.max(np.abs(p0 - p1)):.2e}")
    prm = params[0]
    order = np.argsort(prm["c"], 1)
    log("вклады термов (зерно 1), низкий / средний / высокий:")
    for j, fname in enumerate(fam.FEATURES):
        bb = prm["b"][j][order[j]]
        log(f"  {fname}: " + " / ".join(f"{x:+.2f}" for x in bb) + f"  (размах {bb.max() - bb.min():.2f})")


def lines_p(F, valid, model_file):
    """Как openlin_ts1.anfisa_lines, но с моделью из model_file."""
    import cloudpickle
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import branches, directional, length_m, nms, stitch
    from experiments.openlin_fam import FAM_NAME, OUT, all_windows, quantile_norm, win_mask
    from experiments.openlin_zone_lines_v3 import LINK_MIN_LEN_M
    wins, n = all_windows(valid)
    fit = win_mask(valid.shape, wins, n) & valid
    Z = quantile_norm(F, fit if fit.any() else valid)
    preds = cloudpickle.load(open(OUT / model_file, "rb"))["models"][FAM_NAME]
    P = np.full(valid.shape, np.nan)
    P[valid] = np.mean([p(Z[valid]) for p in preds], 0)
    D, th = directional(P, valid)
    keep = (D > 0) & valid & nms(D, th)
    lines = [pl for pl in stitch(branches(skeletonize(keep)), lambda *_: None) if length_m(pl) >= LINK_MIN_LEN_M]
    return lines, P


def evaluate(log):
    import pandas as pd

    from experiments import openlin_ref_match as rm
    from experiments.openlin_anfisa_lines import rasterize
    from experiments.openlin_fam import OUT, features
    from experiments.openlin_linb4 import pick
    from experiments.openlin_transfer_check import COMP, MAIN_LAYER
    from experiments.openlin_ts1 import anfisa_features, linb2_pick
    from experiments.openlin_ts2 import SITES as TS2_SITES
    from experiments.openlin_ts2 import ref_mask
    from experiments.openlin_ts2_diag import above, shifts
    from experiments.openlin_typesites_fetch import DEST, load_sites
    # ---- Утсйоки
    site = "utsjoki"
    F, valid, _m = features(site, log)
    lines, P = lines_p(F, valid, "frozen_fam4.pkl")
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    o = np.load(M / f"openlin_zone_lines_{site}_v3_настройка1.npz")["C_all"]
    shape20 = o.shape
    r = np.load(M / f"openlin_zone_lines_{site}_v3_рельеф1.npz")["C_all"][:shape20[0], :shape20[1]]
    v20 = np.isfinite(w["shade"][:shape20[0], :shape20[1]])
    l2 = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса2.npz")["C_all"]
    l2f4 = linb2_pick(lines, P, shape20, v20, r.sum() * 0.02, 20.0) > 0
    l1f4 = np.where(v20, rasterize(lines, shape20, 5), 0) > 0
    log(f"Утсйоки: LINB1(FAM4) линий {len(lines)}, {l1f4.sum() * 0.02:.0f} км; LINB2(FAM4) {l2f4.sum() * 0.02:.0f} км; "
        f"общих ячеек с LINB2(FAM3) {100 * (l2f4 & l2).sum() / max(l2.sum(), 1):.0f} %")
    np.savez_compressed(M / f"openlin_zone_lines_{site}_v3_анфиса2_fam4.npz", C_all=l2f4, C=l2f4)
    ang, fid = rm.ref_directions(MAIN_LAYER, w, shape20)
    ang = np.where(v20, ang, np.nan)
    pos = rm.positions(ang, fid, v20, rm.shifts(shape20))
    sets = {"снимок": o, "рельеф": r, "LINB2(FAM3)": l2, "LINB2(FAM4)": l2f4}
    ev = {s: rm.evaluate(m, pos, 20.0) for s, m in sets.items()}
    K = int(np.floor(min(e[0].sum() for e in ev.values()) / rm.LEN_STEP_KM))
    Sv = {s: rm.curve_score(*e, K)["S"] for s, e in ev.items()}
    log(f"  K = {K}; " + "; ".join(f"{s} S = {v:.4f}" for s, v in Sv.items()))
    lines_ref = np.unique(fid[fid > 0])
    dd = []
    for k in range(20):
        half = np.random.default_rng(5000 + k).permutation(lines_ref)[:len(lines_ref) // 2]
        d2 = []
        for sel in (np.isin(lines_ref, half), ~np.isin(lines_ref, half)):
            keep = lines_ref[sel]
            ph = [(rc[np.isin(f, keep)], an[np.isin(f, keep)], f[np.isin(f, keep)]) for rc, an, f in pos]
            d2.append(rm.curve_score(*rm.evaluate(l2f4, ph, 20.0), K)["S"]
                      - rm.curve_score(*rm.evaluate(l2, ph, 20.0), K)["S"])
        dd.append((d2[0] - d2[1]) / 2)
    thr = 1.96 * np.sqrt(2) * float(np.std(dd, ddof=1))
    dS = Sv["LINB2(FAM4)"] - Sv["LINB2(FAM3)"]
    log(f"  ΔS FAM4 − FAM3 = {dS:+.4f}; порог шума половин {thr:.4f} -> "
        + ("больше шума" if abs(dS) > thr else "в пределах шума"))
    # ---- участки: TS2 (разломы) и TS1 (густота LINB1)
    S = load_sites()
    rows = []
    for name, st in S.items():
        d = DEST / name
        if st["lat"] is None or not (d / "ts1_masks.npz").exists():
            continue
        z = np.load(d / "s2_img.npz")
        g = np.load(d / "glo30.npz")["dem"]
        valid20 = (z["count"] > 0) & np.isfinite(z["shade"]) & np.isfinite(g)
        Fs, vs = anfisa_features(z["shade"], g, valid20)
        ln, Ps = lines_p(Fs, vs, "frozen_fam4.pkl")
        mk = np.load(d / "ts1_masks.npz")
        ids_all = np.where(np.isfinite(z["shade"]), rasterize(ln, g.shape, 5), 0)
        budget = mk["рельеф"].sum() * 0.02 if mk["рельеф"].any() else None
        l2s = linb2_pick(ln, Ps, g.shape, np.isfinite(z["shade"]), budget, 20.0) > 0
        area = valid20.sum() * 4e-4
        row = dict(участок=name, LINB1_FAM3_км2=round(mk["анфиса1"].sum() * 0.02 / area, 3),
                   LINB1_FAM4_км2=round((ids_all > 0).sum() * 0.02 / area, 3))
        if name in TS2_SITES:
            ref, _n, _c = ref_mask(name, st, TS2_SITES[name])
            if ref.any():
                sh = shifts(ref, *ref.shape)
                lenm = np.array([len(x) for x in ln], float)

                def k19(m):
                    dist = distance_transform_edt(~m, sampling=(20.0, 20.0))
                    ob = float((dist[ref] <= 300).mean())
                    return above(ob, [float((dist[np.roll(ref, s_, (0, 1)) & valid20] <= 300).mean()) for s_ in sh])
                rk = [k19(pick(ln, ids_all, lenm, lenm, budget,
                               order=np.random.default_rng(10_000 + s_).permutation(len(ln))) > 0) for s_ in range(100)]
                q1, q3 = np.percentile(rk, [25, 75])
                row.update(FAM3_300=k19(mk["анфиса2"]), FAM4_300=k19(l2s), случ_IQR=f"{q1:.0f}-{q3:.0f}")
                log(f"  {name}: выше k из 19 при 300 м — LINB2(FAM3) {row['FAM3_300']}, LINB2(FAM4) {row['FAM4_300']}; "
                    f"случайный {q1:.0f}–{q3:.0f}; разница "
                    + ("читается" if abs(row['FAM4_300'] - row['FAM3_300']) > q3 - q1 else "не читается"))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fam4_sites.csv", index=False, encoding="utf-8-sig")
    for c in ("LINB1_FAM3_км2", "LINB1_FAM4_км2"):
        log(f"  густота {c}: медиана {df[c].median():.2f}, CV {100 * df[c].std() / df[c].mean():.0f} %, "
            f"от {df[c].min():.2f} до {df[c].max():.2f}")


def main():
    from experiments.openlin_fam import OUT, log_to
    ap = argparse.ArgumentParser()
    ap.add_argument("--tract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    log = log_to(OUT / "_fam4.log")
    log("FAM4; объявление audit/FAM4_DECLARATION.md")
    if a.tract:
        run_tracts(log)
    if a.train:
        train(log)
    if a.eval:
        evaluate(log)


if __name__ == "__main__":
    main()
