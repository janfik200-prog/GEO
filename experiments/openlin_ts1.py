"""TS1: Анфиса и старый тракт (тени + рельеф) на участках по видам линеаментов.

Объявление: audit/TS1_DECLARATION.md (13.11.2026, до кода). Описательно,
``TS1_N_TESTS`` = 0: пороги, модели и параметры не меняются ни на одном участке.

Старый тракт — функции ``openlin_zone_lines_v3`` (рабочий набор TWIN, «цепочки»),
набор «весь лист» (``select2(gate=False)``): зона и простирание его не меняют,
поэтому не строятся (передаются «всё поле» и нули). Анфиса — замороженная FAM3,
признаки по формулам ``openlin_fam._features``, линии LINB1, отбор LINB2 до длины
рельефа этого же участка.

Сначала контроль воспроизведения на Утсйоки (UTS1, RELU1, признаки, LINB1, LINB2
совпадают с сохранёнными) — без него участки не считаются:
    python -m experiments.openlin_ts1 --control
    python -m experiments.openlin_ts1 all
    python -m experiments.openlin_ts1 ts_dike_shiprock ts_dunes_namib
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

M = Path("outputs/metrics")
TS1_N_TESTS = 0
OVERLAP_TOL_M = 300.0        # допуск меры S (openlin_ref_match), объявление TS1
CONTROL_OK = Path("outputs/typesites/_ts1_control.json")


# ------------------------------------------------------------------ старый тракт
def tract_lines(opt, hA, hB, valid, res, resp_fn, half_resp, seam):
    """C_all рабочего набора: optics() + run() из main тракта, без зоны."""
    from experiments import openlin_zone_lines_v3 as v3
    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize
    from experiments.openlin_zone_lines import MIN_LEN_M, mask_of, trace
    zone = valid.copy()                                 # не читается (gate=False, сшивка)
    strike = np.zeros(valid.shape, "float32")

    def response(o):
        xx = np.where(valid, standardize(o, valid), 0.0).astype("float32")
        return local_norm(np.where(valid, (resp_fn or ridge_response)(xx), 0.0), valid, res)

    raw = trace(opt, valid, res, local=True, min_len_m=0, response_fn=resp_fn)
    if not raw:          # extend_ends/walk падает на пустом списке (у подделок бывает 0 линий)
        return []
    zz = response(opt)
    # продлеваются только линии >= MIN_LEN_M; если таких нет, extend_ends падает на пустом
    # массиве концов (walk), а по смыслу ничего не продлевает — список остаётся как есть
    if any(x["длина_км"] * 1000 >= MIN_LEN_M for x in raw):
        raw = v3.extend_ends(raw, zz, valid, res, None)
    L = v3.link_same_side(raw, zone, strike, res, None, z=zz, support="mean",
                          offset_m=v3.TWIN_OFFSET_M)
    L = [x for x in L if x["длина_км"] * 1000 >= MIN_LEN_M]
    L = v3.drop_seam_lines(L, seam, res, None)
    W_ = valid.shape[1]
    for x in L:                                         # «цепочки»: мостики через кулисы убираются
        kul = x.get("кулисы")
        if kul is not None and len(kul):
            keep = ~np.isin(x["pts"][:, 0] * W_ + x["pts"][:, 1], kul[:, 0] * W_ + kul[:, 1])
            x["pts"] = x["pts"][keep]
    dts = [distance_transform_edt(~mask_of(trace(h, valid, res, local=True, response_fn=half_resp),
                                           valid.shape), sampling=(res, res)) for h in (hA, hB)]
    _A, C = v3.select2(L, zone, strike, dts, gate=False)
    return C


def seam_of(valid, count=None):
    seam = np.zeros(valid.shape, bool)
    if count is not None:
        seam[:, :-1] |= count[:, :-1] != count[:, 1:]
        seam[:-1, :] |= count[:-1, :] != count[1:, :]
    return seam | ~valid


def run_tract(opt, hA, hB, res, kind, count=None, fakes=0, log=print):
    """kind: 'снимок' (ridge_response, половины снимка) или 'рельеф' (scarp_response)."""
    from experiments.openlin_kora import surrogate
    from experiments.openlin_relief_lines import scarp_response
    from experiments.openlin_zone_lines import mask_of
    resp = scarp_response if kind == "рельеф" else None
    valid = np.isfinite(opt) & np.isfinite(hA) & np.isfinite(hB)
    seam = seam_of(valid, None if kind == "рельеф" else count)
    t0 = time.time()
    C = tract_lines(opt, hA, hB, valid, res, resp, resp, seam)
    out = {"lines": C, "mask": mask_of(C, valid.shape), "valid": valid, "fake_km": []}
    stack = np.stack([np.where(valid, x, np.nan) for x in (opt, hA, hB)])
    for k in range(fakes):
        sg = surrogate(np.nan_to_num(stack), valid, np.random.default_rng(900 + k))
        Cf = tract_lines(sg[0], sg[1], sg[2], valid, res, resp, resp, seam)
        out["fake_km"].append(sum(x["длина_км"] for x in Cf))
    log(f"    {kind}: линий {len(C)}, {sum(x['длина_км'] for x in C):.0f} км; подделки "
        f"{np.round(out['fake_km'], 1).tolist()} км ({time.time() - t0:.0f} с)")
    return out


# ------------------------------------------------------------------ Анфиса
def anfisa_features(img, dem, valid20):
    """Тело openlin_fam._features после чтения файлов — те же формулы и константы."""
    from experiments.openlin_fam import (CELL_M, DEM_SIG_M, GRAD_SIG_M, IMG_SIG_M, RES_IN,
                                         STD_WIN_M, block, multi_ridge, straightness)
    fill = lambda a: np.where(np.isfinite(a), a, np.nanmedian(a))
    img, dem = fill(img.astype("float64")), fill(dem.astype("float64"))
    k = int(CELL_M / RES_IN)
    dark, bright = multi_ridge(img, IMG_SIG_M)
    grad = ndi.gaussian_gradient_magnitude(img, GRAD_SIG_M / RES_IN)
    w = int(STD_WIN_M / RES_IN)
    m1 = ndi.uniform_filter(img, w)
    std = np.sqrt(np.maximum(ndi.uniform_filter(img ** 2, w) - m1 ** 2, 0))
    gy, gx = np.gradient(dem, RES_IN)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    conc, conv = multi_ridge(dem, DEM_SIG_M)
    F = [block(dark, k, "max"), block(bright, k, "max"), block(grad, k, "mean"),
         block(std, k, "mean"), block(slope, k, "mean"), block(conc, k, "max"),
         block(conv, k, "max")]
    F.append(straightness(F[0]))
    F.append(straightness(F[5]))
    valid = block(valid20.astype(float), k, "mean") >= 1.0
    return np.stack(F, -1).astype("float32"), valid


def anfisa_lines(F, valid, log=print):
    """score_map + build (LINB1) на готовых признаках."""
    import cloudpickle
    from skimage.morphology import skeletonize

    from experiments.openlin_anfisa_lines import (branches, directional, length_m, nms,
                                                  stitch)
    from experiments.openlin_fam import (FAM_NAME, OUT, all_windows, quantile_norm,
                                         win_mask)
    from experiments.openlin_zone_lines_v3 import LINK_MIN_LEN_M
    wins, n = all_windows(valid)
    fit = win_mask(valid.shape, wins, n) & valid
    if not fit.any():
        log("    Анфиса: окон 10 км с >= 99 % полных ячеек нет — нормировка по всем полным ячейкам")
        fit = valid
    Z = quantile_norm(F, fit)
    preds = cloudpickle.load(open(OUT / "frozen_fam3.pkl", "rb"))["models"][FAM_NAME]
    P = np.full(valid.shape, np.nan)
    P[valid] = np.mean([p(Z[valid]) for p in preds], 0)
    D, th = directional(P, valid)
    keep = (D > 0) & valid & nms(D, th)
    lines = stitch(branches(skeletonize(keep)), lambda *_: None)
    lines = [pl for pl in lines if length_m(pl) >= LINK_MIN_LEN_M]
    return lines, P


def linb2_pick(lines, P, shape20, valid20, budget_km, res):
    """Отбор LINB2 (main_linb2, вариант с мостиками): по среднему P до бюджета длины."""
    from experiments.openlin_anfisa_lines import length_m, line_mean_p, rasterize
    from experiments.openlin_fam import CELL_M, RES_IN
    k = int(round(CELL_M / RES_IN))
    ids_all = np.where(valid20, rasterize(lines, shape20, k), 0)
    if budget_km is None:
        return ids_all
    cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
    mp = np.array([line_mean_p(pl, P) for pl in lines])
    lenm = np.array([length_m(pl) for pl in lines])
    order = np.lexsort((np.arange(len(lines)), -lenm, -mp))
    cum = np.cumsum(cells[order]) * res / 1000
    n = int(np.searchsorted(cum, budget_km, side="right"))
    keep = np.zeros(len(lines) + 1, bool)
    keep[order[:n] + 1] = True
    return np.where(keep[ids_all], ids_all, 0)


# ------------------------------------------------------------------ контроль на Утсйоки
def control(log):
    from experiments.openlin_fam import DEM, OUT
    from experiments.openlin_relief_lines import load_slopes
    from experiments.openlin_zone_lines import COMP
    site, res = "utsjoki", 20.0
    res_ok = {}
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    hA = np.load(COMP / f"winter_{site}_20m_лист_А.npz")["shade"]
    hB = np.load(COMP / f"winter_{site}_20m_лист_Б.npz")["shade"]
    o = np.load(M / f"openlin_zone_lines_{site}_v3_настройка1.npz")
    r = np.load(M / f"openlin_zone_lines_{site}_v3_рельеф1.npz")
    t = run_tract(w["shade"].astype("float32"), hA, hB, res, "снимок", count=w["count"], log=log)
    res_ok["UTS1"] = bool(np.array_equal(t["mask"], o["C_all"]))
    g, a = load_slopes(site, res)
    t2 = run_tract(g, a, a, res, "рельеф", log=log)
    rr = r["C_all"]
    res_ok["RELU1"] = bool(t2["mask"].shape == rr.shape and np.array_equal(t2["mask"], rr))
    log(f"  UTS1 совпал: {res_ok['UTS1']} (ячеек {int(t['mask'].sum())} / {int(o['C_all'].sum())}); "
        f"RELU1 совпал: {res_ok['RELU1']} ({int(t2['mask'].sum())} / {int(rr.sum())})")
    img = w["shade"]
    dem = np.load(DEM / f"dem_glo30_{site}_лист_20m.npy")
    valid20 = (w["count"] > 0) & np.isfinite(img) & np.isfinite(dem)
    F, valid = anfisa_features(img, dem, valid20)
    q = np.load(OUT / f"feat_{site}.npz")
    res_ok["признаки"] = bool(F.shape == q["F"].shape and np.array_equal(valid, q["valid"])
                              and np.allclose(F, q["F"], atol=1e-5, equal_nan=True))
    lines, P = anfisa_lines(F, valid, log)
    V = np.load(M / f"openlin_zone_lines_{site}_v3_анфиса1_линии.npz")
    saved = [V["rc100"][V["offsets"][i]:V["offsets"][i + 1]] for i in range(len(V["offsets"]) - 1)]
    res_ok["LINB1"] = len(lines) == len(saved) and all(np.array_equal(x, y) for x, y in zip(lines, saved))
    shape20 = o["C_all"].shape
    v20 = np.isfinite(img[:shape20[0], :shape20[1]])
    ids = linb2_pick(lines, P, shape20, v20, rr[:shape20[0], :shape20[1]].sum() * res / 1000, res)
    res_ok["LINB2"] = bool(np.array_equal(ids > 0, np.load(M / f"openlin_zone_lines_{site}_v3_анфиса2.npz")["C_all"]))
    log(f"  признаки: {res_ok['признаки']}; LINB1: {res_ok['LINB1']} ({len(lines)} / {len(saved)}); "
        f"LINB2: {res_ok['LINB2']}")
    CONTROL_OK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_OK.write_text(json.dumps(res_ok, ensure_ascii=False), encoding="utf-8")
    log("КОНТРОЛЬ: " + ("ПРОЙДЕН" if all(res_ok.values()) else "НЕ ПРОЙДЕН — участки не считаются"))


# ------------------------------------------------------------------ участки
def control_b(log):
    """Поправка 1 п. 4(б): снимок s2_img Утсйоки через тот же тракт против UTS1 — описание."""
    from experiments.openlin_typesites_fetch import DEST
    from experiments.openlin_zone_lines import cover
    z = np.load(DEST / "_utsjoki" / "s2_img.npz")
    o = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"]
    V = np.load(M / "openlin_zone_lines_utsjoki_v3_настройка1_линии.npz")
    H, W = (min(a_, b_) for a_, b_ in zip(z["shade"].shape, o.shape))
    t = run_tract(z["shade"][:H, :W], z["half_A"][:H, :W], z["half_B"][:H, :W], 20.0, "снимок",
                  count=z["count"][:H, :W], log=log)
    n_old = len(np.unique(V["C_all_id"])) if "C_all_id" in V.files else None
    log(f"  КОНТРОЛЬ (б), описание: s2_img ({str(z['kind'])}, дат {len(z['dates'])}) — "
        f"{t['mask'].sum() * 0.02:.0f} км, линий {len(t['lines'])}; UTS1 — {o[:H, :W].sum() * 0.02:.0f} км, "
        f"линий {n_old}; cover 300 м {cover(t['mask'], o[:H, :W], 20.0, OVERLAP_TOL_M):.3f}")


def az_hist(segs):
    """Гистограмма азимутов по длине, 12 корзин по 15° (0° — север, по часовой)."""
    h = np.zeros(12)
    for az, ln in segs:
        h[int(az % 180 // 15)] += ln
    return h / max(h.sum(), 1e-9)


def tract_segs(C):
    return [(x["азимут"], x["длина_км"]) for x in C]


def anfisa_line_az(lines, keep_ids=None):
    """Азимут линии целиком — главная ось её точек (PCA), вес — длина линии.

    Шаги ломаной Анфисы идут по сетке 100 м и дают только 0/45/90/135°, поэтому
    гистограмма по шагам (anfisa_segs, первый прогон) — квантование сетки, не
    направление линий (найдено при чтении таблицы 13.11.2026).
    """
    from experiments.openlin_anfisa_lines import length_m
    out = []
    for j, pl in enumerate(lines, 1):
        if keep_ids is not None and j not in keep_ids:
            continue
        q = pl - pl.mean(0)
        if len(pl) < 2 or not q.any():
            continue
        w_, v = np.linalg.eigh(q.T @ q)
        dr, dc = v[:, -1]
        out.append((np.degrees(np.arctan2(dc, -dr)) % 180, length_m(pl) / 1000))
    return out


def recompute_anfisa_az(names, log):
    """Пересчёт азимутов Анфисы в ts1.csv (линии те же: расчёт детерминирован)."""
    import pandas as pd
    from experiments.openlin_relief_lines import ALOS_BAD_M  # noqa: F401 (тот же импорт, что run_site)
    from experiments.openlin_typesites_fetch import DEST, OUT
    df = pd.read_csv(OUT / "ts1.csv", encoding="utf-8-sig")
    for n in names:
        d = DEST / n
        if not (d / "s2_img.npz").exists() or not (d / "ts1_masks.npz").exists():
            continue
        z = np.load(d / "s2_img.npz")
        if not np.isfinite(z["shade"]).any():
            continue
        g = np.load(d / "glo30.npz")["dem"]
        valid20 = (z["count"] > 0) & np.isfinite(z["shade"]) & np.isfinite(g)
        F, valid = anfisa_features(z["shade"], g, valid20)
        lines, P = anfisa_lines(F, valid, log)
        mk = np.load(d / "ts1_masks.npz")
        res = 20.0
        budget = mk["рельеф"].sum() * res / 1000 if mk["рельеф"].any() else None
        ids2 = linb2_pick(lines, P, g.shape, np.isfinite(z["shade"]), budget, res)
        same = bool(np.array_equal(ids2 > 0, mk["анфиса2"]))
        keep2 = set(np.unique(ids2[ids2 > 0]).tolist())
        for s_, keep in (("Анфиса LINB2", keep2), ("Анфиса LINB1 все", None)):
            h = az_hist(anfisa_line_az(lines, keep))
            m = (df["участок"] == n) & (df["набор"] == s_)
            df.loc[m, "азимуты_15"] = " ".join(f"{100 * v:.0f}" for v in h)
            df.loc[m, "главный_азимут"] = int(15 * np.argmax(h) + 7.5)
        log(f"  {n}: азимуты Анфисы по линиям пересчитаны; LINB2 совпал с прогоном: {same}")
    df.to_csv(OUT / "ts1.csv", index=False, encoding="utf-8-sig")


def anfisa_segs(lines, keep_ids=None):
    from experiments.openlin_fam import CELL_M
    out = []
    for j, pl in enumerate(lines, 1):
        if keep_ids is not None and j not in keep_ids:
            continue
        d = np.diff(pl, axis=0)
        for dr, dc in d:
            out.append((np.degrees(np.arctan2(dc, -dr)) % 180, np.hypot(dr, dc) * CELL_M / 1000))
    return out


def run_site(name, site, log):
    from experiments.openlin_relief_lines import ALOS_BAD_M, slope
    from experiments.openlin_typesites_fetch import DEST
    from experiments.openlin_union_recall import add_new
    from experiments.openlin_zone_lines import N_FAKES, cover
    d = DEST / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    res = float(meta["res_m"])
    if not (d / "s2_img.npz").exists():               # меньше двух дат — слой не записан
        log(f"=== {name}: НЕТ ВХОДА — слоя s2_img нет (меньше двух дат)")
        return [dict(участок=name, вид=site["type"], род=site["kind"], снимок="нет входа")],             dict(участок=name, снимок="нет входа")
    z = np.load(d / "s2_img.npz")
    kind = str(z["kind"])
    if not np.isfinite(z["shade"]).any():             # поправка 1 п. 3(д): не подменяется
        log(f"=== {name}: НЕТ ВХОДА — композит пуст (дат меньше MIN_SCENES)")
        return [dict(участок=name, вид=site["type"], род=site["kind"], снимок="нет входа")],             dict(участок=name, снимок="нет входа")
    g = np.load(d / "glo30.npz")["dem"]
    a = np.load(d / "alos.npz")["dem"]
    a = np.where(np.abs(a - g) > ALOS_BAD_M, np.nan, a)
    log(f"=== {name}: {site['type']}; снимок: {kind}, дат {len(z['dates'])}")
    tI = run_tract(z["shade"], z["half_A"], z["half_B"], res, "снимок", count=z["count"],
                   fakes=N_FAKES, log=log)
    sg, sa = slope(g, res), slope(a, res)
    tR = run_tract(sg, sa, sa, res, "рельеф", fakes=N_FAKES, log=log)
    union = tI["mask"] | add_new(tR["mask"], tI["mask"], res)
    valid20 = (z["count"] > 0) & np.isfinite(z["shade"]) & np.isfinite(g)
    F, valid = anfisa_features(z["shade"], g, valid20)
    lines, P = anfisa_lines(F, valid, log)
    v20 = np.isfinite(z["shade"])
    budget = tR["mask"].sum() * res / 1000 if tR["mask"].any() else None
    if budget is None:
        log("    рельеф пуст — Анфиса LINB2 = все линии LINB1")
    ids2 = linb2_pick(lines, P, g.shape, v20, budget, res)
    ids1 = linb2_pick(lines, P, g.shape, v20, None, res)
    log(f"    Анфиса: линий LINB1 {len(lines)}, растр {int((ids1 > 0).sum()) * res / 1000:.0f} км; "
        f"LINB2 {int((ids2 > 0).sum()) * res / 1000:.0f} км (бюджет "
        f"{'нет' if budget is None else f'{budget:.0f} км'})")
    area = valid20.sum() * res * res / 1e6
    keep2 = set(np.unique(ids2[ids2 > 0]).tolist())
    sets = {"снимок": (tI["mask"], tract_segs(tI["lines"]), [x["длина_км"] for x in tI["lines"]], tI["fake_km"]),
            "рельеф": (tR["mask"], tract_segs(tR["lines"]), [x["длина_км"] for x in tR["lines"]], tR["fake_km"]),
            "снимок + рельеф": (union, None, None, None),
            "Анфиса LINB2": (ids2 > 0, anfisa_line_az(lines, keep2), None, None),
            "Анфиса LINB1 все": (ids1 > 0, anfisa_line_az(lines), None, None)}
    rows = []
    for s, (m, segs, lens, fk) in sets.items():
        km = m.sum() * res / 1000
        h = az_hist(segs) if segs else None
        rows.append(dict(участок=name, вид=site["type"], род=site["kind"], снимок=kind, набор=s,
                         км=round(km, 1), км_на_км2=round(km / max(area, 1e-9), 3),
                         линий=len(lens) if lens is not None else (len(keep2) if s == "Анфиса LINB2" else
                                                                     (len(lines) if s.endswith("все") else None)),
                         медиана_км=round(float(np.median(lens)), 2) if lens else None,
                         # длина / длина по трассе (линии подделок — тоже по трассе), не по ячейкам
                         мусор_подделок=(round(float(np.mean(fk)) / sum(lens), 3) if fk and lens else None),
                         азимуты_15=" ".join(f"{100 * v:.0f}" for v in h) if h is not None else None,
                         главный_азимут=int(15 * np.argmax(h) + 7.5) if h is not None and km > 0 else None))
    pairs = [("снимок", "рельеф"), ("Анфиса LINB2", "снимок"), ("Анфиса LINB2", "рельеф"),
             ("Анфиса LINB2", "снимок + рельеф")]
    prow = {f"{x}~{y}": round(cover(sets[x][0], sets[y][0], res, OVERLAP_TOL_M), 3) for x, y in pairs}
    log("    " + "; ".join(f"{r['набор']} {r['км']:.0f} км ({r['км_на_км2']:.2f}/км², мусор "
                          f"{r['мусор_подделок']})" for r in rows))
    log("    перекрытие (300 м): " + ", ".join(f"{k} {v:.2f}" for k, v in prow.items()))
    picture(name, site, g, res, sets, kind)
    np.savez_compressed(d / "ts1_masks.npz", снимок=tI["mask"], рельеф=tR["mask"],
                        анфиса2=ids2 > 0, анфиса1=ids1 > 0, P=P.astype("float32"))
    return rows, dict(участок=name, вид=site["type"], род=site["kind"], снимок=kind, **prow)


def picture(name, site, dem, res, sets, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.openlin_typesites_fetch import OUT, hillshade
    hs = np.mean([hillshade(dem, res, az) for az in (0, 45, 90, 135)], 0)
    fig, ax = plt.subplots(1, 3, figsize=(21, 7.4))
    for a_, (s, col) in zip(ax, (("снимок", "red"), ("рельеф", "deepskyblue"), ("Анфиса LINB2", "gold"))):
        a_.imshow(hs, cmap="gray")
        m = ndi.binary_dilation(sets[s][0], iterations=1)
        a_.imshow(np.ma.masked_where(~m, m), cmap=matplotlib.colors.ListedColormap([col]), alpha=0.9,
                  interpolation="nearest")
        a_.set_title(f"{s}: {sets[s][0].sum() * res / 1000:.0f} км")
        a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f"{name}: {site['type']} — снимок: {kind}")
    fig.savefig(OUT / f"ts1_{name}.png", dpi=75, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sites", nargs="*")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--az-anfisa", action="store_true", help="пересчёт азимутов Анфисы по линиям")
    ap.add_argument("--control-b", action="store_true", help="контроль (б): нужен --utsjoki-control в загрузке")
    a = ap.parse_args()
    from experiments.openlin_typesites_fetch import OUT, load_sites
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "_ts1.log").open("a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    log(f"TS1 {time.strftime('%Y-%m-%d %H:%M')}; объявление audit/TS1_DECLARATION.md")
    if a.control:
        control(log)
        return
    if a.az_anfisa:
        S = load_sites()
        recompute_anfisa_az([n for n in S if S[n]["lat"] is not None], log)
        return
    if a.control_b:
        control_b(log)
        return
    ok = json.loads(CONTROL_OK.read_text(encoding="utf-8")) if CONTROL_OK.exists() else {}
    if not ok or not all(ok.values()):
        raise SystemExit("контроль воспроизведения не пройден или не запускался (--control)")
    S = load_sites()
    names = [n for n in S if S[n]["lat"] is not None] if a.sites == ["all"] else a.sites
    import pandas as pd
    rows, prows = [], []
    # дозапуск части участков дописывает таблицы, а не затирает их
    for f, acc in (("ts1.csv", rows), ("ts1_pairs.csv", prows)):
        if (OUT / f).exists():
            old = pd.read_csv(OUT / f, encoding="utf-8-sig")
            acc += old[~old["участок"].isin(names)].to_dict("records")
    for n in names:
        try:
            r, p = run_site(n, S[n], log)
            rows += r
            prows.append(p)
        except Exception as exc:
            log(f"  {n}: СБОЙ — {exc!r:.300}")
        pd.DataFrame(rows).to_csv(OUT / "ts1.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(prows).to_csv(OUT / "ts1_pairs.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
