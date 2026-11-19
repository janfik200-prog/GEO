"""META1, часть С: синтетические площадки с известной правдой.

Объявление — ``audit/META1_DECLARATION.md`` (раздел «Синтетика» и уточнение 5).
Площадка строится из 12 настоящих зимних сцен Анабара, на той же сетке, и
пишется в формате площадки ветки. Поэтому рабочий тракт
(``openlin_zone_lines_v3``) идёт по ней БЕЗ ИЗМЕНЕНИЙ, тем же кодом:
композит, половины, суррогаты, сшивка — всё его.

Фон:
* ``F1`` — суррогат IAAFT медианы нормированных сцен плюс разность «сцена
  минус медиана» каждой сцены после случайной замены фаз;
* ``F2`` — перестановка блоков 26 x 26 ячеек (520 м, наименьшее чётное число
  ячеек не короче 500 м — наименьшей длины линии) одной перестановкой во всех
  12 сценах; наложение с шагом 13 ячеек и окном Ханна (сумма весов постоянна).

Вписывается (в нормированные сцены, амплитуда a * sigma композита Анабара,
``robust_sigma`` как в опыте с врезками):
* настоящие линии — гребень 2 км x 40 м во всех 12 сценах, на половине мест;
* ``L1`` — тот же гребень только в сценах половины «А», на другой половине мест;
* ``L3`` — пятно с резким краем (порог сглаженного шума в окне врезки) во
  всех сценах, на другой половине мест;
* ``none`` — вторая половина мест пустая.

Запуск из корня:
    python -m experiments.openlin_meta_synth --фон F2 --амп 8 --ложные L1 --зерно 1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")
OUT = Path("outputs/meta1/synth")
BLOCK, STRIDE = 26, 13
AZ_BINS = 12


def site_name(bg, amp, false, seed, perturbed=False):
    return f"m1_{bg}_a{int(amp)}_{false}_s{seed}" + ("_p" if perturbed else "")


def load_scenes():
    """12 сцен композита Анабара: имя, B04, маска годных ячеек."""
    from experiments.openlin_lowsun import scene_cache
    w = np.load(COMP / "winter_anabar_20m_лист.npz")
    out = []
    for s in w["scenes"]:
        z = np.load(scene_cache("anabar") / f"{s}_лист_20m.npz")
        b = z["B04"].astype("float32")
        out.append((str(s), b, np.isfinite(b) & (b > 0)))
    return w, out


def halves_of(names):
    """Те же половины, что build_halves: по плиткам, по дате, через одну."""
    by_tile = {}
    for n in sorted(names, key=lambda n: n.split("_")[2][:8]):
        tile = next(p for p in n.split("_") if p.startswith("T"))
        by_tile.setdefault(tile, []).append(n)
    a = set()
    for _t, g in sorted(by_tile.items()):
        a |= set(g[0::2])
    return a


def hann_shuffle(z_list, good_list, rng):
    """Ф2: перестановка блоков, одна на все сцены; наложение окнами Ханна."""
    H, W = z_list[0].shape
    rr = np.arange(0, H - BLOCK + 1, STRIDE)
    cc = np.arange(0, W - BLOCK + 1, STRIDE)
    pos = np.array([(r, c) for r in rr for c in cc])
    src = pos[rng.permutation(len(pos))]
    win1 = np.hanning(BLOCK + 2)[1:-1]            # периодическое окно, сумма при шаге 13 = const
    win = np.outer(win1, win1)
    outs = []
    for z, good in zip(z_list, good_list):
        num = np.zeros((H, W)); den = np.zeros((H, W))
        zz = np.where(good, z, 0.0); gg = good.astype(float)
        for (r, c), (sr, sc) in zip(pos, src):
            num[r:r + BLOCK, c:c + BLOCK] += win * zz[sr:sr + BLOCK, sc:sc + BLOCK]
            den[r:r + BLOCK, c:c + BLOCK] += win * gg[sr:sr + BLOCK, sc:sc + BLOCK]
        y = np.where((den > 1e-6) & good, num / np.maximum(den, 1e-6), np.nan)
        outs.append(y.astype("float32"))
    return outs


def iaaft_bg(z_list, good_list, rng):
    """Ф1: IAAFT медианы плюс разности сцен с замешанными фазами."""
    from experiments.openlin_kora import surrogate
    from experiments.openlin_null_input import phase_randomize
    cube = np.stack([np.where(g, z, np.nan) for z, g in zip(z_list, good_list)])
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(cube, axis=0)
    mv = np.isfinite(med)
    s = surrogate(np.nan_to_num(med)[None], mv, rng)[0]
    outs = []
    for z, g in zip(z_list, good_list):
        ok = g & mv
        d = np.where(ok, z - med, 0.0)
        dr = phase_randomize(d, ok, rng) if ok.any() else d
        outs.append(np.where(g, s + dr, np.nan).astype("float32"))
    return outs


def blob(shape_win, rng):
    """Пятно с резким краем: порог 0 сглаженного шума, связная часть у центра."""
    from scipy.ndimage import gaussian_filter, label
    h, w = shape_win
    f = gaussian_filter(rng.normal(size=(h, w)), min(h, w) / 8.0)
    yy, xx = np.mgrid[:h, :w]
    f -= np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2) * f.std()  # к краю окна ниже
    m = f > 0
    lab, _n = label(m)
    c = lab[h // 2, w // 2]
    return (lab == c) if c else m


def insert_blob(x, center, amp, sigma, res_m, mask):
    from scipy.ndimage import gaussian_filter

    from experiments.openlin_azimuth_response import INS_LEN_M
    from experiments.openlin_winter import DETREND_M
    s = DETREND_M / res_m / 2.0
    half = int(INS_LEN_M / 2 / res_m + 3 * s) + 2
    r0, c0 = max(int(center[0]) - half, 0), max(int(center[1]) - half, 0)
    r1, c1 = min(int(center[0]) + half + 1, x.shape[0]), min(int(center[1]) + half + 1, x.shape[1])
    win = np.zeros((r1 - r0, c1 - c0))
    k = int(INS_LEN_M / 2 / res_m)
    cr, cc = int(center[0]) - r0, int(center[1]) - c0
    a0, b0 = max(cr - k, 0), max(cc - k, 0)
    sub = mask[: min(2 * k, win.shape[0] - a0), : min(2 * k, win.shape[1] - b0)]
    win[a0:a0 + sub.shape[0], b0:b0 + sub.shape[1]] = -amp * sigma * sub
    x[r0:r1, c0:c1] += win - gaussian_filter(win, s, mode="nearest")


def build(bg, amp, false, seed, log=print):
    from experiments.openlin_azimuth_response import INS_LEN_M, insert_window, sites, track
    from experiments.openlin_lowsun import scene_cache
    from experiments.openlin_winter import normalize
    from src.detection_limit import robust_sigma

    name = site_name(bg, amp, false, seed)
    rng = np.random.default_rng(10_000 + 100 * seed + {"F1": 1, "F2": 2}[bg])
    w, scenes = load_scenes()
    res_m = float(w["res_m"])
    shade = w["shade"]
    sigma = robust_sigma(shade[np.isfinite(shade)])
    names = [s for s, _b, _g in scenes]
    half_a = halves_of(names)
    z_list, stats = [], []
    for _n, b, g in scenes:
        v = b[g]
        med = float(np.median(v)); q1, q3 = np.percentile(v, [25, 75])
        sd = max(float(q3 - q1) / 1.349, 1e-6)
        z_list.append(normalize(b, g)); stats.append((med, sd))
    goods = [g for _s, _b, g in scenes]
    bgz = hann_shuffle(z_list, goods, rng) if bg == "F2" else iaaft_bg(z_list, goods, rng)
    valid_any = np.isfinite(shade)
    pts = sites(shade.shape, res_m, valid_any, np.random.default_rng(17 + seed))
    n_true = len(pts) // 2
    true_c, false_c = pts[:n_true], pts[n_true:]
    az = np.array([(b_ + 0.5) * 180.0 / AZ_BINS for b_ in range(AZ_BINS)])
    t_az = az[np.arange(n_true) % AZ_BINS]
    f_az = az[np.arange(len(false_c)) % AZ_BINS]
    blobs = [blob((2 * int(INS_LEN_M / 2 / res_m),) * 2, np.random.default_rng(500 + i + 37 * seed))
             for i in range(len(false_c))]
    out_dir = scene_cache(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for (sname, _b, g), z, (med, sd) in zip(scenes, bgz, stats):
        x = np.where(np.isfinite(z), z, 0.0).astype("float64")
        for ctr, a_ in zip(true_c, t_az):
            insert_window(x, ctr, a_, amp, sigma, res_m)
        if false == "L1" and sname in half_a:
            for ctr, a_ in zip(false_c, f_az):
                insert_window(x, ctr, a_, amp, sigma, res_m)
        if false == "L3":
            for ctr, bm in zip(false_c, blobs):
                insert_blob(x, ctr, amp, sigma, res_m, bm)
        ok = g & np.isfinite(z)
        raw = np.where(ok, np.maximum(x * sd + med, 1.0), 0.0).astype("float32")
        np.savez_compressed(out_dir / f"{sname}_лист_20m.npz", B04=raw)
    # композит площадки — тем же кодом, что openlin_winter
    from experiments.openlin_winter import detrend, stack_scenes
    got = {s: {"B04": np.load(out_dir / f"{s}_лист_20m.npz")["B04"],
               "sun": 0.0, "az": float(a_)} for s, a_ in zip(names, w["azimuth"])}
    comp, count, _meta = stack_scenes(got, lambda *_: None)
    valid = np.isfinite(comp["B04"])
    sh = detrend(comp["B04"], valid, res_m)
    np.savez_compressed(COMP / f"winter_{name}_20m_лист.npz", shade=sh,
                        count=count.astype("int16"), res_m=res_m, x0=w["x0"], y_top=w["y_top"],
                        epsg=w["epsg"], sun=w["sun"], azimuth=w["azimuth"],
                        scenes=w["scenes"], B04=comp["B04"])
    # зоны и площадка — копии анабарских: весь лист (C_all) от зон не зависит
    for src, dst in ((M / "openlin_anabar.npz", M / f"openlin_{name}.npz"),
                     (M / "openlin_kora_stable_anabar.npz", M / f"openlin_kora_stable_{name}.npz")):
        if not dst.exists():
            shutil.copyfile(src, dst)
    # правда
    OUT.mkdir(parents=True, exist_ok=True)
    truth = np.zeros(shade.shape, bool)
    for ctr, a_ in zip(true_c, t_az):
        p = track(shade.shape, ctr, a_, res_m)
        truth[p[:, 0], p[:, 1]] = True
    np.savez_compressed(OUT / f"{name}_truth.npz", truth=truth, true_c=true_c, t_az=t_az,
                        false_c=false_c, f_az=f_az, false=false, amp=amp, sigma=sigma)
    log(f"{name}: sigma {sigma:.4f}, мест {len(pts)} (настоящих {n_true}, "
        f"ложных «{false}» {len(false_c) if false != 'none' else 0}), половина «А» {len(half_a)} сцен")
    return name


def perturbed_copy(name, log=print):
    """Площадка для устойчивости листа: те же сцены после perturb (правило ветки)."""
    from experiments.openlin_iso import perturb
    from experiments.openlin_lowsun import scene_cache
    from experiments.openlin_winter import detrend, stack_scenes
    pname = name + "_p"
    src, dst = scene_cache(name), scene_cache(pname)
    dst.mkdir(parents=True, exist_ok=True)
    w = dict(np.load(COMP / f"winter_{name}_20m_лист.npz"))
    got = {}
    for f in sorted(src.glob("*_лист_20m.npz")):
        b = np.load(f)["B04"]
        g = b > 0
        bp = perturb(np.where(g, b, 0.0), g)
        gp = perturb(g.astype("float32"), np.ones_like(g)) > 0.999   # край данных не размазывается
        bp = np.where(gp, np.maximum(bp, 1.0), 0.0).astype("float32")
        np.savez_compressed(dst / f.name, B04=bp)
        got[f.name.split("_лист")[0]] = {"B04": bp, "sun": 0.0, "az": 0.0}
    comp, count, _m = stack_scenes(got, lambda *_: None)
    valid = np.isfinite(comp["B04"])
    w.update(shade=detrend(comp["B04"], valid, float(w["res_m"])), count=count.astype("int16"),
             B04=comp["B04"])
    np.savez_compressed(COMP / f"winter_{pname}_20m_лист.npz", **w)
    for s, d in ((M / f"openlin_{name}.npz", M / f"openlin_{pname}.npz"),
                 (M / f"openlin_kora_stable_{name}.npz", M / f"openlin_kora_stable_{pname}.npz")):
        if not d.exists():
            shutil.copyfile(s, d)
    log(f"{pname}: пересэмплированная копия")
    return pname


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--фон", dest="bg", choices=("F1", "F2"), required=True)
    ap.add_argument("--амп", dest="amp", type=float, choices=(4.0, 8.0, 16.0), required=True)
    ap.add_argument("--ложные", dest="false", choices=("none", "L1", "L3"), required=True)
    ap.add_argument("--зерно", dest="seed", type=int, choices=(1, 2), required=True)
    ap.add_argument("--пересэмпл", dest="pert", action="store_true")
    a = ap.parse_args()
    name = build(a.bg, a.amp, a.false, a.seed)
    if a.pert:
        perturbed_copy(name)
    print(json.dumps({"site": name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
