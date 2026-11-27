"""ANZ2: Анфиса-карта с масштабом 20 м (тонкие линии) — 12 карт входа, врезки снимка шириной 20/40/80/160 м.

Объявление audit/ANZ2_DECLARATION.md. Сеть, обучение, порог — как ANZ1; линии — цепочка ANZ1b.
    python -m experiments.openlin_anz2 --run
    python -m experiments.openlin_anz2 --windows
ANZ2_N_TESTS = 0.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from experiments import openlin_anz1 as A1
from experiments import openlin_anz1b as B
from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments import openlin_syn1 as S1
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU
from experiments.openlin_fam import IMG_SIG_M, OUT, RES_IN
from experiments.openlin_iso import Q_LOW
from experiments.openlin_zone_lines_v3 import CONF_TOL_M

ANZ2_N_TESTS = 0
TEACHER = "Анфиса-карта (ANZA) с масштабом 20 м, линии LIRA с гистерезисом (ANZ2); учитель — искусственные линии"
THIN_M = RES_IN                                  # 20 м — одна клетка (объявление п. 1)
IMG_SIG_ANZ2 = (THIN_M,) + tuple(IMG_SIG_M)      # ширины врезок снимка в ANZ2 (п. 3)
OUT_DIR = OUT / "anz2"


def feature_stack2(img, dem, alos_raw, valid):
    """12 карт: 10 карт ANZ1 + тёмная и светлая полоса снимка на масштабе 20 м (п. 2)."""
    X = A1.feature_stack(img, dem, alos_raw, valid)
    extra = [A1._rank_normal(np.asarray(C2.local_norm_nz(C1.ridge_dir(img, (THIN_M,), s)[0], valid), "float64"), valid)
             for s in (+1, -1)]
    return np.concatenate([X, np.stack(extra)])


def _is_thin(ins):
    parts = ins.get("parts") or [ins]
    return any(abs(float(p.get("width_m", 0)) - THIN_M) < 1e-6 for p in parts)


def run(log, epochs=A1.EPOCHS):
    import torch
    from src.detection_limit import robust_sigma
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = {"объявление": "audit/ANZ2_DECLARATION.md", "учитель": TEACHER, "ANZ2_N_TESTS": ANZ2_N_TESTS}
    res["ширины врезок снимка"] = list(IMG_SIG_ANZ2)
    res["хэш набора ширин"] = __import__("hashlib").sha1(repr(IMG_SIG_ANZ2).encode()).hexdigest()[:10]
    train_sites = S1.SITES[:4]
    assert "utsjoki" not in train_sites
    old_sig = S.IMG_SIG_M
    S.IMG_SIG_M = IMG_SIG_ANZ2               # подмена только внутри прогона (п. 3)
    try:
        log(f"  версии: torch {torch.__version__}, cuda {torch.version.cuda}; устройство {A1.device()}; ширины врезок снимка "
            f"{S.IMG_SIG_M} (в общем коде {old_sig}; подмена только на время прогона, хэш "
            f"{__import__('hashlib').sha1(repr(S.IMG_SIG_M).encode()).hexdigest()[:10]}); вход 12 карт (+ полосы 20 м); листы обучения {', '.join(train_sites)}")
        tr, real, fakeB = [], {}, {}
        for i, site in enumerate(train_sites):
            img, dem, raw, valid, _m = A1._sheet(site)
            f_img, f_dem, f_alos = S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_A + i)
            plan = S.make_plan(valid.shape, valid, RES_IN,
                               np.random.default_rng(S1.SEED_FAKE_A + i + S1.SEED_PLAN_SHIFT), S1.N_PROPOSALS)
            sig_i, sig_d = robust_sigma(f_img[valid].astype("float64")), S.relief_sigma(f_dem, valid)
            i2, d2, a2, truth = S.apply_plan(f_img, f_dem, f_alos, plan, sig_i, sig_d, RES_IN, valid=valid)
            a2[~np.isfinite(raw)] = np.nan
            lab = A1.label_map(truth, valid)
            tr.append((feature_stack2(i2, d2, a2, valid), lab))
            log(f"  {site}: врезок {len(plan)} (узких 20 м {sum(_is_thin(p) for p in plan)}), клеток «да» {int((lab == 1).sum())}")
            real[site] = (feature_stack2(img, dem, raw, valid), valid)
            fakeB[site] = (feature_stack2(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + i), valid), valid)
        net = A1.train_net(tr, log, epochs)
        del tr
        torch.save(net.state_dict(), OUT_DIR / "anz2_net.pt")
        maps = {k: A1.predict(net, X) for k, (X, _v) in real.items()}
        mapsB = {k: A1.predict(net, X) for k, (X, _v) in fakeB.items()}
        allr = np.concatenate([maps[k][real[k][1]] for k in maps])
        thr = float(np.quantile(allr, A1.Q_TOP))
        log(f"=== порог карты: {A1.Q_TOP:.0%} оценок настоящих клеток листов обучения = {thr:.4f} (правило ANZ1)")
        ar = af = 0.0
        for k in maps:
            v = real[k][1]
            a_r = float((maps[k][v] >= thr).sum()) * RES_IN ** 2 / 1e6
            a_f = float((mapsB[k][v] >= thr).sum()) * RES_IN ** 2 / 1e6
            ar += a_r; af += a_f
            log(f"  {k}: площадь выше порога снимок {a_r:.2f} км², подделка Б {a_f:.2f} км² ({a_f / max(a_r, 1e-9):.3f})")
        log(f"  сводно по площади Б/снимок {af / max(ar, 1e-9):.3f}")
        res["порог"] = {"значение": thr, "площадь Б/снимок": af / max(ar, 1e-9)}

        def chain(site, sc, sB, v):
            low = float(np.quantile(sB[v], Q_LOW))
            L, base, nb, kmb, polys = B.lines_hyst(sc, v, thr, low, log, f"{site} снимок")
            st_r = B.lines_hyst.stats
            LB, baseB, *_x = B.lines_hyst(sB, v, thr, low, log, f"{site} подделка Б")
            st_b = B.lines_hyst.stats
            km = lambda m: m.sum() * RES_IN / 1000
            r = {"нижний уровень": low, "км линий": km(L), "км линий подделки Б": km(LB), "км до сшивки": km(base),
                 "км до сшивки подделки Б": km(baseB), "мостиков": nb,
                 "доля шума до сшивки": km(baseB) / max(km(base), 1e-9),
                 "доля шума после сшивки": km(LB) / max(km(L), 1e-9),
                 "покрытие снимок": B.coverage(sc, v, L, thr), "покрытие подделка Б": B.coverage(sB, v, LB, thr),
                 "область снимок": st_r, "область подделка Б": st_b, "лист обучения": site != "utsjoki"}
            log(f"  {site}: нижний уровень {low:.4f}; линий {km(L):.1f} км (Б {km(LB):.1f}); доля шума до сшивки "
                f"{r['доля шума до сшивки']:.3f}, после {r['доля шума после сшивки']:.3f}")
            return L, polys, r
        out = {}
        for k in train_sites:
            L, polys, res[k] = chain(k, maps[k], mapsB[k], real[k][1])
            out[k] = (maps[k], L, polys)
        iu = S1.SITES.index("utsjoki")
        img, dem, raw, valid, _m = A1._sheet("utsjoki")
        su = A1.predict(net, feature_stack2(img, dem, raw, valid))
        sBu = A1.predict(net, feature_stack2(*S1.joint_fake(img, dem, raw, valid, S1.SEED_FAKE_B + iu), valid))
        Lu, polys_u, res["utsjoki"] = chain("utsjoki", su, sBu, valid)
        # (а) врезки в настоящий фон Утсйоки — отдельно узкие и прежние ширины
        plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(S1.SEED_PLAN_UTS), S1.N_PROPOSALS)
        i2, d2, a2, truth = S.apply_plan(img, dem, raw, plan, robust_sigma(img[valid].astype("float64")),
                                         S.relief_sigma(dem, valid), RES_IN, valid=valid)
        a2[~np.isfinite(raw)] = np.nan
        si = A1.predict(net, feature_stack2(i2, d2, a2, valid))
        Li, *_r = B.lines_hyst(si, valid, thr, res["utsjoki"]["нижний уровень"], log, "utsjoki + врезки")
        cells, dirs = A1.line_dirs(Li)
        cov = S1.recall_of_insertions(cells, dirs, truth, plan, CONF_TOL_M, TAU)
        amps = np.array([p["amp"] for p in plan])
        thin = np.array([_is_thin(p) for p in plan])
        rng_s = np.random.default_rng(A1.SEED_RANDOM)
        shifts = [(int(rng_s.integers(valid.shape[0])), int(rng_s.integers(valid.shape[1]))) for _ in range(A1.N_SHIFTS)]
        covs_rnd = [S1.recall_of_insertions((cells + np.array(sh)) % np.array(valid.shape), dirs, truth, plan,
                                            CONF_TOL_M, TAU) for sh in shifts]
        rows = []
        for grp, gm in (("узкие 20 м", thin), ("прежние ширины", ~thin)):
            for lo, hi in zip(S1.AMP_BINS[:-1], S1.AMP_BINS[1:]):
                sel = gm & (amps >= lo) & (amps < hi) & np.isfinite(cov)
                rate = float(np.mean(cov[sel] >= S1.FOUND_FRAC)) if sel.any() else float("nan")
                rnd = float(np.mean([np.mean(c[sel] >= S1.FOUND_FRAC) for c in covs_rnd])) if sel.any() else float("nan")
                rows.append({"группа": grp, "корзина": f"{lo:g}–{hi:g}σ", "врезок": int(sel.sum()), "найдено": rate,
                             "случайный сдвиг": rnd})
                log(f"  (а) {grp} {lo:g}–{hi:g}σ: врезок {int(sel.sum())}, найдено {rate:.3f}; сдвиги {rnd:.3f}")
        log("  (а) узкие 20 м: врезка субпиксельная (σ = 10 м) — «находит ли сеть подделку 20 м», не «находит ли тонкую "
            "настоящую линию»; с ANZ1 сравнимы только 40–160 м и приблизительно (план врезок другой)")
        res["(а)"] = rows
    finally:
        S.IMG_SIG_M = old_sig
    fb_, fa_ = sum(res[k]["км линий подделки Б"] for k in train_sites), sum(res[k]["км линий"] for k in train_sites)
    log(f"=== сводно по листам обучения (оптимистично, не контроль): линии {fa_:.0f} км, Б {fb_:.0f} км, доля шума "
        f"{fb_ / max(fa_, 1e-9):.3f}; описательно, оценка снизу, не точность")
    try:
        b1 = json.loads((B.OUT_DIR / "anz1b_итог.json").read_text(encoding="utf-8"))
        for k in ("utsjoki", "anabar"):
            log(f"  рядом ANZ1b {k}: линий {b1[k]['км линий']:.1f} км, доля шума {b1[k]['доля шума после сшивки']:.3f}")
    except (OSError, KeyError) as e:
        log(f"  ANZ1b рядом не прочитан: {e}")
    A1.OUT_DIR, A1.TEACHER, A1.NAME = OUT_DIR, TEACHER, "ANZ2"
    A1._save("utsjoki", su, Lu, polys_u, res["utsjoki"]["доля шума после сшивки"], S1.load_site("utsjoki")[4], log)
    sa, La, pa = out["anabar"]
    A1._save("anabar", sa, La, pa, res["anabar"]["доля шума после сшивки"], S1.load_site("anabar")[4], log)
    res["код"] = {"anz2": __import__("experiments.openlin_len1", fromlist=["sha"]).sha(__file__)}
    (OUT_DIR / "anz2_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"  {OUT_DIR / 'anz2_итог.json'}; время {time.time() - t0:.0f} с")


def main():
    global OUT_DIR
    from experiments.openlin_fam import log_to
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--run" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else A1.EPOCHS
        if "--crop" in sys.argv:
            A1.CROP = int(sys.argv[sys.argv.index("--crop") + 1])
            OUT_DIR = OUT / f"anz2_crop{A1.CROP}"
            OUT_DIR.mkdir(parents=True, exist_ok=True)
        log = log_to(OUT_DIR / "_anz2.log")
        log(f"ANZ2; audit/ANZ2_DECLARATION.md; кусок {A1.CROP or 'нет'}; эпох {epochs}")
        run(log, epochs)
    elif "--windows" in sys.argv:
        A1.OUT_DIR, A1.NAME = OUT_DIR, "ANZ2"
        A1.windows(log_to(OUT_DIR / "_anz2_windows.log"))


if __name__ == "__main__":
    main()
