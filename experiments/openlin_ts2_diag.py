"""TS2-разбор: на каком шаге Анфиса теряет опубликованные разломы (описательно).

Те же эталоны, участки и 19 сдвигов по нормали, что в TS2 (audit/TS2_DECLARATION.md).
Для каждого шага Анфисы — «выше k из 19 сдвигов»:
  признаки (9 каналов, после quantile_norm) и карта P — среднее значение на ячейках
  эталона (сетка 100 м) против сдвинутого эталона; сила D (directional) — то же;
  LINB1 и LINB2 — полнота при 300 м, как в TS2. Рельеф тракта — для сравнения.
Порогов нет, ничего не меняется. Запуск: python -m experiments.openlin_ts2_diag
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from experiments.openlin_ts2 import MIN_SHIFT_M, SEEDS, SITES, ref_mask

FEAT = ("тёмная полоса", "светлая полоса", "градиент", "разброс", "уклон", "вогнутость",
        "выпуклость", "прямизна тёмной", "прямизна вогнутости")


def shifts(ref, H, W):
    rr_, cc_ = np.nonzero(ref)
    q = np.stack([rr_, cc_], 1).astype(float)
    q -= q.mean(0)
    _w, v = np.linalg.eigh(q.T @ q)
    nrm = v[:, 0]
    out = []
    for s in SEEDS:
        r = np.random.default_rng(s)
        t = r.uniform(MIN_SHIFT_M, 10.0 * min(H, W) - 1) * r.choice([-1, 1])
        out.append((int(round(t * nrm[0] / 20)), int(round(t * nrm[1] / 20))))
    return out


def above(obs, nul):
    return int((np.asarray(nul) < obs).sum())


def main():
    from experiments.openlin_anfisa_lines import directional
    from experiments.openlin_fam import (all_windows, block, log_to, quantile_norm,
                                         win_mask)
    from experiments.openlin_ts1 import anfisa_features
    from experiments.openlin_typesites_fetch import DEST, OUT, load_sites
    log = log_to(OUT / "_ts2_diag.log")
    log("TS2-разбор: шаги Анфисы против опубликованных разломов (выше k из 19 сдвигов)")
    S = load_sites()
    for name, kind in SITES.items():
        site = S[name]
        ref, _n, _c = ref_mask(name, site, kind)
        if not ref.any():
            continue
        H, W = ref.shape
        sh = shifts(ref, H, W)
        z = np.load(DEST / name / "s2_img.npz")
        g = np.load(DEST / name / "glo30.npz")["dem"]
        valid20 = (z["count"] > 0) & np.isfinite(z["shade"]) & np.isfinite(g)
        F, valid = anfisa_features(z["shade"], g, valid20)
        wins, n = all_windows(valid)
        fit = win_mask(valid.shape, wins, n) & valid
        Z = quantile_norm(F, fit if fit.any() else valid)
        mk = np.load(DEST / name / "ts1_masks.npz")
        P = mk["P"].astype(float)
        D, _th = directional(P, valid)
        ref100 = block(ref.astype(float), 5, "max") > 0
        refs100 = [block(np.roll(ref, s_, (0, 1)).astype(float), 5, "max") > 0 for s_ in sh]

        def cont(M):
            o = np.nanmean(M[ref100 & valid])
            nul = [np.nanmean(M[r_ & valid]) for r_ in refs100]
            return o, nul
        parts = []
        for j, f in enumerate(FEAT):
            o, nul = cont(Z[..., j])
            parts.append(f"{f} {above(o, nul)}")
        log(f"=== {name}: эталон {ref.sum() * 0.02:.0f} км")
        log("  признаки (среднее z на эталоне, выше k из 19): " + "; ".join(parts))
        for lab, M in (("карта P", P), ("сила D", D)):
            o, nul = cont(M)
            log(f"  {lab}: на эталоне {o:+.3f}, сдвиги медиана {np.median(nul):+.3f} -> выше {above(o, nul)} из 19")
        # контроль отбора (по аудиту): случайный отбор линий LINB1 до того же бюджета, 100 раз
        from experiments.openlin_anfisa_lines import rasterize
        from experiments.openlin_ts1 import anfisa_lines
        lines, _P = anfisa_lines(F, valid, lambda *_: None)
        ids_all = np.where(np.isfinite(z["shade"]), rasterize(lines, ref.shape, 5), 0)
        cells = np.bincount(ids_all.ravel(), minlength=len(lines) + 1)[1:]
        budget = mk["рельеф"].sum() * 0.02
        rk = []
        for s_ in range(100):
            order = np.random.default_rng(10_000 + s_).permutation(len(lines))
            nsel = int(np.searchsorted(np.cumsum(cells[order]) * 0.02, budget, side="right"))
            keep = np.zeros(len(lines) + 1, bool)
            keep[order[:nsel] + 1] = True
            m = keep[ids_all]
            dist = distance_transform_edt(~m, sampling=(20.0, 20.0))
            o = float((dist[ref] <= 300).mean())
            nul = [float((dist[np.roll(ref, s2, (0, 1)) & valid20] <= 300).mean()) for s2 in sh]
            rk.append(above(o, nul))
        log(f"  случайный отбор LINB1 до того же бюджета ({budget:.0f} км), 100 раз: выше k из 19 — "
            f"медиана {np.median(rk):.1f}, квартили {np.percentile(rk, 25):.1f}–{np.percentile(rk, 75):.1f}")
        for lab, key in (("рельеф тракта", "рельеф"), ("Анфиса LINB1", "анфиса1"), ("Анфиса LINB2", "анфиса2")):
            m = mk[key]
            dist = distance_transform_edt(~m, sampling=(20.0, 20.0))
            o = float((dist[ref] <= 300).mean())
            nul = [float((dist[np.roll(ref, s_, (0, 1)) & valid20] <= 300).mean()) for s_ in sh]
            log(f"  {lab:14s} {m.sum() * 0.02:5.0f} км: полнота 300 м {100 * o:.0f} % "
                f"(сдвиги медиана {100 * np.median(nul):.0f}) -> выше {above(o, nul)} из 19")


if __name__ == "__main__":
    main()
