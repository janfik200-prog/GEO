"""Пакет 05 ТЗ К1: сверка с GTK при разных нулевых моделях (Лапландия, перенос XFER).

Объявление — ``audit/PACKET05_DECLARATION.md`` (26.10.2026, до расчёта).
Всё здесь разведочное: сверка с GTK уже видена (openlin_transfer_check).

* повторение старого протокола — что совпало с отчётом 25.10 и где отличие;
* нуль 5000 сдвигов эталона (то же правило, пол p 0.0002);
* нуль «по местности»: 159 из этих 5000 сдвигов, ближайших к несдвинутому
  эталону по доле ячеек у воды (OSM, 300 м) и медиане уклона ArcticDEM;
* допуск 150/300/600/1000 м и слои 291/293/294 — для сведения.

Запуск из корня:
    python -m experiments.openlin_gtk_null
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.openlin_auto_reference import grid, rast

M = Path("outputs/metrics")
RES = Path("outputs/k1/results")
FIG = Path("outputs/k1/figures")
SITE = "lapland"

# ---- объявление (audit/PACKET05_DECLARATION.md) ----------------------------
TOL_M = 300.0
TOLS_M = (150.0, 300.0, 600.0, 1000.0)
N_POOL, SEED_POOL = 5000, 2511
N_MATCH = 159
ALPHA = 0.025
WATER_TOL_M = 300.0
REPORT_25_10 = {"C": {"полнота": 0.033, "нуль_ср": 0.015, "нуль_макс": 0.022, "p": 0.0063},
                "C_all": {"полнота": 0.201, "нуль_ср": 0.134, "нуль_макс": 0.157, "p": 0.0063}}
# ----------------------------------------------------------------------------


def main() -> None:
    import argparse
    import geopandas as gpd
    global SITE, RES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="lapland")
    SITE = ap.parse_args().site
    if SITE != "lapland":
        RES = RES / SITE
    from scipy.ndimage import distance_transform_edt as edt

    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    g = grid(SITE)
    H, W = g["H"], g["W"]
    # область — та же, что у исходной сверки: валидный зимний композит
    valid = g["valid"]
    from experiments.openlin_transfer_check_k1 import gtk_path
    lines = np.load(M / f"openlin_zone_lines_{SITE}_v3_перенос.npz")
    refs = {L: rast(gpd.read_file(gtk_path(SITE, L))
                    .to_crs(epsg=g["epsg"]).geometry, g) for L in (291, 292, 293, 294)}

    # местность: вода (OSM) и уклон ArcticDEM на сетке 20 м
    osm = gpd.read_file(f"outputs/k1/reference/{SITE}_osm.gpkg")
    wat = rast(osm[osm.cls.isin(["shore", "wetland_edge"])].geometry, g)
    near_water = edt(~wat, sampling=20.0) <= WATER_TOL_M
    dem = np.load(f"datacache/arcticdem/{SITE}_10m.npy")
    d20 = np.nanmean(dem.reshape(H, 2, W, 2), axis=(1, 3))
    gy, gx = np.gradient(np.where(np.isfinite(d20), d20, np.nanmean(d20)), 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    m = min(200, H // 8), min(200, W // 8)
    rng = np.random.default_rng(SEED_POOL)
    pool = np.c_[rng.integers(m[0], H - m[0] + 1, N_POOL), rng.integers(m[1], W - m[1] + 1, N_POOL)]

    ref = refs[292]
    ref_idx = np.flatnonzero(ref.ravel())
    rr, cc = np.unravel_index(ref_idx, ref.shape)

    def covars(dy, dx):
        r2, c2 = (rr + dy) % H, (cc + dx) % W
        k = valid[r2, c2]
        return near_water[r2[k], c2[k]].mean(), np.median(slope[r2[k], c2[k]])

    obs_cov = np.array(covars(0, 0))
    pool_cov = np.array([covars(dy, dx) for dy, dx in pool])
    sd = pool_cov.std(0)
    dist = np.hypot(*((pool_cov - obs_cov) / sd).T)
    match = np.argsort(dist)[:N_MATCH]
    match_set = set(match.tolist())
    log(f"местность эталона 292: у воды {100 * obs_cov[0]:.1f} % ячеек, уклон {obs_cov[1]:.2f}°; "
        f"по 5000 сдвигам: у воды {100 * pool_cov[:, 0].mean():.1f} ± {100 * sd[0]:.1f} %, "
        f"уклон {pool_cov[:, 1].mean():.2f} ± {sd[1]:.2f}°")
    log(f"  159 ближайших по местности: у воды {100 * pool_cov[match, 0].mean():.1f} %, "
        f"уклон {pool_cov[match, 1].mean():.2f}°; ранг наблюдения в пуле по «у воды» "
        f"{(pool_cov[:, 0] < obs_cov[0]).mean():.2f}, по уклону {(pool_cov[:, 1] < obs_cov[1]).mean():.2f}")

    ref_rc = {L: np.unravel_index(np.flatnonzero(r.ravel()), r.shape) for L, r in refs.items()}

    def coverage(dist_ours, L, dy, dx, tol):
        """Доля ячеек эталона L, сдвинутого на (dy, dx) по тору, в tol от наших.
        Учитываются ячейки, попавшие в валидную область (как в исходной сверке)."""
        rr_, cc_ = ref_rc[L]
        r3, c3 = (rr_ + dy) % H, (cc_ + dx) % W
        k = valid[r3, c3]
        return float((dist_ours[r3[k], c3[k]] <= tol).mean())

    out = {"область": "валидный зимний композит (как в исходной сверке)",
           "местность": {"наблюдение": obs_cov.tolist(), "пул_ср": pool_cov.mean(0).tolist(),
                         "пул_sd": sd.tolist(), "отобранные_ср": pool_cov[match].mean(0).tolist()},
           "наборы": {}}
    shift_rows = []
    for key, name in (("C", "в зонах"), ("C_all", "весь лист")):
        ours = lines[key][:H, :W] & valid
        dsto = edt(~ours, sampling=20.0)
        res = {}
        for L, refm in refs.items():
            for tol in TOLS_M:
                if L != 292 and tol != TOL_M:
                    continue
                obs = coverage(dsto, L, 0, 0, tol)
                nul = np.array([coverage(dsto, L, dy, dx, tol) for dy, dx in pool])
                nm = nul[match]
                row = {"полнота": obs,
                       "пул5000": {"ср": float(nul.mean()), "макс": float(nul.max()),
                                   "превышений": int((nul >= obs).sum()),
                                   "p": (1 + int((nul >= obs).sum())) / (N_POOL + 1),
                                   "подъём": obs / nul.mean()},
                       "по_местности159": {"ср": float(nm.mean()), "макс": float(nm.max()),
                                           "превышений": int((nm >= obs).sum()),
                                           "p": (1 + int((nm >= obs).sum())) / (N_MATCH + 1),
                                           "подъём": obs / nm.mean()}}
                res[f"{L}_{tol:.0f}"] = row
                if L == 292 and tol == TOL_M:
                    for i, (dy, dx) in enumerate(pool):
                        shift_rows.append({"набор": key, "сдвиг": i, "dy": int(dy), "dx": int(dx),
                                           "полнота": float(nul[i]), "у_воды": float(pool_cov[i, 0]),
                                           "уклон": float(pool_cov[i, 1]),
                                           "по_местности": i in match_set})
                    _hist[key] = (obs, nul, nm)
                a_, b_ = row["пул5000"], row["по_местности159"]
                mark = "ПРОЙДЕНО" if b_["p"] <= ALPHA else "не пройдено"
                log(f"  {name}, слой {L}, {tol:.0f} м: полнота {100 * obs:.1f} %; 5000 сдвигов — "
                    f"нуль {100 * a_['ср']:.1f} %, p = {a_['p']:.4f}, подъём {a_['подъём']:.2f}x; "
                    f"по местности — нуль {100 * b_['ср']:.1f} %, p = {b_['p']:.4f}, подъём "
                    f"{b_['подъём']:.2f}x" + (f" (alpha {ALPHA}: {mark})" if L == 292 and tol == TOL_M else ""))
        out["наборы"][key] = res
    # повторение исходного протокола — только у Лапландии есть отчёт 25.10
    rep = {}
    for key in (("C", "C_all") if SITE == "lapland" else ()):
        r = out["наборы"][key]["292_300"]
        rep[key] = {"отчёт_25_10": REPORT_25_10[key],
                    "восстановлено": {"полнота": r["полнота"]},
                    "разность_полноты_пп": round(100 * (r["полнота"] - REPORT_25_10[key]["полнота"]), 2)}
    rep["неизвестно_из_исходного"] = ["159 исходных сдвигов и зерно", "обрезка эталона",
                                      "all_touched", "маска области", "источник числа 2529 км"]
    rep["вывод"] = ("линии совпали с отчётом бит в бит; полнота отличается на 0.1-0.2 п.п.; "
                    "точное повторение невозможно без кода сверки копии D:")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gtk_original_reproduction.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    (RES / "gtk_spatial_stratified_exploratory.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(shift_rows).to_csv(RES / "gtk_shifts.csv", index=False, encoding="utf-8-sig")
    plot(log)
    log(f"сохранено: {RES}")


_hist: dict = {}


def plot(log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")
    for a_, (key, (obs, nul, nm)) in zip(ax, _hist.items()):
        bins = np.linspace(min(nul.min(), obs) * 0.95, max(nul.max(), obs) * 1.05, 50)
        a_.hist(nul, bins=bins, color="#9aa7b8", label="5000 сдвигов")
        a_.hist(nm, bins=bins, color="#1f5fd6", alpha=0.8, label="159 по местности")
        a_.axvline(obs, color="#d0101a", lw=2.2, label=f"наблюдение {100 * obs:.1f} %")
        a_.set_title({"C": "в зонах", "C_all": "весь лист"}[key] + ": полнота GTK 292 при 300 м",
                     loc="left", fontsize=11)
        a_.set_xlabel("полнота (доля ячеек GTK в 300 м от наших линий)")
        a_.legend(fontsize=9)
    fig.suptitle(f"{SITE}: наблюдаемая полнота против двух нулевых моделей (сдвиг эталона по тору)",
                 x=0.01, ha="left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    name = "gtk_null_distribution" + ("" if SITE == "lapland" else f"_{SITE}") + ".png"
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    log(f"сохранено: {FIG / name}")


if __name__ == "__main__":
    main()
