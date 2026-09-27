"""Ветка 3, часть Б: статистики соседства как признаки, а не только среднее.

ЗАЧЕМ. Часть А меняла форму окрестности уже на готовой карте. Здесь окрестность
входит в саму модель: к 58 сырым признакам добавляются сводки их соседства,
которых среднее не даёт, — местный контраст (значение минус сглаженное:
насколько ячейка выделяется на фоне соседей), разброс (пестрота окрестности),
квантили (низ и верх распределения в окне) — и сглаживание вытянутым ядром
вместо круглого. Окно — 10 км, то есть внутри заданного веткой диапазона
7.5-20 км.

ЧЕСТНОСТЬ. Протокол прежний: 4 полосы по X + 4 по Y, разрыв 15 км, калибровка
внутри полосы по собственным обучающим прогнозам. Признаки соседства считаются
по карте признаков целиком (меток в них нет — это то же самое, что делалось с
масштабами в ветке 1). Постобработка карты у ВСЕХ входов одна и та же и выбрана
заранее — форма, победившая внутри обучающих полос в части А; вариант с круглым
ядром 2 км (рабочая точка ветки 2) печатается рядом для сопоставления. Никакой
подстройки постобработки под конкретный вход не делается.

Выход: `outputs/metrics/branch3_features_maps.csv`,
       `outputs/metrics/branch3_features_folds.csv`,
       `outputs/metrics/branch3_features_paired.csv`,
       `data/processed/branch3_feature_cache.npz`,
       `outputs/branch3_features.png`.
Запуск из корня: ``python -m experiments.branch3_neighbourhood_features``
(дымовой прогон: ``BRANCH3_SMOKE=1`` — одна полоса на вход).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, neighbourhood as nb)
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               poly_design)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes, smooth_field  # noqa: E402
from experiments.nn_geo_mpm import SEED, strip_ids  # noqa: E402
from experiments.branch3_shapes import (NW_AZIMUTH, ORIENT_KM, build_forms,  # noqa: E402
                                        detrend, score_cells)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SMOKE = bool(os.environ.get("BRANCH3_SMOKE"))
#: Окно статистик соседства, км.
WIN_KM = 10.0
#: Постобработка: форма, выбранная внутри обучающих полос в части А, и рабочая
#: точка ветки 2 для сопоставления.
POST_MAIN = "вдоль северо-западное 5 км, 4:1"
POST_ALT = "круглое 2 км"
BASE_INPUT = "58 сырые"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def build_inputs(X: np.ndarray, shape, dx: float, theta_loc: np.ndarray) -> dict:
    """Входы: 58 сырых признаков и они же плюс сводки своего соседства."""
    sig = WIN_KM * 1000.0 / dx
    cols = range(X.shape[1])
    stack = lambda f: np.column_stack([f(X[:, j]) for j in cols])   # noqa: E731
    sm = stack(lambda v: smooth_field(v, shape, sig))
    out = {BASE_INPUT: X,
           f"58 + контраст {WIN_KM:g} км": np.column_stack([X, X - sm]),
           f"58 + разброс {WIN_KM:g} км":
               np.column_stack([X, stack(lambda v: nb.local_spread(v, shape, sig))]),
           f"58 + квантили {WIN_KM:g} км": np.column_stack(
               [X] + [stack(lambda v, q=q: nb.local_quantile(
                   v, shape, int(round(WIN_KM * 1000.0 / dx)) | 1, q))
                   for q in (0.2, 0.8)]),
           f"58 + вдоль северо-запада {WIN_KM:g} км": np.column_stack(
               [X, stack(lambda v: nb.aniso_smooth(
                   v, shape, sig, sig / 4.0, nb.theta_from_azimuth(NW_AZIMUTH)))]),
           f"58 + вдоль местных структур {WIN_KM:g} км": np.column_stack(
               [X, stack(lambda v: nb.steered_smooth(
                   v, shape, sig, sig / 4.0, theta_loc))])}
    return {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in out.items()}


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)
    pr_flat = prognoz.ravel()
    top_c = pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)

    w = feat["lin_dens"].to_numpy(float) * feat["lin_aniso_r"].to_numpy(float)
    theta_loc, coh = nb.orientation_field(feat["lin_dir_sin"].to_numpy(float),
                                          feat["lin_dir_cos"].to_numpy(float), w,
                                          shape, ORIENT_KM * 1000.0 / meta.dx)
    coh_n = np.clip(coh / max(np.quantile(coh[valid], 0.9), 1e-6), 0.0, 1.0)
    ws = np.average(np.sin(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    wc = np.average(np.cos(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    forms = build_forms(shape, meta.dx, theta_loc, coh_n,
                        float(0.5 * np.arctan2(ws, wc)))

    cols58 = [c for c in honest_features() if c in feat.columns]
    X58 = feat[cols58].to_numpy(dtype=float)
    t0 = time.time()
    INPUTS = build_inputs(X58, shape, meta.dx, theta_loc)
    log(f"входы собраны за {time.time() - t0:.0f}с: " +
        ", ".join(f"{k} ({v.shape[1]} стлб)" for k, v in INPUTS.items()))

    strips, edges_by_ax = {}, {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        strips[ax] = np.where(valid, sid, -1)
        edges_by_ax[ax] = (sid, edges, coord)

    # ------------------------------------------------------------------
    # Обучение по полосам (переиспользуется из кэша при повторном запуске)
    # ------------------------------------------------------------------
    cache_path = config.PROCESSED_DIR / ("branch3_feature_smoke.npz" if SMOKE
                                         else "branch3_feature_cache.npz")
    preds = dict(np.load(cache_path, allow_pickle=True)) if cache_path.exists() else {}
    if preds:
        log(f"из кэша {cache_path.name}: {len(preds)} прогнозов по полосам")
    axes = ("X",) if SMOKE else ("X", "Y")
    ks = range(1) if SMOKE else range(N_STRIPS)

    done = 0
    for inp, Xin in INPUTS.items():
        for ax in axes:
            sid, edges, coord = edges_by_ax[ax]
            for k in ks:
                key = f"{inp}|{ax}{k}"
                if key in preds:
                    continue
                te = np.flatnonzero((sid == k) & valid)
                tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                     (coord > edges[k + 1] + STRIP_BUFFER_M)) & valid)
                t0 = time.time()
                allx = np.concatenate([tr, te])
                m = HistBackgroundEnsemble(random_state=SEED).fit(Xin[tr], y[tr])
                p = m.predict_proba(Xin[allx])[:, 1]
                p_tr, p_te = np.sort(p[:tr.size]), p[tr.size:]
                buf = np.full(y.size, np.nan, dtype=np.float32)
                buf[te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
                preds[key] = buf
                au = roc_auc_score(y[te], p[tr.size:]) if y[te].sum() else float("nan")
                log(f"  [{inp} | {ax}{k}] обучение {tr.size}, тест {te.size}, "
                    f"ROC={au:.4f} ({time.time() - t0:.0f}с)")
                done += 1
                if done % 4 == 0:
                    np.savez_compressed(cache_path, **preds)
    np.savez_compressed(cache_path, **preds)
    log(f"прогнозов по полосам всего {len(preds)}")

    # ------------------------------------------------------------------
    # Оценка: одна и та же постобработка у всех входов
    # ------------------------------------------------------------------
    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    map_rows, fold_rows, assembled = [], [], {}
    for inp in INPUTS:
        for post in (POST_MAIN, POST_ALT):
            acc_ax = {}
            for ax in axes:
                acc = np.full(y.size, np.nan)
                for k in ks:
                    key = f"{inp}|{ax}{k}"
                    if key not in preds:
                        continue
                    mask = strips[ax] == k
                    te = np.flatnonzero(mask)
                    p = np.where(np.isfinite(preds[key]), preds[key], np.nan).astype(float)
                    acc[te] = forms[post][1](p, mask)[te]
                    fold_rows.append({"вход": inp, "постобработка": post,
                                      "полоса": f"{ax}{k}",
                                      **score_cells(acc, te, y, crit, D)})
                acc_ax[ax] = acc
            s = (mean_axes(acc_ax["X"], acc_ax["Y"]) if len(axes) == 2 else acc_ax["X"])
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            assembled[f"{inp} | {post}"] = s
            top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
            sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
            tpr = sr >= np.quantile(sr[pool], 1 - config.CREF_AREA)
            tcr = cr >= np.quantile(cr[pool], 1 - config.CREF_AREA)
            agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
            map_rows.append({
                "вход": inp, "постобработка": post, "столбцов": INPUTS[inp].shape[1],
                "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                "совпадение_без_тренда": ((tpr & tcr & valid).sum()
                                          / max((tcr & valid).sum(), 1)),
                "согласие_каппа": agr["kappa"],
                "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                "точек_19": int((top_m & valid)[unb].sum()),
                "точек_73": int((top_m & valid)[allp].sum()),
                "новых_ячеек": int((top_m & ~top_c & valid).sum())})

    mp = pd.DataFrame(map_rows)
    mp.to_csv(ROOT / "outputs" / "metrics" / "branch3_features_maps.csv",
              index=False, encoding="utf-8-sig")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ROOT / "outputs" / "metrics" / "branch3_features_folds.csv",
                 index=False, encoding="utf-8-sig")
    for post in (POST_MAIN, POST_ALT):
        print(f"\n=== Статистики соседства в признаках, постобработка «{post}» ===")
        print(mp[mp["постобработка"] == post]
              .sort_values("совпадение_без_тренда", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}; случайный уровень совпадения 0.10")

    pr_rows = []
    for post in (POST_MAIN, POST_ALT):
        g = folds[folds["постобработка"] == post]
        base = g[g["вход"] == BASE_INPUT].set_index("полоса")
        for inp in INPUTS:
            if inp == BASE_INPUT:
                continue
            b = g[g["вход"] == inp].set_index("полоса")
            for metric in ("совпадение", "совпадение_без_тренда", "roc_auc"):
                d = (b[metric] - base[metric]).dropna()
                if d.size < 3 or np.allclose(d, 0):
                    continue
                pr_rows.append({"постобработка": post, "вход": inp, "метрика": metric,
                                "разница": float(d.mean()),
                                "лучше_в": int((d > 0).sum()), "полос": int(d.size),
                                "p_уилкоксон": float(
                                    sps.wilcoxon(d, alternative="greater").pvalue)})
    paired = pd.DataFrame(pr_rows)
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch3_features_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные тесты по полосам: вход со статистиками против 58 сырых ===")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if len(paired) else "нет сравнений")

    # заверка точками: тем же протоколом, что в прошлых прогонах
    main_rows = mp[mp["постобработка"] == POST_MAIN]
    keys = {f"{BASE_INPUT} | {POST_MAIN}", f"{BASE_INPUT} | {POST_ALT}",
            f"{main_rows.loc[main_rows['точек_73'].idxmax(), 'вход']} | {POST_MAIN}",
            f"{main_rows.loc[main_rows['совпадение_верх_10'].idxmax(), 'вход']} "
            f"| {POST_MAIN}"}
    sel = {k: assembled[k] for k in sorted(keys)}
    alpha = 0.05 / len(sel)
    prot = assessment.run_protocol({**sel, "criterial": crit}, list(sel), meta, valid,
                                   alpha, baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "branch3_points_verdict.csv",
                           index=False, encoding="utf-8-sig")
    print()
    print("=== Точки заверки (19 непредвзятых, косвенная проверка) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size          # noqa: E731
    order = (mp[mp["постобработка"] == POST_MAIN]
             .sort_values("совпадение_без_тренда", ascending=False)["вход"].tolist())
    show = [("Критериальный прогноз", crit, "")]
    for inp in [BASE_INPUT] + [o for o in order if o != BASE_INPUT][:3]:
        r = mp[(mp["вход"] == inp) & (mp["постобработка"] == POST_MAIN)].iloc[0]
        show.append((inp, assembled[f"{inp} | {POST_MAIN}"],
                     f"совпадение {r['совпадение_верх_10']:.0%}, "
                     f"без тренда {r['совпадение_без_тренда']:.0%}"))
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 4.6))
    for ax, (title, v, sub) in zip(np.atleast_1d(axs), show):
        ax.imshow(rank(v), cmap="magma")
        ax.set_title(f"{title}\n{sub}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Ветка 3: сводки соседства в признаках (постобработка «{POST_MAIN}»)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch3_features.png", dpi=130)
    log("OK: outputs/branch3_features.png")


if __name__ == "__main__":
    main()
