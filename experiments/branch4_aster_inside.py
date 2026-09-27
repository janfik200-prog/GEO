"""Ветка 4, шаг 2: ASTER там, где съёмка есть — обучение и оценка внутри контура.

ЗАЧЕМ. На шаге 1 (`branch4_aster_block.py`) блок ASTER ухудшил карту, но этот
результат нельзя приписать спектрам: за пределами контура съёмки все 24 столбца
пропущены, а бустинг обрабатывает пропуск как отдельную ветку решения — значит,
вместе с блоком модель получает 24 копии признака «здесь съёмки нет». Контроль
из одного двоичного столбца контура уже воспроизвёл почти всю просадку.

Здесь подвох снят целиком: рабочая область сужена до контура (15 956 ячеек),
модели и обучаются, и проверяются только на ней. Пропусков в блоке нет, входы
отличаются ровно спектральными столбцами. Это и есть честная проверка «даёт ли
ASTER что-то там, где он снят», и одновременно вопрос о максимальной модели для
листа против переносимого набора.

Полосы, разрыв 15 км, калибровка внутри полосы и постобработка — те же, что на
шаге 1; метрики считаются на ячейках контура.

Выход: `outputs/metrics/branch4_inside_maps.csv`, `..._folds.csv`,
       `..._paired.csv`, `data/processed/branch4_inside_cache.npz`.
Запуск из корня: ``python -m experiments.branch4_aster_inside``
"""

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
                 criterial_target, features_v2)
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               poly_design)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.nn_geo_mpm import SEED, strip_ids  # noqa: E402
from experiments.branch3_shapes import ORIENT_KM, build_forms, score_cells  # noqa: E402
from experiments.branch4_aster_block import AST_INDEX, AST_SERVICE, POST_ALT  # noqa: E402
from experiments.branch4_aster_block import BASE_INPUT, POST_MAIN  # noqa: E402
from src import neighbourhood as nb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    y = criterial_target.training_positives(prognoz).astype(int)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)

    ast = [c for c in feat.columns
           if features_v2.feature_group(c) == "ast" and c not in AST_SERVICE]
    cols58 = [c for c in honest_features() if c in feat.columns]
    cols37 = [c for c in honest_features("перенос") if c in feat.columns]
    X58 = feat[cols58].to_numpy(dtype=float)
    XA = feat[ast].to_numpy(dtype=float)
    inside = np.isfinite(XA).all(axis=1)
    work = valid & inside                      # рабочая область — только контур
    pool = np.flatnonzero(work)
    INPUTS = {BASE_INPUT: X58,
              f"58 + ASTER ({len(ast)})": np.column_stack([X58, XA]),
              f"58 + индексы ASTER ({len(AST_INDEX)})":
                  np.column_stack([X58, feat[list(AST_INDEX)].to_numpy(float)]),
              f"только ASTER ({len(ast)})": XA,
              f"{len(cols37)} перенос": feat[cols37].to_numpy(dtype=float)}
    log(f"рабочая область — контур ASTER: {pool.size} ячеек, позитивов "
        f"{int(y[pool].sum())} ({y[pool].mean():.3f}); пропусков в блоке нет")

    w = feat["lin_dens"].to_numpy(float) * feat["lin_aniso_r"].to_numpy(float)
    theta_loc, coh = nb.orientation_field(feat["lin_dir_sin"].to_numpy(float),
                                          feat["lin_dir_cos"].to_numpy(float), w,
                                          shape, ORIENT_KM * 1000.0 / meta.dx)
    coh_n = np.clip(coh / max(np.quantile(coh[work], 0.9), 1e-6), 0.0, 1.0)
    ws = np.average(np.sin(2 * theta_loc), weights=np.where(work, coh, 0.0))
    wc = np.average(np.cos(2 * theta_loc), weights=np.where(work, coh, 0.0))
    forms = build_forms(shape, meta.dx, theta_loc, coh_n,
                        float(0.5 * np.arctan2(ws, wc)))

    strips, edges_by_ax = {}, {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        strips[ax] = np.where(work, sid, -1)
        edges_by_ax[ax] = (sid, edges, coord)

    cache_path = config.PROCESSED_DIR / "branch4_inside_cache.npz"
    preds = dict(np.load(cache_path, allow_pickle=True)) if cache_path.exists() else {}
    done = 0
    for inp, Xin in INPUTS.items():
        for ax in ("X", "Y"):
            sid, edges, coord = edges_by_ax[ax]
            for k in range(N_STRIPS):
                key = f"{inp}|{ax}{k}"
                if key in preds:
                    continue
                te = np.flatnonzero((sid == k) & work)
                tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                     (coord > edges[k + 1] + STRIP_BUFFER_M)) & work)
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

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), work)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), work)
    pr_flat = prognoz.ravel()
    top_c = (pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)) & work

    map_rows, fold_rows = [], []
    for inp in INPUTS:
        for post in (POST_MAIN, POST_ALT):
            acc_ax = {}
            for ax in ("X", "Y"):
                acc = np.full(y.size, np.nan)
                for k in range(N_STRIPS):
                    mask = strips[ax] == k
                    te = np.flatnonzero(mask)
                    p = preds[f"{inp}|{ax}{k}"].astype(float)
                    acc[te] = forms[post][1](np.where(np.isfinite(p), p, np.nan),
                                             mask)[te]
                    row = {"вход": inp, "постобработка": post, "полоса": f"{ax}{k}"}
                    row.update(score_cells(acc, te, y, crit, D))
                    fold_rows.append(row)
                acc_ax[ax] = acc
            s = mean_axes(acc_ax["X"], acc_ax["Y"])
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            sc = score_cells(s, pool, y, crit, D)
            top_m = (s >= np.quantile(s[pool], 1 - config.CREF_AREA)) & work
            agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
            map_rows.append({
                "вход": inp, "постобработка": post, "столбцов": INPUTS[inp].shape[1],
                "совпадение_верх_10": sc["совпадение"],
                "совпадение_без_тренда": sc["совпадение_без_тренда"],
                "согласие_каппа": agr["kappa"], "карта_roc_auc": sc["roc_auc"],
                "точек_19": int(top_m[unb].sum()), "точек_73": int(top_m[allp].sum()),
                "новых_ячеек": int((top_m & ~top_c).sum())})

    mp = pd.DataFrame(map_rows)
    mp.to_csv(ROOT / "outputs" / "metrics" / "branch4_inside_maps.csv",
              index=False, encoding="utf-8-sig")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ROOT / "outputs" / "metrics" / "branch4_inside_folds.csv",
                 index=False, encoding="utf-8-sig")
    for post in (POST_MAIN, POST_ALT):
        print(f"\n=== Внутри контура ASTER, постобработка «{post}» ===")
        print(mp[mp["постобработка"] == post]
              .drop(columns=["постобработка"])
              .sort_values("совпадение_без_тренда", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта внутри контура: точек 19 -> {int(top_c[unb].sum())}, "
          f"из {allp.size} всех -> {int(top_c[allp].sum())}; "
          f"случайный уровень совпадения 0.10.")

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
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch4_inside_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные тесты по полосам против 58 сырых (внутри контура) ===")
    print(paired[paired["постобработка"] == POST_MAIN]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("готово")


if __name__ == "__main__":
    main()
