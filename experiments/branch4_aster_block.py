"""Ветка 4: чего стоит блок ASTER на карте, а не в важности признаков.

ЗАЧЕМ. Группа ASTER ни разу не измерялась по протоколу полос на уровне карты:
в рабочий набор 58 признаков она не входит (снимок покрывает 70% листа, и набор
собирался из слоёв со сплошным покрытием), а единственная её проверка —
перестановочная важность на `dataset_v6` (20.08.2026), где `ast_kaolin` оказался
вторым по силе признаком из 85. Отсюда вопрос ветки: даёт ли блок прирост на
карте, и не покупается ли этот прирост самим контуром съёмки.

ПОДВОХ, РАДИ КОТОРОГО ЗАВЕДЕНЫ КОНТРОЛИ. Контур ASTER — не случайная область:
он покрывает 70% листа, и внутри него лежат все 19 непредвзятых точек заверки.
Модель может выучить не спектральный признак, а сам факт «здесь съёмка есть»,
потому что граница контура коррелирует с географией, а у мишени сильный
региональный наклон. Поэтому меряется:

* вход «58 + контур ASTER» — только двоичный признак покрытия, без единого
  спектрального столбца: если он повторяет прирост блока, прирост даёт контур;
* метрики отдельно ВНУТРИ контура (там, где данные есть) и по всему листу;
* вход «только ASTER» — сколько блок может сам по себе.

Заодно измеряется цена перехода на смежную территорию: набор переноса из 37
признаков против 58 и против 58 + ASTER на одних и тех же полосах.

ПРОТОКОЛ. Тот же: 4 полосы по X + 4 по Y, разрыв 15 км, калибровка внутри
полосы. Постобработка у всех входов одна и та же и выбрана заранее в ветке 3
(ядро вдоль северо-запада 5 км x 1.25 км); круглое 2 км печатается рядом.

Выход: `outputs/metrics/branch4_aster_maps.csv` (карты целиком и внутри контура),
       `outputs/metrics/branch4_aster_folds.csv`,
       `outputs/metrics/branch4_aster_paired.csv`,
       `outputs/metrics/branch4_points_verdict.csv`,
       `data/processed/branch4_aster_cache.npz`,
       `outputs/branch4_aster.png`.
Запуск из корня: ``python -m experiments.branch4_aster_block``
(дымовой прогон: ``BRANCH4_SMOKE=1`` — одна полоса на вход).
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
                 criterial_target, features_v2)
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               poly_design, trend_share)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.nn_geo_mpm import SEED, strip_ids  # noqa: E402
from experiments.branch3_shapes import (ORIENT_KM, build_forms, detrend,  # noqa: E402
                                        score_cells)
from src import neighbourhood as nb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SMOKE = bool(os.environ.get("BRANCH4_SMOKE"))
#: Служебные столбцы покрытия: убраны из датасета v6 как риск утечки, здесь
#: тоже не используются (`ast_n_obs` — артефакт витков съёмки).
AST_SERVICE = ("ast_valid_frac", "ast_n_obs")
#: Спектральные индексы изменённости — геологически осмысленное подмножество.
AST_INDEX = ("ast_aloh", "ast_alter", "ast_carb", "ast_ferric", "ast_ferrous",
             "ast_kaolin")
POST_MAIN = "вдоль северо-западное 5 км, 4:1"
POST_ALT = "круглое 2 км"
BASE_INPUT = "58 сырые"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


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

    ast = [c for c in feat.columns
           if features_v2.feature_group(c) == "ast" and c not in AST_SERVICE]
    cols58 = [c for c in honest_features() if c in feat.columns]
    cols37 = [c for c in honest_features("перенос") if c in feat.columns]
    X58 = feat[cols58].to_numpy(dtype=float)
    XA = feat[ast].to_numpy(dtype=float)
    inside = np.isfinite(XA).all(axis=1)
    pool_f = np.flatnonzero(valid & inside)
    INPUTS = {BASE_INPUT: X58,
              f"58 + ASTER ({len(ast)})": np.column_stack([X58, XA]),
              f"58 + индексы ASTER ({len(AST_INDEX)})":
                  np.column_stack([X58, feat[list(AST_INDEX)].to_numpy(float)]),
              f"только ASTER ({len(ast)})": XA,
              "58 + контур ASTER (контроль)":
                  np.column_stack([X58, inside.astype(float)]),
              f"{len(cols37)} перенос": feat[cols37].to_numpy(dtype=float)}
    log(f"ASTER: {len(ast)} столбцов (служебные {', '.join(AST_SERVICE)} убраны); "
        f"контур покрывает {inside[valid].mean():.3f} рабочих ячеек "
        f"({pool_f.size} из {pool.size}); позитивов внутри контура "
        f"{int(y[pool_f].sum())} из {int(y[pool].sum())}")
    log("входы: " + ", ".join(f"{k} ({v.shape[1]} стлб)" for k, v in INPUTS.items()))

    # постобработка карты — форма, выбранная в ветке 3
    w = feat["lin_dens"].to_numpy(float) * feat["lin_aniso_r"].to_numpy(float)
    theta_loc, coh = nb.orientation_field(feat["lin_dir_sin"].to_numpy(float),
                                          feat["lin_dir_cos"].to_numpy(float), w,
                                          shape, ORIENT_KM * 1000.0 / meta.dx)
    coh_n = np.clip(coh / max(np.quantile(coh[valid], 0.9), 1e-6), 0.0, 1.0)
    ws = np.average(np.sin(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    wc = np.average(np.cos(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    forms = build_forms(shape, meta.dx, theta_loc, coh_n,
                        float(0.5 * np.arctan2(ws, wc)))

    strips, edges_by_ax = {}, {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        strips[ax] = np.where(valid, sid, -1)
        edges_by_ax[ax] = (sid, edges, coord)

    cache_path = config.PROCESSED_DIR / ("branch4_aster_smoke.npz" if SMOKE
                                         else "branch4_aster_cache.npz")
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
    # Оценка: по всему листу и отдельно внутри контура съёмки
    # ------------------------------------------------------------------
    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    log(f"точки заверки внутри контура ASTER: непредвзятых {int(inside[unb].sum())} "
        f"из {unb.size}, всех {int(inside[allp].sum())} из {allp.size}")

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
                    p = preds[key].astype(float)
                    p = np.where(np.isfinite(p), p, np.nan)
                    acc[te] = forms[post][1](p, mask)[te]
                    row = {"вход": inp, "постобработка": post, "полоса": f"{ax}{k}"}
                    row.update(score_cells(acc, te, y, crit, D))
                    f_cells = np.flatnonzero(mask & inside)
                    sc_f = score_cells(acc, f_cells, y, crit, D)
                    row.update({f"{k_}_контур": v for k_, v in sc_f.items()})
                    fold_rows.append(row)
                acc_ax[ax] = acc
            s = (mean_axes(acc_ax["X"], acc_ax["Y"]) if len(axes) == 2 else acc_ax["X"])
            s = np.where(np.isfinite(s), s, np.nanmin(s))
            assembled[f"{inp} | {post}"] = s
            top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
            sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
            tpr = sr >= np.quantile(sr[pool], 1 - config.CREF_AREA)
            tcr = cr >= np.quantile(cr[pool], 1 - config.CREF_AREA)
            agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
            sc_f = score_cells(s, pool_f, y, crit, D)
            map_rows.append({
                "вход": inp, "постобработка": post, "столбцов": INPUTS[inp].shape[1],
                "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                "совпадение_без_тренда": ((tpr & tcr & valid).sum()
                                          / max((tcr & valid).sum(), 1)),
                "согласие_каппа": agr["kappa"],
                "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
                "совпадение_контур": sc_f["совпадение"],
                "без_тренда_контур": sc_f["совпадение_без_тренда"],
                "roc_контур": sc_f["roc_auc"],
                "тренд_R2": float(trend_share(s[:, None], D)[0]),
                "точек_19": int((top_m & valid)[unb].sum()),
                "точек_73": int((top_m & valid)[allp].sum()),
                "новых_ячеек": int((top_m & ~top_c & valid).sum())})

    mp = pd.DataFrame(map_rows)
    mp.to_csv(ROOT / "outputs" / "metrics" / "branch4_aster_maps.csv",
              index=False, encoding="utf-8-sig")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ROOT / "outputs" / "metrics" / "branch4_aster_folds.csv",
                 index=False, encoding="utf-8-sig")
    cols_show = ["вход", "столбцов", "совпадение_верх_10", "совпадение_без_тренда",
                 "согласие_каппа", "карта_roc_auc", "совпадение_контур",
                 "без_тренда_контур", "roc_контур", "тренд_R2", "точек_19", "точек_73"]
    for post in (POST_MAIN, POST_ALT):
        print(f"\n=== Блок ASTER на карте, постобработка «{post}» ===")
        print(mp[mp["постобработка"] == post][cols_show]
              .sort_values("без_тренда_контур", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}; случайный уровень совпадения 0.10. "
          "«контур» — метрики только по ячейкам со съёмкой ASTER "
          f"({pool_f.size} ячеек).")

    pr_rows = []
    for post in (POST_MAIN, POST_ALT):
        g = folds[folds["постобработка"] == post]
        base = g[g["вход"] == BASE_INPUT].set_index("полоса")
        for inp in INPUTS:
            if inp == BASE_INPUT:
                continue
            b = g[g["вход"] == inp].set_index("полоса")
            for metric in ("совпадение", "совпадение_без_тренда", "roc_auc",
                           "совпадение_контур", "совпадение_без_тренда_контур",
                           "roc_auc_контур"):
                d = (b[metric] - base[metric]).dropna()
                if d.size < 3 or np.allclose(d, 0):
                    continue
                pr_rows.append({"постобработка": post, "вход": inp, "метрика": metric,
                                "разница": float(d.mean()),
                                "лучше_в": int((d > 0).sum()), "полос": int(d.size),
                                "p_уилкоксон": float(
                                    sps.wilcoxon(d, alternative="greater").pvalue)})
    paired = pd.DataFrame(pr_rows)
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch4_aster_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные тесты по полосам против 58 сырых (постобработка ветки 3) ===")
    sub = paired[paired["постобработка"] == POST_MAIN] if len(paired) else paired
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if len(sub) else "нет сравнений")

    main_rows = mp[mp["постобработка"] == POST_MAIN]
    keys = {f"{BASE_INPUT} | {POST_MAIN}",
            f"{main_rows.loc[main_rows['точек_73'].idxmax(), 'вход']} | {POST_MAIN}",
            f"{main_rows.loc[main_rows['без_тренда_контур'].idxmax(), 'вход']} "
            f"| {POST_MAIN}",
            f"{main_rows.loc[main_rows['совпадение_верх_10'].idxmax(), 'вход']} "
            f"| {POST_MAIN}"}
    sel = {k: assembled[k] for k in sorted(keys)}
    alpha = 0.05 / len(sel)
    prot = assessment.run_protocol({**sel, "criterial": crit}, list(sel), meta, valid,
                                   alpha, baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "branch4_points_verdict.csv",
                           index=False, encoding="utf-8-sig")
    print()
    print("=== Точки заверки (19 непредвзятых, косвенная проверка) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size          # noqa: E731
    order = (main_rows.sort_values("без_тренда_контур", ascending=False)["вход"]
             .tolist())
    show = [("Критериальный прогноз", crit, "")]
    for inp in [BASE_INPUT] + [o for o in order if o != BASE_INPUT][:3]:
        r = main_rows[main_rows["вход"] == inp].iloc[0]
        show.append((inp, assembled[f"{inp} | {POST_MAIN}"],
                     f"совпадение {r['совпадение_верх_10']:.0%}, "
                     f"без тренда {r['совпадение_без_тренда']:.0%}, "
                     f"внутри контура {r['без_тренда_контур']:.0%}"))
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 4.8))
    edge = inside.reshape(shape).astype(float)
    for ax, (title, v, sub_) in zip(np.atleast_1d(axs), show):
        ax.imshow(rank(v), cmap="magma")
        ax.contour(edge, levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(f"{title}\n{sub_}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ветка 4: блок ASTER на карте (голубым — контур съёмки, "
                 "постобработка ветки 3)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch4_aster.png", dpi=130)
    log("OK: outputs/branch4_aster.png")


if __name__ == "__main__":
    main()
