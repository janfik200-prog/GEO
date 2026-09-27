"""Ветка 5, шаг 2: цена смены сборки и согласование шкал перед переносом.

ЗАЧЕМ. Шаг 1 (`branch5_transfer_shift.py`) показал две разные вещи. Сдвиг
распределений между листом и смежной территорией мал: ни один признак набора
переноса не выходит за обучающий диапазон больше чем в 7% ячеек. Зато одни и те
же места, собранные дважды (сетка листа и широкая сетка), дают разные значения:
сырые каналы Sentinel-2 смещены до 0.6 стандартного отклонения, а признаки
линеаментов между сборками почти не воспроизводятся.

Значит главный риск переноса — не «там другая геология», а «там другая сборка».
И этот риск, в отличие от геологического, измерим напрямую: широкая сетка
накрывает лист целиком, поэтому для 22.9 тыс. ячеек есть мишень (критериальный
top-10%) и ДВЕ независимые сборки признаков. Можно обучить модель на сборке
листа и проверить её на сборке широкой сетки в тех же самых местах — разница в
качестве и есть цена смены сборки, без единого предположения о геологии.

ЧТО ДЕЛАЕТСЯ.

1. Протокол полос (8 полос, разрыв 15 км, постобработка ветки 3). Каждая модель
   учится на сборке листа, а отложенная полоса считается тремя способами:
   А — сборка листа (потолок), Б — сборка широкой сетки как есть (наивный
   перенос), В — сборка широкой сетки после согласования шкал.

2. Согласование — поквантильное: значение признака переводится из шкалы широкой
   сборки в шкалу сборки листа по паре выборок ОДНИХ И ТЕХ ЖЕ мест. Это не
   предположение, а измеренная поправка: обе выборки описывают одну территорию,
   поэтому разница между ними — свойство сборки, а не геологии. Внутри протокола
   поправка считается только по обучающим ячейкам полосы.

3. Для смежной территории поправка, оценённая на перекрытии, применяется к
   широкой сборке целиком. Здесь допущение всё же есть — что смещение сборки
   одинаково по всей широкой сетке; оно куда слабее обычного для этой задачи
   допущения «распределения признаков на двух территориях совпадают», которое
   стирает и настоящие геологические различия.

4. Сдаваемая карта: модель обучается на всех ячейках листа сразу (метрики всё
   равно берутся из полос), усредняется по нескольким зёрнам — это убирает
   прямоугольные швы калибровки по полосам, — и применяется к согласованной
   широкой сетке.

Выход: `outputs/metrics/branch5_build_cost.csv`, `branch5_build_paired.csv`,
       `data/processed/forecast_wide_v2.parquet`, `outputs/branch5_wide_map.png`.
Запуск из корня: ``python -m experiments.branch5_alignment``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target)
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import (N_STRIPS, STRIP_BUFFER_M,  # noqa: E402
                                               poly_design)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes  # noqa: E402
from experiments.nn_geo_mpm import SEED, strip_ids  # noqa: E402
from experiments.branch3_shapes import ORIENT_KM, build_forms, score_cells  # noqa: E402
from experiments.branch4_aster_block import POST_MAIN  # noqa: E402
from src import neighbourhood as nb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEEDS = (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def qmap(src_ref: np.ndarray, tgt_ref: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Перевод значений из шкалы `src_ref` в шкалу `tgt_ref` по квантилям."""
    s = np.sort(src_ref[np.isfinite(src_ref)])
    t = tgt_ref[np.isfinite(tgt_ref)]
    if s.size < 50 or t.size < 50:
        return v
    q = np.searchsorted(s, v, side="left") / s.size
    out = np.quantile(t, np.clip(q, 0.0, 1.0))
    return np.where(np.isfinite(v), out, np.nan)


def align(src: np.ndarray, tgt: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Поквантильное согласование всех столбцов `X` по паре выборок src/tgt."""
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = qmap(src[:, j], tgt[:, j], X[:, j])
    return out


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

    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    cols = [c for c in honest_features("перенос")
            if c in feat.columns and c in wide.columns]
    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)
    in_sheet = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
                (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    tree = cKDTree(np.column_stack([wx[in_sheet], wy[in_sheet]]))
    _, idx = tree.query(np.column_stack([cx, cy]), k=1)
    w_in = wide.loc[in_sheet].reset_index(drop=True)

    Xs = feat[cols].to_numpy(dtype=float)                    # сборка листа
    Xw = w_in.loc[idx, cols].to_numpy(dtype=float)           # широкая сборка, те же места
    log(f"{len(cols)} признаков; лист {pool.size} рабочих ячеек, широкая сетка "
        f"{len(wide)} ячеек, перекрытие {int(in_sheet.sum())}")

    w = feat["lin_dens"].to_numpy(float) * feat["lin_aniso_r"].to_numpy(float)
    theta_loc, coh = nb.orientation_field(feat["lin_dir_sin"].to_numpy(float),
                                          feat["lin_dir_cos"].to_numpy(float), w,
                                          shape, ORIENT_KM * 1000.0 / meta.dx)
    coh_n = np.clip(coh / max(np.quantile(coh[valid], 0.9), 1e-6), 0.0, 1.0)
    ws = np.average(np.sin(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    wc = np.average(np.cos(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    forms = build_forms(shape, meta.dx, theta_loc, coh_n,
                        float(0.5 * np.arctan2(ws, wc)))
    post = forms[POST_MAIN][1]

    VARIANTS = ("А: сборка листа", "Б: широкая сборка как есть",
                "В: широкая сборка согласована")
    acc = {v: {ax: np.full(y.size, np.nan) for ax in ("X", "Y")} for v in VARIANTS}
    strips, fold_rows = {}, []
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, edges = strip_ids(coord)
        strips[ax] = np.where(valid, sid, -1)
        for k in range(N_STRIPS):
            te = np.flatnonzero((sid == k) & valid)
            tr = np.flatnonzero(((coord < edges[k] - STRIP_BUFFER_M) |
                                 (coord > edges[k + 1] + STRIP_BUFFER_M)) & valid)
            t0 = time.time()
            m = HistBackgroundEnsemble(random_state=SEED).fit(Xs[tr], y[tr])
            # поправка сборки — только по обучающим ячейкам полосы
            Xw_al = np.array(Xs, dtype=float)
            Xw_al[te] = align(Xw[tr], Xs[tr], Xw[te])
            mask = strips[ax] == k
            for v, Xv in zip(VARIANTS, (Xs, Xw, Xw_al)):
                p_tr = np.sort(m.predict_proba(Xs[tr])[:, 1])
                p_te = m.predict_proba(Xv[te])[:, 1]
                cal = np.full(y.size, np.nan)
                cal[te] = np.searchsorted(p_tr, p_te, side="left") / max(p_tr.size, 1)
                acc[v][ax][te] = post(cal, mask)[te]
                row = {"вариант": v, "полоса": f"{ax}{k}",
                       "roc_auc_сырой": (roc_auc_score(y[te], p_te)
                                         if y[te].sum() else np.nan)}
                row.update(score_cells(acc[v][ax], te, y, crit, D))
                fold_rows.append(row)
            log(f"  [{ax}{k}] обучение {tr.size}, тест {te.size}, ROC "
                + ", ".join(f"{v.split(':')[0]}={r['roc_auc_сырой']:.4f}"
                            for v, r in zip(VARIANTS, fold_rows[-3:]))
                + f" ({time.time() - t0:.0f}с)")

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    pr_flat = prognoz.ravel()
    top_c = (pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)) & valid

    map_rows = {}
    for v in VARIANTS:
        s = mean_axes(acc[v]["X"], acc[v]["Y"])
        s = np.where(np.isfinite(s), s, np.nanmin(s))
        map_rows[v] = s
        sc = score_cells(s, pool, y, crit, D)
        top_m = (s >= np.quantile(s[pool], 1 - config.CREF_AREA)) & valid
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        map_rows[v] = s
        fold_rows.append({"вариант": v, "полоса": "КАРТА",
                          "roc_auc_сырой": np.nan, **sc,
                          "каппа": agr["kappa"],
                          "точек_19": int(top_m[unb].sum()),
                          "точек_73": int(top_m[allp].sum())})
    res = pd.DataFrame(fold_rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "branch5_build_cost.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Цена смены сборки: те же места, те же мишени, другая сборка ===")
    print(res[res["полоса"] == "КАРТА"]
          .drop(columns=["полоса", "roc_auc_сырой"])
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}")
    print("\nСреднее по 8 полосам:")
    print(res[res["полоса"] != "КАРТА"].groupby("вариант")
          [["roc_auc_сырой", "roc_auc", "совпадение", "совпадение_без_тренда"]]
          .mean().round(4).to_string())

    g = res[res["полоса"] != "КАРТА"]
    base = g[g["вариант"] == VARIANTS[0]].set_index("полоса")
    pr_rows = []
    for a, b in ((VARIANTS[1], VARIANTS[0]), (VARIANTS[2], VARIANTS[0]),
                 (VARIANTS[2], VARIANTS[1])):
        ga = g[g["вариант"] == a].set_index("полоса")
        gb = g[g["вариант"] == b].set_index("полоса")
        for metric in ("roc_auc_сырой", "совпадение", "совпадение_без_тренда"):
            d = (ga[metric] - gb[metric]).dropna()
            if d.size < 3 or np.allclose(d, 0):
                continue
            pr_rows.append({"что": a, "против": b, "метрика": metric,
                            "разница": float(d.mean()),
                            "лучше_в": int((d > 0).sum()), "полос": int(d.size),
                            "p_уилкоксон": float(
                                sps.wilcoxon(d, alternative="greater").pvalue)})
    paired = pd.DataFrame(pr_rows)
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch5_build_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные сравнения по полосам ===")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------
    # Сдаваемая карта: обучение на всех ячейках листа, усреднение по зёрнам
    # ------------------------------------------------------------------
    Xw_all = wide[cols].to_numpy(dtype=float)
    Xw_all_al = align(Xw[valid], Xs[valid], Xw_all)
    out_rng = []
    for j, c in enumerate(cols):
        af = Xs[valid, j]
        af = af[np.isfinite(af)]
        lo, hi = af.min(), af.max()
        for name, M in (("как есть", Xw_all), ("согласована", Xw_all_al)):
            b = M[~in_sheet, j]
            b = b[np.isfinite(b)]
            out_rng.append({"признак": c, "сборка": name,
                            "вне_диапазона": float(np.mean((b < lo) | (b > hi)))})
    orng = pd.DataFrame(out_rng).groupby("сборка")["вне_диапазона"].mean()
    log("доля ячеек смежной территории вне обучающего диапазона (среднее по "
        f"признакам): как есть {orng['как есть']:.4f}, "
        f"после согласования {orng['согласована']:.4f}")

    scores = []
    for sd in SEEDS:
        t0 = time.time()
        m = HistBackgroundEnsemble(random_state=sd).fit(Xs[valid], y[valid])
        p_tr = np.sort(m.predict_proba(Xs[valid])[:, 1])
        p = m.predict_proba(Xw_all_al)[:, 1]
        scores.append(np.searchsorted(p_tr, p, side="left") / p_tr.size)
        log(f"  зерно {sd}: широкая сетка посчитана ({time.time() - t0:.0f}с)")
    score = np.mean(scores, axis=0)
    spread = np.std(scores, axis=0)
    log(f"разброс между зёрнами: медиана {np.median(spread):.4f}, "
        f"90-й процентиль {np.quantile(spread, 0.9):.4f} (шкала 0-1)")

    # постобработка рабочей точки ветки 3: ядро вдоль северо-запада 5 x 1.25 км
    km = 1000.0 / WIDE_META.dx
    score_sm = nb.aniso_smooth(score, WIDE_META.shape, 5 * km, 1.25 * km,
                               nb.theta_from_azimuth(315.0))
    log("сглаживание широкой карты ядром рабочей точки: спирмен с несглаженной "
        f"{sps.spearmanr(score, score_sm).statistic:.3f}")

    out = wide[["row", "col", "x", "y"]].copy()
    out["score"] = score.astype(np.float32)
    out["score_сглаженный"] = score_sm.astype(np.float32)
    out["разброс_зёрен"] = spread.astype(np.float32)
    out["в_габаритах_листа"] = in_sheet
    out.to_parquet(config.PROCESSED_DIR / "forecast_wide_v2.parquet", index=False)
    old = pd.read_parquet(config.PROCESSED_DIR / "forecast_wide.parquet")
    r = sps.spearmanr(old["score"].to_numpy(), score).statistic
    t_new = score >= np.quantile(score, 0.9)
    t_old = old["score"].to_numpy() >= np.quantile(old["score"].to_numpy(), 0.9)
    log(f"против прежней карты `forecast_wide`: спирмен {r:.3f}, общих ячеек в "
        f"верхних 10% {100 * (t_new & t_old).sum() / max(t_new.sum(), 1):.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [WIDE_META.x0, WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx,
           WIDE_META.y0, WIDE_META.y_top]
    fig, axs = plt.subplots(1, 4, figsize=(23, 6.6))
    for ax, (title, v, cmap) in zip(axs, (
            ("Прежняя карта переноса (без согласования)",
             old["score"].to_numpy(), "viridis"),
            ("Согласованная сборка, среднее по 5 зёрнам", score, "viridis"),
            ("То же после сглаживания рабочей точки", score_sm, "viridis"),
            ("Разброс между зёрнами", spread, "magma"))):
        im = ax.imshow(v.reshape(WIDE_META.shape), origin="upper", cmap=cmap,
                       extent=ext)
        ax.plot([cx.min(), cx.max(), cx.max(), cx.min(), cx.min()],
                [cy.min(), cy.min(), cy.max(), cy.max(), cy.min()],
                color="white", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Ветка 5: перенос на смежную территорию (белым — границы "
                 "основного листа, где есть критериальный расчёт)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch5_wide_map.png", dpi=130)
    log("OK: outputs/branch5_wide_map.png, data/processed/forecast_wide_v2.parquet")


if __name__ == "__main__":
    main()
