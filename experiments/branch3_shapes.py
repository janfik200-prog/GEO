"""Ветка 3, часть А: форма окрестности при сглаживании готового прогноза.

ЗАЧЕМ. Рабочая точка ветки 2 — бустинг на 58 сырых признаках плюс круглое
сглаживание карты с сигмой 2 км. Круглое ядро усредняет одинаково во все
стороны, а рудные зоны здесь вытянуты: усреднять вдоль зоны осмысленно, поперёк
— значит смешивать зону с вмещающим фоном. Здесь проверяется, даёт ли вытянутая
окрестность больше, чем круглая той же площади, и помогает ли вычитание широкого
фона (местный контраст) — при том же способе выбора, что в ветке 2: форма и
ширина окна выбираются на внутренних фолдах обучающей части, тестовая полоса при
выборе не участвует.

ЧТО ПЕРЕБИРАЕТСЯ (36 форм):
* круглое ядро, сигма 1-20 км — то, что делалось до сих пор;
* ядро, вытянутое вдоль ОБЩЕГО простирания линеаментов листа (азимут считается
  по данным), вдоль 5-20 км при вытянутости 2:1 и 4:1;
* ядро вдоль северо-западного простирания (315 градусов) — направление
  тектоно-флюидитных зон по литературе, а не по слоям критериальной формулы;
* ядро вдоль МЕСТНОГО простирания: ориентация меняется от ячейки к ячейке по
  полю направлений линеаментов (:func:`src.neighbourhood.orientation_field`);
* то же с подмешиванием круглого ядра там, где направления не согласованы;
* резкость: сглаженное на 2 км минус доля широкого фона (10 или 20 км) — прямая
  атака на региональный наклон, который в ветке 1 съел весь рекорд совпадения.

ЧЕМ ВЫБИРАЕМ. Совпадением верхних 10% ПОСЛЕ снятия тренда (как в ветке 2): из
прогноза и из критериальной оценки вычитается своя квадратичная поверхность по
координатам. Отчитываемся обеими метриками — и сырым совпадением, и без тренда.

Переобучения нет: берутся уже посчитанные прогнозы по полосам (внешние — из
`branch1_multiscale_cache.npz`, внутренние — из `branch2_inner_cache.npz`).

Выход: `outputs/metrics/branch3_shapes_inner.csv` (внутренняя поверхность выбора),
       `outputs/metrics/branch3_shapes_maps.csv` (карты целиком по всем формам),
       `outputs/metrics/branch3_shapes_folds.csv` (по полосам),
       `outputs/metrics/branch3_shapes_paired.csv` (парные тесты по 8 полосам),
       `outputs/branch3_shapes.png`.
Запуск из корня: ``python -m experiments.branch3_shapes``.
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
                 criterial_target, neighbourhood as nb)
from experiments.honest_feature_signal import N_STRIPS, poly_design  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.branch1_multiscale_ensemble import mean_axes, smooth_field  # noqa: E402
from experiments.nn_geo_mpm import strip_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: Ключ рабочей точки ветки 2 в кэшах внешних и внутренних прогнозов.
OUTER_KEY = "бустинг | 58 сырые (планка)"
INNER_KEY = "58 сырые|бустинг|по умолчанию"
#: Ширина окна вдоль структур, км (задание ветки: от 7.5 до 20 км, плюс мелкие
#: для сопоставимости с рабочей точкой 2 км).
ALONG_KM = (5.0, 10.0, 15.0, 20.0)
ISO_KM = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0)
RATIOS = (2.0, 4.0)
#: Сигма сглаживания поля направлений, км.
ORIENT_KM = 5.0
#: Северо-западное простирание тектоно-флюидитных зон по литературе.
NW_AZIMUTH = 315.0
BASE = "круглое 2 км"

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def detrend(v: np.ndarray, cells: np.ndarray, D: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(D[cells], v[cells], rcond=None)
    return v - D @ beta


def score_cells(pred: np.ndarray, cells: np.ndarray, y: np.ndarray,
                crit: np.ndarray, D: np.ndarray) -> dict:
    """Совпадение верхних 10% на наборе ячеек — как есть и после снятия тренда."""
    p, area = pred[cells], config.CREF_AREA
    ok = np.isfinite(p)
    if ok.sum() < 50 or y[cells].sum() == 0:
        return {"совпадение": np.nan, "совпадение_без_тренда": np.nan, "roc_auc": np.nan}
    top = p >= np.nanquantile(p, 1 - area)
    pr = detrend(np.where(np.isfinite(pred), pred, np.nanmedian(p)), cells, D)[cells]
    cr = detrend(crit, cells, D)[cells]
    tp, tc = pr >= np.quantile(pr, 1 - area), cr >= np.quantile(cr, 1 - area)
    return {"совпадение": float(y[cells][top].sum() / max(y[cells].sum(), 1)),
            "совпадение_без_тренда": float((tp & tc).sum() / max(tc.sum(), 1)),
            "roc_auc": float(roc_auc_score(y[cells], np.where(ok, p, np.nanmin(p))))}


def build_forms(shape, dx: float, theta_loc: np.ndarray, coh: np.ndarray,
                theta_glob: float) -> dict:
    """Формы окрестности: имя -> (семейство, функция (прогноз, маска) -> прогноз)."""
    km = lambda v: v * 1000.0 / dx                                   # noqa: E731
    forms = {}
    for s in ISO_KM:
        forms[f"круглое {s:g} км"] = ("круглое", (
            lambda p, m, s=s: smooth_field(p, shape, km(s), mask=m)))
    for name, th in (("общее простирание", theta_glob),
                     ("северо-западное", nb.theta_from_azimuth(NW_AZIMUTH))):
        for a in ALONG_KM:
            for r in RATIOS:
                forms[f"вдоль {name} {a:g} км, {r:g}:1"] = (f"вдоль {name}", (
                    lambda p, m, a=a, r=r, th=th: nb.aniso_smooth(
                        p, shape, km(a), km(a / r), th, mask=m)))
    for a in ALONG_KM:
        for r in RATIOS:
            forms[f"вдоль местного простирания {a:g} км, {r:g}:1"] = (
                "вдоль местного простирания", (
                    lambda p, m, a=a, r=r: nb.steered_smooth(
                        p, shape, km(a), km(a / r), theta_loc, mask=m)))
            forms[f"вдоль местного простирания {a:g} км, {r:g}:1 + круглое"] = (
                "вдоль местного простирания + круглое", (
                    lambda p, m, a=a, r=r: nb.steered_smooth(
                        p, shape, km(a), km(a / r), theta_loc, mask=m,
                        iso_weight=1.0 - coh)))
    for bg in (10.0, 20.0):
        for alpha in (0.5, 1.0):
            forms[f"резкость: 2 км минус {alpha:g}x фон {bg:g} км"] = ("резкость", (
                lambda p, m, bg=bg, alpha=alpha: (
                    (1.0 + alpha) * smooth_field(p, shape, km(2.0), mask=m)
                    - alpha * smooth_field(p, shape, km(bg), mask=m))))
    return forms


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

    # поле направлений структур
    w = feat["lin_dens"].to_numpy(float) * feat["lin_aniso_r"].to_numpy(float)
    theta_loc, coh = nb.orientation_field(feat["lin_dir_sin"].to_numpy(float),
                                          feat["lin_dir_cos"].to_numpy(float), w,
                                          shape, ORIENT_KM * 1000.0 / meta.dx)
    ws = np.average(np.sin(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    wc = np.average(np.cos(2 * theta_loc), weights=np.where(valid, coh, 0.0))
    theta_glob = float(0.5 * np.arctan2(ws, wc))
    coh_n = np.clip(coh / max(np.quantile(coh[valid], 0.9), 1e-6), 0.0, 1.0)
    log(f"поле направлений: покрытие {float((coh > 0).mean()):.3f}, согласованность "
        f"медиана {float(np.median(coh[valid])):.3f}; общее простирание — азимут "
        f"{float(nb.axis_azimuth_deg(theta_glob)):.0f} градусов")

    forms = build_forms(shape, meta.dx, theta_loc, coh_n, theta_glob)
    log(f"форм окрестности {len(forms)}: " + ", ".join(sorted({f[0] for f in
                                                               forms.values()})))

    strips = {}
    for ax, coord in (("X", cx), ("Y", cy)):
        sid, _ = strip_ids(coord)
        strips[ax] = np.where(valid, sid, -1)

    # ------------------------------------------------------------------
    # Внутренние фолды: какая форма выигрывает внутри обучающей части
    # ------------------------------------------------------------------
    inner = dict(np.load(config.PROCESSED_DIR / "branch2_inner_cache.npz",
                         allow_pickle=True))
    keys = [k for k in inner if k.endswith(INNER_KEY)]
    log(f"внутренних прогнозов рабочей точки {len(keys)}")

    rows = []
    for key in keys:
        ax_k, j, *_ = key.split("|")
        te = np.flatnonzero(np.isfinite(inner[key]))
        p0 = np.where(np.isfinite(inner[key]), inner[key], np.nan).astype(float)
        mask = np.zeros(y.size, dtype=bool)
        mask[te] = True
        for fname, (fam, fn) in forms.items():
            sc = score_cells(fn(p0, mask), te, y, crit, D)
            rows.append({"полоса": ax_k, "внутр_фолд": int(j), "форма": fname,
                         "семейство": fam, **sc})
        log(f"  внутр. {ax_k}<-{j}: ячеек {te.size}, форм {len(forms)}")
    inner_df = pd.DataFrame(rows)
    inner_df.to_csv(ROOT / "outputs" / "metrics" / "branch3_shapes_inner.csv",
                    index=False, encoding="utf-8-sig")

    by_fold = inner_df.groupby(["полоса", "форма"])[
        ["совпадение", "совпадение_без_тренда"]].mean()
    chosen = {f: by_fold.loc[f]["совпадение_без_тренда"].idxmax()
              for f in inner_df["полоса"].unique()}
    pooled = (inner_df.groupby(["форма", "семейство"])[
        ["совпадение", "совпадение_без_тренда", "roc_auc"]].mean()
        .sort_values("совпадение_без_тренда", ascending=False))
    pooled.to_csv(ROOT / "outputs" / "metrics" / "branch3_shapes_pooled.csv",
                  encoding="utf-8-sig")
    best_pooled = pooled.index[0][0]
    print("\n=== Внутри обучающих полос: лучшие формы (среднее по 32 внутренним "
          "фолдам) ===")
    print(pooled.head(12).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nРабочая точка ветки 2 ({BASE}) внутри обучения: "
          f"{pooled.loc[(BASE, 'круглое'), 'совпадение_без_тренда']:.3f}")
    print("\nВыбор по каждой полосе отдельно: " +
          ", ".join(f"{f}:{v}" for f, v in sorted(chosen.items())))

    # ------------------------------------------------------------------
    # Внешняя оценка: все формы на тестовых полосах
    # ------------------------------------------------------------------
    z = np.load(config.PROCESSED_DIR / "branch1_multiscale_cache.npz", allow_pickle=True)
    per_ax = {ax: np.array(z["maps"].item()[OUTER_KEY][ax], dtype=float)
              for ax in ("X", "Y")}

    def assemble(form_of_fold) -> tuple[np.ndarray, list]:
        """Собрать карту: в каждой полосе своя форма, сглаживание внутри полосы."""
        acc_ax, frows = {}, []
        for ax in ("X", "Y"):
            acc = np.full(y.size, np.nan)
            for k in range(N_STRIPS):
                mask = strips[ax] == k
                te = np.flatnonzero(mask)
                fname = form_of_fold(f"{ax}{k}")
                acc[te] = forms[fname][1](per_ax[ax], mask)[te]
                frows.append({"полоса": f"{ax}{k}", "форма": fname,
                              **score_cells(acc, te, y, crit, D)})
            acc_ax[ax] = acc
        s = mean_axes(acc_ax["X"], acc_ax["Y"])
        return np.where(np.isfinite(s), s, np.nanmin(s)), frows

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)

    map_rows, fold_rows, assembled = [], [], {}
    variants = {f: (lambda _f, f=f: f) for f in forms}
    variants["выбрано внутри обучения (по полосам)"] = lambda f: chosen[f]
    variants["выбрано внутри обучения (общая форма)"] = lambda _f: best_pooled
    for vname, fn in variants.items():
        s, frows = assemble(fn)
        assembled[vname] = s
        for r in frows:
            r["вариант"] = vname
        fold_rows += frows
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        sr, cr = detrend(s, pool, D), detrend(crit, pool, D)
        tpr = sr >= np.quantile(sr[pool], 1 - config.CREF_AREA)
        tcr = cr >= np.quantile(cr[pool], 1 - config.CREF_AREA)
        agr = crit_reference.agreement(s, crit, pool, config.CREF_AREA)
        map_rows.append({
            "вариант": vname,
            "семейство": forms[vname][0] if vname in forms else "выбор",
            "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
            "совпадение_без_тренда": ((tpr & tcr & valid).sum()
                                      / max((tcr & valid).sum(), 1)),
            "согласие_каппа": agr["kappa"],
            "карта_roc_auc": roc_auc_score(y[pool], s[pool]),
            "фолды_roc_auc": float(np.nanmean([r["roc_auc"] for r in frows])),
            "точек_19": int((top_m & valid)[unb].sum()),
            "точек_73": int((top_m & valid)[allp].sum()),
            "новых_ячеек": int((top_m & ~top_c & valid).sum())})
    mp = pd.DataFrame(map_rows).sort_values("совпадение_без_тренда", ascending=False)
    mp.to_csv(ROOT / "outputs" / "metrics" / "branch3_shapes_maps.csv",
              index=False, encoding="utf-8-sig")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ROOT / "outputs" / "metrics" / "branch3_shapes_folds.csv",
                 index=False, encoding="utf-8-sig")

    print("\n=== Карты целиком: формы окрестности (верх таблицы по совпадению без "
          "тренда) ===")
    print(mp.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nРабочая точка ветки 2 и выбранные внутри обучения:")
    keep = [BASE, "выбрано внутри обучения (по полосам)",
            "выбрано внутри обучения (общая форма)"]
    print(mp[mp["вариант"].isin(keep)].to_string(index=False,
                                                 float_format=lambda v: f"{v:.3f}"))
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}; случайный уровень совпадения 0.10")

    # парные тесты по 8 полосам против рабочей точки ветки 2
    base = folds[folds["вариант"] == BASE].set_index("полоса")
    pr_rows = []
    for vname in variants:
        if vname == BASE:
            continue
        g = folds[folds["вариант"] == vname].set_index("полоса")
        for metric in ("совпадение", "совпадение_без_тренда", "roc_auc"):
            d = (g[metric] - base[metric]).dropna()
            if d.size < 3 or np.allclose(d, 0):
                continue
            pr_rows.append({"вариант": vname, "метрика": metric,
                            "разница": float(d.mean()),
                            "лучше_в": int((d > 0).sum()), "полос": int(d.size),
                            "p_уилкоксон": float(
                                sps.wilcoxon(d, alternative="greater").pvalue)})
    paired = pd.DataFrame(pr_rows).sort_values(["метрика", "разница"],
                                               ascending=[True, False])
    paired.to_csv(ROOT / "outputs" / "metrics" / "branch3_shapes_paired.csv",
                  index=False, encoding="utf-8-sig")
    print("\n=== Парные тесты по 8 полосам против рабочей точки (круглое 2 км) ===")
    sel = paired[paired["вариант"].isin(keep + [best_pooled])]
    print(sel.to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if len(sel) else "нет сравнений")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size          # noqa: E731
    posthoc = mp.iloc[0]["вариант"]
    show = [("Критериальный прогноз", crit, "")]
    for vname in [BASE, "выбрано внутри обучения (общая форма)",
                  "выбрано внутри обучения (по полосам)", posthoc]:
        if vname not in assembled:
            continue
        r = mp[mp["вариант"] == vname].iloc[0]
        title = vname if vname != posthoc else f"лучшее задним числом:\n{posthoc}"
        show.append((title, assembled[vname],
                     f"совпадение {r['совпадение_верх_10']:.0%}, "
                     f"без тренда {r['совпадение_без_тренда']:.0%}"))
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 4.6))
    for ax, (title, v, sub) in zip(np.atleast_1d(axs), show):
        ax.imshow(rank(v), cmap="magma")
        ax.set_title(f"{title}\n{sub}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ветка 3: форма окрестности при сглаживании прогноза "
                 "(бустинг, 58 сырых признаков)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch3_shapes.png", dpi=130)
    log("OK: outputs/branch3_shapes.png")


if __name__ == "__main__":
    main()
