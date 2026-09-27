"""Итоговая карта прогноза: лучший измеренный вариант и его новые площади.

Отличие от `honest_forecast_main.py` — не в протоколе (он тот же: сплошной
прогноз по полосам с разрывом 15 км, калибровка по обучающему распределению,
среднее схем X и Y), а в модели. Здесь взят вариант, победивший в
`honest_raise_agreement.py`: 58 признаков плюс их гауссово сглаженные версии
на 2/5/10 км, а готовый прогноз сглажен на 10 км.

Почему это не подгонка. Мишень крупноблочная и плавная, а признаки
попиксельные, 500 м: приведение масштаба модели к масштабу явления —
содержательный шаг, а не подбор под метрику. Сглаживается признак и уже
посчитанный прогноз, но никогда мишень, и обучение по-прежнему не ближе 15 км
от предсказываемой ячейки. Масштаб 20 км проверен и хуже (34.6% против
44.1%), координаты в признаках не помогают вовсе (43.3%) — то есть модель
ничего не выигрывает от знания «где», весь результат несут признаки.

Ограничение, из-за которого сглаживание нельзя выбирать по одному согласию.
Первый прогон (постобработка 10 км) дал карту, которая целиком является
плавным трендом: поверхность из 64 гауссовых базисов по одним координатам
воспроизводит её с R² = 0.9999, тогда как для карты на 58 признаках R² = 0.82,
а для самой критериальной карты 0.93. Согласие при этом росло (44% против 28%),
но геологически такая карта бесполезна: это один эллипс 25 x 15 км, «перспективен
центр листа», а не оконтуренные узлы. Поэтому здесь меряется не только согласие,
но и доля тренда, и рабочей считается конфигурация, у которой структуры не
меньше, чем у критериальной карты (R² <= 0.93): нельзя выдавать за прогноз
то, что подробнее исходной карты не стало.

Продукт скрипта — не совпадение с критериальной картой (цель проекта, уточнённая
02.09.2026, — прогноз, а не копия формулы), а ДВЕ площади:

* согласованная — верхние 10% обоих методов сразу; там, где новый метод
  подтверждает старый, разговор с геологами короткий;
* новая — верхние 10% модели вне верхних 10% формулы; это и есть то, ради
  чего метод строился, и именно её нужно отдавать на экспертную оценку
  (блок 9 технологической схемы).

Выход: `data/processed/forecast_main_final.parquet`,
`outputs/honest_forecast_final_map.png`,
`outputs/metrics/honest_forecast_final.csv`.

Запуск из корня репозитория: ``python -m experiments.honest_forecast_final``.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src import cell_mask, config, crit_reference, criterial_target  # noqa: E402
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402
from experiments.honest_forecast_points import SCALES_KM  # noqa: E402
from experiments.honest_raise_agreement import score_variant, smooth_field  # noqa: E402

#: Проверяемые уровни постобработки прогноза, км (0 — без сглаживания).
POST_KM = (0.0, 2.0, 5.0, 10.0)
#: Порог доли тренда: структуры должно быть не меньше, чем в критериальной карте.
TREND_MAX = 0.93

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def trend_share(s: np.ndarray, basis: np.ndarray, valid: np.ndarray) -> float:
    """Доля карты, объяснимая плавной поверхностью по одним координатам.

    Считается внутривыборочно и намеренно: вопрос не «предсказуема ли карта по
    координатам вне обучения», а «остаётся ли в ней хоть что-то, кроме
    крупномасштабного тренда». R² около единицы означает, что карта и есть
    поверхность вида «пятно посреди листа», сколько бы признаков в неё ни вошло.
    """
    m = valid & np.isfinite(s)
    coef, *_ = np.linalg.lstsq(basis[m], s[m], rcond=None)
    return float(1 - np.var(s[m] - basis[m] @ coef) / np.var(s[m]))


def main() -> None:
    meta, prognoz_grid = criterial_target.load_prognoz_grid()
    prognoz = prognoz_grid.ravel()
    y = criterial_target.training_positives(prognoz_grid).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    cols = [c for c in honest_features() if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    shape = (meta.prf, meta.pic)

    parts = [X]
    for km in SCALES_KM:
        sig = km * 1000.0 / meta.dx
        parts.append(np.column_stack([smooth_field(X[:, j], shape, sig)
                                      for j in range(X.shape[1])]))
    X_ms = np.column_stack(parts)
    log(f"{len(cols)} признаков на масштабах {SCALES_KM} -> {X_ms.shape[1]} столбцов")

    s_raw = score_variant(X_ms, y, cx, cy)
    log("прогноз по полосам посчитан")

    basis = np.column_stack([np.ones(len(prognoz)), rbf_basis(cx, cy, RBF_GRID)])
    top_c = prognoz <= np.quantile(prognoz[pool], config.CREF_AREA)
    rows, fields = [], {}
    for km in POST_KM:
        s = (s_raw if km == 0 else
             smooth_field(s_raw, shape, km * 1000.0 / meta.dx))
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        agr = crit_reference.agreement(s, crit_reference.criterial_score(prognoz),
                                       pool, config.CREF_AREA)
        tr = trend_share(s, basis, valid)
        rows.append({
            "постобработка_км": km, "признаков": X_ms.shape[1],
            "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
            "доля_тренда_R2": tr, "структура_ок": tr <= TREND_MAX,
            "roc_auc": roc_auc_score(y[pool], s[pool]),
            "pr_auc": average_precision_score(y[pool], s[pool]),
            "capture": agr["capture"], "kappa": agr["kappa"], "rho": agr["rho"],
            "ячеек_новых": int((top_m & ~top_c & valid).sum()),
        })
        fields[km] = s
        log(f"  сглаживание {km:4.0f} км: совпадение={rows[-1]['совпадение_верх_10']:.1%} "
            f"тренд R2={tr:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "honest_forecast_final.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Согласие против сохранности структуры (обучение не ближе 15 км) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"порог структуры: доля тренда <= {TREND_MAX} (столько же в критериальной карте)")

    ok = res[res["структура_ок"]]
    if ok.empty:
        raise RuntimeError("все варианты выродились в тренд — карту выпускать нельзя")
    best = ok.loc[ok["совпадение_верх_10"].idxmax()]
    km = best["постобработка_км"]
    s = fields[km]
    log(f"выбрана постобработка {km:.0f} км: лучшее согласие среди сохранивших структуру")

    top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
    both = top_c & top_m & valid
    only_m = top_m & ~top_c & valid
    only_c = top_c & ~top_m & valid

    out = pd.DataFrame({"row": np.repeat(np.arange(meta.prf), meta.pic),
                        "col": np.tile(np.arange(meta.pic), meta.prf),
                        "x": cx, "y": cy, "score": s, "prognoz": prognoz,
                        "top_модель": top_m, "top_формула": top_c,
                        "новая_площадь": only_m, "valid": valid})
    out_path = config.PROCESSED_DIR / "forecast_main_final.parquet"
    out.to_parquet(out_path, index=False)
    log(f"OK: {out_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    ext = [meta.x0, meta.x0 + meta.pic * meta.dx, meta.y0, meta.y_top]
    axes[0].imshow(np.where(valid, prognoz, np.nan).reshape(shape), origin="upper",
                   cmap="magma_r", extent=ext)
    axes[0].set_title("Критериальный метод\n(разломы, фации, палеодолины)")
    axes[1].imshow(np.where(valid, s, np.nan).reshape(shape), origin="upper",
                   cmap="magma", extent=ext)
    axes[1].set_title(f"Прогноз модели, {X_ms.shape[1]} признаков\n"
                      f"постобработка {km:.0f} км, обучение не ближе 15 км")
    code = np.where(both, 3, np.where(only_c, 2, np.where(only_m, 1, 0)))
    axes[2].imshow(code.reshape(shape), origin="upper", extent=ext, vmin=0, vmax=3,
                   cmap=ListedColormap(["#1b1b2b", "#d95f02", "#4f7fbf", "#f2d024"]))
    axes[2].set_title(f"Совпадение верхних 10%: {best['совпадение_верх_10']:.0%}")
    km2 = int(only_m.sum()) * (meta.dx * meta.dy) / 1e6
    axes[2].legend(handles=[
        Patch(color="#f2d024", label=f"оба метода — {int(both.sum())} яч."),
        Patch(color="#4f7fbf", label=f"только формула — {int(only_c.sum())} яч."),
        Patch(color="#d95f02", label=f"новая площадь — {int(only_m.sum())} яч. "
                                     f"({km2:.0f} км²)"),
    ], loc="lower left", fontsize=8)
    for ax in axes:
        ax.set_xlabel("X, м")
    axes[0].set_ylabel("Y, м")
    fig.tight_layout()
    fig_path = ROOT / "outputs" / "honest_forecast_final_map.png"
    fig.savefig(fig_path, dpi=130)
    log(f"OK: {fig_path}")


if __name__ == "__main__":
    main()
