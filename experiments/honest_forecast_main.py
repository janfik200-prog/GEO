"""Карта прогноза по основному листу, где каждая ячейка предсказана «вслепую».

Что здесь считается честным. Ячейка получает прогноз от модели, которая
обучалась только на ячейках, удалённых от неё не менее чем на `STRIP_BUFFER_M`
(15 км). Модель не видела не только саму ячейку, но и всю её окрестность,
поэтому не может подставить ответ соседа. Разбиение — полосы: четыре по X
покрывают лист целиком без пересечений, четыре по Y тоже, и каждая схема даёт
сплошную карту. Итоговый прогноз — среднее калиброванных схем.

Зачем так. Прежний прогон (`train_test_main_sheet.py`, случайная CV по ячейкам)
дал совпадение с критериальным прогнозом 92%, но проверка 02.09.2026 показала,
что столько же даёт модель на ОДНИХ КООРДИНАТАХ: при плотной сетке 500 м
случайно отложенная ячейка почти всегда окружена обучающими, и измерялась
пространственная интерполяция. Здесь этой лазейки нет.

Признаки — набор из `honest_feature_selection.py` и
`honest_feature_service_check.py`: 58 столбцов, группы
`s2 s2raw l8 s1 ter relief2 lin dist` (спутниковые съёмки, рельеф, линеаменты,
гидросеть). Что и почему выброшено:

* `gm` и `pf` (гравика, потенциальные поля) — не как геологически неважные, а
  как непереносимые: `gm_gr_all` объясняется координатами на 87%, модель берёт
  такой признак вместо координат, и на отложенной полосе группа `gm` даёт
  ROC-AUC 0.320 — ниже случайного. Замена абсолютного значения на локальную
  аномалию (минус сглаженное поле 20 км) их не спасает — 0.361.
* `*_n_obs`, `*_valid_frac` — служебные столбцы съёмки. Они не геология, но
  сами по себе дают ROC-AUC 0.734, и весь этот вклад приходится на `n_obs`
  (0.738), то есть на геометрию витков спутника, а `valid_frac` не даёт ничего
  (0.506). Полосы витков протяжённые и с резкими границами, поэтому переносятся
  между полосами обучения не хуже геологии — типичный артефакт, который легко
  принять за результат.

Измеренная переносимость (полосы + буфер 15 км, одна модель,
`honest_final_control.py`): набор 0.753 против 0.663 у гладкого
пространственного поля из 64 гауссовых базисов — модели, знающей только «где»
и ни одного признака. Собственный вклад признаков сверх географии
+0.090 ROC-AUC. Именно эта планка, а не случайные 0.5, и вынесена в таблицу
результатов строкой «ТОЛЬКО география».

Выход: `data/processed/forecast_main_honest.parquet`,
       `outputs/metrics/honest_forecast_main.csv`,
       `outputs/honest_forecast_main_map.png`.
Запуск из корня репозитория: ``python -m experiments.honest_forecast_main``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import assessment, cell_mask, config, crit_reference, criterial_target  # noqa: E402
from src.model import HistBackgroundEnsemble  # noqa: E402
from experiments.honest_feature_signal import N_STRIPS, STRIP_BUFFER_M  # noqa: E402
from experiments.honest_final_control import RBF_GRID, rbf_basis  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def strip_ids(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(coord.min(), coord.max(), N_STRIPS + 1)
    return np.clip(np.digitize(coord, edges[1:-1]), 0, N_STRIPS - 1), edges


def blind_scores(X: np.ndarray, y: np.ndarray, coord: np.ndarray, tag: str):
    """Сплошной прогноз по полосам одной оси: полоса предсказана моделью вне неё.

    Возвращает сырые вероятности и калиброванные. Калибровка нужна потому, что
    полосы считают разные модели, и их вероятности несопоставимы между собой:
    склеенная карта ранжирует ячейки внутри полосы верно, а между полосами —
    нет, из-за чего общий top-10% собирается почти из одной полосы. Значение
    переводится в квантиль относительно распределения ЭТОЙ ЖЕ модели на её
    ОБУЧАЮЩИХ ячейках. Тестовая полоса в калибровку не входит — иначе это была
    бы подсказка о доле перспективных площадей в ней (а она сильно разная: от
    28 позитивов в одной полосе до 1194 в другой).
    """
    sid, edges = strip_ids(coord)
    raw = np.full(len(y), np.nan)
    cal = np.full(len(y), np.nan)
    for k in range(N_STRIPS):
        te = sid == k
        tr = (coord < edges[k] - STRIP_BUFFER_M) | (coord > edges[k + 1] + STRIP_BUFFER_M)
        model = HistBackgroundEnsemble(random_state=config.CRIT_SEED)
        model.fit(X[tr], y[tr])
        p_te = model.predict_proba(X[te])[:, 1]
        p_tr = np.sort(model.predict_proba(X[tr])[:, 1])
        raw[te] = p_te
        cal[te] = np.searchsorted(p_tr, p_te, side="left") / p_tr.size
        log(f"  [{tag} {k}] обучение {int(tr.sum())} ячеек ({int(y[tr].sum())} поз.), "
            f"тест {int(te.sum())} ({int(y[te].sum())} поз.), "
            f"ROC-AUC={roc_auc_score(y[te], p_te):.4f}")
    return raw, cal


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    feat_all = criterial_target.training_features("v5_rebuilt")
    spec = json.loads((config.PROCESSED_DIR / "honest_feature_set.json")
                      .read_text(encoding="utf-8"))
    cols = [c for c in spec["признаки"] if c in feat_all.columns]
    X = feat_all[cols].to_numpy(dtype=float)

    # Пул оценки считается по полному набору столбцов — так же, как в прежних
    # прогонах, иначе доля пропусков считалась бы от другого знаменателя.
    valid = cell_mask.build_valid_mask(feat_all)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    shape = (meta.prf, meta.pic)

    log(f"признаков {len(cols)} (набор переносимых, {spec['дата']}), "
        f"позитивов {int(y.sum())} из {y.size}; пул оценки {pool.size}")
    log(f"схема: полосы по {N_STRIPS} на ось, буфер {STRIP_BUFFER_M/1000:.0f} км")

    # Кэш прогонов по полосам: обучение 8 ансамблей — 6 минут, а всё, что ниже,
    # это уже пересчитываемая за секунды отчётность.
    cache = config.PROCESSED_DIR / "honest_blind_scores.npz"
    if cache.exists():
        z = np.load(cache)
        sx, sy, cxs, cys = z["sx"], z["sy"], z["cx"], z["cy"]
        gx_, gy_ = z["gx"], z["gy"]
        log(f"прогнозы по полосам взяты из кэша {cache.name}")
    else:
        sx, cxs = blind_scores(X, y, cx, "X")
        sy, cys = blind_scores(X, y, cy, "Y")
        # Та же процедура на решётке гауссовых базисов: модель, знающая только
        # «где», без единого признака. Это планка «сколько даёт одна география» —
        # мишень крупноблочная и плавная, поэтому часть совпадения достижима
        # вообще без данных, и вычитать надо именно её, а не случайные 10%.
        RB = rbf_basis(cx, cy, RBF_GRID)
        _, gx_ = blind_scores(RB, y, cx, "гео-X")
        _, gy_ = blind_scores(RB, y, cy, "гео-Y")
        np.savez(cache, sx=sx, sy=sy, cx=cxs, cy=cys, gx=gx_, gy=gy_)
    rank = lambda v: sps.rankdata(v) / v.size  # noqa: E731
    s_avg = (cxs + cys) / 2
    s_geo = (gx_ + gy_) / 2

    rows = []
    for name, s in (("полосы по X, сырые", sx), ("полосы по Y, сырые", sy),
                    ("полосы по X, калибр.", cxs), ("полосы по Y, калибр.", cys),
                    ("ТОЛЬКО география (планка)", s_geo),
                    ("среднее калибр. (итог)", s_avg)):
        # agreement ждёт полные массивы по сетке, pool — индексы внутрь них.
        agr = crit_reference.agreement(s, crit_reference.criterial_score(prognoz),
                                       pool, config.CREF_AREA)
        top_c = prognoz.ravel() <= np.quantile(prognoz.ravel()[pool], config.CREF_AREA)
        top_m = s >= np.quantile(s[pool], 1 - config.CREF_AREA)
        rows.append({"схема": name,
                     "roc_auc": roc_auc_score(y[pool], s[pool]),
                     "pr_auc": average_precision_score(y[pool], s[pool]),
                     "agr_capture": agr["capture"], "agr_kappa": agr["kappa"],
                     "agr_rho": agr["rho"],
                     "overlap_top10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                     "IoU": (top_c & top_m).sum() / max((top_c | top_m).sum(), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "metrics" / "honest_forecast_main.csv", index=False)
    print("\n=== Прогноз вслепую против критериального (весь лист) ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nБаза: ROC-AUC случайного 0.5000, PR-AUC {y[pool].mean():.4f}, "
          f"совпадение top-10% случайного {config.CREF_AREA:.1%}, capture 1.0")

    # Косвенная заверка точками — оруденение иного типа, поэтому не первичная
    # метрика, а проверка «не противоречит ли».
    log("косвенная заверка точками")
    prot = assessment.run_protocol({"honest": s_avg, "criterial": -prognoz.ravel()},
                                   ["honest"], meta, valid, 0.05,
                                   baseline="criterial", log=log)
    print("\n=== Точки заверки (косвенная, иной тип оруденения) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "honest_forecast_verdict.csv",
                           index=False)

    out = pd.DataFrame({"row": np.repeat(np.arange(meta.prf), meta.pic),
                        "col": np.tile(np.arange(meta.pic), meta.prf),
                        "x": cx, "y": cy,
                        "score_blind_x": cxs, "score_blind_y": cys,
                        "score_blind": s_avg, "score_geo": s_geo,
                        "prognoz": prognoz.ravel(),
                        "target": y, "valid": valid})
    out_path = config.PROCESSED_DIR / "forecast_main_honest.parquet"
    out.to_parquet(out_path, index=False)
    log(f"OK: {out_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_c = (prognoz.ravel() <= np.quantile(prognoz.ravel()[pool],
                                            config.CREF_AREA)).reshape(shape)
    top_m = (s_avg >= np.quantile(s_avg[pool], 1 - config.CREF_AREA)).reshape(shape)
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    axes[0].imshow(rank(-prognoz.ravel()).reshape(shape), cmap="magma")
    axes[0].set_title("Критериальный прогноз\n(эталон геологов)")
    axes[1].imshow(rank(s_avg).reshape(shape), cmap="magma")
    axes[1].set_title(f"Модель, прогноз вслепую\n(обучение не ближе "
                      f"{STRIP_BUFFER_M/1000:.0f} км, {len(cols)} признаков)")
    agree = np.zeros(shape, dtype=int)
    agree[top_c & ~top_m] = 1
    agree[~top_c & top_m] = 2
    agree[top_c & top_m] = 3
    from matplotlib.colors import ListedColormap
    axes[2].imshow(agree, cmap=ListedColormap(["#eeeeee", "#3b6fa0", "#a04a3b", "#ffd34d"]),
                   vmin=0, vmax=3, interpolation="nearest")
    ov = res.loc[res["схема"] == "среднее калибр. (итог)", "overlap_top10"].iloc[0]
    axes[2].set_title(f"Совпадение top-10%: {ov:.0%}\n"
                      "жёлтый — оба, синий — только геологи, красный — только модель")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig_path = ROOT / "outputs" / "honest_forecast_main_map.png"
    fig.savefig(fig_path, dpi=130)
    log(f"OK: {fig_path}")


if __name__ == "__main__":
    main()
