"""4 карты: ансамбль/критериальный × основной лист/смежная территория.

Панели:

1. **Ансамбль, основной лист** — честный out-of-sample скор: три
   leave-one-object-out фолда (:func:`criterial_target.leave_one_object_out`,
   103 признака = `dataset_v5_rebuilt` минус `ast`/`ls`, буфер
   `config.CRIT_HOLDOUT_BUFFER_M`), сшитые в одну карту — каждый объект
   получает балл фолда, где он был held-out, фон — среднее по 3 фолдам.
2. **Ансамбль, смежная территория** — та же модель, но обучена на ВСЕХ 3
   объектах (`experiments/forecast_wide.py`) и применена к `dataset_wide.parquet`.
   Это уже не проверка обобщения (её не с чем сверить за пределами листа), а
   экстраполяция.
3. **Критериальный, основной лист** — нативный `prognoz.pgrid` заказчика
   (реальные данные, не реконструкция).
4. **Критериальный (реконструкция формулы), смежная территория** — здесь
   нативных данных заказчика нет (Задача 4 реестра заблокирована), поэтому
   используется восстановленная формула ГИС Интегро
   (`config.TAXONOMY_WEIGHTS`/`TAXONOMY_TRANSFORMS`, см.
   `src/features.py::taxonomy_weighted_distance`, Spearman 0.9984 с нативным
   prognoz на листе) — взвешенное L1-расстояние (мера Плюты) до 6 факторов
   (`tect1/tect2/paleo/facies/struct/magm`). ВАЖНАЯ ОГОВОРКА: сами шейп-файлы
   факторов лежат строго в границах исходного листа (проверено — bbox
   совпадает с `prognoz.pgrid` почти точно), за его пределами `dist_*` —
   это расстояние до объекта, которого там физически не картировали, а не
   геологический сигнал. Реконструкция формулы верна, но за пределами листа
   эта панель показывает не «критериальный прогноз», а артефакт удалённости
   от границы картирования — цвет тем ближе к фону, чем дальше ячейка от
   листа. Читать эту панель с максимальной осторожностью.

Каждая панель нормализована НЕЗАВИСИМО (`robust_normalize_01` по своим
собственным квантилям) — совместная нормировка критериальной пары ломается:
нативный `prognoz` уже приведён заказчиком к узкому диапазону, а
реконструированное взвешенное L1-расстояние ничем не ограничено и на широкой
сетке принимает на порядок больший размах, поэтому совместная шкала схлопывала
бы нативный `prognoz` в одну сплошную заливку. Цвет читается как ОТНОСИТЕЛЬНЫЙ
паттерн внутри своей территории, не как абсолютно сопоставимое число между
панелями.

Выход: `outputs/forecast_compare_4panel.png`.

Запуск из корня репозитория: ``python -m experiments.forecast_compare_4panel``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, criterial_target, features, features_v2, integro_grid  # noqa: E402
from src.data_loader import load_layer  # noqa: E402
from src.model import BackgroundEnsemble  # noqa: E402
from src.utils import robust_normalize_01  # noqa: E402
from src.vector_features import distance_raster  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402

DROP_GROUPS = ("ast", "ls")


def ensemble_main_map() -> tuple[np.ndarray, integro_grid.GridMeta]:
    """Сшитая out-of-sample карта: held-out фолд для своего объекта, среднее по фолдам для фона."""
    X, labels_flat, coords, feat_names, meta = criterial_target.build_dataset(dataset="v5_rebuilt")
    keep_idx = [i for i, c in enumerate(feat_names) if features_v2.feature_group(c) not in DROP_GROUPS]
    X = X[:, keep_idx]
    print(f"Ансамбль (основной лист): {len(keep_idx)} признаков, 3 фолда LOO...")

    results = criterial_target.leave_one_object_out(X, labels_flat, coords)
    stitched = np.zeros(labels_flat.size)
    fold_stack = np.column_stack([r["score_all"] for r in results])
    stitched[:] = fold_stack.mean(axis=1)
    for r in results:
        held = r["held_idx"]
        stitched[held] = r["score_all"][held]
    return stitched, meta


def taxonomy_score(meta: integro_grid.GridMeta, shp_dir: Path) -> np.ndarray:
    """Взвешенное L1-расстояние ГИС Интегро (реконструированная формула) на сетке ``meta``."""
    dist_by_role = {}
    for role in config.TAXONOMY_TRANSFORMS:
        path = shp_dir / f"{config.LAYER_FILES[role]}.shp"
        gdf = load_layer(path)
        dist_by_role[role] = distance_raster(meta, gdf.geometry.values).ravel()
    dist = features.taxonomy_weighted_distance(dist_by_role)
    return -dist    # инверсия полярности: больше = перспективнее (как ансамбль/prognoz)


def main() -> None:
    ens_main, main_meta = ensemble_main_map()
    print("Ансамбль (смежная территория): читаю forecast_wide.parquet...")
    ens_wide = pd.read_parquet(config.PROCESSED_DIR / "forecast_wide.parquet")["score"].to_numpy()

    print("Критериальный (основной лист): читаю prognoz.pgrid...")
    _, prognoz = criterial_target.load_prognoz_grid()
    crit_main_raw = -prognoz.ravel().astype(float)   # инверсия: меньше prognoz = перспективнее

    print("Критериальный (реконструкция формулы, смежная территория)...")
    shp_dir = config.GOLD_TARGET_PGRID.parents[1] / config.SHP_SUBDIR
    crit_wide_raw = taxonomy_score(WIDE_META, shp_dir)

    # Независимая нормировка каждой панели (см. докстринг — совместная шкала
    # для критериальной пары нечитаема из-за разного масштаба величин).
    ens_main_n = robust_normalize_01(ens_main)
    ens_wide_n = robust_normalize_01(ens_wide)
    crit_main_n = robust_normalize_01(crit_main_raw)
    crit_wide_n = robust_normalize_01(crit_wide_raw)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def extent(m):
        return [m.x0, m.x0 + m.pic * m.dx, m.y0, m.y_top]

    panels = [
        (ens_main_n.reshape(main_meta.shape), main_meta, "Ансамбль — основной лист (honest LOO)"),
        (ens_wide_n.reshape(WIDE_META.shape), WIDE_META, "Ансамбль — смежная территория (экстраполяция)"),
        (crit_main_n.reshape(main_meta.shape), main_meta, "Критериальный — основной лист (нативный prognoz)"),
        (crit_wide_n.reshape(WIDE_META.shape), WIDE_META,
         "Критериальный — смежная территория\n(реконструкция формулы — вне листа не заверено!)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 20), constrained_layout=True)
    for ax, (grid, meta, title) in zip(axes.ravel(), panels):
        im = ax.imshow(grid, origin="upper", cmap="viridis", vmin=0, vmax=1, extent=extent(meta))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.045, label="score, норм. [0,1] (шкала своя для панели)")
    fig.suptitle("Основной лист vs смежная территория: ансамбль (независимые признаки) "
                 "и критериальный ориентир", fontsize=13)

    out_path = config.PROJECT_ROOT / "outputs" / "forecast_compare_4panel.png"
    fig.savefig(out_path, dpi=130)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
