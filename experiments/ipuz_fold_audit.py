"""WP04 ТЗ ИПУЗ: геометрический аудит пространственного разбиения.

Замороженный ``fold_leakage_check.json`` повторяет логические флаги самой
процедуры разбиения: если бы в ней была ошибка, проверка повторила бы ошибку.
Здесь геометрия пересчитывается независимо — по координатам ячеек из
``target_mask.csv`` и спискам индексов из ``fold_cells.json``:

  * минимальное расстояние обучение↔тест меряется KD-деревом, а не сравнением
    границ полос;
  * проверяется, что буфер действительно вынул все ячейки ближе 15 км;
  * каждая валидная ячейка на каждой оси попадает в тест ровно один раз;
  * площадь теста и площадь обучения считаются по числу ячеек.

Отдельно печатается, где процедура смотрит на отложенную полосу: нормировки
признаков в конвейере нет (бустинг работает с пропусками сам), калибровка
ранга берёт распределение только обучающих ячеек, но квадратичный тренд в
метрике ``совпадение_без_тренда`` подгоняется по всей сетке — это свойство
метрики, а не обучения, и оно вынесено в отчёт явно.

Запуск::

    python -m experiments.ipuz_fold_audit --freeze <каталог_заморозки>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CELL_M = 500.0
BUFFER_M = 15_000.0
CELL_KM2 = (CELL_M / 1000.0) ** 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    freeze = a.freeze.resolve()
    mask_csv = freeze / "01_inputs" / "target_mask.csv"
    cells_json = freeze / "02_protocol" / "fold_cells.json"
    manifest = freeze / "02_protocol" / "fold_manifest.csv"
    for p in (mask_csv, cells_json, manifest):
        if not p.exists():
            raise SystemExit("Не найден входной файл: %s" % p)
    out = (a.out or freeze / "02_protocol").resolve()

    t = pd.read_csv(mask_csv)
    x = t["x"].to_numpy(float)
    y = t["y"].to_numpy(float)
    valid = t["valid"].to_numpy(bool)
    pos = t["pos_train_2295"].to_numpy(bool)
    folds = json.loads(cells_json.read_text(encoding="utf-8"))
    man = pd.read_csv(manifest).set_index("фолд")

    строки = []
    for fid, part in folds.items():
        te = np.asarray(part["test"], dtype=int)
        tr = np.asarray(part["train"], dtype=int)
        pts_te = np.column_stack([x[te], y[te]])
        pts_tr = np.column_stack([x[tr], y[tr]])
        dist, _ = cKDTree(pts_tr).query(pts_te, k=1)
        dmin = float(dist.min())

        ось = man.loc[fid, "ось"]
        coord = x if ось == "X" else y
        lo = float(man.loc[fid, "граница_от"])
        hi = float(man.loc[fid, "граница_до"])
        # Ячейки, которые буфер обязан был исключить из обучения.
        должны_быть_вне = np.flatnonzero(
            valid & (coord >= lo - BUFFER_M) & (coord <= hi + BUFFER_M))
        протечка = np.intersect1d(tr, np.setdiff1d(должны_быть_вне, te))

        строки.append({
            "фолд": fid, "ось": ось,
            "ячеек_тест": int(te.size), "ячеек_обучение": int(tr.size),
            "позитивов_тест": int(pos[te].sum()),
            "позитивов_обучение": int(pos[tr].sum()),
            "площадь_теста_км2": round(te.size * CELL_KM2, 1),
            "площадь_обучения_км2": round(tr.size * CELL_KM2, 1),
            "мин_расстояние_м": round(dmin, 1),
            "разрыв_не_меньше_15км": bool(dmin >= BUFFER_M),
            "пересечение_train_test": int(np.intersect1d(te, tr).size),
            "тест_вне_валидной_маски": int((~valid[te]).sum()),
            "обучение_вне_валидной_маски": int((~valid[tr]).sum()),
            "ячеек_в_буфере_попало_в_обучение": int(протечка.size),
            "манифест_мин_разрыв_м": float(man.loc[fid, "разрыв_м"]),
        })

    df = pd.DataFrame(строки)

    # Покрытие: каждая валидная ячейка ровно один раз в тесте на каждой оси.
    покрытие = {}
    for ось in ("X", "Y"):
        счёт = np.zeros(valid.size, dtype=int)
        for fid, part in folds.items():
            if man.loc[fid, "ось"] == ось:
                счёт[np.asarray(part["test"], dtype=int)] += 1
        покрытие[ось] = {
            "валидных_ровно_один_раз": int((счёт[valid] == 1).sum()),
            "валидных_ни_разу": int((счёт[valid] == 0).sum()),
            "валидных_дважды_и_более": int((счёт[valid] > 1).sum()),
            "невалидных_попало_в_тест": int((счёт[~valid] > 0).sum()),
        }

    проверки = {
        "все_фолды_разрыв_15км": bool(df["разрыв_не_меньше_15км"].all()),
        "нет_пересечения_train_test": int(df["пересечение_train_test"].sum()) == 0,
        "тест_только_внутри_маски": int(df["тест_вне_валидной_маски"].sum()) == 0,
        "обучение_только_внутри_маски":
            int(df["обучение_вне_валидной_маски"].sum()) == 0,
        "буфер_вынул_всё_что_должен":
            int(df["ячеек_в_буфере_попало_в_обучение"].sum()) == 0,
        "покрытие_X_полное": покрытие["X"]["валидных_ни_разу"] == 0
            and покрытие["X"]["валидных_дважды_и_более"] == 0,
        "покрытие_Y_полное": покрытие["Y"]["валидных_ни_разу"] == 0
            and покрытие["Y"]["валидных_дважды_и_более"] == 0,
        "минимум_позитивов_соблюдён":
            bool((df["позитивов_тест"] >= 5).all()
                 and (df["позитивов_обучение"] >= 20).all()),
    }

    отчёт = {
        "источник": {"маска": str(mask_csv), "фолды": str(cells_json)},
        "буфер_м": BUFFER_M,
        "метод_расстояния": "cKDTree, евклидово расстояние между центрами ячеек",
        "фолды": строки,
        "покрытие_по_осям": покрытие,
        "проверки": проверки,
        "утечка_в_подготовке_признаков": {
            "нормировка_признаков": "нет: HistGradientBoosting принимает пропуски "
                                    "и не требует масштабирования "
                                    "(experiments/spectral_exposure.py::blind_map)",
            "заполнение_пропусков": "нет: пропуск — значение признака, "
                                    "маска спектра ставит NaN до разбиения",
            "калибровка_ранга": "только обучающие ячейки полосы: "
                                "np.searchsorted(sort(p_train), p_test)",
            "снятие_квадратичного_тренда": "подгоняется по всей сетке, включая "
                                           "отложенные ячейки; это свойство меры "
                                           "сходства двух готовых карт, а не "
                                           "обучения, но при выводах о значимости "
                                           "учитывать обязательно",
        },
        "итог": "PASS" if all(проверки.values()) else "FAIL",
    }
    (out / "fold_geometry_audit.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out / "fold_geometry_audit.csv", index=False, encoding="utf-8")

    cols = ["фолд", "ось", "ячеек_тест", "ячеек_обучение", "позитивов_тест",
            "позитивов_обучение", "мин_расстояние_м", "площадь_теста_км2"]
    print(df[cols].to_string(index=False))
    print()
    for ось, v in покрытие.items():
        print("ось %s: ровно один раз %d, ни разу %d, дважды %d, "
              "невалидных в тесте %d"
              % (ось, v["валидных_ровно_один_раз"], v["валидных_ни_разу"],
                 v["валидных_дважды_и_более"], v["невалидных_попало_в_тест"]))
    print()
    for k, v in проверки.items():
        print("  %-32s %s" % (k, "OK" if v else "ОШИБКА"))
    print("итог:", отчёт["итог"])
    if отчёт["итог"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
