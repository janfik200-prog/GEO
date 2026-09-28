"""Линеаментные поля в основной задаче: есть ли в них рудный сигнал.

ЗАЧЕМ. Вся ветка ``openlin_*`` до сих пор мерилась по КАРТЕ РАЗЛОМОВ, и там
результат отрицательный: три независимых оператора (гребни скора, линии рельефа,
границы доменов) сели на 1.0-1.4 от случайного уровня. Но карта разломов — не
цель проекта. Цель — золото-урановые кластеры, и у неё есть своя, независимая
заверка: точки заверки (геохимия, рудопроявления) и критериальный прогноз как
эталонная поверхность. Здесь линеаментные поля проверяются по НЕЙ.

Это существенно другой вопрос. Поле может плохо совпадать с нарисованными на
карте разломами и при этом нести рудный сигнал: положение разлома на карте
нарисовано с ошибкой в сотни метров и по данным, которых у модели нет, а
рудоконтролирующей может быть не сама линия, а зона повышенной раздробленности.

ЧТО СЧИТАЕТСЯ. Пять полей кладутся на сетку прогноза и прогоняются двумя
штатными протоколами проекта:

* ``crit_reference.run_protocol`` — согласие с критериальным прогнозом на
  тысячах ячеек (чувствительный тест);
* ``assessment.run_protocol`` — 19 точек заверки (независимый, но грубый: шаг
  метрики 0.53 lift на точку).

Обязательные планки те же, что у всей фазы: критериальный прогноз, «минус
расстояние до гидросети» и случайная карта.

ПОКРЫТИЕ. Растр площадки — повёрнутый кадр, он покрывает 72.6% валидных ячеек
листа. Пул ограничивается покрытыми ячейками, и ВСЕ карты, включая планки,
считаются на одном и том же ограниченном пуле — иначе сравнение бессмысленно.

Запуск из корня репозитория:
    python -m experiments.openlin_gold
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target, openlin_features)

M = ROOT / "outputs" / "metrics"

#: Пре-регистрация. Кандидаты объявлены ДО просмотра результатов; знаменатель
#: Бонферрони накопительный и включает все конфигурации, когда-либо проверенные
#: против тех же 19 точек. ``OWN_N_CONFIGS_TOTAL`` не трогаем — на него уже
#: ссылаются опубликованные пороги; своё слагаемое живёт здесь.
OLIN_N_CONFIGS = 5
OLIN_N_TESTS_TOTAL = config.OWN_N_CONFIGS_TOTAL + OLIN_N_CONFIGS

#: Первичная метрика — та же, что у ``VER_PRIMARY``: lift@10% по строго
#: несмещённым точкам. Согласие с критериальным эталоном (AUC) — вторичное, но
#: считается тем же прогоном, потому что на 19 точках шаг метрики 0.53 lift.
OLIN_PRIMARY = {
    "метрика": "lift@0.10 по unbiased_strict",
    "пул": "валидные ячейки, покрытые растром площадки",
    "значимость": f"сдвиговый нуль, alpha = 0.05 / {OLIN_N_TESTS_TOTAL}",
}


def build_layers(meta, log) -> dict[str, np.ndarray]:
    """Пять линеаментных полей на сетке прогноза."""
    z = dict(np.load(M / "openlin_anabar.npz"))
    layers: dict[str, np.ndarray] = {}

    p = M / "openlin_domains_anabar.npz"
    if p.exists():
        layers["olin_edge"] = openlin_features.to_grid(
            np.load(p)["strength"], z, meta)
        log("olin_edge — сила границы доменов (без обучения)")

    p = M / "openlin_anfis_nevada+abitibi_neigh_to_anabar_pred.npz"
    if p.exists():
        layers["olin_score"] = openlin_features.to_grid(
            np.load(p)["score"], z, meta)
        log("olin_score — непрерывный скор переноса (Невада+Абитиби)")

    p = (M / "openlin_lines_ens_nevada+abitibi_густо_логрегрессия"
         "_hard_anabar.npz")
    if p.exists():
        freq = np.load(p)["freq"]
        layers["olin_freq"] = openlin_features.to_grid(freq, z, meta)
        cons = freq >= 0.5
        layers["olin_lindens"] = openlin_features.to_grid(
            cons.astype("float32"), z, meta, how="frac")
        layers["olin_lindist"] = -openlin_features.distance_to(cons, z, meta)
        log("olin_freq / olin_lindens / olin_lindist — частота линии, "
            "плотность и минус расстояние до консенсусных линий")
    return layers


def main() -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v6.parquet")
    valid = cell_mask.build_valid_mask(df)
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    log(f"лист: валидных ячеек {int(valid.sum()):,}")

    layers = build_layers(meta, log)
    if len(layers) != OLIN_N_CONFIGS:
        log(f"ВНИМАНИЕ: посчитано {len(layers)} полей из "
            f"{OLIN_N_CONFIGS} заявленных — знаменатель Бонферрони "
            f"остаётся по ПОЛНОМУ списку, как требует пре-регистрация")

    out = pd.DataFrame(layers)
    out.to_parquet(config.PROCESSED_DIR / "openlin_features.parquet")
    log(f"слои сохранены: {config.PROCESSED_DIR / 'openlin_features.parquet'} "
        f"({out.shape[0]:,} строк x {out.shape[1]} признаков)")

    cover = np.all([np.isfinite(v) for v in layers.values()], axis=0)
    pool_mask = valid & cover
    log(f"покрыто растром: {int(pool_mask.sum()):,} валидных ячеек "
        f"({100 * pool_mask.sum() / valid.sum():.1f}%)")

    scores = {n: np.nan_to_num(v, nan=np.nanmin(v)) for n, v in layers.items()}
    scores["criterial"] = crit
    scores["naive_hydro"] = assessment.naive_hydro_score(df)
    scores["random"] = assessment.random_score(len(df), seed=config.SEED
                                               if hasattr(config, "SEED") else 0)
    own = list(layers)

    alpha = 0.05 / OLIN_N_TESTS_TOTAL
    log(f"пре-регистрация: {OLIN_N_CONFIGS} карт, знаменатель "
        f"{OLIN_N_TESTS_TOTAL}, alpha = {alpha:.6f}")

    # --- 1. критериальный эталон (чувствительный тест) ----------------------
    log("\n=== согласие с критериальным прогнозом ===")
    cref = crit_reference.run_protocol(scores, crit, meta, pool_mask,
                                       0.05 / (config.CREF_ORIENT_PENALTY
                                               * len(own)), log=log)
    print(cref["verdict"].round(4).to_string(index=False))
    cref["verdict"].to_csv(M / "openlin_gold_crit.csv", index=False,
                           encoding="utf-8-sig")

    # --- 2. точки заверки (независимый, но грубый) --------------------------
    log("\n=== заверка по точкам ===")
    ver = assessment.run_protocol(scores, own, meta, pool_mask, alpha,
                                  baseline="criterial", log=log)
    print(ver["verdict"].round(4).to_string(index=False))
    ver["verdict"].to_csv(M / "openlin_gold_points.csv", index=False,
                          encoding="utf-8-sig")

    log(f"\nсохранено: {M / 'openlin_gold_crit.csv'} и "
        f"{M / 'openlin_gold_points.csv'}")


if __name__ == "__main__":
    main()
