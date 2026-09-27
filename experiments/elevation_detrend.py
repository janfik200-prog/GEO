"""Убрать региональный наклон из признаков, а не только из карт.

ПОВОД. Рабочая точка связана с абсолютной высотой на +0.501 по рангу, тогда как
критериальная карта — только на +0.237, и `dem_elev` держит первое место по
важности. Убрать высоту нельзя: без неё совпадение падает 0.362 -> 0.219. Значит
высота несёт что-то нужное, но несёт вместе с региональным наклоном.

ИДЕЯ. Главная метрика проекта оценивает карты ПОСЛЕ снятия с них квадратичной
поверхности координат — то есть смотрит только на локальные отклонения. А
модель обучается на признаках ВМЕСТЕ с наклоном и тратит ёмкость на то, что
метрика потом выбрасывает. Логично согласовать одно с другим: обучать на том же,
что и меряем.

Наклон снимается двумя разными способами, потому что они снимают разное:

* **минус региональная поверхность** — из признака вычитается подогнанный по
  всему листу квадратичный полином от координат. Ровно та операция, которую
  делает метрика. Меток не касается, на новой территории поверхность
  подгоняется заново по её собственным координатам;
* **превышение над окрестностью** — из признака вычитается его же среднее в
  круге радиусом несколько километров (`neighbourhood.local_contrast`). Это
  сильнее: убирает не только гладкий наклон, но и любой плавный фон, оставляя
  чисто локальную аномалию. Для высоты это привычная геоморфологическая
  величина — превышение над местным базисом.

ЧТО СРАВНИВАЕТСЯ. Сначала по одной высоте (заменить её превышением, заменить
остатком от поверхности, убрать совсем), потом то же для ВСЕХ признаков разом,
включая вариант «сырые вместе с превышениями» — вдруг полезны обе шкалы.

Отдельной строкой, вне сравнения, посчитана модель, обученная на всём листе и
применённая к нему же. Это не проверка обобщения, а верхняя оценка того, что
организация увидит на итоговой карте; помещена сюда, чтобы не путать её с
цифрами проверки по полосам.

Диагностика у каждого варианта: доля дисперсии карты, объяснимая координатами, и
ранговая связь карты с высотой — обе должны падать, если приём работает.

РЕЗУЛЬТАТ (07.09.2026) — идея не подтвердилась, все семь вариантов хуже.

| вариант | без тренда | доля тренда в карте | связь с высотой |
|---|---|---|---|
| как сейчас | **0.362** | 0.748 | +0.501 |
| без высоты | 0.219 | 0.589 | −0.031 |
| высота -> превышение 10 км | 0.285 | 0.630 | +0.094 |
| высота -> превышения 5/10/20 км | 0.287 | 0.571 | +0.294 |
| высота -> минус региональная поверхность | 0.228 | 0.588 | +0.009 |
| все признаки минус поверхность | 0.093 | 0.677 | +0.059 |
| все признаки -> превышения 10 км | 0.106 | 0.695 | +0.218 |
| сырые + превышения 10 км | 0.142 | 0.829 | +0.401 |

Приёмы делают ровно то, ради чего вводились — связь с высотой падает с +0.501
до +0.01…+0.09, доля тренда в карте с 0.748 до 0.57–0.63, — и при этом
совпадение падает тоже. Значит наклон в признаках не паразитный: модель по нему
находит нужные места, а метрика снимает наклон с ГОТОВОЙ карты, что не то же
самое, что убрать его на входе. Тотальное снятие наклона со всех признаков
разрушает модель (фолды 0.739 -> 0.484): у спектральных каналов смысл именно в
абсолютных значениях.

Дополнительный довод, что тревога была преувеличена: доля тренда у нашей карты
0.748 против 0.717 у самой критериальной — разница мала, наша карта не
«трендовее» эталона. Зависимость от высоты остаётся, снять её нечем, и
переносимость надо проверять напрямую, а не через эту связь.

Отдельная строка вне сравнения: модель, обученная на всём листе и применённая к
нему же, даёт совпадение верхних 10% **0.767**, без тренда 0.548, каппа 0.741,
ROC 0.970. Это не проверка обобщения, а оценка того, что видно на итоговой
карте.

Выход: `outputs/metrics/elevation_detrend.csv`.
Запуск из корня: ``python -m experiments.elevation_detrend``
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 neighbourhood as nb, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import (make_model, poly_design,  # noqa: E402
                                               trend_share)
from experiments.branch3_shapes import detrend  # noqa: E402
from experiments.spectral_exposure import blind_map, score_map  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH,
                                                     agree_detrended, boot_diff,
                                                     smooth_strips)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ELEV = "dem_elev"
SIGMA_KM = 10.0                      # основная окрестность для превышения
SIGMAS_KM = (5.0, 10.0, 20.0)        # многомасштабный вариант для высоты
T0 = time.time()


def log(msg: str) -> None:
    print("[%6.1fs] %s" % (time.time() - T0, msg), flush=True)


def bump(df: pd.DataFrame, cols, sigma_km: float, meta, valid) -> pd.DataFrame:
    """Превышение над окрестностью: значение минус его среднее в круге радиуса sigma."""
    shape = (meta.prf, meta.pic)
    sig = sigma_km * 1000.0 / meta.dx
    out = {}
    for c in cols:
        out["%s_прев%.0fкм" % (c, sigma_km)] = nb.local_contrast(
            df[c].to_numpy(dtype=float), shape, sig, mask=valid)
    return pd.DataFrame(out, index=df.index)


def flatten(df: pd.DataFrame, cols, pool: np.ndarray, D: np.ndarray) -> pd.DataFrame:
    """Признак минус подогнанная по листу квадратичная поверхность координат."""
    out = {}
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        ok = pool[np.isfinite(v[pool])]
        out["%s_безнаклона" % c] = detrend(np.where(np.isfinite(v), v, np.nan), ok, D)
    return pd.DataFrame(out, index=df.index)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    y = criterial_target.training_positives(prognoz).astype(int)
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    D = poly_design(cx, cy)
    blocks = crit_reference.block_labels(meta)

    cols = [c for c in honest_features("перенос") if c in feat.columns]
    work = spectral_mask.apply(feat[cols])
    base = list(work.columns)
    elev = work[[ELEV]]
    log("признаков %d, ячеек в работе %d" % (work.shape[1], pool.size))


    variants = {
        "как сейчас": work,
        "без высоты": work.drop(columns=[ELEV]),
        "высота -> превышение над окрестностью 10 км":
            pd.concat([work.drop(columns=[ELEV]),
                       bump(elev, [ELEV], SIGMA_KM, meta, valid)], axis=1),
        "высота -> превышения 5/10/20 км":
            pd.concat([work.drop(columns=[ELEV])] +
                      [bump(elev, [ELEV], s, meta, valid) for s in SIGMAS_KM], axis=1),
        "высота -> минус региональная поверхность":
            pd.concat([work.drop(columns=[ELEV]),
                       flatten(elev, [ELEV], pool, D)], axis=1),
        "все признаки минус региональная поверхность":
            flatten(work, base, pool, D),
        "все признаки -> превышения над окрестностью 10 км":
            bump(work, base, SIGMA_KM, meta, valid),
        "сырые признаки + превышения 10 км":
            pd.concat([work, bump(work, base, SIGMA_KM, meta, valid)], axis=1),
    }

    rows, maps = [], {}
    for name, df in variants.items():
        _, aucs, per_ax = blind_map(df.to_numpy(dtype=float), y, pool, cx, cy)
        sm = smooth_strips(per_ax, meta, valid, cx, cy)
        maps[name] = sm
        m = score_map(sm, y, crit, pool, valid, D)
        ok = pool[np.isfinite(sm[pool])]
        rows.append({"вариант": name, "признаков": df.shape[1],
                     "фолды_roc_auc": float(np.mean(aucs)), **m,
                     "доля_тренда_в_карте": float(trend_share(sm[:, None], D)[0]),
                     "связь_с_высотой": float(sps.spearmanr(
                         sm[ok], work[ELEV].to_numpy()[ok]).statistic)})
        log("  %-50s без тренда=%.3f  тренд=%.3f  высота=%+.3f"
            % (name, m["совпадение_без_тренда"], rows[-1]["доля_тренда_в_карте"],
               rows[-1]["связь_с_высотой"]))

    # Вне сравнения: обучение на всём листе — это не проверка обобщения
    mdl = make_model().fit(work.to_numpy(dtype=float)[pool], y[pool])
    p = np.full(y.size, np.nan)
    p[pool] = mdl.predict_proba(work.to_numpy(dtype=float)[pool])[:, 1]
    theta = nb.theta_from_azimuth(NW_AZIMUTH)
    km = lambda v: v * 1000.0 / meta.dx                                # noqa: E731
    p_sm = nb.aniso_smooth(p, (meta.prf, meta.pic), km(ALONG_KM), km(ACROSS_KM),
                           theta, mask=valid)
    m = score_map(p_sm, y, crit, pool, valid, D)
    rows.append({"вариант": "обучение на всём листе (НЕ проверка обобщения)",
                 "признаков": work.shape[1], "фолды_roc_auc": float("nan"), **m,
                 "доля_тренда_в_карте": float(trend_share(p_sm[:, None], D)[0]),
                 "связь_с_высотой": float("nan")})
    log("  %-50s без тренда=%.3f  с трендом=%.3f"
        % ("обучение на всём листе", m["совпадение_без_тренда"],
           m["совпадение_верх_10"]))

    res = pd.DataFrame(rows)
    out = ROOT / "outputs" / "metrics" / "elevation_detrend.csv"
    res.to_csv(out, index=False)
    print()
    print("=== Снятие наклона с признаков ===")
    print(res.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print()
    print("Потолок метрики без тренда — 0.622; 70%% от него это 0.435.")

    ref = "как сейчас"
    ta_ref, tc = agree_detrended(maps[ref], crit, pool, valid, D)
    print()
    print("=== Разность с нынешней рабочей точкой (бутстрэп 10 км) ===")
    for name in maps:
        if name == ref:
            continue
        ta, _ = agree_detrended(maps[name], crit, pool, valid, D)
        b = boot_diff(ta, ta_ref, tc, pool, blocks)
        print("  %-50s %+.3f [%+.3f; %+.3f]"
              % (name, b["разность_средняя"], b["разность_ниж_5"],
                 b["разность_верх_95"]))
    print()
    print("OK: %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
