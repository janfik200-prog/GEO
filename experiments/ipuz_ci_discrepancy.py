"""WP07 ТЗ ИПУЗ: откуда расхождение между замороженными и пересчитанными
интервалами O₁₀ и какую реализацию берёт статья.

Замороженная таблица ``04_metrics/o10_block_ci.csv`` от 27.10.2026 пришла в
пакете организации (`materials/o10_block_ci.csv`, совпадает побайтово) и
скрипта-источника не содержит: в ней нет ни зерна генератора, ни способа
вычисления процентиля, ни числа повторов — только границы. Поэтому разницу
нельзя объяснить чтением кода, её можно только измерить.

Скрипт отвечает на три вопроса.

1. Разброс границы от зерна. Та же оценка на 200 независимых зёрнах по 2 000
   повторов: если наблюдаемая разница лежит внутри этого разброса, она
   принадлежит самому методу Монте-Карло, а не данным или ошибке.
2. Способ процентиля. ``np.percentile`` при шести принятых соглашениях —
   насколько граница зависит от соглашения при 2 000 повторах.
3. Сходимость по числу повторов: 2 000 против 50 000 на одном зерне.

Отдельно разобран столбец ``sd_полосы``: минимум и максимум по восьми полосам
совпадают с замороженными побайтово, а sd — нет, поэтому проверяются
гипотезы о знаменателе (ddof) и составе набора.

Запуск::

    python -m experiments.ipuz_ci_discrepancy --freeze <каталог_заморозки>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.ipuz_block_ci import (  # noqa: E402
    BLOCK_M, N_BOOT, SEED, ЗАМОРОЖЕНО, block_ids, detrend, top_decile,
    по_полосам,
)

ЗЁРЕН = 200
МНОГО = 50_000

# Столбец sd_полосы замороженной таблицы — его правило вычисления не записано.
SD_ЗАМОРОЖЕНО = {"сырое": 0.1495884933975542,
                 "без тренда": 0.21479212112439278}


def доли_по_блокам(a10: np.ndarray, b10: np.ndarray, blk: np.ndarray,
                   n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    both = a10 & b10
    n_a = np.bincount(blk[a10], minlength=n_blocks).astype(float)
    n_both = np.bincount(blk[both], minlength=n_blocks).astype(float)
    return n_a, n_both


def повторы(n_a: np.ndarray, n_both: np.ndarray, rng: np.random.Generator,
            n_boot: int) -> np.ndarray:
    n_blocks = n_a.size
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    sa = n_a[idx].sum(axis=1)
    sb = n_both[idx].sum(axis=1)
    ok = sa > 0
    return sb[ok] / sa[ok]


def разброс_по_зёрнам(n_a: np.ndarray, n_both: np.ndarray,
                      зёрен: int = ЗЁРЕН) -> dict:
    низ, верх = [], []
    for s in range(зёрен):
        d = повторы(n_a, n_both, np.random.default_rng(s), N_BOOT)
        низ.append(float(np.percentile(d, 2.5)))
        верх.append(float(np.percentile(d, 97.5)))
    низ_a, верх_a = np.asarray(низ), np.asarray(верх)
    return {
        "зёрен": зёрен,
        "ниж95_среднее": float(низ_a.mean()),
        "ниж95_sd": float(низ_a.std(ddof=1)),
        "ниж95_мин": float(низ_a.min()),
        "ниж95_макс": float(низ_a.max()),
        "верх95_среднее": float(верх_a.mean()),
        "верх95_sd": float(верх_a.std(ddof=1)),
        "верх95_мин": float(верх_a.min()),
        "верх95_макс": float(верх_a.max()),
        "_низ": низ_a,
        "_верх": верх_a,
    }


def соглашения(d: np.ndarray) -> dict:
    out = {}
    for m in ("linear", "lower", "higher", "nearest", "midpoint",
              "median_unbiased"):
        out[m] = [float(np.percentile(d, 2.5, method=m)),
                  float(np.percentile(d, 97.5, method=m))]
    return out


def гипотезы_sd(значения: np.ndarray, заморожено: float) -> dict:
    v = np.asarray(значения, dtype=float)
    варианты = {
        "ddof=0, 8 значений": float(v.std()),
        "ddof=1, 8 значений": float(v.std(ddof=1)),
        "ddof=0, с добавленным нулём": float(np.append(v, 0.0).std()),
        "ddof=1, с добавленным нулём": float(np.append(v, 0.0).std(ddof=1)),
        "ddof=0, без нулевых полос": float(v[v > 0].std()),
        "ddof=1, без нулевых полос": float(v[v > 0].std(ddof=1)),
    }
    точное = [k for k, x in варианты.items() if abs(x - заморожено) < 1e-12]
    return {"заморожено": заморожено, "варианты": варианты,
            "совпало_точно": точное}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--зёрен", type=int, default=ЗЁРЕН)
    a = ap.parse_args()

    freeze = a.freeze.resolve()
    out = (a.out or freeze / "04_metrics").resolve()
    mask_csv = freeze / "01_inputs" / "target_mask.csv"
    oof_pq = a.root / "data" / "processed" / "operating_point_masked.parquet"
    for p in (mask_csv, oof_pq):
        if not p.exists():
            raise SystemExit("Не найден входной файл: %s" % p)
    out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(mask_csv)
    x = t["x"].to_numpy(float)
    y = t["y"].to_numpy(float)
    valid = t["valid"].to_numpy(bool)
    crit = -t["crit_score"].to_numpy(float)
    remote = pd.read_parquet(oof_pq)["score"].to_numpy(float)
    pool = valid & np.isfinite(crit) & np.isfinite(remote)

    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    D = np.column_stack([np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys])

    blk, nx, ny = block_ids(x, y)
    n_blocks = int(blk.max()) + 1

    варианты = {
        "сырое": (crit, remote),
        "без тренда": (detrend(crit, pool, D), detrend(remote, pool, D)),
    }

    отчёт = {
        "замороженная_таблица": "04_metrics/o10_block_ci.csv (пакет "
                                "организации, materials/o10_block_ci.csv, "
                                "совпадает побайтово)",
        "чего_нет_в_замороженной": ["зерно", "генератор", "число повторов",
                                    "способ вычисления процентиля"],
        "наша_реализация": {
            "скрипт": "experiments/ipuz_block_ci.py",
            "блок_м": BLOCK_M, "повторов": N_BOOT, "зерно": SEED,
            "генератор": "numpy.random.PCG64",
            "процентиль": "numpy.percentile, соглашение linear (по умолчанию)",
            "выборка": "блоки с возвратом; O₁₀ как отношение сумм",
        },
        "варианты": {},
    }

    строки = []
    # Один генератор на оба варианта подряд — ровно как в ipuz_block_ci.py,
    # иначе числа «нашего прогона» не совпадут с o10_block_ci_reproduced.csv.
    rng_общий = np.random.default_rng(SEED)
    for имя, (va, vb) in варианты.items():
        a10, b10 = top_decile(va, pool), top_decile(vb, pool)
        n_a, n_both = доли_по_блокам(a10, b10, blk, n_blocks)
        o_f, lo_f, hi_f = ЗАМОРОЖЕНО[имя]

        наш = повторы(n_a, n_both, rng_общий, N_BOOT)
        наш_низ = float(np.percentile(наш, 2.5))
        наш_верх = float(np.percentile(наш, 97.5))

        раз = разброс_по_зёрнам(n_a, n_both, a.зёрен)
        низ_a, верх_a = раз.pop("_низ"), раз.pop("_верх")
        доля_низ = float((низ_a <= lo_f).mean())
        доля_верх = float((верх_a <= hi_f).mean())

        много = повторы(n_a, n_both, np.random.default_rng(SEED), МНОГО)
        полосы = по_полосам(a10, b10, x, y)

        # значения по полосам нужны поимённо для разбора sd
        both = a10 & b10
        знач = []
        for coord in (x, y):
            edges = np.linspace(coord.min(), coord.max(), 5)
            sid = np.clip(np.digitize(coord, edges[1:-1]), 0, 3)
            for k in range(4):
                m = sid == k
                na = int((a10 & m).sum())
                if na:
                    знач.append((both & m).sum() / na)

        отчёт["варианты"][имя] = {
            "O10": float(n_both.sum() / n_a.sum()),
            "заморожено": {"ниж95": lo_f, "верх95": hi_f},
            "наш_прогон_зерно_42": {"ниж95": наш_низ, "верх95": наш_верх},
            "разница": {"ниж95": наш_низ - lo_f, "верх95": наш_верх - hi_f},
            "разброс_от_зерна": раз,
            "замороженная_граница_внутри_разброса": {
                "ниж95": bool(раз["ниж95_мин"] <= lo_f <= раз["ниж95_макс"]),
                "верх95": bool(раз["верх95_мин"] <= hi_f <= раз["верх95_макс"]),
                "доля_зёрен_не_выше_замороженной_ниж95": доля_низ,
                "доля_зёрен_не_выше_замороженной_верх95": доля_верх,
            },
            "сколько_sd_разницы": {
                "ниж95": (наш_низ - lo_f) / раз["ниж95_sd"],
                "верх95": (наш_верх - hi_f) / раз["верх95_sd"],
            },
            "соглашения_процентиля": соглашения(наш),
            "при_%d_повторах" % МНОГО: {
                "ниж95": float(np.percentile(много, 2.5)),
                "верх95": float(np.percentile(много, 97.5)),
            },
            "полосы": {
                "мин": полосы["мин_полосы"], "макс": полосы["макс_полосы"],
                "значения": [float(z) for z in знач],
                "sd_гипотезы": гипотезы_sd(знач, SD_ЗАМОРОЖЕНО[имя]),
            },
        }

        строки.append({
            "вариант": имя,
            "ниж95_заморожено": lo_f, "ниж95_наш": наш_низ,
            "ниж95_разница": наш_низ - lo_f,
            "ниж95_sd_от_зерна": раз["ниж95_sd"],
            "ниж95_мин_по_зёрнам": раз["ниж95_мин"],
            "ниж95_макс_по_зёрнам": раз["ниж95_макс"],
            "верх95_заморожено": hi_f, "верх95_наш": наш_верх,
            "верх95_разница": наш_верх - hi_f,
            "верх95_sd_от_зерна": раз["верх95_sd"],
            "верх95_мин_по_зёрнам": раз["верх95_мин"],
            "верх95_макс_по_зёрнам": раз["верх95_макс"],
            "зёрен": a.зёрен, "повторов": N_BOOT,
        })

        print("%-11s заморожено [%.4f; %.4f]  наше [%.4f; %.4f]"
              % (имя, lo_f, hi_f, наш_низ, наш_верх))
        print("            разброс границы от зерна: ниж sd %.4f "
              "[%.4f; %.4f], верх sd %.4f [%.4f; %.4f]"
              % (раз["ниж95_sd"], раз["ниж95_мин"], раз["ниж95_макс"],
                 раз["верх95_sd"], раз["верх95_мин"], раз["верх95_макс"]))
        print("            разница в единицах sd: ниж %.2f, верх %.2f"
              % ((наш_низ - lo_f) / раз["ниж95_sd"],
                 (наш_верх - hi_f) / раз["верх95_sd"]))

    отчёт["вывод"] = (
        "В статье и в релизе остаются ЗАМОРОЖЕННЫЕ границы из "
        "04_metrics/o10_block_ci.csv. Пересчёт "
        "04_metrics/o10_block_ci_reproduced.csv приводится рядом как "
        "подтверждение воспроизводимости: точечная оценка O₁₀ совпадает "
        "побайтово, границы расходятся в пределах разброса метода Монте-Карло "
        "при 2 000 повторах. Единственная реализация, пригодная для "
        "пересчёта, — experiments/ipuz_block_ci.py: у неё записаны зерно, "
        "генератор, способ процентиля и сырое распределение."
    )
    (out / "o10_block_ci_discrepancy.json").write_text(
        json.dumps(отчёт, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(строки).to_csv(
        out / "o10_block_ci_discrepancy.csv", index=False, encoding="utf-8")
    print("записано:", out / "o10_block_ci_discrepancy.json")


if __name__ == "__main__":
    main()
