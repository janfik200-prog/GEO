"""WP07 ТЗ ИПУЗ: блочный бутстрэп доверительных интервалов O₁₀.

Лист режется на квадратные блоки 10 км (20×20 ячеек по 500 м) — 8×8 = 64
блока. Повторная выборка идёт блоками, а не ячейками: соседние ячейки
сильно зависимы, и бутстрэп по ячейкам дал бы интервал в разы уже реального.

    O₁₀* = Σ_b |A₁₀ ∩ B₁₀ ∩ b| / Σ_b |A₁₀ ∩ b|   по выбранным с возвратом блокам

Считается для обеих мер сразу: по исходным картам и по картам без
квадратичного тренда. Сохраняются сырое распределение всех повторов, состав
блоков и зерно генератора — интервал можно пересчитать, не повторяя обучение.

Интервал описывает пространственную изменчивость ВНУТРИ одного листа при
фиксированных картах. Это не неопределённость переноса на другой район и не
p-value против 0,10: 22 905 ячеек не являются независимыми наблюдениями.

Запуск::

    python -m experiments.ipuz_block_ci --freeze <каталог_заморозки>
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

SHAPE = (154, 149)
CELL_M = 500.0
BLOCK_M = 10_000.0
N_BOOT = 2000
SEED = 42
FRAC = 0.10

# Замороженная таблица 27.10.2026 — для строки «разница».
ЗАМОРОЖЕНО = {
    "сырое": (0.19860323003055436, 0.06518425404127275, 0.3481482171852734),
    "без тренда": (0.3622872108249673, 0.1755902382640097, 0.5363779722373598),
}


def top_decile(v: np.ndarray, pool: np.ndarray, frac: float = FRAC) -> np.ndarray:
    m = pool & np.isfinite(v)
    thr = float(np.quantile(v[m], 1.0 - frac))
    out = np.zeros(v.size, dtype=bool)
    out[m] = v[m] >= thr
    return out


def detrend(v: np.ndarray, pool: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Вычесть собственный квадратичный тренд карты по X,Y."""
    out = np.full(v.size, np.nan)
    ok = pool & np.isfinite(v)
    coef, *_ = np.linalg.lstsq(D[ok], v[ok], rcond=None)
    out[ok] = v[ok] - D[ok] @ coef
    return out


def block_ids(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Номер блока 10 км для каждой ячейки."""
    bx = np.floor((x - x.min()) / BLOCK_M).astype(int)
    by = np.floor((y - y.min()) / BLOCK_M).astype(int)
    nx, ny = bx.max() + 1, by.max() + 1
    return by * nx + bx, nx, ny


def bootstrap(a10: np.ndarray, b10: np.ndarray, blk: np.ndarray,
              n_blocks: int, rng: np.random.Generator) -> dict:
    both = a10 & b10
    n_a = np.bincount(blk[a10], minlength=n_blocks).astype(float)
    n_both = np.bincount(blk[both], minlength=n_blocks).astype(float)

    idx = rng.integers(0, n_blocks, size=(N_BOOT, n_blocks))
    sa = n_a[idx].sum(axis=1)
    sb = n_both[idx].sum(axis=1)
    ok = sa > 0
    draws = sb[ok] / sa[ok]

    с_ядром = n_a > 0
    по_блокам = np.divide(n_both, n_a, out=np.zeros_like(n_a), where=с_ядром)
    return {
        "O10": float(n_both.sum() / n_a.sum()),
        "ниж95": float(np.percentile(draws, 2.5)),
        "верх95": float(np.percentile(draws, 97.5)),
        "блоков": int(n_blocks),
        "блоков_с_ядром": int(с_ядром.sum()),
        "мин_блока": float(по_блокам[с_ядром].min()),
        "макс_блока": float(по_блокам[с_ядром].max()),
        "sd_блока": float(по_блокам[с_ядром].std()),
        "повторов_годных": int(draws.size),
        "_draws": draws,
        "_по_блокам": по_блокам,
        "_n_a": n_a,
        "_n_both": n_both,
    }


def по_полосам(a10: np.ndarray, b10: np.ndarray, x: np.ndarray,
               y: np.ndarray) -> dict:
    """O₁₀ внутри каждой из восьми полос протокола — столбцы замороженной таблицы.

    Это не вход бутстрэпа: полосы крупнее блоков и повторяют геометрию
    обучения, поэтому выборка идёт блоками. Величина нужна только для прямого
    сравнения с ``o10_block_ci.csv`` от 27.10.2026.
    """
    both = a10 & b10
    vals = []
    for coord in (x, y):
        edges = np.linspace(coord.min(), coord.max(), 5)
        sid = np.clip(np.digitize(coord, edges[1:-1]), 0, 3)
        for k in range(4):
            m = sid == k
            na = int((a10 & m).sum())
            if na:
                vals.append((both & m).sum() / na)
    v = np.asarray(vals, dtype=float)
    # В замороженной таблице столбец sd_полосы вычислен по выборочному
    # правилу (n−1): оно воспроизводит её значения с точностью 1e-3, тогда как
    # генеральное (n) расходится на 0,01. Приводятся оба, чтобы столбцы были
    # сопоставимы и чтобы правило было видно, а не додумывалось.
    return {"мин_полосы": float(v.min()), "макс_полосы": float(v.max()),
            "sd_полосы": float(v.std(ddof=1)),
            "sd_полосы_генеральное": float(v.std()), "полос": int(v.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--name", default="o10_block_ci_reproduced.csv",
                    help="имя выходной таблицы; замороженную o10_block_ci.csv "
                         "от 27.10.2026 перезаписывать нельзя")
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
    print("блоков %d (%d×%d) по %.0f км, ячеек в блоке до %d"
          % (n_blocks, nx, ny, BLOCK_M / 1000, int(BLOCK_M / CELL_M) ** 2))

    варианты = {
        "сырое": (crit, remote),
        "без тренда": (detrend(crit, pool, D), detrend(remote, pool, D)),
    }

    rows, сырьё = [], {}
    rng = np.random.default_rng(SEED)
    for имя, (va, vb) in варианты.items():
        a10, b10 = top_decile(va, pool), top_decile(vb, pool)
        r = bootstrap(a10, b10, blk, n_blocks, rng)
        r.update(по_полосам(a10, b10, x, y))
        o_f, lo_f, hi_f = ЗАМОРОЖЕНО[имя]
        print("%-11s O₁₀ %.6f  95%% [%.4f; %.4f]  "
              "заморожено [%.4f; %.4f]  sd блока %.4f"
              % (имя, r["O10"], r["ниж95"], r["верх95"], lo_f, hi_f,
                 r["sd_блока"]))
        сырьё[имя] = r.pop("_draws")
        rows.append({"вариант": имя, **{k: v for k, v in r.items()
                                        if not k.startswith("_")},
                     "повторов": N_BOOT, "зерно": SEED,
                     "блок_м": BLOCK_M, "генератор": "numpy PCG64",
                     "O10_заморожено": o_f, "ниж95_заморожено": lo_f,
                     "верх95_заморожено": hi_f,
                     "O10_разница": round(r["O10"] - o_f, 12)})
        r.pop("_по_блокам"), r.pop("_n_a"), r.pop("_n_both")

    pd.DataFrame(rows).to_csv(out / a.name, index=False, encoding="utf-8")
    pd.DataFrame({k: v for k, v in сырьё.items()}).to_csv(
        out / "o10_block_ci_draws.csv", index=False, encoding="utf-8")

    bx = np.floor((x - x.min()) / BLOCK_M).astype(int)
    by = np.floor((y - y.min()) / BLOCK_M).astype(int)
    состав = pd.DataFrame({"блок": np.arange(n_blocks)})
    состав["блок_x"] = состав["блок"] % nx
    состав["блок_y"] = состав["блок"] // nx
    состав["ячеек_всего"] = np.bincount(blk, minlength=n_blocks)
    состав["ячеек_в_пуле"] = np.bincount(blk[pool], minlength=n_blocks)
    состав.to_csv(out / "o10_block_composition.csv", index=False,
                  encoding="utf-8")

    (out / "o10_block_ci_meta.json").write_text(json.dumps({
        "блок_м": BLOCK_M, "блоков": n_blocks, "сетка_блоков": [int(nx), int(ny)],
        "повторов": N_BOOT, "зерно": SEED, "генератор": "numpy.random.PCG64",
        "способ": "повторная выборка блоков с возвратом; O₁₀ пересчитывается "
                  "как отношение сумм по выбранным блокам, а не среднее "
                  "блочных O₁₀",
        "знаменатель": "|A₁₀ ∩ блок| — ячейки верхней децили критериальной карты",
        "нельзя": "считать p-value против 0,10 по 22 905 ячейкам как "
                  "независимым наблюдениям; переносить интервал на другой район",
        "сырое_распределение": "o10_block_ci_draws.csv",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("записано:", out / a.name, "+ сырое распределение и состав блоков")


if __name__ == "__main__":
    main()
