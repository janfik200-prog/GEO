"""Правила ANFIS словами: что именно модель считает признаком разлома.

ЗАЧЕМ. Нечёткий вывод не выиграл у логрегрессии ни на 10 признаках (2.07 против
2.09), ни на 55 (1.78 против 2.12) — по качеству карты он не нужен. Но ради
качества его и не предлагали: смысл ANFIS в том, что его посылки читаются как
условия, а логрегрессия даёт только веса. До сих пор центры правил лежали в
``openlin_anfis_*_rules.csv`` числами и ни разу не были прочитаны.

ЧТО СЧИТАЕТСЯ. Модель переобучается (8 секунд), и для каждого правила берутся:

* ДОЛЯ — на скольких ячейках проверочной части это правило доминирует, то есть
  у него наибольшая нормированная сила среди всех шестнадцати;
* ОБОГАЩЕНИЕ — во сколько раз доля размеченных ячеек среди «его» ячеек выше, чем
  по площадке в целом. Это и есть проверка: правило либо действительно
  выделяет разломы, либо просто нарезает фон;
* УСЛОВИЕ — центр правила, переведённый в слова по квантилям обучающей выборки.

Правило читается только если оно доминирует хоть где-то: у рассеянного разбиения
часть правил вырождается и не выигрывает ни одной ячейки — такие честнее
показать отдельно, чем молча выбросить.

ЭТО ОПИСАНИЕ, А НЕ ГИПОТЕЗА. Ни одного p-value, знаменатель Бонферрони не
растёт: обогащение считается на тех же ячейках, на которых уже посчитан lift, и
новым тестом против карты разломов не является.

Запуск из корня репозитория:
    python -m experiments.openlin_anfis_rules --site nevada
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.openlin_anfis import (N_RULES, N_TRAIN_ROWS, SEED,
                                       TARGET_HALFWIDTH_M, TOL_M, block_split,
                                       fit_anfis, rows, soft_target)
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, load_site

M = Path("outputs/metrics")

#: Границы словесной шкалы в единицах нормированного входа (медиана и IQR).
#: 0.75 IQR — это примерно верхний/нижний квартиль: центр дальше него уже
#: осмысленно называть «высоким» или «низким», ближе — «средним».
WORD_HI = 0.75

#: Сколько каналов попадает в формулировку правила. Все десять в одну строку не
#: читаются, а условие держится на самых выраженных: остальные у центра около
#: нуля и означают «неважно».
WORDS_PER_RULE = 4

#: Понятные имена каналов вместо машинных.
NAMES = {
    "gray": "яркость оптики",
    "ndvi": "растительность (NDVI)",
    "vv": "радар VV",
    "vh": "радар VH",
    "coh_winter_vv_COH12": "зимняя когерентность (12 дн.)",
    "coh_summer_vv_COH12": "летняя когерентность (12 дн.)",
    "coh_summer_vv_COH24": "летняя когерентность (24 дн.)",
    "coh_summer_vv_COH48": "летняя когерентность (48 дн.)",
    "coh_summer_vv_rho": "скорость распада когерентности",
    "coh_summer_vv_tau": "время распада когерентности",
}


def words(center: np.ndarray, names: tuple[str, ...]) -> str:
    """Центр правила -> условие словами по самым выраженным каналам."""
    order = np.argsort(-np.abs(center))
    parts = []
    for i in order[:WORDS_PER_RULE]:
        if abs(center[i]) < WORD_HI:
            break
        parts.append(f"{NAMES.get(names[i], names[i])} "
                     f"{'высокая' if center[i] > 0 else 'низкая'}")
    return " И ".join(parts) if parts else "(без выраженных условий)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="nevada")
    ap.add_argument("--rules", type=int, default=N_RULES)
    ap.add_argument("--target", default="smooth", choices=("hard", "smooth"))
    ap.add_argument("--neigh", action="store_true")
    ap.add_argument("--lineaments", action="store_true",
                    help="таргет — подтверждённые линеаменты, а не разломы")
    a = ap.parse_args()

    z = load_site(a.site)
    x, _y, valid, use = build_stack(z, CHANNELS)
    if a.neigh:
        nx, nn = neigh_stack(x, use, valid, float(z["res_m"]))
        x = np.concatenate([x, nx])
        use = tuple(use) + nn
    if a.lineaments:
        from experiments.openlin_anfis_lin import lineament_dist
        dist = lineament_dist(a.site, float(z["res_m"]))
    else:
        dist = z["dist"]
    tr, va = block_split(valid.shape, float(z["res_m"]))
    tr &= valid
    va &= valid

    xtr = rows(x, tr)
    dtr = dist[tr]
    rng = np.random.default_rng(SEED)
    if len(xtr) > N_TRAIN_ROWS:
        sel = rng.choice(len(xtr), N_TRAIN_ROWS, replace=False)
        xtr, dtr = xtr[sel], dtr[sel]
    soft = a.target == "smooth"
    ytr = (soft_target(dtr) if soft
           else (dtr <= TARGET_HALFWIDTH_M).astype("float32"))

    print(f"площадка {a.site}, признаков {len(use)}, таргет {a.target} "
          f"по {'линеаментам' if a.lineaments else 'разломам'}; "
          f"обучение на {len(xtr):,} строках")
    model = fit_anfis(xtr, ytr, a.rules, soft, lambda *_: None)

    # --- какое правило доминирует на каждой проверочной ячейке ---------------
    xva = rows(x, va)
    dva = dist[va]
    win, out = [], []
    with torch.no_grad():
        s = torch.exp(model.log_s).clamp_min(1e-3)
        for i in range(0, len(xva), 65536):
            xb = torch.from_numpy(xva[i:i + 65536])
            d = (xb[:, None, :] - model.c[None]) / s[None]
            w = torch.softmax(-0.5 * (d * d).sum(-1), dim=1)
            win.append(w.argmax(1).numpy())
            f = xb @ model.a.T + model.b[None]
            out.append((w * f).sum(1).numpy())
    win = np.concatenate(win)
    out = np.concatenate(out)

    near = (dva <= TOL_M)
    base = float(near.mean())
    c = model.c.detach().numpy()

    recs = []
    for r in range(a.rules):
        m = win == r
        recs.append({
            "правило": r,
            "доля_ячеек": float(m.mean()),
            "обогащение": (float(near[m].mean()) / base) if m.any() else np.nan,
            "средний_выход": float(out[m].mean()) if m.any() else np.nan,
            "условие": words(c[r], tuple(use)),
        })
    df = pd.DataFrame(recs).sort_values("обогащение", ascending=False)

    live = df[df["доля_ячеек"] > 0.001]
    dead = df[df["доля_ячеек"] <= 0.001]
    print(f"\nдоля размеченных ячеек по площадке (допуск {TOL_M:.0f} м): "
          f"{base:.3f}; правил, доминирующих хоть где-то: {len(live)} "
          f"из {a.rules}\n")
    print("=== правила, отсортированы по обогащению разметкой " + "=" * 22)
    for _, r in live.iterrows():
        print(f"  правило {int(r['правило']):>2}  доля {100 * r['доля_ячеек']:5.1f}%  "
              f"обогащение ×{r['обогащение']:.2f}")
        print(f"      ЕСЛИ {r['условие']}")
    if len(dead):
        print(f"\nвырождены (не выигрывают ни одной ячейки): "
              f"{', '.join(str(int(v)) for v in dead['правило'])}")

    tag = (f"{a.site}{'_neigh' if a.neigh else ''}_{a.target}"
           f"{'_lin' if a.lineaments else ''}")
    path = M / f"openlin_anfis_rules_read_{tag}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nсохранено: {path}")


if __name__ == "__main__":
    main()
