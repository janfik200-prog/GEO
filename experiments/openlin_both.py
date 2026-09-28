"""Оптика и радар вместе: карта уступов, собранная по двум подсветкам.

ОТКУДА ПОСТАНОВКА. По отдельности ни один прибор карту дать не может, и это
не недостаток обработки, а геометрия: структура, идущая ВДОЛЬ подсветки, не
даёт перепада освещённости вовсе. Измерено с обеих сторон и в обе стороны это
ровно ноль, а не «мало»:

* солнце светит с 184° — оптика находит 189 широтных элементов и 0 меридиональных;
* радар светит со 103° — радар находит 38 меридиональных и 0 широтных.

Слепые зоны у приборов разные, поэтому объединение закрывает угловое
пространство почти целиком.

ЧТО ЗДЕСЬ СЧИТАЕТСЯ. Три вещи, невозможные по одному прибору:

1. ОБЩАЯ КАРТА с указанием, кто нашёл каждый элемент, и распределение по
   румбам — видно, осталась ли дыра;
2. ВЗАИМНОЕ ПОДТВЕРЖДЕНИЕ на косых простираниях. Косая структура лежит под
   углом к обеим подсветкам, значит видна обоим приборам. Совпадение здесь —
   это подтверждение двумя разными физиками отклика (отражение света и
   обратное рассеяние радиоволн), то есть свойство поверхности, а не артефакт
   одного тракта. Сравнивать так можно ТОЛЬКО косые: в слепой зоне прибора
   отсутствие линии ничего не значит;
3. ВЫСТРОЕННОСТЬ в цепочки по объединённому списку. По одному прибору эта
   проверка была искалечена цензурой по направлению: выстраиваться в прямую
   внутри одной трети углового пространства нечему.

Запуск из корня репозитория:
    python -m experiments.openlin_both --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.openlin_shade import (_stretch, fdr_curve, lines_from_field,
                                       load_winter)

M = Path("outputs/metrics")

#: Пороги оценки по приборам. Выбраны так, чтобы доля мусора НА УРОВНЕ
#: ЭЛЕМЕНТОВ была сопоставимой: оптика 227 против 44 на подделке (19 %),
#: радар 42 против 10 (24 %). Брать один порог на оба нельзя — шкала оценки
#: у них разная, радар шумнее.
THR = {"оптика": 3.5, "радар": 3.0}
MIN_LEN_M = 500.0

#: Допуск совпадения элементов двух приборов, м.
TOL_M = 400.0

#: Что считается косым простиранием: столько градусов от обеих подсветок.
OBLIQUE_MIN_DEG = 30.0

ILLUM = {"оптика": 184.0, "радар": 103.0}


def load(site: str, tag: str) -> tuple:
    a = dict(np.load(M / f"openlin_shade_{site}{tag}.npz"))
    b = dict(np.load(M / f"openlin_shade_{site}{tag}_подделка.npz"))
    return a["оценка"], b["оценка"], float(a["res_m"])


def oblique(az: np.ndarray) -> np.ndarray:
    """Элементы, лежащие под углом к ОБЕИМ подсветкам."""
    ok = np.ones(len(az), dtype=bool)
    for a0 in ILLUM.values():
        d = np.abs(az - a0 % 180.0)
        d = np.minimum(d, 180.0 - d)
        ok &= d >= OBLIQUE_MIN_DEG
    return ok


def rumb_table(az: np.ndarray) -> list[tuple[str, int]]:
    d = np.abs(az - 90.0)
    d = np.minimum(d, 180.0 - d)
    out = []
    for lo, hi, name in ((0, 30, "широтные"), (30, 60, "косые"),
                         (60, 90, "меридиональные")):
        out.append((name, int(((d >= lo) & (d < hi)).sum())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import pandas as pd
    from scipy.ndimage import binary_dilation, distance_transform_edt

    from experiments.openlin_align import chains, elements

    data = {}
    for name, tag in (("оптика", ""), ("радар", "_радар")):
        sc, scf, res_m = load(a.site, tag)
        tab = fdr_curve(sc, scf)
        thr = THR[name]
        junk = float(np.interp(thr, tab["порог"], tab["доля мусора, %"]))
        mask, df = lines_from_field(sc, res_m, thr, MIN_LEN_M)
        mask_f, df_f = lines_from_field(scf, res_m, thr, MIN_LEN_M)
        cent, az, ln = elements(mask, res_m)
        data[name] = {"маска": mask, "таблица": df, "центры": cent,
                      "азимут": az, "длины": ln, "res": res_m,
                      "мусор_ячейки": junk, "подделка": len(df_f)}
        log(f"{name}: порог {thr:.1f}, элементов {len(df)} "
            f"(на подделке {len(df_f)}, то есть "
            f"{100 * len(df_f) / max(len(df), 1):.0f} % мусора), "
            f"{df['длина_км'].sum():.0f} км")
        log("    " + ", ".join(f"{k} {v}" for k, v in rumb_table(az)))

    res_m = data["оптика"]["res"]
    both_az = np.concatenate([data[k]["азимут"] for k in data])
    both_cent = np.concatenate([data[k]["центры"] for k in data])
    both_ln = np.concatenate([data[k]["длины"] for k in data])
    log(f"\nобъединённый список: {len(both_az)} элементов")
    log("    " + ", ".join(f"{k} {v}" for k, v in rumb_table(both_az)))

    # 2. взаимное подтверждение на косых простираниях
    log("\nвзаимное подтверждение на косых простираниях "
        f"(допуск {TOL_M:.0f} м):")
    for name, other in (("оптика", "радар"), ("радар", "оптика")):
        d = data[name]
        sel = oblique(d["азимут"])
        if sel.sum() == 0:
            log(f"  {name}: косых элементов нет")
            continue
        dt = distance_transform_edt(~data[other]["маска"],
                                    sampling=(res_m, res_m))
        # центр элемента в (x, y) = (столбец, строка)
        cc = d["центры"][sel].astype(int)
        near = dt[cc[:, 1], cc[:, 0]] <= TOL_M
        # случайное ожидание: какая доля площади листа вообще ближе TOL_M
        base = float((dt <= TOL_M).mean())
        log(f"  {name}: косых {int(sel.sum())}, подтверждено "
            f"{int(near.sum())} ({100 * near.mean():.0f} %), случайно "
            f"ожидалось {100 * base:.0f} %, превышение "
            f"{near.mean() / max(base, 1e-9):.1f}x")

    # 3. выстроенность по объединённому списку
    area_km2 = float(np.isfinite(data["оптика"]["маска"]).sum()) * \
        (res_m / 1000.0) ** 2
    rng = np.random.default_rng(7)
    kw = dict(min_elems=4, min_span_m=8000.0, min_fill=0.15)
    n_real = len(chains(both_cent, both_az, both_ln, res_m, area_km2, **kw))
    sh = [len(chains(both_cent, rng.permutation(both_az), both_ln, res_m,
                     area_km2, **kw)) for _ in range(20)]
    sh = np.asarray(sh)
    p = (1 + int((sh >= n_real).sum())) / (1 + len(sh))
    log(f"\nцепочки по объединённому списку: {n_real}, при перемешанных "
        f"азимутах {sh.mean():.1f} (разброс {sh.min()}-{sh.max()})")
    log(f"p по нулю выстроенности: {p:.3f} "
        f"(минимально достижимое {1 / (1 + len(sh)):.3f})")

    # карта
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = load_winter(a.site, res_m)["shade"].astype("float32")
    base = _stretch(img)
    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.6))
    for k, (name, col) in enumerate((("оптика", (1.0, 0.25, 0.1, 1.0)),
                                     ("радар", (0.18, 0.77, 1.0, 1.0)))):
        ax[k].imshow(base, cmap="gray", vmin=0, vmax=1)
        w = binary_dilation(data[name]["маска"], np.ones((3, 3), dtype=bool))
        ov = np.zeros(w.shape + (4,), dtype="float32")
        ov[w] = col
        ax[k].imshow(ov)
        n = len(data[name]["таблица"])
        km = data[name]["таблица"]["длина_км"].sum()
        ax[k].set_title(f"{name}: подсветка с {ILLUM[name]:.0f}°\n"
                        f"{n} элементов, {km:.0f} км", fontsize=13)
    ax[2].imshow(base, cmap="gray", vmin=0, vmax=1)
    for name, col in (("оптика", (1.0, 0.25, 0.1, 1.0)),
                      ("радар", (0.18, 0.77, 1.0, 1.0))):
        w = binary_dilation(data[name]["маска"], np.ones((3, 3), dtype=bool))
        ov = np.zeros(w.shape + (4,), dtype="float32")
        ov[w] = col
        ax[2].imshow(ov)
    ax[2].set_title(f"вместе: {len(both_az)} элементов", fontsize=13)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    out = Path("outputs/steps") / f"уступы_два_прибора_{a.site}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    log(f"\nсохранено: {out}")

    rows = []
    for name in data:
        d = data[name]
        t = d["таблица"].copy()
        t["прибор"] = name
        rows.append(t)
    pd.concat(rows, ignore_index=True).to_csv(
        M / f"openlin_both_{a.site}.csv", index=False, encoding="utf-8-sig")
    log(f"сохранено: {M / f'openlin_both_{a.site}.csv'}")


if __name__ == "__main__":
    main()
