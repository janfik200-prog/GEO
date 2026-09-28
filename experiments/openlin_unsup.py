"""Линеаменты по снимку без обучения: выделение, проверка, и только потом разломы.

ПОСТАНОВКА. Цель — выделить ЛИНЕАМЕНТЫ по космоснимку. Линеамент определяется по
самому изображению: это протяжённый линейный элемент поверхности. Разломы в этой
задаче не цель и не таргет — сопоставление с ними идёт отдельным шагом, ПОСЛЕ
того как карта построена и проверена. Поэтому здесь нет ни обучения, ни разметки:
поле линейности считается прямо из композита.

ЧЕМ ПРОВЕРЯЕТСЯ КАЧЕСТВО, РАЗ РАЗМЕТКИ НЕТ. Тем же, чем его проверяют для
линеаментов вообще, — воспроизводимостью на независимых данных. Раскол здесь
ФИЗИЧЕСКИЙ: оптика (яркость, NDVI) и радар (VV, VH, интерферометрическая
когерентность) — разные приборы с разной физикой отклика. Линия, видимая обоими,
принадлежит поверхности; линия, видимая одним, может быть его артефактом —
облаком и тенью у оптики, спеклом и геометрией съёмки у радара.

ТРИ ЧИСЛА НА ВЫХОДЕ:

1. ВЗАИМНОЕ ПОКРЫТИЕ линий оптики и радара — доля длины каждого списка, лежащая
   в допуске от другого. Это и есть воспроизводимость карты.
2. СОГЛАСИЕ РОЗ АЗИМУТОВ — совпадают ли направления, даже если конкретные
   отрезки разошлись. Роза устойчивее отдельной линии.
3. СОПОСТАВЛЕНИЕ С РАЗЛОМАМИ — описательно: сколько линий проходит по зонам
   разломов и как соотносятся розы. Ни порогов, ни p-value: это не проверка
   метода, а сравнение двух карт.

Запуск из корня репозитория:
    python -m experiments.openlin_unsup --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_anfis import TOL_M
from experiments.openlin_domains import domain_edges
from experiments.openlin_lines import (DENSITY, DENSITY_DEFAULT, line_metrics,
                                       trace_lines)
from experiments.openlin_lines_diag import azimuths, near_fraction
from experiments.openlin_neigh import neigh_stack
from experiments.openlin_unet import CHANNELS, build_stack, load_site

M = Path("outputs/metrics")

#: Физический раскол каналов. Оптика и радар меряют разное разными приборами;
#: когерентность отнесена к радару, потому что считается по тем же снимкам S1.
OPTIC = ("gray", "ndvi")
RADAR = ("vv", "vh", "coh_winter_vv_COH12", "coh_summer_vv_COH12",
         "coh_summer_vv_COH24", "coh_summer_vv_COH48", "coh_summer_vv_rho",
         "coh_summer_vv_tau")

#: Ширина корзины розы азимутов, градусы. 15 — тот же шаг, что в диагностике.
ROSE_BIN = 15


def field(site: str, chans: tuple[str, ...], log) -> tuple:
    """Поле линейности по подмножеству каналов, без всякого обучения."""
    z = load_site(site)
    use = tuple(c for c in chans if c in z)
    x, _y, valid, _u = build_stack(z, use)
    res_m = float(z["res_m"])
    nx, _n = neigh_stack(x, use, valid, res_m, base=use)
    stack = np.concatenate([x, nx])
    strength, _ang = domain_edges(stack, valid, res_m)
    log(f"  {len(use)} каналов -> {stack.shape[0]} признаков, поле посчитано")
    return z, valid, res_m, np.nan_to_num(strength, nan=0.0)


def rose(az: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(az, bins=np.arange(0, 181, ROSE_BIN))
    return h / max(h.sum(), 1)


def confirm(lo: np.ndarray, lr: np.ndarray, valid: np.ndarray, res_m: float,
            n_null: int = 200, seed: int = 0) -> tuple[float, float]:
    """Взаимное покрытие двух списков линий и его случайный уровень.

    Нуль считается сдвигом одного из списков по тору. Дистанционное
    преобразование при этом НЕ пересчитывается: у циклического сдвига маски оно
    равно сдвинутому преобразованию, и это экономит основную часть времени.
    """
    from scipy.ndimage import distance_transform_edt

    if not lo.any() or not lr.any():
        return float("nan"), float("nan")
    dt_o = distance_transform_edt(~lo, sampling=(res_m, res_m))
    dt_r = distance_transform_edt(~lr, sampling=(res_m, res_m))
    obs = 0.5 * (float((dt_r[lo] <= TOL_M).mean())
                 + float((dt_o[lr] <= TOL_M).mean()))

    rng = np.random.default_rng(seed)
    h, w = lo.shape
    vals = []
    for _ in range(n_null):
        sh = (int(rng.integers(h)), int(rng.integers(w)))
        m = np.roll(lr, sh, (0, 1)) & valid
        if not m.any():
            continue
        dt_rs = np.roll(dt_r, sh, (0, 1))
        vals.append(0.5 * (float((dt_rs[lo] <= TOL_M).mean())
                           + float((dt_o[m] <= TOL_M).mean())))
    return obs, float(np.quantile(vals, 0.95)) if vals else float("nan")


def sweep(a, log) -> None:
    """Докуда можно уплотнять: подтверждение против случайного уровня."""
    fields = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        z, valid, res_m, s = field(a.site, chans, log)
        fields[name] = (z, valid, res_m, s)
    z, valid, res_m, _s = fields["оптика"]

    log("\n=== развёртка по плотности; стоп-правило — подтверждение "
        + "=" * 12)
    log(f"{'плотность':<14}{'линий':>7}{'длина, км':>11}{'подтв.':>9}"
        f"{'случайно':>10}{'запас':>8}")
    rows = []
    for dens in DENSITY:
        cut = {}
        for name in fields:
            zz, vv, rr, ss = fields[name]
            cut[name], br = trace_lines(ss, vv, vv, rr, dens)
            if name == "оптика+радар":
                n_lines, km = len(br), cut[name].sum() * rr / 1000
        obs, q95 = confirm(cut["оптика"], cut["радар"], valid, res_m)
        rows.append({"плотность": dens, "линий": n_lines, "длина_км": km,
                     "подтверждение": obs, "случайно_q95": q95,
                     "запас": obs / q95 if q95 else float("nan")})
        log(f"{dens:<14}{n_lines:>7}{km:>11.0f}{obs:>9.3f}{q95:>10.3f}"
            f"{obs / q95 if q95 else float('nan'):>8.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(M / f"openlin_unsup_sweep_{a.site}.csv", index=False,
              encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_unsup_sweep_{a.site}.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default=DENSITY_DEFAULT)
    ap.add_argument("--sweep", action="store_true",
                    help="развёртка по плотности со стоп-правилом подтверждения")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    log(f"площадка {a.site}, плотность '{a.density}'; обучения нет, "
        f"разметка не используется")

    if a.sweep:
        sweep(a, log)
        return

    parts = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        log(f"\n--- {name} ---")
        z, valid, res_m, s = field(a.site, chans, log)
        lines, br = trace_lines(s, valid, valid, res_m, a.density)
        parts[name] = {"z": z, "valid": valid, "res": res_m, "field": s,
                       "lines": lines, "br": br, "az": azimuths(lines, res_m)}
        log(f"  линий {len(br)}, длина {lines.sum() * res_m / 1000:.0f} км")

    z = parts["оптика"]["z"]
    res_m = parts["оптика"]["res"]
    valid = parts["оптика"]["valid"]

    # --- 1. воспроизводимость: оптика против радара -------------------------
    lo, lr = parts["оптика"]["lines"], parts["радар"]["lines"]
    a_or = near_fraction(lo, lr, res_m)
    a_ro = near_fraction(lr, lo, res_m)
    log("\n=== 1. воспроизводимость на независимой физике " + "=" * 22)
    log(f"  доля длины линий ОПТИКИ рядом с линиями радара: {a_or:.3f}")
    log(f"  доля длины линий РАДАРА рядом с линиями оптики: {a_ro:.3f}")
    log(f"  взаимное покрытие: {0.5 * (a_or + a_ro):.3f}")

    # планка: случайный узор той же плотности (сдвиг одного из списков)
    rng = np.random.default_rng(0)
    h, w = lo.shape
    null = []
    for _ in range(200):
        m = np.roll(lr, (int(rng.integers(h)), int(rng.integers(w))), (0, 1)) & valid
        null.append(near_fraction(lo, m, res_m))
    log(f"  та же величина при случайном сдвиге радарных линий: "
        f"{np.mean(null):.3f} (95% квантиль {np.quantile(null, 0.95):.3f})")

    # --- 2. розы азимутов ---------------------------------------------------
    log("\n=== 2. направления " + "=" * 47)
    ro, rr = rose(parts["оптика"]["az"]), rose(parts["радар"]["az"])
    bins = np.arange(0, 180, ROSE_BIN)
    log(f"  главное направление оптики {bins[ro.argmax()] + ROSE_BIN // 2}°, "
        f"радара {bins[rr.argmax()] + ROSE_BIN // 2}°")
    log(f"  корреляция роз: {float(np.corrcoef(ro, rr)[0, 1]):.3f} "
        f"(плоская роза дала бы {1 / len(ro):.3f} в каждой корзине)")

    # --- 3. сопоставление с разломами (описательно) -------------------------
    log("\n=== 3. сопоставление с зонами разломов (описание, не проверка) "
        + "=" * 5)
    dist = z["dist"]
    base = float(((dist <= TOL_M) & valid).sum() / valid.sum())
    az_f = azimuths((dist <= res_m * 0.7) & valid, res_m)
    rf = rose(az_f)
    for name in ("оптика", "радар", "оптика+радар"):
        r = line_metrics(parts[name]["lines"], dist, valid, res_m)
        cr = float(np.corrcoef(rose(parts[name]["az"]), rf)[0, 1])
        log(f"  {name:<13} линий {len(parts[name]['br']):>4}  "
            f"у зон {r['точность']:.3f} (фон {base:.3f})  "
            f"корреляция роз с разломами {cr:+.3f}")
    log(f"  главное направление контуров зон "
        f"{bins[rf.argmax()] + ROSE_BIN // 2}°")

    # --- 4. поштучное подтверждение: отсев случайных линий -------------------
    log("\n=== 4. поштучный отсев: линия остаётся, если второй прибор видит "
        "её вдоль длины " + "=" * 3)
    l_all = parts["оптика+радар"]["lines"]
    # Требование И, а не ИЛИ: линия из совместного поля почти всегда видна
    # хотя бы одному прибору, поэтому «хотя бы один» ничего не отсеивает.
    # Свидетельством является независимое подтверждение ОБОИМИ.
    from experiments.openlin_lines import _branches

    keep_r, n_r, n_all = confirmed_branches(l_all, lr, res_m)
    keep_o, n_o, _ = confirmed_branches(l_all, lo, res_m)
    keep = keep_r & keep_o
    n_keep = sum(1 for p in _branches(keep, res_m) if len(p) >= 3)

    rng2 = np.random.default_rng(1)
    h2, w2 = l_all.shape
    fake = []
    for _ in range(25):
        s1 = (int(rng2.integers(h2)), int(rng2.integers(w2)))
        s2 = (int(rng2.integers(h2)), int(rng2.integers(w2)))
        k1, _p, _n = confirmed_branches(l_all, np.roll(lr, s1, (0, 1)) & valid,
                                        res_m)
        k2, _p, _n = confirmed_branches(l_all, np.roll(lo, s2, (0, 1)) & valid,
                                        res_m)
        fake.append(sum(1 for p in _branches(k1 & k2, res_m) if len(p) >= 3))
    log(f"  подтверждено радаром {n_r}, оптикой {n_o}, ОБОИМИ {n_keep} "
        f"из {n_all} веток")
    log(f"  осталось {keep.sum() * res_m / 1000:.0f} км из "
        f"{l_all.sum() * res_m / 1000:.0f}")
    log(f"  прошло бы случайно: {np.mean(fake):.1f} веток, "
        f"чистота отбора x{n_keep / max(np.mean(fake), 1e-9):.0f}")
    # Подтверждённые в parts не кладём: у них нет таблицы веток и розы, а
    # строка таблицы собирается по ним. Они уходят в npz отдельным ключом.
    out = pd.DataFrame([{
        "набор": n, "линий": len(parts[n]["br"]),
        "длина_км": parts[n]["lines"].sum() * res_m / 1000,
        "главный_азимут": int(bins[rose(parts[n]["az"]).argmax()] + ROSE_BIN // 2),
    } for n in parts] + [{
        "набор": "подтверждённые обоими", "линий": n_keep,
        "длина_км": keep.sum() * res_m / 1000,
        "главный_азимут": int(bins[rose(azimuths(keep, res_m)).argmax()]
                              + ROSE_BIN // 2),
    }])
    out.to_csv(M / f"openlin_unsup_{a.site}_{a.density}.csv", index=False,
               encoding="utf-8-sig")
    np.savez_compressed(M / f"openlin_unsup_{a.site}_{a.density}.npz",
                        **{f"lines_{i}": parts[n]["lines"]
                           for i, n in enumerate(parts)},
                        confirmed=keep,
                        field=parts["оптика+радар"]["field"])
    log(f"\nсохранено: {M / f'openlin_unsup_{a.site}_{a.density}.csv'}")




# --------------------------------------------------------------------------
# Поштучное подтверждение
# --------------------------------------------------------------------------

#: Какую долю своей длины линия обязана иметь подкреплённой вторым прибором.
#: 0.5 — большинство: случайное совпадение обычно задевает линию куском, а общий
#: линеамент виден второму прибору на всём протяжении.
CONFIRM_FRAC = 0.5

#: Насколько направления должны совпадать в точке подтверждения. 20 градусов —
#: та же терпимость, что у сшивки разрывов. Без этого условия подтверждением
#: считалось бы любое ПЕРЕСЕЧЕНИЕ, а пересечений в густой карте много.
CONFIRM_ANGLE_DEG = 20.0


def azimuth_field(lines: np.ndarray, res_m: float) -> np.ndarray:
    """Растр азимутов: в каждой ячейке линии — направление её ветки, градусы."""
    from experiments.openlin_lines import _axis, _branches

    out = np.full(lines.shape, np.nan, dtype="float32")
    for pts in _branches(lines, res_m):
        if len(pts) < 3:
            continue
        ax, _a, _b = _axis(pts)
        out[pts[:, 0], pts[:, 1]] = np.degrees(np.arctan2(ax[1], -ax[0])) % 180
    return out


def confirmed_branches(lines: np.ndarray, other: np.ndarray, res_m: float,
                       min_frac: float = CONFIRM_FRAC,
                       angle_deg: float = CONFIRM_ANGLE_DEG
                       ) -> tuple[np.ndarray, int, int]:
    """Оставить только линии, подтверждённые вторым прибором вдоль длины.

    Для каждой ячейки линии ищется ближайшая ячейка чужой карты; ячейка
    считается подтверждённой, если та ближе ``TOL_M`` И её направление
    отличается не больше чем на ``angle_deg``. Ветка проходит, если так
    подтверждено не меньше ``min_frac`` её длины.

    Возвращает ``(маска прошедших, сколько прошло, сколько всего)``.
    """
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_lines import _axis, _branches

    if not lines.any() or not other.any():
        return np.zeros_like(lines), 0, 0
    dt, idx = distance_transform_edt(~other, sampling=(res_m, res_m),
                                     return_indices=True)
    az_other = azimuth_field(other, res_m)

    keep = np.zeros_like(lines)
    n_pass = n_all = 0
    for pts in _branches(lines, res_m):
        if len(pts) < 3:
            continue
        n_all += 1
        r, c = pts[:, 0], pts[:, 1]
        near = dt[r, c] <= TOL_M
        ax, _a, _b = _axis(pts)
        my_az = np.degrees(np.arctan2(ax[1], -ax[0])) % 180
        oth = az_other[idx[0][r, c], idx[1][r, c]]
        d = np.abs(oth - my_az)
        same = np.minimum(d, 180.0 - d) <= angle_deg
        if float((near & same & np.isfinite(oth)).mean()) >= min_frac:
            keep[r, c] = True
            n_pass += 1
    return keep, n_pass, n_all


if __name__ == "__main__":
    main()
