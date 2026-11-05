"""Мера сверки наборов линий с эталоном GTK, не растущая от густоты набора.

ТОЛЬКО ОБЪЯВЛЕНИЕ (05.11.2026, до расчёта). Реализацию пишет основная сессия
строго по этому тексту; числа по данным до объявления не считались.

ОТКУДА ПОСТАНОВКА. Нынешняя мера (``openlin_transfer_check.recall``) — доля
ячеек эталона в ``TOL_M`` от ЛЮБОЙ нашей линии. Она растёт с густотой набора,
потому что нуль (сдвиг эталона) ≈ доле площади в полосе от набора. Утсйоки:
рельеф (2858 км) 48.5 % при нуле 28.4 %, снимок (1039 км) 20.0 % при нуле
10.9 %; при равной длине наборы почти равны (аудит). Сравнивать методы по сырой
полноте нельзя. Решение Артура 05.11.2026: ввести меру, у которой случайный
уровень мал и которая не растёт от одной густоты.

ЧТО МЕРИТСЯ. Три меры; ГЛАВНАЯ — вторая.

1. ``recall_dir`` (для сведения). Ячейка эталона «найдена», если в ``TOL_M`` от
   неё есть ячейка нашей линии, у которой разница направлений (по модулю 180°)
   <= ``DIR_TOL_DEG``. Значение — доля найденных среди ячеек эталона в общих
   ячейках снимка. Нуль — сдвиг эталона по тору (ниже).
2. ``curve_score`` (ГЛАВНАЯ). Линии набора ранжируются по длине (длинные
   первыми); для сетки длин L берутся первые линии, пока их суммарная длина
   <= L; на каждом L считается ``recall_dir`` (R_k) и его нуль (N_k).
   Число: S = sum_k (R_k - N_k) / sum_k (1 - N_k), k = 1..K. Числитель —
   площадь между кривой и нулём, знаменатель — та же площадь для идеальной
   кривой (R = 1 на всей сетке). S = 0 — как случайный набор, S = 1 — идеал.
3. ``long_lines`` (для сведения). Доля длинных линий эталона, у которых
   >= ``LINE_FRAC`` длины (в ячейках) направленно найдено по п. 1; нуль — тот
   же сдвиг; печатается и (obs - нуль) / (1 - нуль).

ОБЪЯВЛЕНИЕ (05.11.2026, до расчёта). Откуда каждое число:

* ``TOL_M`` = 300 м — ИМПОРТ из ``openlin_transfer_check`` (масштаб эталона:
  растр 300 м при 1:500 000). ``DIR_TOL_DEG`` = 20° — ИМПОРТ из
  ``openlin_zone_lines`` (допуск направления линии к зоне). Новых чисел тут нет.
* Направление нашей ячейки: у масок линий направления нет, линии не
  пересчитываются. Берётся главная ось (собственный вектор большего
  собственного числа) ковариации координат ячеек ТОЙ ЖЕ 8-связной компоненты,
  лежащих в круге радиуса ``DIR_WIN_M`` вокруг ячейки. ``DIR_WIN_M`` =
  ``TOL_M`` = 300 м: окно равно масштабу, на котором эталон вообще знает
  положение линии, и тем же числом, которым ячейке разрешено искать пару; не
  подбирается. Ячейка, у которой в окне меньше двух ячеек компоненты, без
  направления и не находит ничего. На пересечениях компонента общая (как в
  ``openlin_union_recall``) — ось там размыта, это известный дефект меры.
* Направление ячейки эталона: направление звена векторной линии, которое её
  растеризует (``gtk_raster`` с касанием); при сдвиге ячейка уносит свой угол.
* Ранжирование: ДЛИНА линии = ячеек компоненты * шаг сетки (так же считается
  «км набора» в transfer_check). Один признак, объявлен заранее: средний отклик
  в npz не лежит (там маски), а длина не зависит от эталона. Связи (равная
  длина) — по номеру первой ячейки в построчном порядке.
* Сетка L: шаг ``LEN_STEP_KM`` = 10 км — 10-й процентиль длины линии эталона
  (шапка transfer_check: медиана 24 км, 10-й процентиль 10 км); L_k = k * 10 км.
  Линии берутся целиком. Верх сетки K * 10 км, где K =
  floor(минимальная суммарная длина среди наборов ЭТОГО сравнения / 10 км);
  сравнение — таблица, где наборы сопоставляются между собой. K и длины
  выводятся в лог. Затраты на счёт НЕ основание менять шаг.
* Нуль: сдвиг эталона по тору, код ``transfer_check.shift_null`` (поля
  min(200, H/8) ячеек, ячейки вне общих не считаются), карта наших линий на
  месте. Число сдвигов ``N_SHIFTS`` = ``GTK_SHIFT_N`` = 159 и зерно
  ``SEED`` = ``GTK_SHIFT_SEED`` = 11 — ИМПОРТ: p-проверок здесь нет, нужно
  только среднее нуля, и ветка уже использует эти числа для того же эталона.
  Набор сдвигов (dr, dc) строится ОДИН раз и общий для всех L, всех наборов и
  п. 3 — разности между наборами не содержат шума разных сдвигов.
* ``REF_LONG_KM`` = 10 км — тот же 10-й процентиль длины эталона; длина —
  часть линии внутри общих ячеек (то, что сравнивается с нашими линиями).
* ``LINE_FRAC`` = ``ZONE_FRAC`` = 0.5 (ИМПОРТ из ``openlin_zone_lines``: правило
  ветки «линия принадлежит, если не меньше половины её длины лежит в...»).
  СЛАБОЕ МЕСТО: происхождение самого 0.5 — пре-регистрация 22.09.2026, замера
  за ним я не проверял; п. 3 поэтому только для сведения.
* Длинные эталонные линии при сдвиге вдоль простирания частично совпадают
  сами с собой — нуль выше, то есть строже (как в transfer_check).

ГЛАВНАЯ МЕРА И ПРАВИЛО. Главная — ``curve_score`` (п. 2); п. 1 и 3 только для
сведения. Сравнение методов — ТОЛЬКО по главной, на общем K. Порога различия
двух S из замера нет: пока он не измерен (нужен замер разброса S, например по
независимым половинам эталона), сравнение описательное — «больше/меньше», без
слов «значимо» и «доказано».

ГДЕ ПРИМЕНЯЕТСЯ. Площадка настройки Утсйоки, наборы: снимок UTS1
(``_настройка1``); рельеф RELU1 (``_рельеф1``); снимок+рельеф (правило
сложения ``openlin_union_recall.add_new``: ``ADD_FRAC``, ``ADD_TOL_M``);
каждый — «в зонах» (C) и «весь лист» (C_all). Два семейства сравнения: все
«весь лист» вместе и все «в зонах» вместе (свой K у каждого); пара «в зонах /
весь лист» одного метода — отдельное сравнение с K по более короткому.
СЧЁТЧИК: на площадке настройки это выбор метода, p-значений нет, в общий
счётчик ничего не идёт: ``REF_MATCH_N_TESTS`` = 0, alpha и пол p не вводятся
(правило n >= 2/alpha - 1 к нулю проверок неприменимо). Контроль
(Лапландия): мера применяется ТОЛЬКО к замороженному методу, РАЗОВО; если там
понадобится p или порог — ДО прогона отдельное объявление со своей семьёй
(``GTK_N_TESTS`` = 2 относится к старой мере и не переиспользуется).

Запуск (после реализации): python -m experiments.openlin_ref_match --site utsjoki
"""
from __future__ import annotations

import numpy as np

from experiments.openlin_transfer_check import (GTK_SHIFT_N, GTK_SHIFT_SEED,
                                                TOL_M)
from experiments.openlin_zone_lines import DIR_TOL_DEG, ZONE_FRAC

# ---- объявление (05.11.2026, до расчёта) -----------------------------------
DIR_WIN_M = TOL_M              # окно направления нашей ячейки = масштаб эталона
LEN_STEP_KM = 10.0             # 10-й процентиль длины линии эталона (шапка transfer_check)
REF_LONG_KM = 10.0             # тот же процентиль: «длинные линии эталона»
LINE_FRAC = ZONE_FRAC          # 0.5, правило ветки «не меньше половины длины»
N_SHIFTS = GTK_SHIFT_N         # 159, как для этого эталона в transfer_check
SEED = GTK_SHIFT_SEED          # 11
MAIN_MEASURE = "curve_score"   # главная; остальные — для сведения
REF_MATCH_N_TESTS = 0          # p-проверок нет, в счётчик ничего не идёт
# ---------------------------------------------------------------------------

# Реализация (05.11.2026, после коммита объявления 2a93ff3). Как считается
# быстро и без новых чисел: для каждого положения эталона (на месте и на
# каждом сдвиге) ищутся все пары «ячейка эталона — наша ячейка ближе TOL_M»
# (cKDTree.sparse_distance_matrix), отбрасываются пары с разницей направлений
# больше DIR_TOL_DEG, и для ячейки эталона запоминается НАИМЕНЬШИЙ ранг нашей
# линии, которая её находит. Тогда R_k — доля ячеек, у которых этот ранг
# меньше числа линий, вошедших в L_k: одна проходка на положение даёт всю
# кривую. Это ровно определение из шапки, не приближение.


def _axial_diff(a, b):
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def components(mask: np.ndarray):
    from scipy.ndimage import label
    return label(mask, structure=np.ones((3, 3), int))


def our_directions(mask: np.ndarray, res: float, lab=None, n=None) -> np.ndarray:
    """Угол (градусы, [0, 180)) главной оси компоненты в круге DIR_WIN_M; NaN без оси."""
    from scipy.ndimage import find_objects
    from scipy.signal import fftconvolve

    if lab is None:
        lab, n = components(mask)
    rad = int(round(DIR_WIN_M / res))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = ((yy ** 2 + xx ** 2) <= rad ** 2).astype("float64")
    out = np.full(mask.shape, np.nan, dtype="float32")
    H, W = mask.shape
    for j, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        r0, r1 = max(sl[0].start - rad, 0), min(sl[0].stop + rad, H)
        c0, c1 = max(sl[1].start - rad, 0), min(sl[1].stop + rad, W)
        m = (lab[r0:r1, c0:c1] == j).astype("float64")
        rr, cc = np.mgrid[0:r1 - r0, 0:c1 - c0].astype("float64")
        conv = lambda x: fftconvolve(x, disk, mode="same")
        s0 = conv(m)
        sr, sc = conv(m * rr), conv(m * cc)
        srr, scc, src = conv(m * rr * rr), conv(m * cc * cc), conv(m * rr * cc)
        own = m > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            n0 = np.round(s0)
            mr, mc = sr / s0, sc / s0
            crr, ccc, crc = srr / s0 - mr ** 2, scc / s0 - mc ** 2, src / s0 - mr * mc
            th = 0.5 * np.arctan2(2 * crc, crr - ccc)          # ось в (ряд, столбец)
            az = np.degrees(np.arctan2(np.sin(th), -np.cos(th))) % 180.0
        az = np.where(own & (n0 >= 2), az, np.nan)
        blk = out[r0:r1, c0:c1]
        blk[own] = az[own]
    return out


def ref_directions(layer: str, w: dict, shape) -> tuple[np.ndarray, np.ndarray]:
    """Угол звена эталона на его ячейках растра (NaN вне линий) и номер линии."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely.geometry import LineString

    from experiments.openlin_transfer_check import GTK

    g = gpd.read_file(GTK / f"lin500k_{layer}.gpkg").to_crs(epsg=int(w["epsg"]))
    res = float(w["res_m"])
    tr = Affine(res, 0, float(w["x0"]), 0, -res, float(w["y_top"]))
    segs, ids = [], []
    for i, geom in enumerate(g.geometry):
        if geom is None:
            continue
        for p in getattr(geom, "geoms", [geom]):
            xy = np.asarray(p.coords)[:, :2]
            for (xa, ya), (xb, yb) in zip(xy[:-1], xy[1:]):
                az = float(np.degrees(np.arctan2(xb - xa, yb - ya)) % 180.0)
                segs.append((LineString([(xa, ya), (xb, yb)]), az))
                ids.append((LineString([(xa, ya), (xb, yb)]), i + 1))
    ang = rasterize(segs, out_shape=shape, transform=tr, fill=np.nan,
                    all_touched=True, dtype="float32")
    fid = rasterize(ids, out_shape=shape, transform=tr, fill=0,
                    all_touched=True, dtype="int32")
    return ang, fid


def shifts(shape, n: int = N_SHIFTS, seed: int = SEED) -> list:
    """Общий список сдвигов (dr, dc), поля min(200, H/8); строится один раз."""
    H, W = shape
    rng = np.random.default_rng(seed)
    lo_r, lo_c = min(200, H // 8), min(200, W // 8)
    return [(int(rng.integers(lo_r, H - lo_r)), int(rng.integers(lo_c, W - lo_c)))
            for _ in range(n)]


def rank_lines(mask: np.ndarray, res: float):
    """Компоненты по убыванию длины (км), связи — по первой ячейке построчно.

    Возвращает метки, ранг каждой метки (0 — самая длинная) и длины по рангу.
    """
    lab, n = components(mask)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    first = np.full(n, np.iinfo(np.int64).max)
    flat = lab.ravel()
    nz = np.flatnonzero(flat)
    np.minimum.at(first, flat[nz] - 1, nz)
    order = np.lexsort((first, -size))                    # длина, затем первая ячейка
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    return lab, n, rank, size[order] * res / 1000.0


def _min_rank(ref_rc, ref_ang, tree, our_ang, our_rank, res):
    """Для каждой ячейки эталона — наименьший ранг нашей линии, которая её находит."""
    from scipy.spatial import cKDTree

    out = np.full(len(ref_rc), np.iinfo(np.int64).max)
    if not len(ref_rc):
        return out
    t_ref = cKDTree(ref_rc * res)
    pairs = t_ref.sparse_distance_matrix(tree, TOL_M, output_type="ndarray")
    if not len(pairs):
        return out
    i, j = pairs["i"], pairs["j"]
    ok = _axial_diff(ref_ang[i], our_ang[j]) <= DIR_TOL_DEG
    np.minimum.at(out, i[ok], our_rank[j[ok]])
    return out


def positions(ref_ang, fid, valid, shift_list):
    """Ячейки эталона на месте и на каждом сдвиге (только в общих ячейках)."""
    H, W = valid.shape
    rr, cc = np.nonzero(np.isfinite(ref_ang))
    a, f = ref_ang[rr, cc], fid[rr, cc]
    out = []
    for dr, dc in [(0, 0)] + list(shift_list):
        r2, c2 = (rr + dr) % H, (cc + dc) % W
        ok = valid[r2, c2]
        out.append((np.stack([r2[ok], c2[ok]], 1), a[ok], f[ok]))
    return out


def evaluate(mask, pos, res):
    """Для набора: длины по рангу и наименьший ранг у ячеек эталона на каждом положении."""
    from scipy.spatial import cKDTree

    lab, n, rank, lens = rank_lines(mask, res)
    our_dir = our_directions(mask, res, lab, n)
    rr, cc = np.nonzero(mask & np.isfinite(our_dir))
    tree = cKDTree(np.stack([rr, cc], 1) * res)
    our_ang = our_dir[rr, cc]
    our_rank = rank[lab[rr, cc] - 1]
    mins = [_min_rank(rc, ang, tree, our_ang, our_rank, res) for rc, ang, _f in pos]
    return lens, mins


def curve_score(lens, mins, K: int) -> dict:
    """Кривая R_k, N_k на L_k = k*LEN_STEP_KM и S = sum(R-N) / sum(1-N), k = 1..K."""
    cum = np.cumsum(lens)
    L = LEN_STEP_KM * np.arange(1, K + 1)
    cut = np.searchsorted(cum, L, side="right")        # линий с суммой длины <= L
    R = np.array([(mins[0] < c).mean() for c in cut])
    N = np.array([np.mean([(m < c).mean() for m in mins[1:]]) for c in cut])
    S = float((R - N).sum() / (1.0 - N).sum())
    return dict(L=L, R=R, N=N, S=S)


def recall_dir(mins) -> tuple:
    """(направленная полнота всего набора, средняя по сдвигам полнота нуля)."""
    big = np.iinfo(np.int64).max
    return float((mins[0] < big).mean()), float(np.mean([(m < big).mean() for m in mins[1:]]))


def long_lines(mins, pos, res: float) -> dict:
    """Доля линий эталона длиннее REF_LONG_KM с >= LINE_FRAC направленно найденной длины."""
    big = np.iinfo(np.int64).max

    def share(m, f, long_ids):
        found = np.bincount(f, weights=(m < big), minlength=long_ids.size)
        total = np.bincount(f, minlength=long_ids.size)
        sel = long_ids[:len(total)] & (total > 0)
        return float((found[sel] / total[sel] >= LINE_FRAC).mean()) if sel.any() else np.nan

    f0 = pos[0][2]
    length = np.bincount(f0) * res / 1000.0
    long_ids = length >= REF_LONG_KM
    obs = share(mins[0], f0, long_ids)
    null = float(np.nanmean([share(m, p[2], long_ids) for m, p in zip(mins[1:], pos[1:])]))
    return dict(n_long=int(long_ids.sum()), obs=obs, null=null,
                adj=(obs - null) / (1 - null) if null < 1 else np.nan)


def main() -> None:
    import argparse

    from experiments.openlin_transfer_check import COMP, M, MAIN_LAYER
    from experiments.openlin_union_recall import add_new

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="utsjoki")
    a = ap.parse_args()
    if a.site != "utsjoki":
        raise SystemExit("сравнение методов — на площадке настройки; контроль — "
                         "отдельным объявлением")

    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    o = np.load(M / f"openlin_zone_lines_{a.site}_v3_настройка1.npz")
    r = np.load(M / f"openlin_zone_lines_{a.site}_v3_рельеф1.npz")
    # RELU2 (ложбины, объявление 05.11.2026): если прогон есть, в сравнение
    # входят ещё «ложбины» и «снимок + рельеф + ложбины» (то же правило сложения)
    pv = M / f"openlin_zone_lines_{a.site}_v3_ложбины1.npz"
    v = np.load(pv) if pv.exists() else None
    res = float(w["res_m"])
    shape = o["C_all"].shape
    valid = np.isfinite(w["shade"][:shape[0], :shape[1]])
    ang, fid = ref_directions(MAIN_LAYER, w, shape)
    ang = np.where(valid, ang, np.nan)
    sh = shifts(shape)
    pos = positions(ang, fid, valid, sh)
    print(f"площадка {a.site}: эталон {MAIN_LAYER}, ячеек {len(pos[0][0])}; допуск "
          f"{TOL_M:.0f} м и {DIR_TOL_DEG:.0f}°, окно направления {DIR_WIN_M:.0f} м, "
          f"шаг L {LEN_STEP_KM:.0f} км, сдвигов {len(sh)} (зерно {SEED})", flush=True)

    rows, res_by = [], {}
    for fam, key in (("весь лист", "C_all"), ("в зонах", "C")):
        om, rm = o[key][:shape[0], :shape[1]], r[key][:shape[0], :shape[1]]
        sets = {"снимок": om, "рельеф": rm, "снимок + рельеф": om | add_new(rm, om, res)}
        if v is not None:
            vm = v[key][:shape[0], :shape[1]]
            base = sets["снимок + рельеф"]
            sets["ложбины"] = vm
            sets["снимок + рельеф + ложбины"] = base | add_new(vm, base, res)
        ev = {}
        for s, m in sets.items():
            ev[s] = evaluate(m, pos, res)
            print(f"  [{fam}] {s}: посчитано", flush=True)
        K = int(np.floor(min(lens.sum() for lens, _m in ev.values()) / LEN_STEP_KM))
        print(f"\nсемейство «{fam}»: K = {K} (L до {K * LEN_STEP_KM:.0f} км)")
        for s, (lens, mins) in ev.items():
            cs = curve_score(lens, mins, K)
            rd, rn = recall_dir(mins)
            ll = long_lines(mins, pos, res)
            res_by[(fam, s)] = (lens, mins)
            print(f"  {s:16s} {lens.sum():6.0f} км | ГЛАВНАЯ S = {cs['S']:.3f} | "
                  f"при L = {K * LEN_STEP_KM:.0f} км: {100 * cs['R'][-1]:.1f} % "
                  f"(нуль {100 * cs['N'][-1]:.1f}) | весь набор, направленно: "
                  f"{100 * rd:.1f} % (нуль {100 * rn:.1f}) | линий эталона "
                  f">= {REF_LONG_KM:.0f} км найдено: {100 * ll['obs']:.0f} % из {ll['n_long']} "
                  f"(нуль {100 * ll['null']:.0f} %)")
            rows.append(dict(семейство=fam, набор=s, км=lens.sum(), K=K, S=cs["S"],
                             R_K=cs["R"][-1], N_K=cs["N"][-1], полнота_напр=rd,
                             нуль_напр=rn, линий_длинных=ll["n_long"],
                             найдено_линий=ll["obs"], нуль_линий=ll["null"]))
    print("\nпары «в зонах / весь лист» одного метода (K по более короткому):")
    for s in [k for f, k in res_by if f == "весь лист"]:
        lz, mz = res_by[("в зонах", s)]
        la, ma = res_by[("весь лист", s)]
        K = int(np.floor(min(lz.sum(), la.sum()) / LEN_STEP_KM))
        print(f"  {s:16s} K = {K}: S в зонах {curve_score(lz, mz, K)['S']:.3f}, "
              f"весь лист {curve_score(la, ma, K)['S']:.3f}")
    import pandas as pd
    out = M / f"openlin_ref_match_{a.site}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
