"""Оставшиеся гипотезы предметной модели на данных листа: Г1, Г3, Г4, Г5, Г10.

Заверка — только золото, в признаки не входит (Ог3): 5 золотых ореолов
(`геохимические ореолы.shp`) и пробы с золотом по коренным и рыхлым (коды 2 и
1, без шлиховых — Ог6), как в Г11 (`experiments/gmo_h11_svity_halos.py`).
Разломные зоны — слои формулы ⚠ (`dist_tect1/2` ≤ 0, 61% листа): здесь они не
признак карты, а рамка, внутри которой гипотезы и сформулированы.

Записано до расчёта. Общая мера — отношение долей положительных ячеек
«в группе / вне группы» по Мантелю — Хензелю внутри слоёв обнажённости (как в
Г11). Неопределённость: для ореолов — перестановка (каждый ореол целиком
поворачивается и кладётся в случайное место листа под картой, в счёт идут его
ячейки внутри рамки гипотезы; поправлено после первого запуска — крупный
ореол в раздробленную рамку целиком не помещается), для проб — бутстрэп
блоков 10 км. «Работает» — отношение > 1 и p < 0.05 по ореолам, и
то же направление по пробам. «Не работает» — отношение ≤ 1 у обеих заверок.
Иначе — «не различимо». Ореолов пять, поэтому заранее известно: обнаружимы
только сильные эффекты.

Г1. Внутри разломных зон, на обнажённых ячейках с ASTER, верхняя треть по
    сигналу 2.33 мкм (К13: средний ранг `ast_mgoh` и `ast_carb`) богаче
    золотом, чем остальная зона. Слои — обнажённость × преобладающая свита
    (каолинит в Г2 следил за свитой; 2.33 мкм может так же).
    Справочно — отношение «зона / вне зоны» (не тест: знаменатели разные).
Г3. Ячейки у регионального несогласия (⚠ `geo2_contact` ≤ 2 км) внутри
    разломных зон богаче золотом, чем остальная зона. Справочно — у
    несогласия вне зон. Подтверждение знания, не независимость: оба слоя —
    входы формулы.
Г4. Ветвь переноса по форме ореолов. Длинная ось ореола (минимальный
    описанный прямоугольник, удлинение ≥ 1.5) сравнивается с простиранием
    разломов — СЗ 315°, Билляхская зона и слои `glub_*_nw` — и с простиранием
    базальных горизонтов: ближайшего отрезка несогласия чехла с фундаментом
    (⚠ `geo2_strike`, это азимут контакта, а не разлома; поправлено до
    расчёта) и контактов свит в 2 км вокруг ореола (`svita_labeled`). «Растворённая» — не меньше
    4 из удлинённых ореолов ближе 30° к разломам и дальше от контактов,
    «обломочная» — наоборот, иначе «не различимо». Случайно ближе 30° к
    заданному направлению — с вероятностью 1/3.
Г5. Внутри магнитоактивных блоков (региональное поле `pf_mag_up10` выше
    медианы листа) ячейки с местным магнитным дефицитом (остаток
    `pf_mag_up1` − `pf_mag_up10`, нижняя треть внутри блоков) богаче
    золотом, чем остальной блок. Чувствительность — остаток съёмки
    `gm_mag_ost_35`.
Г10. Внутри разломных зон ячейки верхней трети звена «место накопления»
    пробной карты v0.2 (вариант А: узлы линеаментов, согласованность
    линеаментов, свиты списка) богаче золотом, чем остальная зона. По
    объектам отдельно: узлы `linc_node_dens`, складки `linc_aniso_r`,
    спрямлённые долины `linv_dens`; справочно вариант Б — ⚠ узлы
    `geo2_fault_node`. Спрямлённые долины — с оговоркой: пробы по рыхлым
    отбирают в долинах.

Выход: консоль и `outputs/13_генетическая_модель/gmo_hyp_remaining.png`.

Запуск::

    python -X utf8 -m experiments.gmo_hyp_remaining
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gmo_h11_svity_halos as h  # noqa: E402
from src import anabar_inset, mapstyle, outputs, spectral_mask  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
LINC = ROOT / "data/processed/lineament_clean.parquet"
KMAP = ROOT / "data/processed/knowledge_map_v0_2.parquet"
SVITY = ROOT / "data/processed/svita_labeled.gpkg"
CRS = anabar_inset.SHEET_CRS
h.N_PERM, h.N_BOOT = 1000, 1000
VERDICTS: list[tuple[str, str, str]] = []


def load() -> tuple[pd.DataFrame, list, gpd.GeoDataFrame]:
    df = h.load_cells()
    cols = ["row", "col", "dist_tect1", "dist_tect2", "geo2_contact", "geo2_fault_node",
            "geo2_strike_cos", "geo2_strike_sin", "ast_mgoh", "ast_carb",
            "pf_mag_up1", "pf_mag_up10", "gm_mag_ost_35"]
    df = df.merge(pd.read_parquet(DATA, columns=cols), on=["row", "col"], how="left")
    df = df.merge(pd.read_parquet(LINC, columns=["row", "col", "linc_node_dens",
                                                 "linc_aniso_r", "linv_dens"]),
                  on=["row", "col"], how="left")
    km = pd.read_parquet(KMAP, columns=["row", "col", "звено Н место накопления"])
    df = df.merge(km.rename(columns={"звено Н место накопления": "link_N"}),
                  on=["row", "col"], how="left")
    df["zone"] = (df.dist_tect1 <= 0) | (df.dist_tect2 <= 0)
    halos = gpd.read_file(h.SHP / "геохимические ореолы.shp").set_crs(CRS, allow_override=True)
    au_h = [shapely.Polygon(g.coords) for g in halos[halos.label == "Au"].geometry]
    pts = gpd.read_file(h.SHP / "геохимическое_опробование.shp").set_crs(CRS, allow_override=True)
    pts = pts[pts.code.isin([1, 2]) & pts.label.astype(str).str.contains("Au")]
    return df, au_h, pts


LAST_NULL: dict = {}
SHEET: pd.DataFrame | None = None     # все ячейки листа под картой (load)
FOOT: list[list[np.ndarray]] = []       # следы ореолов на сетке по поворотам


def footprints(polys, n_rot: int = 36) -> list[list[np.ndarray]]:
    """След ореола на ячейках 500 м при n_rot поворотах вокруг центра.

    Каждый след — массив (dr, dc, доля ячейки под ореолом) относительно
    ячейки центра; доли — растр 50 м, как в `halo_fraction`.
    """
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely import affinity
    out = []
    for p in polys:
        rots = []
        for k in range(n_rot):
            q = affinity.rotate(p, 360.0 * k / n_rot, origin="centroid")
            c = q.centroid
            q = affinity.translate(q, -c.x, -c.y)
            half = int(np.ceil(max(abs(v) for v in q.bounds) / h.CELL)) + 1
            n = 2 * half + 1
            T = from_origin(-(half + 0.5) * h.CELL, (half + 0.5) * h.CELL, h.CELL / h.SUB, h.CELL / h.SUB)
            r = features.rasterize([(q, 1)], out_shape=(n * h.SUB, n * h.SUB), transform=T,
                                   fill=0, dtype="uint8")
            f = r.reshape(n, h.SUB, n, h.SUB).mean(axis=(1, 3))
            rr, cc = np.nonzero(f > 0)
            rots.append(np.column_stack([rr - half, cc - half, f[rr, cc]]))
        out.append(rots)
    return out


def fast_perm(d: pd.DataFrame, n_perm: int = 1000, thr: float = 0.1) -> np.ndarray:
    """Нуль ореолов: каждый ореол с поворотом — в случайное место ЛИСТА, целиком
    под картой; в счёт идут только его ячейки, попавшие в рамку гипотезы `d`.
    Требовать, чтобы ореол целиком лёг в раздробленную рамку (обнажённые
    ячейки зон), нельзя: крупные ореолы туда не помещаются вовсе."""
    nrow, ncol = int(SHEET.row.max()) + 1, int(SHEET.col.max()) + 1
    sheet = np.zeros((nrow, ncol), bool)
    sheet[SHEET.row, SHEET.col] = True
    idx = np.full((nrow, ncol), -1)
    idx[d.row.to_numpy(), d.col.to_numpy()] = np.arange(len(d))
    cr, cc_ = SHEET.row.to_numpy(), SHEET.col.to_numpy()
    out, ncell = [], []
    global LAST_NULL
    redo = 0
    while len(out) < n_perm:
        pos = np.zeros(len(d), bool)
        placed = 0
        for rots in FOOT:
            for _try in range(500):
                f = rots[h.rng.integers(len(rots))]
                j = h.rng.integers(len(cr))
                r = cr[j] + f[:, 0].astype(int)
                c = cc_[j] + f[:, 1].astype(int)
                if r.min() < 0 or c.min() < 0 or r.max() >= nrow or c.max() >= ncol:
                    continue
                if not sheet[r, c].all():
                    continue
                k = idx[r, c]
                hit = (k >= 0) & (f[:, 2] >= thr)
                pos[k[hit]] = True
                placed += 1
                break
        if placed < len(FOOT):      # раскладка не урезается, а пересоздаётся
            redo += 1
            continue
        out.append(h.mh_ratio(pos, d))
        ncell.append(int(pos.sum()))
    out = np.array(out)
    LAST_NULL = {"nan": float(np.isnan(out).mean()), "inf": float(np.isinf(out).mean()),
                 "ячеек": float(np.mean(ncell)), "q95": float(np.nanpercentile(out[np.isfinite(out)], 95)),
                 "пересоздано": redo}
    return out


def test(name: str, frame: pd.DataFrame, group: np.ndarray, au_h, pts, strata=None) -> dict:
    """Отношение М-Х «группа / остальная рамка» по ореолам и пробам."""
    d = frame.reset_index(drop=True).copy()
    d["in_list"] = np.asarray(group, bool)
    if strata is not None:
        d["stratum"] = np.asarray(strata)
    pos_h = h.halo_fraction(au_h, d) >= 0.1
    pos_p = h.point_cells(pts, d)
    r_h, r_p = h.mh_ratio(pos_h, d), h.mh_ratio(pos_p, d)
    used = [p for p in au_h if (h.halo_fraction([p], d) >= 0.1).any()]
    perm = fast_perm(d)
    p_h = h.perm_p(perm, r_h) if not np.isnan(r_h) else np.nan
    try:
        lo, hi = h.block_boot(pos_p, d) if pos_p.any() else (np.nan, np.nan)
    except IndexError:          # во всех повторах отношение не определено
        lo, hi = np.nan, np.nan
    res = {"проверка": name, "ячеек": len(d), "в группе": f"{d.in_list.mean():.0%}",
           "ореолов": len(used), "ореол-ячеек": int(pos_h.sum()),
           "в группе ореол": int(pos_h[d.in_list].sum()), "отн. ореолы": round(r_h, 2),
           "p ореолы": round(p_h, 3), "проб-ячеек": int(pos_p.sum()),
           "в группе проб": int(pos_p[d.in_list].sum()),
           "отн. пробы": round(r_p, 2), "95% пробы": f"[{lo:.2f}; {hi:.2f}]"}
    print(f"   {name}: ореолы {r_h:.2f} (p={p_h:.3f}, {len(used)} ореолов, "
          f"{int(pos_h[d.in_list].sum())}/{int(pos_h.sum())} ячеек в группе); "
          f"пробы {r_p:.2f} [{lo:.2f}; {hi:.2f}] ({int(pos_p[d.in_list].sum())}/"
          f"{int(pos_p.sum())}); группа — {d.in_list.mean():.0%} рамки", flush=True)
    n = LAST_NULL
    print(f"      нуль: ореол-ячеек в рамке в среднем {n['ячеек']:.0f} (наблюдено {int(pos_h.sum())}); "
          f"95-й перцентиль отношения {n['q95']:.2f}; не определено {n['nan']:.0%}, "
          f"бесконечно {n['inf']:.0%}; пересоздано {n['пересоздано']}", flush=True)
    res["нуль q95"] = round(n["q95"], 2)
    return res


def verdict(code: str, r: dict, extra: str = "") -> None:
    rh, rp, ph = r["отн. ореолы"], r["отн. пробы"], r["p ореолы"]
    if rh > 1 and ph < 0.05 and rp > 1:
        v = "работает"
    elif rh <= 1 and rp <= 1:
        v = "не работает"
    else:
        v = "не различимо"
    VERDICTS.append((code, v, extra))
    print(f"   => {code}: {v}. {extra}")


def tercile_top(s: pd.Series) -> np.ndarray:
    """Верхняя треть; при массе нулей (нет линий) — ячейки с ненулевым значением."""
    q = s.quantile(2 / 3)
    return ((s >= q) if q > s.min() else (s > s.min())).to_numpy()


def g1(df, au_h, pts) -> list[dict]:
    print("\nГ1: сигнал 2.33 мкм внутри разломных зон")
    ref = test("справочно: зона / вне зоны", df, df.zone, au_h, pts)
    exp = spectral_mask.exposure_mask(df)
    z = df[df.zone & exp & df.ast_mgoh.notna() & df.ast_carb.notna()].copy()
    k13 = (z.ast_mgoh.rank(pct=True) + z.ast_carb.rank(pct=True)) / 2
    strata = z.stratum.astype(str) + "|" + z.svk_main.fillna("вне").astype(str)
    r = test("К13 верхняя треть / остальная зона (обнажённые)", z, tercile_top(k13), au_h, pts,
             strata=strata)
    r2 = test("то же, слои только по обнажённости", z, tercile_top(k13), au_h, pts)
    verdict("Г1", r, f"Справочно положение в зоне: {ref['отн. ореолы']} по ореолам.")
    return [ref, r, r2]


def g3(df, au_h, pts) -> list[dict]:
    print("\nГ3: несогласие в разломных зонах (⚠ оба слоя — входы формулы)")
    z = df[df.zone]
    out = [test(f"у несогласия ≤ {d / 1000:g} км / остальная зона", z, (z.geo2_contact <= d),
                au_h, pts) for d in (2000, 1000, 4000)]
    nz = df[~df.zone]
    out.append(test("справочно: у несогласия ≤ 2 км вне зон", nz, nz.geo2_contact <= 2000,
                    au_h, pts))
    verdict("Г3", out[0], "Подтверждение знания формулы, не независимость.")
    return out


def axis_deg(geom) -> tuple[float, float]:
    """Азимут длинной оси (0–180°, от севера по часовой) и удлинение."""
    rect = shapely.minimum_rotated_rectangle(geom)
    c = np.asarray(rect.exterior.coords)[:4]
    e = [c[1] - c[0], c[2] - c[1]]
    L = [np.hypot(*v) for v in e]
    v = e[int(np.argmax(L))]
    return float(np.degrees(np.arctan2(v[0], v[1])) % 180), max(L) / max(min(L), 1.0)


def mean_axis(angles_deg: np.ndarray, w: np.ndarray) -> float:
    a = np.radians(2 * angles_deg)
    return float(np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2 % 180)


def ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def g4(df, au_h) -> pd.DataFrame:
    print("\nГ4: ветвь переноса по форме ореолов")
    sv = gpd.read_file(SVITY).to_crs(CRS)
    bedrock = sv[~sv.svita.str.contains("отложения|горизонт|долериты|диабазы")]
    edges = shapely.union_all(list(bedrock.geometry.boundary))
    frame = shapely.union_all(list(sv.geometry)).buffer(-300)
    edges = edges.intersection(frame)
    rows = []
    pts = gpd.GeoSeries(gpd.points_from_xy(df.x, df.y), crs=CRS)
    for i, p in enumerate(au_h):
        az, el = axis_deg(p)
        near = df.iloc[np.flatnonzero(pts.distance(p.centroid).to_numpy() <= 2000)]
        # geo2_strike — sin/cos удвоенного АЗИМУТА ближайшего отрезка несогласия
        # (src/geology_v2.contact_features: arctan2(dx, dy))
        us = float(np.degrees(np.arctan2(near.geo2_strike_sin.mean(),
                                         near.geo2_strike_cos.mean())) / 2 % 180)
        fs = 135.0
        seg = edges.intersection(p.buffer(2000))
        lines = [g for g in getattr(seg, "geoms", [seg]) if not g.is_empty]
        angs, ws = [], []
        for ln in lines:
            for part in getattr(ln, "geoms", [ln]):
                c = np.asarray(part.coords)
                v = np.diff(c, axis=0)
                angs += list(np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 180)
                ws += list(np.hypot(v[:, 0], v[:, 1]))
        cs = mean_axis(np.array(angs), np.array(ws)) if ws else np.nan
        rows.append({"ореол": i + 1, "площадь, км²": round(p.area / 1e6, 1),
                     "ось, °": round(az), "удлинение": round(el, 2),
                     "разломы, °": round(fs), "Δ к разломам": round(ang_diff(az, fs)),
                     "несогласие, °": round(us), "Δ к несогласию": round(ang_diff(az, us)),
                     "контакты, °": round(cs) if np.isfinite(cs) else None,
                     "длина контактов, км": round(sum(ws) / 1e3, 1),
                     "Δ к контактам": round(ang_diff(az, cs)) if np.isfinite(cs) else None})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t["разломы ~ несогласие, °"] = [round(ang_diff(135, u)) for u in t["несогласие, °"]]
    print(t[["ореол", "разломы ~ несогласие, °"]].to_string(index=False))
    # После ревью: ореолы, где направления разломов и несогласия ближе 30°,
    # ветви не различают — не учитываются (иначе правило смещено к «обломочной»)
    el = t[(t["удлинение"] >= 1.5) & (t["разломы ~ несогласие, °"] >= 30)]
    base = el[["Δ к несогласию", "Δ к контактам"]].fillna(90).min(axis=1)
    diss = int(((el["Δ к разломам"] <= 30) & (base > el["Δ к разломам"])).sum())
    clast = int(((base <= 30) & (el["Δ к разломам"] > base)).sum())
    v = ("не проверяема" if len(el) < 4 else "растворённая" if diss >= 4
         else "обломочная" if clast >= 4 else "не различимо")
    VERDICTS.append(("Г4", v, f"удлинённых {len(el)} из 5; вдоль разломов {diss}, вдоль контактов {clast}"))
    print(f"   => Г4: {v} (удлинённых {len(el)}; вдоль разломов {diss}, вдоль контактов {clast})")
    return t


def g5(df, au_h, pts) -> list[dict]:
    print("\nГ5: местный магнитный дефицит в магнитоактивном блоке")
    act = df[df.pf_mag_up10 > df.pf_mag_up10.median()].copy()
    res = act.pf_mag_up1 - act.pf_mag_up10
    r = test("дефицит (нижняя треть остатка) / остальной блок", act,
             (res <= res.quantile(1 / 3)).to_numpy(), au_h, pts)
    r2 = test("то же по остатку съёмки gm_mag_ost_35", act,
              (act.gm_mag_ost_35 <= act.gm_mag_ost_35.quantile(1 / 3)).to_numpy(), au_h, pts)
    inact = df[df.pf_mag_up10 <= df.pf_mag_up10.median()].copy()
    res0 = inact.pf_mag_up1 - inact.pf_mag_up10
    r3 = test("справочно: дефицит вне магнитоактивных блоков", inact,
              (res0 <= res0.quantile(1 / 3)).to_numpy(), au_h, pts)
    r4 = test("справочно: магнитоактивный блок / остальной лист", df,
              (df.pf_mag_up10 > df.pf_mag_up10.median()).to_numpy(), au_h, pts)
    verdict("Г5", r, f"По остатку съёмки: ореолы {r2['отн. ореолы']}, пробы {r2['отн. пробы']}.")
    return [r, r2, r3, r4]


def g10(df, au_h, pts) -> list[dict]:
    print("\nГ10: места накопления внутри разломных зон")
    z = df[df.zone & df.link_N.notna()]
    out = [test("звено Н карты v0.2, верхняя треть / остальная зона", z, tercile_top(z.link_N),
                au_h, pts)]
    for col, nm in (("linc_node_dens", "узлы линеаментов"), ("linc_aniso_r", "согласованность линеаментов"),
                    ("linv_dens", "спрямлённые долины"), ("geo2_fault_node", "⚠ узлы разломов формулы")):
        s = -z[col] if col == "geo2_fault_node" else z[col]   # у формулы — расстояние
        top = tercile_top(s) if s.nunique() > 3 else (s > 0).to_numpy()
        out.append(test(f"{nm}: верхняя треть / остальная зона", z, top, au_h, pts))
    # Контроль, добавленный ПОСЛЕ первого запуска (спрямлённые долины дали
    # 2.4 по ореолам и 5.8 по пробам): опробование идёт по долинам, поэтому
    # спрямлённые долины сравниваются с обычными на том же расстоянии до
    # тальвега — слои «обнажённость × расстояние до тальвега».
    zz = z.merge(pd.read_parquet(LINC, columns=["row", "col", "dist_thalweg"]),
                 on=["row", "col"], how="left")
    dcls = np.digitize(zz.dist_thalweg, [250, 500, 1000])
    out.append(test("контроль: спрямлённые долины при равном расстоянии до тальвега", zz,
                    tercile_top(zz.linv_dens), au_h, pts,
                    strata=zz.stratum.astype(str) + "|" + pd.Series(dcls).astype(str)))
    near = zz[zz.dist_thalweg <= 500]
    out.append(test("контроль: только ячейки ≤ 500 м от тальвега", near,
                    tercile_top(near.linv_dens), au_h, pts))
    # После ревью: один порог (linv > 0), пробы раздельно по коренным и
    # рыхлым, без каждого ореола по очереди, ячейки накопления стока.
    grp = (zz.linv_dens > 0).to_numpy()
    st = zz.stratum.astype(str) + "|" + pd.Series(dcls).astype(str)
    print(f"   спрямлённые долины (linv > 0) — {grp.mean():.0%} рамки")
    for code, nm in ((2, "по коренным"), (1, "по рыхлым")):
        out.append(test(f"контроль: пробы {nm}", zz, grp, au_h, pts[pts.code == code], strata=st))
    for i in range(len(au_h)):
        out.append(test(f"контроль: без ореола {i + 1}", zz, grp,
                        [p for j, p in enumerate(au_h) if j != i], pts, strata=st))
    verdict("Г10", out[0], "Вариант А, звено Н целиком.")
    return out


def draw(df, au_h, pts, t4) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.6))
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    x0 = (df.x - (df.col + 0.5) * h.CELL).iloc[0]
    ytop = (df.y + (df.row + 0.5) * h.CELL).iloc[0]
    ext = [x0, x0 + ncol * h.CELL, ytop - nrow * h.CELL, ytop]
    exp = spectral_mask.exposure_mask(df)
    k13 = ((df.ast_mgoh.rank(pct=True) + df.ast_carb.rank(pct=True)) / 2).where(exp & df.zone)
    act = df.pf_mag_up10 > df.pf_mag_up10.median()
    res = (df.pf_mag_up1 - df.pf_mag_up10).where(act)
    panels = [(k13, "Г1: сигнал 2.33 мкм в разломных зонах,\nобнажённые ячейки (доля 0–1)", "viridis"),
              (res, "Г5: остаток магнитного поля в магнитоактивных\nблоках (поле на 1 км минус на 10 км)", "RdBu_r"),
              (df.link_N.where(df.zone), "Г10: звено «место накопления» v0.2\nвнутри разломных зон (доля 0–1)", "viridis")]
    for ax, (v, ttl, cm) in zip(axs, panels):
        a = np.full((nrow, ncol), np.nan)
        a[df.row, df.col] = v
        lo, hi = np.nanpercentile(a, [2, 98])
        im = ax.imshow(a, extent=ext, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75)
        gpd.GeoSeries(au_h).boundary.plot(ax=ax, color="#e67e22", linewidth=2)
        ax.scatter(pts.geometry.x, pts.geometry.y, s=14, color="#c0392b", edgecolor="k",
                   linewidth=0.3, zorder=5)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
        ax.ticklabel_format(style="plain"); ax.tick_params(labelsize=6)
    for _, r in t4.iterrows():
        c = au_h[int(r["ореол"]) - 1].centroid
        axs[2].annotate(str(int(r["ореол"])), (c.x, c.y), fontsize=10, weight="bold",
                        color="white", ha="center", va="center")
    fig.suptitle("Лист R-48-XI,XII, сетка 500 м. Оранжевые контуры — ореолы Au, "
                 "красные точки — пробы Au по коренным и рыхлым (заверка, не признак)", fontsize=11)
    fig.tight_layout()
    out = outputs.png("gmo_hyp_remaining.png")
    mapstyle.save_fig(fig, out, dpi=130)
    print(f"\nРисунок: {out}")


def main() -> None:
    global SHEET, FOOT
    df, au_h, pts = load()
    SHEET = df[["row", "col"]]
    FOOT = footprints(au_h)
    # проверка нуля: на всём листе «зона / вне зоны» — сравнить с прежней
    # перестановкой Г11 нельзя напрямую, печатается средний нулевой уровень
    print(f"Ячеек {len(df)}; в разломных зонах {df.zone.mean():.0%}; ореолов Au {len(au_h)}, "
          f"проб Au {len(pts)}")
    pc = h.point_cells(pts, df)
    hf = h.halo_fraction([p.buffer(1000) for p in au_h], df) > 0
    print(f"Пробы и ореолы — одна ли заверка: из {int(pc.sum())} ячеек с пробами "
          f"{int((pc & hf).sum())} внутри ореолов или ближе 1 км к ним")
    print("\nКалибровка нуля на постановке Г11 (свиты списка, весь лист; прежняя "
          "перестановка дала p = 0.39):")
    test("калибровка: свиты списка / вне", df, df.in_list, au_h, pts)
    g1(df, au_h, pts)
    g3(df, au_h, pts)
    t4 = g4(df, au_h)
    g5(df, au_h, pts)
    g10(df, au_h, pts)
    print("\nИтог:")
    for c, v, e in VERDICTS:
        print(f"   {c}: {v}. {e}")
    draw(df, au_h, pts, t4)


if __name__ == "__main__":
    main()
