"""Пробная карта на знаниях v0, вариант А: только свои каналы, без слоёв формулы.

Цель: впервые пройти цепочку предметной модели целиком — признак → критерий →
объект → звено → итог ячейки с уверенностью и объяснением. Это НЕ итоговая
карта: направления влияния — наше предложение (ВНИГНИ ещё не ответили на пять
вопросов по объектам), веса не заданы, звено «сохранность» не наблюдается
(этап IV пишется по литературе). Модель — черновик
`docs II presentation/Предметная-модель-v1-черновик.md`.

Входные данные: `data/processed/dataset_v5_rebuilt.parquet` (каналы),
`data/processed/k23_svity_grid.parquet` (свиты, К23). Слоёв формулы ⚠ нет.

Правила свёртки (раздел 6 черновика):
* признак → доля 0–1 по рангу среди наблюдённых ячеек с направлением
  (↑ — больше лучше, ↓ — меньше лучше, ◇ — ближе к нулю лучше: кромка по
  тилт-производной); спектральные каналы — только на обнажённых ячейках
  (`spectral_mask`), под растительностью «не наблюдается», а не ноль;
* критерий — среднее своих признаков, объект — среднее своих критериев;
* звено — нечёткое «или» (максимум) по объектам: места накопления и
  источники — альтернативы;
* итог — мягкое «и»: среднее геометрическое наблюдённых звеньев; ненаблюдённое
  звено не обнуляет ячейку, а снижает уверенность (доля наблюдённых звеньев);
* Ог1: разломная зона закрывает не больше одного звена — для каждой ячейки
  она отдаётся тому из звеньев И, П, Н, где даёт наибольший итог, из двух
  других убирается.

Ожидаемый результат: `data/processed/knowledge_map_v0.parquet`,
`outputs/13_генетическая_модель/gmo_knowledge_map_v0.png`, объяснение
верхних ячеек и сверка — только как заверка, не мишень: с критериальной
картой ВНИГНИ и с золотом (5 ореолов, 45 проб без шлиховых).

Запуск::

    python -X utf8 -m experiments.gmo_knowledge_map_v0
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import mapstyle, outputs, spectral_mask  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data/processed/dataset_v5_rebuilt.parquet"
K23 = ROOT / "data/processed/k23_svity_grid.parquet"
OUT = ROOT / "data/processed/knowledge_map_v0.parquet"
TAG = "_v0"
FAULT = "О1 разломная зона"

#: Предметная модель в виде данных: звено → объект → критерий → [(признак, направление)].
#: «доля» — признак уже доля 0–1 со смыслом, ранг не нужен.
MODEL = {
    "И источник": {
        FAULT: {"К2 близость к разломной зоне": [("lin_dist", "↓")],
                "К4 разломные границы в полях": [("pf_gr_tdr", "◇"), ("pf_mag_tdr", "◇"),
                                                 ("gm_gr_1G_25", "↑")]},
        "О4 интрузии": {"К11 приповерхностные магнитные тела": [("pf_mag_asig", "↑"),
                                                               ("gm_mag_ost_35", "↑")]},
        "О6 диафторез": {"К13 полоса 2.33 мкм": [("ast_mgoh", "↑"), ("ast_carb", "↑")]},
    },
    "М мобилизация": {
        "О7 реактивация и выщелачивание": {
            "К15 серицит-мусковит": [("ast_aloh", "↑"), ("ast_alter", "↑")],
            "К16 окисное железо": [("ast_ferric", "↑"), ("s2_iron_ox", "↑"), ("l8_iron_ox", "↑")],
            "К17 железистые силикаты": [("ast_ferrous", "↑"), ("l8_ferrous", "↑")]},
    },
    "П путь": {
        FAULT: {"К3 раздробленность блока": [("lin_dens", "↑")],
                "К5 простирание вдоль СЗ": [("nw_align", "↑")]},
    },
    "Н место накопления": {
        FAULT: {"К1 узлы пересечения": [("lin_node_dens", "↑")]},
        "О9 складки": {"К21 согласованность линеаментов": [("lin_aniso_r", "↑")]},
        "О14 впадины": {"К26 долины разломного заложения": [("dist_dnara", "↓")]},
    },
    "С сохранность": {},     # этап IV — по литературе, канала пока нет
}
SPECTRAL = ("ast_", "s2_", "l8_")

#: Вмещающая среда звена: не альтернатива местам накопления, а условие для
#: них (О11, ответ ВНИГНИ 4: места накопления — в этих свитах). Поэтому
#: место накопления = min(доля свит из списка, лучшее из мест) — нечёткое «и».
#: В первой сборке О11 стоял в «или» и насыщал звено: свиты — 86% листа.
HOST = {"Н место накопления": ("О11 свиты из списка", "svk_list")}


def load() -> pd.DataFrame:
    feats = sorted({f for objs in MODEL.values() for crits in objs.values()
                    for fl in crits.values() for f, _ in fl} - {"nw_align"})
    df = pd.read_parquet(DATA, columns=["row", "col", "x", "y", spectral_mask.BARE_COL,
                                        "lin_dir_cos", "lin_dir_sin"] + feats)
    df = df.merge(pd.read_parquet(K23, columns=["row", "col", "svk_cover", "svk_list"]),
                  on=["row", "col"], how="left")
    # Северо-запад: азимут 315°, удвоенный — 270°: cos = 0, sin = −1. Совпадение
    # направления линеаментов с ним — проекция на (0, −1). Соглашение об азимуте
    # (от севера по часовой) взято из словаря признаков — проверить по картам.
    df["nw_align"] = -df.lin_dir_sin
    exposed = spectral_mask.exposure_mask(df)
    for c in df.columns:
        if c.startswith(SPECTRAL) and c != spectral_mask.BARE_COL:
            df.loc[~exposed, c] = np.nan
    return df[df.svk_cover.fillna(0) >= 0.5].reset_index(drop=True)


def membership(s: pd.Series, how: str) -> pd.Series:
    if how == "доля":
        return s.clip(0, 1)
    if how == "◇":
        s = -s.abs()
    elif how == "↓":
        s = -s
    return s.rank(pct=True)


def link_scores(df: pd.DataFrame, skip_fault: set[str]) -> tuple[pd.DataFrame, dict]:
    """Оценки звеньев; `skip_fault` — звенья, где разломная зона не считается."""
    scores, parts = {}, {}
    for link, objs in MODEL.items():
        obj_vals = {}
        for obj, crits in objs.items():
            if obj == FAULT and link in skip_fault:
                continue
            crit_vals = [pd.concat([membership(df[f], h) for f, h in fl], axis=1).mean(axis=1)
                         for fl in crits.values()]
            obj_vals[obj] = pd.concat(crit_vals, axis=1).mean(axis=1)
        if obj_vals:
            t = pd.DataFrame(obj_vals)
            scores[link] = t.max(axis=1)
            if link in HOST:
                scores[link] = np.fmin(scores[link], df[HOST[link][1]].clip(0, 1))
            seen = t.notna().any(axis=1)
            lead = pd.Series(None, index=df.index, dtype=object)
            lead[seen] = t[seen].idxmax(axis=1)
            parts[link] = lead
        else:
            scores[link] = pd.Series(np.nan, index=df.index)
            parts[link] = pd.Series(None, index=df.index, dtype=object)
    return pd.DataFrame(scores), parts


def soft_and(sc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    obs = sc.notna()
    logm = np.log(sc.clip(lower=1e-3)).where(obs)
    return np.exp(logm.mean(axis=1)), obs.sum(axis=1) / sc.shape[1]


def build(df: pd.DataFrame) -> pd.DataFrame:
    links_with_fault = [l for l, o in MODEL.items() if FAULT in o]
    best = None
    for keep in links_with_fault:
        sc, parts = link_scores(df, skip_fault=set(links_with_fault) - {keep})
        tot, conf = soft_and(sc)
        cand = sc.add_prefix("звено ").assign(итог=tot, уверенность=conf, разлом_закрыл=keep)
        for l in MODEL:
            cand[f"объект {l}"] = parts[l]
        if best is None:
            best = cand
        else:
            better = cand.итог > best.итог
            best[better] = cand[better]
    return pd.concat([df[["row", "col", "x", "y"]], best], axis=1)


def verify(m: pd.DataFrame) -> None:
    """Только заверка: с критериальной картой и с золотом. Не мишень."""
    from src import criterial_target
    meta, prog = criterial_target.load_prognoz_grid()
    p = np.asarray(prog).reshape(meta.prf, meta.pic)[m.row, m.col]
    ok = np.isfinite(p) & m.итог.notna().to_numpy()
    from scipy.stats import spearmanr
    rho = spearmanr(m.итог[ok], p[ok]).statistic
    top_k = m.итог[ok] >= m.итог[ok].quantile(0.9)
    top_p = p[ok] >= np.quantile(p[ok], 0.9)
    print(f"\nЗаверка с картой ВНИГНИ (не мишень): ранговая корреляция {rho:+.2f}; "
          f"верхние 10% совпадают на {np.mean(top_p[top_k.to_numpy()]):.0%} "
          f"(случайно — 10%)")

    import geopandas as gpd
    import shapely
    from experiments import gmo_h11_svity_halos as h
    from src import anabar_inset
    crs = anabar_inset.SHEET_CRS
    pts = gpd.read_file(h.SHP / "геохимическое_опробование.shp").set_crs(crs, allow_override=True)
    au = pts[pts.code.isin([1, 2]) & pts.label.astype(str).str.contains("Au")]
    halos = gpd.read_file(h.SHP / "геохимические ореолы.shp").set_crs(crs, allow_override=True)
    hp = [shapely.Polygon(g.coords) for g in halos[halos.label == "Au"].geometry]
    cells = m.assign(svk_cover=1.0, svk_list=1.0, s2_bare_frac=0.0)
    pos_p = h.point_cells(au, cells)
    pos_h = h.halo_fraction(hp, cells) >= 0.1
    r = m.итог.rank(pct=True)
    for name, pos in (("пробы Au", pos_p), ("ореолы Au", pos_h)):
        print(f"   {name}: {int(pos.sum())} ячеек; доля в верхних 20% карты "
              f"{np.mean(r[pos] >= 0.8):.0%} (случайно — 20%), средний ранг {r[pos].mean():.2f}")


def explain(m: pd.DataFrame, n: int = 3) -> None:
    print("\nОбъяснение верхних ячеек:")
    for _, c in m.nlargest(n, "итог").iterrows():
        parts = [f"{l.split()[0]} {c['звено ' + l]:.2f} ({c['объект ' + l]})"
                 for l in MODEL if pd.notna(c["звено " + l])]
        print(f"   ячейка {int(c.row)},{int(c.col)}: итог {c.итог:.2f}, уверенность "
              f"{c.уверенность:.1f}; разломная зона закрыла «{c.разлом_закрыл}»")
        print("      " + "; ".join(parts))


def draw(m: pd.DataFrame) -> None:
    nrow, ncol = int(m.row.max()) + 1, int(m.col.max()) + 1
    panels = [f"звено {l}" for l in MODEL if l != "С сохранность"] + ["итог", "уверенность"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, col in zip(axs.ravel(), panels):
        a = np.full((nrow, ncol), np.nan)
        a[m.row, m.col] = m[col]
        im = ax.imshow(a, cmap="viridis" if col != "уверенность" else "Greys",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(col.replace("звено ", "звено: "), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, label="доля 0–1")
    fig.suptitle("Пробная карта на знаниях v0, вариант А (свои каналы), лист R-48-XI,XII, "
                 "сетка 500 м. Звено «сохранность» не наблюдается", fontsize=11)
    fig.tight_layout()
    out = outputs.png(f"gmo_knowledge_map{TAG}.png")
    mapstyle.save_fig(fig, out, dpi=130)
    print(f"\nКарта: {out}")


def main() -> None:
    # Ключ --без-долин: v0.1 без dist_dnara. В v0 он вёл звено «место
    # накопления», и карта повторяла современную речную сеть (Г6: палеорельеф
    # не наблюдается; пробы по рыхлым отбирают в долинах — совпадение с ними
    # ложное).
    global TAG
    if "--без-долин" in sys.argv:
        del MODEL["Н место накопления"]["О14 впадины"]
        TAG = "_v0_1"
    df = load()
    m = build(df)
    m.to_parquet(OUT.with_name(f"knowledge_map{TAG}.parquet"), index=False)
    print(f"Ячеек {len(m)}")
    for l in MODEL:
        s = m[f"звено {l}"]
        print(f"   звено {l}: наблюдается в {s.notna().mean():.0%} ячеек")
    print(f"   уверенность: {m.уверенность.value_counts().sort_index().round(2).to_dict()}")
    print("   разломная зона закрыла:", m.разлом_закрыл.value_counts().to_dict())
    print("   ведущий объект места накопления:",
          m["объект Н место накопления"].value_counts().to_dict())
    explain(m)
    verify(m)
    draw(m)


if __name__ == "__main__":
    main()
