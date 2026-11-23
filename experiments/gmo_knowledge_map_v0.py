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
    python -X utf8 -m experiments.gmo_knowledge_map_v0 --чистые-линеаменты   # v0.2
    python -X utf8 -m experiments.gmo_knowledge_map_v0 --вариант-б           # v0.2, вариант Б
    python -X utf8 -m experiments.gmo_knowledge_map_v0 --чистые-линеаменты --сохранность

Ключ --v0-3 — v0.2 с сохранностью плюс К14 и вмещающая среда по разрезу
(`V03_NOTE`), выход `knowledge_map_v0_3.parquet`; v0.2 не меняется.

Ключ --сохранность добавляет звено С (`preservation`, выход с суффиксом `_s`);
с 22.11.2026 — по ответу ВНИГНИ 19.11: палеопонижения и разломы под покровом.

Вариант Б — v0.2 плюс слои формулы ⚠ (`FORMULA`); с картой ВНИГНИ не
сверяется (Ог2), выход `knowledge_map_v0_2_b.parquet`.
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
LINC = ROOT / "data/processed/lineament_clean.parquet"
CLEAN = False   # --чистые-линеаменты: lin_* заменяются на linc_* (вне долин)
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

#: Вариант Б (ключ --вариант-б, поверх v0.2): к своим каналам добавлены слои
#: критериальной формулы ⚠ — ровно там, где они записаны в таблице критериев
#: черновика (разд. 5). Ог2: вариант Б с картой ВНИГНИ не сверяется — это
#: были бы её же входы. Не добавлены: ⚠`geo2_strike_*` в К5 и К21 — это
#: азимут несогласия, а не простирание разломов (выяснено на Г4, 05.11.2026);
#: К6 (несогласие × разломная зона) — производный, ждёт решения по Ог1.
FORMULA = {
    ("И источник", FAULT, "К2 близость к разломной зоне"): [("dist_tect1", "↓"), ("dist_tect2", "↓")],
    ("П путь", FAULT, "К3 раздробленность блока"): [("dens_tect", "↑")],
    ("Н место накопления", FAULT, "К1 узлы пересечения"): [("geo2_fault_node", "↓")],
    ("П путь", "О2 региональное несогласие", "К7 близость к несогласию"): [("geo2_contact", "↓")],
    ("И источник", "О4 интрузии", "К9 близость к дайке"): [("dist_magm", "↓")],
    ("И источник", "О4 интрузии", "К10 насыщенность магматитами"): [("dens_magm", "↑")],
    ("Н место накопления", "О10 коры выветривания", "К19 близость к коре"): [("dist_struct", "↓")],
    ("Н место накопления", "О13 грубообломочные фации", "К25 благоприятная фация"): [("dist_facies", "↓"),
                                                                             ("geo2_facies2", "↓")],
    ("Н место накопления", "О14 впадины", "К26 палеодолины"): [("dist_paleo", "↓")],
}
VARIANT = "А"

#: Звено «С сохранность» (ключ --сохранность). Редакция 22.11.2026 по ответу
#: ВНИГНИ 19.11 (вопрос 2): золото сохраннее «там, где были понижения и
#: лакуны, в которые легли рифейские породы. рядом с разломами, которые были
#: резко и плотно перекрыты более молодыми отложениями»; вопрос 1: там, где
#: рифей полностью размыт, золота нет. Прежние К28а (уровень среза по шкале)
#: и К28б (близость к эрозионному краю рифея) ответу не отвечают и сняты
#: (решение руководителя 22.11). Источник — карта распределения свит
#: (`experiments/gmo_svity_section.py`, срез 500 м). С — фильтр площади:
#: в мягкое «и» не входит, итог не меняет, умножает уверенность.
#: * К28а запрет: рифей полностью размыт (фундамент на поверхности) — С = 0;
#: * К28г палеопонижение кровли фундамента (О14 на этапе III): чем ниже
#:   кровля региональной плоскости, тем лучше (ранг); наблюдается, где кровля
#:   не глубже 500 м; масштаб — десятки км, мелкие лакуны не видны;
#: * К28д разлом под сплошным покровом (О1): кромка в потенциальных полях
#:   (К4, тилт около нуля) там, где фундамент перекрыт рифеем; в варианте Б
#:   ещё ⚠ близость к разломным зонам формулы;
#: * два места сохранности — альтернативы, «или» (максимум) К28г и К28д.
#: Кембрий над рифеем сохранности не мешает (К29а «рифей сохранён»).
PRESERVE = False
#: v0.3 (ключ --v0-3, 22.11.2026) — v0.2 с сохранностью плюс то, что модель
#: может заполнить сама, без ответов ВНИГНИ:
#: * К14 магнитный дефицит в магнитоактивном блоке (О7, М/I–II; Г5): внутри
#:   блока, где региональное поле `pf_mag_up10` выше медианы листа, —
#:   остаток `pf_mag_up1` − `pf_mag_up10`, меньше — лучше; вне блока
#:   разрушать нечего — «не наблюдается»;
#: * К23 по разрезу: вмещающая свита из списка ВНИГНИ не глубже 500 м
#:   (`*_in500` карты свит; ответ ВНИГНИ 19.11 о глубине), а не только на
#:   поверхности: под чехлом, кембрием и юсмастахской свитой свита
#:   списка теперь видна. Где разрез не восстановлен — прежняя доля по
#:   поверхности `svk_list`.
#: * линеаменты с поправкой на фон (`experiments/gmo_lineaments_background.py`,
#:   способ Г: без отрезков вдоль контактов + ранг в страте «свита ×
#:   расчленённость»; принято руководителем 22.11.2026): К3 плотность —
#:   прошла все критерии; К1 узлы — с пометкой «промах на пороге η²
#:   0.055 при 0.05»; К2 близость и К5 простирание — тем же способом. Вне
#:   страт, где линии видны (половина листа), — «не наблюдается».
#:   К21 согласованность снят: не воспроизводится по половинам азимутов
#:   (0.30 при стоп-правиле 0.5) — О9 складки в варианте А не наблюдаются;
#: * О2 региональное несогласие своим каналом (раньше — только слой формулы
#:   ⚠): К7 — насколько несогласие (кровля фундамента) далеко от
#:   поверхности ячейки: под рифеем — глубина кровли по карте свит (не
#:   глубже 500 м), на выходах фундамента — расстояние до ближайшего рифея;
#:   К6 — сопряжение несогласия с разломной зоной (ранг К7 × ранг К2).
#:   Ог1: К6 несёт разломную зону, поэтому считается только там, где
#:   разломная зона отдана звену «путь».
V03 = False
#: v0.3a (ключ --v0-3a): v0.3 плюс радиусы критериев по мировой литературе
#: (`Параметры-критериев-по-литературе-черновик.md`, 22.11.2026, надёжность
#: низкая) и О2 кандидатом в место накопления (базальный горизонт на
#: несогласии; решение руководителя 22.11.2026). Радиус — полная мера до
#: первого числа, линейный спад до второго:
#: * К2 (линеаменты как малые разломы): 1 → 2 км по метрическому расстоянию,
#:   только в стратах, где линии видны (иначе «не наблюдается», Ог4);
#: * К4 и К28д (кромки потенциальных полей): 5 → 10 км от смены знака тилта;
#: * К7 (несогласие, в путях и в местах накопления): 1 → 2 км.
#: Правило «одна поверхность — одно звено» для О2, как Ог1 для разломной зоны.
V03A = False
#: v0.3b (ключ --v0-3b): форма спада из литературы (полная мера, затем
#: линейный спад до вдвое большего расстояния — как 4.5 → 9 км у К2), а
#: масштаб — свой для листа: полная мера в ближайших `Q_CORE` ячейках по
#: распределению канала, но не меньше ячейки (500 м). Литературные радиусы на ячейке 500 м насыщают
#: каналы (кромки магнитного поля — мера 1 в 100 % ячеек).
V03B = False
Q_CORE = 0.25
SPREAD = 2.0          # во сколько раз предел спада дальше полной меры
R_K2, R_K4, R_K7 = ("радиус", 1000.0, 2000.0), ("радиус", 5000.0, 10000.0), ("радиус", 1000.0, 2000.0)
UNC = "О2 региональное несогласие"
FOLDS = "О9 складки"
EDGE = "О1п кромки потенциальных полей"
#: v0.3c (ключ --v0-3c, решения руководителя 22.11.2026) — v0.3b плюс:
#: * К4 (кромки гравитационного и магнитного полей) вынесен из О1 в
#:   отдельный объект «О1п»: внутри О1 он усреднялся с близостью к
#:   линеаментам и не влиял на карту ни при каком радиусе. Это второе,
#:   независимое от рельефа наблюдение разломов (по полям, а не по
#:   линеаментам) — роли И и П; как и О1, закрывает одно звено;
#: * кровля фундамента — одна поверхность: несогласие (О2) и складки (О9)
#:   построены по ней (глубина и выпуклость) и вместе закрывают одно
#:   звено — «путь» или «место накопления». Палеопонижения (К28г) с той же
#:   поверхности остаются в звене «сохранность»: оно фильтр и в итог не
#:   входит.
V03C = False
#: v0.3d (ключ --v0-3d, 23.11.2026) — v0.3c плюс то, что модель заполнила сама:
#: * О8 палеорельеф, К20 (М/II): кровля фундамента из карты свит — это рельеф
#:   перед отложением рифея. Г6 доказала лишь, что современный рельеф его не
#:   повторяет. Денудация высвобождает золото на палеоподнятиях: канал —
#:   превышение кровли над региональной плоскостью (`basement_resid_m`, ↑),
#:   только где кровля не глубже 500 м. Та же поверхность, что у несогласия и
#:   складок, — входит в ту же группу Ог1а (одно звено на поверхность);
#: * О3 локальные несогласия, К8 (Н/III) — кандидат: несогласие выходов свит с
#:   согласным пологим залеганием (`discord_m`, ↑). Не отличает несогласие от
#:   разлома со смещением и от смены мощности (вопрос к ВНИГНИ).
#: Кварцевые жилы по индексам ASTER TIR не добавлены: `astir_qi/ci` проверены
#: 06.08.2026 (src/aster_tir.py) — повторяют растительность и уклон, геологии нет.
V03D = False
PALEO = "О8 палеорельеф"
LOCAL_UNC = "О3 локальные несогласия (кандидат)"
LBG = ROOT / "data/processed/lineament_bg.parquet"
SECTION_CH = ROOT / "data/processed/section_channels.parquet"
LBG_MAP = {"lin_dens": "lincxrank_dens", "lin_node_dens": "lincxrank_node_dens",
           "lin_dist": "lincxrank_dist", "lin_dir_sin": "lincb_dir_sin"}
SECTION_HOST = False
SVITY_GRID = ROOT / "data/processed/svity_section_grid.parquet"
MAX_DEPTH_M = 500.0


def load() -> pd.DataFrame:
    feats = sorted({f for objs in MODEL.values() for crits in objs.values()
                    for fl in crits.values() for f, _ in fl}
                   - {"nw_align", "mag_deficit", "unc_dist", "unc_fault", "fold_convex", "discord_m",
                      "basement_resid_m",
                      "lin_dist_m", "edge_gr_m", "edge_mag_m"})
    if V03A:
        feats = sorted(set(feats) | {"pf_gr_tdr", "pf_mag_tdr"})
    if V03:
        feats = sorted(set(feats) | {"pf_mag_up1", "pf_mag_up10"})
    df = pd.read_parquet(DATA, columns=["row", "col", "x", "y", spectral_mask.BARE_COL,
                                        "lin_dir_cos", "lin_dir_sin"] + feats)
    df = df.merge(pd.read_parquet(K23, columns=["row", "col", "svk_cover", "svk_list"]),
                  on=["row", "col"], how="left")
    lc = pd.read_parquet(LINC)
    df = df.merge(lc[["row", "col", "dist_thalweg"]], on=["row", "col"], how="left")
    if CLEAN:
        # Отрезки вдоль тальвегов убраны (experiments/gmo_lineaments_clean.py)
        lin = [c for c in df.columns if c.startswith("lin_")]
        df = df.drop(columns=lin).merge(
            lc[["row", "col"] + [c.replace("lin_", "linc_") for c in lin]]
            .rename(columns=lambda c: c.replace("linc_", "lin_")), on=["row", "col"], how="left")
    # Северо-запад. Угол линеамента в `src/lineaments.py` — от востока в кадре
    # со строкой 0 на севере, а не азимут: lin_dir_sin = sin 2θ = −sin(2·азимут).
    # Отрезок СЗ–ЮВ даёт lin_dir_sin = +1 (проверено синтетическим отрезком
    # 05.11.2026). В v0–v0.2 стояло −lin_dir_sin — К5 поощрял северо-восток.
    if V03:
        bg = pd.read_parquet(LBG, columns=["row", "col"] + list(LBG_MAP.values()))
        df = df.drop(columns=[c for c in LBG_MAP if c in df.columns]).merge(
            bg.rename(columns={v: k for k, v in LBG_MAP.items()}), on=["row", "col"], how="left")
        df = df.merge(unconformity(df[["row", "col"]]), on=["row", "col"], how="left")
        df["unc_fault"] = (membership(df.unc_dist, "↓") * membership(df.lin_dist, "↓"))
        df = df.merge(pd.read_parquet(SECTION_CH, columns=["row", "col", "fold_convex",
                                                           "discord_m"]),
                      on=["row", "col"], how="left")
        df = df.merge(pd.read_parquet(SVITY_GRID, columns=["row", "col", "basement_resid_m"]),
                      on=["row", "col"], how="left")
        if V03A:
            obs = pd.read_parquet(LBG, columns=["row", "col", "bg_observed_x"])
            dm = lc[["row", "col", "linc_dist"]].merge(obs, on=["row", "col"])
            dm["lin_dist_m"] = dm.linc_dist.where(dm.bg_observed_x)
            df = df.merge(dm[["row", "col", "lin_dist_m"]], on=["row", "col"], how="left")
            for c, out in (("pf_gr_tdr", "edge_gr_m"), ("pf_mag_tdr", "edge_mag_m")):
                df[out] = edge_distance(df, c)
            df["unc_fault"] = membership(df.unc_dist, R_K7) * membership(df.lin_dist_m, R_K2)
    df["nw_align"] = df.lin_dir_sin
    if V03:
        # К14: дефицит только внутри магнитоактивного блока (медиана — по листу)
        act = df.pf_mag_up10 > df.pf_mag_up10.median()
        df["mag_deficit"] = (df.pf_mag_up1 - df.pf_mag_up10).where(act)
    if SECTION_HOST:
        keys = ("il", "br", "lb", "ul", "kt")
        sg = pd.read_parquet(SVITY_GRID, columns=["row", "col"] + [f"{k}_in500" for k in keys])
        v = sg[[f"{k}_in500" for k in keys]]
        sg["svk_sec"] = np.where((v == 1).any(axis=1), 1.0, np.where((v == 0).all(axis=1), 0.0, np.nan))
        df = df.merge(sg[["row", "col", "svk_sec"]], on=["row", "col"], how="left")
        df["svk_list_surface"] = df.svk_list
        df["svk_list"] = df.svk_sec.fillna(df.svk_list)
    exposed = spectral_mask.exposure_mask(df)
    for c in df.columns:
        if c.startswith(SPECTRAL) and c != spectral_mask.BARE_COL:
            df.loc[~exposed, c] = np.nan
    return df[df.svk_cover.fillna(0) >= 0.5].reset_index(drop=True)


def edge_distance(df: pd.DataFrame, col: str) -> pd.Series:
    """Расстояние до кромки в поле — смены знака тилт-производной, м."""
    from scipy.ndimage import distance_transform_edt
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    a = np.full((nrow, ncol), np.nan)
    a[df.row, df.col] = df[col]
    s = np.sign(a)
    edge = np.zeros_like(a, bool)
    for dr, dc in ((0, 1), (1, 0)):
        x, y = s[:nrow - dr, :ncol - dc], s[dr:, dc:]
        hit = (x * y) < 0
        edge[:nrow - dr, :ncol - dc] |= hit
        edge[dr:, dc:] |= hit
    d = distance_transform_edt(~edge) * 500.0
    return pd.Series(d[df.row, df.col], index=df.index).where(df[col].notna())


def unconformity(cells: pd.DataFrame) -> pd.DataFrame:
    """К7 своим каналом: насколько региональное несогласие далеко от поверхности, м."""
    from scipy.ndimage import distance_transform_edt
    g = cells.merge(pd.read_parquet(SVITY_GRID, columns=["row", "col", "rif_thick",
                                                         "basement_depth", "reliab"]),
                    on=["row", "col"], how="left")
    ok = g.reliab == 1
    rif, base = ok & (g.rif_thick > 1), ok & (g.rif_thick <= 1)
    nrow, ncol = int(g.row.max()) + 1, int(g.col.max()) + 1
    grid = np.ones((nrow, ncol), bool)
    grid[g.row[rif], g.col[rif]] = False
    d = distance_transform_edt(grid) * 500.0
    out = pd.Series(np.nan, index=g.index)
    out[rif] = g.basement_depth[rif].where(g.basement_depth[rif] <= MAX_DEPTH_M)
    out[base] = d[g.row[base], g.col[base]]
    return pd.DataFrame({"row": g.row, "col": g.col, "unc_dist": out})


def membership(s: pd.Series, how) -> pd.Series:
    if isinstance(how, tuple) and how[0] == "квантиль":
        # не меньше ячейки: у магнитного поля смена знака тилта — больше чем в
        # четверти ячеек, и квантиль даёт 0
        core = max(float(s.quantile(how[1])), how[3])
        how = ("радиус", core, core * how[2])
    if isinstance(how, tuple):
        # («радиус», полная, предел): 1 до полной, линейный спад до 0 к пределу (м);
        # дальше предела — 0 («наблюдается, но далеко»), пропуск остаётся пропуском
        _, core, rmax = how
        return (1 - (s - core) / (rmax - core)).clip(0, 1).where(s.notna())
    if how == "доля":
        return s.clip(0, 1)
    if how == "◇":
        s = -s.abs()
    elif how == "↓":
        s = -s
    return s.rank(pct=True)


def link_scores(df: pd.DataFrame, skip_fault: set[str],
                skip_unc: frozenset = frozenset(),
                skip: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Оценки звеньев; `skip_fault` / `skip_unc` — звенья, где разломная зона /
    несогласие не считается (одна поверхность — одно звено); `skip` —
    объект → звенья, где он не считается (общий случай для v0.3c)."""
    skip = dict(skip or {})
    skip.setdefault(FAULT, set()).update(skip_fault)
    skip.setdefault(UNC, set()).update(skip_unc)
    scores, parts = {}, {}
    for link, objs in MODEL.items():
        obj_vals = {}
        for obj, crits in objs.items():
            if link in skip.get(obj, ()):
                continue
            # Ог1: К6 несёт разломную зону — только там, где она отдана этому звену
            crit_vals = [pd.concat([membership(df[f], h) for f, h in fl], axis=1).mean(axis=1)
                         for c, fl in crits.items()
                         if not (c.startswith("К6") and link in skip[FAULT])]
            obj_vals[obj] = pd.concat(crit_vals, axis=1).mean(axis=1)
        if obj_vals:
            t = pd.DataFrame(obj_vals)
            scores[link] = t.max(axis=1)
            if link in HOST:
                scores[link] = np.fmin(scores[link], df[HOST[link][1]].clip(0, 1))
                if V03:
                    # np.fmin отбрасывает пропуск: без наблюдённого места звено
                    # получало оценку одной вмещающей свиты. Ог4: не наблюдается.
                    scores[link] = scores[link].where(t.notna().any(axis=1))
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


def preservation(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Звено С: оценка, ведущий объект и множитель уверенности.

    Множитель определён и там, где С не наблюдается (разрез не восстановлен,
    кровля фундамента глубже среза и нет полей): берётся у ближайшей ячейки,
    где С посчитано, — иначе закрытые ячейки выходили бы увереннее соседей.
    """
    from scipy.ndimage import distance_transform_edt
    g = df[["row", "col"]].merge(
        pd.read_parquet(SVITY_GRID, columns=["row", "col", "rif_thick", "basement_depth",
                                             "basement_resid_m", "reliab"]),
        on=["row", "col"], how="left")
    g.index = df.index
    ok = g.reliab == 1
    rif = ok & (g.rif_thick > 1)                 # рифей сохранился, в т.ч. под чехлом
    gone = ok & (g.rif_thick <= 1)               # К28а: рифей размыт — запрет
    k28g = (-g.basement_resid_m).where(rif & (g.basement_depth <= MAX_DEPTH_M)).rank(pct=True)
    edge = ([membership(df[c], R_K4) for c in ("edge_gr_m", "edge_mag_m")] if V03A
            else [membership(df[c], "◇") for c in ("pf_gr_tdr", "pf_mag_tdr")])
    if VARIANT == "Б":
        edge += [membership(df[c], "↓") for c in ("dist_tect1", "dist_tect2")]
    k28d = pd.concat(edge, axis=1).mean(axis=1).where(rif)
    s = pd.concat([k28g, k28d], axis=1).max(axis=1)
    s[gone] = 0.0
    lead = pd.Series(None, index=df.index, dtype=object)
    lead[k28g.notna() & (k28g >= k28d.fillna(-1))] = "О14 палеопонижение кровли фундамента"
    lead[k28d.notna() & (k28d > k28g.fillna(-1))] = "О1 разлом под покровом"
    lead[gone] = "запрет: рифей размыт"
    nrow, ncol = int(df.row.max()) + 1, int(df.col.max()) + 1
    known = np.zeros((nrow, ncol), bool)
    known[df.row[s.notna()], df.col[s.notna()]] = True
    filled = np.zeros((nrow, ncol))
    filled[df.row, df.col] = s.fillna(0.0)
    ir, ic = distance_transform_edt(~known, return_distances=False, return_indices=True)
    mult = pd.Series(filled[ir, ic][df.row, df.col], index=df.index)
    print(f"   С: запрет (рифей размыт) {gone.mean():.1%}; К28г наблюдается {k28g.notna().mean():.0%}, "
          f"К28д {k28d.notna().mean():.0%}; ведущий: {lead.value_counts().to_dict()}")
    return s, lead, mult


def build(df: pd.DataFrame) -> pd.DataFrame:
    import itertools
    # Группы «один объект (одна поверхность) — одно звено»: Ог1 для разломной
    # зоны; в v0.3a — несогласие; в v0.3c — кромки полей отдельно и кровля
    # фундамента (несогласие + складки) как одна поверхность.
    groups = {"разлом": {FAULT}, "несогласие": {UNC}}
    if V03C:
        groups = {"разлом": {FAULT}, "кромки": {EDGE}, "поверхность": {UNC, FOLDS, PALEO}}
    opts = {}
    for gname, objs in groups.items():
        links = [l for l, o in MODEL.items() if objs & set(o)]
        opts[gname] = (links, links if (gname == "разлом" or len(links) > 1) else [None])
    best = None
    for combo in itertools.product(*[o[1] or [None] for o in opts.values()]):
        skip = {}
        for (gname, (links, _)), keep in zip(opts.items(), combo):
            if keep is None:
                continue
            for obj in groups[gname]:
                skip.setdefault(obj, set()).update(set(links) - {keep})
        sc, parts = link_scores(df, skip_fault=skip.get(FAULT, set()), skip=skip)
        tot, conf = soft_and(sc)
        names = dict(zip(opts, combo))
        cand = sc.add_prefix("звено ").assign(итог=tot, уверенность=conf,
                                              разлом_закрыл=names.get("разлом"),
                                              несогласие_закрыло=names.get("несогласие",
                                                                           names.get("поверхность")),
                                              кромки_закрыли=names.get("кромки"))
        for l in MODEL:
            cand[f"объект {l}"] = parts[l]
        if best is None:
            best = cand
        else:
            if V03:
                # Ог1 выбирает звено для разломной зоны сначала по числу
                # наблюдённых звеньев, потом по итогу: итог — среднее по
                # наблюдённым, и иначе выгодно оставить звено пустым (в v0.3
                # без складок место накопления держится только на узлах).
                more = cand.уверенность > best.уверенность + 1e-9
                same = (cand.уверенность - best.уверенность).abs() <= 1e-9
                better = more | (same & (cand.итог > best.итог))
            else:
                better = cand.итог > best.итог
            best[better] = cand[better]
    if PRESERVE:
        s, lead, mult = preservation(df)
        best["звено С сохранность"], best["объект С сохранность"] = s, lead
        links = [f"звено {l}" for l in MODEL]
        observed = best[links].notna().sum(axis=1) / len(links)
        best["уверенность"] = observed * mult
    return pd.concat([df[["row", "col", "x", "y", "dist_thalweg"]], best], axis=1)


def verify(m: pd.DataFrame) -> None:
    """Только заверка: с критериальной картой и с золотом. Не мишень."""
    from scipy.stats import spearmanr as sp
    print("\nПовторяет ли карта речную сеть: ρ с расстоянием до тальвега — "
          + ", ".join(f"{c.replace('звено ', '')} {sp(m[c], m.dist_thalweg, nan_policy='omit').statistic:+.2f}"
                      for c in m.columns if c.startswith("звено ") and m[c].notna().any())
          + f", итог {sp(m.итог, m.dist_thalweg, nan_policy='omit').statistic:+.2f}")
    if VARIANT == "Б":
        print("\nВариант Б с картой ВНИГНИ не сверяется (Ог2): слои формулы — её же входы.")
        return
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
    panels = ([f"звено {l}" for l in MODEL if PRESERVE or l != "С сохранность"]
              + ["итог", "уверенность"])
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.5)) if PRESERVE else plt.subplots(2, 3, figsize=(15, 9.5))
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    for ax, col in zip(axs.ravel(), panels):
        a = np.full((nrow, ncol), np.nan)
        a[m.row, m.col] = m[col]
        im = ax.imshow(a, cmap="viridis" if col != "уверенность" else "Greys",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(col.replace("звено ", "звено: "), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75, label="доля 0–1")
    ver = TAG.lstrip("_").replace("_", ".")
    kind = "А (свои каналы)" if VARIANT == "А" else "Б (свои каналы и слои формулы ⚠)"
    fig.suptitle(f"Пробная карта на знаниях {ver}, вариант {kind}, лист R-48-XI,XII, "
                 "сетка 500 м. " + ("Сохранность — фильтр: умножает уверенность, итог не меняет"
                                   if PRESERVE else "Звено «сохранность» не наблюдается"), fontsize=11)
    fig.tight_layout()
    out = outputs.png(f"gmo_knowledge_map{TAG}.png")
    mapstyle.save_fig(fig, out, dpi=130)
    print(f"\nКарта: {out}")


def configure(argv: list[str]) -> None:
    """Версия карты по ключам: флаги и правки MODEL (без расчёта)."""
    # Ключ --без-долин: v0.1 без dist_dnara. В v0 он вёл звено «место
    # накопления», и карта повторяла современную речную сеть (Г6: палеорельеф
    # не наблюдается; пробы по рыхлым отбирают в долинах — совпадение с ними
    # ложное).
    global TAG, CLEAN, VARIANT, PRESERVE, V03, SECTION_HOST, V03A, V03B, V03C, V03D, R_K2, R_K4, R_K7
    argv = list(argv)
    if "--v0-3d" in argv:
        V03D = True
        argv.append("--v0-3c")
    if "--v0-3c" in argv:
        V03C = True
        argv.append("--v0-3b")
    PRESERVE = "--сохранность" in argv
    if "--v0-3b" in argv:
        V03B = True
        argv.append("--v0-3a")
    if "--v0-3a" in argv:
        V03A = True
        argv.append("--v0-3")
    if "--v0-3" in argv:
        V03 = SECTION_HOST = PRESERVE = True
        argv.append("--чистые-линеаменты")
        MODEL["М мобилизация"]["О7 реактивация и выщелачивание"][
            "К14 магнитный дефицит в блоке"] = [("mag_deficit", "↓")]
        MODEL["П путь"]["О2 региональное несогласие"] = {
            "К7 близость к несогласию": [("unc_dist", "↓")],
            "К6 сопряжение с разломной зоной": [("unc_fault", "доля")]}
        # К21 по линеаментам снят (не воспроизводится); складки — по форме
        # кровли фундамента из карты свит (experiments/gmo_section_channels.py):
        # выпуклость вверх — антиформа. Направление «замок лучше крыла» —
        # предварительное, ждёт литературы и ответа ВНИГНИ.
        MODEL["Н место накопления"]["О9 складки"] = {
            "К21 антиформа кровли фундамента": [("fold_convex", "↑")]}
    b = "--вариант-б" in argv
    if "--без-долин" in argv or "--чистые-линеаменты" in argv or b:
        del MODEL["Н место накопления"]["О14 впадины"]
        TAG = "_v0_1"
    if "--чистые-линеаменты" in argv or b:
        # v0.2: v0.1 + линеаменты без отрезков вдоль долин
        CLEAN, TAG = True, "_v0_2"
    if b:
        # Вариант Б поверх v0.2; О14 возвращается только с палеодолинами формулы.
        # Слои формулы добавляются в конце — после радиусов v0.3a, иначе те
        # затёрли бы ⚠-признаки в К2 и К7.
        VARIANT, TAG = "Б", "_v0_2_b"
    if V03B:
        # нижний предел — ячейка для горизонтальных расстояний; у К7 канал под
        # рифеем — глубина кровли (до 500 м), предел в ячейку его насыщал бы
        R_K2 = R_K4 = ("квантиль", Q_CORE, SPREAD, 500.0)
        R_K7 = ("квантиль", Q_CORE, SPREAD, 0.0)
    if V03A:
        f = MODEL["И источник"][FAULT]
        f["К2 близость к разломной зоне"] = [("lin_dist_m", R_K2)]
        f["К4 разломные границы в полях"] = [("edge_gr_m", R_K4), ("edge_mag_m", R_K4),
                                             ("gm_gr_1G_25", "↑")]
        MODEL["П путь"][UNC]["К7 близость к несогласию"] = [("unc_dist", R_K7)]
        MODEL["Н место накопления"][UNC] = {
            "К7 базальный горизонт на несогласии": [("unc_dist", R_K7)]}
    if V03D:
        MODEL["М мобилизация"][PALEO] = {"К20 палеоподнятие кровли фундамента":
                                         [("basement_resid_m", "↑")]}
        MODEL["Н место накопления"][LOCAL_UNC] = {"К8 несогласное залегание выходов свит":
                                                  [("discord_m", "↑")]}
    if V03C:
        k4 = MODEL["И источник"][FAULT].pop("К4 разломные границы в полях")
        for link in ("И источник", "П путь"):
            MODEL[link][EDGE] = {"К4 разломные границы в полях": list(k4)}
    if b:
        for (link, obj, crit), fl in FORMULA.items():
            MODEL[link].setdefault(obj, {}).setdefault(crit, []).extend(fl)
    if V03A:
        TAG = ("_v0_3d" if V03D else "_v0_3c" if V03C else "_v0_3b" if V03B else "_v0_3a") + ("_b" if b else "")
    elif V03:
        TAG = "_v0_3"
    elif PRESERVE:
        TAG += "_s"


def main() -> None:
    configure(sys.argv[1:])
    df = load()
    m = build(df)
    m.to_parquet(OUT.with_name(f"knowledge_map{TAG}.parquet"), index=False)
    print(f"Ячеек {len(m)}")
    for l in MODEL:
        s = m[f"звено {l}"]
        print(f"   звено {l}: наблюдается в {s.notna().mean():.0%} ячеек")
    print(f"   уверенность: {m.уверенность.round(1).value_counts().sort_index().to_dict()}")
    print("   разломная зона закрыла:", m.разлом_закрыл.value_counts().to_dict())
    if m.несогласие_закрыло.notna().any():
        print("   несогласие (поверхность) закрыло:", m.несогласие_закрыло.value_counts().to_dict())
    if m.кромки_закрыли.notna().any():
        print("   кромки полей закрыли:", m.кромки_закрыли.value_counts().to_dict())
    print("   ведущий объект места накопления:",
          m["объект Н место накопления"].value_counts().to_dict())
    explain(m)
    verify(m)
    draw(m)


if __name__ == "__main__":
    main()
