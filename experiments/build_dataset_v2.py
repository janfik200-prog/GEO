"""Этапы 3 и 9: сборка датасета из групп признаков, посаженных на общую сетку.

Один сборщик на две версии (``DATASETS``), потому что механика склейки и весь
контроль качества у них общие, а различаются они только списком источников:

* ``v2`` (этап 3) — рельеф, линеаменты, Sentinel-2;
* ``v3`` (этап 9) — то же плюс четыре съёмки этапа 8: Sentinel-1, PALSAR-2,
  ASTER, Landsat 8/9.

Запуск: ``python -m experiments.build_dataset_v2 v3``.


Сетка не меняется: та же ``prognoz.pgrid`` (154x149, 500 м), тот же C-порядок
ячеек. Поэтому «сборка» здесь — не пересчёт координат, а склейка таблиц,
посаженных на сетку каждым модулем-источником, и КОНТРОЛЬ того, что склейка
корректна.

Группы (реестр — ``config.V2_FEATURE_GROUPS``):

* ``gm``, ``ls``, ``dist``, ``relief_v1`` — из ``dataset_v1.parquet``;
* ``ter`` — ``terrain_v2.parquet`` (этап 2d);
* ``lin`` — ``lineament_features.parquet`` (этап 2c);
* ``s2`` — ``s2_features.parquet`` (этап 2a).

Условные группы собираются gracefully: если файла нет (ветка закрыта
стоп-правилом или ещё не досчитана), группа пропускается с записью в отчёт, а
не роняет сборку. Так же будет с аэрогаммой, если заказчик её пришлёт.

Контроль качества (всё пишется в ``dataset_v2_sources.md``):

* совпадение длины таблиц и порядка ячеек;
* доля NaN по каждой группе, отдельно по валидным ячейкам;
* число обусловленности корреляционной матрицы группы и всего пула —
  ловушка на алгебраические дубли (так были пойманы ``gm_gr_all`` в v1 и
  ``dem_curv`` в v2);
* самые связанные пары признаков ИЗ РАЗНЫХ групп — ловушка на скрытое
  дублирование между источниками (линеаменты против рельефа).

Выход: ``data/processed/dataset_v2.parquet``, ``dataset_v2_sources.md``,
``outputs/dataset_v2_preview.png``.
Запуск из корня: ``python -m experiments.build_dataset_v2``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import cell_mask, config, features_v2, integro_grid  # noqa: E402

PARTS = {
    "ter": ("terrain_v2.parquet", "src/terrain_v2.py, Copernicus DEM GLO-30"),
    "lin": ("lineament_features.parquet", "src/lineaments.py, отмывка того же DEM"),
    "s2": ("s2_features.parquet", "src/s2_composite.py, Sentinel-2 L2A (COG на AWS)"),
}

#: Датасет v3 (этап 9) = v2 плюс четыре съёмки этапа 8. Отдельный набор, а не
#: правка PARTS: ``dataset_v2.parquet`` — зафиксированный вход этапов 4 и 4b, и
#: если его пересобрать, их результаты перестанут воспроизводиться.
#: ВНИМАНИЕ: «датасет v3» и «пул v3» — разные вещи. Пул v3 = пул v2 + 8
#: геологических факторных слоёв (``config.V3_GEO_FACTORS``), он про состав
#: признаков; датасет v3 — про состав источников.
PARTS_V3 = {
    **PARTS,
    "s1": ("s1_features.parquet", "src/sat_sources.py, Sentinel-1 RTC (Planetary Computer)"),
    "psr": ("psr_features.parquet", "src/sat_sources.py, ALOS PALSAR-2, годовые мозаики"),
    "ast": ("ast_features.parquet", "src/sat_sources.py, ASTER L1T, мозаика 2001/2006"),
    "l8": ("l8_features.parquet", "src/sat_sources.py, Landsat 8/9 Collection 2 L2"),
}

#: Датасет v4 = v3 плюс тепловой диапазон ASTER. Снова отдельный набор, а не
#: правка PARTS_V3, по той же причине: на ``dataset_v3.parquet`` посчитан этап 9
#: с его абляциями, и пересборка сделала бы их невоспроизводимыми.
PARTS_V4 = {
    **PARTS_V3,
    "astir": ("astir_features.parquet", "src/aster_tir.py, ASTER TIR, архив LP DAAC"),
}

#: Датасет v5 = v3 плюс трансформанты потенциальных полей (этап «можно
#: посчитать сейчас», 06.08.2026). Собирается ОТ v3, не от v4: отрицательный
#: результат ASTER TIR (см. PARTS_V4) в пул не идёт.
PARTS_V5 = {
    **PARTS_V3,
    "pf": ("potfield_features.parquet",
           "src/potential_fields.py, БПФ на грав_маг.pgrid (без новых скачиваний)"),
    "opt": ("optical_derived_features.parquet",
            "src/optical_derived.py, индексы из уже собранных s2_b*/ast_b* "
            "(без новых скачиваний)"),
    "geo2": ("geology_v2_features.parquet",
             "src/geology_v2.py, те же шейпы fasii/svita_new/glub_raz_nw/glub_r_nw, "
             "что и в группе geo (без новых скачиваний)"),
    "relief2": ("relief_extra_features.parquet",
                "src/relief_v2_extra.py, тот же Copernicus DEM, что и в группе ter "
                "(кэш datacache/anabar_dem, без новых скачиваний)"),
}

DATASETS = {"v2": PARTS, "v3": PARTS_V3, "v4": PARTS_V4, "v5": PARTS_V5}

#: Паспорт групп: что это, откуда исходник, как посажено на сетку. Пишется в
#: отчёт, чтобы источник каждого признака читался без чтения кода модулей.
PASSPORT = {
    "gm": ("Гравиметрия и магнитометрия, 17 трансформант",
           "Комплект заказчика: `data/SBORKA_DOP/ГРАВИКА_МАГНИТКА/грав_маг.pgrid`, "
           "17 гридов 500 м",
           "билинейная интерполяция; поля направлений (`gr_1GFI_25`, "
           "`mag_1GFI_25`) — через sin/cos, иначе 179° и −179° оказались бы "
           "противоположными"),
    "ls": ("Мультиспектральный снимок, 7 каналов",
           "Комплект заказчика: `data/SBORKA_DOP/КОСМОСНИМОК/landsat_fragm.pgrid`, "
           "30 м",
           "агрегация `average` до 500 м; DN=0 трактуется как NoData (фон "
           "повёрнутой сцены), редкий тёмный пиксель при этом теряется"),
    "geo": ("Геологические факторные слои: расстояния и плотности",
            "Комплект заказчика: `data/Gis-integro/shp_dbf/*.shp` — "
            "`glub_raz_nw` (разломы 1), `glub_r_nw` (разломы 2), `dayki_buf` "
            "(дайки), `kory` (коры выветривания), `fasii` (фации), "
            "`gr_dol_vp_poly` (палеодолины)",
            "растры расстояния до объекта и плотности (суммарная длина для "
            "разломов, площадь для даек) в радиусе `config.DENSITY_RADIUS` = "
            "2500 м"),
    "dist": ("Расстояние до гидросети",
             "Комплект заказчика: `data/SBORKA_DOP/ТОПО/dnl.shp`, `dnara.shp`",
             "растр расстояния; отдельная группа от `geo`, потому что "
             "гидросеть — не рудоконтролирующий фактор, а разметка смещения "
             "точек опробования"),
    "relief_v1": ("Рельеф из комплекта заказчика",
                  "`data/SBORKA_DOP/РЕЛЬЕФ/topo5_new.pgrid`, 100 м",
                  "`average` до 500 м, пересчёт в метры (×0.2 — единицы "
                  "источника метры ×5); дублирует `dem_elev` (Spearman +1.00), "
                  "оставлен как исторический контроль"),
    "ter": ("Производные рельефа: высота, уклон, TPI, врез, локальный размах",
            "Copernicus DEM GLO-30, 30 м, тайлы 1°×1° "
            "(`https://copernicus-dem-30m.s3.amazonaws.com`, анонимно). SRTM "
            "непригоден: предел 60° с.ш., лист на 71°",
            "DEM перепроецируется в метрическую систему сетки с шагом "
            "`TER_RES_M` = 100 м, производные считаются на изотропной сетке, "
            "затем агрегация на 500 м средним и std"),
    "lin": ("Линеаменты: плотность, плотность узлов, расстояние, анизотропия",
            "тот же Copernicus DEM GLO-30 — не оптика: под сплошной тундрой "
            "98 % пикселей заняты растительностью, оптический «линеамент» чаще "
            "всего граница растительного сообщества",
            "отмывка по 8 азимутам (высота солнца 30°) → Canny (σ=2) → "
            "преобразование Хафа (порог 8, длина ≥10 px, разрыв ≤3 px); "
            "расхождение чётных и нечётных азимутов — стоп-правило ветки"),
    "s2": ("9 каналов + 5 отношений (mean и std) + качество",
           "STAC `sentinel-2-l2a`, `https://earth-search.aws.element84.com/v1`",
           "2019–2025, июль–август, облачность <40 %, до 30 сцен на тайл, "
           "≥3 наблюдения на пиксель, маска SCL (оставлены классы 4, 5, 7), "
           "медианный композит 100 м → 500 м (mean+std). Вегетационная маска "
           "выключена: при NDVI>0.35 валидных ячеек не оставалось"),
    "s2raw": ("Подгруппа: только сырые каналы Sentinel-2, без отношений",
              "тот же композит, что и `s2`",
              "служит для абляций — проверки, не создают ли отношения каналов "
              "мнимого прироста"),
    "l8": ("6 каналов + температура поверхности + 4 отношения (mean и std)",
           "STAC `landsat-c2-l2` (Collection 2 Level-2), Planetary Computer",
           "2013–2025, июль–август, облачность <40 %, до 15 сцен на path, "
           "отбраковка по битам QA 1/3/4/5, масштабы SR (2.75e-05, −0.2) и "
           "ST (0.00341802, 149.0)"),
    "s1": ("Радар C-диапазона: VV, VH, их отношение (mean и std)",
           "STAC `sentinel-1-rtc` (радиометрически-террейн-корректированный), "
           "Planetary Computer",
           "2018–2025, июнь–сентябрь, до 12 сцен на орбиту, порог шума −35 дБ"),
    "psr": ("Радар L-диапазона: HH, HV, отношение, угол падения (mean и std)",
            "STAC `alos-palsar-mosaic` — готовые годовые мозаики, "
            "Planetary Computer",
            "2015–2021, поляризации HH/HV, калибровочное смещение −83.0 дБ"),
    "ast": ("9 каналов VNIR+SWIR и 6 минеральных индексов (mean и std)",
            "STAC `aster-l1t`, Planetary Computer",
            "2000–2008, июнь–сентябрь, облачность <60 %, до 12 сцен, ≥1 "
            "наблюдение. Только архив: SWIR-детектор отказал в апреле 2008, "
            "отсюда самая высокая доля пропусков среди всех групп"),
    "astir": ("5 тепловых каналов ASTER и 3 индекса Ниномии (mean и std)",
              "LP DAAC, продукт `AST_L1T.004` (поканальные GeoTIFF), доступ "
              "через Earthdata Login; каталог Planetary Computer для TIR не "
              "годится — он обрывается на 31.12.2006",
              "2000–2026, июнь–сентябрь, только дневные сцены, облачность "
              "<20 %, квота 8 сцен на полуградусную клетку, ≥3 наблюдения. "
              "DN → радианс по канальным коэффициентам, нормировка на 300 К "
              "по каналу 13 (иначе индекс читает прогрев склона), индексы "
              "считаются посценно и лишь затем усредняются медианой"),
    "pf": ("Вертикальная производная, tilt-derivative, аналитический сигнал, "
           "продолжение вверх (1/2/5/10 км), приведение магнитки к полюсу, "
           "скользящая корреляция гравика-магнитка — 16 признаков",
           "Комплект заказчика: `data/SBORKA_DOP/ГРАВИКА_МАГНИТКА/грав_маг.pgrid` "
           "(те же 2 поля `gr_all`/`mg_all`, что и в группе `gm`)",
           "БПФ на нативной сетке 305x455 (шире листа — запас против краевого "
           "разрыва периода), окно Тьюки 10% на затухание к краю, инклинация/"
           "склонение для RTP — IGRF (`ppigrf`) в центре листа, эпоха "
           "2020-07-01 (точная дата съёмки не указана)"),
    "opt": ("NDRE, IRECI (красный край Sentinel-2), Mg-OH индекс ASTER — 3 признака",
            "те же колонки `s2_b04..b8a` и `ast_b08/b09`, что уже в `dataset_v3.parquet`",
            "отношение СРЕДНИХ по ячейке (не среднее отношений — поканальной "
            "ковариации внутри ячейки в датасете нет, поэтому std не считается); "
            "`ast_mgoh` заметно коррелирует с уже посчитанным `ast_carb` "
            "(Spearman -0.77) — оговорка о частичном дублировании, не постулат"),
    "geo2": ("Раздельные расстояния по фациям (CODE_F 1/2), расстояние до "
             "контакта чехла с фундаментом + осевое простирание контакта, "
             "расстояние до узлов пересечения разломных систем — 6 признаков",
             "Комплект заказчика: `data/Gis-integro/shp_dbf/fasii.shp`, "
             "`svita_new.shp`, `glub_raz_nw.shp`, `glub_r_nw.shp` (те же файлы, "
             "что и в группе `geo`)",
             "`fasii` разбит по `CODE_F` на два дистанционных растра вместо "
             "одного общего `dist_facies`; контакт — внешняя граница "
             "объединения `svita_new` (внутренние швы между свитами уходят "
             "при union), простирание — осевая пара `sin(2*az)`/`cos(2*az)` "
             "касательной ближайшего сегмента границы; узлы разломов — "
             "непустые полигон-полигон пересечения `glub_raz_nw` x `glub_r_nw` "
             "(оба слоя — буферизованные разломные коридоры, не линии)"),
    "relief2": ("Кривизна на масштабе 5 км, внутриячеечный разброс TRI, "
                "log1p площади водосбора — 3 признака",
                "тот же Copernicus DEM GLO-30, что и в группе `ter` (кэш "
                "`datacache/anabar_dem`, повторного скачивания нет)",
                "кривизна — лапласиан DEM 100 м, сглаженного гауссианом sigma "
                "~2.5 км (масштаб намеренно вне 500 м/2 км, занятых `dem_tpi_500`/"
                "`dem_tpi_2km`); TRI (Riley et al., 1999) считался и на среднее, "
                "и на std по ячейке — среднее оказалось дублем `dem_slope` "
                "(Spearman 0.999) и в пул не вошло, в пул идёт только std; "
                "водосбор — D8 по заполненному рельефу (`src/catchments.py`) на "
                "самой целевой сетке `dem_elev`, `log1p` числа ячеек водосбора"),
}

#: Космические съёмки: какой аппарат снимал. Отдельная таблица, потому что по
#: имени группы это не читается, а для интерпретации признака важно, что именно
#: измеряет сенсор — отражённый свет, собственное излучение или обратное
#: рассеяние радиоволны.
SATELLITES = {
    "ls": ("Landsat 7", "ETM+ (Enhanced Thematic Mapper Plus)", "NASA / USGS",
           "оптика VNIR+SWIR, 30 м", "фрагмент из комплекта заказчика, "
           "дата съёмки в комплекте не указана"),
    "s2": ("Sentinel-2A / 2B (с 2024 также 2C)", "MSI (MultiSpectral Instrument)",
           "ESA, программа Copernicus", "оптика VNIR+SWIR, 10–20 м",
           "композит по 2019–2025, июль–август"),
    "l8": ("Landsat 8 и Landsat 9", "OLI/TIRS и OLI-2/TIRS-2", "NASA / USGS",
           "оптика 30 м, тепловой канал 100 м (даёт `l8_lst`)",
           "композит по 2013–2025, июль–август"),
    "s1": ("Sentinel-1A / 1B / 1C", "C-SAR, радар C-диапазона (5.4 ГГц)",
           "ESA, программа Copernicus", "RTC-продукт 10 м, поляризации VV/VH",
           "композит по 2018–2025, июнь–сентябрь"),
    "psr": ("ALOS-2", "PALSAR-2, радар L-диапазона (1.2 ГГц)", "JAXA (Япония)",
            "годовые мозаики 25 м, поляризации HH/HV",
            "мозаики 2015–2021; длинная волна проникает под растительный "
            "покров глубже, чем C-диапазон"),
    "ast": ("Terra", "ASTER", "прибор METI/JAXA на аппарате NASA",
            "VNIR 15 м, SWIR 30 м, TIR 90 м",
            "архив 2000–04.2008: SWIR-детектор вышел из строя, новых сцен нет"),
    "astir": ("Terra", "ASTER, подсистема TIR", "прибор METI/JAXA на аппарате NASA",
              "5 каналов 8.1–11.6 мкм, 90 м",
              "весь архив 2000–2026: отказ 2008 года коснулся только SWIR, "
              "тепловые детекторы работают. Единственный источник, где виден "
              "кварц: его спектральная подпись — полосы остаточных лучей "
              "на 8–9.5 мкм, в видимом диапазоне и SWIR кварц прозрачен"),
    "ter": ("TerraSAR-X + TanDEM-X (через продукт Copernicus DEM GLO-30)",
            "радарная интерферометрия X-диапазона", "DLR (Германия) / ESA",
            "ЦМР 30 м", "съёмка 2011–2015; это не снимок, а высота, "
            "восстановленная по паре радарных изображений"),
    "lin": ("то же, что `ter`", "—", "—", "—",
            "линеаменты считаются по отмывке той же ЦМР"),
}

NOT_SATELLITE = {
    "gm": "наземная и аэрогеофизическая съёмка (гравика, магнитка)",
    "geo": "векторная геологическая карта (оцифрованные слои)",
    "dist": "топооснова: гидросеть",
    "relief_v1": "топооснова: рельеф из комплекта заказчика",
    "pf": "производные той же наземной/аэрогеофизической съёмки, что и `gm`",
    "opt": "индексы поверх уже учтённых съёмок `s2`/`ast` — не отдельный источник",
    "geo2": "векторная геологическая карта (те же оцифрованные слои, что и `geo`)",
    "relief2": "топооснова: та же ЦМР, что и `ter`",
}


def group_of(col: str) -> str | None:
    return features_v2.feature_group(col)


def cond_number(tbl: pd.DataFrame, valid: np.ndarray) -> float:
    M = tbl.to_numpy(dtype=float)[valid]
    M = M[np.isfinite(M).all(axis=1)]
    if M.shape[0] < M.shape[1] + 2 or M.shape[1] < 2:
        return float("nan")
    sd = M.std(axis=0)
    M = M[:, sd > 0]
    if M.shape[1] < 2:
        return float("nan")
    return float(np.linalg.cond(np.corrcoef(M, rowvar=False)))


def main(version: str = "v2") -> None:
    parts = DATASETS[version]
    report: list[str] = []
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    n_cells = meta.prf * meta.pic
    print(f"Сетка: {meta.prf}x{meta.pic} = {n_cells} ячеек, шаг {meta.dx} м")

    report.append(f"# Датасет {version}: источники и контроль качества")
    report.append("")
    report.append(f"Сетка: `{config.GOLD_TARGET_PGRID.name}` {meta.prf}x{meta.pic}, "
                  f"шаг {meta.dx} м, CRS `{proj4}`")
    report.append("")
    report.append("Все группы посажены на ЭТУ ЖЕ сетку своими модулями, поэтому "
                  "пересчёта координат при сборке нет — контролируется только "
                  "совпадение длины и порядка ячеек.")
    report.append("")
    report.append("## Группы")

    out = df.copy()
    report.append("")
    report.append("| группа | источник | признаков | статус |")
    report.append("|---|---|---|---|")
    for name, prefixes in config.V2_FEATURE_GROUPS.items():
        if name in parts:
            continue
        cols = [c for c in df.columns if group_of(c) == name]
        if cols:
            report.append(f"| `{name}` | dataset_v1.parquet | {len(cols)} | из v1 |")

    for name, (fname, src) in parts.items():
        path = config.PROCESSED_DIR / fname
        if not path.exists():
            print(f"ГРУППА {name}: файла {fname} нет — пропущена")
            report.append(f"| `{name}` | {src} | 0 | **ПРОПУЩЕНА: нет {fname}** |")
            continue
        part = pd.read_parquet(path)
        assert len(part) == n_cells, f"{fname}: {len(part)} строк вместо {n_cells}"
        dup = [c for c in part.columns if c in out.columns]
        assert not dup, f"{fname}: колонки уже есть в датасете: {dup}"
        out = pd.concat([out, part], axis=1)
        print(f"группа {name}: +{len(part.columns)} признаков из {fname}")
        report.append(f"| `{name}` | {src} | {len(part.columns)} | добавлена |")

    present = sorted({g for c in out.columns if (g := group_of(c))})

    report.append("")
    report.append("## Паспорт групп: что это, откуда и как посажено на сетку")
    report.append("")
    report.append("| группа | что это | исходный источник | обработка |")
    report.append("|---|---|---|---|")
    for g in present:
        if g in PASSPORT:
            what, src, how = PASSPORT[g]
            report.append(f"| `{g}` | {what} | {src} | {how} |")

    report.append("")
    report.append("## Космические съёмки: какой аппарат снимал")
    report.append("")
    report.append("| группа | аппарат | сенсор | оператор | что и с каким "
                  "разрешением | период съёмки |")
    report.append("|---|---|---|---|---|---|")
    for g in present:
        if g in SATELLITES:
            report.append("| `" + g + "` | " + " | ".join(SATELLITES[g]) + " |")
    report.append("")
    report.append("Не космические съёмки: " + "; ".join(
        f"`{g}` — {txt}" for g, txt in NOT_SATELLITE.items() if g in present) + ".")

    valid = np.asarray(cell_mask.build_valid_mask(out))
    print(f"валидных ячеек: {int(valid.sum())} из {len(out)}")

    # --- NaN и обусловленность по группам ---
    report.append("")
    report.append("## Контроль по группам (по валидным ячейкам)")
    report.append("")
    report.append("| группа | признаков | NaN, % | cond корр. матрицы |")
    report.append("|---|---|---|---|")
    print(f"\n{'группа':<12}{'признаков':>10}{'NaN, %':>9}{'cond':>12}")
    groups: dict[str, list[str]] = {}
    for c in out.columns:
        g = group_of(c)
        if g and pd.api.types.is_numeric_dtype(out[c]):
            groups.setdefault(g, []).append(c)
    for g, cols in sorted(groups.items()):
        sub = out[cols]
        nan_pct = 100.0 * np.isnan(sub.to_numpy(dtype=float)[valid]).mean()
        cond = cond_number(sub, valid)
        print(f"{g:<12}{len(cols):>10}{nan_pct:>9.1f}{cond:>12.3g}")
        report.append(f"| `{g}` | {len(cols)} | {nan_pct:.1f} | {cond:.3g} |")

    # --- Скрытое дублирование МЕЖДУ группами ---
    from scipy.stats import spearmanr

    all_cols = [c for cols in groups.values() for c in cols]
    M = out[all_cols].to_numpy(dtype=float)[valid]
    ok = np.isfinite(M).all(axis=1)
    rho = spearmanr(M[ok]).statistic
    pairs = []
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            gi, gj = group_of(all_cols[i]), group_of(all_cols[j])
            if gi != gj:
                pairs.append((abs(rho[i, j]), all_cols[i], all_cols[j], rho[i, j]))
    pairs.sort(reverse=True)
    print("\nСамые связанные пары ИЗ РАЗНЫХ групп (скрытое дублирование источников):")
    report.append("")
    report.append("## Самые связанные пары из разных групп")
    report.append("")
    report.append("| признак A | признак B | Spearman |")
    report.append("|---|---|---|")
    for _, a, b, r in pairs[:10]:
        print(f"  {a:<18} ~ {b:<18} rho = {r:+.2f}")
        report.append(f"| `{a}` | `{b}` | {r:+.2f} |")

    report.append("")
    report.append("## Полный список признаков по группам")
    report.append("")
    for g, cols in sorted(groups.items()):
        report.append(f"**`{g}`** ({len(cols)}): " + " ".join(f"`{c}`" for c in cols))
        report.append("")
    no_group = [c for c in out.columns
                if group_of(c) is None and c not in ("row", "col", "x", "y")]
    if no_group:
        report.append("Вне групп (служебные, в обучении не участвуют): "
                      + " ".join(f"`{c}`" for c in no_group) + ".")
        report.append("")
    report.append("Столбцы `row`, `col`, `x`, `y` — геометрия ячейки, не признаки.")
    report.append("")
    report.append("## Что в датасет НЕ входит")
    report.append("")
    report.append("Стоп-лист постановки — прямые признаки минерагенической карты "
                  "(геохимические ореолы, геохимическое опробование, привнос "
                  "урана, точки рудопроявлений) и критериальный скор `prognoz` "
                  "с промежуточными `new_calc_prop*` "
                  "(`config.GOLD_FEATURES_STOP`). Всё это используется только "
                  "для заверки и сравнения методов, но никогда как признак.")

    out_path = config.PROCESSED_DIR / f"dataset_{version}.parquet"
    out.to_parquet(out_path, index=False)
    md = config.PROCESSED_DIR / f"dataset_{version}_sources.md"
    md.write_text("\n".join(report) + "\n", encoding="utf-8")

    # --- Превью: по одному представителю каждой новой группы ---
    shape = (meta.prf, meta.pic)
    show = [c for c in ("dem_incision", "lin_dens", "s2_clay", "s2_ndvi",
                        "s1_vv_vh", "psr_hh_hv", "ast_aloh", "l8_lst",
                        "s2_b11", "dem_tpi_2km") if c in out.columns][:6]
    if show:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, c in zip(axes.ravel(), show):
            img = np.where(valid.reshape(shape), out[c].to_numpy().reshape(shape), np.nan)
            lo, hi = np.nanpercentile(img, [2, 98])
            im = ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi)
            ax.set_title(c, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        for ax in axes.ravel()[len(show):]:
            ax.axis("off")
        fig.suptitle(f"Датасет {version}: {len(all_cols)} числовых признаков, "
                     f"{int(valid.sum())} валидных ячеек", fontsize=13)
        fig.tight_layout()
        out_png = ROOT / "outputs" / f"dataset_{version}_preview.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")

    print(f"\nСохранено: {out_path}\n           {md}")


if __name__ == "__main__":
    ver = sys.argv[1] if len(sys.argv) > 1 else "v2"
    if ver not in DATASETS:
        raise SystemExit(f"неизвестный датасет {ver!r}; доступны: {list(DATASETS)}")
    main(ver)
