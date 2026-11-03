"""Разметка полигонов `svita_new` по легенде ГГК-200: какая свита в каком полигоне.

Цель: канал для критерия К23 предметной модели — «ячейка в свите из списка
ВНИГНИ» (ответ 4 от 02.11.2026: ильинская, бурдурская, лабастахская,
усть-ильинская, котуйканская). В `svita_new.dbf` названий нет: 101 полигон,
у 66 код цвета `KRAP` = 0, у остальных два кода без расшифровки.

Входные данные:
* `data/Gis-integro/shp_dbf/svita_new.shp` — полигоны, оцифрованные с листа
  R-48-XI,XII (совмещение с растром проверено 21.10.2026, `src/ggk200_base`);
* `datacache/ggk200/R-48-XI_XII/*.tif` — привязанный скан листа;
* легенда листа — `datacache/ggk200/R-48-XI_XII/r-48-xixiigl.jpg`.

Как размечено (03–04.11.2026):
1. **По подписи** — индекс подразделения прочитан глазами на скане внутри
   полигона или по его штриховке и положению (фрагменты с контурами и
   номерами полигонов). Это основной способ для всех крупных полигонов.
2. **По блоку номеров** — номера полигонов идут блоками по подразделениям:
   893–947 четвертичные (Q IV), 953–962 сартанский горизонт (Q III sr),
   977–1005 долериты и диабазы, 995 и 1008–1051 коренные свиты. Внутри блока
   проверено глазами 59 полигонов из 79, исключений нет.
3. **По цвету** — мелкие полигоны коренных свит без подписи: ближайший по
   цветности (без яркости) полигон из размеченных по подписи. Отмечены
   «спорно», если ближайшая другая свита почти так же близка.

Ожидаемый результат:
* `data/processed/svita_labeled.gpkg` и `.csv` — полигон, индекс, подсвита,
  свита, серия, входит ли в список ВНИГНИ, способ разметки, уверенность;
* `outputs/13_генетическая_модель/gmo_svity_labels.png` — карта на подложке
  ГГК-200 для проверки геологом.

`svita_new` покрывает не весь лист: котуйканской свиты, кембрия и архея в нём
нет. Эта часть (04.11.2026) оцифрована со скана по цвету —
:func:`digitize_outside`; край слоя на северо-западе совпадает с контактом
котуйканской и усть-ильинской.

Оговорки:
* внутри полигона 1012 (усть-ильинская) не отделён блок котуйканской,
  опущенный по разлому у р. Тыкы-Пастах;
* в котуйканской не отделены мелкие окна усть-ильинской и останцы kt2;
* юсмастахская — кайма до 1 км вокруг кембрия, приблизительно.
Для канала К23 все три не важны: перепутаны только свиты из списка ВНИГНИ
между собой.

Запуск::

    python -X utf8 -m experiments.gmo_svita_labels
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import anabar_inset, ggk200_base, mapstyle, outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SHP = ROOT / "data/Gis-integro/shp_dbf/svita_new.shp"
OUT = ROOT / "data/processed/svita_labeled"
MAIN = (105.98, 70.65, 108.02, 71.35)

#: Подразделение легенды → (свита, серия, в списке ВНИГНИ).
UNITS = {
    "PR3ul":  ("усть-ильинская", "билляхская", True),
    "PR2lb":  ("лабастахская", "мукунская", True),
    "PR2br3": ("бурдурская", "мукунская", True),
    "PR2br2": ("бурдурская", "мукунская", True),
    "PR2br1": ("бурдурская", "мукунская", True),
    "PR2br":  ("бурдурская", "мукунская", True),     # подсвита не определена
    "PR2il2": ("ильинская", "мукунская", True),
    "PR2il1": ("ильинская", "мукунская", True),
    "PR2il":  ("ильинская", "мукунская", True),
    "QIV":    ("современные отложения", "четвертичные", False),
    "QIIIsr": ("сартанский горизонт", "четвертичные", False),
    "vbPR3":  ("позднепротерозойские диабазы", "интрузии", False),
    "bP2-T1": ("позднепермские — раннетриасовые долериты", "интрузии", False),
    "intr":   ("долериты или диабазы", "интрузии", False),
}

#: Прочитано по подписи на скане (или по штриховке и положению — помечено).
BY_LABEL = {
    1009: "PR3ul", 1012: "PR3ul", 1013: "PR3ul", 1014: "PR3ul", 1015: "PR3ul",
    1016: "PR2lb", 1017: "PR2lb", 1020: "PR2lb",
    1019: "PR2lb",   # подписи внутри нет: штриховка лабастахской, останец на бурдурской
    1021: "PR2br1", 1024: "PR2br1", 1041: "PR2br1",
    1038: "PR2br2", 1039: "PR2br2",
    1030: "PR2br3", 1031: "PR2br3", 1037: "PR2br3",
    1042: "PR2il2", 1043: "PR2il2", 1051: "PR2il2",
    1049: "PR2il1", 1050: "PR2il1",
    977: "vbPR3", 978: "vbPR3", 979: "vbPR3", 980: "vbPR3", 981: "vbPR3",
    982: "vbPR3", 989: "bP2-T1",
}
#: Проверено глазами внутри блоков номеров (четвертичные).
SEEN_Q = {893, 894, 895, 896, 899, 900, 901, 902, 903, 911, 912, 914, 915,
          917, 918, 919, 920, 922, 929, 931, 935, 936, 938, 940, 946, 947,
          958, 959, 960, 961}


def block_unit(pid: int) -> str | None:
    """Подразделение по блоку номеров; None — коренная свита, нужен цвет."""
    if 893 <= pid <= 947:
        return "QIV"
    if 953 <= pid <= 962:
        return "QIIIsr"
    if 977 <= pid <= 1005 and pid != 995:
        return "intr"
    return None


def polygon_colors(g: gpd.GeoDataFrame) -> pd.DataFrame:
    """Медианный цвет скана внутри каждого полигона, без чёрных линий и подписей."""
    import rasterio
    from rasterio import features
    from scipy import ndimage

    with rasterio.open(ggk200_base.sheet_tif("R-48-XI,XII")) as src:
        img = src.read().transpose(1, 2, 0)
        lab = features.rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)),
                                 out_shape=img.shape[:2], transform=src.transform, fill=0)
    rows = []
    for i, pid in enumerate(g.ID):
        m = lab == i + 1
        core = ndimage.binary_erosion(m, iterations=3)
        px = img[core if core.sum() >= 30 else m].astype(float)
        px = px[px.sum(1) > 250]
        rows.append((pid, *np.median(px, 0)))
    return pd.DataFrame(rows, columns=["ID", "R", "G", "B"]).set_index("ID")


def chroma(c: pd.DataFrame) -> np.ndarray:
    """Цветность без яркости: сканы листа местами темнее, оттенок устойчивее."""
    s = c[["R", "G", "B"]].to_numpy()
    return s / s.sum(1, keepdims=True)


def label_all(g: gpd.GeoDataFrame) -> pd.DataFrame:
    col = polygon_colors(g)
    ref = col.loc[[p for p in BY_LABEL if BY_LABEL[p].startswith("PR")]]
    ref_svita = np.array([UNITS[BY_LABEL[p]][0] for p in ref.index])
    ref_unit = np.array([BY_LABEL[p] for p in ref.index])
    ref_ch = chroma(ref)

    out = []
    for pid in g.ID:
        if pid in BY_LABEL:
            unit, how, conf = BY_LABEL[pid], "по подписи", "уверенно"
            if pid == 1019:
                how = "по штриховке и положению"
        elif (u := block_unit(pid)) is not None:
            unit = u
            how = "по подписи" if pid in SEEN_Q else "по блоку номеров"
            conf = "уверенно"
        else:
            d = np.linalg.norm(ref_ch - chroma(col.loc[[pid]]), axis=1)
            k = int(d.argmin())
            sv = ref_svita[k]
            other = d[ref_svita != sv].min()
            unit = ref_unit[k]
            # подсвиту по цвету не различаем: только свита
            unit = {"бурдурская": "PR2br", "ильинская": "PR2il"}.get(sv, unit)
            how = "по цвету"
            conf = "спорно" if other < 1.5 * d[k] else "вероятно"
        sv, series, listed = UNITS[unit]
        out.append(dict(ID=int(pid), unit=unit, svita=sv, series=series,
                        in_vnigni_list=listed, method=how, confidence=conf))
    return pd.DataFrame(out)


#: Контакт юсмастахской и котуйканской свит в северо-западном углу листа —
#: тёмная линия от сартанских отложений к югу, снята со скана по пикселям
#: (x, y). Западнее неё и вне кембрия — юсмастахская; оцифровка приблизительная.
JUS_EDGE = [(224, 232), (218, 268), (212, 340), (208, 380), (205, 420)]
JUS_TOP, JUS_BOTTOM = 100, 420
NW_MAX_X = 2500          # розовое западнее — котуйканская, восточнее — докембрий юго-востока


def digitize_outside(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Оцифровка со скана той части листа, которой нет в `svita_new`.

    Контакты котуйканской и усть-ильинской на скане почти не различаются по
    цвету, а автоматическое деление по тёмным линиям течёт через разрывы
    контуров. Поэтому классы берутся по цвету скана после медианного
    сглаживания (линии и подписи не красят участок):
    зелёное — кембрий (тёмное) или интрузии (светлое), жёлтое — четвертичные
    отложения, малиновое — архей, розовое — докембрий. Розовое на
    северо-западе — котуйканская; кусок в юго-западном углу, отрезанный слоем,
    — усть-ильинская (подпись PR3ul у западной рамки); полоса у кембрия
    западнее контакта `JUS_EDGE` — юсмастахская.
    """
    import rasterio
    import shapely
    from rasterio import features
    from scipy import ndimage

    with rasterio.open(ggk200_base.sheet_tif("R-48-XI,XII")) as src:
        img = src.read().transpose(1, 2, 0).astype(float)
        T = src.transform
    inl = features.rasterize(((p, 1) for p in g.geometry), out_shape=img.shape[:2],
                             transform=T, fill=0).astype(bool)
    sm = np.stack([ndimage.median_filter(img[..., c], size=9) for c in range(3)], -1)
    R, G, B = sm[..., 0], sm[..., 1], sm[..., 2]
    green = G > R + 10
    cls = np.select([green & (G < 200), green, G - B > 22,
                     (R - G > 45) & (B - G > -5)], [1, 2, 3, 4], 5)
    # сгладить: доля каждого класса в окне ~150 м, побеждает большинство
    frac = np.stack([ndimage.uniform_filter((cls == k).astype(float), 11)
                     for k in range(1, 6)], -1)
    cls = frac.argmax(-1) + 1
    cls[inl] = 0

    yy, xx = np.mgrid[:cls.shape[0], :cls.shape[1]]
    edge_x = np.interp(yy, [p[1] for p in JUS_EDGE], [p[0] for p in JUS_EDGE])
    # юсмастахская лежит каймой вокруг кембрия: полоса до 1 км от него
    # (пиксель скана ~15 x 45 м), западнее контакта и в пределах угла
    near_cm = ndimage.distance_transform_edt(cls != 1, sampling=(45.0, 14.7)) <= 1000
    jus = ((cls == 5) & near_cm & (yy >= JUS_TOP) & (yy <= JUS_BOTTOM)
           & (xx < edge_x))

    pink, n = ndimage.label(cls == 5)
    cx = ndimage.center_of_mass(np.ones_like(pink), pink, range(1, n + 1))
    unit = np.zeros(cls.shape, np.uint8)
    unit[cls == 1] = 1                      # кембрий
    unit[cls == 2] = 2                      # интрузии
    unit[cls == 3] = 3                      # четвертичные
    unit[cls == 4] = 4                      # архей
    for k, (cy, cxx) in enumerate(cx, start=1):
        m = pink == k
        # Край svita_new на северо-западе и есть контакт котуйканской с
        # усть-ильинской: севернее него подписи PR3kt1, южнее — PR3ul, а за
        # краем других подписей нет. Мелкие замкнутые контуры внутри (окна
        # усть-ильинской, как у западной рамки на строке ~1440, и останцы kt2)
        # не отделяются: обе свиты в списке, канал К23 от этого не меняется.
        unit[m] = 4 if cxx >= NW_MAX_X else 5
    unit[jus] = 7

    codes = {1: "Cm1", 2: "intr", 3: "Q", 4: "AR", 5: "PR3kt", 7: "PR3jus"}
    sheet = anabar_inset.sheet_lonlat()
    rows = []
    for geom, v in features.shapes(unit, mask=unit > 0, transform=T):
        p = shapely.geometry.shape(geom).intersection(sheet)
        if p.is_empty:
            continue
        rows.append(dict(unit=codes[int(v)], geometry=p))
    out = gpd.GeoDataFrame(rows, crs=4326).dissolve("unit").reset_index().explode(index_parts=False)
    km2 = out.to_crs(anabar_inset.SHEET_CRS).area / 1e6
    out = out[km2.to_numpy() >= 0.25].reset_index(drop=True)   # мельче ячейки — шум
    out["ID"] = np.arange(2001, 2001 + len(out))
    out["method"] = "по цвету скана"
    out["confidence"] = np.where(out.unit == "PR3jus", "приблизительно", "вероятно")
    return out


UNITS.update({
    "PR3kt":  ("котуйканская", "билляхская", True),
    "PR3jus": ("юсмастахская", "билляхская", False),
    "Cm1":    ("кембрий (немакит-далдынская, медвежинская)", "кембрий", False),
    "Q":      ("четвертичные отложения", "четвертичные", False),
    "AR":     ("архей и нижний протерозой", "фундамент", False),
})

COLORS = {"котуйканская": "#7b241c", "усть-ильинская": "#c0392b", "лабастахская": "#e67e22",
          "бурдурская": "#f1c40f", "ильинская": "#8e44ad"}


def draw_map(g: gpd.GeoDataFrame) -> Path:
    # лист в долготе-широте с поправкой на широту выходит почти квадратным:
    # легенда и врезка — справа за рамкой, чтобы не закрывать юг листа
    fig, ax = plt.subplots(figsize=(13, 8.2))
    fig.subplots_adjust(left=0.06, right=0.58, top=0.93, bottom=0.08)
    ggk200_base.draw(ax, bounds=MAIN, decimate=2, alpha=0.55, sheets=("R-48-XI,XII",))
    rock = g[g.in_vnigni_list]
    for sv, c in COLORS.items():
        part = rock[rock.svita == sv]
        part[part.confidence == "уверенно"].plot(ax=ax, facecolor=c, edgecolor="k",
                                                 linewidth=0.3, alpha=0.55)
        part[part.confidence != "уверенно"].plot(ax=ax, facecolor=c, edgecolor="k",
                                                 linewidth=0.8, alpha=0.55, hatch="xx")
    g[~g.in_vnigni_list].plot(ax=ax, facecolor="0.85", edgecolor="0.4",
                              linewidth=0.3, alpha=0.8)
    g[g.source == "svita_new"].dissolve().boundary.plot(ax=ax, color="#1f3a93", linewidth=1.4)
    ax.set_xlim(MAIN[0], MAIN[2])
    ax.set_ylim(MAIN[1], MAIN[3])
    handles = [Patch(facecolor=c, alpha=0.55, edgecolor="k", label=f"{sv} свита")
               for sv, c in COLORS.items()]
    handles += [Patch(facecolor="white", edgecolor="k", hatch="xx",
                      label="не по подписи: по цвету скана или полигона"),
                Patch(facecolor="0.85", edgecolor="0.4",
                      label="четвертичные отложения, долериты и диабазы"),
                Patch(facecolor="none", edgecolor="#1f3a93", linewidth=1.4,
                      label="граница слоя svita_new; за ней — оцифровка со скана")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9, framealpha=0.92,
              title="Свиты из списка ВНИГНИ (ответ 4 от 02.11.2026)", title_fontsize=9)
    ax.set_xlabel("долгота, ° в.д.")
    ax.set_ylabel("широта, ° с.ш.")
    ax.set_title("Лист R-48-XI,XII: свиты по слою svita_new и скану ГГК-200",
                 fontsize=12)
    anabar_inset.draw(ax.inset_axes([1.04, 0.0, 0.42, 0.45]), fontsize=7,
                      title=False, legend=False)
    out = outputs.png("gmo_svity_labels.png")
    mapstyle.save_fig(fig, out, dpi=140)
    return out


def main() -> None:
    raw = gpd.read_file(SHP)
    g = ggk200_base.to_lonlat(raw)
    lab = label_all(g)
    g = g.merge(lab, on="ID")

    # площади — в проекции листа, а не в градусах
    area = raw.set_crs(anabar_inset.SHEET_CRS).geometry.area / 1e6
    g["area_km2"] = area.round(3).to_numpy()
    g["source"] = "svita_new"
    g["note"] = ""
    # Полигон 1012 не отделяет блок котуйканской, опущенный по разлому
    # (подпись PR3kt1 у р. Тыкы-Пастах). Для К23 это не важно — обе свиты в
    # списке, — но название свиты на этом блоке неверно.
    g.loc[g.ID == 1012, "note"] = "внутри блок котуйканской по разлому, не отделён"

    scan = digitize_outside(g)
    meta = scan.unit.map(UNITS)
    scan["svita"] = meta.str[0]
    scan["series"] = meta.str[1]
    scan["in_vnigni_list"] = meta.str[2].astype(bool)
    scan["area_km2"] = (scan.to_crs(anabar_inset.SHEET_CRS).area / 1e6).round(3).to_numpy()
    scan["source"] = "скан ГГК-200"
    scan["note"] = ""
    g = pd.concat([g, scan[g.columns.drop(["KRAP", "LINKS_QTY", "L_CODE", "B_CODE"],
                                          errors="ignore")]], ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
    # площадь листа — его трапеция по рамке скана (2° x 40'), а не
    # прямоугольник SHEET_BOX в проекции, который на ~400 км² больше
    import rasterio, shapely
    with rasterio.open(ggk200_base.sheet_tif("R-48-XI,XII")) as src:
        frame = shapely.box(*src.bounds)
    sheet_km2 = gpd.GeoSeries([frame], crs=4326).to_crs(anabar_inset.SHEET_CRS).area.iloc[0] / 1e6

    g.to_crs(anabar_inset.SHEET_CRS).to_file(OUT.with_suffix(".gpkg"), driver="GPKG")
    g.drop(columns="geometry").to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8")

    print(f"Полигонов {len(g)}; площадь листа {sheet_km2:.0f} км², слоя {g.area_km2.sum():.0f} км²")
    t = (g.groupby(["svita", "unit"]).agg(n=("ID", "size"), km2=("area_km2", "sum"))
         .round(1).reset_index())
    print(t.to_string(index=False))
    print()
    print(g.groupby(["method", "confidence"]).agg(n=("ID", "size"), km2=("area_km2", "sum"))
          .round(2).to_string())
    print()
    print("Размечено по цвету:")
    print(g[g.method == "по цвету"][["ID", "svita", "confidence", "area_km2"]]
          .to_string(index=False))
    print(f"\nКарта: {draw_map(g)}")
    print(f"Слой: {OUT.with_suffix('.gpkg')}")


if __name__ == "__main__":
    main()
