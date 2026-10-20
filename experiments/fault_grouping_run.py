"""Задача №3, пилот: группировка даек в зоны против разломных зон карты.

ЗАЧЕМ ПИЛОТ И ПОЧЕМУ НА ДАЙКАХ. Задача просит группировать отдельные разломы
геологической карты в зоны. Слоя отдельных разломов в переданных данных нет:
``glub_raz_nw`` — 4 полигона, ``glub_r_nw`` — 2, и это уже результат искомого
обобщения. Единственная имеющаяся популяция отдельных ориентированных объектов —
дайки (151 полигон, роль ``magm``), причём роза их азимутов, взвешенная длиной,
сама повторяет СВ-систему разломов. То есть у пилота есть заранее известный
правильный ответ, и метод группировки (:mod:`src.fault_grouping`) проверяется
ДО того, как в него вложена векторизация разломов с растра ГГК-200 нейросетью.

Логика решения: не воспроизводит зоны там, где ответ известен, — вкладываться в
векторизацию незачем; воспроизводит — векторизация становится следующим шагом,
а метод к её выходу уже готов (``axial_segments`` принимает и линии).

ПОРЯДОК:

1. оси даек (описанный прямоугольник) -> моды азимута -> зоны внутри моды;
2. цели — ячейки внутри коридоров карты, отдельно СВ и СЗ. Азимуты самих
   коридоров снимаются ТОЙ ЖЕ функцией ``axial_segments``, а не вписываются
   числом: кандидатом к цели идёт мода, ближайшая к её азимуту;
3. протокол — тот же, что у задачи №2: lift@10% со сдвиговым null, он же после
   вычитания рельефа, плюс IoU маски зон с целью (тоже со сдвиговым null).
   Планки сравниваются с кандидатом на РАВНОЙ площади маски: IoU у карт разного
   размера несравним, поэтому планке отдаётся верхняя доля площади, равная
   реализованной площади зон кандидата.

ПЛАНКИ (не гипотезы, шкала). ``random`` — пол. ``dens_magm`` и ``-dist_magm`` —
дайки БЕЗ всякой группировки: если они дают то же, группировка не добавляет
ничего, и это главный вопрос пилота. ``shuffle_az`` — тот же метод при
перемешанных между дайками азимутах: сохраняет плотность и положения, разрушает
со-направленность. ``one_zone`` — тривиальное обобщение «все дайки одной зоной».

Выход: ``outputs/metrics/fault_grouping_summary.csv``,
``outputs/metrics/fault_grouping_zones.csv``, ``outputs/fault_grouping.png``.
Запуск из корня: ``python -m experiments.fault_grouping_run``.

РЕЗУЛЬТАТ (21.08.2026, dataset_v6, 5000 сдвигов, накопительный порог
alpha = 0.05 / (68 + 6) = 0.00068). Вердикт по пре-регистрированному критерию
:data:`config.FG_PRIMARY` — НЕ ВОСПРОИЗВОДИТ, на обеих целях.

СВ, ПЕРВИЧНАЯ ЦЕЛЬ. Критерий (1) не выполнен, причём на 0.1°: мода даек 52.0°
против 41.9° у коридоров карты, |Δ| = 10.1° при допуске 10°. Дайковый рой,
который по розе «повторяет СВ-систему», на деле развёрнут относительно неё на
десять градусов. IoU 0.439 при p = 0.0002 (это пол шкалы 5000 сдвигов), но
планка БЕЗ группировки ``-dist_magm`` на той же площади даёт 0.426 — перевес
0.013. Значимость есть, содержательного превосходства над «просто близостью к
дайке» нет.

СЗ, ВТОРИЧНАЯ ЦЕЛЬ. Направление совпало отлично: мода 145.0° против 144.3°.
IoU 0.420 против 0.339 у обеих планок равной площади и 0.375 у q95 нуля — это
единственный замер, где группировка даёт что-то сверх «дайки как они есть». Но
p = 0.0104, то есть до накопительного порога не хватает полутора порядков.

ПОЧЕМУ — ВИДНО ПО ДИАГНОСТИКЕ ДО СВЕРКИ, А НЕ ПО ТАБЛИЦЕ. Внутри СВ-моды
101 дайка растянута по перпендикулярному сносу на 84.3 км при медианном зазоре
0.42 км, и зазоров шире порога слияния всего два. Поэтому 90 даек из 101
сливаются в ОДНУ полосу шириной 50.8 км, накрывающую 78.8% пула: коридорной
структуры в положениях даек просто нет, разделять нечего. У СЗ-моды зазоров
пять, зоны выходят ýже (35.4 / 13.7 / 11.0 / 5.0 км) и попадают в диапазон
ширин коридоров карты (8-21 км) — там метод ведёт себя осмысленно.

ЧТО ВИДНО НА КАРТЕ (``outputs/fault_grouping.png``). У СЗ направление
воспроизведено с точностью 0.7°, но полосы СМЕЩЕНЫ: серые коридоры карты видны
вдоль краёв красных зон, а не под ними. Ровно тот случай, ради которого метрика
считается по площади, а не по розе, — роза совпала бы идеально.

LIFT@10% ДЛЯ КАРТ ЗОН НЕИНФОРМАТИВЕН, И ЭТО СЛЕДУЕТ ИЗ УСТРОЙСТВА КАРТЫ, А НЕ
ИЗ РЕЗУЛЬТАТА. Внутри тела зоны поле постоянно (гауссов спад начинается за
границей), поэтому порядок ячеек внутри задают только хвосты СОСЕДНИХ зон.
Измерено: у СВ-кандидата зоны не пересекаются вовсе (0% пула в двух зонах),
верхние 10% поля — это часть большой полосы, ближайшая к двум малым, и целевых
ячеек там нет ни одной, откуда lift = 0.000. Читать столбец как «карта хуже
случайной» нельзя; ответ на вопрос задачи даёт IoU. Столбец оставлен в таблице
как есть: он был пре-регистрирован, и убирать его после взгляда на результат
нельзя.

ПЛАНКИ ВЕДУТ СЕБЯ КАК ОБЯЗАНЫ, и это проверка самого протокола. ``random`` —
lift 1.026 / 0.960 при p = 0.16 / 0.94, пол на месте. ``shuffle_az`` (положения
и плотность сохранены, со-направленность разрушена) даёт IoU 0.373 / 0.270 —
ниже кандидата на обеих целях, то есть направление в группировке работает.

ГЛАВНОЕ ЧИСЛО ПИЛОТА — НЕ У КАНДИДАТА, А У ПЛАНКИ. Близость к дайке без всякой
группировки даёт на СВ-коридоре lift 1.679 (p = 0.0002) и 1.566 после вычитания
рельефа (p = 0.0010), плотность даек — 1.777 / 1.703. Дайки и СВ-разломы
действительно сопространственны, что для заполненных трещин и ожидается, но
обобщение в зоны эту связь не усиливает ни на одной цели.

ЧТО ЭТО ЗНАЧИТ ДЛЯ ЗАДАЧИ №3. Сам метод работоспособен: на синтетике
(``tests/test_fault_grouping.py``) он восстанавливает заданные рои и их
геометрию, а на СЗ-моде выдаёт зоны в диапазоне ширин карты. Ограничение —
вход: 151 объект медианной длины 4 км, растянутый по сносу почти без разрывов.
Строгое чтение решающего правила («не воспроизводит — в векторизацию не
вкладываться») закрывает переход к нейросетевой векторизации ГГК-200; диагноз
«ограничение во входе, а не в алгоритме» его смягчает, но это уже суждение, а
не измерение. Дешёвая проверка самого диагноза — прогнать тот же метод на
рельефных линеаментах (:mod:`src.lineaments`, сотни отрезков, единственная
ветка задачи №8, прошедшая стоп-правило): если дело в разреженности входа,
там зоны обязаны сузиться. Отдельный прогон требует своей пре-регистрации и
своих слагаемых в ledger.

РЕЗУЛЬТАТ БЛОКА 2 (21.08.2026: две популяции x два оператора x 2 цели, 5000
сдвигов, накопительный порог alpha = 0.05 / (68 + 24) = 0.00054). Вердикт —
НЕ ВОСПРОИЗВОДИТ во всех восьми сочетаниях.

ПРЕДСКАЗАНИЕ, РАДИ КОТОРОГО БЛОК ЗАТЕВАЛСЯ, НЕ ПОДТВЕРДИЛОСЬ, И ЭТО БЫЛО
ЗАПИСАНО ДО ПРОГОНА. Диагноз блока 1 звучал как «зоны слиплись из-за
разреженного входа». Измерение популяции линеаментов ДО всякой сверки показало
обратное: 3087 отрезков дают медианный зазор в ``rho`` 26-56 МЕТРОВ и ни одного
зазора шире порога слияния ни в одной из трёх мод. Плотный вход зоны не сужает,
а склеивает вернее — разрыв есть только у разреженного. Отсюда и второй
оператор (разрез по долинам плотности ``rho``), пре-регистрированный этим же
решением, до сверки; см. блок ``FG_RHO_KDE_BW_M`` в :mod:`src.config`.

НАПРАВЛЕНИЕ ПОПАДАЕТ ВПЕРВЫЕ, ПОЛОЖЕНИЕ — НЕТ. Мода линеаментов 39.0° против
41.9° у СВ-коридоров: |Δ| = 2.9°, критерий (1) выполнен впервые за ветку (у
даек было 10.1°). Но моды под СЗ-систему у рельефа нет вовсе — ближайшая 167.0°
при |Δ| = 22.7°. Рельеф листа несёт СВ-направление карты и не несёт СЗ.

ЗОНЫ ВЫРОЖДАЮТСЯ В ЛИСТ. Плотность ``rho`` у всех трёх мод почти ровная: пиков
1-3 при контрасте max/median 2.09-4.17. Разрешения хватало заведомо — полоса
ядра 5 км против размаха 90 км даёт ~18 разрешаемых элементов при ширине
коридоров карты 8-21 км. Оба оператора дают 1-2 зоны, накрывающие 93-100% пула;
на цели СЗ маска — ровно 100% пула, и все шесть строк таблицы совпадают до
четвёртого знака, включая ``random``: мерить там нечего.

ЧИСЛА. Лучшее у линеаментов — ``kde`` на СВ: IoU 0.379 при null q95 0.373 и
p = 0.0090 против порога 0.00054. ``dbscan`` на СВ: 0.373 при q95 0.373,
p = 0.0540.

ПЛАНКА ПЕРЕИГРЫВАЕТ МЕТОД — САМЫЙ ЧИСТЫЙ РЕЗУЛЬТАТ БЛОКА. ``shuffle_az``, то
есть та же группировка при перемешанных между объектами азимутах, даёт на
линеаментах/kde/СВ IoU 0.387 (p = 0.0006) против 0.379 у метода, а на
дайках/kde/СВ — 0.396 против 0.370. Разрушение со-направленности результат не
ухудшает: «зона» здесь держится на плотности объектов, а не на согласованности
их направлений, то есть группировать по направлению на этом листе нечего.

СТАРЫЕ ЧИСЛА НЕ ИЗМЕНИЛИСЬ. Блок 1 (дайки/``dbscan``) воспроизвёлся до
четвёртого знака. Порог при этом ужесточился с 0.00068 до 0.00054; вердикты
перечитаны по новому порогу и остались прежними.

ЧТО ЭТО ЗАКРЫВАЕТ. Гипотеза «структура есть, нужен лишь более плотный вход»
проверена на популяции в 20 раз плотнее и отвергнута: ни дайки, ни рельефные
линеаменты не содержат коридорной структуры в ``rho`` на масштабе коридоров
карты. Правдоподобное объяснение подсказывает сам слой: ``glub_raz_nw`` — это
ГЛУБИННЫЕ разломы, а они выделяются по геофизике и региональной схеме, а не по
поверхностной морфологии. Тогда переданные 6 полигонов — не обобщение
наблюдаемых линейных объектов по их геометрии, и собрать их группировкой
поверхностных объектов нельзя ни кластеризацией, ни нейросетью. Проверить это
на имеющихся данных нечем, поэтому объяснение остаётся гипотезой, а измеренный
факт — отрицательный результат на обеих доступных популяциях.

СЛЕДСТВИЕ ДЛЯ НЕЙРОСЕТЕВОЙ ВЕТКИ ЗАДАЧИ. Ждать «зон» от векторизации ГГК-200
не следует — её ценность в самом слое отдельных разломов, который у заказчика
и запрошен письмом. Если слой появится, метод к нему готов без правок
(``axial_segments`` принимает линии), а прогон на нём требует своей
пре-регистрации и своих слагаемых в ledger сверх нынешних 92.
"""
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (cell_mask, config, fault_grouping as fg,  # noqa: E402
                 fracture_zones as fz, integro_grid)
from src.data_loader import find_base_dir, load_all_layers  # noqa: E402

#: Цель -> роль слоя разломов и колонка расстояния до него в датасете.
TARGETS: dict[str, tuple[str, str]] = {
    "СВ (glub_r_nw)": ("tect2", "dist_tect2"),
    "СЗ (glub_raz_nw)": ("tect1", "dist_tect1"),
}


def _load_dataset() -> pd.DataFrame:
    """Датасет из ``data/processed``, иначе из самого свежего ``data/dataset_v*.zip``."""
    for name in sorted(config.PROCESSED_DIR.glob("dataset_v*.parquet"), reverse=True):
        return pd.read_parquet(name)
    zips = sorted((ROOT / "data").glob("dataset_v*.zip"), reverse=True)
    if not zips:
        raise FileNotFoundError(
            "не найден ни data/processed/dataset_v*.parquet, ни data/dataset_v*.zip")
    with zipfile.ZipFile(zips[0]) as z:
        member = next(n for n in z.namelist() if n.endswith(".parquet"))
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        z.extract(member, config.PROCESSED_DIR)
    return pd.read_parquet(config.PROCESSED_DIR / member)


def _mask_of_top(score: np.ndarray, meta, pool: np.ndarray,
                 area: float) -> np.ndarray:
    """Маска верхней доли ``area`` карты по пулу — планка равной с зонами площади.

    Без неё IoU планки и кандидата несравнимы: у карты вдвое большей площади IoU
    механически другой, и разница читалась бы как содержательная.
    """
    flat = np.asarray(score, dtype=float).ravel().copy()
    ref = flat[pool]
    thr = np.nanquantile(ref, 1.0 - area)
    out = np.zeros(flat.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        out[pool] = np.nan_to_num(flat[pool], nan=-np.inf) >= thr
    return out.reshape(meta.shape)


def _zone_maps(seg: pd.DataFrame, az_target: float, meta,
               operator: str = "dbscan") -> tuple[np.ndarray, np.ndarray, list, float]:
    """Зоны моды, ближайшей к азимуту цели: поле, маска, список зон, азимут моды.

    Кандидатом к цели идёт ровно одна мода — та, чьё направление ближе всего к
    направлению самих коридоров карты. Брать «все зоны сразу» нельзя: тогда
    кандидат к СВ-цели включал бы и СЗ-зоны, и совпадение мерилось бы у смеси.
    """
    modes = fg.direction_modes(seg["az_deg"].to_numpy(),
                               weights=seg["length_m"].to_numpy())
    if modes["modes_deg"].size == 0:
        return np.zeros(meta.shape), np.zeros(meta.shape, dtype=bool), [], float("nan")
    i = int(np.argmin(fg.axial_diff_deg(modes["modes_deg"], az_target)))
    az_mode = float(modes["modes_deg"][i])
    lab = fg.assign_modes(seg["az_deg"].to_numpy(), modes["modes_deg"])
    pick = fg.ZONE_OPERATORS[operator](seg, az_mode, idx=np.where(lab == i)[0])
    return fg.zone_field(pick, meta), fg.zone_mask(pick, meta), pick, az_mode


def _one_zone(seg: pd.DataFrame, az_deg: float) -> list:
    """Планка «все дайки одной зоной»: то же направление, без разделения на сгущения."""
    along, rho = fg.along_rho(seg["x"].to_numpy(), seg["y"].to_numpy(), az_deg)
    half = 0.5 * seg["length_m"].to_numpy()
    return [fg.Zone(az_deg, float(rho.min()), float(rho.max()),
                    float((along - half).min()), float((along + half).max()),
                    float(seg["length_m"].sum()), np.arange(len(seg)))]


def _dyke_population(layers) -> tuple[pd.DataFrame, dict, str]:
    """Дайки: 151 полигон роли ``magm``, планки без группировки — их плотность и близость."""
    seg = fg.axial_segments(layers["magm"].geometry)
    return seg, {"dens_magm": "dens_magm", "-dist_magm": "-dist_magm"}, "magm"


def _lineament_population(meta) -> tuple[pd.DataFrame, dict, str]:
    """Рельефные линеаменты: отрезки Хафа по многоазимутной отмывке.

    Берётся ровно боевой путь :mod:`src.lineaments` без единого изменения
    параметров: это единственная ветка задачи №8, прошедшая стоп-правило
    воспроизводимости, и переопределять её пороги здесь означало бы проверять
    уже другую популяцию.

    Отрезки приходят в пикселях рабочего растра (шаг ``TER_RES_M``, поле
    ``TER_PAD_PX`` до начала сетки), поэтому переводятся в координаты сетки и
    подаются как линии — :func:`src.fault_grouping.axial_segments` принимает их
    наравне с полигонами.
    """
    from shapely.geometry import LineString

    from src import lineaments, terrain_v2

    elev = terrain_v2.projected_dem(meta)
    elev = np.where(np.isfinite(elev), elev, np.nanmedian(elev))
    segments = lineaments.extract_lines(lineaments.edge_map(elev))
    pad, res = config.TER_PAD_PX, config.TER_RES_M

    def to_map(px, py):
        return (meta.x0 + (px - pad + 0.5) * res, meta.y_top - (py - pad + 0.5) * res)

    lines = [LineString([to_map(*a), to_map(*b)]) for a, b in segments]
    return (fg.axial_segments(lines),
            {"lin_dens": "lin_dens", "-lin_dist": "-lin_dist"}, "lin")


#: Популяция -> загрузчик. Дайки идут первыми: на них ветка пре-регистрирована.
POPULATIONS: tuple[str, ...] = ("дайки", "линеаменты")


def _plank_score(name: str, df: pd.DataFrame) -> np.ndarray:
    """Скор планки по имени: ведущий минус означает «ближе — лучше»."""
    if name.startswith("-"):
        return -df[name[1:]].to_numpy(dtype=float)
    return df[name].to_numpy(dtype=float)


def _mode_diagnostics(seg: pd.DataFrame, log) -> dict:
    """Роза, моды и структура ``rho`` — печатается ДО всякой сверки с целями.

    Зазоры в ``rho`` — единственная величина, по которой этап 2 вообще может
    разделить зоны оператором ``dbscan``: если разрывов шире порога слияния нет,
    вся мода обязана слипнуться в одну полосу, и число зон определяется входом,
    а не методом.
    """
    modes = fg.direction_modes(seg["az_deg"].to_numpy(),
                               weights=seg["length_m"].to_numpy())
    lab = fg.assign_modes(seg["az_deg"].to_numpy(), modes["modes_deg"])
    modes = dict(modes)
    modes["labels"] = lab
    log("моды азимута (вес — суммарная длина объектов):")
    for i, (az, w) in enumerate(zip(modes["modes_deg"], modes["modes_weight"])):
        sub = seg.iloc[np.where(lab == i)[0]]
        rho = np.sort(fg.along_rho(sub["x"].to_numpy(), sub["y"].to_numpy(),
                                   float(az))[1])
        gaps = np.diff(rho)
        # Профиль плотности rho: то, по чему режет второй оператор. Число пиков
        # и контраст «пик/долина» отвечают на вопрос, есть ли в популяции
        # коридорная структура ВООБЩЕ, до всякой сверки с картой. Полоса ядра
        # 5 км против размаха ~90 км — это ~18 разрешаемых элементов при ширине
        # коридоров карты 8-21 км, то есть разрешения на них заведомо хватает.
        grid = np.arange(rho.min(), rho.max() + config.FG_RHO_KDE_STEP_M,
                         config.FG_RHO_KDE_STEP_M)
        dens = (sub["length_m"].to_numpy()[None, :] * np.exp(
            -0.5 * ((grid[:, None] - rho[None, :]) / config.FG_RHO_KDE_BW_M) ** 2)
        ).sum(axis=1)
        peaks = int(((dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:])).sum())
        log(f"    {az:5.1f}°  {w / 1000:7.0f} км  объектов {len(sub):4d}  "
            f"размах rho {(rho[-1] - rho[0]) / 1000:5.1f} км  "
            f"зазор медианный {np.median(gaps) / 1000:.3f} км, "
            f"максимальный {gaps.max() / 1000:.1f} км, "
            f"шире порога {int((gaps > config.FG_RHO_EPS_M).sum())}; "
            f"плотность rho: пиков {peaks}, контраст max/median "
            f"{dens.max() / np.median(dens):.2f}")
    return modes


def _target_azimuth(ref: pd.DataFrame) -> float:
    """Осевое среднее азимутов полигонов карты, взвешенное их длиной."""
    return float(np.degrees(np.arctan2(
        np.average(np.sin(2 * np.radians(ref["az_deg"])), weights=ref["length_m"]),
        np.average(np.cos(2 * np.radians(ref["az_deg"])), weights=ref["length_m"]),
    )) / 2.0 % 180.0)


def main(n_shifts: int | None = None,
         populations: tuple[str, ...] | None = None,
         operators: tuple[str, ...] | None = None) -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    n_shifts = config.FG_N_SHIFTS if n_shifts is None else n_shifts
    populations = POPULATIONS if populations is None else populations
    operators = tuple(fg.ZONE_OPERATORS) if operators is None else operators

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    df = _load_dataset()
    pool = np.where(cell_mask.build_valid_mask(df))[0]
    log(f"сетка {meta.prf}x{meta.pic}, валидных ячеек {pool.size}, "
        f"площадь {pool.size * meta.dx * meta.dy / 1e6:.0f} км2")

    base = find_base_dir()
    with tempfile.TemporaryDirectory() as ad:
        layers, _ = load_all_layers(base / config.SHP_SUBDIR, Path(ad))

    controls = fz.relief_control_matrix(df)
    alpha = 0.05 / (config.FG_N_TESTS_PRIOR + config.FG_N_TESTS)
    log(f"порог Бонферрони: alpha = 0.05 / ({config.FG_N_TESTS_PRIOR} + "
        f"{config.FG_N_TESTS}) = {alpha:.5f}; сдвигов {n_shifts}")

    # Цели и их азимуты — общие для всех популяций и операторов.
    targets = {}
    for tname, (role, dist_col) in TARGETS.items():
        ref = fg.axial_segments(layers[role].geometry)
        idx = fz.fault_corridor_cells(df[dist_col].to_numpy(dtype=float), 0.0, pool=pool)
        share = idx.size / pool.size
        targets[tname] = (idx, _target_azimuth(ref))
        log(f"цель «{tname}»: полигонов {len(ref)}, азимут {targets[tname][1]:.1f}°, "
            f"{idx.size} ячеек ({share:.1%} пула), потолок lift@{config.FG_AREA:.0%} = "
            f"{min(1.0, config.FG_AREA / share) / config.FG_AREA:.2f}")

    rows, zone_rows, crit = [], [], {}
    for pname in populations:
        seg, planks, _ = (_dyke_population(layers) if pname == "дайки"
                          else _lineament_population(meta))
        log(f"=== популяция «{pname}»: объектов {len(seg)}, длина медианно "
            f"{seg['length_m'].median() / 1000:.2f} км, суммарно "
            f"{seg['length_m'].sum() / 1000:.0f} км ===")
        _mode_diagnostics(seg, log)

        rng = np.random.default_rng(config.VER_SEED)
        seg_shuf = seg.copy()
        seg_shuf["az_deg"] = rng.permutation(seg["az_deg"].to_numpy())
        rnd = rng.random(len(df))

        for op in operators:
            for tname, (target, az_ref) in targets.items():
                field, mask, picked, az_mode = _zone_maps(seg, az_ref, meta, operator=op)
                d_az = float(fg.axial_diff_deg(az_mode, az_ref))
                area_zones = float(mask.ravel()[pool].mean())
                log(f"--- {pname} / {op} / {tname}: мода {az_mode:.1f}° "
                    f"(|Δ| = {d_az:.1f}°), зон {len(picked)}, маска {area_zones:.1%} пула")
                for z in picked:
                    zone_rows.append({"популяция": pname, "оператор": op, "цель": tname,
                                      **fg.zone_table([z]).iloc[0].to_dict()})

                one = _one_zone(seg, az_mode)
                f_shuf, m_shuf, _, _ = _zone_maps(seg_shuf, az_ref, meta, operator=op)
                cands = {
                    "зоны (метод)": (field, mask, False),
                    "shuffle_az": (f_shuf, m_shuf, True),
                    "one_zone": (fg.zone_field(one, meta), fg.zone_mask(one, meta), True),
                    **{p: (_plank_score(p, df), None, True) for p in planks},
                    "random": (rnd, None, True),
                }
                for cname, (score, cmask, is_plank) in cands.items():
                    # Карты зон приходят формы сетки, признаки датасета — плоские;
                    # протокол работает с плоскими.
                    score = np.asarray(score, dtype=float).ravel()
                    res = fz.screen_candidate(score, df, meta, target, pool,
                                              controls=controls, area=config.FG_AREA,
                                              n_shifts=n_shifts)
                    m = (_mask_of_top(score, meta, pool, area_zones)
                         if cmask is None else cmask)
                    iou = fg.mask_iou_pvalue(m, target, meta, pool=pool,
                                             n_shifts=n_shifts)
                    rows.append({
                        "популяция": pname, "оператор": op, "цель": tname,
                        "карта": cname, "планка": is_plank,
                        "площадь_маски": float(np.asarray(m).ravel()[pool].mean()),
                        **res, "IoU": iou["observed"], "p_IoU": iou["p"],
                        "IoU_null_q95": iou["null_q95"],
                    })
                    log(f"    {cname:14} lift={res['lift']:.3f} (p={res['p']:.4f})  "
                        f"остаток={res['lift_resid']:.3f} (p={res['p_resid']:.4f})  "
                        f"IoU={iou['observed']:.3f} (p={iou['p']:.4f}, "
                        f"null q95 {iou['null_q95']:.3f})")
                crit[(pname, op, tname)] = (az_ref, az_mode, d_az, tuple(planks))

    out = pd.DataFrame(rows)
    mdir = ROOT / "outputs" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(mdir / "fault_grouping_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(zone_rows).to_csv(mdir / "fault_grouping_zones.csv",
                                   index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 92}\nИТОГ (alpha = {alpha:.5f}; критерий см. config.FG_PRIMARY)\n{'=' * 92}")
    for (pname, op, tname), (az_ref, az_mode, d_az, planks) in crit.items():
        sub = out[(out["популяция"] == pname) & (out["оператор"] == op)
                  & (out["цель"] == tname)]
        print(f"\n{pname} / оператор {op} / цель {tname}")
        print(f"{'карта':<15}{'площадь':>9}{'lift':>7}{'p':>9}{'ост.':>8}{'p ост.':>9}"
              f"{'IoU':>7}{'p IoU':>9}")
        for _, r in sub.iterrows():
            print(f"{r['карта']:<15}{r['площадь_маски']:>8.1%}{r['lift']:>7.3f}"
                  f"{r['p']:>9.4f}{r['lift_resid']:>8.3f}{r['p_resid']:>9.4f}"
                  f"{r['IoU']:>7.3f}{r['p_IoU']:>9.4f}")

        # Вердикт по пре-регистрированному критерию, а не по одному p: цель
        # занимает больше трети пула, и отдельно взятая значимость тут дёшева.
        cand = sub[sub["карта"] == "зоны (метод)"].iloc[0]
        base_planks = sub[sub["карта"].isin(planks)]
        c1 = d_az <= config.FG_AZ_BANDWIDTH_DEG
        c2 = min(cand["p"], cand["p_IoU"]) < alpha
        c3 = bool((cand["IoU"] > base_planks["IoU"]).all())
        print(f"  критерий FG_PRIMARY: "
              f"(1) мода {az_mode:.1f}° при карте {az_ref:.1f}°, |Δ| = {d_az:.1f}° "
              f"<= {config.FG_AZ_BANDWIDTH_DEG:.0f}° — {'да' if c1 else 'НЕТ'}; "
              f"(2) p < alpha — {'да' if c2 else 'НЕТ'}; "
              f"(3) IoU выше планок без группировки "
              f"({cand['IoU']:.3f} против {base_planks['IoU'].max():.3f}) — "
              f"{'да' if c3 else 'НЕТ'}")
        print(f"  ВЕРДИКТ: {'ВОСПРОИЗВОДИТ' if (c1 and c2 and c3) else 'НЕ ВОСПРОИЗВОДИТ'}")

    log(f"\nсохранено: {mdir / 'fault_grouping_summary.csv'}")
    for pname in populations:
        seg, _, _ = (_dyke_population(layers) if pname == "дайки"
                     else _lineament_population(meta))
        _figure(meta, df, layers, seg, pname, operators, log)


def _figure(meta, df, layers, seg, pname, operators, log) -> None:
    """Зоны популяции против коридоров карты: строка на оператор, столбец на цель."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(len(operators), len(TARGETS),
                             figsize=(6.2 * len(TARGETS), 6.4 * len(operators)),
                             squeeze=False)
    X, Y = meta.cell_centers()
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    for row, op in zip(axes, operators):
        for ax, (tname, (role, dist_col)) in zip(row, TARGETS.items()):
            az_ref = _target_azimuth(fg.axial_segments(layers[role].geometry))
            corridor = (df[dist_col].to_numpy(dtype=float) <= 0).reshape(meta.shape)
            _, mask, picked, az_mode = _zone_maps(seg, az_ref, meta, operator=op)

            # Однотонные слои, а не градиентные палитры: величина здесь бинарная
            # («внутри / снаружи»), и цветовая шкала внушала бы несуществующий
            # порядок, а на просвет давала бы третий цвет вместо перекрытия.
            ax.imshow(np.where(corridor, 1.0, np.nan),
                      cmap=ListedColormap(["#9e9e9e"]), vmin=0, vmax=1,
                      extent=extent, origin="upper")
            ax.imshow(np.where(mask, 1.0, np.nan), cmap=ListedColormap(["#d62728"]),
                      alpha=0.40, vmin=0, vmax=1, extent=extent, origin="upper")
            ax.plot(*_object_lines(seg), color="k", lw=0.5)
            ax.set_title(f"{op} · {tname}: коридор карты (серое) против зон (красное)\n"
                         f"азимут карты {az_ref:.1f}°, мода {az_mode:.1f}°, "
                         f"зон {len(picked)}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Задача №3: группировка «{pname}» в зоны против разломных зон карты",
                 fontsize=11)
    out_png = ROOT / "outputs" / f"fault_grouping_{'dykes' if pname == 'дайки' else 'lin'}.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"карта: {out_png}")


def _object_lines(seg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Оси объектов одним разорванным полилинием (NaN между отрезками)."""
    a = np.radians(seg["az_deg"].to_numpy())
    half = 0.5 * seg["length_m"].to_numpy()
    x, y = seg["x"].to_numpy(), seg["y"].to_numpy()
    dx, dy = half * np.sin(a), half * np.cos(a)
    xs = np.column_stack([x - dx, x + dx, np.full_like(x, np.nan)]).ravel()
    ys = np.column_stack([y - dy, y + dy, np.full_like(y, np.nan)]).ravel()
    return xs, ys


if __name__ == "__main__":
    main()
