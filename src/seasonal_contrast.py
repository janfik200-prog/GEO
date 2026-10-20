"""Контраст коридоров разломов на снимках разных сезонов.

Третий прибор семейства шага 0, и он меряет ДРУГОЙ сомножитель, чем два первых.

Шаг 0 (:mod:`src.detection_limit`) и накопление вдоль линии
(:mod:`src.line_accumulation`) двигали ПОРОГ обнаружения: 1.48 -> 0.60 sigma. Там
они и упёрлись — чтобы дотянуть порог до реального контраста коридоров
0.194 sigma, нужен прямолинейный разлом длиной около 95 км при поперечнике листа
77 км. Ограничение геометрическое, детектором его не снять. Остаётся второй
сомножитель: САМ КОНТРАСТ. Его и меряет этот модуль — той же функцией
:func:`src.detection_limit.corridor_contrast`, но на композитах других сезонов.

ПОЧЕМУ СЕЗОН МОЖЕТ ЧТО-ТО ДАТЬ. Все композиты проекта — медианы за июль-август,
то есть снимки сплошного растительного покрова (98.3% пикселей листа — трава).
Всё, что живёт неделями и связано с водой, такая медиана стирает по построению.
А в мерзлоте разлом проявляется именно так: зона трещиноватости — это зона
проницаемости, по ней разгружаются подземные воды, зимой они намерзают наледями,
весной там иначе лежит и сходит снег. Лёд против тундры — это не 0.19 sigma.

ПОЧЕМУ РАДАР ОСНОВНОЙ. Оба оптических конвейера проекта бракуют снег по
построению: у Sentinel-2 класс 11 не входит в ``S2_SCL_KEEP``, у Landsat бит 5
входит в ``L8_QA_REJECT_BITS``. На 71 градусе с.ш. это выбрасывает февраль-май
целиком. Sentinel-1 RTC не зависит ни от освещения, ни от облаков, и над листом
даёт 63-85 сцен в КАЖДОМ окне (зонд каталога 26.08.2026). Landsat участвует
только в весеннем окне и только с ослабленной маской
(``config.SEA_L8_QA_REJECT_BITS``, снег сохранён).

ЭТО ОПИСАНИЕ, А НЕ ПРОВЕРКА ГИПОТЕЗЫ. Ни одного p-value, ни одной alpha;
знаменатель Бонферрони НЕ растёт, ровно как у шага 0. :func:`shift_reference`
даёт ШКАЛУ — какой контраст получается, если тот же снимок сдвинут относительно
коридоров, — чтобы число «0.3 sigma» можно было прочитать. Это не тест, и
называть его тестом нельзя: сравнение с квантилем сдвигов здесь служит
единицей измерения, а не решающим правилом.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ЗАКРЫТОЙ ЗАДАЧИ №2C. Там (:mod:`src.snowmelt`) мерились
СРОКИ схода снега как признак и проверялись гипотезой против карты разломов
(20 тестов в ledger, лучший остаточный lift 1.359). Здесь другая величина
(контраст, а не срок) и другой протокол (описание, а не гипотеза). Тот прогон,
кстати, и подсказал это направление: он показал, что размах срока схода по листу
всего 5.3 дня — то есть режим таяния листом почти не различается, а вопрос, есть
ли разница в самой ЯРКОСТИ зимой и весной, остался незаданным.
"""
from __future__ import annotations

import numpy as np

from . import config, detection_limit as dl, integro_grid, stac_grid

#: Какие скалярные поля поддержаны у какого сенсора. Реестр, а не проверка по
#: месту: ``config.SEA_CANDIDATES`` пре-регистрирован, и опечатка в нём обязана
#: падать до скачивания сцен, а не после сведения композита.
SENSOR_FIELDS: dict[str, tuple[str, ...]] = {
    "s1": ("vv_db",),
    "l8": ("red", "ndwi"),
}


def window_months(window: str) -> tuple[int, ...]:
    """Месяцы сезонного окна по имени; ``KeyError`` вместо тихого пустого списка."""
    if window not in config.SEA_WINDOWS:
        raise ValueError(f"неизвестное окно {window!r}, "
                         f"допустимы {tuple(config.SEA_WINDOWS)}")
    return config.SEA_WINDOWS[window]


def window_items(kind: str, meta: integro_grid.GridMeta, window: str) -> list:
    """Сцены сенсора над листом в заданном сезонном окне.

    Квота на группу здесь ВЫШЕ штатной (``config.SEA_MAX_PER_ORBIT``): летняя
    квота ограничивает разнообразие ракурса ради текстурных задач, а тут считается
    медиана яркости на террейн-корректированных данных, и лишняя сцена только
    достраивает пропуски покрытия.
    """
    months = window_months(window)
    if kind == "s1":
        return stac_grid.search(config.S1_COLLECTION, meta, config.SEA_YEARS,
                                months, group_key="sat:relative_orbit",
                                max_per_group=config.SEA_MAX_PER_ORBIT)
    if kind == "l8":
        # У Landsat квота ШТАТНАЯ, а не поднятая. Поднимать её у Sentinel-1
        # заставляло покрытие: одна сцена RTC накрывает 10.9% листа (измерено в
        # блоке SM_*), и без запаса сцен медиана не набирается. Кадр WRS-2 —
        # 185x180 км, то есть накрывает весь лист 77x74.5 км целиком, покрытие
        # тут не ограничение. ИЗМЕРЕНО 26.08.2026: при квоте 45 весеннее окно
        # набирало 272 сцены по 7 маршрутам — часы скачивания ради выборки,
        # которая и так избыточна. Отбор внутри маршрута идёт по облачности,
        # так что штатные 15 — это 15 самых чистых сцен маршрута.
        return stac_grid.search(
            config.L8_COLLECTION, meta, config.L8_YEARS, months,
            query={"eo:cloud_cover": {"lt": config.L8_MAX_CLOUD},
                   "platform": {"in": list(config.L8_PLATFORMS)}},
            group_key="landsat:wrs_path",
            max_per_group=config.L8_MAX_SCENES_PER_PATH)
    raise ValueError(f"сенсор {kind!r} в этой ветке не поддержан "
                     "(оптические маски проекта бракуют снег, см. шапку модуля)")


def _read_scenes(items, reader, workers: int | None = None) -> tuple[list, int]:
    """Прочитать сцены, пропуская нечитаемые; вернуть ``(сцены, сколько пропущено)``.

    ИЗМЕРЕНО 26.08.2026: одна сцена RTC (S1B от 08.03.2021) отдавала
    ``RasterioIOError: not recognized as being in a supported file format`` и
    после всех повторов :func:`src.stac_grid.read_on_grid`, то есть битой была
    сама она, а не соединение. Она уронила ЦЕЛОЕ окно из 76 сцен вместе с 45
    минутами уже скачанного. Композит — это медиана, ей одна сцена из 76 не
    нужна; поэтому нечитаемая сцена выбывает, а не отменяет окно.

    Число пропущенных возвращается наружу, а не проглатывается: если их вдруг
    десятки, окно посчитано по обрезанной выборке, и это обязано быть видно.

    Сцены читаются в ``config.SEA_READ_WORKERS`` потоков — ожидание сети, не
    счёт (см. комментарий у константы). Порядок результата сохраняется входным,
    хотя медиане он и безразличен: воспроизводимость композита не должна
    зависеть от того, какая сцена в этот раз пришла первой.
    """
    from concurrent.futures import ThreadPoolExecutor

    workers = config.SEA_READ_WORKERS if workers is None else workers
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        results = [_read_one(reader, it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda it: _read_one(reader, it), items))
    scenes = [r for r in results if r is not _FAILED]
    return scenes, len(results) - len(scenes)


#: Метка нечитаемой сцены. Отдельный объект, а не ``None``: ``None`` — это
#: значение, которое читатель в принципе может вернуть, и тогда исправная сцена
#: молча попала бы в битые.
_FAILED = object()


def _read_one(reader, item):
    """Одна сцена или :data:`_FAILED`. Исключение не всплывает наружу пула."""
    try:
        return reader(item)
    except Exception:                                 # noqa: BLE001
        return _FAILED


def _require_enough(scenes: list, failed: int, items) -> None:
    """Отказаться считать окно, если после отсева сцен осталось меньше минимума.

    Терпимость к битой сцене не должна превращаться в терпимость к пустому окну:
    композит из трёх уцелевших сцен дал бы число, неотличимое по виду от
    честного, но посчитанное на другой выборке.
    """
    if len(scenes) < config.SEA_MIN_ITEMS:
        raise RuntimeError(
            f"прочитано {len(scenes)} сцен из {len(items)} (битых {failed}), "
            f"минимум {config.SEA_MIN_ITEMS} — окно не считается")


def window_field(kind: str, field: str, meta: integro_grid.GridMeta, items,
                 res_m: float | None = None, min_obs: int | None = None
                 ) -> tuple[np.ndarray, float, int]:
    """Медианный композит окна -> скалярное поле, шаг растра и число битых сцен.

    Поля:

    * ``vv_db`` (Sentinel-1) — то же γ0 VV в дБ, что во всех остальных ветках,
      чтобы сезонные числа сравнивались с летними напрямую;
    * ``red`` (Landsat) — красный канал, тот же, которым мерился летний контраст;
    * ``ndwi`` (Landsat) — ``(green - nir) / (green + nir)``. Взят не «какой-нибудь
      водный индекс», а именно тот, которым наледи выделяют по Landsat: его контур
      не зависит от наличия снега вокруг, в отличие от снегового NDSI.

    Landsat читается с ОСЛАБЛЕННОЙ маской качества (снег сохранён), иначе весеннее
    окно вымаскировалось бы целиком.
    """
    from . import sat_sources

    # Имя поля проверяется ДО всякой работы: ошибка в реестре кандидатов иначе
    # всплывала бы после скачивания и сведения десятков сцен.
    allowed = SENSOR_FIELDS.get(kind)
    if allowed is None:
        raise ValueError(f"сенсор {kind!r} в этой ветке не поддержан")
    if field not in allowed:
        raise ValueError(f"у {kind} поддержаны {allowed}, получено {field!r}")

    res_m = res_m or config.SAT_RES_M
    if kind == "s1":
        from cache_paths import S1_ANABAR

        # Только VV: VH этой ветке не нужен, а качать его — удвоить трафик на
        # четырёх окнах. Для мокрого снега и льда стандарт всё равно VV.
        bands = {"vv": sat_sources.S1_BANDS["vv"]}
        scenes, failed = _read_scenes(
            items, lambda it: stac_grid.scene_on_grid(
                it, bands, meta, S1_ANABAR, res_m=res_m, nodata=config.S1_NODATA))
        _require_enough(scenes, failed, items)
        comp, _ = stac_grid.median_stack(scenes, min_obs=min_obs)
        return stac_grid.to_db(comp["vv"]), res_m, failed

    from cache_paths import L8_ANABAR

    def mask_fn(item, m, r):
        return sat_sources.l8_mask(item, m, r,
                                   reject_bits=config.SEA_L8_QA_REJECT_BITS)

    # Только три канала из семи: red для яркости, green+nir08 для NDWI. Полный
    # L8_BANDS означал бы вдвое больше трафика ради каналов, которые ветке не нужны.
    bands = {k: sat_sources.L8_BANDS[k] for k in ("red", "green", "nir08")}
    scenes, failed = _read_scenes(
        items, lambda it: stac_grid.scene_on_grid(it, bands, meta, L8_ANABAR,
                                                  res_m=res_m, mask_fn=mask_fn))
    _require_enough(scenes, failed, items)
    comp, _ = stac_grid.median_stack(scenes, min_obs=min_obs)
    if field == "red":
        return comp["red"], res_m, failed
    green, nir = comp["green"], comp["nir08"]
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = (green - nir) / (green + nir)
    return np.where(np.isfinite(nd), nd, np.nan), res_m, failed


def cells_from_image(img: np.ndarray, meta: integro_grid.GridMeta, res_m: float,
                     pool: np.ndarray) -> np.ndarray:
    """Растр -> значения на ячейках сетки, вне пула — NaN.

    Пул зануляется ЗДЕСЬ, до всякого сдвига: у сдвиговой шкалы невалидные ячейки
    обязаны ехать вместе с картой, иначе сдвинутый снимок сравнивался бы с
    коридором на другом наборе ячеек (та же логика, что в
    :func:`src.assessment.spatial_null_pvalue`).
    """
    cells = stac_grid.to_cells({"gray": img}, meta, res_m=res_m, prefix="sea_")
    values = cells["sea_gray"].to_numpy(dtype=float)
    keep = np.zeros(values.shape, dtype=bool)
    keep[np.asarray(pool, dtype=int)] = True
    return np.where(keep, values, np.nan)


def shift_reference(values: np.ndarray, dist_to_fault: np.ndarray, sigma: float,
                    meta: integro_grid.GridMeta, n_shifts: int | None = None,
                    seed: int | None = None, **contrast_kw) -> np.ndarray:
    """Контраст того же снимка, ЦИКЛИЧЕСКИ СДВИНУТОГО относительно коридоров.

    Это ШКАЛА, а не тест: у широкого коридора против кольца контраст не нулевой
    сам по себе (лист неоднороден), и без этой величины число «0.3 sigma» нечем
    прочитать. Двигается СНИМОК, коридор стоит — так сохраняется пространственная
    автокорреляция снимка, в отличие от перемешивания ячеек.

    К сдвигу добавляются случайные отражения по обеим осям: чистый тор оставляет
    крупный градиент яркости листа приблизительно на месте, и без отражений шкала
    оказалась бы уже реальной изменчивости.
    """
    n_shifts = config.SEA_N_SHIFTS if n_shifts is None else n_shifts
    rng = np.random.default_rng(config.SEA_SEED if seed is None else seed)
    img = np.asarray(values, dtype=float).reshape(meta.prf, meta.pic)
    out = np.empty(n_shifts, dtype=float)
    for k in range(n_shifts):
        sh = np.roll(img, (int(rng.integers(meta.prf)), int(rng.integers(meta.pic))),
                     axis=(0, 1))
        if rng.random() < 0.5:
            sh = sh[::-1]
        if rng.random() < 0.5:
            sh = sh[:, ::-1]
        got = dl.corridor_contrast(sh.ravel(), dist_to_fault, sigma, **contrast_kw)
        out[k] = got["contrast_median"]
    return out
