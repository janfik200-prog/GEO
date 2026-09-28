"""Кривая предела обнаружения: с какого контраста конвейер видит разлом.

Шаг 0 ветки «чистка стекла». Задачи №8 (:mod:`src.lineaments_sat`) и №2
(:mod:`src.fracture_zones`) закрылись отрицательно, но ни одна не различила два
разных диагноза:

* **разлома нет** — на листе действительно отсутствует ориентированная текстура
  на километровых масштабах, и тогда улучшать конвейер бессмысленно;
* **разлом есть, но не виден** — сигнал утоплен в шуме композита и в подмене
  рельефом, и тогда у чистки входа есть измеримый запас.

Различить их наблюдением нельзя: обе гипотезы предсказывают один и тот же
отрицательный результат. Различает контролируемая врезка — в РЕАЛЬНЫЙ композит
листа, со всей его травой, спеклом и отмывкой рельефа, вставляется синтетический
разлом известного контраста, ширины и длины, после чего тот же самый конвейер
(те же функции, не копия) спрашивается, нашёл ли он его.

ЧТО ЗДЕСЬ МЕРИТСЯ. Не «работает/не работает», а ПОРОГ: минимальная амплитуда,
с которой доля обнаружений превышает долю ложных тревог. Ложная тревога меряется
тем же прибором на той же площадке при амплитуде 0 (:data:`config.DL_AMPLITUDES`
начинается с нуля намеренно) — без этой планки число «поймано» ничего не значит,
ровно как показала задача №8 своей долей линий «без соответствия на карте».

ЕДИНИЦА АМПЛИТУДЫ — устойчивый разброс яркости самого композита
(:func:`robust_sigma`), а не дБ и не коэффициент отражения. Так контраст
сравним между четырьмя съёмками разной физики, и в тех же единицах меряется
контраст НАСТОЯЩИХ закартированных разломов (:func:`corridor_contrast`) — то
есть кривая и реальность попадают на одну ось.

ДВА ДЕТЕКТОРА, потому что закрывались две разные ветки:

* :func:`hough_response` — пороговая ветка задачи №8 (Кэнни -> вероятностный
  Хаф), обнаружение = среди отрезков есть попадающий во вставку по положению и
  азимуту с допусками :data:`config.DL_MATCH_TOL_M` / ``DL_MATCH_TOL_DEG``;
* :func:`tensor_response` — непрерывная ветка задачи №2, отклик = превышение
  медианной когерентности на следе вставки над медианой фона.

ДВА ПРОФИЛЯ (:data:`config.DL_PROFILES`), потому что детекторы реагируют на
разное: Кэнни ищет КРАЙ и ему нужен перепад (``scarp`` — уступ сброса), тензор
реагирует на согласованность направления градиента и берёт одинаково край и
жёлоб (``ridge`` — ложбина вдоль нарушения). Мерить одним профилем значило бы
подсуживать одной из веток.

СИСТЕМА УГЛОВ — растровая и ровно та же, что у :func:`src.fracture_zones
.structure_tensor`: ось задаётся углом ``a`` с направляющим вектором
``(drow, dcol) = (sin a, cos a)``, то есть ``a = arctan2(drow, dcol)``, ось col
на восток, ось row на юг. Перевод из географического азимута —
:func:`axis_from_geographic`. Одна система на модуль и на тензор выбрана
сознательно: направление вставки и направление, которое возвращает тензор,
обязаны сравниваться без пересчёта.

ЭТО ПРИБОР, А НЕ ГИПОТЕЗА. Модуль не считает ни одного p-value против карты
разломов и не касается точек заверки, поэтому в накопительный знаменатель
Бонферрони (:data:`config.FZ_N_TESTS_PRIOR`) ветка НЕ ВХОДИТ. Карта разломов
используется один раз и описательно — в :func:`corridor_contrast`, чтобы
измерить, какой контраст дают настоящие разломы, и понять, куда на кривой
попадает реальность.
"""
from __future__ import annotations

import numpy as np

from . import config


# --------------------------------------------------------------------------
# Единица амплитуды и система углов
# --------------------------------------------------------------------------

def robust_sigma(image: np.ndarray) -> float:
    """Устойчивый разброс яркости композита: ``MAD * 1.4826``.

    Устойчивый, а не обычное СКО: у композитов есть выбросы (остаточные облака у
    оптики, яркие блики у радара), и СКО меряло бы их, а не типичный разброс. На
    нормальном распределении множитель 1.4826 приводит MAD к сигме, поэтому
    «амплитуда 1.0» читается как «перепад в одну сигму фона».
    """
    v = np.asarray(image, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def _canonical_axis(deg: float) -> float:
    """Угол оси в полуинтервал ``[0, 180)`` — с защитой от заворота на границе.

    ИЗМЕРЕНО тестом: ``cos(90°)`` в double равен ``6.1e-17``, а не нулю, поэтому
    у восточного направления ``arctan2`` возвращает угол чуть МЕНЬШЕ нуля, и
    ``% 180`` заворачивает его в ровно ``180.0``. Для оси это тот же ноль
    (у оси нет начала и конца), но значение выходит за полуинтервал, в котором
    строятся розы азимутов (:data:`config.LIN_SAT_ROSE_BINS` бинов на ``[0, 180)``),
    и переполняло бы последний бин. Допуск 1e-9 градуса заведомо меньше любой
    осмысленной здесь величины: азимуты приходят целыми градусами.
    """
    a = float(deg) % 180.0
    return 0.0 if 180.0 - a < 1e-9 else a


def axis_from_geographic(azimuth_deg: float) -> float:
    """Географический азимут (от севера по часовой) -> угол оси в растре.

    Север — это направление УБЫВАНИЯ строки (ось row смотрит на юг), поэтому
    направляющий вектор географического азимута ``b`` равен
    ``(drow, dcol) = (-cos b, sin b)``, а угол оси — его ``arctan2(drow, dcol)``.
    """
    b = np.radians(azimuth_deg)
    return _canonical_axis(np.degrees(np.arctan2(-np.cos(b), np.sin(b))))


def geographic_from_axis(axis_deg: float) -> float:
    """Обратный перевод: угол оси в растре -> географический азимут (0..180)."""
    a = np.radians(axis_deg)
    return _canonical_axis(np.degrees(np.arctan2(np.cos(a), -np.sin(a))))


def axis_difference_deg(a: float, b: float) -> float:
    """Кратчайший угол между двумя ОСЯМИ (не векторами): 0..90 градусов.

    У оси нет начала и конца — 10° и 190° это одно направление, — поэтому
    разность берётся по модулю 180 и сворачивается в 0..90. Та же логика, что у
    удвоенного угла в :func:`src.fracture_zones.structure_tensor`.
    """
    d = abs(float(a) - float(b)) % 180.0
    return float(min(d, 180.0 - d))


# --------------------------------------------------------------------------
# Геометрия вставки
# --------------------------------------------------------------------------

def track_coords(shape: tuple[int, int], center_rc: tuple[float, float],
                 axis_deg: float, res_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Координаты вдоль (``s``) и поперёк (``t``) оси вставки, в метрах.

    ``s`` отсчитывается от центра вставки вдоль её направляющего вектора,
    ``t`` — по нормали к нему; знак ``t`` различает борта, что нужно уступу.
    """
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    dr = (rr - center_rc[0]) * res_m
    dc = (cc - center_rc[1]) * res_m
    a = np.radians(axis_deg)
    u_r, u_c = np.sin(a), np.cos(a)          # направляющий вектор оси
    n_r, n_c = np.cos(a), -np.sin(a)         # нормаль (u . n = 0)
    return dr * u_r + dc * u_c, dr * n_r + dc * n_c


def fault_masks(shape: tuple[int, int], center_rc: tuple[float, float],
                axis_deg: float, length_m: float, width_m: float,
                res_m: float, background_pad_m: float | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """След вставки и фон вокруг неё: ``(line, background)``.

    ``line`` — пиксели самого объекта (поперёк не дальше полуширины плюс пиксель,
    вдоль — не дальше половины длины). ``background`` — пиксели дальше
    :data:`config.DL_BACKGROUND_PAD_M` от ОТРЕЗКА (а не от его центра): ближе
    фон содержал бы размазанный гауссианой окна отклик самой вставки и занижал
    бы контраст против самого себя.
    """
    pad = config.DL_BACKGROUND_PAD_M if background_pad_m is None else background_pad_m
    s, t = track_coords(shape, center_rc, axis_deg, res_m)
    half = length_m / 2.0
    line = (np.abs(t) <= width_m / 2.0 + res_m) & (np.abs(s) <= half)
    # Расстояние до ОТРЕЗКА: вдоль оси за его концами оно растёт, внутри — это
    # просто |t|.
    dist = np.hypot(np.clip(np.abs(s) - half, 0.0, None), t)
    return line, dist > pad


def insert_fault(image: np.ndarray, center_rc: tuple[float, float],
                 axis_deg: float, *, length_m: float, width_m: float,
                 amplitude: float, sigma: float, res_m: float,
                 profile: str = "scarp") -> np.ndarray:
    """Врезка синтетического разлома в реальный композит.

    ``amplitude`` — в долях ``sigma`` (:func:`robust_sigma`), а не в единицах
    источника. Профили поперёк оси:

    * ``scarp`` — уступ: сглаженная ступень ``tanh``, полный перепад между
      бортами равен ``amplitude * sigma``. Это то, что даёт настоящий сброс, и
      единственное, на что вообще способен реагировать Кэнни как детектор края;
    * ``ridge`` — жёлоб/гребень: гауссиана с пиком ``amplitude * sigma`` и
      сигмой в полуширину, то есть на ``+-width/2`` спад в ``e^-0.5``.

    Вдоль оси объект гасится за концами гауссовым хвостом с масштабом в
    собственную ширину: обрубленный край дал бы поперечную ступень на торце,
    то есть паразитный второй объект под 90° к вставленному.
    """
    if profile not in config.DL_PROFILES:
        raise ValueError(f"неизвестный профиль {profile!r}, "
                         f"допустимы {config.DL_PROFILES}")
    img = np.asarray(image, dtype=float)
    if amplitude == 0.0:
        return img.copy()

    s, t = track_coords(img.shape, center_rc, axis_deg, res_m)
    half_w = max(width_m / 2.0, res_m / 2.0)
    if profile == "scarp":
        cross = 0.5 * np.tanh(t / half_w)
    else:
        cross = np.exp(-0.5 * (t / half_w) ** 2)
    over = np.clip(np.abs(s) - length_m / 2.0, 0.0, None)
    taper = np.exp(-0.5 * (over / max(width_m, res_m)) ** 2)
    return img + amplitude * sigma * cross * taper


def cut_window(image: np.ndarray, center_rc: tuple[int, int], length_m: float,
               res_m: float, pad_m: float | None = None
               ) -> tuple[np.ndarray, tuple[float, float]]:
    """Вырезка вокруг врезки и положение её центра внутри вырезки.

    Оба детектора локальны (Кэнни локален, окно тензора не больше
    ``FZ_SCALES_M[-1]``), поэтому мерить на вырезке законно и в десятки раз
    дешевле, чем гонять весь лист. Полуразмер — половина длины плюс
    :data:`config.DL_WINDOW_PAD_M`, чтобы за отступом фона осталось ещё столько
    же чистого фона на оценку медианы.
    """
    pad = config.DL_WINDOW_PAD_M if pad_m is None else pad_m
    half_px = int(np.ceil((length_m / 2.0 + pad) / res_m))
    r0 = max(0, int(center_rc[0]) - half_px)
    c0 = max(0, int(center_rc[1]) - half_px)
    r1 = min(image.shape[0], int(center_rc[0]) + half_px + 1)
    c1 = min(image.shape[1], int(center_rc[1]) + half_px + 1)
    return image[r0:r1, c0:c1], (center_rc[0] - r0, center_rc[1] - c0)


def sample_centers(image: np.ndarray, length_m: float, res_m: float, n: int,
                   rng: np.random.Generator, pad_m: float | None = None,
                   min_finite: float = 0.5) -> list[tuple[int, int]]:
    """Случайные центры врезок: вырезка целиком внутри растра и не пустая.

    ``min_finite`` — минимальная доля непропущенных пикселей вырезки. Врезка в
    угол, где композит наполовину NaN, мерила бы не обнаружимость, а покрытие
    съёмкой.
    """
    pad = config.DL_WINDOW_PAD_M if pad_m is None else pad_m
    half_px = int(np.ceil((length_m / 2.0 + pad) / res_m))
    rows, cols = image.shape
    if rows <= 2 * half_px + 1 or cols <= 2 * half_px + 1:
        raise ValueError(
            f"растр {rows}x{cols} меньше окна врезки {2 * half_px + 1} px: "
            "уменьшите length_m или pad_m")
    out: list[tuple[int, int]] = []
    # Запас попыток: часть площадок отбраковывается по доле NaN, и без потолка
    # цикл на пустом композите не кончился бы никогда.
    for _ in range(50 * max(1, n)):
        if len(out) >= n:
            break
        r = int(rng.integers(half_px, rows - half_px))
        c = int(rng.integers(half_px, cols - half_px))
        win, _ = cut_window(image, (r, c), length_m, res_m, pad_m=pad)
        if np.isfinite(win).mean() >= min_finite:
            out.append((r, c))
    return out


# --------------------------------------------------------------------------
# Детекторы — те же функции, что в закрывшихся ветках
# --------------------------------------------------------------------------

def coherence_fields(image: np.ndarray, res_m: float,
                     scales_m: tuple[float, ...] | None = None
                     ) -> dict[str, np.ndarray]:
    """Поля когерентности по масштабам: ``{'coh_300': ..., 'coh_1km': ...}``.

    Отдельно от :func:`tensor_response` ради нуль-модели: при амплитуде 0 врезка
    не меняет картинку (:func:`insert_fault` там точный no-op), поэтому поле на
    площадке одно на все ширины, и пересчитывать его по разу на ширину значило бы
    вчетверо переплатить за самую массовую часть свипа.
    """
    from . import fracture_zones as fz

    scales_m = scales_m or config.FZ_SCALES_M
    return {f"coh_{fz._scale_tag(s)}": fz.structure_tensor(image, s, res_m=res_m)["coh"]
            for s in scales_m}


def tensor_response(image: np.ndarray, center_rc: tuple[float, float],
                    axis_deg: float, *, length_m: float, width_m: float,
                    res_m: float, scales_m: tuple[float, ...] | None = None,
                    fields: dict[str, np.ndarray] | None = None
                    ) -> dict[str, float]:
    """Отклик непрерывной ветки: превышение когерентности на следе вставки над фоном.

    По масштабу на число: ``coh_<масштаб>`` = медиана когерентности на следе
    минус медиана по фону. Медиана, а не среднее, по той же причине, по которой
    единица амплитуды устойчивая, — у когерентности тяжёлый правый хвост.

    Отклик меряется на СОБСТВЕННОЙ ширине объекта на всех масштабах окна.
    Верхние масштабы размазывают его и занижают — это не артефакт замера, а
    ровно то свойство детектора, которое кривая и должна показать.

    ``fields`` — уже посчитанные поля :func:`coherence_fields` для ЭТОГО же
    изображения; передаются только ради экономии в нуль-модели, на результат не
    влияют.
    """
    fields = coherence_fields(image, res_m, scales_m) if fields is None else fields
    line, back = fault_masks(image.shape, center_rc, axis_deg, length_m,
                             width_m, res_m)
    out: dict[str, float] = {}
    for key, coh in fields.items():
        on = coh[line & np.isfinite(coh)]
        off = coh[back & np.isfinite(coh)]
        out[key] = (float(np.median(on) - np.median(off))
                    if on.size and off.size else float("nan"))
    return out


def hough_response(image: np.ndarray, center_rc: tuple[float, float],
                   axis_deg: float, *, length_m: float, res_m: float,
                   tol_m: float | None = None, tol_deg: float | None = None
                   ) -> dict[str, float]:
    """Отклик пороговой ветки: есть ли среди отрезков Хафа попадающий во вставку.

    Отрезок считается попавшим, когда выполнены оба условия сразу: его ось
    отличается от оси вставки не больше чем на ``tol_deg``, и его середина
    отстоит от ОТРЕЗКА вставки не дальше ``tol_m`` поперёк и не дальше
    ``половина длины + tol_m`` вдоль. Второе условие нужно, чтобы отрезок на
    продолжении вставки, за много километров от неё, не засчитывался попаданием.

    Возвращает число попавших отрезков и общее число найденных — второе нужно
    как контекст: обнаружение среди трёх отрезков и среди трёхсот стоит разного.
    """
    from . import lineaments_sat

    tol_m = config.DL_MATCH_TOL_M if tol_m is None else tol_m
    tol_deg = config.DL_MATCH_TOL_DEG if tol_deg is None else tol_deg
    segments = lineaments_sat.lines_from_image(image)
    half = length_m / 2.0
    a = np.radians(axis_deg)
    u_r, u_c = np.sin(a), np.cos(a)
    n_r, n_c = np.cos(a), -np.sin(a)

    n_match = 0
    for (x0, y0), (x1, y1) in segments:
        seg_axis = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180.0
        if axis_difference_deg(seg_axis, axis_deg) > tol_deg:
            continue
        dr = ((y0 + y1) / 2.0 - center_rc[0]) * res_m
        dc = ((x0 + x1) / 2.0 - center_rc[1]) * res_m
        s = dr * u_r + dc * u_c
        t = dr * n_r + dc * n_c
        if abs(t) <= tol_m and abs(s) <= half + tol_m:
            n_match += 1
    return {"hough_n_match": float(n_match),
            "hough_n_segments": float(len(segments))}


def prepare_probe(image: np.ndarray, center_rc: tuple[int, int], axis_deg: float,
                  *, amplitude: float, length_m: float, width_m: float,
                  res_m: float, sigma: float, profile: str = "scarp"
                  ) -> tuple[np.ndarray, tuple[float, float]]:
    """Вырезка вокруг площадки со вставленным разломом и центр вставки внутри неё.

    Вырезка делается ДО вставки, а сигма берётся снаружи (по всему композиту): и
    то и другое ради сопоставимости площадок между собой. Считай сигму по
    вырезке — амплитуда «1.0» означала бы на спокойном участке и на изрезанном
    физически разный перепад, и кривая смешивала бы два эффекта.

    Вынесено из :func:`probe_once` отдельной функцией, чтобы новые детекторы
    (:mod:`src.line_accumulation`) мерились на ТОМ ЖЕ окне с той же вставкой, а
    не на своей копии: иначе сравнение порогов между детекторами включало бы
    разницу площадок.
    """
    win, ctr = cut_window(image, center_rc, length_m, res_m)
    hit = insert_fault(win, ctr, axis_deg, length_m=length_m, width_m=width_m,
                       amplitude=amplitude, sigma=sigma, res_m=res_m,
                       profile=profile)
    return hit, ctr


def probe_once(image: np.ndarray, center_rc: tuple[int, int], axis_deg: float, *,
               amplitude: float, length_m: float, width_m: float, res_m: float,
               sigma: float, profile: str = "scarp",
               scales_m: tuple[float, ...] | None = None) -> dict[str, float]:
    """Одна врезка: вырезка -> вставка -> оба детектора -> строка откликов."""
    hit, ctr = prepare_probe(image, center_rc, axis_deg, amplitude=amplitude,
                             length_m=length_m, width_m=width_m, res_m=res_m,
                             sigma=sigma, profile=profile)
    row = tensor_response(hit, ctr, axis_deg, length_m=length_m,
                          width_m=width_m, res_m=res_m, scales_m=scales_m)
    row.update(hough_response(hit, ctr, axis_deg, length_m=length_m,
                              res_m=res_m))
    return row


# --------------------------------------------------------------------------
# Порог обнаружения против нуль-модели
# --------------------------------------------------------------------------

def detection_threshold(null_values: np.ndarray, q: float | None = None) -> float:
    """Порог отклика по нуль-модели: квантиль откликов при нулевой амплитуде.

    Нуль-модель — те же врезки на тех же площадках, но без сигнала
    (:data:`config.DL_AMPLITUDES` начинается с 0.0). Её квантиль и есть уровень
    ложной тревоги конвейера: у Хафа на реальном композите всегда найдётся
    отрезок нужного азимута рядом с любой точкой, и без этой планки доля
    «поймано» читалась бы как успех.
    """
    q = config.DL_NULL_Q if q is None else q
    v = np.asarray(null_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.quantile(v, q))


def detection_rate(values: np.ndarray, threshold: float) -> float:
    """Доля врезок, чей отклик строго превысил порог нуль-модели."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(threshold):
        return float("nan")
    return float((v > threshold).mean())


def limit_from_curve(amplitudes: np.ndarray, rates: np.ndarray,
                     target_rate: float = 0.5) -> float:
    """Предел обнаружения: амплитуда, с которой доля обнаружений берёт ``target_rate``.

    Линейная интерполяция между узлами свипа — сетка амплитуд грубая
    (:data:`config.DL_AMPLITUDES`), и возвращать ближайший узел значило бы
    округлять предел до множителя 2. ``inf``, если планка не взята нигде: это
    осмысленный ответ «в пределах шкалы не обнаруживается», а не ошибка.
    """
    a = np.asarray(amplitudes, dtype=float)
    r = np.asarray(rates, dtype=float)
    order = np.argsort(a)
    a, r = a[order], r[order]
    ok = np.isfinite(r)
    a, r = a[ok], r[ok]
    hit = np.where(r >= target_rate)[0]
    if hit.size == 0:
        return float("inf")
    i = int(hit[0])
    if i == 0:
        return float(a[0])
    r0, r1 = r[i - 1], r[i]
    if r1 == r0:
        return float(a[i])
    return float(a[i - 1] + (target_rate - r0) * (a[i] - a[i - 1]) / (r1 - r0))


# --------------------------------------------------------------------------
# Калибровка: контраст НАСТОЯЩИХ разломов в тех же единицах
# --------------------------------------------------------------------------

def corridor_contrast(values: np.ndarray, dist_to_fault: np.ndarray,
                      sigma: float, corridor_m: float = 0.0,
                      ring_m: tuple[float, float] = (2000.0, 6000.0)
                      ) -> dict[str, float]:
    """Контраст закартированных разломов в единицах амплитуды свипа.

    Единственное место ветки, где карта разломов вообще используется, и
    используется ОПИСАТЕЛЬНО: никакого p-value, никакого отбора. Вопрос один —
    какой перепад яркости дают настоящие разломы, то есть куда на кривой
    предела обнаружения попадает реальность.

    Считаются два числа, потому что двум профилям врезки отвечает разное:

    * ``contrast_median`` — модуль разности медиан «внутри коридора» и «в кольце
      вокруг него», делённый на ``sigma``. Это прямой аналог профиля ``ridge``:
      насколько коридор в среднем темнее или светлее окружения;
    * ``contrast_spread`` — разность межквартильных размахов там же, делённая на
      ``sigma``. Аналог профиля ``scarp``: у уступа медиана внутри коридора не
      сдвигается (коридор накрывает оба борта), а разброс внутри растёт.

    Кольцо, а не «весь остальной лист»: разломы на щите не разбросаны случайно, и
    сравнение с далёким фоном мерило бы региональный тренд яркости, а не перепад
    на нарушении.
    """
    v = np.asarray(values, dtype=float)
    d = np.asarray(dist_to_fault, dtype=float)
    ok = np.isfinite(v) & np.isfinite(d)
    inside = ok & (d <= corridor_m)
    ring = ok & (d > ring_m[0]) & (d <= ring_m[1])
    if inside.sum() < 2 or ring.sum() < 2 or not np.isfinite(sigma) or sigma <= 0:
        return {"contrast_median": float("nan"), "contrast_spread": float("nan"),
                "n_inside": float(inside.sum()), "n_ring": float(ring.sum())}

    def iqr(x: np.ndarray) -> float:
        q1, q3 = np.percentile(x, [25, 75])
        return float(q3 - q1)

    return {
        "contrast_median": float(abs(np.median(v[inside]) - np.median(v[ring])) / sigma),
        "contrast_spread": float((iqr(v[inside]) - iqr(v[ring])) / sigma),
        "n_inside": float(inside.sum()), "n_ring": float(ring.sum()),
    }
