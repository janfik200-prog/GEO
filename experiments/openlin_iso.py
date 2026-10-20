"""Генератор линий без встроенной прямизны.

ОТКУДА ПОСТАНОВКА. SHAP-разбор отбора линий показал, что длинная прямая линия
считалась скорее артефактом. Причина оказалась не в классификаторе, а в самих
линиях: AUC одного признака «настоящая против подделки» у всех четырёх прежних
генераторов (old и hessian, по компонентам и по сырым каналам) на обеих
площадках — 0.46-0.54 для длины и 0.46-0.53 для прямолинейности. Из шума
получаются линии той же длины и прямизны, что из настоящего снимка.

Прямизну закладывает ядро оператора. У всех прежних оно вытянутое: окно границы
2 км x 200 м, ориентированный гессиан, накопление вдоль прямой. Такое ядро
делает прямую линию из чего угодно, и никакой классификатор уже не отличит по
прямизне настоящее от ложного.

ЧТО ЗДЕСЬ. Отклик считается ИЗОТРОПНЫМ ядром на масштабе 1-2 ячеек (10-20 м), то
есть без всякого предпочтения направления и длины.

ТРАССИРОВКА: ПОЧЕМУ ДВЕ ВЕРСИИ. Первая версия («порог -> скелет -> ветви»)
прошла оба условия, но рвала непрерывные объекты. Это заметил глазом
пользователь, а замер на руслах (про них точно известно, что они сплошные)
показал, где именно: в отклике русло покрыто на 71 % кусками по километру,
после скелета — 48 % кусками по 230 м, после разбиения по узлам на одном русле
674 ветви, после отсева коротких остаётся 9 %. Виноваты три вещи:

* скелет строится по порогу-«пятну», а у русла на снегу две полосы (светлое
  дно и тёмный берег) — скелет ложится мимо осевой и обрастает отростками;
* ветви резались в каждом узле без сшивки;
* отростки срезались до 1000 м — настройка под сетку 100 м, на шаге 10 м это
  сто ячеек, то есть половина всего, что висит на узле.

Вторая версия («осевая») берёт осевую линию подавлением немаксимумов поперёк
гребня (по собственному вектору гессиана), гистерезис по 8-связности, срезает
только отростки в несколько ячеек и сшивает через узлы по наименьшему
повороту. Последнее — правило ЛОКАЛЬНОЙ гладкости (поворот до 35°): плавный
меандр оно проходит, глобальную прямизну не навязывает. Навязало бы — первое
условие это покажет.

КРИТЕРИИ ОБЪЯВЛЕНЫ ДО ПРОГОНА, см. константы ниже. Проверка на гидросеть
печатается для сведения, но условием не служит: заранее её как порог не
объявляли, а добавлять условие задним числом нельзя. Русла используются только
как объект с заведомо сплошной формой — для проверки непрерывности трассировки,
а не как разметка линеаментов.

Запуск из корня репозитория:
    python -m experiments.openlin_iso --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_lines import (JOIN_ANGLE_DEG, _axis, _branches,
                                       _components, _degree,
                                       _join_through_nodes, _path_length,
                                       _ridge_nms)

COMP = Path("datacache/s2_lowsun/composite")
M = Path("outputs/metrics")

# ---- пре-регистрация (21.09.2026, до первого прогона) ----------------------
#: Условие 1: прямизна различает настоящие линии и линии с подделки.
MIN_AUC_STRAIGHT = 0.60
#: Условие 2: линии переживают пересэмплирование входа...
MIN_COVER = 0.50
#: ...и переживают его заметно лучше, чем случайно совпала бы сдвинутая копия.
MIN_COVER_OVER_NULL = 2.0
# ---- дополнение (21.09.2026, после замера разрывов, ДО прогона осевой) -----
#: Условие 3: трассировка не теряет непрерывность, которая есть в отклике. В
#: самом отклике русло покрыто на 71 % кусками в среднем по 966 м; после всех
#: шагов должно остаться не меньше половины русла кусками не короче километра.
MIN_RIVER_COVER = 0.50
MIN_RIVER_RUN_M = 1000.0
#: Сколько конфигураций проверено против условий 1-2 в этом модуле: гребни и
#: Canny в первом прогоне, осевая — во втором, осевая со сшивкой разрывов — в
#: третьем, осевая с широкими масштабами — в четвёртом, сшивка с опорой на
#: данные — в пятом, сшивка по порогу гистерезиса — в шестом, она же с
#: локальным порогом — в седьмом.
N_CONFIGS = 8
#: Какой бы вариант ни прошёл, окончательно он проверяется на ДРУГОМ фрагменте,
#: не участвовавшем в подборе: после пяти попыток на одном куске местности
#: риск подобрать трассировку под этот кусок уже не пренебрежим.
# ---- сшивка разрывов (21.09.2026, параметры заданы ДО прогона) -------------
#: Наибольший разрыв, который сшивается, м. Осевая линия рвётся на седловинах
#: отклика; по замеру на руслах это и есть главная потеря: куски по 100-150 м,
#: которые в сумме составляют русло, по одному не проходят фильтр длины.
LINK_GAP_M = 80.0
#: Наибольший поворот на КАЖДОМ из двух концов, градусы, — тот же, что при
#: сшивке через узлы.
LINK_ANGLE_DEG = 35.0
#: По скольким ячейкам от конца берётся его направление: 5 ячеек = 50 м.
#: ИТОГ (21.09.2026): русло 39 % кусками по 347 м (было 26 % по 267), но на
#: подделке линий 161 против 10 и AUC прямизны 0.501 — правило «продолжай
#: направление через пустоту» само рисует линии из шума.
LINK_END_PX = 5
# ---- сшивка с опорой на данные (21.09.2026, параметры заданы ДО прогона) ---
#: Разрыв сшивается, только если вдоль мостика есть хотя бы слабый гребень:
#: не меньше ``SUPPORT_FRAC`` его ячеек (с допуском в одну ячейку вбок) имеют
#: отклик выше квантиля ``SUPPORT_Q`` по всему фрагменту. На настоящем русле
#: разрыв возникает там, где отклик просел ниже порога гистерезиса или
#: седловина сбила подавление немаксимумов, — но он там есть. В шуме сплошного
#: слабого гребня вдоль нужного направления, как правило, нет. Связность тогда
#: задаёт снимок, а не правило «продолжай направление».
SUPPORT_Q = 0.75
SUPPORT_FRAC = 0.8
#: ИТОГ (21.09.2026): условие не отбросило ни одного мостика на настоящем
#: снимке (895 линий, русло 39.0 % — один в один со сшивкой вслепую). Допуск
#: в ячейку вбок и доля 80 % выполняются почти везде рядом с гребнями.
#: Замер опоры под ВСЕМИ геометрически допустимыми мостиками при этом показал,
#: что сама опора различает настоящее и подделку сильно: AUC 0.85, минимум
#: выше 90-го процентиля у 77 % настоящих мостиков и у 23 % поддельных.
# ---- сшивка по порогу гистерезиса (21.09.2026, задано ДО прогона) ----------
#: Мостик разрешён, только если ВСЕ его ячейки выше нижнего порога
#: гистерезиса ``Q_LOW``, без допуска вбок. Порог не подбирается по таблице
#: замера, а берётся тот, что уже стоит в трассировке: мостик — это участок,
#: который гистерезис сам принял бы как часть линии, если бы подавление
#: немаксимумов не выкусило его на седловине.
SUPPORT_STRICT_Q = None          # = Q_LOW, присваивается ниже
# ---------------------------------------------------------------------------

# ---- ОТЛОЖЕННЫЙ ФРАГМЕНТ (21.09.2026, зафиксировано ДО выбора места и сборки)
#: Кандидат заморожен: «осевая+сшивка по порогу» со всеми параметрами как есть.
HOLDOUT_GENERATOR = "осевая+сшивка по порогу"
#: Сколько фазовых подделок. На участке настройки у лучшего кандидата осталось
#: 18 поддельных линий, и 90-процентный интервал AUC прямизны вышел 0.45-0.68:
#: проверка потеряла мощность. Десять подделок дают порядка 180 линий.
HOLDOUT_N_FAKES = 10
#: Условие А — прямизна различает настоящее и подделку (та же планка 0.60).
HOLDOUT_MIN_AUC = 0.60
#: Условие Б — доля мусора по числу линий: среднее число линий на одну
#: подделку против числа настоящих. Прямая мера того, ради чего вводилась
#: прямизна: не рисует ли генератор линии из шума.
HOLDOUT_MAX_JUNK = 0.10
#: Условие В — устойчивость и условие Г — непрерывность русла: как было
#: (MIN_COVER, MIN_COVER_OVER_NULL, MIN_RIVER_COVER, MIN_RIVER_RUN_M), допуск
#: 30 м не меняется.
#: Правило выбора места: квадрат того же размера, целиком внутри листа, по
#: возможности дальше всего от участка настройки. Кандидаты — четыре угла листа
#: в порядке убывания удалённости; берётся первый, у которого не меньше
#: ``HOLDOUT_MIN_SCENES`` сцен с низким солнцем накрывают его целиком и не
#: меньше ``HOLDOUT_MIN_RIVER_KM`` км русел слоя ``dnl`` (без них условие Г не
#: посчитать). Место выбирается правилом, а не глазом.
HOLDOUT_MIN_SCENES = 5
HOLDOUT_MIN_RIVER_KM = 50.0
#: ИТОГ (21.09.2026). Правило выбрало юго-западный угол, центр (660, 110),
#: 44 км от участка настройки (самый дальний угол не накрыт ни одной сценой).
#: Те же 9 дат, что на участке настройки. Линий 493, на подделку 9.3.
#:   А) AUC прямизны 0.512, 90 % интервал 0.457-0.567 — НЕ пройдено. Мощность
#:      на этот раз есть (93 поддельные линии), интервал исключает 0.60:
#:      прямизна наших линий не отличается от прямизны линий из шума.
#:      0.66-0.73 на участке настройки не перенеслись.
#:   Б) мусора по числу линий 1.9 % — пройдено.
#:   В) устойчивость 0.727 при нуле 0.027 — пройдено.
#:   Г) русло 18.5 % кусками по 193 м — НЕ пройдено.
#:   Для сведения: у воды 30.7 % линий при 4.3 % площади (7.1x).
#: Вывод: генератор надёжно даёт настоящие и устойчивые линии, но прямизна их
#: от шумовых не отличает, и больше всего их на гидросети.
# ---- ЛОКАЛЬНЫЙ ПОРОГ И ТРЕТИЙ ФРАГМЕНТ (21.09.2026, ДО правки) -------------
#: На отложенном фрагменте почти все линии ушли в контрастный расчленённый угол,
#: а тонкие чёткие линии гладкого плато остались ниже порога: квантили
#: гистерезиса брались по всему кадру, и результат зависел от того, какой самый
#: контрастный кусок попал в кадр. Теперь отклик нормируется по окрестности.
#: Окно одно, без перебора: намного шире линии (20 м) и шага долин (~1 км),
#: но уже типичного пятна одной местности (5-10 км).
LOCAL_WIN_M = 2000.0
#: Восьмая конфигурация. Отлаживается на двух уже израсходованных фрагментах
#: (участок настройки и юго-западный угол), проверяется на ТРЕТЬЕМ — выбранном
#: тем же правилом, с теми же условиями А-Г. До заморозки третий фрагмент не
#: собирается и не просматривается.
HOLDOUT2_GENERATOR = "осевая+сшивка по порогу, локально"
HOLDOUT_USED = ((660, 110),)
# ---------------------------------------------------------------------------

#: Масштабы изотропного ядра, ячейки. На 10 м это 10 и 20 м: ширина тени от
#: уступа в первые метры при солнце 8-10°.
SIGMAS = (1.0, 2.0)

#: Широкий набор: добавлены 40 и 80 м. Снежное дно русла и полосы на плечах
#: долин шириной 30-100 м, и при ядре в 10-20 м гессиан откликается не в
#: середине такой полосы, а на обеих её кромках. Одно русло превращалось в две
#: рваные кромки в 20-40 м от осевой. Ядро остаётся круглым: выбирается только
#: ширина, протяжённость и направление по-прежнему не задаются.
#: ИТОГ (21.09.2026): гипотеза не подтвердилась. Линии остались на кромках
#: полос, покрытие русла упало с 26 % до 20 %, а широкое ядро сгладило шум в
#: длинные ровные гребни: линий на подделке 173 против 10, AUC прямизны 0.519.
#: Оставлено как проваленная конфигурация — в счёт N_CONFIGS.
SIGMAS_WIDE = (1.0, 2.0, 4.0, 8.0)

#: Гистерезис по квантилям отклика: у настоящего входа и у подделки берутся
#: СВОИ квантили, поэтому линий выходит поровну и сравнивается только форма.
Q_HIGH = 0.97
Q_LOW = 0.90
SUPPORT_STRICT_Q = Q_LOW

#: Минимальная длина линии, м, и допуск совпадения при проверке устойчивости.
MIN_LEN_M = 200.0
TOL_M = 30.0

#: Отростки, которые срезаются у осевой линии, м: ступеньки и шипы в
#: несколько ячеек. Не 1000 м, как у ``openlin_lines`` — та настройка сделана
#: под шаг 100 м и на 10 м срезает половину всего, что висит на узле.
SPUR_M = 60.0


def ridge_response(img: np.ndarray, sigmas: tuple = SIGMAS) -> np.ndarray:
    """Линейность по гессиану: сильная кривизна поперёк, слабая вдоль.

    Ядро гауссово и круглое, направление не выбирается заранее. Разность
    модулей собственных чисел гасит пятна (кривизна одинакова во все стороны)
    и оставляет узкие гребни и ложбины на масштабе ядра — и только на нём,
    протяжённость ядро не задаёт.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    best = np.zeros(img.shape, dtype="float32")
    for s in sigmas:
        h = hessian_matrix(img, sigma=s, order="rc",
                           use_gaussian_derivatives=True)
        l1, l2 = hessian_matrix_eigvals(h)
        big = np.maximum(np.abs(l1), np.abs(l2))
        small = np.minimum(np.abs(l1), np.abs(l2))
        best = np.maximum(best, (s ** 2 * (big - small)).astype("float32"))
    return best


def hysteresis8(r: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Гистерезис по 8-связности.

    У ``skimage.filters.apply_hysteresis_threshold`` связность крестовая, и
    тонкая диагональная линия шириной в одну ячейку распадается на отдельные
    точки — ровно там, где она идёт наискосок.
    """
    from scipy.ndimage import label

    low = r >= lo
    lab, n = label(low, structure=np.ones((3, 3), dtype=int))
    if not n:
        return low
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[r >= hi])] = True
    keep[0] = False
    return keep[lab]


def lines_ridge(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from skimage.filters import apply_hysteresis_threshold

    r = np.where(valid, ridge_response(img), 0.0)
    v = r[valid]
    hi, lo = np.quantile(v, Q_HIGH), np.quantile(v, Q_LOW)
    return apply_hysteresis_threshold(r, lo, hi)


def lines_axial(img: np.ndarray, valid: np.ndarray,
                sigmas: tuple = SIGMAS) -> np.ndarray:
    """Осевая линия гребня: подавление немаксимумов, потом гистерезис.

    Квантили порога те же, что у первой версии, и берутся по всему отклику, а
    не по уцелевшим после подавления ячейкам — меняется только то, КАК линия
    утончается, а не СКОЛЬКО сильного отклика в неё попадает.
    """
    r = np.where(valid, ridge_response(img, sigmas), 0.0)
    v = r[valid]
    hi, lo = np.quantile(v, Q_HIGH), np.quantile(v, Q_LOW)
    _s, ridge = _ridge_nms(r, valid, 1.0)
    return hysteresis8(np.where(ridge, r, 0.0), lo, hi)


def lines_canny(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from skimage.feature import canny

    return canny(img, sigma=SIGMAS[-1], low_threshold=Q_LOW,
                 high_threshold=Q_HIGH, use_quantiles=True, mask=valid)


def _describe(pts: np.ndarray, res_m: float, extra_path_m: float = 0.0
              ) -> dict | None:
    if len(pts) < 3:
        return None
    ax, _a, _b = _axis(pts)
    proj = (pts - pts.mean(0)) @ ax
    length = float(proj.max() - proj.min()) * res_m
    if length < MIN_LEN_M:
        return None
    path = _path_length(pts, res_m) + extra_path_m
    return {"длина_км": length / 1000.0,
            "прямолинейность": min(length / path, 1.0) if path else 0,
            "pts": pts}


def to_lines(raw: np.ndarray, res_m: float) -> tuple[np.ndarray, list]:
    """Первая версия: круглое замыкание, скелет, ветви не короче порога."""
    from skimage.morphology import binary_closing, disk, skeletonize

    sk = skeletonize(binary_closing(raw, disk(1)))
    keep = np.zeros_like(sk)
    rows = []
    for pts in _branches(sk, res_m):
        d = _describe(pts, res_m)
        if d is None:
            continue
        keep[pts[:, 0], pts[:, 1]] = True
        rows.append(d)
    return keep, rows


def to_lines_joined(raw: np.ndarray, res_m: float) -> tuple[np.ndarray, list]:
    """Вторая версия: короткие отростки прочь, сшивка через узлы.

    В сшитую линию возвращаются и пиксели самих узлов. Без них в каждом месте
    сшивки остаётся дырка в одну-две ячейки, длина пути занижается, и
    прямолинейность сшитой линии завышается. Сшитых линий у настоящего снимка
    больше, чем у подделки, — значит без этой поправки завысился бы именно AUC
    прямизны, то есть проверка обманула бы сама себя.
    """
    from scipy.ndimage import binary_dilation, label
    from skimage.morphology import skeletonize

    sk = skeletonize(raw)
    n_prune = max(1, int(round(SPUR_M / res_m)))
    for _ in range(3):
        deg = _degree(sk)
        spur = np.zeros_like(sk)
        for pts in _components(sk & (deg <= 2)):
            ends = int((deg[pts[:, 0], pts[:, 1]] == 1).sum())
            if len(pts) <= n_prune and ends == 1:
                spur[pts[:, 0], pts[:, 1]] = True
        if not spur.any():
            break
        sk &= ~spur

    deg = _degree(sk)
    node = sk & (deg > 2)
    lab, n = label(sk & (deg <= 2), structure=np.ones((3, 3), dtype=int))
    segs = []
    if n:
        order = np.argsort(lab.ravel(), kind="stable")
        vals = lab.ravel()[order]
        a = np.searchsorted(vals, np.arange(1, n + 1), side="left")
        b = np.searchsorted(vals, np.arange(1, n + 1), side="right")
        segs = [np.stack(np.unravel_index(order[i:j], sk.shape), axis=1)
                for i, j in zip(a, b)]
    groups = _join_through_nodes(sk, segs, JOIN_ANGLE_DEG) if segs else []

    keep = np.zeros_like(sk)
    rows = []
    h, w = sk.shape
    for grp in groups:
        pts = np.concatenate([segs[i] for i in grp])
        if len(grp) > 1:
            r0, c0 = np.maximum(pts.min(0) - 2, 0)
            r1, c1 = np.minimum(pts.max(0) + 3, (h, w))
            box = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            box[pts[:, 0] - r0, pts[:, 1] - c0] = True
            glue = binary_dilation(box, np.ones((3, 3), dtype=bool)) & \
                node[r0:r1, c0:c1]
            add = np.argwhere(glue) + (r0, c0)
            if len(add):
                pts = np.concatenate([pts, add])
        d = _describe(pts, res_m)
        if d is None:
            continue
        keep[pts[:, 0], pts[:, 1]] = True
        rows.append(d)
    return keep, rows


def _pieces(raw: np.ndarray, res_m: float) -> list[np.ndarray]:
    """Куски после срезки отростков и сшивки через узлы — без фильтра длины."""
    from scipy.ndimage import binary_dilation, label
    from skimage.morphology import skeletonize

    sk = skeletonize(raw)
    n_prune = max(1, int(round(SPUR_M / res_m)))
    for _ in range(3):
        deg = _degree(sk)
        spur = np.zeros_like(sk)
        for pts in _components(sk & (deg <= 2)):
            ends = int((deg[pts[:, 0], pts[:, 1]] == 1).sum())
            if len(pts) <= n_prune and ends == 1:
                spur[pts[:, 0], pts[:, 1]] = True
        if not spur.any():
            break
        sk &= ~spur

    deg = _degree(sk)
    node = sk & (deg > 2)
    lab, n = label(sk & (deg <= 2), structure=np.ones((3, 3), dtype=int))
    if not n:
        return []
    order = np.argsort(lab.ravel(), kind="stable")
    vals = lab.ravel()[order]
    a = np.searchsorted(vals, np.arange(1, n + 1), side="left")
    b = np.searchsorted(vals, np.arange(1, n + 1), side="right")
    segs = [np.stack(np.unravel_index(order[i:j], sk.shape), axis=1)
            for i, j in zip(a, b)]
    out = []
    h, w = sk.shape
    for grp in _join_through_nodes(sk, segs, JOIN_ANGLE_DEG):
        pts = np.concatenate([segs[i] for i in grp])
        if len(grp) > 1:                       # вернуть пиксели узлов
            r0, c0 = np.maximum(pts.min(0) - 2, 0)
            r1, c1 = np.minimum(pts.max(0) + 3, (h, w))
            box = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            box[pts[:, 0] - r0, pts[:, 1] - c0] = True
            glue = binary_dilation(box, np.ones((3, 3), dtype=bool)) & \
                node[r0:r1, c0:c1]
            add = np.argwhere(glue) + (r0, c0)
            if len(add):
                pts = np.concatenate([pts, add])
        out.append(pts)
    return out


def link_gaps(pieces: list[np.ndarray], res_m: float,
              support: np.ndarray | None = None,
              frac: float = SUPPORT_FRAC) -> list[np.ndarray]:
    """Сшивка через разрывы по ЛОКАЛЬНОМУ направлению концов.

    ``openlin_lines._link_gaps`` здесь не годится: он проверяет соосность с
    главной осью всей ветки, а у извилистого русла эта ось не совпадает с
    направлением на конце. Правило превращается в «сшивай только прямые
    куски», то есть снова навязывает прямизну.

    Здесь направление берётся по последним ``LINK_END_PX`` ячейкам, и разрыв
    обязан продолжать выход ИЗ ОБОИХ концов с поворотом не больше
    ``LINK_ANGLE_DEG``. Пары сшиваются от ближних к дальним, каждый конец —
    не больше одного раза. Пиксели мостика входят в линию, иначе длина пути
    занизилась бы и прямолинейность завысилась.
    """
    from scipy.spatial import cKDTree
    from skimage.draw import line as draw_line

    from experiments.openlin_lines import _end_dirs

    gap_px = LINK_GAP_M / res_m
    cos_lim = np.cos(np.radians(LINK_ANGLE_DEG))
    ends, dirs, owner = [], [], []
    for i, pts in enumerate(pieces):
        if len(pts) < 2:
            continue
        e, d = _end_dirs(pts, LINK_END_PX)
        for k in (0, 1):
            ends.append(e[k].astype("float64"))
            dirs.append(d[k])
            owner.append(i)
    if len(ends) < 2:
        return pieces
    ends_a = np.asarray(ends)
    tree = cKDTree(ends_a)
    cand = []
    for p, q in tree.query_pairs(gap_px):
        if owner[p] == owner[q]:
            continue
        v = ends_a[q] - ends_a[p]
        dist = float(np.hypot(*v))
        if dist < 1e-6:
            continue
        u = v / dist
        if dirs[p] @ u >= cos_lim and dirs[q] @ (-u) >= cos_lim:
            if support is not None:
                rr, cc = draw_line(int(ends_a[p][0]), int(ends_a[p][1]),
                                   int(ends_a[q][0]), int(ends_a[q][1]))
                if support[rr, cc].mean() < frac:
                    continue
            cand.append((dist, p, q))
    cand.sort()

    parent = list(range(len(pieces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used = np.zeros(len(ends), dtype=bool)
    bridges: dict[int, list[np.ndarray]] = {}
    for _dist, p, q in cand:
        if used[p] or used[q]:
            continue
        a, b = find(owner[p]), find(owner[q])
        if a == b:                          # замкнули бы кольцо
            continue
        used[p] = used[q] = True
        parent[b] = a
        rr, cc = draw_line(int(ends_a[p][0]), int(ends_a[p][1]),
                           int(ends_a[q][0]), int(ends_a[q][1]))
        bridges.setdefault(owner[p], []).append(np.stack([rr, cc], axis=1))

    groups: dict[int, list[np.ndarray]] = {}
    for i, pts in enumerate(pieces):
        groups.setdefault(find(i), []).append(pts)
        groups[find(i)].extend(bridges.get(i, []))
    return [np.unique(np.concatenate(g), axis=0) for g in groups.values()]


def to_lines_linked(raw: np.ndarray, res_m: float) -> tuple[np.ndarray, list]:
    """Третья версия: как осевая, плюс сшивка разрывов; длина — ПОСЛЕ сшивки."""
    keep = np.zeros(raw.shape, dtype=bool)
    rows = []
    for pts in link_gaps(_pieces(raw, res_m), res_m):
        d = _describe(pts, res_m)
        if d is None:
            continue
        keep[pts[:, 0], pts[:, 1]] = True
        rows.append(d)
    return keep, rows


def axial_with_support(img: np.ndarray, valid: np.ndarray) -> tuple:
    """Осевая маска и карта опоры для сшивки: где отклик выше квантиля."""
    from scipy.ndimage import maximum_filter

    r = np.where(valid, ridge_response(img), 0.0)
    thr = np.quantile(r[valid], SUPPORT_Q)
    return lines_axial(img, valid), maximum_filter(r, size=3) >= thr


def axial_with_strict_support(img: np.ndarray, valid: np.ndarray) -> tuple:
    """Осевая маска и карта опоры: отклик выше нижнего порога гистерезиса."""
    r = np.where(valid, ridge_response(img), 0.0)
    thr = np.quantile(r[valid], SUPPORT_STRICT_Q)
    return lines_axial(img, valid), r >= thr


def local_norm(r: np.ndarray, valid: np.ndarray, res_m: float,
               win_m: float = LOCAL_WIN_M) -> np.ndarray:
    """Отклик в сигмах своей окрестности.

    Нормируется логарифм отклика: сам отклик с тяжёлым хвостом, и локальные
    среднее и разброс без логарифма определялись бы парой самых сильных линий
    в окне. Среднее и разброс считаются с учётом маски, иначе край кадра
    выглядел бы как резкий спад контраста.
    """
    from scipy.ndimage import uniform_filter

    k = max(int(round(win_m / res_m)), 3)
    eps = max(float(np.quantile(r[valid], 0.01)), 1e-9)
    lr = np.log(np.where(valid, r, 0.0) + eps).astype("float64")
    w = valid.astype("float64")
    den = np.maximum(uniform_filter(w, k, mode="nearest"), 1e-6)
    m = uniform_filter(np.where(valid, lr, 0.0), k, mode="nearest") / den
    sq = uniform_filter(np.where(valid, lr * lr, 0.0), k, mode="nearest") / den
    sd = np.sqrt(np.maximum(sq - m * m, 1e-12))
    z = (lr - m) / sd
    floor = float(z[valid].min()) if valid.any() else 0.0
    return np.where(valid, z, floor).astype("float32")


def oriented_response(img: np.ndarray, along: float, sigmas: tuple = SIGMAS,
                      workers: int = 2) -> np.ndarray:
    """Кривизна поперёк линии после усреднения ВДОЛЬ неё, максимум по углам.

    «Чистка снимка вдоль линии»: шум вдоль линии независим, а линия тянется,
    поэтому усреднение вытянутым окном (поперёк — прежние ``sigmas``, вдоль —
    ``along`` ячеек) гасит шум и оставляет линию. Направление заранее
    неизвестно: окно поворачивается с шагом, равным его угловой ширине, и
    берётся наибольший отклик. Считается в частотной области — гауссово окно и
    вторая производная поперёк дают множитель, зависящий только от угла.
    Замер и выбор длины окна — :mod:`openlin_clean_probe`.
    """
    from scipy import fft

    H, W = img.shape
    F = fft.rfft2(img, workers=workers)
    fy = fft.fftfreq(H)[:, None].astype("float32")
    fx = fft.rfftfreq(W)[None, :].astype("float32")
    best = np.zeros((H, W), dtype="float32")
    for s in sigmas:
        la = max(float(along), s)
        n_th = int(np.ceil(np.pi * la / s))
        for k in range(n_th):
            th = np.pi * k / n_th
            fa = fy * np.cos(th) + fx * np.sin(th)        # частота вдоль линии
            fn = -fy * np.sin(th) + fx * np.cos(th)       # частота поперёк
            h = (2 * np.pi * fn) ** 2 * np.exp(-2 * np.pi ** 2 * (la ** 2 * fa ** 2
                                                                  + s ** 2 * fn ** 2))
            r = fft.irfft2(F * h, s=(H, W), workers=workers)
            np.maximum(best, (s ** 2) * np.abs(r).astype("float32"), out=best)
    return best


def axial_local_with_support(img: np.ndarray, valid: np.ndarray,
                             res_m: float = 10.0, response_fn=None) -> tuple:
    """Осевая линия и опора для сшивки — по отклику, нормированному локально.

    Всё остальное как у «сшивки по порогу»: те же квантили гистерезиса и тот
    же порог опоры, только считаются они по нормированному отклику.
    ``response_fn`` подменяет сам отклик (по умолчанию — ``ridge_response``).
    """
    r = np.where(valid, (response_fn or ridge_response)(img), 0.0)
    z = local_norm(r, valid, res_m)
    v = z[valid]
    hi, lo = np.quantile(v, Q_HIGH), np.quantile(v, Q_LOW)
    _s, ridge = _ridge_nms(z, valid, 1.0)
    mask = hysteresis8(np.where(ridge, z, float(v.min())), lo, hi)
    return mask, z >= np.quantile(v, SUPPORT_STRICT_Q)


def to_lines_strict(raw_sup: tuple, res_m: float) -> tuple[np.ndarray, list]:
    """Сшивка разрывов, где под мостиком ВСЕ ячейки выше порога.

    Доля передаётся параметром, а не подменой глобальной константы: при запуске
    через ``python -m`` модуль работает как ``__main__``, и подмена через
    повторный импорт ушла бы в другую копию модуля — строгий режим тихо не
    включился бы.
    """
    return to_lines_supported(raw_sup, res_m, frac=1.0)


def to_lines_supported(raw_sup: tuple, res_m: float,
                       frac: float = SUPPORT_FRAC) -> tuple[np.ndarray, list]:
    """Как ``to_lines_linked``, но мостик обязан лежать на слабом гребне."""
    raw, sup = raw_sup
    keep = np.zeros(raw.shape, dtype=bool)
    rows = []
    for pts in link_gaps(_pieces(raw, res_m), res_m, support=sup, frac=frac):
        d = _describe(pts, res_m)
        if d is None:
            continue
        keep[pts[:, 0], pts[:, 1]] = True
        rows.append(d)
    return keep, rows


#: имя -> (отклик в маску, маска в линии)
GENERATORS = {"гребни": (lines_ridge, to_lines),
              "Canny": (lines_canny, to_lines),
              "осевая": (lines_axial, to_lines_joined),
              "осевая+сшивка": (lines_axial, to_lines_linked),
              "осевая 10-80 м": (lambda im, va: lines_axial(im, va, SIGMAS_WIDE),
                                 to_lines_joined),
              "осевая+сшивка по гребню": (axial_with_support,
                                          to_lines_supported),
              "осевая+сшивка по порогу": (axial_with_strict_support,
                                          to_lines_strict),
              "осевая+сшивка по порогу, локально": (axial_local_with_support,
                                                    to_lines_strict)}


def perturb(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Пересэмплирование: сдвиг на полъячейки, шаг 15 м и обратно на 10 м."""
    from scipy.ndimage import shift, zoom

    x = shift(np.where(valid, img, 0.0), (0.5, 0.5), order=1, mode="nearest")
    small = zoom(x, 2.0 / 3.0, order=1)
    back = zoom(small, (img.shape[0] / small.shape[0],
                        img.shape[1] / small.shape[1]), order=1)
    return back[:img.shape[0], :img.shape[1]].astype("float32")


def mutual_cover(a: np.ndarray, b: np.ndarray, res_m: float) -> float:
    """Средняя доля длины каждого списка, лежащая в допуске от другого."""
    from scipy.ndimage import distance_transform_edt

    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    da = distance_transform_edt(~a, sampling=(res_m, res_m))
    db = distance_transform_edt(~b, sampling=(res_m, res_m))
    return 0.5 * (float((db[a] <= TOL_M).mean())
                  + float((da[b] <= TOL_M).mean()))


def river_lines(shape, res_m: float, x0: float, y_top: float, epsg: int,
                valid: np.ndarray) -> np.ndarray:
    """Осевые линии русел (слой ``dnl``) на сетке фрагмента, в одну ячейку."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from skimage.morphology import skeletonize

    from experiments.build_dataset_v1 import read_sidecar_proj4
    from src import config

    p = Path(config.TOPO_SHP_DIR) / "dnl.shp"
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(read_sidecar_proj4(p), allow_override=True)
    g = g.to_crs(epsg=epsg)
    tr = Affine(res_m, 0, x0, 0, -res_m, y_top)
    r = rasterize(((s, 1) for s in g.geometry if s is not None),
                  out_shape=shape, transform=tr, fill=0, all_touched=True,
                  dtype="uint8").astype(bool)
    return skeletonize(r & valid)


def river_continuity(m: np.ndarray, river: np.ndarray, res_m: float
                     ) -> tuple[float, int, float]:
    """Доля русла в допуске от линий, число кусков и средний сплошной кусок."""
    from scipy.ndimage import binary_dilation, distance_transform_edt, label

    d = distance_transform_edt(~m, sampling=(res_m, res_m))
    cov = river & (d <= TOL_M)
    frac = cov.sum() / max(river.sum(), 1)
    k = np.ones((3, 3), dtype=bool)
    _lab, n = label(binary_dilation(cov, k) & binary_dilation(river, k))
    run_m = river.sum() * res_m * frac / max(n, 1)
    return float(frac), int(n), float(run_m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--channel", default="shade")
    ap.add_argument("--only", default=None,
                    help="прогнать один генератор по имени")
    ap.add_argument("--frag-tag", default="",
                    help="какой фрагмент: '' — участок настройки, "
                         "'_отлож' — юго-западный угол")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_line_rank import auc
    from experiments.openlin_null_input import phase_randomize
    from experiments.openlin_shadow import standardize

    z = dict(np.load(COMP / f"winter_{a.site}_10m{a.frag_tag}.npz"))
    res_m = float(z["res_m"])
    img = z[a.channel].astype("float32")
    valid = np.isfinite(img)
    img = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    fake = np.where(valid, phase_randomize(img, valid,
                                           np.random.default_rng(100)), 0.0)
    fake = fake.astype("float32")
    river = river_lines(img.shape, res_m, float(z["x0"]), float(z["y_top"]),
                        int(z["epsg"]), valid)
    log(f"площадка {a.site}: фрагмент {img.shape[0]} x {img.shape[1]}, шаг "
        f"{res_m:.0f} м, канал {a.channel}; русел для проверки "
        f"непрерывности {river.sum() * res_m / 1000:.0f} км")
    log(f"условия (объявлены до прогона): 1) AUC прямолинейности >= "
        f"{MIN_AUC_STRAIGHT:.2f}; 2) покрытие при пересэмплировании >= "
        f"{MIN_COVER:.2f} и >= {MIN_COVER_OVER_NULL:.0f}x нуля; 3) русло "
        f"покрыто >= {100 * MIN_RIVER_COVER:.0f} % кусками >= "
        f"{MIN_RIVER_RUN_M:.0f} м; конфигураций всего {N_CONFIGS}")

    rng = np.random.default_rng(7)
    rows_out = []
    masks = {}
    gens = {k: v for k, v in GENERATORS.items()
            if a.only is None or k == a.only}
    for name, (gen, liner) in gens.items():
        t0 = time.time()
        m_real, l_real = liner(gen(img, valid), res_m)
        m_fake, l_fake = liner(gen(fake, valid), res_m)
        y = np.r_[np.ones(len(l_real)), np.zeros(len(l_fake))]
        st = np.r_[[r["прямолинейность"] for r in l_real],
                   [r["прямолинейность"] for r in l_fake]]
        ln = np.r_[[r["длина_км"] for r in l_real],
                   [r["длина_км"] for r in l_fake]]
        auc_st, auc_ln = auc(y, st), auc(y, ln)

        m_pert, _ = liner(gen(perturb(img, valid), valid), res_m)
        cov = mutual_cover(m_real, m_pert, res_m)
        null = []
        for _ in range(20):
            dy = int(rng.integers(100, img.shape[0] - 100))
            dx = int(rng.integers(100, img.shape[1] - 100))
            null.append(mutual_cover(m_real, np.roll(np.roll(m_pert, dy, 0),
                                                     dx, 1), res_m))
        null = float(np.mean(null))
        r_cov, r_n, r_run = river_continuity(m_real, river, res_m)
        np.save(M / f"_iso_mask_{a.site}_{name}.npy", m_real)

        ok1 = auc_st >= MIN_AUC_STRAIGHT
        ok2 = cov >= MIN_COVER and cov >= MIN_COVER_OVER_NULL * null
        ok3 = r_cov >= MIN_RIVER_COVER and r_run >= MIN_RIVER_RUN_M
        masks[name] = m_real
        rows_out.append({"генератор": name, "линий": len(l_real),
                         "на подделке": len(l_fake),
                         "AUC прямолинейности": auc_st, "AUC длины": auc_ln,
                         "покрытие": cov, "нуль покрытия": null,
                         "русло покрыто": r_cov, "кусков на русле": r_n,
                         "сплошной кусок, м": r_run,
                         "условие 1": ok1, "условие 2": ok2, "условие 3": ok3})
        mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
        log(f"\n{name} ({time.time() - t0:.0f} с): линий {len(l_real)}, "
            f"на подделке {len(l_fake)}")
        log(f"  1) AUC прямолинейности {auc_st:.3f}, длины {auc_ln:.3f}  -> "
            f"{mark[ok1]}")
        log(f"  2) покрытие при пересэмплировании {cov:.3f}, у сдвинутой копии "
            f"{null:.3f} ({cov / max(null, 1e-9):.1f}x)  -> {mark[ok2]}")
        log(f"  3) русло покрыто {100 * r_cov:.1f} %, кусков {r_n}, средний "
            f"сплошной кусок {r_run:.0f} м  -> {mark[ok3]}")

    # для сведения, не условие: садятся ли линии на гидросеть
    try:
        from scipy.ndimage import distance_transform_edt

        from experiments.openlin_hydro_check import hydro_mask

        water = hydro_mask(img.shape, res_m, float(z["x0"]), float(z["y_top"]),
                           int(z["epsg"]), lambda *_: None)
        dw = distance_transform_edt(~water, sampling=(res_m, res_m))
        base = float((dw[valid] <= 100).mean())
        log("\nдля сведения (порогом не служит): доля линий ближе 100 м к "
            f"воде, по площади {100 * base:.1f} %")
        for name, m in masks.items():
            obs = float((dw[m & valid] <= 100).mean())
            log(f"  {name}: {100 * obs:.1f} % ({obs / max(base, 1e-9):.2f}x)")
    except Exception as e:                               # слои могут не лечь
        log(f"\nпроверка на гидросеть не посчиталась: {type(e).__name__}: {e}")

    import pandas as pd
    tag = (f"_{a.only}" if a.only else "") + a.frag_tag
    pd.DataFrame(rows_out).to_csv(M / f"openlin_iso_{a.site}{tag}.csv",
                                  index=False, encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_iso_{a.site}{tag}.csv'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    from experiments.openlin_shade import _stretch

    fig, ax = plt.subplots(1, len(masks) + 1, figsize=(6.4 * (len(masks) + 1),
                                                         6.9))
    base_img = _stretch(np.where(valid, img, np.nan))
    ax[0].imshow(base_img, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("зимний композит, 10 м", fontsize=14)
    for k, (name, m) in enumerate(masks.items(), start=1):
        ax[k].imshow(base_img, cmap="gray", vmin=0, vmax=1)
        w = binary_dilation(m, np.ones((3, 3), dtype=bool))
        ov = np.zeros(w.shape + (4,), dtype="float32")
        ov[w] = (1.0, 0.2, 0.1, 0.9)
        ax[k].imshow(ov)
        r = next(x for x in rows_out if x["генератор"] == name)
        ax[k].set_title(f"{name}: {r['линий']} линий\nAUC прямизны "
                        f"{r['AUC прямолинейности']:.2f}, русло "
                        f"{100 * r['русло покрыто']:.0f} % кусками по "
                        f"{r['сплошной кусок, м']:.0f} м", fontsize=12)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    out = Path("outputs/steps") / f"изотропные_линии_{a.site}{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=105)
    log(f"сохранено: {out}")




# --------------------------------------------------------------------------
# Отложенный фрагмент
# --------------------------------------------------------------------------

def pick_holdout_center(site: str, log, skip: tuple = ()
                        ) -> tuple[int, int] | None:
    """Место отложенного фрагмента — правилом, объявленным заранее."""
    from experiments.openlin_lowsun import (FRAG_CENTER, FRAG_KM, find_scenes,
                                            scene_grid)
    from experiments.openlin_unet import load_site

    z = load_site(site)
    h, w = z["gray"].shape
    big = float(z["res_m"])
    half = int(round(FRAG_KM * 500 / big))
    r0, c0 = FRAG_CENTER
    cands = [(half, half), (half, w - half), (h - half, half),
             (h - half, w - half)]
    cands.sort(key=lambda rc: -np.hypot(rc[0] - r0, rc[1] - c0))
    for rc in cands:
        if tuple(int(v) for v in rc) in skip:
            log(f"  центр {rc}: уже израсходован — пропуск")
            continue
        dist_km = np.hypot(rc[0] - r0, rc[1] - c0) * big / 1000
        g10 = scene_grid(site, FRAG_KM, 10.0, rc)
        items = find_scenes(g10, lambda *_: None)
        valid = np.ones((g10.height, g10.width), dtype=bool)
        riv = river_lines(valid.shape, 10.0, g10.x0, g10.y_top, g10.epsg,
                          valid)
        river_km = riv.sum() * 10.0 / 1000
        ok = len(items) >= HOLDOUT_MIN_SCENES and \
            river_km >= HOLDOUT_MIN_RIVER_KM
        log(f"  центр {rc}: {dist_km:.0f} км от участка настройки, сцен "
            f"{len(items)}, русел {river_km:.0f} км -> "
            f"{'БЕРЁМ' if ok else 'не подходит'}")
        if ok:
            return rc
    return None


def run_holdout(site: str, tag: str, log,
                gen_name: str = HOLDOUT_GENERATOR) -> None:
    """Замороженный кандидат на отложенном фрагменте, условия А-Г."""
    from experiments.openlin_line_rank import auc
    from experiments.openlin_null_input import phase_randomize
    from experiments.openlin_shadow import standardize

    gen, liner = GENERATORS[gen_name]
    z = dict(np.load(COMP / f"winter_{site}_10m{tag}.npz"))
    res_m = float(z["res_m"])
    img = z["shade"].astype("float32")
    valid = np.isfinite(img)
    img = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    river = river_lines(img.shape, res_m, float(z["x0"]), float(z["y_top"]),
                        int(z["epsg"]), valid)
    log(f"отложенный фрагмент {tag}: {img.shape[0]} x {img.shape[1]}, "
        f"русел {river.sum() * res_m / 1000:.0f} км; генератор "
        f"«{gen_name}», подделок {HOLDOUT_N_FAKES}")

    t0 = time.time()
    m_real, l_real = liner(gen(img, valid), res_m)
    fakes = []
    for k in range(HOLDOUT_N_FAKES):
        f = np.where(valid, phase_randomize(img, valid,
                                            np.random.default_rng(100 + k)),
                     0.0).astype("float32")
        _m, lf = liner(gen(f, valid), res_m)
        fakes.append(lf)
    pooled = [l for lf in fakes for l in lf]
    per_fake = np.array([len(lf) for lf in fakes])
    log(f"  линий: настоящих {len(l_real)}, на подделках {per_fake.tolist()} "
        f"({time.time() - t0:.0f} с)")

    st = np.array([l["прямолинейность"] for l in l_real])
    stf = np.array([l["прямолинейность"] for l in pooled])
    y = np.r_[np.ones(len(st)), np.zeros(len(stf))]
    a_auc = auc(y, np.r_[st, stf]) if len(stf) else float("nan")
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(st), len(st))
        j = rng.integers(0, len(stf), len(stf)) if len(stf) else []
        if len(stf):
            bs.append(auc(np.r_[np.ones(len(i)), np.zeros(len(j))],
                          np.r_[st[i], stf[j]]))
    junk = float(per_fake.mean()) / max(len(l_real), 1)

    rng7 = np.random.default_rng(7)
    m_pert, _ = liner(gen(perturb(img, valid), valid), res_m)
    cov = mutual_cover(m_real, m_pert, res_m)
    null = float(np.mean([mutual_cover(
        m_real, np.roll(np.roll(m_pert, int(rng7.integers(100, img.shape[0] - 100)), 0),
                        int(rng7.integers(100, img.shape[1] - 100)), 1), res_m)
        for _ in range(20)]))
    r_cov, r_n, r_run = river_continuity(m_real, river, res_m)

    okA = a_auc >= HOLDOUT_MIN_AUC
    okB = junk <= HOLDOUT_MAX_JUNK
    okV = cov >= MIN_COVER and cov >= MIN_COVER_OVER_NULL * null
    okG = r_cov >= MIN_RIVER_COVER and r_run >= MIN_RIVER_RUN_M
    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    lo, hi = (np.percentile(bs, [5, 95]) if bs else (np.nan, np.nan))
    log(f"  А) AUC прямизны {a_auc:.3f} (90 % интервал {lo:.3f}-{hi:.3f}), "
        f"порог {HOLDOUT_MIN_AUC:.2f}  -> {mark[okA]}")
    log(f"  Б) мусора по числу линий {100 * junk:.1f} %, порог "
        f"{100 * HOLDOUT_MAX_JUNK:.0f} %  -> {mark[okB]}")
    log(f"  В) устойчивость {cov:.3f}, нуль {null:.3f} "
        f"({cov / max(null, 1e-9):.1f}x)  -> {mark[okV]}")
    log(f"  Г) русло покрыто {100 * r_cov:.1f} %, кусков {r_n}, сплошной "
        f"кусок {r_run:.0f} м  -> {mark[okG]}")

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_hydro_check import hydro_mask
    water = hydro_mask(img.shape, res_m, float(z["x0"]), float(z["y_top"]),
                       int(z["epsg"]), lambda *_: None)
    dw = distance_transform_edt(~water, sampling=(res_m, res_m))
    base = float((dw[valid] <= 100).mean())
    obs = float((dw[m_real & valid] <= 100).mean())
    log(f"  для сведения: у воды (100 м) {100 * obs:.1f} % линий при "
        f"{100 * base:.1f} % площади ({obs / max(base, 1e-9):.2f}x)")
    gtag = "" if gen_name == HOLDOUT_GENERATOR else "_локально"
    np.save(M / f"_iso_mask_{site}_holdout{tag}{gtag}.npy", m_real)
    import pandas as pd
    pd.DataFrame([{"фрагмент": tag, "генератор": gen_name,
                   "линий": len(l_real), "на подделку": per_fake.mean(),
                   "AUC прямизны": a_auc, "AUC 5%": lo, "AUC 95%": hi,
                   "мусор по числу": junk, "устойчивость": cov,
                   "нуль устойчивости": null, "русло покрыто": r_cov,
                   "сплошной кусок, м": r_run, "А": okA, "Б": okB,
                   "В": okV, "Г": okG}]).to_csv(
        M / f"openlin_iso_holdout_{site}{tag}{gtag}.csv", index=False,
        encoding="utf-8-sig")
    log(f"сохранено: {M / f'openlin_iso_holdout_{site}{tag}{gtag}.csv'}")


def holdout_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--pick", action="store_true",
                    help="только выбрать место по правилу")
    ap.add_argument("--tag", default="_отлож")
    ap.add_argument("--gen", default=HOLDOUT_GENERATOR)
    ap.add_argument("--skip-used", action="store_true",
                    help="при выборе места пропустить израсходованные")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    if a.pick:
        rc = pick_holdout_center(a.site, log,
                                 HOLDOUT_USED if a.skip_used else ())
        log(f"выбран центр: {rc}")
        return
    run_holdout(a.site, a.tag, log, a.gen)


if __name__ == "__main__":
    import sys

    if "--holdout" in sys.argv:
        sys.argv.remove("--holdout")
        holdout_main()
    else:
        main()
