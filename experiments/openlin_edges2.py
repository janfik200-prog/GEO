"""Оператор линии: исправление третьего шага.

ЧТО БЫЛО НЕ ТАК. Прежний оператор сравнивал средние двух окон, сдвинутых на
±300 м от предполагаемой линии, и брал максимум по двенадцати направлениям.
Измерение на синтетике (``openlin_operator_fix``) показало две беды:

1. Он ищет СТУПЕНЬ — смену типа поверхности, — а не линию. Узкий линеамент
   (борозда, дайка, спрямлённая долина) попадает в зазор между окнами, и слева
   и справа от него оказывается одна и та же местность: разность средних около
   нуля. На чистом фоне отклик на линию вчетверо слабее, чем на ступень того же
   контраста, а в текстуре линия не видна вовсе — отклик на ней 0.93 от фона.
2. На коррелированной текстуре БЕЗ единой линии он рисует ковёр из вытянутых
   «перьев»: максимум по направлениям всегда найдёт угол, под которым два окна
   случайно разошлись. Именно этот ковёр и трассировался дальше как линеаменты.

ЧТО ЗДЕСЬ. Два оператора и способ их сложить:

* :func:`edges_ridge` — отклик на линию: узкая центральная полоса против
  полусуммы двух боковых;
* :func:`edges_step` — отклик на ступень, то есть прежний смысл, но скачок
  нормирован на разброс ВНУТРИ окон;
* :func:`edges_both` — робастная нормировка обоих и максимум: настоящими бывают
  и линии, и границы блоков.

Нормировка на разброс внутри окон — обычная t-статистика: граница или линия
значима, когда скачок велик по сравнению с тем, как величина гуляет внутри
самих участков. У пёстрого места знаменатель растёт вместе с числителем, и
ложный отклик подавляется.
"""
from __future__ import annotations

import numpy as np

from experiments.openlin_domains import (N_ANGLES, OFFSET_M, WIN_LONG_M,
                                         WIN_SHORT_M, _kernel, _smooth)

#: Ширина центральной полосы, м. 100 м = одна ячейка растра: узкий линеамент на
#: снимке имеет именно такую ширину, шире брать — размажем его вместе с фоном.
CENTER_M = 100.0

#: Окно локального фона, м. Из поля вычитается его же медиана по окну 5 км:
#: это снимает общий уровень «пёстро/ровно» и оставляет локальные выбросы,
#: то есть собственно линии. Линия занимает малую долю такого окна и переживает
#: вычитание, а ковёр из перьев — нет.
BG_M = 5000.0


def _moments(comp: np.ndarray, valid: np.ndarray, sq: np.ndarray,
             k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и дисперсия каждой компоненты в окне ``k``."""
    m = _smooth(comp, valid, k)
    v = np.maximum(_smooth(sq, valid, k) - m ** 2, 0.0)
    return m, v


def _angle_stack(comp: np.ndarray, valid: np.ndarray, res_m: float, mode: str,
                 n_angles: int) -> np.ndarray:
    """Нормированный отклик по каждому направлению: ``(n_angles, H, W)``."""
    sq = (comp ** 2).astype("float32")
    s_par, s_perp = WIN_LONG_M / res_m, WIN_SHORT_M / res_m
    out = []
    for i in range(n_angles):
        th = np.pi * i / n_angles
        kp = _kernel(s_par, s_perp, th, OFFSET_M / res_m)
        km = _kernel(s_par, s_perp, th, -OFFSET_M / res_m)
        mp, vp = _moments(comp, valid, sq, kp)
        mm, vm = _moments(comp, valid, sq, km)
        if mode == "ridge":
            kc = _kernel(s_par, CENTER_M / res_m, th, 0.0)
            n_eff = float(kc.sum() ** 2 / (kc ** 2).sum())
            mc, vc = _moments(comp, valid, sq, kc)
            num = ((mc - 0.5 * (mp + mm)) ** 2).sum(0)
            den = (vc + 0.25 * (vp + vm)).sum(0) / n_eff + 1e-9
        else:
            n_eff = float(kp.sum() ** 2 / (kp ** 2).sum())
            num = ((mp - mm) ** 2).sum(0)
            den = (vp + vm).sum(0) / n_eff + 1e-9
        out.append(np.sqrt(num / den))
    return np.stack(out).astype("float32")


def _peak(st: np.ndarray, n_angles: int, valid: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """Острота отклика по направлениям: максимум минус медиана по ним.

    У настоящей линии выделяется одно направление, а остальные одиннадцать
    остаются низкими. Контрастное пятно внутри окна, наоборот, даёт отклик сразу
    при многих направлениях, и после вычитания медианы от него мало что
    остаётся. На синтетике это поднимает долю линии, попавшей в верхние 20 %
    поля, с 67 до 69 %, а для границы блоков — с 73 до 78 %.
    """
    best = st.max(0) - np.median(st, axis=0)
    idx = st.argmax(0)
    ang = (np.degrees(np.pi * idx / n_angles)).astype("float32")
    return np.where(valid, best, np.nan).astype("float32"), ang


def edges_ridge(comp: np.ndarray, valid: np.ndarray, res_m: float,
                n_angles: int = N_ANGLES) -> tuple[np.ndarray, np.ndarray]:
    """Сила ЛИНИИ: центральная полоса против двух боковых, в единицах разброса."""
    st = _angle_stack(comp, valid, res_m, "ridge", n_angles)
    return _peak(st, n_angles, valid)


def edges_step(comp: np.ndarray, valid: np.ndarray, res_m: float,
               n_angles: int = N_ANGLES) -> tuple[np.ndarray, np.ndarray]:
    """Сила СТУПЕНИ: прежний смысл, но с нормировкой на разброс внутри окон."""
    st = _angle_stack(comp, valid, res_m, "step", n_angles)
    return _peak(st, n_angles, valid)


def detrend(s: np.ndarray, valid: np.ndarray, res_m: float,
            win_m: float = BG_M) -> np.ndarray:
    """Вычесть локальный фон поля и поделить на локальный разброс.

    Поле силы систематически выше там, где местность пестрее, — это свойство
    места, а не наличия линии. Приведение к локальному фону убирает этот
    перекос, и порог трассировки начинает означать одно и то же по всей
    площадке.
    """
    from scipy.ndimage import uniform_filter

    k = max(int(round(win_m / res_m)), 3)
    x = np.where(valid, np.nan_to_num(s, nan=0.0), 0.0).astype("float64")
    w = valid.astype("float64")
    num = uniform_filter(x, k, mode="nearest")
    den = np.maximum(uniform_filter(w, k, mode="nearest"), 1e-3)
    mean = num / den
    var = np.maximum(uniform_filter(x ** 2, k, mode="nearest") / den
                     - mean ** 2, 1e-12)
    out = (np.nan_to_num(s, nan=0.0) - mean) / np.sqrt(var)
    return np.where(valid, out, np.nan).astype("float32")


#: Длина подотрезка, м, и сколько их укладывается вдоль окна. Отклик считается
#: на каждом подотрезке отдельно, а берётся НАИМЕНЬШИЙ: тогда одно контрастное
#: пятно внутри окна не растягивается в «перо» на всю его длину — линия обязана
#: подтверждаться по всей длине окна, а не в среднем по нему.
SEG_M = 500.0
N_SEG = 4


def _kernel2(sigma_long: float, sigma_short: float, theta: float,
             off_perp: float, off_par: float = 0.0) -> np.ndarray:
    """Вытянутое окно со сдвигом и поперёк линии, и вдоль неё."""
    r = int(3 * max(sigma_long, sigma_short) + abs(off_perp) + abs(off_par) + 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype("float64")
    uy, ux = -np.cos(theta), np.sin(theta)      # вдоль линии
    vy, vx = np.sin(theta), np.cos(theta)       # поперёк линии
    py = yy - off_perp * vy - off_par * uy
    px = xx - off_perp * vx - off_par * ux
    a = py * uy + px * ux
    b = py * vy + px * vx
    k = np.exp(-0.5 * ((a / sigma_long) ** 2 + (b / sigma_short) ** 2))
    return k / k.sum()


def edges_seg(comp: np.ndarray, valid: np.ndarray, res_m: float,
              mode: str = "ridge", n_angles: int = N_ANGLES,
              n_seg: int = N_SEG, seg_m: float = SEG_M
              ) -> tuple[np.ndarray, np.ndarray]:
    """Отклик, который обязан держаться на всех подотрезках окна.

    ``mode='ridge'`` — центр против боков (линия), ``mode='step'`` — бок против
    бока (граница блоков). По каждому подотрезку считается нормированный на
    внутренний разброс отклик, итог по направлению — наименьший из них.
    """
    best = np.full(valid.shape, -np.inf, dtype="float32")
    ang = np.zeros(valid.shape, dtype="float32")
    sq = (comp ** 2).astype("float32")
    s_seg = seg_m / res_m / 2.0                 # сигма подотрезка
    s_perp = WIN_SHORT_M / res_m
    s_cent = CENTER_M / res_m
    offs = (np.arange(n_seg) - (n_seg - 1) / 2) * seg_m / res_m

    for i in range(n_angles):
        th = np.pi * i / n_angles
        worst = None
        for off in offs:
            kp = _kernel2(s_seg, s_perp, th, OFFSET_M / res_m, off)
            km = _kernel2(s_seg, s_perp, th, -OFFSET_M / res_m, off)
            n_eff = float(kp.sum() ** 2 / (kp ** 2).sum())
            mp, vp = _moments(comp, valid, sq, kp)
            mm, vm = _moments(comp, valid, sq, km)
            if mode == "ridge":
                kc = _kernel2(s_seg, s_cent, th, 0.0, off)
                mc, vc = _moments(comp, valid, sq, kc)
                num = ((mc - 0.5 * (mp + mm)) ** 2).sum(0)
                den = (vc + 0.25 * (vp + vm)).sum(0) / n_eff + 1e-9
            else:
                num = ((mp - mm) ** 2).sum(0)
                den = (vp + vm).sum(0) / n_eff + 1e-9
            s = np.sqrt(num / den).astype("float32")
            worst = s if worst is None else np.minimum(worst, s)
        upd = worst > best
        best = np.where(upd, worst, best)
        ang = np.where(upd, np.degrees(th), ang)
    return np.where(valid, best, np.nan), ang


# ---------------------------------------------------------------------------
# Другая геометрия: короткое окно плюс прослеживание вдоль линии
# ---------------------------------------------------------------------------

#: Длина локального окна, м. Вдвое короче прежних 2 км: на километре даже
#: заметно изогнутая линия остаётся почти прямой, и окно её не теряет.
#: Подобрано по синтетике: 300 м слишком шумно (34 % длины прямой в верхних
#: 20 % поля), 500 м — 77 %, 1 км — 99 % при 97 % на дуге.
LOCAL_M = 1000.0

#: Поперечные размеры локального окна: центральная полоса и боковые.
LOCAL_CENTER_M = 100.0
LOCAL_SIDE_M = 150.0
LOCAL_OFFSET_M = 200.0

#: Накопление вдоль линии: длина шага и число шагов. Протяжённость набирается
#: не размером окна, а тем, что соседние отрезки поддерживают друг друга, и
#: поддержка идёт по дуге, потому что на каждом шаге направление берётся своё.
#: Перетекание в соседние направления (ANG_MIX) пробовали — оно только поднимает
#: медиану по направлениям и роняет пик: 82 % против 99 % на прямой.
ACC_M = 1000.0
N_ITER = 4
ANG_MIX = 0.0


def _local_stack(comp: np.ndarray, valid: np.ndarray, res_m: float,
                 n_angles: int) -> np.ndarray:
    """Локальный отклик на линию: короткое окно, по каждому направлению."""
    sq = (comp ** 2).astype("float32")
    s_par = LOCAL_M / res_m
    out = []
    for i in range(n_angles):
        th = np.pi * i / n_angles
        kc = _kernel(s_par, LOCAL_CENTER_M / res_m, th, 0.0)
        kp = _kernel(s_par, LOCAL_SIDE_M / res_m, th, LOCAL_OFFSET_M / res_m)
        km = _kernel(s_par, LOCAL_SIDE_M / res_m, th, -LOCAL_OFFSET_M / res_m)
        n_eff = float(kc.sum() ** 2 / (kc ** 2).sum())
        mc, vc = _moments(comp, valid, sq, kc)
        mp, vp = _moments(comp, valid, sq, kp)
        mm, vm = _moments(comp, valid, sq, km)
        num = ((mc - 0.5 * (mp + mm)) ** 2).sum(0)
        den = (vc + 0.25 * (vp + vm)).sum(0) / n_eff + 1e-9
        out.append(np.sqrt(num / den))
    return np.stack(out).astype("float32")


def edges_curve(comp: np.ndarray, valid: np.ndarray, res_m: float,
                n_angles: int = 18, n_iter: int = N_ITER,
                acc_m: float = ACC_M, ang_mix: float = ANG_MIX
                ) -> tuple[np.ndarray, np.ndarray]:
    """Отклик на линию произвольной кривизны.

    Схема другая, чем у прежнего оператора. Там протяжённость задавалась
    размером окна — 2 км вдоль линии, — и всё, что не укладывалось в прямой
    отрезок такой длины, рвалось, а любое контрастное пятно, наоборот,
    растягивалось в «перо» на всю длину окна.

    Здесь окно короткое (300 м), а длина набирается накоплением: отклик
    размазывается вдоль своего же направления на 400 м и частично перетекает в
    соседние направления, и так несколько раз. Линия, плавно меняющая азимут,
    накапливает поддержку по дуге; одиночное пятно поддержки не получает,
    потому что продолжения у него нет.
    """
    st = _local_stack(comp, valid, res_m, n_angles)
    s_acc = acc_m / res_m
    for _ in range(n_iter):
        moved = np.empty_like(st)
        for i in range(n_angles):
            th = np.pi * i / n_angles
            k = _kernel(s_acc, LOCAL_CENTER_M / res_m, th, 0.0)
            moved[i] = _smooth(st[i][None], valid, k)[0]
        # перетекание в соседние направления: поддержка идёт по дуге
        st = ((1.0 - ang_mix) * moved
              + 0.5 * ang_mix * (np.roll(moved, 1, axis=0)
                                 + np.roll(moved, -1, axis=0)))
    return _peak(st.astype("float32"), n_angles, valid)


def _robust(s: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Привести поле к общей шкале: медиана и межквартильный размах."""
    v = s[valid & np.isfinite(s)]
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25)) or 1.0
    return ((s - med) / iqr).astype("float32")


def edges_both(comp: np.ndarray, valid: np.ndarray, res_m: float,
               n_angles: int = N_ANGLES, do_detrend: bool = True
               ) -> tuple[np.ndarray, np.ndarray]:
    """Линия или ступень: оба оператора в общей шкале, берётся больший."""
    sr, ar = edges_ridge(comp, valid, res_m, n_angles)
    ss, as_ = edges_step(comp, valid, res_m, n_angles)
    if do_detrend:
        sr = detrend(sr, valid, res_m)
        ss = detrend(ss, valid, res_m)
    sr, ss = _robust(sr, valid), _robust(ss, valid)
    take_ridge = sr >= ss
    best = np.where(take_ridge, sr, ss).astype("float32")
    ang = np.where(take_ridge, ar, as_).astype("float32")
    return np.where(valid, best, np.nan), ang
