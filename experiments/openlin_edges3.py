"""Шаг 3 заново: детектор линий по кривизне, без перебора направлений.

ПОЧЕМУ ПРЕЖНИЙ ОПЕРАТОР ЗАМЕНЁН. Он сравнивал средние в двух вытянутых окнах и
брал максимум по двенадцати направлениям. У такой конструкции два врождённых
порока, и оба измерены на синтетике:

* максимум по направлениям на любой текстуре систематически выбирает
  наибольшее случайное отклонение, поэтому на чистом коррелированном шуме
  оператор рисует ковёр из вытянутых «перьев» — их и трассировали как
  линеаменты;
* длина окна (2 км) задавала длину линии: всё, что не укладывалось в прямой
  отрезок такой длины, рвалось, а одно контрастное пятно растягивалось на всю
  длину окна.

ЧТО ВМЕСТО. Классический детектор протяжённых структур по вторым производным
(Штегер, Франжи). Линия — это место, где поверхность сильно изогнута ПОПЕРЁК
одного направления и почти не изогнута ВДОЛЬ него. Направление не перебирается,
а получается из собственного вектора, поэтому артефакта «максимум по углам»
нет в принципе. Длина ничем не задаётся: оператор локальный, и протяжённость
возникает сама, когда соседние ячейки продолжают друг друга.

МНОГОКАНАЛЬНОСТЬ. Сжатое описание — восемь компонент, и кривизна считается по
каждой, а затем складывается в одну матрицу вторых моментов
``T = sum_c H_c H_c^T``. Её собственные числа отвечают на нужный вопрос: линия
там, где одно велико, а второе мало.

МНОГОМАСШТАБНОСТЬ. Линеаменты бывают шириной от ячейки до километра, поэтому
расчёт повторяется на нескольких масштабах сглаживания, и берётся лучший.
Нормировка на масштаб (множитель sigma^2) уравнивает их отклики.
"""
from __future__ import annotations

import numpy as np

#: Масштабы сглаживания, м. Нижний — одна ячейка растра (узкая борозда),
#: верхний — километр (широкая зона). Промежуточные с шагом вдвое: у гауссовых
#: производных отклик меняется плавно, и более частая сетка ничего не добавляет.
SCALES_M = (100.0, 200.0, 400.0, 800.0)

#: Окно локального фона, м: из готового поля вычитается его медиана по окну и
#: делится на локальный разброс. Порог трассировки после этого означает одно и
#: то же на пёстрых и на ровных участках.
BG_M = 5000.0


def _smooth_masked(x: np.ndarray, valid: np.ndarray, sigma_px: float
                   ) -> np.ndarray:
    """Гауссово сглаживание стека (C,H,W), нормированное на маску кадра."""
    from scipy.ndimage import gaussian_filter

    m = valid.astype("float32")
    num = gaussian_filter(np.where(valid[None], x, 0.0).astype("float32"),
                          (0, sigma_px, sigma_px), mode="nearest")
    den = gaussian_filter(m, sigma_px, mode="nearest")
    return num / np.maximum(den, 1e-3)[None]


def _curvature_tensor(x: np.ndarray, valid: np.ndarray, sigma_px: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сумма по каналам вторых производных: три числа матрицы 2x2 на ячейку."""
    s = _smooth_masked(x, valid, sigma_px)
    # вторые производные конечными разностями по уже сглаженному стеку
    gy, gx = np.gradient(s, axis=(1, 2))
    hyy, hyx = np.gradient(gy, axis=(1, 2))
    hxy, hxx = np.gradient(gx, axis=(1, 2))
    hxy = 0.5 * (hxy + hyx)
    # нормировка на масштаб: у гауссовых вторых производных отклик падает
    # как 1/sigma^2, и без множителя крупные масштабы всегда проигрывали бы
    k = sigma_px ** 2
    hxx, hyy, hxy = hxx * k, hyy * k, hxy * k
    # T = sum_c H_c H_c^T — симметричная матрица 2x2
    txx = (hxx * hxx + hxy * hxy).sum(0)
    tyy = (hyy * hyy + hxy * hxy).sum(0)
    txy = (hxx * hxy + hxy * hyy).sum(0)
    return txx, tyy, txy


def _line_strength(txx: np.ndarray, tyy: np.ndarray, txy: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Сила линии и её азимут из матрицы вторых моментов кривизны.

    Собственные числа: большое отвечает кривизне поперёк линии, малое — вдоль.
    Сила — разность их корней: она велика, когда кривизна есть только в одном
    направлении, и близка к нулю у пятна, где кривизна во все стороны
    одинаковая.
    """
    half = 0.5 * (txx + tyy)
    rad = np.sqrt(np.maximum((0.5 * (txx - tyy)) ** 2 + txy ** 2, 0.0))
    lam1 = np.maximum(half + rad, 0.0)      # поперёк линии
    lam2 = np.maximum(half - rad, 0.0)      # вдоль линии
    strength = np.sqrt(lam1) - np.sqrt(lam2)
    # собственный вектор при lam1 — поперёк линии; азимут линии ему перпендикуляр
    vy, vx = txy, lam1 - txx
    nrm = np.hypot(vx, vy)
    flat = nrm < 1e-12
    vy = np.where(flat, 1.0, vy / np.maximum(nrm, 1e-12))
    vx = np.where(flat, 0.0, vx / np.maximum(nrm, 1e-12))
    # поворот на 90 градусов даёт направление вдоль линии
    ang = np.degrees(np.arctan2(vx, vy)) % 180.0
    return strength.astype("float32"), ang.astype("float32")


def detrend(s: np.ndarray, valid: np.ndarray, res_m: float,
            win_m: float = BG_M) -> np.ndarray:
    """Привести поле к локальному фону: вычесть медиану окна, поделить на разброс."""
    from scipy.ndimage import uniform_filter

    k = max(int(round(win_m / res_m)), 3)
    x = np.where(valid, np.nan_to_num(s, nan=0.0), 0.0).astype("float64")
    w = valid.astype("float64")
    den = np.maximum(uniform_filter(w, k, mode="nearest"), 1e-3)
    mean = uniform_filter(x, k, mode="nearest") / den
    var = np.maximum(uniform_filter(x ** 2, k, mode="nearest") / den
                     - mean ** 2, 1e-12)
    out = (np.nan_to_num(s, nan=0.0) - mean) / np.sqrt(var)
    return np.where(valid, out, np.nan).astype("float32")


#: Накопление вдоль линии: длина ядра и число проходов. Кривизна — величина
#: точечная, и сама по себе она не отличает короткий всплеск от протяжённой
#: линии. Поэтому отклик подтягивается вдоль СВОЕГО ЖЕ направления, взятого из
#: собственного вектора: настоящая линия накапливает поддержку по всей длине,
#: одиночное пятно — нет. Направление на каждом шаге берётся заново, поэтому
#: поддержка идёт и по дуге.
ACC_M = 800.0
ACC_ITER = 2

#: На сколько градусов ячейка считается «своей» для ориентации при накоплении.
#: 18 ориентаций с шагом 10 градусов: мельче — дороже и без выигрыша.
ACC_ANGLES = 18


def accumulate(s: np.ndarray, ang: np.ndarray, valid: np.ndarray, res_m: float,
               length_m: float = ACC_M, n_iter: int = ACC_ITER,
               n_angles: int = ACC_ANGLES) -> np.ndarray:
    """Подтянуть отклик вдоль его собственного направления."""
    from experiments.openlin_domains import _kernel, _smooth

    out = np.nan_to_num(s, nan=0.0).astype("float32")
    sigma = max(length_m / res_m / 2.0, 1.0)
    for _ in range(n_iter):
        acc = np.zeros_like(out)
        for i in range(n_angles):
            th_deg = 180.0 * i / n_angles
            d = np.abs(ang - th_deg)
            near = np.minimum(d, 180.0 - d) <= (90.0 / n_angles)
            if not near.any():
                continue
            k = _kernel(sigma, 0.8, np.radians(th_deg), 0.0)
            sm = _smooth(np.where(near, out, 0.0)[None], valid, k)[0]
            acc = np.maximum(acc, sm)
        out = acc.astype("float32")
    return np.where(valid, out, np.nan).astype("float32")


def edges_hessian(comp: np.ndarray, valid: np.ndarray, res_m: float,
                  scales_m: tuple[float, ...] = SCALES_M,
                  do_detrend: bool = True, acc_iter: int = ACC_ITER
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Сила линии и её азимут: максимум по масштабам сглаживания."""
    best = np.full(valid.shape, -np.inf, dtype="float32")
    ang = np.zeros(valid.shape, dtype="float32")
    x = np.nan_to_num(comp, nan=0.0).astype("float32")
    for sm in scales_m:
        txx, tyy, txy = _curvature_tensor(x, valid, max(sm / res_m, 0.8))
        s, a = _line_strength(txx, tyy, txy)
        # приводим масштабы к общей шкале: у каждого свой уровень шума
        v = s[valid & np.isfinite(s)]
        med = float(np.median(v))
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25)) or 1.0
        s = (s - med) / iqr
        upd = s > best
        best = np.where(upd, s, best)
        ang = np.where(upd, a, ang)
    out = np.where(valid, best, np.nan).astype("float32")
    if acc_iter:
        out = accumulate(out, ang, valid, res_m, n_iter=acc_iter)
    if do_detrend:
        out = detrend(out, valid, res_m)
    return out, ang
