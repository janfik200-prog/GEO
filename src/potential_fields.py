"""Трансформанты потенциальных полей: БПФ на нативной сетке ``грав_маг.pgrid``.

Аудит (``docs II presentation/Аудит-признаков-что-не-вытащено.md``) нашёл, что
из 17 нативных слоёв гравики/магнитки посчитаны только регионально-остаточное
разделение (window 35) и один масштаб градиента (window 25); нет первой
вертикальной производной, амплитудно-независимого трассировщика кромок
(tilt-derivative), аналитического сигнала, многомасштабного продолжения
вверх, приведения магнитки к полюсу и скользящей корреляции гравика-магнитка.
Все эти трансформанты считаются здесь, БЕЗ новых скачиваний — вход тот же
``config.GRAVMAG_PGRID``, уже на диске.

Почему БПФ на НАТИВНОЙ сетке, а не на целевом листе прогноза: нативная сетка
305x455 (шаг 500 м) шире листа 149x154 на ~78 ячеек по X и ~150 по Y. БПФ
предполагает периодичность поля, и на краю массива период обрывается —
разрыв даёт паразитную высокочастотную энергию по всему спектру. С таким
запасом обрезка обратно на целевую сетку (после ресемплинга) уносит
краевой артефакт за пределы листа. Дополнительно применяется окно Тьюки
(:data:`config.PF_TAPER_FRAC` края) поверх среднего, вычтенного перед БПФ.

Формулы (Blakely, 1995, «Potential Theory in Gravity and Magnetic
Applications»), гармоническое поле в верхнем полупространстве:

* первая вертикальная производная: спектр умножается на ``|k|``;
* горизонтальные производные: спектр умножается на ``i*kx`` / ``i*ky``;
* tilt-derivative = ``arctan2(Vz, sqrt(Vx^2+Vy^2))`` — в [-90°, +90°],
  ноль на кромке источника независимо от его амплитуды;
* аналитический сигнал = ``sqrt(Vx^2+Vy^2+Vz^2)`` — максимум над кромкой,
  не зависит от направления намагниченности (полезен для магнитки на
  приполярных широтах, где обычное поле искажено наклонением);
* продолжение вверх на высоту h: спектр умножается на ``exp(-|k|*h)``;
* приведение к полюсу (RTP, только магнитка, индуцированная намагниченность
  = направлению поля): спектр умножается на ``k^2 / Theta_f^2``, где
  ``Theta_f = i*(kx*cosI*cosD + ky*cosI*sinD) + k*sinI``. На высоких
  наклонениях (лист на 71 с.ш., I=84.74°) оператор хорошо обусловлен —
  типичная низкоширотная неустойчивость RTP здесь не возникает.

Инклинация/склонение — из IGRF (``ppigrf``) в центре листа, эпоха
2020-07-01 (:data:`config.PF_INCLINATION_DEG`, :data:`config.PF_DECLINATION_DEG`) —
точная дата съёмки магнитки в комплекте заказчика не указана, это приближение.

Вертикальная производная существует только для одного поля из двух: свойство
``gr_2V_25`` в нативной сетке уже является ВТОРОЙ производной гравики
(см. журнал 06.08.2026) — то есть Integro сам считал вертикальные производные
только для силы тяжести, а не магнитки; асимметрия унаследована от исходных
данных, не признак ошибки в этом модуле.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, integro_grid


PF_COLS: tuple[str, ...] = (
    "pf_gr_vz", "pf_gr_tdr", "pf_gr_asig",
    "pf_gr_up1", "pf_gr_up2", "pf_gr_up5", "pf_gr_up10",
    "pf_mag_vz", "pf_mag_tdr", "pf_mag_asig",
    "pf_mag_up1", "pf_mag_up2", "pf_mag_up5", "pf_mag_up10", "pf_mag_rtp",
    "pf_gm_corr",
)


def _taper2d(shape: tuple[int, int], frac: float) -> np.ndarray:
    """Окно Тьюки по обеим осям — гасит поле к краю массива перед БПФ."""
    from scipy.signal.windows import tukey

    ny, nx = shape
    wy = tukey(ny, frac) if ny > 1 else np.ones(ny)
    wx = tukey(nx, frac) if nx > 1 else np.ones(nx)
    return np.outer(wy, wx)


def _fill_nan(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Заполнить NaN ближайшим валидным значением (БПФ не терпит дыр)."""
    valid = np.isfinite(arr)
    if valid.all():
        return arr.astype(np.float64), valid
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    return filled.astype(np.float64), valid


def _wavenumbers(shape: tuple[int, int], dx: float, dy: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Угловые волновые числа ``(KX, KY, K)`` в географических осях.

    ``KX`` — вдоль столбцов (восток, знак верный по построению сетки).
    Строки сетки идут с севера на юг (:meth:`integro_grid.GridMeta.cell_centers`),
    поэтому частота вдоль строк меняет знак на "географический север".
    """
    ny, nx = shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_row = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY_row = np.meshgrid(kx, ky_row)
    KY = -KY_row
    K = np.hypot(KX, KY)
    return KX, KY, K


def _prepare_spectrum(field: np.ndarray, dx: float, dy: float, taper_frac: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    filled, valid = _fill_nan(field)
    mean = float(filled.mean())
    anom = (filled - mean) * _taper2d(filled.shape, taper_frac)
    KX, KY, K = _wavenumbers(filled.shape, dx, dy)
    F = np.fft.fft2(anom)
    return F, KX, KY, K, mean, valid


def _apply_filter(F: np.ndarray, filt: np.ndarray) -> np.ndarray:
    filt = np.nan_to_num(filt, nan=0.0)
    return np.fft.ifft2(F * filt).real


def vertical_derivative(field: np.ndarray, dx: float, dy: float,
                         taper_frac: float = config.PF_TAPER_FRAC) -> np.ndarray:
    """Первая вертикальная производная ``Vz`` (спектр x ``|k|``)."""
    F, _, _, K, _, valid = _prepare_spectrum(field, dx, dy, taper_frac)
    out = _apply_filter(F, K).astype(np.float32)
    out[~valid] = np.nan
    return out


def tilt_derivative(field: np.ndarray, dx: float, dy: float,
                     taper_frac: float = config.PF_TAPER_FRAC) -> np.ndarray:
    """Tilt-derivative, радианы в [-pi/2, pi/2] — ноль на кромке источника."""
    F, KX, KY, K, _, valid = _prepare_spectrum(field, dx, dy, taper_frac)
    vz = _apply_filter(F, K)
    dfdx = _apply_filter(F, 1j * KX)
    dfdy = _apply_filter(F, 1j * KY)
    thg = np.hypot(dfdx, dfdy)
    with np.errstate(invalid="ignore"):
        tdr = np.arctan2(vz, thg).astype(np.float32)
    tdr[~valid] = np.nan
    return tdr


def analytic_signal(field: np.ndarray, dx: float, dy: float,
                     taper_frac: float = config.PF_TAPER_FRAC) -> np.ndarray:
    """Амплитуда аналитического сигнала, максимум над кромкой источника."""
    F, KX, KY, K, _, valid = _prepare_spectrum(field, dx, dy, taper_frac)
    vz = _apply_filter(F, K)
    dfdx = _apply_filter(F, 1j * KX)
    dfdy = _apply_filter(F, 1j * KY)
    asig = np.sqrt(dfdx ** 2 + dfdy ** 2 + vz ** 2).astype(np.float32)
    asig[~valid] = np.nan
    return asig


def upward_continuation(field: np.ndarray, dx: float, dy: float, height_m: float,
                         taper_frac: float = config.PF_TAPER_FRAC) -> np.ndarray:
    """Продолжение поля вверх на ``height_m`` (спектр x ``exp(-|k|*h)``)."""
    F, _, _, K, mean, valid = _prepare_spectrum(field, dx, dy, taper_frac)
    filt = np.exp(-K * height_m)
    out = (_apply_filter(F, filt) + mean).astype(np.float32)
    out[~valid] = np.nan
    return out


def reduce_to_pole(field: np.ndarray, dx: float, dy: float,
                    inclination_deg: float, declination_deg: float,
                    taper_frac: float = config.PF_TAPER_FRAC) -> np.ndarray:
    """Приведение магнитной аномалии к полюсу (индуцированная намагниченность)."""
    F, KX, KY, K, _, valid = _prepare_spectrum(field, dx, dy, taper_frac)
    incl = np.deg2rad(inclination_deg)
    decl = np.deg2rad(declination_deg)
    a = KX * np.cos(incl) * np.cos(decl) + KY * np.cos(incl) * np.sin(decl)
    theta_f = 1j * a + K * np.sin(incl)
    with np.errstate(divide="ignore", invalid="ignore"):
        filt = np.where(K > 0, (K.astype(complex) ** 2) / (theta_f ** 2), 0.0)
    out = _apply_filter(F, filt).astype(np.float32)
    out[~valid] = np.nan
    return out


def moving_correlation(a: np.ndarray, b: np.ndarray, win_px: int) -> np.ndarray:
    """Скользящая (локальная) корреляция Пирсона двух полей, окно ``win_px`` ячеек."""
    from scipy.ndimage import uniform_filter

    fa, valid_a = _fill_nan(a)
    fb, valid_b = _fill_nan(b)
    valid = valid_a & valid_b

    ma = uniform_filter(fa, win_px)
    mb = uniform_filter(fb, win_px)
    maa = uniform_filter(fa * fa, win_px)
    mbb = uniform_filter(fb * fb, win_px)
    mab = uniform_filter(fa * fb, win_px)
    cov = mab - ma * mb
    va = maa - ma ** 2
    vb = mbb - mb ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(va * vb)
    corr = np.clip(corr, -1.0, 1.0).astype(np.float32)
    corr[~valid] = np.nan
    return corr


def potential_field_features(target_meta: integro_grid.GridMeta,
                              proj4: str | None = None) -> pd.DataFrame:
    """Таблица ``PF_COLS`` на ячейках целевой сетки (C-порядок, как датасет).

    Все трансформанты считаются на нативной сетке ``грав_маг.pgrid`` (БПФ
    требует запаса от края целевого листа, см. докстринг модуля), затем
    билинейно ресемплируются на целевую сетку.
    """
    native_meta, arrays = integro_grid.load_pgrid_dataset(config.GRAVMAG_PGRID)
    proj4 = proj4 or integro_grid.read_grid_proj4(config.GRAVMAG_PGRID)
    dx, dy = native_meta.dx, native_meta.dy

    gr = arrays["gr_all"].astype(np.float64)
    mag = arrays["mg_all"].astype(np.float64)

    native: dict[str, np.ndarray] = {
        "gr_vz": vertical_derivative(gr, dx, dy),
        "gr_tdr": tilt_derivative(gr, dx, dy),
        "gr_asig": analytic_signal(gr, dx, dy),
        "mag_vz": vertical_derivative(mag, dx, dy),
        "mag_tdr": tilt_derivative(mag, dx, dy),
        "mag_asig": analytic_signal(mag, dx, dy),
        "mag_rtp": reduce_to_pole(mag, dx, dy, config.PF_INCLINATION_DEG,
                                   config.PF_DECLINATION_DEG),
    }
    for h in config.PF_UPWARD_HEIGHTS_M:
        suffix = f"{h / 1000.0:g}"
        native[f"gr_up{suffix}"] = upward_continuation(gr, dx, dy, h)
        native[f"mag_up{suffix}"] = upward_continuation(mag, dx, dy, h)

    gr_res = arrays["gr_ost_35"].astype(np.float64)
    mag_res = arrays["mag_ost_35"].astype(np.float64)
    native["gm_corr"] = moving_correlation(gr_res, mag_res, config.PF_CORR_WIN_PX)

    out = {}
    for name, arr in native.items():
        dst = integro_grid.resample_to_grid(
            arr.astype(np.float32), native_meta, target_meta, "bilinear", proj4)
        out["pf_" + name] = dst.ravel()
    return pd.DataFrame(out)[list(PF_COLS)]
