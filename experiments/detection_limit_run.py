"""Шаг 0 «чистка стекла»: кривая предела обнаружения на реальных композитах листа.

ЧТО ЭТОТ ПРОГОН ОТВЕЧАЕТ. Задачи №8 и №2 закрылись отрицательно, но не различили
два диагноза: «разлома нет» и «разлом есть, но не виден». Наблюдением их не
различить — обе гипотезы дают один и тот же отрицательный результат. Различает
контролируемая врезка: в реальный композит вставляется синтетический разлом
известного контраста, и меряется, с какого контраста конвейер начинает его
находить. Результат — ПОРОГ в долях разброса яркости листа, и рядом с ним
контраст НАСТОЯЩИХ закартированных разломов в тех же единицах. Их сравнение и
есть ответ:

* реальный контраст ВЫШЕ порога -> конвейер обязан был видеть разломы и не
  увидел; значит дело в конвейере, и у чистки входа (топокоррекция, отношения
  каналов, направленные фильтры) есть измеримый запас;
* реальный контраст НИЖЕ порога -> разломы физически не проявлены в яркости на
  этом листе с этим разрешением; тереть стекло бессмысленно, и отрицательный
  результат задач №8/№2 получает физическое объяснение вместо методического.

ЭТО ПРИБОР, А НЕ ГИПОТЕЗА. Ни одного p-value против карты разломов здесь не
считается, точки заверки не участвуют, поэтому знаменатель Бонферрони
(``config.FZ_N_TESTS_PRIOR``) прогон НЕ УВЕЛИЧИВАЕТ. Карта разломов используется
один раз и описательно — в блоке калибровки.

БЛОКИ (порядок фиксирован до прогона, как CANDIDATES в fracture_zones_run):

* **калибровка** — контраст коридоров ``glub_raz_nw``/``glub_r_nw`` против
  кольца вокруг них, в долях ``robust_sigma`` композита;
* **A. свип амплитуда x ширина** при длине ``DL_LENGTH_PRIMARY`` и профиле
  ``scarp`` — основной результат;
* **B. свип амплитуда x профиль** при ширине ``DL_WIDTH_PRIMARY`` — проверка,
  что вывод не держится на выборе профиля;
* **C. свип амплитуда x длина** — только по флагу ``--lengths``: окна под 10 км
  вчетверо дороже, а вопрос вторичный.

ИЗМЕРЕНО В БЛОКЕ B (прогон 25.08.2026), против ожидания, записанного здесь до
прогона: жёлоб обнаруживается ЛЕГЧЕ уступа, причём и Кэнни тоже (у Sentinel-1
предел Хафа 1.56 против 2.28, у когерентности 300 м 2.61 против 3.07). Ожидание
было «Кэнни — детектор края, уступ обязан даваться легче»; оно неверно, и прибор
здесь ни при чём. Причина в параметризации: при равной АМПЛИТУДЕ жёлоб высотой A
и полушириной w/2 даёт градиент вдвое круче, чем уступ с полным перепадом A на
ту же ширину, и предъявляет детектору края ДВА склона вместо одного. То есть
«амплитуда 1.0» у двух профилей — равный контраст, но не равная градиентная
энергия. Вывод ветки от этого не зависит (оба профиля дают предел много выше
реального контраста), но сравнивать реальность нужно с профилем ``ridge``:
:func:`src.detection_limit.corridor_contrast` меряет разность медиан коридора и
кольца, а это жёлобная величина, не уступная.

ПРО СЛЕПОТУ ``coh_3km``. В основном свипе верхний масштаб тензора не обнаруживает
НИЧЕГО ни на одной ширине и ни у одного источника даже при амплитуде 4 sigma
(0 из 16 комбинаций). Читать это как «масштаб 3 км бесполезен» НЕЛЬЗЯ без блока
C: длина врезки в основном свипе равна ``DL_LENGTH_PRIMARY`` = 3 км, то есть
гауссиана окна ровно накрывает объект целиком и обязана его размазать. Вопрос
«виден ли на 3 км объект ДЛИННЕЕ окна» решает только свип по длине.

ПАРНЫЙ ДИЗАЙН. Площадки (центры и азимуты врезок) выбираются ОДИН раз на
источник и переиспользуются на всех амплитудах, включая нулевую. Поэтому кривая
и её нуль-модель стоят на одних и тех же участках листа, и разница между ними не
может быть следствием того, что нулю достался спокойный угол.

АЗИМУТЫ ВРЕЗОК — не случайные, а две реальные системы листа: СЗ 145° и СВ 45°
(пики роз карты разломов, измерены в experiments/lineaments_sat_run.py). Мерить
обнаружимость под случайным азимутом значило бы отвечать на вопрос, который
никто не задавал.

Выход: ``outputs/metrics/detection_limit_raw.csv`` (все врезки),
``detection_limit_curve.csv`` (доли обнаружений и пороги),
``detection_limit_calibration.csv`` (контраст настоящих разломов),
``outputs/detection_limit.png``.

Запуск из корня: ``python -m experiments.detection_limit_run``
(``--quick`` — сокращённая сетка для проверки, что прогон вообще едет;
``--sources s1,psr`` — подмножество съёмок; ``--lengths`` — добавить блок C).

РЕЗУЛЬТАТ (25.08.2026, все четыре источника, 60 врезок на комбинацию, 6960 врезок
в сыром файле; композиты полностью из локального кэша).

СНАЧАЛА ПРОВЕРКА САМОГО ПРИБОРА. Частота ложной тревоги на нулевой амплитуде
равна 0.05 у всех трёх когерентностей на всех источниках и 0.033-0.05 у Хафа
(ниже из-за совпадений на целочисленных счётчиках). Это в точности заложенное
``DL_NULL_Q`` = 0.95, то есть нуль-модель откалибрована и порог означает ровно
то, что должен.

ПРЕДЕЛ ОБНАРУЖЕНИЯ, доли sigma фона (амплитуда, на которой доля обнаружений
берёт 0.5; лучшее по каждому источнику, ширина 100 м, профиль scarp):
Sentinel-1 1.52, PALSAR-2 1.56, Sentinel-2 1.78, Landsat 1.81 — все через Хаф,
кроме Sentinel-2, где чуть лучше ``coh_300``. Профиль ridge при ширине 500 м даёт
1.48-2.13 у Хафа. Лучшая комбинация всей таблицы — 1.48.

КОНТРАСТ НАСТОЯЩИХ КОРИДОРОВ КАРТЫ в тех же единицах: 0.086-0.194 sigma по СЗ
(максимум у Sentinel-2 и Sentinel-1) и 0.003-0.089 по СВ. То есть реальные
разломы В 8-20 РАЗ СЛАБЕЕ порога, ниже которого конвейер их не видит ни одним
детектором, ни на одной ширине, ни на одном источнике.

ВЫВОД ВЕТКИ: диагноз ФИЗИЧЕСКИЙ, а не методический. Разломы этого листа не
проявлены в яркости на рабочем разрешении 100 м — не «конвейер плох», а «сигнала
такой величины в этих данных нет». Отрицательные результаты задач №8 и №2
получают объяснение, независимое от них самих: шаг 0 ПРЕДСКАЗЫВАЕТ их исход,
не пользуясь ни одним их числом. Чистка входа (топокоррекция, отношения каналов,
направленные фильтры) поднимает отношение сигнал/шум в разы, а не на порядок,
поэтому от неё разумно ждать сдвига порога с 1.5 до, скажем, 0.7 sigma — этого
не хватает. Смысл имеет только смена ВЕЛИЧИНЫ (не яркость: интерферометрическая
когерентность, сроки фенологии, потенциальные поля) или РАЗРЕШЕНИЯ.

ЗАВИСИМОСТЬ ОТ ГЕОМЕТРИИ, блок A и блок C:

* по ширине — чем уже, тем лучше: 100 м даёт лучший предел, 1000 м у большинства
  комбинаций не обнаруживается вовсе. Широкий объект при равной амплитуде даёт
  пологий градиент, и оба детектора градиентные;
* по длине — чем длиннее, тем лучше: лучший предел 2.86 sigma при 1 км, 2.28 при
  3 км, 1.47 при 10 км. Но даже разлом длиной 10 км требует 1.5 sigma.

ПРО ВЕРХНИЙ МАСШТАБ ТЕНЗОРА. В блоке A ``coh_3km`` не обнаруживает ничего (0 из
16 комбинаций) — но там длина врезки равна 3 км, то есть окно накрывает объект
целиком, и слепота могла быть артефактом постановки. Блок C это разделил: на
объекте длиной 10 км ``coh_3km`` просыпается (доля обнаружений при 4 sigma растёт
до 0.28 у Landsat и 0.48 у Sentinel-2 против ложной тревоги 0.05), но планку 0.5
не берёт нигде. Значит масштаб 3 км в ``FZ_SCALES_M`` слаб по существу, а не
только был неудачно проверен.

ЧЕГО ЭТОТ ПРОГОН НЕ ЗАКРЫВАЕТ. Калибровка меряет контраст ШИРОКОГО коридора
(8432 ячейки внутри против 5742 в кольце), а не резкого следа разлома внутри
него. Если настоящий след на порядок контрастнее коридора в среднем, оценка
0.19 sigma занижена. Против этого работает согласие с задачей №8 (её Хаф ловил
54-59% карты при допуске 2 км — неотличимо от случайного узора), но прямая
проверка требует вектора самих разломных линий, которого в переданных данных
нет, и остаётся отложенной.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (cell_mask, config, detection_limit as dl,  # noqa: E402
                 fracture_zones as fz, integro_grid, stac_grid)

#: Азимуты врезок — географические пики роз карты разломов (СЗ и СВ системы),
#: измерены в experiments/lineaments_sat_run.py: tect1/СЗ 145°, tect2/СВ 45°.
FAULT_AZIMUTHS_GEO: tuple[float, ...] = (145.0, 45.0)

#: Колонка расстояния до коридора для каждой системы — для блока калибровки.
CALIBRATION_TARGETS: dict[str, str] = {"СЗ (tect1)": "dist_tect1",
                                       "СВ (tect2)": "dist_tect2"}

#: Сокращённая сетка ``--quick``: только чтобы убедиться, что прогон едет.
QUICK_AMPLITUDES: tuple[float, ...] = (0.0, 1.0, 4.0)
QUICK_WIDTHS: tuple[float, ...] = (200.0, 1000.0)
QUICK_REPEATS: int = 8


def _log_factory(t0):
    def log(msg=""):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)
    return log


# --------------------------------------------------------------------------
# Композиты
# --------------------------------------------------------------------------

def _composite(kind: str, meta, log):
    """Серый композит источника: из кэша шага 0, иначе собрать и положить туда.

    Кэшируется ПОСЧИТАННОЕ, а не скачанное (см. cache_paths.DETECTION_LIMIT):
    сведение Sentinel-2 из 180 сцен идёт минутами даже с локального диска, а свип
    перезапускается часто.
    """
    from cache_paths import DETECTION_LIMIT

    path = DETECTION_LIMIT / f"{kind}_gray.npz"
    if path.exists():
        z = np.load(path)
        log(f"  композит из кэша {path.name}: {z['img'].shape}, шаг {float(z['res_m']):.0f} м")
        return z["img"], float(z["res_m"])

    from experiments.lineaments_sat_run import MIN_ITEMS, _fetch_items, _grayscale

    items = _fetch_items(kind, meta)
    if len(items) < MIN_ITEMS:
        log(f"  сцен {len(items)} < {MIN_ITEMS} — источник пропущен")
        return None, None
    img, res_m, _ = _grayscale(kind, meta, items)
    np.savez_compressed(path, img=img, res_m=res_m)
    log(f"  композит собран из {len(items)} сцен и закэширован: {path.name}")
    return img, res_m


# --------------------------------------------------------------------------
# Калибровка: контраст настоящих разломов
# --------------------------------------------------------------------------

def _calibration(kind: str, img, res_m: float, sigma: float, meta, df, pool, log):
    """Контраст закартированных коридоров в тех же единицах, что амплитуда свипа."""
    cells = stac_grid.to_cells({"gray": img}, meta, res_m=res_m, prefix="dl_")
    values = cells["dl_gray"].to_numpy(dtype=float)
    mask = np.zeros(values.shape, dtype=bool)
    mask[pool] = True
    values = np.where(mask, values, np.nan)

    rows = []
    for name, col in CALIBRATION_TARGETS.items():
        if col not in df.columns:
            log(f"  нет колонки {col} — калибровка по «{name}» пропущена")
            continue
        got = dl.corridor_contrast(values, df[col].to_numpy(dtype=float), sigma)
        rows.append({"источник": kind, "система": name, **got})
        log(f"  {name}: контраст медиан {got['contrast_median']:.3f} sigma, "
            f"разброса {got['contrast_spread']:+.3f} sigma "
            f"(ячеек внутри {got['n_inside']:.0f}, в кольце {got['n_ring']:.0f})")
    return rows


# --------------------------------------------------------------------------
# Свипы
# --------------------------------------------------------------------------

def _sites(img, res_m: float, length_m: float, n: int, rng):
    """Площадки врезок: центры плюс азимут, чередующийся между двумя системами."""
    centers = dl.sample_centers(img, length_m, res_m, n, rng)
    axes = [dl.axis_from_geographic(FAULT_AZIMUTHS_GEO[i % len(FAULT_AZIMUTHS_GEO)])
            for i in range(len(centers))]
    return list(zip(centers, axes))


def _null_rows(kind, img, res_m, sites, widths, length_m, scales_m):
    """Нуль-модель: те же площадки без сигнала, все ширины с одного поля.

    При амплитуде 0 врезка — точный no-op, поэтому поле когерентности на
    площадке одно на все ширины (:func:`src.detection_limit.coherence_fields`).
    Хаф от ширины не зависит вовсе и считается один раз.
    """
    rows = []
    for center, axis in sites:
        win, ctr = dl.cut_window(img, center, length_m, res_m)
        fields = dl.coherence_fields(win, res_m, scales_m)
        hough = dl.hough_response(win, ctr, axis, length_m=length_m, res_m=res_m)
        for w in widths:
            resp = dl.tensor_response(win, ctr, axis, length_m=length_m,
                                      width_m=w, res_m=res_m, fields=fields)
            rows.append({"источник": kind, "профиль": "null", "амплитуда": 0.0,
                         "ширина_м": w, "длина_м": length_m,
                         "азимут": dl.geographic_from_axis(axis),
                         **resp, **hough})
    return rows


def _signal_rows(kind, img, res_m, sigma, sites, *, amplitudes, widths, length_m,
                 profile, scales_m, log, label):
    rows = []
    total = len([a for a in amplitudes if a > 0]) * len(widths)
    done = 0
    for amp in [a for a in amplitudes if a > 0]:
        for w in widths:
            for center, axis in sites:
                row = dl.probe_once(img, center, axis, amplitude=amp,
                                    length_m=length_m, width_m=w, res_m=res_m,
                                    sigma=sigma, profile=profile,
                                    scales_m=scales_m)
                rows.append({"источник": kind, "профиль": profile,
                             "амплитуда": amp, "ширина_м": w, "длина_м": length_m,
                             "азимут": dl.geographic_from_axis(axis), **row})
            done += 1
            log(f"    {label}: {done}/{total} (амплитуда {amp}, ширина {w:.0f} м)")
    return rows


# --------------------------------------------------------------------------
# Кривая из сырых откликов
# --------------------------------------------------------------------------

def _curve(raw: pd.DataFrame, detectors: tuple[str, ...]) -> pd.DataFrame:
    """Сырые отклики -> доли обнаружений и предел, по детектору и по комбинации.

    Порог берётся из строк ``профиль == "null"`` С ТОЙ ЖЕ шириной и длиной:
    нуль-модель у каждой геометрии своя, потому что след вставки, по которому
    меряется отклик, у разных ширин разный.
    """
    out = []
    is_null = raw["профиль"].astype(str).str.startswith("null")
    null = raw[is_null]
    signal = raw[~is_null]
    keys = ["источник", "профиль", "ширина_м", "длина_м"]
    for (src, prof, w, ln), grp in signal.groupby(keys, sort=False):
        ref = null[(null["источник"] == src) & (null["ширина_м"] == w)
                   & (null["длина_м"] == ln)]
        # Нуль-модель, размеченная по профилю (``null:ridge``), берётся своя.
        # Нужна там, где сама СТАТИСТИКА зависит от профиля — у согласованного
        # фильтра line_accumulation формула для жёлоба и уступа разная, и общий
        # нуль сравнивал бы разные величины. Прогон шага 0 размечает нуль просто
        # как ``null``, и для него ветка не срабатывает.
        own = ref[ref["профиль"] == f"null:{prof}"]
        if not own.empty:
            ref = own
        for det in detectors:
            if det not in raw.columns:
                continue
            thr = dl.detection_threshold(ref[det].to_numpy(dtype=float))
            amps, rates = [], []
            for amp, sub in grp.groupby("амплитуда", sort=True):
                amps.append(float(amp))
                rates.append(dl.detection_rate(sub[det].to_numpy(dtype=float), thr))
            # Нулевая амплитуда входит в кривую как точка (0, доля ложных тревог):
            # без неё кривая начиналась бы неизвестно откуда.
            fa = dl.detection_rate(ref[det].to_numpy(dtype=float), thr)
            amps, rates = [0.0, *amps], [fa, *rates]
            out.append({
                "источник": src, "профиль": prof, "ширина_м": w, "длина_м": ln,
                "детектор": det, "порог_нуля": thr, "ложная_тревога": fa,
                "предел_ампл": dl.limit_from_curve(np.array(amps), np.array(rates)),
                **{f"доля@{a:g}": r for a, r in zip(amps, rates)},
            })
    return pd.DataFrame(out)


def _plot(curve: pd.DataFrame, calib: pd.DataFrame, out_png: Path, log) -> None:
    """Кривые обнаружения по источникам; вертикаль — контраст настоящих разломов."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = curve[(curve["профиль"] == "scarp")
                 & (curve["длина_м"] == config.DL_LENGTH_PRIMARY)]
    if main.empty:
        log("нечего рисовать: основной блок пуст")
        return
    sources = list(dict.fromkeys(main["источник"]))
    dets = list(dict.fromkeys(main["детектор"]))
    fig, axes = plt.subplots(len(dets), len(sources), squeeze=False,
                             figsize=(3.6 * len(sources), 3.2 * len(dets)),
                             sharex=True, sharey=True)
    # Порядок колонок в DataFrame задан объединением ключей словарей по строкам и
    # амплитуде не соответствует — без сортировки линия шла бы зигзагом.
    amp_cols = sorted((c for c in main.columns if c.startswith("доля@")),
                      key=lambda c: float(c.split("@")[1]))
    amps = [float(c.split("@")[1]) for c in amp_cols]
    for i, det in enumerate(dets):
        for j, src in enumerate(sources):
            ax = axes[i][j]
            sub = main[(main["детектор"] == det) & (main["источник"] == src)]
            for _, r in sub.iterrows():
                ax.plot(amps, [r[c] for c in amp_cols], marker="o", lw=1.4,
                        label=f"{r['ширина_м']:.0f} м")
            real = calib[calib["источник"] == src]["contrast_median"]
            if len(real) and np.isfinite(real).any():
                ax.axvline(float(np.nanmax(real)), color="crimson", ls="--", lw=1.2)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(src, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{det}\nдоля обнаружений", fontsize=8)
            if i == len(dets) - 1:
                ax.set_xlabel("амплитуда, доли sigma фона", fontsize=8)
            ax.grid(alpha=0.3)
    axes[0][-1].legend(title="ширина", fontsize=7, title_fontsize=7)
    fig.suptitle("Шаг 0: предел обнаружения врезанного разлома на реальных композитах\n"
                 "красная штриховая — контраст настоящих коридоров карты разломов",
                 fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"карта: {out_png}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default="s2,l8,s1,psr",
                    help="съёмки через запятую (s2,l8,s1,psr)")
    ap.add_argument("--quick", action="store_true",
                    help="сокращённая сетка: проверить, что прогон едет")
    ap.add_argument("--lengths", action="store_true",
                    help="добавить блок C — свип по длине (дорогой)")
    args = ap.parse_args()

    t0 = time.time()
    log = _log_factory(t0)

    amplitudes = QUICK_AMPLITUDES if args.quick else config.DL_AMPLITUDES
    widths = QUICK_WIDTHS if args.quick else config.DL_WIDTHS_M
    n_rep = QUICK_REPEATS if args.quick else config.DL_N_REPEATS
    scales_m = config.FZ_SCALES_M
    detectors = (*(f"coh_{fz._scale_tag(s)}" for s in scales_m), "hough_n_match")

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    df = pd.read_parquet(sorted(config.PROCESSED_DIR.glob("dataset_v*.parquet"))[-1])
    pool = np.where(cell_mask.build_valid_mask(df))[0]
    log(f"сетка {meta.prf}x{meta.pic}, валидных ячеек {pool.size}")
    log(f"свип: амплитуды {amplitudes}, ширины {widths} м, "
        f"длина {config.DL_LENGTH_PRIMARY:.0f} м, врезок на комбинацию {n_rep}")
    log("знаменатель Бонферрони НЕ увеличивается: ветка не считает p-value "
        "против карты разломов и не касается точек заверки")

    raw_rows: list[dict] = []
    calib_rows: list[dict] = []
    for kind in [s.strip() for s in args.sources.split(",") if s.strip()]:
        log(f"--- {kind} ---")
        img, res_m = _composite(kind, meta, log)
        if img is None:
            continue
        sigma = dl.robust_sigma(img)
        log(f"  sigma композита (MAD*1.4826) = {sigma:.4g}, "
            f"валидных пикселей {np.isfinite(img).mean():.1%}")
        calib_rows += _calibration(kind, img, res_m, sigma, meta, df, pool, log)

        rng = np.random.default_rng(config.DL_SEED)
        sites = _sites(img, res_m, config.DL_LENGTH_PRIMARY, n_rep, rng)
        log(f"  площадок врезки {len(sites)} (парный дизайн: те же на всех амплитудах)")

        # Ширина блока B добавляется в нуль-модель явно: порог берётся по ТОЙ ЖЕ
        # геометрии, что сигнал, а в сокращённой сетке --quick список ширин
        # свипа её не содержит.
        null_widths = tuple(dict.fromkeys((*widths, config.DL_WIDTH_PRIMARY)))
        raw_rows += _null_rows(kind, img, res_m, sites, null_widths,
                               config.DL_LENGTH_PRIMARY, scales_m)
        log(f"    нуль-модель посчитана (ширины {null_widths})")
        raw_rows += _signal_rows(kind, img, res_m, sigma, sites,
                                 amplitudes=amplitudes, widths=widths,
                                 length_m=config.DL_LENGTH_PRIMARY,
                                 profile="scarp", scales_m=scales_m, log=log,
                                 label="блок A")
        raw_rows += _signal_rows(kind, img, res_m, sigma, sites,
                                 amplitudes=amplitudes,
                                 widths=(config.DL_WIDTH_PRIMARY,),
                                 length_m=config.DL_LENGTH_PRIMARY,
                                 profile="ridge", scales_m=scales_m, log=log,
                                 label="блок B")

        if args.lengths:
            for ln in config.DL_LENGTHS_M:
                if ln == config.DL_LENGTH_PRIMARY:
                    continue
                sites_l = _sites(img, res_m, ln, n_rep,
                                 np.random.default_rng(config.DL_SEED))
                raw_rows += _null_rows(kind, img, res_m, sites_l,
                                       (config.DL_WIDTH_PRIMARY,), ln, scales_m)
                raw_rows += _signal_rows(kind, img, res_m, sigma, sites_l,
                                         amplitudes=amplitudes,
                                         widths=(config.DL_WIDTH_PRIMARY,),
                                         length_m=ln, profile="scarp",
                                         scales_m=scales_m, log=log,
                                         label=f"блок C ({ln:.0f} м)")

    if not raw_rows:
        log("ни один источник не собран — нечего считать")
        return

    raw = pd.DataFrame(raw_rows)
    curve = _curve(raw, detectors)
    calib = pd.DataFrame(calib_rows)

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("detection_limit_raw", raw),
                        ("detection_limit_curve", curve),
                        ("detection_limit_calibration", calib)):
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    log(f"сохранено: {out_dir / 'detection_limit_curve.csv'} и рядом")

    _report(curve, calib, detectors)
    _plot(curve, calib, ROOT / "outputs" / "detection_limit.png", log)


def _report(curve: pd.DataFrame, calib: pd.DataFrame, detectors) -> None:
    """Итог: предел по каждому детектору против контраста настоящих разломов."""
    print(f"\n{'=' * 84}\nШАГ 0: ПРЕДЕЛ ОБНАРУЖЕНИЯ (в долях sigma фона)\n{'=' * 84}")
    if not calib.empty:
        print("\nКонтраст НАСТОЯЩИХ коридоров карты разломов:")
        print(f"{'источник':<10}{'система':<14}{'медианы':>10}{'разброса':>11}")
        for _, r in calib.iterrows():
            print(f"{r['источник']:<10}{r['система']:<14}"
                  f"{r['contrast_median']:>10.3f}{r['contrast_spread']:>11.3f}")

    main = curve[(curve["профиль"] == "scarp")
                 & (curve["длина_м"] == config.DL_LENGTH_PRIMARY)]
    print("\nПредел обнаружения (амплитуда, на которой доля обнаружений берёт 0.5;\n"
          "inf = не обнаруживается нигде в пределах шкалы):")
    print(f"{'источник':<10}{'детектор':<16}{'ширина':>9}{'ложн.тревога':>14}{'предел':>10}")
    for _, r in main.sort_values(["источник", "детектор", "ширина_м"]).iterrows():
        lim = "inf" if not np.isfinite(r["предел_ампл"]) else f"{r['предел_ампл']:.2f}"
        print(f"{r['источник']:<10}{r['детектор']:<16}{r['ширина_м']:>8.0f}м"
              f"{r['ложная_тревога']:>14.2f}{lim:>10}")

    ridge = curve[(curve["профиль"] == "ridge")
                  & (curve["длина_м"] == config.DL_LENGTH_PRIMARY)]
    if not ridge.empty:
        print(f"\nБлок B, профиль ridge (ширина {config.DL_WIDTH_PRIMARY:.0f} м) — "
              "именно с ним сопоставим контраст коридоров:")
        print(f"{'источник':<10}{'детектор':<16}{'ложн.тревога':>14}{'предел':>10}")
        for _, r in ridge.sort_values(["источник", "детектор"]).iterrows():
            lim = "inf" if not np.isfinite(r["предел_ампл"]) else f"{r['предел_ампл']:.2f}"
            print(f"{r['источник']:<10}{r['детектор']:<16}"
                  f"{r['ложная_тревога']:>14.2f}{lim:>10}")

    lengths = sorted(curve["длина_м"].unique())
    if len(lengths) == 1:
        print(f"\nОГОВОРКА: длина врезки одна ({lengths[0]:.0f} м), поэтому "
              "слепота верхнего масштаба тензора\nнеотделима от того, что окно "
              "накрывает объект целиком. Разделяет только --lengths.")

    if calib.empty or main.empty:
        return
    contrasts = calib["contrast_median"].to_numpy(dtype=float)
    # Сравнение идёт с ЛУЧШИМ пределом по всей таблице, включая профиль ridge, а
    # не только с блоком A. Две причины. Первая: corridor_contrast меряет
    # разность медиан коридора и кольца — величина жёлобная, и сопоставлять её
    # надо с жёлобом. Вторая: вопрос ветки — «мог ли хоть какой-нибудь детектор
    # это увидеть», поэтому планка обязана быть самой выгодной для обнаружения,
    # иначе отрицательный вывод получен нечестной планкой.
    at_len = curve[curve["длина_м"] == config.DL_LENGTH_PRIMARY].copy()
    at_len["предел_ампл"] = at_len["предел_ампл"].replace([np.inf], np.nan)
    best = at_len["предел_ампл"].min()
    if np.isfinite(best):
        row = at_len.loc[at_len["предел_ампл"].idxmin()]
        print(f"\nЛучшая комбинация таблицы: {row['источник']}, {row['детектор']}, "
              f"профиль {row['профиль']}, ширина {row['ширина_м']:.0f} м")
    if not np.isfinite(contrasts).any() or not np.isfinite(best):
        # Без обоих чисел вывод не формулируется. Молчать честнее, чем выдать
        # «физический» вердикт там, где просто нечего сравнивать.
        print("\nВЫВОД НЕ ФОРМУЛИРУЕТСЯ: "
              f"{'калибровка пуста' if not np.isfinite(contrasts).any() else ''}"
              f"{' и ' if not np.isfinite(contrasts).any() and not np.isfinite(best) else ''}"
              f"{'предел не взят ни на одной ширине' if not np.isfinite(best) else ''}.")
        return
    real = float(np.nanmax(contrasts))
    print(f"\nЛучший предел по таблице: {best:.2f} sigma; "
          f"контраст настоящих разломов: {real:.2f} sigma.")
    if real > best:
        print("ВЫВОД: настоящие разломы КОНТРАСТНЕЕ порога — конвейер обязан был их\n"
              "видеть. Диагноз методический, у чистки входа есть запас.")
    else:
        print("ВЫВОД: настоящие разломы СЛАБЕЕ порога — в яркости на этом листе и\n"
              "этом разрешении они не проявлены. Диагноз физический, чистка входа\n"
              "яркости не поможет; смысл имеет только смена величины или разрешения.")


if __name__ == "__main__":
    main()
