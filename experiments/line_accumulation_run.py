"""Пункт 1 «мутного стекла»: опускает ли накопление вдоль линии порог обнаружения.

ОТКУДА ВОПРОС. Шаг 0 (experiments/detection_limit_run.py) измерил порог
1.48-1.81 sigma при контрасте настоящих коридоров карты 0.086-0.194 sigma —
разрыв в 8-20 раз — и заодно показал, что единственный рычаг, который порог
реально двигал, это ДЛИНА: 2.86 sigma при 1 км, 2.28 при 3 км, 1.47 при 10 км.
Ширина, источник и профиль давали доли. Оба детектора шага 0 длиной при этом
почти не пользуются: Кэнни решает попиксельно, тензор усредняет в изотропном
окне. Отсюда прогон: те же врезки, те же площадки, тот же нуль — но два
детектора, для которых длина основная величина (:mod:`src.line_accumulation`).

ЧТО СПРАШИВАЕТСЯ, ОДНОЙ СТРОКОЙ: падает ли порог с 1.5 sigma хотя бы к 0.5.
Планка 0.5 названа ДО прогона и не подлежит пересмотру после: она вчетверо
слабее нынешнего порога и всё ещё втрое выше реального контраста коридоров,
то есть даже её взятие не означало бы, что разломы станут видны, — означало бы
только, что направление живое и стоит следующего шага.

ДВА ДЕТЕКТОРА ПО КРАЯМ ШКАЛЫ ЧЕСТНОСТИ, и вместе они разделяют два разных «нет»:

* ``acc_matched`` — согласованный фильтр, которому СООБЩЕНЫ положение, азимут,
  ширина и профиль вставки. Это верхняя граница: лучше не может быть ни один
  линейный детектор, потому что искать уже нечего;
* ``acc_link_span`` — сращивание фрагментов, полностью слепой (о вставке не
  знает ничего, как Хаф шага 0), пороги на пиксель ослаблены ровно вдвое.

Слепой провалился, а граница взяла -> дело в переборе, метод искать стоит.
Провалилась граница -> перебирать нечего, и вся семья «накопление вдоль линии»
закрыта разом — измерением, а не мнением.

ПОБОЧНОЕ ЧИСЛО, КОТОРОЕ ВАЖНЕЕ ГЛАВНОГО. Наивно накопление по ``N`` отсчётам
давит шум в ``sqrt(N)``: 30 отсчётов на 3 км дали бы 5.5 раз, почти закрыв
разрыв. Но ``sqrt(N)`` верен лишь для НЕЗАВИСИМЫХ отсчётов, а у композита есть
своя текстура с километровой корреляцией. Статистика ``acc_matched`` нормирована
под ``|N(0, 1)|``, поэтому отклонение её нуль-модели от табличного квантиля прямо
меряет, сколько независимости есть на самом деле
(:func:`src.line_accumulation.independence_ratio`). Это ограничение ВСЕГО класса
накопительных детекторов, а не одного нашего.

БЛОКИ (порядок зафиксирован до прогона):

* **D. основной** — свип амплитуда x ДЛИНА при ширине ``DL_WIDTH_PRIMARY`` и
  профиле ``ACC_PROFILE_PRIMARY`` (жёлоб — именно с ним сопоставим контраст
  коридоров, см. блок B шага 0);
* **E. контроль профиля** — то же при ``scarp`` и одной длине
  ``DL_LENGTH_PRIMARY``: вывод не должен держаться на выборе профиля.

ПАРНЫЙ ДИЗАЙН И НУЛЬ — как в шаге 0: площадки выбираются один раз на источник и
длину и переиспользуются на всех амплитудах, включая нулевую. Нуль-модель здесь
размечена ПО ПРОФИЛЮ (``null:ridge``/``null:scarp``), потому что формула
согласованного фильтра для жёлоба и уступа разная и общий нуль сравнивал бы
разные величины.

ЭТО ПРИБОР, А НЕ ГИПОТЕЗА. Ни одного p-value против карты разломов, точки
заверки не участвуют, знаменатель Бонферрони НЕ растёт. Карта разломов
используется один раз и описательно — в блоке калибровки, тем же кодом, что в
шаге 0.

Выход: ``outputs/metrics/line_accumulation_raw.csv``,
``line_accumulation_curve.csv``, ``line_accumulation_calibration.csv``,
``outputs/line_accumulation.png``.

Запуск из корня: ``python -m experiments.line_accumulation_run``
(``--quick`` — сокращённая сетка; ``--sources s1,psr`` — подмножество съёмок).

РЕЗУЛЬТАТ (26.08.2026, четыре источника, 60 врезок на комбинацию, 5760 врезок в
сыром файле; композиты из того же кэша, что у шага 0).

ПРОВЕРКА ПРИБОРА. Частота ложной тревоги обоих новых детекторов равна 0.05 на
всех источниках и всех длинах — ровно заложенное ``DL_NULL_Q``. Нуль-модель
откалибрована, порог означает то, что должен.

ПОРОГ ОБНАРУЖЕНИЯ, доли sigma фона, профиль жёлоба, лучшее по источнику:

===========  ======  ======  ======  =========  ==============
детектор     s2      s1      psr     l8         как у шага 0
===========  ======  ======  ======  =========  ==============
coh_300      1.38    1.73    1.91    2.43       изотропное окно
hough        0.98    1.47    1.52    1.87       попиксельно
acc_link     0.89    1.29    1.39    1.64       слепой, +10-15%
acc_matched  0.60    0.62    0.73    1.10       верхняя граница
===========  ======  ======  ======  =========  ==============

ГЛАВНОЕ ЧИСЛО, и оно снято ПРИ РАВНОЙ ДЛИНЕ, чтобы выигрыш детектора не
складывался с выигрышем геометрии: на 10 км согласованный фильтр опускает порог
против Хафа в 1.6-2.4 раза (s2 0.60 против 0.98, s1 0.62 против 1.47,
psr 0.73 против 1.52, l8 1.10 против 1.87). Сквозной итог ветки — лучшее по всей
таблице упало с опубликованных шагом 0 1.48 sigma до 0.60, то есть в 2.5 раза,
но здесь часть выигрыша дала длина, а часть детектор.

Рычаг длины подтверждён: у трёх источников из четырёх предел монотонно падает от
1 к 10 км (s1 2.20 -> 1.37 -> 0.62), у Landsat немонотонно (1.90 -> 2.44 ->
1.10), и это записано как есть, а не сглажено.

ПЛАНКА 0.5, названная ДО прогона, НЕ ВЗЯТА: лучшее 0.60. Отрицательный результат
остаётся отрицательным, планка не двигается.

СЛЕПОЙ ПРОТИВ ГРАНИЦЫ. Сращивание фрагментов выигрывает у Хафа на 11 комбинациях
из 12 и нигде не проигрывает, но выигрыш скромный: чаще 10-20%, максимум 40%
(s2 на 3 км: 0.89 против 1.48), а на s2/10 км эффекта нет вовсе (0.98 против
0.98). Направление устойчиво, порядок величины — нет. Разрыв между слепым и
границей (на 10 км 0.98-1.64 против 0.60-1.10) показывает, что заметная часть
потерь — цена ПЕРЕБОРА, а не отсутствие сигнала.

ПОЧЕМУ НЕ sqrt(N): ГЛАВНЫЙ ПОБОЧНЫЙ РЕЗУЛЬТАТ. Номинально маска линии содержит
52 / 151 / 499 пикселей на 1 / 3 / 10 км. Эффективно, по нуль-модели
согласованного фильтра, независимых отсчётов там 1.4-3.6 / 1.8-5.4 / 10.9-18.1,
то есть 1-7% от номинала. Линия длиной 1-3 км — это фактически ОДИН независимый
отсчёт: композит имеет собственную текстуру с километровой корреляцией, и
накапливать в ней нечего. Между 3 и 10 км эффективное число растёт в 3-7 раз при
росте длины в 3.3 раза, то есть не медленнее линейного, — накопление включается
только после того, как длина превысит масштаб текстуры. Это ограничение ВСЕГО
класса накопительных методов, а не одного нашего детектора, и оно же объясняет,
почему в шаге 0 длина оказалась единственным работающим рычагом.

ЧТО ИЗ ЭТОГО СЛЕДУЕТ КОЛИЧЕСТВЕННО. Чтобы дотянуть 0.60 до реального контраста
коридоров 0.194 sigma, нужен множитель ~10 по числу независимых отсчётов, то
есть прямолинейный разлом длиной около 95 км. Поперечник листа — 77 км. Такой
разлом на лист НЕ ПОМЕЩАЕТСЯ, и это ограничение геометрическое: его не снять ни
выбором детектора, ни чисткой входа.

ИСПРАВЛЕНО ПОСЛЕ ПРОГОНА, честно и намеренно. В ``_report`` до запуска стояла
фраза «провалилась верхняя граница -> весь класс накопительных методов
закрывается». Измерение её не подтвердило: падение в 2.5 раза и не вышедшая на
полку зависимость от длины — это не закрытие класса. Формулировка исправлена,
планка 0.5 оставлена на месте. Тот же порядок действий, что с ожиданием про
Кэнни в блоке B шага 0: неверным оказалось ожидание, а не прибор.

ЧЕГО ЭТОТ ПРОГОН НЕ ЗАКРЫВАЕТ. Верхняя граница считана при ИЗВЕСТНОЙ геометрии
и на прямой линии. Настоящие разломы искривлены, и накопление вдоль кривой
набирает меньше, чем вдоль прямой той же длины, — то есть 95 км это оценка
снизу, оптимистичная. Слепой перебор по азимуту и смещению (полноценное
преобразование Радона) здесь не считался: при непройденной планке у границы он
отвечал бы на вопрос, который уже решён не в его пользу.
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
                 fracture_zones as fz, integro_grid, line_accumulation as acc)
from experiments.detection_limit_run import (  # noqa: E402
    _calibration, _composite, _curve, _log_factory, _sites)

#: Планка, названная до прогона: «направление живое, если порог упал хотя бы сюда».
TARGET_LIMIT: float = 0.5

#: Сокращённая сетка ``--quick`` — только чтобы убедиться, что прогон едет.
QUICK_AMPLITUDES: tuple[float, ...] = (0.0, 1.0, 4.0)
QUICK_LENGTHS: tuple[float, ...] = (config.DL_LENGTH_PRIMARY,)
QUICK_REPEATS: int = 6


def _rows(kind, img, res_m, sigma, sites, *, amplitudes, length_m, profile,
          scales_m, log, label):
    """Все амплитуды одной геометрии, включая нулевую (нуль-модель).

    Нуль считается ТЕМ ЖЕ вызовом, что сигнал, а не отдельной веткой кода: при
    амплитуде 0 врезка — точный no-op, и любое расхождение путей означало бы, что
    порог берётся не с той же величины, которую он потом отсекает.
    """
    rows = []
    total = len(amplitudes)
    for k, amp in enumerate(sorted(amplitudes), start=1):
        for center, axis in sites:
            row = acc.probe_once(img, center, axis, amplitude=amp,
                                 length_m=length_m, width_m=config.DL_WIDTH_PRIMARY,
                                 res_m=res_m, sigma=sigma, profile=profile,
                                 scales_m=scales_m)
            rows.append({"источник": kind,
                         "профиль": f"null:{profile}" if amp == 0.0 else profile,
                         "амплитуда": amp, "ширина_м": config.DL_WIDTH_PRIMARY,
                         "длина_м": length_m,
                         "азимут": dl.geographic_from_axis(axis), **row})
        log(f"    {label}: {k}/{total} (амплитуда {amp})")
    return rows


def _plot(curve: pd.DataFrame, calib: pd.DataFrame, out_png: Path, log) -> None:
    """Кривые обнаружения по длинам; вертикаль — контраст настоящих разломов."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = curve[curve["профиль"] == config.ACC_PROFILE_PRIMARY]
    if main.empty:
        log("нечего рисовать: основной блок пуст")
        return
    sources = list(dict.fromkeys(main["источник"]))
    dets = list(dict.fromkeys(main["детектор"]))
    fig, axes = plt.subplots(len(dets), len(sources), squeeze=False,
                             figsize=(3.6 * len(sources), 2.9 * len(dets)),
                             sharex=True, sharey=True)
    amp_cols = sorted((c for c in main.columns if c.startswith("доля@")),
                      key=lambda c: float(c.split("@")[1]))
    amps = [float(c.split("@")[1]) for c in amp_cols]
    for i, det in enumerate(dets):
        for j, src in enumerate(sources):
            ax = axes[i][j]
            sub = main[(main["детектор"] == det) & (main["источник"] == src)]
            for _, r in sub.sort_values("длина_м").iterrows():
                ax.plot(amps, [r[c] for c in amp_cols], marker="o", lw=1.4,
                        label=f"{r['длина_м'] / 1000:.0f} км")
            real = calib[calib["источник"] == src]["contrast_median"]
            if len(real) and np.isfinite(real).any():
                ax.axvline(float(np.nanmax(real)), color="crimson", ls="--", lw=1.2)
            ax.axhline(0.5, color="0.5", ls=":", lw=1.0)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(src, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{det}\nдоля обнаружений", fontsize=8)
            if i == len(dets) - 1:
                ax.set_xlabel("амплитуда, доли sigma фона", fontsize=8)
            ax.grid(alpha=0.3)
    axes[0][-1].legend(title="длина", fontsize=7, title_fontsize=7)
    fig.suptitle("Накопление вдоль линии: порог обнаружения по длине врезки\n"
                 "красная штриховая — контраст настоящих коридоров карты разломов",
                 fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"карта: {out_png}")


def _report(curve: pd.DataFrame, calib: pd.DataFrame, raw: pd.DataFrame,
            sheet_km: float = float("nan")) -> None:
    """Итог: пороги по детекторам и длинам против реального контраста."""
    print(f"\n{'=' * 88}\nНАКОПЛЕНИЕ ВДОЛЬ ЛИНИИ: ПОРОГ ОБНАРУЖЕНИЯ (доли sigma фона)\n{'=' * 88}")
    if not calib.empty:
        print("\nКонтраст НАСТОЯЩИХ коридоров карты разломов:")
        print(f"{'источник':<10}{'система':<14}{'медианы':>10}{'разброса':>11}")
        for _, r in calib.iterrows():
            print(f"{r['источник']:<10}{r['система']:<14}"
                  f"{r['contrast_median']:>10.3f}{r['contrast_spread']:>11.3f}")

    main = curve[curve["профиль"] == config.ACC_PROFILE_PRIMARY]
    print(f"\nБлок D, профиль {config.ACC_PROFILE_PRIMARY} "
          f"(inf = не обнаруживается нигде в пределах шкалы):")
    print(f"{'источник':<10}{'детектор':<16}{'длина':>8}{'ложн.тревога':>14}{'предел':>10}")
    for _, r in main.sort_values(["источник", "детектор", "длина_м"]).iterrows():
        lim = "inf" if not np.isfinite(r["предел_ампл"]) else f"{r['предел_ампл']:.2f}"
        print(f"{r['источник']:<10}{r['детектор']:<16}{r['длина_м'] / 1000:>7.0f}км"
              f"{r['ложная_тревога']:>14.2f}{lim:>10}")

    ctrl = curve[curve["профиль"] == "scarp"]
    if not ctrl.empty:
        print("\nБлок E, контроль профиля (scarp):")
        print(f"{'источник':<10}{'детектор':<16}{'длина':>8}{'предел':>10}")
        for _, r in ctrl.sort_values(["источник", "детектор", "длина_м"]).iterrows():
            lim = "inf" if not np.isfinite(r["предел_ампл"]) else f"{r['предел_ампл']:.2f}"
            print(f"{r['источник']:<10}{r['детектор']:<16}"
                  f"{r['длина_м'] / 1000:>7.0f}км{lim:>10}")

    # --- сколько независимости на самом деле есть у фона ---
    mt = main[main["детектор"] == "acc_matched"]
    if not mt.empty and "acc_nominal_n" in raw.columns:
        print("\nСКОЛЬКО НЕЗАВИСИМЫХ ОТСЧЁТОВ РЕАЛЬНО ДАЁТ ЛИНИЯ\n"
              "(номинально — число пикселей маски; эффективно — с поправкой на\n"
              "связность фона, измеренной по нуль-модели согласованного фильтра):")
        print(f"{'источник':<10}{'длина':>8}{'номинально':>12}{'эффективно':>12}{'доля':>8}")
        for _, r in mt.sort_values(["источник", "длина_м"]).iterrows():
            sub = raw[(raw["источник"] == r["источник"])
                      & (raw["длина_м"] == r["длина_м"])
                      & (raw["профиль"] == f"null:{config.ACC_PROFILE_PRIMARY}")]
            nominal = float(np.nanmedian(sub["acc_nominal_n"])) if len(sub) else np.nan
            frac = acc.independence_ratio(r["порог_нуля"])
            eff = nominal * frac
            print(f"{r['источник']:<10}{r['длина_м'] / 1000:>7.0f}км{nominal:>12.0f}"
                  f"{eff:>12.1f}{frac:>8.3f}")

    # --- вердикт ---
    if calib.empty or main.empty:
        print("\nВЫВОД НЕ ФОРМУЛИРУЕТСЯ: нет калибровки или пустой основной блок.")
        return
    acc_only = main[main["детектор"].isin(("acc_matched", "acc_link_span"))].copy()
    acc_only["предел_ампл"] = acc_only["предел_ампл"].replace([np.inf], np.nan)
    if not np.isfinite(acc_only["предел_ампл"]).any():
        print("\nВЫВОД: ни один накопительный детектор не берёт планку 0.5 доли\n"
              "обнаружений НИ НА ОДНОЙ амплитуде шкалы. Накопление вдоль линии не\n"
              "даёт ничего сверх шага 0.")
        return
    best_row = acc_only.loc[acc_only["предел_ампл"].idxmin()]
    best = float(best_row["предел_ампл"])
    real = float(np.nanmax(calib["contrast_median"].to_numpy(dtype=float)))
    print(f"\nЛучший накопительный предел: {best:.2f} sigma "
          f"({best_row['источник']}, {best_row['детектор']}, "
          f"длина {best_row['длина_м'] / 1000:.0f} км).")
    print(f"Порог шага 0 без накопления: 1.48 sigma; контраст настоящих "
          f"разломов: {real:.3f} sigma.")
    if best <= real:
        print("ВЫВОД: накопление довело порог НИЖЕ реального контраста — разломы\n"
              "принципиально обнаружимы, и ветку надо открывать заново.")
    elif best <= TARGET_LIMIT:
        print(f"ВЫВОД: планка {TARGET_LIMIT} взята — направление живое. До реального\n"
              f"контраста ещё {best / real:.1f} раза, но рычаг длины работает и\n"
              "следующий шаг (слепой перебор по азимуту, большая длина) осмыслен.")
    else:
        print(f"ВЫВОД: планка {TARGET_LIMIT} НЕ взята (лучшее {best:.2f}), разрыв до\n"
              f"реального контраста остаётся {best / real:.1f}-кратным.")
        # ВНИМАНИЕ. Здесь до прогона стояла фраза «провалилась верхняя граница,
        # значит весь класс накопительных методов закрывается». ИЗМЕРЕНИЕ ЕЁ НЕ
        # ПОДТВЕРДИЛО и формулировка исправлена: порог упал с 1.48 до 0.60 sigma,
        # то есть в 2.5 раза, и зависимость от длины НЕ вышла на полку (у трёх
        # источников из четырёх предел монотонно падает до самой большой длины
        # шкалы). Закрывать класс на таких числах было бы подгонкой вывода под
        # заранее написанный текст. Планка при этом не двигается: 0.5 названа до
        # прогона, 0.60 её не берёт, и это остаётся отрицательным результатом.
        print("Но закрыть на этом ВЕСЬ класс нельзя: зависимость от длины не вышла\n"
              "на полку, и предел продолжает падать до конца шкалы. Вопрос поэтому\n"
              "не «работает ли накопление» (работает), а «хватит ли длины».")
        if np.isfinite(sheet_km):
            need_km = best_row["длина_м"] / 1000.0 * (best / real) ** 2
            print(f"\nСКОЛЬКО ДЛИНЫ НЕ ХВАТАЕТ. Эффективное число независимых отсчётов\n"
                  f"растёт с длиной не медленнее линейного (замерено между 3 и 10 км),\n"
                  f"а накопление давит шум как его корень. Чтобы дотянуть {best:.2f} до\n"
                  f"{real:.3f} sigma, нужен множитель {(best / real) ** 2:.0f} по отсчётам,\n"
                  f"то есть прямолинейный разлом длиной около {need_km:.0f} км.")
            if need_km > sheet_km:
                print(f"Поперечник листа — {sheet_km:.0f} км. Такой разлом на лист НЕ\n"
                      "ПОМЕЩАЕТСЯ: ограничение геометрическое, и его не снять ни выбором\n"
                      "детектора, ни чисткой входа.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default="s2,l8,s1,psr",
                    help="съёмки через запятую (s2,l8,s1,psr)")
    ap.add_argument("--quick", action="store_true",
                    help="сокращённая сетка: проверить, что прогон едет")
    args = ap.parse_args()

    t0 = time.time()
    log = _log_factory(t0)

    amplitudes = QUICK_AMPLITUDES if args.quick else config.DL_AMPLITUDES
    lengths = QUICK_LENGTHS if args.quick else config.ACC_LENGTHS_M
    n_rep = QUICK_REPEATS if args.quick else config.DL_N_REPEATS
    scales_m = config.FZ_SCALES_M
    detectors = (*(f"coh_{fz._scale_tag(s)}" for s in scales_m),
                 "hough_n_match", "acc_matched", "acc_link_span")

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    df = pd.read_parquet(sorted(config.PROCESSED_DIR.glob("dataset_v*.parquet"))[-1])
    pool = np.where(cell_mask.build_valid_mask(df))[0]
    log(f"сетка {meta.prf}x{meta.pic}, валидных ячеек {pool.size}")
    log(f"свип: амплитуды {amplitudes}, длины {lengths} м, "
        f"ширина {config.DL_WIDTH_PRIMARY:.0f} м, врезок на комбинацию {n_rep}")
    log(f"ослабление конвейера ровно вдвое: Кэнни {config.ACC_EDGE_PCTL:.0f}/"
        f"{config.ACC_EDGE_LOW_PCTL:.0f} перцентиль, отрезок от "
        f"{config.ACC_MIN_LEN_PX} px, порог Хафа {config.ACC_HOUGH_THRESHOLD}")
    log("знаменатель Бонферрони НЕ увеличивается: ветка не считает p-value "
        "против карты разломов и не касается точек заверки")

    raw_rows: list[dict] = []
    calib_rows: list[dict] = []
    sheet_km = float("nan")
    for kind in [s.strip() for s in args.sources.split(",") if s.strip()]:
        log(f"--- {kind} ---")
        img, res_m = _composite(kind, meta, log)
        if img is None:
            continue
        # Поперечник листа нужен вердикту: если требуемая длина разлома больше
        # его, ограничение геометрическое, а не методическое.
        sheet_km = max(img.shape) * res_m / 1000.0
        sigma = dl.robust_sigma(img)
        log(f"  sigma композита (MAD*1.4826) = {sigma:.4g}, "
            f"валидных пикселей {np.isfinite(img).mean():.1%}")
        calib_rows += _calibration(kind, img, res_m, sigma, meta, df, pool, log)

        for ln in lengths:
            # Свой генератор на каждую длину: окна разного размера, и общий
            # поток дал бы длинам разные площадки не по замыслу, а по остатку.
            sites = _sites(img, res_m, ln, n_rep, np.random.default_rng(config.DL_SEED))
            log(f"  длина {ln / 1000:.0f} км: площадок {len(sites)}")
            raw_rows += _rows(kind, img, res_m, sigma, sites,
                              amplitudes=amplitudes, length_m=ln,
                              profile=config.ACC_PROFILE_PRIMARY,
                              scales_m=scales_m, log=log,
                              label=f"блок D ({ln / 1000:.0f} км)")

        ctrl_len = config.DL_LENGTH_PRIMARY
        if ctrl_len in tuple(lengths):
            sites = _sites(img, res_m, ctrl_len, n_rep,
                           np.random.default_rng(config.DL_SEED))
            raw_rows += _rows(kind, img, res_m, sigma, sites,
                              amplitudes=amplitudes, length_m=ctrl_len,
                              profile="scarp", scales_m=scales_m, log=log,
                              label="блок E (контроль профиля)")

    if not raw_rows:
        log("ни один источник не собран — нечего считать")
        return

    raw = pd.DataFrame(raw_rows)
    curve = _curve(raw, detectors)
    calib = pd.DataFrame(calib_rows)

    out_dir = ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("line_accumulation_raw", raw),
                        ("line_accumulation_curve", curve),
                        ("line_accumulation_calibration", calib)):
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    log(f"сохранено: {out_dir / 'line_accumulation_curve.csv'} и рядом")

    _report(curve, calib, raw, sheet_km)
    _plot(curve, calib, ROOT / "outputs" / "line_accumulation.png", log)


if __name__ == "__main__":
    main()
