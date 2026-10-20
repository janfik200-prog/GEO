"""Задача №2C: сроки схода снега по Sentinel-1 против карты разломов.

Продолжение ``experiments/fracture_zones_run.py``. Блок 1 там закрыл площадную
ТЕКСТУРУ снимков: единственный кандидат со значимым сырым p (``s1_vv_vh_std``,
p = 0.0008) терял значимость после вычитания рельефа. Здесь измеряется величина
другой природы — не уровень сигнала, а РЕЖИМ: когда сходит снег и сколько
длится мокрая фаза (см. :mod:`src.snowmelt`).

Порядок:

1. по каждому году :data:`config.SM_YEARS` собирается временной ряд σ0 VV и
   сводится в корзины по дню года;
2. год без мокрой фазы отбраковывается (:func:`src.snowmelt.sheet_wet_amplitude`),
   остальные выравниваются по сдвигу весны и сводятся в ОДНУ климатическую
   кривую (:func:`src.snowmelt.align_and_pool`), из которой считаются метрики;
3. полученные карты проходят ТОТ ЖЕ протокол, что и блок 1
   (:func:`src.fracture_zones.screen_candidate`): lift@10% ячеек-целей,
   сдвиговый null и та же метрика после вычитания рельефа. Планки и цели те же,
   поэтому числа двух блоков сравниваются напрямую.

Диагностика физики печатается до всякой сверки: средняя по листу сезонная
кривая за каждый год. Если у неё нет провала мокрой фазы, мерить сроки не по
чему, и ветка закрывается на этом шаге, а не после протокола.

Выход: ``outputs/metrics/snowmelt_summary.csv``,
``data/processed/snowmelt_features.parquet``, ``outputs/snowmelt_preview.png``.
Запуск из корня: ``python -m experiments.snowmelt_run``.

РЕЗУЛЬТАТ (20.08.2026, dataset_v6, 5000 сдвигов, те же цели и планки, что в
блоке 1). Кандидат, значимый после контроля рельефа, — НИ ОДИН.

ГЛАВНОЕ, ЧТО СДЕЛАНО: снято ограничение, закрывшее однолетний пилот. По одному
году дата таяния определялась в 11% пикселей (одна сцена RTC покрывает медианно
10.9% листа). После сведения лет с выравниванием — 20 заполненных корзин из 25
на пиксель, медиана и q10 совпадают, у 100% пикселей корзин не меньше порога.
То есть ветка проверена по существу, а не упёрлась в покрытие.

ФИЗИКА ПОДТВЕРЖДАЕТСЯ НА ЛИСТЕ И НЕ ПОДТВЕРЖДАЕТСЯ НА ПИКСЕЛЕ. Средняя по листу
кривая даёт провал мокрой фазы 6.0-8.5 дБ в каждом полном году (2017: -15.8 дБ
на 164-й день, 2018: -16.6 / 144, 2019: -14.5 / 148, 2026: -16.7 / 154) при
пороге Наглера-Ротта 3 дБ. Но медиана ПОПИКСЕЛЬНОЙ глубины провала — 4.49 дБ,
и размах срока схода по листу всего 5.3 дня, то есть чуть больше одной корзины
в 5 дней. Величина, которую мы просим коррелировать с разломами, сама едва
разрешима.

ЧИСЛА (первичная цель «узлы», в скобках «коридор»). Лучший кандидат —
``sm_drop_db``: сырой lift 1.474 (1.081), p = 0.0196; после вычитания рельефа
1.359 (1.064), p = 0.0422. Это ЛУЧШИЙ остаточный lift за оба блока — планки
блока 1 держались на 1.185, — но до накопительного порога он не дотягивает на
полтора порядка. ``sm_wet_len`` 1.136 -> 0.995. ``sm_recovery_doy`` 0.954 ->
0.914.

ЗНАК СРОКА СХОДА ОБРАТНЫЙ И ЗАРАНЕЕ НЕ ЗАФИКСИРОВАН. ``sm_melt_doy`` даёт lift
0.163 (0.317): поздний сход снега узлов разломов ИЗБЕГАЕТ. Проверять
``-sm_melt_doy`` постфактум нельзя — направление связи физика допускает в обе
стороны (трещиноватость и дренирует, и накапливает надувы), и выбор знака после
взгляда на таблицу удвоил бы число тестов задним числом. Если ветку продолжать,
знак надо пре-регистрировать ДО прогона.

КОНТРОЛЬ НА АРТЕФАКТ СЪЁМКИ ЧИСТЫЙ. ``sm_n_bins`` (карта покрытия, не физика)
даёт 0.307 (0.321) — геометрия съёмки с целями не связана. Но на карте
``outputs/snowmelt_preview.png`` край полосы захвата виден как ПРЯМАЯ яркая
линия в северо-западном углу во всех панелях сразу. Глазами такую линию легко
принять за линеамент; это ровно тот способ ошибиться, ради которого ветка
меряется протоколом, а не рисунком.

ЧТО ВИДНО НА КАРТЕ. Рисунок срока схода — отчётливо дендритовый, повторяет
долинную сеть. Это и есть причина, по которой остаток после вычитания рельефа
падает: сроки таяния несут прежде всего рельеф, а не структуру.

ГОД 2021 ИСКЛЮЧЁН ПО ФИЗИКЕ, А НЕ ПО РЕЗУЛЬТАТУ. Все его майско-июньские сцены
S1B отдали 403 после 4 попыток; размах кривой 1.8 дБ (ниже порога мокрой фазы),
«минимум» на 128-й день против 144-164 у полных лет. Правило отбраковки
сформулировано по этой кривой ДО прогона протокола на четырёх годах и зашито в
код (:func:`src.snowmelt.sheet_wet_amplitude`). Прогон с 2021 сохранён для
сверки: ``outputs/metrics/snowmelt_summary_s2021.csv``. Он даёт тот же вердикт,
но хуже числа (``sm_drop_db`` 0.853 -> 1.185, попиксельная глубина провала 2.81
дБ, то есть ниже порога Наглера). С учётом этой пары прогонов накопительный
знаменатель Бонферрони равен 44, а не 34, как печатает прогон: alpha = 0.00114.

ЧТО ЭТО НЕ ЗАКРЫВАЕТ. Ряд остаётся коротким: 4 года и 140 сцен RTC. Коллекция
GRD даёт вдвое больше сцен и 7 относительных орбит против 2, но её яркость
зависит от геометрии наблюдения, поэтому орбиты нельзя складывать в один ряд без
радиометрической коррекции за рельеф. Именно ``sm_drop_db`` — единственная
величина за оба блока, пережившая контроль рельефа с lift выше планок, — и есть
то, что стоит перепроверять на удлинённом ряде с пре-регистрацией знака.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (cell_mask, config, fracture_zones as fz,  # noqa: E402
                 integro_grid, snowmelt as sm, stac_grid)


def _fetch_year(meta, year: int, log) -> tuple[np.ndarray, np.ndarray] | None:
    """Временной ряд σ0 VV за один год: ``(дни года, куб (n_сцен, H, W) в дБ)``."""
    from cache_paths import S1_ANABAR

    items = stac_grid.search(config.SM_COLLECTION, meta, years=(year,),
                             months=config.SM_MONTHS)
    if not items:
        log(f"  {year}: сцен нет, год пропущен")
        return None
    bands = {config.SM_POLARIZATION: (config.SM_POLARIZATION, 1)}
    doys, layers = [], []
    for i, it in enumerate(sorted(items, key=lambda x: x.datetime)):
        try:
            # Повторы делает сам stac_grid.read_on_grid; здесь остаётся только
            # решение «пропустить сцену»: одна недоступная сцена не должна
            # ронять сезон, ряд и так нерегулярный и пропуск гасится корзинами.
            d = stac_grid.scene_on_grid(it, bands, meta, S1_ANABAR,
                                        res_m=config.SM_RES_M,
                                        nodata=config.S1_NODATA)
        except Exception as exc:                      # noqa: BLE001
            log(f"    сцена {it.id} пропущена после "
                f"{config.SAT_FETCH_RETRIES} попыток: {type(exc).__name__} {exc}")
            continue
        layers.append(stac_grid.to_db(d[config.SM_POLARIZATION],
                                      floor_db=config.S1_FLOOR_DB))
        doys.append(it.datetime.timetuple().tm_yday)
        if (i + 1) % 10 == 0:
            log(f"    {i + 1}/{len(items)} сцен")
    if not layers:
        return None
    return np.array(doys, dtype=int), np.stack(layers)


def _log_season_curve(centers: np.ndarray, cube: np.ndarray, year: int, log) -> None:
    """Средняя по листу сезонная кривая — проверка физики до всякой сверки."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        curve = np.nanmean(cube, axis=(1, 2))
    ok = np.isfinite(curve)
    if ok.sum() < config.SM_MIN_BINS:
        log(f"  {year}: корзин с данными {ok.sum()} — кривой сезона нет")
        return
    lo = int(np.nanargmin(np.where(ok, curve, np.inf)))
    log(f"  {year}: минимум sigma0 = {curve[lo]:.1f} дБ на дне года {centers[lo]:.0f}; "
        f"размах кривой {np.nanmax(curve) - np.nanmin(curve):.1f} дБ")
    marks = " ".join(f"{int(c)}:{v:.1f}" for c, v in zip(centers[ok], curve[ok]))
    log(f"      кривая (день:дБ) {marks}")


def preview(feats: pd.DataFrame | None = None, meta=None,
            years: tuple[int, ...] | None = None) -> Path | None:
    """Карта листа по метрикам таяния — глазная проверка до всякой статистики.

    Смотреть здесь надо на две вещи. Первая: заполнена ли карта (панель
    ``sm_n_bins``) — именно нехватка корзин закрыла однолетний пилот. Вторая:
    похож ли рисунок срока на дренажную сеть или на ярусы рельефа; во втором
    случае контроль рельефа в протоколе всё равно съест величину, и это видно
    заранее.

    Вызывается из :func:`main`, но работает и отдельно, по сохранённому
    parquet: ``python -m experiments.snowmelt_run preview``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if feats is None:
        path = config.PROCESSED_DIR / "snowmelt_features.parquet"
        if not path.exists():
            print(f"нет {path} — сначала прогон main()")
            return None
        feats = pd.read_parquet(path)
    if meta is None:
        meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)

    panels = [("sm_melt_doy", "срок схода снега, день года", "viridis"),
              ("sm_recovery_doy", "возврат к летнему уровню, день года", "viridis"),
              ("sm_drop_db", "глубина мокрой фазы, дБ", "magma"),
              ("sm_wet_len", "длительность мокрой фазы, дней", "magma"),
              ("sm_n_bins", "корзин с данными", "cividis")]
    panels = [p for p in panels if p[0] in feats.columns]
    if not panels:
        print("в признаках нет ни одной колонки таяния — рисовать нечего")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (col, title, cmap) in zip(axes, panels):
        img = feats[col].to_numpy(dtype=float).reshape(meta.prf, meta.pic)
        finite = np.isfinite(img)
        # Пределы по процентилям, а не по min/max: одиночная корзина-выброс
        # иначе съедает весь диапазон цвета и карта выглядит пустой.
        lo, hi = ((np.nanquantile(img, 0.02), np.nanquantile(img, 0.98))
                  if finite.any() else (0.0, 1.0))
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1.0)
        ax.set_title(f"{title}\nопределено {finite.mean():.0%} ячеек", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.72)
    # Годы именно СВЕДЁННЫЕ, а не config.SM_YEARS: год без мокрой фазы
    # отбраковывается на входе, и подпись обязана это показывать.
    shown = "не указаны" if years is None else ", ".join(str(y) for y in years)
    fig.suptitle("Задача 2C: сроки схода снега по Sentinel-1 "
                 f"(сведённые годы: {shown}; климатическая кривая после выравнивания)",
                 fontsize=10)
    out_png = ROOT / "outputs" / "snowmelt_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main(years: tuple[int, ...] | None = None, screen: bool = True,
         n_shifts: int | None = None) -> None:
    t0 = time.time()

    def log(msg):
        # Консоль Windows здесь cp1251, и один символ вне её таблицы (например
        # sigma) роняет прогон на print — уже после того, как отработала вся
        # закачка. Для часовой ветки это неприемлемо, поэтому вывод чинится, а
        # не падает.
        line = f"[{time.time() - t0:5.0f}s] {msg}"
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            line.encode(enc)
        except UnicodeEncodeError:
            line = line.encode(enc, errors="replace").decode(enc)
        print(line, flush=True)

    years = years or config.SM_YEARS
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    log(f"сетка {meta.prf}x{meta.pic}, шаг растра {config.SM_RES_M:.0f} м, "
        f"годы {years}, месяцы {config.SM_MONTHS}")

    binned_years, melt_doys, used_years, centers = [], [], [], None
    for year in years:
        got = _fetch_year(meta, year, log)
        if got is None:
            continue
        doys, cube = got
        centers, binned = sm.bin_by_doy(doys, cube)
        del cube                                  # сырой куб года больше не нужен
        _log_season_curve(centers, binned, year, log)
        amp = sm.sheet_wet_amplitude(binned)
        if not np.isfinite(amp) or amp < config.SM_WET_DROP_DB:
            # Год без мокрой фазы не даёт опорной даты — см. sheet_wet_amplitude.
            log(f"  {year}: размах кривой {amp:.1f} дБ < порога "
                f"{config.SM_WET_DROP_DB:.1f} дБ — мокрой фазы нет, год не сводится")
            continue
        md = sm.sheet_melt_doy(centers, binned)
        nb = np.isfinite(binned).sum(axis=0)
        log(f"  {year}: сдвиг сезона (средний по листу минимум) = {md:.0f} день года; "
            f"корзин на пиксель медианно {np.median(nb):.0f} из {binned.shape[0]}")
        binned_years.append(binned.astype(np.float32))
        melt_doys.append(md)
        used_years.append(year)

    if not binned_years:
        log("ни одного года с данными — ветка закрывается здесь")
        return

    # Годы сводятся в ОДНУ климатическую кривую: по одному году дата таяния
    # определяется лишь в 11% пикселей (измерено пилотом), см. align_and_pool.
    pooled = sm.align_and_pool(binned_years, melt_doys)
    del binned_years
    nb = np.isfinite(pooled).sum(axis=0)
    log(f"сведено {len(melt_doys)} лет ({used_years}); корзин на пиксель после сведения: "
        f"медиана {np.median(nb):.0f}, q10 {np.quantile(nb, 0.1):.0f}, "
        f"доля пикселей с >= {config.SM_MIN_BINS} корзин {np.mean(nb >= config.SM_MIN_BINS):.1%}")
    rasters = sm.pooled_rasters(centers, pooled)
    del pooled

    feats = sm.melt_features(meta, rasters)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats_path = config.PROCESSED_DIR / "snowmelt_features.parquet"
    feats.to_parquet(feats_path, index=False)
    log(f"признаки на ячейки: {feats.shape} -> {feats_path}")
    for c in sm.SNOWMELT_COLS:
        if c in feats.columns:
            v = feats[c].to_numpy(dtype=float)
            log(f"    {c:<18} NaN {np.isnan(v).mean():5.1%}  "
                f"медиана {np.nanmedian(v):8.2f}  разброс {np.nanstd(v):6.2f}")
    try:
        png = preview(feats, meta, tuple(used_years))
    except Exception as exc:                          # noqa: BLE001
        # Картинка — вспомогательный выход; ронять из-за неё часовой прогон,
        # у которого уже сохранены признаки, нельзя.
        log(f"предпросмотр не нарисован: {type(exc).__name__} {exc}")
    else:
        if png is not None:
            log(f"карта: {png}")
    if not screen:
        return
    _screen(meta, feats, n_shifts, log)


def _screen(meta, feats: pd.DataFrame, n_shifts, log) -> None:
    """Тот же протокол, что в блоке 1: цели, планки, null, контроль рельефа."""
    from experiments.fracture_zones_run import PLANKS, _load_dataset, _targets, _score_of

    df = _load_dataset()
    valid = cell_mask.build_valid_mask(df)
    pool = np.where(valid)[0]
    targets = _targets(df, pool)
    controls = fz.relief_control_matrix(df)

    cands = [c for c in sm.SNOWMELT_COLS if c in feats.columns and c != "sm_n_years"]
    # Знаменатель Бонферрони НАКОПИТЕЛЬНЫЙ: к 24 тестам блока 1 добавляются
    # тесты этой ветки. Цели и планки те же, набор точек тот же, поэтому порог
    # обязан учитывать оба блока — иначе «перебирай, пока не выиграет».
    n_tests = config.FZ_N_TESTS_BLOCK1 + 2 * len(cands)
    alpha = 0.05 / n_tests
    log(f"кандидатов {len(cands)}, тестов в этой ветке {2 * len(cands)}, "
        f"накопительно {n_tests}, alpha = {alpha:.5f}")

    rows = []
    for target_name, fault_idx in targets.items():
        for name in [*PLANKS, *cands]:
            score = (feats[name].to_numpy(dtype=float) if name in cands
                     else _score_of(name, df, len(df)))
            res = fz.screen_candidate(score, df, meta, fault_idx, pool,
                                      controls=controls, n_shifts=n_shifts)
            rows.append({"цель": target_name, "карта": name,
                         "планка": name in PLANKS,
                         "NaN": float(np.isnan(score[pool]).mean()), **res})
            log(f"  [{target_name}] {name:18} lift={res['lift']:.3f} (p={res['p']:.4f})  "
                f"остаток lift={res['lift_resid']:.3f} (p={res['p_resid']:.4f})")

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "snowmelt_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 82}\nИТОГ 2C (накопительный порог alpha = {alpha:.5f})\n{'=' * 82}")
    for target_name in out["цель"].unique():
        sub = out[out["цель"] == target_name].sort_values("lift_resid", ascending=False)
        print(f"\nЦель: {target_name}")
        print(f"{'карта':<20}{'NaN':>6}{'lift':>8}{'p':>9}{'lift ост':>10}{'p ост':>9}  вердикт")
        for _, r in sub.iterrows():
            verdict = ("планка" if r["планка"] else
                       "ЗНАЧИМ после контроля" if r["p_resid"] < alpha else "-")
            print(f"{r['карта']:<20}{r['NaN']:>5.0%}{r['lift']:>8.3f}{r['p']:>9.4f}"
                  f"{r['lift_resid']:>10.3f}{r['p_resid']:>9.4f}  {verdict}")
    log(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        print(f"карта: {preview()}")
    else:
        main()
