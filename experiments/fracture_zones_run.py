"""Задача №2: зоны трещиноватости по снимкам против карты разломов.

Два блока, в порядке цены:

1. **Скрининг того, что уже посчитано** (сеть не нужна). В датасете уже лежит
   субъячеечный разброс каждого спутникового канала (``*_std`` из
   :func:`src.stac_grid.to_cells`) и радарные отношения ``s1_vv_vh`` /
   ``psr_hh_hv``. Это готовые площадные меры текстуры — то есть уже кандидаты в
   «зоны трещиноватости», которые никто не проверял против карты разломов.
   Каждый прогоняется через протокол :func:`src.fracture_zones.screen_candidate`:
   lift@10% ячеек-целей, сдвиговый null и та же метрика после вычитания
   рельефа.
2. **Структурный тензор по композитам** (нужны скачанные сцены, см.
   :func:`_raster_block`). Непрерывное поле когерентности вместо пороговых линий
   задачи №8 — см. докстринг :mod:`src.fracture_zones`. Перед сверкой с
   разломами каждая пара источник-масштаб обязана пройти стоп-правило
   воспроизводимости между половинами выборки сцен; не прошедшая до сверки не
   допускается и в знаменатель Бонферрони не входит.

ЦЕЛЬ СВЕРКИ выбрана по измерению, а не по умолчанию (см. FZ_NODE_CORRIDOR_M в
:mod:`src.config`): широкие коридоры разломов накрывают 57.8% листа и дают
потолок lift@10% всего 1.73, поэтому первичная цель — пересечение СЗ и СВ
систем (14.3% площади, потолок 6.98), геологически это узлы структурного
каркаса.

ПЛАНКИ обязательны и печатаются в той же таблице: случайный скор (пол),
шероховатость рельефа ``relief2_tri_std`` (что даёт один рельеф) и рельефная
линеаментная ветка ``lin_dens``/``lin_node_dens`` (единственная ветка, прошедшая
стоп-правило задачи №8). Без них lift не читается.

Выход: ``outputs/metrics/fracture_zones_summary.csv`` (блок 1),
``outputs/metrics/fracture_zones_repro.csv`` и ``fracture_zones_tensor.csv``,
``outputs/fracture_zones_tensor.png`` (блок 2).
Запуск из корня: ``python -m experiments.fracture_zones_run``.

РЕЗУЛЬТАТ БЛОКА 1 (20.08.2026, dataset_v6, 5000 сдвигов, порог Бонферрони
alpha = 0.05/24 = 0.00208). Повторён 21.08.2026 вместе с блоком 2 — все числа
воспроизвелись до четвёртого знака.

НИ ОДИН кандидат не значим после контроля рельефа — ни на одной из двух целей.

Самая содержательная строка таблицы — ``s1_vv_vh_std`` (субъячеечный разброс
отношения поляризаций Sentinel-1) на цели «коридор»: сырое p = 0.0008, то есть
ЕДИНСТВЕННЫЙ кандидат, проходящий Бонферрони в сыром виде. После вычитания
рельефа lift падает 1.179 -> 1.070, а p = 0.1290. Это диагноз задачи №8
(«снимок пересказывает рельеф через более шумный канал»), воспроизведённый на
площадных признаках и полученный ИЗМЕРЕНИЕМ, а не хрупким стоп-правилом: там
ветка закрывалась тем, что список отрезков не переживал смену половины
выборки, здесь — тем, что сигнал не переживает вычитание рельефа.

Планки ведут себя ровно как обязаны, и это проверка самого протокола:

* ``random`` — lift 0.957/0.994 при p = 0.82/0.66: пол на месте;
* ``relief2_tri_std`` — сырой lift 1.313, остаток 0.770. Колонка сама входит в
  FZ_RELIEF_CONTROLS, так что её остаток обязан схлопнуться; он схлопывается —
  контроль рельефа действительно вычитает то, что должен;
* ``lin_dens``/``lin_node_dens`` (рельефная ветка, единственная прошедшая
  стоп-правило задачи №8) держат верх остаточного рейтинга (1.185 и 1.149), но
  и они до alpha не дотягивают.

Отдельно стоит потолок метрики. На первичной цели «узлы» он равен 6.98, а
лучший результат по таблице — 1.36. То есть на 10% площади ни одна карта не
находит узлы пересечения разломных систем сколько-нибудь заметно лучше
случайной.

Сырые p при этом почти везде незначимы даже у карт с lift 1.3 — это работает
сдвиговый null: карта с сильной пространственной автокорреляцией набирает
lift 1.3 случайно без труда. Тем самым подтверждается и замечание задачи №8
(«доля авто без карты неотличима от случайного узора сравнимой плотности»), и
решение зашить нуль-модель в протокол с самого начала: без неё половина
таблицы читалась бы как успех.

РЕЗУЛЬТАТ БЛОКА 2 (21.08.2026; Sentinel-2 180 сцен, Landsat 8/9 90,
Sentinel-1 24, PALSAR-2 56 — все четыре источника задачи №8). Накопительный
порог alpha = 0.05/68 = 0.000735 (44 теста прошлых прогонов плюс 24 своих).
Значимых после контроля рельефа — НЕТ.

ГЛАВНОЕ И НЕОЖИДАННОЕ: СТОП-ПРАВИЛО ПРОЙДЕНО ВСЕМИ ДВЕНАДЦАТЬЮ ПАРАМИ. Задача
№8 на этих же снимках закрылась ровно тем, что список отрезков не переживал
смену половины выборки. Непрерывное поле когерентности переживает: Sentinel-2
rho = 0.956/0.978/0.986, Landsat 0.925/0.960/0.971, Sentinel-1
0.678/0.799/0.817, PALSAR-2 0.633/0.670/0.610 при пороге 0.5. То есть хрупкой
была БИНАРИЗАЦИЯ Canny->Hough, а не снимок, и это измеренное подтверждение
довода, ради которого блок 2 вообще затевался. Радар при этом воспроизводится
заметно хуже оптики — у него меньше сцен и есть спекл, — но порог берёт и он.

Но воспроизводимость поля — это вход в протокол, а не вывод: устойчиво
воспроизводиться может и рельеф с освещением. Дальше поле не проходит.

ЧИСЛА. Лучшая карта — ``fzs1_coh_3km`` на цели «узлы»: lift 1.282 (p = 0.2535),
после вычитания рельефа 1.346 (p = 0.2210). Единственный случай за все блоки,
когда остаток ВЫШЕ сырого: контроль рельефа эту карту не ослабляет. Но p = 0.22
против порога 0.0007, а потолок метрики на этой цели 6.98. На цели «коридор» не
поднимается никто: максимум 1.052.

ОПТИКА ИДЁТ НИЖЕ ЕДИНИЦЫ, И ЭТО СИСТЕМАТИЧНО. На цели «узлы» Sentinel-2 даёт
0.911/0.759/0.460, Landsat 1.167/0.990/0.699 — с ростом масштаба обе съёмки
уходят всё дальше вниз, то есть высокая когерентность оптики систематически
ИЗБЕГАЕТ узлов. Радар ведёт себя иначе (PALSAR-2 1.188/1.237/1.164). Читать это
как «оптика противопоказана» нельзя: знак не пре-регистрировался, а p везде
далеки от порога. Но направление согласуется с тем, что оптика на этом листе
меряет задернование (98.3% пикселей — трава), а не породу.

ПОЧЕМУ, ВИДНО ПО САМОМУ ПОЛЮ, А НЕ ПО ТАБЛИЦЕ. Медиана когерентности PALSAR-2
падает с масштабом: 0.140 на 300 м, 0.035 на 1 км, 0.009 на 3 км; значений выше
0.9 нет вовсе. Связь когерентности с энергией тензора при этом почти нулевая
(+0.234 / +0.109 / -0.028), то есть поле не сводится к «где сильный перепад
яркости». Смысл цифр прямой: НА ЭТОМ ЛИСТЕ У СНИМКА НЕТ ОРИЕНТИРОВАННОЙ
ТЕКСТУРЫ НА КИЛОМЕТРОВЫХ МАСШТАБАХ. Для дендритовой долинной сети это ожидаемо
(она ветвится и изгибается, прямых кусков в 3 км у неё нет), а вот разлом прямой
на километры — и будь в снимке разломная ткань, когерентность 3 км зажглась бы
вдоль неё. Она не зажигается: на карте ``outputs/fracture_zones_tensor.png``
поля всех трёх источников — изотропные пятна без единой линейной формы, хотя на
самих композитах слева долинная сеть читается отчётливо.

Реализация тензора при этом проверена отдельно: на синтетических полосах тесты
требуют высокой когерентности и восстановления направления с точностью до
половины лепестка розы (``tests/test_fracture_zones.py``). Низкие значения —
свойство данных, а не кода.

ЧТО ЭТО НЕ ЗАКРЫВАЕТ. Первое: кандидатом пре-регистрирована ГОЛАЯ когерентность.
Она нормирована и не знает про контраст, поэтому одинаково велика и на резкой
границе, и на слабом, но согласованном градиенте. Естественный следующий
кандидат — когерентность, взвешенная энергией (в поле уже считаются
``dir_sin``/``dir_cos`` именно как ``coh * sin2a``), но это ОТДЕЛЬНАЯ гипотеза,
и вводить её надо пре-регистрацией до прогона, а не после взгляда на таблицу.
Второе: прогон на трёх источниках (без Sentinel-2, 62 теста) дал те же числа до
четвёртого знака — сверка воспроизводима, но это и значит, что добавление
источника ветку не спасает.
"""
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import cell_mask, config, fracture_zones as fz, integro_grid  # noqa: E402

#: Планки. Не кандидаты: нужны, чтобы число lift вообще можно было прочитать.
#: ``relief2_tri_std`` входит и в FZ_RELIEF_CONTROLS — её остаток обязан быть
#: около нуля, это заодно проверка, что контроль рельефа действительно работает.
PLANKS: tuple[str, ...] = ("random", "relief2_tri_std", "lin_dens", "lin_node_dens")

#: Источники блока 2 — те же четыре съёмки, что проверяла задача №8.
FZ2_SOURCES: tuple[str, ...] = ("s2", "l8", "s1", "psr")

#: Кандидаты фиксируются ДО прогона (пре-регистрация, как VER_PRIMARY): по одному
#: на физический канал, а не «все *_std, какие нашлись», — иначе знаменатель
#: Бонферрони растёт быстрее, чем содержательность перебора.
CANDIDATES: tuple[str, ...] = (
    # радар: отношение поляризаций почти не зависит от местного угла падения,
    # то есть свободно от геометрического артефакта, которым задача №8
    # объяснила свой отрицательный результат
    "s1_vv_vh", "s1_vv_vh_std", "s1_vv_std", "s1_vh_std",
    "psr_hh_hv", "psr_hh_std",
    # оптика: та же текстура, но через покров (98.3% пикселей листа — трава)
    "l8_red_std", "l8_blue_std",
    "s2_b11_std", "s2_b8a_std", "s2_b02_std", "ast_b04_std",
)


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


def _targets(df: pd.DataFrame, pool: np.ndarray) -> dict[str, np.ndarray]:
    """Цели сверки: узлы (пересечение систем) — первичная, коридор — вторичная."""
    d1 = df["dist_tect1"].to_numpy(dtype=float)
    d2 = df["dist_tect2"].to_numpy(dtype=float)
    c = config.FZ_NODE_CORRIDOR_M
    nodes = fz.fault_corridor_cells(np.maximum(d1, d2), c, pool=pool)
    corridor = fz.fault_corridor_cells(np.minimum(d1, d2), c, pool=pool)
    return {"узлы (СЗ и СВ)": nodes, "коридор (СЗ или СВ)": corridor}


def _score_of(name: str, df: pd.DataFrame, n_cells: int) -> np.ndarray:
    if name == "random":
        return np.random.default_rng(config.VER_SEED).random(n_cells)
    return df[name].to_numpy(dtype=float)


def main(n_shifts: int | None = None,
         sources: tuple[str, ...] | None = None) -> None:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    df = _load_dataset()
    valid = cell_mask.build_valid_mask(df)
    pool = np.where(valid)[0]
    log(f"сетка {meta.prf}x{meta.pic}, валидных ячеек {pool.size}, колонок {df.shape[1]}")

    targets = _targets(df, pool)
    for name, idx in targets.items():
        share = idx.size / pool.size
        log(f"цель «{name}»: {idx.size} ячеек ({share:.1%} пула), "
            f"потолок lift@{config.FZ_AREA:.0%} = {min(1.0, config.FZ_AREA / share) / config.FZ_AREA:.2f}")

    controls = fz.relief_control_matrix(df)
    have = [c for c in config.FZ_RELIEF_CONTROLS if c in df.columns]
    log(f"контроли рельефа: {len(have)} колонок {have}")

    names = [n for n in (*PLANKS, *CANDIDATES) if n == "random" or n in df.columns]
    missing = [n for n in (*PLANKS, *CANDIDATES) if n not in names]
    if missing:
        log(f"нет в датасете, пропущены: {missing}")
    # Знаменатель Бонферрони: каждый кандидат проверяется дважды (сырой и
    # остаток), планки в перебор не входят — они не гипотезы, а шкала.
    n_tests = 2 * len([n for n in names if n in CANDIDATES])
    alpha = 0.05 / max(1, n_tests)
    log(f"кандидатов {n_tests // 2}, тестов {n_tests}, порог Бонферрони alpha = {alpha:.5f}")

    rows = []
    for target_name, fault_idx in targets.items():
        for name in names:
            score = _score_of(name, df, len(df))
            # Доля пропусков печатается рядом с lift не для полноты: NaN по
            # протоколу — промах, поэтому у неполного источника метрика
            # механически занижена и с остальными несравнима. На этом листе так
            # только у ASTER (сцен на весь лист не хватает, см. AST_MIN_OBS).
            nan_frac = float(np.isnan(score[pool]).mean())
            res = fz.screen_candidate(
                score, df, meta, fault_idx, pool,
                controls=controls, n_shifts=n_shifts)
            rows.append({"цель": target_name, "карта": name,
                         "планка": name in PLANKS, "NaN": nan_frac, **res})
            log(f"  [{target_name}] {name:16} lift={res['lift']:.3f} (p={res['p']:.4f})  "
                f"остаток lift={res['lift_resid']:.3f} (p={res['p_resid']:.4f})")

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "fracture_zones_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 78}\nИТОГ (порог значимости alpha = {alpha:.5f})\n{'=' * 78}")
    for target_name in targets:
        sub = out[out["цель"] == target_name].sort_values("lift_resid", ascending=False)
        print(f"\nЦель: {target_name}")
        print(f"{'карта':<18}{'NaN':>6}{'lift':>7}{'p':>9}{'lift ост.':>11}{'p ост.':>9}  вердикт")
        for _, r in sub.iterrows():
            verdict = ("планка" if r["планка"] else
                       "ЗНАЧИМ после контроля" if r["p_resid"] < alpha else
                       "не значим после контроля")
            print(f"{r['карта']:<18}{r['NaN']:>5.0%}{r['lift']:>7.3f}{r['p']:>9.4f}"
                  f"{r['lift_resid']:>11.3f}{r['p_resid']:>9.4f}  {verdict}")

    log(f"\nсохранено: {out_path}")
    _raster_block(meta, df, pool, targets, controls, log, n_shifts, sources)


def _source_fields(kind: str, meta, log):
    """Композит источника и поля тензора по нему и по двум половинам выборки.

    Возвращает ``(поле_полное, поле_чётных, поле_нечётных, res_m, сцен, img)``
    либо ``None``, если сцен меньше порога задачи №8.
    """
    from experiments.lineaments_sat_run import (MIN_ITEMS, NAMES, _date_key,
                                                _fetch_items, _grayscale)
    from src import lineaments_sat

    items = _fetch_items(kind, meta)
    log(f"--- {NAMES[kind]}: найдено сцен {len(items)} ---")
    if len(items) < MIN_ITEMS:
        log(f"  меньше {MIN_ITEMS} сцен — источник пропущен")
        return None

    if kind == "psr":
        # У PALSAR-2 «сцена» — это годовая мозаика ОДНОЙ ПЛИТКИ, а не снимок на
        # весь лист. Деление по порядку дат разрезало бы лист ПРОСТРАНСТВЕННО:
        # половины покрыли бы разные куски, и стоп-правило считалось бы на узком
        # перекрытии. Независимая ось здесь — год съёмки, а не номер плитки.
        even = [it for it in items if it.datetime.year % 2 == 0]
        odd = [it for it in items if it.datetime.year % 2 == 1]
    else:
        even, odd = lineaments_sat.split_even_odd(items, key=_date_key(kind))
    # Порог наблюдений на пиксель у половин снимается: половинная выборка при
    # SAT_MIN_OBS = 3 почти вся уходит в NaN, и стоп-правило мерило бы покрытие,
    # а не согласие полей. Так же сделано в задаче №8.
    half_min_obs = 1 if kind != "s2" else None
    img, res_m, _ = _grayscale(kind, meta, items)
    img_a, _, _ = _grayscale(kind, meta, even, min_obs=half_min_obs)
    img_b, _, _ = _grayscale(kind, meta, odd, min_obs=half_min_obs)
    log(f"  композит: шаг {res_m:.0f} м, валидных пикселей "
        f"{np.isfinite(img).mean():.1%}, половины {len(even)}/{len(odd)}")
    return (fz.zone_field(img, res_m=res_m), fz.zone_field(img_a, res_m=res_m),
            fz.zone_field(img_b, res_m=res_m), res_m, len(items), img)


def _raster_block(meta, df, pool, targets, controls, log, n_shifts=None,
                  sources: tuple[str, ...] | None = None) -> None:
    """Блок 2: непрерывное поле когерентности вместо пороговых линий задачи №8.

    Порядок жёсткий и в нём весь смысл ветки:

    1. по каждому источнику считается поле тензора на трёх масштабах;
    2. СТОП-ПРАВИЛО: поле должно воспроизводиться между независимыми половинами
       выборки сцен (:func:`src.fracture_zones.reproducibility`, порог
       :data:`config.LIN_MIN_SPEARMAN`). Не прошедшая пара источник-масштаб до
       сверки с разломами НЕ ДОПУСКАЕТСЯ — и потому не входит в знаменатель
       Бонферрони. Это не экономия тестов, а логика: карта, которая меняется от
       смены половины сцен, не описывает ничего на земле;
    3. выжившие проходят ТОТ ЖЕ протокол, что блок 1 и ветка 2C.

    Кандидат — только КОГЕРЕНТНОСТЬ. Энергия тензора это модуль градиента, то
    есть контраст, а он на этом листе задаётся рельефом и освещением; направление
    (``dir_sin``/``dir_cos``) осмысленно как роза азимутов, но не как скор для
    lift. Список зафиксирован до прогона, как CANDIDATES выше.
    """
    try:
        import cache_paths  # noqa: F401
    except ImportError:
        log("блок 2: нет cache_paths.py — пропущен "
            "(в рабочем дереве файл удалён, вернуть: git checkout cache_paths.py)")
        return

    fields, feats, cands, rep_rows = {}, {}, [], []
    # Подмножество источников — чтобы считать по тому, что уже в кэше, не
    # дожидаясь самой долгой закачки; знаменатель Бонферрони накопительный,
    # так что досчитанный позже источник свои тесты прибавит, а не обнулит.
    for kind in (sources or FZ2_SOURCES):
        try:
            got = _source_fields(kind, meta, log)
        except Exception as exc:                      # noqa: BLE001
            # Отказ одного источника (нет сцен, обрыв сети) не должен уносить
            # остальные: каждый источник — самостоятельная проверка.
            log(f"  источник {kind} упал: {type(exc).__name__} {exc}")
            continue
        if got is None:
            continue
        field, field_a, field_b, res_m, n_items, img = got
        fields[kind] = (field, img, res_m)
        feats[kind] = fz.zone_features(meta, img, prefix=f"fz{kind}_", res_m=res_m)

        for s in config.FZ_SCALES_M:
            tag = fz._scale_tag(s)
            rep = fz.reproducibility(field_a[f"coh_{tag}"], field_b[f"coh_{tag}"])
            passed = np.isfinite(rep["rho"]) and rep["rho"] >= config.LIN_MIN_SPEARMAN
            log(f"  когерентность {tag:>3}: воспроизводимость rho = {rep['rho']:+.3f} "
                f"(порог {config.LIN_MIN_SPEARMAN}) -> "
                f"{'допущена к сверке' if passed else 'ОТБРОШЕНА'}")
            rep_rows.append({"источник": kind, "масштаб": tag, "сцен": n_items,
                             "rho_половин": rep["rho"],
                             "площадь": rep["covered_frac"], "допущена": passed})
            if passed:
                cands.append((kind, f"fz{kind}_coh_{tag}"))

    if not fields:
        log("блок 2: ни одного источника с данными — кэш сцен пуст, "
            "сначала нужна закачка композитов")
        return
    _save_reproducibility(rep_rows, log)
    _preview_coherence(fields, log)

    if not cands:
        log("блок 2: ни одна пара источник-масштаб не прошла стоп-правило — "
            "сверка с картой разломов не проводится")
        return

    # Знаменатель накопительный: к прогонам блока 1 и ветки 2C добавляются
    # только ДОПУЩЕННЫЕ кандидаты — отброшенные стоп-правилом не проверялись.
    n_tests = config.FZ_N_TESTS_PRIOR + 2 * len(cands)
    alpha = 0.05 / n_tests
    log(f"блок 2: допущено кандидатов {len(cands)}, тестов {2 * len(cands)}, "
        f"накопительно {n_tests}, alpha = {alpha:.6f}")

    rows = []
    for target_name, fault_idx in targets.items():
        for kind, col in cands:
            score = feats[kind][col].to_numpy(dtype=float)
            res = fz.screen_candidate(score, df, meta, fault_idx, pool,
                                      controls=controls, n_shifts=n_shifts)
            rows.append({"цель": target_name, "карта": col,
                         "NaN": float(np.isnan(score[pool]).mean()), **res})
            log(f"  [{target_name}] {col:20} lift={res['lift']:.3f} (p={res['p']:.4f})  "
                f"остаток lift={res['lift_resid']:.3f} (p={res['p_resid']:.4f})")

    out = pd.DataFrame(rows)
    out_path = ROOT / "outputs" / "metrics" / "fracture_zones_tensor.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 82}\nИТОГ БЛОКА 2 (накопительный порог alpha = {alpha:.6f})\n{'=' * 82}")
    for target_name in out["цель"].unique():
        sub = out[out["цель"] == target_name].sort_values("lift_resid", ascending=False)
        print(f"\nЦель: {target_name}")
        print(f"{'карта':<22}{'NaN':>6}{'lift':>8}{'p':>9}{'lift ост':>10}{'p ост':>9}  вердикт")
        for _, r in sub.iterrows():
            verdict = "ЗНАЧИМ после контроля" if r["p_resid"] < alpha else "-"
            print(f"{r['карта']:<22}{r['NaN']:>5.0%}{r['lift']:>8.3f}{r['p']:>9.4f}"
                  f"{r['lift_resid']:>10.3f}{r['p_resid']:>9.4f}  {verdict}")
    log(f"сохранено: {out_path}")


def _save_reproducibility(rep_rows: list[dict], log) -> None:
    out_path = ROOT / "outputs" / "metrics" / "fracture_zones_repro.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rep_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"стоп-правило сохранено: {out_path}")


def _preview_coherence(fields: dict, log) -> None:
    """Композит и когерентность на трёх масштабах — по строке на источник."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_col = 1 + len(config.FZ_SCALES_M)
    fig, axes = plt.subplots(len(fields), n_col,
                             figsize=(4.2 * n_col, 4.2 * len(fields)))
    axes = np.atleast_2d(axes)
    for row, (kind, (field, img, res_m)) in zip(axes, fields.items()):
        lo, hi = np.nanpercentile(img, [2, 98])
        row[0].imshow(img, cmap="gray", vmin=lo, vmax=hi)
        row[0].set_title(f"{kind}: композит, шаг {res_m:.0f} м", fontsize=9)
        for ax, s in zip(row[1:], config.FZ_SCALES_M):
            tag = fz._scale_tag(s)
            c = field[f"coh_{tag}"]
            ax.imshow(c, cmap="magma", vmin=0.0, vmax=np.nanpercentile(c, 98))
            ax.set_title(f"когерентность {tag}", fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Задача №2, блок 2: структурный тензор по композитам "
                 "(непрерывное поле вместо пороговых линий)", fontsize=11)
    out_png = ROOT / "outputs" / "fracture_zones_tensor.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"карта: {out_png}")


if __name__ == "__main__":
    main()
