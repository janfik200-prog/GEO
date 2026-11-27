# SYN1: Анфиса на искусственных линиях — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Анфиса (FAM) учится отличать врезанные в подделку линии от фона и без учителя-человека выбирает видимые линии из кандидатов CAND3 на Утсйоки и Анабаре.

**Architecture:** Общая фазовая подделка снимка, GLO-30 и ALOS → врезки линий разных типов с известной геометрией → тот же тракт кандидатов CAND2/CAND3 и 20 признаков ANFL0b → метка «у врезки» → FAM → порог по подделке Б (мусор ≤ 20 %) → перенос долей длины на Утсйоки (проверка) и Анабар (цель).

**Tech Stack:** Python 3.12, numpy, scipy, torch (CPU), geopandas/shapely, pytest; модули ветки `experiments/openlin_*`, `src/detection_limit.py`.

**Spec:** `docs/superpowers/specs/2026-11-23-anfisa-synthetic-teacher-design.md`

## Global Constraints
- Порядок ветки: `audit/SYN1_DECLARATION.md` → `openlin-registrar` → код → `openlin-keeper` → прогон → `openlin-artifact-hunter` → окна → `openlin-auditor` → отчёт.
- Только CPU: `CUDA_VISIBLE_DEVICES=""`. Запуск модулей из корня: `python -m experiments.<модуль>`.
- Числа — из замера или арифметики порога; каждое — с источником в объявлении; после прогона не двигаются.
- GTK и разметка не входят ни в обучение, ни в выбор порога. Лапландия не трогается.
- Код, комментарии, сообщения — по-русски. Правки — только редактором (heredoc в ветке молча не применяется), проверка grep.
- git: коммит и push — только с разрешения Артура; без подписи ассистента.
- Подпись всех карт, окон и векторов: «учитель — искусственные линии (SYN1)».

## Review Focus
- Врезка у края листа или у дыры данных: геометрия обрезается — метка и подсчёт полноты не должны падать и не должны считать обрезанную врезку найденной по части вне маски.
- Пересечение врезок с настоящими линиями на проверочном фоне: «найдено» засчитывается только по следу врезки, не по соседней настоящей черте (допуск и направление).
- Канал без кандидатов на подделке (пустой пул канала): обучение и признак «до линии другого канала» работают.
- Порог, при котором на настоящем листе выбрано 0 км: правило «≤ 20 %» не должно делиться на ноль; печатать и останавливаться.
- Подделка с теми же фазами для трёх источников разной формы/маски: формы и маски совпадают, иначе стоп.

---

### Task 0: Объявление и регистратор
**Files:** Create: `audit/SYN1_DECLARATION.md`
- [ ] **Step 1:** Записать объявление по замыслу (разделы «Как устроено» и «Уточнения»), с источником каждого числа: диапазоны врезок (контраст log-равномерно 0,25–8σ; ширина — масштабы каналов `IMG_SIG_M`/`DEM_SIG_M`; длина log-равномерно от `LINK_MIN_LEN_M` до наибольшей длины, которую вмещает лист с запасом; изгиб, неравномерность контраста, доля составных и смешанных, разрывы — с обоснованием), число врезок на лист и правило непересечения, допуск метки (`CONF_TOL_M` = 60 м — допуск подтверждения ветки), зёрна, правило порога, условия (а)–(в), `SYN1_N_TESTS = 0`.
- [ ] **Step 2:** Вызвать `openlin-registrar`; внести его поправку; ничего не кодить до неё.

### Task 1: Общая фазовая подделка
**Files:** Create: `experiments/openlin_syn_insert.py`; Test: `tests/test_openlin_syn.py`
**Interfaces:** Produces: `joint_phase_randomize(fields: list[np.ndarray], valid: np.ndarray, rng) -> list[np.ndarray]` — одни случайные фазы для всех полей; амплитудный спектр каждого поля сохранён; среднее и разброс по `valid` как у исходного (как `openlin_null_input.phase_randomize`).
- [ ] **Step 1: тест** `test_joint_fake_keeps_cross_correlation`: два коррелированных поля 256×256 (b = a + шум); у подделки корреляция по ячейкам отличается от исходной не более чем на 0,05; `test_joint_fake_breaks_lines`: поле с прямой полосой — после подделки отклик `ridge_dir` вдоль линии не выделяется (максимум по строке полосы < 3× медианы); `test_joint_fake_seed`: разные зёрна → разные поля, одно зерно → одинаковые.
- [ ] **Step 2:** `pytest tests/test_openlin_syn.py -k joint -q` → FAIL (нет функции).
- [ ] **Step 3:** реализовать (rfft2 каждого поля, общий массив случайных фаз).
- [ ] **Step 4:** тот же pytest → PASS.

### Task 2: Врезки
**Files:** Modify: `experiments/openlin_syn_insert.py`; Test: `tests/test_openlin_syn.py`
**Interfaces:** Produces:
- `relief_sigma(dem: np.ndarray, valid) -> float` — `robust_sigma` рельефа после детренда `openlin_winter.DETREND_M`.
- `make_plan(shape, valid, res_m: float, rng, n: int) -> list[dict]` — словари: `kind` ∈ {"тёмная","светлая","перепад","ложбина","гребень","уступ","смешанная","составная"}, `center`, `az_deg`, `length_m`, `width_m`, `amp`, `bend`, `pieces` (для составной — список отрезков вдоль оси с разрывами). Врезки не пересекаются (зазор ≥ ширины окна детренда/2), центр — в годной части.
- `apply_plan(img, dem, alos, plan, sig_img, sig_dem, res_m) -> tuple[img2, dem2, alos2, truth]` — `truth`: список массивов ячеек (строка, столбец) следа каждой врезки внутри `valid`; рельефные виды — одинаково в `dem` и `alos`; «смешанная» — в рельеф и полосой в снимок на одной оси. Врезка — через `src.detection_limit.insert_fault` с детрендом как `openlin_azimuth_response.insert_window`.
- [ ] **Step 1: тесты** `test_plan_no_overlap` (n=40 на 1000×1000: попарные расстояния следов ≥ заданного зазора); `test_dark_band_visible` (тёмная полоса 4σ на шуме → средний z канала 1 на следе > 2); `test_relief_into_both` (вид «ложбина»: `dem2 − dem` == `alos2 − alos` с точностью 1e-9; снимок не изменён); `test_composite_has_gaps` (у составной след состоит из ≥ 2 кусков); `test_truth_inside_valid` (часть врезки за маской — в `truth` только годные ячейки).
- [ ] **Step 2:** pytest → FAIL. **Step 3:** реализовать. **Step 4:** pytest → PASS.

### Task 3: Тракт по массивам (без смены поведения)
**Files:** Modify: `experiments/openlin_anfl.py` (`candidates`), `experiments/openlin_anfl0b.py` (`line_features`); Test: `tests/test_openlin_syn.py`
**Interfaces:** Produces: `openlin_anfl.candidates_arr(img, dem, valid, log, site_check: str | None = None) -> list[dict]` (пул CAND3 с `poly`); `candidates(site, log)` вызывает её. `openlin_anfl0b.line_features_arr(pool, img, dem, alos_raw, valid, log) -> np.ndarray` (20 признаков); `line_features(site, …)` читает ALOS и вызывает её.
- [ ] **Step 1: тест** `test_arr_equals_site_path` (кусок Утсйоки 600×600: пул и признаки через `*_arr` совпадают с прежним путём, сравнение sha пула и `np.allclose` признаков).
- [ ] **Step 2–4:** FAIL → рефакторинг → PASS. Отметить в объявлении: sha `openlin_anfl.py`/`openlin_anfl0b.py` меняются, поведение то же (тест).

### Task 4: Метка «у врезки» и сборка обучения
**Files:** Create: `experiments/openlin_syn1.py`; Test: `tests/test_openlin_syn.py`
**Interfaces:** Consumes: Tasks 1–3. Produces:
- `labels_from_truth(pool, truth: list[np.ndarray], shape, tol_m: float) -> np.ndarray` — 1, если ≥ 50 % ячеек кандидата в `tol_m` от следа какой-либо врезки, иначе 0.
- `site_sample(site, seed: int, n_ins: int, log) -> dict` — подделка (Task 1, зерно `seed`) + врезки (Task 2) → кандидаты → признаки → метки; ключи `X`, `y`, `pool`, `truth`, `plan`, `valid`.
- [ ] **Step 1: тесты** `test_labels_from_truth` (кандидат на следе → 1, параллельный в 300 м → 0, наполовину на следе → 1).
- [ ] **Step 2–4:** FAIL → реализовать → PASS.

### Task 5: Обучение, порог, перенос
**Files:** Modify: `experiments/openlin_syn1.py`
**Interfaces:** Produces:
- `train_syn(samples: list[dict], log) -> (model, params)` — `openlin_anfl.train` на склеенных `X`, `y` листов обучения.
- `choose_threshold(real: list[dict], fakeB: list[dict], model, log) -> (thr: float, share: float)` — `real`/`fakeB`: кандидаты и признаки настоящих листов без врезок и подделок Б (зерно своё); наименьший порог по сетке `openlin_anfl.N_THR` квантилей, при котором Σкм выбранного на подделках Б ≤ 0,20 · Σкм выбранного на настоящих; `share` — доля км пула, выбранная на настоящих при этом пороге. При 0 км на настоящих — печать и `SystemExit`.
- [ ] **Step 1: тест** `test_choose_threshold_toy` (игрушечные оценки: подделка ниже снимка → порог существует, доля в (0,1]; подделка = снимок → правило не выполнимо → `SystemExit`).
- [ ] **Step 2–4:** FAIL → реализовать → PASS.

### Task 6: Проверки и вывод
**Files:** Modify: `experiments/openlin_syn1.py`
**Interfaces:** Produces: `run(log)` (ключ `--run`): обучение на `fam_r1–r3` и Анабаре (подделки А с врезками) → порог (Task 5) → Утсйоки: (а) настоящий фон + врезки (своё зерно): доля найденных врезок по корзинам контраста и по видам при доле длины `share`, рядом — потолок (все кандидаты); (б) мусор подделки Б ≤ 20 % по каждому листу; (в) распределения признаков: кандидаты-врезки против выбранных настоящих (10/50/90 %); → выбор на настоящих Утсйоки и Анабаре без врезок; npz, gpkg (`openlin_anfl.export_gpkg`, подпись учителя), json-итог, правила Анфисы, sha кода/входов. `windows(log)` (ключ `--windows`) — окна как ANFL0b в `outputs/fam/syn1_<лист>/`.
- [ ] **Step 1:** прогон на куске одного листа (уменьшенное `n_ins`) — цепочка доходит до конца, все файлы пишутся в `outputs/fam/` под именем `syn1_*`.
- [ ] **Step 2:** `pytest tests/test_openlin_syn.py tests/test_openlin_invariants.py -q` → PASS.

### Task 7: Прогон и проверки ветки
- [ ] **Step 1:** `openlin-keeper` — проверка кода против объявления, тесты, быстрый кусок.
- [ ] **Step 2:** `CUDA_VISIBLE_DEVICES="" python -m experiments.openlin_syn1 --run` (в фоне).
- [ ] **Step 3:** `openlin-artifact-hunter` по выбору на Анабаре и Утсйоки (тяга к долинам и краям пятен — рядом с CAND3/ANFL0b, швы, вода, освещение).
- [ ] **Step 4:** `python -m experiments.openlin_syn1 --windows`.
- [ ] **Step 5:** `openlin-auditor`; итог — `audit/SYN1_RESULTS.md`; отчёт Артуру. Коммит — только с его разрешения.
