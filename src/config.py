"""Все настройки проекта.

Единственное место для констант пайплайна. В остальных модулях не должно быть
хардкодных чисел — они импортируют значения отсюда.
"""

from pathlib import Path

# --- Геометрия сетки ---
CELL_SIZE: int = 500          # размер ячейки сетки, м
RANDOM_STATE: int = 42

# --- Поиск данных ---
# Каталоги-кандидаты, в которых ищется папка shp_dbf с эталонным слоем.
BASE_DIR_CANDIDATES: list[Path] = [
    Path.cwd(),
    Path.cwd() / "data" / "Gis-integro",
    Path.cwd().parent / "data" / "Gis-integro",
    Path("/mnt/data/prog_zip"),
    Path("/mnt/data"),
    Path(r"C:\Users\janfi\OneDrive\Desktop\Прочее\Прогноз"),
]
# Архивы с данными, которые распаковываются, если папка не найдена напрямую.
ZIP_CANDIDATES: list[Path] = [
    Path.cwd() / "data" / "Прогноз.zip",
    Path("/mnt/data/Прогноз.zip"),
    Path.cwd() / "Прогноз.zip",
]
SHP_SUBDIR: str = "shp_dbf"             # подпапка с шейп-файлами
MASK_SENTINEL: str = "svita_new.shp"    # признак валидной папки shp_dbf
OUT_SUBDIR: str = "ml_png_result"       # подпапка для результата
OUT_PNG_NAME: str = "forecast_ml_final.png"

# --- Датасет v1 (золото, лист R-48-XI,XII): источники и общая сетка ---
# Целевая сетка = сетка критериального расчёта prognoz.pgrid (500 м, 154x149):
# метки и baseline лягут на неё без пересчёта.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
GOLD_TARGET_PGRID: Path = PROJECT_ROOT / "data" / "Gis-integro" / "Расчет" / "prognoz.pgrid"
SBORKA_DIR: Path = PROJECT_ROOT / "data" / "SBORKA_DOP"
GRAVMAG_PGRID: Path = SBORKA_DIR / "ГРАВИКА_МАГНИТКА" / "грав_маг.pgrid"
LANDSAT_PGRID: Path = SBORKA_DIR / "КОСМОСНИМОК" / "landsat_fragm.pgrid"
TOPO5_PGRID: Path = SBORKA_DIR / "РЕЛЬЕФ" / "topo5_new.pgrid"
TOPO_SHP_DIR: Path = SBORKA_DIR / "ТОПО"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
TOPO5_TO_METERS: float = 0.2      # единицы topo5_new = метры*5 (регрессия на Copernicus DEM, 19.07)
LANDSAT_NODATA: int = 0           # DN=0 в landsat_fragm — фон повёрнутой сцены, не данные
ANGLE_PROPS: tuple[str, ...] = ("gr_1GFI_25", "mag_1GFI_25")  # поля направлений: ресемплинг через sin/cos

# --- Слои (роль -> базовое имя shapefile без расширения) ---
LAYER_FILES: dict[str, str] = {
    "mask": "svita_new",         # маска территории (свиты)
    "facies": "fasii",           # литолого-фациальный
    "tect1": "glub_raz_nw",      # разломы СЗ
    "tect2": "glub_r_nw",        # разломы СВ
    "paleo": "gr_dol_vp_poly",   # палеодолины
    "struct": "kory",            # коры выветривания
    "magm": "dayki_buf",         # дайки (магматогенный)
}
# Базовые слои-факторы; всё остальное в папке трактуется как точки рудопроявлений.
BASE_LAYER_NAMES: set[str] = set(LAYER_FILES.values())

# --- Близость: преобразование расстояния -> близость (transform, quantile) ---
PROXIMITY_PARAMS: dict[str, tuple[str, float]] = {
    "facies": ("cbrt", 0.78),
    "paleo": ("cbrt", 0.76),
    "struct": ("sqrt", 0.72),
    "magm": ("sqrt", 0.42),
    "tect1": ("cbrt", 0.74),
    "tect2": ("cbrt", 0.74),
}

# --- Обучение (presence-background) ---
# positive — только реальные рудопроявления, фон — случайная выборка ячеек.
# Псевдометки по geo_score намеренно НЕ используются (вносят циркулярность).
TRAIN_N_BACKGROUND: int = 3000   # размер случайного фона для боевого обучения
TRAIN_SEED: int = 42
MIN_POS_CELLS: int = 10           # минимум реальных точек для запуска ML

# --- Random Forest ---
RF_N_ESTIMATORS: int = 800
RF_MAX_DEPTH: int = 7
RF_MIN_SAMPLES_LEAF: int = 12
RF_MIN_SAMPLES_SPLIT: int = 20

# --- Gradient Boosting (кандидат; параметры подобраны validation.tune_gb, сиды 1,7) ---
GB_N_ESTIMATORS: int = 200
GB_MAX_DEPTH: int = 3
GB_LEARNING_RATE: float = 0.05
GB_SUBSAMPLE: float = 0.8

# --- Ансамбль RF+GB с усреднением по фону (кандидат против чистого RF) ---
# На каждом раунде берётся подвыборка отрицательного класса (фона) долей
# ENS_BG_FRAC, на (позитивы + подвыборка) обучаются RF и GB; итоговая вероятность
# — среднее по всем 2*ENS_N_ROUNDS моделям. Снижает дисперсию из-за случайного
# выбора фона и за счёт декорреляции RF/GB.
ENS_N_ROUNDS: int = 8            # число фоновых подвыборок (тюнинг на сидах 1,7)
ENS_BG_FRAC: float = 0.7         # доля отрицательного класса в каждой подвыборке

# --- Сглаживание прогнозной поверхности ---
SMOOTH_PASSES: int = 4             # проходов сглаживания итоговой карты

# --- Обогащённые признаки (mineral-systems из тех же векторных слоёв) ---
# Плотность каналов/источника и многомасштабная близость — то, чего нет в
# «расстоянии до ближайшего». Набор отобран честной held-out проверкой
# (experiments/feat_prune.py): из 9 кандидатов значимый вклад дали ровно эти 4
# (top-5% lift +0.69x к базовому, 95% ДИ [+0.28, +1.15]). Отброшены структурные
# узлы и широкие ореолы палео/магмы/тектоники (важность ниже средней).
DENSITY_RADIUS: int = 2500            # радиус для плотности разломов/даек, м
WIDE_PROXIMITY_TRANSFORM: str = "sqrt"  # трансформация широкого ореола
WIDE_PROXIMITY_Q: float = 0.95        # квантиль масштаба широкого ореола
WIDE_PROXIMITY_ROLES: tuple[str, ...] = ("facies", "struct")  # слои с широким ореолом

# --- Признаки рельефа (Copernicus DEM GLO-30, см. src/dem_features.py) ---
# Производные топографии — независимый от векторных слоёв канал. Набор прошёл
# честную held-out проверку (experiments/dem_prune.py, 8 сидов): top-5% +1.04x
# (95% ДИ [+0.54,+1.55]), top-10% +0.56x ([+0.17,+0.95]). dem_tpi (топоположение)
# — важнейший признак из всех, ловит палеодолины. Данные качаются с AWS в кэш;
# при недоступности признаки заполняются нейтрально (graceful fallback).
DEM_PAD: float = 0.05                 # padding bbox сетки для DEM, градусы
DEM_RES: tuple[float, float] = (0.004, 0.0016)  # шаг DEM-растра (lon, lat), ~150 м на 71° N
DEM_TPI_WINDOW: int = 15              # окно topographic position index, пикселей (~2 км)
DEM_ROUGH_WINDOW: int = 9             # окно шероховатости, пикселей

# --- Признаки модели (geo_score сюда НЕ входит) ---
FEATURE_COLS: list[str] = [
    "prox_facies", "prox_paleo", "prox_struct", "prox_magm",
    "prox_tect1", "prox_tect2", "tect_combo", "tect_intersection",
    "tect_magm_intersection", "tect_struct_intersection",
    "paleo_struct_intersection", "coincidence_score", "tect_only_penalty",
    # обогащение (отобрано прунингом, см. выше):
    "dens_tect", "dens_magm", "prox_facies_wide", "prox_struct_wide",
    # рельеф Copernicus DEM (отобрано dem_prune, см. src/dem_features.py):
    "dem_elev", "dem_slope", "dem_curv", "dem_tpi", "dem_rough",
]

# --- Критериальный baseline = метод ГИС Интегро «Таксономия по критериям» ---
# Точная реализация нативного расчёта ГИС Интегро (data/Gis-integro/Расчет):
#   1) расстояние до фактора -> степенная трансформация (симметризация гистограммы);
#   2) min-max нормировка каждого критерия в [0, 1];
#   3) эталон = минимум по критерию (= 0 после min-max; расстояние 0 = перспективно);
#   4) взвешенное манхэттенское (L1) расстояние до эталона (мера Плюты), меньше = лучше.
# Метрика (L1) и нормировка подтверждены символами igk_prognose.dll и подбором по
# нативному prognoz.prognoz.property (Spearman ≈ 0.98). Трансформации и вес struct=0.5
# взяты из постановки «Блок Прогноз».
# Используется ТОЛЬКО как baseline в валидации, не в боевом ML-прогнозе.
TAXONOMY_TRANSFORMS: dict[str, str] = {
    "facies": "cbrt",   # литолого-фациальный (lyth):   x^(1/3)
    "paleo": "cbrt",    # палеогеоморфологический:      x^(1/3)
    "struct": "sqrt",   # структурно-литологический:    sqrt(x)
    "magm": "sqrt",     # магматогенный:                sqrt(x)
    "tect1": "cbrt",    # тектонический СЗ (tect_nw):    x^(1/3)
    "tect2": "cbrt",    # тектонический СВ (tect_ne):    x^(1/3)
}
# Веса критериев восстановлены регрессией prognoz на min-max нормированные критерии
# (OLS без интерсепта, R^2 = 1.000, Spearman = 1.000 с нативным prognoz.prognoz.property).
# Понижен магматогенный фактор (magm ≈ 0.5), тектоника — максимальный вес.
TAXONOMY_WEIGHTS: dict[str, float] = {
    "tect2": 1.000,   # тектонический СВ (tect_ne)
    "tect1": 0.997,   # тектонический СЗ (tect_nw)
    "paleo": 0.923,   # палеогеоморфологический
    "facies": 0.891,  # литолого-фациальный (lyth)
    "struct": 0.849,  # структурно-литологический
    "magm": 0.493,    # магматогенный (понижен)
}

# --- Итоговая поверхность: веса локального бонуса ---
LOCAL_BONUS_WEIGHTS: dict[str, float] = {
    "tect_intersection": 0.38,
    "tect_magm_intersection": 0.37,
    "tect_struct_intersection": 0.25,
}

# --- Золотые зоны: ОТНОСИТЕЛЬНЫЙ отбор top-N связных ядер ---
# Берём фиксированное число сильнейших связных пятен из пула верхних ячеек, а не
# пересечение абсолютных порогов. Устойчиво к смене распределения/набора
# признаков (старый абсолютный фильтр «пустел» при добавлении DEM).
GOLD_SEED_Q: float = 0.98       # пул кандидатов = верхние (1−q) доли по prospectivity
GOLD_MAX_ZONES: int = 30        # предохранитель-cap: не больше стольких ядер (от шумовой россыпи)
GOLD_ZONE_MIN_CELLS: int = 4    # минимальный размер связной зоны (компактность)

# --- Честная валидация (presence-background + пространственный CV) ---
VAL_N_SPLITS: int = 5            # число фолдов пространственной CV
VAL_BLOCK_SIZE: int = 10000      # размер пространственного блока для группировки, м
VAL_N_BACKGROUND: int = 2000     # размер случайной фоновой выборки
VAL_SEED: int = 42
VAL_AREAS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)  # доли площади top-N для метрики
VAL_SEEDS: tuple[int, ...] = (1, 7, 13, 21, 42, 99)      # сиды для повторной CV
PERM_N: int = 80             # число перестановок для permutation-теста
PERM_AREA: float = 0.10      # площадь top-N для lift в permutation-тесте
PERM_RF_TREES: int = 300     # деревьев RF в permutation (компромисс скорость/шум)

# --- Контроль качества / визуализация ---
POINT_COVERAGE_Q: float = 0.85  # верхние 15% прогноза для метрики покрытия точек
N_DISPLAY_CLASSES: int = 20     # число классов на карте
SHOW_POINTS: bool = False       # показывать точки рудопроявлений на карте
