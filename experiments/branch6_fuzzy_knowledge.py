"""Ветка 6: знаниевая нечёткая карта по минерагеническим ролям.

ЗАЧЕМ. Все модели проекта учатся на критериальном прогнозе — значит их согласие
с ним частично куплено самим обучением, а на смежной территории, где мишени
нет, они работают по похожести признаков. Нечёткая модель — единственное
семейство, которое мишень НЕ видит вовсе: принадлежности задаются в физических
величинах по геологической трактовке признака, агрегация — оператором, а не
подобранными весами. Поэтому её согласие с критериальной картой нельзя списать
на подгонку, а перенос на смежную территорию не требует переобучения.

ЧТО ЗАДАНО РУКАМИ И ОТКУДА.

ГДЕ МОДЕЛЬ ВСЁ-ТАКИ КАСАЕТСЯ МИШЕНИ. Принадлежности, направления, пороги и
выбор варианта по умолчанию мишени не видят. Но постобработочное сглаживание
(5 км x 1.25 км вдоль северо-запада) взято готовым из ветки 3, где форма
выбиралась по согласию с критериальной картой. Поэтому вариант «без
сглаживания» считается и печатается всегда — он и есть полностью независимая
цифра.

* Роли — `data/processed/mineragenic_feature_set.json` (03.09.2026): околорудные
  изменения, окисление сульфидов, структурный контроль, ловушка-палеодолина.
* Направление благоприятности каждого признака — из геологии (CLAUDE.md,
  раздел о признаках золотого оруденения), а НЕ из корреляции с мишенью: иначе
  мишень попадёт в модель через чёрный ход.
* Роль «глубинные тела и поля» (gm/pf) исключена по двум причинам сразу:
  направление благоприятности для остаточной гравики и магнитного поля из
  литературы однозначно не выводится (магнитные минимумы над разрушением
  магнетита против положительных аномалий над телами), а сами группы измеренно
  непереносимы на смежную территорию.
* `dist_dnara` исключён: гидросеть оцифрована только в пределах листа.

ПОРОГИ. Функция принадлежности — линейный подъём между двумя порогами.
Геологи организации пороги в физических величинах не задавали, поэтому они
берутся как процентили распределения ПО ЛИСТУ (60-й и 90-й для «чем больше, тем
благоприятнее»). Это компромисс, и он честно ограничивает силу утверждения:
пороги зависят от данных листа, но не от мишени. Главное — на смежной
территории применяются ТЕ ЖЕ числа в физических единицах, а не пересчитанные по
её собственному распределению: иначе перенос превратился бы в нормировку и
стёр реальные различия территорий.

ОПЕРАТОРЫ. Внутри роли и между ролями перебираются нечёткое ИЛИ, среднее,
гамма-операторы, произведение и минимум. Выбор варианта по совпадению с
критериальной картой был бы подгонкой, поэтому вариант по умолчанию выбран по
свойству, мишени не касающемуся, — контрастности (нечёткое ИЛИ по 9-14
скоррелированным индексам насыщается к единице и обнуляет различия между
ячейками, это измеряется и показывается). Остальные варианты печатаются как
проверка чувствительности.

Пропуск в слое — нейтральная принадлежность 0.5 (жёсткое И иначе убивало бы
ячейку: ASTER покрывает 70% листа и 39% широкой сетки).

Выход: `outputs/metrics/branch6_fuzzy_variants.csv`, `branch6_fuzzy_points.csv`,
       `data/processed/fuzzy_thresholds.json`, `data/processed/fuzzy_wide.parquet`,
       `outputs/branch6_fuzzy.png`.
Запуск из корня: ``python -m experiments.branch6_fuzzy_knowledge``
"""

import json
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (assessment, cell_mask, config, crit_reference,  # noqa: E402
                 criterial_target)
from experiments.honest_feature_signal import poly_design, trend_share  # noqa: E402
from experiments.fetch_wide_area import WIDE_META  # noqa: E402
from experiments.branch3_shapes import detrend  # noqa: E402
from src import neighbourhood as nb  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="An input array is constant.*")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T0 = time.time()

NW_AZIMUTH = 315.0          # простирание Билляхской тектоно-флюидитной зоны
ORIENT_KM = 5.0

#: роль -> [(признак, направление благоприятности)]
#: "выше" — чем больше значение, тем благоприятнее; "ниже" — наоборот;
#: "по модулю" — благоприятен любой сильный отклон от нуля (перегиб рельефа);
#: "северо-запад" — ось структур близка к простиранию рудных зон.
ROLES: dict[str, list[tuple[str, str]]] = {
    # Гидротермальная проработка: Al-OH (серицит-мусковит), каолинит, карбонаты,
    # Mg-OH (хлорит) — прямое соответствие описанной для Хаптасыннахской зоны
    # серицитизации и карбонатизации. Изменчивость индекса в ячейке (_std) —
    # признак контрастной, зональной проработки, а не ровного фона.
    "околорудные изменения": [
        ("ast_aloh", "выше"), ("ast_aloh_std", "выше"),
        ("ast_kaolin", "выше"), ("ast_kaolin_std", "выше"),
        ("ast_alter", "выше"), ("ast_alter_std", "выше"),
        ("ast_carb", "выше"), ("ast_carb_std", "выше"),
        ("ast_mgoh", "выше")],
    # Железные шляпы: окисление сульфидов рудных зон даёт окисные и закисные
    # формы железа, видимые и в ASTER, и в Sentinel-2, и в Landsat.
    "окисление сульфидов": [
        ("ast_ferric", "выше"), ("ast_ferric_std", "выше"),
        ("ast_ferrous", "выше"), ("ast_ferrous_std", "выше"),
        ("s2_iron_ox", "выше"), ("s2_iron_ox_std", "выше"),
        ("s2_ferrous", "выше"), ("s2_ferrous_std", "выше"),
        ("s2_ferric", "выше"), ("s2_ferric_std", "выше"),
        ("s2_clay_std", "выше"),
        ("l8_iron_ox", "выше"), ("l8_iron_ox_std", "выше"),
        ("l8_ferrous", "выше")],
    # Структурный контроль: густая сеть линеаментов, узлы их пересечения,
    # близость к линеаменту, согласованное простирание, перегибы рельефа.
    "структурный контроль": [
        ("lin_dens", "выше"), ("lin_node_dens", "выше"),
        ("lin_dist", "ниже"), ("lin_aniso_r", "выше"),
        ("__ось_сз", "северо-запад"),
        ("relief2_curv_5km", "по модулю"), ("relief2_tri_std", "выше")],
    # Ловушка: палеодолинный контроль — тальвеги, крупный водосбор, депрессии.
    "ловушка-палеодолина": [
        ("dem_incision", "выше"), ("relief2_catch_log", "выше"),
        ("dem_tpi_500", "ниже")],
}
NA_MODES = ("пропуск не учитывается", "пропуск = нейтральные 0.5")
DEFAULT_NA = NA_MODES[0]
DEFAULT_INNER = "среднее"
DEFAULT_OUTER = "гамма 0.7"


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def axis_membership(df: pd.DataFrame, shape, dx: float) -> np.ndarray:
    """Близость местной оси структур к северо-западному простиранию (0..1).

    Направления линеаментов заданы только в тех ячейках, где линеамент прошёл,
    поэтому берётся сглаженное поле ориентаций (как в ветке 3), а не сырые
    столбцы.
    """
    w = df["lin_dens"].to_numpy(float) * df["lin_aniso_r"].to_numpy(float)
    theta, _ = nb.orientation_field(df["lin_dir_sin"].to_numpy(float),
                                    df["lin_dir_cos"].to_numpy(float), w, shape,
                                    ORIENT_KM * 1000.0 / dx)
    return 0.5 * (1.0 + np.cos(2.0 * (theta - nb.theta_from_azimuth(NW_AZIMUTH))))


def fit_thresholds(df: pd.DataFrame, valid: np.ndarray) -> dict:
    """Пороги принадлежности в физических величинах — по распределению листа."""
    th = {}
    for role, items in ROLES.items():
        for col, direction in items:
            if col not in df.columns or direction == "северо-запад":
                continue
            v = df.loc[valid, col].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size < 100:
                continue
            if direction == "по модулю":
                v = np.abs(v)
            lo, hi = (60, 90) if direction != "ниже" else (40, 10)
            th[col] = [float(np.percentile(v, lo)), float(np.percentile(v, hi))]
    return th


def memberships(df: pd.DataFrame, th: dict, shape, dx: float) -> dict:
    """Принадлежности по ролям: роль -> матрица (ячейки x признаки)."""
    out, names = {}, {}
    for role, items in ROLES.items():
        cols, used = [], []
        for col, direction in items:
            if direction == "северо-запад":
                cols.append(axis_membership(df, shape, dx))
                used.append(col)
                continue
            if col not in df.columns or col not in th:
                continue
            v = df[col].to_numpy(float)
            if direction == "по модулю":
                v = np.abs(v)
            a, b = th[col]
            mu = np.clip((v - a) / (b - a), 0.0, 1.0) if b != a else np.full(v.size, 0.5)
            cols.append(np.where(np.isfinite(v), mu, np.nan))
            used.append(col)
        out[role] = np.column_stack(cols)
        names[role] = used
    return out, names


def aggregate(M: np.ndarray, op: str) -> np.ndarray:
    """Агрегация принадлежностей (по столбцам), пропуски не участвуют.

    Пропуск (слоя нет в этой ячейке) не заменяется числом: он выпадает из
    агрегации, а если у роли не измерен ни один признак — роль становится
    пропуском и выпадает из агрегации между ролями. Произведение и нечёткое ИЛИ
    берутся в степени 1/n по числу измеренных термов: иначе пропуск работает как
    множитель-единица и ячейка вне съёмки получает БОЛЬШЕ, чем измеренная, —
    контур съёмки снова становится признаком, только с обратным знаком.
    Минимум оставлен как есть (он по своей природе зависит от числа термов) и
    годится только для сравнения. Заполнение пропуска
    нейтральной 0.5, как советует классическое изложение нечёткой логики, здесь
    измеренно вредно: 30% листа без съёмки ASTER получали ровно 0.5 при том, что
    типичная измеренная принадлежность роли около 0.21, и весь верхний дециль
    роли уезжал за контур съёмки. Вариант с 0.5 остался как сравнение
    (см. NA_MODES).
    """
    n = np.maximum(np.isfinite(M).sum(axis=1), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prod = np.nanprod(M, axis=1) ** (1.0 / n)          # среднее геометрическое
        orr = 1.0 - np.nanprod(1.0 - M, axis=1) ** (1.0 / n)
        if op == "среднее":
            return np.nanmean(M, axis=1)
        if op == "нечёткое ИЛИ":
            return orr
        if op == "произведение":
            return prod
        if op == "минимум":
            return np.nanmin(M, axis=1)
        if op.startswith("гамма"):
            g = float(op.split()[1])
            return prod ** (1 - g) * orr ** g
    raise ValueError(op)


def as_mode(mem: dict, mode: str) -> dict:
    """Правило обращения с пропуском: нейтральные 0.5 или выпадение из агрегации."""
    if mode == NA_MODES[0]:
        return mem
    return {r: np.where(np.isfinite(M), M, 0.5) for r, M in mem.items()}


def fuzzy_score(mem: dict, inner: str, outer: str) -> np.ndarray:
    roles = np.column_stack([aggregate(M, inner) for M in mem.values()])
    return aggregate(roles, outer)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    shape = (meta.prf, meta.pic)
    # мишень одна на весь скрипт — верхние 10% критериального прогноза ВНУТРИ
    # рабочего пула; раньше ROC считался по квантилю всей сетки, а совпадение —
    # по квантилю пула, и это были слегка разные множества позитивов
    feat = criterial_target.training_features("v5_rebuilt")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()
    crit = crit_reference.criterial_score(prognoz)
    D = poly_design(cx, cy)
    pr_flat = prognoz.ravel()
    y = np.zeros(prognoz.size, dtype=int)
    top_c = (pr_flat <= np.quantile(pr_flat[pool], config.CREF_AREA)) & valid
    y[top_c] = 1

    th = fit_thresholds(feat, valid)
    (config.PROCESSED_DIR / "fuzzy_thresholds.json").write_text(
        json.dumps({"дата": "2026-09-05", "пороги": th,
                    "правило": "линейный подъём между процентилями 60 и 90 "
                               "распределения листа (для направления «ниже» — 40 и 10)"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    df_cols = set(feat.columns)
    mem, mem_names = memberships(feat, th, shape, meta.dx)
    cr_g = detrend(crit, pool, D)
    tcr_g = (cr_g >= np.quantile(cr_g[pool], 1 - config.CREF_AREA)) & valid
    log(f"ролей {len(mem)}, признаков в них "
        + ", ".join(f"{r}: {M.shape[1]}" for r, M in mem.items()))

    print("\n=== Насыщение операторов внутри роли (разброс принадлежности роли) ===")
    sat = []
    for role, M in mem.items():
        for op in ("нечёткое ИЛИ", "среднее", "гамма 0.7"):
            v = aggregate(M, op)[valid]
            sat.append({"роль": role, "оператор": op, "признаков": M.shape[1],
                        "измерено": float(np.mean(np.isfinite(v))),
                        "медиана": float(np.nanmedian(v)),
                        "разброс": float(np.nanstd(v)),
                        "межквартильный": float(np.nanpercentile(v, 75)
                                                - np.nanpercentile(v, 25))})
    sat = pd.DataFrame(sat)
    print(sat.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pts = assessment.load_verification_points(meta)
    allp = assessment.filter_points(pts["cell"].to_numpy(), valid)
    unb = assessment.filter_points(assessment.unbiased_cells(pts), valid)
    km = 1000.0 / meta.dx
    rows, maps = [], {}
    for na in NA_MODES:
      mem_na = as_mode(mem, na)
      for inner in ("среднее", "нечёткое ИЛИ", "гамма 0.7"):
          for outer in ("гамма 0.7", "гамма 0.5", "гамма 0.9", "среднее",
                        "произведение", "минимум"):
              s = fuzzy_score(mem_na, inner, outer)
              s = np.where(np.isfinite(s), s, np.nanmin(s[pool]))
              for post in ("без сглаживания", "сглаживание рабочей точки"):
                  v = (s if post == "без сглаживания" else
                       nb.aniso_smooth(np.where(valid, s, np.nan), shape, 5 * km,
                                       1.25 * km, nb.theta_from_azimuth(NW_AZIMUTH),
                                       mask=valid))
                  v = np.where(np.isfinite(v), v, np.nanmin(v[valid]))
                  name = f"{inner} | {outer} | {post}" + (
                      "" if na == DEFAULT_NA else f" | {na}")
                  maps[name] = v
                  top_m = (v >= np.quantile(v[pool], 1 - config.CREF_AREA)) & valid
                  vr, cr = detrend(v, pool, D), detrend(crit, pool, D)
                  tpr = vr >= np.quantile(vr[pool], 1 - config.CREF_AREA)
                  tcr = cr >= np.quantile(cr[pool], 1 - config.CREF_AREA)
                  agr = crit_reference.agreement(v, crit, pool, config.CREF_AREA)
                  rows.append({
                      "пропуски": na,
                      "внутри роли": inner, "между ролями": outer,
                      "постобработка": post,
                      "вырожденная": bool(np.std(v[pool]) < 1e-9
                                          or abs(top_m[pool].mean() - 0.10) > 0.02),
                      "реализованная_площадь": float(top_m[pool].mean()),
                      "совпадение_верх_10": (top_c & top_m).sum() / max(top_c.sum(), 1),
                      "совпадение_без_тренда": ((tpr & tcr & valid).sum()
                                                / max((tcr & valid).sum(), 1)),
                      "согласие_каппа": agr["kappa"],
                      "roc_auc": roc_auc_score(y[pool], v[pool]),
                      "тренд_R2": float(trend_share(v[:, None], D)[0]),
                      "точек_19": int(top_m[unb].sum()),
                      "точек_73": int(top_m[allp].sum()),
                      "новых_ячеек": int((top_m & ~top_c).sum())})
    var = pd.DataFrame(rows)
    var.to_csv(ROOT / "outputs" / "metrics" / "branch6_fuzzy_variants.csv",
               index=False, encoding="utf-8-sig")
    print("\n=== Нечёткая карта против критериальной (обучения нет вовсе) ===")
    print("Исключены варианты, где верхний дециль не набирается: карта постоянна "
          "или из-за массовых связок на нуле отсекается не 10% площади, а больше "
          f"— {int(var['вырожденная'].sum())} из {len(var)}.")
    print(var[~var["вырожденная"]].sort_values("совпадение_без_тренда", ascending=False)
          .drop(columns=["вырожденная"]).head(18)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nСреднее по правилу обращения с пропуском:")
    print(var[~var["вырожденная"]].groupby("пропуски")[
        ["совпадение_верх_10", "совпадение_без_тренда", "roc_auc", "точек_19"]]
        .mean().round(3).to_string())
    print(f"\nКритериальная карта: точек 19 -> {int(top_c[unb].sum())}, "
          f"73 -> {int(top_c[allp].sum())}; случайный уровень совпадения 0.10.")

    sel = {k: maps[k] for k in
           (f"{DEFAULT_INNER} | {DEFAULT_OUTER} | без сглаживания",
            f"{DEFAULT_INNER} | {DEFAULT_OUTER} | сглаживание рабочей точки",
            f"нечёткое ИЛИ | {DEFAULT_OUTER} | сглаживание рабочей точки")}
    # порог значимости — накопительная поправка на все опробованные варианты
    alpha = 0.05 / len(var)
    prot = assessment.run_protocol({**sel, "criterial": crit}, list(sel), meta, valid,
                                   alpha, baseline="criterial", log=log)
    prot["verdict"].to_csv(ROOT / "outputs" / "metrics" / "branch6_fuzzy_points.csv",
                           index=False, encoding="utf-8-sig")
    # «случайный уровень 0.10» — ожидание для независимых бросков, а не для двух
    # автокоррелированных карт с общей маской; сдвиговый null даёт настоящую
    # планку, блочный бутстрэп — интервал
    cp = crit_reference.run_protocol(sel, crit, meta, valid, alpha, log=log)
    cp["verdict"].to_csv(ROOT / "outputs" / "metrics" / "branch6_fuzzy_vs_crit.csv",
                         index=False, encoding="utf-8-sig")
    print()
    print("=== Согласие с критериальным эталоном: сдвиговый null и блочный интервал ===")
    print(cp["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("=== Точки заверки (19 непредвзятых, косвенная проверка) ===")
    print(prot["verdict"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------
    # Проверка допущений: верны ли заданные руками направления
    # ------------------------------------------------------------------
    # Слабый результат знаниевой модели объясняется двояко: роли не работают
    # ИЛИ направления благоприятности заданы неверно. Это разделяется прямым
    # измерением: у каждой принадлежности считается ROC-AUC против мишени.
    # Выше 0.5 — направление подтверждается данными, ниже — опровергается.
    # Сама модель этими числами НЕ пользуется, они только объясняют результат.
    dir_rows, flip = [], {}
    blocks = crit_reference.block_labels(meta, config.CREF_BLOCK)
    rng_b = np.random.default_rng(config.CRIT_SEED)
    uniq = np.unique(blocks[pool])
    draws = [np.concatenate([pool[blocks[pool] == b] for b in
                             rng_b.choice(uniq, uniq.size, replace=True)])
             for _ in range(120)]
    dirs_of = {c: d for role in ROLES for c, d in ROLES[role]}
    for role, cols_used in mem_names.items():
        for j, col in enumerate(cols_used):
            direction = dirs_of[col]
            mu = mem[role][:, j]
            ok = pool[np.isfinite(mu[pool])]
            au = float(roc_auc_score(y[ok], mu[ok])) if y[ok].sum() else float("nan")
            # интервал по блокам 10 км: на 22.9 тыс. соседних ячеек AUC 0.51
            # неотличим от 0.5, бинарное «подтверждается» без интервала обманчиво
            bs = []
            for idx in draws:
                i = idx[np.isfinite(mu[idx])]
                if 0 < y[i].sum() < i.size:
                    bs.append(roc_auc_score(y[i], mu[i]))
            lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
                      ) if len(bs) > 20 else (float("nan"), float("nan"))
            dir_rows.append({"роль": role, "признак": col,
                             "направление": direction,
                             "измерено_ячеек": int(ok.size), "auc_к_мишени": au,
                             "интервал_низ": lo, "интервал_верх": hi,
                             "подтверждается": bool(lo > 0.5),
                             "опровергается": bool(hi < 0.5)})
            flip[(role, j)] = au < 0.5
    dirs = pd.DataFrame(dir_rows)
    dirs.to_csv(ROOT / "outputs" / "metrics" / "branch6_fuzzy_directions.csv",
                index=False, encoding="utf-8-sig")
    print("\n=== Заданные руками направления против данных ===")
    print(dirs.sort_values("auc_к_мишени", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("Подтверждается интервалом: "
          f"{int(dirs['подтверждается'].sum())} из {len(dirs)}, опровергается "
          f"{int(dirs['опровергается'].sum())}, неотличимо от случайного "
          f"{int((~dirs['подтверждается'] & ~dirs['опровергается']).sum())}; "
          f"медиана |AUC-0.5| = {float((dirs['auc_к_мишени'] - 0.5).abs().median()):.3f}")

    print("\n=== Каждая роль поодиночке (среднее внутри роли) ===")
    role_rows = []
    for role, M in mem.items():
        v = aggregate(M, DEFAULT_INNER)
        v = np.where(np.isfinite(v), v, np.nanmin(v[pool]))
        top_m = (v >= np.quantile(v[pool], 1 - config.CREF_AREA)) & valid
        vr = detrend(v, pool, D)
        tpr = vr >= np.quantile(vr[pool], 1 - config.CREF_AREA)
        role_rows.append({"роль": role, "признаков": M.shape[1],
                          "совпадение_верх_10": (top_c & top_m).sum() / top_c.sum(),
                          "совпадение_без_тренда": ((tpr & tcr_g).sum()
                                                    / tcr_g.sum()),
                          "roc_auc": float(roc_auc_score(y[pool], v[pool])),
                          "точек_19": int(top_m[unb].sum())})
    print(pd.DataFrame(role_rows)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Верхняя граница: направления развёрнуты по мишени. Это НЕ модель —
    # мишень подсмотрена, — а ответ на вопрос «сколько вообще можно выжать из
    # этих ролей, если все направления угаданы правильно».
    mem_or = {r: np.column_stack([(1.0 - M[:, j]) if flip.get((r, j)) else M[:, j]
                                  for j in range(M.shape[1])])
              for r, M in mem.items()}
    for outer in (DEFAULT_OUTER, "среднее"):
        v = fuzzy_score(mem_or, DEFAULT_INNER, outer)
        v = np.where(np.isfinite(v), v, np.nanmin(v[pool]))
        v = nb.aniso_smooth(np.where(valid, v, np.nan), shape, 5 * km, 1.25 * km,
                            nb.theta_from_azimuth(NW_AZIMUTH), mask=valid)
        v = np.where(np.isfinite(v), v, np.nanmin(v[valid]))
        top_m = (v >= np.quantile(v[pool], 1 - config.CREF_AREA)) & valid
        vr = detrend(v, pool, D)
        tpr = vr >= np.quantile(vr[pool], 1 - config.CREF_AREA)
        log(f"верхняя граница (направления по мишени, {DEFAULT_INNER} | {outer}): "
            f"совпадение {(top_c & top_m).sum() / top_c.sum():.3f}, без тренда "
            f"{(tpr & tcr_g).sum() / tcr_g.sum():.3f}, ROC "
            f"{roc_auc_score(y[pool], v[pool]):.3f}, точек {int(top_m[unb].sum())} из 19")

    # ------------------------------------------------------------------
    # Перенос без переобучения: те же пороги в физических величинах
    # ------------------------------------------------------------------
    wide = pd.read_parquet(config.PROCESSED_DIR / "dataset_wide.parquet")
    wshape = WIDE_META.shape
    miss = [c for role in ROLES for c, d in ROLES[role]
            if d != "северо-запад" and c not in wide.columns]
    mem_w, _ = memberships(wide, th, wshape, WIDE_META.dx)
    sw = fuzzy_score(mem_w, DEFAULT_INNER, DEFAULT_OUTER)
    sw = np.where(np.isfinite(sw), sw, np.nanmin(sw))
    sw_sm = nb.aniso_smooth(sw, wshape, 5 * km, 1.25 * km,
                            nb.theta_from_azimuth(NW_AZIMUTH))
    wx, wy = wide["x"].to_numpy(float), wide["y"].to_numpy(float)
    in_sheet = ((wx >= cx.min() - 250) & (wx <= cx.max() + 250) &
                (wy >= cy.min() - 250) & (wy <= cy.max() + 250))
    ml = pd.read_parquet(config.PROCESSED_DIR / "forecast_wide_v2.parquet")
    log(f"перенос: нет на широкой сетке {len(miss)} признаков {miss}; "
        f"доля ячеек с пропуском ASTER {float(np.mean(~np.isfinite(wide['ast_kaolin']))):.3f}")
    out = wide[["row", "col", "x", "y"]].copy()
    out["fuzzy"] = sw.astype(np.float32)
    out["fuzzy_сглаженный"] = sw_sm.astype(np.float32)
    out["в_габаритах_листа"] = in_sheet
    out.to_parquet(config.PROCESSED_DIR / "fuzzy_wide.parquet", index=False)

    adj = ~in_sheet
    for nm, a, b in (("вся сетка", sw_sm, ml["score_сглаженный"].to_numpy()),
                     ("смежная территория", sw_sm[adj],
                      ml["score_сглаженный"].to_numpy()[adj])):
        ta = a >= np.quantile(a, 0.9)
        tb = b >= np.quantile(b, 0.9)
        log(f"нечёткая карта против модели переноса ({nm}): спирмен "
            f"{sps.spearmanr(a, b).statistic:.3f}, общих ячеек в верхних 10% "
            f"{100 * (ta & tb).sum() / max(ta.sum(), 1):.1f}%")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rank = lambda v: sps.rankdata(v).reshape(shape) / v.size          # noqa: E731
    best = maps[f"{DEFAULT_INNER} | {DEFAULT_OUTER} | сглаживание рабочей точки"]
    r = var[(var["внутри роли"] == DEFAULT_INNER) & (var["между ролями"] == DEFAULT_OUTER)
            & (var["постобработка"] == "сглаживание рабочей точки")].iloc[0]
    fig = plt.figure(figsize=(19, 6.4))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(rank(crit), cmap="magma")
    ax.set_title("Критериальный прогноз", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(rank(best), cmap="magma")
    ax.set_title(f"Нечёткая карта ({DEFAULT_INNER} / {DEFAULT_OUTER})\n"
                 f"совпадение {r['совпадение_верх_10']:.0%}, без тренда "
                 f"{r['совпадение_без_тренда']:.0%}, точек {int(r['точек_19'])} из 19",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 4, 3)
    for role, M in mem.items():
        ax.hist(aggregate(M, DEFAULT_INNER)[valid], bins=50, histtype="step",
                density=True, label=role)
    ax.set_title("Принадлежность по ролям на листе", fontsize=9)
    ax.set_xlabel("принадлежность (1 — благоприятно)")
    ax.set_ylabel("плотность")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 4, 4)
    im = ax.imshow(sw_sm.reshape(wshape), origin="upper", cmap="viridis",
                   extent=[WIDE_META.x0, WIDE_META.x0 + WIDE_META.pic * WIDE_META.dx,
                           WIDE_META.y0, WIDE_META.y_top])
    ax.plot([cx.min(), cx.max(), cx.max(), cx.min(), cx.min()],
            [cy.min(), cy.min(), cy.max(), cy.max(), cy.min()], color="white", lw=1.2)
    ax.set_title("Перенос без переобучения: те же пороги\nна широкой сетке", fontsize=9)
    ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Ветка 6: знаниевая нечёткая модель по минерагеническим ролям",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "branch6_fuzzy.png", dpi=130)
    log("OK: outputs/branch6_fuzzy.png, data/processed/fuzzy_wide.parquet")


if __name__ == "__main__":
    main()
