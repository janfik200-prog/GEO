"""SYN1: Анфиса учится на искусственных линиях (audit/SYN1_DECLARATION.md).

    python -m experiments.openlin_syn1 --run        # обучение, порог, проверки (а)–(в), выбор, npz/gpkg/json
    python -m experiments.openlin_syn1 --windows    # окна (после охотника)
    ... --crop 600 [--n 200]                        # отладка: центральные куски, вывод в outputs/fam/_syn1_crop600/

Задача 4 плана: метка «у врезки» и сборка обучения для одного листа — подделка А
(общая фазовая, Task 1) → врезки (Task 2) → кандидаты CAND3 по массивам → 20 признаков
ANFL0b → метки. Задача 5 — порог по подделке Б. Задача 6 — проверки на Утсйоки и вывод.
SYN1_N_TESTS = 0. Лидар не используется. Только процессор (CUDA_VISIBLE_DEVICES="").
GTK и разметка геологов сюда не входят.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from experiments import openlin_anfl as A
from experiments import openlin_anfl0b as B
from experiments import openlin_cand1 as C1
from experiments import openlin_cand2 as C2
from experiments import openlin_syn_insert as S
from experiments.openlin_acon1 import TAU, angdiff   # τ = π/8 = 22,5° — допуск направления LSD (поправка 1 п. 3)
from experiments.openlin_fam import DEM, OUT, RES_IN
from experiments.openlin_zone_lines_v3 import CONF_FRAC   # 0,3 — доля подтверждения тракта: печать рядом (поправка 1 п. 3)
from experiments.openlin_zone_lines_v3 import CONF_TOL_M   # 60 м — допуск подтверждения ветки (объявление)

SYN1_N_TESTS = 0
SYN1_N_TESTS_PRIOR = 160               # поправка 1: RELU4 154 + 6; сдвигов нет, α не определяется
SITES = ("fam_r1", "fam_r2", "fam_r3", "anabar", "utsjoki")   # i — номер листа в этом списке (объявление)
SEED_FAKE_A = 3100                     # подделка А листа i — 3100 + i (обучение)
SEED_FAKE_B = 3300                     # подделка Б листа i — 3300 + i (порог)
SEED_PLAN_SHIFT = 100                  # план врезок — зерно подделки + 100 (3200 + i; решение контролёра)
N_PROPOSALS = 2000                     # предложений врезок на лист — бюджет поиска (поправка 1 п. 1)
LABEL_FRAC = A.LABEL_FRAC              # 0,5: «да» — ≥ 50 % ячеек в допуске (как CAND3/ANFL0)
NEAR_M = S.SEP_M                       # 1000 м = LOCAL_WIN_M/2: печать доли «нет» у следа (поправка 1)
AMP_BINS = S.AMP_SIG[0] * 2.0 ** np.arange(6)   # 5 корзин контраста по октавам 0,25…8σ (поправка 1)


# ---------------------------------------------------------------- метки
def _trace_dist_m(truth, shape) -> np.ndarray:
    """Расстояние (м) от каждой ячейки до ближайшей ячейки следа любой врезки; без следов — inf."""
    m = np.zeros(shape, bool)
    for t in truth:
        if len(t):
            m[t[:, 0], t[:, 1]] = True
    if not m.any():
        return np.full(shape, np.inf, "float32")
    return (distance_transform_edt(~m) * RES_IN).astype("float32")


def _cell_fracs(pool, d):
    """Доля ячеек кандидата в допуске при поле расстояний d (булево ≤ tol уже применено)."""
    return np.array([d[p["cells"][:, 0], p["cells"][:, 1]].mean() for p in pool], "float64")


def labels_from_truth(pool, truth: list[np.ndarray], shape, tol_m: float) -> np.ndarray:
    """Метки кандидатов по следам врезок (поправка 1, registrar): 1 — ≥ LABEL_FRAC ячеек кандидата
    в tol_m от следа какой-либо врезки; 0 — ни одной ячейки в tol_m; −1 — часть (> 0 и < LABEL_FRAC):
    «не учится», в обучение не входит (иначе Анфису учат отбрасывать куски врезок)."""
    if not len(pool):
        return np.zeros(0, "int64")
    near = _trace_dist_m(truth, shape) <= tol_m
    f = _cell_fracs(pool, near)
    return np.where(f >= LABEL_FRAC, 1, np.where(f > 0, -1, 0)).astype("int64")


# ---------------------------------------------------------------- сборка обучения
def _bins(values):
    return np.histogram(np.clip(values, AMP_BINS[0], AMP_BINS[-1]), AMP_BINS)[0]


def _log_plan(plan, n_ins, sig_img, sig_dem, log):
    """Печать плана: принято врезок, км, по видам и корзинам контраста (поправка 1 п. 1)."""
    kinds = {k: sum(p["kind"] == k for p in plan) for k in S.KINDS}
    log(f"    σ снимка {sig_img:.4g}, σ рельефа {sig_dem:.4g}; врезок принято {len(plan)} из {n_ins} предложений, "
        f"{sum(p['length_m'] for p in plan) / 1000:.1f} км; по видам: "
        + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    log("    по корзинам контраста " + ", ".join(
        f"{a:g}–{b:g}σ {n}" for a, b, n in zip(AMP_BINS[:-1], AMP_BINS[1:], _bins([p["amp"] for p in plan]))))


def joint_fake(img, dem, alos_raw, valid, seed: int):
    """Общая фазовая подделка [снимок, GLO-30, ALOS] зерном seed; маска NaN ALOS возвращается."""
    alos_raw = np.asarray(alos_raw, "float64")
    miss = ~np.isfinite(alos_raw)
    # подделке нужен конечный вход: NaN ALOS → медиана, после подделки маска NaN возвращается,
    # чтобы признак «доля без ALOS» работал как на настоящем листе
    alos_in = np.where(miss, np.nanmedian(alos_raw), alos_raw)
    f_img, f_dem, f_alos = S.joint_phase_randomize([img, dem, alos_in], np.asarray(valid, bool),
                                                   np.random.default_rng(seed))
    f_alos = f_alos.astype("float64")
    f_alos[miss] = np.nan
    return f_img, f_dem, f_alos


def sample_arrays(img, dem, alos_raw, valid, seed: int, n_ins: int, log, site: str = "массивы") -> dict:
    """Подделка А (зерно seed) + врезки (план — зерно seed + SEED_PLAN_SHIFT) → кандидаты → признаки →
    метки, по массивам в памяти. img/dem — как из load_inputs (NaN заполнены), alos_raw — ALOS с NaN."""
    valid = np.asarray(valid, bool)
    miss = ~np.isfinite(np.asarray(alos_raw, "float64"))
    f_img, f_dem, f_alos = joint_fake(img, dem, alos_raw, valid, seed)
    # контраст врезок — в σ того фона, в который врезаем (подделки; решение контролёра)
    from src.detection_limit import robust_sigma
    sig_img = robust_sigma(f_img[valid].astype("float64"))
    sig_dem = S.relief_sigma(f_dem, valid)
    plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(seed + SEED_PLAN_SHIFT), n_ins)
    img2, dem2, alos2, truth = S.apply_plan(f_img, f_dem, f_alos, plan, sig_img, sig_dem, RES_IN, valid=valid)
    alos2[miss] = np.nan
    log(f"  {site}: зерно подделки {seed}, плана {seed + SEED_PLAN_SHIFT}")
    _log_plan(plan, n_ins, sig_img, sig_dem, log)
    pool = A.candidates_arr(img2, dem2, valid, log)
    n0 = len(pool)
    pool = [p for p in pool if valid[p["cells"][:, 0], p["cells"][:, 1]].any()]   # как B.site_data
    X = B.line_features_arr(pool, img2, dem2, alos2, valid, log, site=site)
    y = labels_from_truth(pool, truth, valid.shape, CONF_TOL_M)
    d = _trace_dist_m(truth, valid.shape)
    no = np.flatnonzero(y == 0)
    near_no = (np.array([d[pool[i]["cells"][:, 0], pool[i]["cells"][:, 1]].min() for i in no]) < NEAR_M
               if len(no) else np.zeros(0, bool))
    log(f"    кандидатов {len(pool)} (без годных ячеек выведено {n0 - len(pool)}): «да» {(y == 1).sum()}, "
        f"«нет» {(y == 0).sum()}, вне обучения (часть на следе) {(y == -1).sum()}; "
        f"доля «нет» ближе {NEAR_M:.0f} м к следу {near_no.mean() if len(no) else float('nan'):.3f}")
    return {"X": X, "y": y, "pool": pool, "truth": truth, "plan": plan, "valid": valid,
            "sig_img": sig_img, "sig_dem": sig_dem}


# ---------------------------------------------------------------- обучение и порог
JUNK_RATE = 0.20                       # правило ветки: на подделке Б не более 20 % км от выбранного на снимке


def train_syn(samples: list[dict], log):
    """Анфиса на склеенных листах обучения; строки с меткой −1 («часть врезки») отбрасываются."""
    X = np.concatenate([s["X"] for s in samples], 0)
    y = np.concatenate([s["y"] for s in samples], 0)
    keep = y != -1
    log(f"  SYN1 обучение: строк {len(y)}, отброшено «часть» {int((~keep).sum())}, "
        f"«да» {int((y[keep] == 1).sum())}, «нет» {int((y[keep] == 0).sum())}")
    return A.train(X[keep], y[keep].astype("float64"), log)


def _km_union(pool, sel, shape) -> float:
    """км выбранного — объединение ячеек (как openlin_anfl0b.km_of)."""
    return float(A.sel_mask(pool, sel, shape).sum() * RES_IN / 1000)


def choose_threshold(real: list[dict], fakeB: list[dict], model, log):
    """Наименьший порог сетки (N_THR квантилей оценок настоящих листов, 2–98 %), при котором
    Σкм выбранного на подделках Б ≤ JUNK_RATE · Σкм выбранного на настоящих. Возвращает (thr, share),
    share = км выбранного на настоящих / км всего пула настоящих. Не выполнимо → SystemExit."""
    s_real = [np.asarray(model(r["X"]), "float64") for r in real]
    s_fake = [np.asarray(model(f["X"]), "float64") for f in fakeB]
    allr = np.concatenate(s_real) if s_real else np.zeros(0)
    if not len(allr):
        log("SYN1: у настоящих листов нет кандидатов — порог не определить")
        raise SystemExit("SYN1: порог не определён — пул настоящих листов пуст")
    km_pool = sum(_km_union(r["pool"], np.ones(len(r["pool"]), bool), r["valid"].shape) for r in real)
    if HONEST:                                 # SYN1b: верхний узел той же сетки (98 %), без правила 20 % (SYN1B_DECLARATION)
        thr = float(np.quantile(allr, 0.98))
        km_real = sum(_km_union(r["pool"], s >= thr, r["valid"].shape) for r, s in zip(real, s_real))
        km_fake = sum(_km_union(f["pool"], s >= thr, f["valid"].shape) for f, s in zip(fakeB, s_fake))
        if km_real == 0:
            raise SystemExit("SYN1b: при верхнем узле сетки выбрано 0 км")
        share = km_real / km_pool
        log(f"  SYN1b порог (верхний узел сетки, 98 %) {thr:.6g}: км на подделках Б {km_fake:.1f}, км на настоящих "
            f"{km_real:.1f}, доля мусора сводно {km_fake / km_real:.3f}, доля пула {share:.3f}")
        return thr, float(share)
    for thr in np.quantile(allr, np.linspace(0.02, 0.98, A.N_THR)):
        km_real = sum(_km_union(r["pool"], s >= thr, r["valid"].shape) for r, s in zip(real, s_real))
        km_fake = sum(_km_union(f["pool"], s >= thr, f["valid"].shape) for f, s in zip(fakeB, s_fake))
        if km_real > 0 and km_fake <= JUNK_RATE * km_real:
            share = km_real / km_pool
            log(f"  SYN1 порог {thr:.6g}: км на подделках Б {km_fake:.1f}, км на настоящих {km_real:.1f}, "
                f"доля пула {share:.3f}")
            return float(thr), float(share)
    log(f"SYN1: ни один порог сетки из {A.N_THR} не даёт км(подделка Б) ≤ {JUNK_RATE} · км(снимок)")
    raise SystemExit(f"SYN1: правило {JUNK_RATE:.0%} недостижимо ни при одном пороге сетки — остановка (поправка 1)")


def site_sample(site, seed: int, n_ins: int, log) -> dict:
    """Лист целиком: входы load_inputs + ALOS 20 м (обрезан как в openlin_anfl0b.line_features) →
    sample_arrays. Зерно подделки А выбирает вызывающий (3100 + i)."""
    img, dem, raw, valid, meta = load_site(site)
    d = sample_arrays(img, dem, raw, valid, seed, n_ins, log, site=site)
    d["meta"] = meta
    return d


CROP: int | None = None                # отладка: центральный кусок CROP×CROP ячеек (ключ --crop, тест)


def load_site(site):
    """Входы листа: load_inputs + ALOS 20 м как есть (с NaN), обрезан до формы valid
    (как openlin_anfl0b.line_features). При CROP — центральный кусок; meta сдвигается на кусок."""
    img, dem, valid, meta = C1.load_inputs(site)
    raw = np.load(DEM / f"dem_alos30_{site}_лист_20m.npy").astype("float64")[:valid.shape[0], :valid.shape[1]]
    if CROP:
        H, W = valid.shape
        r0, c0 = max(H // 2 - CROP // 2, 0), max(W // 2 - CROP // 2, 0)
        sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
        img, dem, raw, valid = img[sl].copy(), dem[sl].copy(), raw[sl].copy(), valid[sl].copy()
        meta = dict(meta, x0=meta["x0"] + c0 * RES_IN, y_top=meta["y_top"] - r0 * RES_IN)
    return img, dem, raw, valid, meta


# ---------------------------------------------------------------- задача 6: найдена ли врезка
TEACHER = "искусственные линии (SYN1)"   # подпись выхода (объявление, «Выход»); не GTK
SEED_PLAN_UTS = 3400                     # план врезок Утсйоки для (а) — зерно 3400 (объявление)
FOUND_FRAC = LABEL_FRAC                  # 0,5: врезка найдена — ≥ 50 % следа покрыто (определение; поправка 1 п. 3)
FOUND_FRAC_PRINT = CONF_FRAC             # 0,3: то же — печать рядом (правило подтверждения тракта, поправка 1 п. 3)
WIN_KM = (("utsjoki", 6), ("anabar", 8))  # окна как ANFL0b: Утсйоки 6 км, Анабар 8 км


def _dir_of(t):
    """Угол направления (строка, столбец)-вектора t в соглашении ядер: t = ±(−cos a, sin a), a ∈ [0, π)."""
    return np.arctan2(t[..., 1], -t[..., 0]) % np.pi


def insertion_dirs(ins: dict, cells, res_m: float = RES_IN) -> np.ndarray:
    """Местное направление врезки (касательная дуги) в ячейках её следа: ближайшая точка дуги
    (шаг res_m/4, как truth_cells) → длина дуги s → касательная u·cos κs + n·sin κs."""
    cells = np.asarray(cells, float).reshape(-1, 2)
    if not len(cells):
        return np.zeros(0)
    s = np.concatenate([np.arange(a, b + res_m / 8, res_m / 4) for a, b in ins["pieces"]])
    _i, j = cKDTree(S.curve_points(ins, s, res_m)).query(cells)
    u, n = S._frame(ins["az_deg"])
    ks = ins["curv"] * s[j]
    return _dir_of(np.cos(ks)[:, None] * u + np.sin(ks)[:, None] * n)


def cand_dirs(pool, img, dem) -> list[np.ndarray]:
    """Направление кандидата в каждой его ячейке — направление ядра своего канала (θ отклика C1.response,
    нормаль (sin θ, cos θ), линия (−cos θ, sin θ)), как поле «dir» в openlin_acon1.traced."""
    out = [None] * len(pool)
    chs = np.array([p["ch"] for p in pool], int)
    for ch in range(len(C1.CHANNELS)):
        idx = np.flatnonzero(chs == ch)
        if not len(idx):
            continue
        th = C1.response(ch, img, dem)[1] % np.pi
        for i in idx:
            c = pool[i]["cells"]
            out[i] = th[c[:, 0], c[:, 1]].astype("float64")
        del th
    return out


def recall_of_insertions(cells, dirs, truth, plan, tol_m: float, tau: float, res_m: float = RES_IN) -> np.ndarray:
    """Доля следа каждой врезки, покрытая ячейками выбора: ячейка следа покрыта, если в tol_m от неё есть
    ячейка выбора с направлением в ±tau от местного направления врезки (решение контролёра: параллельная
    настоящая черта другого направления врезку не находит). cells — (N, 2) ячейки выбора, dirs — (N,)
    их направления. Врезка без ячеек следа — NaN."""
    cov = np.full(len(truth), np.nan)
    cells = np.asarray(cells).reshape(-1, 2)
    dirs = np.asarray(dirs, float)
    tree = cKDTree(cells) if len(cells) else None
    for i, (t, ins) in enumerate(zip(truth, plan)):
        if not len(t):
            continue
        if tree is None:
            cov[i] = 0.0
            continue
        a = insertion_dirs(ins, t, res_m)
        nb = tree.query_ball_point(np.asarray(t, float), tol_m / res_m + 1e-6)
        hit = [len(js) > 0 and bool((angdiff(dirs[js], a[k]) <= tau).any()) for k, js in enumerate(nb)]
        cov[i] = float(np.mean(hit))
    return cov


def verdict_a(rates, f: float):
    """Вердикт (а) по долям найденных врезок в 5 корзинах контраста (поправка 1 п. 2): доля в наибольшей
    корзине > f, > доли в наименьшей и ρ Спирмена (корзина, доля) по непустым корзинам > 0. Пустая крайняя
    корзина (NaN) — «не определено», засчитывается как НЕ выполнено (решение контролёра).
    Возвращает (выполнено, текст, определён, ρ)."""
    from scipy.stats import spearmanr
    rates = np.asarray(rates, float)
    ok = np.isfinite(rates)
    rho = float(spearmanr(np.arange(len(rates))[ok], rates[ok])[0]) if ok.sum() >= 2 else float("nan")
    top, bot = rates[-1], rates[0]
    if not (np.isfinite(top) and np.isfinite(bot)):
        return False, "НЕ ВЫПОЛНЕНО (не определено: в крайней корзине нет врезок)", False, rho
    passed = bool(top > f and top > bot and rho > 0)
    return passed, ("ВЫПОЛНЕНО" if passed else "ПРОВАЛ — «Анфиса не переносит врезки на настоящий фон»"), True, rho


def _cells_dirs(pool, dirs, sel):
    idx = np.flatnonzero(sel)
    if not len(idx):
        return np.zeros((0, 2), int), np.zeros(0)
    return (np.concatenate([pool[i]["cells"] for i in idx]), np.concatenate([dirs[i] for i in idx]))


# ---------------------------------------------------------------- задача 6: прогон
def _pool_X(img, dem, alos, valid, log, label, site_check=None) -> dict:
    """Кандидаты CAND3 по массивам + 20 признаков ANFL0b (линии без годных ячеек — вне пула, как B.site_data)."""
    pool = A.candidates_arr(img, dem, valid, log, site_check=site_check)
    n0 = len(pool)
    pool = [p for p in pool if valid[p["cells"][:, 0], p["cells"][:, 1]].any()]
    log(f"  {label}: кандидатов {len(pool)} (без годных ячеек выведено {n0 - len(pool)}), "
        f"{_km_union(pool, np.ones(len(pool), bool), valid.shape):.1f} км")
    X = B.line_features_arr(pool, img, dem, alos, valid, log, site=label)
    return {"pool": pool, "X": X, "valid": valid}


def share_threshold(score, d, share) -> float:
    """Числовой порог листа по перенесённой доле: top_by_km на пуле листа при share · км пула
    (км пула — объединение ячеек, как ANFL0b); порог = наименьшая оценка выбранных. Пусто → +inf."""
    pool, shape = d["pool"], d["valid"].shape
    budget = share * _km_union(pool, np.ones(len(pool), bool), shape) * 1000 / RES_IN   # в ячейках
    order = np.argsort(-np.asarray(score), kind="stable")
    m = np.zeros(shape, bool)
    n_cells, last = 0, None
    for i in order:                      # бюджет и отбор — по объединению ячеек (кресты линий не считаются дважды)
        c = pool[i]["cells"]
        add = int((~m[c[:, 0], c[:, 1]]).sum()) if len(c) else 0
        if n_cells + add > budget:
            break
        m[c[:, 0], c[:, 1]] = True
        n_cells += add
        last = i
    return float(score[last]) if last is not None else float("inf")


def _q3(X):
    return [[float(v) for v in np.quantile(X[:, k], [0.1, 0.5, 0.9])] if len(X) else [float("nan")] * 3
            for k in range(X.shape[1])]


HONEST = False                           # SYN1b (ключ --honest): порог — верхний узел сетки, доля мусора печатается
JUNK_SITE: dict = {}                     # доля мусора по листам (км подделки Б / км снимка при числовом пороге листа)
FAKEB_ANABAR: list = []                  # подделка Б Анабара (для его доли мусора)


def _save(site, d, score, thr, share, meta, log) -> dict:
    """npz + gpkg выбора на настоящем листе (оценка ≥ числового порога листа), подпись учителя SYN1."""
    sel = score >= thr
    chs = np.array([p["ch"] for p in d["pool"]], int)
    km_pool = _km_union(d["pool"], np.ones(len(sel), bool), d["valid"].shape)
    km = _km_union(d["pool"], sel, d["valid"].shape)
    area = d["valid"].sum() * RES_IN ** 2 / 1e6
    log(f"  {site}: выбрано {int(sel.sum())} из {len(sel)} линий, {km:.1f} км ({100 * km / max(km_pool, 1e-9):.1f} % "
        f"пула), {km / max(area, 1e-9):.2f} км/км²; по каналам: "
        + ", ".join(f"{C1.CHANNELS[ch]} {int((sel & (chs == ch)).sum())}" for ch in range(5)))
    np.savez_compressed(OUT / f"syn1_{site}.npz", score=score, sel=sel, ch=chs, X=d["X"], thr=thr, share=share,
                        teacher=TEACHER)
    path = OUT / f"syn1_{site}.gpkg"
    if any(len(d["pool"][i]["poly"]) >= 2 for i in np.flatnonzero(sel)):
        old = A.TEACHER
        A.TEACHER = TEACHER                   # export_gpkg пишет глобальную подпись — на время меняем
        try:
            A.export_gpkg(path, d["pool"], sel, score, meta)
        finally:
            A.TEACHER = old
    else:                                     # пустой выбор — пустой слой с той же схемой
        import geopandas as gpd
        gpd.GeoDataFrame({"канал": [], "оценка": [], "учитель": [], "длина_м": []}, geometry=[],
                         crs=f"EPSG:{int(meta['epsg'])}").to_file(path, driver="GPKG")
    if site in JUNK_SITE and path.exists():    # поле «доля мусора листа» в векторах (SYN1b)
        import geopandas as gpd
        g = gpd.read_file(path)
        g["доля_мусора_листа"] = float(JUNK_SITE[site])
        g.to_file(path, driver="GPKG")
    log(f"  {OUT / f'syn1_{site}.npz'}; {path}")
    return {"выбрано линий": int(sel.sum()), "из": len(sel), "км": km, "доля пула": km / max(km_pool, 1e-9),
            "км/км²": km / max(area, 1e-9), "порог": thr}


def code_sha() -> dict:
    from experiments.openlin_len1 import sha
    return {n: sha(f) for n, f in (("syn1", __file__), ("syn_insert", S.__file__), ("anfl", A.__file__),
                                   ("anfl0b", B.__file__), ("cand1", C1.__file__), ("cand2", C2.__file__),
                                   ("cand3", A.C3.__file__))}


def input_sha() -> dict:
    from experiments.openlin_len1 import sha
    from experiments.openlin_transfer_check import COMP
    return {s: {"S2": sha(COMP / f"winter_{s}_20m_лист.npz"), "GLO-30": sha(DEM / f"dem_glo30_{s}_лист_20m.npy"),
                "ALOS": sha(DEM / f"dem_alos30_{s}_лист_20m.npy")} for s in SITES}


def run(log, n_ins: int = N_PROPOSALS):
    from src.detection_limit import robust_sigma
    log(f"SYN1_N_TESTS = {SYN1_N_TESTS}; SYN1_N_TESTS_PRIOR = {SYN1_N_TESTS_PRIOR} (поправка 1: сдвигов нет, "
        f"α не определяется); предложений врезок на лист {n_ins}; кусок {CROP or 'нет (весь лист)'}")
    res = {"объявление": "audit/SYN1_DECLARATION.md", "учитель": TEACHER, "SYN1_N_TESTS": SYN1_N_TESTS,
           "SYN1_N_TESTS_PRIOR": SYN1_N_TESTS_PRIOR, "предложений врезок": n_ins, "кусок": CROP}
    train_sites = SITES[:4]                    # fam_r1, fam_r2, fam_r3, anabar (i = 0…3); utsjoki (i = 4) — проверка
    samples, real, fakeB, metas = [], [], [], {}
    for i, site in enumerate(train_sites):
        log(f"=== лист обучения {site} (i = {i})")
        img, dem, raw, valid, metas[site] = load_site(site)
        samples.append(sample_arrays(img, dem, raw, valid, SEED_FAKE_A + i, n_ins, log, site=f"{site}: подделка А"))
        real.append(_pool_X(img, dem, raw, valid, log, f"{site}: снимок без врезок"))
        f_img, f_dem, f_alos = joint_fake(img, dem, raw, valid, SEED_FAKE_B + i)
        fakeB.append(_pool_X(f_img, f_dem, f_alos, valid, log, f"{site}: подделка Б (зерно {SEED_FAKE_B + i})"))
        del img, dem, raw, f_img, f_dem, f_alos
    log("=== обучение Анфисы на подделках А с врезками: " + ", ".join(train_sites))
    res["обучение"] = {s: {"врезок": len(d["plan"]), "кандидатов": len(d["y"]), "да": int((d["y"] == 1).sum()),
                           "нет": int((d["y"] == 0).sum()), "часть": int((d["y"] == -1).sum())}
                       for s, d in zip(train_sites, samples)}
    model, params = train_syn(samples, log)
    del samples
    A_FEATS = A.FEATS
    A.FEATS = B.FEATS                          # rules_text печатает имена из A.FEATS — на время 20 признаков ANFL0b
    try:
        rules = A.rules_text(params)
    finally:
        A.FEATS = A_FEATS
    log("=== правила Анфисы\n" + rules)
    res["правила"] = rules

    log("=== порог по подделкам Б листов обучения")
    thr, share = choose_threshold(real, fakeB, model, log)
    junk_tr = {}
    for site, r, f in zip(train_sites, real, fakeB):
        kr = _km_union(r["pool"], model(r["X"]) >= thr, r["valid"].shape)
        kf = _km_union(f["pool"], model(f["X"]) >= thr, f["valid"].shape)
        junk_tr[site] = kf / kr if kr else float("inf")
        log(f"  {site}: при пороге снимок {kr:.1f} км, подделка Б {kf:.1f} км, отношение {junk_tr[site]:.3f} "
            f"(лист обучения: оценка оптимистична; по построению, не засчитывается)")
    res["порог"] = {"порог обучения": thr, "доля длины пула": share, "подделка Б / снимок на листах обучения": junk_tr}
    JUNK_SITE.update(junk_tr)
    FAKEB_ANABAR[:] = [fakeB[train_sites.index("anabar")]]   # для доли мусора Анабара при его числовом пороге
    del fakeB

    # ---------------- Утсйоки (настройка, на нём не учили)
    iu = SITES.index("utsjoki")
    log(f"=== utsjoki (i = {iu}) — проверка, в обучение не входил")
    img, dem, raw, valid, meta_u = load_site("utsjoki")
    ru = _pool_X(img, dem, raw, valid, log, "utsjoki: снимок без врезок", site_check=None if CROP else "utsjoki")
    s_ru = model(ru["X"])
    thr_u = share_threshold(s_ru, ru, share)
    km_ru = _km_union(ru["pool"], s_ru >= thr_u, valid.shape)
    log(f"  числовой порог Утсйоки при доле {share:.3f}: {thr_u:.6g}; выбрано на снимке {km_ru:.1f} км")
    if km_ru == 0:
        log("SYN1: на настоящем Утсйоки при перенесённой доле выбрано 0 км — остановка (поправка 1)")
        raise SystemExit("SYN1: 0 км выбранного на настоящем Утсйоки — остановка")

    # (а) настоящий фон + врезки, тот же числовой порог
    log(f"=== (а) настоящий Утсйоки + врезки (план — зерно {SEED_PLAN_UTS}), тот же числовой порог {thr_u:.6g}")
    sig_img = robust_sigma(img[valid].astype("float64"))
    sig_dem = S.relief_sigma(dem, valid)
    plan = S.make_plan(valid.shape, valid, RES_IN, np.random.default_rng(SEED_PLAN_UTS), n_ins)
    img2, dem2, alos2, truth = S.apply_plan(img, dem, raw, plan, sig_img, sig_dem, RES_IN, valid=valid)
    alos2[~np.isfinite(raw)] = np.nan
    _log_plan(plan, n_ins, sig_img, sig_dem, log)
    ri = _pool_X(img2, dem2, alos2, valid, log, "utsjoki: снимок + врезки")
    s_ri = model(ri["X"])
    sel_i = s_ri >= thr_u
    all_i = np.ones(len(sel_i), bool)
    f_sel = _km_union(ri["pool"], sel_i, valid.shape) / max(_km_union(ri["pool"], all_i, valid.shape), 1e-9)
    dirs = cand_dirs(ri["pool"], img2, dem2)
    del img2, dem2, alos2
    cov_s = recall_of_insertions(*_cells_dirs(ri["pool"], dirs, sel_i), truth, plan, CONF_TOL_M, TAU)
    cov_a = recall_of_insertions(*_cells_dirs(ri["pool"], dirs, all_i), truth, plan, CONF_TOL_M, TAU)
    has = np.isfinite(cov_s)
    log(f"  f — доля длины пула, выбранная на снимке с врезками: {f_sel:.3f}; врезок со следом в годной части "
        f"{int(has.sum())} из {len(plan)}")
    amp = np.array([p["amp"] for p in plan])
    b = np.digitize(np.clip(amp, AMP_BINS[0], AMP_BINS[-1]), AMP_BINS[1:-1])     # 0…4, как _bins
    found = has & (np.nan_to_num(cov_s) >= FOUND_FRAC)
    found30 = has & (np.nan_to_num(cov_s) >= FOUND_FRAC_PRINT)
    ceil = has & (np.nan_to_num(cov_a) >= FOUND_FRAC)

    def rate(m, grp):
        n = int((has & grp).sum())
        return n, int((m & grp).sum()), (float((m & grp).sum()) / n if n else float("nan"))
    bins_tab = []
    log(f"  корзина | врезок | найдено ≥ {FOUND_FRAC:.0%} | ≥ {FOUND_FRAC_PRINT:.0%} | потолок (все кандидаты) | "
        f"0,5·потолок (печать)")
    for k in range(len(AMP_BINS) - 1):
        n, k50, r50 = rate(found, b == k)
        _n, k30, r30 = rate(found30, b == k)
        _n, kc, rc = rate(ceil, b == k)
        bins_tab.append({"корзина": f"{AMP_BINS[k]:g}–{AMP_BINS[k + 1]:g}σ", "врезок": n, "найдено": k50,
                         "доля": r50, "найдено_30": k30, "доля_30": r30, "потолок": kc, "доля_потолка": rc})
        log(f"  {AMP_BINS[k]:g}–{AMP_BINS[k + 1]:g}σ | {n} | {k50} ({r50:.3f}) | {k30} ({r30:.3f}) | {kc} ({rc:.3f}) | "
            f"{0.5 * rc:.3f}")
    kinds_tab = {}
    for kd in S.KINDS:
        g = np.array([p["kind"] == kd for p in plan], bool)
        n, k50, r50 = rate(found, g)
        _n, k30, r30 = rate(found30, g)
        _n, kc, rc = rate(ceil, g)
        kinds_tab[kd] = {"врезок": n, "найдено": k50, "доля": r50, "доля_30": r30, "доля_потолка": rc}
        log(f"  вид «{kd}»: {k50}/{n} ({r50:.3f}); ≥ {FOUND_FRAC_PRINT:.0%}: {k30}; потолок {kc} ({rc:.3f})")
    rates = np.array([t["доля"] for t in bins_tab])
    ns = np.array([t["врезок"] for t in bins_tab])
    ok = np.isfinite(rates)
    top, bot = rates[-1], rates[0]
    passed_a, txt_a, det_a, rho = verdict_a(rates, f_sel)
    log(f"  (а): доля в наибольшей корзине {top:.3f} > f {f_sel:.3f}: {bool(top > f_sel)}; > доли в наименьшей "
        f"{bot:.3f}: {bool(top > bot)}; ρ Спирмена (корзина, доля) по {int(ok.sum())} корзинам {rho:+.3f} > 0: "
        f"{bool(rho > 0)} → {txt_a}")
    if ns.min() < 10:
        log(f"  внимание: наименьшая корзина — {int(ns.min())} врезок; вердикт (а) шумный (поправка 1 п. 1)")
    res["а"] = {"вердикт": passed_a, "определён": det_a, "итог": txt_a, "f": f_sel, "rho": rho, "порог": thr_u,
                "врезок": len(plan), "врезок со следом": int(has.sum()), "сигма снимка": sig_img,
                "сигма рельефа": sig_dem, "корзины": bins_tab, "виды": kinds_tab}

    # (в) для сведения: признаки кандидатов-врезок против выбранных настоящих
    y_i = labels_from_truth(ri["pool"], truth, valid.shape, CONF_TOL_M)
    Xm, Xs = ri["X"][y_i == 1], ru["X"][s_ru >= thr_u]
    qm, qs = _q3(Xm), _q3(Xs)
    log(f"=== (в) для сведения: признаки 10/50/90 % — кандидаты у врезок ({len(Xm)}) | выбранные на снимке ({len(Xs)})")
    for k, name in enumerate(B.FEATS):
        log(f"  «{name}»: {qm[k][0]:.3f} / {qm[k][1]:.3f} / {qm[k][2]:.3f} | {qs[k][0]:.3f} / {qs[k][1]:.3f} / "
            f"{qs[k][2]:.3f}")
    res["в"] = {"кандидатов у врезок": len(Xm), "выбранных на снимке": len(Xs),
                "признаки": {name: {"у врезок": qm[k], "выбранные": qs[k]} for k, name in enumerate(B.FEATS)}}
    del ri, dirs

    # (б) подделка Б Утсйоки, тот же числовой порог
    log(f"=== (б) подделка Б Утсйоки (зерно {SEED_FAKE_B + iu}), без врезок, порог {thr_u:.6g}")
    f_img, f_dem, f_alos = joint_fake(img, dem, raw, valid, SEED_FAKE_B + iu)
    fu = _pool_X(f_img, f_dem, f_alos, valid, log, "utsjoki: подделка Б")
    del f_img, f_dem, f_alos
    km_fu = _km_union(fu["pool"], model(fu["X"]) >= thr_u, valid.shape)
    ratio = km_fu / km_ru
    verdict_b = bool(ratio <= JUNK_RATE)
    txt_b = "шум подделки отброшен" if verdict_b else "шум подделки НЕ отброшен"
    if HONEST:                                 # SYN1b: условия нет, только доля мусора листа
        txt_b = f"доля мусора Утсйоки {ratio:.0%} (около {ratio:.0%} линий может быть шумом)"
    JUNK_SITE["utsjoki"] = ratio
    log(f"  (б): км на подделке Б {km_fu:.1f} / км на снимке {km_ru:.1f} = {ratio:.3f} ≤ {JUNK_RATE}: {verdict_b} → "
        f"{txt_b} (проверяется отказ от фазового шума, не от швов, краёв пятен, воды — поправка 1)")
    res["б"] = {"вердикт": verdict_b, "итог": txt_b, "км подделки": km_fu, "км снимка": km_ru, "отношение": ratio}
    del fu, img, dem, raw

    # ---------------- выбор на настоящих листах без врезок
    log("=== выбор на настоящих Утсйоки и Анабаре (без врезок)")
    res["utsjoki"] = _save("utsjoki", ru, s_ru, thr_u, share, meta_u, log)
    ra = real[SITES.index("anabar")]
    s_ra = model(ra["X"])
    thr_a = share_threshold(s_ra, ra, share)
    log(f"  числовой порог Анабара при доле {share:.3f}: {thr_a:.6g} (Анабар в обучении: «обучено без меток, "
        f"в том числе на фоне цели»)")
    fa = FAKEB_ANABAR[0]
    km_fa = _km_union(fa["pool"], model(fa["X"]) >= thr_a, fa["valid"].shape)
    km_ra = _km_union(ra["pool"], s_ra >= thr_a, ra["valid"].shape)
    JUNK_SITE["anabar"] = km_fa / km_ra if km_ra else float("inf")
    log(f"  доля мусора Анабара (подделка Б {SEED_FAKE_B + SITES.index('anabar')}, тот же числовой порог): "
        f"{km_fa:.1f} / {km_ra:.1f} км = {JUNK_SITE['anabar']:.3f}")
    res["доля мусора по листам"] = dict(JUNK_SITE)
    res["anabar"] = _save("anabar", ra, s_ra, thr_a, share, metas["anabar"], log)
    res["код"] = code_sha()
    res["входы"] = input_sha()
    (OUT / "syn1_итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                                        encoding="utf-8")
    log(f"  {OUT / 'syn1_итог.json'}")
    log(f"=== итог: (а) {txt_a}; (б) {txt_b}")
    return res


def windows(log):
    """Окна как ANFL0b (1×3: зимний S2 / S2 + линии / рельеф + линии): кандидаты пересобираются
    (детерминированно) на настоящих входах, выбор — из npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for site, win_km in WIN_KM:
        img, dem, _raw, valid, _meta = load_site(site)
        pool = A.candidates_arr(img, dem, valid, log)
        pool = [p for p in pool if valid[p["cells"][:, 0], p["cells"][:, 1]].any()]   # как _pool_X
        sel = np.load(OUT / f"syn1_{site}.npz")["sel"]
        assert len(sel) == len(pool), "пул не совпал с прогоном"
        dst = OUT / f"syn1_{site}"
        dst.mkdir(exist_ok=True)
        gy, gx = np.gradient(dem, RES_IN)
        az = np.radians(315)
        shade = np.clip(0.6 + 0.4 * (-(gx * np.sin(az) - gy * np.cos(az)) / np.maximum(np.hypot(gx, gy), 1e-3)
                                     * np.tanh(np.hypot(gx, gy) * 10)), 0, 1)
        H, Wd = img.shape
        wp = int(win_km * 1000 / RES_IN)
        k = 0
        for i in range(3):
            for j in range(3):
                k += 1
                r0 = max(int((i + 0.5) * H / 3 - wp / 2), 0)
                c0 = max(int((j + 0.5) * Wd / 3 - wp / 2), 0)
                fig, axs = plt.subplots(1, 3, figsize=(27, 9.4))
                for ax, (t, a, show) in zip(axs, (("зимний S2, без линий", img, False), ("S2 + линии", img, True),
                                                  ("рельеф GLO-30 + линии", shade, True))):
                    bl = a[r0:r0 + wp, c0:c0 + wp]
                    lo, hi = np.nanpercentile(bl, [2, 98])
                    ax.imshow(bl, cmap="gray", vmin=lo, vmax=hi,
                              extent=(0, bl.shape[1] * RES_IN / 1000, bl.shape[0] * RES_IN / 1000, 0))
                    ax.set_title(t, fontsize=14)
                    if show:
                        for flag, col, ms in ((False, "#9bd", 0.8), (True, "red", 1.8)):
                            for idx in np.flatnonzero(sel == flag):
                                y = pool[idx]["cells"] - [r0, c0]
                                m = (y[:, 0] >= 0) & (y[:, 0] < wp) & (y[:, 1] >= 0) & (y[:, 1] < wp)
                                if m.any():
                                    y = y[m] * RES_IN / 1000
                                    ax.plot(y[:, 1], y[:, 0], ".", ms=ms, color=col)
                    ax.set_xlim(0, win_km)
                    ax.set_ylim(win_km, 0)
                junk = ""
                jf = OUT / "syn1_итог.json"
                if HONEST and jf.exists():           # SYN1b: формулировка поправки 1 п. 4
                    j = json.loads(jf.read_text(encoding="utf-8")).get("доля мусора по листам", {}).get(site)
                    if j is not None:
                        junk = (f"\nОписательная карта: отношение км подделки к км снимка при том же пороге ≈ {j:.0%} — "
                                f"доля линий, неотличимых от шума метода (не точность); линии слабее ~2σ не видны")
                fig.suptitle(f"SYN1{'b' if HONEST else ''}: Анфиса, учитель — искусственные линии; {site}, окно {k} "
                             f"({win_km}×{win_km} км), строка {r0}, столбец {c0}. Красные — выбрала Анфиса, голубые — "
                             f"отброшенные кандидаты{junk}", fontsize=13)
                fig.tight_layout()
                p = dst / f"окно_{k}.png"
                fig.savefig(p, dpi=70)
                plt.close(fig)
                log(f"  {p}")


def main():
    global CROP, OUT, HONEST
    from experiments.openlin_fam import log_to
    n_ins = N_PROPOSALS
    if "--honest" in sys.argv:                # SYN1b: свой каталог, выходы SYN1 не трогаются
        HONEST = True
        OUT = OUT / "syn1b"
        OUT.mkdir(parents=True, exist_ok=True)
    if "--crop" in sys.argv:                  # отладка: куски, свой каталог вывода — настоящие выходы не трогаются
        CROP = int(sys.argv[sys.argv.index("--crop") + 1])
        OUT = OUT / f"_syn1_crop{CROP}"
        OUT.mkdir(parents=True, exist_ok=True)
        if "--n" in sys.argv:
            n_ins = int(sys.argv[sys.argv.index("--n") + 1])
    head = "sha256 кода " + ", ".join(f"{k} {v}" for k, v in code_sha().items())
    if "--run" in sys.argv:
        log = log_to(OUT / "_syn1.log")
        log(f"SYN1; audit/SYN1_DECLARATION.md; {head}")
        run(log, n_ins=n_ins)
    elif "--windows" in sys.argv:
        log = log_to(OUT / "_syn1_windows.log")
        log(f"SYN1 окна; {head}")
        windows(log)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
