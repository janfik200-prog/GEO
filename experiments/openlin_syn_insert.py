"""SYN1: вставка синтетических линий в реальные поля листа.

Здесь же общая фазовая подделка: одни и те же случайные фазы для всех полей,
поэтому взаимные спектры (связи между каналами) сохраняются, а протяжённые
линии разрушены.
"""
from __future__ import annotations

import numpy as np


def joint_phase_randomize(fields: list[np.ndarray], valid: np.ndarray,
                          rng) -> list[np.ndarray]:
    """Фазовая подделка списка полей с ОДНИМ общим массивом случайных фаз.

    Как openlin_null_input.phase_randomize для одного поля: невалидные ячейки
    заполняются медианой, амплитудный спектр каждого поля сохраняется, фазы
    случайны с ph[0,0]=0 (эрмитова симметрия обеспечивается rfft2/irfft2),
    среднее и разброс по valid возвращаются как у исходного поля. Общие фазы
    сохраняют разность фаз между полями, то есть взаимную корреляцию.
    """
    valid = np.asarray(valid, bool)
    for f in fields:
        if np.shape(f) != valid.shape:
            raise ValueError(
                f"форма поля {np.shape(f)} не совпадает с valid {valid.shape}")
    if not fields:
        return []
    ph = None
    out = []
    for img in fields:
        med = float(np.nanmedian(img[valid]))
        x = np.nan_to_num(np.where(valid, img, med).astype("float64"), nan=med)
        f = np.fft.rfft2(x)
        if ph is None:                       # один массив фаз на все поля
            ph = rng.uniform(-np.pi, np.pi, f.shape)
            ph[0, 0] = 0.0
        y = np.fft.irfft2(np.abs(f) * np.exp(1j * ph), s=x.shape)
        y = (y - y.mean()) / (y.std() or 1.0)
        out.append((y * x[valid].std() + x[valid].mean()).astype("float32"))
    return out


# =========================================================== врезки (задача 2)
# Все числа — из audit/SYN1_DECLARATION.md, разделы «Врезки» и «Поправка 1».
from experiments.openlin_fam import DEM_SIG_M, IMG_SIG_M, RES_IN  # noqa: E402
from experiments.openlin_iso import LOCAL_WIN_M  # noqa: E402
from experiments.openlin_winter import DETREND_M  # noqa: E402
from experiments.openlin_zone_lines_v3 import LINK_MIN_LEN_M, V3_GAP_M  # noqa: E402

IMG_KINDS = ("тёмная", "светлая", "перепад")          # снимок (каналы CAND 1–3)
DEM_KINDS = ("ложбина", "гребень", "уступ")           # рельеф (каналы CAND 4–5 + уступ)
KINDS = IMG_KINDS + DEM_KINDS + ("смешанная", "составная")   # 8 видов, равновероятно (O'Leary)
# Профиль и знак: тёмная полоса/ложбина ниже окружения (канал 1/4 CAND — λmax > 0),
# светлая/гребень выше; перепад/уступ — ступень (сторону задаёт азимут 0–360°).
PROFILE = {"тёмная": ("ridge", -1.0), "светлая": ("ridge", +1.0),
           "перепад": ("scarp", +1.0), "ложбина": ("ridge", -1.0),
           "гребень": ("ridge", +1.0), "уступ": ("scarp", +1.0)}
AMP_SIG = (0.25, 8.0)             # контраст, log-равномерно, в σ канала (объявление; 5 октав)
LEN_M = (LINK_MIN_LEN_M, 6636.0)  # длина, log-равномерно: минимум тракта — 99,9 % пула CAND3 (замер)
STRAIGHT_MIN = 0.698              # хорда/дуга равномерно в [0,698; 1] — 10 % прямизны пула (замер)
CV_ALONG = 0.122                  # КВ контраста вдоль линии — медиана замера
CORR_ALONG_M = DETREND_M / 2.0    # длина корреляции множителя — окно детренда/2 (объявление)
N_PIECES = (2, 4)                 # составная: 2–4 куска, равновероятно
GAP_MAX_M = V3_GAP_M              # разрывы составной — равномерно до 900 м (правило сшивки)
SEP_M = LOCAL_WIN_M / 2.0         # 1000 м между следами: окна нормировки не пересекаются


def relief_sigma(dem: np.ndarray, valid, res_m: float = RES_IN) -> float:
    """σ рельефа: robust_sigma GLO-30 после того же детренда DETREND_M, что у композита."""
    from experiments.openlin_winter import detrend
    from src.detection_limit import robust_sigma
    valid = np.asarray(valid, bool)
    d = detrend(np.asarray(dem, "float64"), valid, res_m)
    return robust_sigma(d[valid])


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _theta_from_ratio(q: float) -> float:
    """Центральный угол дуги θ по отношению хорда/дуга q = sin(θ/2)/(θ/2) (бисекция)."""
    if q >= 1.0 - 1e-12:
        return 0.0
    lo, hi = 0.0, 2.0 * np.pi
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if np.sin(m / 2) / (m / 2) > q:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def _width(kind, rng):
    """Ширина — масштаб своего канала: снимок IMG_SIG_M, рельеф DEM_SIG_M (равновероятно)."""
    return float(rng.choice(IMG_SIG_M if kind in IMG_KINDS else DEM_SIG_M))


def insertion(kind: str, center, az_deg: float, length_m: float, amp: float,
              rng, *, bend: float = 1.0, curv_sign: float = 1.0,
              width_m: float | None = None, cv: float = CV_ALONG) -> dict:
    """Одна врезка (словарь плана).

    ``length_m`` — длина линии (сумма кусков, без разрывов); ``amp`` — один контраст
    в σ на всю врезку (поправка 1 п. 4); ``bend`` — хорда/дуга всего пролёта;
    ``az_deg`` — географический азимут касательной в центре. Ширина, куски и
    множитель — из ``rng``. ``parts`` — составляющие: вид, отрезок дуги [s0, s1]
    (м от центра), ширина. ``pieces`` — отрезки следа (у составной — куски,
    между ними разрывы).
    """
    if kind not in KINDS:
        raise ValueError(f"неизвестный вид {kind!r}")
    L = float(length_m)
    if kind == "составная":
        k = int(rng.integers(N_PIECES[0], N_PIECES[1] + 1))
        kinds = [str(v) for v in rng.choice(IMG_KINDS + DEM_KINDS, size=k, replace=False)]
        lens = rng.dirichlet(np.ones(k)) * L
        gaps = rng.uniform(0.0, GAP_MAX_M, k - 1)
        span = L + float(gaps.sum())
        s, pieces = -span / 2, []
        for i in range(k):
            pieces.append((s, s + float(lens[i])))
            s += float(lens[i]) + (float(gaps[i]) if i < k - 1 else 0.0)
        parts = [dict(kind=kd, s0=a, s1=b, width_m=_width(kd, rng))
                 for kd, (a, b) in zip(kinds, pieces)]
    else:
        span = L
        pieces = [(-L / 2, L / 2)]
        if kind == "смешанная":
            kinds = [str(rng.choice(DEM_KINDS)), str(rng.choice(("тёмная", "светлая")))]
        else:
            kinds = [kind]
        parts = [dict(kind=kd, s0=-L / 2, s1=L / 2,
                      width_m=float(width_m) if width_m is not None else _width(kd, rng))
                 for kd in kinds]
    theta = _theta_from_ratio(float(bend))
    curv = float(np.sign(curv_sign) or 1.0) * theta / span
    return dict(kind=kind, center=(float(center[0]), float(center[1])),
                az_deg=float(az_deg), length_m=L, span_m=span, amp=float(amp),
                width_m=parts[0]["width_m"], bend=float(bend), curv=curv,
                cv=float(cv), mod_seed=int(rng.integers(2 ** 31)),
                pieces=pieces, parts=parts)


def _frame(az_deg):
    """Направляющий вектор и нормаль в (строка, столбец): u = (−cos a, sin a), n ⟂ u
    (те же, что у insert_fault после axis_from_geographic, с точностью до знака оси)."""
    a = np.radians(az_deg)
    return np.array([-np.cos(a), np.sin(a)]), np.array([np.sin(a), np.cos(a)])


def curve_points(ins: dict, s_m, res_m: float) -> np.ndarray:
    """Точки дуги (строка, столбец, дробные) для длин дуги ``s_m`` (м) от центра."""
    u, n = _frame(ins["az_deg"])
    k = ins["curv"]
    s = np.asarray(s_m, float)
    if abs(k) < 1e-12:
        a, b = s, np.zeros_like(s)
    else:
        a, b = np.sin(k * s) / k, (1 - np.cos(k * s)) / k
    c = np.asarray(ins["center"], float)
    return c + (a[:, None] * u + b[:, None] * n) / res_m


def trace_points(ins: dict, res_m: float) -> np.ndarray:
    """Весь пролёт дуги (с разрывами составной) с шагом res_m/2 — для зазора между врезками."""
    h = ins["span_m"] / 2
    return curve_points(ins, np.arange(-h, h + res_m / 4, res_m / 2), res_m)


def _coords(ins, r0, c0, shape, res_m):
    """Длина дуги s и знаковое расстояние t до дуги (м) для ячеек окна."""
    u, n = _frame(ins["az_deg"])
    rr, cc = np.mgrid[r0:r0 + shape[0], c0:c0 + shape[1]]
    dr = (rr - ins["center"][0]) * res_m
    dc = (cc - ins["center"][1]) * res_m
    k = ins["curv"]
    if abs(k) < 1e-12:
        return dr * u[0] + dc * u[1], dr * n[0] + dc * n[1]
    sg, R = np.sign(k), 1.0 / abs(k)
    # центр окружности O = c + n/κ; точка x = p(s) + t·n(s), n(s) = n·cos κs − u·sin κs
    wr, wc = dr - n[0] / k, dc - n[1] / k
    rho = np.maximum(np.hypot(wr, wc), 1e-9)
    nr, nc = -sg * wr / rho, -sg * wc / rho
    ang = np.arctan2(-(nr * u[0] + nc * u[1]), nr * n[0] + nc * n[1])
    return ang / k, sg * (R - rho)


def _along(ins, res_m):
    """Гладкий множитель контраста вдоль дуги: среднее 1 и КВ ``cv`` на самой линии
    (как замер — КВ вдоль каждой линии), корреляция — гаусс с σ = CORR_ALONG_M.
    Возвращает (s, m) для интерполяции."""
    from scipy.ndimage import gaussian_filter1d
    h = ins["span_m"] / 2
    pad = 3 * CORR_ALONG_M
    s = np.arange(-h - pad, h + pad + res_m, res_m)
    if ins["cv"] <= 0:
        return s, np.ones_like(s)
    z = gaussian_filter1d(np.random.default_rng(ins["mod_seed"]).normal(size=s.size),
                          CORR_ALONG_M / res_m, mode="reflect")
    on = np.abs(s) <= h
    z = (z - z[on].mean()) / (z[on].std() or 1.0)
    return s, np.maximum(1.0 + ins["cv"] * z, 0.0)


def _part_field(ins, part, r0, c0, shape, res_m):
    """Поле одной составляющей в окне (в σ) — как insert_fault (профиль поперёк,
    гауссов хвост за концами с масштабом в ширину), но вдоль дуги и с множителем."""
    s, t = _coords(ins, r0, c0, shape, res_m)
    prof, sign = PROFILE[part["kind"]]
    w = part["width_m"]
    half_w = max(w / 2.0, res_m / 2.0)
    if prof == "scarp":
        # Ступень ±0,5 не гаснет поперёк: с хвостом только вдоль s она дала бы полосу
        # длиной L через всё окно, торцы которой — паразитные поперечные линии до края
        # окна и шов на его границе (ревью задачи 2). Поэтому ступень локализуется
        # поперёк тем же детрендом, что у композита, но одномерным: минус её гауссово
        # сглаживание по t с σ = DETREND_M/2 (свёртка ступени = 0,5·erf(t/(√2·σ))).
        # Вдали (|t| ≳ 3σ) профиль → 0, у линии ступень почти та же (erf там ≈ 0).
        from scipy.special import erf
        sd_m = DETREND_M / 2.0
        cross = 0.5 * np.tanh(t / half_w) - 0.5 * erf(t / (np.sqrt(2.0) * sd_m))
    else:
        cross = np.exp(-0.5 * (t / half_w) ** 2)
    over = np.maximum(np.maximum(part["s0"] - s, s - part["s1"]), 0.0)
    taper = np.exp(-0.5 * (over / max(w, res_m)) ** 2)
    sa, ma = _along(ins, res_m)
    mult = np.interp(np.clip(s, part["s0"], part["s1"]), sa, ma)
    return sign * ins["amp"] * cross * taper * mult


def _add(x, ins, parts, sigma, res_m):
    """Добавить составляющие в поле на месте, в окне, с детрендом как insert_window."""
    from scipy.ndimage import gaussian_filter
    if not parts:
        return
    sd = DETREND_M / res_m / 2.0
    p = trace_points(ins, res_m)
    wmax = max(q["width_m"] for q in parts)
    # у ступени поле поперёк живёт до ~3σ детренда (1D-локализация), и 2D-детренд
    # размазывает его ещё на 3σ: окно 6σ, иначе на его границе шов (замер в тесте)
    n_sd = 6 if any(PROFILE[q["kind"]][0] == "scarp" for q in parts) else 3
    pad = int(n_sd * sd + 4 * wmax / res_m) + 2
    r0 = max(int(np.floor(p[:, 0].min())) - pad, 0)
    c0 = max(int(np.floor(p[:, 1].min())) - pad, 0)
    r1 = min(int(np.ceil(p[:, 0].max())) + pad + 1, x.shape[0])
    c1 = min(int(np.ceil(p[:, 1].max())) + pad + 1, x.shape[1])
    if r1 <= r0 or c1 <= c0:
        return
    f = np.zeros((r1 - r0, c1 - c0))
    for q in parts:
        f += _part_field(ins, q, r0, c0, f.shape, res_m)
    f *= sigma
    x[r0:r1, c0:c1] += f - gaussian_filter(f, sd, mode="nearest")


def truth_cells(ins: dict, shape, valid, res_m: float) -> np.ndarray:
    """Ячейки (строка, столбец) следа: только куски (без разрывов), только в valid."""
    out = []
    for a, b in ins["pieces"]:
        s = np.arange(a, b + res_m / 8, res_m / 4)
        out.append(np.rint(curve_points(ins, s, res_m)).astype(int))
    p = np.unique(np.concatenate(out), axis=0)
    ok = (p[:, 0] >= 0) & (p[:, 0] < shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < shape[1])
    p = p[ok]
    return p[np.asarray(valid, bool)[p[:, 0], p[:, 1]]]


def make_plan(shape, valid, res_m: float, rng, n: int) -> list[dict]:
    """План врезок: ``n`` предложений в случайном порядке (бюджет поиска; в прогоне —
    2000 на лист, объявление); предложение принимается, если центр в valid и весь
    его след (с разрывами) не ближе SEP_M к следам принятых. Возвращает принятые."""
    from scipy.spatial import cKDTree
    valid = np.asarray(valid, bool)
    plan, acc, tree = [], np.zeros((0, 2)), None
    for _ in range(int(n)):
        kind = str(rng.choice(KINDS))
        center = (rng.uniform(0, shape[0]), rng.uniform(0, shape[1]))
        ins = insertion(kind, center, rng.uniform(0.0, 360.0),
                        _logu(rng, *LEN_M), _logu(rng, *AMP_SIG), rng,
                        bend=rng.uniform(STRAIGHT_MIN, 1.0),
                        curv_sign=float(rng.choice((-1.0, 1.0))))
        if not valid[int(center[0]), int(center[1])]:
            continue
        p = trace_points(ins, res_m)
        if tree is not None and tree.query(p)[0].min() * res_m < SEP_M:
            continue
        plan.append(ins)
        acc = np.vstack([acc, p])
        tree = cKDTree(acc)
    return plan


def apply_plan(img, dem, alos, plan, sig_img: float, sig_dem: float,
               res_m: float, valid=None):
    """Врезать план: виды снимка — в img (σ = sig_img), виды рельефа — одним и тем же
    приращением в dem и alos (σ = sig_dem). ``valid`` по умолчанию — конечные
    ячейки всех трёх полей. Возвращает (img2, dem2, alos2, truth)."""
    img2 = np.array(img, dtype="float64")
    dem2 = np.array(dem, dtype="float64")
    alos2 = np.array(alos, dtype="float64")
    if valid is None:
        valid = np.isfinite(img2) & np.isfinite(dem2) & np.isfinite(alos2)
    d_dem = np.zeros(dem2.shape)
    truth = []
    for ins in plan:
        _add(img2, ins, [q for q in ins["parts"] if q["kind"] in IMG_KINDS], sig_img, res_m)
        _add(d_dem, ins, [q for q in ins["parts"] if q["kind"] in DEM_KINDS], sig_dem, res_m)
        truth.append(truth_cells(ins, img2.shape, valid, res_m))
    return img2, dem2 + d_dem, alos2 + d_dem, truth
