"""Пары-близнецы для сшивки: настоящие разрывы с известным ответом и их двойники.

ОТКУДА ПОСТАНОВКА. В работе ANZA-LIRA (github.com/fims9000/anza_lira,
``path_completion/pair_classifier.py``) правило «соединять или нет» учится на
парах-близнецах: к каждому настоящему разрыву подбирается «трудный
отрицательный» — пара концов с той же геометрией, но из разных линий. Тогда
геометрия ничего не подсказывает, и решать приходится по тому, что лежит
МЕЖДУ концами. У авторов разрывы искусственные, и на этом работа остановилась.
Решение пользователя 09.10.2026: сделать близнецов на наших данных.

ОТКУДА У НАС НАСТОЯЩИЕ РАЗРЫВЫ. Лист снят дважды (две половины сцен). Обрывки
берутся из одной половины («рабочей»), ответ — из другой («опорной»):
  «+»  ОДИН ОБЪЕКТ: оба обрывка лежат на ОДНОЙ цельной линии опорной половины
       и оба конца разрыва — в допуске от неё. Линия опорной половины — одна
       связная цепочка трассировки без сшивки, то есть там разрыва нет; в
       рабочей половине его сделал шум съёмки, а не мы.
  «−»  СОСЕДИ: обрывки лежат на ДВУХ РАЗНЫХ линиях опорной половины, и эти
       линии идут рядом — их проекции на общее направление перекрываются
       больше ``V6_OVERLAP_MAX_M``. По правилу самой ветки это две параллельные
       линии, а не одна разорванная. Это тот случай, на который жаловался
       просмотр: «сшивает два близлежащих линеамента».
  «?»  НЕ ЯСНО: разные линии опорной половины друг за другом. Это либо разные
       объекты, либо один, порванный в обеих половинах, — различить нечем,
       в близнецы такие пары не идут и считаются отдельно.
Пары-кандидаты — по геометрии рабочей сшивки, поэтому «+» и «−» уже прошли
одни и те же ворота; затем каждому «+» подбирается ближайший «−» по разрыву,
боковому сдвигу и расхождению направлений (без повторов).

Принадлежность обрывка линии — прежние константы ветки: не меньше ``CONF_FRAC``
длины в ``CONF_TOL_M`` от линии; опорные линии — от ``MIN_LEN_M``.

ЧТО МЕРИТСЯ. На близнецах — два правила, которые у нас уже есть: средний
отклик вдоль прямой и слабое звено самого широкого пути, оба по отклику
РАБОЧЕЙ половины (опорная в оценку не входит, она только даёт ответ). Для
каждого: разделение «+» и «−» (AUROC) и при действующем пороге — доля
соединённых настоящих разрывов и доля склеенных соседей. Считается в обе
стороны: рабочая «Б» при опорной «А» и наоборот.

ОГОВОРКИ. (1) Примеры смещены к сильным линиям: цельная линия в опорной
половине — заметный уступ; слабые линеаменты, порванные в обеих половинах,
сюда не попадают. (2) «−» — только соседи; отрицательных вида «разные объекты
на одной оси» надёжно получить нельзя (см. «?»).

Это замер, а не прогон: рабочий набор не меняется, в счётчик не идёт.
Близнецы сохраняются в ``outputs/metrics/openlin_twins_{site}.npz`` — на них
можно учить правило сшивки.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_twins.log``).
  Близнецов: 591 (рабочая «Б» при опорной «А») и 382 (наоборот). Из ~26.8 тыс.
  пар-кандидатов ответ есть у 9 %: «+» 591, «−» 824, «?» 990, остальные вне
  опорных линий.
  ОБА ДЕЙСТВУЮЩИХ ПРАВИЛА СОСЕДЕЙ ОТ ОДНОГО ОБЪЕКТА ПОЧТИ НЕ ОТЛИЧАЮТ:
    прямая (средний отклик): AUROC 0.65 / 0.57; соединяет настоящих 84 / 84 %,
                             склеивает соседей 69 / 70 %;
    путь (слабое звено):     AUROC 0.63 / 0.60; настоящих 86 / 90 %, соседей
                             75 / 76 %;
    оба сразу:               настоящих 75 / 79 %, соседей 58 / 60 %.
  То есть отклик между концами — плохой признак для этой задачи: между двумя
  близкими параллельными линиями он не падает до фона.
  ГЕОМЕТРИЯ РАЗЛИЧАЕТ ЛУЧШЕ ОТКЛИКА: до подбора медианный боковой сдвиг у «+»
  49 м, у «−» 164 м. Подбор близнецов свёл это только до 49 против 121 м —
  близких по сдвигу «−» не хватает, близнецы неполные.
  КАЧЕСТВО РАЗМЕТКИ (просмотр первого листа из четырёх, 6 пар): «+» выглядят
  верно; среди «−» есть настоящие соседи (две дуги, два края наледи) и есть
  сомнительные — ветвление одного объекта, которое опорная половина записала
  двумя перекрывающимися линиями. Долю таких по шести примерам назвать нельзя.
  Вердикт по разметке — за просмотром пользователя.

Запуск из корня репозитория:
    python -m experiments.openlin_twins --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import openlin_zone_lines_v3 as v3
from experiments.openlin_widest_probe import candidate_pairs, supports
from experiments.openlin_zone_lines import M, MIN_LEN_M, OUT, _adiff, trace

OUT_DIR = OUT.parent / "линеаменты_близнецы"
N_SHOW = 24                  # пар-близнецов на картинках (4 листа по 6)


def id_raster(lines, shape) -> np.ndarray:
    """Растр номеров линий (0 — пусто)."""
    out = np.zeros(shape, dtype="int32")
    for k, r in enumerate(lines, 1):
        out[r["pts"][:, 0], r["pts"][:, 1]] = k
    return out


def membership(cand, near_id, dist) -> np.ndarray:
    """Номер опорной линии, которой принадлежит обрывок (0 — никакой)."""
    out = np.zeros(len(cand), dtype="int32")
    for i, r in enumerate(cand):
        p = r["pts"]
        ok = dist[p[:, 0], p[:, 1]] <= v3.CONF_TOL_M
        if ok.mean() < v3.CONF_FRAC:
            continue
        ids, cnt = np.unique(near_id[p[ok, 0], p[ok, 1]], return_counts=True)
        if cnt.max() / len(p) >= v3.CONF_FRAC:
            out[i] = ids[cnt.argmax()]
    return out


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    x = np.concatenate([pos, neg])
    x = np.where(np.isfinite(x), x, -1e9)
    r = rankdata(x)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / max(len(pos) * len(neg), 1))


def match_twins(feat: np.ndarray, pos: np.ndarray, neg: np.ndarray):
    """Каждому «+» — ближайший по геометрии «−», без повторов."""
    from scipy.optimize import linear_sum_assignment

    f = feat / np.maximum(feat[np.r_[pos, neg]].std(0), 1e-9)
    cost = np.linalg.norm(f[pos][:, None, :] - f[neg][None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    return pos[ri], neg[ci]


#: Признаки пары. Геометрия — из обрывков, отклик — только рабочей половины.
FEATURES = ("разрыв_м", "сдвиг_вбок_м", "расхождение_направлений",
            "косина_мостика", "длина_короткого_м", "длина_длинного_м",
            "средний_по_прямой", "минимум_по_прямой", "слабое_звено_пути",
            "извилистость_пути", "отклик_прямо_за_концом_мин",
            "отклик_прямо_за_концом_макс")
BLOCKS = 4                   # лист делится 4 x 4 — блоки для проверки «по месту»


def pair_table(z, valid, cand, pairs, meta, feat, sm, sw, straight, paths, cls,
               res_m: float, work: str, tm: float, tw: float) -> dict:
    """Признаки пар с известным ответом («+» и «−») для подбора правила."""
    from skimage.draw import line as draw_line

    from experiments.openlin_lines import _axis

    H, W = z.shape
    zf = np.where(valid, z, np.nan)
    idx = np.flatnonzero((cls == 1) | (cls == 2))
    X = np.zeros((len(idx), len(FEATURES)), dtype="float32")

    def ahead(i, end, n_cells):
        """Средний отклик на ПРЯМОМ продолжении обрывка за его концом.

        У разорванной линии он есть (линия идёт дальше своим направлением), у
        соседей конец упирается в фон — сосед лежит сбоку, а не впереди.
        """
        pts = cand[i]["pts"]
        ax, _a, _b = _axis(pts)
        d = np.asarray(ax, float)
        d = d * (np.sign((np.asarray(end, float) - pts.mean(0)) @ d) or 1.0)
        e = np.asarray(end, float) + d * n_cells
        rr, cc = draw_line(int(end[0]), int(end[1]), int(round(e[0])), int(round(e[1])))
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[ok][2:], cc[ok][2:]              # две ячейки у конца — на гребне
        return float(np.nanmean(zf[rr, cc])) if len(rr) else np.nan

    for row, k in enumerate(idx):
        p, qq, _kul = pairs[k]
        _d, _p, _q, i, j, _sg = meta[k]
        along, across, ang = feat[k]
        li, lj = cand[i]["длина_км"] * 1000, cand[j]["длина_км"] * 1000
        n_cells = max(int(round(np.hypot(qq[0] - p[0], qq[1] - p[1]))), 3)
        ai, aj = ahead(i, p, n_cells), ahead(j, qq, n_cells)
        st = straight[k]
        X[row] = (along, across, ang, np.degrees(np.arctan2(across, max(along, 1e-9))),
                  min(li, lj), max(li, lj), sm[k],
                  float(np.nanmin(zf[st[:, 0], st[:, 1]])) if len(st) else np.nan,
                  sw[k] if np.isfinite(sw[k]) else np.nan,
                  len(paths[k]) / max(len(st), 1),
                  np.nanmin([ai, aj]), np.nanmax([ai, aj]))
    mid = np.array([((pairs[k][0][0] + pairs[k][1][0]) / 2,
                     (pairs[k][0][1] + pairs[k][1][1]) / 2) for k in idx])
    block = ((mid[:, 0] * BLOCKS // H) * BLOCKS + mid[:, 1] * BLOCKS // W).astype("int16")
    return dict(X=X, y=(cls[idx] == 1).astype("int8"), block=block,
                half=np.full(len(idx), work), ok_m=sm[idx] > tm, ok_w=sw[idx] > tw)


def oriented(layer, p, q, res_m: float, order: int) -> np.ndarray:
    """Вырезка вдоль пары концов: концы на одной горизонтали (как у LIRA)."""
    from scipy.ndimage import map_coordinates

    a, b = np.asarray(p, float), np.asarray(q, float)
    u = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
    n = np.array([-u[1], u[0]])
    mid = (a + b) / 2
    hl = (v3.V5_GAP_M / 2 + MIN_LEN_M) / res_m       # разрыв + по линии с каждой стороны
    ht = v3.V4_OFFSET_M / res_m                       # допустимый боковой сдвиг
    gl, gt = np.meshgrid(np.arange(-hl, hl + 1), np.arange(-ht, ht + 1))
    rr = mid[0] + u[0] * gl + n[0] * gt
    cc = mid[1] + u[1] * gl + n[1] * gt
    return map_coordinates(layer, (rr, cc), order=order, mode="constant", cval=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shade import _stretch
    from experiments.openlin_shadow import standardize
    from experiments.openlin_zone_lines import COMP

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    halves = v3.build_halves(a.site, log)
    valid = (np.isfinite(w["shade"]) & np.isfinite(halves["А"])
             & np.isfinite(halves["Б"]))
    shape = valid.shape

    def prepare(o):
        xx = np.where(valid, standardize(o, valid), 0.0).astype("float32")
        z = local_norm(np.where(valid, ridge_response(xx), 0.0), valid, res_m)
        raw = trace(o, valid, res_m, local=True, min_len_m=0)
        return z, v3.extend_ends(raw, z, valid, res_m)

    P = {n: prepare(halves[n]) for n in ("А", "Б")}
    log(f"половины готовы ({time.time() - t0:.0f} с)")
    q = v3.V7_BRIDGE_Q
    saved, table = {}, []
    for work, ref in (("Б", "А"), ("А", "Б")):
        z, raw = P[work]
        ref_lines = [r for r in P[ref][1] if r["длина_км"] * 1000 >= MIN_LEN_M]
        rid = id_raster(ref_lines, shape)
        dist, (ir, ic) = distance_transform_edt(rid == 0, sampling=(res_m, res_m),
                                                return_indices=True)
        near_id = rid[ir, ic]
        cand, pairs, meta = candidate_pairs(raw, res_m)
        mem = membership(cand, near_id, dist)
        n = len(pairs)
        cls = np.zeros(n, dtype="int8")            # 1 «+», 2 «−» соседи, 3 «?», 0 вне опоры
        feat = np.zeros((n, 3))                    # разрыв, |сдвиг вбок|, расхождение направлений
        for k, ((p, qq, _kul), (_d, _p, _q, i, j, _sg)) in enumerate(zip(pairs, meta)):
            t = np.radians(2 * np.array([cand[i]["азимут"], cand[j]["азимут"]]))
            st = (np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())) / 2) % 180.0
            al = np.radians(st)
            u = np.array([-np.cos(al), np.sin(al)])
            nn_ = np.array([np.sin(al), np.cos(al)])
            v = np.asarray(qq, float) - np.asarray(p, float)
            feat[k] = (abs(v @ u) * res_m, abs(v @ nn_) * res_m,
                       _adiff(cand[i]["азимут"], cand[j]["азимут"]))
            mi, mj = mem[i], mem[j]
            if not mi or not mj:
                continue
            if dist[p] > v3.CONF_TOL_M or dist[qq] > v3.CONF_TOL_M \
                    or near_id[p] != mi or near_id[qq] != mj:
                continue                           # конец разрыва не на своей опорной линии
            if mi == mj:
                cls[k] = 1
                continue
            pi_, pj_ = ref_lines[mi - 1]["pts"], ref_lines[mj - 1]["pts"]
            ti, tj = pi_ @ u, pj_ @ u
            lo_, hi_ = max(ti.min(), tj.min()), min(ti.max(), tj.max())
            if (hi_ - lo_) * res_m <= v3.V6_OVERLAP_MAX_M:
                cls[k] = 3
                continue
            # «−» только если на перекрытии опорные линии идут ПОРОЗНЬ, не ближе
            # V4_SIDE_MIN_M: иначе это ветвление одного объекта, записанное
            # опорной половиной двумя линиями, а не соседи
            ni = np.median((pi_ @ nn_)[(ti >= lo_) & (ti <= hi_)])
            nj = np.median((pj_ @ nn_)[(tj >= lo_) & (tj <= hi_)])
            cls[k] = 2 if abs(ni - nj) * res_m >= v3.V4_SIDE_MIN_M else 4
        rng = np.random.default_rng(31)
        sm, sw, nm, nw, straight, paths = supports(z, valid, pairs, res_m, rng)
        tm, tw = float(np.nanquantile(nm, q)), float(np.quantile(nw, q))
        pos, neg = np.flatnonzero(cls == 1), np.flatnonzero(cls == 2)
        log(f"\n=== рабочая «{work}», опорная «{ref}»: пар {n}; «+» один объект "
            f"{len(pos)}, «−» соседи {len(neg)}, ветвление (не в счёт) "
            f"{int((cls == 4).sum())}, «?» не ясно {int((cls == 3).sum())}, "
            f"вне опоры {int((cls == 0).sum())} ({time.time() - t0:.0f} с)")
        table.append(pair_table(z, valid, cand, pairs, meta, feat, sm, sw, straight,
                                paths, cls, res_m, work, tm, tw))
        if len(pos) < 10 or len(neg) < 10:
            log("  близнецов слишком мало — замер невозможен")
            continue
        tp, tn = match_twins(feat, pos, neg)
        med = lambda idx: np.round(np.median(feat[idx], 0), 0)
        log(f"  близнецов {len(tp)}; геометрия «+» (разрыв м / сдвиг м / угол °): "
            f"{med(tp)}, «−»: {med(tn)}; до подбора «+» {med(pos)}, «−» {med(neg)}")
        for name, s, thr in (("прямая (средний отклик)", sm, tm),
                             ("путь (слабое звено)", sw, tw)):
            log(f"  {name:26s}: AUROC {auroc(s[tp], s[tn]):.3f} | при пороге "
                f"{thr:.2f} соединено настоящих {100 * (s[tp] > thr).mean():.1f} %, "
                f"склеено соседей {100 * (s[tn] > thr).mean():.1f} %")
        both_p = (sm[tp] > tm) & (sw[tp] > tw); both_n = (sm[tn] > tm) & (sw[tn] > tw)
        any_p = (sm[tp] > tm) | (sw[tp] > tw); any_n = (sm[tn] > tm) | (sw[tn] > tw)
        log(f"  оба правила сразу        : соединено настоящих {100 * both_p.mean():.1f} %, "
            f"склеено соседей {100 * both_n.mean():.1f} %")
        log(f"  хотя бы одно             : соединено настоящих {100 * any_p.mean():.1f} %, "
            f"склеено соседей {100 * any_n.mean():.1f} %")
        unk = np.flatnonzero(cls == 3)
        if len(unk):
            log(f"  для сведения, «?»: прямая принимает {100 * (sm[unk] > tm).mean():.1f} %, "
                f"путь {100 * (sw[unk] > tw).mean():.1f} %")
        saved[work] = dict(tp=tp, tn=tn, feat=feat, sm=sm, sw=sw, tm=tm, tw=tw,
                           pairs=pairs, meta=meta, cand=cand, rid=rid)

    if table:
        # таблица признаков всех пар с ответом — для подбора правила
        # (:mod:`openlin_twins_fit`); признаки только из рабочей половины
        np.savez_compressed(M / f"openlin_twins_{a.site}_таблица.npz",
                            X=np.concatenate([t["X"] for t in table]),
                            y=np.concatenate([t["y"] for t in table]),
                            блок=np.concatenate([t["block"] for t in table]),
                            половина=np.concatenate([t["half"] for t in table]),
                            прошла_прямая=np.concatenate([t["ok_m"] for t in table]),
                            прошёл_путь=np.concatenate([t["ok_w"] for t in table]),
                            признаки=np.array(FEATURES))
    if not saved:
        return
    np.savez_compressed(
        M / f"openlin_twins_{a.site}.npz",
        **{f"{k}_{work}": np.asarray(v) for work, s in saved.items()
           for k, v in (("геометрия_плюс", s["feat"][s["tp"]]),
                        ("геометрия_минус", s["feat"][s["tn"]]),
                        ("прямая_плюс", s["sm"][s["tp"]]), ("прямая_минус", s["sm"][s["tn"]]),
                        ("путь_плюс", s["sw"][s["tp"]]), ("путь_минус", s["sw"][s["tn"]]),
                        ("концы_плюс", [(*s["pairs"][k][0], *s["pairs"][k][1]) for k in s["tp"]]),
                        ("концы_минус", [(*s["pairs"][k][0], *s["pairs"][k][1]) for k in s["tn"]]))})

    # ---- картинки: близнецы рядом, вырезки повёрнуты вдоль пары концов
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    work = "Б" if "Б" in saved else next(iter(saved))
    s = saved[work]
    base = np.nan_to_num(_stretch(np.where(valid, halves[work], np.nan)), nan=0.5)
    fid = id_raster(s["cand"], shape)
    rng = np.random.default_rng(5)
    pick = rng.choice(len(s["tp"]), min(N_SHOW, len(s["tp"])), replace=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()

    def draw(ax, k, title):
        p, qq = s["pairs"][k][0], s["pairs"][k][1]
        i, j = s["meta"][k][3], s["meta"][k][4]
        im = oriented(base, p, qq, res_m, 1)
        fr = oriented(fid, p, qq, res_m, 0)
        rf = oriented(s["rid"], p, qq, res_m, 0)
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ov = np.zeros(im.shape + (4,), "float32")
        ov[rf > 0] = (0.10, 0.70, 0.20, 0.55)                 # опорная половина
        ov[(fr > 0)] = (0.55, 0.80, 0.95, 0.9)                # прочие обрывки
        ov[(fr == i + 1) | (fr == j + 1)] = (0.10, 0.35, 0.95, 1.0)
        ax.imshow(ov)
        c = (im.shape[1] - 1) / 2
        hg = np.hypot(qq[0] - p[0], qq[1] - p[1]) / 2
        ax.plot([c - hg, c + hg], [(im.shape[0] - 1) / 2] * 2, "o", color="#e8a400", ms=5)
        ax.set_title(f"{title}: разрыв {s['feat'][k][0]:.0f} м, сдвиг {s['feat'][k][1]:.0f} м · "
                     f"прямая {'да' if s['sm'][k] > s['tm'] else 'нет'}, "
                     f"путь {'да' if s['sw'][k] > s['tw'] else 'нет'}",
                     loc="left", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    for sheet in range(0, len(pick), 6):
        rows = pick[sheet:sheet + 6]
        fig, ax = plt.subplots(len(rows), 2, figsize=(19, 3.6 * len(rows)),
                               facecolor="#fcfcfb", squeeze=False)
        for r_, t in enumerate(rows):
            draw(ax[r_, 0], s["tp"][t], "«+» один объект")
            draw(ax[r_, 1], s["tn"][t], "«−» соседи")
        fig.suptitle(f"Близнецы: рабочая половина «{work}» (снимок и синие обрывки; "
                     f"тёмно-синие — пара, жёлтые точки — её концы), зелёным — линии "
                     f"опорной половины. Вырезка {2 * (v3.V5_GAP_M / 2 + MIN_LEN_M) / 1000:.1f}"
                     f" × {2 * v3.V4_OFFSET_M / 1000:.1f} км, повёрнута вдоль пары.",
                     x=0.01, ha="left", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT_DIR / f"близнецы_{sheet // 6 + 1}.png", dpi=85,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    log(f"\nкартинки: {OUT_DIR}; близнецы сохранены в openlin_twins_{a.site}.npz "
        f"({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
