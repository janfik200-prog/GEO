"""Подтверждение по отклику: голосование двенадцати сцен вместо двух половин.

ОТКУДА ПОСТАНОВКА. Глазная проверка 28.09.2026 по картам стадий: детектор
«видит всё» — карта отклика покрывает все линеаменты, которые геолог видит на
снимке. Полнота теряется ПОСЛЕ детектора, и одно из двух мест — подтверждение.
Прежнее правило сравнивало готовую линию с готовыми линиями двух половин, то
есть дважды проходило через хрупкий шаг «отклик -> линия», и упиралось в
неравные половины (8 сцен против 4; на 77 % листа половина «Б» — одна сцена).
Оценка по двойнику: порядка 300 настоящих линий (около 15 %) терялось именно
на подтверждении — в том числе линеаменты из «бусин», которые в половинах не
собираются в линию вовсе.

ЧТО ВМЕСТО. Для каждой из 12 сцен строится своя карта отклика (тот же
детектор, что рисует линии). Линия голосует сценами: сцена «видит» линию, если
средний отклик вдоль неё выше, чем у 95 % копий той же линии, перенесённых в
случайные места этой сцены. Линия подтверждена, если её видят не меньше двух
сцен и больше, чем видят её же перенесённые копии в 95 % случаев. Сцены
работают одинаково, геометрия линии в сценах заново не собирается.

ЭТО ЗАМЕНА ТОЛЬКО ПОДТВЕРЖДЕНИЯ. Тракт до него (поиск теней, продолжение,
сшивка, отсев швов) — тот же, функции те же. Двойник строится на стопке из
композита и всех 12 сцен с общими случайными фазами, как прежде на стопке
«композит + две половины», и проходит весь тракт вместе с голосованием.

РЕЗУЛЬТАТ ПРОБЫ (28.09.2026, вырезка 24 x 24 км, один двойник; в счётчик не
идёт): В ТАКОМ ВИДЕ НЕ РАБОТАЕТ. Двойник подтверждается почти как снимок:
весь лист 70 % против 90 %, мусор этапа подтверждения 78 %; в зонах 50 %
против 75 %, мусор 67 % (порог 20 %). Причина — двойной учёт: линии
ищутся на композите, а композит — медиана тех же сцен, что голосуют (на 77 %
листа — трёх сцен, каждая — треть медианы). Линия проведена там, где медиана
сильна, то есть там, где сильны сами сцены, и «сцена видит линию» заложено
в отбор. Сдвиговый нуль этого не ловит — он сравнивает со случайными
местами, а не с отобранными так же; поймал двойник, прошедший весь тракт.
Нуль «места такой же силы в композите» отвергнут без прогона: такими местами
окажутся другие настоящие линеаменты, и сигнал исчезнет вместе с шумом.
Следующий кандидат — перекрёстная схема: линии ищутся на композите одной
половины, голосуют сцены другой, и наоборот.

РЕЗУЛЬТАТ ПРОБЫ ПЕРЕКРЁСТНОЙ СХЕМЫ (05.10.2026, та же вырезка, один двойник;
в счётчик не идёт): ТОЖЕ НЕ РАБОТАЕТ. Линии ищутся на композите одной
половины, судят сцены другой (общих сцен нет). Подтверждение у снимка 87 и
84 %, у двойника 66 и 71 %; мусор этапа 76-85 % при пороге 20 %. Двойного
учёта здесь уже нет, значит причина глубже: двойник сохраняет связь между
каналами (текстура поверхности у половин общая, корреляция 0.72 против 0.84
у снимка), и повышенный отклик в том же месте другой съёмки даёт любая
устойчивая текстура, а не только линеамент. Подтверждение ПО ОТКЛИКУ
проверяет «поверхность та же», а не «здесь линия». Прежнее подтверждение по
готовым линиям половин работает (мусор 6 %) именно потому, что требует от
каждой половины заново собрать линию длиной от 500 м, чего шум почти не
умеет. Семейство «подтверждать по отклику» закрыто: и прямое, и перекрёстное.

Запуск из корня репозитория:
    python -m experiments.openlin_vote --site anabar
    python -m experiments.openlin_vote --проба      # вырезка, один двойник
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.openlin_zone_lines import (COMP, DIR_TOL_DEG, M, MIN_LEN_M,
                                            N_FAKES, STAB_MIN, STAB_OVER_NULL,
                                            STAB_TOL_M, ZONE_FRAC, _adiff,
                                            _line_strike, cover, mask_of,
                                            trace, zone_and_strike)
from experiments.openlin_zone_lines_v3 import (CACHE, NDVI_N_TESTS,
                                               NDVI_N_TESTS_PRIOR,
                                               STAGE_JUNK_MAX,
                                               V3_MEDIAN_KM_MIN,
                                               close_along_strike,
                                               drop_seam_lines, extend_ends,
                                               link_same_side)

# ---- пре-регистрация (28.09.2026, объявлено до прогона) ---------------------
#: Копий линии для её собственного нуля. Уровень на линию 0.05 требует пол
#: 1/(n+1) < 0.05, то есть n >= 20; для 95-го процентиля нужно не меньше 20
#: значений, чтобы хоть одно стояло выше него. Берётся 40: пол 0.024, и
#: даже если часть копий упадёт вне сцены, двадцать обычно останутся.
LINE_SHIFTS = 40
LINE_MIN_NULL = 20           # меньше копий в сцене — сцена по линии не голосует
#: Порог «сцена видит линию» — 95-й процентиль копий, как у опоры мостика
#: (доля ложных срабатываний сцены 5 % по построению).
LINE_Q = 0.95
#: Порог голосов — 95-й процентиль голосов перенесённых копий той же линии.
VOTE_Q = 0.95
#: Сцена голосует по линии, только если покрывает хотя бы её половину (сцены
#: лежат плитками и накрывают лист не целиком).
SCENE_COVER = 0.5
#: Повтор — это совпадение хотя бы двух съёмок: меньше двух голосов не
#: подтверждают ничего, какой бы ни была статистика.
MIN_VOTES = 2
#: Условия приёмки — прежние четыре (мусор этапа подтверждения <= 20 %,
#: сдвиговый нуль, устойчивость, медиана длины), для двух наборов.
#: Счётчик: после NDVI 57, этот прогон добавляет два набора по три условия.
VOTE_N_TESTS_PRIOR = NDVI_N_TESTS_PRIOR + NDVI_N_TESTS      # 57
VOTE_N_TESTS = 6
VOTE_ALPHA = 0.05 / (VOTE_N_TESTS_PRIOR + VOTE_N_TESTS)     # 0.000794
#: Сдвиговый нуль всего набора: n >= 2/alpha - 1 = 2519; берётся 3000,
#: пол 0.00033, запас alpha / пол = 2.4x.
SET_SHIFTS = 3000
SEED = 41
# ---------------------------------------------------------------------------


def scene_files():
    """Сцены в том же порядке, что у половин (по дате)."""
    return sorted(CACHE.glob("*_лист_20m.npz"), key=lambda f: f.name.split("_")[2][:8])


def scene_shade(b: np.ndarray, res_m: float) -> np.ndarray:
    """Одна сцена обработкой полного композита: выравнивание яркости и детренд."""
    from experiments.openlin_winter import detrend, normalize

    good = np.isfinite(b) & (b > 0)
    return detrend(normalize(b.astype("float32"), good), good, res_m)


def response(img: np.ndarray, valid: np.ndarray, res_m: float) -> np.ndarray:
    """Тот же отклик, что внутри трассировки: гребень в сигмах окрестности."""
    from experiments.openlin_iso import local_norm, ridge_response
    from experiments.openlin_shadow import standardize

    if not valid.any():                     # сцена сюда не достаёт — не голосует
        return np.zeros(valid.shape, "float32")
    x = np.where(valid, standardize(img, valid), 0.0).astype("float32")
    z = local_norm(np.where(valid, ridge_response(x), 0.0), valid, res_m)
    return np.where(valid, z, 0.0).astype("float32")


def tract(o, valid, zone, strike, seam, res_m, log=None) -> list:
    """Тракт до подтверждения — те же шаги, что в openlin_zone_lines_v3."""
    raw = trace(o, valid, res_m, local=True, min_len_m=0)
    zz = response(o, valid, res_m)
    raw = extend_ends(raw, zz, valid, res_m, log)
    L = link_same_side(raw, zone, strike, res_m, log, z=zz)
    L = [x for x in L if x["длина_км"] * 1000 >= MIN_LEN_M]
    return drop_seam_lines(L, seam, res_m, log)


def gated(lines, zone, strike) -> np.ndarray:
    """Маска «линия в зоне согласия и вдоль её простирания» — прежний отбор."""
    out = np.zeros(len(lines), bool)
    for i, r in enumerate(lines):
        p = r["pts"]
        out[i] = (zone[p[:, 0], p[:, 1]].mean() >= ZONE_FRAC and
                  _adiff(r["азимут"], _line_strike(p, strike)) <= DIR_TOL_DEG)
    return out


def twin_stack(channels, valids, rng, n_iter: int | None = None):
    """Двойник стопки с ОБЩИМИ случайными фазами, по одному каналу за раз.

    Та же процедура, что :func:`openlin_kora.surrogate` (IAAFT: общий поворот
    фаз, затем чередование «вернуть амплитуды — вернуть гистограмму»), но у
    каждого канала своя маска данных (сцены покрывают лист не целиком), и в
    памяти держится один канал: 13 каналов листа разом — несколько гигабайт.
    """
    from experiments.openlin_kora import IAAFT_ITER

    n_iter = IAAFT_ITER if n_iter is None else n_iter
    H, W = channels[0].shape
    theta = rng.uniform(-np.pi, np.pi, (H, W // 2 + 1))
    theta[0, 0] = 0.0
    turn = np.exp(1j * theta)
    for x, v in zip(channels, valids):
        if not v.any():
            yield np.full(x.shape, np.nan, "float32")
            continue
        X = np.where(v, x, np.median(x[v])).astype("float64")
        F = np.fft.rfft2(X)
        amp = np.abs(F)
        Y = np.fft.irfft2(F * turn, s=X.shape)
        target = np.sort(X[v])

        def remap(Y):
            out = Y.copy()
            out[v] = target[np.argsort(np.argsort(Y[v]))]
            return out

        for _ in range(n_iter):
            Y = remap(Y)
            Y = np.fft.irfft2(amp * np.exp(1j * np.angle(np.fft.rfft2(Y))), s=X.shape)
        yield np.where(v, remap(Y), np.nan).astype("float32")


def line_votes(lines, zs, vs, rng):
    """Голоса сцен по каждой линии и её собственный нуль.

    Возвращает (confirmed, votes, thr, vthr, voter): thr[l, s] — порог сцены s
    для линии l, vthr[l] — порог голосов, voter[l, s] — голосует ли сцена.
    """
    S = len(zs)
    H, W = zs[0].shape
    nL = len(lines)
    thr = np.full((nL, S), np.inf, "float32")
    voter = np.zeros((nL, S), bool)
    votes = np.zeros(nL, int)
    vthr = np.full(nL, np.inf)
    for i, r in enumerate(lines):
        p = r["pts"]
        dr = np.concatenate([[0], rng.integers(200, H - 200, LINE_SHIFTS)])
        dc = np.concatenate([[0], rng.integers(200, W - 200, LINE_SHIFTS)])
        rr = (p[:, 0][None, :] + dr[:, None]) % H
        cc = (p[:, 1][None, :] + dc[:, None]) % W
        hits = np.zeros((S, LINE_SHIFTS + 1), bool)
        for s in range(S):
            val = vs[s][rr, cc]
            cov = val.mean(1)
            if cov[0] < SCENE_COVER:
                continue
            m = np.where(cov >= SCENE_COVER,
                         (zs[s][rr, cc] * val).sum(1) / np.maximum(val.sum(1), 1),
                         np.nan)
            null = m[1:][np.isfinite(m[1:])]
            if len(null) < LINE_MIN_NULL:
                continue
            voter[i, s] = True
            thr[i, s] = np.quantile(null, LINE_Q)
            hits[s] = np.nan_to_num(m, nan=-np.inf) > thr[i, s]
        v = hits.sum(0)
        votes[i] = v[0]
        vthr[i] = np.quantile(v[1:], VOTE_Q)
    confirmed = (votes >= MIN_VOTES) & (votes > vthr)
    return confirmed, votes, thr, vthr, voter


def set_shift_null(lines, zs, vs, thr, vthr, voter, sel, n_shifts, rng):
    """Сдвиговый нуль всего набора: линии едут по листу целиком, сцены стоят.

    Для каждого сдвига голосование повторяется с ТЕМИ ЖЕ порогами каждой
    линии; статистика — доля подтверждённых. Возвращает долю нуля и массив.
    """
    idx = np.flatnonzero(sel)
    if not len(idx):
        return np.zeros(0)
    P = np.concatenate([lines[i]["pts"] for i in idx])
    seg = np.concatenate([np.full(len(lines[i]["pts"]), k) for k, i in enumerate(idx)])
    n = np.bincount(seg).astype("float64")
    T, V, Q = thr[idx], voter[idx], vthr[idx]
    H, W = zs[0].shape
    out = np.empty(n_shifts)
    for g in range(n_shifts):
        dr, dc = int(rng.integers(200, H - 200)), int(rng.integers(200, W - 200))
        rr, cc = (P[:, 0] + dr) % H, (P[:, 1] + dc) % W
        v = np.zeros(len(idx), int)
        for s in range(len(zs)):
            val = vs[s][rr, cc]
            cnt = np.bincount(seg, weights=val, minlength=len(idx))
            sm = np.bincount(seg, weights=zs[s][rr, cc] * val, minlength=len(idx))
            m = sm / np.maximum(cnt, 1)
            v += (V[:, s] & (cnt / n >= SCENE_COVER) & (m > T[:, s])).astype(int)
        out[g] = ((v >= MIN_VOTES) & (v > Q)).mean()
    return out


def cross_probe(site: str) -> None:
    """Проба перекрёстной схемы: линии с половины А судят сцены Б, и наоборот.

    Независимость полная: у половин нет общих сцен, так что двойного учёта,
    погубившего прямое голосование, здесь нет. Правило: линию подтверждают,
    если её видит не меньше половины независимых сцен, покрывающих это место
    (на большей части листа у половины Б одна сцена на плитку, у А — две).
    Проба на той же вырезке 24 x 24 км, один двойник; в счётчик не идёт.
    """
    from scipy.ndimage import distance_transform_edt

    from experiments.openlin_zone_lines_v3 import build_halves

    def log(m=""):
        print(m, flush=True)

    t0 = time.time()
    crop = (slice(400, 1600), slice(300, 1500))
    rng = np.random.default_rng(SEED)
    w = dict(np.load(COMP / f"winter_{site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    H0, W0 = w["shade"].shape
    k = int(round(100 / res_m))
    zone, strike = zone_and_strike(site, k)
    zone, strike = zone[:H0, :W0][crop], strike[:H0, :W0][crop]
    halves = build_halves(site, lambda *_: None)
    comp = {n: halves[n][crop].astype("float32") for n in ("А", "Б")}
    by_tile: dict[str, list] = {}
    for f in scene_files():
        tile = next(p for p in f.name.split("_") if p.startswith("T"))
        by_tile.setdefault(tile, []).append(f)
    member = {"А": [], "Б": []}
    for _t, group in sorted(by_tile.items()):
        member["А"] += group[0::2]; member["Б"] += group[1::2]
    shades = {n: [scene_shade(np.load(f)["B04"], res_m)[crop] for f in member[n]]
              for n in member}
    valid = np.isfinite(comp["А"]) & np.isfinite(comp["Б"])
    seam = ~valid
    log(f"ПРОБА перекрёстной схемы: вырезка 24 x 24 км, один двойник; сцен в "
        f"половинах {len(member['А'])} и {len(member['Б'])}")

    def judge(o, sc_other):
        vs = [np.isfinite(s) for s in sc_other]
        zs = [response(s, v, res_m) for s, v in zip(sc_other, vs)]
        L = tract(np.where(valid, o, np.nan), valid, zone, strike, seam, res_m)
        _c, votes, _t, _q, voter = line_votes(L, zs, vs, rng)
        n_v = voter.sum(1)
        conf = (votes >= 1) & (2 * votes >= n_v)
        return L, conf, n_v

    def both(cA, cB, sA, sB, name):
        out = {}
        for src, o, other in (("А", cA, sB), ("Б", cB, sA)):
            L, conf, n_v = judge(o, other)
            out[src] = (L, conf)
            log(f"  {name}, линии с половины {src}: {len(L)}, подтверждено сценами "
                f"другой половины {int(conf.sum())} ({100 * conf.mean() if len(L) else 0:.0f} %); "
                f"судей на линию (медиана) {int(np.median(n_v)) if len(L) else 0} "
                f"({time.time() - t0:.0f} с)")
        return out

    real = both(comp["А"], comp["Б"], shades["А"], shades["Б"], "снимок")
    chans = [comp["А"], comp["Б"]] + shades["А"] + shades["Б"]
    vals = [np.isfinite(c) for c in chans]
    tw = list(twin_stack(chans, vals, np.random.default_rng(900)))
    nA = len(shades["А"])
    fake = both(tw[0], tw[1], tw[2:2 + nA], tw[2 + nA:], "двойник")

    log("")
    tot_r = tot_f = n_r = n_f = 0
    for src in ("А", "Б"):
        (Lr, cr), (Lf, cf) = real[src], fake[src]
        r, f = cr.mean() if len(Lr) else 0, cf.mean() if len(Lf) else 0
        tot_r += int(cr.sum()); tot_f += int(cf.sum()); n_r += len(Lr); n_f += len(Lf)
        log(f"линии с половины {src}: подтверждение у снимка {100 * r:.0f} %, у "
            f"двойника {100 * f:.0f} %, мусор этапа {100 * f / max(r, 1e-9):.0f} % (порог 20 %)")
    r, f = tot_r / max(n_r, 1), tot_f / max(n_f, 1)
    log(f"вместе: снимок {tot_r} из {n_r}, двойник {tot_f} из {n_f}; мусор этапа "
        f"{100 * f / max(r, 1e-9):.0f} %; подтверждённых у двойника {tot_f} против "
        f"{tot_r} у снимка ({100 * tot_f / max(tot_r, 1):.0f} %)")

    ref = np.load(M / f"openlin_zone_lines_{site}_v3.npz")["C_all"][crop]
    Ls = [x for s in ("А", "Б") for x, c in zip(*real[s]) if c]
    dold = distance_transform_edt(~ref, sampling=(res_m, res_m))
    same = sum(1 for x in Ls if (dold[x["pts"][:, 0], x["pts"][:, 1]] <= 60).mean() >= 0.3)
    km = sum(x["длина_км"] for x in Ls)
    log(f"для сведения: подтверждено {len(Ls)} линий ({km:.0f} км, с повторами между "
        f"половинами); совпадают с действующим набором {same}, вне его {len(Ls) - same}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--перекрёстно", dest="cross", action="store_true",
                    help="проба перекрёстной схемы на вырезке")
    ap.add_argument("--проба", dest="probe", action="store_true",
                    help="вырезка 24 x 24 км, один двойник, 200 сдвигов — проверка кода")
    a = ap.parse_args()
    if a.cross:
        cross_probe(a.site)
        return

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_iso import perturb

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    opt = w["shade"].astype("float32")
    valid = np.isfinite(opt)
    H0, W0 = opt.shape
    k = int(round(100 / res_m))
    zone, strike = zone_and_strike(a.site, k)
    zone = zone[:H0, :W0] & valid
    strike = strike[:H0, :W0]
    up = lambda x: np.repeat(np.repeat(x, k, 0), k, 1)[:H0, :W0]
    zone = up(close_along_strike(zone[::k, ::k], strike[::k, ::k], 100.0)) & valid
    cnt = w["count"][:H0, :W0]
    seam = np.zeros(valid.shape, bool)
    seam[:, :-1] |= cnt[:, :-1] != cnt[:, 1:]
    seam[:-1, :] |= cnt[:-1, :] != cnt[1:, :]
    seam |= ~valid

    crop = (slice(None), slice(None))
    n_fakes, n_set = N_FAKES, SET_SHIFTS
    if a.probe:
        crop = (slice(400, 1600), slice(300, 1500))
        n_fakes, n_set = 1, 200
        log("ПРОБА: вырезка 24 x 24 км, один двойник, 200 сдвигов — это проверка "
            "кода, не прогон, в счётчик не идёт")
    opt, valid, zone, strike, seam = (x[crop] for x in (opt, valid, zone, strike, seam))

    files = scene_files()
    shades = [scene_shade(np.load(f)["B04"], res_m)[crop] for f in files]
    svalid = [np.isfinite(s) for s in shades]
    zs = [response(s, v, res_m) for s, v in zip(shades, svalid)]
    log(f"сцен {len(files)}, карты отклика построены ({time.time() - t0:.0f} с); "
        f"порог значимости 0.05/{VOTE_N_TESTS_PRIOR + VOTE_N_TESTS} = {VOTE_ALPHA:.6f}")

    def run(o, zmaps, vmaps, lg=None):
        L = tract(o, valid, zone, strike, seam, res_m, lg)
        conf, votes, thr, vthr, voter = line_votes(L, zmaps, vmaps, rng)
        g = gated(L, zone, strike)
        return L, conf, votes, thr, vthr, voter, g

    L, conf, votes, thr, vthr, voter, g = run(opt, zs, svalid, log)
    km = lambda sel: sum(L[i]["длина_км"] for i in np.flatnonzero(sel))
    log(f"снимок: линий {len(L)}; в зонах {int(g.sum())}, подтверждено "
        f"{int((conf & g).sum())} ({km(conf & g):.0f} км); весь лист "
        f"подтверждено {int(conf.sum())} ({km(conf):.0f} км) ({time.time() - t0:.0f} с)")
    nv = voter.sum(1)
    log(f"  голосующих сцен на линию (25/50/75 %): "
        f"{' / '.join(str(int(v)) for v in np.percentile(nv, [25, 50, 75]))}; "
        f"голосов у подтверждённых (50 %): {int(np.median(votes[conf])) if conf.any() else 0}")

    # ---- двойники через весь тракт и голосование ---------------------------
    fk = {"A_g": 0, "C_g": 0, "A": 0, "C": 0}
    for j in range(n_fakes):
        chans = twin_stack([opt] + shades, [valid] + svalid,
                           np.random.default_rng(900 + j))
        o_t = next(chans)
        zt, vt = [], []
        for s_t, v in zip(chans, svalid):
            zt.append(response(s_t, v, res_m)); vt.append(v)
        Lt, ct, _v, _t, _q, _w, gt = run(np.where(valid, o_t, np.nan), zt, vt)
        fk["A"] += len(Lt); fk["C"] += int(ct.sum())
        fk["A_g"] += int(gt.sum()); fk["C_g"] += int((ct & gt).sum())
        log(f"  двойник {j + 1}: линий {len(Lt)}, подтверждено {int(ct.sum())}; "
            f"в зонах {int(gt.sum())} / {int((ct & gt).sum())} ({time.time() - t0:.0f} с)")
        del zt, vt

    mark = {True: "ПРОЙДЕНО", False: "НЕ пройдено"}
    results = {}
    for name, sel, fa, fc in (("в зонах", g, fk["A_g"], fk["C_g"]),
                              ("весь лист", np.ones(len(L), bool), fk["A"], fk["C"])):
        real = (conf & sel).sum() / max(sel.sum(), 1)
        fake = fc / max(fa, 1)
        junk = fake / max(real, 1e-9)
        null = set_shift_null(L, zs, svalid, thr, vthr, voter, sel, n_set, rng)
        p = float((1 + (null >= real).sum()) / (1 + len(null)))
        Ls = [L[i] for i in np.flatnonzero(conf & sel)]
        med = float(np.median([x["длина_км"] for x in Ls])) if Ls else 0.0
        results[name] = (Ls, real, fake, junk, p, med)
        log(f"\n{name}: подтверждено {len(Ls)} из {int(sel.sum())} "
            f"({100 * real:.1f} %), у двойника {100 * fake:.1f} %")
        log(f"  мусор этапа подтверждения: {100 * junk:.1f} % (порог "
            f"{100 * STAGE_JUNK_MAX:.0f} %) -> {mark[junk <= STAGE_JUNK_MAX]}")
        log(f"  сдвиговый нуль ({len(null)} сдвигов): доля {100 * null.mean():.1f} % "
            f"(макс {100 * null.max():.1f} %) против {100 * real:.1f} %, p = {p:.5f} "
            f"(порог {VOTE_ALPHA:.6f}) -> {mark[p <= VOTE_ALPHA]}")
        log(f"  медиана длины: {med:.2f} км (порог {V3_MEDIAN_KM_MIN}) -> "
            f"{mark[med >= V3_MEDIAN_KM_MIN]}")

    # ---- устойчивость: возмущённый композит, те же сцены ---------------------
    Lp, cp, *_rest, gp = run(perturb(opt, valid), zs, svalid)
    for name, Ls, Lps in (("в зонах", results["в зонах"][0],
                           [Lp[i] for i in np.flatnonzero(cp & gp)]),
                          ("весь лист", results["весь лист"][0],
                           [Lp[i] for i in np.flatnonzero(cp)])):
        mC, mP = mask_of(Ls, valid.shape), mask_of(Lps, valid.shape)
        stab = cover(mC, mP, res_m, STAB_TOL_M)
        rr = np.random.default_rng(7)
        null = np.mean([cover(mC, np.roll(np.roll(mP, int(rr.integers(200, valid.shape[0] - 200)), 0),
                                          int(rr.integers(200, valid.shape[1] - 200)), 1),
                              res_m, STAB_TOL_M) for _ in range(20)])
        ok = stab >= STAB_MIN and stab >= STAB_OVER_NULL * max(null, 1e-9)
        log(f"устойчивость, {name}: {stab:.3f}, нуль {null:.3f} -> {mark[ok]}")

    # ---- для сведения: сравнение с подтверждением половинами -----------------
    ref = np.load(M / f"openlin_zone_lines_{a.site}_v3.npz")
    from scipy.ndimage import distance_transform_edt
    for name, key in (("в зонах", "C"), ("весь лист", "C_all")):
        Ls = results[name][0]
        old = ref[key][crop]
        dold = distance_transform_edt(~old, sampling=(res_m, res_m))
        same = sum(1 for x in Ls if (dold[x["pts"][:, 0], x["pts"][:, 1]] <= 60).mean() >= 0.3)
        mnew = mask_of(Ls, valid.shape)
        dnew = distance_transform_edt(~mnew, sampling=(res_m, res_m))
        from scipy.ndimage import label
        lab, n_old = label(old, np.ones((3, 3), bool))
        on = lab > 0
        tot = np.bincount(lab[on], minlength=n_old + 1)[1:]
        hit = np.bincount(lab[on & (dnew <= 60)], minlength=n_old + 1)[1:]
        lost = int((hit / np.maximum(tot, 1) < 0.3).sum())
        log(f"для сведения, {name}: голосованием {len(Ls)} линий, из них совпадают "
            f"с подтверждёнными половинами {same}, новых {len(Ls) - same}; "
            f"кусков прежнего набора, которых голосование не подтвердило: {lost} из {n_old}")

    if not a.probe:
        np.savez_compressed(
            M / f"openlin_vote_{a.site}.npz",
            A=mask_of([L[i] for i in np.flatnonzero(g)], valid.shape),
            C=mask_of(results["в зонах"][0], valid.shape),
            A_all=mask_of(L, valid.shape),
            C_all=mask_of(results["весь лист"][0], valid.shape), zone=zone)
    log(f"\nготово ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
