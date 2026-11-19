"""LINB1, разбор (настройка, описательно): точность линий Анфисы по группам среднего P, D, длины, прямизны.

Скрипт, давший outputs/fam/_linb_diag.log (11.11.2026); перенесён в репозиторий по аудиту LINB2.
Запуск: python -m experiments.openlin_linb_diag
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from experiments import openlin_ref_match as rm
from experiments.openlin_transfer_check import COMP, MAIN_LAYER, TOL_M
from experiments.openlin_anfisa_lines import rasterize, length_m
w = dict(np.load(COMP / "winter_utsjoki_20m_лист.npz")); res = float(w["res_m"])
V = np.load("outputs/metrics/openlin_zone_lines_utsjoki_v3_анфиса1_линии.npz")
flat, off = V["rc100"], V["offsets"]
lines = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
K = np.load("outputs/fam/linb1_utsjoki_карта.npz"); P, D = K["P"], K["D"]
shape = np.load("outputs/metrics/openlin_zone_lines_utsjoki_v3_настройка1.npz")["C_all"].shape
valid = np.isfinite(w["shade"][:shape[0], :shape[1]])
ids = np.where(valid, rasterize(lines, shape, 5), 0)
ang, fid = rm.ref_directions(MAIN_LAYER, w, shape); ang = np.where(valid, ang, np.nan)
pos = rm.positions(ang, fid, valid, rm.shifts(shape)[:30])
m = ids > 0
lab, n = rm.components(m)
d = rm.our_directions(m, res, lab, n)
rr, cc = np.nonzero(m & np.isfinite(d))
def hit(p):
    rc, a, _ = p
    t = cKDTree(rc * res); q = t.query_ball_point(np.stack([rr, cc], 1) * res, TOL_M)
    return np.array([any(rm._axial_diff(a[j], d[r, c]) <= 20 for j in js) for js, r, c in zip(q, rr, cc)])
h0 = hit(pos[0]); hn = np.mean([hit(p) for p in pos[1:6]], 0)
lid = ids[rr, cc]
nl = len(lines)
tot = np.bincount(lid, minlength=nl + 1)[1:]; hh = np.bincount(lid, weights=h0, minlength=nl + 1)[1:]
nn = np.bincount(lid, weights=hn, minlength=nl + 1)[1:]
L = np.array([length_m(pl) for pl in lines]) / 1000
mP = np.array([np.nanmean(P[pl[:, 0].astype(int), pl[:, 1].astype(int)]) for pl in lines])
mD = np.array([np.nanmean(D[pl[:, 0].astype(int), pl[:, 1].astype(int)]) for pl in lines])
chord = np.array([np.hypot(*(pl[-1] - pl[0])) * 0.1 for pl in lines])
st = chord / np.maximum(L, 1e-6)
ok = tot > 0
def table(x, name, q=(0, .2, .4, .6, .8, 1)):
    e = np.quantile(x[ok], q)
    print(f"\n{name}: квантильные группы (точность = доля ячеек у эталона; нуль — 5 сдвигов)")
    for a_, b_ in zip(e[:-1], e[1:]):
        s = ok & (x >= a_) & (x <= b_)
        print(f"  {a_:7.2f}–{b_:7.2f}: линий {s.sum():5d}, км {L[s].sum():6.0f}, точность "
              f"{hh[s].sum() / tot[s].sum():.3f}, нуль {nn[s].sum() / tot[s].sum():.3f}")
print(f"всего: точность {hh.sum() / tot.sum():.3f}, нуль {nn.sum() / tot.sum():.3f}")
table(mP, "среднее P вдоль линии"); table(mD, "среднее D вдоль линии")
table(L, "длина, км"); table(st, "прямизна (хорда/длина)")
