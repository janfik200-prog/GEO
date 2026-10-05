"""Окна «сшивка как сейчас / строгая сшивка» — для оценки глазом до прогона.

ОТКУДА ПОСТАНОВКА. Просмотр 05.10.2026: «сшивка всё ещё работает криво».
ЗАМЕР по рабочему набору (12 сцен, 3202 линии от 500 м): две трети
подтверждённой длины — сшитые линии (1790 км из 2613); у линий из трёх и
более обрывков треть длины — мостики, то есть прямые отрезки, дорисованные
через разрыв; сшитые линии в полтора раза кривее целых (кривизна 3.0 % против
2.0 %) и подтверждаются хуже (41-50 % против 63 %). Сшивку раскручивали под
условие «медиана длины >= 1.5 км»: разрыв до 1192 м, сдвиг вбок до 600 м.

СТРОГАЯ СШИВКА. Обрывок соединяется только с тем, который его прямо
продолжает: боковой сдвиг не больше 60 м — это допуск дрожания концов
трассировки (``CONF_TOL_M``), то есть «на одной прямой» с точностью прибора.
Кулисные цепочки не склеиваются и остаются отдельными отрезками. Остальные
условия прежние: направление, разрыв с опорой мостика на отклик, запрет
перекрытия. Это ПРОСМОТР, не прогон: проверок на двойнике здесь нет, в
счётчик не идёт.

Запуск из корня репозитория:
    python -m experiments.openlin_link_compare --site anabar
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import experiments.openlin_zone_lines_v3 as v3
from experiments.openlin_zone_lines import COMP, MIN_LEN_M, OUT, mask_of, trace

STRICT_OFFSET_M = v3.CONF_TOL_M      # 60 м
OUT_DIR = OUT.parent / "линеаменты_сшивка"
N_WINDOWS = 14
WIN = 250                            # 5 x 5 км


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt, uniform_filter

    from experiments.openlin_shade import _stretch
    from experiments.openlin_vote import response

    t0 = time.time()
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    h = v3.build_halves(a.site, lambda *_: None)
    valid = np.isfinite(img) & np.isfinite(h["А"]) & np.isfinite(h["Б"])
    cnt = w["count"]
    seam = np.zeros(valid.shape, bool)
    seam[:, :-1] |= cnt[:, :-1] != cnt[:, 1:]
    seam[:-1, :] |= cnt[:-1, :] != cnt[1:, :]
    seam |= ~valid
    z = response(img, valid, res_m)
    raw = v3.extend_ends(trace(img, valid, res_m, local=True, min_len_m=0), z, valid, res_m)
    ids = np.zeros(valid.shape, "int32")
    for i, r in enumerate(raw):
        ids[r["pts"][:, 0], r["pts"][:, 1]] = i + 1
    dts = [distance_transform_edt(~mask_of(trace(x, valid, res_m, local=True), valid.shape),
                                  sampling=(res_m, res_m)) for x in (h["А"], h["Б"])]

    def variant(offset_m, cut_kulisa=False):
        keep = v3.V4_OFFSET_M
        v3.V4_OFFSET_M = offset_m
        try:
            L = v3.link_same_side([dict(r) for r in raw], None, None, res_m, None, z=z)
        finally:
            v3.V4_OFFSET_M = keep
        L = v3.drop_seam_lines([x for x in L if x["длина_км"] * 1000 >= MIN_LEN_M],
                               seam, res_m)
        if cut_kulisa:
            # цепочка остаётся одной линией (длина и отбор — по цепочке), но
            # ячейки мостиков через кулисы убираются: их никто не находил
            W_ = valid.shape[1]
            for x in L:
                kul = x.get("кулисы")
                if kul is not None and len(kul):
                    code = x["pts"][:, 0] * W_ + x["pts"][:, 1]
                    x["pts"] = x["pts"][~np.isin(code, kul[:, 0] * W_ + kul[:, 1])]
        conf = np.array([all((d[x["pts"][:, 0], x["pts"][:, 1]] <= v3.CONF_TOL_M).mean()
                             >= v3.CONF_FRAC for d in dts) for x in L])
        return L, conf

    def describe(name, L, conf):
        ln = np.array([x["длина_км"] for x in L])
        br = np.array([(ids[x["pts"][:, 0], x["pts"][:, 1]] == 0).mean() for x in L])
        cv = []
        for x in L:
            p = x["pts"].astype(float); q = p - p.mean(0)
            vt = np.linalg.svd(q, full_matrices=False)[2]; t = q @ vt[0]
            cv.append(np.abs(q @ vt[1]).mean() / max(t.max() - t.min(), 1))
        cv = np.array(cv)
        print(f"{name}: линий {len(L)}, подтверждено {int(conf.sum())} "
              f"({ln[conf].sum():.0f} км), медиана длины подтверждённых "
              f"{np.median(ln[conf]):.2f} км, кривизна {100 * np.median(cv[conf]):.1f} %, "
              f"мостики — {100 * (br[conf] * ln[conf]).sum() / ln[conf].sum():.0f} % длины "
              f"({time.time() - t0:.0f} с)", flush=True)

    Lo, co = variant(v3.V4_OFFSET_M)
    Ls, cs = variant(STRICT_OFFSET_M)
    Lc, cc = variant(v3.V4_OFFSET_M, cut_kulisa=True)
    describe(f"как сейчас (сдвиг до {v3.V4_OFFSET_M:.0f} м)", Lo, co)
    describe(f"строгая (сдвиг до {STRICT_OFFSET_M:.0f} м)  ", Ls, cs)
    describe("цепочки без мостиков через кулисы", Lc, cc)

    def masks(L, conf):
        return (mask_of([x for x, c in zip(L, conf) if c], valid.shape),
                mask_of([x for x, c in zip(L, conf) if not c], valid.shape))

    (oc, on), (sc, sn), (kc, kn) = masks(Lo, co), masks(Ls, cs), masks(Lc, cc)
    rawm = ids > 0
    for nm, m in (("как сейчас", oc), ("строгая", sc), ("цепочки", kc)):
        print(f"  {nm}: найденных детектором ячеек в подтверждённом наборе "
              f"{(m & rawm).sum() * res_m / 1000:.0f} км, дорисованных "
              f"{(m & ~rawm).sum() * res_m / 1000:.0f} км", flush=True)
    # окна — там, где два варианта расходятся сильнее всего
    diff = uniform_filter(((oc | on) ^ (sc | sn)).astype("float32"), WIN)
    half = WIN // 2
    diff[:half] = 0; diff[-half:] = 0; diff[:, :half] = 0; diff[:, -half:] = 0
    base = _stretch(np.where(valid, img, np.nan))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
    for i in range(1, N_WINDOWS + 1):
        r, c = np.unravel_index(int(np.argmax(diff)), diff.shape)
        diff[max(r - WIN, 0):r + WIN, max(c - WIN, 0):c + WIN] = 0
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        fig, ax = plt.subplots(2, 2, figsize=(19, 19.6), facecolor="#fcfcfb")
        for axx, pair, title in (
                (ax[0, 0], None, "снимок — смотрите сами"),
                (ax[0, 1], (oc, on), "как сейчас: красные подтверждены, жёлтые — нет"),
                (ax[1, 0], (sc, sn), "строгая сшивка: только прямое продолжение"),
                (ax[1, 1], (kc, kn), "цепочки: куски те же, мостики через кулисы убраны")):
            axx.imshow(base[sl], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if pair is not None:
                ov = np.zeros((WIN, WIN, 4), "float32")
                ov[pair[1][sl]] = (0.95, 0.60, 0.0, 0.95)
                ov[pair[0][sl]] = (0.85, 0.05, 0.05, 0.95)
                axx.imshow(ov)
            axx.set_title(title, loc="left", fontsize=15)
            axx.set_xticks([]); axx.set_yticks([])
        fig.suptitle(f"Окно {i} · 5 × 5 км · центр ({r}, {c}) · север сверху",
                     x=0.01, ha="left", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT_DIR / f"{i:02d}_окно_{r}_{c}.png", dpi=80,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"готово: {OUT_DIR} ({time.time() - t0:.0f} с)", flush=True)


if __name__ == "__main__":
    main()
