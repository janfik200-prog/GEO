"""Корреляция 160 признаков шага 1: сколько в них на самом деле независимого.

ЧТО ЗА ПРИЗНАКИ. Шаг 1 прежнего конвейера: 10 каналов композита (яркость, NDVI,
радар VV/VH, шесть слоёв интерферометрической когерентности) и для каждого из
них на трёх масштабах окрестности (500 м, 1 км, 2 км) пять признаков
структурного тензора и гессиана (:mod:`openlin_neigh`): когерентность ``coh``,
энергия ``eng``, направление ``cos``/``sin`` (в удвоенном угле) и гребневость
``rdg``. Итого 10 + 10 x 3 x 5 = 160.

ЗАЧЕМ. 160 признаков — это число колонок, а не число независимых сведений о
местности. Если большая их часть повторяет друг друга, то и сжатие в 8
компонент, и любые выводы по «важности признаков» читаются иначе.

ЧТО СЧИТАЕТСЯ:

* ранговая корреляция Спирмена по случайной выборке ячеек — признаки с тяжёлыми
  хвостами, и линейная корреляция определялась бы редкими выбросами;
* эффективное число независимых признаков — отношение участия собственных чисел
  корреляционной матрицы (сумма в квадрате, делённая на сумму квадратов), и
  сколько компонент набирают 90 % разброса;
* группы почти-дубликатов: иерархическая кластеризация по полной связи на
  расстоянии 1 - |r|, срез на |r| >= 0.9 — внутри группы любые два признака
  коррелируют не слабее;
* средний |r| по типам пар: тот же канал на другом масштабе, тот же признак у
  другого канала, и так далее.

Направления (``cos``/``sin``) — величины круговые, и их корреляция с остальными
по построению мала; это свойство признака, а не находка.

Запуск из корня репозитория:
    python -m experiments.openlin_feat_corr --site anabar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

M = Path("outputs/metrics")
OUT = Path("outputs/steps")

#: Сколько ячеек брать для корреляции. Соседние ячейки 100-метровой сетки
#: сильно связаны сглаживанием в 0.5-2 км, так что больше выборка ничего не
#: добавит, кроме времени.
N_SAMPLE = 40_000

#: Порог «почти дубликатов».
DUP_R = 0.9

#: Цвета расходящейся шкалы: синий <-> красный с серой серединой (эталонная
#: палитра dataviz). Середина серая, чтобы «нет связи» читалось как «ничего».
NEG, MID, POS = "#2a78d6", "#f0efec", "#e34948"


def build_features(site: str) -> tuple[np.ndarray, list[str], list[tuple]]:
    """160 признаков по валидным ячейкам и их описание (канал, масштаб, тип)."""
    from experiments.openlin_neigh import SCALES_M, neigh_stack
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    nx, nnames = neigh_stack(x, use, valid, res_m, base=use)
    stack = np.concatenate([x, nx])
    names = list(use) + list(nnames)
    meta = [(c, "сырой", "канал") for c in use]
    for n in nnames:
        meta.append(_parse(n, use, SCALES_M))
    rows = stack[:, valid].T
    return rows, names, meta


def _parse(name: str, use: tuple, scales: tuple) -> tuple:
    """Разобрать имя признака окрестности на (канал, масштаб, тип)."""
    ch = next((c for c in sorted(use, key=len, reverse=True)
               if name.startswith(c)), name)
    rest = name[len(ch):].strip("_")
    kind = next((k for k in ("coh", "eng", "cos", "sin", "rdg") if k in rest),
                rest)
    sc = next((f"{int(s)}" for s in scales if f"{int(s)}" in rest), "?")
    return ch, sc, kind


def spearman(rows: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    r = np.apply_along_axis(rankdata, 0, rows)
    c = np.corrcoef(r, rowvar=False)
    return np.nan_to_num(c, nan=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--select", action="store_true",
                    help=f"убрать дубликаты и оставить {N_KEEP} признаков")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    rows, names, meta = build_features(a.site)
    n = len(names)
    rng = np.random.default_rng(42)
    sel = rng.choice(len(rows), min(N_SAMPLE, len(rows)), replace=False)
    const = rows[sel].std(0) < 1e-9
    log(f"площадка {a.site}: признаков {n}, ячеек в выборке {len(sel)}"
        + (f", постоянных признаков {int(const.sum())}" if const.any() else ""))
    c = spearman(rows[sel])
    np.fill_diagonal(c, 1.0)

    # эффективное число независимых признаков
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)[::-1]
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    k90 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1)
    k99 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1)
    log(f"\nэффективное число независимых признаков (отношение участия): "
        f"{pr:.1f} из {n}")
    log(f"компонент на 90 % разброса: {k90}, на 99 %: {k99}; первая компонента "
        f"несёт {100 * ev[0] / ev.sum():.0f} %")

    # группы почти-дубликатов
    d = 1.0 - np.abs(c)
    np.fill_diagonal(d, 0.0)
    link = linkage(squareform(d, checks=False), method="complete")
    lab = fcluster(link, t=1.0 - DUP_R, criterion="distance")
    n_groups = len(np.unique(lab))
    big = pd.Series(lab).value_counts()
    log(f"\nгрупп почти-дубликатов (|r| >= {DUP_R} внутри группы): "
        f"{n_groups}; одиночек {int((big == 1).sum())}; самые крупные — "
        + ", ".join(str(v) for v in big.head(5).tolist()))
    for g in big.index[:5]:
        members = [names[i] for i in np.flatnonzero(lab == g)]
        log(f"  группа из {len(members)}: " + ", ".join(members[:8])
            + (" …" if len(members) > 8 else ""))

    # средний |r| по типам пар
    iu = np.triu_indices(n, 1)
    ch = np.array([m[0] for m in meta])
    sc = np.array([m[1] for m in meta])
    kd = np.array([m[2] for m in meta])
    absr = np.abs(c[iu])
    same_ch = ch[iu[0]] == ch[iu[1]]
    same_sc = sc[iu[0]] == sc[iu[1]]
    same_kd = kd[iu[0]] == kd[iu[1]]
    kinds = [("тот же канал и тип, другой масштаб", same_ch & same_kd & ~same_sc),
             ("тот же канал и масштаб, другой тип", same_ch & same_sc & ~same_kd),
             ("тот же тип и масштаб, другой канал", ~same_ch & same_kd & same_sc),
             ("всё разное", ~same_ch & ~same_kd & ~same_sc)]
    log("\nсредний |r| по типам пар:")
    for name, m in kinds:
        if m.any():
            log(f"  {name:36s} {absr[m].mean():.2f} (медиана "
                f"{np.median(absr[m]):.2f}, пар {int(m.sum())})")
    log("\nсредний |r| одного и того же признака между масштабами, по типам:")
    for k in ("coh", "eng", "cos", "sin", "rdg"):
        m = same_ch & ~same_sc & (kd[iu[0]] == k) & same_kd
        if m.any():
            log(f"  {k:4s} {absr[m].mean():.2f}")

    # самые сильные пары между РАЗНЫМИ каналами
    cross = ~same_ch
    top = np.argsort(-absr * cross)[:10]
    log("\nсамые сильные связи между разными каналами:")
    for t in top:
        i, j = iu[0][t], iu[1][t]
        log(f"  {c[i, j]:+.2f}  {names[i]}  ~  {names[j]}")

    pd.DataFrame(c, index=names, columns=names).to_csv(
        M / f"openlin_feat_corr_{a.site}.csv", encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_feat_corr_{a.site}.csv'}")

    plot(c, names, meta, leaves_list(link), lab, a.site, pr, n_groups, log)
    if a.select:
        chosen = select_main(c, names, meta, lab, a.site, log)
        plot_selected(c, names, chosen, a.site, log)
        save_stack(a.site, chosen, log)


#: Сколько признаков оставить. Эффективное число независимых по отношению
#: участия собственных чисел — 20.0 из 160.
N_KEEP = 20


def dedup(c: np.ndarray, lab: np.ndarray) -> list[int]:
    """Из каждой группы почти-дубликатов — один, самый представительный.

    Представительность — сумма квадратов корреляций со всеми 160 признаками:
    из двух почти одинаковых признаков остаётся тот, через который лучше видны
    остальные.
    """
    score = (c ** 2).sum(0)
    keep = []
    for g in np.unique(lab):
        idx = np.flatnonzero(lab == g)
        keep.append(int(idx[np.argmax(score[idx])]))
    return sorted(keep)


def greedy_select(c: np.ndarray, cand: list[int], k: int
                  ) -> tuple[list[int], list[float]]:
    """Жадный отбор столбцов: каждый шаг — наибольший прирост объяснённого.

    Работает на корреляционной матрице всех признаков: после выбора признака
    его вклад вычитается из остаточной матрицы (шаг разложения Холецкого), и
    дубликат уже выбранного больше ничего добавить не может. Возвращает номера
    и накопленную долю разброса ВСЕХ признаков, объяснённую выбранными.
    """
    r = c.astype("float64").copy()
    total = float(np.trace(c))
    chosen, share = [], []
    left = list(cand)
    for _ in range(k):
        gain = [(r[:, j] ** 2).sum() / r[j, j] if r[j, j] > 1e-9 else -1.0
                for j in left]
        j = left[int(np.argmax(gain))]
        v = r[:, j] / np.sqrt(r[j, j])
        r = r - np.outer(v, v)
        chosen.append(j)
        left.remove(j)
        share.append(1.0 - float(np.trace(r)) / total)
    return chosen, share


def select_main(c, names, meta, lab, site, log) -> None:
    """Убрать дубликаты, оставить ``N_KEEP`` признаков, сохранить их стек."""
    import pandas as pd

    n = len(names)
    kept = dedup(c, lab)
    log(f"\nпосле удаления дубликатов (|r| >= {DUP_R}): {len(kept)} из {n}")
    chosen, share = greedy_select(c, kept, N_KEEP)
    ev = np.sort(np.clip(np.linalg.eigvalsh(c), 0, None))[::-1]
    pc_share = float(ev[:N_KEEP].sum() / ev.sum())
    log(f"{N_KEEP} признаков объясняют {100 * share[-1]:.0f} % разброса всех "
        f"{n}; {N_KEEP} главных компонент — {100 * pc_share:.0f} % (потолок для "
        f"любых {N_KEEP} чисел)")
    sub = np.abs(c[np.ix_(chosen, chosen)])
    np.fill_diagonal(sub, 0)
    log(f"наибольший |r| между выбранными: {sub.max():.2f}, средний "
        f"{sub[np.triu_indices(N_KEEP, 1)].mean():.2f}")

    rows = []
    log(f"\n{'№':>3} {'признак':34s} {'канал':22s} {'масштаб':>8s} {'тип':>6s}"
        f" {'накоплено':>10s}")
    for k, (j, s) in enumerate(zip(chosen, share), start=1):
        ch, sc, kd = meta[j]
        rows.append({"порядок": k, "признак": names[j], "канал": ch,
                     "масштаб_м": sc, "тип": kd, "накоплено": s})
        log(f"{k:3d} {names[j]:34s} {ch:22s} {sc:>8s} {kd:>6s} {100 * s:9.0f}%")
    df = pd.DataFrame(rows)
    log("\nсостав: по типу " + str(df["тип"].value_counts().to_dict())
        + "; по масштабу " + str(df["масштаб_м"].value_counts().to_dict()))
    df.to_csv(M / f"openlin_feat{N_KEEP}_{site}.csv", index=False,
              encoding="utf-8-sig")
    log(f"сохранено: {M / f'openlin_feat{N_KEEP}_{site}.csv'}")
    return chosen


def save_stack(site: str, chosen: list[int], log) -> None:
    """Сохранить отобранные признаки растром — вход для следующих шагов."""
    from experiments.openlin_neigh import neigh_stack
    from experiments.openlin_unet import build_stack, load_site
    from experiments.openlin_unsup import OPTIC, RADAR

    z = load_site(site)
    use = tuple(c for c in OPTIC + RADAR if c in z)
    res_m = float(z["res_m"])
    x, _y, valid, use = build_stack(z, use)
    nx, nnames = neigh_stack(x, use, valid, res_m, base=use)
    stack = np.concatenate([x, nx])[chosen].astype("float32")
    names = (list(use) + list(nnames))
    names = [names[i] for i in chosen]
    p = M / f"openlin_feat{N_KEEP}_{site}.npz"
    np.savez_compressed(p, stack=stack, names=np.array(names), valid=valid,
                        res_m=res_m, x0=z["x0"], y_top=z["y_top"],
                        epsg=z["epsg"])
    log(f"сохранено: {p} ({stack.shape[0]} x {stack.shape[1]} x "
        f"{stack.shape[2]})")


def plot_selected(c, names, chosen, site, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    sub = c[np.ix_(chosen, chosen)]
    lab = [names[j].replace("coh_summer_vv_", "лет.").replace(
        "coh_winter_vv_", "зим.") for j in chosen]
    fig, ax = plt.subplots(figsize=(11, 10), facecolor="#fcfcfb")
    im = ax.imshow(sub, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(lab)), lab, rotation=70, ha="right", fontsize=10,
                  color="#52514e")
    ax.set_yticks(range(len(lab)), lab, fontsize=10, color="#52514e")
    for s in ax.spines.values():
        s.set_visible(False)
    off = np.abs(sub - np.eye(len(sub)))
    ax.set_title(f"{len(chosen)} отобранных признаков: корреляция между ними\n"
                 f"наибольший |r| = {off.max():.2f}", fontsize=14,
                 color="#0b0b0b", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    fig.tight_layout()
    p = OUT / f"признаки_{len(chosen)}_{site}.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


def plot(c, names, meta, order, lab, site, pr, n_groups, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    ink, ink2 = "#0b0b0b", "#52514e"
    n = len(names)
    fig, ax = plt.subplots(1, 2, figsize=(22, 11.4), facecolor="#fcfcfb")

    # слева: по устройству признаков — канал за каналом
    chans = []
    for m in meta:
        if m[0] not in chans:
            chans.append(m[0])
    struct = sorted(range(n), key=lambda i: (
        chans.index(meta[i][0]), meta[i][1] != "сырой",
        int(meta[i][1]) if meta[i][1].isdigit() else 0, meta[i][2]))
    im = ax[0].imshow(c[np.ix_(struct, struct)], cmap=cmap, vmin=-1, vmax=1,
                      interpolation="nearest")
    bounds, centers, labels = [], [], []
    for chn in chans:
        idx = [k for k, i in enumerate(struct) if meta[i][0] == chn]
        bounds.append(idx[-1] + 0.5)
        centers.append((idx[0] + idx[-1]) / 2)
        labels.append(chn.replace("coh_", "ког ").replace("_vv_", " "))
    for b in bounds[:-1]:
        ax[0].axhline(b, color="#fcfcfb", lw=1.6)
        ax[0].axvline(b, color="#fcfcfb", lw=1.6)
    ax[0].set_xticks(centers, labels, rotation=60, ha="right", fontsize=10,
                     color=ink2)
    ax[0].set_yticks(centers, labels, fontsize=10, color=ink2)
    ax[0].set_title("по устройству: канал за каналом\n(внутри канала — сырой, "
                    "потом 500 м, 1 км, 2 км × пять признаков)", fontsize=14,
                    color=ink, loc="left")

    # справа: по сходству — средняя связь, шесть блоков, подписанных по типу
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    d = 1.0 - np.abs(c)
    np.fill_diagonal(d, 0.0)
    link = linkage(squareform(d, checks=False), method="average")
    order = leaves_list(link)
    blk = fcluster(link, 6, criterion="maxclust")[order]
    ax[1].imshow(c[np.ix_(order, order)], cmap=cmap, vmin=-1, vmax=1,
                 interpolation="nearest")
    tname = {"coh": "когерентность", "eng": "энергия", "cos": "направление cos",
             "sin": "направление sin", "rdg": "гребень", "канал": "сырые каналы"}
    start = 0
    for k in range(1, n + 1):
        if k == n or blk[k] != blk[start]:
            if k - start >= 3:
                ax[1].add_patch(plt.Rectangle((start - 0.5, start - 0.5),
                                              k - start, k - start, fill=False,
                                              ec=ink, lw=1.2))
                kinds = [meta[order[i]][2] for i in range(start, k)]
                top = max(set(kinds), key=kinds.count)
                label = tname.get(top, top)
                if top != "канал" and kinds.count("канал") >= 3:
                    label += " + сырые каналы"
                ax[1].text(start + 1, k - 2, f"{label} ({k - start})",
                           va="bottom", ha="left", fontsize=11, color=ink,
                           bbox=dict(fc="#fcfcfb", ec="none", alpha=0.85,
                                     pad=1.5))
            start = k
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("по сходству: блоки — один и тот же ТИП признака у всех\n"
                    f"10 каналов и 3 масштабов; эффективно независимых ≈ {pr:.0f}",
                    fontsize=14, color=ink, loc="left")
    for a_ in ax:
        for s in a_.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("ранговая корреляция Спирмена", color=ink2, fontsize=12)
    cb.outline.set_visible(False)
    fig.suptitle(f"Корреляция 160 признаков шага 1, площадка {site}",
                 fontsize=17, color=ink, x=0.02, ha="left")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"корреляция_160_признаков_{site}.png"
    fig.savefig(p, dpi=90, facecolor=fig.get_facecolor())
    log(f"сохранено: {p}")


if __name__ == "__main__":
    main()
