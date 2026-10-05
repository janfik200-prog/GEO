"""Парные снимки для глазной проверки: слева чистый кусок, справа — с разметкой.

ЗАЧЕМ. Все проверки ветки — машинные: подделка, сдвиговый нуль, устойчивость,
артефакты. Ни одна из них не отвечает на вопрос, видит ли ЧЕЛОВЕК линеамент
там, где его провёл детектор. Здесь лист режется на окна вокруг найденных
линий, и каждое окно сохраняется дважды: сначала без разметки (чтобы смотреть
непредвзято), потом с ней.

ЧТО В ПАПКЕ.
  ``00_обзор.png``       — весь лист с пронумерованными окнами;
  ``NN_окно_*.png``      — окно с линиями: слева чистое, справа с разметкой;
  ``контроль_N_*.png``   — окно В ЗОНЕ СОГЛАСИЯ, где детектор не нашёл ничего:
                           нужно, чтобы видеть разницу между «есть линия» и
                           «похожий рельеф без линии».

Окно 5 x 5 км при шаге сетки 20 м — 250 ячеек, каждая рисуется в несколько
экранных точек, так что линии видно без приближения. Север сверху, масштабная
линейка в километре.

Запуск из корня репозитория:
    python -m experiments.openlin_review_tiles --site anabar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.openlin_zone_lines import COMP, M, OUT

WIN_KM = 5.0                 # сторона окна
MAX_WINDOWS = 40             # предел числа окон
COVER_TARGET = 0.85          # докуда набирать окна по доле ячеек линий
N_CONTROL = 6                # контрольных окон без линий
OUT_DIR = OUT.parent / "линеаменты_проверка"   # прямо в outputs/, а не в outputs/steps/


def pick_windows(mask, n_cell: int, max_n: int, target: float):
    """Окна, жадно покрывающие больше всего ячеек линий."""
    from scipy.ndimage import uniform_filter

    total = int(mask.sum())
    taken = np.zeros_like(mask)
    half = n_cell // 2
    out = []
    work = mask.copy()
    while len(out) < max_n and taken.sum() < target * total:
        dens = uniform_filter(work.astype("float32"), n_cell)
        dens[:half, :] = 0; dens[-half:, :] = 0
        dens[:, :half] = 0; dens[:, -half:] = 0
        r, c = np.unravel_index(int(np.argmax(dens)), dens.shape)
        if dens[r, c] <= 0:
            break
        sl = (slice(r - half, r + half), slice(c - half, c + half))
        taken[sl] |= mask[sl]
        work[sl] = False
        out.append((int(r), int(c), int(mask[sl].sum())))
    return out


def pick_control(zone, lines_any, n_cell: int, n: int, rng):
    """Окна в зоне согласия, где линий нет вовсе."""
    from scipy.ndimage import uniform_filter

    half = n_cell // 2
    z = uniform_filter(zone.astype("float32"), n_cell)
    l = uniform_filter(lines_any.astype("float32"), n_cell)
    ok = (z > 0.3) & (l == 0)
    ok[:half, :] = False; ok[-half:, :] = False
    ok[:, :half] = False; ok[:, -half:] = False
    idx = np.flatnonzero(ok.ravel())
    if not len(idx):
        return []
    pick = rng.choice(idx, min(n * 8, len(idx)), replace=False)
    out, taken = [], []
    for i in pick:
        r, c = np.unravel_index(int(i), ok.shape)
        if any(abs(r - rr) < n_cell and abs(c - cc) < n_cell for rr, cc in taken):
            continue
        taken.append((r, c)); out.append((int(r), int(c), 0))
        if len(out) == n:
            break
    return out


def draw_pair(img, C, A, zone, rc, n_cell: int, res_m: float, path: Path,
              title: str, control: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, c = rc
    half = n_cell // 2
    sl = (slice(r - half, r + half), slice(c - half, c + half))
    base = img[sl]
    fig, ax = plt.subplots(1, 2, figsize=(19, 10), facecolor="#fcfcfb")
    for k in (0, 1):
        ax[k].imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[k].set_xticks([]); ax[k].set_yticks([])
    # разметка в одну ячейку и без утолщения: толстая полоса закрывает сам
    # объект, ради которого сделан снимок
    ov = np.zeros(base.shape + (4,), "float32")
    # в контрольном окне подтверждённых линий нет по построению, а жёлтые —
    # найденные, но не подтверждённые — показываются: по ним видно, где пропуск,
    # в детекторе (ничего нет) или в подтверждении (жёлтая линия есть)
    ov[A[sl] & ~C[sl]] = (0.95, 0.60, 0.0, 0.9)
    if not control:
        ov[C[sl]] = (0.85, 0.05, 0.05, 0.95)
    ax[1].imshow(ov)
    # масштабная линейка в 1 км
    px = 1000.0 / res_m
    for k in (0, 1):
        ax[k].plot([10, 10 + px], [base.shape[0] - 10] * 2, color="#111", lw=3)
        ax[k].text(10, base.shape[0] - 16, "1 км", color="#111", fontsize=12)
    ax[0].set_title("без разметки — смотрите сами", loc="left", fontsize=14)
    ax[1].set_title("с разметкой: красные — подтверждены двумя съёмками, "
                    "жёлтые — одной" if not control
                    else "подтверждённых линий нет; жёлтые — найдены, но не подтверждены",
                    loc="left", fontsize=14)
    fig.suptitle(title, x=0.01, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--набор", dest="set_name", default="весь",
                    choices=("весь", "зоны"))
    ap.add_argument("--сцены", dest="scenes", default="12", choices=("12", "все"),
                    help="набор на прежних 12 сценах или на всех 55 (своя папка)")
    ap.add_argument("--цепочки", dest="chains", action="store_true",
                    help="набор «цепочки» (мостики через кулисы убраны), своя папка")
    ap.add_argument("--только-контроль", dest="only_ctrl", action="store_true",
                    help="дорисовать контрольные окна, не трогая остальные")
    ap.add_argument("--контролей", dest="n_ctrl", type=int, default=N_CONTROL)
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    from experiments.openlin_shade import _stretch

    t0 = time.time()
    sfx = "_все" if a.scenes == "все" else ""
    out_dir = OUT_DIR.with_name(OUT_DIR.name + sfx)
    w = dict(np.load(COMP / f"winter_{a.site}_20m_лист{sfx}.npz"))
    res_m = float(w["res_m"])
    img = w["shade"].astype("float32")
    valid = np.isfinite(img)
    if a.chains:
        out_dir = OUT_DIR.with_name(OUT_DIR.name + "_цепочки")
    d = np.load(M / f"openlin_zone_lines_{a.site}_v3{'_цеп' if a.chains else sfx}.npz")
    zone = d["zone"]
    # по умолчанию показывается ПОЛНЫЙ набор (отбор только по подтверждению
    # двумя съёмками); --набор зоны оставляет прежний, отфильтрованный зоной
    if a.set_name == "весь" and "C_all" in d:
        A, C = d["A_all"], d["C_all"]
    else:
        A, C = d["A"], d["C"]
    shape = C.shape
    img = img[:shape[0], :shape[1]]
    valid = valid[:shape[0], :shape[1]]
    base = _stretch(np.where(valid, img, np.nan))
    n_cell = int(WIN_KM * 1000 / res_m)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("контроль_*.png" if a.only_ctrl else "*.png"):
        old.unlink()

    wins = [] if a.only_ctrl else pick_windows(C, n_cell, MAX_WINDOWS, COVER_TARGET)
    rng = np.random.default_rng(3)
    # Контроль — окно в зоне согласия без ПОДТВЕРЖДЁННЫХ линий. Окон совсем без
    # линий (с учётом неподтверждённых) на полном наборе не осталось ни одного:
    # детектор находит что-нибудь в любом квадрате 5 x 5 км.
    ctrl = pick_control(zone, C, n_cell, a.n_ctrl, rng)
    log(f"окон с линиями: {len(wins)}, контрольных: {len(ctrl)} "
        f"({time.time() - t0:.0f} с)")

    x0, y_top, epsg = float(w["x0"]), float(w["y_top"]), int(w["epsg"])

    def place(r, c):
        from pyproj import Transformer
        tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x0 + c * res_m, y_top - r * res_m)
        return f"{lat:.3f}° с.ш., {lon:.3f}° в.д."

    for i, (r, c, n) in enumerate(wins, 1):
        km = n * res_m * res_m / 1e6
        p = out_dir / f"{i:02d}_окно_{r}_{c}.png"
        draw_pair(base, C, A, zone, (r, c), n_cell, res_m, p,
                  f"Окно {i} из {len(wins)} · {WIN_KM:.0f} × {WIN_KM:.0f} км · "
                  f"{place(r, c)} · север сверху")
        log(f"  {p.name}: ячеек линий {n} ({km:.2f} км2)")
    for i, (r, c, _n) in enumerate(ctrl, 1):
        p = out_dir / f"контроль_{i}_окно_{r}_{c}.png"
        draw_pair(base, C, A, zone, (r, c), n_cell, res_m, p,
                  f"Контроль {i} · в зоне согласия, подтверждённых линий нет · "
                  f"{place(r, c)} · север сверху", control=True)
        log(f"  {p.name}")
    if a.only_ctrl:
        # обзор с рамками окон не перерисовывается: окон в этом режиме нет
        log(f"\nготово: {out_dir} ({time.time() - t0:.0f} с)")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_dilation

    fig, ax = plt.subplots(figsize=(13, 12), facecolor="#fcfcfb")
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    ov = np.zeros(shape + (4,), "float32")
    ov[binary_dilation(C, np.ones((5, 5), bool))] = (0.78, 0.09, 0.09, 1.0)
    ax.imshow(ov)
    half = n_cell // 2
    for i, (r, c, _n) in enumerate(wins, 1):
        ax.add_patch(plt.Rectangle((c - half, r - half), n_cell, n_cell,
                                   fill=False, ec="#2a78d6", lw=1.6))
        ax.text(c - half, r - half - 25, str(i), color="#2a78d6", fontsize=13,
                weight="bold")
    for i, (r, c, _n) in enumerate(ctrl, 1):
        ax.add_patch(plt.Rectangle((c - half, r - half), n_cell, n_cell,
                                   fill=False, ec="#1d7a3c", lw=1.6, ls="--"))
        ax.text(c - half, r - half - 25, f"к{i}", color="#1d7a3c", fontsize=13,
                weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Весь лист: {int(C.sum() * res_m * res_m / 1e6)} км2 линий, "
                 f"синие рамки — окна с линиями, зелёные пунктирные — контроль",
                 loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "00_обзор.png", dpi=100, facecolor=fig.get_facecolor())
    log(f"\nготово: {out_dir} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
