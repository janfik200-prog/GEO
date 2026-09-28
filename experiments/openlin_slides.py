"""Схемы шагов метода для презентации: на реальных данных, в палитре слайдов.

Пять картинок, по одной на шаг. Везде, где можно, показан настоящий фрагмент
Анабара, а не рисованная схема: аудитории нужно увидеть, что оператор делает с
их снимком, а не что он делает в принципе.

Палитра взята из самой презентации (Cambria/Calibri, тёмно-синий текст,
терракотовый акцент, тёплый фон карточек), чтобы картинки не выбивались из
оформления.

Запуск из корня репозитория:
    python -m experiments.openlin_slides
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from experiments.openlin_anfis import TOL_M
from experiments.openlin_domains import (N_ANGLES, OFFSET_M, WIN_LONG_M,
                                         WIN_SHORT_M, _kernel, _smooth,
                                         edges_from_components)
from experiments.openlin_gate import components
from experiments.openlin_lines import DENSITY, _branches, _link_gaps, _ridge_nms
from experiments.openlin_neigh import _one_channel
from experiments.openlin_unet import build_stack, load_site, CHANNELS
from experiments.openlin_unsup import OPTIC, RADAR

OUT = Path("outputs/slides")
M = Path("outputs/metrics")

# --- палитра презентации ---------------------------------------------------
INK, INK2 = "#1B2A32", "#6E7F88"
ACC = "#C15B33"          # терракотовый акцент
SLATE = "#3E6D85"        # второй акцент, холодный
GREY = "#9AA5AB"
CARD, SURF = "#F4EFEA", "#FBFAF8"
EDGE = "#D8D2C8"

#: Последовательная шкала для полей: тёплый светлый -> тёмный сине-серый.
#: Однотонная по построению — величина читается светлотой, а не сменой цвета.
RAMP = LinearSegmentedColormap.from_list(
    "deck", ["#F7F3EE", "#D9DDDA", "#A9B6BB", "#6E8794", "#3E6D85", "#1B2A32"])

DENS = "предельно"
CROP = (250, 430, 300, 520)      # строки и столбцы фрагмента для врезок


def style(ax, title=None, sub=None):
    ax.set_facecolor(SURF)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    if title:
        ax.set_title(title, fontsize=12, color=INK, fontweight="bold", pad=8)
    if sub:
        ax.set_xlabel(sub, fontsize=9.5, color=INK2)


def robust(a, lo=2, hi=98):
    v = a[np.isfinite(a)]
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / ((p2 - p1) or 1.0), 0, 1)


def crop(a):
    r0, r1, c0, c1 = CROP
    return a[r0:r1, c0:c1]


def fat(mask, k=2):
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, np.ones((k, k), dtype=bool))


# ===========================================================================
# Шаг 1
# ===========================================================================

def step1(z, valid, res_m):
    """Одна ячейка описывается устройством окрестности, а не своим значением."""
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    f = _one_channel(gray, valid, 500.0 / res_m)       # масштаб 500 м

    fig = plt.figure(figsize=(13.2, 3.5), facecolor=SURF)
    gs = fig.add_gridspec(1, 4, wspace=0.12, left=0.015, right=0.985,
                          top=0.84, bottom=0.10)

    ax = fig.add_subplot(gs[0])
    ax.imshow(crop(robust(gray)), cmap="gray")
    style(ax, "снимок как есть", "яркость в каждой ячейке")

    ax = fig.add_subplot(gs[1])
    ax.imshow(crop(f["coh"]), cmap=RAMP)
    style(ax, "вытянутость", "0 — хаос, 1 — строгая полосатость")

    ax = fig.add_subplot(gs[2])
    ax.imshow(crop(robust(f["rdg"])), cmap=RAMP)
    style(ax, "выступ или впадина", "светлый гребень / тёмная борозда")

    # схема эллипсов: что именно меряется
    ax = fig.add_subplot(gs[3])
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.25)
    for i, (w, h, ang, cap) in enumerate([
            (0.12, 0.12, 0, "плато\nперепадов нет"),
            (0.62, 0.10, 35, "уступ, борозда\nвытянутость ≈ 1"),
            (0.42, 0.40, 0, "пёстрая текстура\nвытянутость ≈ 0")]):
        ax.add_patch(Ellipse((0.5 + i, 0.72), w, h, angle=ang,
                             facecolor=ACC if i == 1 else GREY, alpha=0.85,
                             edgecolor=INK, lw=0.8))
        ax.text(0.5 + i, 0.30, cap, ha="center", va="center", fontsize=9,
                color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.set_title("что меряет структурный тензор", fontsize=12, color=INK,
                 fontweight="bold", pad=8)
    ax.set_xlabel("форма «облака направлений» в окрестности", fontsize=9.5,
                  color=INK2)

    fig.savefig(OUT / "step1.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("step1 готов")


# ===========================================================================
# Шаг 2
# ===========================================================================

def step2(site):
    """Сжатие 160 -> 8 двумя способами и почему берутся оба."""
    from sklearn.decomposition import PCA

    z, valid, res_m, comp = components(site, OPTIC + RADAR)
    # доля объяснённого разброса — для этого нужен PCA без обрезки
    x, _y, _v, use = build_stack(z, CHANNELS)
    from experiments.openlin_neigh import neigh_stack
    nx, _n = neigh_stack(x, use, valid, res_m, base=use)
    stack = np.concatenate([x, nx])
    rows = stack[:, valid].T.astype("float32")
    sel = np.random.default_rng(42).choice(len(rows), 30000, replace=False)
    full = PCA(n_components=30, random_state=42).fit(rows[sel])
    ev = full.explained_variance_ratio_

    mae = np.load(M / f"openlin_mae_{site}_{DENS}.npz")["confirmed"]
    pca_l = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")["confirmed"]

    fig = plt.figure(figsize=(13.2, 3.6), facecolor=SURF)
    gs = fig.add_gridspec(1, 4, wspace=0.22, left=0.05, right=0.985,
                          top=0.84, bottom=0.16)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURF)
    k = np.arange(1, 21)
    ax.bar(k, 100 * ev[:20], color=[ACC if i < 8 else GREY for i in range(20)],
           linewidth=0)
    ax.axvline(8.5, color=INK2, lw=1, ls="--")
    ax.text(8.8, max(100 * ev[:20]) * 0.8, "берём 8", fontsize=9.5, color=INK2)
    ax.set_xlabel("номер компоненты", fontsize=9.5, color=INK2)
    ax.set_ylabel("доля разброса, %", fontsize=9.5, color=INK2)
    ax.set_title("8 компонент хватает", fontsize=12, color=INK,
                 fontweight="bold", pad=8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.text(0.5, 0.92, f"первые 8 дают {100 * ev[:8].sum():.0f} %",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=INK)

    for j in (0, 1):
        ax = fig.add_subplot(gs[j + 1])
        ax.imshow(crop(comp[j]), cmap=RAMP)
        style(ax, f"компонента {j + 1}",
              "линейная смесь 160 признаков" if j == 0 else "")

    ax = fig.add_subplot(gs[3])
    ax.set_facecolor(SURF)
    # Пересечение меряется по ДЛИНЕ с допуском 250 м, а не по связным кускам:
    # у двух списков куски режутся в разных местах, и счёт компонент
    # пересечения занижал бы совпадение в разы.
    def km(m):
        return float(m.sum()) * res_m / 1000.0

    vals = [km(pca_l), km(mae), km(pca_l | mae)]
    ax.bar([0, 1, 2], vals, 0.62, color=[SLATE, GREY, ACC], linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 18, f"{v:.0f}", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["главные\nкомпоненты", "автоэнкодер", "вместе"],
                       fontsize=9.5, color=INK2)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("длина линий, км", fontsize=9.5, color=INK2)
    ax.set_title("способы находят разное", fontsize=12, color=INK,
                 fontweight="bold", pad=8)
    ax.set_xlabel("по положению совпадает 53–60 % длины", fontsize=9,
                  color=ACC)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)

    fig.savefig(OUT / "step2.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("step2 готов")
    return z, valid, res_m, comp


# ===========================================================================
# Шаг 3
# ===========================================================================

def step3(z, valid, res_m, comp):
    """Схема двух окон, перебор направлений и готовое поле силы."""
    strength, ang = edges_from_components(comp, valid, res_m)
    strength = np.nan_to_num(strength, nan=0.0)

    fig = plt.figure(figsize=(13.2, 4), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.0], wspace=0.2,
                          left=0.02, right=0.98, top=0.86, bottom=0.12)

    # --- схема окон --------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(SURF)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.05, 1.02)
    ax.add_patch(Rectangle((-1.25, 0.10), 2.5, 0.52, facecolor=CARD,
                           edgecolor=ACC, lw=1.4))
    ax.add_patch(Rectangle((-1.25, -0.62), 2.5, 0.52, facecolor=CARD,
                           edgecolor=ACC, lw=1.4))
    ax.plot([-1.35, 1.35], [0, 0], color=INK, lw=2.4)
    ax.text(0, 0.36, "окно A", ha="center", fontsize=11, color=INK)
    ax.text(0, -0.36, "окно B", ha="center", fontsize=11, color=INK)
    ax.text(-1.2, -0.86, "воображаемая линия под углом θ", ha="left",
            va="center", fontsize=9.5, color=INK2)
    ax.annotate("", xy=(-1.25, 0.78), xytext=(1.25, 0.78),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(0, 0.84, "2 км вдоль линии", ha="center", fontsize=9.5, color=INK2)
    ax.annotate("", xy=(-1.33, 0.10), xytext=(-1.33, 0.62),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(-1.30, 0.36, "200 м", ha="left", fontsize=9, color=INK2)
    ax.annotate("", xy=(1.33, 0.0), xytext=(1.33, 0.36),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(1.29, 0.19, "300 м", ha="right", fontsize=9, color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.set_title("два окна по разные стороны линии", fontsize=12, color=INK,
                 fontweight="bold", pad=8)
    ax.set_xlabel("сравниваются средние портреты участков, а не яркость",
                  fontsize=9.5, color=INK2)

    # --- отклик по направлениям в одной ячейке -----------------------------
    r0, r1, c0, c1 = CROP
    sub = strength[r0:r1, c0:c1]
    rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
    rr += r0
    cc += c0
    resp = []
    for i in range(N_ANGLES):
        th = np.pi * i / N_ANGLES
        kp = _kernel(WIN_LONG_M / res_m, WIN_SHORT_M / res_m, th, OFFSET_M / res_m)
        km = _kernel(WIN_LONG_M / res_m, WIN_SHORT_M / res_m, th, -OFFSET_M / res_m)
        d = _smooth(comp[:, rr - 70:rr + 70, cc - 70:cc + 70],
                    valid[rr - 70:rr + 70, cc - 70:cc + 70], kp) - \
            _smooth(comp[:, rr - 70:rr + 70, cc - 70:cc + 70],
                    valid[rr - 70:rr + 70, cc - 70:cc + 70], km)
        resp.append(float(np.sqrt((d[:, 70, 70] ** 2).sum())))
    resp = np.asarray(resp)
    deg = np.arange(N_ANGLES) * 180 / N_ANGLES

    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(SURF)
    best = int(resp.argmax())
    ax.bar(deg, resp, width=11, color=[ACC if i == best else GREY
                                       for i in range(N_ANGLES)], linewidth=0)
    ax.set_xlabel("направление линии, градусы", fontsize=9.5, color=INK2)
    ax.set_ylabel("сила границы, сигмы", fontsize=9.5, color=INK2)
    ax.set_title("перебор 12 направлений", fontsize=12, color=INK,
                 fontweight="bold", pad=8)
    ax.text(deg[best], resp[best] * 1.02, f"максимум\n{deg[best]:.0f}°",
            ha="center", fontsize=9, color=ACC)
    ax.set_ylim(0, resp.max() * 1.25)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)

    ax = fig.add_subplot(gs[2])
    ax.imshow(crop(strength), cmap=RAMP)
    style(ax, "готовое поле силы границы",
          "тёмное — участки по сторонам различаются")

    fig.savefig(OUT / "step3.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("step3 готов")
    return strength


# ===========================================================================
# Шаг 4
# ===========================================================================

def step4(strength, valid, res_m):
    """Цепочка трассировки на одном и том же фрагменте."""
    from scipy.ndimage import label
    from skimage.morphology import skeletonize

    sigma_m, hi_a, lo_a, min_len, min_str = DENSITY[DENS]
    s, ridge = _ridge_nms(strength, valid, sigma_m / res_m)
    v = s[valid]
    hi, lo = np.quantile(v, 1 - hi_a), np.quantile(v, 1 - lo_a)
    cand = ridge & (s >= lo) & valid
    lab, n = label(cand, structure=np.ones((3, 3), dtype=int))
    seeded = np.zeros(n + 1, dtype=bool)
    seeded[np.unique(lab[cand & (s >= hi)])] = True
    seeded[0] = False
    hyst = seeded[lab]
    sk = skeletonize(hyst)
    parts = _branches(hyst, res_m)
    links = _link_gaps(parts, hyst.shape, res_m) & valid
    final = np.zeros_like(valid)
    for pts in _branches(hyst | links, res_m):
        if len(pts) < 3:
            continue
        c = (pts - pts.mean(0)).astype("float64")
        _u, _sv, vt = np.linalg.svd(c, full_matrices=False)
        proj = c @ vt[0]
        length = float(proj.max() - proj.min()) * res_m
        if length >= min_len:
            final[pts[:, 0], pts[:, 1]] = True

    panels = [(crop(s), "1. сглаженное поле", "убран пиксельный шум", False),
              (crop(ridge), "2. гребень", "подавление немаксимумов поперёк", True),
              (crop(hyst), "3. гистерезис", "два порога: ядро и продолжение", True),
              (crop(sk), "4. скелет", "линия толщиной в ячейку", True),
              (crop(final), "5. отсев и сшивка", "короткое убрано, разрывы сшиты", True)]

    fig = plt.figure(figsize=(13.2, 3.1), facecolor=SURF)
    gs = fig.add_gridspec(1, 5, wspace=0.10, left=0.01, right=0.99,
                          top=0.82, bottom=0.12)
    for i, (img, t, sub, is_mask) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if is_mask:
            ax.imshow(crop(robust(strength)), cmap="gray", alpha=0.35)
            ax.imshow(np.ma.masked_where(~fat(img), np.zeros_like(img, float)),
                      cmap=ListedColormap([ACC]))
        else:
            ax.imshow(img, cmap=RAMP)
        style(ax, t, sub)
    fig.savefig(OUT / "step4.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("step4 готов")


# ===========================================================================
# Шаг 5
# ===========================================================================

def step5(site, valid, res_m):
    """Подтверждение двумя приборами и случайный уровень."""
    d = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    lo, lr, la = d["lines_0"], d["lines_1"], d["lines_2"]
    keep = d["confirmed"]
    z = load_site(site)
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))

    fig = plt.figure(figsize=(13.2, 4.1), facecolor=SURF)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.16,
                          left=0.01, right=0.985, top=0.84, bottom=0.17)

    for i, (m, t, sub, col) in enumerate([
            (lo, "видит оптика", "яркость и NDVI", SLATE),
            (lr, "видит радар", "VV, VH, когерентность", ACC),
            (keep, "подтверждено обоими", "вдоль длины и по направлению", INK)]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(crop(gray), cmap="gray", alpha=0.55)
        ax.imshow(np.ma.masked_where(~fat(crop(m)), np.zeros(crop(m).shape)),
                  cmap=ListedColormap([col]))
        style(ax, t, sub)

    ax = fig.add_subplot(gs[3])
    ax.set_facecolor(SURF)
    names = ["прослежено", "подтвердил\nрадар", "подтвердила\nоптика",
             "подтвердили\nоба", "прошло бы\nслучайно"]
    vals = [613, 393, 222, 153, 0.9]
    cols = [GREY, ACC, SLATE, INK, EDGE]
    ax.bar(range(5), vals, 0.66, color=cols, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15 if v > 2 else 1.4,
                f"{v:.0f}" if v >= 2 else f"{v:.1f}",
                ha="center", fontsize=10, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1500)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=9.5, color=INK2)
    ax.set_ylabel("линий, шкала логарифмическая", fontsize=9, color=INK2)
    ax.set_title("отбор и его цена", fontsize=12, color=INK,
                 fontweight="bold", pad=8)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)
    ax.set_xlabel("подтвердили оба — 153 против 0.9 случайных: чистота ×174",
                  fontsize=9, color=ACC)

    fig.savefig(OUT / "step5.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    print("step5 готов")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    site = "anabar"
    z = load_site(site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])

    step1(z, valid, res_m)
    z2, valid2, res2, comp = step2(site)
    strength = step3(z2, valid2, res2, comp)
    step4(strength, valid2, res2)
    step5(site, valid, res_m)
    print(f"\nвсе схемы в {OUT}")


if __name__ == "__main__":
    main()
