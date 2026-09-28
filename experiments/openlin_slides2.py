"""Крупные схемы для презентации: одна картинка — один слайд.

Отличие от :mod:`experiments.openlin_slides`: там пять сводных полос, по полосе
на шаг, и каждая панель на слайде выходит мелкой. Здесь те же данные разложены
на отдельные картинки под формат «картинка во всю ширину слайда», подписи
крупные, на панель приходится вчетверо больше площади.

Всё, что можно, показано на настоящем фрагменте Анабара.

Запуск из корня репозитория:
    python -m experiments.openlin_slides2
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle

from experiments.openlin_domains import (N_ANGLES, OFFSET_M, WIN_LONG_M,
                                         WIN_SHORT_M, _kernel, _smooth,
                                         edges_from_components)
from experiments.openlin_gate import components
from experiments.openlin_lines import DENSITY, _branches, _link_gaps, _ridge_nms
from experiments.openlin_neigh import _one_channel
from experiments.openlin_unet import CHANNELS, build_stack, load_site
from experiments.openlin_unsup import OPTIC, RADAR

OUT = Path("outputs/slides")
M = Path("outputs/metrics")

INK, INK2 = "#1B2A32", "#6E7F88"
ACC = "#C15B33"
SLATE = "#3E6D85"
GREY = "#9AA5AB"
CARD, SURF = "#F4EFEA", "#FBFAF8"
EDGE = "#D8D2C8"

RAMP = LinearSegmentedColormap.from_list(
    "deck", ["#F7F3EE", "#D9DDDA", "#A9B6BB", "#6E8794", "#3E6D85", "#1B2A32"])

DENS = "предельно"
CROP = (250, 430, 300, 520)

#: Размеры шрифтов. Картинка идёт на слайд во всю ширину, то есть уменьшается
#: примерно в 1.05 раза, — значит подпись в 18 pt читается из зала как 17 pt.
T_SIZE, S_SIZE, N_SIZE = 18, 13.5, 13


def style(ax, title=None, sub=None):
    ax.set_facecolor(SURF)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    if title:
        ax.set_title(title, fontsize=T_SIZE, color=INK, fontweight="bold",
                     pad=10)
    if sub:
        ax.set_xlabel(sub, fontsize=S_SIZE, color=INK2)


def bare(ax):
    """Оси графика: без верхней и правой рамки, подписи крупные."""
    ax.set_facecolor(SURF)
    ax.tick_params(colors=INK2, labelsize=N_SIZE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(EDGE)


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


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor=SURF)
    plt.close(fig)
    print(f"{name} готов")


# ===========================================================================
# Шаг 1
# ===========================================================================

def p_features(z, valid, res_m):
    """Одна и та же местность: яркость и две величины окрестности.

    Величины окрестности кладутся полупрозрачным слоем поверх того же снимка:
    так видно, каким местам снимка они отвечают, а не просто «две новые карты».
    """
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    f = _one_channel(gray, valid, 500.0 / res_m)
    base = crop(robust(gray))

    fig = plt.figure(figsize=(13.2, 5.5), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, wspace=0.08, left=0.012, right=0.988,
                          top=0.87, bottom=0.14)
    panels = [(None, "снимок как есть", "яркость в каждой ячейке"),
              (f["coh"], "вытянутость текстуры",
               "тёмное — перепады выстроены в одну сторону"),
              (robust(f["rdg"]), "выступ или впадина",
               "светлый гребень / тёмная борозда")]
    for i, (over, t, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(base, cmap="gray")
        if over is not None:
            ax.imshow(crop(over), cmap=RAMP, alpha=0.58)
        style(ax, t, sub)
    fig.text(0.5, 0.02, "две правые панели — тот же снимок, поверх него "
             "полупрозрачно наложена посчитанная величина", ha="center",
             fontsize=S_SIZE, color=INK2)
    save(fig, "p_features.png")


#: Размер окошка-примера, ячеек. 60 = 6 км: втрое шире окрестности 2 км,
#: то есть в кадр попадает не только сам объект, но и его окружение.
EX_PX = 60


def _example(coh, eng, valid, kind):
    """Место на площадке, где окрестность выглядит нужным образом.

    Величины усредняются по окну размером с будущее окошко: иначе выбирается
    одиночная ячейка-выброс, а вокруг неё оказывается обычная текстура.
    """
    from scipy.ndimage import uniform_filter

    k = EX_PX
    c = uniform_filter(np.nan_to_num(coh), k)
    e = uniform_filter(np.nan_to_num(eng), k)
    ok = uniform_filter(valid.astype("float32"), k) > 0.999
    # край кадра отрезаем: окошко должно целиком лежать внутри растра
    ok[:k, :] = ok[-k:, :] = ok[:, :k] = ok[:, -k:] = False

    lo_e, hi_e = np.quantile(e[ok], [0.10, 0.75])
    if kind == "плато":
        score = np.where(ok & (e <= lo_e), -e, -np.inf)
    elif kind == "уступ":
        score = np.where(ok & (e >= np.quantile(e[ok], 0.5)), c, -np.inf)
    else:                                   # пёстрая текстура
        score = np.where(ok & (e >= hi_e), -c, -np.inf)
    r, cc = np.unravel_index(int(np.argmax(score)), score.shape)
    h = k // 2
    return (slice(r - h, r + h), slice(cc - h, cc + h))


def p_ellipse(z, valid, res_m):
    """Форма «облака направлений» и по фрагменту площадки на каждый случай."""
    gray = np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"])))
    f = _one_channel(gray, valid, 500.0 / res_m)
    g = robust(gray)

    cases = [("плато", 0.16, 0.16, 0, "плато", "перепадов нет"),
             ("уступ", 0.80, 0.13, 35, "уступ или борозда",
              "одно направление"),
             ("пёстрая", 0.52, 0.50, 0, "пёстрая текстура",
              "все направления")]

    fig = plt.figure(figsize=(12.6, 5.4), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.52], hspace=0.16,
                          wspace=0.10, left=0.015, right=0.985, top=0.90,
                          bottom=0.09)
    for i, (kind, w, h, angle, title, sub) in enumerate(cases):
        box = _example(f["coh"], f["eng"], valid, kind)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(g[box], cmap="gray")
        style(ax, title, sub)

        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(SURF)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.42, 0.42)
        ax.add_patch(Ellipse((0, 0), w, h, angle=angle,
                             facecolor=ACC if i == 1 else GREY, alpha=0.85,
                             edgecolor=INK, lw=1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(EDGE)
    fig.text(0.5, 0.015, "внизу — эллипс направлений для фрагмента над ним "
             "(6 × 6 км, реальные участки листа)", ha="center",
             fontsize=S_SIZE, color=INK2)
    save(fig, "p_ellipse.png")


# ===========================================================================
# Шаг 2
# ===========================================================================

def p_scree(site):
    """Сколько разброса объясняют первые компоненты."""
    from sklearn.decomposition import PCA

    from experiments.openlin_neigh import neigh_stack

    z = load_site(site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    x, _y, _v, use = build_stack(z, CHANNELS)
    nx, _n = neigh_stack(x, use, valid, res_m, base=use)
    stack = np.concatenate([x, nx])
    rows = stack[:, valid].T.astype("float32")
    sel = np.random.default_rng(42).choice(len(rows), 30000, replace=False)
    ev = PCA(n_components=30, random_state=42).fit(
        rows[sel]).explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11.5, 5.0), facecolor=SURF)
    bare(ax)
    k = np.arange(1, 21)
    ax.bar(k, 100 * ev[:20], color=[ACC if i < 8 else GREY for i in range(20)],
           linewidth=0)
    ax.axvline(8.5, color=INK2, lw=1.2, ls="--")
    ax.text(8.9, max(100 * ev[:20]) * 0.78,
            f"первые 8 дают {100 * ev[:8].sum():.0f} %", fontsize=S_SIZE,
            color=INK)
    ax.set_xlabel("номер компоненты", fontsize=S_SIZE, color=INK2)
    ax.set_ylabel("доля разброса, %", fontsize=S_SIZE, color=INK2)
    fig.tight_layout()
    save(fig, "p_scree.png")
    return ev


def p_components(comp):
    """Первые две компоненты как карты."""
    fig = plt.figure(figsize=(11.0, 5.3), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.015, right=0.985,
                          top=0.88, bottom=0.09)
    for j in (0, 1):
        ax = fig.add_subplot(gs[j])
        ax.imshow(crop(comp[j]), cmap=RAMP)
        style(ax, f"компонента {j + 1}", "линейная смесь 160 признаков")
    save(fig, "p_components.png")


def p_two_ways(site, res_m):
    """Два способа сжатия дают разные линии, поэтому берутся оба."""
    mae = np.load(M / f"openlin_mae_{site}_{DENS}.npz")["confirmed"]
    pca_l = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")["confirmed"]

    def km(m):
        return float(m.sum()) * res_m / 1000.0

    vals = [km(pca_l), km(mae), km(pca_l | mae)]
    fig, ax = plt.subplots(figsize=(10.5, 5.0), facecolor=SURF)
    bare(ax)
    ax.bar([0, 1, 2], vals, 0.6, color=[SLATE, GREY, ACC], linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 22, f"{v:.0f} км", ha="center", fontsize=T_SIZE,
                color=INK)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["главные компоненты", "автоэнкодер", "вместе"],
                       fontsize=S_SIZE, color=INK2)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("длина подтверждённых линий, км", fontsize=S_SIZE,
                  color=INK2)
    ax.set_xlabel("по положению совпадает 53–60 % длины", fontsize=S_SIZE,
                  color=ACC)
    fig.tight_layout()
    save(fig, "p_two_ways.png")


# ===========================================================================
# Шаг 3
# ===========================================================================

def p_windows():
    """Геометрия двух окон по разные стороны воображаемой линии."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.05, 1.05)
    ax.add_patch(Rectangle((-1.25, 0.10), 2.5, 0.52, facecolor=CARD,
                           edgecolor=ACC, lw=1.8))
    ax.add_patch(Rectangle((-1.25, -0.62), 2.5, 0.52, facecolor=CARD,
                           edgecolor=ACC, lw=1.8))
    ax.plot([-1.4, 1.4], [0, 0], color=INK, lw=3.0)
    ax.text(0, 0.36, "окно A", ha="center", va="center", fontsize=T_SIZE,
            color=INK)
    ax.text(0, -0.36, "окно B", ha="center", va="center", fontsize=T_SIZE,
            color=INK)
    ax.text(-1.35, -0.88, "воображаемая линия под углом θ", ha="left",
            va="center", fontsize=S_SIZE, color=INK2)
    ax.annotate("", xy=(-1.25, 0.80), xytext=(1.25, 0.80),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(0, 0.87, "2 км вдоль линии", ha="center", fontsize=S_SIZE,
            color=INK2)
    ax.annotate("", xy=(-1.34, 0.10), xytext=(-1.34, 0.62),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(-1.30, 0.36, "200 м", ha="left", va="center", fontsize=S_SIZE,
            color=INK2)
    ax.annotate("", xy=(1.34, 0.0), xytext=(1.34, 0.36),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(1.30, 0.19, "300 м", ha="right", va="center", fontsize=S_SIZE,
            color=INK2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
    fig.tight_layout()
    save(fig, "p_windows.png")


def p_angles(valid, res_m, comp, strength):
    """Отклик одной ячейки на 12 направлений."""
    r0, r1, c0, c1 = CROP
    sub = strength[r0:r1, c0:c1]
    rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
    rr += r0
    cc += c0
    resp = []
    for i in range(N_ANGLES):
        th = np.pi * i / N_ANGLES
        box = (slice(rr - 70, rr + 70), slice(cc - 70, cc + 70))
        kp = _kernel(WIN_LONG_M / res_m, WIN_SHORT_M / res_m, th,
                     OFFSET_M / res_m)
        km_ = _kernel(WIN_LONG_M / res_m, WIN_SHORT_M / res_m, th,
                      -OFFSET_M / res_m)
        d = (_smooth(comp[:, box[0], box[1]], valid[box], kp)
             - _smooth(comp[:, box[0], box[1]], valid[box], km_))
        resp.append(float(np.sqrt((d[:, 70, 70] ** 2).sum())))
    resp = np.asarray(resp)
    deg = np.arange(N_ANGLES) * 180 / N_ANGLES
    best = int(resp.argmax())

    fig, ax = plt.subplots(figsize=(11.0, 5.0), facecolor=SURF)
    bare(ax)
    ax.bar(deg, resp, width=11, color=[ACC if i == best else GREY
                                       for i in range(N_ANGLES)], linewidth=0)
    ax.text(deg[best], resp[best] * 1.03, f"максимум {deg[best]:.0f}°",
            ha="center", fontsize=T_SIZE, color=ACC)
    ax.set_xlabel("направление предполагаемой линии, градусы", fontsize=S_SIZE,
                  color=INK2)
    ax.set_ylabel("сила границы, сигмы", fontsize=S_SIZE, color=INK2)
    ax.set_ylim(0, resp.max() * 1.22)
    fig.tight_layout()
    save(fig, "p_angles.png")


def p_field(strength):
    """Готовое поле силы границы на том же фрагменте."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4), facecolor=SURF)
    im = ax.imshow(crop(strength), cmap=RAMP)
    style(ax, None, "тёмное — участки по разные стороны различаются сильнее")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("сигмы", fontsize=S_SIZE, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=N_SIZE)
    cb.outline.set_edgecolor(EDGE)
    fig.tight_layout()
    save(fig, "p_field.png")


# ===========================================================================
# Шаг 4
# ===========================================================================

def p_trace(strength, valid, res_m):
    """Цепочка трассировки, разложенная на две картинки по три панели."""
    from scipy.ndimage import label
    from skimage.morphology import skeletonize

    sigma_m, hi_a, lo_a, min_len, _min_str = DENSITY[DENS]
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
        if float(proj.max() - proj.min()) * res_m >= min_len:
            final[pts[:, 0], pts[:, 1]] = True

    groups = [("p_trace_a.png",
               [(s, "1. сглаженное поле", "убран пиксельный шум", False),
                (ridge, "2. гребень", "немаксимумы поперёк подавлены", True),
                (hyst, "3. два порога", "ядро линии и её продолжение", True)]),
              ("p_trace_b.png",
               [(sk, "4. скелет", "линия толщиной в одну ячейку", True),
                (final, "5. отсев и сшивка", "разрывы сшиты, короткое убрано",
                 True)])]
    for name, panels in groups:
        fig = plt.figure(figsize=(13.2 if len(panels) == 3 else 9.2, 5.3),
                         facecolor=SURF)
        gs = fig.add_gridspec(1, len(panels), wspace=0.08, left=0.012,
                              right=0.988, top=0.88, bottom=0.09)
        for i, (img, t, sub, is_mask) in enumerate(panels):
            ax = fig.add_subplot(gs[i])
            if is_mask:
                ax.imshow(crop(robust(strength)), cmap="gray", alpha=0.35)
                m = crop(img)
                ax.imshow(np.ma.masked_where(~fat(m), np.zeros(m.shape)),
                          cmap=ListedColormap([ACC]))
            else:
                ax.imshow(crop(img), cmap=RAMP)
            style(ax, t, sub)
        save(fig, name)


# ===========================================================================
# Шаг 5
# ===========================================================================

def p_confirm(site, res_m):
    """Что видит оптика, что радар и что остаётся после пересечения."""
    d = np.load(M / f"openlin_unsup_{site}_{DENS}.npz")
    lo, lr, keep = d["lines_0"], d["lines_1"], d["confirmed"]
    z = load_site(site)
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))

    fig = plt.figure(figsize=(13.2, 5.3), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, wspace=0.08, left=0.012, right=0.988,
                          top=0.88, bottom=0.09)
    for i, (m, t, sub, col) in enumerate([
            (lo, "видит оптика", "яркость и NDVI", SLATE),
            (lr, "видит радар", "VV, VH, когерентность", ACC),
            (keep, "подтверждено обоими", "вдоль длины и по направлению",
             INK)]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(crop(gray), cmap="gray", alpha=0.55)
        mm = crop(m)
        ax.imshow(np.ma.masked_where(~fat(mm), np.zeros(mm.shape)),
                  cmap=ListedColormap([col]))
        style(ax, t, sub)
    save(fig, "p_confirm.png")


def p_final_map(site="anabar"):
    """Итог: карта линеаментов по всей площадке и фрагмент крупным планом."""
    uni = np.load(M / f"openlin_union_{site}.npz")["confirmed"]
    z = load_site(site)
    res_m = float(z["res_m"])
    valid = np.isfinite(z["gray"])
    gray = robust(np.nan_to_num(z["gray"], nan=float(np.nanmedian(z["gray"]))))
    h, w = gray.shape
    r0, r1, c0, c1 = CROP

    fig = plt.figure(figsize=(12.6, 5.6), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.06,
                          left=0.012, right=0.988, top=0.93, bottom=0.10)

    ax = fig.add_subplot(gs[0])
    # вне кадра снимка — фон слайда, иначе пустые углы читаются как местность
    ax.imshow(np.ma.masked_where(~valid, gray), cmap="gray", alpha=0.85)
    ax.imshow(np.ma.masked_where(~fat(uni, 3), np.zeros_like(gray)),
              cmap=ListedColormap([ACC]))
    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, facecolor="none",
                           edgecolor=SLATE, lw=1.8))
    style(ax, None, f"{w * res_m / 1000:.0f} × {h * res_m / 1000:.0f} км, "
          "255 линий, 1010 км")

    ax = fig.add_subplot(gs[1])
    ax.imshow(crop(gray), cmap="gray", alpha=0.85)
    ax.imshow(np.ma.masked_where(~fat(crop(uni), 2),
                                 np.zeros(crop(gray).shape)),
              cmap=ListedColormap([ACC]))
    for s in ax.spines.values():
        s.set_color(SLATE)
        s.set_linewidth(1.8)
    style(ax, None, f"фрагмент в рамке: {(c1 - c0) * res_m / 1000:.0f} × "
          f"{(r1 - r0) * res_m / 1000:.0f} км")
    for s in ax.spines.values():           # style() вернул тонкую рамку
        s.set_color(SLATE)
        s.set_linewidth(1.8)
    save(fig, "p_final_map.png")


def p_classes_map(site="anabar"):
    """Карта с классами уверенности: что отдаём вместо бинарного отсева."""
    from experiments.openlin_lines import _branches

    z = np.load(M / f"openlin_confidence_{site}.npz")
    a, b, c = z["A"], z["B"], z["C"]
    site_z = load_site(site)
    res_m = float(site_z["res_m"])
    valid = np.isfinite(site_z["gray"])
    gray = robust(np.nan_to_num(site_z["gray"],
                                nan=float(np.nanmedian(site_z["gray"]))))
    r0, r1, c0, c1 = CROP

    def km(m):
        return m.sum() * res_m / 1000.0

    def draw(ax, m_a, m_b, m_c, base, k):
        ax.imshow(base, cmap="gray", alpha=0.85)
        for m, col in ((m_c & ~m_b, GREY), (m_b & ~m_a, SLATE), (m_a, ACC)):
            ax.imshow(np.ma.masked_where(~fat(m, k), np.zeros(base.shape)),
                      cmap=ListedColormap([col]))

    fig = plt.figure(figsize=(12.6, 5.8), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.06,
                          left=0.012, right=0.988, top=0.93, bottom=0.16)

    ax = fig.add_subplot(gs[0])
    draw(ax, a, b, c, np.ma.masked_where(~valid, gray), 3)
    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, facecolor="none",
                           edgecolor=INK, lw=1.4, ls="--"))
    style(ax, None, "вся площадка, 86 × 77 км")

    ax = fig.add_subplot(gs[1])
    draw(ax, crop(a), crop(b), crop(c), crop(gray), 2)
    style(ax, None, "фрагмент в рамке, 22 × 18 км")

    handles = [
        Line2D([], [], color=ACC, lw=3.5,
               label=f"A — видят оба: {km(a):.0f} км, чистота ×196"),
        Line2D([], [], color=SLATE, lw=3.5,
               label=f"B — плюс {km(b) - km(a):.0f} км, ×108"),
        Line2D([], [], color=GREY, lw=3.5,
               label=f"C — плюс {km(c) - km(b):.0f} км, ×12"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                     frameon=False, fontsize=S_SIZE,
                     bbox_to_anchor=(0.5, 0.055))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.5, 0.015, "чистота — во сколько раз число линий класса выше "
             "случайного уровня; классы вложены: A ⊂ B ⊂ C", ha="center",
             fontsize=S_SIZE, color=INK2)
    save(fig, "p_classes_map.png")


def p_waterfall():
    """Сколько линий остаётся на каждом условии отбора."""
    names = ["прослежено", "подтвердил\nрадар", "подтвердила\nоптика",
             "подтвердили\nоба", "прошло бы\nслучайно"]
    vals = [613, 393, 222, 153, 0.9]
    cols = [GREY, ACC, SLATE, INK, EDGE]

    fig, ax = plt.subplots(figsize=(11.5, 5.0), facecolor=SURF)
    bare(ax)
    ax.bar(range(5), vals, 0.62, color=cols, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.18 if v > 2 else 1.4,
                f"{v:.0f}" if v >= 2 else f"{v:.1f}", ha="center",
                fontsize=T_SIZE, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2200)
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=S_SIZE, color=INK2)
    ax.set_ylabel("линий, логарифмическая шкала", fontsize=S_SIZE, color=INK2)
    fig.tight_layout()
    save(fig, "p_waterfall.png")


# ===========================================================================
# Плотность и перенос
# ===========================================================================

def p_density():
    """Запас подтверждения на пяти ступенях плотности."""
    df = pd.read_csv(M / "openlin_unsup_sweep_anabar.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11.5, 5.0), facecolor=SURF)
    bare(ax)
    x = df["линий"].to_numpy()
    y = df["запас"].to_numpy()
    ax.plot(x, y, "-o", color=ACC, lw=2.4, ms=9)
    ax.axhline(1.0, color=INK2, lw=1.2, ls="--")
    ax.text(x.max(), 1.06, "случайный уровень", ha="right", fontsize=S_SIZE,
            color=INK2)
    for xi, yi, name in zip(x, y, df["плотность"]):
        ax.annotate(f"{name}\n×{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 16), ha="center", fontsize=S_SIZE, color=INK)
    ax.set_xlabel("прослежено линий", fontsize=S_SIZE, color=INK2)
    ax.set_ylabel("во сколько раз выше случайного", fontsize=S_SIZE,
                  color=INK2)
    ax.set_ylim(0, y.max() * 1.35)
    ax.set_xlim(0, x.max() * 1.12)
    fig.tight_layout()
    save(fig, "p_density.png")


def p_transfer():
    """Перенос ранжирования линий между площадками."""
    a = pd.read_csv(M / "openlin_line_rank_anabar_to_nevada.csv",
                    encoding="utf-8-sig").set_index("модель")
    b = pd.read_csv(M / "openlin_line_rank_nevada_to_anabar.csv",
                    encoding="utf-8-sig").set_index("модель")
    rows = [("ANFIS", a.loc["ANFIS", "AUC_перенос"],
             b.loc["ANFIS", "AUC_перенос"]),
            ("логрегрессия", a.loc["логрегрессия", "AUC_перенос"],
             b.loc["логрегрессия", "AUC_перенос"]),
            ("только длина", 0.693, 0.710)]

    fig, ax = plt.subplots(figsize=(11.5, 5.0), facecolor=SURF)
    bare(ax)
    idx = np.arange(len(rows))
    ax.bar(idx - 0.19, [r[1] for r in rows], 0.36, color=SLATE, linewidth=0,
           label="Анабар → Невада")
    ax.bar(idx + 0.19, [r[2] for r in rows], 0.36, color=ACC, linewidth=0,
           label="Невада → Анабар")
    for i, r in enumerate(rows):
        ax.text(i - 0.19, r[1] + 0.012, f"{r[1]:.3f}", ha="center",
                fontsize=S_SIZE, color=INK)
        ax.text(i + 0.19, r[2] + 0.012, f"{r[2]:.3f}", ha="center",
                fontsize=S_SIZE, color=INK)
    ax.axhline(0.5, color=INK2, lw=1.2, ls="--")
    ax.text(len(rows) - 0.55, 0.512, "случайный порядок", fontsize=S_SIZE,
            color=INK2)
    ax.set_xticks(idx)
    ax.set_xticklabels([r[0] for r in rows], fontsize=S_SIZE, color=INK2)
    ax.set_ylim(0.45, 0.86)
    ax.set_ylabel("AUC на чужой площадке", fontsize=S_SIZE, color=INK2)
    leg = ax.legend(fontsize=S_SIZE, frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    save(fig, "p_transfer.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    site = "anabar"
    z = load_site(site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])

    p_features(z, valid, res_m)
    p_ellipse(z, valid, res_m)
    p_scree(site)
    z2, valid2, res2, comp = components(site, OPTIC + RADAR)
    p_components(comp)
    p_two_ways(site, res2)
    p_windows()
    strength, _ang = edges_from_components(comp, valid2, res2)
    strength = np.nan_to_num(strength, nan=0.0)
    p_angles(valid2, res2, comp, strength)
    p_field(strength)
    p_trace(strength, valid2, res2)
    p_confirm(site, res_m)
    p_waterfall()
    p_final_map(site)
    p_density()
    p_transfer()
    print(f"\nвсе картинки в {OUT}")


if __name__ == "__main__":
    main()
