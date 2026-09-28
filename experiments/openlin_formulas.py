"""Формульные карточки для презентации: весь счёт метода в виде формул.

Каждая карточка — отдельная картинка на отдельный слайд: строка формулы плюс
короткая расшифровка под ней. Формулы набраны matplotlib mathtext, поэтому
кириллица держится ВНЕ ``$...$``: внутри математического шрифта её нет.

Источник формул — сам код, а не пересказ:

* шаг 1 — :mod:`experiments.openlin_neigh` (``_gsm``, ``_one_channel``);
* шаг 2 — :func:`experiments.openlin_domains.domain_edges` и
  :mod:`experiments.openlin_mae`;
* шаг 3 — :func:`experiments.openlin_domains._kernel`, ``_smooth``,
  ``edges_from_components``;
* шаг 4 — :func:`experiments.openlin_lines._ridge_nms`, ``trace_lines``;
* шаг 5 — :func:`experiments.openlin_unsup.confirmed_branches`, ``confirm``.

Запуск из корня репозитория:
    python -m experiments.openlin_formulas
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/slides")

INK, INK2 = "#1B2A32", "#6E7F88"
ACC = "#C15B33"
SURF = "#FBFAF8"
EDGE = "#D8D2C8"

#: Ширина карточки в дюймах. Слайд 13.33", поля по 0.6" — картинка идёт почти
#: во всю ширину, поэтому масштабирование на слайде близко к 1:1.
WIDTH = 12.1

F_SIZE = 25          # формула
G_SIZE = 14.5        # расшифровка

#: Отступы строки, дюймы: до расшифровки, после неё и поле сверху.
GAP_GLOSS, GAP_ROW, PAD_TOP = 0.14, 0.46, 0.26

matplotlib.rcParams["mathtext.fontset"] = "cm"


def _heights(rows: list[tuple[str, str]]) -> list[float]:
    """Высота набранной формулы, дюймы. Дробь под корнем вдвое выше строки,
    поэтому место под расшифровку считается, а не задаётся константой."""
    fig = plt.figure(figsize=(WIDTH, 4))
    r = fig.canvas.get_renderer()
    out = []
    for tex, _g in rows:
        t = fig.text(0.035, 0.5, tex, fontsize=F_SIZE)
        out.append(t.get_window_extent(renderer=r).height / fig.dpi)
        t.remove()
    plt.close(fig)
    return out


def card(name: str, rows: list[tuple[str, str]]) -> None:
    """Карточка из строк ``(формула, расшифровка)``."""
    fh = _heights(rows)
    gh = [0.30 if g else 0.0 for _t, g in rows]
    row_h = [f + (GAP_GLOSS + g if g else 0.0) + GAP_ROW
             for f, g in zip(fh, gh)]
    h = PAD_TOP + sum(row_h)

    fig = plt.figure(figsize=(WIDTH, h), facecolor=SURF)
    y = PAD_TOP                                  # курсор сверху, дюймы
    for i, ((tex, gloss), f, g, rh) in enumerate(zip(rows, fh, gh, row_h)):
        if i:
            yy = 1.0 - (y - GAP_ROW * 0.55) / h
            fig.add_artist(plt.Line2D([0.035, 0.965], [yy, yy], color=EDGE,
                                      lw=0.8, transform=fig.transFigure))
        fig.text(0.035, 1.0 - y / h, tex, fontsize=F_SIZE, color=INK,
                 va="top", ha="left")
        if gloss:
            fig.text(0.035, 1.0 - (y + f + GAP_GLOSS) / h, gloss,
                     fontsize=G_SIZE, color=INK2, va="top", ha="left")
        y += rh
    fig.savefig(OUT / name, dpi=200, facecolor=SURF)
    plt.close(fig)
    print(f"{name} готов")


CARDS: dict[str, list[tuple[str, str]]] = {
    # ---------------------------------------------------------------- шаг 1
    "f_tensor.png": [
        (r"$I_\sigma=\dfrac{(I\cdot v)*G_\sigma}{v*G_\sigma},"
         r"\qquad \sigma=1.5$",
         "сглаживание, нормированное на маску кадра v: пустые углы снимка "
         "не затягиваются внутрь"),
        (r"$\nabla I_\sigma=(I_x,\;I_y)$",
         "градиент — направление наибольшего роста яркости"),
        (r"$J_{xx}=\langle I_x^2\rangle_\rho,\quad "
         r"J_{yy}=\langle I_y^2\rangle_\rho,\quad "
         r"J_{xy}=\langle I_xI_y\rangle_\rho$",
         "структурный тензор: усредняются произведения, а не сами градиенты; "
         "ρ = 500 м, 1 км, 2 км"),
        (r"$\lambda_\pm=\dfrac{J_{xx}+J_{yy}}{2}\pm"
         r"\dfrac{1}{2}\sqrt{(J_{xx}-J_{yy})^2+4J_{xy}^2}$",
         "оси эллипса направлений в окрестности"),
    ],
    "f_five.png": [
        (r"$C=\dfrac{\lambda_+-\lambda_-}{\lambda_++\lambda_-}\in[0,1],"
         r"\qquad E=\log(\lambda_++\lambda_-)$",
         "вытянутость текстуры и её контрастность"),
        (r"$\cos2\theta,\;\sin2\theta,\qquad "
         r"\theta=\frac{1}{2}\arctan\dfrac{2J_{xy}}{J_{xx}-J_{yy}}$",
         "направление в удвоенном угле: линия 5° и линия 185° — одна и та же"),
        (r"$R=\lambda_k(H),\;\;k=\arg\max_k|\lambda_k|,\qquad "
         r"H=\nabla\nabla^{\!\top}I_{\rho/2}$",
         "знаковый гребень: тёмная борозда и светлый хребет получают разный "
         "знак"),
        (r"$\tilde f=\dfrac{f-\mathrm{med}\,f}{\mathrm{IQR}\,f},"
         r"\qquad 10\times3\times5+10=160$",
         "робастная нормировка и итог шага: 160 чисел в каждой ячейке"),
    ],
    # ---------------------------------------------------------------- шаг 2
    "f_pca.png": [
        (r"$\Sigma=\frac{1}{N}\sum_i(f_i-\bar f)(f_i-\bar f)^{\!\top},"
         r"\qquad \Sigma=U\Lambda U^{\!\top}$",
         "форма облака из 662 200 точек в пространстве 160 признаков"),
        (r"$u=\Lambda_8^{-1/2}U_8^{\!\top}(f-\bar f)\in\mathbb{R}^8$",
         "8 компонент с отбеливанием: каждая делится на свой разброс и входит "
         "на равных"),
        (r"$\|u_a-u_b\|_2=\sqrt{(f_a-f_b)^{\!\top}\Sigma^{-1}(f_a-f_b)}$",
         "поэтому обычное расстояние между участками измеряется в сигмах"),
    ],
    "f_ae.png": [
        (r"$u=\mathrm{Enc}(f\odot m),\qquad \hat f=\mathrm{Dec}(u),"
         r"\qquad m\sim\mathrm{Bern}(0.5)$",
         "половина ячеек скрыта, сеть восстанавливает их по окружению"),
        (r"$\mathcal{L}=\dfrac{\sum(\hat f-f)^2(1-m)\,v}{\sum(1-m)\,v}$",
         "потеря считается только на скрытых и только на валидных ячейках"),
        (r"$d\in\{1,2,4\},\;\;k=3\;\;\Rightarrow\;\;RF=15$",
         "расширения свёрток дают рецептивное поле 15 ячеек = 1.5 км, "
         "понижения разрешения нет"),
        (r"$u\in\mathbb{R}^8$",
         "тот же размер, что у компонент: латент подставляется в то же место "
         "оператора"),
    ],
    # ---------------------------------------------------------------- шаг 3
    "f_window.png": [
        (r"$e_\theta=(-\cos\theta,\,\sin\theta),\qquad "
         r"n_\theta=(\sin\theta,\,\cos\theta)$",
         "вдоль предполагаемой линии и поперёк неё"),
        (r"$K^{\pm}_\theta(p)=\exp\!\left(-\dfrac{(p\!\cdot\!e_\theta)^2}"
         r"{2\sigma_\parallel^2}-\dfrac{(p\!\cdot\!n_\theta\mp\delta)^2}"
         r"{2\sigma_\perp^2}\right)$",
         "два вытянутых окна, сдвинутых поперёк линии в разные стороны"),
        (r"$\sigma_\parallel=2\;\mathrm{km},\qquad "
         r"\sigma_\perp=200\;\mathrm{m},\qquad \delta=300\;\mathrm{m}$",
         "вдоль линии длинное, поперёк узкое, между центрами окон 600 м"),
        (r"$\bar u^{\pm}_\theta=\dfrac{(u\cdot v)*K^{\pm}_\theta}"
         r"{v*K^{\pm}_\theta}$",
         "средний портрет участка; свёртка снова нормирована на маску кадра"),
    ],
    "f_edge.png": [
        (r"$S_\theta=\left\|\bar u^{+}_\theta-\bar u^{-}_\theta\right\|_2$",
         "сила границы под углом θ: длина разности двух восьмимерных "
         "портретов, в сигмах"),
        (r"$S=\max_{\theta}S_\theta,\qquad A=\arg\max_{\theta}S_\theta$",
         "поле силы и поле азимута линии"),
        (r"$\theta_i=\dfrac{\pi i}{12},\quad i=0,\dots,11$",
         "12 направлений с шагом 15°; 216 свёрток через быстрое "
         "преобразование Фурье, 4–5 с на площадку"),
    ],
    # ---------------------------------------------------------------- шаг 4
    "f_ridge.png": [
        (r"$H=\nabla\nabla^{\!\top}S_\sigma,\qquad "
         r"n=\mathrm{eig}(H,\lambda_{\min})$",
         "направление поперёк линии — наибольшая кривизна поля силы"),
        (r"$\mathrm{NMS}:\;\;S(p)\geq S(p+n)\;\wedge\;S(p)\geq S(p-n)"
         r"\;\wedge\;\lambda_{\min}<0$",
         "от полосы остаётся гребень толщиной в одну ячейку"),
        (r"$h=Q_{0.55}(S),\qquad l=Q_{0.20}(S)$",
         "два порога по квантилям площади: ядро линии и её продолжение"),
        (r"$L=\{c:\,S(c)\geq l\;\wedge\;c\leftrightarrow\{S\geq h\}\}$",
         "продолжение берётся, только если оно связно с ядром"),
    ],
    "f_geom.png": [
        (r"$a=\arg\max_{\|a\|=1}\sum_i\left((p_i-\bar p)\!\cdot\!a\right)^2,"
         r"\qquad \ell=\max_i(p_i\!\cdot\!a)-\min_i(p_i\!\cdot\!a)$",
         "главная ось ветки и её размах вдоль этой оси"),
        (r"$P=\sum_i\|p_{i+1}-p_i\|,\qquad "
         r"\|p_{i+1}-p_i\|\in\{1,\sqrt{2}\}$",
         "длина пути с учётом диагоналей"),
        (r"$\mathrm{str}=\ell/P\geq0.45,\qquad \ell\geq400\;\mathrm{m}$",
         "фильтр по прямолинейности и длине"),
        (r"$g\leq2\;\mathrm{km},\qquad \angle\leq20^\circ$",
         "сшивка разрывов: концы соединяются, только если смотрят друг "
         "на друга"),
    ],
    # ---------------------------------------------------------------- шаг 5
    "f_confirm.png": [
        (r"$d(c)=\min_{q\in L'}\|c-q\|\leq\tau,\qquad \tau=250\;\mathrm{m}$",
         "рядом с ячейкой есть линия второго прибора"),
        (r"$\Delta A(c)=\min\!\left(|A'-A|,\,180^\circ-|A'-A|\right)"
         r"\leq\alpha,\quad \alpha=20^\circ$",
         "и она идёт в ту же сторону"),
        (r"$\varphi(L,L')=\dfrac{1}{|L|}\sum_{c\in L}"
         r"\mathbb{1}\!\left[d(c)\leq\tau\;\wedge\;\Delta A(c)\leq\alpha\right]$",
         "доля длины ветки, подтверждённая вторым прибором"),
        (r"$L\;\mathrm{ok}\Leftrightarrow\varphi(L,L_{\mathrm{opt}})\geq0.5\;\;\wedge"
         r"\;\;\varphi(L,L_{\mathrm{rad}})\geq0.5$",
         "нужны оба прибора по отдельности, а не любой из них"),
    ],
    "f_null.png": [
        (r"$L'_s=\mathrm{roll}(L',\,s),\qquad s\sim U(H)\times U(W)$",
         "тороидальный сдвиг: число линий, длины и направления те же, "
         "привязка к местности случайная"),
        (r"$N_0=\mathbb{E}_s\,\#\{L:\;\varphi(L,L'_s)\geq0.5\}=0.9$",
         "сколько веток прошло бы фильтр случайно, 25 сдвигов"),
        (r"$N/N_0=153/0.9=174$",
         "чистота отбора"),
    ],
    # ------------------------------------------------------- метрики и ранг
    "f_density.png": [
        (r"$\mathrm{cov}=\dfrac{1}{2}\left("
         r"\dfrac{|\{c\in L_{\mathrm{opt}}:d_{\mathrm{rad}}(c)\leq\tau\}|}"
         r"{|L_{\mathrm{opt}}|}+"
         r"\dfrac{|\{c\in L_{\mathrm{rad}}:d_{\mathrm{opt}}(c)\leq\tau\}|}"
         r"{|L_{\mathrm{rad}}|}\right)$",
         "взаимное покрытие двух карт по длине"),
        (r"$\mathrm{margin}=\dfrac{\mathrm{cov}}"
         r"{Q_{0.95}\left(\mathrm{cov}_{s}\right)}>1$",
         "запас над случайным уровнем — правило остановки при уплотнении"),
    ],
    "f_auc.png": [
        (r"$\mathrm{AUC}=P\!\left(\hat y(L_+)>\hat y(L_-)\right)$",
         "вероятность того, что подтверждённая линия окажется выше "
         "неподтверждённой"),
        (r"$p=\dfrac{1+\#\{\mathrm{AUC}_\pi\geq\mathrm{AUC}\}}{1+n_\pi},"
         r"\qquad n_\pi=10^4$",
         "перестановочный ноль: метки на чужой площадке перемешиваются "
         "10 000 раз"),
        (r"$p=10^{-4}<\alpha=2.22\cdot10^{-4}$",
         "порог с поправкой на всё семейство проверок"),
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in CARDS.items():
        card(name, rows)
    print(f"\nвсе карточки в {OUT}")


if __name__ == "__main__":
    main()
