"""Подбор правила сшивки на парах с известным ответом (близнецы).

ОТКУДА ПОСТАНОВКА. :mod:`openlin_twins` дал пары концов с ответом из второй
половины съёмки: «+» один разорванный объект, «−» соседние параллельные линии.
Замер на них: оба действующих правила (средний отклик вдоль прямой и слабое
звено пути) соединяют ~85 % настоящих разрывов, но и склеивают ~70 % соседей.
Здесь проверяется, можно ли по тем же данным подобрать правило лучше.

ЧТО СРАВНИВАЕТСЯ, на одной таблице пар:
  — два действующих правила, как есть;
  — подобранное правило на трёх наборах признаков: только отклик, только
    геометрия, всё вместе. Две модели: логистическая регрессия (веса читаются)
    и градиентный бустинг (ловит нелинейность).
Признаки — только из рабочей половины; опорная даёт только ответ.

КАК ПРОВЕРЯЕТСЯ. Оценка — всегда на парах, которых модель не видела:
  «по месту»   — лист разбит на блоки 4 x 4, блоки делятся на 4 группы, модель
                 учится на трёх и проверяется на четвёртой (соседние пары
                 одного уступа не попадают и в обучение, и в проверку);
  «по съёмке»  — учится на парах одной рабочей половины, проверяется на парах
                 другой (места те же, снимок другой).
Сравнение с действующим правилом — при ТОЙ ЖЕ доле соединённых настоящих
разрывов, что у него: сколько соседей при этом склеивается.

Это подбор и замер, а не прогон тракта: рабочий набор не меняется, в счётчик
не идёт. Чтобы правило пошло в сшивку, нужен объявленный полный прогон.

РЕЗУЛЬТАТ (09.10.2026, лог ``outputs/metrics/_twins_fit.log``). Пар с ответом
1982: «+» 973, «−» 1009 (после чистки «−» от ветвлений — 330 пар убрано).
  Рабочее правило: соединяет настоящих 83.6 %, склеивает соседей 59.2 %.
  Подобранное правило при тех же 84 % настоящих склеивает соседей:
    по всем признакам, логистическая — 24.4 % «по месту», 29.5 % «по съёмке»;
    только геометрия                 — 27.0 % / 29.4 %;
    только отклик                    — 53.8 % / 52.7 % (почти как сейчас).
  Бустинг везде чуть хуже логистической — нелинейности тут нет.
  ПОЧТИ ВСЁ РАЗЛИЧЕНИЕ ДАЁТ ОДИН ПРИЗНАК — БОКОВОЙ СДВИГ (вклад 0.24, у
  остальных не больше 0.02): медиана у «+» 55 м, у «−» 199 м. Новый признак
  «отклик прямо за концом» не сработал (вклад 0.001). Отклик между концами в
  любом виде добавляет мало.
  Один только предел бокового сдвига, вместе с рабочим правилом:
    <= 60 м: настоящих 47 %, соседей  8 %;   <= 150 м: 77 % / 24 %;
    <= 100 м:          65 %,         13 %;   <= 200 м: 80 % / 33 %;
    <= 600 м (сейчас): 84 %,         59 %.
  ОГОВОРКА О КРУГЕ: боковой сдвиг частично вшит в саму разметку. «+» — оба
  конца в 60 м от ОДНОЙ линии, значит их сдвиг заведомо мал; «−» — опорные
  линии не ближе 60 м друг к другу, значит сдвиг заведомо не мал. Поэтому
  0.86 завышено на неизвестную величину. По существу вывод всё равно верен
  (соседи — это и есть линии со сдвигом вбок), но точным цифрам верить нельзя.
  СВЯЗЬ С ПРЕЖНИМ: строгая сшивка (сдвиг до 60 м) уже пробовалась 05.10.2026 и
  отклонена просмотром — «отсеялось нужное», теряется 300 км. Близнецы дают
  тому цифру: предел 60 м режет половину настоящих разрывов.

Запуск из корня репозитория (после openlin_twins):
    python -m experiments.openlin_twins_fit --site anabar
"""
from __future__ import annotations

import argparse

import numpy as np

from experiments.openlin_zone_lines import M

SETS = {"только отклик": slice(6, 12), "только геометрия": slice(0, 6),
        "всё вместе": slice(0, 12)}
N_FOLDS = 4


def auroc(score, y) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score))


def glued_at_recall(score, y, recall: float) -> float:
    """Доля склеенных соседей при пороге, дающем заданную долю настоящих."""
    thr = np.quantile(score[y == 1], 1.0 - recall)
    return float((score[y == 0] >= thr).mean())


def models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "логистическая": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                               StandardScaler(),
                                               LogisticRegression(max_iter=2000)),
        "бустинг": lambda: HistGradientBoostingClassifier(random_state=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    a = ap.parse_args()

    from sklearn.inspection import permutation_importance

    d = np.load(M / f"openlin_twins_{a.site}_таблица.npz", allow_pickle=False)
    X, y = d["X"].astype("float64"), d["y"].astype(int)
    block, half = d["блок"], d["половина"]
    ok_m, ok_w = d["прошла_прямая"], d["прошёл_путь"]
    names = [str(s) for s in d["признаки"]]
    X = np.where(np.isfinite(X), X, np.nan)
    print(f"пар с ответом {len(y)}: «+» {int(y.sum())}, «−» {int((1 - y).sum())}; "
          f"по половинам: " + ", ".join(f"«{h}» {int((half == h).sum())}"
                                        for h in np.unique(half)), flush=True)

    rec_m, rec_w = ok_m[y == 1].mean(), ok_w[y == 1].mean()
    print("\nдействующие правила на этих парах:")
    for nm, ok in (("прямая (рабочее)", ok_m), ("путь", ok_w), ("оба сразу", ok_m & ok_w)):
        print(f"  {nm:18s}: соединяет настоящих {100 * ok[y == 1].mean():.1f} %, "
              f"склеивает соседей {100 * ok[y == 0].mean():.1f} %")
    print(f"  как числа: средний по прямой AUROC {auroc(np.nan_to_num(X[:, 6], nan=-9), y):.3f}, "
          f"слабое звено {auroc(np.nan_to_num(X[:, 8], nan=-9), y):.3f}")

    # группы блоков для проверки «по месту»: блоки тасуются один раз
    ub = np.unique(block)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(rng.permutation(ub), np.arange(len(ub)) % N_FOLDS))
    fold = np.array([fold_of[b] for b in block])

    for check in ("по месту", "по съёмке"):
        print(f"\nпроверка «{check}» (на парах, которых модель не видела); в скобках — "
              f"склеено соседей при {100 * rec_m:.0f} % соединённых настоящих, как у "
              f"рабочего правила ({100 * ok_m[y == 0].mean():.0f} %):")
        for sname, sl in SETS.items():
            for mname, make in models().items():
                oof = np.full(len(y), np.nan)
                if check == "по месту":
                    splits = [(fold != f, fold == f) for f in range(N_FOLDS)]
                else:
                    hs = np.unique(half)
                    splits = [(half != h, half == h) for h in hs]
                for tr, te in splits:
                    m = make().fit(X[tr][:, sl], y[tr])
                    oof[te] = m.predict_proba(X[te][:, sl])[:, 1]
                print(f"  {sname:17s} · {mname:13s}: AUROC {auroc(oof, y):.3f} "
                      f"({100 * glued_at_recall(oof, y, rec_m):.1f} %)", flush=True)

    # какие признаки работают: перестановочная важность на отложенных блоках
    print("\nвклад признаков (бустинг, всё вместе; падение AUROC при перемешивании "
          "признака, среднее по группам блоков):")
    imp = np.zeros(len(names))
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        m = models()["бустинг"]().fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], scoring="roc_auc", n_repeats=10,
                                   random_state=0)
        imp += r.importances_mean / N_FOLDS
    for k in np.argsort(-imp):
        print(f"  {names[k]:30s} {imp[k]:+.3f}   медиана «+» {np.nanmedian(X[y == 1, k]):8.2f}"
              f"   «−» {np.nanmedian(X[y == 0, k]):8.2f}")


if __name__ == "__main__":
    main()
