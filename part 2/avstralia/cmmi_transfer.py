"""Вариант D: перенос модели между территориями для ОДНОГО типа руды (Zn-Pb).

Общее признаковое пространство — глобальная геофизика GMT (mag4km, faa, vgg,
geoid, relief), сэмплируемая по тайлам N..E.. (докачка по точкам). Эксперимент:
  • train США+Канада -> test Австралия (перенос: модель не видела Австралию);
  • within-domain Австралия (spatial CV) — верхняя планка;
  • критериальный индекс (аналог ГИС Интегро) на Австралии — базовый уровень.
И симметрично (train AU -> test USCA). Метрики: ROC-AUC, lift@10%.

Запуск: python3 -m experiments.cmmi_transfer
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from cache_paths import CMMI
from experiments.common import DS, SEED, region_xy, criterial, lift

OCC = str(CMMI / "cmmi_occ.csv")


def evaluate(Xtr, ytr, Xte, yte, lon_te, lat_te):
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=SEED).fit(Xtr, ytr)
    s_tr = gb.predict_proba(Xte)[:, 1]                       # перенос
    s_cr = criterial(Xtr, ytr, Xte)                          # критериальный (обучен на train-территории)
    # within-domain: spatial CV на тестовой территории
    g = np.floor(lon_te / 5).astype(int) * 1000 + np.floor(lat_te / 5).astype(int)
    wd = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xte, yte, g):
        m = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=SEED).fit(Xte[tr], yte[tr])
        wd.append(roc_auc_score(yte[te], m.predict_proba(Xte[te])[:, 1]))
    return {"перенос AUC": roc_auc_score(yte, s_tr), "перенос lift": lift(s_tr, yte),
            "критериальный AUC": roc_auc_score(yte, s_cr), "критериальный lift": lift(s_cr, yte),
            "within-domain AUC": float(np.mean(wd))}


def main():
    t0 = time.time()
    occ = pd.read_csv(OCC, encoding="latin1", low_memory=False)
    print("сэмплирую геофизику для США+Канады ...")
    Xu, yu, lou, lau = region_xy(occ, ["United States of America", "Canada"], (-170, -52), (25, 75), SEED)
    print(f"[{time.time()-t0:4.0f}s] USCA: presence={int(yu.sum())}, bg={int((yu==0).sum())}")
    print("сэмплирую геофизику для Австралии ...")
    Xa, ya, loa, laa = region_xy(occ, ["Australia"], (112, 154), (-44, -10), SEED)
    print(f"[{time.time()-t0:4.0f}s] AU: presence={int(ya.sum())}, bg={int((ya==0).sum())}")

    print(f"\n[{time.time()-t0:4.0f}s] === ПЕРЕНОС: train США+Канада -> test Австралия ===")
    for k, v in evaluate(Xu, yu, Xa, ya, loa, laa).items():
        print(f"   {k:22s} {v:.3f}")
    print(f"\n[{time.time()-t0:4.0f}s] === ПЕРЕНОС: train Австралия -> test США+Канада ===")
    for k, v in evaluate(Xa, ya, Xu, yu, lou, lau).items():
        print(f"   {k:22s} {v:.3f}")
    print(f"[{time.time()-t0:4.0f}s] готово.")


if __name__ == "__main__":
    main()
