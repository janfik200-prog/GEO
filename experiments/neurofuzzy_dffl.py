"""ДФФЛ — глубокое обучение нечётких признаков на ветке с геологией.

ЗАЧЕМ. КАФН дал ту же точность, что бустинг, но остался плоской сетью:
правила в нём пишутся по отдельным признакам. ДФФЛ (`DeepFuzzyFeatureModel`
из `deep-neuro-fuzzy`) устроен иерархично: признаки разбиты на предметные
группы, внутри каждой группы нечёткие правила сворачивают её в несколько
чисел — концептов, а правила следующей стадии пишутся уже по концептам.
Нам это нужно не ради метрики: из обученной модели достаётся трасса решения
по конкретной ячейке — какие правила сработали внутри группы и как группа
вложилась в итоговый балл. Это паспорт перспективной ячейки на языке
обстановок, а не весов дерева.

КАК РАЗБИТЫ ГРУППЫ. Состав блоков задан вручную по предметному смыслу, а не
автоматикой: разломная обстановка, вмещающая среда, магматизм, рельеф,
дистанционный облик. Внутри каждой группы взяты ведущие признаки по вкладу
Шепли (`outputs/metrics/shap_two_branches.csv`), не более четырёх на блок —
иначе число правил внутри блока перестаёт читаться. Имена стадии-сводки
намеренно нейтральные («сводный 1..3»): их смысл раскрывают правила, которые
модель выучит сама, домысливать его заранее нельзя.

ВАЖНО ПРИ ЧТЕНИИ. Цель — верхние 10% критериального балла, а балл собран из
тех же геологических слоёв. Высокая точность на этой ветке означает замыкание
формулы на свои входы, а не геологическое открытие (так же, как в
`shap_two_branches`). Ценность запуска — в читаемости решения.

ЧТО СЧИТАЕТСЯ.
1. Полосовая проверка с разрывом 15 км: ДФФЛ против рабочего бустинга на одних
   и тех же фолдах.
2. Модель на всей площади листа: карта рядом с бустингом и совпадение верхних
   10%.
3. Трассы решения: вклад концептов в балл в среднем по перспективным ячейкам и
   подробный разбор двух ячеек — перспективной и фоновой.

ВХОД: сетка листа, data/processed/dataset_v5_rebuilt.parquet
ВЫХОД: outputs/03_признаковое_пространство/neurofuzzy_dffl.png,
       outputs/metrics/neurofuzzy_dffl.log и .csv
Запуск из корня: python -X utf8 -m experiments.neurofuzzy_dffl
Переменная окружения DFFL_FOLDS ограничивает число полос (для пробы времени).
"""

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NF_ROOT = Path.home() / "deep-neuro-fuzzy"
sys.path.insert(0, str(NF_ROOT / "src"))

from src import (cell_mask, config, crit_reference, criterial_target,  # noqa: E402
                 mapstyle, neighbourhood as nb, outputs, spectral_mask)
from experiments.forecast_wide import honest_features  # noqa: E402
from experiments.honest_feature_signal import make_model, strip_folds  # noqa: E402
from experiments.spectral_exposure_threshold import (ACROSS_KM, ALONG_KM,  # noqa: E402
                                                     NW_AZIMUTH)

from ruanfis import (BootstrapConfig, DecisionLayerConfig, FuzzyTrainer,  # noqa: E402
                     FuzzyVariable, GaussianMembership,
                     HierarchicalModelConfig, StageConfig,
                     StagewisePretrainingConfig, TrainingConfig,
                     TransparentBlockConfig, analyze_concept_flow,
                     build_stagewise_pretrained_hierarchical_model,
                     format_concept_flows)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AREA = 0.10
GAP_M = 15_000.0
MAX_FOLDS = int(os.environ.get("DFFL_FOLDS", "0"))

# предметные блоки: имя, имена концептов, (столбец, читаемое имя переменной)
BLOCKS = [
    ("разломная_обстановка", ("разломы_1", "разломы_2"),
     [("dist_tect1", "расст_разломы_СЗ"),
      ("dist_tect2", "расст_разломы_СВ"),
      ("dens_tect", "плотность_разломов"),
      ("geo2_fault_node", "расст_узлы_разломов")]),
    ("вмещающая_среда", ("среда_1", "среда_2"),
     [("dist_paleo", "расст_палеодолины"),
      ("dist_facies", "расст_фации"),
      ("dist_struct", "расст_коры_выветривания"),
      ("geo2_facies2", "фация")]),
    ("магматизм", ("магматизм_1",),
     [("dist_magm", "расст_дайки"),
      ("dens_magm", "плотность_даек"),
      ("geo2_contact", "расст_контакты")]),
    ("рельеф", ("рельеф_1",),
     [("dem_elev", "высота"),
      ("dem_relief_1km", "расчленённость_1км"),
      ("relief2_curv_5km", "кривизна_5км"),
      ("dem_incision", "врезание")]),
    ("дистанционный_облик", ("облик_1",),
     [("lin_node_dens", "плотность_узлов_линеаментов"),
      ("lin_dist", "расст_линеамент"),
      ("s2_iron_ox", "индекс_окислов_железа")]),
]
SVOD = ("сводный_1", "сводный_2", "сводный_3")
T0 = time.time()
LOG: list[str] = []


def log(msg: str) -> None:
    line = "[%6.1fs] %s" % (time.time() - T0, msg)
    print(line, flush=True)
    LOG.append(line)


def var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership(
        [0.2, 0.5, 0.8], [0.18, 0.18, 0.18],
        term_names=["низкий", "средний", "высокий"]))


def var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership(
        [0.25, 0.75], [0.2, 0.2], term_names=["низкий", "высокий"]))


def build_config(input_dim: int) -> HierarchicalModelConfig:
    """Две стадии: предметные блоки -> сводные концепты -> решающий слой."""
    blocks, start, concepts = [], 0, []
    for name, cnames, feats in BLOCKS:
        idx = tuple(range(start, start + len(feats)))
        start += len(feats)
        blocks.append(TransparentBlockConfig(
            name=name, input_indices=idx,
            variables=tuple(var3(ru) for _, ru in feats),
            n_concepts=len(cnames), concept_names=cnames,
            max_rule_arity=2, max_rules=12,
            rule_generation_mode="prototype", prototype_term_limit=2,
            prototype_sample_size=512, consequent_mode="affine_sigmoid"))
        concepts.extend(cnames)
    return HierarchicalModelConfig(
        input_dim=input_dim,
        stages=(
            StageConfig(name="предметные_блоки", blocks=tuple(blocks)),
            StageConfig(name="сводка", blocks=(TransparentBlockConfig(
                name="сводный_блок",
                input_indices=tuple(range(len(concepts))),
                variables=tuple(var3(c) for c in concepts),
                n_concepts=len(SVOD), concept_names=SVOD,
                max_rule_arity=2, max_rules=24,
                rule_generation_mode="prototype", prototype_term_limit=2,
                prototype_sample_size=512,
                consequent_mode="affine_sigmoid"),)),
        ),
        decision_layer=DecisionLayerConfig(
            name="решение", variables=tuple(var2(c) for c in SVOD),
            output_dim=1, output_names=("перспективность",),
            max_rule_arity=2, max_rules=8,
            rule_generation_mode="prototype", prototype_sample_size=512))


def to_unit(train: np.ndarray, *others: np.ndarray) -> list:
    """Шкалирование в [0,1] по квантилям обучающей части: термы ждут долей."""
    lo = np.nanpercentile(train, 2, axis=0)
    hi = np.nanpercentile(train, 98, axis=0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    f = lambda z: np.clip((z - lo) / span, 0.0, 1.0)
    return [f(train)] + [f(z) for z in others]


def fill(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = X.copy()
    med = np.nanmedian(ref, axis=0)
    bad = ~np.isfinite(out)
    out[bad] = np.take(med, np.where(bad)[1])
    return out


def tt(z: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(z), dtype=torch.float32)


def fit_dffl(Xtr, ytr, Xva, yva, seed: int):
    torch.manual_seed(seed)
    cfg = build_config(Xtr.shape[1])
    model = build_stagewise_pretrained_hierarchical_model(
        cfg, sample_inputs=tt(Xtr), sample_targets=tt(ytr).view(-1, 1),
        validation_inputs=tt(Xva), validation_targets=tt(yva).view(-1, 1),
        bootstrap_config=BootstrapConfig(
            decision_task_type="binary_classification"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="binary_classification", epochs_per_stage=20,
            decision_epochs=15, learning_rate=0.02, batch_size=512,
            stage_selection_metric="f1"))
    trainer = FuzzyTrainer(model, TrainingConfig(
        task_type="binary_classification", max_epochs=40, learning_rate=0.01,
        batch_size=512, patience=8, binary_auto_pos_weight=True,
        gradient_clip_norm=5.0, device="cpu"))
    trainer.fit(tt(Xtr), tt(ytr).view(-1, 1), tt(Xva), tt(yva).view(-1, 1))
    model.eval()
    return model


def predict(model, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = []
    with torch.no_grad():
        for a in range(0, X.shape[0], chunk):
            z = model(tt(X[a:a + chunk]))
            z = z[0] if isinstance(z, tuple) else z
            out.append(torch.sigmoid(z).numpy().ravel())
    return np.concatenate(out)


def main() -> None:
    meta, prognoz = criterial_target.load_prognoz_grid()
    crit = crit_reference.criterial_score(prognoz)
    feat = criterial_target.training_features("v5_rebuilt")
    raw = pd.read_parquet(config.PROCESSED_DIR / "dataset_v5_rebuilt.parquet")
    valid = cell_mask.build_valid_mask(feat)
    pool = np.flatnonzero(valid)
    cx, cy = meta.cell_centers()
    cx, cy = cx.ravel(), cy.ravel()

    dzz = [c for c in honest_features("перенос") if c in feat.columns]
    S = spectral_mask.apply(feat[dzz])
    cols, rus = [], []
    for _, _, feats in BLOCKS:
        for col, ru in feats:
            src = S if col in S.columns else raw
            if col not in src.columns:
                raise KeyError("нет столбца %s" % col)
            cols.append(src[col].to_numpy(float))
            rus.append(ru)
    X = np.column_stack(cols)
    Xf = fill(X, X[pool])
    y = (crit >= np.nanquantile(crit[pool], 1 - AREA)).astype(int)
    log("ячеек листа %d, целевых %d, признаков %d в %d предметных блоках"
        % (pool.size, int(y[pool].sum()), X.shape[1], len(BLOCKS)))
    log("правил в сборке: %d" % build_config(X.shape[1]).generated_rule_count())

    folds = strip_folds(cx[pool], cy[pool], y[pool], GAP_M)
    if MAX_FOLDS:
        folds = folds[:MAX_FOLDS]
    log("полос с разрывом %.0f км: %d" % (GAP_M / 1000, len(folds)))

    dffl, boost = [], []
    for k, (tr, te) in enumerate(folds):
        Xtr, Xte = Xf[pool][tr], Xf[pool][te]
        ytr, yte = y[pool][tr], y[pool][te]
        Utr, Ute = to_unit(Xtr, Xte)
        model = fit_dffl(Utr, ytr, Ute, yte, config.CRIT_SEED + k)
        dffl.append(roc_auc_score(yte, predict(model, Ute)))
        m = make_model().fit(Xtr, ytr)
        boost.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        log("   полоса %d: ДФФЛ %.3f, бустинг %.3f" % (k + 1, dffl[-1],
                                                       boost[-1]))
    log("ветка с геологией: ДФФЛ %.3f ± %.3f | бустинг %.3f ± %.3f"
        % (np.mean(dffl), np.std(dffl), np.mean(boost), np.std(boost)))
    pd.DataFrame([{"модель": "ДФФЛ", "площадь_под_кривой": np.mean(dffl),
                   "разброс": np.std(dffl), "полос": len(folds)},
                  {"модель": "бустинг", "площадь_под_кривой": np.mean(boost),
                   "разброс": np.std(boost), "полос": len(folds)}]).to_csv(
        ROOT / "outputs" / "metrics" / "neurofuzzy_dffl.csv", index=False,
        encoding="utf-8-sig")

    # --- модель на всей площади листа: карта и трассы решения
    log("")
    log("обучение на всей площади листа")
    U_pool, U_all = to_unit(Xf[pool], Xf)
    final = fit_dffl(U_pool, y[pool], U_pool, y[pool], config.CRIT_SEED)
    p_d = predict(final, U_all)
    m = make_model().fit(Xf[pool], y[pool])
    p_b = m.predict_proba(Xf)[:, 1]
    shape = (meta.prf, meta.pic)
    km = 1000.0 / meta.dx
    th = nb.theta_from_azimuth(NW_AZIMUTH)
    sm = lambda v: nb.aniso_smooth(np.nan_to_num(v), shape, ALONG_KM * km,
                                   ACROSS_KM * km, th)
    p_d, p_b = sm(p_d), sm(p_b)
    rho = sps.spearmanr(p_d[valid], p_b[valid]).statistic
    td = p_d >= np.quantile(p_d[valid], 1 - AREA)
    tb = p_b >= np.quantile(p_b[valid], 1 - AREA)
    log("ранговая связь карт ДФФЛ и бустинга %+.3f, общих верхних 10%% %.0f%%"
        % (rho, 100 * (td & tb & valid).sum() / (tb & valid).sum()))

    rng = np.random.default_rng(config.CRIT_SEED)
    top = pool[td[pool]]
    bg = pool[~td[pool]]
    take = rng.choice(top, min(400, top.size), replace=False)
    flows = analyze_concept_flow(final, tt(U_all[take]), top_k_rules=1)
    acc: dict[str, list] = {}
    for f in flows:
        for c in f.decision_concept_contributions:
            acc.setdefault(c.concept_name, []).append(c.contribution[0])
    log("")
    log("ВКЛАД СВОДНЫХ КОНЦЕПТОВ В БАЛЛ, среднее по %d перспективным ячейкам"
        % take.size)
    tot = sum(abs(np.mean(v)) for v in acc.values()) or 1.0
    for name, v in sorted(acc.items(), key=lambda kv: -abs(np.mean(kv[1]))):
        log("   %-12s вклад %+6.3f, доля %4.1f%%"
            % (name, np.mean(v), 100 * abs(np.mean(v)) / tot))

    log("")
    log("ВЕДУЩИЕ ПРАВИЛА: какое правило чаще всего срабатывает в блоке")
    take_bg = rng.choice(bg, min(400, bg.size), replace=False)
    census = {}
    for label, cells in (("перспективные", take), ("фоновые", take_bg)):
        fl = analyze_concept_flow(final, tt(U_all[cells]), top_k_rules=1)
        for f in fl:
            for c in f.hidden_concepts:
                for r in c.top_rule_contributions:
                    key = (c.block_name, c.concept_name, label)
                    census.setdefault(key, {}).setdefault(
                        r.rule_text, []).append(r.normalized_weight)
            for r in f.top_decision_rules:
                key = ("решение", "перспективность", label)
                census.setdefault(key, {}).setdefault(
                    r.rule_text, []).append(r.normalized_weight)
    seen = {k[:2] for k in census}
    for block, concept in sorted(seen):
        log("   %s / %s" % (block, concept))
        for label in ("перспективные", "фоновые"):
            d = census.get((block, concept, label))
            if not d:
                continue
            text, w = max(d.items(), key=lambda kv: len(kv[1]))
            n = sum(len(v) for v in d.values())
            log("      %-14s %s | доля ячеек %.0f%%, вес %.2f"
                % (label, text, 100 * len(w) / n, float(np.mean(w))))

    log("")
    log("РАЗБОР ДВУХ ЯЧЕЕК (перспективная и фоновая)")
    pick = np.array([top[np.argmax(p_d[top])], bg[np.argmin(p_d[bg])]])
    txt = format_concept_flows(analyze_concept_flow(final, tt(U_all[pick]),
                                                    top_k_rules=2), decimals=3)
    for line in txt.splitlines():
        log("   " + line)

    ext = (cx.min() - 250, cx.max() + 250, cy.min() - 250, cy.max() + 250)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.6),
                             constrained_layout=True)
    for axi, (ttl, v) in zip(axes, (("Нечёткая иерархия (ДФФЛ)", p_d),
                                    ("Бустинг (рабочая точка)", p_b))):
        vv = np.where(valid, v, np.nan)
        cuts = np.nanpercentile(vv[valid], mapstyle.CLASS_EDGES[1:-1])
        img = np.where(valid, np.digitize(vv, cuts).astype(float), np.nan)
        axi.imshow(np.ma.masked_invalid(img.reshape(shape)), extent=ext,
                   origin="upper", interpolation="nearest", vmin=-0.5,
                   vmax=4.5, cmap=matplotlib.colors.ListedColormap(
                       mapstyle.CLASS_COLORS))
        axi.set_title(ttl, fontsize=11)
        axi.set_xlabel("X, км")
        axi.set_ylabel("Y, км")
        axi.xaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.yaxis.set_major_formatter(lambda t, _: "%.0f" % (t / 1000))
        axi.set_aspect("equal")
    fig.suptitle("Ветка с геологией: нечёткая иерархия против бустинга "
                 "(классы по доле площади)", fontsize=12)
    path = outputs.png("neurofuzzy_dffl.png")
    mapstyle.save_fig(fig, path, dpi=150)
    log("карта: %s" % path)
    (ROOT / "outputs" / "metrics" / "neurofuzzy_dffl.log").write_text(
        "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
