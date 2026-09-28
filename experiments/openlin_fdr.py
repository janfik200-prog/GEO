"""Как отделить линеаменты от мусора: порог по измеренной доле ложных.

ОТКУДА БЕРУТСЯ ОТРИЦАТЕЛЬНЫЕ ПРИМЕРЫ. Разметки геолога нет, и обучать «линия
или не линия» было не на чем. Фазовая подделка (:mod:`openlin_null_input`) эту
дыру закрывает: у композита сохраняются спектр и текстура, но разрушаются все
протяжённые объекты, поэтому КАЖДАЯ линия, прослеженная по подделке, заведомо
артефакт. Линии с настоящего композита — смесь настоящих и таких же артефактов.

ЧТО СЧИТАЕТСЯ. По каждой линии собираются признаки (длина, прямолинейность,
сила поля вдоль неё, однородность, согласие с направлением поля, доли
подтверждения оптикой и радаром). На них учится классификатор «настоящий вход
против подделки». Его оценка — не вероятность «это разлом», а мера непохожести
на артефакт.

ПОРОГ. Для каждого значения оценки известно, сколько линий даёт подделка и
сколько — настоящий вход. Их отношение (с поправкой на разное общее число) и
есть доля ложных среди оставленных. Порог выбирается по ней, а не на глаз:
«оставить 300 линий при 5 % мусора» — проверяемое утверждение.

Запуск из корня репозитория:
    python -m experiments.openlin_fdr --site anabar --fakes 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.openlin_lines import _axis, _branches, _path_length
from experiments.openlin_null_input import _mask
from experiments.openlin_unet import load_site
from experiments.openlin_unsup import CONFIRM_FRAC, OPTIC, RADAR

M = Path("outputs/metrics")

#: Окно локального фона для признака «сила над окружением», м.
RING_M = 2000.0


def line_features(la: np.ndarray, lo: np.ndarray, lr: np.ndarray,
                  field: np.ndarray, ang: np.ndarray, res_m: float
                  ) -> pd.DataFrame:
    """Признаки каждой прослеженной ветки."""
    from scipy.ndimage import uniform_filter

    from experiments.openlin_confidence import _fracs

    k = int(round(2 * RING_M / res_m))
    bg = uniform_filter(field.astype("float64"), size=k, mode="nearest")
    bg_sd = np.sqrt(np.maximum(
        uniform_filter(field.astype("float64") ** 2, size=k, mode="nearest")
        - bg ** 2, 1e-12))
    dens = uniform_filter(la.astype("float64"), size=k, mode="nearest")

    coords, f_opt, f_rad = _fracs(la, {"оптика": lo, "радар": lr}, res_m)
    rows = []
    for pts, fo, fr in zip(coords, f_opt, f_rad):
        r, c = pts[:, 0], pts[:, 1]
        ax, _a, _b = _axis(pts)
        proj = (pts - pts.mean(0)) @ ax
        length = float(proj.max() - proj.min()) * res_m
        path = _path_length(pts, res_m)
        my_az = float(np.degrees(np.arctan2(ax[1], -ax[0])) % 180)
        d = np.abs(ang[r, c] - my_az)
        d = np.minimum(d, 180.0 - d)
        sv = field[r, c]
        rows.append({
            "длина_км": length / 1000.0,
            "прямолинейность": min(length / path, 1.0) if path else 0.0,
            "сила_средняя": float(sv.mean()),
            "сила_макс": float(sv.max()),
            "сила_над_фоном": float(((sv - bg[r, c]) / bg_sd[r, c]).mean()),
            "однородность": float(1.0 - sv.std() / max(abs(sv.mean()), 1e-6)),
            "согласие_с_полем": float(1.0 - d.mean() / 90.0),
            "соседей_рядом": float(dens[r, c].mean()),
            "подтв_оптика": float(fo),
            "подтв_радар": float(fr),
            "подтв_оба": float(min(fo, fr)),
        })
    return pd.DataFrame(rows)


def one_pass(site: str, density: str, operator: str, fake: bool, seed: int,
             log, compress: str = "pca") -> pd.DataFrame:
    """Признаки всех линий одного прогона (настоящего или поддельного)."""
    from experiments.openlin_edges2 import edges_curve
    from experiments.openlin_edges3 import edges_hessian
    from experiments.openlin_domains import edges_from_components
    from experiments.openlin_gate import components
    from experiments.openlin_lines import trace_lines
    from experiments.openlin_null_input import run

    masks = {}
    for name, chans in (("оптика", OPTIC), ("радар", RADAR),
                        ("оптика+радар", OPTIC + RADAR)):
        rng = np.random.default_rng(seed)
        masks[name] = _mask(site, chans, density, operator, fake, rng,
                            compress)

    # поле и азимут для признаков берём из того же прогона по всем каналам
    rng = np.random.default_rng(seed)
    z = load_site(site)
    valid = np.isfinite(z["gray"])
    res_m = float(z["res_m"])
    if fake:
        from experiments.openlin_null_input import phase_randomize
        z = dict(z)
        for c in OPTIC + RADAR:
            if c in z:
                z[c] = np.where(valid, phase_randomize(z[c], valid, rng),
                                np.nan)
    from sklearn.decomposition import PCA

    from experiments.openlin_neigh import neigh_stack
    from experiments.openlin_unet import build_stack

    use = tuple(c for c in OPTIC + RADAR if c in z)
    x, _y, valid2, _u = build_stack(z, use)
    if compress == "raw":
        stack = x.astype("float32")
    else:
        nx, _n = neigh_stack(x, use, valid2, res_m, base=use)
        stack = np.concatenate([x, nx]).astype("float32")
    rows = stack[:, valid2].T
    if compress == "raw":
        comp = stack
    elif compress == "ae":
        from experiments.openlin_mae import latent_map, train_ae

        model = train_ae(stack, valid2, lambda *_a, **_k: None)
        comp = latent_map(model, stack, valid2)
    else:
        sel = rng.choice(len(rows), min(50_000, len(rows)), replace=False)
        pca = PCA(n_components=8, whiten=True, random_state=42).fit(rows[sel])
        comp = np.zeros((8,) + valid2.shape, dtype="float32")
        comp[:, valid2] = pca.transform(rows).T
    op = {"hessian": edges_hessian, "curve": edges_curve,
          "old": edges_from_components}[operator]
    field, ang = op(comp, valid2, res_m)
    field = np.nan_to_num(field, nan=0.0)

    df = line_features(masks["оптика+радар"], masks["оптика"], masks["радар"],
                       field, ang, res_m)
    df["настоящая"] = int(not fake)
    return df


def fdr_curve(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Сколько линий остаётся при каждом пороге и какая доля из них — мусор."""
    real = df["настоящая"].to_numpy() == 1
    n_real, n_fake = int(real.sum()), int((~real).sum())
    scale = n_real / max(n_fake, 1)          # подделок может быть несколько
    out = []
    for q in np.arange(0.0, 0.96, 0.05):
        t = float(np.quantile(score, q))
        keep_r = int((score[real] >= t).sum())
        keep_f = int((score[~real] >= t).sum())
        if keep_r == 0:
            continue
        out.append({"порог": t,
                    "линий": keep_r,
                    "ожидаемо мусора": keep_f * scale,
                    "доля мусора, %": 100 * keep_f * scale / keep_r})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="anabar")
    ap.add_argument("--density", default="предельно")
    ap.add_argument("--operator", default="hessian",
                    help="один оператор или несколько через запятую: "
                         "их линии объединяются в общий список")
    ap.add_argument("--fakes", type=int, default=2)
    ap.add_argument("--compress", default="pca",
                    help="pca, ae или оба через запятую: сжатие описания")
    a = ap.parse_args()

    def log(m=""):
        print(m, flush=True)

    ops = [o for o in a.operator.split(",") if o]
    log(f"площадка {a.site}, операторы {', '.join(ops)}, "
        f"плотность '{a.density}'")
    comps = [c for c in a.compress.split(",") if c]
    parts = []
    for cmp_ in comps:
        for op in ops:
            tag = f"{op}/{cmp_}"
            t0 = time.time()
            d = one_pass(a.site, a.density, op, False, 7, log, cmp_)
            d["оператор"] = tag
            parts.append(d)
            log(f"  {tag:16s} настоящий вход: {len(d)} линий "
                f"({time.time() - t0:.0f} с)")
            for i in range(a.fakes):
                t0 = time.time()
                f = one_pass(a.site, a.density, op, True, 100 + i, log, cmp_)
                f["оператор"] = tag
                parts.append(f)
                log(f"  {tag:16s} подделка {i + 1}: {len(f)} линий "
                    f"({time.time() - t0:.0f} с)")
    df = pd.concat(parts, ignore_index=True)

    # «оператор» в признаки не идёт: он одинаково представлен в обоих
    # классах и только дал бы модели повод выучить различия между прогонами
    feats = [c for c in df.columns if c not in ("настоящая", "оператор")]
    X = df[feats].to_numpy("float64")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["настоящая"].to_numpy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    score = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    from experiments.openlin_line_rank import auc

    log(f"\nразделимость «настоящая против подделки»: AUC "
        f"{auc(y, score):.3f} (0.5 — метод не отличает)")

    model.fit(X, y)
    w = model[-1].coef_[0]
    order = np.argsort(-np.abs(w))
    log("\nчто отличает настоящие линии (вес в модели):")
    for i in order[:6]:
        log(f"  {feats[i]:20s} {w[i]:+.2f}")

    tab = fdr_curve(df, score)
    log("\nсколько оставить и сколько среди них мусора:")
    log(tab.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    df["оценка"] = score
    # в имя обязательно входит способ сжатия: без него прогон по сырым каналам
    # затирает таблицу, посчитанную по признакам окрестности
    name = "+".join(ops) + "__" + "+".join(comps)
    df.to_csv(M / f"openlin_fdr_{a.site}_{name}.csv", index=False,
              encoding="utf-8-sig")
    log(f"\nсохранено: {M / f'openlin_fdr_{a.site}_{name}.csv'}")


if __name__ == "__main__":
    main()
