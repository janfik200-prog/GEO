"""Обогащение признаков из ИМЕЮЩИХСЯ слоёв — честная held-out проверка.

Данные и постановка не меняются. Из тех же векторных слоёв извлекаем
mineral-systems признаки, которых в FEATURE_COLS нет (там только «расстояние до
ближайшего» + произведения):

  * prox_node   — близость к структурным узлам (пересечения разломов СЗ×СВ);
  * dens_tect   — плотность разломов (длина в радиусе DENS_R);
  * dens_magm   — плотность магматизма (площадь даек в радиусе DENS_R);
  * prox_*_wide — широкий ореол (многомасштабная близость) по 6 базовым слоям.

Сравниваем base(13) vs enriched под тем же 6-сидовым парным held-out протоколом
(как pu_seeds/halo_seeds): bootstrap-95% ДИ на разницу lift@top-10%/5%.
Плюс важность новых признаков на полной модели.

Запуск: python3 -m experiments.feat_enrich
"""

import sys, tempfile, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from src import config
from src.data_loader import find_base_dir, load_all_layers
from src.features import build_grid, build_features, distance_to_proximity
from src.model import BackgroundEnsemble, mark_presence
from src.utils import robust_normalize_01
from src.validation import assign_spatial_blocks
from experiments.pu_seeds import assign_folds

DENS_R = 2500          # радиус для плотности разломов/даек, м
AREAS = (0.05, 0.10)
N_BOOT = 5000

NEW_COLS = [
    "prox_node", "dens_tect", "dens_magm",
    "prox_facies_wide", "prox_paleo_wide", "prox_struct_wide",
    "prox_magm_wide", "prox_tect1_wide", "prox_tect2_wide",
]


def enrich(grid: gpd.GeoDataFrame, layers: dict) -> gpd.GeoDataFrame:
    """Добавить mineral-systems признаки из имеющихся слоёв (см. NEW_COLS)."""
    cent = grid.geometry.centroid

    # --- структурные узлы: пересечения разломов СЗ×СВ ---
    u1 = unary_union(layers["tect1"].geometry)
    u2 = unary_union(layers["tect2"].geometry)
    nodes = u1.intersection(u2)
    if nodes.is_empty:
        dist_node = np.full(len(grid), float(np.sqrt(((grid.total_bounds[2]-grid.total_bounds[0])**2))))
    else:
        dist_node = np.array([g.distance(nodes) for g in cent.values])
    grid["dist_node"] = dist_node
    grid["prox_node"] = distance_to_proximity(dist_node, "sqrt", 0.60)

    # --- плотность разломов: длина разломов в радиусе DENS_R ---
    faults = gpd.GeoDataFrame(
        geometry=list(layers["tect1"].geometry) + list(layers["tect2"].geometry), crs=grid.crs
    )
    fsindex = faults.sindex
    dens_t = np.zeros(len(grid))
    for i, c in enumerate(cent.values):
        buf = c.buffer(DENS_R)
        hit = list(fsindex.query(buf, predicate="intersects"))
        if hit:
            dens_t[i] = faults.geometry.iloc[hit].intersection(buf).length.sum()
    grid["dens_tect"] = robust_normalize_01(dens_t, 0.02, 0.98)

    # --- плотность магматизма: площадь даек в радиусе DENS_R ---
    magm = layers["magm"].reset_index(drop=True)
    msindex = magm.sindex
    dens_m = np.zeros(len(grid))
    for i, c in enumerate(cent.values):
        buf = c.buffer(DENS_R)
        hit = list(msindex.query(buf, predicate="intersects"))
        if hit:
            dens_m[i] = magm.geometry.iloc[hit].intersection(buf).area.sum()
    grid["dens_magm"] = robust_normalize_01(dens_m, 0.02, 0.98)

    # --- широкий ореол (многомасштабная близость) ---
    for role in ("facies", "paleo", "struct", "magm", "tect1", "tect2"):
        grid[f"prox_{role}_wide"] = distance_to_proximity(grid[f"dist_{role}"], "sqrt", 0.95)
    return grid


def fold_lift(X, presence, tr_idx, te_pos, rng, area) -> float:
    pos = tr_idx[presence[tr_idx] == 1]
    neg = tr_idx[presence[tr_idx] == 0]
    bg = rng.choice(neg, size=min(config.VAL_N_BACKGROUND, len(neg)), replace=False)
    model = BackgroundEnsemble(random_state=config.VAL_SEED)
    model.fit(np.vstack([X[pos], X[bg]]),
              np.r_[np.ones(len(pos)), np.zeros(len(bg))].astype(int))
    full = model.predict_proba(X)[:, 1]
    thr = np.quantile(full, 1.0 - area)
    return float((full[te_pos] >= thr).mean()) / area


def boot_ci(diff, seed):
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(N_BOOT)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    BASE_DIR = find_base_dir()
    layers, points = load_all_layers(BASE_DIR / config.SHP_SUBDIR, Path(tempfile.mkdtemp()))
    grid, _, _ = build_grid(layers["mask"], config.CELL_SIZE)
    grid = build_features(grid, layers)
    grid = enrich(grid, layers)
    grid, _ = mark_presence(grid, points)

    base_cols = config.FEATURE_COLS
    enr_cols = base_cols + NEW_COLS
    Xb = grid[base_cols].fillna(0).to_numpy()
    Xe = grid[enr_cols].fillna(0).to_numpy()
    presence = grid["presence"].to_numpy().astype(int)
    blocks = assign_spatial_blocks(grid, config.VAL_BLOCK_SIZE)

    print(f"Точек: {int(presence.sum())} | признаков base={len(base_cols)} enriched={len(enr_cols)} "
          f"(+{len(NEW_COLS)}) | сиды {config.VAL_SEEDS}\n")

    # важность новых признаков на полной модели
    rng0 = np.random.default_rng(config.VAL_SEED)
    pos = np.where(presence == 1)[0]
    neg = np.where(presence == 0)[0]
    bg = rng0.choice(neg, size=min(config.TRAIN_N_BACKGROUND, len(neg)), replace=False)
    full_model = BackgroundEnsemble(random_state=config.VAL_SEED)
    full_model.fit(np.vstack([Xe[pos], Xe[bg]]),
                   np.r_[np.ones(len(pos)), np.zeros(len(bg))].astype(int))
    imp = full_model.feature_importances_
    order = np.argsort(imp)[::-1]
    print("Топ-8 признаков по важности (enriched, * — новый):")
    for k in order[:8]:
        mark = " *" if enr_cols[k] in NEW_COLS else ""
        print(f"   {imp[k]:.3f}  {enr_cols[k]}{mark}")
    print()

    # парное held-out сравнение
    print(f"{'сид':>5}{'base@10':>10}{'enr@10':>10}{'Δ@10':>9}")
    print("-" * 34)
    pairs = {a: {"b": [], "e": []} for a in AREAS}
    for seed in config.VAL_SEEDS:
        folds = assign_folds(blocks, config.VAL_N_SPLITS, seed)
        sm = {a: {"b": [], "e": []} for a in AREAS}
        for f in range(config.VAL_N_SPLITS):
            te = folds == f
            te_pos = np.where(te & (presence == 1))[0]
            if te_pos.size == 0:
                continue
            tr_idx = np.where(~te)[0]
            base_seed = seed * 1000 + f
            for a in AREAS:
                lb = fold_lift(Xb, presence, tr_idx, te_pos, np.random.default_rng(base_seed), a)
                le = fold_lift(Xe, presence, tr_idx, te_pos, np.random.default_rng(base_seed), a)
                sm[a]["b"].append(lb); sm[a]["e"].append(le)
                pairs[a]["b"].append(lb); pairs[a]["e"].append(le)
        b10, e10 = np.mean(sm[0.10]["b"]), np.mean(sm[0.10]["e"])
        print(f"{seed:>5}{b10:>9.2f}x{e10:>9.2f}x{e10-b10:>+8.2f}x")

    print("-" * 34)
    print(f"\nПар (сид×фолд): {len(pairs[0.10]['b'])}")
    for a in AREAS:
        b = np.array(pairs[a]["b"]); e = np.array(pairs[a]["e"]); d = e - b
        ci = boot_ci(d, config.VAL_SEED)
        sig = "ДА" if (ci[0] > 0 or ci[1] < 0) else "НЕТ"
        print(f"top-{int(a*100)}%: base {b.mean():.2f}x | enriched {e.mean():.2f}x | "
              f"Δ {d.mean():+.2f}x (95% ДИ [{ci[0]:+.2f}, {ci[1]:+.2f}]) | "
              f"enr>base {(d>0).mean()*100:.0f}% | значимо: {sig}")


if __name__ == "__main__":
    main()
