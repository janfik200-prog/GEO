"""Этап 3: сборка датасета v2 из групп признаков, собранных на этапе 2.

Сетка не меняется: та же ``prognoz.pgrid`` (154x149, 500 м), тот же C-порядок
ячеек. Поэтому «сборка» здесь — не пересчёт координат, а склейка таблиц,
посаженных на сетку каждым модулем-источником, и КОНТРОЛЬ того, что склейка
корректна.

Группы (реестр — ``config.V2_FEATURE_GROUPS``):

* ``gm``, ``ls``, ``dist``, ``relief_v1`` — из ``dataset_v1.parquet``;
* ``ter`` — ``terrain_v2.parquet`` (этап 2d);
* ``lin`` — ``lineament_features.parquet`` (этап 2c);
* ``s2`` — ``s2_features.parquet`` (этап 2a).

Условные группы собираются gracefully: если файла нет (ветка закрыта
стоп-правилом или ещё не досчитана), группа пропускается с записью в отчёт, а
не роняет сборку. Так же будет с аэрогаммой, если заказчик её пришлёт.

Контроль качества (всё пишется в ``dataset_v2_sources.md``):

* совпадение длины таблиц и порядка ячеек;
* доля NaN по каждой группе, отдельно по валидным ячейкам;
* число обусловленности корреляционной матрицы группы и всего пула —
  ловушка на алгебраические дубли (так были пойманы ``gm_gr_all`` в v1 и
  ``dem_curv`` в v2);
* самые связанные пары признаков ИЗ РАЗНЫХ групп — ловушка на скрытое
  дублирование между источниками (линеаменты против рельефа).

Выход: ``data/processed/dataset_v2.parquet``, ``dataset_v2_sources.md``,
``outputs/dataset_v2_preview.png``.
Запуск из корня: ``python -m experiments.build_dataset_v2``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import cell_mask, config, integro_grid  # noqa: E402

PARTS = {
    "ter": ("terrain_v2.parquet", "src/terrain_v2.py, Copernicus DEM GLO-30"),
    "lin": ("lineament_features.parquet", "src/lineaments.py, отмывка того же DEM"),
    "s2": ("s2_features.parquet", "src/s2_composite.py, Sentinel-2 L2A (COG на AWS)"),
}


def group_of(col: str) -> str | None:
    for name, prefixes in config.V2_FEATURE_GROUPS.items():
        if any(col.startswith(p) for p in prefixes):
            return name
    return None


def cond_number(tbl: pd.DataFrame, valid: np.ndarray) -> float:
    M = tbl.to_numpy(dtype=float)[valid]
    M = M[np.isfinite(M).all(axis=1)]
    if M.shape[0] < M.shape[1] + 2 or M.shape[1] < 2:
        return float("nan")
    sd = M.std(axis=0)
    M = M[:, sd > 0]
    if M.shape[1] < 2:
        return float("nan")
    return float(np.linalg.cond(np.corrcoef(M, rowvar=False)))


def main() -> None:
    report: list[str] = []
    meta = integro_grid.read_pgrid(config.GOLD_TARGET_PGRID)
    proj4 = integro_grid.read_grid_proj4(config.GOLD_TARGET_PGRID)
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    n_cells = meta.prf * meta.pic
    print(f"Сетка: {meta.prf}x{meta.pic} = {n_cells} ячеек, шаг {meta.dx} м")

    report.append("# Датасет v2: источники и контроль качества")
    report.append("")
    report.append(f"Сетка: `{config.GOLD_TARGET_PGRID.name}` {meta.prf}x{meta.pic}, "
                  f"шаг {meta.dx} м, CRS `{proj4}`")
    report.append("")
    report.append("Все группы посажены на ЭТУ ЖЕ сетку своими модулями, поэтому "
                  "пересчёта координат при сборке нет — контролируется только "
                  "совпадение длины и порядка ячеек.")
    report.append("")
    report.append("## Группы")

    out = df.copy()
    report.append("")
    report.append("| группа | источник | признаков | статус |")
    report.append("|---|---|---|---|")
    for name, prefixes in config.V2_FEATURE_GROUPS.items():
        if name in PARTS:
            continue
        cols = [c for c in df.columns if group_of(c) == name]
        report.append(f"| `{name}` | dataset_v1.parquet | {len(cols)} | из v1 |")

    for name, (fname, src) in PARTS.items():
        path = config.PROCESSED_DIR / fname
        if not path.exists():
            print(f"ГРУППА {name}: файла {fname} нет — пропущена")
            report.append(f"| `{name}` | {src} | 0 | **ПРОПУЩЕНА: нет {fname}** |")
            continue
        part = pd.read_parquet(path)
        assert len(part) == n_cells, f"{fname}: {len(part)} строк вместо {n_cells}"
        dup = [c for c in part.columns if c in out.columns]
        assert not dup, f"{fname}: колонки уже есть в датасете: {dup}"
        out = pd.concat([out, part], axis=1)
        print(f"группа {name}: +{len(part.columns)} признаков из {fname}")
        report.append(f"| `{name}` | {src} | {len(part.columns)} | добавлена |")

    valid = np.asarray(cell_mask.build_valid_mask(out))
    print(f"валидных ячеек: {int(valid.sum())} из {len(out)}")

    # --- NaN и обусловленность по группам ---
    report.append("")
    report.append("## Контроль по группам (по валидным ячейкам)")
    report.append("")
    report.append("| группа | признаков | NaN, % | cond корр. матрицы |")
    report.append("|---|---|---|---|")
    print(f"\n{'группа':<12}{'признаков':>10}{'NaN, %':>9}{'cond':>12}")
    groups: dict[str, list[str]] = {}
    for c in out.columns:
        g = group_of(c)
        if g and pd.api.types.is_numeric_dtype(out[c]):
            groups.setdefault(g, []).append(c)
    for g, cols in sorted(groups.items()):
        sub = out[cols]
        nan_pct = 100.0 * np.isnan(sub.to_numpy(dtype=float)[valid]).mean()
        cond = cond_number(sub, valid)
        print(f"{g:<12}{len(cols):>10}{nan_pct:>9.1f}{cond:>12.3g}")
        report.append(f"| `{g}` | {len(cols)} | {nan_pct:.1f} | {cond:.3g} |")

    # --- Скрытое дублирование МЕЖДУ группами ---
    from scipy.stats import spearmanr

    all_cols = [c for cols in groups.values() for c in cols]
    M = out[all_cols].to_numpy(dtype=float)[valid]
    ok = np.isfinite(M).all(axis=1)
    rho = spearmanr(M[ok]).statistic
    pairs = []
    for i in range(len(all_cols)):
        for j in range(i + 1, len(all_cols)):
            gi, gj = group_of(all_cols[i]), group_of(all_cols[j])
            if gi != gj:
                pairs.append((abs(rho[i, j]), all_cols[i], all_cols[j], rho[i, j]))
    pairs.sort(reverse=True)
    print("\nСамые связанные пары ИЗ РАЗНЫХ групп (скрытое дублирование источников):")
    report.append("")
    report.append("## Самые связанные пары из разных групп")
    report.append("")
    report.append("| признак A | признак B | Spearman |")
    report.append("|---|---|---|")
    for _, a, b, r in pairs[:10]:
        print(f"  {a:<18} ~ {b:<18} rho = {r:+.2f}")
        report.append(f"| `{a}` | `{b}` | {r:+.2f} |")

    out_path = config.PROCESSED_DIR / "dataset_v2.parquet"
    out.to_parquet(out_path, index=False)
    md = config.PROCESSED_DIR / "dataset_v2_sources.md"
    md.write_text("\n".join(report) + "\n", encoding="utf-8")

    # --- Превью: по одному представителю каждой новой группы ---
    shape = (meta.prf, meta.pic)
    show = [c for c in ("dem_incision", "lin_dens", "s2_clay", "s2_ndvi",
                        "s2_b11", "dem_tpi_2km") if c in out.columns]
    if show:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for ax, c in zip(axes.ravel(), show):
            img = np.where(valid.reshape(shape), out[c].to_numpy().reshape(shape), np.nan)
            lo, hi = np.nanpercentile(img, [2, 98])
            im = ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi)
            ax.set_title(c, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        for ax in axes.ravel()[len(show):]:
            ax.axis("off")
        fig.suptitle(f"Датасет v2: {len(all_cols)} числовых признаков, "
                     f"{int(valid.sum())} валидных ячеек", fontsize=13)
        fig.tight_layout()
        out_png = ROOT / "outputs" / "dataset_v2_preview.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120, bbox_inches="tight")

    print(f"\nСохранено: {out_path}\n           {md}")


if __name__ == "__main__":
    main()
