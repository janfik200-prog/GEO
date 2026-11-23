"""Каналы предметной модели одной таблицей — для словаря признаков и базы.

Каналы карты на знаниях (`experiments/gmo_knowledge_map_v0.py`, v0.3c–d)
посчитаны в разных расчётах и лежат в разных файлах: линеаменты с
поправкой на фон, карта распределения свит, каналы из неё, кромки полей,
магнитный дефицит. Здесь они собираются на всю сетку листа (22 946 ячеек,
адрес row, col, x, y), чтобы базу можно было наполнить ими как обычным
набором (`GIS-agent-db/scripts/db_load_datasets.py knowledge_channels`), а
связи «признак — критерий» в модели ставились на настоящие коды, а не
висели пробелами. Описание каждого кода — `src/feature_dict.py`
(`KNOWLEDGE_COLS`, группа «know»).

Значения те же, что в карте на знаниях, с одной оговоркой: магнитный
дефицит и кромки полей здесь посчитаны по всей сетке листа, а в карте — по
ячейкам с разметкой свит (медиана регионального поля по ним отличается в
третьем знаке).

Запуск::

    python -X utf8 -m experiments.build_knowledge_channels
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gmo_knowledge_map_v0 as km  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = ROOT / "data/processed/knowledge_channels.parquet"


def main() -> None:
    d = pd.read_parquet(km.DATA, columns=["row", "col", "x", "y", "pf_gr_tdr", "pf_mag_tdr",
                                          "pf_mag_up1", "pf_mag_up10"])
    bg = pd.read_parquet(km.LBG, columns=["row", "col", "bg_observed_x", "lincxrank_dens",
                                          "lincxrank_node_dens", "lincxrank_dist", "lincb_dir_sin"])
    d = d.merge(bg, on=["row", "col"], how="left")
    lc = pd.read_parquet(km.LINC, columns=["row", "col", "linc_dist"])
    d = d.merge(lc, on=["row", "col"], how="left")
    d["lincx_dist_m"] = d.linc_dist.where(d.bg_observed_x.fillna(False).astype(bool))
    d["edge_gr_m"] = km.edge_distance(d, "pf_gr_tdr")
    d["edge_mag_m"] = km.edge_distance(d, "pf_mag_tdr")
    act = d.pf_mag_up10 > d.pf_mag_up10.median()
    d["mag_deficit"] = (d.pf_mag_up1 - d.pf_mag_up10).where(act)
    d = d.merge(km.unconformity(d[["row", "col"]]), on=["row", "col"], how="left")
    d = d.merge(pd.read_parquet(km.SECTION_CH, columns=["row", "col", "fold_convex", "fold_slope",
                                                       "discord_m"]),
                on=["row", "col"], how="left")
    keys = ("il", "br", "lb", "ul", "kt")
    sg = pd.read_parquet(km.SVITY_GRID, columns=["row", "col", "rif_thick", "basement_depth",
                                                 "basement_resid_m"]
                         + [f"{k}_in500" for k in keys])
    d = d.merge(sg, on=["row", "col"], how="left")
    d = d.merge(pd.read_parquet(km.K23, columns=["row", "col", "svk_list", "svk_ul"]),
                on=["row", "col"], how="left")
    cols = ["row", "col", "x", "y", "lincxrank_dens", "lincxrank_node_dens", "lincxrank_dist",
            "lincx_dist_m", "lincb_dir_sin", "edge_gr_m", "edge_mag_m", "mag_deficit", "unc_dist",
            "fold_convex", "fold_slope", "discord_m", "rif_thick", "basement_depth",
            "basement_resid_m"] + [f"{k}_in500" for k in keys] + ["svk_list", "svk_ul"]
    out = d[cols].astype({c: "float64" for c in cols[4:]})
    out.to_parquet(OUT, index=False)
    print(f"ячеек {len(out)}, каналов {len(cols) - 4}: {OUT}")
    print((100 * out[cols[4:]].notna().mean()).round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
