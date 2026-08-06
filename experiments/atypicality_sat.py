"""Этап 9: лестница нетипичности на датасете v3 — все бесплатные съёмки в пуле.

ЧТО ДОБАВИЛОСЬ К ЭТАПУ 4b
-------------------------
Этап 8 собрал четыре съёмки сверх Sentinel-2 и пересобрал сам S2. В пул они
входят НЕ целиком: правила отброса (``src.features_v2.SAT_DROPPED``) писались по
измерениям, а не по факту скачивания.

* ``s1`` (5 призн.) — Sentinel-1 RTC, C-диапазон, 24 сцены, 100% покрытия;
* ``psr`` (7) — ALOS PALSAR-2, L-диапазон, 56 годовых мозаик, 100%;
* ``ast`` (21) — ASTER SWIR, мозаика 2001/2006, покрытие 69.7% — УСЛОВНАЯ группа;
* ``l8`` (2) — от 90 сцен Landsat 8/9 уцелел только тепловой канал: оптика
  повторяет Sentinel-2 с |r| = 0.83..1.00;
* ``s2raw`` (9) — сырые средние каналов Sentinel-2, включая красный край.
  В пуле v2 они были выброшены как избыточные (отношения предсказывали их на
  R^2 = 0.97-0.99), но на композите из 180 сцен вместо 60 это перестало
  выполняться (R^2 = 0.37-0.91): при 21 наблюдении на пиксель вместо 6 каналы
  разъехались с отношениями. Отдельной группой — чтобы решение принимала
  абляция, а не задним числом переписанное правило.

ЧТО В ЭТОМ ПРОГОНЕ ПРОВЕРЯЕТСЯ, А ЧТО НЕТ
-----------------------------------------
Первичная метрика и порог значимости — прежние, из ``config.VER_PRIMARY``, и
менять их по результату нельзя: они пре-зарегистрированы до этапа 3.

Главный вопрос НЕ «стал ли lift выше» — при 19 несмещённых точках одна точка
стоит 0.53 lift, и почти любой сдвиг укладывается в шум. Главный вопрос —
абляции: даёт ли хоть одна новая съёмка что-то, чего нет в уже собранном.
Ответ «нет» здесь такой же полноценный результат, как «да»: он закрывает ветку
и экономит место в пуле, где каждый лишний признак оплачивается шумом.

ЧЕГО ЖДАТЬ ОТ ASTER, ЧЕСТНО И ДО ПРОГОНА
----------------------------------------
Группа ``ast`` заполнена на 69.7% валидных ячеек, а пропуски импутируются
медианой. Значит 30% площади получают по всем 21 признаку ровно нулевое
отклонение, то есть выглядят «типичными» просто потому, что ASTER их не снял.
Если у группы появится прирост, его придётся проверить на этот артефакт
(диагностика печатается в конце: связь скора с картой покрытия ASTER), иначе
«аномалия» окажется границей чужой сцены 07.06.2001.

Выход: ``outputs/metrics/atypicality_sat*.csv``, ``outputs/atypicality_sat_map.png``,
``outputs/atypicality_sat_ablation.png``.

Запуск из корня: ``python -m experiments.atypicality_sat``.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cell_mask, config, features_v2  # noqa: E402

from experiments import atypicality_v2  # noqa: E402

#: Сырые каналы S2 возвращаются в пул отдельной группой (см. заголовок).
RESTORE = tuple(config.V2_FEATURE_GROUPS["s2raw"])


def aster_coverage_check(stem: Path) -> None:
    """Не объясняется ли скор простым «здесь ASTER снимал, а здесь нет».

    Считается после прогона по сохранённым картам: доля площади с покрытием
    ASTER среди верхних 10% скора против доли по всему листу. Заметный перекос
    означает, что детектор нашёл границу сцены, а не геологию.
    """
    npz = Path(f"{stem}_scores.npz")
    if not npz.exists():
        return
    df = pd.read_parquet(config.PROCESSED_DIR / "dataset_v3.parquet")
    if "ast_aloh" not in df.columns:
        return
    valid = np.asarray(cell_mask.build_valid_mask(df))
    cov = np.isfinite(df["ast_aloh"].to_numpy(float))
    base = cov[valid].mean()
    print(f"\nДиагностика ASTER: покрытие {base:.1%} валидных ячеек.")
    print("Доля покрытых среди верхних 10% скора (перекос = найдена граница сцены):")
    with np.load(npz) as z:
        for key in sorted(z.files):
            if key == "valid":
                continue
            s = z[key][valid]
            top = np.argsort(s)[-int(0.10 * s.size):]
            frac = cov[valid][top].mean()
            flag = "   <-- ПЕРЕКОС" if abs(frac - base) > 0.15 else ""
            print(f"  {key:<28}{frac:>7.1%}{flag}")


if __name__ == "__main__":
    stem = "atypicality_sat"
    atypicality_v2.main(include_factors=True, stem_name=stem, pool_name="sat",
                        dataset="dataset_v3.parquet", restore=RESTORE)
    aster_coverage_check(ROOT / "outputs" / "metrics" / stem)
