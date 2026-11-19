#!/usr/bin/env bash
# META1, часть С: одна синтетическая сцена (аргумент BG:AMP:FALSE:SEED).
# площадка -> тракт -> пересэмплированная копия -> тракт -> оценка; готовые
# шаги пропускаются. Запускается из experiments/openlin_meta_run.sh параллельно.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
L=outputs/meta1
IFS=: read -r BG AMP FALSE SEED <<< "$1"
N="m1_${BG}_a${AMP}_${FALSE}_s${SEED}"
echo "=== $N начало $(date +%H:%M)"
[ -f "datacache/s2_lowsun/composite/winter_${N}_20m_лист.npz" ] || \
  $PY -m experiments.openlin_meta_synth --фон $BG --амп $AMP --ложные $FALSE --зерно $SEED > "$L/_synth_${N}.log" 2>&1
[ -f "outputs/metrics/openlin_zone_lines_${N}_v3_бл.npz" ] || \
  $PY -m experiments.openlin_zone_lines_v3 --site "$N" > "$L/_zl_${N}.log" 2>&1
[ -f "datacache/s2_lowsun/composite/winter_${N}_p_20m_лист.npz" ] || \
  $PY -c "from experiments.openlin_meta_synth import perturbed_copy; perturbed_copy('$N')" >> "$L/_synth_${N}.log" 2>&1
[ -f "outputs/metrics/openlin_zone_lines_${N}_p_v3_бл.npz" ] || \
  $PY -m experiments.openlin_zone_lines_v3 --site "${N}_p" > "$L/_zl_${N}_p.log" 2>&1
[ -f "$L/synth/${N}_eval.csv" ] || \
  $PY -m experiments.openlin_meta_eval --часть С --сцена "$N" > "$L/_eval_${N}.log" 2>&1
echo "=== $N готово $(date +%H:%M)"
