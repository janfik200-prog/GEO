#!/usr/bin/env bash
# META1, часть С: все 36 синтетических сцен, по P штук параллельно (по
# умолчанию 5: процесс занимает 3–5 ГБ, ядер 16). Сначала Ф2 — по нему вывод.
# Готовые шаги каждой сцены пропускаются, поэтому скрипт можно перезапускать.
# Только процессор. Запуск из корня: bash experiments/openlin_meta_run.sh [P]
set -euo pipefail
P=${1:-5}
mkdir -p outputs/meta1/synth
ORDER=""
for BG in F2 F1; do for AMP in 4 8 16; do for FALSE in none L1 L3; do for SEED in 1 2; do
  ORDER="$ORDER $BG:$AMP:$FALSE:$SEED"
done; done; done; done
# первая — сцена, на которой проверялся код
ORDER="F2:8:L1:1 $(echo $ORDER | tr ' ' '\n' | grep -v '^F2:8:L1:1$' | tr '\n' ' ')"
echo $ORDER | tr ' ' '\n' | xargs -P "$P" -I{} bash experiments/openlin_meta_scene.sh {}
echo "=== всё $(date +%H:%M)"
