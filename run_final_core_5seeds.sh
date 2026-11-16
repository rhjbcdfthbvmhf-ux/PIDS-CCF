#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for DATASET in IndianPines PaviaU Salinas; do
  for SEED in 23 37 51 71 101; do
    LOG_PREFIX="final_${DATASET}_seed${SEED}"

    "$PY" src/train.py \
      --data-root data/raw --output-root runs_final_core_clean \
      --dataset "$DATASET" --variant clean --seed "$SEED" --epochs 100 \
      --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
      --backbone ccf --learning-rate 3e-4 --weight-decay 1e-4 \
      --validation-interval 5 --patience 35 --evaluation-draws 10 \
      --selection-objective clean \
      > "${LOG_PREFIX}_clean.log" 2>&1

    "$PY" src/train.py \
      --data-root data/raw --output-root runs_final_core_stage1 \
      --dataset "$DATASET" --variant pids --seed "$SEED" --epochs 100 \
      --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
      --backbone ccf --training-compound --learning-rate 3e-4 --weight-decay 1e-4 \
      --validation-interval 5 --patience 35 --evaluation-draws 10 \
      --selection-objective robust \
      > "${LOG_PREFIX}_stage1.log" 2>&1

    "$PY" src/train.py \
      --data-root data/raw --output-root runs_final_core_stage2 \
      --dataset "$DATASET" --variant pids --seed "$SEED" --epochs 80 \
      --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
      --backbone ccf --training-compound --learning-rate 1e-4 --weight-decay 1e-4 \
      --initial-run "runs_final_core_stage1/${DATASET}/pids/seed_${SEED}" \
      --validation-interval 5 --patience 35 --evaluation-draws 10 \
      --selection-objective robust \
      > "${LOG_PREFIX}_stage2.log" 2>&1

    SELECTION="runs_final_selection/${DATASET}/seed_${SEED}/selection.json"
    "$PY" src/select_stage.py \
      --stage1 "runs_final_core_stage1/${DATASET}/pids/seed_${SEED}" \
      --stage2 "runs_final_core_stage2/${DATASET}/pids/seed_${SEED}" \
      --output "$SELECTION" \
      > "${LOG_PREFIX}_selection.log" 2>&1

    SELECTED_RUN=$(
      "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_run_dir"])' \
        "$SELECTION"
    )
    "$PY" src/scene_router.py \
      --clean-run "runs_final_core_clean/${DATASET}/clean/seed_${SEED}" \
      --robust-run "$SELECTED_RUN" --data-root data/raw \
      --train-draws 10 --validation-draws 3 --test-draws 10 \
      --scene-size 512 --batch-size 512 --workers 4 \
      --output "runs_final_router/${DATASET}/seed_${SEED}/router.json" \
      > "${LOG_PREFIX}_router.log" 2>&1
  done
done
