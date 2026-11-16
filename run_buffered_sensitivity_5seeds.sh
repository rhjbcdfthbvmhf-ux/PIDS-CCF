#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# A radius of 10 ensures that two 11x11 patches centered in development and
# test sets share no pixels. Indian Pines cannot retain all 16 test classes
# under this constraint and is therefore documented rather than force-fit.
for DATASET in PaviaU Salinas; do
  for SEED in 23 37 51 71 101; do
    PREFIX="buffered_${DATASET}_seed${SEED}"
    CLEAN_ROOT="runs_buffered_full_no_overlap/clean_stage"
    STAGE1_ROOT="runs_buffered_full_no_overlap/pids_stage1"
    STAGE2_ROOT="runs_buffered_full_no_overlap/pids_stage2"
    SELECT_PATH="runs_buffered_full_no_overlap/selection/${DATASET}/seed_${SEED}/selection.json"

    if [[ ! -f "${CLEAN_ROOT}/${DATASET}/clean/seed_${SEED}/metrics.json" ]]; then
      "$PY" src/train.py \
        --data-root data/raw --output-root "$CLEAN_ROOT" \
        --dataset "$DATASET" --variant clean --seed "$SEED" --epochs 100 \
        --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
        --split-mode buffered --buffer-radius 10 --backbone ccf \
        --learning-rate 3e-4 --weight-decay 1e-4 \
        --validation-interval 5 --patience 35 --evaluation-draws 10 \
        --selection-objective clean \
        > "${PREFIX}_clean.log" 2>&1
    fi

    if [[ ! -f "${STAGE1_ROOT}/${DATASET}/pids/seed_${SEED}/metrics.json" ]]; then
      "$PY" src/train.py \
        --data-root data/raw --output-root "$STAGE1_ROOT" \
        --dataset "$DATASET" --variant pids --seed "$SEED" --epochs 100 \
        --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
        --split-mode buffered --buffer-radius 10 --backbone ccf --training-compound \
        --learning-rate 3e-4 --weight-decay 1e-4 \
        --validation-interval 5 --patience 35 --evaluation-draws 10 \
        --selection-objective robust \
        > "${PREFIX}_stage1.log" 2>&1
    fi

    if [[ ! -f "${STAGE2_ROOT}/${DATASET}/pids/seed_${SEED}/metrics.json" ]]; then
      "$PY" src/train.py \
        --data-root data/raw --output-root "$STAGE2_ROOT" \
        --dataset "$DATASET" --variant pids --seed "$SEED" --epochs 80 \
        --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
        --split-mode buffered --buffer-radius 10 --backbone ccf --training-compound \
        --learning-rate 1e-4 --weight-decay 1e-4 \
        --initial-run "${STAGE1_ROOT}/${DATASET}/pids/seed_${SEED}" \
        --validation-interval 5 --patience 35 --evaluation-draws 10 \
        --selection-objective robust \
        > "${PREFIX}_stage2.log" 2>&1
    fi

    if [[ ! -f "$SELECT_PATH" ]]; then
      "$PY" src/select_stage.py \
        --stage1 "${STAGE1_ROOT}/${DATASET}/pids/seed_${SEED}" \
        --stage2 "${STAGE2_ROOT}/${DATASET}/pids/seed_${SEED}" \
        --output "$SELECT_PATH" \
        > "${PREFIX}_selection.log" 2>&1
    fi
  done
done

